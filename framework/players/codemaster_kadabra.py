"""Kadabra codemaster -- GloVe-first clue search with one LLM danger check.

Kadabra is Abra evolved.  Abra (``codemaster_glove`` / ``guesser_glove``) is a
pure static-embedding pair, and it self-plays the single-team track at **6.80**
turns -- better than the LLM champion's ~7.3 -- for one structural reason: a
codemaster and a guesser that share the *same* vector space agree with each
other perfectly.  The clue the codemaster believes points at three of our words
is decoded by the guesser using literally the same cosines.  That coupling is
the engine, and Kadabra keeps it.

Abra's two fatal flaws are both places where the embedding is silent rather
than wrong:

1. **No assassin sense beyond cosine.**  GloVe knows that ``SURGE`` is near
   ``LEAD`` but not that a human reads ``PAGEANT`` as ``STATE`` (Miss ...).
   Lateral, cultural, pun-shaped associations are invisible to a 2014 co-
   occurrence matrix, and one assassin pick is the whole game.
2. **Out-of-vocabulary collapse on themed boards.**  ``XENOMORPH`` is GloVe
   rank 375895; ``TIKTOK`` is not in GloVe 6B at all.  On a slang/pop-culture
   pool the cosines degenerate and Abra guesses at random.

So: the embedding proposes, and **one** LLM call disposes.  The pipeline is

    board -> vectorised clue search over the GloVe vocabulary
          -> legality + stem filter (identical in behaviour to the champion's)
          -> top-K candidates with their intended targets
          -> ONE Anthropic call: "does any of these pull toward the assassin?"
          -> veto / penalise / pick, clue number from the embedding margins

with a deterministic fallback under every step.  The LLM is never on the
critical path: if the key is missing, the SDK is absent, the call times out or
the reply is unparseable, the embedding's own argmax is issued with a
conservative number.  That is the Abra 6.80 baseline, not a broken agent.

Competition invariants (a violation is a disqualification, not a bad score):

* every clue passes ``clue_is_legal`` -- one alphabetic English word, never a
  sub-word of, nor a superword of, any unrevealed board word;
* ``get_clue`` never raises and never returns anything but ``[str, int]``;
* one move is bounded by a hard wall (``MOVE_WALL_S``) well under the event's
  60 s soft limit, enforced by abandoning the worker thread rather than by
  politely asking it to stop.

Python 3.9 compatible.  Depends on numpy only through the vector cache loader,
which degrades to ``None`` when numpy or the cache is absent.
"""

import json
import os
import random
import re
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Framework hook-up
# ---------------------------------------------------------------------------

try:
    from players.codemaster import Codemaster
except Exception:  # pragma: no cover - keeps the file runnable standalone
    class Codemaster(object):
        def __init__(self):
            self.move_history = []

        def set_move_history(self, move_history):
            self.move_history = move_history

        def get_move_history(self):
            return self.move_history


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_VERSION = "kadabra build 2026-08-07"

DEFAULT_MODEL = "claude-sonnet-5"
MODEL_ENV = "KADABRA_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"

#: The tournament key, substituted by the packaging script when a real zip is
#: cut.  What lives in git is a placeholder, and only a value that looks like a
#: real key (``sk-``) is ever used, so the checked-in constant behaves exactly
#: like no key at all.
HARDCODED_API_KEY = "KADABRA-KEY-PLACEHOLDER"

DEBUG_PREFIX = "[Kadabra]"
QUIET_ENV = "KADABRA_QUIET"

#: Per-move cooperative budget: every stage checks it and degrades.
MOVE_DEADLINE_S = 38.0
#: Non-cooperative backstop.  Python cannot interrupt a thread blocked in a
#: socket read, so the only honest way to bound it is to stop waiting.
MOVE_WALL_S = 45.0
#: Per-attempt HTTP timeout.
CALL_TIMEOUT_S = 20.0

# -- the embedding search ---------------------------------------------------

#: How deep into the (frequency-ordered) cache a clue may be drawn from.  The
#: same 20k prefix Abra uses: rarer words make clues no partner decodes.
DEFAULT_CLUE_VOCAB = 20000

#: A clue must beat the danger ceiling by this much before it counts at all.
MIN_MARGIN = 0.010
#: Padding added to a danger word's cosine before it enters the ceiling.  The
#: assassin's is deliberately large: the whole point of Kadabra over Abra is
#: that an assassin pull is not merely a bad word, it is a lost game.
ASSASSIN_PAD = 0.10
OPPONENT_PAD = 0.045
CIVILIAN_PAD = 0.020
#: A clue whose weakest bought word is below this cosine is mush -- the
#: ``LADDER`` shape from the champion's forensics, where nothing on the board
#: clears 0.15 and the "target" and the assassin are indistinguishable.
MIN_ABS_COSINE = 0.150
#: Below this margin the last word of the group is not worth claiming, so the
#: number drops by one even though the group scored.
THIN_MARGIN = 0.045
#: How many own words one clue may claim.
MAX_NUMBER = 4
#: Rows of the score matrix walked when collecting legal candidates.
SCAN_ROWS = 600
#: Candidates handed to the LLM danger check.
N_CANDIDATES = 6
#: Longest clue we will issue.  Cosine does not care that
#: ``INSTRUMENTATION`` is a mouthful, but a stranger's guesser and a human
#: judge reviewing clue spirit both do.
MAX_CLUE_LENGTH = 13

# -- the danger check -------------------------------------------------------

#: Assassin rating (0-10) at or above which a clue is dead, whatever it scored.
LLM_VETO_SCORE = 5.0
#: Assassin rating at or above which a surviving clue is narrowed to one word.
LLM_NUMBER_CAP_SCORE = 3.0
#: Sub-veto penalties, in units of the embedding score (groups are ~1.0 apart).
LLM_ASSASSIN_WEIGHT = 0.16
LLM_OPPONENT_WEIGHT = 0.05
#: Nudge toward the clue the model itself named, when it survived the veto.
LLM_PICK_BONUS = 0.45
#: With no usable danger check at all, a clue is issued unchecked -- so it is
#: issued one notch below what the embedding claims.
#:
#: Measured, not assumed.  A paired 30-seed offline sweep (no API at all, so
#: every clue takes this path) went 5.43 at a cap of 2, **4.73 at 3** and 4.67
#: at 4, with **0 assassin deaths in all three arms** and no movement at all on
#: the slang or organiser pools.  The safety work is done by the margin gates
#: below, not by the cap, so the tight cap was buying nothing and costing 0.7
#: turns; 3 keeps one notch of conservatism on a clue nothing audited.
UNCHECKED_NUMBER_CAP = 3
#: Under this much remaining budget the danger check is skipped entirely.
CHECK_MIN_SECONDS = 8.0
#: Cosine to the assassin at which a clue is refused outright, whatever else it
#: has going for it.  Used on the paths where our own words are unpriceable and
#: a comparative margin would therefore veto everything.
ASSASSIN_ABS_VETO = 0.25

TEAM_TOTALS = {"Red": 9, "Blue": 8}

#: Offline clue vocabulary, used only when the vector cache is unavailable
#: *and* the LLM is unreachable.  Deliberately generic, common English nouns:
#: this path exists to keep the game legal and finishable, not to play well.
_FALLBACK_VOCAB = (
    "ANIMAL", "BUILDING", "CITY", "COLOUR", "DANGER", "EARTH", "FIRE",
    "FOREST", "GAME", "HISTORY", "JOURNEY", "KITCHEN", "LIGHT", "MACHINE",
    "MUSIC", "NATURE", "OCEAN", "PEOPLE", "PLANT", "POWER", "RIVER", "ROYAL",
    "SCIENCE", "SIGNAL", "SPACE", "SPEECH", "SPORT", "STORY", "TOOL",
    "TRAVEL", "VILLAGE", "WEATHER", "WINTER", "WORK",
)


def _truthy(value):
    return str(value if value is not None else "").strip().lower() in (
        "1", "true", "yes", "on")


def _quiet_default():
    return _truthy(os.environ.get(QUIET_ENV))


def _emit(message):
    """One prefixed diagnostics line.  Never raises, whatever the console.

    ASCII only, and ``print`` rather than logging: the organisers capture
    stdout, and a Windows code page that cannot encode an em dash would turn a
    diagnostic into a ``UnicodeEncodeError`` mid-game.
    """
    try:
        print("%s %s" % (DEBUG_PREFIX, message))
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 - diagnostics must never break a game
        pass


_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")


def _brief(reason, limit=110):
    """One-line, length-capped exception text -- no payloads, no keys."""
    text = " ".join(str(reason if reason is not None else "").split())
    text = _KEY_PATTERN.sub("sk-<redacted>", text)
    return text[:limit] if len(text) > limit else text


def _key_looks_real(value):
    return bool(value) and str(value).strip().startswith("sk-")


def _resolve_api_key(explicit=None):
    """``(key, source)``.  ``source`` is a label -- never the key itself."""
    if explicit:
        return str(explicit), "constructor kwarg"
    from_env = os.environ.get(KEY_ENV)
    if from_env:
        return from_env, "%s env var" % KEY_ENV
    if _key_looks_real(HARDCODED_API_KEY):
        return HARDCODED_API_KEY, "embedded key"
    return None, None


def _resolve_model(override=None):
    return override or os.environ.get(MODEL_ENV) or DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Word handling and legality
# ---------------------------------------------------------------------------

_ALPHA_RE = re.compile(r"^[A-Z]+$")


def _normalise(word):
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def _is_revealed(board_word):
    text = str(board_word)
    return len(text) > 0 and text[0] == "*"


def clue_is_legal(clue, board_words):
    """Competition legality: single alphabetic word, no sub-word derivation.

    Behaviourally identical to the champion's filter (and to the audit the
    harness runs independently), so a clue that passes here can never be
    counted illegal downstream.
    """
    if not clue:
        return False
    text = str(clue).strip().upper()
    if len(text.split()) != 1:            # multiword
        return False
    if not _ALPHA_RE.match(text):         # hyphens, digits, apostrophes
        return False
    if len(text) < 3:
        return False
    for word in board_words or ():
        if _is_revealed(word):
            continue
        norm = _normalise(word)
        if not norm:
            continue
        if text in norm or norm in text:
            return False
    return True


#: Shortest shared prefix that reads as the same stem to a human judge.
_STEM_PREFIX = 5


def _shares_stem(clue, board_word):
    """``SWIMMER`` / ``SWIMMING``: legal by substring, dubious to a judge.

    Applied on top of ``clue_is_legal`` rather than inside it, so the legality
    predicate keeps exactly the champion's behaviour and this stays a
    candidate-quality filter.
    """
    a, b = _normalise(clue), _normalise(board_word)
    if len(a) < _STEM_PREFIX or len(b) < _STEM_PREFIX:
        return False
    return a[:_STEM_PREFIX] == b[:_STEM_PREFIX]


def _clue_is_clean(clue, board_words):
    if not clue_is_legal(clue, board_words):
        return False
    for word in board_words or ():
        if _is_revealed(word):
            continue
        if _shares_stem(clue, word):
            return False
    return True


def _letters(word):
    return set(_normalise(word))


def _overlap_similarity(clue, board_word):
    """Letter-overlap similarity -- the last-resort offline signal."""
    a, b = _letters(clue), _letters(board_word)
    if not a or not b:
        return 0.0
    shared = len(a & b) / float(len(a | b))
    prefix = 0
    ca, cb = _normalise(clue), _normalise(board_word)
    for x, y in zip(ca, cb):
        if x != y:
            break
        prefix += 1
    return shared + 0.1 * prefix


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: Same cache, same env var as the Abra pair: coupling the two agents to the
#: identical vector space is the whole thesis, and a second copy of the file
#: would be a second thing to keep in sync.
CACHE_ENV = "KADABRA_GLOVE_CACHE"
ABRA_CACHE_ENV = "ABRA_GLOVE_CACHE"

_VEC_LOCK = threading.Lock()
_VEC_STATE = {"loaded": False, "value": None, "path": None}


def cache_path():
    return (os.environ.get(CACHE_ENV)
            or os.environ.get(ABRA_CACHE_ENV)
            or os.path.join(_REPO_ROOT, "data", "glove_cache.npz"))


def vectors():
    """``(words, matrix, index)`` or ``None``.

    ``matrix`` rows are L2-normalised ``float32``, so a cosine is a dot
    product.  Loaded once per process and shared with the Abra partners when
    their module is importable, which keeps one copy of the matrix in memory
    instead of two.
    """
    if _VEC_STATE["loaded"]:
        return _VEC_STATE["value"]
    with _VEC_LOCK:
        if _VEC_STATE["loaded"]:
            return _VEC_STATE["value"]
        value = None
        path = cache_path()
        try:
            from players import glove_common  # noqa: WPS433 - optional
        except Exception:  # noqa: BLE001
            glove_common = None
        if (glove_common is not None
                and not os.environ.get(CACHE_ENV)):
            try:
                value = glove_common.vectors()
                path = glove_common.CACHE_PATH
            except Exception:  # noqa: BLE001
                value = None
        if value is None:
            try:
                import numpy as np  # noqa: WPS433 - optional dependency
                with np.load(path, allow_pickle=False) as data:
                    words = [str(w) for w in data["words"]]
                    matrix = data["vectors"]
                value = (words, matrix,
                         dict((word, i) for i, word in enumerate(words)))
            except Exception:  # noqa: BLE001 - a missing cache is not an error
                value = None
        _VEC_STATE["value"] = value
        _VEC_STATE["path"] = path
        _VEC_STATE["loaded"] = True
        return value


def board_rows(board_words, index):
    """``(kept words, row numbers)`` for board words in the vocabulary."""
    kept, rows = [], []
    for word in board_words:
        row = index.get(_normalise(word).lower())
        if row is None:
            continue
        kept.append(word)
        rows.append(row)
    return kept, rows


# ---------------------------------------------------------------------------
# Wall-clock budget
# ---------------------------------------------------------------------------

class _Deadline(object):
    """A per-move budget every stage degrades against.

    Elapsed time is the larger of the two clocks: wall clock is what the
    organisers' limit is read in and counts a suspended machine against us,
    while the monotonic clock is what a backwards NTP step cannot rewind.
    """

    def __init__(self, budget):
        self.budget = float(budget)
        self.start = time.time()
        self._mono_start = time.monotonic()

    def elapsed(self):
        return max(time.time() - self.start,
                   time.monotonic() - self._mono_start)

    def remaining(self):
        return self.budget - self.elapsed()

    def expired(self, reserve=0.0):
        return self.remaining() <= reserve


def _run_bounded(func, wall):
    """``(value, True)`` when ``func`` returned inside ``wall`` seconds.

    ``(None, False)`` otherwise, and the worker is abandoned exactly where it
    is.  The thread is a daemon, so a wedged connection cannot hold the process
    open; an exception inside ``func`` is re-raised in the caller.
    """
    if wall is None or wall <= 0:
        return func(), True
    box = {}

    def _target():
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    worker = threading.Thread(target=_target)
    worker.daemon = True
    worker.start()
    worker.join(wall)
    if worker.is_alive():
        return None, False
    if "error" in box:
        raise box["error"]
    return box.get("value"), True


# ---------------------------------------------------------------------------
# Minimal Anthropic client: retries, backoff, timeouts, never raises
# ---------------------------------------------------------------------------

class _LLM(object):
    """One-shot chat over the Anthropic Messages API.

    ``chat`` returns ``None`` instead of raising, so every caller can fall back
    deterministically.  The sampling-parameter shape is probed once per model
    because current Claude models differ on whether they accept ``thinking`` /
    ``output_config``, and an SDK too old to have those *Python parameters*
    gets the identical fields through ``extra_body`` -- otherwise extended
    thinking stays on by default and the whole token budget goes to a reasoning
    block that returns no text at all.  (That exact failure cost the champion a
    whole organiser test-run.)
    """

    def __init__(self, model, max_retries=2, timeout=CALL_TIMEOUT_S,
                 api_key=None, warn=None):
        self.model = model
        self.api_key = api_key
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.warn = warn
        self.last_error = None
        self._warned_unavailable = False
        self._client = None
        self._client_failed = False
        self._extra = None
        self._extra_body = False
        self._token_scale = 1
        self._rng = random.Random(0xCAFE)
        self.calls = 0
        self.failures = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def _param_shapes():
        return [
            {"thinking": {"type": "disabled"},
             "output_config": {"effort": "low"}},
            {"thinking": {"type": "disabled"}},
            {},
        ]

    def _get_client(self):
        if self._client is not None or self._client_failed:
            return self._client
        try:
            import anthropic  # noqa: WPS433 - optional dependency, imported late
        except Exception as exc:  # noqa: BLE001
            self.last_error = (type(exc).__name__,
                               "the anthropic package is not importable")
            self._client_failed = True
            return None
        api_key, _ = _resolve_api_key(self.api_key)
        if not api_key:
            self.last_error = ("MissingAPIKey",
                               "no key from the api_key kwarg, %s or the "
                               "embedded constant" % KEY_ENV)
            self._client_failed = True
            return None
        try:
            self._client = anthropic.Anthropic(api_key=api_key, max_retries=0,
                                               timeout=self.timeout)
        except TypeError:
            try:
                self._client = anthropic.Anthropic(api_key=api_key,
                                                   max_retries=0)
            except Exception as exc:  # noqa: BLE001
                self.last_error = (type(exc).__name__, _brief(exc))
                self._client_failed = True
                return None
        except Exception as exc:  # noqa: BLE001
            self.last_error = (type(exc).__name__, _brief(exc))
            self._client_failed = True
            return None
        return self._client

    def _report(self):
        if self.warn is None:
            return
        name, reason = self.last_error or ("NoResponse",
                                           "no usable reply from the model")
        try:
            self.warn("LLM call failed (%s: %s) -- using the embedding result "
                      "unchecked" % (name, _brief(reason)))
        except Exception:  # noqa: BLE001
            pass

    def _report_unavailable(self):
        if self._warned_unavailable:
            return
        self._warned_unavailable = True
        if self.last_error is None:
            self.last_error = ("MissingAPIKey", "no API key available")
        self._report()

    def key_source(self):
        return _resolve_api_key(self.api_key)[1]

    def available(self):
        return self._get_client() is not None

    def chat(self, system, user, max_tokens=700, deadline=None):
        """One completion.  Returns the assistant text, or None on failure."""
        if not self.available():
            self._report_unavailable()
            return None

        attempt = 0
        while attempt <= self.max_retries:
            if deadline is not None and deadline.expired(reserve=1.5):
                return None
            per_call = self.timeout
            if deadline is not None:
                per_call = max(3.0, min(per_call, deadline.remaining() - 0.5))
            text, retryable = self._call(system, user, max_tokens, per_call)
            if text is not None:
                return text
            if not retryable or attempt >= self.max_retries:
                self.failures += 1
                self._report()
                return None
            attempt += 1
            self.retries += 1
            delay = min(4.0, 0.6 * (2 ** (attempt - 1))) * (0.5 + self._rng.random())
            if deadline is not None:
                delay = min(delay, max(0.0, deadline.remaining() - 1.5))
            if delay > 0:
                time.sleep(delay)

        self.failures += 1
        self._report()
        return None

    def _shape_kwargs(self, fields):
        if not fields:
            return {}
        if self._extra_body:
            return {"extra_body": dict(fields)}
        return dict(fields)

    def _empty_reply(self, response, budget):
        """Record a text-less reply; say whether a retry could fix it."""
        stop = getattr(response, "stop_reason", None)
        self.last_error = ("EmptyReply",
                           "no text block (stop_reason=%s)" % stop)
        if stop == "max_tokens" and self._token_scale < 4:
            self._token_scale *= 2
            return True
        return False

    def _call(self, system, user, max_tokens, per_call):
        client = self._get_client()
        if client is None:
            return None, False
        shapes = self._param_shapes() if self._extra is None else [self._extra]
        shape_index = 0
        budget = max(1, int(max_tokens) * self._token_scale)
        while shape_index < len(shapes):
            fields = dict(shapes[shape_index])
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=budget,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    timeout=per_call,
                    **self._shape_kwargs(fields)
                )
            except Exception as exc:  # noqa: BLE001
                blob = ("%s %s" % (type(exc).__name__, exc)).lower()
                if (fields and not self._extra_body
                        and isinstance(exc, TypeError)
                        and "unexpected keyword argument" in blob):
                    # The installed SDK has no such Python parameter.  Re-send
                    # the same fields through extra_body, which lands them
                    # verbatim in the request JSON the server reads.
                    self._extra_body = True
                    continue
                unsupported = ("does not support" in blob
                               or "unexpected keyword" in blob
                               or "extra inputs are not permitted" in blob)
                if unsupported and shape_index + 1 < len(shapes):
                    shape_index += 1
                    continue
                self.last_error = (type(exc).__name__, _brief(exc))
                return None, self._retryable(blob)
            else:
                self._extra = shapes[shape_index]
                self.calls += 1
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.input_tokens += getattr(usage, "input_tokens", 0) or 0
                    self.output_tokens += getattr(usage, "output_tokens", 0) or 0
                parts = []
                for block in getattr(response, "content", None) or ():
                    if getattr(block, "type", None) == "text":
                        parts.append(getattr(block, "text", "") or "")
                text = "".join(parts)
                if text.strip():
                    return text, False
                return None, self._empty_reply(response, budget)
        self.last_error = ("UnsupportedParameters",
                           "no accepted sampling-parameter shape for %s"
                           % self.model)
        return None, True

    @staticmethod
    def _retryable(blob):
        for token in ("timeout", "timed out", "rate", "overloaded", "429",
                      "500", "502", "503", "529", "connection", "temporarily"):
            if token in blob:
                return True
        return False

    def usage_summary(self):
        return {"model": self.model, "calls": self.calls,
                "retries": self.retries, "failures": self.failures,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

def _extract_json(text, opener, closer):
    """Pull the first balanced JSON value out of a chatty response."""
    if not text:
        return None
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:  # noqa: BLE001
                    return None
    return None


def parse_candidates(text):
    """``[(clue, [targets], claimed number)]`` from a brainstorm reply."""
    out = []
    data = _extract_json(text, "[", "]")
    if isinstance(data, list):
        for item in data:
            targets, number = [], None
            if isinstance(item, dict):
                clue = (item.get("clue") or item.get("word")
                        or item.get("CLUE") or "")
                raw = item.get("targets") or item.get("words") or []
                if isinstance(raw, (list, tuple)):
                    targets = [t for t in (_normalise(x) for x in raw) if t]
                try:
                    number = int(item.get("number"))
                except (TypeError, ValueError):
                    number = None
            elif isinstance(item, str):
                clue = item
            else:
                continue
            clue = _normalise(clue)
            if not clue:
                continue
            if number is None:
                number = len(targets) or 1
            out.append((clue, targets, max(1, number)))
    if out:
        return out
    for token in re.findall(r"\b[A-Za-z]{3,}\b", text or "")[:10]:
        out.append((_normalise(token), [], 1))
    return out


def _coerce_rating(value):
    """A 0-10 rating out of whatever the model felt like emitting."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value or "").strip().lower()
        table = {"none": 0.0, "no": 0.0, "low": 2.0, "medium": 5.0,
                 "moderate": 5.0, "high": 8.0, "yes": 8.0, "severe": 10.0}
        if text in table:
            return table[text]
        match = re.search(r"\d+(?:\.\d+)?", text)
        if not match:
            return None
        number = float(match.group(0))
    if number != number:  # NaN
        return None
    return max(0.0, min(10.0, number))


def parse_danger(text, clues):
    """``({clue: (assassin, opponent)}, picked clue or None)``.

    Tolerant on purpose: the reply is one small JSON object, but a model that
    wraps it in prose, uses ``"risk"`` instead of ``"assassin"`` or answers with
    a bare number per clue must all still be readable.  Anything genuinely
    unreadable comes back empty, and an empty danger read is treated as *no
    check performed* rather than as a clean bill of health.
    """
    wanted = dict((_normalise(c), c) for c in clues)
    ratings = {}
    pick = None

    data = _extract_json(text, "{", "}")
    if isinstance(data, dict):
        for key, value in data.items():
            norm = _normalise(key)
            if norm in ("BEST", "PICK", "CHOICE", "CHOSEN", "SAFEST"):
                candidate = _normalise(value if isinstance(value, str) else "")
                if candidate in wanted:
                    pick = wanted[candidate]
                continue
            clue = wanted.get(norm)
            if clue is None:
                continue
            assassin = opponent = None
            if isinstance(value, dict):
                for field, target in (("assassin", "a"), ("risk", "a"),
                                      ("assassin_risk", "a"),
                                      ("opponent", "o"), ("opponents", "o"),
                                      ("opponent_risk", "o"), ("enemy", "o")):
                    if field in value:
                        rating = _coerce_rating(value[field])
                        if rating is None:
                            continue
                        if target == "a" and assassin is None:
                            assassin = rating
                        elif target == "o" and opponent is None:
                            opponent = rating
            else:
                assassin = _coerce_rating(value)
            if assassin is None and opponent is None:
                continue
            ratings[clue] = (assassin if assassin is not None else 0.0,
                             opponent if opponent is not None else 0.0)

    if pick is None and text:
        match = re.search(r"(?:best|pick|choice|safest)\W{0,12}([A-Za-z]{3,})",
                          text, re.IGNORECASE)
        if match:
            pick = wanted.get(_normalise(match.group(1)))
    return ratings, pick


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class AICodemaster(Codemaster):
    """GloVe-first clue search, LLM-checked for danger."""

    def __init__(self, team="Red", **kwargs):
        super(AICodemaster, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        self.opponent = "Blue" if self.team == "Red" else "Red"

        self.model = _resolve_model(kwargs.get("model"))
        self.deadline_s = float(kwargs.get("deadline", MOVE_DEADLINE_S))
        #: ``0`` runs the pipeline on the calling thread (tests).
        self.move_wall_s = float(kwargs.get("move_wall", MOVE_WALL_S))
        self.quiet = bool(kwargs.get("quiet", _quiet_default()))
        self.verbose = bool(kwargs.get("verbose", False))

        self.clue_vocab = int(kwargs.get("clue_vocab", DEFAULT_CLUE_VOCAB))
        self.max_number = int(kwargs.get("max_number", MAX_NUMBER))
        self.n_candidates = int(kwargs.get("n_candidates", N_CANDIDATES))
        self.scan_rows = int(kwargs.get("scan_rows", SCAN_ROWS))
        self.min_margin = float(kwargs.get("min_margin", MIN_MARGIN))
        self.assassin_pad = float(kwargs.get("assassin_pad", ASSASSIN_PAD))
        self.opponent_pad = float(kwargs.get("opponent_pad", OPPONENT_PAD))
        self.civilian_pad = float(kwargs.get("civilian_pad", CIVILIAN_PAD))
        self.min_abs_cosine = float(kwargs.get("min_abs_cosine",
                                               MIN_ABS_COSINE))
        self.thin_margin = float(kwargs.get("thin_margin", THIN_MARGIN))
        self.max_clue_length = int(kwargs.get("max_clue_length",
                                              MAX_CLUE_LENGTH))

        self.check_enabled = bool(kwargs.get("danger_check", True))
        self.veto_score = float(kwargs.get("veto_score", LLM_VETO_SCORE))
        self.number_cap_score = float(kwargs.get("number_cap_score",
                                                 LLM_NUMBER_CAP_SCORE))
        self.assassin_weight = float(kwargs.get("assassin_weight",
                                                LLM_ASSASSIN_WEIGHT))
        self.opponent_weight = float(kwargs.get("opponent_weight",
                                                LLM_OPPONENT_WEIGHT))
        self.pick_bonus = float(kwargs.get("pick_bonus", LLM_PICK_BONUS))
        self.unchecked_cap = int(kwargs.get("unchecked_cap",
                                            UNCHECKED_NUMBER_CAP))

        self.llm = _LLM(self.model,
                        max_retries=int(kwargs.get("max_retries", 1)),
                        timeout=float(kwargs.get("call_timeout",
                                                 CALL_TIMEOUT_S)),
                        api_key=kwargs.get("api_key"),
                        warn=self._warn)

        self.words = []
        self.maps = []
        self.move_history = []

        self._turn = 0
        self._turn_path = "embedding"
        self._turn_candidates = 0
        #: clues already given this game -- repeats were 36.8% fatal for the
        #: champion, and they cost nothing to ban.
        self._issued = set()
        self.checks_run = 0
        self.checks_vetoed = 0
        self.number_capped = 0
        #: one entry per danger check, so a lost game is diagnosable offline
        self.check_log = []

        self._announce()

    # -- diagnostics -------------------------------------------------------

    def _say(self, message):
        if not self.quiet:
            _emit("codemaster(%s) %s" % (self.team, message))

    def _warn(self, message):
        """Printed even in quiet mode -- a silent degradation is the one
        failure mode indistinguishable from a bad agent."""
        _emit("WARNING: %s" % message)

    def _announce(self):
        self._say("init: version=%s model=%s" % (AGENT_VERSION, self.model))
        source = self.llm.key_source()
        if source:
            self._say("api key: found via %s (value never printed)" % source)
        else:
            self._warn("no API key found (checked the api_key kwarg, %s and "
                       "the embedded HARDCODED_API_KEY) -- clues will be pure "
                       "embedding, issued unchecked" % KEY_ENV)
        try:
            import anthropic  # noqa: WPS433
            self._say("anthropic package: ok (version %s)"
                      % getattr(anthropic, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001
            self._warn("anthropic package not importable (%s: %s) -- clues "
                       "will be issued unchecked"
                       % (type(exc).__name__, _brief(exc)))
        loaded = vectors()
        if loaded is None:
            self._warn("GloVe cache NOT loaded (looked in %s) -- the primary "
                       "clue engine is unavailable and every clue will come "
                       "from the offline letter-overlap fallback"
                       % _VEC_STATE.get("path"))
        else:
            self._say("GloVe cache: %d words x %d dims from %s"
                      % (len(loaded[0]), loaded[1].shape[1],
                         _VEC_STATE.get("path")))

    def _log_turn(self, result, before):
        if self.quiet:
            return
        self._say("clue %d: %s %s | path=%s candidates=%d | check %d run, "
                  "%d vetoed, %d number-capped"
                  % (self._turn, result[0], result[1], self._turn_path,
                     self._turn_candidates,
                     self.checks_run - before[0],
                     self.checks_vetoed - before[1],
                     self.number_capped - before[2]))

    # -- framework hooks ---------------------------------------------------

    def set_game_state(self, words, maps):
        self.words = list(words)
        self.maps = list(maps)

    def set_move_history(self, move_history):
        self.move_history = list(move_history or [])

    def get_clue(self):
        """Return ``[clue, number]``.  Never raises, never illegal."""
        self._turn += 1
        self._turn_path = "embedding"
        self._turn_candidates = 0
        before = (self.checks_run, self.checks_vetoed, self.number_capped)
        try:
            result, finished = _run_bounded(self._get_clue_inner,
                                            self.move_wall_s)
            if not finished:
                self._turn_path = "wall-fallback"
                self._warn("get_clue exceeded its %.0f s hard wall -- "
                           "answering from the offline fallback"
                           % self.move_wall_s)
                result = self._fallback_clue()
        except BaseException as exc:  # noqa: BLE001 - a crash is a disqualification
            self._turn_path = "crash-fallback"
            self._warn("clue pipeline raised (%s: %s) -- using offline fallback"
                       % (type(exc).__name__, _brief(exc)))
            try:
                result = self._fallback_clue()
            except BaseException:  # noqa: BLE001
                result = ["SIGNAL", 1]
        result = self._sanitise(result)
        try:
            self._issued.add(_normalise(result[0]))
            self._log_turn(result, before)
        except Exception:  # noqa: BLE001 - bookkeeping never breaks a game
            pass
        return result

    def _sanitise(self, result):
        """Last line of defence: shape, legality and range, no matter what."""
        clue, number = None, 1
        try:
            clue = _normalise(result[0])
            number = int(result[1])
        except Exception:  # noqa: BLE001
            clue = None
        if not clue or not clue_is_legal(clue, self._unrevealed()):
            clue = self._safe_word()
            number = 1
        own = self._split_board()[0]
        ceiling = max(1, min(self.max_number, len(own) or 1))
        if number < 1 or number > ceiling:
            number = max(1, min(ceiling, number if number >= 1 else 1))
        return [clue, int(number)]

    def _safe_word(self):
        """Any legal word at all -- used only when everything else failed."""
        board = self._unrevealed()
        for word in _FALLBACK_VOCAB:
            if clue_is_legal(word, board) and _normalise(word) not in self._issued:
                return word
        for word in _FALLBACK_VOCAB:
            if clue_is_legal(word, board):
                return word
        return "SIGNAL"

    # -- board -------------------------------------------------------------

    def _unrevealed(self):
        return [w for w in self.words if not _is_revealed(w)]

    def _split_board(self):
        own, opp, civ, assassin = [], [], [], []
        for i, word in enumerate(self.words):
            if _is_revealed(word):
                continue
            kind = self.maps[i] if i < len(self.maps) else "Civilian"
            if kind == self.team:
                own.append(word)
            elif kind == self.opponent:
                opp.append(word)
            elif kind == "Assassin":
                assassin.append(word)
            else:
                civ.append(word)
        return own, opp, civ, assassin

    # -- pipeline ----------------------------------------------------------

    def _get_clue_inner(self):
        deadline = _Deadline(self.deadline_s)
        own, opp, civ, assassin = self._split_board()
        if not own:
            self._turn_path = "offline-fallback (no words left)"
            return self._fallback_clue()

        candidates = self._embedding_candidates(own, opp, civ, assassin)
        self._turn_candidates = len(candidates)

        budget_ok = not deadline.expired(reserve=CHECK_MIN_SECONDS)
        if not candidates:
            # The embedding has nothing to say -- every own word left is out of
            # the vocabulary (``PLATYPUS`` is GloVe rank 68860; ``XENOMORPH``
            # 375895), or no clue clears the safety gates.  This is exactly the
            # hole Kadabra exists to plug, so the turn's one LLM call is spent
            # *generating* the clue instead of checking one.
            if self.llm.available() and budget_ok:
                return self._llm_clue(own, opp, civ, assassin, deadline)
            self._turn_path = "offline-fallback (no legal candidate)"
            return self._fallback_clue()

        ratings, pick = {}, None
        if self.check_enabled and self.llm.available() and budget_ok:
            ratings, pick = self._danger_check(candidates, own, opp, assassin,
                                               deadline)

        return self._select(candidates, ratings, pick, own)

    # -- the embedding engine ----------------------------------------------

    def _embedding_candidates(self, own, opp, civ, assassin):
        """``[(clue, targets, number, score, margin)]``, best first.

        The whole Abra thesis in one matrix product: every clue in the
        vocabulary scored against every board word at once, then, for each
        group size ``k``, the margin between the *weakest* word the clue would
        buy and the strongest thing it must not.  A group only counts if that
        margin is positive and the weakest bought word is itself a real
        association (``min_abs_cosine``) -- the second condition is what stops
        the "nothing on this board clears 0.15, so the assassin and our target
        are indistinguishable" clue that killed the champion four times.
        """
        loaded = vectors()
        if loaded is None:
            return []
        try:
            import numpy as np  # noqa: WPS433
        except Exception:  # noqa: BLE001
            return []
        words, matrix, index = loaded

        board = own + opp + civ + assassin
        kept, rows = board_rows(board, index)
        if not rows:
            return []
        own_set, opp_set, ass_set = set(own), set(opp), set(assassin)
        own_cols = [i for i, w in enumerate(kept) if w in own_set]
        if not own_cols:
            return []

        limit = int(min(self.clue_vocab, matrix.shape[0]))
        sims = matrix[:limit].dot(matrix[rows].T)

        # A board word missing from the vocabulary is not "safe"; it is a word
        # the partner cannot reach and we cannot price.  Own words missing are
        # simply unclaimable (they are not in own_cols).  Danger words missing
        # get no column, which is the one place this agent is structurally
        # blind -- and exactly where the LLM check earns its call.
        ceiling = np.full(limit, -1.0, dtype="float32")
        for group, pad in ((ass_set, self.assassin_pad),
                           (opp_set, self.opponent_pad)):
            cols = [i for i, w in enumerate(kept) if w in group]
            if cols:
                ceiling = np.maximum(ceiling, sims[:, cols].max(axis=1) + pad)
        civ_cols = [i for i, w in enumerate(kept)
                    if w not in own_set and w not in opp_set and w not in ass_set]
        if civ_cols:
            ceiling = np.maximum(ceiling,
                                 sims[:, civ_cols].max(axis=1) + self.civilian_pad)

        own_sims = sims[:, own_cols]
        order = np.argsort(-own_sims, axis=1)
        own_sorted = np.take_along_axis(own_sims, order, axis=1)

        max_k = int(min(self.max_number, len(own_cols)))
        best_score = np.full(limit, -1e9, dtype="float32")
        best_k = np.zeros(limit, dtype="int32")
        best_margin = np.zeros(limit, dtype="float32")
        for k in range(1, max_k + 1):
            weakest = own_sorted[:, k - 1]
            margin = weakest - ceiling
            # Score: a word is worth ~1.0, and margin buys at most ~1.0 more,
            # so a genuinely safe 2 always beats a marginal 3 and never beats a
            # safe 3.  Clipping stops one huge cosine from buying a whole word.
            score = (k + 4.0 * np.clip(margin, 0.0, 0.25)
                     + 0.4 * np.clip(own_sorted[:, 0], 0.0, 1.0))
            ok = (margin >= self.min_margin) & (weakest >= self.min_abs_cosine)
            score = np.where(ok, score, -1e9)
            better = score > best_score
            best_score = np.where(better, score, best_score)
            best_k = np.where(better, k, best_k)
            best_margin = np.where(better, margin, best_margin)

        board_words = self._unrevealed()
        ranked = np.argsort(-best_score)
        out = []
        seen_stems = set()
        for row in ranked[:self.scan_rows]:
            if best_score[row] < -1e8:
                break
            clue = _normalise(words[row])
            if not clue or clue in self._issued:
                continue
            if len(clue) > self.max_clue_length:
                continue
            if not _clue_is_clean(clue, board_words):
                continue
            stem = clue[:4]
            if stem in seen_stems:      # LUGGAGE / LUGGED add no information
                continue
            seen_stems.add(stem)
            k = int(best_k[row])
            margin = float(best_margin[row])
            targets = [kept[own_cols[j]] for j in order[row][:k]]
            number = k
            if k > 1 and margin < self.thin_margin:
                # The group scored, but its last word is only barely ours.
                # Buy one fewer word rather than the whole group.
                number = k - 1
            out.append((clue, targets, int(number), float(best_score[row]),
                        margin))
            if len(out) >= self.n_candidates:
                break
        return out

    # -- the danger check --------------------------------------------------

    def _danger_check(self, candidates, own, opp, assassin, deadline):
        """ONE call.  "Would a human guesser read any of these as the assassin?"

        This is the only thing the LLM does in this agent, and it is the one
        thing a static embedding provably cannot do: the champion's forensics
        recorded four assassin deaths on clues whose cosine to the assassin was
        unremarkable and whose *cultural* pull was obvious (``PAGEANT`` ->
        ``STATE``, ``SINGER`` -> ``WASHER``).
        """
        if not candidates:
            return {}, None
        clues = [c[0] for c in candidates]
        system = (
            "You are a Codenames safety reviewer. A spymaster has shortlisted "
            "one-word clues. Your only job is to say how strongly an ordinary "
            "human guesser would be pulled toward the ASSASSIN word, or toward "
            "the opposing team's words, by each clue. Think about puns, "
            "compound words, brand names, film and song titles and everyday "
            "phrases, not just dictionary meaning -- those are the links that "
            "lose games. Be strict: a plausible link is a high rating."
        )
        lines = [
            "ASSASSIN (guessing it loses the game instantly): %s"
            % (", ".join(assassin) if assassin else "(none left)"),
            "Opposing team's words: %s"
            % (", ".join(opp) if opp else "(none left)"),
            "Our own words: %s" % ", ".join(own),
            "",
            "Candidate clues, each with the own words it is meant to point at:",
        ]
        for clue, targets, number, _score, _margin in candidates:
            lines.append("- %s (%d) -> %s" % (clue, number, ", ".join(targets)))
        lines.extend([
            "",
            "For each candidate clue rate, from 0 (no pull at all) to 10 "
            "(a guesser would very likely say it):",
            '  "assassin": how strongly the clue pulls toward the assassin '
            'word,',
            '  "opponent": how strongly it pulls toward the strongest opposing '
            'word.',
            'Then name the clue you would actually give as "best".',
            "",
            'Respond with ONLY a JSON object, e.g. '
            '{"CLUE": {"assassin": 0, "opponent": 3}, "best": "CLUE"}',
        ])

        text = self.llm.chat(system, "\n".join(lines), max_tokens=500,
                             deadline=deadline)
        if not text:
            return {}, None
        ratings, pick = parse_danger(text, clues)
        if not ratings:
            self._warn("danger check reply was unparseable -- issuing the "
                       "embedding pick unchecked")
            return {}, None
        self.checks_run += 1
        try:
            self.check_log.append({
                "turn": self._turn,
                "ratings": dict((c, list(v)) for c, v in ratings.items()),
                "pick": pick,
            })
        except Exception:  # noqa: BLE001
            pass
        return ratings, pick

    # -- selection ---------------------------------------------------------

    def _select(self, candidates, ratings, pick, own):
        """Combine the embedding score with the danger read."""
        scored = []
        for clue, targets, number, score, margin in candidates:
            rating = ratings.get(clue)
            final = score
            vetoed = False
            if rating is not None:
                assassin_rating, opponent_rating = rating
                if assassin_rating >= self.veto_score:
                    vetoed = True
                final -= self.assassin_weight * assassin_rating
                final -= self.opponent_weight * opponent_rating
                if clue == pick:
                    final += self.pick_bonus
            scored.append((final, vetoed, clue, targets, number, rating))

        survivors = [s for s in scored if not s[1]]
        if not survivors:
            # Everything the embedding liked, the model calls fatal.  Do not
            # argue with it: fall through to a deterministic clue instead.
            self.checks_vetoed += len(scored)
            self._turn_path = "offline-fallback (all candidates vetoed)"
            return self._fallback_clue()
        self.checks_vetoed += len(scored) - len(survivors)

        survivors.sort(key=lambda item: -item[0])
        _final, _vetoed, clue, targets, number, rating = survivors[0]

        if rating is None:
            # Unchecked: either no key, a failed call, or the model said
            # nothing about this particular clue.  An unchecked clue is not a
            # clean clue, so it is issued smaller.
            self._turn_path = "embedding (unchecked)"
            if number > self.unchecked_cap:
                self.number_capped += 1
            number = min(number, self.unchecked_cap)
        else:
            self._turn_path = "embedding + check"
            if rating[0] >= self.number_cap_score and number > 1:
                # Survived the veto but the model still smells the assassin:
                # buy one word, so a wrong guess cannot cascade.
                self.number_capped += 1
                number = 1

        number = max(1, min(int(number), self.max_number, len(own)))
        return [clue, number]

    # -- the out-of-vocabulary rescue --------------------------------------

    def _llm_clue(self, own, opp, civ, assassin, deadline):
        """ONE call, used *instead of* the danger check when GloVe is blind.

        The turn's call budget is one either way: normally it audits the
        embedding's shortlist, and here -- where there is no shortlist, because
        the words left are outside the 60k vocabulary -- it writes the clue.
        Whatever comes back is still filtered for legality and, wherever the
        clue itself is in vocabulary, still measured against the assassin.
        """
        system = (
            "You are an expert Codenames spymaster. Give a single-word clue "
            "that points at your own team's words and at nothing else. "
            "Guessing the assassin loses the game instantly, so a clue with "
            "any pull toward it is worthless however good it is otherwise. "
            "Think about how an ordinary person free-associates -- puns, "
            "brands, films, everyday phrases -- not just dictionary meaning."
        )
        lines = [
            "Your team's remaining words: %s" % ", ".join(own),
            "ASSASSIN (never point at this): %s"
            % (", ".join(assassin) if assassin else "(none left)"),
            "Opposing team's words: %s"
            % (", ".join(opp) if opp else "(none left)"),
            "Neutral words: %s" % (", ".join(civ) if civ else "(none left)"),
        ]
        if self._issued:
            lines.append("Clues already given this game (do not repeat): %s"
                         % ", ".join(sorted(self._issued)))
        lines.extend([
            "",
            "Propose 5 candidate clues, best first. Each clue must be a single "
            "ordinary English word, must not be any board word, and must not "
            "contain or be contained by any board word.",
            "",
            'Respond with ONLY a JSON array, e.g. '
            '[{"clue": "MAMMAL", "targets": ["PLATYPUS"], "number": 1}]',
        ])
        text = self.llm.chat(system, "\n".join(lines), max_tokens=500,
                             deadline=deadline)
        if not text:
            self._turn_path = "offline-fallback (rescue call failed)"
            return self._fallback_clue()

        board = self._unrevealed()
        own_set = set(_normalise(w) for w in own)
        best = None
        for clue, targets, number in parse_candidates(text):
            if clue in self._issued or not _clue_is_clean(clue, board):
                continue
            targets = [t for t in targets if t in own_set]
            if self._assassin_veto(clue, own, assassin):
                # The clue is in the vocabulary and it points at the assassin.
                # The model could not see that; the embedding can.
                self.checks_vetoed += 1
                continue
            count = max(1, min(int(number), len(targets) or 1,
                               self.max_number, len(own)))
            best = [clue, count]
            break
        if best is None:
            self._turn_path = "offline-fallback (rescue produced nothing legal)"
            return self._fallback_clue()
        self.checks_run += 1
        self._turn_path = "llm-rescue (own words out of vocabulary)"
        return best

    def _clue_pulls(self, clue, groups):
        """``[best cosine per group]`` for one clue, or ``None``.

        ``None`` means the embedding cannot price the clue at all -- the clue
        itself is out of vocabulary -- which is a *different* answer from "it
        is safe", and every caller treats it as such.  A group with no
        priceable word scores ``None`` in its slot for the same reason.
        """
        loaded = vectors()
        if loaded is None:
            return None
        _words, matrix, index = loaded
        row = index.get(_normalise(clue).lower())
        if row is None:
            return None
        out = []
        try:
            for group in groups:
                _kept, rows = board_rows(group or [], index)
                out.append(float(max(matrix[rows].dot(matrix[row])))
                           if rows else None)
        except Exception:  # noqa: BLE001 - a sensor must never break a game
            return None
        return out

    def _clue_margin(self, clue, own, opp, civ, assassin):
        """Comparative safety of one clue: own pull minus the danger ceiling.

        An own word the vocabulary cannot price contributes nothing, so the
        number understates our own side -- the pessimistic direction, and the
        right one for ranking a list of last-resort clues against each other.
        """
        pulls = self._clue_pulls(clue, (own, assassin, opp, civ))
        if pulls is None:
            return None
        best_own = pulls[0] if pulls[0] is not None else 0.0
        ceiling = -1.0
        for value, pad in ((pulls[1], self.assassin_pad),
                           (pulls[2], self.opponent_pad),
                           (pulls[3], self.civilian_pad)):
            if value is not None:
                ceiling = max(ceiling, value + pad)
        return best_own - ceiling

    def _assassin_veto(self, clue, own, assassin):
        """Does the embedding say this clue points at the assassin?

        Deliberately *not* the comparative margin: on the boards this is used
        for, our own words are the ones the vocabulary cannot see, so a
        comparison against them would veto everything.  Two absolute rules
        instead -- a strong pull on the assassin in its own right, or a pull
        stronger than anything the clue has on the own words we *can* price.
        """
        pulls = self._clue_pulls(clue, (own, assassin))
        if pulls is None or pulls[1] is None:
            return False
        if pulls[1] >= ASSASSIN_ABS_VETO:
            return True
        return pulls[0] is not None and pulls[1] >= pulls[0]

    # -- offline fallback --------------------------------------------------

    def _fallback_clue(self):
        """Deterministic, legal, instant.  No network needed.

        Two tiers.  If the vector cache is loaded the bundled vocabulary is
        ranked *in embedding space* against the same danger ceiling the main
        engine uses -- this path is reached on the boards where the main engine
        found nothing, which are exactly the boards where a blind clue is most
        likely to be fatal.  ``CITY`` scored top by letter overlap on a board
        whose assassin was ``STATE``; the cosine sees that instantly.

        Only with no vectors at all does it fall through to letter overlap,
        which is not meaning and does not pretend to be: it exists so a missing
        cache produces a legal, finishable game rather than a crash.  Number is
        always 1.
        """
        own, opp, civ, assassin = self._split_board()
        board = self._unrevealed()

        pool = [w for w in _FALLBACK_VOCAB
                if clue_is_legal(w, board) and _normalise(w) not in self._issued]
        if not pool:
            pool = [w for w in _FALLBACK_VOCAB if clue_is_legal(w, board)]
        ranked = []
        for word in pool:
            margin = self._clue_margin(word, own, opp, civ, assassin)
            if margin is not None:
                ranked.append((margin, word))
        if ranked:
            ranked.sort(key=lambda pair: (-pair[0], pair[1]))
            return [ranked[0][1], 1]

        best_clue, best_score = None, None
        for clue in pool or _FALLBACK_VOCAB:
            if not clue_is_legal(clue, board):
                continue
            total = 0.0
            for word in own:
                total += _overlap_similarity(clue, word)
            for word in opp + civ:
                total -= 0.5 * _overlap_similarity(clue, word)
            for word in assassin:
                total -= 4.0 * _overlap_similarity(clue, word)
            if best_score is None or total > best_score:
                best_clue, best_score = clue, total
        if best_clue is None:
            best_clue = self._safe_word()
        return [best_clue, 1]

    # -- diagnostics -------------------------------------------------------

    def usage_summary(self):
        summary = self.llm.usage_summary()
        summary.update({"checks_run": self.checks_run,
                        "checks_vetoed": self.checks_vetoed,
                        "number_capped": self.number_capped})
        return summary
