"""Kadabra guesser -- GloVe ranking primary, one LLM call where GloVe fails.

The point of the Kadabra pair is coupling.  ``codemaster_kadabra`` picks its
clue by maximising the cosine margin between the words it is buying and
everything dangerous, in exactly the vector space this file ranks with -- so on
a clue from its own partner, the guesser is not *guessing* what the spymaster
meant, it is recomputing it.  That is why the pure-embedding Abra pair
self-plays the single-team track at 6.80 turns, ahead of a much cleverer LLM
pair at ~7.3.

The embedding fails in two specific, detectable ways, and only there does this
file spend an API call:

1. **The clue is out of vocabulary.**  A stranger's codemaster says ``TIKTOK``
   or ``XENOMORPH`` or a brainstormed mashup like ``LASTDRINK``; GloVe 6B has
   no row for it and cosine ranking degenerates to noise.  This is a recorded
   killer: the champion once handed Abra ``LASTDRINK`` and Abra, ranking by
   letter overlap, guessed the assassin ``NAIL``.
2. **The margins are mush.**  The clue is in vocabulary but nothing on the
   board clears a real association, or the top two words are within noise of
   each other.  ``NAIL``-``BOX`` at 0.179 against ``NAIL``-``FILE`` at 0.193 is
   not a ranking, it is a coin flip, and one of those two words was the
   assassin.

In both cases: ONE call, ranking the remaining board words, and the LLM order
replaces the cosine order for the rest of the turn.  Never more than one call
per clue -- the ranking is cached and simply filtered as words are revealed.

The stop rule reads a **fresh** board state on every decision, with one
correction the engine forces.  ``game.Game.run`` calls, per guess::

    set_board(live_list)   # our copy
    get_answer()
    _accept_guess()        # the engine reveals the word in its own list
    keep_guessing()        # <- no set_board in between

so at ``keep_guessing`` time our board is exactly one guess stale: the word we
just guessed still looks unrevealed and would be re-offered as "the next
candidate", making every confidence ratio 1.0 and the stop rule vacuous.
``_pending`` closes that gap.  The colour is free: ``_accept_guess`` only
leaves the same side to move when the word belonged to the guessing team, so a
pending guess at ``keep_guessing`` time is always one of ours.

Competition invariants: ``get_answer`` returns an unrevealed board word or
``None``, never raises, and is bounded by a hard wall; clue number ``0`` (or an
absurdly large number) means unlimited guesses and is handled explicitly.

Python 3.9 compatible.
"""

import json
import os
import random
import re
import sys
import threading
import time

try:
    from players.guesser import Guesser
except Exception:  # pragma: no cover - keeps the file runnable standalone
    class Guesser(object):
        def __init__(self):
            self.move_history = []

        def set_move_history(self, move_history):
            self.move_history = move_history


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_VERSION = "kadabra build 2026-08-07"

DEFAULT_MODEL = "claude-sonnet-5"
MODEL_ENV = "KADABRA_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"
HARDCODED_API_KEY = "KADABRA-KEY-PLACEHOLDER"

DEBUG_PREFIX = "[Kadabra]"
QUIET_ENV = "KADABRA_QUIET"

MOVE_DEADLINE_S = 38.0
MOVE_WALL_S = 45.0
CALL_TIMEOUT_S = 20.0

TEAM_TOTALS = {"Red": 9, "Blue": 8}

# -- when the embedding is trusted -----------------------------------------

#: A cosine below this is not an association, it is noise.  If the *best* word
#: on the board is below it, the clue is mush and the LLM ranks instead.
TRUST_MIN_TOP = 0.20
#: If the top two words are this close, the embedding has no opinion about
#: which one the clue means -- and one of them can be the assassin.
TRUST_MIN_GAP = 0.030
#: A board word the vocabulary has never seen (``PLATYPUS`` is GloVe rank
#: 68860) is scored last by construction, so a turn containing one is ranked on
#: incomplete information.  Above this cosine the top pick is emphatic enough
#: that the unrankable word is very unlikely to be the intended one, and the
#: call is not worth making.
TRUST_CLEAR_TOP = 0.350

# -- the stop rule ---------------------------------------------------------

#: Cosine floor for guessing on inside the clue number.  Coupled with the
#: codemaster's ``MIN_ABS_COSINE`` (0.15): a word our own partner bought always
#: clears this, a word it did not usually does not.
GLOVE_CONTINUE_ABS = 0.130
#: ...or a strong relative signal, for a clue whose whole scale is low.
GLOVE_CONTINUE_RATIO = 0.62
#: The traditional bonus (+1) guess: deliberately a high bar in both terms.
GLOVE_BONUS_ABS = 0.300
GLOVE_BONUS_RATIO = 0.85

#: The same two gates on the LLM's 0-100 scale.
LLM_CONTINUE_ABS = 35.0
LLM_CONTINUE_RATIO = 0.55
LLM_BONUS_ABS = 65.0
LLM_BONUS_RATIO = 0.85

#: Letter overlap is not meaning.  On that path, take the number and no more.
OFFLINE_BONUS = False

#: An unlimited ("sweep") clue invites over-reach; demand a clearer signal.
SWEEP_RATIO = 0.80


def _truthy(value):
    return str(value if value is not None else "").strip().lower() in (
        "1", "true", "yes", "on")


def _quiet_default():
    return _truthy(os.environ.get(QUIET_ENV))


def _emit(message):
    try:
        print("%s %s" % (DEBUG_PREFIX, message))
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 - diagnostics must never break a game
        pass


_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")


def _brief(reason, limit=110):
    text = " ".join(str(reason if reason is not None else "").split())
    text = _KEY_PATTERN.sub("sk-<redacted>", text)
    return text[:limit] if len(text) > limit else text


def _key_looks_real(value):
    return bool(value) and str(value).strip().startswith("sk-")


def _resolve_api_key(explicit=None):
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


def _normalise(word):
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def _is_revealed(board_word):
    text = str(board_word)
    return len(text) > 0 and text[0] == "*"


def _letters(word):
    return set(_normalise(word))


def _overlap_similarity(clue, board_word):
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
# Vectors -- the same cache, and where possible the same loaded copy, as the
# Abra partners and the Kadabra codemaster.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_ENV = "KADABRA_GLOVE_CACHE"
ABRA_CACHE_ENV = "ABRA_GLOVE_CACHE"

_VEC_LOCK = threading.Lock()
_VEC_STATE = {"loaded": False, "value": None, "path": None}


def cache_path():
    return (os.environ.get(CACHE_ENV)
            or os.environ.get(ABRA_CACHE_ENV)
            or os.path.join(_REPO_ROOT, "data", "glove_cache.npz"))


def vectors():
    """``(words, matrix, index)`` or ``None`` when unavailable."""
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
        if glove_common is not None and not os.environ.get(CACHE_ENV):
            try:
                value = glove_common.vectors()
                path = glove_common.CACHE_PATH
            except Exception:  # noqa: BLE001
                value = None
        if value is None:
            try:
                import numpy as np  # noqa: WPS433
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
    """``(value, True)`` when ``func`` returned inside ``wall`` seconds."""
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
# Minimal Anthropic client (see codemaster_kadabra for the full commentary)
# ---------------------------------------------------------------------------

class _LLM(object):
    """One-shot chat over the Anthropic Messages API.  Never raises."""

    def __init__(self, model, max_retries=1, timeout=CALL_TIMEOUT_S,
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
        self._rng = random.Random(0xBEEF)
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
            import anthropic  # noqa: WPS433 - optional dependency
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
            self.warn("LLM call failed (%s: %s) -- ranking from the embedding"
                      % (name, _brief(reason)))
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

    def chat(self, system, user, max_tokens=600, deadline=None):
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


def parse_scores(text, options):
    """``{board word: score}`` from a ranking reply; tolerant of chattiness."""
    index = dict((_normalise(word), word) for word in options)

    scores = {}
    data = _extract_json(text, "{", "}")
    if isinstance(data, dict):
        for key, value in data.items():
            word = index.get(_normalise(key))
            if word is None:
                continue
            try:
                scores[word] = float(value)
            except (TypeError, ValueError):
                continue
    if scores:
        return scores

    ordered = _extract_json(text, "[", "]")
    if isinstance(ordered, list):
        rank = 0
        for entry in ordered:
            if isinstance(entry, dict):
                entry = entry.get("word") or entry.get("board_word") or ""
            word = index.get(_normalise(entry))
            if word is not None and word not in scores:
                scores[word] = float(max(1, 100 - 15 * rank))
                rank += 1
    if scores:
        return scores

    rank = 0
    for token in re.findall(r"[A-Za-z]{2,}", text or ""):
        word = index.get(_normalise(token))
        if word is not None and word not in scores:
            scores[word] = float(max(1, 100 - 15 * rank))
            rank += 1
    return scores


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class AIGuesser(Guesser):
    """Cosine ranking, with one LLM call exactly where cosine cannot see."""

    def __init__(self, team="Red", **kwargs):
        super(AIGuesser, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        self.opponent = "Blue" if self.team == "Red" else "Red"

        self.model = _resolve_model(kwargs.get("model"))
        self.deadline_s = float(kwargs.get("deadline", MOVE_DEADLINE_S))
        #: ``0`` runs the pipeline on the calling thread (tests).
        self.move_wall_s = float(kwargs.get("move_wall", MOVE_WALL_S))
        self.quiet = bool(kwargs.get("quiet", _quiet_default()))
        self.verbose = bool(kwargs.get("verbose", False))
        self.allow_bonus = bool(kwargs.get("allow_bonus", True))
        self.llm_rescue = bool(kwargs.get("llm_rescue", True))

        self.trust_min_top = float(kwargs.get("trust_min_top", TRUST_MIN_TOP))
        self.trust_min_gap = float(kwargs.get("trust_min_gap", TRUST_MIN_GAP))
        self.trust_clear_top = float(kwargs.get("trust_clear_top",
                                                TRUST_CLEAR_TOP))
        self.sweep_ratio = float(kwargs.get("sweep_ratio", SWEEP_RATIO))

        self.llm = _LLM(self.model,
                        max_retries=int(kwargs.get("max_retries", 1)),
                        timeout=float(kwargs.get("call_timeout",
                                                 CALL_TIMEOUT_S)),
                        api_key=kwargs.get("api_key"),
                        warn=self._warn)

        self.words = []
        self.move_history = []
        self.clue = ""
        self.num = 0

        # Per-turn state.  The bundled guesser_GPT forgets to reset all of it.
        self.guesses = 0
        self._ranking = None          # [(word, score)] descending
        self._ranking_key = None      # (clue, sweep?) the ranking was built on
        self._ranking_source = "glove"
        #: options the vector cache could not price on this turn
        self._unrankable = []
        self._turn_best = 0.0
        self._capacity = None         # (guess budget, sweep?) frozen per turn
        #: the guess the engine has accepted but our board copy has not seen
        self._pending = None
        #: bumped once per ``get_answer``; a pipeline abandoned at the hard
        #: wall may still be running when the next move starts, and this is how
        #: its late writes are told apart from the live move's.
        self._answer_gen = 0

        self.llm_calls = 0
        self.oov_clues = 0
        self.thin_clues = 0

        self._announce()

    # -- diagnostics -------------------------------------------------------

    def _say(self, message):
        if not self.quiet:
            _emit("guesser(%s) %s" % (self.team, message))

    def _warn(self, message):
        _emit("WARNING: %s" % message)

    def _announce(self):
        self._say("init: version=%s model=%s" % (AGENT_VERSION, self.model))
        source = self.llm.key_source()
        if source:
            self._say("api key: found via %s (value never printed)" % source)
        else:
            self._warn("no API key found (checked the api_key kwarg, %s and "
                       "the embedded HARDCODED_API_KEY) -- out-of-vocabulary "
                       "clues will fall back to letter overlap" % KEY_ENV)
        loaded = vectors()
        if loaded is None:
            self._warn("GloVe cache NOT loaded (looked in %s) -- the primary "
                       "ranking engine is unavailable"
                       % _VEC_STATE.get("path"))
        else:
            self._say("GloVe cache: %d words x %d dims from %s"
                      % (len(loaded[0]), loaded[1].shape[1],
                         _VEC_STATE.get("path")))

    # -- framework hooks ---------------------------------------------------

    def set_board(self, words):
        self.words = list(words)

    def set_move_history(self, move_history):
        self.move_history = list(move_history or [])

    def set_clue(self, clue, num):
        self.clue = str(clue or "")
        try:
            self.num = int(num)
        except (TypeError, ValueError):
            self.num = 1
        self.guesses = 0
        self._ranking = None
        self._ranking_key = None
        self._unrankable = []
        self._turn_best = 0.0
        self._capacity = None
        self._pending = None
        print("The clue is:", clue, num)
        return [clue, num]

    def get_answer(self):
        """Return one unrevealed board word.  Never raises, never invalid."""
        self._answer_gen += 1
        try:
            word, finished = _run_bounded(self._get_answer_inner,
                                          self.move_wall_s)
            if finished:
                return word
            self._warn("get_answer exceeded its %.0f s hard wall -- answering "
                       "from the offline ranking" % self.move_wall_s)
            options = self._options()
            if not options:
                return None
            pick = self._fallback_pick(options)
            self._pending = pick
            self._ranking = None
            self._ranking_source = "offline"
            return pick
        except BaseException:  # noqa: BLE001 - a crash is a disqualification
            try:
                options = self._options()
            except Exception:  # noqa: BLE001
                return None
            return options[0] if options else None

    def keep_guessing(self):
        try:
            return self._keep_guessing_inner()
        except BaseException:  # noqa: BLE001
            return False

    # -- guessing ----------------------------------------------------------

    def _pending_own(self):
        """The just-guessed word our board copy has not caught up with yet.

        ``None`` once the engine hands us a refreshed board (the word then
        reads as revealed), so this only corrects the one call where the copy
        is stale: ``keep_guessing``.
        """
        word = self._pending
        if word and word in self.words:
            return word
        return None

    def _options(self):
        pending = self._pending_own()
        return [w for w in self.words
                if not _is_revealed(w) and w != pending]

    def _get_answer_inner(self):
        generation = self._answer_gen
        options = self._options()
        if not options:
            return None
        ranking = self._get_ranking(options)
        for word, score in ranking:
            if word in options:
                self.guesses += 1
                self._claim_pending(word, generation)
                self._log_guess(word, score, ranking, options)
                return word
        self.guesses += 1
        word = self._fallback_pick(options)
        self._claim_pending(word, generation)
        return word

    def _claim_pending(self, word, generation):
        """Record the guess we are about to return -- unless we are a ghost.

        A pipeline abandoned at the hard wall is still running, and when its
        socket finally lets go it walks the rest of that function.  Writing
        ``_pending`` then would hide a word nobody ever guessed, which is the
        staleness bug this attribute exists to fix, arriving from the other
        direction.
        """
        if generation == self._answer_gen:
            self._pending = word

    def _log_guess(self, word, score, ranking, options):
        if self.quiet:
            return
        try:
            live = [(w, s) for w, s in ranking if w in options]
            runner_up = live[1][1] if len(live) > 1 else 0.0
            self._say("guess %d/%s: %s (score %.3f, margin over runner-up "
                      "%.3f, source=%s)"
                      % (self.guesses, self.num, word, score,
                         score - runner_up, self._ranking_source))
        except Exception:  # noqa: BLE001 - diagnostics never break a game
            pass

    # -- ranking -----------------------------------------------------------

    def _get_ranking(self, options):
        """Ranking for this clue, computed once per turn and then filtered.

        Recomputing after every correct guess would multiply the API cost for
        no signal: the clue has not changed and revealed words simply drop out
        of ``options``.
        """
        key = (_normalise(self.clue), self._is_sweep())
        if self._ranking is not None and self._ranking_key == key:
            live = [(w, s) for w, s in self._ranking if w in options]
            return live or [(word, 0.0) for word in options]

        scores, source = self._glove_scores(options)
        if source == "glove":
            # A usable cosine ranking.  Only mush earns a call.
            if self._needs_rescue(scores, options):
                rescued = self._llm_scores(options)
                if rescued:
                    scores, source = rescued, "llm"
                elif self._unrankable and self._all_mush(scores):
                    # No rescue available and the cosine ranking is mush.  A
                    # word the vocabulary cannot see is scored at the trust
                    # floor rather than at -1: it therefore beats a ranking
                    # that is *entirely* noise and loses to any real
                    # association.  For a matched pair this is right by
                    # construction -- our codemaster only issues a clue this
                    # weak when its own target is the word neither of us can
                    # price -- and the conservative gates below apply either
                    # way, so a stranger's weak clue costs at most the bonus
                    # guess we would not have taken.
                    for word in self._unrankable:
                        # Ordered among themselves by letter overlap, which is
                        # a weak signal but the only one left; the constant
                        # keeps them all above the noise and below any real
                        # association.
                        scores[word] = (self.trust_min_top
                                        + 0.02 * _overlap_similarity(self.clue,
                                                                     word))
                    source = "offline"
        else:
            # ``"oov"`` (the clue has no vector) or ``None`` (no cache at all).
            # The embedding has nothing to say, so the LLM is the primary
            # ranker here rather than a rescue, and letter overlap is the floor
            # under both.
            rescued = self._llm_scores(options)
            if rescued:
                scores, source = rescued, "llm"
            else:
                scores = dict((w, 100.0 * _overlap_similarity(self.clue, w))
                              for w in options)
                source = "offline"
                self._warn("no usable ranking for clue %r (out of vocabulary, "
                           "no LLM) -- guessing from letter overlap"
                           % (self.clue,))

        ranking = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        ranking = [(w, s) for w, s in ranking if w in options]
        if not ranking:
            ranking = [(word, 0.0) for word in options]

        self._ranking = ranking
        self._ranking_key = key
        self._ranking_source = source
        self._turn_best = ranking[0][1]
        return ranking

    def _glove_scores(self, options):
        """``({word: cosine}, "glove")``, or ``({}, "oov")``, or ``(None-ish)``.

        Three distinguishable outcomes, because they call for three different
        responses: a usable cosine ranking, a clue the vocabulary has never
        heard of (rescue), and no vector cache at all (the LLM is primary).
        """
        loaded = vectors()
        if loaded is None:
            return {}, None
        _words, matrix, index = loaded
        clue_row = index.get(_normalise(self.clue).lower())
        if clue_row is None:
            self.oov_clues += 1
            self._say("clue %r is out of the GloVe vocabulary" % (self.clue,))
            return {}, "oov"
        kept, rows = board_rows(options, index)
        if not rows:
            return {}, "oov"
        sims = matrix[rows].dot(matrix[clue_row])
        scores = dict((word, float(value)) for word, value in zip(kept, sims))
        # A board word the vocabulary lacks must stay guessable, just last --
        # and the fact that there was one is remembered, because a ranking that
        # could not see every option is a ranking on incomplete information.
        self._unrankable = [w for w in options if w not in scores]
        for word in self._unrankable:
            scores[word] = -1.0
        return scores, "glove"

    def _all_mush(self, scores):
        """Is *every* word the vocabulary can price merely noise for this clue?

        The gate on promoting unrankable words, and it is a narrower question
        than the one that earns a rescue call.  A ranking can deserve the call
        because it is incomplete while still holding a perfectly real
        association -- and in that case the word it can see is a better bet
        than the word it cannot.  Promoting there cost a recorded slang game
        (``TATTOO 2``: ``GOTH`` correctly, then the unrankable ``AUTOTUNE``,
        which was the assassin).
        """
        unrankable = set(self._unrankable)
        priceable = [s for w, s in scores.items() if w not in unrankable]
        return bool(priceable) and max(priceable) < self.trust_min_top

    def _needs_rescue(self, scores, options):
        """Is the cosine ranking mush?

        Two ways, both recorded killers: nothing on the board is a real
        association (so the "best" word is arbitrary), or the top two are
        inside noise of each other (so the ranking has no opinion, and one of
        the two can be the assassin).
        """
        if not self.llm_rescue or len(options) < 2:
            return False
        values = sorted(scores.values(), reverse=True)
        if not values:
            return True
        if self._unrankable and values[0] < self.trust_clear_top:
            self.thin_clues += 1
            self._say("ranking is incomplete: %s out of vocabulary and no "
                      "emphatic top pick (%.3f)"
                      % (", ".join(self._unrankable), values[0]))
            return True
        if values[0] < self.trust_min_top:
            self.thin_clues += 1
            self._say("cosine ranking is mush (best %.3f < %.3f)"
                      % (values[0], self.trust_min_top))
            return True
        if len(values) > 1 and (values[0] - values[1]) < self.trust_min_gap:
            self.thin_clues += 1
            self._say("cosine ranking has no opinion (top two within %.3f)"
                      % (values[0] - values[1]))
            return True
        return False

    def _llm_scores(self, options):
        """ONE call, only from ``_get_ranking``.  ``{}`` on any failure."""
        if not self.llm_rescue or not self.llm.available():
            return {}
        deadline = _Deadline(self.deadline_s)
        system = (
            "You are an expert Codenames field operative. You see the board "
            "words but not the key. Work out which words the spymaster's clue "
            "points at. Guessing the assassin loses the game instantly, so "
            "only rate a word highly when the link is real."
        )
        number = ("unlimited -- the spymaster expects you to finish off words "
                  "from earlier clues too" if self.num <= 0 else str(self.num))
        prompt = [
            "Clue: %s  (number of words: %s)" % (self.clue, number),
            "",
            "Remaining board words: %s" % ", ".join(options),
        ]
        context = self._history_context()
        if context:
            prompt.extend(["", context])
        prompt.extend([
            "",
            "Score how strongly the clue points at each word, from 100 "
            "(certain) down to 0 (unrelated). List every word you would "
            "seriously consider, most likely first.",
            "",
            'Respond with ONLY a JSON object, e.g. '
            '{"WHALE": 95, "SHIP": 70, "BEACH": 40}',
        ])
        text = self.llm.chat(system, "\n".join(prompt), max_tokens=500,
                             deadline=deadline)
        if not text:
            return {}
        scores = parse_scores(text, options)
        if not scores:
            return {}
        self.llm_calls += 1
        # Unmentioned words are still guessable, just below everything named.
        for word in options:
            scores.setdefault(word, 0.0)
        return scores

    def _history_context(self):
        lines = []
        for move in (self.move_history or [])[-12:]:
            if not move:
                continue
            actor = str(move[0])
            if actor == "%s_Codemaster" % self.team and len(move) >= 3:
                lines.append("Earlier clue from your spymaster: %s %s"
                             % (move[1], move[2]))
            elif actor == "%s_Guesser" % self.team and len(move) >= 3:
                lines.append("You guessed %s -> %s"
                             % (move[1], str(move[2]).strip("*")))
        if not lines:
            return ""
        return "Recent history:\n" + "\n".join(lines[-8:])

    def _fallback_pick(self, options):
        scored = [(_overlap_similarity(self.clue, word), word)
                  for word in options]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored[0][1]

    # -- stop rule ---------------------------------------------------------

    def _counts(self):
        """``(our words still hidden, unrevealed words remaining)``.

        A pending guess counts as found: the engine only asks us to keep
        guessing when the word it just accepted was one of ours.
        """
        marker = "*%s*" % self.team
        found = sum(1 for w in self.words if str(w) == marker)
        if self._pending_own():
            found += 1
        total = TEAM_TOTALS.get(self.team, 9)
        return max(0, total - found), len(self._options())

    def _turn_capacity(self):
        """``(guesses this clue is worth, sweep?)`` -- frozen at turn start.

        Clue number ``0`` means unlimited guesses, and some codemasters spell
        the same thing as a huge integer.  Either way the ceiling is how many
        of our own words were still hidden when the clue arrived: guessing past
        that can only hit somebody else's word.  Freezing the pair matters --
        both terms move as words are revealed, and recomputing mid-turn would
        shrink the budget under our own feet.
        """
        if self._capacity is None:
            own_left, _ = self._counts()
            own_left = max(1, own_left)
            if self.num <= 0 or self.num > own_left:
                self._capacity = (own_left, True)
            else:
                self._capacity = (max(1, self.num), False)
        return self._capacity

    def _is_sweep(self):
        return self._turn_capacity()[1]

    def _keep_guessing_inner(self):
        options = self._options()
        if not options:
            return False

        limit, sweep = self._turn_capacity()

        # Nothing of ours left to find: stop, whatever the number said.
        own_left_now, _ = self._counts()
        if own_left_now <= 0:
            return False
        # Hard cap: the clue number, plus the traditional bonus guess when the
        # number was finite.
        if self.guesses >= limit + (0 if sweep else 1):
            return False

        ranking = self._get_ranking(options)
        remaining = [(w, s) for w, s in ranking if w in options]
        if not remaining:
            return False

        best = self._turn_best or (ranking[0][1] if ranking else 0.0)
        if best <= 0:
            return False
        score = remaining[0][1]
        ratio = score / float(best)

        bonus = (not sweep) and self.guesses >= limit
        allow = self._gate(score, ratio, bonus, sweep)
        if self.verbose:
            sys.stderr.write("[Kadabra G] continue=%s score=%.3f ratio=%.2f "
                             "bonus=%s sweep=%s src=%s\n"
                             % (allow, score, ratio, bonus, sweep,
                                self._ranking_source))
        return allow

    def _gate(self, score, ratio, bonus, sweep):
        """The stop rule, per ranking source.

        The two sources live on incomparable scales -- a cosine of 0.30 is a
        strong association, a model score of 0.30 is nothing -- so each gets
        its own absolute floor, and both share a relative one.
        """
        if bonus and not self.allow_bonus:
            return False
        source = self._ranking_source
        if source == "offline":
            # Letter overlap.  Take the number, never the bonus.
            return not bonus
        if source == "llm":
            floor = LLM_BONUS_ABS if bonus else LLM_CONTINUE_ABS
            need = LLM_BONUS_RATIO if bonus else LLM_CONTINUE_RATIO
        else:
            floor = GLOVE_BONUS_ABS if bonus else GLOVE_CONTINUE_ABS
            need = GLOVE_BONUS_RATIO if bonus else GLOVE_CONTINUE_RATIO
        if sweep:
            need = max(need, self.sweep_ratio)
        return score >= floor and ratio >= need

    # -- diagnostics -------------------------------------------------------

    def usage_summary(self):
        summary = self.llm.usage_summary()
        summary.update({"rescue_calls": self.llm_calls,
                        "oov_clues": self.oov_clues,
                        "thin_clues": self.thin_clues})
        return summary
