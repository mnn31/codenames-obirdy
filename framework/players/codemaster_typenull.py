"""TYPE: NULL -- the minimalist codemaster (IEEE CoG 2026 Codenames entry).

The null hypothesis for the oBirdy line: *maybe the pipeline is not earning its
cost*.  ``codemaster_obirdy`` spends three to five LLM calls on a single clue
(brainstorm, a panel of simulated guessers, a danger probe) plus an offline
embedding sensor.  This agent spends **exactly one** -- two only when the first
reply comes back unusable -- and puts every ounce of the difference into the
prompt.

Design contract
---------------
1. **One call per turn.**  ``_CLUE_SYSTEM`` + ``_clue_user`` ask the model to do
   the whole pipeline inside one completion: group its own words, propose a
   clue, then *simulate a key-blind teammate ranking the entire board* and
   self-check the assassin against that ranking.  The panel and the probe are
   not deleted, they are folded into the prompt.
2. **All safety is deterministic.**  Nothing the model says is trusted:

   * legality is re-derived here (``clue_is_legal``) -- single alphabetic
     English word, >= 3 letters, no sub-word derivation in either direction
     against any unrevealed board word.  Same rule the bundled
     ``codemaster_GPT`` polices itself with and the arena audits;
   * targets are intersected with our own unrevealed words, so a clue cannot
     buy a word that is not ours;
   * the clue **number is computed, never taken**: it is the length of the
     leading run of our own words in the model's own predicted ranking, capped
     by the claimed count, by the surviving target count and by
     ``MAX_CLUE_NUMBER``.  A model that predicts its own first guess is not
     ours has vetoed itself;
   * the assassin appearing anywhere in the top ``ASSASSIN_TOP_K`` of that
     ranking is an outright veto;
   * a clue already given this game is rejected (read back from
     ``get_move_history``, so it survives our own state being reset).
3. **One retry, then determinism.**  An invalid reply is re-asked once with the
   specific reason appended.  A second failure falls through to
   ``_fallback_clue``: a pure function of the board over a bundled vocabulary,
   always legal, never repeating.
4. **Never crash, never hang.**  ``get_clue`` catches ``BaseException`` and is
   bounded by a hard ~45 s wall (``MOVE_WALL_S``) enforced off-thread, because
   a socket read cannot be interrupted cooperatively.

What is deliberately *absent* (this is the experiment): candidate generation
over subsets, panel sampling, the danger probe, the bundled GloVe similarity
table, race awareness, endgame sweeps.  If this ties the champion, that
machinery is not paying for itself.

API key: ``api_key`` kwarg, else ``ANTHROPIC_API_KEY``, else the
``HARDCODED_API_KEY`` constant (ignored while it holds the checked-in
placeholder, which does not start with ``sk-``).

Environment (all optional): ``ANTHROPIC_API_KEY``, ``TYPENULL_MODEL``,
``TYPENULL_QUIET=1``.  ``TYPENULL_PROVIDER=openai_compat`` swaps the inline
client for a plain OpenAI-shaped HTTP endpoint (``TYPENULL_BASE_URL``, key from
``TYPENULL_COMPAT_KEY`` or ``HF_TOKEN``) for cheap plumbing runs; the default
and the competition configuration remain Anthropic.

Python 3.9 compatible.  Imports nothing from the harness and nothing from its
partner guesser.
"""

import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

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

AGENT_VERSION = "type-null build 2026-08-07"

DEFAULT_MODEL = "claude-sonnet-5"
MODEL_ENV = "TYPENULL_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"
QUIET_ENV = "TYPENULL_QUIET"
DEBUG_PREFIX = "[TypeNull]"

#: Cooperative budget for one ``get_clue``.
MOVE_DEADLINE_S = 38.0
#: Per-attempt HTTP timeout.
CALL_TIMEOUT_S = 18.0
#: Non-cooperative ceiling.  Past this the LLM path is abandoned where it
#: stands and the deterministic fallback answers instead, so no move can
#: exceed roughly this figure however badly the network behaves.  The event's
#: soft limit is 60 s.
MOVE_WALL_S = 45.0
WALL_WARNING = ("get_clue exceeded its %.0f s hard wall -- abandoning the LLM "
                "path and clueing from the offline fallback")
FALLBACK_WARNING = "LLM call failed (%s: %s) -- using offline fallback"

#: Ceiling on a clue number.  More words per clue is mechanically more
#: assassin exposure per turn; the champion's own measured mean is ~1.4 and
#: every attempt in this repo to push it higher bought turns and paid in
#: deaths.
MAX_CLUE_NUMBER = 3
#: The assassin anywhere this deep in the model's own predicted teammate
#: ranking kills the clue outright.
ASSASSIN_TOP_K = 3
#: Shortest clue we will issue.  Two-letter "words" are almost always a
#: parsing artefact rather than a clue, and the judges review clue spirit.
MIN_LEGAL_CLUE_LEN = 3

MAX_TOKEN_SCALE = 4
MIN_ANTHROPIC = (0, 60)
MIN_ANTHROPIC_TEXT = "0.60"
OLD_SDK_WARNING = ("anthropic %s is older than the tested minimum %s -- please "
                   "pip install -U anthropic; continuing with compatibility "
                   "mode")

#: The tournament key, substituted into this line when a submission zip is cut.
#: What lives in git is the placeholder, and the agent ignores it because it
#: does not start with ``sk-``.
HARDCODED_API_KEY = "TYPENULL-KEY-PLACEHOLDER"

PROVIDER_ENV = "TYPENULL_PROVIDER"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_COMPAT = "openai_compat"
BASE_URL_ENV = "TYPENULL_BASE_URL"
COMPAT_KEY_ENV = "TYPENULL_COMPAT_KEY"
COMPAT_FALLBACK_KEY_ENV = "HF_TOKEN"
DEFAULT_COMPAT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_COMPAT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

_ALPHA_RE = re.compile(r"^[A-Z]+$")
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _truthy(value):
    return str(value if value is not None else "").strip().lower() in (
        "1", "true", "yes", "on")


def _quiet_default():
    return _truthy(os.environ.get(QUIET_ENV))


def _emit(message):
    """One prefixed diagnostics line.  Never raises, whatever the console."""
    try:
        print("%s %s" % (DEBUG_PREFIX, message))
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass


def _brief(reason, limit=110):
    """One-line, length-capped exception text -- no payloads, no keys."""
    text = " ".join(str(reason if reason is not None else "").split())
    text = _KEY_PATTERN.sub("sk-<redacted>", text)
    return text[:limit] if len(text) > limit else text


def _version_tuple(text):
    parts = []
    for chunk in str(text or "").split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _anthropic_version():
    try:
        import anthropic  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return None, False
    text = str(getattr(anthropic, "__version__", "") or "unknown")
    parsed = _version_tuple(text)
    return text, bool(parsed) and parsed < MIN_ANTHROPIC


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


def _resolve_provider(value=None):
    name = str(value or os.environ.get(PROVIDER_ENV) or "").strip().lower()
    return PROVIDER_COMPAT if name == PROVIDER_COMPAT else PROVIDER_ANTHROPIC


def _resolve_model(provider, override=None):
    if override:
        return override
    from_env = os.environ.get(MODEL_ENV)
    if from_env:
        return from_env
    return (DEFAULT_COMPAT_MODEL if provider == PROVIDER_COMPAT
            else DEFAULT_MODEL)


def _normalise(word):
    """Uppercase, letters only (board words may carry punctuation)."""
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def _is_revealed(board_word):
    text = str(board_word)
    return len(text) > 0 and text[0] == "*"


def clue_is_legal(clue, board_words):
    """Competition legality: single alphabetic word, no sub-word derivation.

    Mirrors the rule inside the bundled ``codemaster_GPT`` and the independent
    audit ``harness/arena.py`` performs, so a clue that passes here can never
    be counted illegal downstream.
    """
    if not clue:
        return False
    text = str(clue).strip().upper()
    if len(text.split()) != 1:
        return False
    if not _ALPHA_RE.match(text):
        return False
    if len(text) < MIN_LEGAL_CLUE_LEN:
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


def _letters(word):
    return set(_normalise(word))


def _shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _similarity(clue, board_word):
    """Cheap offline association proxy: letter overlap + shared prefix.

    Blind to meaning.  It exists so the API-down path still produces a legal,
    non-degenerate clue, not because it is any good.
    """
    a, b = _normalise(clue), _normalise(board_word)
    if not a or not b:
        return 0.0
    la, lb = _letters(a), _letters(b)
    jaccard = len(la & lb) / float(len(la | lb))
    prefix = _shared_prefix(a, b) / float(max(len(a), len(b)))
    return 0.7 * jaccard + 0.3 * prefix


# ---------------------------------------------------------------------------
# Wall-clock budget
# ---------------------------------------------------------------------------

class _Deadline(object):
    """Elapsed time is the larger of two clocks, so neither a suspended
    machine nor a backwards NTP step can hide time from the budget."""

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

    ``(None, False)`` otherwise, and the worker is abandoned where it is.
    Python cannot interrupt a thread blocked in a socket read, so the only
    honest way to bound such a call is to stop waiting for it.  The thread is a
    daemon, so a wedged connection cannot hold the process open either.  An
    exception inside ``func`` is re-raised in the caller.
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
# Minimal chat client: retries, backoff, timeouts, never raises
# ---------------------------------------------------------------------------

class _LLM(object):
    """One-completion chat wrapper over Anthropic (default) or an
    OpenAI-shaped HTTP endpoint.  ``chat`` returns ``None`` rather than
    raising, so every caller can fall back deterministically."""

    def __init__(self, model, max_retries=1, timeout=CALL_TIMEOUT_S,
                 provider=None, base_url=None, api_key=None, warn=None):
        self.provider = _resolve_provider(provider)
        self.model = model
        self.api_key = api_key
        self.base_url = str(base_url
                            or os.environ.get(BASE_URL_ENV)
                            or DEFAULT_COMPAT_BASE_URL)
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
        self._rng = random.Random(0x4E554C4C)
        self.calls = 0
        self.failures = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def _param_shapes():
        # Most -> least featureful.  Extended thinking is disabled
        # deliberately: newer models enable it by default and would spend the
        # whole token budget on a reasoning block, returning no text at all.
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
            import anthropic  # noqa: WPS433
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
        """Announce a fallback.  Quiet mode does not cover these."""
        if self.warn is None:
            return
        name, reason = self.last_error or ("NoResponse",
                                           "no usable reply from the model")
        try:
            self.warn(FALLBACK_WARNING % (name, _brief(reason)))
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
        if self.provider == PROVIDER_COMPAT:
            return "compat env var" if self._compat_key() else None
        return _resolve_api_key(self.api_key)[1]

    def _compat_key(self):
        return (os.environ.get(COMPAT_KEY_ENV)
                or os.environ.get(COMPAT_FALLBACK_KEY_ENV))

    def available(self):
        if self.provider == PROVIDER_COMPAT:
            return bool(self._compat_key())
        return self._get_client() is not None

    def chat(self, system, user, max_tokens=900, deadline=None):
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

            if self.provider == PROVIDER_COMPAT:
                text, retryable = self._call_compat(system, user, max_tokens,
                                                    per_call)
            else:
                text, retryable = self._call_anthropic(system, user,
                                                       max_tokens, per_call)
            if text is not None:
                return text
            if not retryable or attempt >= self.max_retries:
                self.failures += 1
                self._report()
                return None

            attempt += 1
            self.retries += 1
            delay = min(4.0, 0.6 * (2 ** (attempt - 1))) * (
                0.5 + self._rng.random())
            if deadline is not None:
                delay = min(delay, max(0.0, deadline.remaining() - 1.5))
            if delay > 0:
                time.sleep(delay)

        self.failures += 1
        self._report()
        return None

    def _shape_kwargs(self, fields):
        """Modern SDKs take these as named parameters; older ones raise
        ``TypeError`` before a request is made, so the identical fields go
        through ``extra_body`` and land verbatim in the request JSON."""
        if not fields:
            return {}
        if self._extra_body:
            return {"extra_body": dict(fields)}
        return dict(fields)

    def _empty_reply(self, response, budget):
        stop = getattr(response, "stop_reason", None)
        self.last_error = (
            "EmptyResponse",
            "no text block in the reply (stop_reason=%s, budget %d tokens)"
            % (stop, budget))
        if stop == "max_tokens" and self._token_scale < MAX_TOKEN_SCALE:
            self._token_scale = min(MAX_TOKEN_SCALE, self._token_scale * 4)
            return True
        return False

    def _call_anthropic(self, system, user, max_tokens, per_call):
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

    def _call_compat(self, system, user, max_tokens, per_call):
        key = self._compat_key()
        if not key:
            self.last_error = ("MissingAPIKey",
                               "no %s / %s" % (COMPAT_KEY_ENV,
                                               COMPAT_FALLBACK_KEY_ENV))
            return None, False
        payload = {
            "model": self.model,
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer %s" % key,
                     "Content-Type": "application/json",
                     "Accept": "application/json"},
            method="POST")
        try:
            handle = urllib.request.urlopen(request, timeout=per_call)
            try:
                body = handle.read()
            finally:
                try:
                    handle.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            blob = ("%s %s" % (type(exc).__name__, exc)).lower()
            code = getattr(exc, "code", None)
            retryable = (self._retryable(blob)
                         or code in (408, 409, 425, 429, 500, 502, 503, 504,
                                     529))
            self.last_error = (type(exc).__name__, _brief(exc))
            return None, retryable

        try:
            data = json.loads(body.decode("utf-8", "replace"))
            message = (data.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text") or "")
                    elif isinstance(block, str):
                        parts.append(block)
                content = "".join(parts)
            if not isinstance(content, str):
                return None, False
            usage = data.get("usage") or {}
            self.input_tokens += int(usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or 0)
        except Exception as exc:  # noqa: BLE001
            self.last_error = (type(exc).__name__, _brief(exc))
            return None, False
        self.calls += 1
        return content, False

    @staticmethod
    def _retryable(blob):
        markers = ("timeout", "timed out", "rate limit", "429", "overloaded",
                   "500", "502", "503", "529", "connection", "temporarily")
        return any(marker in blob for marker in markers)

    def usage_summary(self):
        return {"model": self.model, "provider": self.provider,
                "calls": self.calls, "retries": self.retries,
                "failures": self.failures,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

def _json_objects(text):
    """Every balanced ``{...}`` span in ``text`` that parses as JSON.

    The prompt asks for prose reasoning then one JSON object, but models
    sometimes brace something in the prose or wrap the answer in a fence.
    Scanning for *all* candidates and letting the caller pick the last usable
    one is more robust than committing to the first ``{``.
    """
    found = []
    if not text:
        return found
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
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
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        found.append(json.loads(text[start:i + 1]))
                    except Exception:  # noqa: BLE001
                        pass
                    start = -1
    return found


def _as_word_list(value):
    """Coerce whatever the model put in a list field into board-word tokens."""
    out = []
    if isinstance(value, str):
        value = re.split(r"[,\n]", value)
    if not isinstance(value, (list, tuple)):
        return out
    for entry in value:
        if isinstance(entry, dict):
            entry = (entry.get("word") or entry.get("board_word")
                     or entry.get("name") or "")
        token = _normalise(entry)
        if token:
            out.append(token)
    return out


def parse_clue_reply(text):
    """``dict`` of the fields we care about, or ``None``.

    Deliberately permissive about *shape* and completely unforgiving about
    *content*: every value it returns is re-validated against the board by the
    caller.
    """
    for data in reversed(_json_objects(text)):
        if not isinstance(data, dict):
            continue
        clue = data.get("clue")
        if clue is None:
            continue
        parsed = {
            "clue": _normalise(clue),
            "targets": _as_word_list(data.get("targets")),
            "ranking": _as_word_list(data.get("ranking")),
            "avoid": _as_word_list(data.get("avoid")),
        }
        try:
            parsed["number"] = int(data.get("number"))
        except (TypeError, ValueError):
            parsed["number"] = len(parsed["targets"]) or 1
        if parsed["clue"]:
            return parsed
    return None


# ---------------------------------------------------------------------------
# The one prompt
# ---------------------------------------------------------------------------

_CLUE_SYSTEM = (
    "You are the {team} Codemaster in Codenames. You see the secret key; your "
    "teammate does not.\n"
    "Your teammate hears only ONE English word and ONE number, then touches "
    "that many board words. They touch words in the order those words feel "
    "most connected to your clue.\n"
    "\n"
    "Outcomes per touched word: your colour -> you keep going; a civilian -> "
    "turn over; the opponent's colour -> turn over and they gain a word; the "
    "ASSASSIN -> you lose the game instantly.\n"
    "\n"
    "So the only question that matters is not 'is my clue clever' but 'what "
    "will a key-blind teammate actually touch first, second, third'. A clue "
    "that pulls three of your words but pulls the assassin harder is a losing "
    "clue.\n"
    "\n"
    "Clue legality, enforced by the judges: exactly one English word, letters "
    "only, no hyphen, no proper-noun tricks, no digits, no invented "
    "portmanteaus, and it may not contain any board word nor be contained in "
    "one (TIME is illegal next to TIMES; SEAHORSE is illegal next to HORSE). "
    "It must be a word an ordinary English speaker knows, because a teammate "
    "who has never heard the word cannot decode it.\n"
)

_CLUE_USER = """BOARD (only unrevealed words are listed)

YOUR WORDS ({own_n}) -- these are what you want touched:
{own}

OPPONENT WORDS ({opp_n}) -- touching one of these hands them a word and ends your turn:
{opp}

CIVILIANS ({civ_n}) -- touching one ends your turn:
{civ}

ASSASSIN -- if your teammate touches this word you LOSE IMMEDIATELY:
{assassin}
{leftovers}{banned}
Work through these steps in order. Keep the prose under 120 words and use no
curly braces anywhere except in the final JSON object.

STEP 1. Find the two or three tightest groups among YOUR WORDS. A group is only
real if one ordinary word connects every member more strongly than it connects
anything else on the board.

STEP 2. Pick your best candidate clue for the best group. Prefer a common,
concrete word. Reject anything that is a rare word, a compound you invented, a
proper noun, or that shares a stem with a board word.

STEP 3. SIMULATE YOUR TEAMMATE. They cannot see colours. Rank the SIX words on
the whole board -- yours, opponent, civilian and the assassin all together --
that they would reach for first on hearing your clue, most likely first. Be
honest and adversarial here: this ranking is the only defence the team has.

STEP 4. SELF-CHECK. Look at your ranking.
  (a) If "{assassin}" appears anywhere in the top {assassin_k}, the clue is
      dead. Throw it away and return to STEP 2 with a different clue.
  (b) Count how many words at the very TOP of the ranking, before the first
      non-{team} word, are yours. That count is the largest honest number. If
      it is zero, the clue is dead -- return to STEP 2.
  (c) Never claim a number bigger than that count, however many words you
      wanted the clue to cover.

STEP 5. Output the final answer as one JSON object and nothing after it:

{{"clue": "WORD",
 "targets": ["THE", "BOARD", "WORDS", "IT", "BUYS"],
 "ranking": ["SIX", "BOARD", "WORDS", "TEAMMATE", "REACHES", "FOR"],
 "avoid": ["dangerous board words this clue brushes against"],
 "number": N}}

"ranking" must contain board words only, spelled exactly as printed above.
"number" must be at most {max_number} and at most the length of "targets"."""


#: A deterministic vocabulary for the API-down path.  Chosen to be common,
#: concrete and unlikely to be a board word.
_FALLBACK_VOCAB = (
    "ANIMAL", "BALANCE", "BRIDGE", "CAPTAIN", "CARGO", "CLIMATE", "COMFORT",
    "COUNTRY", "CULTURE", "DANGER", "DESERT", "DEVICE", "DINNER", "DRIVER",
    "EMPIRE", "ENERGY", "ESCAPE", "FABRIC", "FACTORY", "FARMER", "FESTIVAL",
    "FOSSIL", "FRIEND", "FUTURE", "GADGET", "GARDEN", "GLACIER", "HARVEST",
    "HAZARD", "HELMET", "HISTORY", "HOLIDAY", "HUNTER", "ISLAND", "JOURNEY",
    "JUNGLE", "KITCHEN", "LANTERN", "LEGEND", "LIBRARY", "LIQUID", "MAGNET",
    "MARKET", "MEADOW", "MEMORY", "MERCHANT", "MESSAGE", "MINERAL", "MIRROR",
    "MONSTER", "MORNING", "MOUNTAIN", "MUSEUM", "MYSTERY", "NATURE", "OFFICE",
    "PACKAGE", "PARADE", "PATTERN", "PICNIC", "PILLOW", "PIONEER", "POCKET",
    "POWDER", "PRISON", "PUZZLE", "QUARRY", "RAINBOW", "RECIPE", "RESCUE",
    "RHYTHM", "RIDDLE", "SAFARI", "SEASON", "SECRET", "SHELTER", "SIGNAL",
    "SILENCE", "SKETCH", "SPEECH", "STORAGE", "STRANGER", "SUMMIT", "SURFACE",
    "SYMBOL", "TALENT", "TAVERN", "TEMPEST", "THUNDER", "TICKET", "TRADITION",
    "TRAFFIC", "TREASURE", "TROPHY", "TUNNEL", "VALLEY", "VICTORY", "VILLAGE",
    "VOYAGE", "WEALTH", "WHISPER", "WILDLIFE", "WINTER", "WIZARD", "WONDER",
    "WORKSHOP",
)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class AICodemaster(Codemaster):
    """One LLM call per clue; everything else is deterministic."""

    def __init__(self, team="Red", **kwargs):
        super(AICodemaster, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        self.opponent = "Blue" if self.team == "Red" else "Red"

        self.provider = _resolve_provider(kwargs.get("provider"))
        self.base_url = kwargs.get("base_url")
        self.model = _resolve_model(self.provider, kwargs.get("model"))
        self.deadline_s = float(kwargs.get("deadline", MOVE_DEADLINE_S))
        #: ``0`` runs the pipeline on the calling thread (tests).
        self.move_wall_s = float(kwargs.get("move_wall", MOVE_WALL_S))
        self.max_number = int(kwargs.get("max_number", MAX_CLUE_NUMBER))
        self.assassin_top_k = int(kwargs.get("assassin_top_k", ASSASSIN_TOP_K))
        #: One semantic retry.  ``0`` makes the agent strictly single-call.
        self.retries = int(kwargs.get("clue_retries", 1))
        self.quiet = bool(kwargs.get("quiet", _quiet_default()))
        self.verbose = bool(kwargs.get("verbose", False))

        self.llm = _LLM(self.model,
                        max_retries=int(kwargs.get("max_retries", 1)),
                        timeout=float(kwargs.get("call_timeout",
                                                 CALL_TIMEOUT_S)),
                        provider=self.provider,
                        base_url=self.base_url,
                        api_key=kwargs.get("api_key"),
                        warn=self._warn)

        self.words = []
        self.maps = []
        self.move_history = []

        #: Clues this agent issued, so it can never repeat one.  Rebuilt from
        #: ``move_history`` on every turn as well, so it survives state loss.
        self._given = []
        self.turns = 0
        self.rejected = 0
        self.assassin_vetoes = 0
        self.number_capped = 0
        self.fallbacks = 0
        self.retried = 0
        self.clue_log = []

        self._announce()

    # -- diagnostics -------------------------------------------------------

    def _say(self, message):
        if not self.quiet:
            _emit("codemaster(%s) %s" % (self.team, message))

    def _warn(self, message):
        """Printed even in quiet mode -- a silent fallback is indistinguishable
        from a bad agent, which is exactly how a whole tournament run was once
        played on the offline path without anyone noticing."""
        _emit("WARNING: %s" % message)

    def _announce(self):
        self._say("init: version=%s model=%s provider=%s wall=%.0fs"
                  % (AGENT_VERSION, self.model, self.provider,
                     self.move_wall_s))
        source = self.llm.key_source()
        if source:
            self._say("api key: found via %s (value never printed)" % source)
        else:
            self._warn("no API key found (checked the api_key kwarg, %s, and "
                       "the embedded HARDCODED_API_KEY) -- every clue will "
                       "come from the offline fallback" % KEY_ENV)
        if self.provider == PROVIDER_ANTHROPIC:
            try:
                import anthropic  # noqa: WPS433

                self._say("anthropic package: ok (version %s)"
                          % getattr(anthropic, "__version__", "unknown"))
            except Exception as exc:  # noqa: BLE001
                self._warn("anthropic package not importable (%s: %s) -- every "
                           "clue will come from the offline fallback"
                           % (type(exc).__name__, _brief(exc)))
            else:
                version, too_old = _anthropic_version()
                if too_old:
                    self._warn(OLD_SDK_WARNING % (version, MIN_ANTHROPIC_TEXT))

    # -- framework hooks ---------------------------------------------------

    def set_game_state(self, words, maps):
        self.words = list(words)
        self.maps = list(maps)

    def set_move_history(self, move_history):
        self.move_history = list(move_history or [])

    def get_clue(self):
        """Return ``[clue, number]``.  Never raises, never illegal."""
        self.turns += 1
        try:
            result, finished = _run_bounded(self._get_clue_inner,
                                            self.move_wall_s)
            if finished and result:
                return result
            if not finished:
                self._warn(WALL_WARNING % self.move_wall_s)
            return self._fallback_clue()
        except BaseException:  # noqa: BLE001 - a crash is a disqualification
            try:
                return self._fallback_clue()
            except BaseException:  # noqa: BLE001
                return ["SIGNAL", 1]

    # -- the pipeline ------------------------------------------------------

    def _get_clue_inner(self):
        own, opp, civ, assassin = self._split_board()
        if not own:
            return self._fallback_clue()

        deadline = _Deadline(self.deadline_s)
        if not self.llm.available():
            self.fallbacks += 1
            return self._fallback_clue()

        system = _CLUE_SYSTEM.format(team=self.team)
        feedback = ""
        attempts = max(1, self.retries + 1)
        last_reason = "no reply"
        for attempt in range(attempts):
            if deadline.expired(reserve=2.0):
                break
            user = self._clue_user(own, opp, civ, assassin) + feedback
            text = self.llm.chat(system, user, max_tokens=900,
                                 deadline=deadline)
            if self.verbose:
                sys.stderr.write("[TypeNull CM] %r\n" % (text,))
            parsed = parse_clue_reply(text)
            if parsed is None:
                last_reason = "the reply contained no usable JSON object"
            else:
                result, last_reason = self._validate(parsed, own, assassin)
                if result is not None:
                    self._record(result[0])
                    self._log(result, parsed, attempt)
                    return result
            self.rejected += 1
            if attempt + 1 < attempts:
                self.retried += 1
                feedback = ("\n\nYour previous answer was rejected: %s. "
                            "Give a different clue that fixes exactly that, "
                            "in the same JSON format." % last_reason)

        self.fallbacks += 1
        self._warn("no valid clue from the model (%s) -- using the offline "
                   "fallback" % last_reason)
        return self._fallback_clue()

    def _clue_user(self, own, opp, civ, assassin):
        return _CLUE_USER.format(
            own=", ".join(own) or "(none)", own_n=len(own),
            opp=", ".join(opp) or "(none)", opp_n=len(opp),
            civ=", ".join(civ) or "(none)", civ_n=len(civ),
            assassin=assassin or "(none)",
            assassin_k=self.assassin_top_k,
            team=self.team,
            max_number=self.max_number,
            leftovers=self._leftover_block(own),
            banned=self._banned_block(),
        )

    def _leftover_block(self, own):
        """Own words earlier clues targeted and the guesser never found.

        Read back from the move history rather than from our own state, so it
        is correct even after a restart, and so it costs no extra call: the
        model is simply told which of its words are already half-clued.
        """
        left = [w for w in own if w in self._clued_words()]
        if not left:
            return ""
        return ("\nAlready hinted at by an earlier clue this game (your "
                "teammate may still connect them): %s\n" % ", ".join(left))

    def _banned_block(self):
        given = self._issued_clues()
        if not given:
            return ""
        return ("\nClues already used this game -- you may NOT repeat any of "
                "these: %s\n" % ", ".join(sorted(given)))

    # -- deterministic validation -----------------------------------------

    def _validate(self, parsed, own, assassin):
        """``(result, reason)``.  ``result`` is ``[clue, number]`` or ``None``.

        Nothing the model said is trusted here.  The number in particular is
        *computed* from the model's own predicted teammate ranking, not taken
        from its "number" field: the field is only an upper bound.
        """
        clue = parsed["clue"]
        board = self.words or []

        if not clue_is_legal(clue, board):
            return None, ("%r is not a legal clue -- it must be one English "
                          "word of at least %d letters and must not contain "
                          "or be contained in any board word"
                          % (clue, MIN_LEGAL_CLUE_LEN))
        if clue in self._issued_clues():
            return None, "%r was already used as a clue this game" % clue

        # The model echoes board words back as bare tokens; a board word may
        # carry punctuation or a space (the competition pool is secret and may
        # be slangier than the bundled one), so everything is matched through
        # the same normalisation the legality check uses.
        index = dict((_normalise(w), w) for w in self._unrevealed())
        own_set = set(own)

        def resolve(token):
            return index.get(token)

        targets = []
        for token in parsed["targets"]:
            word = resolve(token)
            if word in own_set and word not in targets:
                targets.append(word)
        if not targets:
            return None, ("none of the words you listed in \"targets\" is one "
                          "of your own unrevealed words")

        ranking = [resolve(token) for token in parsed["ranking"]]
        ranking = [word for word in ranking if word is not None]

        # (a) hard assassin veto on the model's own simulation
        if assassin and assassin in ranking[:self.assassin_top_k]:
            self.assassin_vetoes += 1
            return None, ("your own ranking puts the assassin %r in the top "
                          "%d, which loses the game outright"
                          % (assassin, self.assassin_top_k))

        # (b) the number is the leading run of our own words in that ranking
        claimed = max(1, int(parsed["number"] or 1))
        if ranking:
            safe = 0
            for word in ranking:
                if word in own_set:
                    safe += 1
                else:
                    break
            if safe == 0:
                return None, ("your own ranking says your teammate's FIRST "
                              "pick (%r) is not one of your words" % ranking[0])
        else:
            # No usable simulation came back.  Refuse to guess wide.
            safe = 1

        number = max(1, min(claimed, len(targets), safe, self.max_number))
        if number < claimed:
            self.number_capped += 1
        #: what the clue is actually buying, as board words -- the leftover
        #: hint in later prompts reads this back.
        parsed["resolved_targets"] = targets
        return [clue, number], ""

    # -- board reading -----------------------------------------------------

    def _split_board(self):
        own, opp, civ, assassin = [], [], [], ""
        for word, kind in zip(self.words, self.maps):
            if _is_revealed(word):
                continue
            if kind == self.team:
                own.append(word)
            elif kind == self.opponent:
                opp.append(word)
            elif kind == "Assassin":
                assassin = word
            else:
                civ.append(word)
        return own, opp, civ, assassin

    def _unrevealed(self):
        return [w for w in self.words if not _is_revealed(w)]

    def _issued_clues(self):
        """Our own clues this game, from the move history plus our own log.

        The history is authoritative (it survives our state being reset); the
        local list covers the turn currently being played.
        """
        seen = set(self._given)
        prefix = self.team + "_Codemaster"
        for entry in self.move_history or ():
            try:
                if entry[0] == prefix:
                    token = _normalise(entry[1])
                    if token:
                        seen.add(token)
            except Exception:  # noqa: BLE001
                continue
        return seen

    def _clued_words(self):
        """Own words an earlier clue this game claimed, still unrevealed.

        Taken from our own clue log, because the framework's move history
        records the clue but not what it was aiming at.  It is only a prompt
        hint -- the model is told which of its words are already half-clued so
        it can reuse that association instead of paying for it twice -- so
        losing it after a restart costs nothing.
        """
        clued = set()
        for entry in self.clue_log:
            for word in entry.get("targets") or ():
                clued.add(word)
        return clued

    def _record(self, clue):
        token = _normalise(clue)
        if token and token not in self._given:
            self._given.append(token)

    def _log(self, result, parsed, attempt):
        targets = parsed.get("resolved_targets") or parsed.get("targets") or []
        entry = {"clue": result[0], "number": result[1],
                 "claimed": parsed.get("number"),
                 "targets": list(targets),
                 "attempt": attempt + 1}
        self.clue_log.append(entry)
        self._say("clue=%s %s (claimed %s, targets %s, attempt %d, "
                  "calls=%d) | ranking=%s"
                  % (result[0], result[1], parsed.get("number"),
                     ",".join(targets) or "-",
                     attempt + 1, self.llm.calls,
                     ",".join(parsed.get("ranking") or ()) or "-"))

    # -- the offline path --------------------------------------------------

    def _fallback_clue(self, allow_repeat=False):
        """Deterministic, API-free, always-legal, never-repeating clue.

        Letter overlap is blind to meaning; this exists so a dead API still
        produces a legal move, not because it plays well.  Because it is a
        pure function of the board, the repeat filter is what stops a dead API
        turning the whole game into the same clue over and over.
        """
        own, opp, civ, assassin = self._split_board()
        board = self.words or []
        issued = set() if allow_repeat else self._issued_clues()

        scored = []
        for index, clue in enumerate(_FALLBACK_VOCAB):
            if clue in issued or not clue_is_legal(clue, board):
                continue
            total = 0.0
            for word in self._unrevealed():
                sim = _similarity(clue, word)
                if word in own:
                    total += sim
                elif word == assassin:
                    total -= 4.0 * sim
                elif word in opp:
                    total -= 0.7 * sim
                else:
                    total -= 0.3 * sim
            scored.append((total, -index, clue))

        if not scored:
            for candidate in ("SIGNAL", "OBJECT", "SUBJECT", "TOPIC", "IDEA",
                              "THING", "MATTER"):
                if candidate not in issued and clue_is_legal(candidate, board):
                    self._record(candidate)
                    return [candidate, 1]
            if not allow_repeat:
                return self._fallback_clue(allow_repeat=True)
            return ["SIGNAL", 1]

        scored.sort(reverse=True)
        clue = scored[0][2]
        self._record(clue)
        self._say("clue=%s 1 (offline fallback)" % clue)
        return [clue, 1]

    # -- accounting --------------------------------------------------------

    def usage_summary(self):
        summary = self.llm.usage_summary()
        summary["agent"] = AGENT_VERSION
        summary["turns"] = self.turns
        summary["rejected"] = self.rejected
        summary["retried"] = self.retried
        summary["assassin_vetoes"] = self.assassin_vetoes
        summary["number_capped"] = self.number_capped
        summary["fallbacks"] = self.fallbacks
        if self.clue_log:
            summary["clue_log"] = list(self.clue_log)
        return summary
