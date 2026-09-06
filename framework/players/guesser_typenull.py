"""TYPE: NULL -- the minimalist guesser (IEEE CoG 2026 Codenames entry).

Partner to ``codemaster_typenull``.  Same null hypothesis: **one LLM call per
turn**, all the intelligence in the prompt, all the safety in deterministic
post-processing.

Design contract
---------------
1. **One call per turn.**  On the first ``get_answer`` after a clue, one
   completion ranks *every* remaining board word 0-100 for "is this one of
   ours for this clue" and, separately, names the words it would not touch at
   any price.  That ranking is cached for the rest of the turn, so guesses two
   and three cost nothing and ``keep_guessing`` costs nothing.  The champion
   samples the model repeatedly; this asks once and asks better.
2. **The stop rule is arithmetic, not a second prompt.**  ``keep_guessing``
   never calls the model.  It continues only while

       guesses < allowance   and   next >= max(ABS_FLOOR, REL_FRACTION * top)

   where ``next`` is the confidence of the word that would actually be
   guessed next, penalised if the model listed it under "avoid".  The first
   guess of a turn is mandatory (the engine takes one regardless), so the
   floor is only ever consulted from the second guess on.
3. **The board is read correctly between guesses.**  ``game.Game.run`` calls
   ``set_board`` -> ``get_answer`` -> ``_accept_guess`` -> ``keep_guessing``
   with **no** ``set_board`` in between, so our copy of the board is exactly
   one guess stale at the moment the stop rule runs.  ``_pending`` holds the
   word the engine has accepted but our copy has not seen; it is excluded from
   the options and credited to our own count.  The colour is free: the engine
   only leaves the same side to move when the word belonged to the guessing
   team, so a pending guess at ``keep_guessing`` time is always one of ours.
   (This is a recorded, fatal bug in this repo's history -- three assassin
   deaths -- and it is not being reintroduced.)
4. **num = 0 means unlimited.**  Handled: the allowance becomes
   ``SWEEP_CAP`` guesses under a raised confidence floor, so an unlimited
   clue from a stranger's codemaster cannot walk the whole board.
5. **Never crash, never hang.**  ``get_answer`` and ``keep_guessing`` catch
   ``BaseException``; the LLM path is bounded by a hard ~45 s wall enforced
   off-thread, falling through to a deterministic letter-overlap pick.

API key: ``api_key`` kwarg, else ``ANTHROPIC_API_KEY``, else the
``HARDCODED_API_KEY`` constant (ignored while it holds the checked-in
placeholder).

Environment (all optional): ``ANTHROPIC_API_KEY``, ``TYPENULL_MODEL``,
``TYPENULL_QUIET=1``, ``TYPENULL_PROVIDER=openai_compat`` (+
``TYPENULL_BASE_URL``, ``TYPENULL_COMPAT_KEY`` / ``HF_TOKEN``).

Python 3.9 compatible.  Imports nothing from the harness and nothing from its
partner codemaster.
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
    from players.guesser import Guesser
except Exception:  # pragma: no cover - keeps the file runnable standalone
    class Guesser(object):
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

MOVE_DEADLINE_S = 38.0
CALL_TIMEOUT_S = 18.0
MOVE_WALL_S = 45.0
WALL_WARNING = ("%s exceeded its %.0f s hard wall -- abandoning the LLM path "
                "and answering from the offline fallback")
FALLBACK_WARNING = "LLM call failed (%s: %s) -- using offline fallback"

#: Absolute confidence a word needs before it is worth a *non-mandatory*
#: guess.  Scores are the model's own 0-100 "this is one of ours" scale.
ABS_FLOOR = 50.0
#: ...and it must also be this fraction of the turn's best word.  A clue whose
#: second word is half as convincing as its first is a clue that bought one
#: word, whatever number the codemaster said.
REL_FRACTION = 0.60
#: Multiplier applied to a word the model explicitly listed as one it would
#: not touch.  It can still be guessed if it is the only option (the first
#: guess is mandatory), but it will essentially never clear the floor.
AVOID_PENALTY = 0.35
#: An unlimited (num == 0) clue: how many guesses it may buy, and the floor
#: each of them must clear.  A stranger's codemaster opening with 0 must not
#: be able to walk our guesser into the assassin.
SWEEP_CAP = 4
SWEEP_FLOOR = 65.0
#: Take one guess past the clue number when the leftover is overwhelming?
#: **Off.**  Across 329 recorded games in this repo the bonus guess hit one of
#: our own words 15% of the time and the assassin 6%; the first guess of a
#: turn hits ours 91%.  The minimalist trade is to not buy that lottery ticket.
ALLOW_BONUS = False
BONUS_FLOOR = 85.0

MAX_TOKEN_SCALE = 4
MIN_ANTHROPIC = (0, 60)
MIN_ANTHROPIC_TEXT = "0.60"
OLD_SDK_WARNING = ("anthropic %s is older than the tested minimum %s -- please "
                   "pip install -U anthropic; continuing with compatibility "
                   "mode")

HARDCODED_API_KEY = "TYPENULL-KEY-PLACEHOLDER"

PROVIDER_ENV = "TYPENULL_PROVIDER"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_COMPAT = "openai_compat"
BASE_URL_ENV = "TYPENULL_BASE_URL"
COMPAT_KEY_ENV = "TYPENULL_COMPAT_KEY"
COMPAT_FALLBACK_KEY_ENV = "HF_TOKEN"
DEFAULT_COMPAT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_COMPAT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

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
    try:
        print("%s %s" % (DEBUG_PREFIX, message))
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass


def _brief(reason, limit=110):
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
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def _is_revealed(board_word):
    text = str(board_word)
    return len(text) > 0 and text[0] == "*"


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
    """Offline association proxy: letter overlap + shared prefix.  Blind to
    meaning; it exists so an API-down turn still returns a legal word."""
    a, b = _normalise(clue), _normalise(board_word)
    if not a or not b:
        return 0.0
    la, lb = _letters(a), _letters(b)
    jaccard = len(la & lb) / float(len(la | lb))
    prefix = _shared_prefix(a, b) / float(max(len(a), len(b)))
    return 0.7 * jaccard + 0.3 * prefix


def _confidence_bucket(score):
    if score >= 80:
        return "high"
    if score >= 60:
        return "medium"
    if score >= 40:
        return "low"
    return "weak"


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
    """``(value, True)`` when ``func`` returned inside ``wall`` seconds, else
    ``(None, False)`` with the daemon worker abandoned where it stands."""
    if wall is None or wall <= 0:
        return func(), True
    box = {}

    def _target():
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001
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
# Minimal chat client
# ---------------------------------------------------------------------------

class _LLM(object):
    """See ``codemaster_typenull._LLM`` -- deliberately duplicated so each
    submitted file is self-contained."""

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
        self._rng = random.Random(0x4E554C4D)
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
    """Every balanced ``{...}`` span in ``text`` that parses as JSON."""
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


def _bracket_list(text):
    """The first balanced ``[...]`` span that parses as JSON."""
    if not text:
        return None
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
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:  # noqa: BLE001
                        return None
    return None


def parse_guess_reply(text, options):
    """``(scores, avoid)`` for the words in ``options``.

    Four fallbacks deep, because a guesser that cannot read its own model's
    reply plays letter overlap: the ``{"scores": {...}}`` object we asked for,
    then any object mapping board words to numbers, then a bare ordered array,
    then board words in order of first mention.
    """
    index = {}
    for word in options:
        index[_normalise(word)] = word

    scores = {}
    avoid = set()

    for data in reversed(_json_objects(text)):
        if not isinstance(data, dict):
            continue
        raw = data.get("scores")
        mapping = raw if isinstance(raw, dict) else data
        local = {}
        for key, value in mapping.items():
            word = index.get(_normalise(key))
            if word is None:
                continue
            try:
                local[word] = float(value)
            except (TypeError, ValueError):
                continue
        for entry in (data.get("avoid") or ()):
            if isinstance(entry, dict):
                entry = entry.get("word") or ""
            word = index.get(_normalise(entry))
            if word is not None:
                avoid.add(word)
        if local:
            scores = local
            break

    if scores:
        return scores, avoid

    ordered = _bracket_list(text)
    if isinstance(ordered, list):
        rank = 0
        for entry in ordered:
            if isinstance(entry, dict):
                entry = (entry.get("word") or entry.get("board_word") or "")
            word = index.get(_normalise(entry))
            if word is not None and word not in scores:
                scores[word] = float(max(1, 100 - 15 * rank))
                rank += 1
    if scores:
        return scores, avoid

    rank = 0
    for token in re.findall(r"[A-Za-z]{2,}", text or ""):
        word = index.get(_normalise(token))
        if word is not None and word not in scores:
            scores[word] = float(max(1, 100 - 15 * rank))
            rank += 1
    return scores, avoid


# ---------------------------------------------------------------------------
# The one prompt
# ---------------------------------------------------------------------------

_GUESS_SYSTEM = (
    "You are the {team} field operative in Codenames. You cannot see the "
    "secret key. Your codemaster has given you one word and one number, and "
    "you must decide which board words that clue is pointing at.\n"
    "\n"
    "Every board word is one of four things and you do not know which: one of "
    "YOURS (good), a CIVILIAN (turn over), the OPPONENT's (turn over, they "
    "gain a word), or the ASSASSIN (you lose the game on the spot). Exactly "
    "one assassin is on the board at all times.\n"
    "\n"
    "That asymmetry is the whole job. Being right about one more word wins a "
    "turn; being wrong about one word can lose the game. Confidence is what "
    "you are being asked for, not enthusiasm: a word you would bet the game "
    "on scores 90, a word that merely fits the theme scores 40, and a word "
    "you are reaching for scores 10.\n"
)

_GUESS_USER = """CLUE: {clue}
NUMBER: {number_text}

REMAINING BOARD WORDS ({n}):
{words}
{history}
Think it through briefly (under 100 words, no curly braces), then answer.

STEP 1. What is the clue's most specific sense? A codemaster picks the sense
that separates their words from everyone else's, not the most common sense.

STEP 2. Score EVERY word listed above from 0 to 100: how confident are you
that the codemaster meant that word by "{clue}"? Most words should score low.
Do not spread scores evenly -- the point of the number is that roughly
{number_text} words should stand clearly above the rest.

STEP 3. WHICH WOULD YOU NOT TOUCH? Name the words that connect to "{clue}"
loosely enough that a codemaster would have avoided them -- generic, thematic
or second-meaning links. These are where the assassin and the opponent's words
hide. A word can score moderately and still belong here.

STEP 4. Output one JSON object and nothing after it:

{{"scores": {{{example}}},
 "avoid": ["WORDS", "YOU", "WOULD", "NOT", "TOUCH"]}}

Every one of the {n} board words above must appear in "scores", spelled
exactly as printed."""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class AIGuesser(Guesser):
    """One LLM call per turn; a deterministic, arithmetic stop rule."""

    def __init__(self, team="Red", **kwargs):
        super(AIGuesser, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        self.opponent = "Blue" if self.team == "Red" else "Red"

        self.provider = _resolve_provider(kwargs.get("provider"))
        self.base_url = kwargs.get("base_url")
        self.model = _resolve_model(self.provider, kwargs.get("model"))
        self.deadline_s = float(kwargs.get("deadline", MOVE_DEADLINE_S))
        #: ``0`` runs the pipeline on the calling thread (tests).
        self.move_wall_s = float(kwargs.get("move_wall", MOVE_WALL_S))
        self.abs_floor = float(kwargs.get("abs_floor", ABS_FLOOR))
        self.rel_fraction = float(kwargs.get("rel_fraction", REL_FRACTION))
        self.avoid_penalty = float(kwargs.get("avoid_penalty", AVOID_PENALTY))
        self.sweep_cap = int(kwargs.get("sweep_cap", SWEEP_CAP))
        self.sweep_floor = float(kwargs.get("sweep_floor", SWEEP_FLOOR))
        self.allow_bonus = bool(kwargs.get("allow_bonus", ALLOW_BONUS))
        self.bonus_floor = float(kwargs.get("bonus_floor", BONUS_FLOOR))
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
        self.move_history = []
        self.clue = ""
        self.num = 0

        # -- per-turn state.  The bundled guesser_GPT never resets its count.
        self.guesses = 0
        self._scores = None
        self._avoid = set()
        self._scores_key = None
        self._source = "offline"
        #: the guess the engine has accepted but our board copy has not seen
        self._pending = None
        #: bumped per ``get_answer``; a pipeline abandoned at the wall must not
        #: later write its pick into ``_pending`` and hide a word nobody
        #: guessed.
        self._answer_gen = 0

        self.turns = 0
        self.stopped_early = 0
        self.guess_log = []

        self._announce()

    # -- diagnostics -------------------------------------------------------

    def _say(self, message):
        if not self.quiet:
            _emit("guesser(%s) %s" % (self.team, message))

    def _warn(self, message):
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
                       "the embedded HARDCODED_API_KEY) -- every guess will "
                       "come from the offline fallback" % KEY_ENV)
        if self.provider == PROVIDER_ANTHROPIC:
            try:
                import anthropic  # noqa: WPS433

                self._say("anthropic package: ok (version %s)"
                          % getattr(anthropic, "__version__", "unknown"))
            except Exception as exc:  # noqa: BLE001
                self._warn("anthropic package not importable (%s: %s) -- every "
                           "guess will come from the offline fallback"
                           % (type(exc).__name__, _brief(exc)))
            else:
                version, too_old = _anthropic_version()
                if too_old:
                    self._warn(OLD_SDK_WARNING % (version, MIN_ANTHROPIC_TEXT))

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
        if self.num < 0:
            self.num = 0
        self.turns += 1
        self.guesses = 0
        self._scores = None
        self._avoid = set()
        self._scores_key = None
        self._pending = None
        print("The clue is:", clue, num)
        return [clue, num]

    def get_answer(self):
        """Return one unrevealed board word.  Never raises."""
        self._answer_gen += 1
        try:
            word, finished = _run_bounded(self._get_answer_inner,
                                          self.move_wall_s)
            if finished:
                return word
            self._warn(WALL_WARNING % ("get_answer", self.move_wall_s))
            options = self._options()
            if not options:
                return None
            pick = self._fallback_pick(options)
            self.guesses += 1
            self._pending = pick
            self._source = "offline"
            return pick
        except BaseException:  # noqa: BLE001 - a crash is a disqualification
            try:
                options = self._options()
                if not options:
                    return None
                self.guesses += 1
                self._pending = options[0]
                return options[0]
            except BaseException:  # noqa: BLE001
                return None

    def keep_guessing(self):
        """No LLM call ever happens here.  Pure arithmetic on the cached
        confidences, read against a board that is one guess stale."""
        try:
            return self._keep_guessing_inner()
        except BaseException:  # noqa: BLE001
            return False

    # -- board state -------------------------------------------------------

    def _pending_own(self):
        """The just-guessed word our board copy has not caught up with.

        ``None`` once the engine hands us a refreshed board (the word then
        reads as revealed), so this only ever corrects the one call where the
        copy is stale: ``keep_guessing``.
        """
        word = self._pending
        if word and word in self.words:
            return word
        return None

    def _options(self):
        pending = self._pending_own()
        return [w for w in self.words
                if not _is_revealed(w) and w != pending]

    # -- guessing ----------------------------------------------------------

    def _get_answer_inner(self):
        generation = self._answer_gen
        options = self._options()
        if not options:
            return None

        ranking = self._ranking(options)
        word = ranking[0][0] if ranking else self._fallback_pick(options)
        self.guesses += 1
        self._claim_pending(word, generation)
        try:
            self._log_guess(word, ranking)
        except Exception:  # noqa: BLE001 - diagnostics never break a game
            pass
        if self.verbose:
            sys.stderr.write("[TypeNull G] %s\n" % (word,))
        return word

    def _claim_pending(self, word, generation):
        """Record the guess we are about to return -- unless we are a ghost.

        A pipeline abandoned at the hard wall is still running; when its socket
        finally lets go it walks the rest of that function.  Writing
        ``_pending`` then would hide a word from ``_options`` that nobody ever
        guessed.
        """
        if generation == self._answer_gen:
            self._pending = word

    def _ranking(self, options):
        """``[(word, effective score)]`` descending, for live options only.

        Computed once per turn.  Recomputing after every correct guess would
        double the API cost for no signal: the clue has not changed and
        revealed words simply drop out of ``options``.
        """
        scores = self._turn_scores(options)
        ranked = []
        for word in options:
            value = float(scores.get(word, 0.0))
            if word in self._avoid:
                value *= self.avoid_penalty
            ranked.append((word, value))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    def _turn_scores(self, options):
        key = _normalise(self.clue), self.num
        if self._scores is not None and self._scores_key == key:
            return self._scores

        scores = {}
        avoid = set()
        if self.llm.available():
            deadline = _Deadline(self.deadline_s)
            text = self.llm.chat(_GUESS_SYSTEM.format(team=self.team),
                                 self._guess_user(options),
                                 max_tokens=1100, deadline=deadline)
            if self.verbose:
                sys.stderr.write("[TypeNull G] %r\n" % (text,))
            if text:
                scores, avoid = parse_guess_reply(text, options)

        if scores:
            self._source = "llm"
        else:
            self._source = "offline"
            self._warn("no usable LLM ranking for clue %r -- guessing from the "
                       "offline letter-overlap fallback" % (self.clue,))
            scores = dict((word, 100.0 * _similarity(self.clue, word))
                          for word in options)
            avoid = set()

        self._scores = scores
        self._avoid = avoid
        self._scores_key = key
        return scores

    def _guess_user(self, options):
        example = ", ".join('"%s": 0' % word for word in options[:3])
        return _GUESS_USER.format(
            clue=self.clue or "(none)",
            number_text=("unlimited" if self.num == 0 else str(self.num)),
            n=len(options),
            words=", ".join(options),
            example=example,
            history=self._history_block(),
        )

    def _history_block(self):
        """What earlier clues this game bought, so a leftover association is
        available without a second call.  Read from the engine's own log."""
        lines = []
        for entry in self.move_history or ():
            try:
                if entry[0] == self.team + "_Codemaster":
                    lines.append("  clue %s %s" % (entry[1], entry[2]))
                elif entry[0] == self.team + "_Guesser":
                    lines.append("    -> we touched %s (%s)"
                                 % (entry[1], entry[2]))
            except Exception:  # noqa: BLE001
                continue
        # The engine appends this turn's clue to the history *before* handing
        # it to us, so the trailing clue line is the one already at the top of
        # the prompt.  Repeating it there would just be noise.
        while lines and lines[-1].startswith("  clue "):
            lines.pop()
        if not lines:
            return ""
        return ("\nEARLIER THIS GAME (your codemaster may still be leaning on "
                "an old clue):\n" + "\n".join(lines) + "\n")

    def _fallback_pick(self, options):
        best = None
        best_score = None
        for word in options:
            score = _similarity(self.clue, word)
            if best_score is None or score > best_score:
                best, best_score = word, score
        return best or options[0]

    # -- the stop rule -----------------------------------------------------

    def _allowance(self):
        """How many guesses this turn may take, before confidence is read.

        ``num == 0`` is the framework's "unlimited"; it is capped rather than
        honoured, because a stranger's codemaster opening with 0 must not be
        able to walk us into the assassin.
        """
        if self.num == 0:
            return self.sweep_cap
        base = max(1, self.num)
        return base + 1 if self.allow_bonus else base

    def _floor(self):
        if self.num == 0:
            return self.sweep_floor
        if self.guesses >= max(1, self.num):
            return self.bonus_floor      # only reachable when allow_bonus
        return self.abs_floor

    def _keep_guessing_inner(self):
        if self.guesses >= self._allowance():
            return False
        options = self._options()
        if not options:
            return False
        if self._scores is None:
            # No cached confidences (the wall fired, or the API is down).  One
            # word per clue is the safe reading of a signal we do not have.
            return False

        ranked = self._ranking(options)
        if not ranked:
            return False
        word, value = ranked[0]

        # The relative test is anchored on the turn's *original* best, not on
        # the best surviving option: once the top pick has been guessed the
        # leader is whatever is left, and comparing it against itself makes
        # the whole rule vacuous.  That is exactly how this repo's recorded
        # stale-board bug produced a confidence ratio of 1.0 on every turn.
        floor = max(self._floor(), self.rel_fraction * self._turn_best())

        go = value >= floor and word not in self._avoid
        if not go:
            self.stopped_early += 1
        self._say("keep_guessing=%s after %d/%s | next=%s conf=%.0f "
                  "floor=%.0f%s"
                  % (go, self.guesses,
                     self.num if self.num > 0 else "unlimited",
                     word, value, floor,
                     " (on the avoid list)" if word in self._avoid else ""))
        return bool(go)

    def _turn_best(self):
        """The highest confidence this turn, avoid-penalty included, over the
        board as it looked when the ranking was built."""
        if not self._scores:
            return 0.0
        best = 0.0
        for word, value in self._scores.items():
            value = float(value)
            if word in self._avoid:
                value *= self.avoid_penalty
            if value > best:
                best = value
        return best

    # -- diagnostics -------------------------------------------------------

    def _log_guess(self, word, ranking):
        value = dict(ranking).get(word, 0.0)
        runner = ranking[1][1] if len(ranking) > 1 else 0.0
        margin = 0.0
        if value > 0:
            margin = max(0.0, (value - runner) / float(value))
        self.guess_log.append({"clue": self.clue, "num": self.num,
                               "word": word, "score": value,
                               "source": self._source})
        self._say("guess %d/%s: %s (conf %.0f, %s, margin %.2f) | clue=%s %s "
                  "| ranking=%s"
                  % (self.guesses, self.num if self.num > 0 else "unlimited",
                     word, value, _confidence_bucket(value), margin,
                     self.clue, self.num, self._source))

    # -- accounting --------------------------------------------------------

    def usage_summary(self):
        summary = self.llm.usage_summary()
        summary["agent"] = AGENT_VERSION
        summary["turns"] = self.turns
        summary["stopped_early"] = self.stopped_early
        summary["allow_bonus"] = self.allow_bonus
        if self.guess_log:
            summary["guess_log"] = list(self.guess_log)
        return summary
