"""Lycanroc guesser -- IEEE CoG 2026 Codenames AI Competition entry.

**A behaviourally unchanged fork of ``guesser_obirdy``.**  Lycanroc is a
codemaster experiment: it attacks the champion's clue *pace* at the candidate-
generation stage and leaves every other stage -- scorer, panel, danger probe,
embedding sensor, and this guesser -- exactly as the champion shipped them.
The only edits in this file are the two identity strings below, so a mixed
tournament log can tell which build spoke.  Diffing this against
``guesser_obirdy.py`` should show nothing else.

Self-contained by design: nothing is imported from the evaluation harness or
from the partner codemaster.  Client, retry/backoff, deadline budgeting,
parsing and the offline fallback are all inlined.

Pipeline for one clue
---------------------
1. ``set_clue`` resets per-turn state.  (The bundled ``guesser_GPT`` never
   resets its guess counter between turns and silently stops guessing for the
   rest of the game once the cumulative count passes a clue number; that bug
   is deliberately not reproduced here.)
2. On the first ``get_answer`` of a turn, several sampled LLM calls score the
   remaining board words against the clue.  Word order is shuffled per sample
   to defeat position bias; scores are averaged and blended with a rank-based
   Borda term.
3. ``get_answer`` returns the strongest word, validated to be an exact,
   unrevealed board word.  A deterministic similarity fallback covers parse
   failures and a dead API, so a legal word is always returned.
4. ``keep_guessing`` applies a dynamic stop rule: the mandatory first guess is
   free, further guesses continue while the next candidate stays confident
   relative to the turn's best, and the bonus (+1) guess is only taken when an
   earlier, unexhausted clue still has a strong leftover association.

A note on the engine's call order, because the stop rule depends on it.
``game.Game.run`` calls ``set_board`` (which hands us a **copy**), then
``get_answer``, then reveals the guessed word *in its own list*, then calls
``keep_guessing`` -- with no ``set_board`` in between.  Our copy is therefore
one guess stale exactly when the stop rule runs, so the word we just guessed
still looks unrevealed and would be re-offered as "the next candidate", making
every confidence ratio 1.0.  ``_pending`` closes that gap.  It can also be
*coloured* for free: ``_accept_guess`` only leaves the same side to move when
the guessed word belonged to the guessing team, so a pending guess at
``keep_guessing`` time is always one of our own words.

Clue number ``0`` means *unlimited guesses* (framework README), and some
codemasters spell the same thing as a huge integer.  Both are handled as a
"sweep" turn: earlier clues' rankings are blended in at a discount (the
convention our own codemaster's sweep gate assumes), the confidence bar is
raised, and the ceiling is however many of our own words are genuinely still
hidden -- guessing past that can only hit somebody else's word.

Risk is adapted to the track: single-team games (no opposing moves in the
history) treat a wrong pick as a lost turn, two-team games treat it as a lost
turn *and* a gift.  The assassin is feared in both.

Nothing here reads the framework's ``results/`` log files: those may contain
other entrants' games and the organisers treat reading them as leakage.  All
state is derived from the board, the clue and ``get_move_history``, and no
state is carried across games (agents are re-instantiated per game anyway).

API key: the ``api_key`` kwarg, else ``ANTHROPIC_API_KEY``, else the
``HARDCODED_API_KEY`` constant below (which the packaging script fills in, and
which is ignored while it still holds the checked-in placeholder).  No
environment variable has to be set for tournament play.

Environment (all optional): ``ANTHROPIC_API_KEY``, ``OBIRDY_MODEL`` (model
override), ``OBIRDY_QUIET=1`` (silence the diagnostics, warnings excepted).
``OBIRDY_PROVIDER=openai_compat`` swaps the inline
client for a plain OpenAI-shaped HTTP endpoint (``OBIRDY_BASE_URL``, key from
``OBIRDY_COMPAT_KEY`` or ``HF_TOKEN``) for cheap plumbing runs; the default and
the competition configuration remain Anthropic.

Python 3.9 compatible.
"""

import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

try:  # pragma: no cover - trivial
    from concurrent.futures import ThreadPoolExecutor
except ImportError:  # pragma: no cover
    ThreadPoolExecutor = None

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

#: Version of record; printed at init so a tournament log says which build ran.
AGENT_VERSION = "lycanroc build 2026-08-07 (guesser: mega-pidgeot, unchanged)"

DEFAULT_MODEL = "claude-sonnet-5"
MODEL_ENV = "OBIRDY_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"

# -- SDK floor --------------------------------------------------------------
#
# See the codemaster for the full post-mortem; duplicated so each submission
# file stands alone.  Short version: ``anthropic==0.29.0`` has no ``thinking``
# parameter on ``Messages.create``, the sampling-shape probe read that
# ``TypeError`` as the model refusing the field and dropped it, extended
# thinking stayed on by default, and the model then spent the whole
# ``max_tokens`` budget on a reasoning block and returned no text block at all.
# ``_call_anthropic`` now re-sends the same fields through ``extra_body``
# (present in every release back to 0.29, copied verbatim into the request
# JSON), and this floor says so out loud at startup.
MIN_ANTHROPIC = (0, 60)
MIN_ANTHROPIC_TEXT = "0.60"
OLD_SDK_WARNING = ("anthropic %s is older than the tested minimum %s -- please "
                   "pip install -U anthropic; continuing with compatibility "
                   "mode")
#: Ceiling on how far an empty reply may widen the token budget.
MAX_TOKEN_SCALE = 4


def _version_tuple(text):
    """``"0.29.0" -> (0, 29, 0)``, stopping at the first non-numeric part."""
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
    """``(version string, is it below the tested floor?)``.

    ``(None, False)`` when the package is not importable -- that case has its
    own, louder warning.
    """
    try:
        import anthropic  # noqa: WPS433 - optional dependency, imported late
    except Exception:  # noqa: BLE001
        return None, False
    text = str(getattr(anthropic, "__version__", "") or "unknown")
    parsed = _version_tuple(text)
    return text, bool(parsed) and parsed < MIN_ANTHROPIC


# -- diagnostics ------------------------------------------------------------
#
# See the codemaster for the rationale; duplicated so each submission file
# stands alone.  ``print`` because the organisers' harness captures stdout, one
# line each, ASCII only, and a fallback is never silent.
DEBUG_PREFIX = "[Lycanroc]"
QUIET_ENV = "OBIRDY_QUIET"
FALLBACK_WARNING = "LLM call failed (%s: %s) -- using offline fallback"


def _truthy(value):
    return str(value if value is not None else "").strip().lower() in (
        "1", "true", "yes", "on")


def _quiet_default():
    """Verbose by default: the organisers asked to be able to see the run."""
    return _truthy(os.environ.get(QUIET_ENV))


def _emit(message):
    """One prefixed diagnostics line.  Never raises, whatever the console."""
    try:
        print("%s %s" % (DEBUG_PREFIX, message))
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 - diagnostics must never break a game
        pass


#: Belt and braces: an SDK that ever echoed the credential back inside an error
#: message must not get it printed into a shared tournament log.
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")


def _brief(reason, limit=110):
    """One-line, length-capped exception text -- no payloads, no keys."""
    text = " ".join(str(reason if reason is not None else "").split())
    text = _KEY_PATTERN.sub("sk-<redacted>", text)
    return text[:limit] if len(text) > limit else text


def _confidence_bucket(margin):
    """Coarse label for how far clear the pick is of the runner-up."""
    if margin >= 0.20:
        return "high"
    if margin >= 0.07:
        return "medium"
    return "low"


#: The tournament key, substituted into this line by
#: ``harness/package_submission.py`` when the submission zip is cut.  What
#: lives in git is the placeholder; a real key is never committed.
#:
#: It has to travel inside the file because the organisers run every entry from
#: one environment and will not set a per-team variable -- two teams asking for
#: ``ANTHROPIC_API_KEY`` would clash.  Only a value that *looks* like a real key
#: is used (see ``_key_looks_real``), so the checked-in placeholder can never be
#: mistaken for one: with it in place the agent behaves exactly as it does with
#: no key at all.
HARDCODED_API_KEY = "OBIRDY-KEY-PLACEHOLDER"


def _key_looks_real(value):
    """Anthropic keys start with ``sk-``; a placeholder never does."""
    return bool(value) and str(value).strip().startswith("sk-")


def _resolve_api_key(explicit=None):
    """``(key, source)`` for the Anthropic client.

    Order: constructor kwarg, then ``ANTHROPIC_API_KEY`` (our own development
    workflow), then the embedded constant.  ``source`` is a short human label
    for the diagnostics banner -- **never the key or any part of it**.
    """
    if explicit:
        return str(explicit), "constructor kwarg"
    from_env = os.environ.get(KEY_ENV)
    if from_env:
        return from_env, "%s env var" % KEY_ENV
    if _key_looks_real(HARDCODED_API_KEY):
        return HARDCODED_API_KEY, "embedded key"
    return None, None


#: Provider switch; see the codemaster for the rationale.  ``anthropic`` is the
#: default and the competition configuration; ``openai_compat`` speaks plain
#: OpenAI chat-completions over stdlib HTTP so plumbing can be exercised on a
#: free router without spending competition credits or adding a dependency.
PROVIDER_ENV = "OBIRDY_PROVIDER"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_COMPAT = "openai_compat"
BASE_URL_ENV = "OBIRDY_BASE_URL"
COMPAT_KEY_ENV = "OBIRDY_COMPAT_KEY"
COMPAT_FALLBACK_KEY_ENV = "HF_TOKEN"
DEFAULT_COMPAT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_COMPAT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def _resolve_provider(value=None):
    """Anything we do not recognise falls back to the competition provider."""
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


MOVE_DEADLINE_S = 45.0
CALL_TIMEOUT_S = 22.0

#: Absolute ceiling on one move, whatever the pipeline is doing.
#:
#: ``MOVE_DEADLINE_S`` is the budget every stage *degrades* against, and that is
#: cooperative: a stage only notices the clock when it is next asked.  A single
#: HTTP call wedged in a socket read is never asked, so the deadline can be
#: consulted correctly at every check and still be blown by hundreds of seconds
#: between two of them.  This is the non-cooperative half.  Past this wall the
#: LLM path is abandoned where it stands and the deterministic offline answer is
#: given instead, so no move can exceed roughly this figure however badly the
#: network, a proxy or the machine behaves.  The event's soft limit is 60 s and
#: the pipeline's own budget is 45 s, which leaves the wall 5 s of headroom on
#: one side and 10 s on the other.
MOVE_WALL_S = 50.0
WALL_WARNING = ("%s exceeded its %.0f s hard wall -- abandoning the LLM path "
                "and answering from the offline fallback")

#: Total words per team on the standard 5x5 board (red moves first).
TEAM_TOTALS = {"Red": 9, "Blue": 8}

#: Relative-confidence thresholds (candidate score / turn-best score).
CONTINUE_BASE_SOLO = 0.45
CONTINUE_BASE_DUEL = 0.52
BONUS_THRESHOLD = 0.80

#: Confidence floor for an unlimited ("sweep") turn.  A clue of 0 invites
#: over-reach: the guesser has no number to stop at, and the words it is meant
#: to mop up were clued turns ago, so the blended leftover signal has to be
#: close to the turn's best before another guess is worth the risk.
SWEEP_MIN_CONFIDENCE = 0.90


# ---------------------------------------------------------------------------
# Deterministic offline fallback association table
# ---------------------------------------------------------------------------

_FALLBACK_ASSOC = {
    "CREATURE": ("BEAR", "LION", "HORSE", "MOUSE", "WHALE", "DUCK", "EAGLE",
                 "SHARK", "RABBIT", "PENGUIN", "OCTOPUS", "KANGAROO", "DOG",
                 "CAT", "BUFFALO", "SCORPION", "DRAGON", "HAWK", "ROBIN",
                 "WORM", "SEAL", "CRANE", "BAT", "FISH"),
    "MYTH": ("ANGEL", "GHOST", "GIANT", "DWARF", "UNICORN", "CENTAUR",
             "PHOENIX", "DRAGON", "WITCH", "ATLANTIS"),
    "NATION": ("AFRICA", "AMERICA", "AUSTRALIA", "CANADA", "CHINA", "EGYPT",
               "ENGLAND", "EUROPE", "FRANCE", "GERMANY", "GREECE", "INDIA",
               "MEXICO", "TURKEY", "ANTARCTICA"),
    "PLANET": ("MERCURY", "JUPITER", "SATURN", "MOON", "STAR", "SPACE",
               "SATELLITE", "TELESCOPE", "ALIEN", "ORBIT"),
    "VEHICLE": ("CAR", "TRAIN", "PLANE", "JET", "SHIP", "VAN", "LIMOUSINE",
                "AMBULANCE", "HELICOPTER", "ENGINE", "TRACK"),
    "WEAPON": ("BOMB", "MISSILE", "PISTOL", "KNIFE", "SPIKE", "WHIP", "BOW",
               "TORCH", "FIGHTER", "SOLDIER", "WAR", "STRIKE"),
    "MUSIC": ("BAND", "PIANO", "FLUTE", "BUGLE", "HORN", "OPERA", "CONCERT",
              "NOTE", "CONDUCTOR", "STRING", "ORGAN", "SOUND"),
    "SPORT": ("BALL", "COURT", "FIELD", "RACKET", "CRICKET", "PITCH", "BAT",
              "STADIUM", "GAME", "PLAY", "SWING", "MATCH"),
    "MEAL": ("APPLE", "CARROT", "CHOCOLATE", "HONEY", "KETCHUP", "LEMON",
             "OLIVE", "ORANGE", "PIE", "PUMPKIN", "BERRY", "NUT", "HAM"),
    "CLOTHES": ("BELT", "BOOT", "CAP", "CLOAK", "DRESS", "GLOVE", "HOOD",
                "PANTS", "SHOE", "SOCK", "SUIT", "TIE", "COTTON"),
    "TOOL": ("DRILL", "FORK", "HOOK", "NAIL", "NEEDLE", "PIN", "BRUSH",
             "SCALE", "PIPE", "LOCK", "KEY", "SWITCH", "PLATE"),
    "JOB": ("AGENT", "DOCTOR", "LAWYER", "NURSE", "PILOT", "TEACHER", "COOK",
            "SPY", "SCIENTIST", "PIRATE", "NINJA", "THIEF", "POLICE"),
    "ROYAL": ("KING", "QUEEN", "PRINCESS", "KNIGHT", "CROWN", "COURT",
              "RULER", "CASTLE", "TEMPLE"),
    "BUILDING": ("CHURCH", "HOSPITAL", "HOTEL", "SCHOOL", "THEATER", "TOWER",
                 "SKYSCRAPER", "EMBASSY", "BANK", "SHOP", "LAB", "PYRAMID",
                 "BRIDGE", "WALL", "MINE"),
    "METAL": ("COPPER", "GOLD", "IRON", "LEAD", "MARBLE", "DIAMOND", "IVORY",
              "ROCK", "MINE", "BOLT", "CHAIN"),
    "WEATHER": ("ICE", "SNOW", "WIND", "COLD", "WAVE", "STREAM", "AIR",
                "FIRE", "WATER", "SNOWMAN", "CLIFF", "BEACH"),
    "PLANT": ("GRASS", "MAPLE", "ROSE", "ROOT", "FOREST", "TRUNK", "STRAW",
              "LOG", "BARK", "SPINE", "PALM"),
    "BODY": ("ARM", "EYE", "FACE", "FOOT", "HAND", "HEAD", "HEART", "MOUTH",
             "THUMB", "TOOTH", "SPINE", "CHEST", "TAIL"),
    "PAPERWORK": ("BILL", "CARD", "CHECK", "CONTRACT", "FILE", "MAIL", "NOTE",
                  "PAPER", "POST", "PRESS", "DRAFT", "CODE", "NOVEL"),
    "GAMBLE": ("CASINO", "DICE", "ROULETTE", "LUCK", "CHANCE", "CLUB", "DECK",
               "JACK", "POOL", "BET", "STOCK"),
    "MACHINE": ("ROBOT", "SERVER", "SCREEN", "TABLET", "BATTERY", "LASER",
                "MICROSCOPE", "VACUUM", "WASHER", "ENGINE", "MODEL"),
    "MOTION": ("DANCE", "FALL", "FLY", "ROW", "SWING", "TRIP", "CYCLE",
               "DROP", "SLIP", "WAKE", "PASS", "CAST"),
    "SHAPE": ("CIRCLE", "SQUARE", "TRIANGLE", "ROUND", "LINE", "POINT",
              "FIGURE", "BOX", "RING", "CROSS", "BLOCK"),
    "DARK": ("SHADOW", "NIGHT", "DEATH", "POISON", "DISEASE", "GHOST",
             "SOUL", "PIT", "HOLE", "GRAVE"),
    "LIGHT": ("DAY", "SUN", "TORCH", "FIRE", "STAR", "GLASS", "MOON",
              "SPOT", "RAY", "GLOW"),
    "MONEY": ("BANK", "BILL", "GOLD", "POUND", "MILLIONAIRE", "STOCK",
              "CHANGE", "CHARGE", "MINT", "BUCK"),
    "OCEAN": ("BEACH", "SHIP", "WHALE", "SHARK", "SEAL", "PORT", "SCUBA",
              "WAVE", "FISH", "BERMUDA", "ATLANTIS", "OCTOPUS"),
    "CLASSROOM": ("PUPIL", "TEACHER", "BOARD", "DESK", "CHAIR", "TABLE",
                  "DEGREE", "SCHOOL", "STAFF", "CLASS"),
}


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _normalise(word):
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def _is_revealed(board_word):
    text = str(board_word)
    return len(text) > 0 and text[0] == "*"


def _letters(word):
    return set(ch for ch in word if ch.isalpha())


def _shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _similarity(clue, board_word):
    """Cheap deterministic association proxy in roughly [0, 2]."""
    word = _normalise(board_word)
    clue = _normalise(clue)
    if not word or not clue:
        return 0.0
    score = 0.0
    if word in _FALLBACK_ASSOC.get(clue, ()):
        score += 1.0
    a, b = _letters(clue), _letters(word)
    union = a | b
    if union:
        score += 0.6 * (len(a & b) / float(len(union)))
    score += 0.15 * min(_shared_prefix(clue, word), 3)
    score += 0.001 * (len(word) % 7)
    return score


# ---------------------------------------------------------------------------
# Wall-clock budget
# ---------------------------------------------------------------------------

class _Deadline(object):
    """A per-move budget the whole pipeline degrades against.

    Elapsed time is the *larger* of what the two clocks say.  Wall clock is the
    measure the organisers' 60 s limit is read in, and it is the one that counts
    a suspended machine against us; the monotonic clock is the one a backwards
    NTP step cannot rewind.  Taking the max means neither can hide elapsed time
    from the budget.
    """

    def __init__(self, budget):
        self.budget = float(budget)
        self.start = time.time()
        self._mono_start = time.monotonic()
        self.end = self.start + self.budget

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
    is.  Python cannot interrupt a thread blocked in a socket read, so the only
    honest way to bound a call that may never return is to stop waiting for it:
    the thread is a daemon, so a wedged connection cannot hold the process open
    either, and its only remaining effect is bookkeeping the caller has already
    stopped reading.

    An exception inside ``func`` is re-raised in the caller, so wrapping a
    pipeline in this changes nothing about how its failures are handled.
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
    """See the codemaster for the rationale; kept duplicated so each
    submission file stands alone."""

    def __init__(self, model, max_retries=2, timeout=CALL_TIMEOUT_S,
                 provider=None, base_url=None, api_key=None, warn=None):
        self.provider = _resolve_provider(provider)
        self.model = model
        self.api_key = api_key
        self.base_url = str(base_url
                            or os.environ.get(BASE_URL_ENV)
                            or DEFAULT_COMPAT_BASE_URL)
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        #: called with one already-formatted warning line; see ``_report``
        self.warn = warn
        #: ``(exception class name, brief reason)`` of the last failed attempt
        self.last_error = None
        self._warned_unavailable = False
        self._client = None
        self._client_failed = False
        self._extra = None
        #: True once this SDK build has been shown to lack the named
        #: parameters, so they travel in ``extra_body`` instead.
        self._extra_body = False
        #: Multiplier on ``max_tokens``, raised once if a reply comes back with
        #: no text because the budget ran out.
        self._token_scale = 1
        self._rng = random.Random(0xB18D)
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
        except Exception as exc:
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
            # ``timeout`` here is belt and braces.  Every call site already
            # passes a per-attempt ``timeout``, but the SDK's own default is
            # 600 s, so any path that ever forgot one would inherit a ten-minute
            # ceiling instead of ours.  Setting it on the client means the
            # floor is ``CALL_TIMEOUT_S`` even then.
            self._client = anthropic.Anthropic(api_key=api_key, max_retries=0,
                                               timeout=self.timeout)
        except TypeError:
            # An SDK too old for the constructor keyword; per-call timeouts
            # still apply.
            try:
                self._client = anthropic.Anthropic(api_key=api_key,
                                                   max_retries=0)
            except Exception as exc:
                self.last_error = (type(exc).__name__, _brief(exc))
                self._client_failed = True
                return None
        except Exception as exc:
            self.last_error = (type(exc).__name__, _brief(exc))
            self._client_failed = True
            return None
        return self._client

    # -- warnings ----------------------------------------------------------

    def _report(self):
        """Announce a fallback.  Always -- quiet mode does not cover these."""
        if self.warn is None:
            return
        name, reason = self.last_error or ("NoResponse",
                                           "no usable reply from the model")
        try:
            self.warn(FALLBACK_WARNING % (name, _brief(reason)))
        except Exception:  # noqa: BLE001
            pass

    def _report_unavailable(self):
        """The no-client case, reported once rather than on every call."""
        if self._warned_unavailable:
            return
        self._warned_unavailable = True
        if self.last_error is None:
            self.last_error = ("MissingAPIKey", "no API key available")
        self._report()

    def key_source(self):
        """Where a key would come from, as a label.  Never the key itself."""
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

            if self.provider == PROVIDER_COMPAT:
                text, retryable = self._call_compat(system, user, max_tokens,
                                                    per_call)
            else:
                text, retryable = self._call_anthropic(system, user, max_tokens,
                                                       per_call)
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
        """How this SDK build has to be handed the shape's fields.

        Modern releases take them as named parameters.  Older ones have no such
        parameter and raise ``TypeError`` before a request is ever made, so the
        identical fields go through ``extra_body`` and land verbatim in the
        request JSON.  Same wire format, same server behaviour.
        """
        if not fields:
            return {}
        if self._extra_body:
            return {"extra_body": dict(fields)}
        return dict(fields)

    def _empty_reply(self, response, budget):
        """Record a text-less reply and say whether a retry could fix it.

        Returning ``""`` as though it were an answer is how the 0.29 regression
        stayed invisible: an SDK that cannot disable extended thinking spends
        the entire ``max_tokens`` budget on a reasoning block and emits no text
        block.  That is a failure; when the budget is what ran out, widening it
        once is worth one retry.
        """
        stop = getattr(response, "stop_reason", None)
        self.last_error = (
            "EmptyResponse",
            "no text block in the reply (stop_reason=%s, budget %d tokens) -- "
            "the request may be spending its whole budget on reasoning"
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
            except Exception as exc:
                blob = ("%s %s" % (type(exc).__name__, exc)).lower()
                if (fields and not self._extra_body
                        and isinstance(exc, TypeError)
                        and "unexpected keyword argument" in blob):
                    # The installed SDK has no such Python parameter.  That is
                    # a client-side gap, not the model declining the field, so
                    # re-send the same request with the fields in ``extra_body``
                    # instead of dropping what the model actually needs.
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
                except Exception:
                    pass
        except Exception as exc:
            blob = ("%s %s" % (type(exc).__name__, exc)).lower()
            code = getattr(exc, "code", None)
            retryable = (self._retryable(blob)
                         or code in (408, 409, 425, 429, 500, 502, 503, 504, 529))
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
        except Exception as exc:
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
                "calls": self.calls,
                "retries": self.retries, "failures": self.failures,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


def _run_parallel(jobs, deadline, max_workers=4):
    if not jobs:
        return []
    if ThreadPoolExecutor is None or len(jobs) == 1:
        results = []
        for job in jobs:
            try:
                results.append(job())
            except Exception:
                results.append(None)
        return results
    results = [None] * len(jobs)
    workers = max(1, min(max_workers, len(jobs)))
    # Deliberately *not* ``with ThreadPoolExecutor(...)``.  The context manager
    # exits through ``shutdown(wait=True)``, which joins every worker -- so a
    # job wedged in a socket read blows straight through the deadline that
    # ``future.result(timeout=...)`` just honoured, and the whole point of the
    # timeout is lost at the closing brace.  Abandon instead: give up on the
    # future, shut the pool down without waiting, and let the stragglers finish
    # into results nobody reads.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(job): i for i, job in enumerate(jobs)}
        for future, index in futures.items():
            try:
                timeout = None
                if deadline is not None:
                    timeout = max(0.1, deadline.remaining())
                results[index] = future.result(timeout=timeout)
            except BaseException:  # noqa: BLE001 - a dead job is just ``None``
                results[index] = None
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:      # Python < 3.9 has no ``cancel_futures``
            pool.shutdown(wait=False)
    return results


# ---------------------------------------------------------------------------
# Parsing
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
                except Exception:
                    return None
    return None


def _parse_scores(text, options):
    """``{board word: score}`` from a scoring reply; tolerant of chattiness."""
    index = {}
    for word in options:
        index[_normalise(word)] = word

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

    # Fallback: an ordered JSON array, or bare words in order of preference.
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
    """Sampled-ranking guesser with a dynamic, risk-aware stop rule."""

    def __init__(self, team="Red", **kwargs):
        super(AIGuesser, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        self.opponent = "Blue" if self.team == "Red" else "Red"

        self.provider = _resolve_provider(kwargs.get("provider"))
        self.base_url = kwargs.get("base_url")
        self.model = _resolve_model(self.provider, kwargs.get("model"))
        self.deadline_s = float(kwargs.get("deadline", MOVE_DEADLINE_S))
        #: The non-cooperative backstop under ``deadline_s``; see
        #: ``MOVE_WALL_S``.  ``0`` disables it (tests that want the
        #: pipeline to run on the calling thread).
        self.move_wall_s = float(kwargs.get("move_wall", MOVE_WALL_S))
        self.samples = int(kwargs.get("samples", 2))
        self.top_k = int(kwargs.get("top_k", 8))
        self.allow_bonus = bool(kwargs.get("allow_bonus", True))
        self.verbose = bool(kwargs.get("verbose", False))
        #: Diagnostics are ON unless asked otherwise -- the organisers wanted to
        #: be able to see what the agent is doing.  ``quiet`` (kwarg) or
        #: ``OBIRDY_QUIET=1`` silences everything except fallback warnings.
        self.quiet = bool(kwargs.get("quiet", _quiet_default()))
        #: How much of an earlier clue's strength carries into a sweep turn.
        #: At 1.0 a word that was the clear top pick of an earlier clue enters
        #: the sweep at full strength, which is what the confidence floor below
        #: is calibrated against; discounting it as well would put every
        #: leftover under the floor and make sweeps unable to do their job.
        self.leftover_weight = float(kwargs.get("leftover_weight", 1.0))
        #: Confidence floor applied on an unlimited-guess turn.
        self.sweep_confidence = float(
            kwargs.get("sweep_confidence", SWEEP_MIN_CONFIDENCE))

        self.llm = _LLM(self.model,
                        max_retries=int(kwargs.get("max_retries", 2)),
                        timeout=float(kwargs.get("call_timeout", CALL_TIMEOUT_S)),
                        provider=self.provider,
                        base_url=self.base_url,
                        api_key=kwargs.get("api_key"),
                        warn=self._warn)

        self.words = []
        self.move_history = []
        self.clue = ""
        self.num = 0

        # Per-turn state.  The bundled guesser_GPT forgets to reset this.
        self.guesses = 0
        self._ranking = None          # [(word, score)] descending
        self._ranking_key = None      # (clue, sweep?) the ranking was built on
        self._turn_best = 0.0
        self._capacity = None         # (guess budget, sweep?) frozen per turn
        #: the guess the engine has accepted but our board copy has not seen
        self._pending = None
        #: bumped once per ``get_answer``.  A pipeline abandoned at the hard
        #: wall may still be running when the next move starts; this is how its
        #: late writes are told apart from the live move's.
        self._answer_gen = 0

        #: clue -> ranking, so leftovers from earlier turns cost no extra calls
        self._clue_rankings = {}
        #: clue -> number, to know which earlier clues were never exhausted
        self._clue_numbers = {}
        #: "llm" or "offline" -- where this turn's ranking came from
        self._ranking_source = "offline"

        self._announce()

    # -- diagnostics -------------------------------------------------------

    def _say(self, message):
        """A diagnostics line, suppressed by quiet mode."""
        if not self.quiet:
            _emit("guesser(%s) %s" % (self.team, message))

    def _warn(self, message):
        """A warning.  Printed even in quiet mode -- that is the whole point."""
        _emit("WARNING: %s" % message)

    def _announce(self):
        """Say what this process actually loaded, before a game depends on it.

        Three facts, because these are the three ways a submission silently
        turns into a much worse agent: wrong build, wrong model, no key.  A
        missing key is a warning rather than a note: it is the difference
        between our agent and a letter-overlap heuristic.
        """
        self._say("init: version=%s model=%s provider=%s samples=%d"
                  % (AGENT_VERSION, self.model, self.provider, self.samples))

        source = self.llm.key_source()
        if source:
            self._say("api key: found via %s (value never printed)" % source)
        else:
            self._warn("no API key found (checked the api_key kwarg, %s, and "
                       "the embedded HARDCODED_API_KEY) -- every guess will "
                       "come from the offline fallback" % KEY_ENV)

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

    def _log_guess(self, word, ranking):
        """One line per guess: the pick, how clear it was, and from where.

        The margin is over the runner-up rather than over the turn's best: on
        the mandatory first guess every ratio to the best is 1.0 by
        construction, which says nothing at all.
        """
        if self.quiet:
            return
        live = [(w, s) for w, s in ranking if w in self._options() or w == word]
        picked = dict(live).get(word, 0.0)
        runner = live[1][1] if len(live) > 1 else 0.0
        margin = 0.0
        if picked > 0:
            margin = max(0.0, (picked - runner) / float(picked))
        self._say("guess %d/%s: %s (confidence %s, margin %.2f) | clue=%s %s "
                  "| ranking=%s"
                  % (self.guesses, self.num if self.num > 0 else "unlimited",
                     word, _confidence_bucket(margin), margin, self.clue,
                     self.num, self._ranking_source))

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
        self._turn_best = 0.0
        self._capacity = None
        self._pending = None
        if self.clue:
            self._clue_numbers[_normalise(self.clue)] = self.num
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
            # Abandoned mid-flight: answer from the offline ranking, which is
            # letter overlap but is instant and always legal.  The generation
            # counter is what stops the abandoned worker from later writing its
            # own pick into ``_pending`` and hiding a word we never guessed.
            self._warn(WALL_WARNING % ("get_answer", self.move_wall_s))
            options = self._options()
            if not options:
                return None
            pick = self._fallback_pick(options)
            self._pending = pick
            self._ranking = None
            self._ranking_source = "offline"
            return pick
        except BaseException:  # noqa: BLE001 - a crash is a disqualification
            options = self._options()
            if not options:
                return None
            return options[0]

    def keep_guessing(self):
        try:
            return self._keep_guessing_inner()
        except BaseException:  # noqa: BLE001
            return False

    # -- guessing ----------------------------------------------------------

    def _pending_own(self):
        """The just-guessed word our board copy has not caught up with yet.

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
                if self.verbose:
                    sys.stderr.write("[oBirdy G] %s (%.1f)\n" % (word, score))
                try:
                    self._log_guess(word, ranking)
                except Exception:  # noqa: BLE001 - diagnostics never break a game
                    pass
                return word

        self.guesses += 1
        word = self._fallback_pick(options)
        self._claim_pending(word, generation)
        try:
            self._log_guess(word, [(word, 1.0)])
        except Exception:  # noqa: BLE001
            pass
        return word

    def _claim_pending(self, word, generation):
        """Record the guess we are about to return -- unless we are a ghost.

        A pipeline abandoned at the hard wall is still running, and when its
        socket finally lets go it walks the rest of this function.  Writing
        ``_pending`` then would hide a word from ``_options`` that nobody ever
        guessed, which is the staleness bug this attribute exists to fix,
        arriving from the other direction.
        """
        if generation == self._answer_gen:
            self._pending = word

    def _get_ranking(self, options):
        """Ranking of the words for this clue, computed once per turn.

        Recomputing after every correct guess would double or triple the API
        cost for no signal: the clue has not changed, and revealed words simply
        drop out of ``options``, so the cached order is filtered instead.
        """
        key = (_normalise(self.clue), self._is_sweep())
        if self._ranking is not None and self._ranking_key == key:
            live = [(w, s) for w, s in self._ranking if w in options]
            return live or [(word, 0.0) for word in options]

        scores = {}
        if self.llm.available():
            deadline = _Deadline(self.deadline_s)
            scores = self._llm_scores(options, deadline)

        if scores:
            self._ranking_source = "llm"
        else:
            # The offline path.  It is legal and deterministic, but it is
            # letter overlap rather than meaning, so it is worth saying out
            # loud that this is what is deciding the guess.
            self._ranking_source = "offline"
            self._warn("no usable LLM ranking for clue %r -- guessing from the "
                       "offline similarity fallback" % (self.clue,))
            scores = dict(
                (word, 100.0 * _similarity(self.clue, word)) for word in options)

        ranking = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        ranking = [(w, s) for w, s in ranking if w in options]
        if not ranking:
            ranking = [(word, 0.0) for word in options]

        clue_key = _normalise(self.clue)
        if clue_key:
            # Store the pure single-clue signal: later turns read it back as
            # leftover memory, and blending it with itself would drift.
            self._clue_rankings[clue_key] = ranking

        if self._is_sweep():
            ranking = self._blend_leftovers(ranking, options)

        self._ranking = ranking
        self._ranking_key = key
        self._turn_best = ranking[0][1]
        return ranking

    def _blend_leftovers(self, ranking, options):
        """Fold earlier clues' rankings into an unlimited ("sweep") turn.

        A number of 0 means "you already know the rest" -- the codemaster is
        leaning on words it clued before.  Carrying those earlier rankings in
        at a discount is exactly the convention our codemaster's sweep gate
        assumes, and it degrades to the plain ranking when there is no
        leftover memory (e.g. a foreign codemaster opening with 0).
        """
        if not self._clue_rankings:
            return ranking
        scale = ranking[0][1] if ranking else 100.0
        if scale <= 0:
            scale = 100.0
        scores = dict(ranking)
        current = _normalise(self.clue)
        for clue, previous in self._clue_rankings.items():
            if clue == current or not previous:
                continue
            top = previous[0][1]
            if top <= 0:
                continue
            for word, value in previous:
                if word not in options:
                    continue
                carried = self.leftover_weight * (value / float(top)) * scale
                if carried > scores.get(word, 0.0):
                    scores[word] = carried
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

    def _llm_scores(self, options, deadline):
        """Average several shuffled-order samples into one score per word."""
        samples = max(1, self.samples)
        if deadline.remaining() < 12.0:
            samples = 1

        jobs = []
        for sample in range(samples):
            shuffled = list(options)
            random.Random(7919 * (sample + 1) + len(options)).shuffle(shuffled)
            jobs.append(self._score_job(shuffled, deadline))

        replies = _run_parallel(jobs, deadline, max_workers=samples)

        totals = {}
        seen = 0
        for reply in replies:
            if not reply:
                continue
            parsed = _parse_scores(reply, options)
            if not parsed:
                continue
            seen += 1
            ordered = sorted(parsed.items(), key=lambda pair: -pair[1])
            for rank, (word, value) in enumerate(ordered):
                # Blend the model's own score with a rank-based Borda term so
                # one sample's inflated scale cannot dominate the panel.  The
                # Borda weight is kept low so a genuinely weak second choice
                # still reads as weak to the stop rule.
                borda = max(0.0, 100.0 - 20.0 * rank)
                totals[word] = totals.get(word, 0.0) + 0.7 * float(value) + 0.3 * borda
        if not seen:
            return {}
        return dict((word, value / float(seen)) for word, value in totals.items())

    def _score_job(self, options, deadline):
        system = (
            "You are an expert Codenames field operative. You see the board "
            "words but not the key. You must work out which words the "
            "spymaster's clue points at. Guessing the assassin loses the game "
            "instantly, so only rate a word highly when the link is real."
        )
        context = self._history_context()
        number = ("unlimited (the spymaster expects you to also finish off "
                  "words from earlier clues)" if self.num <= 0 else str(self.num))
        prompt = [
            "Clue: %s  (number of words: %s)" % (self.clue, number),
            "",
            "Remaining board words: %s" % ", ".join(options),
        ]
        if context:
            prompt.append("")
            prompt.append(context)
        prompt.extend([
            "",
            "Score how strongly the clue points at each of the %d most likely "
            "words, from 100 (certain) down to 0 (unrelated). Only list words "
            "you would actually consider." % min(self.top_k, len(options)),
            "",
            'Respond with ONLY a JSON object, e.g. '
            '{"WHALE": 95, "SHIP": 70, "BEACH": 40}',
        ])
        text = "\n".join(prompt)

        def job():
            return self.llm.chat(system, text, max_tokens=500, deadline=deadline)

        return job

    def _history_context(self):
        """Earlier own clues and what they turned up -- cheap extra signal."""
        lines = []
        for move in (self.move_history or [])[-12:]:
            if not move:
                continue
            actor = str(move[0])
            if actor == "%s_Codemaster" % self.team and len(move) >= 3:
                lines.append("Earlier clue from your spymaster: %s %s"
                             % (move[1], move[2]))
            elif actor == "%s_Guesser" % self.team and len(move) >= 3:
                outcome = str(move[2]).strip("*")
                lines.append("You guessed %s -> %s" % (move[1], outcome))
        if not lines:
            return ""
        return "Recent history:\n" + "\n".join(lines[-8:])

    def _fallback_pick(self, options):
        scored = [(_similarity(self.clue, word), word) for word in options]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored[0][1]

    # -- stop rule ---------------------------------------------------------

    def _turn_capacity(self):
        """(guesses this clue is worth, sweep?) -- frozen at the turn's start.

        Clue number ``0`` means unlimited guesses; some codemasters spell the
        same thing as a huge integer.  Either way the ceiling is how many of
        our own words were still hidden when the clue arrived -- guessing past
        that can only hit somebody else's word.  Freezing the pair matters:
        both terms move as words are revealed, and recomputing mid-turn would
        shrink the budget under our own feet (and re-trigger scoring calls).
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

        # Nothing left of ours to find: stop, whatever the number said.
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
        confidence = remaining[0][1] / float(best)

        if not sweep and self.guesses >= limit:
            return self._bonus_guess_ok(remaining[0][0], confidence)

        threshold = self._continue_threshold()
        if sweep:
            # An unlimited clue invites over-reach; demand a clearer signal
            # rather than a small surcharge on the usual bar.
            threshold = max(threshold, self.sweep_confidence)
        if self.verbose:
            sys.stderr.write("[oBirdy G] continue? conf=%.2f thr=%.2f sweep=%s\n"
                             % (confidence, threshold, sweep))
        return confidence >= threshold

    def _continue_threshold(self):
        base = CONTINUE_BASE_DUEL if self._two_team_mode() else CONTINUE_BASE_SOLO

        own_left, unrevealed = self._counts()
        if unrevealed:
            density = own_left / float(unrevealed)
            # A board dense in our own colour makes a marginal pick safer.
            base -= 0.30 * (density - 0.36)

        base += self._urgency()
        return max(0.30, min(0.75, base))

    def _urgency(self):
        """Threshold adjustment for the two-team race (negative = gamble more).

        Red structurally starts one word behind, so "more words left than the
        opponent" is not by itself evidence of losing; only a gap of more than
        one is.  When the opponent is one or two words from winning, a turn
        given away is usually the game, which is worth real risk to avoid.
        """
        if not self._two_team_mode():
            return 0.0
        own_left, opp_left = self._words_left()
        adjustment = 0.0
        if own_left > opp_left + 1:
            adjustment -= 0.06
        if opp_left <= 2:
            adjustment -= 0.10
        return adjustment

    def _must_gamble(self):
        """True when passing the turn probably loses and the +1 can still win.

        Every two-team loss in the phase-2 evaluation was the same 8-8 photo
        finish: both teams grinding out one word per clue, and red -- which
        starts a word behind -- running out of turns first.  In that position
        the safe play is not safe: declining the bonus guess concedes the race.

        The bonus buys exactly **one** word, so it can only end the game when
        one word is all we need.  Needing two or more turns the gamble into a
        pure coin flip: a hit still does not win, while a miss can lose on the
        spot (the assassin, or an opponent word that completes their board).
        eval_c_default seed 14 was exactly that -- two words short with the
        opponent one from home, the +1 taken anyway, AIR the assassin.  So the
        race condition is kept and a winnability condition is added.
        """
        if not self._two_team_mode():
            return False
        own_left, opp_left = self._words_left()
        return 0 < opp_left <= 2 and own_left <= 1

    def _bonus_guess_ok(self, word, confidence):
        """Take the +1 guess only on a genuinely strong leftover association.

        Unless we are about to lose the race, in which case a strong current
        association is enough on its own -- there is no later turn to spend it.
        """
        if not self.allow_bonus:
            return False
        desperate = self._must_gamble()
        threshold = BONUS_THRESHOLD
        if desperate:
            threshold = max(0.5, BONUS_THRESHOLD + self._urgency())
        if confidence < threshold:
            return False
        own_left, unrevealed = self._counts()
        if unrevealed and own_left / float(unrevealed) < 0.3 and not desperate:
            return False
        return desperate or self._leftover_supports(word)

    def _leftover_supports(self, word):
        """Does an earlier, unexhausted clue also point strongly at ``word``?"""
        current = _normalise(self.clue)
        pending = self._pending_own()
        for clue, ranking in self._clue_rankings.items():
            if clue == current:
                continue
            live = [(w, s) for w, s in ranking if not _is_revealed(w)
                    and w in self.words and w != pending]
            if not live or live[0][0] != word:
                continue
            top = ranking[0][1] if ranking else 0.0
            if top > 0 and live[0][1] / float(top) >= 0.75:
                return True
        return False

    # -- game state --------------------------------------------------------

    def _counts(self):
        """(own words still hidden, unrevealed words remaining).

        A pending guess counts as found: the engine only asks us to keep
        guessing when the word it just accepted was one of ours.
        """
        marker = "*%s*" % self.team
        found = sum(1 for w in self.words if str(w) == marker)
        if self._pending_own():
            found += 1
        total = TEAM_TOTALS.get(self.team, 9)
        unrevealed = len(self._options())
        return max(0, total - found), unrevealed

    def _two_team_mode(self):
        for move in self.move_history or ():
            if move and str(move[0]).startswith(self.opponent):
                return True
        return False

    def _words_left(self):
        """(our words still hidden, the opponent's words still hidden)."""
        own_marker = "*%s*" % self.team
        opp_marker = "*%s*" % self.opponent
        own_found = sum(1 for w in self.words if str(w) == own_marker)
        opp_found = sum(1 for w in self.words if str(w) == opp_marker)
        if self._pending_own():
            own_found += 1
        own_left = TEAM_TOTALS.get(self.team, 9) - own_found
        opp_left = TEAM_TOTALS.get(self.opponent, 8) - opp_found
        return max(0, own_left), max(0, opp_left)

    # -- diagnostics -------------------------------------------------------

    def usage_summary(self):
        return self.llm.usage_summary()
