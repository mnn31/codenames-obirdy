"""oBirdy codemaster -- IEEE CoG 2026 Codenames AI Competition entry.

Self-contained by design: this file imports nothing from the evaluation
harness and nothing from its partner guesser.  Everything it needs (LLM
client, retry/backoff, deadline budgeting, clue legality, an offline
fallback) is inlined below.

Pipeline for one clue
---------------------
1. Read the key grid, split the unrevealed board into own / opponent /
   civilian / assassin.  Carry over own words that earlier clues targeted but
   the guesser never found (read back from ``get_move_history``).
2. **Brainstorm** candidate clues with the LLM over promising subsets of own
   words (plus a handful of deterministic fallback candidates so the pipeline
   is never empty).
3. **Legality filter** -- single alphabetic English word, no sub-word
   derivation in either direction against any unrevealed board word.  Mirrors
   the check inside the bundled ``codemaster_GPT`` and the arena audit.
4. **Simulate** -- a panel of sampled "generic guesser" LLM calls (no key
   knowledge) ranks the board for every surviving candidate.
5. **Score** -- risk-adjusted expected value: own words found before the first
   mistake, minus civilian / opponent penalties, minus a dominating assassin
   penalty.  The number reported is what the panel actually finds, not what we
   wish it found.
5a. **Embedding danger sensor** (offline, free) -- an optional bundled table of
   GloVe cosines between a clue vocabulary and the board pool.  If a candidate
   pulls on the assassin at least as hard as on the weakest word its number is
   asking for, it is penalised heavily; if the candidate is not in the table's
   vocabulary at all it is mildly demoted, because a clue no static embedding
   knows is a clue an embedding-based partner cannot decode.  Degrades to a
   no-op when the data file is absent.
5b. **Danger probe** -- the panel only reports the words it happened to rank,
   so a clue that quietly points at the assassin can score perfectly while the
   assassin never appears at all.  The probe asks a no-key guesser persona to
   rate 0-10 how strongly the winning clue points at named danger words.  A
   meaningful assassin rating vetoes it, and selection falls through to the
   next-best candidate, which is probed in turn (two probes per turn at most).
   Opponent pull is a softer penalty.  A finalist that could not be probed
   (deadline, API failure) is treated with suspicion, not trust.

5c. **Race awareness** (two-team track only) -- the two-team score is binary,
   so being out-raced 8-3 costs exactly what an assassin death costs and
   risk-aversion is worthless once the game is already lost.  From the move
   history the codemaster reads each side's words-per-turn pace, projects who
   reaches their last word first, and escalates the three clue-number knobs
   toward ``preset="ambitious"`` in proportion to how many turns behind that
   projection puts us.  Level or ahead, nothing changes.  Inert in single-team
   play, where the opposing team never moves.  Both safety nets stay armed at
   every escalation level.

6. **Endgame sweep** -- clue number ``0`` means *unlimited guesses* (framework
   README).  When only a couple of own words remain, the chosen clue covers
   some of them and every one it misses was already targeted by an earlier
   clue, and no assassin or opponent word sits inside the sweep window, the
   number becomes ``0`` so the guesser can also finish off its leftovers.

Robustness contract: ``get_clue`` never raises and never returns a malformed
or illegal clue.  If the API is unreachable the deterministic fallback still
produces a legal ``[clue, number]`` pair.

Nothing here reads the framework's ``results/`` log files: those may contain
other entrants' games and the organisers treat reading them as leakage.  All
state comes from the board, the key grid and ``get_move_history``, and no
state is carried across games (agents are re-instantiated per game anyway).

The only file this agent ever reads is its own optional similarity table
(``oBirdy/obirdy_simtable.bin.gz``, shipped in a per-team subfolder of the
players directory -- the competition rules allow additional submitted files).
Everything still works without it.

API key: the ``api_key`` kwarg, else ``ANTHROPIC_API_KEY``, else the
``HARDCODED_API_KEY`` constant below (which the packaging script fills in, and
which is ignored while it still holds the checked-in placeholder).  No
environment variable has to be set for tournament play.

Environment (all optional): ``ANTHROPIC_API_KEY``, ``OBIRDY_MODEL`` (model
override), ``OBIRDY_SIMTABLE`` (table path), ``OBIRDY_QUIET=1`` (silence the
diagnostics, warnings excepted), ``OBIRDY_RACE_MODE=0`` (turn duel race
awareness off).
``OBIRDY_PROVIDER=openai_compat`` swaps the inline client for a plain
OpenAI-shaped HTTP endpoint (``OBIRDY_BASE_URL``, key from
``OBIRDY_COMPAT_KEY`` or ``HF_TOKEN``) for cheap plumbing runs; the default and
the competition configuration remain Anthropic.

Python 3.9 compatible.
"""

import gzip
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

#: Version of record; printed at init so a tournament log says which build ran.
AGENT_VERSION = "mega-pidgeot-x build 2026-08-08"

DEFAULT_MODEL = "claude-opus-5"
MODEL_ENV = "OBIRDY_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"

# -- SDK floor --------------------------------------------------------------
#
# The organisers ran our submission against ``anthropic==0.29.0`` and the agent
# quietly turned into its own fallback.  ``Messages.create`` in that release has
# no ``thinking`` parameter, so the sampling-shape probe in ``_LLM`` read the
# resulting ``TypeError`` as "the model refuses this" and dropped the field --
# which left extended thinking ON by default, and the model then spent the whole
# ``max_tokens`` budget on a reasoning block and returned no text at all
# (``stop_reason=max_tokens``, zero text blocks).  An empty string is not an
# exception, so nothing was logged: the brainstorm parsed 0 candidates every
# turn (``candidates=3``, the three deterministic fallbacks) and the panel came
# back empty (``path=unsimulated``).
#
# Two defences.  ``_call_anthropic`` now re-sends the same fields through
# ``extra_body`` -- supported by every release back to 0.29 and copied verbatim
# into the request JSON, which is what the *server* reads -- so an old SDK asks
# for exactly what a new one asks for.  And this floor makes the situation
# visible at startup rather than in a post-mortem.
MIN_ANTHROPIC = (0, 60)
MIN_ANTHROPIC_TEXT = "0.60"
OLD_SDK_WARNING = ("anthropic %s is older than the tested minimum %s -- please "
                   "pip install -U anthropic; continuing with compatibility "
                   "mode")
#: Ceiling on how far an empty reply may widen the token budget (see
#: ``_LLM._empty_reply``).  4x turns 900 into 3600, comfortably above any
#: reasoning block plus its answer.
MAX_TOKEN_SCALE = 4


def _version_tuple(text):
    """``"0.29.0" -> (0, 29, 0)``.  Stops at the first non-numeric component,
    so release candidates and dev suffixes compare on their numeric prefix."""
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

    ``(None, False)`` when the package is not importable at all -- that case
    has its own, louder warning and is not this check's business.
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
# ``print`` rather than logging or stderr, because the organisers' harness
# captures stdout and that is where they will look.  Everything is one line,
# prefixed, and says which of our two agents produced it.
#
# The reason this exists: a test run of our zip went the whole game on the
# deterministic offline fallback (every clue number 1, TOOL / ROYAL / SPEECH)
# because the key never reached the process, and nothing in the output said so.
# A silent fallback is indistinguishable from a bad agent.
DEBUG_PREFIX = "[oBirdy]"
QUIET_ENV = "OBIRDY_QUIET"
#: ASCII only.  These lines land in whatever console the organisers run, and a
#: Windows code page that cannot encode an em dash would turn a diagnostic into
#: a ``UnicodeEncodeError`` mid-game.
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


#: Provider switch.  ``anthropic`` (the default) is the competition
#: configuration and is untouched by this option existing.  ``openai_compat``
#: speaks plain OpenAI chat-completions over stdlib HTTP to any base URL --
#: it exists so pipeline plumbing can be exercised end to end against a free
#: router without spending competition credits, and it adds no dependency.
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


#: Hard wall-clock budget for one ``get_clue`` call.  The event's soft limit is
#: 60 s; we stay well inside it and degrade (fewer candidates / samples) as the
#: deadline approaches.
MOVE_DEADLINE_S = 45.0

#: Per-attempt HTTP timeout.
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

# Risk weights (per own word found = +1.0).
ASSASSIN_PENALTY = 9.0      # must dominate everything else
OPPONENT_PENALTY = 1.3      # two-team: also hands the opponent a word
OPPONENT_PENALTY_SOLO = 0.7  # single-team: a blue word is just a lost turn
CIVILIAN_PENALTY = 0.7      # a lost turn
BONUS_GUESS_WEIGHT = 0.35   # chance the guesser takes one extra guess

#: Fraction of panel samples that must agree a prefix is all-own before the
#: clue number is allowed to grow to that length.  0.5 = simple majority.
#: At the default two samples this already means "either sample agrees", so it
#: is a knob for tightening, not loosening.
MAJORITY_FRACTION = 0.5

#: How far a panel-supported prefix may outrun the brainstorm's own claimed
#: target count.  The brainstorm never looked at how a guesser would read the
#: board; the panel did, so its verdict could earn some slack.
#:
#: Held at 0.  A sweep found (bonus 0.55 / civilian 0.45 / slack 1) far ahead
#: on 20 haiku seeds (6.55 vs 8.35) and on the first 15 sonnet seeds (6.33 vs
#: 7.20), but a second, independent 15-seed sonnet run of that same config came
#: back at 9.80 with three assassin losses.  Reading those three games, two
#: died on guess 2 or 3 of a raised number walking into an assassin the panel
#: had never ranked -- more words per clue is mechanically more assassin
#: exposure per turn, and our repellence term cannot price a word the panel
#: never surfaced.  Until the panel is asked about the assassin directly, the
#: extra half-word per clue is not worth a 10% loss rate.
#:
#: Asking directly is what the danger probe below does, and the probe now ships
#: on -- but the ambitious numbers behind it lost their own paired eval (solo
#: 7.87 with an assassin death, two-team 65%), so they stay parked in
#: ``preset="pidgeot"`` and this default stays at 0.
CLAIMED_SLACK = 0

# Repellence: the terms above only fire when the panel actually *reaches* the
# dangerous word, i.e. when every earlier pick was one of ours.  A partner
# guesser we did not co-design will not walk the panel's exact order, so a clue
# whose ranking contains the assassin anywhere near the top is dangerous even
# when our own simulated panel would stop before it.  These weights punish mere
# *presence* in the ranking, decayed by position, independently of the
# expected-value term.
ASSASSIN_PRESENCE_PENALTY = 2.5
OPPONENT_PRESENCE_PENALTY = 0.35
#: Positional decay: penalty at rank ``i`` is ``weight / (i + PRESENCE_DECAY)``.
PRESENCE_DECAY = 1.0

# Danger probe.  The panel answers "what would a guesser pick?"; the probe
# answers "how hard does this clue pull on *this* word?", which is the question
# the panel silently declines to answer for any word it never ranked.
#
# **On by default as of Pidgeot.**  It was built as the safety net under a
# raised clue number, but the evidence says clue *safety* is the binding
# failure on its own: the shipped defaults died on PAGEANT 2 (seed 10, STATE
# never surfaced by the panel) and on five of eight games against an embedding
# guesser.  So the probe ships and the raised numbers do not.
#
# It is deliberately cheap.  Only the *winning* candidate is probed -- one
# extra call per turn.  If that call vetoes, selection falls through to the
# next-best candidate and probes that, and then stops: two calls per turn is
# the ceiling.  Probing the top three (the old shape) tripled the cost to rank
# clues that were never going to be chosen.
PROBE_TOP_K = 2             # candidates the probe may walk in one turn
PROBE_VETO_SCORE = 5.0      # assassin rating at/above which the clue dies
PROBE_VETO_PENALTY = 50.0   # dwarfs any EV a vetoed clue could have earned
#: Below this, "the assassin is as strong as our own words" is rating noise.
PROBE_RELATIVE_FLOOR = 3.0
#: Require the embedding sensor to agree before the *relative* veto fires.
#:
#: Held **off**, and the first validation battery is why.  Across its 38 games
#: the probe vetoed 15 times: 12 relative (assassin rated 3-4 while our own
#: marginal target was rated no higher) and 3 absolute.  Every one of the 15
#: fell through to a clue the probe rated *strictly safer* on the assassin,
#: those arms recorded 0 assassin deaths, and the solo arm's 0.4-turn gap
#: against the pre-probe run was carried just as much by the eight games in
#: which no veto fired at all.  There is no false-positive evidence to spend a
#: safety net on.  The knob exists because the two absolute vetoes that killed
#: a clue our own words out-rated (assassin 5 against margin 6 and 8) are the
#: shape that *would* justify it, and re-arming should be one kwarg.
PROBE_VETO_REQUIRES_EMBED = False
#: A surviving clue whose probe still rated the assassin this high asks for one
#: word only.
#:
#: The veto answers "is this clue poisoned?".  This answers a different and
#: cheaper question: "the clue is defensible, but how many words should we buy
#: on it?".  Both of the 2026-08-03 battery's assassin deaths came through here
#: -- ``PLATE 2`` into ``WASHER`` and ``NPC 2`` into ``QUEST``, both rated
#: **exactly 4.0** on the assassin, both surviving the absolute veto by one
#: point and the relative veto because our own marginal target was rated higher
#: (6 and 5), and in both the assassin was taken as **guess 2 of the 2**.
#: Neither clue was bad: ``PLATE`` pulls ``FORK`` at 0.298 against ``WASHER`` at
#: 0.146, and the embedding sensor is right to call it clean.  The *number* was
#: the exposure.
#:
#: Measured over the 657 issued clues in the 98 recorded games that carry
#: per-clue probe detail: ratings of 5+ are already vetoed, and exactly **8**
#: issued clues sit in this top surviving band -- of which **2 were fatal**.
#: Capping all 8 at one word costs 8 words in 657 clues (0.012 words per clue,
#: about one word every dozen games) and removes the guess-2 exposure that
#: killed both games.  A cheaper trade than lowering the veto, which would have
#: thrown the clues away entirely.
PROBE_NUMBER_CAP_SCORE = 4.0
PROBE_ASSASSIN_WEIGHT = 1.2   # sub-veto assassin pull, quadratic in the rating
PROBE_OPPONENT_WEIGHT = 0.25  # opponent pull, linear in the mean rating
#: A finalist the probe could not reach is *not* trusted: it loses this much
#: score and its number falls back to what the brainstorm itself claimed.
PROBE_MISSING_PENALTY = 0.25
PROBE_UNPROBED_CAP = 2      # ceiling for an unprobed clue with no claimed count
PROBE_MAX_OPPONENT = 6      # opponent words asked about, to bound the prompt
PROBE_REFERENCE_WORDS = 2   # own words mixed in, so the list is not all danger
PROBE_MIN_SECONDS = 6.0     # under this much budget the probe is skipped
PROBE_TIGHT_SECONDS = 12.0  # under this much budget only the leader is probed

# Embedding danger sensor.  The probe costs a call and can only ask about a
# handful of words; this costs nothing and sees the whole board, but only for
# words a static embedding knows.  The two fail in different directions, which
# is the point of running both.
#
# Evidence (2026-07-31, offline, from the raw game records): four of the five
# assassin deaths in ``sweep_obirdy_cm_abra_g_default_solo`` and the one death
# in ``reeval_c_fixed`` are visible here -- SURGE pulls LEAD at 0.281 against
# its own target CHARGE at 0.169, PAGEANT pulls STATE at 0.198 against SPOT at
# 0.124 -- and the fifth (EGGSHELL) is not an English word the embedding knows
# at all, which is the other half of the sensor.
SIMTABLE_ENV = "OBIRDY_SIMTABLE"
#: The shipped (format 2, themed-coverage) file carries a versioned name so an
#: unzip that declines to overwrite an older oBirdy/ folder can never leave a
#: stale table in play -- the organisers hit exactly that.  Legacy names are
#: still searched as fallbacks.
SIMTABLE_FILENAME_V2 = "obirdy_simtable_v2.bin.gz"
SIMTABLE_FILENAME = "obirdy_simtable.bin.gz"
#: Per-team subfolder for auxiliary files, which is where the organisers want
#: them: several entries ship data files and a flat ``players/`` directory
#: would have them colliding on name.  The bare module directory is still
#: searched afterwards so an older layout keeps working.
SIMTABLE_SUBDIR = "oBirdy"
#: Format magics, newest last; every one of them is 7 bytes, so the JSON header
#: always starts at the same offset.  Version 1 packs each stored pair into one
#: 16-bit word (10-bit board index, 6-bit code) and so cannot describe a board
#: vocabulary past 1024 words; version 2 keeps the indices and the codes in
#: parallel arrays, which buys a full 16-bit index -- enough for the themed
#: tournament pools -- and is *smaller* on the wire, because a run of ascending
#: indices and a run of 6-bit codes each compress far better apart than
#: interleaved.  Both are read here: a data file older than 2026-08-03 must
#: still load, since the agent and its table can be unpacked from different
#: zips.
_SIMTABLE_MAGICS = (b"OBSIM1\n", b"OBSIM2\n")
#: Version 1's magic, kept under its old name for callers that predate v2.
_SIMTABLE_MAGIC = _SIMTABLE_MAGICS[0]
EMBED_SENSOR = True
#: How far under the weakest word the number is buying the assassin must sit.
#: Measured over 2308 recorded in-vocabulary clue turns: 0.02 fires on 8.0% of
#: turns and covers every one of the codemaster-side deaths above.
EMBED_MARGIN = 0.02
#: Dominates any expected-value difference between candidates without reaching
#: the probe's outright veto, so a probe veto still wins the argument.
EMBED_PENALTY = 6.0
#: A clue outside the table's vocabulary is *suspicious*, not condemned: this
#: is a tie-break, so an in-vocabulary candidate wins at equal score.  LASTDRINK
#: and EGGSHELL both died exactly here -- an embedding partner given a word its
#: vocabulary lacks falls back to letter overlap and guesses essentially at
#: random.
EMBED_OOV_PENALTY = 0.05
#: How deep into the offline vocabulary ``_fallback_clue`` will walk looking for
#: a clue the sensor does not flag.  Bounded because every step costs a little
#: overlap score, and a clue nobody can decode is its own kind of lost turn.
FALLBACK_SENSOR_TOP_K = 8
#: Tie-break weight on the incoming candidate order in ``_unsimulated_clue``.
#: Small enough that it can never outweigh a sensor flag (worth ``EMBED_PENALTY``
#: = 6.0), large enough to survive the probe's own re-sort.
UNSIMULATED_ORDER_STEP = 0.001


#: Named configurations.  ``preset`` seeds the knobs; any explicit kwarg still
#: wins, so a sweep can vary one dimension of a preset.  The default preset is
#: the shipped behaviour and must stay empty.
PRESETS = {
    "pidgeotto": {},
    "pidgeot": {},
    # The clue-number push that lost three games to the assassin in Pidgeotto.
    # Parked: it failed its own paired eval even with the probe armed.  Kept as
    # a named config so re-arming it is a one-word change, not an archaeology
    # exercise.
    "ambitious": {
        "bonus_guess_weight": 0.55,
        "civilian_penalty": 0.45,
        "claimed_slack": 1,
    },
}
DEFAULT_PRESET = "pidgeot"


# ---------------------------------------------------------------------------
# Race awareness (two-team track only)
# ---------------------------------------------------------------------------
#
# The two-team score is binary.  Losing a race 8-3 is worth exactly as much as
# walking into the assassin, so risk-aversion has no value once the game is
# already lost -- and it has a great deal of value while the game is won.  The
# shipped conservative numbers were priced against the *single-team* score,
# where a death costs 25 against an average of about 7, and that pricing must
# not change: everything below is inert unless the opposing team has visibly
# moved, which never happens in a single-team game (``game.Game.run`` only ever
# gives the red team the move when ``single_team`` is set, so no ``Blue_*``
# entry can appear in ``move_history``).
#
# Evidence.  Against the Abra pair (GloVe codemaster + GloVe guesser, perfectly
# coupled, clue numbers 2-3) the shipped build is 0-3 with **no assassin
# deaths**: seed 100 lost 8-3 at their ~2.7 words per turn against our 1.0,
# seed 101 lost 8-5, seed 102 lost an 8-8 photo finish by one word.  The 90%
# duel record that promoted Pidgeot was measured only against the Rattata
# heuristic pair, which scores about 1.3 words per turn -- slower than we do.
#
# What the levers are.  Only ``claimed_slack`` mechanically raises a clue
# number; ``bonus_guess_weight`` and ``civilian_penalty`` change *which*
# candidate wins, favouring ones whose panel prefix runs longer and tolerating
# a civilian on the way.  The three together are ``preset="ambitious"``, whose
# measured cost is roughly one assassin death per 10-15 solo games even with
# both safety nets armed.  In a race we are projected to lose that trade is
# correct; while ahead it is not; the scaling below is the whole difference.
RACE_MODE = True
RACE_MODE_ENV = "OBIRDY_RACE_MODE"

#: Words per turn assumed for a team we have not watched yet.  The opponent
#: prior is deliberately *competent* (roughly the Abra pair's 2-3 clue numbers
#: discounted for misses) rather than the Rattata pace we happen to have the
#: most games against.  Our own prior is the shipped agent's measured rate.
RACE_OPP_PACE_PRIOR = 1.5
RACE_OWN_PACE_PRIOR = 1.2
#: Pseudo-turns of prior mixed into each pace, so one unlucky turn cannot
#: convince the agent it is losing and one lucky turn cannot convince it it is
#: winning.  1.0 = the prior counts for exactly one observed turn.
RACE_PRIOR_WEIGHT = 1.0
#: A pace can never read as zero: a team that has found nothing yet is slow,
#: not stopped, and dividing by it must stay finite.
RACE_MIN_PACE = 0.25
#: Turns behind at which escalation reaches the ambitious values.  Turns are
#: integers, so the deficit is too: 1 turn behind is a half-escalated arm and
#: 2 or more turns behind is the full one.
RACE_FULL_DEFICIT = 2.0
#: Scale at or above which ``claimed_slack`` is allowed to rise.  It is the one
#: knob that literally puts another word on the clue, and more words per clue
#: is mechanically more assassin exposure per turn, so it gets its own gate
#: rather than riding the same ramp as the scoring weights.
#:
#: Deficits are whole turns and ``RACE_FULL_DEFICIT`` is 2, so the shipped
#: scale only ever takes three values: 0 (level or ahead), 0.5 (one turn
#: behind) and 1.0 (two or more).  0.5 therefore means "arm the number lift as
#: soon as we are a full turn behind", which is the aggressive reading and is
#: chosen deliberately: the photo-finish losses are precisely the games one
#: extra word would have turned, and Abra seed 102 was lost 8-8 by one word on
#: a turn this reads at 0.5.  If the do-no-harm regression against the slower
#: opponent comes back with an assassin death, raising this to 0.75 withdraws
#: the number lift from those games and leaves everything else alone -- that is
#: the first lever to reach for, not turning the feature off.
RACE_SLACK_SCALE = 0.5

#: Board composition, mirroring ``game.Game``.  Used only by offline analysis
#: (``harness/race_audit.py``) reconstructing counts from a recorded history;
#: the live agent reads the board instead, which is authoritative.
TEAM_TOTALS = {"Red": 9, "Blue": 8}


def _race_default():
    """Race awareness is on unless the environment explicitly turns it off."""
    value = os.environ.get(RACE_MODE_ENV)
    if value is None or str(value).strip() == "":
        return RACE_MODE
    return _truthy(value)


def race_counts(move_history, team):
    """Tally one recorded history from ``team``'s point of view.

    Two different tallies, because they answer two different questions:

    * ``own_pace_words`` / ``opp_pace_words`` count a team's *own* colour
      revealed on that team's *own* turn -- how productive a turn is, which is
      what a pace estimate means.  A word we hand the opponent by guessing it
      for them is not evidence about how fast they clue.
    * ``own_revealed`` / ``opp_revealed`` count a colour revealed by anybody,
      which is what "how many are left" means.  The live agent reads that off
      the board instead; this exists so offline analysis can reconstruct it.

    A pure function of the history, so the agent and the offline auditor can
    never drift apart.
    """
    opponent = "Blue" if team == "Red" else "Red"
    own_marker = "*%s*" % team
    opp_marker = "*%s*" % opponent
    counts = {"own_turns": 0, "opp_turns": 0, "own_pace_words": 0,
              "opp_pace_words": 0, "own_revealed": 0, "opp_revealed": 0,
              "opponent_moved": False}
    for move in move_history or ():
        if not move:
            continue
        actor = str(move[0])
        if actor.startswith(opponent):
            counts["opponent_moved"] = True
        if actor == "%s_Codemaster" % team:
            counts["own_turns"] += 1
        elif actor == "%s_Codemaster" % opponent:
            counts["opp_turns"] += 1
        elif actor.endswith("_Guesser") and len(move) >= 3:
            revealed = str(move[2])
            if revealed == own_marker:
                counts["own_revealed"] += 1
                if actor.startswith(team):
                    counts["own_pace_words"] += 1
            elif revealed == opp_marker:
                counts["opp_revealed"] += 1
                if actor.startswith(opponent):
                    counts["opp_pace_words"] += 1
    return counts


def race_state(move_history, team, own_left, opp_left,
               full_deficit=RACE_FULL_DEFICIT):
    """Where the two-team race stands, from ``team``'s side, before its clue.

    ``None`` when the opponent has not moved yet -- a single-team game forever,
    and our own opening clue in a duel, where there is nothing to read.

    The projection is in **turns**, and turns are integers: needing 1.2 turns
    means needing 2.  Red moves first and therefore wins a tie, so red's
    deficit is ``own_need - opp_need`` and blue's carries a +1.  Positive means
    we are projected to lose by that many turns.
    """
    counts = race_counts(move_history, team)
    if not counts["opponent_moved"]:
        return None
    own_left = max(0, int(own_left))
    opp_left = max(0, int(opp_left))
    weight = float(RACE_PRIOR_WEIGHT)
    own_pace = max(RACE_MIN_PACE,
                   (counts["own_pace_words"] + weight * RACE_OWN_PACE_PRIOR)
                   / (counts["own_turns"] + weight))
    opp_pace = max(RACE_MIN_PACE,
                   (counts["opp_pace_words"] + weight * RACE_OPP_PACE_PRIOR)
                   / (counts["opp_turns"] + weight))
    own_proj = own_left / own_pace
    opp_proj = opp_left / opp_pace
    # ceil without importing math: the file's import list is part of the
    # self-containment contract and is asserted by a test.
    own_need = int(own_proj) + (1 if own_proj > int(own_proj) else 0)
    opp_need = int(opp_proj) + (1 if opp_proj > int(opp_proj) else 0)
    deficit = own_need - opp_need + (0 if team == "Red" else 1)
    scale = deficit / float(full_deficit or RACE_FULL_DEFICIT)
    state = dict(counts)
    state.update({
        "own_left": own_left,
        "opp_left": opp_left,
        "own_pace": round(own_pace, 3),
        "opp_pace": round(opp_pace, 3),
        "own_need": own_need,
        "opp_need": opp_need,
        "deficit": deficit,
        "scale": max(0.0, min(1.0, scale)),
    })
    return state


def race_knobs(scale, bonus_guess_weight, civilian_penalty, claimed_slack,
               slack_scale=RACE_SLACK_SCALE):
    """The three number knobs, escalated ``scale`` of the way to ambitious.

    Interpolated toward ``PRESETS["ambitious"]`` and never past it: at
    ``scale`` 0 these are exactly the values handed in, at 1 they are exactly
    the ambitious ones.  A configuration that is *already* more ambitious than
    the target keeps its own value rather than being pulled back -- escalation
    may only ever raise ambition.
    """
    target = PRESETS["ambitious"]
    scale = max(0.0, min(1.0, float(scale)))
    bonus = float(bonus_guess_weight)
    if target["bonus_guess_weight"] > bonus:
        bonus += scale * (target["bonus_guess_weight"] - bonus)
    civilian = float(civilian_penalty)
    if target["civilian_penalty"] < civilian:
        civilian -= scale * (civilian - target["civilian_penalty"])
    slack = int(claimed_slack)
    if scale >= slack_scale and target["claimed_slack"] > slack:
        slack = int(target["claimed_slack"])
    return bonus, civilian, slack


# ---------------------------------------------------------------------------
# Deterministic offline fallback vocabulary
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

_FALLBACK_VOCAB = tuple(sorted(set(list(_FALLBACK_ASSOC.keys()) + [
    "ANIMAL", "CAPTAIN", "CARGO", "CHASE", "CLIMATE", "COMFORT", "CONFLICT",
    "COUNTRY", "CULTURE", "DANGER", "DESERT", "DEVICE", "DINNER", "DRIVER",
    "EMPIRE", "ENERGY", "ESCAPE", "FABRIC", "FACTORY", "FARMER", "FESTIVAL",
    "FLAVOUR", "FOSSIL", "FRIEND", "FUTURE", "GADGET", "GARDEN", "GLACIER",
    "HARBOUR", "HARVEST", "HAZARD", "HELMET", "HISTORY", "HOLIDAY", "HUNTER",
    "ISLAND", "JOURNEY", "JUNGLE", "KITCHEN", "LADDER", "LANTERN", "LEGEND",
    "LIBRARY", "LIQUID", "MAGNET", "MARKET", "MEADOW", "MEMORY", "MERCHANT",
    "MESSAGE", "MINERAL", "MIRROR", "MONSTER", "MORNING", "MOTOR", "MOUNTAIN",
    "MUSEUM", "MYSTERY", "NATURE", "NUMBER", "OFFICE", "PACKAGE", "PARADE",
    "PATTERN", "PICNIC", "PILLOW", "PIONEER", "POCKET", "POWDER", "PRISON",
    "PUZZLE", "QUARRY", "RAINBOW", "RECIPE", "RESCUE", "RHYTHM", "RIDDLE",
    "RIVER", "SAFARI", "SEASON", "SECRET", "SHELTER", "SIGNAL", "SILENCE",
    "SKETCH", "SOUVENIR", "SPEECH", "STORAGE", "STORY", "STRANGER", "SUMMIT",
    "SUPPER", "SURFACE", "SYMBOL", "TALENT", "TAVERN", "TEMPEST", "THUNDER",
    "TICKET", "TRADITION", "TRAFFIC", "TREASURE", "TROPHY", "TUNNEL",
    "VALLEY", "VICTORY", "VILLAGE", "VOYAGE", "WEALTH", "WHISPER", "WILDLIFE",
    "WINTER", "WIZARD", "WONDER", "WORKSHOP",
])))


# ---------------------------------------------------------------------------
# Bundled similarity table (optional data file)
# ---------------------------------------------------------------------------

_SIMTABLE_STATE = {"loaded": False, "table": None, "path": None}


def _module_dir():
    """Directory this agent file lives in (the framework's ``players/``)."""
    return os.path.dirname(os.path.abspath(__file__))


class _SimTable(object):
    """Read-only view of ``obirdy_simtable.bin.gz``.

    Built by ``harness/simtable.py`` from a GloVe cache; see that module for
    the layout.  Loading parses two word lists and keeps the numeric sections
    as raw bytes, so there is no matrix, no numpy and no per-entry Python
    object -- a lookup slices two or three bytes per stored pair, depending on
    the format version.

    Every failure mode (file absent, truncated, wrong magic, unreadable) ends
    in ``load`` returning ``None`` and the sensor becoming a no-op.  A
    submission that cannot find its own data file must still play.
    """

    __slots__ = ("board_words", "board_index", "clue_words", "clue_index",
                 "_offsets", "_entries", "_codes", "levels", "floor",
                 "code_bits", "code_mask", "version", "path")

    def __init__(self, header, board_words, clue_words, offsets, entries,
                 codes=None):
        self.path = None
        self.board_words = board_words
        self.board_index = dict((w, i) for i, w in enumerate(board_words))
        self.clue_words = clue_words
        self.clue_index = dict((w, i) for i, w in enumerate(clue_words))
        self._offsets = offsets
        #: Version 1: packed (index << code_bits) | code, 2 bytes each.
        #: Version 2: bare uint16 board indices, with ``_codes`` alongside.
        self._entries = entries
        self._codes = codes
        self.levels = [float(x) for x in header.get("levels") or ()]
        self.floor = float(header.get("floor") or 0.0)
        self.code_bits = int(header.get("code_bits") or 6)
        self.code_mask = (1 << self.code_bits) - 1
        self.version = int(header.get("version")
                           or (1 if codes is None else 2))

    # -- loading -----------------------------------------------------------

    @staticmethod
    def search_paths(path=None):
        """Where the table may live, best first.

        1. an explicit ``simtable_path`` kwarg, or ``OBIRDY_SIMTABLE``;
        2. ``<module dir>/oBirdy/`` -- the submitted layout;
        3. ``<module dir>/`` -- the pre-2026-08 layout, kept so an older
           checkout or an already-unpacked zip still finds its data;
        4. ``<repo>/data/`` -- development only.
        """
        if path:
            return [path]
        found = []
        from_env = os.environ.get(SIMTABLE_ENV)
        if from_env:
            found.append(from_env)
        here = _module_dir()
        found.append(os.path.join(here, SIMTABLE_SUBDIR, SIMTABLE_FILENAME_V2))
        found.append(os.path.join(here, SIMTABLE_SUBDIR, SIMTABLE_FILENAME))
        found.append(os.path.join(here, SIMTABLE_FILENAME))
        found.append(os.path.join(os.path.dirname(os.path.dirname(here)),
                                  "data", SIMTABLE_FILENAME_V2))
        found.append(os.path.join(os.path.dirname(os.path.dirname(here)),
                                  "data", SIMTABLE_FILENAME))
        return found

    @classmethod
    def load(cls, path=None):
        for candidate in cls.search_paths(path):
            try:
                if not candidate or not os.path.exists(candidate):
                    continue
                with gzip.open(candidate, "rb") as handle:
                    blob = handle.read()
                table = cls._parse(blob)
                if table is not None:
                    table.path = candidate
                return table
            except Exception:  # noqa: BLE001 - a bad data file is not fatal
                continue
        return None

    @classmethod
    def _parse(cls, blob):
        version = 0
        for candidate, magic in enumerate(_SIMTABLE_MAGICS, start=1):
            if blob.startswith(magic):
                version = candidate
                break
        if not version:
            return None
        start = len(_SIMTABLE_MAGICS[version - 1])
        stop = blob.index(b"\n", start)
        header = json.loads(blob[start:stop].decode("utf-8"))
        header.setdefault("version", version)
        n_board = int(header["n_board"])
        n_clue = int(header["n_clue"])
        n_entries = int(header.get("n_entries", -1))
        parts = blob[stop + 1:].split(b"\n", n_board + n_clue)
        if len(parts) != n_board + n_clue + 1:
            return None
        board = [p.decode("utf-8") for p in parts[:n_board]]
        clues = [p.decode("utf-8") for p in parts[n_board:n_board + n_clue]]
        tail = parts[-1]
        offsets_bytes = 4 * (n_clue + 1)
        offsets = tail[:offsets_bytes]
        body = tail[offsets_bytes:]
        if len(offsets) != offsets_bytes:
            return None
        if version == 1:
            return cls(header, board, clues, offsets, body)
        # Version 2: n_entries uint16 indices, then n_entries code bytes.  The
        # split is what n_entries is for, so a header that does not declare it
        # (or declares a length the body cannot hold) is unreadable rather than
        # silently mis-sliced.
        if n_entries < 0 or len(body) < 3 * n_entries:
            return None
        return cls(header, board, clues, offsets, body[:2 * n_entries],
                   body[2 * n_entries:3 * n_entries])

    @classmethod
    def shared(cls, path=None):
        """Load once per process; both teams' codemasters share the result."""
        if _SIMTABLE_STATE["loaded"] and _SIMTABLE_STATE["path"] == path:
            return _SIMTABLE_STATE["table"]
        table = cls.load(path)
        _SIMTABLE_STATE["table"] = table
        _SIMTABLE_STATE["path"] = path
        _SIMTABLE_STATE["loaded"] = True
        return table

    # -- lookup ------------------------------------------------------------

    def _offset(self, row):
        return int.from_bytes(self._offsets[4 * row:4 * row + 4], "little")

    def has_clue(self, clue):
        return _normalise(clue).lower() in self.clue_index

    def pulls(self, clue, words):
        """``{board word: cosine}`` for the stored pairs, or ``None`` if OOV.

        Only pairs at or above the table's floor are stored, so a board word
        missing from the result pulls *less* than the floor -- which is the
        answer the caller wants, not a gap in the data.
        """
        row = self.clue_index.get(_normalise(clue).lower())
        if row is None:
            return None
        wanted = {}
        for word in words or ():
            column = self.board_index.get(_normalise(word))
            if column is not None:
                wanted[column] = word
        out = {}
        if not wanted:
            return out
        entries = self._entries
        codes = self._codes
        levels = self.levels
        if codes is None:                       # version 1: one packed word
            for i in range(self._offset(row), self._offset(row + 1)):
                packed = int.from_bytes(entries[2 * i:2 * i + 2], "little")
                word = wanted.get(packed >> self.code_bits)
                if word is not None:
                    out[word] = levels[packed & self.code_mask]
            return out
        for i in range(self._offset(row), self._offset(row + 1)):
            word = wanted.get(int.from_bytes(entries[2 * i:2 * i + 2],
                                             "little"))
            if word is not None:
                out[word] = levels[codes[i] & self.code_mask]
        return out


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

_ALPHA_RE = re.compile(r"^[A-Z]+$")


def _normalise(word):
    """Uppercase and keep letters only (board words may contain punctuation)."""
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def _is_revealed(board_word):
    text = str(board_word)
    return len(text) > 0 and text[0] == "*"


def clue_is_legal(clue, board_words):
    """Competition legality: single alphabetic word, no sub-word derivation.

    Mirrors the rule the bundled ``codemaster_GPT`` polices itself with, and
    the independent audit the arena performs, so a clue that passes here can
    never be counted illegal downstream.
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
    """Thin chat wrapper over one of two providers.

    * ``anthropic`` (default): the Messages API through the optional SDK, whose
      import stays lazy so the other provider works without it installed.  Key
      from ``ANTHROPIC_API_KEY`` only -- never a literal.  The sampling-parameter
      shape is probed once per model, because current Claude models differ on
      whether they accept ``thinking`` / ``output_config``.
    * ``openai_compat``: OpenAI-shaped ``POST {base}/chat/completions`` over
      stdlib ``urllib``, key from ``OBIRDY_COMPAT_KEY`` or ``HF_TOKEN``.  No
      third-party package is involved, so the submission file stays
      self-contained.

    Both providers share one retry policy: transient failures back off
    exponentially with jitter, always respecting the caller's deadline, and
    ``chat`` returns ``None`` instead of raising so every caller can fall back
    deterministically.
    """

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
        self._extra = None            # resolved lazily (see _param_shapes)
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

    # -- shapes ------------------------------------------------------------

    @staticmethod
    def _param_shapes():
        # Most→least featureful.  Thinking is disabled deliberately: these are
        # short structured judgements where latency and tokens matter more
        # than deliberation, and thinking is on by default on newer models.
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

    # -- call --------------------------------------------------------------

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
        identical fields go through ``extra_body``, which lands them verbatim in
        the request JSON.  Same wire format, same server behaviour.
        """
        if not fields:
            return {}
        if self._extra_body:
            return {"extra_body": dict(fields)}
        return dict(fields)

    def _empty_reply(self, response, budget):
        """Record a text-less reply and say whether a retry could fix it.

        Returning ``""`` as though it were an answer is precisely how the 0.29
        regression stayed invisible for a whole tournament run: an SDK that
        cannot disable extended thinking spends the entire ``max_tokens``
        budget on a reasoning block and emits no text block at all.  That is a
        failure and gets reported as one; when the *budget* is what ran out,
        widening it once is worth one retry.
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
        """One Messages attempt -> ``(text or None, retry?)``."""
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
        """One OpenAI-shaped attempt over stdlib HTTP -> ``(text, retry?)``."""
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


def _run_parallel(jobs, deadline, max_workers=6):
    """Run zero-argument callables concurrently; failures become ``None``."""
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
# Response parsing
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
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    return None
    return None


def _parse_candidates(text):
    """``[{"clue": ..., "targets": [...]}, ...]`` from a brainstorm reply."""
    out = []
    data = _extract_json(text, "[", "]")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                clue = item.get("clue") or item.get("word") or item.get("CLUE")
                targets = item.get("targets") or item.get("words") or []
            elif isinstance(item, str):
                clue, targets = item, []
            else:
                continue
            if not clue:
                continue
            clue = _normalise(clue)
            if not clue:
                continue
            clean = []
            if isinstance(targets, (list, tuple)):
                for target in targets:
                    norm = _normalise(target)
                    if norm:
                        clean.append(norm)
            out.append((clue, clean))
    if out:
        return out
    # Last-ditch: scrape bare uppercase words out of the reply.
    for token in re.findall(r"[A-Za-z]{3,}", text or ""):
        out.append((token.upper(), []))
    return out[:20]


def _parse_panel(text, clues, board_words):
    """``{clue: [ordered board words]}`` from a panel reply."""
    board_index = {}
    for word in board_words:
        board_index[_normalise(word)] = word
    result = {}
    data = _extract_json(text, "{", "}")
    if isinstance(data, dict):
        for key, value in data.items():
            clue = _normalise(key)
            if clue not in clues:
                continue
            ordered = []
            if isinstance(value, (list, tuple)):
                for entry in value:
                    if isinstance(entry, dict):
                        entry = entry.get("word") or entry.get("board_word") or ""
                    word = board_index.get(_normalise(entry))
                    if word is not None and word not in ordered:
                        ordered.append(word)
            if ordered:
                result[clue] = ordered
    return result


def _coerce_score(value):
    """A 0-10 rating out of whatever shape the model felt like emitting."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return None
        number = float(match.group(0))
    elif isinstance(value, dict):
        for key in ("score", "rating", "value", "pull"):
            if key in value:
                return _coerce_score(value[key])
        return None
    else:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return max(0.0, min(10.0, number))


def _parse_probe(text, clues, board_words):
    """``{clue: {board word: 0-10 rating}}`` from a danger-probe reply.

    Accepts both shapes a model plausibly returns: the nested
    ``{"OCEAN": {"POISON": 2}}`` a batched prompt asks for, and the flat
    ``{"POISON": 2}`` an isolated single-clue prompt tends to produce.
    """
    board_index = {}
    for word in board_words:
        board_index[_normalise(word)] = word
    clue_set = set(clues)
    data = _extract_json(text, "{", "}")
    if not isinstance(data, dict):
        return {}

    def ratings(mapping):
        out = {}
        for key, value in mapping.items():
            word = board_index.get(_normalise(key))
            if word is None:
                continue
            score = _coerce_score(value)
            if score is not None:
                out[word] = score
        return out

    result = {}
    for key, value in data.items():
        clue = _normalise(key)
        if clue in clue_set and isinstance(value, dict):
            scored = ratings(value)
            if scored:
                result[clue] = scored
    if result:
        return result
    if len(clue_set) == 1:
        scored = ratings(data)
        if scored:
            return {next(iter(clue_set)): scored}
    return {}


# ---------------------------------------------------------------------------
# Deterministic similarity (offline fallback + candidate prior)
# ---------------------------------------------------------------------------

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
# The agent
# ---------------------------------------------------------------------------

class AICodemaster(Codemaster):
    """Simulation-scored, risk-adjusted codemaster."""

    def __init__(self, team="Red", **kwargs):
        super(AICodemaster, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        self.opponent = "Blue" if self.team == "Red" else "Red"

        # A preset only supplies defaults; an explicit kwarg always wins, and
        # an unknown preset name degrades to the shipped configuration rather
        # than raising inside a live game.
        self.preset = str(kwargs.get("preset") or DEFAULT_PRESET)
        preset_values = PRESETS.get(self.preset, PRESETS[DEFAULT_PRESET])

        def option(name, default):
            if name in kwargs:
                return kwargs[name]
            return preset_values.get(name, default)

        self.provider = _resolve_provider(option("provider", None))
        self.base_url = option("base_url", None)
        self.model = _resolve_model(self.provider, kwargs.get("model"))
        self.deadline_s = float(kwargs.get("deadline", MOVE_DEADLINE_S))
        #: The non-cooperative backstop under ``deadline_s``; see
        #: ``MOVE_WALL_S``.  ``0`` disables it (tests that want the
        #: pipeline to run on the calling thread).
        self.move_wall_s = float(kwargs.get("move_wall", MOVE_WALL_S))
        self.n_candidates = int(kwargs.get("n_candidates", 12))
        self.max_simulated = int(kwargs.get("max_simulated", 8))
        self.panel_samples = int(kwargs.get("panel_samples", 2))
        self.max_targets = int(kwargs.get("max_targets", 4))
        self.verbose = bool(kwargs.get("verbose", False))
        #: Diagnostics are ON unless asked otherwise -- the organisers wanted to
        #: be able to see what the agent is doing.  ``quiet`` (kwarg) or
        #: ``OBIRDY_QUIET=1`` silences everything except fallback warnings.
        self.quiet = bool(kwargs.get("quiet", _quiet_default()))

        # Tunable risk model (swept offline; see harness/reports).
        self.majority_fraction = float(
            option("majority_fraction", MAJORITY_FRACTION))
        self.bonus_guess_weight = float(
            option("bonus_guess_weight", BONUS_GUESS_WEIGHT))
        self.civilian_penalty = float(
            option("civilian_penalty", CIVILIAN_PENALTY))
        self.claimed_slack = int(option("claimed_slack", CLAIMED_SLACK))
        self.assassin_presence = float(
            option("assassin_presence", ASSASSIN_PRESENCE_PENALTY))
        self.opponent_presence = float(
            option("opponent_presence", OPPONENT_PRESENCE_PENALTY))

        # Race awareness: duel-only, and structurally inert anywhere the
        # opposing team never moves.  ``_race_*`` are the values scoring
        # actually reads; until a duel escalates they are the three above.
        self.race_mode = bool(option("race_mode", _race_default()))
        self.race_full_deficit = float(option("race_full_deficit",
                                              RACE_FULL_DEFICIT))
        self.race_slack_scale = float(option("race_slack_scale",
                                             RACE_SLACK_SCALE))
        self._race = None
        self._race_log = []
        self._race_bonus = self.bonus_guess_weight
        self._race_civilian = self.civilian_penalty
        self._race_slack = self.claimed_slack

        # Danger probe.  On by default as of Pidgeot: clue safety, not clue
        # ambition, is what the evidence says is losing games.
        self.probe_enabled = bool(option("probe", True))
        self.probe_top_k = int(option("probe_top_k", PROBE_TOP_K))
        self.probe_veto_score = float(option("probe_veto_score",
                                             PROBE_VETO_SCORE))
        self.probe_veto_penalty = float(option("probe_veto_penalty",
                                               PROBE_VETO_PENALTY))
        self.probe_relative_veto = bool(option("probe_relative_veto", True))
        self.probe_relative_floor = float(option("probe_relative_floor",
                                                 PROBE_RELATIVE_FLOOR))
        self.probe_veto_requires_embed = bool(
            option("probe_veto_requires_embed", PROBE_VETO_REQUIRES_EMBED))
        self.probe_assassin_weight = float(option("probe_assassin_weight",
                                                  PROBE_ASSASSIN_WEIGHT))
        self.probe_opponent_weight = float(option("probe_opponent_weight",
                                                  PROBE_OPPONENT_WEIGHT))
        self.probe_missing_penalty = float(option("probe_missing_penalty",
                                                  PROBE_MISSING_PENALTY))
        self.probe_unprobed_cap = int(option("probe_unprobed_cap",
                                             PROBE_UNPROBED_CAP))
        self.probe_max_opponent = int(option("probe_max_opponent",
                                             PROBE_MAX_OPPONENT))
        self.probe_reference_words = int(option("probe_reference_words",
                                                PROBE_REFERENCE_WORDS))
        self.probe_number_cap_score = float(
            option("probe_number_cap_score", PROBE_NUMBER_CAP_SCORE))
        self.probes_run = 0
        self.probes_vetoed = 0
        #: clues kept but narrowed to one word by ``PROBE_NUMBER_CAP_SCORE``
        self.probe_number_capped = 0
        #: per-candidate probe detail, so a lost game can be diagnosed offline
        #: without re-running it.  Empty whenever the probe is off.
        self.probe_log = []

        # Embedding danger sensor: free, offline, and blind in exactly the
        # places the LLM probe can see.
        self.embed_sensor = bool(option("embed_sensor", EMBED_SENSOR))
        self.embed_margin = float(option("embed_margin", EMBED_MARGIN))
        self.embed_penalty = float(option("embed_penalty", EMBED_PENALTY))
        self.embed_oov_penalty = float(option("embed_oov_penalty",
                                              EMBED_OOV_PENALTY))
        self.simtable_path = option("simtable_path", None)
        self.embed_flagged = 0
        self.embed_oov = 0
        self.embed_log = []
        #: Clues the sensor has flagged this game.  The probe reads it for
        #: corroboration and the probe log records it either way, so a battery
        #: can tell an agreed veto from a probe-only one without a re-run.
        self._embed_flagged_clues = set()

        #: Score the top-N candidates with their own panel call instead of one
        #: batched call for all of them.  A batched call lets a strong clue's
        #: obvious answer bleed into a weaker clue's ranking; isolating the
        #: contenders removes that contamination at N extra calls per sample.
        self.panel_isolated = int(option("panel_isolated", 0))

        # Endgame sweep: clue number 0 means "unlimited guesses" (framework
        # README).  Used only when the panel plus the guesser's leftover
        # memory demonstrably covers every own word left and the risk gate
        # below is clean -- an unlimited clue with a stray assassin is a loss.
        #
        # Off by default as of Pidgeotto.  Once the guesser's sweep-turn
        # confidence floor was raised to 0.90 the gate fired in 1 of 12 slang
        # games and a paired A/B came back exactly level (8.92 both arms, same
        # assassin rate).  An unlimited clue is the highest-variance move we
        # can make, so a measured gain of zero is not worth its tail; the
        # machinery stays behind this flag rather than being deleted, and the
        # guesser still handles a foreign codemaster's 0 either way.
        self.allow_sweep = bool(option("sweep", False))
        self.sweep_max_words = int(option("sweep_max_words", 3))
        self.sweeps_issued = 0

        self.llm = _LLM(self.model,
                        max_retries=int(kwargs.get("max_retries", 2)),
                        timeout=float(kwargs.get("call_timeout", CALL_TIMEOUT_S)),
                        provider=self.provider,
                        base_url=self.base_url,
                        api_key=kwargs.get("api_key"),
                        warn=self._warn)

        self.words = []
        self.maps = []
        self.move_history = []

        #: turn counter, for the per-clue diagnostics line
        self._turn = 0
        #: which branch of the pipeline produced the last clue
        self._turn_path = "panel"
        self._turn_candidates = 0

        #: own words we have already pointed at, so leftovers can be re-clued
        self._clued_targets = set()
        #: clues we have already given this game -- never give one twice
        self._issued = set()
        #: (board signature, clue) -> panel rankings, valid within a board state
        self._panel_cache = {}
        self._rng = random.Random(0xC0DE ^ hash(self.team) % 100000)

        self._announce()

    # -- diagnostics -------------------------------------------------------

    def _say(self, message):
        """A diagnostics line, suppressed by quiet mode."""
        if not self.quiet:
            _emit("codemaster(%s) %s" % (self.team, message))

    def _warn(self, message):
        """A warning.  Printed even in quiet mode -- that is the whole point."""
        _emit("WARNING: %s" % message)

    def _announce(self):
        """Say what this process actually loaded, before a game depends on it.

        Four facts, because these are the four ways a submission silently turns
        into a much worse agent: wrong build, wrong model, no key, no data file.
        A missing key is a warning rather than a note: it is the difference
        between our agent and a letter-overlap heuristic.
        """
        self._say("init: version=%s model=%s preset=%s provider=%s race=%s"
                  % (AGENT_VERSION, self.model, self.preset, self.provider,
                     "on" if self.race_mode else "off"))

        source = self.llm.key_source()
        if source:
            self._say("api key: found via %s (value never printed)" % source)
        else:
            self._warn("no API key found (checked the api_key kwarg, %s, and "
                       "the embedded HARDCODED_API_KEY) -- every clue will "
                       "come from the offline fallback" % KEY_ENV)

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

        if not self.embed_sensor:
            self._say("similarity table: sensor disabled")
            return
        table = self._simtable()
        if table is None:
            self._say("similarity table: NOT FOUND (searched %s) -- the "
                      "embedding danger sensor is off"
                      % ", ".join(_SimTable.search_paths(self.simtable_path)))
        else:
            self._say("similarity table: loaded from %s (format %d, %d clues, "
                      "%d board words)" % (table.path or "an explicit path",
                                           table.version,
                                           len(table.clue_words),
                                           len(table.board_words)))

    def _log_turn(self, result, before):
        """One line per clue: what came out, and what the safety nets did."""
        if self.quiet:
            return
        probes = self.probes_run - before[0]
        vetoed = self.probes_vetoed - before[1]
        flagged = self.embed_flagged - before[2]
        oov = self.embed_oov - before[3]
        race = ""
        if self._race is not None:
            # Duel only: in a single-team game this line is byte-for-byte what
            # it was before race awareness existed.
            race = (" | race %+d turns (pace %.2f vs %.2f) escalation %.2f"
                    % (self._race["deficit"], self._race["own_pace"],
                       self._race["opp_pace"], self._race["scale"]))
        self._say("clue %d: %s %s | path=%s candidates=%d | probe %d run, %d "
                  "vetoed | sensor %d flagged, %d out-of-vocabulary%s"
                  % (self._turn, result[0], result[1], self._turn_path,
                     self._turn_candidates, probes, vetoed, flagged, oov,
                     race))

    # -- framework hooks ---------------------------------------------------

    def set_game_state(self, words, maps):
        self.words = list(words)
        self.maps = list(maps)

    def set_move_history(self, move_history):
        self.move_history = list(move_history or [])

    def get_clue(self):
        """Return ``[clue, number]``.  Never raises, never illegal."""
        self._turn += 1
        self._turn_path = "panel"
        self._turn_candidates = 0
        before = (self.probes_run, self.probes_vetoed,
                  self.embed_flagged, self.embed_oov)
        try:
            result, finished = _run_bounded(self._get_clue_inner,
                                            self.move_wall_s)
            if not finished:
                # The pipeline is still blocked somewhere it cannot be
                # interrupted.  Stop waiting for it and answer offline.
                self._turn_path = "wall-fallback"
                self._warn(WALL_WARNING % ("get_clue", self.move_wall_s))
                result = self._fallback_clue()
        except BaseException as exc:  # noqa: BLE001 - a crash here is a disqualification
            self._turn_path = "crash-fallback"
            self._warn("clue pipeline raised (%s: %s) -- using offline fallback"
                       % (type(exc).__name__, _brief(exc)))
            try:
                result = self._fallback_clue()
            except BaseException:
                result = ["SIGNAL", 1]
        try:
            self._issued.add(_normalise(result[0]))
        except Exception:  # noqa: BLE001 - bookkeeping must never break a game
            pass
        try:
            self._log_turn(result, before)
        except Exception:  # noqa: BLE001 - diagnostics must never break a game
            pass
        return result

    # -- pipeline ----------------------------------------------------------

    def _get_clue_inner(self):
        deadline = _Deadline(self.deadline_s)
        own, opp, civ, assassin = self._split_board()
        self._race_update(own, opp)

        if not own:
            self._turn_path = "offline-fallback (no words left)"
            return self._fallback_clue()
        if len(own) == 1 and not self.llm.available():
            self._turn_path = "offline-fallback (no LLM)"
            return self._fallback_clue()

        leftovers = [w for w in own if _normalise(w) in self._clued_targets]

        candidates = []
        if self.llm.available() and not deadline.expired(reserve=6.0):
            candidates = self._brainstorm(own, opp, civ, assassin, leftovers,
                                          deadline)
        candidates = self._filter_candidates(candidates, own)

        # Always keep a couple of deterministic options in play so the panel
        # has something legal to score even if the brainstorm came back empty.
        for clue in self._heuristic_candidates(limit=3):
            if all(clue != existing[0] for existing in candidates):
                candidates.append((clue, []))

        if not candidates:
            self._turn_path = "offline-fallback (no legal candidate)"
            return self._fallback_clue()

        simulate = candidates[:self.max_simulated]
        self._turn_candidates = len(simulate)
        panel = {}
        if self.llm.available() and not deadline.expired(reserve=5.0):
            panel = self._simulate(simulate, deadline)

        if not panel:
            return self._unsimulated_clue(simulate, own, opp, assassin,
                                          deadline)

        scored = []
        for clue, targets in simulate:
            rankings = panel.get(clue)
            if not rankings:
                continue
            score, number = self._score_candidate(rankings, own, opp, civ,
                                                  assassin, len(targets))
            # The sensor is free, so it prices every candidate before anything
            # decides which one is worth an API call.
            score -= self._embedding_penalty(clue, rankings, own, assassin,
                                             number, targets)
            scored.append((clue, targets, rankings, score, number))

        chosen = None
        if scored and self.probe_enabled:
            chosen = self._probe_select(scored, own, opp, assassin, deadline)

        if chosen is not None:
            self._turn_path = "panel+probe"
            clue, number = chosen
        else:
            self._turn_path = "panel (unprobed)"
            best = None
            for clue, targets, rankings, score, number in scored:
                key = (score, number, clue)
                if best is None or key > best[0]:
                    best = (key, clue, number)
            if best is None:
                clue, targets = simulate[0]
                return [clue, max(1, min(2, len(targets) or 1, len(own)))]
            _, clue, number = best
        number = max(1, min(int(number), len(own)))

        # Remember what we pointed at so leftovers survive into later turns.
        chosen_targets = dict(simulate).get(clue) or []
        for target in chosen_targets[:number]:
            self._clued_targets.add(_normalise(target))
        # ``panel[clue]`` is one ordering per sample, not one flat list; the
        # words we actually pointed at are the leading picks of each ordering.
        for order in panel.get(clue) or []:
            for word in order[:number]:
                if self._colour(word) == self.team:
                    self._clued_targets.add(_normalise(word))

        if self.allow_sweep:
            number = self._maybe_sweep(number, panel.get(clue) or [],
                                       own, opp, assassin)

        if self.verbose:
            sys.stderr.write("[oBirdy CM] %s %d\n" % (clue, number))
        return [clue, number]

    # -- the no-panel path -------------------------------------------------

    def _unsimulated_clue(self, simulate, own, opp, assassin, deadline):
        """Choose a clue with no panel behind it -- safety nets still armed.

        The panel is the only stage of the pipeline that needs the API to be
        healthy.  The embedding sensor does not: it is a table lookup, it is
        free, and it sees the whole board.  Skipping it here is what the
        organisers' run cost us.  With their SDK unable to return any text every
        turn arrived on this branch, took ``simulate[0]`` unexamined, and
        ``TOOL 1`` walked the guesser into ``KNIFE``.  The sensor scores that
        exact pair at 0.328 on the assassin against 0.211 on ``BELT``, the only
        own word ``TOOL`` knows -- a flag by 0.117, well past the 0.02 margin.
        It was never asked.

        So: price every candidate with the sensor, prefer one it does not flag,
        cap a flagged survivor at a single word, and let the probe confirm the
        winner when the clock and the API allow (it costs one call, and a dead
        API lands on "no answer", which is the conservative handling and not a
        pass).  Ordering is the tie-break, so a board where nothing is flagged
        picks exactly what this branch picked before.
        """
        if not simulate:               # unreachable via the pipeline; cheap
            return self._fallback_clue()
        offline = not self.llm.available()
        if offline:
            self._turn_path = "offline-fallback (no LLM)"
            self._warn("no LLM available -- this clue comes from the "
                       "deterministic offline vocabulary")
        else:
            self._turn_path = "unsimulated (panel returned nothing)"

        scored = []
        for index, (clue, targets) in enumerate(simulate):
            number = max(1, min(2, len(targets) or 1, len(own)))
            penalty = self._embedding_penalty(clue, None, own, assassin,
                                              number, targets)
            if penalty >= self.embed_penalty and number > 1:
                # The flag may be about the second word only, which is a
                # different question from whether the clue is safe at all.
                narrowed = self._embedding_penalty(clue, None, own, assassin,
                                                   1, targets)
                if narrowed < self.embed_penalty:
                    number, penalty = 1, narrowed
            # The incoming order is the brainstorm's own preference and is the
            # tie-break, so a board the sensor likes throughout picks exactly
            # what this branch picked before it learned to look.  It rides in
            # the score because the probe re-sorts on that and nothing else.
            scored.append((clue, list(targets), None,
                           -penalty - UNSIMULATED_ORDER_STEP * index, number))
        scored.sort(key=lambda item: -item[3])

        if self.probe_enabled and not offline:
            chosen = self._probe_select(scored, own, opp, assassin, deadline)
            if chosen is not None:
                self._turn_path += " +sensor+probe"
                clue, number = chosen
                return [clue, self._flagged_cap(clue, number, own)]

        self._turn_path += " +sensor"
        clue, _, _, _, number = scored[0]
        return [clue, self._flagged_cap(clue, number, own)]

    def _flagged_cap(self, clue, number, own):
        """One word only for a clue the sensor flagged, whatever else said so.

        On the panel path a flag is a 6.0 score penalty and the competition
        between candidates does the rest.  Here there is no competition to lose:
        if every candidate is flagged one of them still has to be given, and the
        least it can do is ask for a single word.
        """
        number = max(1, min(int(number), len(own)))
        if clue in self._embed_flagged_clues:
            return 1
        return number

    # -- endgame sweep -----------------------------------------------------

    def _maybe_sweep(self, number, rankings, own, opp, assassin):
        """Return 0 (unlimited) when the board can safely be cleared now.

        Requirements, all of them:

        * only a handful of our words are left;
        * the clue we are about to give does *not* already cover them all
          (otherwise a plain number is strictly safer and just as good);
        * every own word the clue misses was already targeted by an earlier
          clue, so a guesser with leftover memory still knows it;
        * no assassin anywhere near the top of any panel sample, and no
          opponent word inside the sweep window.
        """
        own_left = len(own)
        if own_left < 2 or own_left > self.sweep_max_words:
            return number
        if number >= own_left or not rankings:
            return number

        covered = set()
        for order in rankings:
            for word in order[:number]:
                covered.add(_normalise(word))
        leftovers = set(w for w in self._clued_targets)
        missing = set(_normalise(w) for w in own) - covered - leftovers
        if missing:
            return number

        assassin_set = set(assassin)
        opponent_set = set(opp)
        for order in rankings:
            if any(word in assassin_set for word in order[:own_left + 1]):
                return number
            if any(word in opponent_set for word in order[:own_left]):
                return number

        self.sweeps_issued += 1
        for word in own:
            self._clued_targets.add(_normalise(word))
        return 0

    # -- board -------------------------------------------------------------

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

    def _colour(self, word):
        try:
            index = self.words.index(word)
        except ValueError:
            return None
        if index >= len(self.maps):
            return None
        return self.maps[index]

    def _unrevealed(self):
        return [w for w in self.words if not _is_revealed(w)]

    def _issued_clues(self):
        """Clues our team has already given this game.

        Repeating one is never right.  The guesser rebuilds the same ranking
        from the same clue, and its top pick has already been taken off the
        board, so the repeat walks it one step further down a list we chose
        precisely because we wanted its head.  In the recorded battery a
        repeated clue showed up in 38 of 329 games and 14 of those ended on
        the assassin (36.8%) against 6.5% for games with no repeat; both
        assassin deaths our codemaster owns in the Abra cross-pairing
        (``sweep_obirdy_cm_abra_g`` seeds 2 and 6) were repeat games, seed 6
        dying to the second ``NAIL 2``.

        The pathology is worst with the API down, where the deterministic
        fallback is a pure function of the board and re-emits its own top
        choice every turn -- ``TOOL`` six times in one recorded game.

        ``move_history`` is authoritative (it survives anything our own state
        does not); the local set is a belt-and-braces copy.
        """
        issued = set(self._issued)
        prefix = "%s_Codemaster" % self.team
        for move in self.move_history or ():
            if move and str(move[0]) == prefix and len(move) >= 2:
                clue = _normalise(move[1])
                if clue:
                    issued.add(clue)
        return issued

    def _two_team_mode(self):
        """True once the opposing team has visibly moved (two-team track)."""
        for move in self.move_history or ():
            if move and str(move[0]).startswith(self.opponent):
                return True
        return False

    # -- race awareness ----------------------------------------------------

    def _race_update(self, own, opp):
        """Recompute this turn's race state and the knobs it escalates.

        Called once per clue, before anything is scored.  Three ways out, and
        all three land on the shipped conservative values:

        * ``race_mode`` off;
        * the opponent has never moved -- a single-team game, or our own
          opening clue, where there is no race to read;
        * we are level or ahead on the projection, which is the common case
          and the one the safety-first pricing was built for.
        """
        self._race = None
        self._race_bonus = self.bonus_guess_weight
        self._race_civilian = self.civilian_penalty
        self._race_slack = self.claimed_slack
        if not self.race_mode:
            return
        state = race_state(self.move_history, self.team, len(own), len(opp),
                           full_deficit=self.race_full_deficit)
        if state is None:
            return
        bonus, civilian, slack = race_knobs(
            state["scale"], self.bonus_guess_weight, self.civilian_penalty,
            self.claimed_slack, slack_scale=self.race_slack_scale)
        self._race_bonus = bonus
        self._race_civilian = civilian
        self._race_slack = slack
        state = dict(state)
        state["turn"] = self._turn
        state["bonus_guess_weight"] = round(bonus, 3)
        state["civilian_penalty"] = round(civilian, 3)
        state["claimed_slack"] = slack
        self._race = state
        self._race_log.append(state)

    # -- step 2: brainstorm ------------------------------------------------

    def _brainstorm(self, own, opp, civ, assassin, leftovers, deadline):
        system = (
            "You are a world-class Codenames spymaster. You give single-word "
            "clues that a stranger with no knowledge of the key could decode. "
            "You are extremely risk-averse about the assassin word."
        )
        lines = [
            "YOUR words (you want these guessed): %s" % ", ".join(own),
            "OPPONENT words (never hint at these): %s" % (", ".join(opp) or "none"),
            "NEUTRAL bystanders (avoid): %s" % (", ".join(civ) or "none"),
            "ASSASSIN word (instant loss if guessed): %s" % (", ".join(assassin) or "none"),
        ]
        if leftovers:
            lines.append("Still unfound from your earlier clues: %s"
                         % ", ".join(leftovers))
        lines.append("")
        lines.append(
            "Propose %d candidate clues. Each clue must be ONE English word: "
            "no hyphens, no spaces, no proper nouns, not a board word, and it "
            "must not contain any board word as a substring nor be contained "
            "inside one. Each clue targets 1-%d of YOUR words. Prefer clues "
            "whose link to the targets is far stronger than to any opponent, "
            "neutral or assassin word. Include a mix: some safe 1-2 word "
            "clues, some ambitious 3-4 word clues."
            % (self.n_candidates, self.max_targets)
        )
        lines.append(
            'Respond with ONLY a JSON array, e.g. '
            '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
        )
        text = self.llm.chat(system, "\n".join(lines), max_tokens=900,
                             deadline=deadline)
        return _parse_candidates(text)

    def _filter_candidates(self, candidates, own):
        """Legality filter + de-duplication, ordered by an offline prior."""
        board = self.words
        own_set = set(_normalise(w) for w in own)
        seen = set(self._issued_clues())
        kept = []
        for clue, targets in candidates:
            clue = _normalise(clue)
            if not clue or clue in seen:
                continue
            if not clue_is_legal(clue, board):
                continue
            seen.add(clue)
            # Keep only targets that really are unrevealed own words.
            real = []
            for target in targets:
                if target in own_set:
                    matched = next(
                        (w for w in own if _normalise(w) == target), None)
                    if matched is not None and matched not in real:
                        real.append(matched)
            kept.append((clue, real[:self.max_targets]))

        # Prior: prefer more claimed targets, then a mild offline sanity score.
        def prior(item):
            clue, targets = item
            score = 1.2 * len(targets)
            for word in self._unrevealed():
                sim = _similarity(clue, word)
                kind = self._colour(word)
                if kind == self.team:
                    score += 0.3 * sim
                elif kind == "Assassin":
                    score -= 1.5 * sim
                elif kind == self.opponent:
                    score -= 0.4 * sim
                else:
                    score -= 0.2 * sim
            return -score

        kept.sort(key=prior)
        return kept

    def _heuristic_candidates(self, limit=3):
        """Best offline-vocabulary clues -- always legal, never empty-handed."""
        board = self.words
        issued = self._issued_clues()
        scored = []
        for clue in _FALLBACK_VOCAB:
            if clue in issued or not clue_is_legal(clue, board):
                continue
            total = 0.0
            for word in self._unrevealed():
                sim = _similarity(clue, word)
                kind = self._colour(word)
                if kind == self.team:
                    total += sim
                elif kind == "Assassin":
                    total -= 4.0 * sim
                elif kind == self.opponent:
                    total -= 0.7 * sim
                else:
                    total -= 0.3 * sim
            scored.append((total, clue))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [clue for _, clue in scored[:limit]]

    # -- step 4: simulate --------------------------------------------------

    def _simulate(self, candidates, deadline):
        """Sampled generic-guesser panel ranks the board for each candidate."""
        board = self._unrevealed()
        if not board:
            return {}
        clues = [clue for clue, _ in candidates]
        signature = (tuple(sorted(board)), tuple(clues))
        cached = self._panel_cache.get(signature)
        if cached:
            return cached

        samples = self.panel_samples
        remaining = deadline.remaining()
        if remaining < 14.0:
            samples = 1
        if remaining < 6.0:
            return {}

        top_k = min(6, len(board))
        groups = self._panel_groups(clues)
        jobs = []
        for sample in range(samples):
            shuffled = list(board)
            random.Random(1000 + sample).shuffle(shuffled)
            for group in groups:
                jobs.append(
                    self._panel_job(shuffled, group, top_k, sample, deadline))

        replies = _run_parallel(jobs, deadline, max_workers=min(8, len(jobs)))

        merged = {}
        for reply in replies:
            if not reply:
                continue
            parsed = _parse_panel(reply, set(clues), board)
            for clue, ordered in parsed.items():
                merged.setdefault(clue, []).append(ordered)
        self._panel_cache[signature] = merged
        return merged

    def _panel_groups(self, clues):
        """Split the candidates into the clue sets each panel call sees.

        One batched call is cheapest but lets a strong clue's obvious answer
        contaminate a weaker clue's ranking, because the model reads them all
        in one context.  ``panel_isolated`` gives that many leading candidates
        a call of their own and keeps the batched call for the tail.
        """
        isolated = max(0, min(self.panel_isolated, len(clues)))
        if isolated <= 0:
            return [list(clues)]
        groups = [[clue] for clue in clues[:isolated]]
        tail = list(clues[isolated:])
        if tail:
            groups.append(tail)
        return groups

    def _panel_job(self, board, clues, top_k, sample, deadline):
        system = (
            "You are an average Codenames field operative. You can see the "
            "board words but NOT which team they belong to. Given a clue you "
            "say which board words a typical player would pick, in order."
        )
        prompt = [
            "Board words still in play:",
            ", ".join(board),
            "",
            "For each clue below, list the %d board words most associated "
            "with it, most likely first. Judge each clue completely "
            "independently of the others." % top_k,
            "",
            "Clues: %s" % ", ".join(clues),
            "",
            'Respond with ONLY a JSON object mapping each clue to its ordered '
            'list, e.g. {"OCEAN": ["WHALE", "SHIP"]}',
        ]
        text = "\n".join(prompt)

        def job():
            return self.llm.chat(system, text, max_tokens=900, deadline=deadline)

        return job

    # -- step 5b: danger probe ---------------------------------------------

    def _probe_select(self, scored, own, opp, assassin, deadline):
        """Pick a clue, probing as few candidates as the answer allows.

        The leader is probed; if it survives it is the clue.  Only a veto is
        worth a second call, and only one: after ``probe_top_k`` candidates the
        next-best is taken *unprobed*, which is already the conservative
        handling (claimed-count ceiling, missing penalty) rather than a pass.

        Returns ``(clue, number)`` or ``None`` when the probe had nothing to
        say at all, in which case the caller falls back to plain scoring.
        """
        try:
            return self._probe_select_inner(scored, own, opp, assassin,
                                            deadline)
        except Exception:  # noqa: BLE001 - never lose a game to the safety net
            return None

    def _probe_select_inner(self, scored, own, opp, assassin, deadline):
        if not scored:
            return None
        order = sorted(scored, key=lambda item: (-item[3], item[0]))
        budget = max(0, int(self.probe_top_k))
        remaining = deadline.remaining() if deadline is not None else 999.0
        if remaining < PROBE_TIGHT_SECONDS:
            budget = min(budget, 1)

        used = 0
        vetoed = []
        for clue, targets, rankings, score, number in order:
            ratings = None
            if used < budget:
                used += 1
                ratings = self._probe_one(clue, rankings, own, opp, assassin,
                                          deadline)
            mark = len(self.probe_log)
            adjusted, adjusted_number = self._apply_probe(
                score, number, len(targets), ratings, opp, assassin,
                clue=clue)
            if len(self.probe_log) > mark and self.probe_log[-1]["veto"]:
                vetoed.append((adjusted, clue, adjusted_number))
                continue
            return clue, adjusted_number

        if vetoed:
            # Everything we looked at is poisoned: take the least-bad one, at
            # one word only.
            vetoed.sort(key=lambda item: (-item[0], item[1]))
            return vetoed[0][1], vetoed[0][2]
        return None

    def _probe_one(self, clue, rankings, own, opp, assassin, deadline):
        """One probe call for one candidate.  ``None`` means "no answer".

        A crash here must land on *no answer*, not on *no probe*: the caller
        then applies the conservative unprobed handling instead of quietly
        reverting to the optimistic pre-probe number.
        """
        try:
            return self._probe_one_inner(clue, rankings, own, opp, assassin,
                                         deadline)
        except Exception:  # noqa: BLE001
            return None

    def _probe_one_inner(self, clue, rankings, own, opp, assassin, deadline):
        board = self._unrevealed()
        if not board or not self.llm.available():
            return None
        remaining = deadline.remaining() if deadline is not None else 999.0
        if remaining < PROBE_MIN_SECONDS:
            return None
        danger = list(assassin) + list(opp)[:max(0, self.probe_max_opponent)]
        if not danger:
            return None

        words = list(danger)
        for word in self._probe_reference(rankings, own):
            if word not in words:
                words.append(word)
        random.Random(7717 + len(self.probe_log)).shuffle(words)

        reply = self._probe_job(board, [clue], words, deadline)()
        if not reply:
            return None
        ratings = _parse_probe(reply, {clue}, board).get(clue)
        if not ratings:
            return None
        self.probes_run += 1
        return ratings

    def _probe_reference(self, rankings, own):
        """Own words to mix into the probe list so it is not all danger.

        A list made only of the words we fear tells the model what we fear and
        invites a reassuring answer; these also give the assassin's rating
        something on the same scale to be compared against.
        """
        own_set = set(own)
        reference = []
        for order in rankings or ():
            for word in order:
                if word in own_set and word not in reference:
                    reference.append(word)
                if len(reference) >= max(0, self.probe_reference_words):
                    return reference
        for word in own:
            if word not in reference:
                reference.append(word)
            if len(reference) >= max(0, self.probe_reference_words):
                break
        return reference

    def _probe_job(self, board, clues, words, deadline):
        system = (
            "You are an average Codenames field operative. You can see the "
            "board words but NOT which team they belong to. You rate how "
            "strongly a clue points at particular board words, honestly and "
            "without hedging."
        )
        prompt = [
            "Board words still in play:",
            ", ".join(board),
            "",
        ]
        if len(clues) == 1:
            prompt.append("Clue: %s" % clues[0])
        else:
            prompt.append("Clues: %s" % ", ".join(clues))
        prompt.extend([
            "",
            "Rate from 0 to 10 how strongly the clue points at each of these "
            "board words, where 0 means no connection at all and 10 means it "
            "is the first word you would say: %s" % ", ".join(words),
            "",
        ])
        if len(clues) == 1:
            prompt.append(
                'Respond with ONLY a JSON object mapping each word to its '
                'rating, e.g. {"%s": 7}' % words[0])
        else:
            prompt.append(
                'Rate each clue independently of the others. Respond with '
                'ONLY a JSON object mapping each clue to its word ratings, '
                'e.g. {"%s": {"%s": 7}}' % (clues[0], words[0]))
        text = "\n".join(prompt)

        def job():
            return self.llm.chat(system, text, max_tokens=400,
                                 deadline=deadline)

        return job

    def _apply_probe(self, score, number, claimed, ratings, opp, assassin,
                     clue=None):
        """Fold a finalist's probe result into its score and its number.

        ``clue`` is recorded on the log entry and is otherwise unused.  The
        first Pidgeot validation battery logged ratings without it, so a veto
        could be counted but the word it killed could not be named, and
        judging a veto true or false positive afterwards was impossible.
        """
        if not ratings:
            return (score - self.probe_missing_penalty,
                    self._conservative_number(number, claimed))

        assassin_pull = 0.0
        for word in assassin:
            assassin_pull = max(assassin_pull, ratings.get(word, 0.0))

        opponent_pulls = [ratings[word] for word in opp if word in ratings]

        danger = set(assassin) | set(opp)
        reference = [value for word, value in ratings.items()
                     if word not in danger]

        # The comparison that matters is against the *marginal* word we are
        # asking the guesser to reach, not the clue's best one.  Every clue
        # points hard at its own headline word, so requiring the assassin to
        # out-rate that (``max``) made the relative veto structurally unable
        # to fire: ``sweep_pidgeot`` ran 9 probes and vetoed nothing, then
        # died on guess 2 of CLOCK 2 at PART.  A clue is safe only when the
        # assassin is clearly weaker than the weakest word we are counting on.
        margin = min(reference) if reference else None

        veto = assassin_pull >= self.probe_veto_score
        if not veto and self.probe_relative_veto and margin is not None:
            veto = (assassin_pull >= margin
                    and assassin_pull >= self.probe_relative_floor)
            # Optional corroboration: only let the *relative* veto fire when
            # the embedding sensor independently puts the assassin near this
            # clue.  Off by default -- see ``probe_veto_requires_embed``.  The
            # absolute veto is a ceiling, not a comparison, so it is never
            # gated on a second opinion.
            if veto and self.probe_veto_requires_embed:
                veto = (clue is not None
                        and clue in self._embed_flagged_clues)
        self.probe_log.append({
            "clue": clue,
            "assassin": round(assassin_pull, 2),
            "margin": None if margin is None else round(margin, 2),
            "opponent": [round(value, 2) for value in opponent_pulls],
            "embed_flagged": bool(clue is not None
                                  and clue in self._embed_flagged_clues),
            "veto": bool(veto),
        })
        if veto:
            self.probes_vetoed += 1
            return score - self.probe_veto_penalty, 1

        penalty = self.probe_assassin_weight * (assassin_pull / 10.0) ** 2
        if opponent_pulls:
            mean_opponent = sum(opponent_pulls) / float(len(opponent_pulls))
            penalty += self.probe_opponent_weight * (mean_opponent / 10.0)
        if assassin_pull >= self.probe_number_cap_score:
            # Survived the veto, but the probe still put the assassin near the
            # top of what this clue reaches.  Keep the clue and buy one word.
            # See ``PROBE_NUMBER_CAP_SCORE``: both deaths in the 2026-08-03
            # battery were the assassin taken as the second word of a two.
            self.probe_number_capped += 1
            self.probe_log[-1]["number_capped"] = True
            number = 1
        return score - penalty, number

    def _conservative_number(self, number, claimed):
        """The number to use when nobody vouched for the dangerous words.

        Un-does exactly the optimism the probe was meant to underwrite: the
        brainstorm's own claimed target count becomes a hard ceiling again, and
        a candidate that claimed nothing at all falls back to a small cap.
        """
        if claimed:
            return max(1, min(int(number), int(claimed)))
        return max(1, min(int(number), max(1, self.probe_unprobed_cap)))

    # -- step 5a: embedding danger sensor ----------------------------------

    def _simtable(self):
        if not self.embed_sensor:
            return None
        return _SimTable.shared(self.simtable_path)

    def _sensor_targets(self, rankings, own, number, claimed, pulls, floor):
        """The own words this clue's number is actually asking for.

        The panel's leading picks are the honest answer -- they are what a
        guesser was simulated to take.  The brainstorm's own claims come next.
        Failing both, the embedding's own best guesses stand in.  Capped at
        ``number``: a clue is only dangerous relative to the words it is
        actually asking for, and grading it against every own word on the
        board would fire on almost everything.
        """
        limit = max(1, int(number))
        own_set = set(own)
        targets = []
        for order in rankings or ():
            for word in order[:limit]:
                if word in own_set and word not in targets:
                    targets.append(word)
        for source in ((claimed or ()), sorted(
                own, key=lambda w: -pulls.get(w, floor))):
            if len(targets) >= limit:
                break
            for word in source:
                if word in own_set and word not in targets:
                    targets.append(word)
                if len(targets) >= limit:
                    break
        return targets[:limit]

    def _embedding_penalty(self, clue, rankings, own, assassin, number,
                           claimed=()):
        """Static-embedding danger for one candidate, or 0.0 when it is clean.

        Three outcomes, in the order they are checked:

        * the clue is not in the table's vocabulary -- a mild demotion, so an
          in-vocabulary candidate wins a tie.  ``LASTDRINK`` and ``EGGSHELL``
          both killed a game here: an embedding partner handed a word its
          vocabulary lacks falls back to letter overlap;
        * the clue pulls on the assassin at least as hard as on the weakest
          word the number is buying -- the full penalty;
        * anything else -- free.

        A board word missing from the lookup pulls *less* than the table's
        floor, which is an answer and not a gap: a target the embedding barely
        knows is exactly the target a stranger will not reach, so it counts as
        the floor rather than being quietly dropped from the comparison.
        """
        table = self._simtable()
        if table is None or not assassin:
            return 0.0
        pulls = table.pulls(clue, list(assassin) + list(own))
        if pulls is None:
            self.embed_oov += 1
            self.embed_log.append({"clue": clue, "oov": True})
            return self.embed_oov_penalty

        pull = None
        for word in assassin:
            value = pulls.get(word)
            if value is not None and (pull is None or value > pull):
                pull = value
        if pull is None:
            return 0.0            # the assassin is below the floor: no pull

        targets = self._sensor_targets(rankings, own, number, claimed, pulls,
                                       table.floor)
        reference = min([pulls.get(w, table.floor) for w in targets]
                        or [table.floor])

        if pull >= reference - self.embed_margin:
            self.embed_flagged += 1
            self._embed_flagged_clues.add(clue)
            self.embed_log.append({
                "clue": clue,
                "assassin": round(pull, 3),
                "reference": round(reference, 3),
            })
            return self.embed_penalty
        return 0.0

    # -- step 5: score -----------------------------------------------------

    def _score_candidate(self, rankings, own, opp, civ, assassin, claimed):
        """Risk-adjusted EV plus the number the panel actually supports."""
        own_set = set(own)
        opp_set = set(opp)
        assassin_set = set(assassin)
        opponent_penalty = (OPPONENT_PENALTY if self._two_team_mode()
                            else OPPONENT_PENALTY_SOLO)

        max_len = max(len(order) for order in rankings)
        cap = min(max_len, len(own), max(1, self.max_targets + 1))

        # Number: longest prefix that is all-own in a majority of samples.
        number = 1
        for k in range(1, cap + 1):
            good = 0
            for order in rankings:
                prefix = order[:k]
                if len(prefix) == k and all(w in own_set for w in prefix):
                    good += 1
            if good >= self.majority_fraction * len(rankings):
                number = k
            else:
                break
        if claimed:
            # The brainstorm's own target count is a sanity cap, but it is the
            # least verified number in the pipeline: the panel actually looked
            # at the board.  ``claimed_slack`` lets a panel-supported prefix
            # outrun a timid brainstorm by that many words.
            number = min(number, max(1, claimed + self._race_slack))

        total = 0.0
        for order in rankings:
            value = 0.0
            for position, word in enumerate(order[:number + 1]):
                weight = self._race_bonus if position >= number else 1.0
                if word in assassin_set:
                    value -= ASSASSIN_PENALTY * weight
                    break
                if word in own_set:
                    value += 1.0 * weight
                    continue
                if word in opp_set:
                    value -= opponent_penalty * weight
                else:
                    value -= self._race_civilian * weight
                break
            value -= self._presence_penalty(order, opp_set, assassin_set)
            total += value
        score = total / float(len(rankings))

        # Slight preference for consistency across the panel.
        firsts = set(order[0] for order in rankings if order)
        if len(firsts) == 1:
            score += 0.1
        return score, number

    def _presence_penalty(self, order, opp_set, assassin_set):
        """Cost of the assassin (or an opponent word) merely being ranked here.

        Independent of the expected-value walk above, which stops at the first
        mistake and therefore never sees a danger word sitting behind one.  A
        foreign guesser will not reproduce our panel's exact order, so anything
        it might plausibly reach counts -- weighted down by how far from the
        top of the ranking it sits.
        """
        penalty = 0.0
        for position, word in enumerate(order):
            decay = 1.0 / (position + PRESENCE_DECAY)
            if word in assassin_set:
                penalty += self.assassin_presence * decay
            elif word in opp_set:
                penalty += self.opponent_presence * decay
        return penalty

    # -- offline fallback --------------------------------------------------

    def _fallback_clue(self, allow_repeat=False):
        """Deterministic, API-free, always-legal clue.

        This path is a pure function of the board, so without the repeat
        filter a dead API turns the game into the same clue over and over.
        ``allow_repeat`` is the last resort for a board whose whole offline
        vocabulary has already been spent.

        Scoring is letter overlap, which is blind to meaning -- that is how the
        recorded game got six ``TOOL``s and how the organisers' run got one that
        pointed at the assassin.  The embedding sensor is not blind and costs
        nothing, so the offline vocabulary is ranked by overlap and then the
        first candidate the sensor does not flag is taken.  When it flags every
        one of them the best-scoring clue is still given, at a single word.
        """
        own, opp, civ, assassin = self._split_board()
        board = self.words or []
        issued = set() if allow_repeat else self._issued_clues()
        scored = []
        for clue in _FALLBACK_VOCAB:
            if clue in issued or not clue_is_legal(clue, board):
                continue
            total = 0.0
            hits = 0
            for word in self._unrevealed():
                sim = _similarity(clue, word)
                kind = self._colour(word)
                if kind == self.team:
                    total += sim
                    if sim >= 0.45:
                        hits += 1
                elif kind == "Assassin":
                    total -= 4.0 * sim
                elif kind == self.opponent:
                    total -= 0.7 * sim
                else:
                    total -= 0.3 * sim
            # ``index`` keeps the old strictly-greater comparison's tie-break,
            # which was vocabulary order.  Sorting alphabetically instead would
            # silently repick every tied board.
            scored.append((total, -len(scored), clue, hits))
        if not scored:
            for candidate in ("SIGNAL", "OBJECT", "SUBJECT", "TOPIC", "IDEA",
                              "THING", "MATTER"):
                if candidate not in issued and clue_is_legal(candidate, board):
                    return [candidate, 1]
            if not allow_repeat:
                return self._fallback_clue(allow_repeat=True)
            return ["SIGNAL", 1]

        scored.sort(reverse=True)
        best_clue = scored[0][2]
        for _, _, clue, hits in scored[:FALLBACK_SENSOR_TOP_K]:
            number = max(1, min(hits, max(1, len(own))))
            if self._embedding_penalty(clue, None, own, assassin, number,
                                       ()) < self.embed_penalty:
                return [clue, number]
        return [best_clue, 1]

    # -- diagnostics -------------------------------------------------------

    def usage_summary(self):
        summary = self.llm.usage_summary()
        summary["sweeps_issued"] = self.sweeps_issued
        summary["preset"] = self.preset
        summary["probes_run"] = self.probes_run
        summary["probes_vetoed"] = self.probes_vetoed
        summary["probe_number_capped"] = self.probe_number_capped
        summary["embed_table"] = self._simtable() is not None
        summary["embed_flagged"] = self.embed_flagged
        summary["embed_oov"] = self.embed_oov
        if self.probe_log:
            summary["probe_log"] = list(self.probe_log)
        if self.embed_log:
            summary["embed_log"] = list(self.embed_log)
        if self._race_log:
            # Only a duel that actually read the race adds these keys, so a
            # single-team summary is byte-identical to the pre-race build.
            summary["race_mode"] = True
            summary["race_log"] = list(self._race_log)
            summary["race_escalated"] = sum(
                1 for state in self._race_log if state["scale"] > 0.0)
        return summary
