"""Offline unit tests for the Silvally challenger agents.

No network, no API key, no shipped data file: every LLM call is monkeypatched
at the agents' own ``_LLM.chat`` boundary, and every similarity lookup runs
against a synthetic table this file writes itself with its own independent
encoder.  Nothing here depends on ``obirdy_simtable_v2.bin.gz`` staying the
size or shape it is today.

What is actually pinned:

* the density measurement, on two hand-built boards whose answers are known by
  construction (a tangled board and a clean one);
* ``board_scale`` monotonicity -- more danger never buys more ambition, more
  joint pairs never buys less;
* the ceiling -- no route through the morphing can produce a knob value past
  the champion's parked ``ambitious`` preset;
* champion identity -- on a board that does not earn ambition, every knob the
  fork touches equals the value ``codemaster_obirdy`` ships.

Run from the repo root::

    python -m pytest harness/test_silvally.py -q
    python -m unittest harness.test_silvally -v
"""

import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from players import codemaster_obirdy as champion_cm  # noqa: E402
from players import codemaster_silvally as cm_mod  # noqa: E402
from players import guesser_obirdy as champion_g  # noqa: E402
from players import guesser_silvally as g_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEVELS = [round(i / 63.0, 5) for i in range(64)]


def write_simtable(path, board_words, rows, version=2):
    """Write a similarity table in the shipped format.

    ``rows`` is ``{clue: {board word: cosine}}``.  This is an independent
    encoder, not a call into the builder: the reader is being tested against
    the format, not against whatever the builder emits today.
    """
    board = [w.upper() for w in board_words]
    columns = dict((w, i) for i, w in enumerate(board))
    lowered = dict((clue.lower(), values) for clue, values in rows.items())
    clues = sorted(lowered)
    offsets = [0]
    entry_columns = []
    entry_codes = []
    for clue in clues:
        for word, value in sorted(lowered[clue].items()):
            code = min(range(len(LEVELS)), key=lambda i: abs(LEVELS[i] - value))
            entry_columns.append(columns[word.upper()])
            entry_codes.append(code)
        offsets.append(len(entry_columns))

    header = {"format": "obirdy-simtable-%d" % version, "version": version,
              "floor": 0.12, "top_k": 64, "code_bits": 6,
              "n_board": len(board), "n_clue": len(clues),
              "n_entries": len(entry_columns), "levels": LEVELS}
    blob = bytearray(b"OBSIM%d\n" % version)
    blob += json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n"
    blob += ("\n".join(board) + "\n").encode("utf-8")
    blob += ("\n".join(clues) + "\n").encode("utf-8")
    for value in offsets:
        blob += int(value).to_bytes(4, "little")
    if version == 1:
        for column, code in zip(entry_columns, entry_codes):
            blob += ((int(column) << 6) | int(code)).to_bytes(2, "little")
    else:
        for column in entry_columns:
            blob += int(column).to_bytes(2, "little")
        blob += bytes(int(code) for code in entry_codes)
    with gzip.open(path, "wb") as handle:
        handle.write(bytes(blob))
    return path


#: A board built to be *clean*: our four words have no measurable tie to
#: anything that is not ours, the assassin has no tie to anything of ours, and
#: two dedicated clue words each reach a pair of our words well above the
#: 0.33 joint-clue floor.
CLEAN_BOARD = ["WHALE", "SHIP", "PIANO", "FLUTE",
               "TRUCK", "ROAD", "BREAD", "SOUP", "KNIFE"]
CLEAN_OWN = ["WHALE", "SHIP", "PIANO", "FLUTE"]
CLEAN_OPP = ["TRUCK", "ROAD"]
CLEAN_CIV = ["BREAD", "SOUP"]
CLEAN_ASSASSIN = ["KNIFE"]
CLEAN_ROWS = {
    # Board-word rows: nothing crosses colour, and the two own pairs sit only
    # a little above the floor so ``crowding`` stays low.
    "whale": {"SHIP": 0.16},
    "ship": {"WHALE": 0.16},
    "piano": {"FLUTE": 0.16},
    "flute": {"PIANO": 0.16},
    "truck": {"ROAD": 0.15},
    "road": {"TRUCK": 0.15},
    "bread": {"SOUP": 0.15},
    "soup": {"BREAD": 0.15},
    "knife": {"BREAD": 0.14},
    # Clue-word rows: two buyable pairs among our four words, plus one weak
    # cross-pair that only appears once the joint-clue floor is lowered.
    "ocean": {"WHALE": 0.55, "SHIP": 0.52},
    "music": {"PIANO": 0.58, "FLUTE": 0.56},
    "sea": {"WHALE": 0.20, "PIANO": 0.20},
}

#: The same shape of board built to be *tangled*: every one of our words has a
#: strong tie to a word that is not ours, and the assassin sits right next to
#: two of them.
TANGLED_BOARD = list(CLEAN_BOARD)
TANGLED_ROWS = {
    "whale": {"SHIP": 0.40, "TRUCK": 0.42, "KNIFE": 0.38},
    "ship": {"WHALE": 0.40, "ROAD": 0.44},
    "piano": {"FLUTE": 0.40, "BREAD": 0.41},
    "flute": {"PIANO": 0.40, "SOUP": 0.43, "KNIFE": 0.39},
    "truck": {"WHALE": 0.42, "ROAD": 0.45},
    "road": {"SHIP": 0.44, "TRUCK": 0.45},
    "bread": {"PIANO": 0.41, "SOUP": 0.40},
    "soup": {"FLUTE": 0.43, "BREAD": 0.40},
    "knife": {"WHALE": 0.38, "FLUTE": 0.39},
}


class _Table(object):
    """Context manager owning one synthetic table on disk."""

    def __init__(self, board, rows, version=2):
        self.board = board
        self.rows = rows
        self.version = version
        self.dir = None
        self.path = None

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="silvally-simtable-")
        self.path = os.path.join(self.dir, "table.bin.gz")
        write_simtable(self.path, self.board, self.rows, version=self.version)
        # The loader caches by path; a fresh path per test keeps them isolated.
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})
        return self.path

    def __exit__(self, *exc):
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
        return False


def make_codemaster(path=None, **kwargs):
    """A quiet, offline codemaster pointed at a given table (or none)."""
    options = {"quiet": True, "simtable_path": path, "move_wall": 0.0}
    options.update(kwargs)
    return cm_mod.AICodemaster("Red", **options)


class _PatchedLLM(object):
    """Scripted stand-in for ``_LLM.chat`` in either agent module."""

    def __init__(self, module, responder, available=True):
        self.module = module
        self.responder = responder
        self.available = available
        self.calls = None

    def __enter__(self):
        self._chat = self.module._LLM.chat
        self._available = self.module._LLM.available
        calls = []

        def chat(inner, system, user, max_tokens=600, deadline=None):
            calls.append({"system": system, "user": user})
            return self.responder(system, user)

        self.module._LLM.chat = chat
        self.module._LLM.available = lambda inner: self.available
        self.calls = calls
        return calls

    def __exit__(self, *exc):
        self.module._LLM.chat = self._chat
        self.module._LLM.available = self._available
        return False


# ---------------------------------------------------------------------------
# The ramp
# ---------------------------------------------------------------------------

class TestNormRamp(unittest.TestCase):

    def test_clamps_at_both_ends(self):
        self.assertEqual(cm_mod.norm_ramp(0.0, 0.2, 0.4), 0.0)
        self.assertEqual(cm_mod.norm_ramp(0.2, 0.2, 0.4), 0.0)
        self.assertEqual(cm_mod.norm_ramp(0.4, 0.2, 0.4), 1.0)
        self.assertEqual(cm_mod.norm_ramp(9.9, 0.2, 0.4), 1.0)
        self.assertAlmostEqual(cm_mod.norm_ramp(0.3, 0.2, 0.4), 0.5)

    def test_degenerate_ramp_is_a_step_not_a_crash(self):
        self.assertEqual(cm_mod.norm_ramp(0.5, 0.4, 0.4), 1.0)
        self.assertEqual(cm_mod.norm_ramp(0.3, 0.4, 0.4), 0.0)

    def test_junk_reads_as_zero(self):
        self.assertEqual(cm_mod.norm_ramp(None, 0.2, 0.4), 0.0)
        self.assertEqual(cm_mod.norm_ramp("x", 0.2, 0.4), 0.0)

    def test_guesser_carries_the_same_ramp(self):
        for value in (0.0, 0.19, 0.25, 0.4, 1.0):
            self.assertAlmostEqual(cm_mod.norm_ramp(value, 0.2, 0.4),
                                   g_mod.norm_ramp(value, 0.2, 0.4))


# ---------------------------------------------------------------------------
# The joint-pair scan
# ---------------------------------------------------------------------------

class TestJointPairs(unittest.TestCase):

    def test_finds_the_pairs_a_clue_reaches(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            table = cm_mod._SimTable.load(path)
            pairs, covered = table.joint_pairs(CLEAN_OWN, 0.33)
            self.assertEqual(covered, sorted(CLEAN_OWN))
            self.assertEqual(pairs, set([("SHIP", "WHALE"),
                                         ("FLUTE", "PIANO")]))

    def test_floor_is_respected(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            table = cm_mod._SimTable.load(path)
            # 0.60 is above every stored reading: nothing is buyable.
            self.assertEqual(table.joint_pairs(CLEAN_OWN, 0.60)[0], set())
            # Lowering the floor can only ever admit more pairs, and the weak
            # ``sea`` row is the one that appears when it drops.
            counts = [len(table.joint_pairs(CLEAN_OWN, floor)[0])
                      for floor in (0.60, 0.50, 0.33, 0.19, 0.13)]
            self.assertEqual(counts, sorted(counts))
            self.assertIn(("PIANO", "WHALE"),
                          table.joint_pairs(CLEAN_OWN, 0.19)[0])
            self.assertNotIn(("PIANO", "WHALE"),
                             table.joint_pairs(CLEAN_OWN, 0.33)[0])

    def test_out_of_vocabulary_words_leave_the_denominator(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            table = cm_mod._SimTable.load(path)
            pairs, covered = table.joint_pairs(
                CLEAN_OWN + ["XENOMORPH", "HOGWARTS"], 0.33)
            self.assertEqual(covered, sorted(CLEAN_OWN))
            self.assertEqual(len(pairs), 2)

    def test_one_word_has_no_pairs(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            table = cm_mod._SimTable.load(path)
            self.assertEqual(table.joint_pairs(["WHALE"], 0.33),
                             (set(), ["WHALE"]))
            self.assertEqual(table.joint_pairs([], 0.33), (set(), []))

    def test_both_format_versions_agree(self):
        results = []
        for version in (1, 2):
            with _Table(CLEAN_BOARD, CLEAN_ROWS, version=version) as path:
                table = cm_mod._SimTable.load(path)
                self.assertEqual(table.version, version)
                results.append(table.joint_pairs(CLEAN_OWN, 0.33))
        self.assertEqual(results[0], results[1])

    def test_slow_path_matches_the_typed_view(self):
        """``_index_view`` is an optimisation and must change no answer."""

        class _NoView(cm_mod._SimTable):
            __slots__ = ()

            def _index_view(self):
                return None

        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            fast = cm_mod._SimTable.load(path).joint_pairs(CLEAN_OWN, 0.33)
            slow = _NoView.load(path).joint_pairs(CLEAN_OWN, 0.33)
        self.assertEqual(fast, slow)


# ---------------------------------------------------------------------------
# The density measurement
# ---------------------------------------------------------------------------

class TestDensity(unittest.TestCase):

    def _measure(self, path, rows_own=None):
        agent = make_codemaster(path)
        return agent, agent._measure_density(rows_own or CLEAN_OWN, CLEAN_OPP,
                                             CLEAN_CIV, CLEAN_ASSASSIN)

    def test_clean_board_reads_clean(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent, density = self._measure(path)
        self.assertTrue(density["measured"])
        self.assertEqual(density["coverage"], 1.0)
        self.assertEqual(density["own_left"], 4)
        self.assertEqual(density["unrevealed"], 9)
        # Every tie is at or just above the 0.12 floor.
        self.assertLess(density["crowding"], 0.20)
        # Nothing of ours touches anything that is not ours.
        self.assertLess(density["collision"], 0.14)
        # The assassin's only stored tie is to a civilian.
        self.assertLess(density["assassin"], 0.14)
        # Two buyable pairs out of six.
        self.assertEqual(density["pairs"], 2)
        self.assertEqual(density["pair_total"], 6)
        self.assertAlmostEqual(density["pair_supply"], 0.333, places=2)

    def test_tangled_board_reads_tangled(self):
        with _Table(TANGLED_BOARD, TANGLED_ROWS) as path:
            agent, density = self._measure(path)
        self.assertTrue(density["measured"])
        self.assertGreater(density["crowding"], 0.38)
        self.assertGreater(density["collision"], 0.38)
        self.assertGreater(density["assassin"], 0.35)
        # At most the one pair its own board rows happen to supply, well under
        # the clean board's, and irrelevant either way: the danger readings
        # have already withdrawn the licence.
        self.assertLessEqual(density["pairs"], 1)
        self.assertEqual(cm_mod.board_scale(density), 0.0)

    def test_the_two_boards_order_as_expected(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            _, clean = self._measure(path)
        with _Table(TANGLED_BOARD, TANGLED_ROWS) as path:
            _, tangled = self._measure(path)
        for key in ("crowding", "collision", "assassin"):
            self.assertLess(clean[key], tangled[key], key)
        self.assertGreater(clean["pair_supply"], tangled["pair_supply"])
        self.assertGreater(cm_mod.board_scale(clean),
                           cm_mod.board_scale(tangled))
        self.assertEqual(cm_mod.board_scale(tangled), 0.0)

    def test_similarity_is_symmetrised(self):
        """A tie stored in only one direction still counts."""
        rows = dict((clue, dict(values))
                    for clue, values in CLEAN_ROWS.items())
        rows["whale"] = {"TRUCK": 0.50}        # one direction only
        rows["truck"] = {"ROAD": 0.15}
        with _Table(CLEAN_BOARD, rows) as path:
            agent, density = self._measure(path)
        self.assertGreater(density["collision"], 0.12)

    def test_missing_table_measures_nothing(self):
        agent = make_codemaster(os.path.join(tempfile.gettempdir(),
                                             "silvally-absent.bin.gz"))
        self.assertIsNone(agent._measure_density(CLEAN_OWN, CLEAN_OPP,
                                                 CLEAN_CIV, CLEAN_ASSASSIN))

    def test_one_own_word_measures_nothing(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            self.assertIsNone(agent._measure_density(["WHALE"], CLEAN_OPP,
                                                     CLEAN_CIV,
                                                     CLEAN_ASSASSIN))

    def test_result_is_cached_per_board_state(self):
        """The scan walks the whole table; it runs once per board state."""

        class _Counted(cm_mod._SimTable):
            __slots__ = ()
            scans = []

            def joint_pairs(self, words, floor):
                _Counted.scans.append(tuple(words))
                return cm_mod._SimTable.joint_pairs(self, words, floor)

        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            agent._simtable = lambda: _Counted.load(path)
            first = agent._measure_density(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                           CLEAN_ASSASSIN)
            second = agent._measure_density(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                            CLEAN_ASSASSIN)
            self.assertIs(first, second)
            self.assertEqual(len(_Counted.scans), 1)
            # A different board state is a different measurement.
            agent._measure_density(CLEAN_OWN[:3], CLEAN_OPP, CLEAN_CIV,
                                   CLEAN_ASSASSIN)
            self.assertEqual(len(_Counted.scans), 2)
        del _Counted.scans[:]

    def test_unknown_words_lower_coverage(self):
        own = CLEAN_OWN + ["XENOMORPH", "HOGWARTS", "TARDIS"]
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            density = agent._measure_density(own, CLEAN_OPP, CLEAN_CIV,
                                             CLEAN_ASSASSIN)
        self.assertLess(density["coverage"], 1.0)


# ---------------------------------------------------------------------------
# Morphing: monotonicity and the ceiling
# ---------------------------------------------------------------------------

BASE_DENSITY = {
    "measured": True,
    "coverage": 1.0,
    "crowding": 0.20,
    "collision": 0.15,
    "assassin": 0.13,
    "pair_supply": 0.60,
    "own_left": 5,
}


def density_with(**overrides):
    out = dict(BASE_DENSITY)
    out.update(overrides)
    return out


class TestMorphingMonotonicity(unittest.TestCase):

    def test_a_clean_board_with_pairs_escalates_fully(self):
        self.assertEqual(cm_mod.board_scale(BASE_DENSITY), 1.0)

    def test_more_crowding_never_buys_more_ambition(self):
        previous = None
        for crowding in (0.20, 0.26, 0.28, 0.30, 0.32, 0.34, 0.40):
            scale = cm_mod.board_scale(density_with(crowding=crowding))
            if previous is not None:
                self.assertLessEqual(scale, previous,
                                     "crowding=%s" % crowding)
            previous = scale
        self.assertEqual(previous, 0.0)

    def test_more_collision_never_buys_more_ambition(self):
        previous = None
        for collision in (0.12, 0.20, 0.24, 0.28, 0.30, 0.36):
            scale = cm_mod.board_scale(density_with(collision=collision))
            if previous is not None:
                self.assertLessEqual(scale, previous)
            previous = scale
        self.assertEqual(previous, 0.0)

    def test_a_central_assassin_never_buys_more_ambition(self):
        previous = None
        for assassin in (0.12, 0.22, 0.26, 0.30, 0.32, 0.45):
            scale = cm_mod.board_scale(density_with(assassin=assassin))
            if previous is not None:
                self.assertLessEqual(scale, previous)
            previous = scale
        self.assertEqual(previous, 0.0)

    def test_more_pairs_never_buy_less_ambition(self):
        previous = None
        for supply in (0.0, 0.10, 0.15, 0.25, 0.35, 0.45, 1.0):
            scale = cm_mod.board_scale(density_with(pair_supply=supply))
            if previous is not None:
                self.assertGreaterEqual(scale, previous)
            previous = scale
        self.assertEqual(previous, 1.0)

    def test_no_pairs_means_no_ambition_however_clean(self):
        self.assertEqual(
            cm_mod.board_scale(density_with(pair_supply=0.0, crowding=0.10,
                                            collision=0.10, assassin=0.10)),
            0.0)

    def test_any_single_danger_is_enough_to_withdraw_ambition(self):
        for key in ("crowding", "collision", "assassin"):
            self.assertEqual(cm_mod.board_scale(density_with(**{key: 0.9})),
                             0.0, key)

    def test_a_board_the_table_cannot_see_is_conservative(self):
        self.assertEqual(cm_mod.board_scale(density_with(coverage=0.3)), 0.0)
        self.assertEqual(cm_mod.board_scale(density_with(measured=False)), 0.0)
        self.assertEqual(cm_mod.board_scale(None), 0.0)
        self.assertEqual(cm_mod.board_scale({}), 0.0)


class TestCeilingClamp(unittest.TestCase):
    """No route through the morphing may pass the parked ``ambitious`` values."""

    CEILING = champion_cm.PRESETS["ambitious"]

    def test_the_fork_carries_the_champion_ceiling_unchanged(self):
        self.assertEqual(cm_mod.PRESETS["ambitious"], self.CEILING)
        self.assertEqual(self.CEILING, {"bonus_guess_weight": 0.55,
                                        "civilian_penalty": 0.45,
                                        "claimed_slack": 1})

    def test_scale_is_clamped_into_the_unit_interval(self):
        self.assertEqual(cm_mod.board_scale(density_with(pair_supply=99.0)),
                         1.0)
        self.assertEqual(
            cm_mod.board_scale(density_with(crowding=-5.0, collision=-5.0,
                                            assassin=-5.0)),
            1.0)

    def test_max_scale_kwarg_lowers_but_never_raises_the_ceiling(self):
        self.assertEqual(cm_mod.board_scale(BASE_DENSITY, max_scale=0.4), 0.4)
        self.assertEqual(cm_mod.board_scale(BASE_DENSITY, max_scale=5.0), 1.0)

    def test_full_escalation_lands_exactly_on_ambitious(self):
        bonus, civilian, slack = cm_mod.race_knobs(
            1.0, champion_cm.BONUS_GUESS_WEIGHT,
            champion_cm.CIVILIAN_PENALTY, champion_cm.CLAIMED_SLACK)
        self.assertAlmostEqual(bonus, self.CEILING["bonus_guess_weight"])
        self.assertAlmostEqual(civilian, self.CEILING["civilian_penalty"])
        self.assertEqual(slack, self.CEILING["claimed_slack"])

    def test_a_live_turn_never_exceeds_the_ceiling(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            for supply in (0.0, 0.3, 0.6, 1.0):
                agent = make_codemaster(path)
                agent._measure_density = lambda *a, **k: density_with(
                    pair_supply=supply)
                agent._race_update(CLEAN_OWN, CLEAN_OPP)
                agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                    CLEAN_ASSASSIN)
                self.assertLessEqual(agent._race_bonus,
                                     self.CEILING["bonus_guess_weight"] + 1e-9)
                self.assertGreaterEqual(
                    agent._race_civilian,
                    self.CEILING["civilian_penalty"] - 1e-9)
                self.assertLessEqual(agent._race_slack,
                                     self.CEILING["claimed_slack"])

    def test_a_broken_ramp_cannot_push_past_the_ceiling(self):
        """Even mis-swept bounds only ever reach ``ambitious``."""
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path, adapt_pair_lo=-5.0,
                                    adapt_pair_hi=-4.0, adapt_crowd_lo=99.0,
                                    adapt_crowd_hi=100.0, adapt_max_scale=9.0)
            agent._race_update(CLEAN_OWN, CLEAN_OPP)
            agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                CLEAN_ASSASSIN)
        self.assertLessEqual(agent._race_bonus,
                             self.CEILING["bonus_guess_weight"] + 1e-9)
        self.assertGreaterEqual(agent._race_civilian,
                                self.CEILING["civilian_penalty"] - 1e-9)
        self.assertLessEqual(agent._race_slack, self.CEILING["claimed_slack"])

    def test_the_slack_gate_still_guards_the_extra_word(self):
        """``claimed_slack`` only rises past the champion's own gate."""
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            below = make_codemaster(path)
            below._measure_density = lambda *a, **k: density_with(
                crowding=0.30)          # danger 0.5 -> scale 0.5 ... just at
            below._race_update(CLEAN_OWN, CLEAN_OPP)
            below._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                CLEAN_ASSASSIN)
            self.assertAlmostEqual(below._board_scale, 0.5, places=6)
            self.assertEqual(below._race_slack, 1)   # 0.5 is the gate, and it
            #                                         is inclusive by design

            timid = make_codemaster(path)
            timid._measure_density = lambda *a, **k: density_with(
                crowding=0.315)         # danger > 0.5 -> scale < 0.5
            timid._race_update(CLEAN_OWN, CLEAN_OPP)
            timid._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                CLEAN_ASSASSIN)
            self.assertLess(timid._board_scale, 0.5)
            self.assertEqual(timid._race_slack, champion_cm.CLAIMED_SLACK)


# ---------------------------------------------------------------------------
# Champion identity
# ---------------------------------------------------------------------------

class TestChampionIdentical(unittest.TestCase):
    """Where the board earns nothing, the fork *is* the champion."""

    SHARED_CONSTANTS = (
        "ASSASSIN_PENALTY", "OPPONENT_PENALTY", "OPPONENT_PENALTY_SOLO",
        "CIVILIAN_PENALTY", "BONUS_GUESS_WEIGHT", "MAJORITY_FRACTION",
        "CLAIMED_SLACK", "ASSASSIN_PRESENCE_PENALTY",
        "OPPONENT_PRESENCE_PENALTY", "PRESENCE_DECAY", "PROBE_TOP_K",
        "PROBE_VETO_SCORE", "PROBE_VETO_PENALTY", "PROBE_RELATIVE_FLOOR",
        "PROBE_NUMBER_CAP_SCORE", "PROBE_ASSASSIN_WEIGHT",
        "PROBE_OPPONENT_WEIGHT", "PROBE_MISSING_PENALTY", "EMBED_SENSOR",
        "EMBED_MARGIN", "EMBED_PENALTY", "EMBED_OOV_PENALTY",
        "MOVE_DEADLINE_S", "MOVE_WALL_S", "CALL_TIMEOUT_S",
        "RACE_FULL_DEFICIT", "RACE_SLACK_SCALE", "RACE_OPP_PACE_PRIOR",
        "RACE_OWN_PACE_PRIOR",
    )

    def test_every_inherited_constant_is_unchanged(self):
        for name in self.SHARED_CONSTANTS:
            self.assertEqual(getattr(cm_mod, name),
                             getattr(champion_cm, name), name)

    def test_guesser_constants_are_unchanged(self):
        for name in ("CONTINUE_BASE_SOLO", "CONTINUE_BASE_DUEL",
                     "BONUS_THRESHOLD", "SWEEP_MIN_CONFIDENCE",
                     "MOVE_DEADLINE_S", "MOVE_WALL_S", "TEAM_TOTALS"):
            self.assertEqual(getattr(g_mod, name),
                             getattr(champion_g, name), name)

    def test_tangled_board_reproduces_the_shipped_knobs(self):
        with _Table(TANGLED_BOARD, TANGLED_ROWS) as path:
            agent = make_codemaster(path)
            agent._race_update(CLEAN_OWN, CLEAN_OPP)
            agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                CLEAN_ASSASSIN)
        self.assertEqual(agent._board_scale, 0.0)
        self.assertEqual(agent._race_bonus, champion_cm.BONUS_GUESS_WEIGHT)
        self.assertEqual(agent._race_civilian, champion_cm.CIVILIAN_PENALTY)
        self.assertEqual(agent._race_slack, champion_cm.CLAIMED_SLACK)
        self.assertEqual(agent._adapt_log, [])

    def test_no_table_reproduces_the_shipped_knobs(self):
        agent = make_codemaster(os.path.join(tempfile.gettempdir(),
                                             "silvally-nothing-here.gz"))
        agent._race_update(CLEAN_OWN, CLEAN_OPP)
        agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV, CLEAN_ASSASSIN)
        self.assertIsNone(agent._density)
        self.assertEqual(agent._race_bonus, champion_cm.BONUS_GUESS_WEIGHT)
        self.assertEqual(agent._race_civilian, champion_cm.CIVILIAN_PENALTY)
        self.assertEqual(agent._race_slack, champion_cm.CLAIMED_SLACK)

    def test_adapt_off_reproduces_the_shipped_knobs_on_any_board(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path, adapt=False)
            agent._race_update(CLEAN_OWN, CLEAN_OPP)
            agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                CLEAN_ASSASSIN)
        self.assertIsNone(agent._density)
        self.assertEqual(agent._race_bonus, champion_cm.BONUS_GUESS_WEIGHT)
        self.assertEqual(agent._race_civilian, champion_cm.CIVILIAN_PENALTY)
        self.assertEqual(agent._race_slack, champion_cm.CLAIMED_SLACK)

    def test_a_measurement_that_raises_does_not_escalate_or_crash(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)

            def explode(*args, **kwargs):
                raise RuntimeError("table on fire")

            agent._measure_density = explode
            agent._race_update(CLEAN_OWN, CLEAN_OPP)
            agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                CLEAN_ASSASSIN)
        self.assertEqual(agent._race_bonus, champion_cm.BONUS_GUESS_WEIGHT)
        self.assertEqual(agent._race_slack, champion_cm.CLAIMED_SLACK)

    def test_a_tight_deadline_skips_the_measurement(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            agent._race_update(CLEAN_OWN, CLEAN_OPP)
            agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV,
                                CLEAN_ASSASSIN,
                                deadline=cm_mod._Deadline(0.001))
        self.assertIsNone(agent._density)
        self.assertEqual(agent._race_bonus, champion_cm.BONUS_GUESS_WEIGHT)


# ---------------------------------------------------------------------------
# The duel combination
# ---------------------------------------------------------------------------

class TestRaceCombination(unittest.TestCase):
    """Board and race escalation combine as a maximum, never a sum."""

    HISTORY = [
        ["Red_Codemaster", "OCEAN", 1],
        ["Red_Guesser", "WHALE", "*Red*"],
        ["Blue_Codemaster", "ROADS", 3],
        ["Blue_Guesser", "TRUCK", "*Blue*"],
        ["Blue_Guesser", "ROAD", "*Blue*"],
        ["Blue_Guesser", "BREAD", "*Civilian*"],
    ]

    def _agent(self, path, history, density):
        agent = make_codemaster(path)
        agent.set_move_history(history)
        agent._measure_density = lambda *a, **k: density
        agent._race_update(CLEAN_OWN, CLEAN_OPP)
        agent._adapt_update(CLEAN_OWN, CLEAN_OPP, CLEAN_CIV, CLEAN_ASSASSIN)
        return agent

    def test_a_losing_race_escalates_on_a_board_that_would_not(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = self._agent(path, self.HISTORY,
                                density_with(pair_supply=0.0))
        self.assertEqual(agent._board_scale, 0.0)
        self.assertIsNotNone(agent._race)
        self.assertEqual(agent._adapt_scale, agent._race["scale"])

    def test_a_clean_board_escalates_in_a_race_we_are_winning(self):
        won = [["Red_Codemaster", "OCEAN", 2],
               ["Red_Guesser", "WHALE", "*Red*"],
               ["Red_Guesser", "SHIP", "*Red*"],
               ["Blue_Codemaster", "ROADS", 1],
               ["Blue_Guesser", "BREAD", "*Civilian*"]]
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = self._agent(path, won, BASE_DENSITY)
        self.assertEqual(agent._race["scale"], 0.0)
        self.assertEqual(agent._board_scale, 1.0)
        self.assertEqual(agent._adapt_scale, 1.0)

    def test_the_combination_is_a_maximum(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            for supply in (0.0, 0.2, 0.6):
                agent = self._agent(path, self.HISTORY,
                                    density_with(pair_supply=supply))
                expected = max(agent._board_scale, agent._race["scale"])
                self.assertAlmostEqual(agent._adapt_scale, expected)
                self.assertLessEqual(agent._adapt_scale, 1.0)

    def test_solo_play_never_reads_a_race(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = self._agent(path, [["Red_Codemaster", "OCEAN", 1]],
                                BASE_DENSITY)
        self.assertIsNone(agent._race)
        self.assertEqual(agent._adapt_scale, agent._board_scale)


# ---------------------------------------------------------------------------
# Guesser adaptation
# ---------------------------------------------------------------------------

class TestGuesserDanger(unittest.TestCase):

    def test_deeper_clue_numbers_never_read_as_safer(self):
        previous = None
        for number in (1, 2, 3, 4, 9):
            danger = g_mod.turn_danger(number, False, 9, 25)
            if previous is not None:
                self.assertGreaterEqual(danger, previous)
            previous = danger
        self.assertEqual(previous, 1.0)

    def test_a_board_less_ours_never_reads_as_safer(self):
        previous = None
        for own_left in (9, 6, 4, 3, 2, 1):
            danger = g_mod.turn_danger(1, False, own_left, 12)
            if previous is not None:
                self.assertGreaterEqual(danger, previous)
            previous = danger

    def test_a_sweep_is_maximum_depth(self):
        self.assertEqual(g_mod.turn_danger(0, True, 9, 25), 1.0)

    def test_bounded_and_defined_on_an_empty_board(self):
        for args in ((1, False, 0, 0), (3, True, 0, 1), (0, False, 9, 25)):
            self.assertGreaterEqual(g_mod.turn_danger(*args), 0.0)
            self.assertLessEqual(g_mod.turn_danger(*args), 1.0)


class TestGuesserSamples(unittest.TestCase):

    def _guesser(self, **kwargs):
        options = {"quiet": True, "move_wall": 0.0}
        options.update(kwargs)
        return g_mod.AIGuesser("Red", **options)

    def test_samples_rise_with_danger_and_stop_at_the_ceiling(self):
        deadline = g_mod._Deadline(45.0)
        agent = self._guesser()
        seen = []
        for danger in (0.0, 0.25, 0.5, 0.75, 1.0):
            agent._danger = danger
            seen.append(agent._samples_for_turn(deadline))
        self.assertEqual(seen[0], agent.samples)
        self.assertEqual(seen[-1], agent.adapt_max_samples)
        for earlier, later in zip(seen, seen[1:]):
            self.assertLessEqual(earlier, later)
        self.assertTrue(all(s <= agent.adapt_max_samples for s in seen))

    def test_adaptation_off_never_buys_a_sample(self):
        agent = self._guesser(adapt=False)
        agent._danger = 1.0
        self.assertEqual(agent._samples_for_turn(g_mod._Deadline(45.0)),
                         agent.samples)

    def test_a_tight_budget_never_buys_a_sample(self):
        agent = self._guesser()
        agent._danger = 1.0
        self.assertEqual(agent._samples_for_turn(g_mod._Deadline(5.0)),
                         agent.samples)

    def test_danger_is_read_from_the_live_turn(self):
        agent = self._guesser()
        agent.set_board(list(CLEAN_BOARD))
        agent.set_clue("OCEAN", 3)
        self.assertGreater(agent._turn_danger(), 0.0)
        agent.set_clue("OCEAN", 1)
        agent.adapt_mode = False
        self.assertEqual(agent._turn_danger(), 0.0)


class TestGuesserThreshold(unittest.TestCase):

    def _guesser(self, **kwargs):
        options = {"quiet": True, "move_wall": 0.0}
        options.update(kwargs)
        agent = g_mod.AIGuesser("Red", **options)
        agent.set_board(list(CLEAN_BOARD))
        return agent

    def test_a_deeper_clue_demands_more_confidence(self):
        agent = self._guesser()
        agent.set_clue("OCEAN", 1)
        shallow = agent._continue_threshold()
        agent.set_clue("OCEAN", 3)
        deep = agent._continue_threshold()
        self.assertGreater(deep, shallow)

    def test_the_tilt_is_bounded_by_the_gain(self):
        agent = self._guesser()
        agent.set_clue("OCEAN", 3)
        with_adapt = agent._continue_threshold()
        agent.adapt_mode = False
        without = agent._continue_threshold()
        self.assertLessEqual(with_adapt - without,
                             agent.adapt_threshold_gain + 1e-9)

    def test_adaptation_off_matches_the_champion_formula(self):
        for number in (1, 2, 3, 0):
            ours = self._guesser(adapt=False)
            ours.set_clue("OCEAN", number)
            theirs = champion_g.AIGuesser("Red", quiet=True, move_wall=0.0)
            theirs.set_board(list(CLEAN_BOARD))
            theirs.set_clue("OCEAN", number)
            self.assertAlmostEqual(ours._continue_threshold(),
                                   theirs._continue_threshold(),
                                   msg="number=%s" % number)

    def test_the_threshold_stays_inside_the_shipped_clamp(self):
        agent = self._guesser(adapt_threshold_gain=99.0)
        agent.set_clue("OCEAN", 4)
        self.assertLessEqual(agent._continue_threshold(), 0.75)
        self.assertGreaterEqual(agent._continue_threshold(), 0.30)


# ---------------------------------------------------------------------------
# The mandatories still hold with the morphing armed
# ---------------------------------------------------------------------------

FULL_BOARD = [
    "WHALE", "SHIP", "BEACH", "APPLE", "TREE",
    "ROBOT", "LASER", "KING", "CROWN", "DRAGON",
    "PIANO", "FLUTE", "BANK", "GOLD", "DESERT",
    "SNOW", "TOWER", "SHADOW", "NIGHT", "HORSE",
    "CAR", "TRAIN", "DOCTOR", "NURSE", "POISON",
]
FULL_KEY = [
    "Red", "Red", "Red", "Blue", "Blue",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Civilian", "Civilian", "Red", "Civilian", "Blue",
    "Civilian", "Blue", "Red", "Civilian", "Assassin",
]


def scripted(system, user):
    """A plausible brainstorm / panel / probe reply for any prompt."""
    if "spymaster" in system.lower() and "candidate" in user.lower():
        return ('{"clues": [{"clue": "OCEAN", "targets": ["WHALE", "SHIP"]},'
                ' {"clue": "MUSIC", "targets": ["PIANO", "FLUTE"]}]}')
    if "rate" in user.lower() or "0-10" in user or "0 to 10" in user:
        return '{"OCEAN": {"POISON": 1, "WHALE": 9}}'
    return '{"OCEAN": ["WHALE", "SHIP", "BEACH"], "MUSIC": ["PIANO", "FLUTE"]}'


class TestMandatories(unittest.TestCase):

    def test_clues_stay_legal_and_unrepeated_with_morphing_on(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            agent.set_game_state(list(FULL_BOARD), list(FULL_KEY))
            with _PatchedLLM(cm_mod, scripted):
                seen = set()
                for _ in range(4):
                    clue, number = agent.get_clue()
                    self.assertTrue(cm_mod.clue_is_legal(clue, agent.words),
                                    "illegal clue %r" % clue)
                    self.assertNotIn(cm_mod._normalise(clue), seen)
                    seen.add(cm_mod._normalise(clue))
                    self.assertIsInstance(number, int)
                    self.assertGreaterEqual(number, 0)

    def test_a_dead_api_still_produces_a_legal_clue(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            agent.set_game_state(list(FULL_BOARD), list(FULL_KEY))
            with _PatchedLLM(cm_mod, lambda s, u: None, available=False):
                clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, agent.words))
        self.assertGreaterEqual(number, 0)

    def test_the_number_never_exceeds_the_words_we_have_left(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            words = list(FULL_BOARD)
            for index in (0, 1, 2, 5, 6, 10):      # reveal six of our nine
                words[index] = "*Red*"
            agent.set_game_state(words, list(FULL_KEY))
            with _PatchedLLM(cm_mod, scripted):
                _, number = agent.get_clue()
        self.assertLessEqual(number, 3)

    def test_the_guesser_answers_an_unlimited_clue(self):
        agent = g_mod.AIGuesser("Red", quiet=True, move_wall=0.0)
        agent.set_board(list(FULL_BOARD))
        agent.set_clue("OCEAN", 0)
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 95, "SHIP": 60}'):
            word = agent.get_answer()
        self.assertIn(word, FULL_BOARD)
        self.assertTrue(agent._is_sweep())

    def test_the_morphing_telemetry_is_emitted(self):
        printed = []
        original = cm_mod._emit
        cm_mod._emit = lambda message: printed.append(message)
        try:
            with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
                agent = make_codemaster(path, quiet=False)
                agent.set_game_state(list(FULL_BOARD), list(FULL_KEY))
                with _PatchedLLM(cm_mod, scripted):
                    agent.get_clue()
        finally:
            cm_mod._emit = original
        joined = "\n".join(printed)
        self.assertIn("board sensed:", joined)
        self.assertIn("morph", joined)
        self.assertIn("crowd", joined)

    def test_usage_summary_reports_the_morphing(self):
        with _Table(CLEAN_BOARD, CLEAN_ROWS) as path:
            agent = make_codemaster(path)
            agent.set_game_state(list(FULL_BOARD), list(FULL_KEY))
            with _PatchedLLM(cm_mod, scripted):
                agent.get_clue()
            summary = agent.usage_summary()
        self.assertIn("adapt_mode", summary)
        self.assertIn("adapt_escalated", summary)

    def test_the_agents_do_not_import_their_champion(self):
        """Challenger entries stand alone: no import of the forked files."""
        for name in ("codemaster_silvally.py", "guesser_silvally.py"):
            with open(os.path.join(FRAMEWORK_DIR, "players", name)) as handle:
                source = handle.read()
            self.assertNotIn("import codemaster_obirdy", source, name)
            self.assertNotIn("import guesser_obirdy", source, name)
            self.assertNotIn("from players.codemaster_obirdy", source, name)
            self.assertNotIn("from players.guesser_obirdy", source, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
