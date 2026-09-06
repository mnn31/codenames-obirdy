"""Offline unit tests for the Lycanroc fork's candidate-generation stage.

No network, no API key: every LLM call is monkeypatched at the agent's own
``_LLM.chat`` boundary, which is the only place the file touches Anthropic.
Nothing here re-tests the champion's scorer, panel, probe or sensor -- those
are covered by ``harness/test_obirdy.py`` and Lycanroc does not touch them.
What is tested is the five things the fork adds, plus the claim that turning
the fork off leaves the champion behind:

* pair / triple **enumeration** -- every subset, and triples only once enough
  own words remain;
* **table-based ranking** -- the ``min`` rule, the floor, the danger discount,
  and the diversity spread;
* **merge / dedupe** -- position, spelling and the target upgrade;
* **deadline pressure** -- triples dropped first, then pairs, then nothing;
* **disabled == champion** -- the same prompt, the same one call, the same
  clue, the same number, the same usage dict.

Run from the repo root::

    python -m unittest harness.test_lycanroc -v
"""

import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from players import codemaster_lycanroc as lyc  # noqa: E402
from players import codemaster_obirdy as champ  # noqa: E402
from players import guesser_lycanroc as lyc_g  # noqa: E402
from players import guesser_obirdy as champ_g  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BOARD = [
    "WHALE", "SHIP", "BEACH", "APPLE", "TREE",
    "ROBOT", "LASER", "KING", "CROWN", "DRAGON",
    "PIANO", "FLUTE", "BANK", "GOLD", "DESERT",
    "SNOW", "TOWER", "SHADOW", "NIGHT", "HORSE",
    "CAR", "TRAIN", "DOCTOR", "NURSE", "POISON",
]

#: 9 Red / 8 Blue / 7 Civilian / 1 Assassin, the engine's composition.
KEY = [
    "Red", "Red", "Red", "Blue", "Blue",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Civilian", "Civilian", "Red", "Civilian", "Blue",
    "Civilian", "Blue", "Red", "Civilian", "Assassin",
]

RED = [BOARD[i] for i, k in enumerate(KEY) if k == "Red"]
BLUE = [BOARD[i] for i, k in enumerate(KEY) if k == "Blue"]
CIV = [BOARD[i] for i, k in enumerate(KEY) if k == "Civilian"]
ASSASSIN = [BOARD[i] for i, k in enumerate(KEY) if k == "Assassin"]


class _PatchedLLM(object):
    """Scripted ``_LLM.chat``; a context manager so a failure cannot leak it."""

    def __init__(self, module, responder, available=True):
        self.module = module
        self.responder = responder
        self.is_available = available
        self.calls = []

    def __enter__(self):
        self._chat = self.module._LLM.chat
        self._available = self.module._LLM.available
        calls = self.calls
        responder = self.responder

        def chat(inner_self, system, user, max_tokens=600, deadline=None):
            calls.append({"system": system, "user": user,
                          "max_tokens": max_tokens})
            return responder(system, user)

        self.module._LLM.chat = chat
        self.module._LLM.available = lambda inner_self: self.is_available
        return self.calls

    def __exit__(self, *exc):
        self.module._LLM.chat = self._chat
        self.module._LLM.available = self._available
        return False


class _FakeDeadline(object):
    """A deadline that reports a fixed number of seconds left."""

    def __init__(self, remaining):
        self._remaining = float(remaining)
        self.budget = float(remaining)

    def elapsed(self):
        return 0.0

    def remaining(self):
        return self._remaining

    def expired(self, reserve=0.0):
        return self._remaining <= reserve


class _FakeTable(object):
    """A hand-built stand-in for ``_SimTable``, for the ranking tests.

    ``rows`` is ``{clue word: {board word: cosine}}``.  Only the three
    attributes ``board_clue_index`` reads are provided plus the packed byte
    sections, built here by an encoder independent of the agent's reader --
    so a format bug in either one shows up as a disagreement rather than as
    two copies of the same mistake.
    """

    def __init__(self, rows, version=2):
        board_words = sorted({w for row in rows.values() for w in row})
        clue_words = sorted(rows)
        levels = sorted({round(v, 6) for row in rows.values()
                         for v in row.values()})
        code_bits = 6
        self.board_words = board_words
        self.board_index = dict((w, i) for i, w in enumerate(board_words))
        self.clue_words = clue_words
        self.clue_index = dict((w, i) for i, w in enumerate(clue_words))
        self.levels = levels
        self.floor = min(levels) if levels else 0.0
        self.code_bits = code_bits
        self.code_mask = (1 << code_bits) - 1
        self.version = version
        self.path = "fake"

        offsets = bytearray()
        entries = bytearray()
        codes = bytearray()
        cursor = 0
        for clue in clue_words:
            offsets += cursor.to_bytes(4, "little")
            for word, value in sorted(rows[clue].items()):
                column = self.board_index[word]
                code = levels.index(round(value, 6))
                if version == 1:
                    entries += ((column << code_bits) | code).to_bytes(2, "little")
                else:
                    entries += column.to_bytes(2, "little")
                    codes.append(code)
                cursor += 1
        offsets += cursor.to_bytes(4, "little")
        self._offsets = bytes(offsets)
        self._entries = bytes(entries)
        self._codes = bytes(codes) if version == 2 else None


def make_codemaster(module=lyc, **kwargs):
    kwargs.setdefault("quiet", True)
    kwargs.setdefault("api_key", "sk-ant-test")
    kwargs.setdefault("move_wall", 0)          # run on the calling thread
    agent = module.AICodemaster("Red", **kwargs)
    agent.set_game_state(list(BOARD), list(KEY))
    return agent


# ---------------------------------------------------------------------------
# 1. Enumeration
# ---------------------------------------------------------------------------

class TestGroupEnumeration(unittest.TestCase):
    """``rank_groups`` must look at every subset, not a sample of them."""

    def _index(self, words, cosine=0.9):
        """An index where one clue row links every word at ``cosine``."""
        return dict((lyc._normalise(w), {0: cosine}) for w in words)

    def test_every_pair_is_enumerated(self):
        own = ["ALPHA", "BETA", "GAMMA", "DELTA"]
        ranked = lyc.rank_groups(self._index(own), own, {}, 2, 0.1)
        self.assertEqual(len(ranked), 6)                     # C(4, 2)
        self.assertEqual(
            sorted(tuple(sorted(item[1])) for item in ranked),
            [("ALPHA", "BETA"), ("ALPHA", "DELTA"), ("ALPHA", "GAMMA"),
             ("BETA", "DELTA"), ("BETA", "GAMMA"), ("DELTA", "GAMMA")])

    def test_every_triple_is_enumerated(self):
        own = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
        ranked = lyc.rank_groups(self._index(own), own, {}, 3, 0.1)
        self.assertEqual(len(ranked), 10)                    # C(5, 3)
        for item in ranked:
            self.assertEqual(len(item[1]), 3)
            self.assertEqual(len(set(item[1])), 3)

    def test_group_larger_than_the_board_is_empty(self):
        own = ["ALPHA", "BETA"]
        self.assertEqual(lyc.rank_groups(self._index(own), own, {}, 3, 0.1), [])

    def test_a_word_the_table_never_saw_joins_no_group(self):
        own = ["ALPHA", "BETA", "STRANGER"]
        index = self._index(["ALPHA", "BETA"])
        ranked = lyc.rank_groups(index, own, {}, 2, 0.1)
        self.assertEqual([item[1] for item in ranked], [("ALPHA", "BETA")])

    def test_triples_wait_for_enough_own_words(self):
        """The agent asks about triples only past its own threshold."""
        rows = {}
        for clue in ("linker",):
            rows[clue] = dict((w, 0.5) for w in RED)
        table = _FakeTable(rows)
        agent = make_codemaster(pair_min_own_for_triples=5)
        agent._pair_index = lyc.board_clue_index(table, BOARD)
        agent._pair_index_key = tuple(lyc._normalise(w) for w in BOARD)

        plan = agent._pair_groups(RED[:4], BLUE, ASSASSIN, None)
        self.assertEqual([size for size, _, _ in plan], [2])
        plan = agent._pair_groups(RED[:5], BLUE, ASSASSIN, None)
        self.assertEqual([size for size, _, _ in plan], [2, 3])

    def test_a_single_own_word_asks_nothing(self):
        agent = make_codemaster()
        self.assertEqual(agent._pair_groups(RED[:1], BLUE, ASSASSIN, None), [])


# ---------------------------------------------------------------------------
# 2. Table-based ranking
# ---------------------------------------------------------------------------

class TestPairRanking(unittest.TestCase):

    def test_the_reader_agrees_with_an_independent_encoder(self):
        rows = {"ocean": {"WHALE": 0.6, "SHIP": 0.4},
                "royal": {"KING": 0.7, "CROWN": 0.5}}
        for version in (1, 2):
            index = lyc.board_clue_index(_FakeTable(rows, version=version),
                                         ["WHALE", "SHIP", "KING", "CROWN"])
            self.assertIsNotNone(index, version)
            self.assertAlmostEqual(index["WHALE"][0], 0.6, places=5)
            self.assertAlmostEqual(index["SHIP"][0], 0.4, places=5)
            self.assertAlmostEqual(index["KING"][1], 0.7, places=5)

    def test_index_keeps_only_the_columns_asked_for(self):
        rows = {"ocean": {"WHALE": 0.6, "SHIP": 0.4, "BEACH": 0.3}}
        index = lyc.board_clue_index(_FakeTable(rows), ["WHALE", "BEACH"])
        self.assertEqual(sorted(index), ["BEACH", "WHALE"])

    def test_no_table_is_no_index(self):
        self.assertIsNone(lyc.board_clue_index(None, BOARD))

    def test_a_board_the_table_cannot_price_is_no_index(self):
        rows = {"ocean": {"WHALE": 0.6}}
        self.assertIsNone(lyc.board_clue_index(_FakeTable(rows), ["NOTHERE"]))

    def test_a_corrupt_table_is_no_index_rather_than_a_wrong_one(self):
        rows = {"ocean": {"WHALE": 0.6, "SHIP": 0.4}}
        table = _FakeTable(rows)
        table._offsets = table._offsets[:3]      # truncated
        self.assertIsNone(lyc.board_clue_index(table, ["WHALE", "SHIP"]))

    def test_the_joint_score_is_the_minimum_not_the_mean(self):
        """A clue that nails one word and misses the other is worth nothing."""
        index = {"WHALE": {0: 0.95}, "SHIP": {0: 0.05},
                 "KING": {1: 0.40}, "CROWN": {1: 0.40}}
        ranked = lyc.rank_groups(index, ["WHALE", "SHIP", "KING", "CROWN"],
                                 {}, 2, 0.1)
        best = ranked[0]
        self.assertEqual(tuple(sorted(best[1])), ("CROWN", "KING"))
        self.assertAlmostEqual(best[3], 0.40, places=5)
        # The lopsided pair survives the floor of 0.1 only at its own minimum.
        lopsided = [i for i in ranked if tuple(sorted(i[1])) == ("SHIP", "WHALE")]
        self.assertEqual(lopsided, [])

    def test_the_floor_drops_a_weak_pair(self):
        index = {"WHALE": {0: 0.20}, "SHIP": {0: 0.20}}
        self.assertEqual(lyc.rank_groups(index, ["WHALE", "SHIP"], {}, 2, 0.28),
                         [])
        kept = lyc.rank_groups(index, ["WHALE", "SHIP"], {}, 2, 0.15)
        self.assertEqual(len(kept), 1)

    def test_a_pair_needs_one_clue_that_reaches_both(self):
        """Two different clue rows, one per word, is not a joint clue."""
        index = {"WHALE": {0: 0.9}, "SHIP": {1: 0.9}}
        self.assertEqual(lyc.rank_groups(index, ["WHALE", "SHIP"], {}, 2, 0.1),
                         [])

    def test_danger_discounts_but_the_floor_is_read_undiscounted(self):
        index = {"WHALE": {0: 0.50, 1: 0.35}, "SHIP": {0: 0.50, 1: 0.35}}
        danger = {0: 0.60}          # row 0 also screams at the assassin
        ranked = lyc.rank_groups(index, ["WHALE", "SHIP"], danger, 2, 0.30,
                                 danger_weight=0.6, danger_free=0.15)
        self.assertEqual(len(ranked), 1)
        # Row 0 scores 0.50 - 0.6 * 0.45 = 0.23; row 1 scores a clean 0.35, so
        # the clean-but-weaker witness wins.
        self.assertEqual(ranked[0][2], 1)
        self.assertAlmostEqual(ranked[0][0], 0.35, places=5)
        self.assertAlmostEqual(ranked[0][4], 0.0, places=5)

    def test_harmless_pull_is_free(self):
        index = {"WHALE": {0: 0.50}, "SHIP": {0: 0.50}}
        clean = lyc.rank_groups(index, ["WHALE", "SHIP"], {}, 2, 0.1)
        mild = lyc.rank_groups(index, ["WHALE", "SHIP"], {0: 0.14}, 2, 0.1)
        self.assertAlmostEqual(clean[0][0], mild[0][0], places=6)

    def test_danger_rows_take_the_worst_pull(self):
        index = {"POISON": {0: 0.2, 1: 0.5}, "CAR": {0: 0.4}}
        self.assertEqual(lyc.danger_rows(index, ["POISON", "CAR"]),
                         {0: 0.4, 1: 0.5})

    def test_ranking_is_deterministic(self):
        index = {"WHALE": {0: 0.5}, "SHIP": {0: 0.5}, "KING": {0: 0.5}}
        first = lyc.rank_groups(index, ["WHALE", "SHIP", "KING"], {}, 2, 0.1)
        second = lyc.rank_groups(index, ["KING", "WHALE", "SHIP"], {}, 2, 0.1)
        self.assertEqual([sorted(i[1]) for i in first],
                         [sorted(i[1]) for i in second])

    def test_diversity_spreads_the_groups_over_the_board(self):
        """The recorded failure: four calls about one cluster of three words."""
        ranked = [
            (0.66, ("GAS", "OIL"), 0, 0.66, 0.0),
            (0.41, ("FIRE", "STAFF"), 1, 0.41, 0.0),
            (0.40, ("FIRE", "GAS"), 2, 0.40, 0.0),
            (0.38, ("FIRE", "OIL"), 3, 0.38, 0.0),
            (0.35, ("HEART", "BEAR"), 4, 0.35, 0.0),
        ]
        picked = lyc.select_diverse_groups(ranked, 3, overlap_penalty=0.08)
        self.assertEqual([item[1] for item in picked],
                         [("GAS", "OIL"), ("FIRE", "STAFF"), ("HEART", "BEAR")])
        words = set()
        for item in picked:
            words.update(item[1])
        self.assertEqual(len(words), 6)

    def test_diversity_never_discards_a_clearly_better_group(self):
        ranked = [
            (0.90, ("A", "B"), 0, 0.90, 0.0),
            (0.85, ("A", "C"), 1, 0.85, 0.0),
            (0.10, ("D", "E"), 2, 0.10, 0.0),
        ]
        picked = lyc.select_diverse_groups(ranked, 2, overlap_penalty=0.08)
        self.assertEqual([item[1] for item in picked], [("A", "B"), ("A", "C")])

    def test_diversity_respects_its_limit(self):
        ranked = [(0.5, ("A", "B"), 0, 0.5, 0.0), (0.4, ("C", "D"), 1, 0.4, 0.0)]
        self.assertEqual(lyc.select_diverse_groups(ranked, 0), [])
        self.assertEqual(len(lyc.select_diverse_groups(ranked, 1)), 1)
        self.assertEqual(len(lyc.select_diverse_groups(ranked, 9)), 2)


class TestShippedTableRanking(unittest.TestCase):
    """The same rules against the real bundled table, when it is present."""

    @classmethod
    def setUpClass(cls):
        cls.table = lyc._SimTable.shared()
        if cls.table is None:
            raise unittest.SkipTest("bundled similarity table not found")

    def test_the_real_board_ranks_real_pairs(self):
        index = lyc.board_clue_index(self.table, BOARD)
        self.assertIsNotNone(index)
        danger = lyc.danger_rows(index, ASSASSIN + BLUE)
        ranked = lyc.rank_groups(index, RED, danger, 2, lyc.PAIR_FLOOR)
        self.assertTrue(ranked, "no pair cleared the floor on the test board")
        pairs = set(tuple(sorted(item[1])) for item in ranked)
        # PIANO/FLUTE and SHIP/WHALE are both red pairs on this board and are
        # the two a human spymaster would see first.
        self.assertIn(("FLUTE", "PIANO"), pairs)
        self.assertIn(("SHIP", "WHALE"), pairs)
        for score, group, _, joint, _ in ranked:
            self.assertGreaterEqual(joint, lyc.PAIR_FLOOR)
            self.assertLessEqual(len(group), 2)

    def test_the_index_costs_a_fraction_of_the_move_budget(self):
        import time
        start = time.time()
        lyc.board_clue_index(self.table, BOARD)
        self.assertLess(time.time() - start, 5.0)

    def test_the_index_is_built_once_per_board(self):
        agent = make_codemaster()
        first = agent._pair_columns()
        second = agent._pair_columns()
        self.assertIs(first, second)


# ---------------------------------------------------------------------------
# 3. Merge / dedupe
# ---------------------------------------------------------------------------

class TestMerge(unittest.TestCase):

    def test_general_ordering_survives(self):
        general = [("ALPHA", ["WHALE"]), ("BETA", [])]
        extra = [("GAMMA", ["KING", "CROWN"])]
        merged = lyc.merge_candidates(general, extra)
        self.assertEqual([c for c, _ in merged], ["ALPHA", "BETA", "GAMMA"])

    def test_a_duplicate_keeps_its_position_and_spelling(self):
        general = [("Alpha", ["WHALE"]), ("BETA", [])]
        extra = [("ALPHA", ["WHALE", "SHIP"])]
        merged = lyc.merge_candidates(general, extra)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0][0], "Alpha")

    def test_a_duplicate_upgrades_to_the_richer_target_claim(self):
        """The number rule caps at the claimed count, so the claim matters."""
        general = [("ANCHOR", ["SHIP"])]
        extra = [("ANCHOR", ["SHIP", "WHALE"])]
        merged = lyc.merge_candidates(general, extra)
        self.assertEqual(merged, [("ANCHOR", ["SHIP", "WHALE"])])

    def test_a_duplicate_never_downgrades(self):
        general = [("ANCHOR", ["SHIP", "WHALE"])]
        extra = [("ANCHOR", [])]
        merged = lyc.merge_candidates(general, extra)
        self.assertEqual(merged, [("ANCHOR", ["SHIP", "WHALE"])])

    def test_duplicates_inside_one_reply_collapse(self):
        merged = lyc.merge_candidates([("A", []), ("A", ["WHALE"])])
        self.assertEqual(merged, [("A", ["WHALE"])])

    def test_malformed_entries_are_skipped_not_fatal(self):
        merged = lyc.merge_candidates([("A", ["WHALE"]), None, 7, ("", ["X"])])
        self.assertEqual(merged, [("A", ["WHALE"])])

    def test_empty_inputs(self):
        self.assertEqual(lyc.merge_candidates(None, [], ()), [])


# ---------------------------------------------------------------------------
# 4. Deadline pressure
# ---------------------------------------------------------------------------

class TestDeadlinePressure(unittest.TestCase):
    """Group calls are the first thing dropped, triples before pairs."""

    def setUp(self):
        rows = {"linker": dict((w, 0.9) for w in RED)}
        self.table = _FakeTable(rows)

    def _agent(self, **kwargs):
        agent = make_codemaster(**kwargs)
        agent._pair_index = lyc.board_clue_index(self.table, BOARD)
        agent._pair_index_key = tuple(lyc._normalise(w) for w in BOARD)
        return agent

    def test_a_roomy_budget_asks_about_pairs_and_triples(self):
        agent = self._agent()
        plan = agent._pair_groups(RED, BLUE, ASSASSIN, _FakeDeadline(40.0))
        self.assertEqual([size for size, _, _ in plan], [2, 3])
        self.assertEqual(agent.pair_dropped_for_time, 0)

    def test_triples_go_first(self):
        agent = self._agent()
        remaining = (lyc.PAIR_MIN_SECONDS + lyc.TRIPLE_MIN_SECONDS) / 2.0
        plan = agent._pair_groups(RED, BLUE, ASSASSIN, _FakeDeadline(remaining))
        self.assertEqual([size for size, _, _ in plan], [2])
        self.assertEqual(agent.pair_dropped_for_time, 1)

    def test_pairs_go_next(self):
        agent = self._agent()
        plan = agent._pair_groups(RED, BLUE, ASSASSIN,
                                  _FakeDeadline(lyc.PAIR_MIN_SECONDS - 1.0))
        self.assertEqual(plan, [])
        self.assertEqual(agent.pair_dropped_for_time, 1)

    def test_a_squeezed_turn_still_makes_the_general_call(self):
        agent = self._agent()
        with _PatchedLLM(lyc, lambda s, u: '[{"clue":"OCEAN","targets":[]}]') as calls:
            agent._brainstorm(RED, BLUE, CIV, ASSASSIN, [],
                              _FakeDeadline(lyc.PAIR_MIN_SECONDS - 1.0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(agent.pair_calls, 0)

    def test_no_deadline_object_is_treated_as_unlimited(self):
        agent = self._agent()
        plan = agent._pair_groups(RED, BLUE, ASSASSIN, None)
        self.assertEqual([size for size, _, _ in plan], [2, 3])

    def test_the_extra_call_ceiling_holds(self):
        agent = self._agent()
        with _PatchedLLM(lyc, lambda s, u: '[{"clue":"OCEAN","targets":[]}]') as calls:
            agent._brainstorm(RED, BLUE, CIV, ASSASSIN, [], _FakeDeadline(40.0))
        self.assertEqual(len(calls), 1 + lyc.PAIR_MAX_EXTRA_CALLS)
        self.assertEqual(agent.pair_calls, lyc.PAIR_MAX_EXTRA_CALLS)

    def test_the_ceiling_is_a_knob_and_it_binds(self):
        agent = self._agent(pair_max_extra_calls=1)
        with _PatchedLLM(lyc, lambda s, u: '[{"clue":"OCEAN","targets":[]}]') as calls:
            agent._brainstorm(RED, BLUE, CIV, ASSASSIN, [], _FakeDeadline(40.0))
        self.assertEqual(len(calls), 2)

    def test_no_table_means_no_group_calls(self):
        agent = make_codemaster()
        agent._pair_index = None
        agent._pair_index_key = tuple(lyc._normalise(w) for w in BOARD)
        with _PatchedLLM(lyc, lambda s, u: '[{"clue":"OCEAN","targets":[]}]') as calls:
            agent._brainstorm(RED, BLUE, CIV, ASSASSIN, [], _FakeDeadline(40.0))
        self.assertEqual(len(calls), 1)

    def test_a_ranking_crash_costs_nothing_but_the_feature(self):
        agent = self._agent()

        def boom(*args, **kwargs):
            raise RuntimeError("ranking exploded")

        original = lyc.rank_groups
        lyc.rank_groups = boom
        try:
            with _PatchedLLM(lyc, lambda s, u: '[{"clue":"OCEAN","targets":[]}]') as calls:
                out = agent._brainstorm(RED, BLUE, CIV, ASSASSIN, [],
                                        _FakeDeadline(40.0))
        finally:
            lyc.rank_groups = original
        self.assertEqual(len(calls), 1)
        self.assertEqual(out, [("OCEAN", [])])


# ---------------------------------------------------------------------------
# 5. The group prompt
# ---------------------------------------------------------------------------

class TestGroupPrompt(unittest.TestCase):

    def _prompt(self, groups, per_group=5):
        agent = make_codemaster()
        return agent._group_brainstorm_prompt(groups, RED, BLUE, CIV,
                                              ASSASSIN, [], per_group)

    def test_it_names_every_group_and_the_count(self):
        system, user = self._prompt([("KING", "CROWN"), ("WHALE", "SHIP")])
        self.assertIn("KING + CROWN", user)
        self.assertIn("WHALE + SHIP", user)
        self.assertIn("give 5 single-word clues", user)
        self.assertIn("ALL 2 words", user)

    def test_it_lists_the_words_to_avoid(self):
        _, user = self._prompt([("KING", "CROWN")])
        avoid = user.rsplit("Avoid any pull toward these words:", 1)[1]
        for word in ASSASSIN + BLUE + CIV:
            self.assertIn(word, avoid)

    def test_the_persona_is_the_champions(self):
        system, _ = self._prompt([("KING", "CROWN")])
        self.assertEqual(system, champ.AICodemaster.__dict__.get(
            "BRAINSTORM_SYSTEM", system))
        self.assertIn("extremely risk-averse about the assassin", system)

    def test_a_triple_prompt_says_three(self):
        _, user = self._prompt([("KING", "CROWN", "GOLD")], per_group=4)
        self.assertIn("ALL 3 words", user)
        self.assertIn("KING + CROWN + GOLD", user)

    def test_the_reply_shape_is_the_one_the_parser_reads(self):
        _, user = self._prompt([("KING", "CROWN")])
        self.assertIn('"clue"', user)
        self.assertIn('"targets"', user)
        parsed = lyc._parse_candidates(
            '[{"clue":"ROYAL","targets":["KING","CROWN"]}]')
        self.assertEqual(parsed, [("ROYAL", ["KING", "CROWN"])])


# ---------------------------------------------------------------------------
# 6. Disabled == champion
# ---------------------------------------------------------------------------

class TestChampionIdentical(unittest.TestCase):
    """With the fork off, this file must be the champion in every way we can
    check offline: the same prompt, the same one call, the same clue, the same
    number, and a usage dict with no extra keys."""

    OFF = {"pair_brainstorm": False}

    def test_the_general_prompt_is_byte_identical(self):
        ours = make_codemaster(**self.OFF)
        theirs = make_codemaster(module=champ)
        system, user = ours._general_brainstorm_prompt(RED, BLUE, CIV,
                                                       ASSASSIN, ["KING"])
        recorded = {}

        def chat(inner_self, sys_text, user_text, max_tokens=600, deadline=None):
            recorded["system"] = sys_text
            recorded["user"] = user_text
            return "[]"

        original = champ._LLM.chat
        champ._LLM.chat = chat
        try:
            theirs._brainstorm(RED, BLUE, CIV, ASSASSIN, ["KING"], None)
        finally:
            champ._LLM.chat = original
        self.assertEqual(system, recorded["system"])
        self.assertEqual(user, recorded["user"])

    def test_one_call_and_the_champions_parse(self):
        agent = make_codemaster(**self.OFF)
        reply = '[{"clue":"ROYAL","targets":["KING","CROWN"]},{"clue":"SEA","targets":["WHALE"]}]'
        with _PatchedLLM(lyc, lambda s, u: reply) as calls:
            out = agent._brainstorm(RED, BLUE, CIV, ASSASSIN, [],
                                    _FakeDeadline(40.0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_tokens"], 900)
        self.assertEqual(out, champ._parse_candidates(reply))

    def _scripted(self, module):
        """A responder good enough to drive a whole turn deterministically."""

        def responder(system, user):
            if "candidate clues" in user or "groups of YOUR words" in user:
                return json.dumps([
                    {"clue": "ROYAL", "targets": ["KING", "CROWN"]},
                    {"clue": "VOYAGE", "targets": ["SHIP"]},
                ])
            if "most associated" in user:
                return json.dumps({"ROYAL": ["CROWN", "KING", "GOLD"],
                                   "VOYAGE": ["SHIP", "WHALE", "DESERT"]})
            return json.dumps({"assassin": 0, "opponents": {}})

        return responder

    def test_the_same_board_gives_the_same_clue_and_number(self):
        theirs = make_codemaster(module=champ)
        with _PatchedLLM(champ, self._scripted(champ)):
            expected = theirs.get_clue()
        ours = make_codemaster(**self.OFF)
        with _PatchedLLM(lyc, self._scripted(lyc)):
            got = ours.get_clue()
        self.assertEqual(got, expected)

    def test_the_usage_dict_grows_no_keys(self):
        theirs = make_codemaster(module=champ)
        with _PatchedLLM(champ, self._scripted(champ)):
            theirs.get_clue()
        ours = make_codemaster(**self.OFF)
        with _PatchedLLM(lyc, self._scripted(lyc)):
            ours.get_clue()
        self.assertEqual(sorted(ours.usage_summary()),
                         sorted(theirs.usage_summary()))

    def test_the_fork_on_adds_its_keys_only_when_it_fired(self):
        rows = {"linker": dict((w, 0.9) for w in RED)}
        agent = make_codemaster()
        agent._pair_index = lyc.board_clue_index(_FakeTable(rows), BOARD)
        agent._pair_index_key = tuple(lyc._normalise(w) for w in BOARD)
        with _PatchedLLM(lyc, self._scripted(lyc)):
            agent.get_clue()
        summary = agent.usage_summary()
        self.assertIn("pair_calls", summary)
        self.assertGreaterEqual(summary["pair_calls"], 1)
        self.assertGreaterEqual(summary["pair_groups_asked"], 1)
        self.assertIn("pair_upgrades", summary)


class TestBrainstormAccounting(unittest.TestCase):
    """``pair_new_candidates`` and ``pair_upgrades`` are the two ways a group
    call can change the panel's menu, and they are counted separately."""

    def _agent(self):
        rows = {"linker": dict((w, 0.9) for w in RED)}
        agent = make_codemaster()
        agent._pair_index = lyc.board_clue_index(_FakeTable(rows), BOARD)
        agent._pair_index_key = tuple(lyc._normalise(w) for w in BOARD)
        return agent

    def _run(self, general, group):
        agent = self._agent()

        def responder(system, user):
            return group if "groups of YOUR words" in user else general

        with _PatchedLLM(lyc, responder):
            agent._brainstorm(RED, BLUE, CIV, ASSASSIN, [], _FakeDeadline(40.0))
        return agent

    def test_a_brand_new_clue_counts_as_new(self):
        agent = self._run(json.dumps([{"clue": "VOYAGE", "targets": ["SHIP"]}]),
                          json.dumps([{"clue": "ROYAL",
                                       "targets": ["PIANO", "FLUTE"]}]))
        self.assertEqual(agent.pair_new_candidates, 1)
        self.assertEqual(agent.pair_upgrades, 0)

    def test_a_raised_claim_on_an_existing_clue_counts_as_an_upgrade(self):
        agent = self._run(json.dumps([{"clue": "ROYAL", "targets": ["PIANO"]}]),
                          json.dumps([{"clue": "ROYAL",
                                       "targets": ["PIANO", "FLUTE"]}]))
        self.assertEqual(agent.pair_new_candidates, 0)
        self.assertEqual(agent.pair_upgrades, 1)

    def test_a_repeat_of_the_same_claim_counts_as_neither(self):
        same = json.dumps([{"clue": "ROYAL", "targets": ["PIANO"]}])
        agent = self._run(same, same)
        self.assertEqual(agent.pair_new_candidates, 0)
        self.assertEqual(agent.pair_upgrades, 0)


# ---------------------------------------------------------------------------
# 7. The mechanism, demonstrated
# ---------------------------------------------------------------------------

class TestThePaceMechanism(unittest.TestCase):
    """The whole thesis in one scripted turn.

    ``number = min(panel_supported_prefix, max(1, claimed + claimed_slack))``
    with ``claimed_slack = 0``, so a general brainstorm that claims one word
    caps the clue at one *even when the panel would have confirmed two*.  Here
    the panel says ``ROYAL`` reaches PIANO then FLUTE -- both ours -- and the
    only difference between the two agents is that Lycanroc also asked a
    question whose answer claims the pair.  The champion issues 1; Lycanroc
    issues 2, off the same panel and past the same probe.

    Scripted, not measured: it demonstrates that the mechanism is wired, not
    how often a live model supplies the second target.  That is what the live
    smoke is for.
    """

    def _responder(self, pair_reply):
        def responder(system, user):
            if "groups of YOUR words" in user:
                return pair_reply
            if "candidate clues" in user:
                # The documented failure mode: a timid single-target claim.
                return json.dumps([{"clue": "ROYAL", "targets": ["PIANO"]}])
            if "most associated" in user:
                return json.dumps({"ROYAL": ["PIANO", "FLUTE", "DESERT"]})
            # Probe: every danger word rated 0, so nothing is vetoed or capped.
            return json.dumps(dict((w, 0) for w in BLUE + ASSASSIN + CIV))

        return responder

    def _agent(self, module=lyc, **kwargs):
        rows = {"linker": dict((w, 0.9) for w in RED)}
        agent = make_codemaster(module=module, **kwargs)
        if module is lyc:
            agent._pair_index = lyc.board_clue_index(_FakeTable(rows), BOARD)
            agent._pair_index_key = tuple(lyc._normalise(w) for w in BOARD)
        return agent

    def test_the_champion_is_capped_at_one_by_its_own_claim(self):
        agent = self._agent(module=champ)
        with _PatchedLLM(champ, self._responder("[]")):
            clue, number = agent.get_clue()
        self.assertEqual([clue, number], ["ROYAL", 1])

    def test_the_fork_issues_two_off_the_same_panel(self):
        agent = self._agent()
        pair_reply = json.dumps(
            [{"clue": "ROYAL", "targets": ["PIANO", "FLUTE"]}])
        with _PatchedLLM(lyc, self._responder(pair_reply)):
            clue, number = agent.get_clue()
        self.assertEqual([clue, number], ["ROYAL", 2])

    def test_the_panel_still_has_the_final_word_on_the_number(self):
        """A pair claim the panel does not support buys nothing."""
        agent = self._agent()

        def responder(system, user):
            if "groups of YOUR words" in user:
                return json.dumps(
                    [{"clue": "ROYAL", "targets": ["PIANO", "FLUTE"]}])
            if "candidate clues" in user:
                return json.dumps([{"clue": "ROYAL", "targets": ["PIANO"]}])
            if "most associated" in user:
                # FLUTE is not second: an opponent word is.
                return json.dumps({"ROYAL": ["PIANO", "KING", "FLUTE"]})
            return json.dumps(dict((w, 0) for w in BLUE + ASSASSIN + CIV))

        with _PatchedLLM(lyc, responder):
            clue, number = agent.get_clue()
        self.assertEqual([clue, number], ["ROYAL", 1])

    def test_a_probe_veto_still_beats_a_pair_clue(self):
        agent = self._agent()

        def responder(system, user):
            if "groups of YOUR words" in user:
                return json.dumps(
                    [{"clue": "ROYAL", "targets": ["PIANO", "FLUTE"]}])
            if "candidate clues" in user:
                return json.dumps([{"clue": "VOYAGE", "targets": ["SHIP"]}])
            if "most associated" in user:
                return json.dumps({"ROYAL": ["PIANO", "FLUTE"],
                                   "VOYAGE": ["SHIP", "WHALE"]})
            ratings = dict((w, 0) for w in BLUE + CIV)
            if "ROYAL" in user:
                ratings[ASSASSIN[0]] = 9        # absolute veto territory
            else:
                ratings[ASSASSIN[0]] = 0
            return json.dumps(ratings)

        with _PatchedLLM(lyc, responder):
            clue, _number = agent.get_clue()
        self.assertNotEqual(clue, "ROYAL")
        self.assertGreaterEqual(agent.probes_vetoed, 1)


# ---------------------------------------------------------------------------
# 8. The mandatories the fork must not have broken
# ---------------------------------------------------------------------------

class TestInvariants(unittest.TestCase):

    def _scripted_bad(self, system, user):
        """A hostile reply: board words, junk, and an illegal clue."""
        if "groups of YOUR words" in user:
            return json.dumps([
                {"clue": "KING", "targets": ["KING", "CROWN"]},       # board word
                {"clue": "DRAG", "targets": ["KING", "CROWN"]},       # in DRAGON
                {"clue": "TWO WORDS", "targets": ["KING", "CROWN"]},
                {"clue": "REGAL", "targets": ["KING", "CROWN"]},
            ])
        if "candidate clues" in user:
            return json.dumps([{"clue": "SEAFARING", "targets": ["SHIP"]}])
        if "most associated" in user:
            return json.dumps({"REGAL": ["CROWN", "KING", "GOLD"],
                               "SEAFARING": ["SHIP", "WHALE"]})
        return json.dumps({"assassin": 0, "opponents": {}})

    def _agent_with_table(self, **kwargs):
        rows = {"linker": dict((w, 0.9) for w in RED)}
        agent = make_codemaster(**kwargs)
        agent._pair_index = lyc.board_clue_index(_FakeTable(rows), BOARD)
        agent._pair_index_key = tuple(lyc._normalise(w) for w in BOARD)
        return agent

    def test_group_candidates_still_face_the_legality_filter(self):
        agent = self._agent_with_table()
        with _PatchedLLM(lyc, self._scripted_bad):
            clue, number = agent.get_clue()
        self.assertTrue(lyc.clue_is_legal(clue, BOARD), clue)
        self.assertNotIn(clue, ("KING", "DRAG"))
        self.assertIsInstance(number, int)
        self.assertGreaterEqual(number, 0)

    def test_the_arena_audit_passes(self):
        from harness.arena import _audit_clue
        agent = self._agent_with_table()
        with _PatchedLLM(lyc, self._scripted_bad):
            clue, number = agent.get_clue()
        self.assertIsNone(_audit_clue(clue, number, BOARD))

    def test_a_dead_api_still_returns_a_legal_clue(self):
        agent = self._agent_with_table()

        def dead(system, user):
            raise RuntimeError("API down")

        with _PatchedLLM(lyc, dead):
            clue, number = agent.get_clue()
        self.assertTrue(lyc.clue_is_legal(clue, BOARD), clue)
        self.assertGreaterEqual(number, 1)

    def test_a_clue_is_never_repeated(self):
        agent = self._agent_with_table()
        seen = set()
        with _PatchedLLM(lyc, self._scripted_bad):
            for _ in range(4):
                clue, _number = agent.get_clue()
                self.assertNotIn(clue, seen)
                seen.add(clue)

    def test_the_number_never_exceeds_the_own_words_left(self):
        agent = self._agent_with_table()
        own = [w for w, k in zip(BOARD, KEY) if k == "Red"]
        with _PatchedLLM(lyc, self._scripted_bad):
            _clue, number = agent.get_clue()
        self.assertLessEqual(number, len(own))

    def test_the_scorer_knobs_are_the_champions(self):
        """The whole thesis: the fork does not buy pace with risk."""
        ours = make_codemaster()
        theirs = make_codemaster(module=champ)
        for name in ("bonus_guess_weight", "civilian_penalty", "claimed_slack",
                     "assassin_presence", "opponent_presence",
                     "majority_fraction", "probe_veto_score",
                     "probe_veto_penalty", "probe_relative_floor",
                     "probe_number_cap_score", "probe_top_k",
                     "embed_margin", "embed_penalty", "embed_oov_penalty",
                     "race_mode", "race_full_deficit", "race_slack_scale",
                     "preset", "max_targets", "max_simulated", "n_candidates"):
            self.assertEqual(getattr(ours, name), getattr(theirs, name), name)
        for name in ("ASSASSIN_PENALTY", "OPPONENT_PENALTY",
                     "OPPONENT_PENALTY_SOLO", "CIVILIAN_PENALTY",
                     "BONUS_GUESS_WEIGHT", "CLAIMED_SLACK", "PRESETS",
                     "DEFAULT_PRESET", "MOVE_DEADLINE_S", "MOVE_WALL_S",
                     "CALL_TIMEOUT_S"):
            self.assertEqual(getattr(lyc, name), getattr(champ, name), name)

    def test_the_guesser_is_the_champions(self):
        for name in dir(champ_g):
            if name.startswith("__") or name in ("AGENT_VERSION",
                                                 "DEBUG_PREFIX"):
                continue
            value = getattr(champ_g, name)
            if isinstance(value, (int, float, str, tuple, frozenset, bool)):
                self.assertEqual(getattr(lyc_g, name, None), value, name)

    def test_the_guesser_class_still_answers(self):
        guesser = lyc_g.AIGuesser("Red", quiet=True, api_key=None)
        guesser.set_board(list(BOARD))
        guesser.set_clue("ROYAL", 2)
        answer = guesser.get_answer()
        self.assertIn(answer, BOARD)


if __name__ == "__main__":
    unittest.main()
