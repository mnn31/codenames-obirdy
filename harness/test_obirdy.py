"""Offline unit tests for the oBirdy agents.

No network, no API key: every LLM call is monkeypatched at the agents' own
``_LLM.chat`` boundary, which is the only place either file touches Anthropic.

Run from the repo root::

    python -m harness.test_obirdy
    python -m unittest harness.test_obirdy -v
"""

import gzip
import json
import os
import sys
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from players import codemaster_obirdy as cm_mod  # noqa: E402
from players import guesser_obirdy as g_mod  # noqa: E402
from players.guesser_obirdy import CONTINUE_BASE_DUEL  # noqa: E402


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

# 9 Red / 8 Blue / 7 Civilian / 1 Assassin, matching the engine's composition.
KEY = [
    "Red", "Red", "Red", "Blue", "Blue",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Civilian", "Civilian", "Red", "Civilian", "Blue",
    "Civilian", "Blue", "Red", "Civilian", "Assassin",
]


def patch_llm(module, responder, available=True):
    """Replace ``_LLM.chat``/``available`` with a scripted stand-in."""
    calls = []

    def chat(self, system, user, max_tokens=600, deadline=None):
        calls.append({"system": system, "user": user})
        return responder(system, user)

    module._LLM.chat = chat
    module._LLM.available = lambda self: available
    return calls


def restore_llm(module, original_chat, original_available):
    module._LLM.chat = original_chat
    module._LLM.available = original_available


class _PatchedLLM(object):
    """Context manager so a failed assertion cannot leak a patched module."""

    def __init__(self, module, responder, available=True):
        self.module = module
        self.responder = responder
        self.available = available
        self.calls = None

    def __enter__(self):
        self._chat = self.module._LLM.chat
        self._available = self.module._LLM.available
        self.calls = patch_llm(self.module, self.responder, self.available)
        return self.calls

    def __exit__(self, *exc):
        restore_llm(self.module, self._chat, self._available)
        return False


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------

class TestClueLegality(unittest.TestCase):

    def test_accepts_plain_word(self):
        self.assertTrue(cm_mod.clue_is_legal("OCEAN", BOARD))

    def test_rejects_board_word(self):
        self.assertFalse(cm_mod.clue_is_legal("WHALE", BOARD))

    def test_rejects_substring_of_board_word(self):
        # "RAG" is inside "DRAGON"
        self.assertFalse(cm_mod.clue_is_legal("RAG", BOARD))

    def test_rejects_superstring_of_board_word(self):
        # "CARPET" contains the board word "CAR"
        self.assertFalse(cm_mod.clue_is_legal("CARPET", BOARD))

    def test_rejects_multiword(self):
        self.assertFalse(cm_mod.clue_is_legal("DEEP SEA", BOARD))

    def test_rejects_hyphenated(self):
        self.assertFalse(cm_mod.clue_is_legal("DEEP-SEA", BOARD))

    def test_rejects_digits_and_empty(self):
        self.assertFalse(cm_mod.clue_is_legal("AGENT47", BOARD))
        self.assertFalse(cm_mod.clue_is_legal("", BOARD))
        self.assertFalse(cm_mod.clue_is_legal(None, BOARD))

    def test_revealed_words_do_not_constrain(self):
        board = list(BOARD)
        board[0] = "*Red*"          # WHALE revealed
        self.assertTrue(cm_mod.clue_is_legal("WHALE", board))

    def test_matches_arena_audit(self):
        """Anything we emit must also pass the harness' independent audit."""
        from harness.arena import _audit_clue
        for clue in ("OCEAN", "MYTH", "VOYAGE", "HARVEST"):
            if cm_mod.clue_is_legal(clue, BOARD):
                self.assertIsNone(_audit_clue(clue, 2, BOARD), clue)
        for clue in ("WHALE", "CARPET", "RAG", "DEEP SEA"):
            self.assertFalse(cm_mod.clue_is_legal(clue, BOARD))
            self.assertIsNotNone(_audit_clue(clue, 2, BOARD), clue)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing(unittest.TestCase):

    def test_candidates_from_clean_json(self):
        text = '[{"clue": "ocean", "targets": ["whale", "ship"]}]'
        self.assertEqual(cm_mod._parse_candidates(text),
                         [("OCEAN", ["WHALE", "SHIP"])])

    def test_candidates_from_chatty_json(self):
        text = ('Sure! Here are my clues:\n```json\n'
                '[{"clue":"OCEAN","targets":["WHALE"]},'
                ' {"clue":"ROYAL","targets":["KING","CROWN"]}]\n```\nHope that helps.')
        parsed = cm_mod._parse_candidates(text)
        self.assertEqual([c for c, _ in parsed], ["OCEAN", "ROYAL"])

    def test_candidates_from_garbage_still_yields_something(self):
        parsed = cm_mod._parse_candidates("ocean royal voyage")
        self.assertTrue(parsed)

    def test_candidates_from_empty(self):
        self.assertEqual(cm_mod._parse_candidates(""), [])
        self.assertEqual(cm_mod._parse_candidates(None), [])

    def test_panel_ignores_unknown_words_and_clues(self):
        text = '{"OCEAN": ["WHALE", "NOTONBOARD", "SHIP"], "BOGUS": ["APPLE"]}'
        parsed = cm_mod._parse_panel(text, {"OCEAN"}, BOARD)
        self.assertEqual(parsed, {"OCEAN": ["WHALE", "SHIP"]})

    def test_probe_from_nested_json(self):
        text = '{"OCEAN": {"POISON": 2, "whale": 9}}'
        self.assertEqual(cm_mod._parse_probe(text, {"OCEAN"}, BOARD),
                         {"OCEAN": {"POISON": 2.0, "WHALE": 9.0}})

    def test_probe_from_flat_json_when_one_clue_was_asked(self):
        """An isolated single-clue prompt usually gets the bare word map back."""
        text = '{"POISON": 2, "WHALE": 9}'
        self.assertEqual(cm_mod._parse_probe(text, {"OCEAN"}, BOARD),
                         {"OCEAN": {"POISON": 2.0, "WHALE": 9.0}})

    def test_probe_flat_json_is_ambiguous_for_several_clues(self):
        text = '{"POISON": 2, "WHALE": 9}'
        self.assertEqual(cm_mod._parse_probe(text, {"OCEAN", "GLOW"}, BOARD), {})

    def test_probe_from_chatty_json(self):
        text = ('Sure, here you go:\n```json\n{"OCEAN": {"POISON": 1}}\n```\n'
                'Hope that helps!')
        self.assertEqual(cm_mod._parse_probe(text, {"OCEAN"}, BOARD),
                         {"OCEAN": {"POISON": 1.0}})

    def test_probe_ignores_unknown_words_and_clues(self):
        text = '{"OCEAN": {"POISON": 3, "NOTONBOARD": 9}, "BOGUS": {"WHALE": 9}}'
        self.assertEqual(cm_mod._parse_probe(text, {"OCEAN"}, BOARD),
                         {"OCEAN": {"POISON": 3.0}})

    def test_probe_from_garbage(self):
        for text in ("!!!", "", None, "[1, 2, 3]", '{"OCEAN": "very risky"}'):
            self.assertEqual(cm_mod._parse_probe(text, {"OCEAN"}, BOARD), {})

    def test_probe_scores_are_coerced_and_clamped(self):
        self.assertEqual(cm_mod._coerce_score(99), 10.0)
        self.assertEqual(cm_mod._coerce_score(-5), 0.0)
        self.assertEqual(cm_mod._coerce_score("8/10"), 8.0)
        self.assertEqual(cm_mod._coerce_score(" 3.5 "), 3.5)
        self.assertEqual(cm_mod._coerce_score({"score": 4}), 4.0)
        self.assertIsNone(cm_mod._coerce_score("high"))
        self.assertIsNone(cm_mod._coerce_score(True))
        self.assertIsNone(cm_mod._coerce_score(None))

    def test_probe_tolerates_string_ratings(self):
        text = '{"OCEAN": {"POISON": "7", "WHALE": "10 out of 10"}}'
        self.assertEqual(cm_mod._parse_probe(text, {"OCEAN"}, BOARD),
                         {"OCEAN": {"POISON": 7.0, "WHALE": 10.0}})

    def test_guesser_scores_from_json(self):
        scores = g_mod._parse_scores('{"WHALE": 95, "ship": 70}', BOARD)
        self.assertEqual(scores, {"WHALE": 95.0, "SHIP": 70.0})

    def test_guesser_scores_from_ordered_array(self):
        scores = g_mod._parse_scores('["WHALE", "SHIP"]', BOARD)
        self.assertGreater(scores["WHALE"], scores["SHIP"])

    def test_guesser_scores_from_prose(self):
        scores = g_mod._parse_scores("I would pick WHALE, then SHIP.", BOARD)
        self.assertGreater(scores["WHALE"], scores["SHIP"])

    def test_guesser_scores_from_junk(self):
        self.assertEqual(g_mod._parse_scores("!!!", BOARD), {})


# ---------------------------------------------------------------------------
# Codemaster behaviour
# ---------------------------------------------------------------------------

class TestCodemaster(unittest.TestCase):

    def _agent(self, **kwargs):
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    def test_constructor_signature_matches_baseline(self):
        agent = cm_mod.AICodemaster("Blue")
        self.assertEqual(agent.team, "Blue")
        self.assertEqual(agent.opponent, "Red")

    def test_offline_fallback_is_legal(self):
        agent = self._agent()
        with _PatchedLLM(cm_mod, lambda s, u: None, available=False):
            clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)
        self.assertGreaterEqual(number, 1)

    def test_api_failure_still_returns_legal_clue(self):
        agent = self._agent()
        with _PatchedLLM(cm_mod, lambda s, u: None, available=True):
            clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)
        self.assertGreaterEqual(number, 1)

    def test_illegal_llue_suggestions_are_filtered_out(self):
        """A brainstorm full of illegal clues must not leak one through."""
        agent = self._agent()

        def responder(system, user):
            if "spymaster" in system:
                return ('[{"clue":"WHALE","targets":["WHALE"]},'
                        ' {"clue":"CARPET","targets":["CAR"]},'
                        ' {"clue":"DEEP SEA","targets":["WHALE"]}]')
            return '{"OCEAN": ["WHALE"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)

    def test_a_clue_is_never_given_twice(self):
        """The guesser's top pick for a repeat is already off the board."""
        agent = self._agent()

        def responder(system, user):
            if "spymaster" in system:
                return ('[{"clue":"OCEAN","targets":["WHALE","SHIP"]},'
                        ' {"clue":"ROYAL","targets":["KING"]}]')
            return '{"OCEAN": ["WHALE", "SHIP"], "ROYAL": ["CROWN"]}'

        with _PatchedLLM(cm_mod, responder):
            first, _ = agent.get_clue()
            self.assertEqual(first, "OCEAN")
            second, _ = agent.get_clue()
        self.assertNotEqual(second, "OCEAN")

    def test_repeats_are_read_back_from_move_history(self):
        """Our own state may be fresh; the transcript is authoritative."""
        agent = self._agent()
        agent.set_move_history([
            ["Red_Codemaster", "OCEAN", 2],
            ["Red_Guesser", "WHALE", "*Red*", True],
            ["Blue_Codemaster", "ROYAL", 1],
        ])
        issued = agent._issued_clues()
        self.assertIn("OCEAN", issued)
        self.assertNotIn("ROYAL", issued)   # the opponent's clue is not ours

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            return '{"OCEAN": ["WHALE", "SHIP"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, _ = agent.get_clue()
        self.assertNotEqual(clue, "OCEAN")
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)

    def test_offline_fallback_does_not_loop_on_one_clue(self):
        """A dead API is a pure function of the board; it must still vary."""
        agent = self._agent()
        seen = []
        with _PatchedLLM(cm_mod, lambda s, u: None, available=False):
            for _ in range(5):
                clue, number = agent.get_clue()
                seen.append(clue)
                self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)
        self.assertEqual(len(set(seen)), len(seen), seen)

    def test_exhausted_vocabulary_still_returns_a_legal_clue(self):
        """Last resort: repeating beats returning nothing."""
        agent = self._agent()
        agent._issued = set(cm_mod._FALLBACK_VOCAB) | {
            "SIGNAL", "OBJECT", "SUBJECT", "TOPIC", "IDEA", "THING", "MATTER"}
        with _PatchedLLM(cm_mod, lambda s, u: None, available=False):
            clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)
        self.assertGreaterEqual(number, 1)

    def test_simulation_picks_the_safe_clue(self):
        """Panel says clue A hits reds, clue B hits the assassin -> pick A."""
        agent = self._agent()

        def responder(system, user):
            if "spymaster" in system:
                return ('[{"clue":"OCEAN","targets":["WHALE","SHIP","BEACH"]},'
                        ' {"clue":"VENOM","targets":["WHALE"]}]')
            return ('{"OCEAN": ["WHALE", "SHIP", "BEACH", "APPLE"],'
                    ' "VENOM": ["POISON", "WHALE"]}')

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(number, 3)   # WHALE/SHIP/BEACH are all Red

    def test_number_does_not_exceed_what_the_panel_finds(self):
        """Brainstorm claims 3 targets; the panel only supports 1."""
        agent = self._agent()

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP","BEACH"]}]'
            # APPLE is Blue, so the all-own prefix stops after WHALE.
            return '{"OCEAN": ["WHALE", "APPLE", "SHIP"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertEqual(number, 1)

    def test_assassin_penalty_dominates_word_count(self):
        rankings = [["WHALE", "SHIP", "POISON"]]
        own = ["WHALE", "SHIP", "BEACH"]
        agent = self._agent()
        safe, safe_n = agent._score_candidate([["WHALE", "SHIP"]], own, ["APPLE"],
                                              [], ["POISON"], 2)
        risky, _ = agent._score_candidate(rankings, own, ["APPLE"], [],
                                          ["POISON"], 3)
        self.assertGreater(safe, risky)
        self.assertEqual(safe_n, 2)

    def test_assassin_anywhere_in_the_ranking_is_penalised(self):
        """The old rule only saw danger inside number+1; a stranger's guesser
        will not stop where our panel does, so deep placements count too."""
        agent = self._agent()
        own = ["WHALE", "SHIP", "BEACH"]
        clean = [["WHALE", "SHIP", "APPLE", "TREE", "CAR", "TRAIN"]]
        lurking = [["WHALE", "SHIP", "APPLE", "TREE", "CAR", "POISON"]]
        clean_score, _ = agent._score_candidate(clean, own, ["APPLE", "TREE"],
                                                [], ["POISON"], 2)
        lurking_score, _ = agent._score_candidate(lurking, own, ["APPLE", "TREE"],
                                                  [], ["POISON"], 2)
        self.assertGreater(clean_score, lurking_score)

    def test_assassin_repellence_decays_with_position(self):
        agent = self._agent()
        own = ["WHALE", "SHIP", "BEACH"]
        near = [["WHALE", "SHIP", "BEACH", "POISON", "CAR", "TRAIN"]]
        far = [["WHALE", "SHIP", "BEACH", "CAR", "TRAIN", "POISON"]]
        near_score, _ = agent._score_candidate(near, own, [], [], ["POISON"], 3)
        far_score, _ = agent._score_candidate(far, own, [], [], ["POISON"], 3)
        self.assertGreater(far_score, near_score)

    def test_opponent_repellence_is_weaker_than_assassin_repellence(self):
        agent = self._agent()
        own = ["WHALE", "SHIP", "BEACH"]
        with_opponent = [["WHALE", "SHIP", "BEACH", "CAR", "TRAIN", "APPLE"]]
        with_assassin = [["WHALE", "SHIP", "BEACH", "CAR", "TRAIN", "POISON"]]
        opp_score, _ = agent._score_candidate(
            with_opponent, own, ["APPLE"], [], ["POISON"], 3)
        assassin_score, _ = agent._score_candidate(
            with_assassin, own, ["APPLE"], [], ["POISON"], 3)
        clean = [["WHALE", "SHIP", "BEACH", "CAR", "TRAIN", "DESERT"]]
        clean_score, _ = agent._score_candidate(
            clean, own, ["APPLE"], [], ["POISON"], 3)
        self.assertGreater(clean_score, opp_score)
        self.assertGreater(opp_score, assassin_score)

    def test_repellence_weights_are_configurable(self):
        own = ["WHALE", "SHIP", "BEACH"]
        lurking = [["WHALE", "SHIP", "BEACH", "CAR", "TRAIN", "POISON"]]
        off = cm_mod.AICodemaster("Red", assassin_presence=0.0,
                                  opponent_presence=0.0)
        off.set_game_state(list(BOARD), list(KEY))
        on = self._agent(assassin_presence=8.0)
        off_score, _ = off._score_candidate(lurking, own, [], [], ["POISON"], 3)
        on_score, _ = on._score_candidate(lurking, own, [], [], ["POISON"], 3)
        self.assertGreater(off_score, on_score)

    def test_majority_fraction_controls_how_far_the_number_stretches(self):
        own = ["WHALE", "SHIP", "BEACH"]
        # One sample supports a prefix of 3, the other only 1.
        rankings = [["WHALE", "SHIP", "BEACH"], ["WHALE", "APPLE", "TREE"]]
        lenient = self._agent(majority_fraction=0.5)
        strict = self._agent(majority_fraction=1.0)
        _, lenient_n = lenient._score_candidate(rankings, own, ["APPLE", "TREE"],
                                                [], ["POISON"], 3)
        _, strict_n = strict._score_candidate(rankings, own, ["APPLE", "TREE"],
                                              [], ["POISON"], 3)
        self.assertEqual(lenient_n, 3)
        self.assertEqual(strict_n, 1)

    def test_claimed_slack_lets_the_panel_outrun_a_timid_brainstorm(self):
        own = ["WHALE", "SHIP", "BEACH"]
        rankings = [["WHALE", "SHIP", "BEACH"]]
        capped = self._agent()           # slack is held at 0 by default
        loose = self._agent(claimed_slack=1)
        _, capped_n = capped._score_candidate(rankings, own, [], [],
                                              ["POISON"], 1)
        _, loose_n = loose._score_candidate(rankings, own, [], [],
                                            ["POISON"], 1)
        self.assertEqual(capped_n, 1)
        self.assertEqual(loose_n, 2)

    def test_claimed_slack_never_invents_unsupported_words(self):
        """Slack raises the cap, it does not raise what the panel found."""
        own = ["WHALE", "SHIP", "BEACH"]
        rankings = [["WHALE", "APPLE", "SHIP"]]     # APPLE is Blue
        agent = self._agent(claimed_slack=3)
        _, number = agent._score_candidate(rankings, own, ["APPLE"], [],
                                           ["POISON"], 3)
        self.assertEqual(number, 1)

    def test_panel_groups_default_to_one_batched_call(self):
        agent = self._agent()
        self.assertEqual(agent._panel_groups(["A", "B", "C", "D"]),
                         [["A", "B", "C", "D"]])

    def test_panel_isolation_splits_the_leading_candidates(self):
        agent = self._agent(panel_isolated=2)
        self.assertEqual(agent._panel_groups(["A", "B", "C", "D"]),
                         [["A"], ["B"], ["C", "D"]])
        # No tail left over means no empty trailing group.
        self.assertEqual(agent._panel_groups(["A", "B"]), [["A"], ["B"]])

    def test_isolated_panel_still_scores_every_candidate(self):
        agent = self._agent(panel_isolated=2, panel_samples=1)
        seen = []

        def responder(system, user):
            if "spymaster" in system:
                return ('[{"clue":"OCEAN","targets":["WHALE","SHIP"]},'
                        ' {"clue":"ROYAL","targets":["KING"]}]')
            seen.append(user)
            if "OCEAN" in user:
                return '{"OCEAN": ["WHALE", "SHIP"]}'
            return '{"ROYAL": ["KING"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        # Each leading candidate gets a panel prompt to itself; the offline
        # fallback candidates share the batched tail call.
        headers = [line for prompt in seen for line in prompt.splitlines()
                   if line.startswith("Clues: ")]
        self.assertIn("Clues: OCEAN", headers)
        self.assertIn("Clues: ROYAL", headers)
        self.assertEqual(clue, "OCEAN")

    def test_two_team_detection_from_move_history(self):
        agent = self._agent()
        self.assertFalse(agent._two_team_mode())
        agent.set_move_history([["Red_Codemaster", "OCEAN", 2],
                                ["Blue_Codemaster", "ROYAL", 1]])
        self.assertTrue(agent._two_team_mode())

    def test_leftover_targets_are_remembered(self):
        agent = self._agent()

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            return '{"OCEAN": ["WHALE", "SHIP"]}'

        with _PatchedLLM(cm_mod, responder):
            agent.get_clue()
        self.assertIn("WHALE", agent._clued_targets)

    def test_panel_picks_are_remembered_as_clued_targets(self):
        """The panel's own leading picks count as pointed-at, not just the
        brainstorm's declared targets -- the guesser will chase the former."""
        agent = self._agent()

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"OCEAN","targets":[]}]'
            return '{"OCEAN": ["WHALE", "SHIP", "APPLE"]}'

        with _PatchedLLM(cm_mod, responder):
            agent.get_clue()
        self.assertIn("WHALE", agent._clued_targets)
        self.assertNotIn("APPLE", agent._clued_targets)   # Blue, never ours

    def test_deadline_is_respected(self):
        """A slow API must not blow the per-move budget."""
        agent = self._agent(deadline=3.0)

        def responder(system, user):
            time.sleep(1.2)
            return None

        started = time.time()
        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        elapsed = time.time() - started
        self.assertLess(elapsed, 12.0)
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)

    def test_never_raises_on_garbage_state(self):
        agent = cm_mod.AICodemaster("Red")
        agent.set_game_state([], [])
        clue, number = agent.get_clue()
        self.assertIsInstance(clue, str)
        self.assertGreaterEqual(int(number), 1)

    def test_exception_inside_pipeline_is_contained(self):
        agent = self._agent()

        def boom(system, user):
            raise RuntimeError("simulated crash")

        with _PatchedLLM(cm_mod, boom):
            clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)


class TestCodemasterSweep(unittest.TestCase):
    """Clue number 0 = unlimited guesses; only safe in a narrow endgame.

    Retired by default in Pidgeotto (a paired A/B came back exactly level and
    the move is the highest-variance one available), so these tests opt in
    explicitly and the gate itself is still expected to work.
    """

    def _endgame(self, own_left=3, **kwargs):
        """Board where only ``own_left`` Red words are still hidden."""
        kwargs.setdefault("sweep", True)
        agent = cm_mod.AICodemaster("Red", **kwargs)
        board = list(BOARD)
        reds = [i for i, kind in enumerate(KEY) if kind == "Red"]
        for index in reds[own_left:]:
            board[index] = "*Red*"
        agent.set_game_state(board, list(KEY))
        return agent, board

    def test_sweep_when_leftovers_cover_the_rest(self):
        agent, board = self._endgame(own_left=3)
        # WHALE / SHIP / BEACH are the three Red words still hidden.
        agent._clued_targets.update({"SHIP", "BEACH"})

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"MAMMAL","targets":["WHALE"]}]'
            # Everything after WHALE is a harmless civilian.
            return '{"MAMMAL": ["WHALE", "SNOW", "DESERT", "TOWER"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertEqual(number, 0)
        self.assertEqual(agent.sweeps_issued, 1)
        self.assertTrue(cm_mod.clue_is_legal(clue, board), clue)

    def test_no_sweep_without_leftover_coverage(self):
        agent, _ = self._endgame(own_left=3)

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"MAMMAL","targets":["WHALE"]}]'
            return '{"MAMMAL": ["WHALE", "SNOW", "DESERT", "TOWER"]}'

        with _PatchedLLM(cm_mod, responder):
            _, number = agent.get_clue()
        self.assertEqual(number, 1)
        self.assertEqual(agent.sweeps_issued, 0)

    def test_no_sweep_when_assassin_is_in_the_window(self):
        agent, _ = self._endgame(own_left=3)
        agent._clued_targets.update({"SHIP", "BEACH"})

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"MAMMAL","targets":["WHALE"]}]'
            # POISON is the assassin and sits inside the sweep window.
            return '{"MAMMAL": ["WHALE", "HORSE", "POISON", "TREE"]}'

        with _PatchedLLM(cm_mod, responder):
            _, number = agent.get_clue()
        self.assertEqual(number, 1)
        self.assertEqual(agent.sweeps_issued, 0)

    def test_no_sweep_when_opponent_word_is_in_the_window(self):
        agent, _ = self._endgame(own_left=3)
        agent._clued_targets.update({"SHIP", "BEACH"})

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"MAMMAL","targets":["WHALE"]}]'
            # APPLE is Blue.
            return '{"MAMMAL": ["WHALE", "APPLE", "HORSE", "TREE"]}'

        with _PatchedLLM(cm_mod, responder):
            _, number = agent.get_clue()
        self.assertEqual(number, 1)

    def test_no_sweep_early_in_the_game(self):
        agent = cm_mod.AICodemaster("Red")
        agent.set_game_state(list(BOARD), list(KEY))   # all 9 reds hidden
        agent._clued_targets.update(set(w for w in BOARD))

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"MAMMAL","targets":["WHALE"]}]'
            return '{"MAMMAL": ["WHALE", "SNOW", "DESERT", "TOWER"]}'

        with _PatchedLLM(cm_mod, responder):
            _, number = agent.get_clue()
        self.assertGreaterEqual(number, 1)
        self.assertEqual(agent.sweeps_issued, 0)

    def test_sweep_is_off_by_default(self):
        self.assertFalse(cm_mod.AICodemaster("Red").allow_sweep)

    def test_sweep_can_be_disabled(self):
        agent, _ = self._endgame(own_left=3, sweep=False)
        agent._clued_targets.update({"SHIP", "BEACH"})

        def responder(system, user):
            if "spymaster" in system:
                return '[{"clue":"MAMMAL","targets":["WHALE"]}]'
            return '{"MAMMAL": ["WHALE", "SNOW", "DESERT", "TOWER"]}'

        with _PatchedLLM(cm_mod, responder):
            _, number = agent.get_clue()
        self.assertEqual(number, 1)

    def test_zero_passes_the_arena_audit(self):
        from harness.arena import _audit_clue
        self.assertIsNone(_audit_clue("MAMMAL", 0, BOARD))
        self.assertIsNotNone(_audit_clue("MAMMAL", -1, BOARD))


# ---------------------------------------------------------------------------
# Danger probe + the pidgeot preset
# ---------------------------------------------------------------------------

def is_probe(system):
    """The danger probe is the only call whose persona rates pull strength."""
    return "rate how strongly" in system


def is_brainstorm(system):
    return "spymaster" in system


class TestPresets(unittest.TestCase):
    """A preset seeds knobs; explicit kwargs and shipped defaults still win."""

    def test_default_is_pidgeot_and_ships_the_probe_on(self):
        """Intentional behaviour change: the probe is now a shipped default.

        Clue safety, not clue ambition, is what the forensics say is losing
        games -- seed 10's PAGEANT 2 into STATE and five of eight against an
        embedding guesser, all with the probe switched off.
        """
        agent = cm_mod.AICodemaster("Red")
        self.assertEqual(agent.preset, "pidgeot")
        self.assertTrue(agent.probe_enabled)
        self.assertTrue(agent.embed_sensor)
        # The *numbers* stay conservative: only the safety nets are new.
        self.assertEqual(agent.claimed_slack, cm_mod.CLAIMED_SLACK)
        self.assertEqual(agent.bonus_guess_weight, cm_mod.BONUS_GUESS_WEIGHT)
        self.assertEqual(agent.civilian_penalty, cm_mod.CIVILIAN_PENALTY)

    def test_the_probe_walks_two_candidates_at_most(self):
        self.assertEqual(cm_mod.AICodemaster("Red").probe_top_k, 2)

    def test_ambitious_preset_is_parked_but_still_reachable(self):
        agent = cm_mod.AICodemaster("Red", preset="ambitious")
        self.assertEqual(agent.claimed_slack, 1)
        self.assertAlmostEqual(agent.bonus_guess_weight, 0.55)
        self.assertAlmostEqual(agent.civilian_penalty, 0.45)

    def test_explicit_kwargs_override_the_preset(self):
        agent = cm_mod.AICodemaster("Red", preset="ambitious",
                                    claimed_slack=0, probe=False)
        self.assertEqual(agent.claimed_slack, 0)
        self.assertFalse(agent.probe_enabled)

    def test_unknown_preset_degrades_to_the_shipped_config(self):
        agent = cm_mod.AICodemaster("Red", preset="mewtwo")
        self.assertTrue(agent.probe_enabled)
        self.assertEqual(agent.claimed_slack, cm_mod.CLAIMED_SLACK)

    def test_probe_knobs_are_kwargs_tunable(self):
        agent = cm_mod.AICodemaster("Red", probe=True, probe_top_k=1,
                                    probe_veto_score=9.5,
                                    probe_veto_penalty=3.0,
                                    probe_assassin_weight=0.0,
                                    probe_opponent_weight=2.0,
                                    probe_missing_penalty=1.5,
                                    probe_unprobed_cap=1)
        self.assertTrue(agent.probe_enabled)
        self.assertEqual(agent.probe_top_k, 1)
        self.assertAlmostEqual(agent.probe_veto_score, 9.5)
        self.assertAlmostEqual(agent.probe_veto_penalty, 3.0)
        self.assertAlmostEqual(agent.probe_assassin_weight, 0.0)
        self.assertAlmostEqual(agent.probe_opponent_weight, 2.0)
        self.assertAlmostEqual(agent.probe_missing_penalty, 1.5)
        self.assertEqual(agent.probe_unprobed_cap, 1)

    def test_sensor_knobs_are_kwargs_tunable(self):
        agent = cm_mod.AICodemaster("Red", embed_sensor=False,
                                    embed_margin=0.5, embed_penalty=1.0,
                                    embed_oov_penalty=2.0)
        self.assertFalse(agent.embed_sensor)
        self.assertAlmostEqual(agent.embed_margin, 0.5)
        self.assertAlmostEqual(agent.embed_penalty, 1.0)
        self.assertAlmostEqual(agent.embed_oov_penalty, 2.0)

    def test_preset_reaches_usage_summary(self):
        agent = cm_mod.AICodemaster("Red", preset="ambitious")
        summary = agent.usage_summary()
        self.assertEqual(summary["preset"], "ambitious")
        self.assertEqual(summary["probes_run"], 0)
        self.assertEqual(summary["probes_vetoed"], 0)
        self.assertEqual(summary["embed_flagged"], 0)
        self.assertEqual(summary["embed_oov"], 0)


class TestDangerProbe(unittest.TestCase):
    """The probe prices the assassin the panel never surfaced."""

    def _agent(self, **kwargs):
        # The sensor is a *separate* signal with its own tests; keep it out of
        # the probe's fixtures so a fixture clue's GloVe neighbourhood cannot
        # quietly decide these outcomes.
        kwargs.setdefault("embed_sensor", False)
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    # -- scenario: two clues, only one of which secretly pulls the assassin --

    BRAINSTORM = ('[{"clue":"OCEAN","targets":["WHALE","SHIP","BEACH"]},'
                  ' {"clue":"GLOW","targets":["LASER","ROBOT"]}]')
    PANEL = ('{"OCEAN": ["WHALE", "SHIP", "BEACH"],'
             ' "GLOW": ["LASER", "ROBOT"]}')

    def _responder(self, probe_replies):
        """Brainstorm + panel are fixed; only the probe replies vary."""

        def responder(system, user):
            if is_brainstorm(system):
                return self.BRAINSTORM
            if is_probe(system):
                for clue, reply in probe_replies.items():
                    if "Clue: %s" % clue in user or "Clues: %s" % clue in user:
                        return reply
                return None
            return self.PANEL

        return responder

    def test_high_assassin_pull_kills_the_best_scoring_clue(self):
        agent = self._agent()
        responder = self._responder({
            "OCEAN": '{"POISON": 9, "WHALE": 8, "APPLE": 1}',
            "GLOW": '{"POISON": 0, "LASER": 9, "APPLE": 1}',
        })
        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertEqual(clue, "GLOW")
        self.assertEqual(agent.probes_vetoed, 1)

    def test_the_same_clue_wins_when_the_probe_comes_back_clean(self):
        """Control arm: identical scenario, harmless assassin ratings.

        Only the winner is probed, so a clean leader costs exactly one call --
        the whole point of the shipped shape.
        """
        agent = self._agent()
        responder = self._responder({
            "OCEAN": '{"POISON": 0, "WHALE": 9, "APPLE": 1}',
            "GLOW": '{"POISON": 0, "LASER": 9, "APPLE": 1}',
        })
        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(agent.probes_vetoed, 0)
        self.assertEqual(agent.probes_run, 1)

    #: Three scored candidates, so the two-probe ceiling has something to fall
    #: through *to*.
    BRAINSTORM3 = ('[{"clue":"OCEAN","targets":["WHALE","SHIP","BEACH"]},'
                   ' {"clue":"GLOW","targets":["LASER","ROBOT"]},'
                   ' {"clue":"ROYAL","targets":["CROWN"]}]')
    PANEL3 = ('{"OCEAN": ["WHALE", "SHIP", "BEACH"],'
              ' "GLOW": ["LASER", "ROBOT"], "ROYAL": ["CROWN"]}')

    def _poisonous(self, probed):
        def responder(system, user):
            if is_brainstorm(system):
                return self.BRAINSTORM3
            if is_probe(system):
                for line in user.splitlines():
                    if line.startswith("Clue: "):
                        probed.append(line[len("Clue: "):])
                return '{"POISON": 9, "WHALE": 8, "LASER": 8, "CROWN": 8}'
            return self.PANEL3

        return responder

    def test_a_second_veto_exhausts_the_budget_and_stops_probing(self):
        """Two probes is the ceiling: the third candidate goes unprobed.

        An unprobed candidate is not a trusted one -- it takes the missing
        penalty and the conservative number -- but it beats knowingly giving a
        clue the probe has just called poisoned.
        """
        agent = self._agent()
        probed = []
        with _PatchedLLM(cm_mod, self._poisonous(probed)):
            clue, number = agent.get_clue()
        self.assertEqual(len(probed), 2)
        self.assertEqual(agent.probes_vetoed, 2)
        self.assertNotIn(clue, probed)
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)
        self.assertLessEqual(number, cm_mod.PROBE_UNPROBED_CAP)

    def test_every_candidate_vetoed_still_returns_one_of_them(self):
        """Nothing survives even an unlimited budget: least-bad, one word."""
        agent = self._agent(probe_top_k=99)
        probed = []
        with _PatchedLLM(cm_mod, self._poisonous(probed)):
            clue, number = agent.get_clue()
        self.assertEqual(sorted(probed), ["GLOW", "OCEAN", "ROYAL"])
        self.assertIn(clue, ("OCEAN", "GLOW", "ROYAL"))
        self.assertEqual(number, 1)
        self.assertEqual(agent.probes_vetoed, 3)

    def test_relative_veto_fires_below_the_absolute_threshold(self):
        """An assassin that outranks our own targets is a veto at any level."""
        agent = self._agent()
        ratings = {"POISON": 4.0, "WHALE": 3.0}
        score, number = agent._apply_probe(5.0, 3, 3, ratings,
                                           ["APPLE"], ["POISON"])
        self.assertLess(score, -40.0)
        self.assertEqual(number, 1)

    def test_relative_veto_ignores_a_board_nobody_associates_with(self):
        agent = self._agent()
        ratings = {"POISON": 1.0, "WHALE": 0.0}
        score, number = agent._apply_probe(5.0, 3, 3, ratings,
                                           ["APPLE"], ["POISON"])
        self.assertGreater(score, 4.0)
        self.assertEqual(number, 3)

    def test_relative_veto_compares_against_the_marginal_target(self):
        """sweep_pidgeot seed 13: 9 probes run, nothing vetoed, dead anyway.

        Every clue points hard at its headline word, so comparing the
        assassin against the *best* reference could essentially never fire.
        The word at risk is the weakest one we are counting on.
        """
        agent = self._agent()
        # WHALE is the headline word, SHIP the second word the number buys.
        ratings = {"POISON": 4.5, "WHALE": 9.0, "SHIP": 4.0}
        score, number = agent._apply_probe(5.0, 2, 2, ratings,
                                           ["APPLE"], ["POISON"])
        self.assertLess(score, -40.0)
        self.assertEqual(number, 1)
        self.assertEqual(agent.probes_vetoed, 1)

    def test_an_assassin_under_every_target_is_not_vetoed(self):
        agent = self._agent()
        ratings = {"POISON": 2.0, "WHALE": 9.0, "SHIP": 7.0}
        score, number = agent._apply_probe(5.0, 2, 2, ratings,
                                           ["APPLE"], ["POISON"])
        self.assertGreater(score, 4.0)
        self.assertEqual(number, 2)
        self.assertEqual(agent.probes_vetoed, 0)

    def test_probe_detail_is_recorded_for_offline_forensics(self):
        """The battery only had counters, which could not say why."""
        agent = self._agent()
        agent._apply_probe(5.0, 2, 2, {"POISON": 4.5, "WHALE": 9.0,
                                       "SHIP": 4.0, "APPLE": 6.0},
                           ["APPLE"], ["POISON"])
        entry = agent.probe_log[-1]
        self.assertAlmostEqual(entry["assassin"], 4.5)
        self.assertAlmostEqual(entry["margin"], 4.0)
        self.assertEqual(entry["opponent"], [6.0])
        self.assertTrue(entry["veto"])
        self.assertEqual(agent.usage_summary()["probe_log"], agent.probe_log)

    def test_the_log_names_the_clue_the_veto_killed(self):
        """The first battery could count vetoes but not name them.

        Fifteen vetoes across 38 games and no way to say which word each one
        rejected, so no way to judge any of them a false positive offline.
        """
        agent = self._agent()
        responder = self._responder({
            "OCEAN": '{"POISON": 9, "WHALE": 8, "APPLE": 1}',
            "GLOW": '{"POISON": 0, "LASER": 9, "APPLE": 1}',
        })
        with _PatchedLLM(cm_mod, responder):
            clue, _ = agent.get_clue()
        killed = [entry for entry in agent.probe_log if entry["veto"]]
        self.assertEqual([entry["clue"] for entry in killed], ["OCEAN"])
        self.assertEqual(agent.probe_log[-1]["clue"], clue)

    def test_the_log_says_whether_the_sensor_agreed(self):
        agent = self._agent()
        agent._embed_flagged_clues.add("OCEAN")
        ratings = {"POISON": 4.0, "WHALE": 3.0}
        agent._apply_probe(5.0, 3, 3, ratings, ["APPLE"], ["POISON"],
                           clue="OCEAN")
        agent._apply_probe(5.0, 3, 3, ratings, ["APPLE"], ["POISON"],
                           clue="GLOW")
        self.assertTrue(agent.probe_log[0]["embed_flagged"])
        self.assertFalse(agent.probe_log[1]["embed_flagged"])

    def test_corroboration_is_off_by_default(self):
        """No false-positive evidence to spend the net on -- see the constant."""
        self.assertFalse(cm_mod.AICodemaster("Red").probe_veto_requires_embed)

    def test_corroboration_holds_back_an_uncorroborated_relative_veto(self):
        # 3.0 for the same reason as above: it isolates the veto from the
        # number cap, which fires at 4.0 whatever the veto decided.
        agent = self._agent(probe_veto_requires_embed=True)
        ratings = {"POISON": 3.0, "WHALE": 3.0}
        score, number = agent._apply_probe(5.0, 3, 3, ratings,
                                           ["APPLE"], ["POISON"], clue="OCEAN")
        self.assertGreater(score, 4.0)
        self.assertEqual(number, 3)
        self.assertEqual(agent.probes_vetoed, 0)
        self.assertFalse(agent.probe_log[-1]["veto"])

    def test_corroboration_lets_a_seconded_relative_veto_through(self):
        agent = self._agent(probe_veto_requires_embed=True)
        agent._embed_flagged_clues.add("OCEAN")
        ratings = {"POISON": 4.0, "WHALE": 3.0}
        score, number = agent._apply_probe(5.0, 3, 3, ratings,
                                           ["APPLE"], ["POISON"], clue="OCEAN")
        self.assertLess(score, -40.0)
        self.assertEqual(number, 1)
        self.assertTrue(agent.probe_log[-1]["veto"])

    def test_corroboration_never_gates_the_absolute_veto(self):
        """A 5-of-10 pull on the assassin is a ceiling, not a comparison."""
        agent = self._agent(probe_veto_requires_embed=True)
        ratings = {"POISON": 9.0, "WHALE": 8.0}
        score, number = agent._apply_probe(5.0, 3, 3, ratings,
                                           ["APPLE"], ["POISON"], clue="OCEAN")
        self.assertLess(score, -40.0)
        self.assertEqual(number, 1)

    def test_corroboration_is_kwarg_tunable_and_reaches_the_relative_veto(self):
        loose = self._agent()
        strict = self._agent(probe_veto_requires_embed=True)
        self.assertFalse(loose.probe_veto_requires_embed)
        self.assertTrue(strict.probe_veto_requires_embed)
        ratings = {"POISON": 4.0, "WHALE": 3.0}
        self.assertTrue(loose._apply_probe(5.0, 3, 3, ratings, ["APPLE"],
                                           ["POISON"], clue="OCEAN")[0] < -40)
        self.assertTrue(strict._apply_probe(5.0, 3, 3, ratings, ["APPLE"],
                                            ["POISON"], clue="OCEAN")[0] > 4)

    def test_probe_log_stays_out_of_a_probe_free_summary(self):
        agent = cm_mod.AICodemaster("Red")
        self.assertNotIn("probe_log", agent.usage_summary())

    def test_relative_veto_can_be_disabled(self):
        # Rated 3.0 rather than 4.0 deliberately: at 3.0 the relative veto is
        # still armed to fire (it ties the marginal target and clears the
        # floor), but the number cap is not, so the number surviving intact is
        # evidence about the veto alone.
        agent = self._agent(probe_relative_veto=False)
        ratings = {"POISON": 3.0, "WHALE": 3.0}
        score, number = agent._apply_probe(5.0, 3, 3, ratings,
                                           ["APPLE"], ["POISON"])
        self.assertGreater(score, 4.0)
        self.assertEqual(number, 3)

    def test_opponent_pull_is_a_softer_penalty_than_assassin_pull(self):
        """Same rating on both words: the assassin must cost strictly more."""
        agent = self._agent()
        clean, _ = agent._apply_probe(5.0, 2, 2,
                                      {"POISON": 0.0, "APPLE": 0.0,
                                       "WHALE": 9.0},
                                      ["APPLE"], ["POISON"])
        opponent, _ = agent._apply_probe(5.0, 2, 2,
                                         {"POISON": 0.0, "APPLE": 4.0,
                                          "WHALE": 9.0},
                                         ["APPLE"], ["POISON"])
        assassin, _ = agent._apply_probe(5.0, 2, 2,
                                         {"POISON": 4.0, "APPLE": 0.0,
                                          "WHALE": 9.0},
                                         ["APPLE"], ["POISON"])
        self.assertGreater(clean, opponent)
        self.assertGreater(opponent, assassin)

    # -- conservative fallback --------------------------------------------

    def test_unprobed_candidate_loses_its_claimed_slack(self):
        """No answer about the danger words means no extra word on the clue.

        Only visible with the slack armed, so these four run the parked
        ``ambitious`` numbers: the shipped default has no slack to lose.
        """
        agent = self._agent(preset="ambitious")

        def responder(system, user):
            if is_brainstorm(system):
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            if is_probe(system):
                return None            # the probe call failed
            return '{"OCEAN": ["WHALE", "SHIP", "BEACH"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(number, 2)    # claimed 2, not claimed + slack

    def test_a_clean_probe_earns_the_extra_word(self):
        agent = self._agent(preset="ambitious")

        def responder(system, user):
            if is_brainstorm(system):
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            if is_probe(system):
                return '{"POISON": 0, "WHALE": 9}'
            return '{"OCEAN": ["WHALE", "SHIP", "BEACH"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(number, 3)

    def test_unparseable_probe_reply_is_treated_as_unprobed(self):
        agent = self._agent(preset="ambitious")

        def responder(system, user):
            if is_brainstorm(system):
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            if is_probe(system):
                return "looks risky to me, honestly"
            return '{"OCEAN": ["WHALE", "SHIP", "BEACH"]}'

        with _PatchedLLM(cm_mod, responder):
            _, number = agent.get_clue()
        self.assertEqual(number, 2)

    def test_probe_crash_is_contained_and_stays_conservative(self):
        agent = self._agent(preset="ambitious")

        def responder(system, user):
            if is_brainstorm(system):
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            if is_probe(system):
                raise RuntimeError("probe exploded")
            return '{"OCEAN": ["WHALE", "SHIP", "BEACH"]}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)
        self.assertEqual(number, 2)

    def test_unprobed_candidate_without_claimed_targets_is_capped(self):
        agent = self._agent()
        self.assertEqual(agent._apply_probe(5.0, 4, 0, None, [], [])[1],
                         cm_mod.PROBE_UNPROBED_CAP)
        self.assertEqual(agent._apply_probe(5.0, 4, 0, None, [], [])[0],
                         5.0 - agent.probe_missing_penalty)

    SCORED3 = [
        ("OCEAN", ["WHALE"], [["WHALE", "SHIP"]], 2.0, 2),
        ("GLOW", ["LASER"], [["LASER"]], 1.0, 1),
        ("ROYAL", ["SHIP"], [["SHIP"]], 0.5, 1),
    ]

    def test_probe_is_skipped_when_the_budget_is_gone(self):
        agent = self._agent()
        calls = []

        def responder(system, user):
            calls.append(user)
            return '{"POISON": 0}'

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent._probe_select(
                self.SCORED3[:1], ["WHALE", "SHIP"], ["APPLE"], ["POISON"],
                cm_mod._Deadline(1.0))
        self.assertEqual(calls, [])
        # No answer is not a pass: the leader still loses its slack.
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(number, 1)

    def test_a_thin_budget_probes_only_the_leader(self):
        agent = self._agent(probe_top_k=3)
        calls = []

        def responder(system, user):
            calls.append(user)
            return '{"POISON": 9, "WHALE": 1}'      # veto everything

        with _PatchedLLM(cm_mod, responder):
            agent._probe_select(
                self.SCORED3, ["WHALE", "SHIP", "LASER"], ["APPLE"],
                ["POISON"], cm_mod._Deadline(cm_mod.PROBE_TIGHT_SECONDS - 3.0))
        self.assertEqual(len(calls), 1)

    def test_a_full_budget_walks_past_a_veto(self):
        agent = self._agent(probe_top_k=3)
        calls = []

        def responder(system, user):
            calls.append(user)
            return '{"POISON": 9, "WHALE": 1}'

        with _PatchedLLM(cm_mod, responder):
            agent._probe_select(
                self.SCORED3, ["WHALE", "SHIP", "LASER"], ["APPLE"],
                ["POISON"], cm_mod._Deadline(40.0))
        self.assertEqual(len(calls), 3)

    def test_probe_is_skipped_when_the_api_is_unavailable(self):
        agent = self._agent()
        with _PatchedLLM(cm_mod, lambda s, u: '{"POISON": 0}', available=False):
            result = agent._probe_select(self.SCORED3[:1], ["WHALE"],
                                         ["APPLE"], ["POISON"],
                                         cm_mod._Deadline(40.0))
        self.assertEqual(result, ("OCEAN", 1))
        self.assertEqual(agent.probes_run, 0)

    # -- prompt shape ------------------------------------------------------

    def test_only_the_winning_candidate_is_probed(self):
        """One extra call per turn is the shipped cost."""
        agent = self._agent()
        prompts = []

        def responder(system, user):
            if is_brainstorm(system):
                return self.BRAINSTORM
            if is_probe(system):
                prompts.append(user)
                return '{"POISON": 0}'
            return self.PANEL

        with _PatchedLLM(cm_mod, responder):
            agent.get_clue()
        headers = [line for prompt in prompts for line in prompt.splitlines()
                   if line.startswith("Clue")]
        self.assertEqual(headers, ["Clue: OCEAN"])

    def test_probe_asks_about_the_assassin_and_mixes_in_own_words(self):
        agent = self._agent()
        prompts = []

        def responder(system, user):
            if is_brainstorm(system):
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            if is_probe(system):
                prompts.append(user)
                return '{"POISON": 0}'
            return '{"OCEAN": ["WHALE", "SHIP"]}'

        with _PatchedLLM(cm_mod, responder):
            agent.get_clue()
        asked = [line for prompt in prompts for line in prompt.splitlines()
                 if line.startswith("Rate from 0 to 10")]
        self.assertTrue(asked)
        self.assertIn("POISON", asked[0])
        self.assertIn("APPLE", asked[0])        # an opponent word
        self.assertIn("WHALE", asked[0])        # own reference word

    def test_probe_never_tells_the_panel_persona_the_key(self):
        """The probe persona must stay a no-key guesser, like the panel."""
        agent = self._agent()
        systems = []

        def responder(system, user):
            if is_brainstorm(system):
                return '[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]'
            if is_probe(system):
                systems.append(system)
                return '{"POISON": 0}'
            return '{"OCEAN": ["WHALE", "SHIP"]}'

        with _PatchedLLM(cm_mod, responder):
            agent.get_clue()
        self.assertTrue(systems)
        for system in systems:
            self.assertIn("NOT which team they belong to", system)
            self.assertNotIn("assassin", system.lower())


# ---------------------------------------------------------------------------
# Embedding danger sensor (bundled similarity table)
# ---------------------------------------------------------------------------

SENSOR_LEVELS = [round(i / 63.0, 5) for i in range(64)]


def write_simtable(path, board_words, rows, levels=None, version=1):
    """Write a table in the shipped format.  Mirrors ``harness/simtable.py``.

    ``rows`` is ``{clue: {board word: similarity}}``.  Keeping an independent
    writer here means the reader is tested against the *format*, not against
    whatever the builder happens to emit today.  Clue words are stored
    lower-case (GloVe's own casing) and board words upper-case, exactly as the
    builder writes them.

    ``version`` picks the layout: 1 packs each pair into a 16-bit word, 2 keeps
    the board indices and the codes in parallel arrays.  It defaults to 1 so
    the sensor tests below go on exercising the layout every table shipped
    before 2026-08-03 used -- the reader has to keep reading those.
    """
    import gzip as _gzip

    levels = list(levels if levels is not None else SENSOR_LEVELS)
    board = [w.upper() for w in board_words]
    columns = dict((w, i) for i, w in enumerate(board))
    lowered = dict((clue.lower(), values) for clue, values in rows.items())
    clues = sorted(lowered)
    offsets = [0]
    entry_columns = []
    entry_codes = []
    for clue in clues:
        for word, value in sorted(lowered[clue].items()):
            code = min(range(len(levels)),
                       key=lambda i: abs(levels[i] - value))
            entry_columns.append(columns[word])
            entry_codes.append(code)
        offsets.append(len(entry_columns))

    header = {"format": "obirdy-simtable-%d" % version, "version": version,
              "floor": 0.12, "top_k": 64,
              "code_bits": 6, "n_board": len(board), "n_clue": len(clues),
              "n_entries": len(entry_columns), "levels": levels}
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
    with _gzip.open(path, "wb") as handle:
        handle.write(bytes(blob))
    return path


class TestEmbeddingSensor(unittest.TestCase):
    """The half of clue safety the LLM panel is structurally blind to.

    Five of the eight ``sweep_obirdy_cm_abra_g_default_solo`` games died on an
    assassin our panel never ranked; four are plainly visible as a GloVe
    cosine.  This is that signal, offline and free.
    """

    ROWS = {
        # Clean: strong on ours, nothing on the assassin.
        "OCEAN": {"SHIP": 0.45, "WHALE": 0.40, "BEACH": 0.44},
        # SURGE/LEAD shape: the assassin outranks the word we are buying.
        "TOXIN": {"POISON": 0.30, "WHALE": 0.20},
        # WOODWIND/DECK shape: the assassin is present but far behind.
        "SPARK": {"LASER": 0.50, "POISON": 0.20},
        # The assassin clears the floor and none of ours does.
        "MURK": {"POISON": 0.25},
    }

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="obirdy-simtable-")
        self.path = write_simtable(
            os.path.join(self.tmp, "table.bin.gz"),
            ["WHALE", "SHIP", "BEACH", "LASER", "ROBOT", "CROWN", "POISON"],
            self.ROWS)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})

    def _agent(self, **kwargs):
        kwargs.setdefault("simtable_path", self.path)
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    # -- the reader --------------------------------------------------------

    def test_table_round_trips(self):
        table = cm_mod._SimTable.load(self.path)
        self.assertIsNotNone(table)
        pulls = table.pulls("OCEAN", ["SHIP", "WHALE", "POISON"])
        # Values come back through the codebook, so within one 6-bit step.
        self.assertAlmostEqual(pulls["SHIP"], 0.45, delta=0.02)
        self.assertAlmostEqual(pulls["WHALE"], 0.40, delta=0.02)
        self.assertNotIn("POISON", pulls)      # below the floor, not missing

    def test_unknown_clue_is_none_not_empty(self):
        """``None`` means "no vocabulary"; ``{}`` means "nothing above floor"."""
        table = cm_mod._SimTable.load(self.path)
        self.assertIsNone(table.pulls("LASTDRINK", ["POISON"]))
        self.assertEqual(table.pulls("OCEAN", ["ROBOT"]), {})
        self.assertTrue(table.has_clue("ocean"))
        self.assertFalse(table.has_clue("LASTDRINK"))

    def test_missing_file_is_a_no_op(self):
        agent = self._agent(simtable_path=os.path.join(self.tmp, "nope.gz"))
        self.assertIsNone(agent._simtable())
        self.assertEqual(
            agent._embedding_penalty("TOXIN", [["WHALE"]], ["WHALE"],
                                     ["POISON"], 1), 0.0)

    def test_corrupt_file_is_a_no_op(self):
        bad = os.path.join(self.tmp, "bad.bin.gz")
        with open(bad, "wb") as handle:
            handle.write(b"not a gzip stream at all")
        self.assertIsNone(cm_mod._SimTable.load(bad))

    def test_truncated_file_is_a_no_op(self):
        with open(self.path, "rb") as handle:
            blob = handle.read()
        short = os.path.join(self.tmp, "short.bin.gz")
        with open(short, "wb") as handle:
            handle.write(blob[:len(blob) // 2])
        self.assertIsNone(cm_mod._SimTable.load(short))

    def test_sensor_off_never_loads_the_table(self):
        agent = self._agent(embed_sensor=False)
        self.assertIsNone(agent._simtable())
        self.assertEqual(
            agent._embedding_penalty("TOXIN", [["WHALE"]], ["WHALE"],
                                     ["POISON"], 1), 0.0)

    # -- the rule ----------------------------------------------------------

    def test_hit_assassin_outranks_the_weakest_target(self):
        agent = self._agent()
        penalty = agent._embedding_penalty("TOXIN", [["WHALE"]], ["WHALE"],
                                           ["POISON"], 1)
        self.assertEqual(penalty, agent.embed_penalty)
        self.assertEqual(agent.embed_flagged, 1)
        self.assertAlmostEqual(agent.embed_log[-1]["assassin"], 0.30, places=2)

    def test_miss_target_clearly_ahead_of_the_assassin(self):
        agent = self._agent()
        penalty = agent._embedding_penalty("SPARK", [["LASER"]], ["LASER"],
                                           ["POISON"], 1)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.embed_flagged, 0)

    def test_miss_assassin_below_the_stored_floor(self):
        agent = self._agent()
        self.assertEqual(
            agent._embedding_penalty("OCEAN", [["SHIP", "WHALE"]],
                                     ["SHIP", "WHALE"], ["POISON"], 2), 0.0)

    def test_hit_when_nothing_of_ours_clears_the_floor(self):
        agent = self._agent()
        self.assertEqual(
            agent._embedding_penalty("MURK", [["ROBOT"]], ["ROBOT"],
                                     ["POISON"], 1), agent.embed_penalty)

    def test_a_bigger_number_reaches_a_weaker_target_and_flips_the_verdict(self):
        """The rule is about the *marginal* word the number is buying."""
        agent = self._agent()
        rankings = [["LASER", "ROBOT"]]
        one = agent._embedding_penalty("SPARK", rankings, ["LASER", "ROBOT"],
                                       ["POISON"], 1)
        two = agent._embedding_penalty("SPARK", rankings, ["LASER", "ROBOT"],
                                       ["POISON"], 2)
        self.assertEqual(one, 0.0)        # LASER 0.50 vs POISON 0.20
        self.assertEqual(two, agent.embed_penalty)   # ROBOT is below the floor

    def test_margin_is_tunable(self):
        agent = self._agent(embed_margin=0.4)
        self.assertEqual(
            agent._embedding_penalty("SPARK", [["LASER"]], ["LASER"],
                                     ["POISON"], 1), agent.embed_penalty)

    # -- out of vocabulary -------------------------------------------------

    def test_out_of_vocabulary_clue_is_demoted(self):
        agent = self._agent()
        penalty = agent._embedding_penalty("LASTDRINK", [["WHALE"]],
                                           ["WHALE"], ["POISON"], 1)
        self.assertEqual(penalty, agent.embed_oov_penalty)
        self.assertEqual(agent.embed_oov, 1)
        self.assertTrue(agent.embed_log[-1]["oov"])

    def test_the_oov_demotion_is_only_a_tie_break(self):
        """A word the embedding never learned is suspicious, not condemned."""
        agent = self._agent()
        self.assertLess(agent.embed_oov_penalty, 0.25)
        self.assertLess(agent.embed_oov_penalty, agent.embed_penalty / 10.0)

    def test_an_in_vocabulary_candidate_wins_a_tie(self):
        agent = self._agent(probe=False)
        systems = []

        def responder(system, user):
            systems.append(system)
            if is_brainstorm(system):
                return ('[{"clue":"OCEAN","targets":["WHALE","SHIP"]},'
                        ' {"clue":"LASTDRINK","targets":["WHALE","SHIP"]}]')
            return ('{"OCEAN": ["WHALE", "SHIP"],'
                    ' "LASTDRINK": ["WHALE", "SHIP"]}')

        with _PatchedLLM(cm_mod, responder):
            clue, _ = agent.get_clue()
        self.assertEqual(clue, "OCEAN")

    # -- end to end --------------------------------------------------------

    def test_sensor_moves_the_pick_away_from_a_flagged_clue(self):
        agent = self._agent(probe=False)

        def responder(system, user):
            if is_brainstorm(system):
                return ('[{"clue":"TOXIN","targets":["WHALE","SHIP","BEACH"]},'
                        ' {"clue":"OCEAN","targets":["WHALE"]}]')
            # TOXIN outscores OCEAN on expected value alone.
            return ('{"TOXIN": ["WHALE", "SHIP", "BEACH"],'
                    ' "OCEAN": ["WHALE"]}')

        with _PatchedLLM(cm_mod, responder):
            clue, _ = agent.get_clue()
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(agent.embed_flagged, 1)

        without = self._agent(probe=False, embed_sensor=False)
        with _PatchedLLM(cm_mod, responder):
            clue, _ = without.get_clue()
        self.assertEqual(clue, "TOXIN")

    def test_sensor_counters_reach_usage_summary(self):
        agent = self._agent()
        agent._embedding_penalty("TOXIN", [["WHALE"]], ["WHALE"],
                                 ["POISON"], 1)
        summary = agent.usage_summary()
        self.assertTrue(summary["embed_table"])
        self.assertEqual(summary["embed_flagged"], 1)
        self.assertEqual(len(summary["embed_log"]), 1)


class TestSimTableResolution(unittest.TestCase):
    """Where the codemaster looks for its data file, and in what order.

    The organisers unpack our zip into their own ``players/`` directory, so the
    only thing that can be relied on is *relative to this module*.  Three
    layers, best first: an explicit path (kwarg or env), the per-team subfolder
    the organisers asked for, and the flat module directory we shipped before.
    """

    ROWS = {"OCEAN": {"SHIP": 0.45}, "TOXIN": {"POISON": 0.50}}

    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="obirdy-resolve-")
        self.subdir = os.path.join(self.tmp, cm_mod.SIMTABLE_SUBDIR)
        os.makedirs(self.subdir)
        self._module_dir = cm_mod._module_dir
        cm_mod._module_dir = lambda: self.tmp
        self._env = os.environ.pop(cm_mod.SIMTABLE_ENV, None)
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})

    def tearDown(self):
        import shutil

        cm_mod._module_dir = self._module_dir
        if self._env is None:
            os.environ.pop(cm_mod.SIMTABLE_ENV, None)
        else:
            os.environ[cm_mod.SIMTABLE_ENV] = self._env
        shutil.rmtree(self.tmp, ignore_errors=True)
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})

    def _write(self, path, clue):
        return write_simtable(path, ["SHIP", "POISON"],
                              {clue: self.ROWS[clue]})

    def _subfolder_table(self):
        return self._write(os.path.join(self.subdir, cm_mod.SIMTABLE_FILENAME),
                           "OCEAN")

    def _legacy_table(self):
        return self._write(os.path.join(self.tmp, cm_mod.SIMTABLE_FILENAME),
                           "TOXIN")

    # -- the order ---------------------------------------------------------

    def test_the_search_order_is_subfolder_then_legacy(self):
        paths = cm_mod._SimTable.search_paths()
        self.assertEqual(paths[0], os.path.join(self.subdir,
                                                cm_mod.SIMTABLE_FILENAME_V2))
        self.assertEqual(paths[1], os.path.join(self.subdir,
                                                cm_mod.SIMTABLE_FILENAME))
        self.assertEqual(paths[2], os.path.join(self.tmp,
                                                cm_mod.SIMTABLE_FILENAME))

    def test_the_env_var_comes_first(self):
        os.environ[cm_mod.SIMTABLE_ENV] = "/nowhere/table.bin.gz"
        self.assertEqual(cm_mod._SimTable.search_paths()[0],
                         "/nowhere/table.bin.gz")

    def test_an_explicit_kwarg_beats_everything(self):
        legacy = self._legacy_table()
        self._subfolder_table()
        os.environ[cm_mod.SIMTABLE_ENV] = "/nowhere/table.bin.gz"
        self.assertEqual(cm_mod._SimTable.search_paths(legacy), [legacy])
        table = cm_mod._SimTable.load(legacy)
        self.assertTrue(table.has_clue("TOXIN"))
        self.assertEqual(table.path, legacy)

    # -- what actually loads -----------------------------------------------

    def test_the_subfolder_wins_over_the_legacy_location(self):
        self._legacy_table()
        expected = self._subfolder_table()
        table = cm_mod._SimTable.load()
        self.assertEqual(table.path, expected)
        self.assertTrue(table.has_clue("OCEAN"))
        self.assertFalse(table.has_clue("TOXIN"))

    def test_the_legacy_location_still_works_on_its_own(self):
        expected = self._legacy_table()
        table = cm_mod._SimTable.load()
        self.assertEqual(table.path, expected)
        self.assertTrue(table.has_clue("TOXIN"))

    def test_the_env_var_wins_over_both(self):
        self._legacy_table()
        self._subfolder_table()
        elsewhere = self._write(os.path.join(self.tmp, "elsewhere.bin.gz"),
                                "TOXIN")
        os.environ[cm_mod.SIMTABLE_ENV] = elsewhere
        table = cm_mod._SimTable.load()
        self.assertEqual(table.path, elsewhere)

    def test_no_table_anywhere_is_a_no_op_not_a_crash(self):
        self.assertIsNone(cm_mod._SimTable.load())
        agent = cm_mod.AICodemaster("Red")
        agent.set_game_state(list(BOARD), list(KEY))
        self.assertEqual(
            agent._embedding_penalty("OCEAN", [["WHALE"]], ["WHALE"],
                                     ["POISON"], 1), 0.0)


class TestShippedSimTable(unittest.TestCase):
    """The real data file, if it is checked out."""

    def setUp(self):
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})
        self.table = cm_mod._SimTable.load()
        if self.table is None:
            self.skipTest("no bundled simtable -- python -m harness.simtable")

    def tearDown(self):
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})

    # -- raw-file helpers --------------------------------------------------

    def _path(self):
        return os.path.join(FRAMEWORK_DIR, "players", cm_mod.SIMTABLE_SUBDIR,
                            cm_mod.SIMTABLE_FILENAME_V2)

    def _header(self):
        import gzip as _gzip

        with _gzip.open(self._path(), "rb") as handle:
            blob = handle.read(4096)
        start = len(cm_mod._SIMTABLE_MAGIC)
        return json.loads(blob[start:blob.index(b"\n", start)].decode("utf-8"))

    def _row(self, clue):
        """``[(board column, code)]`` stored for ``clue``, decoded by hand.

        By hand and per version, rather than through ``pulls``: these tests are
        about the bytes on disk, so they must not be able to pass by agreeing
        with the same decoder they are checking.
        """
        table = self.table
        row = table.clue_index[clue.lower()]
        out = []
        for i in range(table._offset(row), table._offset(row + 1)):
            word = int.from_bytes(table._entries[2 * i:2 * i + 2], "little")
            if table.version == 1:
                out.append((word >> table.code_bits, word & table.code_mask))
            else:
                out.append((word, table._codes[i]))
        return out

    def _columns(self, clue):
        return [column for column, _code in self._row(clue)]

    def test_it_ships_in_the_teams_own_subfolder(self):
        """The organisers want auxiliary files under ``players/<team>/``."""
        here = os.path.join(FRAMEWORK_DIR, "players", cm_mod.SIMTABLE_SUBDIR,
                            cm_mod.SIMTABLE_FILENAME_V2)
        self.assertTrue(os.path.exists(here), here)
        self.assertFalse(os.path.exists(
            os.path.join(FRAMEWORK_DIR, "players", cm_mod.SIMTABLE_FILENAME)))
        self.assertEqual(self.table.path, here)

    def test_it_covers_the_framework_word_pool(self):
        with open(os.path.join(FRAMEWORK_DIR, "players",
                               "cm_wordlist.txt")) as handle:
            pool = [w.strip().upper() for w in handle if w.strip()]
        covered = sum(1 for w in pool if w in self.table.board_index)
        self.assertEqual(covered, len(pool))

    def test_it_covers_every_slang_word_an_embedding_can_reach(self):
        """The tournament pool is slang-heavy, so this is the number that counts.

        Two thirds of the slang pool used to be missing.  Most of that was a
        frequency miss and is now fixed; the rest is ``UNREACHABLE_SLANG``,
        which GloVe 6B does not contain in any form.  Asserting against
        ``232 - len(UNREACHABLE_SLANG)`` rather than against 232 keeps this
        test honest in both directions: it fails if a reachable word is
        dropped, and it fails if the unreachable list is quietly padded.
        """
        from harness.secret_pool import SLANG_POOL
        from harness.simtable import UNREACHABLE_SLANG

        pool = [w.strip().upper() for w in SLANG_POOL]
        missing = sorted(w for w in pool if w not in self.table.board_index)
        self.assertEqual(missing, sorted(UNREACHABLE_SLANG))

    def test_the_unreachable_words_are_only_ever_pool_words(self):
        from harness.secret_pool import SLANG_POOL
        from harness.simtable import UNREACHABLE_SLANG

        pool = set(w.strip().upper() for w in SLANG_POOL)
        self.assertEqual(sorted(set(UNREACHABLE_SLANG) - pool), [])
        self.assertEqual(len(set(UNREACHABLE_SLANG)), len(UNREACHABLE_SLANG))

    def test_it_covers_every_themed_pool_completely(self):
        """The half-blind case this vocabulary was widened for.

        The organisers ran us on a 25-word themed board of which the shipped
        1008-word table could price twelve.  Every themed pool, and their board
        verbatim, is now covered outright -- there is no ``UNREACHABLE_THEMED``
        to discount against, because there is no themed word GloVe 6B lacks.
        """
        from harness.secret_pool import (ORGANISER_THEMED_BOARD, THEMED_POOLS)
        from harness.simtable import UNREACHABLE_THEMED

        self.assertEqual(list(UNREACHABLE_THEMED), [])
        pools = dict(THEMED_POOLS)
        pools["organiser"] = ORGANISER_THEMED_BOARD
        for name in sorted(pools):
            missing = sorted(w.strip().upper() for w in pools[name]
                             if w.strip().upper() not in self.table.board_index)
            self.assertEqual(missing, [], name)

    def test_the_board_vocabulary_fits_the_index_width(self):
        """A board word past the index width would alias onto another column."""
        from harness import simtable

        ceiling = simtable.MAX_BOARD_WORDS_BY_VERSION[self.table.version]
        self.assertLess(len(self.table.board_words), ceiling)
        self.assertEqual(len(set(self.table.board_words)),
                         len(self.table.board_words))
        # And the shipped file really does need the wider one: this is the
        # whole reason the format was bumped.
        self.assertGreater(len(self.table.board_words),
                           simtable.LEGACY_MAX_BOARD_WORDS)
        self.assertEqual(self.table.version, 2)

    def test_the_later_banks_never_displace_an_earlier_one(self):
        """Widening the board vocabulary must not cost an existing lookup.

        The first build that merged the generated candidates into one global
        top-64 pushed LADDER->BOX (0.145, the shallowest recorded death) off
        the end of its clue's list and the sensor went silent on it.  The banks
        are ranked separately now -- real pools, themed pools, generated
        guesses -- and this pins that the real-pool bank still reaches as deep
        as the shipped ``top_k`` with two younger banks beside it.
        """
        header = self._header()
        n_pool = int(header["n_pool"])
        n_themed = int(header["n_themed"])
        self.assertGreater(n_pool, 0)
        self.assertGreater(n_themed, 0)
        deepest = 0
        for clue in ("LADDER", "PACKING", "SURGE", "PAGEANT"):
            columns = self._columns(clue)
            deepest = max(deepest, sum(1 for c in columns if c < n_pool))
            # No bank may exceed its own quota either.
            self.assertLessEqual(
                sum(1 for c in columns if n_pool <= c < n_pool + n_themed),
                int(header["themed_top_k"]), clue)
            self.assertLessEqual(
                sum(1 for c in columns if c >= n_pool + n_themed),
                int(header["candidate_top_k"]), clue)
        self.assertGreater(deepest, 32, "pool half of the table got shallower")

    def test_it_knows_the_offline_fallback_vocabulary(self):
        known = sum(1 for w in cm_mod._FALLBACK_VOCAB
                    if self.table.has_clue(w))
        self.assertGreater(known, 0.95 * len(cm_mod._FALLBACK_VOCAB))

    def test_the_recorded_deaths_are_flagged(self):
        """The four codemaster-side deaths this sensor was built for.

        Boards reconstructed offline from ``results_local``; see
        ``docs/versions.md``.  Each entry is (clue, assassin, targets the
        clue's number was buying).
        """
        agent = cm_mod.AICodemaster("Red")
        cases = [
            ("PAGEANT", "STATE", ["PRINCESS", "SPOT"]),   # reeval_c seed 10
            ("SURGE", "LEAD", ["CHARGE"]),                # abra seed 3
            ("PACKING", "OLIVE", ["TRUNK"]),              # abra seed 5
            ("LADDER", "BOX", ["SCALE"]),                 # abra seed 6
        ]
        for clue, assassin, targets in cases:
            agent.embed_flagged = 0
            penalty = agent._embedding_penalty(
                clue, [targets], list(targets), [assassin], len(targets))
            self.assertEqual(penalty, agent.embed_penalty,
                             "%s should have been flagged" % clue)

    def test_the_death_our_codemaster_did_not_own_is_not_flagged(self):
        """abra seed 7: WOODWIND->FLUTE was right; the partner's bonus killed it."""
        agent = cm_mod.AICodemaster("Red")
        self.assertEqual(
            agent._embedding_penalty("WOODWIND", [["FLUTE"]], ["FLUTE"],
                                     ["DECK"], 1), 0.0)

    def test_the_mashup_clues_are_out_of_vocabulary(self):
        """Clues an embedding partner cannot read must stay unreadable to us.

        ``EGGSHELL`` is the interesting one.  It is a real word, at GloVe rank
        73529 -- so the moment the builder started reading a 150k-deep cache it
        appeared in the clue vocabulary and stopped taking the demotion, even
        though the Abra partner that lost a game to it caches 60k words and
        still cannot read it.  ``CLUE_VOCAB_CEILING`` is what holds this line.
        """
        for clue in ("LASTDRINK", "EGGSHELL", "HARBORMEASURE"):
            self.assertFalse(self.table.has_clue(clue), clue)

    def test_the_file_round_trips_through_an_independent_writer(self):
        """Re-encode a slice of the shipped file and read identical values back.

        ``write_simtable`` below is a second implementation of the format, so
        agreeing with it means the shipped file *is* the documented layout and
        not merely something the builder and reader happen to agree on.  A
        slice, because the whole file is 2.2 M entries.
        """
        import tempfile

        clues = self.table.clue_words[::151][:200]
        board = list(self.table.board_words)
        rows = {}
        for clue in clues:
            rows[clue] = self.table.pulls(clue, board) or {}

        tmp = tempfile.mkdtemp(prefix="obirdy-roundtrip-")
        try:
            path = write_simtable(os.path.join(tmp, "rt.bin.gz"), board, rows,
                                  levels=self.table.levels,
                                  version=self.table.version)
            again = cm_mod._SimTable.load(path)
            self.assertIsNotNone(again)
            self.assertEqual(again.version, self.table.version)
            self.assertEqual(again.board_words, board)
            for clue in clues:
                self.assertEqual(again.pulls(clue, board), rows[clue], clue)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_stored_entries_are_structurally_sound(self):
        """Offsets monotonic, columns in range, codes on the shipped codebook."""
        header = self._header()
        n_board = int(header["n_board"])
        n_entries = int(header["n_entries"])
        self.assertEqual(len(self.table.levels), 1 << self.table.code_bits)
        self.assertEqual(self.table.levels, sorted(self.table.levels))
        self.assertEqual(len(self.table._entries), 2 * n_entries)
        if self.table.version >= 2:
            self.assertEqual(len(self.table._codes), n_entries)
        self.assertEqual(self.table._offset(len(self.table.clue_words)),
                         n_entries)

        previous = 0
        for row in range(len(self.table.clue_words)):
            here = self.table._offset(row)
            self.assertEqual(here, previous)
            previous = self.table._offset(row + 1)
            self.assertGreaterEqual(previous, here)
        for clue in ("ocean", "ladder", "surge"):
            pairs = self._row(clue)
            columns = [column for column, _code in pairs]
            self.assertEqual(columns, sorted(columns), clue)
            self.assertTrue(all(0 <= c < n_board for c in columns), clue)
            self.assertTrue(all(0 <= code < len(self.table.levels)
                                for _c, code in pairs), clue)

    def test_it_stays_small_enough_to_ship(self):
        """It travels with the agent, so the budget is a real constraint."""
        self.assertLess(os.path.getsize(self._path()), 5 * 1024 * 1024)


class TestSimTableBuilderRules(unittest.TestCase):
    """The pure filters behind the generated board vocabulary.

    No vectors and no cache, so these run on a bare checkout.
    """

    VOCAB = {"album", "albums", "upload", "uploaded", "wireless", "wirelessly",
             "mustache", "mustaches", "glass", "press", "latte", "ninja",
             "carry", "carries", "bike", "biker"}

    def test_inflected_forms_are_dropped(self):
        from harness.simtable import _is_inflected

        for word in ("ALBUMS", "UPLOADED", "WIRELESSLY", "MUSTACHES",
                     "CARRIES", "BIKER"):
            self.assertTrue(_is_inflected(word, self.VOCAB), word)

    def test_bare_nouns_survive(self):
        from harness.simtable import _is_inflected

        for word in ("ALBUM", "LATTE", "NINJA", "GLASS", "PRESS", "BIKE"):
            self.assertFalse(_is_inflected(word, self.VOCAB), word)

    def test_derivations_of_a_taken_word_are_dropped(self):
        from harness.simtable import _is_derivation

        taken = ["BATCAVE", "SPEEDRUN"]
        self.assertTrue(_is_derivation("CAVE", taken))
        self.assertTrue(_is_derivation("BATCAVES", taken))
        self.assertFalse(_is_derivation("LATTE", taken))

    def test_the_clue_ceiling_is_at_least_the_prefix(self):
        from harness import simtable

        self.assertGreaterEqual(simtable.CLUE_VOCAB_CEILING,
                                simtable.DEFAULT_CLUE_VOCAB)
        self.assertLess(simtable.DEFAULT_CANDIDATE_TOP_K,
                        simtable.DEFAULT_TOP_K)
        self.assertLess(simtable.DEFAULT_THEMED_TOP_K, simtable.DEFAULT_TOP_K)
        self.assertLess(simtable.DEFAULT_CANDIDATE_TOP_K,
                        simtable.DEFAULT_THEMED_TOP_K)

    def test_the_themed_bank_is_every_themed_word_the_pools_can_deal(self):
        """The board vocabulary is derived from the pools, never transcribed."""
        from harness.secret_pool import (ORGANISER_THEMED_BOARD, THEMED_POOLS)
        from harness.simtable import bundled_pools, themed_pool_words

        themed = themed_pool_words()
        self.assertEqual(len(themed), len(set(themed)))
        # Everything a themed pool can deal is either in the themed bank or
        # already in the two real pools -- nothing falls between them.
        covered = set(themed) | set(bundled_pools())
        for name in sorted(THEMED_POOLS):
            for word in THEMED_POOLS[name]:
                self.assertIn(word.strip().upper(), covered, name)
        for word in ORGANISER_THEMED_BOARD:
            self.assertIn(word.strip().upper(), covered)

    def test_the_candidate_seeds_exclude_the_default_pool(self):
        from harness.secret_pool import load_default_pool
        from harness.simtable import candidate_seeds

        seeds = set(candidate_seeds())
        default = set(w.strip().upper() for w in load_default_pool())
        # A handful of themed words are also default-pool words (JAM, STAR);
        # what must not happen is the default pool being seeded wholesale.
        self.assertLess(len(seeds & default), 40)
        self.assertGreater(len(seeds), 400)


class TestSimTableFormatVersions(unittest.TestCase):
    """Both layouts, both directions.

    Version 2 exists because the themed pools took the board vocabulary past
    version 1's 1024-word ceiling.  A submission's agent file and its data file
    can be unpacked from different zips, so the reader has to keep reading
    version 1 as well -- these tests are the only thing standing behind that.
    """

    ROWS = {
        "OCEAN": {"SHIP": 0.45, "WHALE": 0.40, "BEACH": 0.44},
        "TOXIN": {"POISON": 0.30, "WHALE": 0.20},
    }
    BOARD = ["WHALE", "SHIP", "BEACH", "POISON"]

    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="obirdy-versions-")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, version, board=None, rows=None):
        return write_simtable(
            os.path.join(self.tmp, "v%d.bin.gz" % version),
            board if board is not None else self.BOARD,
            rows if rows is not None else self.ROWS, version=version)

    def test_both_versions_read_back_the_same_values(self):
        tables = {}
        for version in (1, 2):
            table = cm_mod._SimTable.load(self._write(version))
            self.assertIsNotNone(table, version)
            self.assertEqual(table.version, version)
            tables[version] = table
        for clue, expected in self.ROWS.items():
            first = tables[1].pulls(clue, self.BOARD)
            second = tables[2].pulls(clue, self.BOARD)
            self.assertEqual(first, second, clue)
            self.assertEqual(sorted(first), sorted(expected), clue)

    def test_a_version_one_file_still_loads(self):
        """The compatibility claim, stated on its own so it cannot be lost."""
        table = cm_mod._SimTable.load(self._write(1))
        self.assertEqual(table.version, 1)
        self.assertIsNone(table._codes)
        self.assertAlmostEqual(table.pulls("OCEAN", ["SHIP"])["SHIP"], 0.45,
                               delta=0.02)

    def test_version_two_carries_a_board_index_version_one_cannot(self):
        from harness import simtable

        # Alphabetic names: the reader normalises board words to letters only,
        # so WORD0001 and WORD0002 would be the same card.
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        board = ["WORD" + letters[i // 676] + letters[i // 26 % 26]
                 + letters[i % 26] for i in range(1500)]
        far = board[1499]
        rows = {"OCEAN": {far: 0.45, board[0]: 0.30}}
        table = cm_mod._SimTable.load(self._write(2, board, rows))
        self.assertEqual(table.board_index[far], 1499)
        got = table.pulls("OCEAN", [far, board[0]])
        self.assertAlmostEqual(got[far], 0.45, delta=0.02)
        self.assertAlmostEqual(got[board[0]], 0.30, delta=0.02)
        # And the builder's own encoder refuses to write it as version 1
        # rather than aliasing column 1499 onto column 475.
        with self.assertRaises(ValueError):
            simtable.encode({}, board, ["ocean"], [0, 1], [1499], [3],
                            version=1)

    def test_the_builders_encoder_and_this_writer_agree(self):
        """Two independent encoders, both versions, byte-for-byte on the body."""
        from harness import simtable

        levels = list(SENSOR_LEVELS)
        columns, codes, offsets = [], [], [0]
        for clue in sorted(c.lower() for c in self.ROWS):
            row = dict((k, v) for k, v in
                       self.ROWS[clue.upper()].items())
            for word, value in sorted(row.items()):
                columns.append(self.BOARD.index(word))
                codes.append(min(range(len(levels)),
                                 key=lambda i: abs(levels[i] - value)))
            offsets.append(len(columns))

        for version in (1, 2):
            mine = cm_mod._SimTable.load(self._write(version))
            blob = simtable.encode(
                {"floor": 0.12, "code_bits": 6, "levels": levels},
                self.BOARD, sorted(c.lower() for c in self.ROWS), offsets,
                columns, codes, version=version)
            path = os.path.join(self.tmp, "builder%d.bin.gz" % version)
            with gzip.open(path, "wb") as handle:
                handle.write(blob)
            theirs = cm_mod._SimTable.load(path)
            self.assertEqual(theirs.version, version)
            self.assertEqual(theirs._offsets, mine._offsets, version)
            self.assertEqual(theirs._entries, mine._entries, version)
            self.assertEqual(theirs._codes, mine._codes, version)

    def test_a_truncated_version_two_body_is_refused_not_mis_sliced(self):
        """A short file must load as ``None``, never as plausible numbers."""
        path = self._write(2)
        with gzip.open(path, "rb") as handle:
            blob = handle.read()
        for cut in (1, 5, 9):
            self.assertIsNone(cm_mod._SimTable._parse(blob[:-cut]))
        self.assertIsNone(cm_mod._SimTable._parse(b"OBSIM9\n{}\n"))


class TestDefaultConfigRegression(unittest.TestCase):
    """What the shipped default does, and what it deliberately now does.

    The old version of this class asserted the probe was *off* by default.
    That assertion was the thing under review, not a fixed point: with the
    probe off, the shipped agent clued PAGEANT 2 into the assassin on eval C
    seed 10 and lost five of eight games against an embedding guesser.  The
    tests below now pin the opposite -- the probe runs, and it costs exactly
    one call when the leader survives it.  Everything the probe does *not*
    touch (the risk weights, the clue number) must still be identical.
    """

    SCENARIO_BRAINSTORM = TestDangerProbe.BRAINSTORM
    SCENARIO_PANEL = TestDangerProbe.PANEL

    def _play(self, **kwargs):
        kwargs.setdefault("embed_sensor", False)
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        systems = []

        def responder(system, user):
            systems.append(system)
            if is_brainstorm(system):
                return self.SCENARIO_BRAINSTORM
            if is_probe(system):
                # Probe-neutral: nothing pulls on anything.
                return '{"POISON": 0, "APPLE": 0, "WHALE": 0, "LASER": 0}'
            return self.SCENARIO_PANEL

        with _PatchedLLM(cm_mod, responder):
            clue, number = agent.get_clue()
        return clue, number, systems

    def test_default_config_calls_the_probe_exactly_once(self):
        """Intentional change: the probe is a shipped default from Pidgeot."""
        clue, number, systems = self._play()
        self.assertEqual(sum(1 for s in systems if is_probe(s)), 1)
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(number, 3)

    def test_probe_neutral_responses_do_not_change_the_pick(self):
        base_clue, base_number, systems = self._play(probe=False)
        self.assertFalse(any(is_probe(system) for system in systems))
        probe_clue, probe_number, _ = self._play()
        self.assertEqual(probe_clue, base_clue)
        self.assertEqual(probe_number, base_number)

    def test_ambitious_preset_only_differs_by_its_own_knobs(self):
        base_clue, _, _ = self._play()
        preset_clue, preset_number, _ = self._play(preset="ambitious")
        self.assertEqual(preset_clue, base_clue)
        self.assertEqual(preset_number, 3)

    def test_disabled_probe_leaves_numbers_exactly_as_before(self):
        """Every knob the probe added is inert while ``probe`` is off."""
        own = ["WHALE", "SHIP", "BEACH"]
        rankings = [["WHALE", "SHIP", "BEACH"]]
        plain = cm_mod.AICodemaster("Red")
        plain.set_game_state(list(BOARD), list(KEY))
        loud = cm_mod.AICodemaster("Red", probe_veto_score=0.0,
                                   probe_missing_penalty=99.0,
                                   probe_unprobed_cap=1)
        loud.set_game_state(list(BOARD), list(KEY))
        self.assertEqual(plain._score_candidate(rankings, own, [], [],
                                                ["POISON"], 3),
                         loud._score_candidate(rankings, own, [], [],
                                               ["POISON"], 3))


class TestAmbitiousComposition(unittest.TestCase):
    """``preset="ambitious"`` must raise the *numbers* and nothing else.

    The clue-number push failed twice, and both failures predate the two
    safety nets: the first ran with no probe at all, the second with the old
    top-3 probe whose relative veto could not fire.  Re-testing it is only
    meaningful if selecting the preset demonstrably keeps the shipped probe
    shape (winning clue, one fall-through, two calls a turn at most) and the
    embedding sensor.  Every assertion below is a way that could silently stop
    being true.
    """

    ROWS = TestEmbeddingSensor.ROWS

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="obirdy-ambitious-")
        self.path = write_simtable(
            os.path.join(self.tmp, "table.bin.gz"),
            ["WHALE", "SHIP", "BEACH", "LASER", "ROBOT", "CROWN", "POISON"],
            self.ROWS)
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})

    def _agent(self, **kwargs):
        kwargs.setdefault("preset", "ambitious")
        kwargs.setdefault("simtable_path", self.path)
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    # -- the knobs ---------------------------------------------------------

    def test_it_raises_the_numbers_and_only_the_numbers(self):
        ambitious = cm_mod.AICodemaster("Red", preset="ambitious")
        shipped = cm_mod.AICodemaster("Red")
        raised = {"bonus_guess_weight", "civilian_penalty", "claimed_slack"}
        self.assertEqual(set(cm_mod.PRESETS["ambitious"]), raised)
        for name in raised:
            self.assertNotEqual(getattr(ambitious, name),
                                getattr(shipped, name))
        for name in ("probe_enabled", "probe_top_k", "probe_veto_score",
                     "probe_relative_veto", "probe_relative_floor",
                     "probe_veto_requires_embed", "probe_veto_penalty",
                     "probe_missing_penalty", "probe_unprobed_cap",
                     "probe_assassin_weight", "probe_opponent_weight",
                     "embed_sensor", "embed_margin", "embed_penalty",
                     "embed_oov_penalty", "majority_fraction",
                     "assassin_presence", "opponent_presence", "allow_sweep"):
            self.assertEqual(getattr(ambitious, name), getattr(shipped, name),
                             "ambitious moved %s" % name)

    def test_both_nets_are_armed(self):
        agent = cm_mod.AICodemaster("Red", preset="ambitious")
        self.assertTrue(agent.probe_enabled)
        self.assertTrue(agent.embed_sensor)
        self.assertEqual(agent.probe_top_k, 2)

    # -- the probe shape ---------------------------------------------------

    BRAINSTORM = ('[{"clue":"OCEAN","targets":["WHALE","SHIP","BEACH"]},'
                  ' {"clue":"GLOW","targets":["LASER","ROBOT"]}]')
    PANEL = ('{"OCEAN": ["WHALE", "SHIP", "BEACH"],'
             ' "GLOW": ["LASER", "ROBOT"]}')

    def _responder(self, probe_replies, record=None):
        def responder(system, user):
            if is_brainstorm(system):
                return self.BRAINSTORM
            if is_probe(system):
                if record is not None:
                    record.append(user)
                for clue, reply in probe_replies.items():
                    if "Clue: %s" % clue in user:
                        return reply
                return None
            return self.PANEL

        return responder

    def test_a_clean_leader_still_costs_exactly_one_probe(self):
        agent = self._agent()
        asked = []
        responder = self._responder({
            "OCEAN": '{"POISON": 0, "WHALE": 8, "APPLE": 1}',
            "GLOW": '{"POISON": 0, "LASER": 9, "APPLE": 1}',
        }, record=asked)
        with _PatchedLLM(cm_mod, responder):
            clue, _ = agent.get_clue()
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(len(asked), 1)
        self.assertEqual(agent.probes_run, 1)

    def test_a_veto_still_falls_through_to_the_next_candidate(self):
        agent = self._agent()
        asked = []
        responder = self._responder({
            "OCEAN": '{"POISON": 9, "WHALE": 8, "APPLE": 1}',
            "GLOW": '{"POISON": 0, "LASER": 9, "APPLE": 1}',
        }, record=asked)
        with _PatchedLLM(cm_mod, responder):
            clue, _ = agent.get_clue()
        self.assertEqual(clue, "GLOW")
        self.assertEqual(len(asked), 2)          # the ceiling, not more
        self.assertEqual(agent.probes_vetoed, 1)
        self.assertEqual([entry["clue"] for entry in agent.probe_log],
                         ["OCEAN", "GLOW"])

    def test_the_relative_veto_still_fires_under_the_raised_numbers(self):
        """The veto that the first two failures never had a working copy of."""
        agent = self._agent()
        ratings = {"POISON": 4.0, "WHALE": 3.0}
        score, number = agent._apply_probe(5.0, 4, 4, ratings, ["APPLE"],
                                           ["POISON"], clue="OCEAN")
        self.assertLess(score, -40.0)
        self.assertEqual(number, 1)

    # -- the sensor --------------------------------------------------------

    def test_the_sensor_still_moves_the_pick(self):
        agent = self._agent(probe=False)

        def responder(system, user):
            if is_brainstorm(system):
                return ('[{"clue":"TOXIN","targets":["WHALE","SHIP","BEACH"]},'
                        ' {"clue":"OCEAN","targets":["WHALE"]}]')
            return ('{"TOXIN": ["WHALE", "SHIP", "BEACH"],'
                    ' "OCEAN": ["WHALE"]}')

        with _PatchedLLM(cm_mod, responder):
            clue, _ = agent.get_clue()
        self.assertEqual(clue, "OCEAN")
        self.assertEqual(agent.embed_flagged, 1)

    def test_the_oov_demotion_survives_the_preset(self):
        agent = self._agent()
        self.assertAlmostEqual(
            agent._embedding_penalty("LASTDRINK", [["WHALE"]], ["WHALE"],
                                     ["POISON"], 1),
            cm_mod.EMBED_OOV_PENALTY)

    # -- what the preset is actually for -----------------------------------

    def test_the_raised_numbers_do_reach_the_clue(self):
        """Otherwise the arm under test is not testing anything."""
        rankings = [["WHALE", "SHIP", "BEACH"]]
        own = ["WHALE", "SHIP", "BEACH"]
        shipped = cm_mod.AICodemaster("Red", embed_sensor=False)
        shipped.set_game_state(list(BOARD), list(KEY))
        ambitious = cm_mod.AICodemaster("Red", preset="ambitious",
                                        embed_sensor=False)
        ambitious.set_game_state(list(BOARD), list(KEY))
        _, timid_number = shipped._score_candidate(
            rankings, own, [], [], ["POISON"], 2)
        _, bold_number = ambitious._score_candidate(
            rankings, own, [], [], ["POISON"], 2)
        self.assertGreater(bold_number, timid_number)


# ---------------------------------------------------------------------------
# Race awareness (two-team track)
# ---------------------------------------------------------------------------

#: Verbatim ``move_history`` from ``results_local/gauntlet_vs_abra.json``, the
#: 0-3 run against the GloVe pair that this whole round exists because of.
#: ``results_local/`` is gitignored, so the two decisive games are copied here
#: rather than read; ``TestRaceAudit`` checks the copies against the files when
#: they happen to be on disk.
#:
#: Seed 100: lost 8-3.  Their three-word clues cleared the board at ~2.7 words
#: a turn while ours delivered 1.0.  No assassin anywhere -- we were simply
#: out-raced, which in a binary-scored track costs exactly what dying costs.
ABRA_SEED_100 = [
    ['Red_Codemaster', 'OIL', 1],
    ['Red_Guesser', 'OLIVE', '*Red*', False],
    ['Blue_Codemaster', 'ARTICLE', 3],
    ['Blue_Guesser', 'PRESS', '*Blue*', True],
    ['Blue_Guesser', 'NOTE', '*Blue*', True],
    ['Blue_Guesser', 'COVER', '*Blue*', False],
    ['Red_Codemaster', 'ARMY', 1],
    ['Red_Guesser', 'FORCE', '*Red*', False],
    ['Blue_Codemaster', 'DARREN', 3],
    ['Blue_Guesser', 'KIWI', '*Blue*', True],
    ['Blue_Guesser', 'STAR', '*Blue*', True],
    ['Blue_Guesser', 'BUCK', '*Blue*', False],
    ['Red_Codemaster', 'SURF', 1],
    ['Red_Guesser', 'WAVE', '*Red*', False],
    ['Blue_Codemaster', 'BITES', 2],
    ['Blue_Guesser', 'TICK', '*Blue*', True],
    ['Blue_Guesser', 'SPIDER', '*Blue*', False],
]

#: Seed 102: lost 8-8 by one word.  Level for most of the game, so the race
#: reading must stay quiet until the endgame, where our last clue needed to
#: buy two words and bought one.
ABRA_SEED_102 = [
    ['Red_Codemaster', 'COMET', 3],
    ['Red_Guesser', 'TAIL', '*Red*', True],
    ['Red_Guesser', 'STAR', '*Red*', True],
    ['Red_Guesser', 'MISSILE', '*Red*', False],
    ['Blue_Codemaster', 'KING', 2],
    ['Blue_Guesser', 'CROWN', '*Blue*', True],
    ['Blue_Guesser', 'LION', '*Blue*', False],
    ['Red_Codemaster', 'CHAIN', 2],
    ['Red_Guesser', 'LINK', '*Red*', False],
    ['Blue_Codemaster', 'TUNES', 2],
    ['Blue_Guesser', 'COMIC', '*Blue*', True],
    ['Blue_Guesser', 'SWING', '*Blue*', False],
    ['Red_Codemaster', 'PERIMETER', 2],
    ['Red_Guesser', 'WALL', '*Red*', True],
    ['Red_Guesser', 'YARD', '*Red*', False],
    ['Blue_Codemaster', 'INLET', 2],
    ['Blue_Guesser', 'ANTARCTICA', '*Blue*', True],
    ['Blue_Guesser', 'FORK', '*Blue*', False],
    ['Red_Codemaster', 'HEAVEN', 1],
    ['Red_Guesser', 'ANGEL', '*Red*', False],
    ['Blue_Codemaster', 'JAPAN', 1],
    ['Blue_Guesser', 'TOKYO', '*Blue*', True],
    ['Blue_Guesser', 'GERMANY', '*Civilian*', False],
    ['Red_Codemaster', 'CIRCULAR', 1],
    ['Red_Guesser', 'CIRCLE', '*Red*', False],
    ['Blue_Codemaster', 'SCRUB', 1],
    ['Blue_Guesser', 'BRUSH', '*Blue*', False],
]

#: A single-team game: red clues, red guesses, forever.  ``game.Game.run``
#: hands the move straight back to red when ``single_team`` is set, so no
#: ``Blue_*`` entry can ever appear.  This is the shape every race feature must
#: be provably inert against.
SOLO_HISTORY = [
    ['Red_Codemaster', 'HARVEST', 2],
    ['Red_Guesser', 'DOCTOR', '*Red*', True],
    ['Red_Guesser', 'SHADOW', '*Red*', False],
    ['Red_Codemaster', 'LANTERN', 1],
    ['Red_Guesser', 'FLUTE', '*Red*', False],
]

#: A duel sitting exactly one turn behind: three words a turn each, but we
#: still have six to their five.  Scale 0.5, which moves the scoring weights
#: and must leave the clue number's ceiling alone.
LEVEL_MINUS_ONE_HISTORY = [
    ['Red_Codemaster', 'ALPHA', 2],
    ['Red_Guesser', 'DOCTOR', '*Red*', True],
    ['Red_Guesser', 'SHADOW', '*Red*', False],
    ['Blue_Codemaster', 'BETA', 2],
    ['Blue_Guesser', 'KING', '*Blue*', True],
    ['Blue_Guesser', 'CROWN', '*Blue*', False],
    ['Red_Codemaster', 'GAMMA', 1],
    ['Red_Guesser', 'FLUTE', '*Red*', False],
    ['Blue_Codemaster', 'DELTA', 1],
    ['Blue_Guesser', 'BANK', '*Blue*', False],
]


def board_with(own_found=0, opp_found=0, team="Red"):
    """``BOARD`` with that many of each colour revealed, from the tail.

    The agent reads the live board, not the history, so an agent-level fixture
    has to make the two agree.  Revealing from the end keeps the leading own
    words (WHALE, SHIP, BEACH) available for the scripted panel replies.
    """
    opponent = "Blue" if team == "Red" else "Red"
    words = list(BOARD)
    for colour, count in ((team, own_found), (opponent, opp_found)):
        indexes = [i for i, kind in enumerate(KEY) if kind == colour]
        for index in reversed(indexes[:len(indexes)]):
            if count <= 0:
                break
            words[index] = "*%s*" % colour
            count -= 1
    return words


def history_prefix(history, turn, team="Red"):
    """The history as it stood just before ``team``'s ``turn``-th clue."""
    prefix = "%s_Codemaster" % team
    seen = 0
    for index, move in enumerate(history):
        if str(move[0]) == prefix:
            seen += 1
            if seen == turn:
                return history[:index]
    raise AssertionError("no turn %d in this history" % turn)


def words_left(history, team="Red"):
    """(ours, theirs) still hidden, counted from the reveals in ``history``."""
    opponent = "Blue" if team == "Red" else "Red"
    own = sum(1 for m in history
              if len(m) >= 3 and str(m[2]) == "*%s*" % team)
    opp = sum(1 for m in history
              if len(m) >= 3 and str(m[2]) == "*%s*" % opponent)
    return (cm_mod.TEAM_TOTALS[team] - own,
            cm_mod.TEAM_TOTALS[opponent] - opp)


def race_at(history, turn, team="Red"):
    """The race state the agent would read before its ``turn``-th clue."""
    prefix = history_prefix(history, turn, team)
    own_left, opp_left = words_left(prefix, team)
    return cm_mod.race_state(prefix, team, own_left, opp_left)


class TestRaceState(unittest.TestCase):
    """The pace reading, against the games it was derived from."""

    def test_a_history_with_no_opponent_move_is_not_a_race(self):
        for turn in (1, 2):
            self.assertIsNone(race_at(SOLO_HISTORY, turn))
        self.assertIsNone(cm_mod.race_state([], "Red", 9, 8))

    def test_our_opening_clue_has_no_race_to_read_either(self):
        """A duel is indistinguishable from a solo game until blue moves."""
        self.assertIsNone(race_at(ABRA_SEED_100, 1))
        self.assertIsNone(race_at(ABRA_SEED_102, 1))

    def test_the_counts_come_off_the_recorded_history(self):
        counts = cm_mod.race_counts(history_prefix(ABRA_SEED_100, 2), "Red")
        self.assertEqual(counts["own_turns"], 1)
        self.assertEqual(counts["opp_turns"], 1)
        self.assertEqual(counts["own_pace_words"], 1)
        self.assertEqual(counts["opp_pace_words"], 3)
        self.assertTrue(counts["opponent_moved"])

    def test_abra_seed_100_escalates_by_turn_two(self):
        """The game the round is named after: 1.0 a turn against 3.0 a turn."""
        second = race_at(ABRA_SEED_100, 2)
        self.assertGreater(second["opp_pace"], second["own_pace"])
        self.assertGreaterEqual(second["deficit"], 2)
        self.assertEqual(second["scale"], 1.0)
        self.assertEqual(race_at(ABRA_SEED_100, 3)["scale"], 1.0)

    def test_abra_seed_102_stays_quiet_until_the_race_is_actually_lost(self):
        """The 8-8 photo finish: level early, one turn short at the end."""
        for turn in (2, 3, 4):
            self.assertEqual(race_at(ABRA_SEED_102, turn)["scale"], 0.0,
                             "turn %d" % turn)
        final = race_at(ABRA_SEED_102, 5)
        self.assertEqual((final["own_left"], final["opp_left"]), (2, 1))
        self.assertEqual(final["deficit"], 1)
        self.assertEqual(final["scale"], 0.5)

    def test_a_word_we_hand_them_moves_the_count_not_their_pace(self):
        """Their remaining pile shrinks; their clueing did not get better."""
        history = [
            ['Red_Codemaster', 'A', 1],
            ['Red_Guesser', 'X', '*Blue*', False],
            ['Blue_Codemaster', 'B', 1],
            ['Blue_Guesser', 'Y', '*Blue*', False],
        ]
        counts = cm_mod.race_counts(history, "Red")
        self.assertEqual(counts["opp_pace_words"], 1)
        self.assertEqual(counts["opp_revealed"], 2)
        self.assertEqual(counts["own_pace_words"], 0)

    def test_with_no_evidence_the_priors_stand(self):
        history = [['Blue_Codemaster', 'B', 1],
                   ['Blue_Guesser', 'Y', '*Civilian*', False]]
        state = cm_mod.race_state(history, "Red", 9, 8)
        self.assertAlmostEqual(state["own_pace"], cm_mod.RACE_OWN_PACE_PRIOR, 3)
        self.assertAlmostEqual(state["opp_pace"],
                               cm_mod.RACE_OPP_PACE_PRIOR / 2.0, 3)

    def test_the_opponent_prior_is_a_competent_one_not_rattata(self):
        """A fresh duel must not assume the slowest opponent we have played."""
        self.assertGreaterEqual(cm_mod.RACE_OPP_PACE_PRIOR, 1.5)
        self.assertGreater(cm_mod.RACE_OPP_PACE_PRIOR,
                           cm_mod.RACE_OWN_PACE_PRIOR)

    def test_an_even_opening_is_not_a_deficit(self):
        """Red is 9 words to blue's 8 and moves first: that is a fair start."""
        history = [['Red_Codemaster', 'A', 3],
                   ['Red_Guesser', 'X', '*Red*', True],
                   ['Red_Guesser', 'Y', '*Red*', True],
                   ['Red_Guesser', 'Z', '*Red*', False],
                   ['Blue_Codemaster', 'B', 3],
                   ['Blue_Guesser', 'P', '*Blue*', True],
                   ['Blue_Guesser', 'Q', '*Blue*', True],
                   ['Blue_Guesser', 'R', '*Blue*', False]]
        state = cm_mod.race_state(history, "Red", 6, 5)
        self.assertLessEqual(state["deficit"], 0)
        self.assertEqual(state["scale"], 0.0)

    def test_blue_loses_the_tie_that_red_wins(self):
        """Moving second is worth a turn, and the deficit has to say so."""
        history = [['Red_Codemaster', 'A', 1],
                   ['Red_Guesser', 'X', '*Red*', False],
                   ['Blue_Codemaster', 'B', 1],
                   ['Blue_Guesser', 'P', '*Blue*', False]]
        as_red = cm_mod.race_state(history, "Red", 6, 6)
        as_blue = cm_mod.race_state(history, "Blue", 6, 6)
        self.assertEqual(as_blue["deficit"], as_red["deficit"] + 1)

    def test_a_scoreless_team_still_has_a_finite_pace(self):
        history = [['Red_Codemaster', 'A', 1],
                   ['Red_Guesser', 'X', '*Civilian*', False]] * 6
        history.append(['Blue_Codemaster', 'B', 1])
        history.append(['Blue_Guesser', 'P', '*Blue*', False])
        state = cm_mod.race_state(history, "Red", 9, 7)
        self.assertGreaterEqual(state["own_pace"], cm_mod.RACE_MIN_PACE)
        self.assertEqual(state["scale"], 1.0)

    def test_being_ahead_never_reads_as_a_negative_escalation(self):
        history = [['Red_Codemaster', 'A', 4],
                   ['Red_Guesser', 'W', '*Red*', True],
                   ['Red_Guesser', 'X', '*Red*', True],
                   ['Red_Guesser', 'Y', '*Red*', True],
                   ['Red_Guesser', 'Z', '*Red*', False],
                   ['Blue_Codemaster', 'B', 1],
                   ['Blue_Guesser', 'P', '*Civilian*', False]]
        state = cm_mod.race_state(history, "Red", 5, 8)
        self.assertLess(state["deficit"], 0)
        self.assertEqual(state["scale"], 0.0)

    def test_a_malformed_history_is_survivable(self):
        for history in ([None], [[]], [["Red_Codemaster"]],
                        [["Blue_Guesser"]], [["Blue_Codemaster", "B", 1]]):
            cm_mod.race_state(history, "Red", 9, 8)


class TestRaceKnobs(unittest.TestCase):
    """How the escalation scales, and where it stops."""

    BASE = (cm_mod.BONUS_GUESS_WEIGHT, cm_mod.CIVILIAN_PENALTY,
            cm_mod.CLAIMED_SLACK)

    def knobs(self, scale, base=None):
        return cm_mod.race_knobs(scale, *(base or self.BASE))

    def test_scale_zero_is_the_shipped_configuration(self):
        self.assertEqual(self.knobs(0.0), self.BASE)

    def test_scale_one_is_exactly_the_ambitious_preset(self):
        target = cm_mod.PRESETS["ambitious"]
        bonus, civilian, slack = self.knobs(1.0)
        self.assertAlmostEqual(bonus, target["bonus_guess_weight"], 6)
        self.assertAlmostEqual(civilian, target["civilian_penalty"], 6)
        self.assertEqual(slack, target["claimed_slack"])

    def test_the_escalation_is_monotone_in_the_scale(self):
        previous = self.knobs(0.0)
        for step in range(1, 21):
            current = self.knobs(step / 20.0)
            self.assertGreaterEqual(current[0], previous[0])
            self.assertLessEqual(current[1], previous[1])
            self.assertGreaterEqual(current[2], previous[2])
            previous = current

    def test_it_never_escalates_past_the_ambitious_values(self):
        target = cm_mod.PRESETS["ambitious"]
        for scale in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 99.0):
            bonus, civilian, slack = self.knobs(scale)
            self.assertLessEqual(bonus, target["bonus_guess_weight"] + 1e-9)
            self.assertGreaterEqual(civilian,
                                    target["civilian_penalty"] - 1e-9)
            self.assertLessEqual(slack, target["claimed_slack"])

    def test_a_negative_scale_clamps_to_the_shipped_values(self):
        self.assertEqual(self.knobs(-3.0), self.BASE)

    def test_the_slack_arms_a_step_later_than_the_weights(self):
        """Only slack literally adds a word, so it is the last knob to move."""
        below = self.knobs(cm_mod.RACE_SLACK_SCALE - 0.01)
        at = self.knobs(cm_mod.RACE_SLACK_SCALE)
        self.assertEqual(below[2], cm_mod.CLAIMED_SLACK)
        self.assertEqual(at[2], cm_mod.PRESETS["ambitious"]["claimed_slack"])
        self.assertGreater(below[0], cm_mod.BONUS_GUESS_WEIGHT)
        self.assertLess(below[1], cm_mod.CIVILIAN_PENALTY)

    def test_an_already_ambitious_configuration_is_never_pulled_back(self):
        """Escalation may raise ambition and may never lower it."""
        bold = (0.8, 0.2, 3)
        for scale in (0.0, 0.5, 1.0):
            self.assertEqual(cm_mod.race_knobs(scale, *bold), bold)


class TestRaceInTheAgent(unittest.TestCase):
    """The wiring: what the codemaster does with the reading."""

    def _agent(self, history, own_found=0, opp_found=0, **kwargs):
        kwargs.setdefault("embed_sensor", False)
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(board_with(own_found, opp_found), list(KEY))
        agent.set_move_history(list(history))
        return agent

    def _own_opp(self, agent):
        own, opp, _, _ = agent._split_board()
        return own, opp

    def test_a_single_team_history_leaves_every_knob_at_its_base(self):
        agent = self._agent(SOLO_HISTORY)
        agent._race_update(*self._own_opp(agent))
        self.assertIsNone(agent._race)
        self.assertEqual(agent._race_bonus, agent.bonus_guess_weight)
        self.assertEqual(agent._race_civilian, agent.civilian_penalty)
        self.assertEqual(agent._race_slack, agent.claimed_slack)

    def test_a_losing_duel_raises_every_knob(self):
        agent = self._agent(history_prefix(ABRA_SEED_100, 2),
                            own_found=1, opp_found=3)
        agent._race_update(*self._own_opp(agent))
        self.assertIsNotNone(agent._race)
        self.assertEqual(agent._race["scale"], 1.0)
        target = cm_mod.PRESETS["ambitious"]
        self.assertAlmostEqual(agent._race_bonus,
                               target["bonus_guess_weight"], 6)
        self.assertAlmostEqual(agent._race_civilian,
                               target["civilian_penalty"], 6)
        self.assertEqual(agent._race_slack, target["claimed_slack"])

    def test_a_winning_duel_keeps_them_conservative(self):
        history = [['Red_Codemaster', 'A', 4],
                   ['Red_Guesser', 'W', '*Red*', True],
                   ['Red_Guesser', 'X', '*Red*', True],
                   ['Red_Guesser', 'Y', '*Red*', True],
                   ['Red_Guesser', 'Z', '*Red*', False],
                   ['Blue_Codemaster', 'B', 1],
                   ['Blue_Guesser', 'P', '*Civilian*', False]]
        agent = self._agent(history)
        agent._race_update(*self._own_opp(agent))
        self.assertEqual(agent._race["scale"], 0.0)
        self.assertEqual(agent._race_bonus, agent.bonus_guess_weight)
        self.assertEqual(agent._race_slack, agent.claimed_slack)

    def test_race_mode_off_is_inert_in_a_losing_duel(self):
        agent = self._agent(history_prefix(ABRA_SEED_100, 2), own_found=1,
                            opp_found=3, race_mode=False)
        agent._race_update(*self._own_opp(agent))
        self.assertIsNone(agent._race)
        self.assertEqual(agent._race_slack, agent.claimed_slack)

    def test_the_environment_can_turn_it_off(self):
        saved = os.environ.get(cm_mod.RACE_MODE_ENV)
        try:
            os.environ[cm_mod.RACE_MODE_ENV] = "0"
            self.assertFalse(cm_mod.AICodemaster("Red").race_mode)
            os.environ[cm_mod.RACE_MODE_ENV] = "1"
            self.assertTrue(cm_mod.AICodemaster("Red").race_mode)
            os.environ.pop(cm_mod.RACE_MODE_ENV)
            self.assertTrue(cm_mod.AICodemaster("Red").race_mode)
        finally:
            if saved is None:
                os.environ.pop(cm_mod.RACE_MODE_ENV, None)
            else:
                os.environ[cm_mod.RACE_MODE_ENV] = saved

    # -- the number the escalation actually buys ---------------------------

    BRAINSTORM = ('[{"clue":"OCEAN","targets":["WHALE","SHIP"]}]')
    PANEL = '{"OCEAN": ["WHALE", "SHIP", "BEACH", "ROBOT"]}'

    def _play(self, history, **kwargs):
        """One scripted turn: the panel supports 3, the brainstorm claims 2."""
        agent = self._agent(history, **kwargs)

        def responder(system, user):
            if is_brainstorm(system):
                return self.BRAINSTORM
            if is_probe(system):
                return '{"POISON": 0, "WHALE": 8, "APPLE": 0}'
            return self.PANEL

        with _PatchedLLM(cm_mod, responder):
            return agent, agent.get_clue()

    def test_the_claimed_ceiling_lifts_only_while_losing(self):
        _, (safe_clue, safe_number) = self._play(SOLO_HISTORY, own_found=3)
        _, (race_clue, race_number) = self._play(
            history_prefix(ABRA_SEED_100, 2), own_found=1, opp_found=3)
        self.assertEqual(safe_clue, race_clue)
        self.assertEqual(safe_number, 2)        # the brainstorm's own claim
        self.assertEqual(race_number, 3)        # +1 of slack, panel-supported

    def test_the_escalated_number_never_exceeds_the_ambitious_arm(self):
        _, (_, race_number) = self._play(history_prefix(ABRA_SEED_100, 2),
                                         own_found=1, opp_found=3)
        _, (_, ambitious_number) = self._play(SOLO_HISTORY, own_found=3,
                                              preset="ambitious")
        self.assertEqual(race_number, ambitious_number)

    def test_one_turn_behind_is_a_half_escalated_turn(self):
        """Deficits are integers, so the shipped scale takes three values."""
        agent, (_, number) = self._play(LEVEL_MINUS_ONE_HISTORY,
                                        own_found=3, opp_found=3)
        self.assertEqual(agent._race["deficit"], 1)
        self.assertEqual(agent._race["scale"], 0.5)
        self.assertLess(agent._race_civilian, agent.civilian_penalty)
        self.assertGreater(agent._race_bonus, agent.bonus_guess_weight)
        self.assertEqual(number, 3)

    def test_the_slack_threshold_is_the_retune_lever(self):
        """Raising it holds the ceiling at one turn behind and only there.

        This is the knob to reach for first if the do-no-harm regression run
        against the slower opponent comes back with a death: it withdraws the
        number lift from the photo-finish games while leaving the scoring
        weights, and the full escalation two turns behind, exactly as they are.
        """
        agent, (_, number) = self._play(LEVEL_MINUS_ONE_HISTORY,
                                        own_found=3, opp_found=3,
                                        race_slack_scale=0.75)
        self.assertEqual(agent._race["scale"], 0.5)
        self.assertEqual(agent._race_slack, agent.claimed_slack)
        self.assertEqual(number, 2)
        hopeless, (_, hopeless_number) = self._play(
            history_prefix(ABRA_SEED_100, 2), own_found=1, opp_found=3,
            race_slack_scale=0.75)
        self.assertEqual(hopeless._race["scale"], 1.0)
        self.assertEqual(hopeless_number, 3)

    # -- inertness, proved rather than asserted ----------------------------

    def test_single_team_play_is_identical_with_race_mode_on_and_off(self):
        with _Captured() as loud:
            on, on_result = self._play(SOLO_HISTORY, own_found=3)
        with _Captured() as quiet:
            off, off_result = self._play(SOLO_HISTORY, own_found=3,
                                         race_mode=False)
        self.assertEqual(on_result, off_result)
        self.assertEqual(loud.text().replace("race=on", "race=off"),
                         quiet.text())

    def test_the_turn_line_gains_a_race_segment_only_in_a_duel(self):
        with _Captured() as solo:
            self._play(SOLO_HISTORY, own_found=3)
        with _Captured() as duel:
            self._play(history_prefix(ABRA_SEED_100, 2), own_found=1,
                       opp_found=3)
        self.assertNotIn("race ", solo.text())
        self.assertIn("escalation 1.00", duel.text())

    def test_the_usage_summary_is_unchanged_in_single_team(self):
        solo, _ = self._play(SOLO_HISTORY, own_found=3)
        duel, _ = self._play(history_prefix(ABRA_SEED_100, 2), own_found=1,
                             opp_found=3)
        self.assertNotIn("race_mode", solo.usage_summary())
        self.assertNotIn("race_log", solo.usage_summary())
        self.assertEqual(duel.usage_summary()["race_escalated"], 1)
        self.assertEqual(len(duel.usage_summary()["race_log"]), 1)

    def test_scoring_reads_the_race_knobs_not_the_configured_ones(self):
        """The knobs are per-turn state, so a stale turn cannot leak forward."""
        agent = self._agent(SOLO_HISTORY)
        own = ["WHALE", "SHIP", "BEACH"]
        rankings = [["WHALE", "SHIP", "BEACH"]]
        base = agent._score_candidate(rankings, own, [], [], ["POISON"], 2)
        agent.set_move_history(history_prefix(ABRA_SEED_100, 2))
        agent._race_update(*self._own_opp(agent))
        raised = agent._score_candidate(rankings, own, [], [], ["POISON"], 2)
        self.assertGreater(raised[1], base[1])
        agent.set_move_history(list(SOLO_HISTORY))
        agent._race_update(*self._own_opp(agent))
        self.assertEqual(agent._score_candidate(rankings, own, [], [],
                                                ["POISON"], 2), base)


class TestRaceAudit(unittest.TestCase):
    """The offline auditor that judges a death against the projection."""

    def setUp(self):
        from harness import race_audit

        self.audit = race_audit

    def _game(self, history, **kwargs):
        game = {"seed": 0, "single_team": False, "winner": "B",
                "red_found": 3, "blue_found": 8, "move_history": history}
        game.update(kwargs)
        return game

    def test_the_auditor_reads_the_same_race_the_agent_did(self):
        turns = self.audit.walk_game(self._game(ABRA_SEED_100))
        self.assertEqual(len(turns), 3)
        self.assertIsNone(turns[0]["race"])
        for index in (1, 2):
            expected = race_at(ABRA_SEED_100, index + 1)
            self.assertEqual(turns[index]["race"]["deficit"],
                             expected["deficit"])
            self.assertEqual(turns[index]["race"]["scale"], expected["scale"])

    def test_a_clean_game_has_no_death_to_judge(self):
        verdict = self.audit.audit_game(self._game(ABRA_SEED_100))
        self.assertFalse(verdict["assassin_death"])
        self.assertEqual(verdict["escalated_turns"], 2)
        self.assertEqual(verdict["max_scale"], 1.0)

    def test_a_death_taken_while_projected_lost_is_priced(self):
        history = history_prefix(ABRA_SEED_100, 3) + [
            ['Red_Codemaster', 'SURF', 2],
            ['Red_Guesser', 'POISON', '*Assassin*', False],
        ]
        verdict = self.audit.audit_game(self._game(history))
        self.assertTrue(verdict["assassin_death"])
        self.assertEqual(verdict["death_turn"], 3)
        self.assertEqual(verdict["death_clue"], "SURF 2")
        self.assertTrue(verdict["death_projected_lost"])

    def test_a_death_taken_while_level_or_ahead_is_not(self):
        history = [['Red_Codemaster', 'A', 4],
                   ['Red_Guesser', 'W', '*Red*', True],
                   ['Red_Guesser', 'X', '*Red*', True],
                   ['Red_Guesser', 'Y', '*Red*', True],
                   ['Red_Guesser', 'Z', '*Red*', False],
                   ['Blue_Codemaster', 'B', 1],
                   ['Blue_Guesser', 'P', '*Civilian*', False],
                   ['Red_Codemaster', 'C', 3],
                   ['Red_Guesser', 'POISON', '*Assassin*', False]]
        verdict = self.audit.audit_game(self._game(history))
        self.assertTrue(verdict["assassin_death"])
        self.assertFalse(verdict["death_projected_lost"])

    def test_an_opponent_death_is_not_ours_to_answer_for(self):
        history = list(ABRA_SEED_100) + [
            ['Blue_Guesser', 'POISON', '*Assassin*', False]]
        self.assertFalse(
            self.audit.audit_game(self._game(history))["assassin_death"])

    def test_single_team_games_are_skipped_entirely(self):
        import tempfile

        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            with open(path, "w") as out:
                json.dump([self._game(SOLO_HISTORY, single_team=True),
                           self._game(ABRA_SEED_100)], out)
            report = self.audit.audit_file(path)
        finally:
            os.unlink(path)
        self.assertEqual(report["games"], 1)
        self.assertEqual(report["skipped_single_team"], 1)
        self.assertEqual(report["deaths"], 0)

    def test_the_copied_fixtures_still_match_the_recorded_run(self):
        """``results_local/`` is gitignored; check the copies when it is here."""
        path = os.path.join(REPO_ROOT, "results_local", "gauntlet_vs_abra.json")
        if not os.path.exists(path):
            self.skipTest("gauntlet record not on disk")
        with open(path, "r") as handle:
            games = dict((g["seed"], g) for g in json.load(handle))
        self.assertEqual(games[100]["move_history"], ABRA_SEED_100)
        self.assertEqual(games[102]["move_history"], ABRA_SEED_102)


# ---------------------------------------------------------------------------
# Guesser behaviour
# ---------------------------------------------------------------------------

class TestGuesser(unittest.TestCase):

    def _agent(self, **kwargs):
        agent = g_mod.AIGuesser("Red", **kwargs)
        agent.set_board(list(BOARD))
        return agent

    def test_constructor_signature_matches_baseline(self):
        agent = g_mod.AIGuesser("Blue")
        self.assertEqual(agent.team, "Blue")

    def test_answer_is_always_a_real_unrevealed_word(self):
        agent = self._agent()
        board = list(BOARD)
        board[0] = "*Red*"
        agent.set_board(board)
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "SHIP": 80}'):
            agent.set_clue("OCEAN", 2)
            answer = agent.get_answer()
        self.assertIn(answer, board)
        self.assertFalse(answer.startswith("*"))
        self.assertEqual(answer, "SHIP")   # WHALE is already revealed

    def test_offline_fallback_returns_a_word(self):
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: None, available=False):
            agent.set_clue("OCEAN", 2)
            answer = agent.get_answer()
        self.assertIn(answer, BOARD)

    def test_unparseable_reply_falls_back(self):
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: "!!! ???"):
            agent.set_clue("OCEAN", 2)
            answer = agent.get_answer()
        self.assertIn(answer, BOARD)

    def test_guess_counter_resets_between_turns(self):
        """The bundled guesser_GPT never does this; we must."""
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "SHIP": 95}'):
            agent.set_clue("OCEAN", 2)
            agent.get_answer()
            agent.get_answer()
            self.assertEqual(agent.guesses, 2)
            agent.set_clue("ROYAL", 2)
            self.assertEqual(agent.guesses, 0)
            agent.get_answer()
            self.assertTrue(agent.keep_guessing())

    def test_stops_when_confidence_collapses(self):
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "SHIP": 4}'):
            agent.set_clue("OCEAN", 2)
            self.assertEqual(agent.get_answer(), "WHALE")
            # The engine reveals the guessed word before asking to continue.
            board = list(BOARD)
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            self.assertFalse(agent.keep_guessing())

    def test_continues_while_confident(self):
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "SHIP": 96}'):
            agent.set_clue("OCEAN", 2)
            agent.get_answer()
            self.assertTrue(agent.keep_guessing())

    def test_hard_cap_at_number_plus_one(self):
        agent = self._agent()
        scores = '{"WHALE": 99, "SHIP": 98, "BEACH": 97, "APPLE": 96}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 2)
            for _ in range(3):
                agent.get_answer()
            self.assertEqual(agent.guesses, 3)
            self.assertFalse(agent.keep_guessing())

    def test_bonus_guess_needs_leftover_support(self):
        agent = self._agent()
        scores = '{"WHALE": 99, "SHIP": 98, "BEACH": 97}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 2)
            agent.get_answer()
            agent.get_answer()
            # Two guesses made against number=2: only a leftover justifies +1.
            self.assertFalse(agent.keep_guessing())

    def test_bonus_guess_taken_with_leftover_support(self):
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: '{"BEACH": 99, "SHIP": 90}'):
            agent.set_clue("SHORE", 1)          # earlier, unexhausted clue
            agent.get_answer()
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "BEACH": 98}'):
            agent.set_clue("OCEAN", 1)
            agent.get_answer()                  # takes WHALE
            board = list(BOARD)
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            self.assertTrue(agent.keep_guessing())

    def _photo_finish_board(self):
        """Blue is two words from winning; red still needs several."""
        board = list(BOARD)
        return self._race_board(reds_revealed=7)

    def _race_board(self, reds_revealed=7):
        """Blue two words from winning; WHALE and SHIP stay ours and hidden.

        ``reds_revealed=7`` leaves two of our words on the board, so the guess
        we are about to make still does not finish the game; ``8`` leaves one,
        which is the real 8-8 photo finish the gamble exists for.
        """
        board = list(BOARD)
        blues = [i for i, kind in enumerate(KEY) if kind == "Blue"]
        for index in blues[:6]:
            board[index] = "*Blue*"
        reds = [i for i, kind in enumerate(KEY) if kind == "Red"
                and BOARD[i] not in ("WHALE", "SHIP")]
        for index in reds[:reds_revealed]:
            board[index] = "*Red*"
        if reds_revealed > len(reds):
            board[BOARD.index("SHIP")] = "*Red*"
        return board

    def test_must_gamble_only_fires_in_a_two_team_photo_finish(self):
        solo = self._agent()
        solo.set_board(self._race_board(reds_revealed=8))
        self.assertFalse(solo._must_gamble())      # single team: no race

        duel = self._agent()
        duel.set_move_history([["Blue_Codemaster", "ROYAL", 2]])
        duel.set_board(list(BOARD))
        self.assertFalse(duel._must_gamble())      # nobody is close yet
        duel.set_board(self._race_board(reds_revealed=8))
        self.assertTrue(duel._must_gamble())       # one word from home

    def test_must_gamble_refuses_when_the_bonus_cannot_win(self):
        """eval_c_default seed 14: two words short, opponent one from home.

        The +1 buys one word, which still would not end the game, while a
        miss can lose it outright.  That gamble has no upside to pay for it.
        """
        duel = self._agent()
        duel.set_move_history([["Blue_Codemaster", "ROYAL", 2]])
        duel.set_board(self._race_board(reds_revealed=7))
        self.assertEqual(duel._words_left(), (2, 2))
        self.assertFalse(duel._must_gamble())

    def test_bonus_guess_skips_leftover_support_when_losing_the_race(self):
        """Declining the +1 concedes the race, so a strong current
        association is enough on its own."""
        agent = self._agent()
        agent.set_move_history([["Blue_Codemaster", "ROYAL", 2]])
        agent.set_board(self._race_board(reds_revealed=7))
        scores = '{"WHALE": 99, "SHIP": 98}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 1)
            # Guessing WHALE leaves exactly one of ours: the +1 can win.
            self.assertEqual(agent.get_answer(), "WHALE")
            self.assertTrue(agent.keep_guessing())

    def test_bonus_guess_still_needs_confidence_when_losing_the_race(self):
        """Gambling is not the same as guessing blindly."""
        agent = self._agent()
        agent.set_move_history([["Blue_Codemaster", "ROYAL", 2]])
        agent.set_board(self._race_board(reds_revealed=7))
        scores = '{"WHALE": 99, "SHIP": 5}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 1)
            self.assertEqual(agent.get_answer(), "WHALE")
            self.assertFalse(agent.keep_guessing())

    def test_position_bias_guard_shuffles_each_sample(self):
        agent = self._agent(samples=3)
        seen = []

        def responder(system, user):
            seen.append(user)
            return '{"WHALE": 90}'

        with _PatchedLLM(g_mod, responder):
            agent.set_clue("OCEAN", 2)
            agent.get_answer()
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3)   # word order differs per sample

    def test_ranking_is_computed_once_per_turn(self):
        agent = self._agent(samples=2)
        calls = []

        def responder(system, user):
            calls.append(user)
            return '{"WHALE": 99, "SHIP": 95, "BEACH": 90}'

        with _PatchedLLM(g_mod, responder):
            agent.set_clue("OCEAN", 3)
            agent.get_answer()
            board = list(BOARD)
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            agent.keep_guessing()
            agent.get_answer()
        self.assertEqual(len(calls), 2)       # samples only, no re-scoring

    def test_two_team_mode_is_stricter(self):
        solo = self._agent()
        duel = self._agent()
        duel.set_move_history([["Blue_Codemaster", "ROYAL", 2]])
        self.assertGreater(duel._continue_threshold(), solo._continue_threshold())

    def test_starting_one_word_behind_is_not_treated_as_losing(self):
        """Red always has 9 words to Blue's 8; that alone is not a deficit."""
        agent = self._agent()
        agent.set_move_history([["Blue_Codemaster", "ROYAL", 2]])
        self.assertEqual(agent._urgency(), 0.0)

    def test_urgency_rises_when_the_opponent_is_about_to_win(self):
        agent = self._agent()
        agent.set_move_history([["Blue_Codemaster", "ROYAL", 2]])
        board = list(BOARD)
        blues = [i for i, kind in enumerate(KEY) if kind == "Blue"]
        for index in blues[:6]:              # Blue is two words from winning
            board[index] = "*Blue*"
        agent.set_board(board)
        self.assertLessEqual(agent._urgency(), -0.10)
        self.assertLess(agent._continue_threshold(), CONTINUE_BASE_DUEL)

    def test_urgency_is_off_in_single_team_games(self):
        agent = self._agent()
        board = list(BOARD)
        blues = [i for i, kind in enumerate(KEY) if kind == "Blue"]
        for index in blues[:6]:
            board[index] = "*Blue*"
        agent.set_board(board)
        self.assertEqual(agent._urgency(), 0.0)

    def test_counts_track_revealed_own_words(self):
        agent = self._agent()
        board = list(BOARD)
        board[0] = "*Red*"
        board[1] = "*Red*"
        board[3] = "*Blue*"
        agent.set_board(board)
        own_left, unrevealed = agent._counts()
        self.assertEqual(own_left, 7)
        self.assertEqual(unrevealed, 22)

    def test_unlimited_clue_keeps_going_while_confident(self):
        agent = self._agent()
        scores = '{"WHALE": 99, "SHIP": 99, "BEACH": 20}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 0)
            board = list(BOARD)
            agent.get_answer()
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            # A clue of 1 would have stopped here; 0 is unlimited, and SHIP is
            # still nearly as strong as the turn's best.
            self.assertEqual(agent.guesses, 1)
            self.assertTrue(agent.keep_guessing())

    def test_unlimited_clue_stops_below_the_sweep_confidence_floor(self):
        """A merely-plausible next word is not enough on an unlimited turn."""
        agent = self._agent()
        scores = '{"WHALE": 99, "SHIP": 60, "BEACH": 40}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 0)
            board = list(BOARD)
            agent.get_answer()
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            self.assertFalse(agent.keep_guessing())

    def test_sweep_confidence_floor_is_configurable(self):
        agent = self._agent(sweep_confidence=0.10)
        scores = '{"WHALE": 99, "SHIP": 60, "BEACH": 40}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 0)
            board = list(BOARD)
            agent.get_answer()
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            self.assertTrue(agent.keep_guessing())

    def test_leftover_from_an_earlier_clue_survives_the_sweep_floor(self):
        """The point of a sweep: a word an earlier clue owned is still taken."""
        agent = self._agent()
        replies = {
            "ROYAL": '{"KING": 99, "CROWN": 97}',
            "OCEAN": '{"WHALE": 99, "SHIP": 20}',
        }

        def responder(system, user):
            return replies["ROYAL"] if "ROYAL" in user else replies["OCEAN"]

        with _PatchedLLM(g_mod, responder):
            agent.set_clue("ROYAL", 1)
            agent.get_answer()                       # KING; CROWN left behind
            board = list(BOARD)
            board[BOARD.index("KING")] = "*Red*"
            agent.set_board(board)

            agent.set_clue("OCEAN", 0)
            agent.get_answer()                       # WHALE
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            # CROWN carries none of OCEAN's meaning, but it was ROYAL's clear
            # top pick among the words still hidden, so the sweep still wants it.
            self.assertTrue(agent.keep_guessing())
            self.assertEqual(agent.get_answer(), "CROWN")

    def test_unlimited_clue_stops_at_own_words_remaining(self):
        agent = self._agent()
        board = list(BOARD)
        reds = [i for i, kind in enumerate(KEY) if kind == "Red"]
        for index in reds[2:]:
            board[index] = "*Red*"          # only 2 Red words left
        agent.set_board(board)
        scores = '{"WHALE": 99, "SHIP": 98, "BEACH": 97, "TREE": 96}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 0)
            for word in ("WHALE", "SHIP"):
                agent.get_answer()
                board[BOARD.index(word)] = "*Red*"
                agent.set_board(board)
            self.assertFalse(agent.keep_guessing())

    def test_huge_number_is_treated_as_unlimited(self):
        agent = self._agent()
        agent.set_clue("OCEAN", 9999)
        self.assertTrue(agent._is_sweep())
        limit, sweep = agent._turn_capacity()
        self.assertTrue(sweep)
        self.assertEqual(limit, 9)          # all nine Red words still hidden

    def test_unlimited_clue_uses_leftover_memory(self):
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: '{"BEACH": 99, "TREE": 10}'):
            agent.set_clue("SHORE", 1)
            self.assertEqual(agent.get_answer(), "BEACH")
        board = list(BOARD)
        board[BOARD.index("BEACH")] = "*Red*"
        agent.set_board(board)
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "TREE": 5}'):
            agent.set_clue("MAMMAL", 0)
            self.assertEqual(agent.get_answer(), "WHALE")
            board[BOARD.index("WHALE")] = "*Red*"
            agent.set_board(board)
            ranking = agent._get_ranking(agent._options())
        # TREE scored 5 for MAMMAL, but SHORE ranked it second: the blended
        # sweep ranking must lift it above equally-weak unrelated words.
        self.assertEqual(ranking[0][0], "TREE")

    def test_unlimited_clue_is_stricter_than_a_finite_one(self):
        agent = self._agent()
        agent.set_clue("OCEAN", 3)
        finite = agent._continue_threshold()
        agent.set_clue("OCEAN", 0)
        self.assertTrue(agent._is_sweep())
        self.assertEqual(finite, agent._continue_threshold())  # base is shared
        # The sweep confidence floor is applied inside _keep_guessing_inner.
        # SHIP sits at 0.60 of the turn's best: over the finite bar, under the
        # sweep floor.  Both arms follow the engine's call order (no set_board
        # between the guess and the stop-rule question).
        scores = '{"WHALE": 100, "SHIP": 60}'
        with _PatchedLLM(g_mod, lambda s, u: scores):
            agent.set_clue("OCEAN", 0)
            self.assertEqual(agent.get_answer(), "WHALE")
            swept = agent.keep_guessing()
            agent2 = self._agent()
            agent2.set_clue("OCEAN", 3)
            self.assertEqual(agent2.get_answer(), "WHALE")
            finite_ok = agent2.keep_guessing()
        self.assertTrue(finite_ok)
        self.assertFalse(swept)

    # -- engine call order -------------------------------------------------
    #
    # game.Game.run does, per guess:
    #     set_board(live_list)        # our set_board takes a *copy*
    #     get_answer()
    #     _accept_guess()             # engine reveals the word in its own list
    #     keep_guessing()             # <- no set_board in between
    # so the stop rule sees a board that is one guess stale.  Before the
    # ``_pending`` fix the just-guessed word was still offered as "the next
    # candidate", which made every confidence ratio 1.0 and the stop rule a
    # no-op.  These tests drive that exact order.

    def _engine_turn(self, agent, clue, num, board, key, max_guesses=6):
        """Replay game.Game.run's inner loop; returns [(word, colour, keep)]."""
        agent.set_clue(clue, num)
        log = []
        keep = True
        while keep and len(log) < max_guesses:
            agent.set_move_history(agent.move_history)
            agent.set_board(board)                 # snapshot BEFORE the guess
            guess = agent.get_answer()
            if guess is None:
                break
            index = board.index(guess)
            colour = key[index]
            board[index] = "*%s*" % colour         # engine reveals its own list
            if colour != agent.team:
                log.append((guess, colour, False))
                break
            keep = agent.keep_guessing()           # our copy is now stale
            log.append((guess, colour, keep))
        return log

    def test_stop_rule_is_not_defeated_by_the_stale_board(self):
        """A weak second candidate must be refused under the engine's order."""
        agent = self._agent()
        board, key = list(BOARD), list(KEY)
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "SHIP": 4}'):
            log = self._engine_turn(agent, "OCEAN", 2, board, key)
        self.assertEqual([entry[0] for entry in log], ["WHALE"])
        self.assertFalse(log[0][2])

    def test_stop_rule_still_continues_on_a_strong_second_candidate(self):
        agent = self._agent()
        board, key = list(BOARD), list(KEY)
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "SHIP": 95}'):
            log = self._engine_turn(agent, "OCEAN", 2, board, key)
        self.assertEqual([entry[0] for entry in log], ["WHALE", "SHIP"])

    def test_pending_guess_is_counted_as_one_of_ours(self):
        """keep_guessing is only reached when the guess was our own word."""
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99, "SHIP": 4}'):
            agent.set_clue("OCEAN", 2)
            self.assertEqual(agent.get_answer(), "WHALE")
            own_left, unrevealed = agent._counts()
        self.assertEqual(agent._pending_own(), "WHALE")
        self.assertEqual(own_left, 8)              # 9 - the pending WHALE
        self.assertEqual(unrevealed, 24)           # WHALE excluded
        self.assertNotIn("WHALE", agent._options())

    def test_pending_clears_once_the_board_catches_up(self):
        agent = self._agent()
        with _PatchedLLM(g_mod, lambda s, u: '{"WHALE": 99}'):
            agent.set_clue("OCEAN", 2)
            agent.get_answer()
        board = list(BOARD)
        board[BOARD.index("WHALE")] = "*Red*"
        agent.set_board(board)
        self.assertIsNone(agent._pending_own())
        self.assertEqual(agent._counts()[0], 8)    # counted once, not twice

    def test_never_raises_on_empty_board(self):
        agent = g_mod.AIGuesser("Red")
        agent.set_board(["*Red*"] * 25)
        agent.set_clue("OCEAN", 2)
        self.assertIsNone(agent.get_answer())
        self.assertFalse(agent.keep_guessing())

    def test_exception_inside_pipeline_is_contained(self):
        agent = self._agent()

        def boom(system, user):
            raise RuntimeError("simulated crash")

        with _PatchedLLM(g_mod, boom):
            agent.set_clue("OCEAN", 2)
            answer = agent.get_answer()
        self.assertIn(answer, BOARD)

    def test_deadline_is_respected(self):
        agent = self._agent(deadline=3.0)

        def responder(system, user):
            time.sleep(1.0)
            return None

        started = time.time()
        with _PatchedLLM(g_mod, responder):
            agent.set_clue("OCEAN", 2)
            answer = agent.get_answer()
        self.assertLess(time.time() - started, 12.0)
        self.assertIn(answer, BOARD)


# ---------------------------------------------------------------------------
# Provider switch (Anthropic by default, OpenAI-compatible for plumbing runs)
# ---------------------------------------------------------------------------

class _PatchedEnv(object):
    """Set/clear environment variables for the duration of a block."""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.saved[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


class _PatchedUrlopen(object):
    """Capture the HTTP request the compat provider would have sent."""

    def __init__(self, module, replies):
        self.module = module
        self.replies = list(replies)
        self.requests = []

    def __enter__(self):
        self._saved = self.module.urllib.request.urlopen
        self._sleep = self.module.time.sleep

        def urlopen(request, timeout=None):
            self.requests.append(request)
            reply = self.replies.pop(0) if self.replies else None
            if isinstance(reply, Exception):
                raise reply
            return _FakeResponse(reply)

        self.module.urllib.request.urlopen = urlopen
        self.module.time.sleep = lambda seconds: None
        return self

    def __exit__(self, *exc):
        self.module.urllib.request.urlopen = self._saved
        self.module.time.sleep = self._sleep
        return False


class _FakeResponse(object):

    def __init__(self, payload):
        if isinstance(payload, (bytes, str)):
            self.body = payload if isinstance(payload, bytes) else payload.encode()
        else:
            self.body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.body

    def close(self):
        pass


class _FakeHTTPError(Exception):

    def __init__(self, code):
        Exception.__init__(self, "HTTP Error %d" % code)
        self.code = code


def compat_reply(text, prompt_tokens=11, completion_tokens=7):
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens}}


class TestProviderSelection(unittest.TestCase):

    MODULES = (cm_mod, g_mod)

    def test_default_is_anthropic(self):
        with _PatchedEnv(OBIRDY_PROVIDER=None, OBIRDY_MODEL=None):
            for module in self.MODULES:
                self.assertEqual(module._resolve_provider(),
                                 module.PROVIDER_ANTHROPIC)
                self.assertEqual(
                    module._resolve_model(module.PROVIDER_ANTHROPIC),
                    module.DEFAULT_MODEL)

    def test_env_selects_the_compat_provider(self):
        with _PatchedEnv(OBIRDY_PROVIDER="openai_compat", OBIRDY_MODEL=None):
            for module in self.MODULES:
                self.assertEqual(module._resolve_provider(),
                                 module.PROVIDER_COMPAT)
                self.assertEqual(module._resolve_model(module.PROVIDER_COMPAT),
                                 module.DEFAULT_COMPAT_MODEL)

    def test_unknown_provider_falls_back_to_anthropic(self):
        with _PatchedEnv(OBIRDY_PROVIDER="ollama"):
            for module in self.MODULES:
                self.assertEqual(module._resolve_provider(),
                                 module.PROVIDER_ANTHROPIC)

    def test_kwarg_beats_the_environment(self):
        with _PatchedEnv(OBIRDY_PROVIDER="openai_compat", OBIRDY_MODEL=None):
            agent = cm_mod.AICodemaster("Red", provider="anthropic")
            self.assertEqual(agent.llm.provider, cm_mod.PROVIDER_ANTHROPIC)
            self.assertEqual(agent.model, cm_mod.DEFAULT_MODEL)

    def test_agents_wire_the_provider_into_their_client(self):
        with _PatchedEnv(OBIRDY_PROVIDER=None, OBIRDY_MODEL=None,
                         OBIRDY_BASE_URL=None):
            codemaster = cm_mod.AICodemaster("Red", provider="openai_compat")
            guesser = g_mod.AIGuesser("Red", provider="openai_compat")
            for agent, module in ((codemaster, cm_mod), (guesser, g_mod)):
                self.assertEqual(agent.llm.provider, module.PROVIDER_COMPAT)
                self.assertEqual(agent.model, module.DEFAULT_COMPAT_MODEL)
                self.assertEqual(agent.llm.base_url,
                                 module.DEFAULT_COMPAT_BASE_URL)

    def test_explicit_model_survives_the_provider_default(self):
        with _PatchedEnv(OBIRDY_MODEL=None):
            agent = cm_mod.AICodemaster("Red", provider="openai_compat",
                                        model="meta-llama/Llama-3.3-70B-Instruct")
            self.assertEqual(agent.model, "meta-llama/Llama-3.3-70B-Instruct")

    def test_model_env_still_wins_over_the_provider_default(self):
        with _PatchedEnv(OBIRDY_MODEL="Qwen/Qwen3-32B"):
            agent = cm_mod.AICodemaster("Red", provider="openai_compat")
            self.assertEqual(agent.model, "Qwen/Qwen3-32B")

    def test_base_url_comes_from_kwarg_or_env(self):
        with _PatchedEnv(OBIRDY_BASE_URL="https://example.invalid/v1"):
            self.assertEqual(cm_mod._LLM("m", provider="openai_compat").base_url,
                             "https://example.invalid/v1")
            self.assertEqual(
                cm_mod._LLM("m", provider="openai_compat",
                            base_url="https://other.invalid/v1").base_url,
                "https://other.invalid/v1")

    def test_compat_availability_follows_the_key(self):
        llm = cm_mod._LLM("m", provider="openai_compat")
        with _PatchedEnv(OBIRDY_COMPAT_KEY=None, HF_TOKEN=None):
            self.assertFalse(llm.available())
        with _PatchedEnv(OBIRDY_COMPAT_KEY=None, HF_TOKEN="hf_token"):
            self.assertTrue(llm.available())
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat", HF_TOKEN=None):
            self.assertTrue(llm.available())

    def test_compat_never_needs_the_anthropic_key(self):
        llm = cm_mod._LLM("m", provider="openai_compat")
        with _PatchedEnv(ANTHROPIC_API_KEY=None, OBIRDY_COMPAT_KEY="sk-compat"):
            self.assertTrue(llm.available())

    def test_provider_is_reported_in_usage(self):
        agent = cm_mod.AICodemaster("Red", provider="openai_compat")
        self.assertEqual(agent.usage_summary()["provider"],
                         cm_mod.PROVIDER_COMPAT)


class TestCompatTransport(unittest.TestCase):
    """Request shape and failure handling, with ``urlopen`` monkeypatched."""

    def _llm(self, module=cm_mod, **kwargs):
        kwargs.setdefault("provider", "openai_compat")
        kwargs.setdefault("base_url", "https://router.invalid/v1")
        return module._LLM("Qwen/Qwen2.5-72B-Instruct", **kwargs)

    def test_request_shape(self):
        llm = self._llm()
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat", HF_TOKEN=None):
            with _PatchedUrlopen(cm_mod, [compat_reply("OCEAN")]) as patched:
                text = llm.chat("system text", "user text", max_tokens=123)
        self.assertEqual(text, "OCEAN")
        self.assertEqual(len(patched.requests), 1)
        request = patched.requests[0]
        self.assertEqual(request.full_url,
                         "https://router.invalid/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers.get("Authorization"),
                         "Bearer sk-compat")
        self.assertEqual(request.headers.get("Content-type"),
                         "application/json")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(payload["max_tokens"], 123)
        self.assertEqual(payload["messages"],
                         [{"role": "system", "content": "system text"},
                          {"role": "user", "content": "user text"}])

    def test_base_url_trailing_slash_is_tolerated(self):
        llm = self._llm(base_url="https://router.invalid/v1/")
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(cm_mod, [compat_reply("OCEAN")]) as patched:
                llm.chat("s", "u")
        self.assertEqual(patched.requests[0].full_url,
                         "https://router.invalid/v1/chat/completions")

    def test_usage_is_accounted(self):
        llm = self._llm()
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(cm_mod, [compat_reply("OCEAN", 40, 9)]):
                llm.chat("s", "u")
        summary = llm.usage_summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["input_tokens"], 40)
        self.assertEqual(summary["output_tokens"], 9)
        self.assertEqual(summary["failures"], 0)

    def test_block_style_content_is_flattened(self):
        llm = self._llm()
        reply = {"choices": [{"message": {"content": [{"text": "OC"},
                                                      {"text": "EAN"}]}}]}
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(cm_mod, [reply]):
                self.assertEqual(llm.chat("s", "u"), "OCEAN")

    def test_transient_error_is_retried_then_succeeds(self):
        llm = self._llm(max_retries=2)
        replies = [_FakeHTTPError(503), _FakeHTTPError(429),
                   compat_reply("OCEAN")]
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(cm_mod, replies) as patched:
                text = llm.chat("s", "u")
        self.assertEqual(text, "OCEAN")
        self.assertEqual(len(patched.requests), 3)
        self.assertEqual(llm.retries, 2)
        self.assertEqual(llm.failures, 0)

    def test_permanent_error_fails_fast(self):
        llm = self._llm(max_retries=2)
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(cm_mod, [_FakeHTTPError(401)]) as patched:
                self.assertIsNone(llm.chat("s", "u"))
        self.assertEqual(len(patched.requests), 1)   # no retry on a bad key
        self.assertEqual(llm.failures, 1)

    def test_retries_are_bounded(self):
        llm = self._llm(max_retries=1)
        replies = [_FakeHTTPError(503), _FakeHTTPError(503),
                   compat_reply("OCEAN")]
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(cm_mod, replies) as patched:
                self.assertIsNone(llm.chat("s", "u"))
        self.assertEqual(len(patched.requests), 2)
        self.assertEqual(llm.failures, 1)

    def test_garbage_body_is_a_clean_failure(self):
        for body in (b"<html>gateway</html>", b"{}", b'{"choices": []}'):
            llm = self._llm()
            with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
                with _PatchedUrlopen(cm_mod, [body]):
                    self.assertIsNone(llm.chat("s", "u"))
            self.assertEqual(llm.failures, 1)
            self.assertEqual(llm.calls, 0)

    def test_missing_key_never_reaches_the_network(self):
        llm = self._llm()
        with _PatchedEnv(OBIRDY_COMPAT_KEY=None, HF_TOKEN=None):
            with _PatchedUrlopen(cm_mod, [compat_reply("OCEAN")]) as patched:
                self.assertIsNone(llm.chat("s", "u"))
        self.assertEqual(patched.requests, [])

    def test_deadline_is_respected_before_the_call(self):
        llm = self._llm()
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(cm_mod, [compat_reply("OCEAN")]) as patched:
                self.assertIsNone(
                    llm.chat("s", "u", deadline=cm_mod._Deadline(0.5)))
        self.assertEqual(patched.requests, [])

    def test_the_guesser_client_behaves_the_same(self):
        llm = self._llm(module=g_mod)
        with _PatchedEnv(OBIRDY_COMPAT_KEY="sk-compat"):
            with _PatchedUrlopen(g_mod, [compat_reply("WHALE")]) as patched:
                self.assertEqual(llm.chat("s", "u"), "WHALE")
        self.assertEqual(patched.requests[0].full_url,
                         "https://router.invalid/v1/chat/completions")


class _FakeAnthropicClient(object):
    """Stands in for the SDK client so no package or key is needed."""

    class _Messages(object):

        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.calls.append(kwargs)
            reply = self.outer.replies.pop(0) if self.outer.replies else None
            if isinstance(reply, Exception):
                raise reply
            return reply

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self._Messages(self)


class _Block(object):

    def __init__(self, text, block_type="text"):
        self.type = block_type
        self.text = text


class _Reply(object):
    """``block_type="thinking"`` reproduces what an SDK too old to disable
    extended thinking gets back: a reasoning block and no text block."""

    def __init__(self, text, input_tokens=5, output_tokens=3,
                 stop_reason="end_turn", block_type="text"):
        self.content = [_Block(text, block_type)]
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": input_tokens,
                                    "output_tokens": output_tokens})()


class TestAnthropicPathUnchanged(unittest.TestCase):
    """The default provider must behave exactly as it did before the switch."""

    def _llm(self, replies, module=cm_mod, **kwargs):
        llm = module._LLM("claude-sonnet-5", **kwargs)
        llm._client = _FakeAnthropicClient(replies)
        return llm

    def test_successful_call_returns_text_and_counts_usage(self):
        llm = self._llm([_Reply("OCEAN", 12, 4)])
        self.assertEqual(llm.chat("system", "user", max_tokens=99), "OCEAN")
        summary = llm.usage_summary()
        self.assertEqual(summary["provider"], cm_mod.PROVIDER_ANTHROPIC)
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["input_tokens"], 12)
        self.assertEqual(summary["output_tokens"], 4)
        sent = llm._client.calls[0]
        self.assertEqual(sent["model"], "claude-sonnet-5")
        self.assertEqual(sent["max_tokens"], 99)
        self.assertEqual(sent["system"], "system")
        self.assertEqual(sent["messages"],
                         [{"role": "user", "content": "user"}])

    def test_unsupported_parameter_falls_back_to_a_simpler_shape(self):
        llm = self._llm([RuntimeError("model does not support thinking"),
                         RuntimeError("model does not support thinking"),
                         _Reply("OCEAN")])
        self.assertEqual(llm.chat("system", "user"), "OCEAN")
        self.assertEqual(len(llm._client.calls), 3)
        self.assertEqual(llm.retries, 0)          # shape probing is not a retry
        self.assertEqual(llm._extra, {})          # cached for later calls

    def test_transient_error_is_retried(self):
        llm = self._llm([RuntimeError("overloaded"), _Reply("OCEAN")],
                        max_retries=2)
        saved = cm_mod.time.sleep
        cm_mod.time.sleep = lambda seconds: None
        try:
            self.assertEqual(llm.chat("system", "user"), "OCEAN")
        finally:
            cm_mod.time.sleep = saved
        self.assertEqual(llm.retries, 1)
        self.assertEqual(llm.failures, 0)

    def test_permanent_error_fails_without_retrying(self):
        llm = self._llm([RuntimeError("invalid request")], max_retries=2)
        self.assertIsNone(llm.chat("system", "user"))
        self.assertEqual(len(llm._client.calls), 1)
        self.assertEqual(llm.failures, 1)

    def test_no_client_means_no_call_and_no_failure_count(self):
        llm = cm_mod._LLM("claude-sonnet-5")
        llm._client_failed = True
        self.assertIsNone(llm.chat("system", "user"))
        self.assertEqual(llm.failures, 0)

    def test_anthropic_path_never_touches_urlopen(self):
        llm = self._llm([_Reply("OCEAN")])
        with _PatchedUrlopen(cm_mod, [compat_reply("NOPE")]) as patched:
            self.assertEqual(llm.chat("system", "user"), "OCEAN")
        self.assertEqual(patched.requests, [])


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

class TestPackaging(unittest.TestCase):
    """The one place a real key is ever written into a file.

    Everything here is about the two ways that can go wrong: substituting into
    the wrong number of places (a keyless zip, or a mangled file), and letting
    the result reach git.
    """

    FAKE_KEY = "sk-ant-api03-FAKEKEYFORTESTSONLY0000000000"

    def setUp(self):
        from harness import package_submission

        self.pkg = package_submission

    def _source(self, name):
        with open(os.path.join(FRAMEWORK_DIR, "players", name)) as handle:
            return handle.read()

    # -- the substitution --------------------------------------------------

    def test_the_placeholder_matches_both_agents(self):
        for module in (cm_mod, g_mod):
            self.assertEqual(module.HARDCODED_API_KEY, self.pkg.PLACEHOLDER)

    def test_it_hits_exactly_one_site_per_file(self):
        for name in self.pkg.AGENT_FILES:
            text, count = self.pkg.substitute_key(self._source(name),
                                                  self.FAKE_KEY)
            self.assertEqual(count, 1, name)
            constants = [line for line in text.splitlines()
                         if line.startswith("HARDCODED_API_KEY")]
            self.assertEqual(constants,
                             ['HARDCODED_API_KEY = "%s"' % self.FAKE_KEY], name)

    def test_the_substituted_file_still_parses_and_reports_the_key(self):
        import ast

        for name in self.pkg.AGENT_FILES:
            text, _ = self.pkg.substitute_key(self._source(name),
                                              self.FAKE_KEY)
            tree = ast.parse(text)          # a mangled file would raise here
            found = [node.value.value for node in ast.walk(tree)
                     if isinstance(node, ast.Assign)
                     and getattr(node.targets[0], "id", None)
                     == "HARDCODED_API_KEY"]
            self.assertEqual(found, [self.FAKE_KEY], name)

    def test_it_changes_exactly_one_line_of_the_file(self):
        """A packaging bug that edits prose would ship a broken agent."""
        for name in self.pkg.AGENT_FILES:
            source = self._source(name)
            text, _ = self.pkg.substitute_key(source, self.FAKE_KEY)
            before = source.splitlines()
            after = text.splitlines()
            self.assertEqual(len(before), len(after), name)
            changed = [i for i, line in enumerate(before)
                       if line != after[i]]
            self.assertEqual(len(changed), 1, name)
            self.assertTrue(before[changed[0]].startswith("HARDCODED_API_KEY"))
            # The prose that explains the constant is untouched.
            self.assertEqual(text.count("HARDCODED_API_KEY"),
                             source.count("HARDCODED_API_KEY"), name)

    def test_a_renamed_constant_is_caught_rather_than_shipped_keyless(self):
        text, count = self.pkg.substitute_key("SOMETHING_ELSE = 'x'\n",
                                              self.FAKE_KEY)
        self.assertEqual(count, 0)
        with self.assertRaises(SystemExit):
            self.pkg.agent_source("codemaster.py", self.FAKE_KEY)

    def test_it_refuses_a_key_that_would_break_the_file(self):
        for bad in ('sk-ant-"quote', "sk-ant-new\nline"):
            with self.assertRaises(SystemExit):
                self.pkg.substitute_key("HARDCODED_API_KEY = \"x\"\n", bad)

    # -- the env file ------------------------------------------------------

    def test_it_reads_the_tournament_entry_and_only_that_one(self):
        import tempfile

        handle, path = tempfile.mkstemp(suffix=".env")
        with os.fdopen(handle, "w") as out:
            out.write("# comment\n"
                      "ANTHROPIC_API_KEY=sk-ant-development\n"
                      "%s=%s\n" % (self.pkg.KEY_VAR, self.FAKE_KEY))
        try:
            values = self.pkg.read_env_file(path)
            self.assertEqual(values[self.pkg.KEY_VAR], self.FAKE_KEY)
            self.assertEqual(self.pkg.resolve_key(path), self.FAKE_KEY)
        finally:
            os.unlink(path)

    def test_a_missing_or_implausible_key_stops_the_build(self):
        import tempfile

        handle, path = tempfile.mkstemp(suffix=".env")
        with os.fdopen(handle, "w") as out:
            out.write("%s=paste-it-here\n" % self.pkg.KEY_VAR)
        saved = os.environ.pop(self.pkg.KEY_VAR, None)
        try:
            with self.assertRaises(SystemExit):
                self.pkg.resolve_key(path)
            with self.assertRaises(SystemExit):
                self.pkg.resolve_key(path + ".missing")
        finally:
            if saved is not None:
                os.environ[self.pkg.KEY_VAR] = saved
            os.unlink(path)

    # -- the zip -----------------------------------------------------------

    def test_a_keyed_zip_must_be_named_so_gitignore_catches_it(self):
        with self.assertRaises(SystemExit):
            self.pkg.build("/tmp/obirdy_submission.zip", self.FAKE_KEY)

    def test_it_builds_a_zip_in_the_layout_the_organisers_asked_for(self):
        import shutil
        import tempfile
        import zipfile

        tmp = tempfile.mkdtemp(prefix="obirdy-package-")
        out = os.path.join(tmp, "obirdy_submission_keyed.zip")
        try:
            self.pkg.build(out, self.FAKE_KEY)
            self.pkg.verify(out, self.FAKE_KEY)
            with zipfile.ZipFile(out) as archive:
                self.assertEqual(sorted(archive.namelist()),
                                 ["INSTRUCTIONS.md",
                                  "codemaster_obirdy.py",
                                  "guesser_obirdy.py",
                                  "oBirdy/obirdy_simtable_v2.bin.gz"])
                for name in self.pkg.AGENT_FILES:
                    source = archive.read(name).decode("utf-8")
                    self.assertEqual(source.count(self.FAKE_KEY), 1, name)
                    self.assertIn('HARDCODED_API_KEY = "%s"' % self.FAKE_KEY,
                                  source)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_placeholder_build_carries_no_key(self):
        import shutil
        import tempfile
        import zipfile

        tmp = tempfile.mkdtemp(prefix="obirdy-package-")
        out = os.path.join(tmp, "obirdy_submission_testing.zip")
        try:
            self.pkg.build(out, None)
            self.pkg.verify(out, None)
            with zipfile.ZipFile(out) as archive:
                for name in self.pkg.AGENT_FILES:
                    source = archive.read(name).decode("utf-8")
                    self.assertIn('HARDCODED_API_KEY = "%s"'
                                  % self.pkg.PLACEHOLDER, source)
                    self.assertNotIn("sk-ant", source)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_keyed_artefact_is_gitignored(self):
        with open(os.path.join(REPO_ROOT, ".gitignore")) as handle:
            patterns = [line.strip() for line in handle]
        self.assertIn("submission/*keyed*", patterns)
        self.assertIn("keyed", os.path.basename(self.pkg.DEFAULT_OUT))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class _AlwaysFailingClient(object):
    """An SDK client whose every call raises the same permanent error."""

    class _Messages(object):

        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.calls.append(kwargs)
            raise self.outer.error

    def __init__(self, error):
        self.error = error
        self.calls = []
        self.messages = self._Messages(self)


class _Captured(object):
    """Capture stdout the way the organisers' harness does."""

    def __enter__(self):
        import io

        self._saved = sys.stdout
        self.buffer = io.StringIO()
        sys.stdout = self.buffer
        return self

    def __exit__(self, *exc):
        sys.stdout = self._saved
        return False

    def text(self):
        return self.buffer.getvalue()

    def lines(self, needle="[oBirdy]"):
        return [line for line in self.text().splitlines() if needle in line]


class TestDiagnostics(unittest.TestCase):
    """Loud by default, silent on request, and never leaking the key.

    The organisers' test run of our zip played a whole game on the
    deterministic offline fallback because the key never reached the process,
    and nothing in the output said so.  Everything here exists to make that
    failure impossible to miss next time.
    """

    def setUp(self):
        self._env = dict((name, os.environ.get(name))
                         for name in (cm_mod.KEY_ENV, cm_mod.QUIET_ENV))
        for name in self._env:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _codemaster(self, **kwargs):
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    def _guesser(self, **kwargs):
        agent = g_mod.AIGuesser("Red", **kwargs)
        agent.set_board(list(BOARD))
        return agent

    # -- the init banner ---------------------------------------------------

    def test_the_codemaster_announces_what_it_loaded(self):
        with _Captured() as out:
            self._codemaster(api_key="sk-ant-test")
        blob = out.text()
        self.assertIn("version=%s" % cm_mod.AGENT_VERSION, blob)
        self.assertIn("model=%s" % cm_mod.DEFAULT_MODEL, blob)
        self.assertIn("api key: found via constructor kwarg", blob)
        self.assertIn("anthropic package:", blob)
        self.assertIn("similarity table:", blob)
        self.assertTrue(all(line.startswith("[oBirdy] ")
                            for line in out.lines()), blob)

    def test_the_guesser_announces_what_it_loaded(self):
        with _Captured() as out:
            self._guesser(api_key="sk-ant-test")
        blob = out.text()
        self.assertIn("version=%s" % g_mod.AGENT_VERSION, blob)
        self.assertIn("model=%s" % g_mod.DEFAULT_MODEL, blob)
        self.assertIn("api key: found via constructor kwarg", blob)
        self.assertIn("anthropic package:", blob)

    def test_a_missing_key_is_a_warning_not_a_note(self):
        for build in (self._codemaster, self._guesser):
            with _Captured() as out:
                build()
            self.assertIn("WARNING: no API key found", out.text())
            self.assertNotIn("api key: found", out.text())

    def test_the_banner_says_where_the_table_came_from(self):
        table = cm_mod._SimTable.load()
        if table is None:
            self.skipTest("no bundled simtable")
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})
        with _Captured() as out:
            self._codemaster()
        self.assertIn("similarity table: loaded from", out.text())
        self.assertIn(cm_mod.SIMTABLE_SUBDIR, out.text())

    def test_a_missing_table_says_where_it_looked(self):
        with _Captured() as out:
            self._codemaster(simtable_path="/nowhere/table.bin.gz")
        self.assertIn("similarity table: NOT FOUND", out.text())
        self.assertIn("/nowhere/table.bin.gz", out.text())

    # -- per turn ----------------------------------------------------------

    def test_the_codemaster_logs_one_line_per_clue(self):
        agent = self._codemaster(api_key="sk-ant-test", max_retries=0)
        # A stand-in client, so the test never leaves the machine.
        agent.llm._client = _AlwaysFailingClient(RuntimeError("nope"))
        with _Captured() as out:
            clue = agent.get_clue()
        lines = [line for line in out.lines() if "clue 1:" in line]
        self.assertEqual(len(lines), 1, out.text())
        self.assertIn("%s %s" % (clue[0], clue[1]), lines[0])
        for field in ("path=", "candidates=", "probe ", "sensor "):
            self.assertIn(field, lines[0])

    def test_the_guesser_logs_its_pick_and_a_confidence_bucket(self):
        agent = self._guesser()
        agent.set_clue("TOOL", 2)
        with _Captured() as out:
            word = agent.get_answer()
        lines = [line for line in out.lines() if "guess 1/2" in line]
        self.assertEqual(len(lines), 1, out.text())
        self.assertIn(word, lines[0])
        self.assertTrue(any(bucket in lines[0]
                            for bucket in ("high", "medium", "low")), lines[0])

    def test_the_confidence_buckets_are_ordered(self):
        self.assertEqual(g_mod._confidence_bucket(0.9), "high")
        self.assertEqual(g_mod._confidence_bucket(0.10), "medium")
        self.assertEqual(g_mod._confidence_bucket(0.0), "low")

    # -- warnings ----------------------------------------------------------

    def test_a_failed_api_call_says_so_instead_of_failing_silently(self):
        for module, build in ((cm_mod, self._codemaster),
                              (g_mod, self._guesser)):
            agent = build(api_key="sk-ant-test", max_retries=0)
            agent.llm._client = _AlwaysFailingClient(
                RuntimeError("invalid request: bad model"))
            with _Captured() as out:
                self.assertIsNone(agent.llm.chat("system", "user"))
            warnings = out.lines("WARNING")
            self.assertEqual(len(warnings), 1, out.text())
            self.assertEqual(
                warnings[0],
                "[oBirdy] WARNING: " + module.FALLBACK_WARNING
                % ("RuntimeError", "invalid request: bad model"))
            self.assertIn("using offline fallback", warnings[0])

    def test_the_warning_names_the_exception_class(self):
        agent = self._codemaster(api_key="sk-ant-test", max_retries=0)
        agent.llm._client = _AlwaysFailingClient(TimeoutError("timed out"))
        with _Captured() as out:
            agent.llm.chat("system", "user")
        self.assertIn("(TimeoutError: timed out)", out.text())

    def test_a_whole_clue_on_the_offline_path_warns(self):
        agent = self._codemaster()          # no key at all
        with _Captured() as out:
            agent.get_clue()
        self.assertIn("WARNING: no LLM available", out.text())
        self.assertIn("path=offline-fallback (no LLM)", out.text())

    def test_a_guess_on_the_offline_path_warns(self):
        agent = self._guesser()
        agent.set_clue("TOOL", 1)
        with _Captured() as out:
            agent.get_answer()
        self.assertIn("WARNING:", out.text())
        self.assertIn("offline similarity fallback", out.text())
        self.assertIn("ranking=offline", out.text())

    def test_an_unavailable_client_warns_once_not_per_call(self):
        for build in (self._codemaster, self._guesser):
            agent = build()
            agent.llm._client_failed = True
            agent.llm.last_error = ("MissingAPIKey", "no API key available")
            with _Captured() as out:
                for _ in range(5):
                    agent.llm.chat("system", "user")
            self.assertEqual(len(out.lines("WARNING")), 1, out.text())

    # -- quiet mode --------------------------------------------------------

    def test_the_quiet_kwarg_silences_everything_but_warnings(self):
        with _Captured() as out:
            agent = self._codemaster(quiet=True, api_key="sk-ant-test",
                                     max_retries=0)
            agent.llm._client = _AlwaysFailingClient(RuntimeError("nope"))
            agent.get_clue()
        blob = out.text()
        self.assertEqual([line for line in out.lines()
                          if "WARNING" not in line], [], blob)
        # ...but the fallbacks still announce themselves.
        self.assertTrue(out.lines("WARNING"), blob)

    def test_the_quiet_env_var_silences_the_guesser_too(self):
        os.environ[g_mod.QUIET_ENV] = "1"
        with _Captured() as out:
            agent = self._guesser(api_key="sk-ant-test")
            agent.set_clue("TOOL", 1)
        self.assertEqual(out.lines(), [], out.text())
        self.assertTrue(agent.quiet)

    def test_quiet_is_off_unless_asked(self):
        self.assertFalse(cm_mod._quiet_default())
        for value in ("1", "true", "YES", "on"):
            os.environ[cm_mod.QUIET_ENV] = value
            self.assertTrue(cm_mod._quiet_default(), value)
            self.assertTrue(g_mod._quiet_default(), value)
        for value in ("", "0", "no", "false"):
            os.environ[cm_mod.QUIET_ENV] = value
            self.assertFalse(cm_mod._quiet_default(), value)

    # -- the key never appears anywhere ------------------------------------

    SECRET = "sk-ant-api03-DEADBEEFdeadbeefSECRETKEYVALUE"

    def test_no_key_material_is_ever_printed(self):
        os.environ[cm_mod.KEY_ENV] = self.SECRET
        saved = (cm_mod.HARDCODED_API_KEY, g_mod.HARDCODED_API_KEY)
        cm_mod.HARDCODED_API_KEY = self.SECRET
        g_mod.HARDCODED_API_KEY = self.SECRET
        try:
            with _Captured() as out:
                codemaster = self._codemaster()
                codemaster.llm._client = _AlwaysFailingClient(
                    RuntimeError("authentication_error: invalid x-api-key %s"
                                 % self.SECRET))
                codemaster.llm.max_retries = 0
                codemaster.get_clue()
                guesser = self._guesser()
                guesser.set_clue("TOOL", 1)
                guesser.get_answer()
            blob = out.text()
        finally:
            cm_mod.HARDCODED_API_KEY, g_mod.HARDCODED_API_KEY = saved

        self.assertTrue(blob.strip(), "the agents printed nothing at all")
        self.assertNotIn(self.SECRET, blob)
        self.assertNotIn(self.SECRET[:12], blob)
        self.assertNotIn("DEADBEEF", blob)
        self.assertNotIn("OBIRDY-KEY-PLACEHOLDER", blob)
        # The source is named; the value is not.
        self.assertIn("api key: found via %s env var" % cm_mod.KEY_ENV, blob)

    def test_an_echoed_key_in_an_error_message_is_redacted(self):
        for module in (cm_mod, g_mod):
            brief = module._brief("invalid x-api-key: %s" % self.SECRET)
            self.assertNotIn(self.SECRET, brief)
            self.assertIn("sk-<redacted>", brief)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

class TestApiKeyResolution(unittest.TestCase):
    """kwarg -> ``ANTHROPIC_API_KEY`` -> the embedded constant.

    The organisers refuse per-team environment variables (two entries both
    wanting ``ANTHROPIC_API_KEY`` would clash), so the tournament key ships
    inside the file.  The env var stays first-of-the-fallbacks because it is
    how we run every offline battery.
    """

    MODULES = (cm_mod, g_mod)

    def setUp(self):
        self._env = dict((m, os.environ.pop(m.KEY_ENV, None))
                         for m in self.MODULES)
        self._constant = dict((m, m.HARDCODED_API_KEY) for m in self.MODULES)

    def tearDown(self):
        for module, value in self._env.items():
            if value is None:
                os.environ.pop(module.KEY_ENV, None)
            else:
                os.environ[module.KEY_ENV] = value
            module.HARDCODED_API_KEY = self._constant[module]

    def test_the_placeholder_is_not_a_key(self):
        for module in self.MODULES:
            self.assertEqual(module.HARDCODED_API_KEY, "OBIRDY-KEY-PLACEHOLDER")
            self.assertFalse(module._key_looks_real(module.HARDCODED_API_KEY))
            self.assertEqual(module._resolve_api_key(), (None, None))

    def test_a_substituted_key_is_used(self):
        for module in self.MODULES:
            module.HARDCODED_API_KEY = "sk-ant-substituted"
            key, source = module._resolve_api_key()
            self.assertEqual(key, "sk-ant-substituted")
            self.assertEqual(source, "embedded key")

    def test_the_env_var_beats_the_embedded_key(self):
        for module in self.MODULES:
            module.HARDCODED_API_KEY = "sk-ant-substituted"
            os.environ[module.KEY_ENV] = "sk-ant-from-env"
            key, source = module._resolve_api_key()
            self.assertEqual(key, "sk-ant-from-env")
            self.assertIn(module.KEY_ENV, source)

    def test_a_kwarg_beats_everything(self):
        for module in self.MODULES:
            module.HARDCODED_API_KEY = "sk-ant-substituted"
            os.environ[module.KEY_ENV] = "sk-ant-from-env"
            self.assertEqual(module._resolve_api_key("sk-ant-explicit"),
                             ("sk-ant-explicit", "constructor kwarg"))

    def test_only_keys_that_look_like_keys_count(self):
        for module in self.MODULES:
            for value in ("", "OBIRDY-KEY-PLACEHOLDER", "<paste key here>",
                          "TODO", "ant-sk-backwards"):
                module.HARDCODED_API_KEY = value
                self.assertEqual(module._resolve_api_key(), (None, None), value)

    def test_the_client_is_built_from_the_resolved_key(self):
        for module in self.MODULES:
            module.HARDCODED_API_KEY = "sk-ant-substituted"
            seen = {}

            class _FakeAnthropicModule(object):
                @staticmethod
                def Anthropic(api_key=None, max_retries=0):
                    seen["key"] = api_key
                    return object()

            saved = sys.modules.get("anthropic")
            sys.modules["anthropic"] = _FakeAnthropicModule
            try:
                self.assertIsNotNone(module._LLM("m")._get_client())
                self.assertEqual(seen["key"], "sk-ant-substituted")
                self.assertIsNotNone(
                    module._LLM("m", api_key="sk-ant-kwarg")._get_client())
                self.assertEqual(seen["key"], "sk-ant-kwarg")
            finally:
                if saved is None:
                    sys.modules.pop("anthropic", None)
                else:
                    sys.modules["anthropic"] = saved

    def test_the_agents_pass_their_kwarg_through(self):
        codemaster = cm_mod.AICodemaster("Red", api_key="sk-ant-cm")
        guesser = g_mod.AIGuesser("Red", api_key="sk-ant-g")
        self.assertEqual(codemaster.llm.api_key, "sk-ant-cm")
        self.assertEqual(guesser.llm.api_key, "sk-ant-g")
        self.assertEqual(codemaster.llm.key_source(), "constructor kwarg")
        self.assertEqual(guesser.llm.key_source(), "constructor kwarg")

    def test_no_key_anywhere_reports_no_source(self):
        for module in self.MODULES:
            self.assertIsNone(module._LLM("m").key_source())


# ---------------------------------------------------------------------------
# The unsimulated path: the safety nets that were not running
# ---------------------------------------------------------------------------

# The organisers' fatal board, transcribed from their log.  KNIFE is the
# assassin; the codemaster is Red.
FATAL_BOARD = [
    "SINK", "POOL", "KNIFE", "ALPS", "CONTRACT",
    "CAR", "PLATE", "TRUNK", "WORM", "RULER",
    "BELT", "CHINA", "PARACHUTE", "FIGHTER", "SPELL",
    "PRESS", "LION", "JAM", "MAPLE", "BEACH",
    "CHICK", "MOUTH", "EMBASSY", "LEMON", "SCHOOL",
]
FATAL_KEY = [
    "Red", "Red", "Assassin", "Blue", "Civilian",
    "Red", "Blue", "Red", "Civilian", "Blue",
    "Red", "Blue", "Red", "Blue", "Civilian",
    "Blue", "Red", "Civilian", "Red", "Blue",
    "Civilian", "Blue", "Civilian", "Red", "Civilian",
]


class TestUnsimulatedPathSafety(unittest.TestCase):
    """``path=unsimulated`` used to take ``simulate[0]`` with nothing checking
    it.  On the organisers' machine that branch ran every turn -- their SDK
    could not return any text -- and it handed out ``TOOL 1`` on a board whose
    assassin is ``KNIFE``.

    The embedding sensor is free and offline: it needs neither the panel nor a
    working API, and it rates ``TOOL`` at 0.341 against ``KNIFE`` and 0.210
    against ``BELT``, the only own word it knows.  That is a flag by a clear
    margin.  It was simply never asked.

    Those two numbers are the exact GloVe cosines; what the table returns is
    the nearest codebook level, and above ~0.30 the Lloyd-max book is coarse on
    purpose (few pairs live out there, and no decision turns on 0.34 versus
    0.35).  The tolerance below is that step, not slack -- the assertion that
    matters is the ordering.
    """

    def setUp(self):
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})
        if cm_mod._SimTable.load() is None:
            self.skipTest("no bundled simtable -- python -m harness.simtable")

    def tearDown(self):
        cm_mod._SIMTABLE_STATE.update({"loaded": False, "table": None,
                                       "path": None})

    def _agent(self, available=True, **kwargs):
        agent = cm_mod.AICodemaster("Red", quiet=True, **kwargs)
        agent.set_game_state(list(FATAL_BOARD), list(FATAL_KEY))
        agent.set_move_history([])
        agent.llm.available = lambda: available
        agent.llm.chat = lambda *args, **kw: None     # API up but answering nothing
        return agent

    def test_the_sensor_sees_the_clue_that_killed_us(self):
        """The verdict itself, straight off the shipped table."""
        table = cm_mod._SimTable.load()
        pulls = table.pulls("TOOL", ["KNIFE", "BELT"])
        self.assertGreater(pulls["KNIFE"], pulls["BELT"])
        self.assertAlmostEqual(pulls["KNIFE"], 0.341, delta=0.02)
        self.assertAlmostEqual(pulls["BELT"], 0.210, delta=0.01)

    def test_tool_is_flagged_and_not_given_on_the_unsimulated_path(self):
        agent = self._agent()
        own, opp, civ, assassin = agent._split_board()
        clue, number = agent._unsimulated_clue(
            [("TOOL", ["BELT"]), ("PLANT", ["CAR", "MAPLE"])],
            own, opp, assassin, cm_mod._Deadline(45.0))
        self.assertIn("TOOL", agent._embed_flagged_clues)
        self.assertEqual(agent.embed_flagged, 1)
        self.assertNotEqual(clue, "TOOL")
        self.assertGreaterEqual(number, 1)

    def test_a_flagged_clue_with_no_alternative_is_capped_at_one_word(self):
        """Somebody has to be given.  The least it can do is ask for one word."""
        agent = self._agent()
        own, opp, civ, assassin = agent._split_board()
        clue, number = agent._unsimulated_clue(
            [("TOOL", ["BELT", "SINK"])], own, opp, assassin,
            cm_mod._Deadline(45.0))
        self.assertEqual([clue, number], ["TOOL", 1])

    def test_a_clean_board_still_takes_the_leading_candidate(self):
        """No flags means no behaviour change from the branch this replaced."""
        agent = self._agent()
        own, opp, civ, assassin = agent._split_board()
        clue, number = agent._unsimulated_clue(
            [("PLANT", ["CAR", "MAPLE"]), ("MEAL", ["POOL"])],
            own, opp, assassin, cm_mod._Deadline(45.0))
        self.assertEqual([clue, number], ["PLANT", 2])
        self.assertEqual(agent.embed_flagged, 0)

    def test_the_path_label_records_that_the_sensor_ran(self):
        agent = self._agent()
        own, opp, civ, assassin = agent._split_board()
        agent._unsimulated_clue([("PLANT", ["CAR"])], own, opp, assassin,
                                cm_mod._Deadline(45.0))
        self.assertTrue(agent._turn_path.startswith(
            "unsimulated (panel returned nothing)"))
        self.assertIn("sensor", agent._turn_path)

    def test_the_offline_branch_keeps_its_label_and_its_warning(self):
        """``path=offline-fallback (no LLM)`` is what the diagnostics assert on."""
        agent = self._agent(available=False)
        own, opp, civ, assassin = agent._split_board()
        with _Captured() as out:
            agent._unsimulated_clue([("PLANT", ["CAR"])], own, opp, assassin,
                                    cm_mod._Deadline(45.0))
        self.assertTrue(agent._turn_path.startswith(
            "offline-fallback (no LLM)"))
        self.assertIn("no LLM available", out.text())

    def test_a_whole_turn_with_a_dead_api_never_gives_tool(self):
        """End to end through ``get_clue`` -- the shape the organisers ran."""
        agent = self._agent()
        clue, number = agent.get_clue()
        self.assertNotEqual(clue, "TOOL")
        self.assertTrue(agent._turn_path.startswith("unsimulated"))

    def _overlap_leader(self, agent):
        """What ``_fallback_clue`` scored by letter overlap alone would pick.

        Reimplemented rather than recorded, so this stays a statement about the
        sensor's effect and not about one board's data.
        """
        best, best_score = None, None
        for clue in cm_mod._FALLBACK_VOCAB:
            if not cm_mod.clue_is_legal(clue, agent.words):
                continue
            total = 0.0
            for word in agent._unrevealed():
                sim = cm_mod._similarity(clue, word)
                kind = agent._colour(word)
                if kind == agent.team:
                    total += sim
                elif kind == "Assassin":
                    total -= 4.0 * sim
                elif kind == agent.opponent:
                    total -= 0.7 * sim
                else:
                    total -= 0.3 * sim
            if best_score is None or total > best_score:
                best, best_score = clue, total
        return best

    def test_the_fallback_is_unchanged_when_the_sensor_is_clean(self):
        agent = self._agent(available=False)
        self.assertEqual(agent._fallback_clue()[0],
                         self._overlap_leader(agent))

    def test_the_deterministic_fallback_skips_a_clue_the_sensor_flags(self):
        """``_fallback_clue`` scored by letter overlap alone, which is blind to
        meaning -- that is how the recorded API-down game gave six ``TOOL``s."""
        agent = self._agent(available=False)
        leader = self._overlap_leader(agent)
        real = agent._embedding_penalty

        def flag_the_leader(clue, *args, **kwargs):
            if clue == leader:
                agent._embed_flagged_clues.add(clue)
                return agent.embed_penalty
            return real(clue, *args, **kwargs)

        agent._embedding_penalty = flag_the_leader
        clue, number = agent._fallback_clue()
        self.assertNotEqual(clue, leader)
        self.assertIn(clue, cm_mod._FALLBACK_VOCAB)

    def test_the_fallback_still_answers_when_every_candidate_is_flagged(self):
        agent = self._agent(available=False)
        leader = self._overlap_leader(agent)
        agent._embedding_penalty = lambda *args, **kwargs: agent.embed_penalty
        self.assertEqual(agent._fallback_clue(), [leader, 1])


class TestThemedBoardPacing(unittest.TestCase):
    """Why the 18 consecutive number-1 clues were not about the board.

    The organisers saw them on a fully-themed board and the natural reading is
    that our panel majority rule cannot fire when every word is near every
    other one.  But the same log shows ``candidates=3`` on nearly every turn --
    a dead brainstorm -- and with the brainstorm dead the only candidates left
    are ``_heuristic_candidates``, which claim no targets at all.  The no-panel
    number is then ``min(2, len(targets) or 1, ...)``, which is 1 by
    construction, on any board whatsoever.
    """

    def _degraded(self, words, key):
        agent = cm_mod.AICodemaster("Red", quiet=True, api_key="sk-ant-test")
        agent.set_game_state(list(words), list(key))
        agent.set_move_history([])
        agent.llm.available = lambda: True
        agent.llm.chat = lambda *args, **kwargs: None
        return agent

    def _play(self, agent, turns=6):
        numbers = []
        for _ in range(turns):
            numbers.append(agent.get_clue()[1])
            for index, word in enumerate(agent.words):
                if (agent.maps[index] == "Red"
                        and not cm_mod._is_revealed(word)):
                    agent.words[index] = "*%s*" % word.lower()
                    break
        return numbers

    def test_a_dead_brainstorm_gives_number_one_on_an_ordinary_board(self):
        numbers = self._play(self._degraded(BOARD, KEY))
        self.assertEqual(set(numbers), {1}, numbers)

    def test_and_the_same_on_the_themed_board(self):
        from harness.secret_pool import generate_board

        spec = generate_board("organiser", 1)
        numbers = self._play(self._degraded(spec["words"], spec["key_grid"]))
        self.assertEqual(set(numbers), {1}, numbers)

    def test_a_live_panel_still_pages_past_one_on_a_themed_board(self):
        """The rule itself is fine: give it rankings and it counts them."""
        from harness.secret_pool import generate_board

        spec = generate_board("organiser", 1)
        agent = cm_mod.AICodemaster("Red", quiet=True, api_key="sk-ant-test")
        agent.set_game_state(list(spec["words"]), list(spec["key_grid"]))
        own = agent._split_board()[0]
        rankings = [list(own[:3]) + ["JAM"], list(own[:3]) + ["JAM"]]
        _, number = agent._score_candidate(rankings, own, [], [], ["QUEST"],
                                           claimed=3)
        self.assertEqual(number, 3)


# ---------------------------------------------------------------------------
# Old-SDK compatibility
# ---------------------------------------------------------------------------

class _OldSDKClient(object):
    """A client whose ``create`` refuses named parameters, like 0.29.0 does.

    ``anthropic==0.29.0`` types ``Messages.create`` explicitly and has no
    ``thinking`` parameter, so passing one never reaches the network -- Python
    raises ``TypeError`` first.  It does accept ``extra_body``, which it copies
    into the request JSON verbatim.  That is exactly what this stands in for.
    """

    class _Messages(object):

        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            for name in ("thinking", "output_config"):
                if name in kwargs:
                    raise TypeError(
                        "Messages.create() got an unexpected keyword "
                        "argument '%s'" % name)
            self.outer.calls.append(kwargs)
            body = kwargs.get("extra_body") or {}
            if body.get("thinking", {}).get("type") == "disabled":
                return _Reply("OCEAN")
            # Thinking left on: the whole budget goes to a reasoning block and
            # no text block comes back at all.
            return _Reply("", output_tokens=kwargs["max_tokens"],
                          stop_reason="max_tokens", block_type="thinking")

    def __init__(self):
        self.calls = []
        self.messages = self._Messages(self)


class TestOldSDKCompatibility(unittest.TestCase):
    """The organisers' machine ran ``anthropic==0.29.0`` and our agent silently
    became its own offline fallback.  These are the two defences."""

    MODULES = (cm_mod, g_mod)

    # -- version floor -----------------------------------------------------

    def test_version_parsing_stops_at_the_first_non_numeric_part(self):
        for module in self.MODULES:
            self.assertEqual(module._version_tuple("0.29.0"), (0, 29, 0))
            self.assertEqual(module._version_tuple("0.60.0"), (0, 60, 0))
            self.assertEqual(module._version_tuple("1.2.3rc1"), (1, 2, 3))
            self.assertEqual(module._version_tuple("unknown"), ())

    def test_the_floor_is_what_the_organisers_ran_plus_headroom(self):
        for module in self.MODULES:
            self.assertLess(module._version_tuple("0.29.0"),
                            module.MIN_ANTHROPIC)
            self.assertGreaterEqual(module._version_tuple("0.120.2"),
                                    module.MIN_ANTHROPIC)

    def test_an_old_sdk_warns_at_startup(self):
        for module in self.MODULES:
            saved = module._anthropic_version
            module._anthropic_version = lambda: ("0.29.0", True)
            try:
                with _Captured() as out:
                    self._build(module)
            finally:
                module._anthropic_version = saved
            warnings = [line for line in out.lines("WARNING")
                        if "older than the tested minimum" in line]
            self.assertEqual(len(warnings), 1, out.text())
            self.assertEqual(
                warnings[0],
                "[oBirdy] WARNING: " + module.OLD_SDK_WARNING
                % ("0.29.0", module.MIN_ANTHROPIC_TEXT))
            self.assertIn("pip install -U anthropic", warnings[0])
            self.assertIn("continuing with compatibility mode", warnings[0])

    def test_a_current_sdk_says_nothing_extra(self):
        for module in self.MODULES:
            saved = module._anthropic_version
            module._anthropic_version = lambda: ("0.120.2", False)
            try:
                with _Captured() as out:
                    self._build(module)
            finally:
                module._anthropic_version = saved
            self.assertNotIn("older than the tested minimum", out.text())

    def _build(self, module):
        if module is cm_mod:
            return module.AICodemaster("Red", api_key="sk-ant-test")
        return module.AIGuesser("Red", api_key="sk-ant-test")

    # -- the compatibility route -------------------------------------------

    def _llm(self, module, client):
        llm = module._LLM("claude-sonnet-5", max_retries=0)
        llm._client = client
        return llm

    def test_named_parameters_the_sdk_lacks_are_re_sent_in_extra_body(self):
        """The whole regression in one assertion.

        Dropping the field (what the shape probe used to do) leaves extended
        thinking on, and the model then spends every token of the budget on a
        reasoning block.  Routing it through ``extra_body`` asks the server for
        the same thing the modern SDK asks for.
        """
        for module in self.MODULES:
            client = _OldSDKClient()
            llm = self._llm(module, client)
            self.assertEqual(llm.chat("system", "user", max_tokens=900),
                             "OCEAN")
            self.assertTrue(llm._extra_body)
            # The most featureful shape survives: nothing was given up.
            self.assertEqual(llm._extra, module._LLM._param_shapes()[0])
            self.assertEqual(len(client.calls), 1)
            sent = client.calls[0]["extra_body"]
            self.assertEqual(sent["thinking"], {"type": "disabled"})
            self.assertEqual(sent["output_config"], {"effort": "low"})
            self.assertNotIn("thinking", client.calls[0])

    def test_the_route_is_remembered_for_later_calls(self):
        for module in self.MODULES:
            client = _OldSDKClient()
            llm = self._llm(module, client)
            llm.chat("system", "user")
            llm.chat("system", "user")
            self.assertEqual(len(client.calls), 2)
            for call in client.calls:
                self.assertIn("extra_body", call)

    def test_a_modern_sdk_never_uses_extra_body(self):
        for module in self.MODULES:
            llm = module._LLM("claude-sonnet-5")
            llm._client = _FakeAnthropicClient([_Reply("OCEAN")])
            self.assertEqual(llm.chat("system", "user"), "OCEAN")
            self.assertFalse(llm._extra_body)
            self.assertNotIn("extra_body", llm._client.calls[0])
            self.assertEqual(llm._client.calls[0]["thinking"],
                             {"type": "disabled"})

    def test_a_server_refusal_still_drops_the_field(self):
        """Client-side gap and server-side refusal are different diagnoses.

        Only the first is worth re-sending; the second means the model really
        will not take the field, and the simpler shape is the right answer.
        """
        for module in self.MODULES:
            llm = module._LLM("claude-sonnet-5", max_retries=0)
            llm._client = _FakeAnthropicClient(
                [RuntimeError("model does not support thinking"),
                 RuntimeError("model does not support thinking"),
                 _Reply("OCEAN")])
            self.assertEqual(llm.chat("system", "user"), "OCEAN")
            self.assertFalse(llm._extra_body)
            self.assertEqual(llm._extra, {})

    # -- empty replies are failures, not answers ---------------------------

    def test_a_reply_with_no_text_is_a_failure_not_an_empty_string(self):
        for module in self.MODULES:
            llm = module._LLM("claude-sonnet-5", max_retries=0)
            llm._client = _FakeAnthropicClient(
                [_Reply("", stop_reason="max_tokens", block_type="thinking")])
            self.assertIsNone(llm.chat("system", "user", max_tokens=900))
            self.assertEqual(llm.last_error[0], "EmptyResponse")
            self.assertIn("stop_reason=max_tokens", llm.last_error[1])
            self.assertEqual(llm.failures, 1)

    def test_an_exhausted_budget_is_retried_wider(self):
        for module in self.MODULES:
            llm = module._LLM("claude-sonnet-5", max_retries=1)
            llm._client = _FakeAnthropicClient(
                [_Reply("", stop_reason="max_tokens", block_type="thinking"),
                 _Reply("OCEAN")])
            saved = module.time.sleep
            module.time.sleep = lambda seconds: None
            try:
                self.assertEqual(llm.chat("system", "user", max_tokens=900),
                                 "OCEAN")
            finally:
                module.time.sleep = saved
            self.assertEqual(llm.retries, 1)
            self.assertEqual(llm._client.calls[0]["max_tokens"], 900)
            self.assertEqual(llm._client.calls[1]["max_tokens"], 3600)
            self.assertLessEqual(llm._token_scale, module.MAX_TOKEN_SCALE)

    def test_an_empty_reply_that_did_not_run_out_of_budget_is_not_retried(self):
        for module in self.MODULES:
            llm = module._LLM("claude-sonnet-5", max_retries=2)
            llm._client = _FakeAnthropicClient(
                [_Reply("", stop_reason="end_turn"), _Reply("OCEAN")])
            self.assertIsNone(llm.chat("system", "user"))
            self.assertEqual(len(llm._client.calls), 1)
            self.assertEqual(llm._token_scale, 1)


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------

class TestSelfContained(unittest.TestCase):

    FILES = ("codemaster_obirdy.py", "guesser_obirdy.py")

    def test_no_cross_imports(self):
        """Submission files may only import stdlib, anthropic, and the ABCs."""
        import ast

        allowed = {
            # ``gzip`` reads the optional bundled similarity table; it is
            # stdlib, so the file stays self-contained in code.
            "gzip",
            "json", "os", "random", "re", "sys", "threading", "time",
            "urllib.request",
            "anthropic", "concurrent.futures",
            "players.codemaster", "players.guesser",
        }
        for name in self.FILES:
            path = os.path.join(FRAMEWORK_DIR, "players", name)
            with open(path, "r") as handle:
                tree = ast.parse(handle.read())
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.add(node.module or "")
            self.assertTrue(modules <= allowed,
                            "%s imports %s" % (name, sorted(modules - allowed)))

    def test_no_hardcoded_key(self):
        for name in self.FILES:
            path = os.path.join(FRAMEWORK_DIR, "players", name)
            with open(path, "r") as handle:
                source = handle.read()
            self.assertNotIn("sk-ant", source, name)


# ---------------------------------------------------------------------------
# Bounding a move: per-attempt timeouts and the hard wall
# ---------------------------------------------------------------------------

class _WedgedChat(object):
    """A ``chat`` that blocks the way a socket read blocks: uninterruptibly.

    This is the shape the 2026-08-03 battery hit.  Four ``get_clue`` calls came
    back at 851-855 s against a 45 s budget with no failure and no warning,
    because the machine went to sleep mid-request for 843 s (``pmset``:
    ``Sleep ... 'Clamshell Sleep' ... 843 secs``) and the wall clock the arena
    measures kept running while every cooperative deadline check was suspended
    with the process.  Sleep is only one way to get there -- a wedged proxy or a
    black-holed connection does the same thing to a live machine -- and in every
    version of it the deadline is consulted correctly and blown anyway, because
    nothing consults it while the thread is inside the socket.
    """

    def __init__(self, seconds=5.0):
        self.seconds = seconds
        self.entered = 0

    def __call__(self, system, user):
        self.entered += 1
        if self.entered == 1:
            time.sleep(self.seconds)
        return None


class TestMoveIsBounded(unittest.TestCase):
    """No move may outlast its wall, whatever the network is doing."""

    WALL = 0.6

    def setUp(self):
        self._env = {}
        for name in ("ANTHROPIC_API_KEY", "OBIRDY_COMPAT_KEY", "HF_TOKEN"):
            self._env[name] = os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._env.items():
            if value is not None:
                os.environ[name] = value

    def _codemaster(self, **kwargs):
        kwargs.setdefault("move_wall", self.WALL)
        kwargs.setdefault("quiet", True)
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    def _guesser(self, **kwargs):
        kwargs.setdefault("move_wall", self.WALL)
        kwargs.setdefault("quiet", True)
        agent = g_mod.AIGuesser("Red", **kwargs)
        agent.set_board(list(BOARD))
        agent.set_clue("OCEAN", 2)
        return agent

    # -- the wall ----------------------------------------------------------

    def test_a_wedged_client_still_returns_a_legal_clue_inside_the_wall(self):
        agent = self._codemaster()
        wedged = _WedgedChat(5.0)
        started = time.time()
        with _PatchedLLM(cm_mod, wedged):
            clue, number = agent.get_clue()
        elapsed = time.time() - started
        self.assertLess(elapsed, self.WALL + 2.0,
                        "get_clue took %.2f s against a %.2f s wall"
                        % (elapsed, self.WALL))
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), clue)
        self.assertGreaterEqual(number, 1)
        self.assertLessEqual(number, 9)
        self.assertEqual(agent._turn_path, "wall-fallback")

    def test_a_wedged_client_still_returns_a_legal_guess_inside_the_wall(self):
        agent = self._guesser()
        started = time.time()
        with _PatchedLLM(g_mod, _WedgedChat(5.0)):
            word = agent.get_answer()
        elapsed = time.time() - started
        self.assertLess(elapsed, self.WALL + 2.0,
                        "get_answer took %.2f s" % elapsed)
        self.assertIn(word, BOARD)
        self.assertEqual(agent._pending, word)

    def test_the_wall_announces_itself_even_in_quiet_mode(self):
        agent = self._codemaster()
        with _Captured() as out:
            with _PatchedLLM(cm_mod, _WedgedChat(5.0)):
                agent.get_clue()
        warnings = [line for line in out.lines("WARNING")
                    if "hard wall" in line]
        self.assertEqual(len(warnings), 1, out.text())
        self.assertIn("offline fallback", warnings[0])

    def test_an_abandoned_guess_pipeline_cannot_claim_a_pending_word(self):
        """The ghost finishes eventually.  Its writes must not land.

        ``_pending`` hides a word from ``_options`` because we have just
        guessed it.  A pipeline abandoned at the wall is still running, and if
        it were allowed to set ``_pending`` on its way out it would hide a word
        nobody guessed -- the staleness bug this attribute exists to fix,
        arriving from the other side.
        """
        agent = self._guesser()
        wedged = _WedgedChat(1.0)
        with _PatchedLLM(g_mod, wedged):
            word = agent.get_answer()
            generation = agent._answer_gen
            # Let the abandoned worker run to completion.
            time.sleep(1.6)
        self.assertEqual(agent._answer_gen, generation)
        self.assertEqual(agent._pending, word,
                         "the abandoned pipeline overwrote _pending")

    def test_a_healthy_pipeline_is_unaffected_by_the_wall(self):
        def responder(system, user):
            if "spymaster" in system:
                return ('[{"clue":"OCEAN","targets":["WHALE","SHIP"]},'
                        ' {"clue":"ROYAL","targets":["KING","CROWN"]}]')
            return '{"OCEAN": ["WHALE", "SHIP"], "ROYAL": ["KING"]}'

        with _PatchedLLM(cm_mod, responder):
            walled = self._codemaster().get_clue()
            # ``move_wall=0`` runs the pipeline on the calling thread.
            direct = self._codemaster(move_wall=0).get_clue()
        self.assertEqual(walled, direct)

    def test_the_wall_sits_between_the_budget_and_the_event_limit(self):
        for module in (cm_mod, g_mod):
            self.assertGreater(module.MOVE_WALL_S, module.MOVE_DEADLINE_S)
            self.assertLess(module.MOVE_WALL_S, 60.0)

    # -- _run_parallel -----------------------------------------------------

    def test_run_parallel_abandons_a_wedged_job_instead_of_joining_it(self):
        """The bug the context manager was hiding.

        ``future.result(timeout=...)`` honoured the deadline and then
        ``with ThreadPoolExecutor(...)`` exited through ``shutdown(wait=True)``
        and joined the wedged worker anyway, so the timeout bought nothing.
        """
        for module in (cm_mod, g_mod):
            jobs = [lambda: time.sleep(4.0),
                    lambda: "quick",
                    lambda: "also quick"]
            started = time.time()
            results = module._run_parallel(jobs, module._Deadline(0.3),
                                           max_workers=3)
            elapsed = time.time() - started
            self.assertLess(elapsed, 2.0,
                            "%s._run_parallel joined a wedged job (%.2f s)"
                            % (module.__name__, elapsed))
            self.assertIsNone(results[0])
            self.assertEqual(results[1:], ["quick", "also quick"])

    # -- the deadline itself -----------------------------------------------

    def test_the_deadline_counts_whichever_clock_ran_further(self):
        """Wall clock counts a suspended machine; monotonic cannot be rewound.

        Taking the larger elapsed of the two means neither a sleeping laptop
        nor a backwards NTP step can hide time from the budget.
        """
        for module in (cm_mod, g_mod):
            deadline = module._Deadline(45.0)
            self.assertGreater(deadline.remaining(), 44.0)

            # The wall clock jumped forward (the machine slept).
            slept = module._Deadline(45.0)
            slept.start -= 100.0
            self.assertLess(slept.remaining(), 0.0)
            self.assertTrue(slept.expired())

            # The wall clock jumped backwards; monotonic still counts.
            stepped = module._Deadline(45.0)
            stepped.start += 3600.0
            stepped._mono_start -= 100.0
            self.assertLess(stepped.remaining(), 0.0)


class TestEveryAttemptHasATimeout(unittest.TestCase):
    """No API attempt may inherit the SDK's own 600 s default."""

    MODULES = (cm_mod, g_mod)

    def _llm(self, module, client, **kwargs):
        llm = module._LLM("claude-sonnet-5", max_retries=0, **kwargs)
        llm._client = client
        return llm

    def test_a_modern_attempt_carries_a_per_attempt_timeout(self):
        for module in self.MODULES:
            llm = self._llm(module, _FakeAnthropicClient([_Reply("OCEAN")]))
            self.assertEqual(llm.chat("system", "user"), "OCEAN")
            sent = llm._client.calls[0]
            self.assertIn("timeout", sent)
            self.assertLessEqual(sent["timeout"], module.CALL_TIMEOUT_S)
            self.assertGreater(sent["timeout"], 0.0)

    def test_the_extra_body_route_carries_one_too(self):
        """The compatibility path re-sends the request; it must not lose the
        timeout on the way."""
        for module in self.MODULES:
            client = _OldSDKClient()
            llm = self._llm(module, client)
            self.assertEqual(llm.chat("system", "user", max_tokens=900),
                             "OCEAN")
            self.assertTrue(llm._extra_body)
            for call in client.calls:
                self.assertIn("timeout", call)
                self.assertLessEqual(call["timeout"], module.CALL_TIMEOUT_S)

    def test_a_deadline_shrinks_the_per_attempt_timeout(self):
        for module in self.MODULES:
            llm = self._llm(module, _FakeAnthropicClient([_Reply("OCEAN")]))
            llm.chat("system", "user", deadline=module._Deadline(8.0))
            self.assertLess(llm._client.calls[0]["timeout"],
                            module.CALL_TIMEOUT_S)

    def test_the_client_itself_defaults_to_our_timeout_not_the_sdks(self):
        for module in self.MODULES:
            seen = {}

            class _FakeAnthropicModule(object):
                @staticmethod
                def Anthropic(api_key=None, max_retries=0, timeout=None):
                    seen["timeout"] = timeout
                    seen["max_retries"] = max_retries
                    return object()

            saved = sys.modules.get("anthropic")
            sys.modules["anthropic"] = _FakeAnthropicModule
            try:
                llm = module._LLM("m", api_key="sk-ant-test")
                self.assertIsNotNone(llm._get_client())
            finally:
                if saved is None:
                    sys.modules.pop("anthropic", None)
                else:
                    sys.modules["anthropic"] = saved
            self.assertEqual(seen["timeout"], module.CALL_TIMEOUT_S)
            self.assertEqual(seen["max_retries"], 0)

    def test_an_sdk_too_old_for_the_constructor_keyword_still_builds(self):
        for module in self.MODULES:
            class _FakeAnthropicModule(object):
                @staticmethod
                def Anthropic(api_key=None, max_retries=0):
                    return "client"

            saved = sys.modules.get("anthropic")
            sys.modules["anthropic"] = _FakeAnthropicModule
            try:
                llm = module._LLM("m", api_key="sk-ant-test")
                self.assertEqual(llm._get_client(), "client")
            finally:
                if saved is None:
                    sys.modules.pop("anthropic", None)
                else:
                    sys.modules["anthropic"] = saved

    def test_the_compat_provider_carries_a_timeout(self):
        for module in self.MODULES:
            seen = {}

            class _Handle(object):
                @staticmethod
                def read():
                    return b'{"choices":[{"message":{"content":"OCEAN"}}]}'

                @staticmethod
                def close():
                    return None

            def fake_urlopen(request, timeout=None):
                seen["timeout"] = timeout
                return _Handle()

            os.environ["OBIRDY_COMPAT_KEY"] = "compat-token"
            saved = module.urllib.request.urlopen
            module.urllib.request.urlopen = fake_urlopen
            try:
                llm = module._LLM("m", max_retries=0,
                                  provider=module.PROVIDER_COMPAT)
                self.assertEqual(llm.chat("system", "user"), "OCEAN")
            finally:
                module.urllib.request.urlopen = saved
                os.environ.pop("OBIRDY_COMPAT_KEY", None)
            self.assertIsNotNone(seen["timeout"])
            self.assertLessEqual(seen["timeout"], module.CALL_TIMEOUT_S)


# ---------------------------------------------------------------------------
# The probe's number cap
# ---------------------------------------------------------------------------

class TestProbeNumberCap(unittest.TestCase):
    """A clue the probe kept, but rated dangerous, buys one word.

    Both assassin deaths in the 2026-08-03 battery are this shape: the probe
    rated the assassin exactly 4.0, the veto did not fire because our own
    marginal target was rated higher, and the assassin was taken as guess 2 of
    a 2.
    """

    def _agent(self, **kwargs):
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    def test_the_cap_sits_below_the_veto(self):
        self.assertLess(cm_mod.PROBE_NUMBER_CAP_SCORE,
                        cm_mod.PROBE_VETO_SCORE)

    def test_a_clue_kept_at_the_cap_asks_for_one_word(self):
        agent = self._agent()
        # Assassin 4, our marginal target 6: no absolute veto (needs 5), no
        # relative veto (4 < 6).  This is PLATE, and this is NPC.
        score, number = agent._apply_probe(
            5.0, 2, 2, {"POISON": 4.0, "WHALE": 6.0, "SHIP": 8.0},
            ["APPLE"], ["POISON"], clue="PLATE")
        self.assertFalse(agent.probe_log[-1]["veto"])
        self.assertGreater(score, -40.0)
        self.assertEqual(number, 1)

    def test_below_the_cap_the_number_survives(self):
        agent = self._agent()
        _, number = agent._apply_probe(
            5.0, 3, 3, {"POISON": 3.0, "WHALE": 6.0, "SHIP": 8.0},
            ["APPLE"], ["POISON"], clue="OCEAN")
        self.assertEqual(number, 3)
        self.assertEqual(agent.probe_number_capped, 0)

    def test_the_cap_is_logged_and_counted(self):
        agent = self._agent()
        agent._apply_probe(5.0, 2, 2, {"POISON": 4.0, "WHALE": 6.0},
                           ["APPLE"], ["POISON"], clue="PLATE")
        self.assertTrue(agent.probe_log[-1]["number_capped"])
        self.assertEqual(agent.probe_number_capped, 1)
        self.assertEqual(agent.usage_summary()["probe_number_capped"], 1)

    def test_the_cap_never_raises_a_number(self):
        agent = self._agent()
        _, number = agent._apply_probe(
            5.0, 1, 1, {"POISON": 9.0, "WHALE": 9.0},
            ["APPLE"], ["POISON"], clue="OCEAN")
        self.assertEqual(number, 1)

    def test_both_recorded_deaths_are_capped(self):
        """Replayed from the ratings ``probe_log`` actually recorded.

        ``final_a_default`` seed 8, ``PLATE 2`` into ``WASHER``:
        ``{"assassin": 4.0, "margin": 6.0}``.
        ``final_b_organiser`` seed 3, ``NPC 2`` into ``QUEST``:
        ``{"assassin": 4.0, "margin": 5.0}``.
        """
        for clue, margin in (("PLATE", 6.0), ("NPC", 5.0)):
            agent = self._agent()
            _, number = agent._apply_probe(
                5.0, 2, 2, {"POISON": 4.0, "WHALE": margin, "SHIP": 9.0},
                ["APPLE"], ["POISON"], clue=clue)
            self.assertFalse(agent.probe_log[-1]["veto"],
                             "%s was not vetoed and must not be" % clue)
            self.assertEqual(number, 1, clue)

    def test_a_probe_that_did_not_answer_is_untouched_by_the_cap(self):
        """No ratings means the conservative number, not the cap."""
        agent = self._agent()
        score, number = agent._apply_probe(5.0, 2, 2, None, ["APPLE"],
                                           ["POISON"], clue="OCEAN")
        self.assertEqual(agent.probe_number_capped, 0)
        self.assertEqual(number, 2)
        self.assertLess(score, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
