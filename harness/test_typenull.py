"""Offline unit tests for the TYPE: NULL agents.

No API key, no network: every LLM call is monkeypatched.  Run with

    python -m pytest harness/test_typenull.py -q

The agents under test are ``framework/players/codemaster_typenull.py`` and
``framework/players/guesser_typenull.py``; nothing else in the repo is touched
by this file.
"""

import json
import os
import sys
import time
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRAMEWORK = os.path.join(_REPO, "framework")
if _FRAMEWORK not in sys.path:
    sys.path.insert(0, _FRAMEWORK)

from players import codemaster_typenull as cm_mod        # noqa: E402
from players import guesser_typenull as g_mod            # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BOARD = [
    "APPLE", "TREE", "ORCHARD", "SEED", "BRANCH",       # red-ish
    "ROCKET", "MOON", "STAR", "COMET",                  # red-ish
    "BANK", "RIVER", "BRIDGE", "TOWER", "CASTLE",       # blue
    "KING", "QUEEN", "KNIGHT",
    "SOCK", "LAMP", "CHAIR", "SPOON", "PAPER", "GLOVE", "CLOUD",   # civilian
    "DEATH",                                            # assassin
]
KEY = (["Red"] * 9 + ["Blue"] * 8 + ["Civilian"] * 7 + ["Assassin"])
assert len(BOARD) == 25 and len(KEY) == 25


def make_codemaster(reply=None, replies=None, **kwargs):
    """A codemaster whose single LLM call returns scripted text.

    ``move_wall=0`` runs the pipeline on the calling thread so a failure shows
    up as a real traceback rather than a wall timeout.
    """
    kwargs.setdefault("move_wall", 0)
    kwargs.setdefault("quiet", True)
    agent = cm_mod.AICodemaster("Red", **kwargs)
    queue = list(replies or ([reply] if reply is not None else []))
    agent.prompts = []

    def fake_chat(system, user, max_tokens=900, deadline=None):
        agent.prompts.append(user)
        agent.llm.calls += 1
        return queue.pop(0) if queue else None

    agent.llm.chat = fake_chat
    agent.llm.available = lambda: True
    agent.set_game_state(list(BOARD), list(KEY))
    return agent


def make_guesser(reply=None, replies=None, **kwargs):
    kwargs.setdefault("move_wall", 0)
    kwargs.setdefault("quiet", True)
    agent = g_mod.AIGuesser("Red", **kwargs)
    queue = list(replies or ([reply] if reply is not None else []))
    agent.prompts = []

    def fake_chat(system, user, max_tokens=1100, deadline=None):
        agent.prompts.append(user)
        agent.llm.calls += 1
        return queue.pop(0) if queue else None

    agent.llm.chat = fake_chat
    agent.llm.available = lambda: True
    agent.set_board(list(BOARD))
    return agent


def clue_reply(clue, targets, ranking, number, avoid=()):
    return ("Some prose reasoning with no braces.\n" + json.dumps({
        "clue": clue, "targets": list(targets), "ranking": list(ranking),
        "avoid": list(avoid), "number": number}))


def score_reply(scores, avoid=()):
    return ("Thinking about the clue first.\n"
            + json.dumps({"scores": scores, "avoid": list(avoid)}))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestLegality(unittest.TestCase):

    def test_single_alphabetic_word_only(self):
        self.assertTrue(cm_mod.clue_is_legal("ORBIT", BOARD))
        self.assertFalse(cm_mod.clue_is_legal("TWO WORDS", BOARD))
        self.assertFalse(cm_mod.clue_is_legal("HYPHEN-ATED", BOARD))
        self.assertFalse(cm_mod.clue_is_legal("N1NE", BOARD))
        self.assertFalse(cm_mod.clue_is_legal("", BOARD))
        self.assertFalse(cm_mod.clue_is_legal(None, BOARD))

    def test_minimum_length(self):
        self.assertFalse(cm_mod.clue_is_legal("AB", BOARD))
        self.assertTrue(cm_mod.clue_is_legal("ABC", BOARD))

    def test_no_board_word_derivation_in_either_direction(self):
        self.assertFalse(cm_mod.clue_is_legal("APPLES", BOARD))    # contains
        self.assertFalse(cm_mod.clue_is_legal("APPLE", BOARD))     # equal
        self.assertFalse(cm_mod.clue_is_legal("MOONLIGHT", BOARD))
        self.assertFalse(cm_mod.clue_is_legal("OON", BOARD))       # contained

    def test_revealed_words_do_not_constrain(self):
        board = ["*Red*", "MOON"]
        self.assertTrue(cm_mod.clue_is_legal("LUNAR", board))
        self.assertFalse(cm_mod.clue_is_legal("MOONS", board))

    def test_guesser_and_codemaster_agree_on_normalisation(self):
        self.assertEqual(cm_mod._normalise(" ice-cream "), "ICECREAM")
        self.assertEqual(g_mod._normalise(" ice-cream "), "ICECREAM")


class TestReplyParsing(unittest.TestCase):

    def test_clue_json_after_prose(self):
        parsed = cm_mod.parse_clue_reply(
            clue_reply("FRUIT", ["APPLE"], ["APPLE", "TREE"], 1))
        self.assertEqual(parsed["clue"], "FRUIT")
        self.assertEqual(parsed["targets"], ["APPLE"])
        self.assertEqual(parsed["number"], 1)

    def test_last_object_wins_over_a_braced_aside(self):
        text = ('I considered {"clue": "WRONG"} first.\n'
                + clue_reply("RIGHT", ["MOON"], ["MOON"], 1))
        self.assertEqual(cm_mod.parse_clue_reply(text)["clue"], "RIGHT")

    def test_unparseable_reply_is_none(self):
        self.assertIsNone(cm_mod.parse_clue_reply("no json here at all"))
        self.assertIsNone(cm_mod.parse_clue_reply(""))
        self.assertIsNone(cm_mod.parse_clue_reply(None))

    def test_missing_number_falls_back_to_target_count(self):
        parsed = cm_mod.parse_clue_reply(
            '{"clue": "SKY", "targets": ["MOON", "STAR"], "ranking": []}')
        self.assertEqual(parsed["number"], 2)

    def test_guess_scores_object(self):
        scores, avoid = g_mod.parse_guess_reply(
            score_reply({"MOON": 90, "STAR": 70}, ["DEATH"]),
            ["MOON", "STAR", "DEATH"])
        self.assertEqual(scores["MOON"], 90.0)
        self.assertIn("DEATH", avoid)

    def test_guess_falls_back_to_a_bare_ordered_array(self):
        scores, _ = g_mod.parse_guess_reply('["STAR", "MOON"]',
                                            ["MOON", "STAR"])
        self.assertGreater(scores["STAR"], scores["MOON"])

    def test_guess_falls_back_to_bare_word_order(self):
        scores, _ = g_mod.parse_guess_reply("I would say STAR then MOON.",
                                            ["MOON", "STAR"])
        self.assertGreater(scores["STAR"], scores["MOON"])

    def test_words_not_on_the_board_are_dropped(self):
        scores, avoid = g_mod.parse_guess_reply(
            score_reply({"MOON": 90, "ZEBRA": 99}, ["ZEBRA"]),
            ["MOON", "STAR"])
        self.assertEqual(set(scores), {"MOON"})
        self.assertEqual(avoid, set())


# ---------------------------------------------------------------------------
# Codemaster
# ---------------------------------------------------------------------------

class TestCodemasterHappyPath(unittest.TestCase):

    def test_one_call_and_a_legal_clue(self):
        agent = make_codemaster(
            clue_reply("LUNAR", ["MOON", "STAR"], ["MOON", "STAR", "COMET"], 2))
        clue, number = agent.get_clue()
        self.assertEqual(clue, "LUNAR")
        self.assertEqual(number, 2)
        self.assertEqual(agent.llm.calls, 1)

    def test_prompt_contains_the_whole_key(self):
        agent = make_codemaster(
            clue_reply("LUNAR", ["MOON"], ["MOON"], 1))
        agent.get_clue()
        prompt = agent.prompts[0]
        for word in ("APPLE", "BANK", "SOCK", "DEATH"):
            self.assertIn(word, prompt)
        self.assertIn("ASSASSIN", prompt)
        self.assertIn("LOSE IMMEDIATELY", prompt)

    def test_revealed_words_leave_the_prompt(self):
        agent = make_codemaster(clue_reply("LUNAR", ["MOON"], ["MOON"], 1))
        board = list(BOARD)
        board[0] = "*Red*"                 # APPLE gone
        agent.set_game_state(board, list(KEY))
        agent.get_clue()
        self.assertNotIn("APPLE", agent.prompts[0])


class TestCodemasterNumberDiscipline(unittest.TestCase):
    """The number is computed from the model's own simulated ranking."""

    def test_number_is_the_leading_run_of_our_own_words(self):
        # MOON, STAR are red; RIVER is blue and truncates the run at 2.
        agent = make_codemaster(
            clue_reply("SKY", ["MOON", "STAR", "COMET"],
                       ["MOON", "STAR", "RIVER", "COMET"], 3))
        self.assertEqual(agent.get_clue(), ["SKY", 2])
        self.assertEqual(agent.number_capped, 1)

    def test_number_never_exceeds_the_claim(self):
        agent = make_codemaster(
            clue_reply("SKY", ["MOON", "STAR", "COMET"],
                       ["MOON", "STAR", "COMET"], 1))
        self.assertEqual(agent.get_clue(), ["SKY", 1])

    def test_number_never_exceeds_the_surviving_targets(self):
        # ROCKET is ours but BANK is not; only one target survives.
        agent = make_codemaster(
            clue_reply("LAUNCH", ["ROCKET", "BANK"],
                       ["ROCKET", "MOON", "STAR"], 3))
        self.assertEqual(agent.get_clue(), ["LAUNCH", 1])

    def test_number_is_capped_by_max_clue_number(self):
        agent = make_codemaster(
            clue_reply("SKY", ["MOON", "STAR", "COMET", "ROCKET"],
                       ["MOON", "STAR", "COMET", "ROCKET"], 4))
        self.assertEqual(agent.get_clue(), ["SKY", cm_mod.MAX_CLUE_NUMBER])

    def test_board_words_are_matched_through_normalisation(self):
        # The secret competition pool may be slangier than the bundled one, so
        # a board word with a space or a hyphen must still be resolvable from
        # the bare token the model echoes back.
        board = list(BOARD)
        board[6] = "ICE CREAM"                  # was MOON, still Red
        agent = make_codemaster(
            clue_reply("DESSERT", ["ICECREAM"], ["ICECREAM", "STAR"], 2))
        agent.set_game_state(board, list(KEY))
        clue, number = agent.get_clue()
        self.assertEqual(clue, "DESSERT")
        self.assertEqual(number, 1)             # one surviving target
        self.assertEqual(agent.clue_log[0]["targets"], ["ICE CREAM"])

    def test_duplicate_targets_are_not_double_counted(self):
        agent = make_codemaster(
            clue_reply("SKY", ["MOON", "MOON", "MOON"],
                       ["MOON", "STAR", "COMET"], 3))
        self.assertEqual(agent.get_clue(), ["SKY", 1])

    def test_no_ranking_means_a_single_word(self):
        agent = make_codemaster(
            clue_reply("SKY", ["MOON", "STAR"], [], 2))
        self.assertEqual(agent.get_clue(), ["SKY", 1])


class TestCodemasterSafety(unittest.TestCase):

    def test_assassin_in_the_top_k_vetoes_the_clue(self):
        agent = make_codemaster(replies=[
            clue_reply("ENDING", ["MOON"], ["MOON", "DEATH"], 1),
            clue_reply("LUNAR", ["MOON"], ["MOON", "STAR"], 1),
        ])
        self.assertEqual(agent.get_clue(), ["LUNAR", 1])
        self.assertEqual(agent.assassin_vetoes, 1)
        self.assertEqual(agent.llm.calls, 2)

    def test_a_ranking_that_opens_on_a_foreign_word_is_rejected(self):
        agent = make_codemaster(replies=[
            clue_reply("WATER", ["MOON"], ["RIVER", "MOON"], 1),
            clue_reply("LUNAR", ["MOON"], ["MOON"], 1),
        ])
        self.assertEqual(agent.get_clue(), ["LUNAR", 1])

    def test_illegal_clue_is_retried_then_falls_back(self):
        agent = make_codemaster(replies=[
            clue_reply("MOONBEAM", ["MOON"], ["MOON"], 1),   # contains MOON
            clue_reply("SEEDLING", ["SEED"], ["SEED"], 1),   # contains SEED
        ])
        clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertEqual(number, 1)
        self.assertEqual(agent.fallbacks, 1)

    def test_retry_prompt_names_the_reason(self):
        agent = make_codemaster(replies=[
            clue_reply("MOONBEAM", ["MOON"], ["MOON"], 1),
            clue_reply("LUNAR", ["MOON"], ["MOON"], 1),
        ])
        agent.get_clue()
        self.assertEqual(len(agent.prompts), 2)
        self.assertIn("rejected", agent.prompts[1])
        self.assertIn("MOONBEAM", agent.prompts[1])

    def test_targets_that_are_not_ours_reject_the_clue(self):
        agent = make_codemaster(replies=[
            clue_reply("ROYAL", ["KING", "QUEEN"], ["KING", "QUEEN"], 2),
            clue_reply("LUNAR", ["MOON"], ["MOON"], 1),
        ])
        self.assertEqual(agent.get_clue(), ["LUNAR", 1])

    def test_clue_retries_zero_makes_it_strictly_single_call(self):
        agent = make_codemaster(
            clue_reply("MOONBEAM", ["MOON"], ["MOON"], 1), clue_retries=0)
        agent.get_clue()
        self.assertEqual(agent.llm.calls, 1)


class TestCodemasterNoRepeats(unittest.TestCase):

    def test_a_clue_already_in_the_move_history_is_rejected(self):
        agent = make_codemaster(replies=[
            clue_reply("LUNAR", ["MOON"], ["MOON"], 1),
            clue_reply("ORBIT", ["MOON"], ["MOON"], 1),
        ])
        agent.set_move_history([["Red_Codemaster", "LUNAR", 1]])
        self.assertEqual(agent.get_clue(), ["ORBIT", 1])

    def test_the_banned_list_reaches_the_prompt(self):
        agent = make_codemaster(clue_reply("ORBIT", ["MOON"], ["MOON"], 1))
        agent.set_move_history([["Red_Codemaster", "LUNAR", 1]])
        agent.get_clue()
        self.assertIn("LUNAR", agent.prompts[0])
        self.assertIn("may NOT repeat", agent.prompts[0])

    def test_the_opponents_clues_are_not_banned_for_us(self):
        agent = make_codemaster(clue_reply("LUNAR", ["MOON"], ["MOON"], 1))
        agent.set_move_history([["Blue_Codemaster", "LUNAR", 1]])
        self.assertEqual(agent.get_clue(), ["LUNAR", 1])

    def test_the_offline_fallback_never_repeats_itself(self):
        agent = make_codemaster(reply=None)
        agent.llm.available = lambda: False
        seen = []
        for _ in range(6):
            clue, number = agent.get_clue()
            self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
            self.assertEqual(number, 1)
            seen.append(clue)
            agent.set_move_history(
                [["Red_Codemaster", c, 1] for c in seen])
        self.assertEqual(len(set(seen)), len(seen))


class TestCodemasterRobustness(unittest.TestCase):

    def test_no_api_still_produces_a_legal_clue(self):
        agent = cm_mod.AICodemaster("Red", move_wall=0, quiet=True,
                                    api_key=None)
        agent.llm.available = lambda: False
        agent.set_game_state(list(BOARD), list(KEY))
        clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertGreaterEqual(number, 1)

    def test_an_exception_inside_the_pipeline_never_escapes(self):
        agent = make_codemaster(clue_reply("LUNAR", ["MOON"], ["MOON"], 1))

        def boom(*_args, **_kwargs):
            raise RuntimeError("kaboom")

        agent._get_clue_inner = boom
        clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertGreaterEqual(number, 1)

    def test_a_wedged_call_is_abandoned_at_the_wall(self):
        agent = cm_mod.AICodemaster("Red", move_wall=0.4, quiet=True)
        agent.set_game_state(list(BOARD), list(KEY))
        agent.llm.available = lambda: True
        agent.llm.chat = lambda *a, **k: time.sleep(30)
        started = time.time()
        clue, number = agent.get_clue()
        self.assertLess(time.time() - started, 5.0)
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertEqual(number, 1)

    def test_an_empty_board_does_not_crash(self):
        agent = make_codemaster(clue_reply("LUNAR", ["MOON"], ["MOON"], 1))
        agent.set_game_state(["*Red*"] * 25, list(KEY))
        clue, number = agent.get_clue()
        self.assertIsInstance(clue, str)
        self.assertGreaterEqual(number, 1)

    def test_garbage_replies_never_produce_a_malformed_clue(self):
        for junk in ("", "???", "{", '{"clue": 42}', '{"clue": "A B"}',
                     '{"clue": "MOON"}', "null"):
            agent = make_codemaster(replies=[junk, junk])
            clue, number = agent.get_clue()
            self.assertTrue(cm_mod.clue_is_legal(clue, BOARD),
                            "illegal clue %r from reply %r" % (clue, junk))
            self.assertIsInstance(number, int)
            self.assertGreaterEqual(number, 1)

    def test_usage_summary_is_json_serialisable(self):
        agent = make_codemaster(clue_reply("LUNAR", ["MOON"], ["MOON"], 1))
        agent.get_clue()
        json.dumps(agent.usage_summary())

    def test_blue_team_reads_the_key_the_other_way_round(self):
        agent = cm_mod.AICodemaster("Blue", move_wall=0, quiet=True)
        agent.set_game_state(list(BOARD), list(KEY))
        own, opp, civ, assassin = agent._split_board()
        self.assertIn("BANK", own)
        self.assertIn("APPLE", opp)
        self.assertEqual(assassin, "DEATH")


# ---------------------------------------------------------------------------
# Guesser
# ---------------------------------------------------------------------------

class TestGuesserBasics(unittest.TestCase):

    def test_one_call_per_turn_however_many_guesses(self):
        agent = make_guesser(score_reply(
            {"MOON": 95, "STAR": 90, "COMET": 88}))
        agent.set_clue("SKY", 3)
        first = agent.get_answer()
        self.assertEqual(first, "MOON")
        self._reveal(agent, first)
        self.assertTrue(agent.keep_guessing())
        second = agent.get_answer()
        self.assertEqual(second, "STAR")
        self.assertEqual(agent.llm.calls, 1)

    def test_the_prompt_lists_every_remaining_word(self):
        agent = make_guesser(score_reply({"MOON": 90}))
        agent.set_clue("SKY", 1)
        agent.get_answer()
        prompt = agent.prompts[0]
        for word in BOARD:
            self.assertIn(word, prompt)
        self.assertIn("NOT TOUCH", prompt)

    def test_the_guess_is_always_an_unrevealed_board_word(self):
        board = ["*Red*"] * 24 + ["DEATH"]
        agent = make_guesser(score_reply({"DEATH": 10}))
        agent.set_board(board)
        agent.set_clue("SKY", 2)
        self.assertEqual(agent.get_answer(), "DEATH")

    def test_no_options_returns_none(self):
        agent = make_guesser(score_reply({}))
        agent.set_board(["*Red*"] * 25)
        agent.set_clue("SKY", 1)
        self.assertIsNone(agent.get_answer())

    def test_guess_counter_resets_between_turns(self):
        agent = make_guesser(replies=[score_reply({"MOON": 90}),
                                      score_reply({"STAR": 90})])
        agent.set_clue("SKY", 1)
        agent.get_answer()
        self.assertEqual(agent.guesses, 1)
        agent.set_clue("NIGHT", 1)
        self.assertEqual(agent.guesses, 0)

    @staticmethod
    def _reveal(agent, word):
        """What the engine does: mutate its own list, no ``set_board`` call."""
        agent.words = ["*Red*" if w == word else w for w in agent.words]


class TestPromptMemory(unittest.TestCase):
    """Leftovers are carried between turns without paying for a second call."""

    def test_the_codemaster_tells_the_model_which_words_are_half_clued(self):
        agent = make_codemaster(replies=[
            clue_reply("SKY", ["MOON", "STAR"], ["MOON", "STAR"], 2),
            clue_reply("ORBIT", ["STAR"], ["STAR"], 1),
        ])
        agent.get_clue()
        board = ["*Red*" if w == "MOON" else w for w in BOARD]
        agent.set_game_state(board, list(KEY))
        agent.set_move_history([["Red_Codemaster", "SKY", 2]])
        agent.get_clue()
        prompt = agent.prompts[1]
        self.assertIn("Already hinted at", prompt)
        self.assertIn("STAR", prompt.split("Already hinted at")[1][:80])

    def test_the_guesser_sees_earlier_turns_but_not_the_current_clue_twice(self):
        agent = make_guesser(replies=[score_reply({"MOON": 90}),
                                      score_reply({"STAR": 90})])
        agent.set_clue("SKY", 1)
        agent.set_move_history([["Red_Codemaster", "SKY", 1]])
        agent.get_answer()
        self.assertNotIn("EARLIER THIS GAME", agent.prompts[0])

        agent.set_move_history([["Red_Codemaster", "SKY", 1],
                                ["Red_Guesser", "MOON", "*Red*", False],
                                ["Red_Codemaster", "NIGHT", 1]])
        agent.set_clue("NIGHT", 1)
        agent.get_answer()
        block = agent.prompts[1]
        self.assertIn("EARLIER THIS GAME", block)
        self.assertIn("we touched MOON", block)
        self.assertEqual(block.count("clue SKY 1"), 1)


class TestGuesserStopRule(unittest.TestCase):

    def test_the_first_guess_is_mandatory_however_weak(self):
        agent = make_guesser(score_reply(dict((w, 3.0) for w in BOARD)))
        agent.set_clue("NONSENSE", 2)
        self.assertIsNotNone(agent.get_answer())

    def test_it_stops_when_the_next_word_falls_off_a_cliff(self):
        agent = make_guesser(score_reply({"MOON": 95, "STAR": 20}))
        agent.set_clue("SKY", 2)
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())
        self.assertEqual(agent.stopped_early, 1)

    def test_it_continues_when_the_next_word_is_comparable(self):
        agent = make_guesser(score_reply({"MOON": 95, "STAR": 88}))
        agent.set_clue("SKY", 2)
        agent.get_answer()
        self.assertTrue(agent.keep_guessing())

    def test_the_absolute_floor_bites_even_on_a_flat_ranking(self):
        agent = make_guesser(score_reply({"MOON": 30, "STAR": 29}))
        agent.set_clue("SKY", 2)
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())

    def test_it_never_exceeds_the_clue_number(self):
        agent = make_guesser(score_reply(
            {"MOON": 99, "STAR": 99, "COMET": 99}))
        agent.set_clue("SKY", 1)
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())

    def test_the_bonus_guess_is_off_by_default(self):
        agent = make_guesser(score_reply(
            {"MOON": 99, "STAR": 99, "COMET": 99}))
        agent.set_clue("SKY", 2)
        agent.get_answer()
        TestGuesserBasics._reveal(agent, "MOON")
        self.assertTrue(agent.keep_guessing())
        agent.get_answer()
        TestGuesserBasics._reveal(agent, "STAR")
        self.assertFalse(agent.keep_guessing())

    def test_the_bonus_guess_needs_its_own_high_floor_when_armed(self):
        scores = {"MOON": 99, "STAR": 99, "COMET": 70}
        agent = make_guesser(score_reply(scores), allow_bonus=True)
        agent.set_clue("SKY", 2)
        agent.get_answer()
        TestGuesserBasics._reveal(agent, "MOON")
        agent.keep_guessing()
        agent.get_answer()
        TestGuesserBasics._reveal(agent, "STAR")
        self.assertFalse(agent.keep_guessing())   # 70 < BONUS_FLOOR

        agent = make_guesser(score_reply(
            {"MOON": 99, "STAR": 99, "COMET": 95}), allow_bonus=True)
        agent.set_clue("SKY", 2)
        agent.get_answer()
        TestGuesserBasics._reveal(agent, "MOON")
        agent.keep_guessing()
        agent.get_answer()
        TestGuesserBasics._reveal(agent, "STAR")
        self.assertTrue(agent.keep_guessing())

    def test_a_word_on_the_avoid_list_is_demoted_and_never_continued_into(self):
        agent = make_guesser(score_reply({"MOON": 95, "DEATH": 94},
                                         avoid=["DEATH"]))
        agent.set_clue("SKY", 2)
        self.assertEqual(agent.get_answer(), "MOON")
        TestGuesserBasics._reveal(agent, "MOON")
        self.assertFalse(agent.keep_guessing())

    def test_no_cached_confidence_means_one_word_only(self):
        agent = make_guesser(reply=None)
        agent.llm.available = lambda: False
        agent.set_clue("SKY", 3)
        agent.get_answer()          # offline ranking is still built...
        agent._scores = None        # ...but if nothing at all is cached:
        self.assertFalse(agent.keep_guessing())


class TestGuesserStaleBoard(unittest.TestCase):
    """The recorded, fatal bug: ``keep_guessing`` runs against a board the
    engine has already advanced, with no ``set_board`` in between."""

    def test_the_pending_guess_is_excluded_from_the_options(self):
        agent = make_guesser(score_reply({"MOON": 95, "STAR": 90}))
        agent.set_clue("SKY", 2)
        self.assertEqual(agent.get_answer(), "MOON")
        # The engine has accepted MOON but has NOT called set_board.
        self.assertNotIn("MOON", agent._options())
        self.assertEqual(agent._pending_own(), "MOON")

    def test_the_stop_rule_reads_the_next_word_not_the_one_just_guessed(self):
        # Without the fix the "next" word is MOON again and the confidence
        # ratio is 1.0 by construction, so the rule can never stop.
        agent = make_guesser(score_reply({"MOON": 95, "STAR": 10}))
        agent.set_clue("SKY", 2)
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())

    def test_a_refreshed_board_clears_the_pending_word(self):
        agent = make_guesser(score_reply({"MOON": 95, "STAR": 90}))
        agent.set_clue("SKY", 2)
        agent.get_answer()
        agent.set_board(["*Red*" if w == "MOON" else w for w in BOARD])
        self.assertIsNone(agent._pending_own())
        self.assertNotIn("MOON", agent._options())

    def test_the_same_word_is_never_guessed_twice_in_a_turn(self):
        agent = make_guesser(score_reply(
            {"MOON": 95, "STAR": 94, "COMET": 93}))
        agent.set_clue("SKY", 3)
        picks = []
        for _ in range(3):
            word = agent.get_answer()
            picks.append(word)
            agent.words = ["*Red*" if w == word else w for w in agent.words]
        self.assertEqual(len(set(picks)), 3)


class TestGuesserUnlimited(unittest.TestCase):

    def test_num_zero_is_capped_not_honoured(self):
        scores = dict((w, 99.0) for w in BOARD)
        agent = make_guesser(score_reply(scores))
        agent.set_clue("SWEEP", 0)
        taken = 0
        while True:
            word = agent.get_answer()
            taken += 1
            agent.words = ["*Red*" if w == word else w for w in agent.words]
            if not agent.keep_guessing():
                break
        self.assertEqual(taken, g_mod.SWEEP_CAP)

    def test_num_zero_uses_the_raised_floor(self):
        agent = make_guesser(score_reply({"MOON": 99, "STAR": 55}))
        agent.set_clue("SWEEP", 0)
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())   # 55 < SWEEP_FLOOR

    def test_a_negative_number_is_treated_as_unlimited(self):
        agent = make_guesser(score_reply({"MOON": 99}))
        agent.set_clue("SWEEP", -3)
        self.assertEqual(agent.num, 0)
        agent.get_answer()

    def test_a_non_integer_number_does_not_crash(self):
        agent = make_guesser(score_reply({"MOON": 99}))
        agent.set_clue("SKY", "two")
        self.assertEqual(agent.num, 1)
        self.assertEqual(agent.get_answer(), "MOON")


class TestGuesserRobustness(unittest.TestCase):

    def test_no_api_still_guesses_a_real_word(self):
        agent = g_mod.AIGuesser("Red", move_wall=0, quiet=True)
        agent.llm.available = lambda: False
        agent.set_board(list(BOARD))
        agent.set_clue("MOONLIGHT", 1)
        self.assertIn(agent.get_answer(), BOARD)

    def test_garbage_replies_still_guess_a_real_word(self):
        for junk in ("", "???", "{", '{"scores": "nope"}', "null",
                     '{"scores": {"NOTAWORD": 9}}'):
            agent = make_guesser(reply=junk)
            agent.set_clue("SKY", 2)
            self.assertIn(agent.get_answer(), BOARD,
                          "bad guess from reply %r" % junk)

    def test_an_exception_inside_the_pipeline_never_escapes(self):
        agent = make_guesser(score_reply({"MOON": 90}))

        def boom(*_args, **_kwargs):
            raise RuntimeError("kaboom")

        agent._get_answer_inner = boom
        agent.set_clue("SKY", 1)
        self.assertIn(agent.get_answer(), BOARD)

    def test_keep_guessing_never_raises(self):
        agent = make_guesser(score_reply({"MOON": 90}))
        agent.set_clue("SKY", 2)
        agent.get_answer()

        def boom(*_args, **_kwargs):
            raise RuntimeError("kaboom")

        agent._keep_guessing_inner = boom
        self.assertFalse(agent.keep_guessing())

    def test_a_wedged_call_is_abandoned_at_the_wall(self):
        agent = g_mod.AIGuesser("Red", move_wall=0.4, quiet=True)
        agent.llm.available = lambda: True
        agent.llm.chat = lambda *a, **k: time.sleep(30)
        agent.set_board(list(BOARD))
        agent.set_clue("SKY", 2)
        started = time.time()
        word = agent.get_answer()
        self.assertLess(time.time() - started, 5.0)
        self.assertIn(word, BOARD)
        self.assertEqual(agent._pending, word)

    def test_usage_summary_is_json_serialisable(self):
        agent = make_guesser(score_reply({"MOON": 90}))
        agent.set_clue("SKY", 1)
        agent.get_answer()
        json.dumps(agent.usage_summary())


# ---------------------------------------------------------------------------
# End to end, inside the real engine
# ---------------------------------------------------------------------------

class TestFullGameOffline(unittest.TestCase):
    """A whole single-team game with both agents on their offline paths.

    Proves the pair finishes legally with no key and no network, which is the
    exact configuration the organisers accidentally ran the champion in.
    """

    def test_a_full_single_team_game_finishes_legally(self):
        import contextlib
        import io
        import tempfile
        import shutil

        from game import Game

        source = os.path.join(_FRAMEWORK, "players", "cm_wordlist.txt")
        sandbox = tempfile.mkdtemp(prefix="typenull-")
        try:
            os.makedirs(os.path.join(sandbox, "players"))
            shutil.copy(source, os.path.join(sandbox, "players",
                                             "cm_wordlist.txt"))
            cwd = os.getcwd()
            os.chdir(sandbox)
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    game = Game(cm_mod.AICodemaster, g_mod.AIGuesser,
                                cm_mod.AICodemaster, g_mod.AIGuesser,
                                seed=7, do_print=True, do_log=False,
                                single_team=True,
                                cmr_kwargs={"quiet": True, "api_key": None},
                                gr_kwargs={"quiet": True, "api_key": None},
                                cmb_kwargs={"quiet": True, "api_key": None},
                                gb_kwargs={"quiet": True, "api_key": None})
                    # No key anywhere: force the offline path deterministically
                    # rather than depending on the environment.
                    for agent in (game.codemaster_red, game.guesser_red):
                        agent.llm.available = lambda: False
                    game.run()
            finally:
                os.chdir(cwd)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

        self.assertIn(game.game_winner, ("R", "B"))
        clues = [entry for entry in game.get_move_history()
                 if entry[0].endswith("_Codemaster")]
        self.assertTrue(clues)
        seen = set()
        # Legality against the *live* board cannot be re-checked here -- the
        # board has moved on -- so the structural half of the rule is asserted
        # instead, plus the no-repeat invariant that a dead API breaks.
        for _, clue, number in clues:
            self.assertIsInstance(number, int)
            self.assertGreaterEqual(number, 1)
            self.assertEqual(len(str(clue).split()), 1)
            self.assertTrue(str(clue).isalpha())
            self.assertNotIn(cm_mod._normalise(clue), seen,
                             "clue %r repeated" % clue)
            seen.add(cm_mod._normalise(clue))


if __name__ == "__main__":
    unittest.main()
