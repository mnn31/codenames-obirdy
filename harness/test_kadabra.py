"""Offline unit tests for the Kadabra agents.

Kadabra is the embedding-first challenger: GloVe proposes, one LLM call
disposes.  Nothing here touches the network -- the LLM is monkeypatched
everywhere it appears, and the tests that need real vectors skip when the
gitignored cache is absent, exactly as ``test_partners`` does.

What these tests are actually protecting, in priority order:

1. **The disqualification rules.**  An illegal clue, a crash, a malformed
   response or a move over the time limit ends our tournament, so each has a
   test that reaches it through the public entry point rather than through an
   internal helper.
2. **The two LLM layers firing where -- and only where -- they should.**  The
   danger check on a shortlist the embedding produced, the rescue call when it
   produced nothing, and never both in one turn.
3. **The stop rule against a stale board**, which is the bug class that cost
   the champion three recorded assassin deaths.

Run from the repo root::

    python -m unittest harness.test_kadabra -v
    python -m pytest harness/test_kadabra.py -q
"""

import os
import sys
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from players import codemaster_kadabra as cm_mod  # noqa: E402
from players import codemaster_obirdy as champion  # noqa: E402
from players import guesser_kadabra as g_mod  # noqa: E402

BOARD = [
    "WHALE", "SHIP", "BEACH", "APPLE", "TREE",
    "ROBOT", "LASER", "KING", "CROWN", "DRAGON",
    "PIANO", "FLUTE", "BANK", "GOLD", "DESERT",
    "SNOW", "TOWER", "SHADOW", "NIGHT", "HORSE",
    "CAR", "TRAIN", "DOCTOR", "NURSE", "POISON",
]

KEY = [
    "Red", "Red", "Red", "Blue", "Blue",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Red", "Red", "Blue", "Blue", "Civilian",
    "Civilian", "Civilian", "Red", "Civilian", "Blue",
    "Civilian", "Blue", "Red", "Civilian", "Assassin",
]

HAVE_VECTORS = cm_mod.vectors() is not None
needs_vectors = unittest.skipUnless(
    HAVE_VECTORS, "GloVe cache absent (python -m harness.glove_data)")


class FakeLLM(object):
    """Stands in for ``_LLM``: scripted replies, counted calls, never a socket.

    ``replies`` is consumed in order and then repeats its last entry, so a test
    that only cares about the first call does not have to script the rest.  A
    reply of ``None`` is a failed call, which is the case every fallback in
    both agents is built around.
    """

    def __init__(self, replies=None, available=True, raises=False):
        self.replies = list(replies or [])
        self._available = available
        self.raises = raises
        self.calls = 0
        self.prompts = []
        self.systems = []
        self.model = "fake"
        self.failures = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def available(self):
        return self._available

    def key_source(self):
        return "test" if self._available else None

    def chat(self, system, user, max_tokens=700, deadline=None):
        self.calls += 1
        self.systems.append(system)
        self.prompts.append(user)
        if self.raises:
            raise RuntimeError("boom")
        if not self.replies:
            return None
        if len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)

    def usage_summary(self):
        return {"model": self.model, "calls": self.calls, "retries": 0,
                "failures": 0, "input_tokens": 0, "output_tokens": 0}


def make_codemaster(llm=None, **kwargs):
    kwargs.setdefault("quiet", True)
    kwargs.setdefault("move_wall", 0)     # run on the calling thread
    agent = cm_mod.AICodemaster("Red", **kwargs)
    if llm is not None:
        agent.llm = llm
    agent.set_game_state(list(BOARD), list(KEY))
    return agent


def make_guesser(llm=None, board=None, **kwargs):
    kwargs.setdefault("quiet", True)
    kwargs.setdefault("move_wall", 0)
    agent = g_mod.AIGuesser("Red", **kwargs)
    if llm is not None:
        agent.llm = llm
    agent.set_board(list(board if board is not None else BOARD))
    return agent


# ---------------------------------------------------------------------------
# Legality -- the disqualification rule
# ---------------------------------------------------------------------------

class TestLegality(unittest.TestCase):

    def test_matches_the_champion_exactly(self):
        """Behavioural equality with the shipped filter, not a re-derivation.

        The competition audit, the champion and this agent must agree on every
        clue, so the test asserts agreement on a corpus rather than restating
        the rule and hoping the two restatements match.
        """
        corpus = [
            "MARINE", "WHALES", "WHA", "WHALE", "DEEP SEA", "", "  ", "AB",
            "SEA-DOG", "SHIP2", "kingdom", "KINGDOM", "CROWNED", "POISONOUS",
            "TOWER", "shadowy", "TRAINING", "A" * 40, "NIGHTFALL", "Doctor",
            "3RD", "N.I.G.H.T", "HORSEPLAY", "GOLDEN",
        ]
        for clue in corpus:
            self.assertEqual(
                cm_mod.clue_is_legal(clue, BOARD),
                champion.clue_is_legal(clue, BOARD),
                "disagreed with the champion on %r" % clue)

    def test_rejects_board_derivations_both_directions(self):
        self.assertFalse(cm_mod.clue_is_legal("WHALES", BOARD))   # superword
        self.assertFalse(cm_mod.clue_is_legal("WHA", BOARD))      # sub-word
        self.assertFalse(cm_mod.clue_is_legal("DEEP SEA", BOARD))  # two words
        self.assertFalse(cm_mod.clue_is_legal("SEA-DOG", BOARD))  # punctuation
        self.assertTrue(cm_mod.clue_is_legal("MARINE", BOARD))

    def test_revealed_words_stop_constraining(self):
        revealed = ["*Red*" if w == "WHALE" else w for w in BOARD]
        self.assertTrue(cm_mod.clue_is_legal("WHALES", revealed))

    def test_stem_guard_is_additional_not_part_of_legality(self):
        # Legal by the competition rule, refused as a candidate: a human judge
        # reads SWIMMING and SWIMMER as the same word.
        board = ["SWIMMING"] + BOARD[1:]
        self.assertTrue(cm_mod.clue_is_legal("SWIMMER", board))
        self.assertFalse(cm_mod._clue_is_clean("SWIMMER", board))
        self.assertTrue(cm_mod._clue_is_clean("POOL", board))


# ---------------------------------------------------------------------------
# Codemaster: shape, robustness, and the offline paths
# ---------------------------------------------------------------------------

class TestCodemasterRobustness(unittest.TestCase):

    def test_always_returns_a_legal_pair(self):
        agent = make_codemaster(FakeLLM(available=False))
        clue, number = agent.get_clue()
        self.assertIsInstance(clue, str)
        self.assertIsInstance(number, int)
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertGreaterEqual(number, 1)
        self.assertLessEqual(number, cm_mod.MAX_NUMBER)

    def test_no_key_no_vectors_still_plays(self):
        agent = make_codemaster(FakeLLM(available=False))
        original = cm_mod.vectors
        cm_mod.vectors = lambda: None
        try:
            clue, number = agent.get_clue()
        finally:
            cm_mod.vectors = original
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertEqual(number, 1)

    def test_a_crashing_pipeline_still_answers(self):
        agent = make_codemaster(FakeLLM(available=False))

        def explode():
            raise ValueError("pipeline died")

        agent._get_clue_inner = explode
        clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertEqual(agent._turn_path, "crash-fallback")

    def test_the_hard_wall_abandons_a_wedged_pipeline(self):
        agent = make_codemaster(FakeLLM(available=False), move_wall=0.2)

        def forever():
            time.sleep(30)
            return ["NEVER", 1]

        agent._get_clue_inner = forever
        started = time.time()
        clue, number = agent.get_clue()
        self.assertLess(time.time() - started, 5.0)
        self.assertEqual(agent._turn_path, "wall-fallback")
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))

    def test_the_sanitiser_repairs_any_junk_the_pipeline_returns(self):
        agent = make_codemaster(FakeLLM(available=False))
        for junk in (["WHALE", 3],          # a board word
                     ["MARINE", 99],        # number out of range
                     ["MARINE", 0],         # a codemaster may not sweep by mistake
                     ["deep sea", 1],       # two words
                     [None, None],
                     "nonsense"):
            agent._get_clue_inner = lambda junk=junk: junk
            clue, number = agent.get_clue()
            self.assertTrue(cm_mod.clue_is_legal(clue, BOARD), junk)
            self.assertGreaterEqual(number, 1)
            self.assertLessEqual(number, cm_mod.MAX_NUMBER)

    def test_never_repeats_a_clue(self):
        agent = make_codemaster(FakeLLM(available=False))
        seen = set()
        for _ in range(6):
            clue, _number = agent.get_clue()
            self.assertNotIn(clue, seen)
            seen.add(clue)

    def test_number_never_exceeds_the_words_we_have_left(self):
        agent = make_codemaster(FakeLLM(available=False))
        board = list(BOARD)
        for i, kind in enumerate(KEY):
            if kind == "Red" and board[i] != "WHALE":
                board[i] = "*Red*"
        agent.set_game_state(board, list(KEY))
        _clue, number = agent.get_clue()
        self.assertEqual(number, 1)

    def test_a_short_or_missing_key_grid_is_survivable(self):
        """The engine is trusted, but a malformed state must not be fatal."""
        for maps in ([], KEY[:5], ["Red"] * 25, ["nonsense"] * 25):
            agent = make_codemaster(FakeLLM(available=False))
            agent.set_game_state(list(BOARD), list(maps))
            clue, number = agent.get_clue()
            self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
            self.assertGreaterEqual(number, 1)

    @needs_vectors
    def test_long_clues_are_refused(self):
        agent = make_codemaster(FakeLLM(available=False), max_clue_length=6)
        own, opp, civ, assassin = agent._split_board()
        for clue, _t, _n, _s, _m in agent._embedding_candidates(
                own, opp, civ, assassin):
            self.assertLessEqual(len(clue), 6)

    def test_usage_summary_is_serialisable(self):
        agent = make_codemaster(FakeLLM(available=False))
        summary = agent.usage_summary()
        for field in ("calls", "checks_run", "checks_vetoed", "number_capped"):
            self.assertIn(field, summary)


# ---------------------------------------------------------------------------
# Codemaster: the danger check
# ---------------------------------------------------------------------------

class TestDangerCheckParsing(unittest.TestCase):

    def test_plain_json(self):
        text = '{"MARINE": {"assassin": 0, "opponent": 2}, "best": "MARINE"}'
        ratings, pick = cm_mod.parse_danger(text, ["MARINE", "OCEANIC"])
        self.assertEqual(ratings["MARINE"], (0.0, 2.0))
        self.assertEqual(pick, "MARINE")

    def test_prose_around_the_json_and_alternative_keys(self):
        text = ('Here is my read.\n'
                '{"MARINE": {"risk": 7, "enemy": 1}, "pick": "OCEANIC"}\n'
                'Hope that helps.')
        ratings, pick = cm_mod.parse_danger(text, ["MARINE", "OCEANIC"])
        self.assertEqual(ratings["MARINE"], (7.0, 1.0))
        self.assertEqual(pick, "OCEANIC")

    def test_word_ratings_and_bare_numbers(self):
        text = '{"MARINE": "high", "OCEANIC": 3}'
        ratings, _pick = cm_mod.parse_danger(text, ["MARINE", "OCEANIC"])
        self.assertEqual(ratings["MARINE"][0], 8.0)
        self.assertEqual(ratings["OCEANIC"][0], 3.0)

    def test_ratings_are_clamped_and_unknown_clues_ignored(self):
        text = '{"MARINE": 400, "NOTACANDIDATE": 9}'
        ratings, _pick = cm_mod.parse_danger(text, ["MARINE"])
        self.assertEqual(ratings["MARINE"][0], 10.0)
        self.assertNotIn("NOTACANDIDATE", ratings)

    def test_garbage_is_empty_not_clean(self):
        ratings, pick = cm_mod.parse_danger("I cannot help with that.",
                                            ["MARINE"])
        self.assertEqual(ratings, {})
        self.assertIsNone(pick)


class TestDangerCheckBehaviour(unittest.TestCase):
    """The check runs against a fixed, hand-built shortlist.

    Substituting ``_embedding_candidates`` keeps these tests about the *rule*
    rather than about whatever the vector cache happens to contain, so they
    pass identically on a checkout with no cache at all.
    """

    SHORTLIST = [
        ("SURGE", ["CHARGE", "CURRENT"], 4, 3.0, 0.12),
        ("MARINE", ["WHALE", "SHIP"], 2, 2.8, 0.10),
        ("OCEANIC", ["WHALE"], 1, 1.9, 0.20),
    ]

    def _agent(self, reply):
        llm = FakeLLM([reply])
        agent = make_codemaster(llm)
        agent._embedding_candidates = lambda *a, **k: list(self.SHORTLIST)
        return agent, llm

    def test_exactly_one_call_per_turn(self):
        agent, llm = self._agent('{"SURGE": 0, "MARINE": 0, "OCEANIC": 0}')
        agent.get_clue()
        self.assertEqual(llm.calls, 1)

    def test_a_vetoed_leader_loses_to_the_runner_up(self):
        agent, _llm = self._agent(
            '{"SURGE": {"assassin": 8}, "MARINE": {"assassin": 0}, '
            '"OCEANIC": {"assassin": 0}}')
        clue, number = agent.get_clue()
        self.assertEqual(clue, "MARINE")
        self.assertEqual(number, 2)
        self.assertEqual(agent.checks_vetoed, 1)

    def test_a_sub_veto_assassin_rating_narrows_the_clue_to_one_word(self):
        agent, _llm = self._agent(
            '{"SURGE": {"assassin": 3}, "MARINE": {"assassin": 9}, '
            '"OCEANIC": {"assassin": 9}}')
        clue, number = agent.get_clue()
        self.assertEqual(clue, "SURGE")
        self.assertEqual(number, 1)
        self.assertEqual(agent.number_capped, 1)

    def test_the_models_own_pick_breaks_a_near_tie(self):
        agent, _llm = self._agent(
            '{"SURGE": {"assassin": 1}, "MARINE": {"assassin": 0}, '
            '"OCEANIC": {"assassin": 0}, "best": "MARINE"}')
        clue, _number = agent.get_clue()
        self.assertEqual(clue, "MARINE")

    def test_every_candidate_vetoed_falls_through_to_the_offline_clue(self):
        agent, _llm = self._agent(
            '{"SURGE": 9, "MARINE": 9, "OCEANIC": 9}')
        clue, number = agent.get_clue()
        self.assertNotIn(clue, ("SURGE", "MARINE", "OCEANIC"))
        self.assertEqual(number, 1)
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))

    def test_an_unchecked_clue_is_issued_smaller(self):
        """No key, a dead API or an unparseable reply are the same case.

        An unchecked clue is not a clean clue, so the number is capped -- the
        embedding's own margin is the only thing vouching for it.
        """
        for llm in (FakeLLM(available=False), FakeLLM([None]),
                    FakeLLM(["sorry, I can't"])):
            agent = make_codemaster(llm)
            agent._embedding_candidates = lambda *a, **k: list(self.SHORTLIST)
            clue, number = agent.get_clue()
            self.assertEqual(clue, "SURGE")
            self.assertEqual(number, cm_mod.UNCHECKED_NUMBER_CAP)

    def test_a_clean_check_unlocks_the_full_number(self):
        """The cap is what an *unchecked* clue pays, not a global ceiling."""
        agent, _llm = self._agent(
            '{"SURGE": {"assassin": 0, "opponent": 0}}')
        clue, number = agent.get_clue()
        self.assertEqual(clue, "SURGE")
        self.assertEqual(number, 4)
        self.assertGreater(number, cm_mod.UNCHECKED_NUMBER_CAP)
        self.assertEqual(agent.number_capped, 0)

    def test_a_raising_client_does_not_reach_the_engine(self):
        agent = make_codemaster(FakeLLM(raises=True))
        agent._embedding_candidates = lambda *a, **k: list(self.SHORTLIST)
        clue, _number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))

    def test_the_prompt_names_the_assassin_and_the_targets(self):
        agent, llm = self._agent('{"SURGE": 0}')
        agent.get_clue()
        prompt = llm.prompts[0]
        self.assertIn("POISON", prompt)          # the assassin
        self.assertIn("SURGE", prompt)
        self.assertIn("WHALE", prompt)           # a target


# ---------------------------------------------------------------------------
# Codemaster: the out-of-vocabulary rescue
# ---------------------------------------------------------------------------

class TestOutOfVocabularyRescue(unittest.TestCase):
    """The turn's one call writes the clue instead of auditing one.

    Reached when the embedding produced no shortlist at all, which on a real
    board means every own word left is outside the 60k vocabulary -- the
    ``PLATYPUS`` / ``XENOMORPH`` case that is Abra's documented collapse.
    """

    def _agent(self, reply, **kwargs):
        llm = FakeLLM([reply])
        agent = make_codemaster(llm, **kwargs)
        agent._embedding_candidates = lambda *a, **k: []
        return agent, llm

    def test_the_rescue_clue_is_used(self):
        agent, llm = self._agent(
            '[{"clue": "MAMMAL", "targets": ["WHALE"], "number": 1}]')
        clue, number = agent.get_clue()
        self.assertEqual(clue, "MAMMAL")
        self.assertEqual(number, 1)
        self.assertEqual(llm.calls, 1)
        self.assertIn("out of vocabulary", agent._turn_path)

    def test_an_illegal_rescue_clue_is_skipped_not_issued(self):
        agent, _llm = self._agent(
            '[{"clue": "WHALES", "targets": ["WHALE"], "number": 2},'
            ' {"clue": "MAMMAL", "targets": ["WHALE"], "number": 1}]')
        clue, _number = agent.get_clue()
        self.assertEqual(clue, "MAMMAL")

    def test_a_rescue_reply_with_nothing_legal_falls_back_offline(self):
        agent, _llm = self._agent('[{"clue": "WHALE", "targets": []}]')
        clue, number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertEqual(number, 1)
        self.assertIn("offline-fallback", agent._turn_path)

    def test_a_failed_rescue_call_falls_back_offline(self):
        agent, _llm = self._agent(None)
        clue, _number = agent.get_clue()
        self.assertTrue(cm_mod.clue_is_legal(clue, BOARD))
        self.assertIn("offline-fallback", agent._turn_path)

    def test_the_embedding_vetoes_a_rescue_clue_it_can_see_is_fatal(self):
        agent, _llm = self._agent(
            '[{"clue": "VENOM", "targets": ["WHALE"], "number": 1},'
            ' {"clue": "MAMMAL", "targets": ["WHALE"], "number": 1}]')
        agent._assassin_veto = lambda clue, own, assassin: clue == "VENOM"
        clue, _number = agent.get_clue()
        self.assertEqual(clue, "MAMMAL")
        self.assertEqual(agent.checks_vetoed, 1)

    def test_the_rescue_number_cannot_exceed_its_own_targets(self):
        agent, _llm = self._agent(
            '[{"clue": "MAMMAL", "targets": ["WHALE"], "number": 4}]')
        _clue, number = agent.get_clue()
        self.assertEqual(number, 1)

    def test_no_rescue_call_when_the_embedding_had_a_shortlist(self):
        """One call per turn, and only one -- never a check *and* a rescue."""
        llm = FakeLLM(['{"MARINE": 0}'])
        agent = make_codemaster(llm)
        agent._embedding_candidates = lambda *a, **k: [
            ("MARINE", ["WHALE"], 1, 2.0, 0.1)]
        agent.get_clue()
        self.assertEqual(llm.calls, 1)
        self.assertIn("check", agent._turn_path)


class TestCandidateParsing(unittest.TestCase):

    def test_full_objects(self):
        parsed = cm_mod.parse_candidates(
            '[{"clue": "MAMMAL", "targets": ["WHALE", "HORSE"], "number": 2}]')
        self.assertEqual(parsed, [("MAMMAL", ["WHALE", "HORSE"], 2)])

    def test_bare_strings_and_missing_numbers(self):
        parsed = cm_mod.parse_candidates('["MAMMAL", "AQUATIC"]')
        self.assertEqual([p[0] for p in parsed], ["MAMMAL", "AQUATIC"])
        self.assertTrue(all(p[2] >= 1 for p in parsed))

    def test_prose_is_scraped_rather_than_dropped(self):
        parsed = cm_mod.parse_candidates("I would say MAMMAL, or AQUATIC.")
        self.assertIn("MAMMAL", [p[0] for p in parsed])


# ---------------------------------------------------------------------------
# Codemaster: the embedding engine itself
# ---------------------------------------------------------------------------

@needs_vectors
class TestEmbeddingEngine(unittest.TestCase):

    def test_candidates_are_legal_ranked_and_bounded(self):
        agent = make_codemaster(FakeLLM(available=False))
        own, opp, civ, assassin = agent._split_board()
        candidates = agent._embedding_candidates(own, opp, civ, assassin)
        self.assertTrue(candidates)
        self.assertLessEqual(len(candidates), agent.n_candidates)
        scores = [c[3] for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for clue, targets, number, _score, _margin in candidates:
            self.assertTrue(cm_mod._clue_is_clean(clue, BOARD), clue)
            self.assertGreaterEqual(number, 1)
            self.assertLessEqual(number, cm_mod.MAX_NUMBER)
            self.assertTrue(set(targets) <= set(own))

    def test_targets_really_are_the_clues_nearest_own_words(self):
        agent = make_codemaster(FakeLLM(available=False))
        own, opp, civ, assassin = agent._split_board()
        clue, targets, _number, _score, _margin = agent._embedding_candidates(
            own, opp, civ, assassin)[0]
        pulls = agent._clue_pulls(clue, [[w] for w in own])
        best = sorted(zip(pulls, own), key=lambda p: -(p[0] or -1))
        self.assertEqual(set(targets),
                         set(w for _v, w in best[:len(targets)]))

    def test_an_assassin_pulling_clue_is_vetoed_by_the_sensor(self):
        agent = make_codemaster(FakeLLM(available=False))
        own = ["WHALE"]
        # POISON is the assassin on this board; a clue that is essentially the
        # assassin must be refused however good it looks elsewhere.
        self.assertTrue(agent._assassin_veto("TOXIN", own, ["POISON"]))
        self.assertFalse(agent._assassin_veto("BLUBBER", own, ["POISON"]))

    def test_an_unpriceable_clue_is_not_reported_as_safe(self):
        agent = make_codemaster(FakeLLM(available=False))
        self.assertIsNone(agent._clue_margin("ZZQQXX", ["WHALE"], [], [],
                                             ["POISON"]))
        self.assertFalse(agent._assassin_veto("ZZQQXX", ["WHALE"], ["POISON"]))

    def test_the_offline_fallback_consults_the_embedding(self):
        """The recorded ``CITY`` -> ``STATE`` death, as a regression test.

        Letter overlap ranked ``CITY`` top on a board whose assassin was
        ``STATE``.  The fallback now ranks its own vocabulary by cosine against
        the same danger ceiling the main engine uses, so the blind path is
        blind about *meaning*, not about danger.
        """
        agent = make_codemaster(FakeLLM(available=False))
        board = ["PLATYPUS", "STATE"] + ["*Red*"] * 23
        maps = ["Red", "Assassin"] + ["Civilian"] * 23
        agent.set_game_state(board, maps)
        own, opp, civ, assassin = agent._split_board()
        clue, number = agent._fallback_clue()
        self.assertEqual(number, 1)
        self.assertTrue(cm_mod.clue_is_legal(clue, board))
        self.assertGreater(agent._clue_margin(clue, own, opp, civ, assassin),
                           agent._clue_margin("CITY", own, opp, civ, assassin))


# ---------------------------------------------------------------------------
# Guesser
# ---------------------------------------------------------------------------

class TestGuesserBasics(unittest.TestCase):

    def test_returns_an_unrevealed_board_word(self):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", 2)
        word = agent.get_answer()
        self.assertIn(word, BOARD)

    def test_revealed_words_are_never_offered(self):
        board = ["*Red*" if w == "WHALE" else w for w in BOARD]
        agent = make_guesser(FakeLLM(available=False), board=board)
        agent.set_clue("MARINE", 2)
        for _ in range(5):
            word = agent.get_answer()
            self.assertNotEqual(word, "*Red*")
            self.assertFalse(str(word).startswith("*"))

    def test_no_options_left_returns_none(self):
        agent = make_guesser(FakeLLM(available=False),
                             board=["*Red*"] * len(BOARD))
        agent.set_clue("MARINE", 1)
        self.assertIsNone(agent.get_answer())
        self.assertFalse(agent.keep_guessing())

    def test_per_turn_state_is_reset(self):
        """``guesser_GPT`` never resets its counter and stops guessing for the
        rest of the game once the cumulative count passes a clue number."""
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", 2)
        agent.get_answer()
        agent.get_answer()
        agent.set_clue("ROYAL", 2)
        self.assertEqual(agent.guesses, 0)
        self.assertIsNone(agent._ranking)
        self.assertIsNone(agent._pending)
        self.assertTrue(agent.keep_guessing())

    def test_a_crashing_pipeline_still_answers(self):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", 1)

        def explode():
            raise ValueError("ranking died")

        agent._get_answer_inner = explode
        self.assertIn(agent.get_answer(), BOARD)

    def test_the_hard_wall_abandons_a_wedged_ranking(self):
        agent = make_guesser(FakeLLM(available=False), move_wall=0.2)
        agent.set_clue("MARINE", 1)

        def forever():
            time.sleep(30)
            return "WHALE"

        agent._get_answer_inner = forever
        started = time.time()
        word = agent.get_answer()
        self.assertLess(time.time() - started, 5.0)
        self.assertIn(word, BOARD)

    def test_a_ghost_worker_cannot_hide_a_word_it_never_guessed(self):
        """The abandoned pipeline finishes late and writes ``_pending``.

        Its generation is stale by then, so the write is dropped -- otherwise a
        word nobody guessed would vanish from ``_options`` for the rest of the
        turn.
        """
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", 1)
        agent._answer_gen = 5
        agent._claim_pending("WHALE", 4)
        self.assertIsNone(agent._pending)
        agent._claim_pending("WHALE", 5)
        self.assertEqual(agent._pending, "WHALE")


class TestGuesserStopRule(unittest.TestCase):

    def _armed(self, clue="MARINE", num=2, scores=None, source="llm"):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue(clue, num)
        ranking = sorted((scores or {}).items(), key=lambda p: -p[1])
        agent._ranking = ranking
        agent._ranking_key = (g_mod._normalise(clue), agent._is_sweep())
        agent._ranking_source = source
        agent._turn_best = ranking[0][1] if ranking else 0.0
        return agent

    def test_the_stale_board_does_not_make_the_rule_vacuous(self):
        """The bug that cost the champion three assassin deaths.

        ``game.Game.run`` reveals the accepted word in its own list and calls
        ``keep_guessing`` with no ``set_board`` in between, so the word just
        guessed still looks unrevealed to us.  Re-offering it makes every
        confidence ratio 1.0 and the stop rule dead code.
        """
        agent = self._armed(scores={"WHALE": 100.0, "SHIP": 10.0})
        agent.get_answer()                       # takes WHALE
        self.assertEqual(agent._pending, "WHALE")
        self.assertNotIn("WHALE", agent._options())
        # The next candidate is SHIP at 0.1 of the turn's best, not WHALE at
        # 1.0, so the threshold can actually refuse it.
        self.assertFalse(agent.keep_guessing())

    def test_it_continues_on_a_strong_next_word(self):
        agent = self._armed(scores={"WHALE": 100.0, "SHIP": 90.0})
        agent.get_answer()
        self.assertTrue(agent.keep_guessing())

    def test_it_stops_at_the_number_plus_the_bonus(self):
        agent = self._armed(num=1, scores={"WHALE": 100.0, "SHIP": 99.0,
                                           "BEACH": 99.0})
        agent.get_answer()
        self.assertTrue(agent.keep_guessing())    # the bonus, strongly earned
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())   # never a second bonus

    def test_the_bonus_needs_a_much_stronger_signal_than_the_number(self):
        agent = self._armed(num=1, scores={"WHALE": 100.0, "SHIP": 60.0})
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())

    def test_letter_overlap_never_takes_the_bonus(self):
        agent = self._armed(num=1, source="offline",
                            scores={"WHALE": 100.0, "SHIP": 100.0})
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())

    def test_number_zero_is_an_unlimited_turn(self):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", 0)
        limit, sweep = agent._turn_capacity()
        self.assertTrue(sweep)
        self.assertEqual(limit, g_mod.TEAM_TOTALS["Red"])

    def test_an_absurd_number_is_also_unlimited_not_a_crash(self):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", 9999)
        self.assertTrue(agent._is_sweep())

    def test_a_non_integer_number_degrades_to_one(self):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", "two")
        self.assertEqual(agent.num, 1)

    def test_an_unlimited_turn_gets_no_bonus_guess(self):
        agent = self._armed(num=0, scores=dict((w, 100.0) for w in BOARD[:12]))
        limit, _sweep = agent._turn_capacity()
        agent.guesses = limit
        self.assertFalse(agent.keep_guessing())

    def test_an_unlimited_turn_demands_a_clearer_signal(self):
        agent = self._armed(num=0, scores={"WHALE": 100.0, "SHIP": 70.0})
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())

    def test_it_stops_once_none_of_our_words_are_left(self):
        board = ["*Red*"] * 9 + list(BOARD[9:])
        agent = make_guesser(FakeLLM(available=False), board=board)
        agent.set_clue("MARINE", 3)
        agent._ranking = [(w, 100.0) for w in board if not w.startswith("*")]
        agent._ranking_key = ("MARINE", agent._is_sweep())
        agent._turn_best = 100.0
        self.assertFalse(agent.keep_guessing())

    def test_the_capacity_is_frozen_for_the_whole_turn(self):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("MARINE", 3)
        first = agent._turn_capacity()
        agent.set_board(["*Red*"] * 8 + list(BOARD[8:]))
        self.assertEqual(agent._turn_capacity(), first)


class TestGuesserRanking(unittest.TestCase):

    def test_an_out_of_vocabulary_clue_spends_the_call(self):
        llm = FakeLLM(['{"WHALE": 95, "SHIP": 40}'])
        agent = make_guesser(llm)
        agent._glove_scores = lambda options: ({}, "oov")
        agent.set_clue("LASTDRINK", 1)
        self.assertEqual(agent.get_answer(), "WHALE")
        self.assertEqual(llm.calls, 1)
        self.assertEqual(agent._ranking_source, "llm")

    def test_one_call_per_turn_however_many_guesses(self):
        llm = FakeLLM(['{"WHALE": 95, "SHIP": 90, "BEACH": 88}'])
        agent = make_guesser(llm)
        agent._glove_scores = lambda options: ({}, "oov")
        agent.set_clue("XENOMORPH", 3)
        agent.get_answer()
        agent.set_board(["*Red*" if w == "WHALE" else w for w in BOARD])
        agent.get_answer()
        self.assertEqual(llm.calls, 1)

    def test_a_confident_cosine_ranking_costs_nothing(self):
        llm = FakeLLM(['{"SHIP": 99}'])
        agent = make_guesser(llm)

        def scores(options):
            out = dict((w, 0.05) for w in options)
            out[options[0]] = 0.62
            return out, "glove"

        agent._glove_scores = scores
        agent.set_clue("MARINE", 1)
        agent.get_answer()
        self.assertEqual(llm.calls, 0)
        self.assertEqual(agent._ranking_source, "glove")

    def test_a_mush_cosine_ranking_is_rescued(self):
        llm = FakeLLM(['{"SHIP": 99}'])
        agent = make_guesser(llm)
        agent._glove_scores = lambda options: (
            dict((w, 0.05) for w in options), "glove")
        agent.set_clue("NAIL", 1)
        self.assertEqual(agent.get_answer(), "SHIP")
        self.assertEqual(llm.calls, 1)

    def test_a_photo_finish_between_the_top_two_is_rescued(self):
        llm = FakeLLM(['{"SHIP": 99}'])
        agent = make_guesser(llm)

        def scores(options):
            out = dict((w, 0.05) for w in options)
            out[options[0]] = 0.400
            out[options[1]] = 0.395     # inside TRUST_MIN_GAP
            return out, "glove"

        agent._glove_scores = scores
        agent.set_clue("NAIL", 1)
        agent.get_answer()
        self.assertEqual(llm.calls, 1)

    def test_a_failed_rescue_leaves_the_cosine_ranking_in_place(self):
        llm = FakeLLM([None])
        agent = make_guesser(llm)

        def scores(options):
            out = dict((w, 0.05) for w in options)
            out["SHIP"] = 0.10
            return out, "glove"

        agent._glove_scores = scores
        agent.set_clue("NAIL", 1)
        self.assertEqual(agent.get_answer(), "SHIP")

    def test_unrankable_words_are_promoted_only_over_genuine_noise(self):
        """``PLATYPUS`` is GloVe rank 68860: unscoreable, so scored last.

        When everything the cache *can* price is noise, the word it cannot
        price is the better bet -- our codemaster only issues a clue that weak
        when its target is the word neither of us can see.  When something
        priceable is a real association, it is not.
        """
        board = ["PLATYPUS"] + list(BOARD[1:])
        llm = FakeLLM([None])

        agent = make_guesser(llm, board=board)
        agent._glove_scores = self._scores_with_unrankable(agent, top=0.05)
        agent.set_clue("MAMMAL", 1)
        self.assertEqual(agent.get_answer(), "PLATYPUS")

        agent = make_guesser(llm, board=board)
        agent._glove_scores = self._scores_with_unrankable(agent, top=0.30)
        agent.set_clue("MAMMAL", 1)
        self.assertEqual(agent.get_answer(), "SHIP")

    @staticmethod
    def _scores_with_unrankable(agent, top):
        def scores(options):
            out = dict((w, 0.02) for w in options if w != "PLATYPUS")
            out["SHIP"] = top
            out["PLATYPUS"] = -1.0
            agent._unrankable = ["PLATYPUS"]
            return out, "glove"
        return scores

    def test_words_the_model_never_mentioned_stay_guessable(self):
        llm = FakeLLM(['{"WHALE": 95}'])
        agent = make_guesser(llm)
        agent._glove_scores = lambda options: ({}, "oov")
        agent.set_clue("LASTDRINK", 3)
        picks = set()
        board = list(BOARD)
        for _ in range(3):
            word = agent.get_answer()
            picks.add(word)
            board = ["*Red*" if w == word else w for w in board]
            agent.set_board(board)
        self.assertEqual(len(picks), 3)

    def test_a_raising_client_does_not_reach_the_engine(self):
        agent = make_guesser(FakeLLM(raises=True))
        agent._glove_scores = lambda options: ({}, "oov")
        agent.set_clue("LASTDRINK", 1)
        self.assertIn(agent.get_answer(), BOARD)

    def test_usage_summary_is_serialisable(self):
        agent = make_guesser(FakeLLM(available=False))
        summary = agent.usage_summary()
        for field in ("calls", "rescue_calls", "oov_clues", "thin_clues"):
            self.assertIn(field, summary)


@needs_vectors
class TestGuesserEmbedding(unittest.TestCase):

    def test_it_really_ranks_on_meaning(self):
        agent = make_guesser(FakeLLM(available=False))
        agent.set_clue("OCEAN", 1)
        self.assertIn(agent.get_answer(), ("WHALE", "SHIP", "BEACH"))

    def test_a_known_clue_costs_no_call(self):
        llm = FakeLLM(['{"KING": 99}'])
        agent = make_guesser(llm)
        agent.set_clue("MONARCH", 1)
        agent.get_answer()
        self.assertEqual(agent._ranking_source, "glove")
        self.assertEqual(llm.calls, 0)


# ---------------------------------------------------------------------------
# The pair, end to end, with no network at all
# ---------------------------------------------------------------------------

class TestPairPlaysAGame(unittest.TestCase):

    def test_a_full_game_is_legal_and_terminates(self):
        codemaster = make_codemaster(FakeLLM(available=False))
        guesser = make_guesser(FakeLLM(available=False))
        board = list(BOARD)
        colours = dict(zip(BOARD, KEY))
        turns = 0
        while turns < 40:
            turns += 1
            codemaster.set_game_state(board, list(KEY))
            clue, number = codemaster.get_clue()
            self.assertTrue(cm_mod.clue_is_legal(clue, board))
            self.assertGreaterEqual(number, 1)
            guesser.set_board(list(board))
            guesser.set_clue(clue, number)
            while True:
                word = guesser.get_answer()
                if word is None:
                    break
                self.assertIn(word, board)
                board = ["*%s*" % colours[w] if w == word else w
                         for w in board]
                if colours[word] != "Red":
                    break
                if not guesser.keep_guessing():
                    break
                guesser.set_board(list(board))
            if all(colours[w] != "Red" for w in board if not str(w).startswith("*")):
                break
            if any(str(w) == "*Assassin*" for w in board):
                break
        self.assertLess(turns, 40, "the pair failed to finish a game")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
