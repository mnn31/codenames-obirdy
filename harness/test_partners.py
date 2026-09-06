"""Offline unit tests for the Abra (GloVe embedding) partner agents.

Abra is partner-zoo infrastructure, not a submission: it is the *semantic
stranger* used to check that our clues survive being handed to a teammate we
did not co-design.  These tests therefore care about two things only:

* it never crashes and never emits an illegal clue, with or without vectors;
* when vectors are present it really does play on meaning.

The vector cache lives in the gitignored ``data/`` directory, so the
embedding-specific tests skip on a fresh checkout rather than fail.

Run from the repo root::

    python -m unittest harness.test_partners -v
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from players import codemaster_glove as cm_mod  # noqa: E402
from players import glove_common  # noqa: E402
from players import guesser_glove as g_mod  # noqa: E402

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

HAVE_VECTORS = glove_common.vectors() is not None
needs_vectors = unittest.skipUnless(
    HAVE_VECTORS, "GloVe cache absent (python -m harness.glove_data)")


class TestGloveCommon(unittest.TestCase):

    def test_clue_legality_rejects_board_derivations(self):
        self.assertFalse(glove_common.clue_is_legal("WHALES", BOARD))
        self.assertFalse(glove_common.clue_is_legal("WHA", BOARD))
        self.assertFalse(glove_common.clue_is_legal("DEEP SEA", BOARD))
        self.assertTrue(glove_common.clue_is_legal("MARINE", BOARD))

    def test_revealed_board_words_stop_constraining_clues(self):
        board = list(BOARD)
        board[BOARD.index("CAR")] = "*Red*"
        self.assertFalse(glove_common.clue_is_legal("CARGO", BOARD))
        self.assertTrue(glove_common.clue_is_legal("CARGO", board))

    def test_missing_cache_is_reported_not_raised(self):
        """A fresh checkout has no data/; that must degrade, not explode."""
        result = glove_common.vectors()
        self.assertTrue(result is None or len(result) == 3)


class TestAbraCodemaster(unittest.TestCase):

    def _agent(self, **kwargs):
        agent = cm_mod.AICodemaster("Red", **kwargs)
        agent.set_game_state(list(BOARD), list(KEY))
        return agent

    def test_constructor_matches_the_framework_signature(self):
        agent = cm_mod.AICodemaster("Blue")
        self.assertEqual(agent.team, "Blue")
        self.assertEqual(agent.opponent, "Red")

    def test_clue_is_legal_and_well_formed(self):
        clue, number = self._agent().get_clue()
        self.assertTrue(glove_common.clue_is_legal(clue, BOARD), clue)
        self.assertGreaterEqual(int(number), 1)
        self.assertLessEqual(int(number), 4)

    def test_empty_board_does_not_crash(self):
        agent = cm_mod.AICodemaster("Red")
        agent.set_game_state([], [])
        clue, number = agent.get_clue()
        self.assertIsInstance(clue, str)
        self.assertGreaterEqual(int(number), 1)

    def test_fallback_runs_when_vectors_are_missing(self):
        agent = self._agent()
        clue, number = agent._fallback_clue()
        self.assertTrue(glove_common.clue_is_legal(clue, BOARD), clue)

    @needs_vectors
    def test_embedding_clue_avoids_the_assassin(self):
        """The clue must not sit closer to POISON than to a Red word."""
        agent = self._agent()
        clue, _ = agent.get_clue()
        _, matrix, index = glove_common.vectors()
        row = index.get(clue.lower())
        self.assertIsNotNone(row, clue)
        own = [w for w, k in zip(BOARD, KEY) if k == "Red"]
        _, own_rows = glove_common.board_rows(own, index)
        _, assassin_rows = glove_common.board_rows(["POISON"], index)
        best_own = max(float(matrix[r].dot(matrix[row])) for r in own_rows)
        assassin = max(float(matrix[r].dot(matrix[row])) for r in assassin_rows)
        self.assertGreater(best_own, assassin)

    @needs_vectors
    def test_margin_tightens_the_number(self):
        loose = self._agent(margin=0.0).get_clue()[1]
        tight = self._agent(margin=0.25).get_clue()[1]
        self.assertGreaterEqual(loose, tight)


class TestAbraGuesser(unittest.TestCase):

    def _agent(self, **kwargs):
        agent = g_mod.AIGuesser("Red", **kwargs)
        agent.set_board(list(BOARD))
        return agent

    def test_answer_is_always_an_unrevealed_board_word(self):
        agent = self._agent()
        board = list(BOARD)
        board[0] = "*Red*"
        agent.set_board(board)
        agent.set_clue("MARINE", 2)
        answer = agent.get_answer()
        self.assertIn(answer, board)
        self.assertNotEqual(answer, "*Red*")

    def test_guess_budget_follows_the_clue_number(self):
        agent = self._agent()
        agent.set_clue("MARINE", 2)
        agent.get_answer()
        self.assertTrue(agent.keep_guessing())     # 1 of 2
        agent.get_answer()
        # 2 of 2 -- only the bonus guess may follow, and only if confident.
        agent.get_answer()
        self.assertFalse(agent.keep_guessing())

    def test_unlimited_clue_number_is_not_treated_as_one(self):
        agent = self._agent()
        agent.set_clue("MARINE", 0)
        self.assertGreater(agent._limit(), 1)

    def test_unknown_clue_word_still_produces_a_guess(self):
        agent = self._agent()
        agent.set_clue("ZZZQQQ", 1)
        self.assertIn(agent.get_answer(), BOARD)

    def test_keep_guessing_never_raises(self):
        agent = g_mod.AIGuesser("Red")
        agent.set_board([])
        agent.set_clue("MARINE", 2)
        self.assertFalse(agent.keep_guessing())

    @needs_vectors
    def test_ranking_is_semantic_not_alphabetical(self):
        agent = self._agent()
        agent.set_clue("OCEAN", 2)
        ranking = agent._rank(agent._options())
        top = [word for word, _ in ranking[:5]]
        self.assertIn("SHIP", top + ["WHALE"])
        self.assertTrue({"WHALE", "SHIP", "BEACH"} & set(top))
        self.assertNotIn("PIANO", top)


class TestAbraPlaysWholeGames(unittest.TestCase):
    """The point of a partner is that it finishes games legally, every time."""

    def test_single_team_batch_completes_without_errors(self):
        from harness import arena, stats
        results = arena.run_batch(
            red_codemaster="players.codemaster_glove.AICodemaster",
            red_guesser="players.guesser_glove.AIGuesser",
            seeds=range(3),
            single_team=True,
        )
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNone(result.get("error"), result.get("traceback"))
        summary = stats.summarize(results)
        self.assertEqual(summary["illegal_clues"]["count"], 0)


class TestSweepEvidence(unittest.TestCase):
    """A battery must not overwrite its own raw records."""

    def test_result_filename_separates_pools_and_tracks(self):
        from harness import sweep

        names = {
            sweep.result_filename("pidgeot", "default", True),
            sweep.result_filename("pidgeot", "slang", True),
            sweep.result_filename("pidgeot", "default", False),
        }
        self.assertEqual(len(names), 3, names)
        self.assertIn("sweep_pidgeot_default_solo.json", names)
        self.assertIn("sweep_pidgeot_default_duel.json", names)


class TestMegaPidgeotBattery(unittest.TestCase):
    """The promotion battery: its plan, its cost, and above all its gate.

    The gate is the whole point -- it is what stops a re-run of a config that
    has already failed twice from being adopted on a margin that seed luck can
    manufacture.  It is a pure function of the recorded results, so all of it
    is testable without spending a credit.
    """

    def setUp(self):
        import json
        import tempfile
        from harness import eval_battery

        self.eb = eval_battery
        self.json = json
        self.tmp = tempfile.mkdtemp(prefix="mega-battery-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_arm(self, arm, pool, single_team, records):
        from harness import sweep

        path = os.path.join(self.tmp,
                            sweep.result_filename(arm, pool, single_team))
        with open(path, "w") as handle:
            self.json.dump(records, handle)
        return path

    @staticmethod
    def _games(n, score, assassin=0, winner="R"):
        out = []
        for i in range(n):
            out.append({"seed": i, "score": score, "winner": winner,
                        "assassin_hit": i < assassin})
        return out

    def _arms(self, incumbent=7.5, challenger=6.5, incumbent_deaths=0,
              challenger_deaths=0):
        eb = self.eb
        return {
            "solo_default": {
                eb.MEGA_DEFAULT_ARM: {"games": 15, "mean_score": incumbent,
                                      "assassin_deaths": incumbent_deaths,
                                      "red_wins": 15},
                eb.MEGA_AMBITIOUS_ARM: {"games": 15, "mean_score": challenger,
                                        "assassin_deaths": challenger_deaths,
                                        "red_wins": 15},
            },
        }

    # -- the plan ----------------------------------------------------------

    def test_the_battery_is_three_runs_and_sixty_games(self):
        battery = self.eb.BATTERIES["mega"]
        self.assertEqual([run.name for run in battery],
                         ["mega_solo_default", "mega_solo_slang",
                          "mega_two_team"])
        self.assertEqual(sum(run.games for run in battery), 60)

    def test_it_is_priced_per_game_at_the_measured_rate(self):
        battery = self.eb.BATTERIES["mega"]
        self.assertAlmostEqual(self.eb.GAME_USD, 0.17)
        for run in battery:
            self.assertIsNotNone(run.usd_per_game)
        self.assertAlmostEqual(sum(run.usd for run in battery), 10.20,
                               places=2)

    def test_both_solo_runs_are_paired_on_the_same_spec(self):
        """An unpaired arm is a different experiment, not a cheaper one."""
        battery = dict((run.name, run) for run in self.eb.BATTERIES["mega"])
        for name in ("mega_solo_default", "mega_solo_slang"):
            run = battery[name]
            self.assertIn(self.eb.MEGA_SPEC, run.argv)
            self.assertEqual([arm for arm, _, _ in run.arms],
                             [self.eb.MEGA_DEFAULT_ARM,
                              self.eb.MEGA_AMBITIOUS_ARM])
        self.assertIn("slang", battery["mega_solo_slang"].argv)
        self.assertNotIn("slang", battery["mega_solo_default"].argv)

    def test_the_spec_files_exist_and_name_the_two_arms(self):
        with open(os.path.join(REPO_ROOT, self.eb.MEGA_SPEC)) as handle:
            spec = self.json.load(handle)
        self.assertEqual(sorted(spec),
                         sorted([self.eb.MEGA_DEFAULT_ARM,
                                 self.eb.MEGA_AMBITIOUS_ARM]))
        self.assertEqual(spec[self.eb.MEGA_DEFAULT_ARM], {})
        self.assertEqual(spec[self.eb.MEGA_AMBITIOUS_ARM],
                         {"cmr": {"preset": "ambitious"}})
        for arm, path in self.eb.MEGA_TWO_TEAM_SPECS.items():
            with open(os.path.join(REPO_ROOT, path)) as handle:
                single = self.json.load(handle)
            self.assertEqual(list(single), [arm])

    def test_the_two_team_run_is_resolved_from_the_solo_result(self):
        run = dict((r.name, r) for r in self.eb.BATTERIES["mega"])["mega_two_team"]
        self.assertIsNotNone(run.resolver)
        argv = self.eb._mega_two_team_argv(self.eb.MEGA_DEFAULT_ARM)
        self.assertIn(self.eb.MEGA_TWO_TEAM_SPECS[self.eb.MEGA_DEFAULT_ARM],
                      argv)
        self.assertIn("--two-team", argv)

    # -- reading the arms back ---------------------------------------------

    def test_arm_stats_counts_games_deaths_and_the_mean(self):
        path = self._write_arm("pidgeot_default", "default", True,
                               self._games(4, 7) + self._games(1, 9,
                                                               assassin=1))
        stats = self.eb.arm_stats(path)
        self.assertEqual(stats["games"], 5)
        self.assertEqual(stats["assassin_deaths"], 1)
        self.assertAlmostEqual(stats["mean_score"], 7.4)

    def test_a_missing_arm_is_zero_games_not_a_crash(self):
        stats = self.eb.arm_stats(os.path.join(self.tmp, "nope.json"))
        self.assertEqual(stats["games"], 0)
        self.assertIsNone(stats["mean_score"])

    def test_a_crashed_game_is_not_counted(self):
        records = self._games(2, 7)
        records.append({"seed": 9, "score": None, "error": "boom"})
        path = self._write_arm("ambitious_nets", "default", True, records)
        self.assertEqual(self.eb.arm_stats(path)["games"], 2)

    def test_a_two_team_arm_has_no_score_but_still_has_wins(self):
        records = [{"seed": i, "score": None, "winner": "R" if i else "B",
                    "assassin_hit": False} for i in range(4)]
        path = self._write_arm("ambitious_nets", "default", False, records)
        stats = self.eb.arm_stats(path)
        self.assertIsNone(stats["mean_score"])
        self.assertEqual(stats["red_wins"], 3)

    # -- picking the arm eval C is spent on --------------------------------

    def test_the_lower_solo_mean_wins_the_eval_c_slot(self):
        self._write_arm("pidgeot_default", "default", True, self._games(4, 8))
        self._write_arm("ambitious_nets", "default", True, self._games(4, 6))
        self.assertEqual(self.eb.winning_solo_arm(self.tmp),
                         self.eb.MEGA_AMBITIOUS_ARM)

    def test_a_tie_confirms_the_incumbent(self):
        self._write_arm("pidgeot_default", "default", True, self._games(4, 7))
        self._write_arm("ambitious_nets", "default", True, self._games(4, 7))
        self.assertEqual(self.eb.winning_solo_arm(self.tmp),
                         self.eb.MEGA_DEFAULT_ARM)

    def test_a_challenger_that_died_never_wins_the_slot(self):
        self._write_arm("pidgeot_default", "default", True, self._games(4, 8))
        self._write_arm("ambitious_nets", "default", True,
                        self._games(4, 5, assassin=1))
        self.assertEqual(self.eb.winning_solo_arm(self.tmp),
                         self.eb.MEGA_DEFAULT_ARM)

    def test_an_unrun_battery_confirms_the_incumbent(self):
        self.assertEqual(self.eb.winning_solo_arm(self.tmp),
                         self.eb.MEGA_DEFAULT_ARM)

    # -- the gate ----------------------------------------------------------

    def test_the_margin_is_seven_tenths_of_a_turn(self):
        self.assertAlmostEqual(self.eb.MEGA_SOLO_MARGIN, 0.7)

    def test_a_clean_win_by_the_margin_promotes(self):
        verdict = self.eb.promotion_verdict(self._arms(7.5, 6.8))
        self.assertTrue(verdict["promote"], verdict["reasons"])
        self.assertAlmostEqual(verdict["solo_margin"], 0.7)

    def test_a_win_short_of_the_margin_does_not_promote(self):
        verdict = self.eb.promotion_verdict(self._arms(7.5, 6.9))
        self.assertFalse(verdict["promote"])
        self.assertIn("solo margin", verdict["reasons"][0])

    def test_the_incumbent_winning_does_not_promote(self):
        self.assertFalse(self.eb.promotion_verdict(
            self._arms(6.5, 7.5))["promote"])

    def test_a_death_in_the_challenger_blocks_a_winning_margin(self):
        verdict = self.eb.promotion_verdict(
            self._arms(8.0, 6.0, challenger_deaths=1))
        self.assertFalse(verdict["promote"])
        self.assertTrue(any("assassin" in r for r in verdict["reasons"]))

    def test_a_death_in_the_shipped_arm_also_blocks(self):
        """The run is then not clean enough to promote anything on."""
        verdict = self.eb.promotion_verdict(
            self._arms(8.0, 6.0, incumbent_deaths=1))
        self.assertFalse(verdict["promote"])

    def test_a_death_in_any_later_stage_blocks(self):
        arms = self._arms(8.0, 6.0)
        arms["two_team"] = {self.eb.MEGA_AMBITIOUS_ARM: {
            "games": 10, "mean_score": None, "assassin_deaths": 1,
            "red_wins": 8}}
        self.assertFalse(self.eb.promotion_verdict(arms)["promote"])

    def test_an_unmeasured_battery_never_promotes(self):
        verdict = self.eb.promotion_verdict({})
        self.assertFalse(verdict["promote"])
        self.assertEqual(verdict["verdict"], "Pidgeot stands")

    def test_slang_carries_no_margin_but_still_carries_the_deaths(self):
        """Half the slang pool is outside the table, so one net is down."""
        arms = self._arms(7.5, 6.8)
        arms["solo_slang"] = {self.eb.MEGA_DEFAULT_ARM: {
            "games": 10, "mean_score": 7.0, "assassin_deaths": 0,
            "red_wins": 10}}
        arms["solo_slang"][self.eb.MEGA_AMBITIOUS_ARM] = {
            "games": 10, "mean_score": 9.9, "assassin_deaths": 0,
            "red_wins": 10}
        self.assertTrue(self.eb.promotion_verdict(arms)["promote"])
        arms["solo_slang"][self.eb.MEGA_AMBITIOUS_ARM]["assassin_deaths"] = 1
        self.assertFalse(self.eb.promotion_verdict(arms)["promote"])

    def test_the_verdict_renders_without_any_results_on_disk(self):
        text = self.eb.format_verdict(out_dir=self.tmp)
        self.assertIn("PIDGEOT STANDS", text)

    # -- the plan still prints ---------------------------------------------

    def test_both_batteries_print(self):
        self.assertIn("MEGA PIDGEOT", self.eb.format_plan(
            self.eb.BATTERIES["mega"], self.eb.MEGA_TITLE,
            self.eb.MEGA_PRICING))
        self.assertIn("PIDGEOT VALIDATION BATTERY", self.eb.format_plan())

    def test_printing_the_plan_spends_nothing(self):
        self.assertEqual(self.eb._main(["--battery", "mega"]), 0)
        self.assertEqual(self.eb._main(["--battery", "mega", "--run", "all"]),
                         2)


class TestRaceBatteryGate(unittest.TestCase):
    """The duel-track gate.

    Two of its four conditions are things no win rate can express: a death
    while ahead blocks even a battery that won everything, and beating an
    absolute threshold is not enough without beating the paired control arm
    that played the same boards.  All of it is a pure function of the recorded
    results, so all of it is testable offline.
    """

    def setUp(self):
        from harness import eval_battery

        self.eb = eval_battery

    @staticmethod
    def _arm(games, wins, deaths=0, illegal=0):
        return {"games": games, "red_wins": wins,
                "win_rate": wins / float(games) if games else None,
                "red_assassin_deaths": deaths, "illegal_clues": illegal,
                "mean_red_found": None, "mean_blue_found": None}

    def _arms(self, on=6, off=1, rattata=9, **kwargs):
        return {"abra_on": self._arm(12, on, **kwargs),
                "abra_off": self._arm(12, off),
                "rattata_on": self._arm(10, rattata),
                "mirror": self._arm(4, 2)}

    @staticmethod
    def _audit(deaths=0, indefensible=0):
        return {"deaths": deaths, "defensible": deaths - indefensible,
                "indefensible": indefensible, "reports": []}

    def test_a_clean_improvement_ships(self):
        verdict = self.eb.race_verdict(self._arms(), self._audit())
        self.assertTrue(verdict["promote"], verdict["reasons"])

    def test_beating_the_threshold_without_beating_the_control_does_not(self):
        """Twelve fresh seeds could be easier; the paired arm cannot be."""
        verdict = self.eb.race_verdict(self._arms(on=6, off=6), self._audit())
        self.assertFalse(verdict["promote"])
        self.assertIn("does not beat", verdict["reasons"][0])

    def test_beating_the_control_without_the_threshold_does_not_either(self):
        verdict = self.eb.race_verdict(self._arms(on=3, off=1), self._audit())
        self.assertFalse(verdict["promote"])
        self.assertTrue(any("gate wants" in r for r in verdict["reasons"]))

    def test_the_do_not_harm_floor_blocks_a_regression(self):
        verdict = self.eb.race_verdict(self._arms(rattata=7), self._audit())
        self.assertFalse(verdict["promote"])
        self.assertTrue(any("Rattata" in r for r in verdict["reasons"]))
        self.assertTrue(self.eb.race_verdict(self._arms(rattata=8),
                                             self._audit())["promote"])

    def test_a_death_taken_while_projected_lost_does_not_block(self):
        """The whole claim of the round: that trade is the correct one."""
        verdict = self.eb.race_verdict(self._arms(deaths=1),
                                       self._audit(deaths=1))
        self.assertTrue(verdict["promote"], verdict["reasons"])
        self.assertEqual(verdict["deaths"], 1)

    def test_a_death_taken_while_level_or_ahead_blocks_everything(self):
        verdict = self.eb.race_verdict(self._arms(deaths=1),
                                       self._audit(deaths=1, indefensible=1))
        self.assertFalse(verdict["promote"])
        self.assertTrue(any("level or ahead" in r for r in verdict["reasons"]))

    def test_an_illegal_clue_anywhere_blocks(self):
        arms = self._arms()
        arms["mirror"]["illegal_clues"] = 1
        self.assertFalse(self.eb.race_verdict(arms, self._audit())["promote"])

    def test_an_unmeasured_battery_never_ships(self):
        verdict = self.eb.race_verdict({}, self._audit())
        self.assertFalse(verdict["promote"])
        self.assertEqual(verdict["verdict"], "Pidgeot stands")

    def test_the_abra_seeds_cover_the_recorded_losses(self):
        """race_off has to reproduce 0-3 before the new seeds mean anything."""
        start, stop = self.eb.RACE_ABRA_SEEDS.split("-")
        self.assertLessEqual(int(start), 100)
        self.assertGreaterEqual(int(stop), 102)

    def test_the_race_battery_prints_and_spends_nothing(self):
        text = self.eb.format_plan(self.eb.BATTERIES["race"],
                                   self.eb.RACE_TITLE, self.eb.RACE_PRICING)
        self.assertIn("RACE AWARENESS", text)
        self.assertIn("race_abra_duel", text)
        self.assertEqual(self.eb._main(["--battery", "race"]), 0)
        self.assertEqual(self.eb._main(["--battery", "race", "--run", "all"]),
                         2)

    def test_the_verdict_renders_without_any_results_on_disk(self):
        import tempfile

        tmp = tempfile.mkdtemp(prefix="race-battery-")
        try:
            text = self.eb.format_race_verdict(out_dir=tmp)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertIn("PIDGEOT STANDS", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
