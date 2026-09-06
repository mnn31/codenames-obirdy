"""The eval batteries: what to run, what it costs, and what the result means.

Nothing here calls an API by itself.  ``python -m harness.eval_battery`` prints
the plan -- every command, its game count, its estimated call count and its
estimated spend -- so the battery can be budgeted before a single credit is
spent.  ``--run`` executes it, and deliberately refuses to do so without
``--yes`` so the print-out stays the default.

Two batteries live here:

``--battery pidgeot``
    the validation battery that tagged Pidgeot.  Already run.
``--battery mega``
    the **promotion** battery: paired ``pidgeot_default`` against
    ``ambitious_nets`` (the parked clue-number push, now composed with both
    safety nets) on solo default and solo slang, then eval C on whichever solo
    arm wins.  Its gate is ``promotion_verdict`` below, which is a pure
    function of the recorded results and is unit-tested offline.
``--battery race``
    the **duel-track** battery for race awareness: paired race-on / race-off
    against the Abra GloVe pair (the 0-3 record this round exists because of),
    a do-not-harm regression against the Rattata heuristics, and a mirror
    sanity run.  Its gate is ``race_verdict``, and it reads every assassin
    death through ``harness.race_audit`` so a death can be judged against the
    projection it was taken under rather than merely counted.

Why the Pidgeot estimate is built from *calls* and not from a per-game
average: the danger probe changes how many calls a game makes, so a flat
dollars-per-game figure from the pre-probe runs would under-price exactly the
thing that battery existed to measure.  That is now measured -- the three
validation runs came in at $0.192 (solo), $0.166 (two-team) and $0.176
(cross-pair) per game -- so the mega battery prices per game at ``GAME_USD``.

Usage::

    python -m harness.eval_battery                       # the plan and the bill
    python -m harness.eval_battery --battery mega        # the promotion plan
    python -m harness.eval_battery --json                # machine-readable
    python -m harness.eval_battery --battery mega --run all --yes
    python -m harness.eval_battery --battery mega --verdict   # read the gate

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

#: Mean USD per Anthropic call across the four measured sonnet runs in
#: harness/reports (eval A 0.00466, eval B 0.00539, eval C 0.00457, the
#: withdrawn numbers variant 0.00467).  Per-game spend in those same runs ran
#: $0.120 - $0.189, which is the sanity check on anything computed here.
CALL_USD = 0.0048

#: Calls per game by run shape, decomposed from the measured runs.
#: eval A: 540 calls / 15 games = 36, of which 108 clues x 3 = 324 codemaster
#: (brainstorm + two panel samples) and 216 guesser (2 samples per turn).
CALLS_SOLO_DEFAULT = 36.0        # both agents ours, no probe
CALLS_TWO_TEAM_DEFAULT = 32.0    # only red is on the API
CALLS_CROSS_OUR_CM = 22.0        # our codemaster, a stranger's guesser
CALLS_CROSS_OUR_G = 17.0         # a stranger's codemaster, our guesser
CALLS_PANEL_BATCHED = 36.0
CALLS_PANEL_ISOLATED = 79.0      # panel_isolated=3 -> 9 codemaster calls/clue

#: Shipped Pidgeot: the probe adds *one* call per clue, not one per finalist.
#: A veto costs a second, and the recorded veto rate is low enough that ~1.1
#: per clue is the right planning figure.  Solo runs ~7.2 clues per game,
#: two-team ~6.4, cross-pairing ~5.
CALLS_SOLO_PROBE = CALLS_SOLO_DEFAULT + 8.0
CALLS_TWO_TEAM_PROBE = CALLS_TWO_TEAM_DEFAULT + 7.0
CALLS_CROSS_OUR_CM_PROBE = CALLS_CROSS_OUR_CM + 6.0


#: Measured USD per game on the Pidgeot validation runs, probe included:
#: val_a 0.1917 (solo, 10), val_c 0.1664 (two-team, 20), val_x 0.1764
#: (cross-pair, 8).  Rounded to the blended figure; the solo shape is the
#: expensive one, so a solo-heavy battery should be read as a floor.
GAME_USD = 0.17


class Run(object):

    def __init__(self, name, purpose, argv, arms, optional=False,
                 usd_per_game=None, resolver=None):
        self.name = name
        self.purpose = purpose
        self.argv = list(argv)
        #: ``[(arm name, games, calls per game), ...]``
        self.arms = list(arms)
        self.optional = bool(optional)
        #: When set, price per game instead of per call.
        self.usd_per_game = usd_per_game
        #: ``() -> argv``, called at execution time.  A run whose command
        #: depends on an earlier run's result (eval C on the winning solo arm)
        #: cannot be written down in advance; ``argv`` is then the placeholder
        #: the plan prints and ``resolver`` is what actually runs.
        self.resolver = resolver

    @property
    def games(self):
        return sum(games for _, games, _ in self.arms)

    @property
    def calls(self):
        return sum(games * rate for _, games, rate in self.arms)

    @property
    def usd(self):
        if self.usd_per_game is not None:
            return self.games * self.usd_per_game
        return self.calls * CALL_USD

    @property
    def command(self):
        return " ".join(self.argv)

    def resolved_argv(self):
        if self.resolver is None:
            return list(self.argv)
        return list(self.resolver())


OBIRDY_CM = "players.codemaster_obirdy.AICodemaster"
OBIRDY_G = "players.guesser_obirdy.AIGuesser"
HEURISTIC_CM = "players.codemaster_heuristic.AICodemaster"
HEURISTIC_G = "players.guesser_heuristic.AIGuesser"


#: The Pidgeot validation battery.  Three runs, one purpose each, sized to the
#: cheapest thing that can falsify the change rather than to what would be nice
#: to know.  Each has a *recorded* predecessor on the same seeds, so a
#: single-arm run is a real comparison and a paired A/B would be paying twice
#: for it.
BATTERY = [
    Run(
        "eval_a_solo",
        "matched solo, default pool: the probe costs a call per clue and the "
        "sensor can reject a clue, so the solo score is what could regress. "
        "Compare against reeval_a_fixed (7.13, 0 assassin, seeds 0-14)",
        ["python", "-m", "harness.arena",
         "--seeds", "0-9", "--single-team", "--jobs", "4",
         "--red-cm", OBIRDY_CM, "--red-g", OBIRDY_G,
         "--out", "results_local/pidgeot_a_solo.json"],
        [("shipped", 10, CALLS_SOLO_PROBE)],
    ),
    Run(
        "eval_c_two_team",
        "two-team vs heuristics, seeds 0-19 -- seed 10 is the PAGEANT 2 into "
        "STATE death this round exists to stop, and it must be in the range",
        ["python", "-m", "harness.arena",
         "--seeds", "0-19", "--jobs", "4",
         "--red-cm", OBIRDY_CM, "--red-g", OBIRDY_G,
         "--blue-cm", HEURISTIC_CM, "--blue-g", HEURISTIC_G,
         "--out", "results_local/pidgeot_c_two_team.json"],
        [("shipped", 20, CALLS_TWO_TEAM_PROBE)],
    ),
    Run(
        "eval_x_abra_our_cm",
        "our codemaster + Abra's GloVe guesser, seeds 0-7: the 5-deaths-in-8 "
        "run. Both new safety nets aim here",
        ["python", "-m", "harness.sweep",
         "--spec", "harness/sweeps/abra_our_cm.json",
         "--seeds", "0-7", "--jobs", "4",
         "--report", "harness/reports/eval_x_abra_pidgeot.txt"],
        [("obirdy_cm_abra_g", 8, CALLS_CROSS_OUR_CM_PROBE)],
    ),
    Run(
        "eval_b_slang",
        "eval B protocol on the slang pool -- optional: the sensor is inert "
        "on most slang words (123 of 232 are in the table) and slang was "
        "already 0-assassin in both arms",
        ["python", "-m", "harness.arena",
         "--seeds", "0-9", "--single-team", "--pool", "slang", "--jobs", "4",
         "--red-cm", OBIRDY_CM, "--red-g", OBIRDY_G,
         "--out", "results_local/pidgeot_b_slang.json"],
        [("shipped", 10, CALLS_SOLO_PROBE)],
        optional=True,
    ),
    Run(
        "ab_no_safety",
        "paired control: shipped vs probe-off/sensor-off, if the runs above "
        "come back ambiguous and the change needs attributing",
        ["python", "-m", "harness.sweep",
         "--spec", "harness/sweeps/pidgeot.json",
         "--seeds", "0-9", "--jobs", "4",
         "--report", "harness/reports/pidgeot_ab_no_safety.txt"],
        [("shipped", 10, CALLS_SOLO_PROBE),
         ("no_safety", 10, CALLS_SOLO_DEFAULT)],
        optional=True,
    ),
]


# ---------------------------------------------------------------------------
# Mega Pidgeot: the promotion battery
# ---------------------------------------------------------------------------

#: The two paired arms.  ``ambitious_nets`` is ``preset="ambitious"`` with
#: nothing else changed, which means the shipped danger probe and the shipped
#: embedding sensor are both armed -- the thing neither previous failure of
#: those numbers had.
MEGA_DEFAULT_ARM = "pidgeot_default"
MEGA_AMBITIOUS_ARM = "ambitious_nets"

#: How much better the ambitious arm's solo mean must be, in turns, on the
#: default pool.  Lower is better on the solo score.  0.7 is deliberately
#: larger than the 0.4-turn gap the Pidgeot validation run showed against its
#: own predecessor, because that gap did not survive its own forensics: the
#: probe provably could not change the output in eight of those ten games and
#: those eight carried the same shift.  A margin that a re-run of the identical
#: config can manufacture is not a reason to change the submission.
MEGA_SOLO_MARGIN = 0.7

MEGA_SPEC = "harness/sweeps/mega_pidgeot.json"
MEGA_TWO_TEAM_SPECS = {
    MEGA_DEFAULT_ARM: "harness/sweeps/mega_c_pidgeot_default.json",
    MEGA_AMBITIOUS_ARM: "harness/sweeps/mega_c_ambitious_nets.json",
}
MEGA_SOLO_DEFAULT_SEEDS = "0-14"
MEGA_SOLO_SLANG_SEEDS = "0-9"
MEGA_TWO_TEAM_SEEDS = "0-9"


def _mega_two_team_argv(arm=None):
    """Eval C on one arm.  ``arm`` defaults to whichever solo arm won."""
    if arm is None:
        arm = winning_solo_arm()
    return ["python", "-m", "harness.sweep",
            "--spec", MEGA_TWO_TEAM_SPECS[arm],
            "--seeds", MEGA_TWO_TEAM_SEEDS, "--two-team", "--jobs", "4",
            "--blue-cm", HEURISTIC_CM, "--blue-g", HEURISTIC_G,
            "--report", "harness/reports/mega_c_two_team.txt"]


MEGA_BATTERY = [
    Run(
        "mega_solo_default",
        "paired solo on the default pool, 15 seeds: the arm that decides "
        "everything. Both arms play the same boards, so this is the margin "
        "the gate reads",
        ["python", "-m", "harness.sweep",
         "--spec", MEGA_SPEC,
         "--seeds", MEGA_SOLO_DEFAULT_SEEDS, "--jobs", "4",
         "--report", "harness/reports/mega_a_solo_default.txt"],
        [(MEGA_DEFAULT_ARM, 15, 0.0), (MEGA_AMBITIOUS_ARM, 15, 0.0)],
        usd_per_game=GAME_USD,
    ),
    Run(
        "mega_solo_slang",
        "paired solo on the slang pool, 10 seeds: the sensor is inert on most "
        "slang words, so this is where raised numbers run with one net down",
        ["python", "-m", "harness.sweep",
         "--spec", MEGA_SPEC, "--pool", "slang",
         "--seeds", MEGA_SOLO_SLANG_SEEDS, "--jobs", "4",
         "--report", "harness/reports/mega_b_solo_slang.txt"],
        [(MEGA_DEFAULT_ARM, 10, 0.0), (MEGA_AMBITIOUS_ARM, 10, 0.0)],
        usd_per_game=GAME_USD,
    ),
    Run(
        "mega_two_team",
        "eval C vs the heuristics, 10 seeds, on whichever solo arm won -- the "
        "command is resolved from the solo results at run time",
        _mega_two_team_argv(MEGA_AMBITIOUS_ARM),
        [("winning solo arm", 10, 0.0)],
        usd_per_game=GAME_USD,
        resolver=_mega_two_team_argv,
    ),
]


def _sweep_result_path(arm: str, pool: str, single_team: bool,
                       out_dir: str = "results_local") -> str:
    from harness import sweep as sweep_mod
    return os.path.join(out_dir,
                        sweep_mod.result_filename(arm, pool, single_team))


def arm_stats(path: str) -> Dict[str, Any]:
    """The three numbers the gate turns on, read back from one arm's records.

    Missing file, unfinished games and crashed games all end in ``games`` not
    matching what was asked for, which the gate treats as "not measured"
    rather than as a pass.
    """
    stats = {"path": path, "games": 0, "mean_score": None,
             "assassin_deaths": 0, "red_wins": 0}
    try:
        with open(path) as handle:
            records = json.load(handle)
    except Exception:  # noqa: BLE001 - an absent arm is a real answer here
        return stats
    scores = []
    for record in records or ():
        if record.get("error"):
            continue
        stats["games"] += 1
        if record.get("assassin_hit"):
            stats["assassin_deaths"] += 1
        if record.get("winner") == "R":
            stats["red_wins"] += 1
        if record.get("score") is not None:
            scores.append(float(record["score"]))
    if scores:
        stats["mean_score"] = sum(scores) / float(len(scores))
    return stats


def read_mega_arms(out_dir: str = "results_local") -> Dict[str, Any]:
    """Every arm of the mega battery that is on disk."""
    arms = {}
    for stage, pool, single in (("solo_default", "default", True),
                                ("solo_slang", "slang", True),
                                ("two_team", "default", False)):
        arms[stage] = dict(
            (arm, arm_stats(_sweep_result_path(arm, pool, single, out_dir)))
            for arm in (MEGA_DEFAULT_ARM, MEGA_AMBITIOUS_ARM))
    return arms


def winning_solo_arm(out_dir: str = "results_local") -> str:
    """The solo arm eval C should be spent on: the lower default-pool mean.

    A tie, a missing arm or an arm that killed itself on the assassin all
    resolve to the shipped default, because eval C is a confirmation run and
    confirming the incumbent is the cheaper mistake.
    """
    arms = read_mega_arms(out_dir)["solo_default"]
    challenger = arms[MEGA_AMBITIOUS_ARM]
    incumbent = arms[MEGA_DEFAULT_ARM]
    if challenger["mean_score"] is None or incumbent["mean_score"] is None:
        return MEGA_DEFAULT_ARM
    if challenger["assassin_deaths"]:
        return MEGA_DEFAULT_ARM
    if challenger["mean_score"] < incumbent["mean_score"]:
        return MEGA_AMBITIOUS_ARM
    return MEGA_DEFAULT_ARM


def promotion_verdict(arms: Dict[str, Any],
                      margin: float = MEGA_SOLO_MARGIN) -> Dict[str, Any]:
    """Does ``ambitious_nets`` become Mega Pidgeot?

    Both conditions, no partial credit:

    * its solo mean on the **default pool** beats the shipped arm by at least
      ``margin`` turns.  Slang is measured and reported but does not carry the
      margin: 123 of its 232 words are outside the similarity table, so the
      embedding sensor is half-blind there and a win bought under a disarmed
      net is not the win being claimed;
    * **0 assassin deaths in every arm that ran**, including the shipped one.
      A death in the shipped arm does not excuse the challenger; it means the
      run is not clean enough to promote anything on.
    """
    reasons = []
    solo = arms.get("solo_default") or {}
    incumbent = solo.get(MEGA_DEFAULT_ARM) or {}
    challenger = solo.get(MEGA_AMBITIOUS_ARM) or {}

    measured = 0.0
    if incumbent.get("mean_score") is None or challenger.get("mean_score") is None:
        reasons.append("solo default pool not measured in both arms")
    else:
        measured = incumbent["mean_score"] - challenger["mean_score"]
        if measured < margin:
            reasons.append(
                "solo margin %+.2f turns, gate wants >= %.2f"
                % (measured, margin))

    deaths = []
    for stage, stage_arms in sorted((arms or {}).items()):
        for arm, stats in sorted((stage_arms or {}).items()):
            if stats.get("games") and stats.get("assassin_deaths"):
                deaths.append("%s/%s: %d" % (stage, arm,
                                             stats["assassin_deaths"]))
    if deaths:
        reasons.append("assassin deaths (" + ", ".join(deaths) + ")")

    return {"promote": not reasons,
            "solo_margin": measured,
            "margin_required": margin,
            "reasons": reasons,
            "verdict": ("ambitious_nets promotes to Mega Pidgeot"
                        if not reasons else "Pidgeot stands")}


def format_verdict(arms: Dict[str, Any] = None,
                   out_dir: str = "results_local") -> str:
    arms = read_mega_arms(out_dir) if arms is None else arms
    lines = ["MEGA PIDGEOT PROMOTION GATE", ""]
    header = "%-14s %-18s %6s %8s %9s %9s" % (
        "stage", "arm", "games", "score", "assassin", "red wins")
    lines.extend([header, "-" * len(header)])
    for stage in ("solo_default", "solo_slang", "two_team"):
        for arm in (MEGA_DEFAULT_ARM, MEGA_AMBITIOUS_ARM):
            stats = (arms.get(stage) or {}).get(arm) or {}
            if not stats.get("games"):
                continue
            score = stats.get("mean_score")
            lines.append("%-14s %-18s %6d %8s %9d %9d" % (
                stage, arm, stats["games"],
                "n/a" if score is None else "%.2f" % score,
                stats.get("assassin_deaths", 0), stats.get("red_wins", 0)))
    verdict = promotion_verdict(arms)
    lines.append("")
    lines.append("solo margin (default pool): %+.2f turns, gate wants >= %.2f"
                 % (verdict["solo_margin"], verdict["margin_required"]))
    for reason in verdict["reasons"]:
        lines.append("  blocked: %s" % reason)
    lines.append(verdict["verdict"].upper())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Race awareness: the duel-track battery
# ---------------------------------------------------------------------------
#
# What this is measuring, and why it is not the mega battery.  The mega arm
# raised the clue numbers *everywhere*; this one raises them only in a duel we
# are projected to lose.  The two tracks are scored differently and must be
# judged differently: the single-team score prices an assassin death at 25
# against a mean near 7, so conservatism is worth buying there, while the
# two-team score is binary, so losing 8-3 costs exactly what dying costs and
# conservatism is worth nothing once we are behind.
#
# The evidence this exists to answer is ``results_local/gauntlet_vs_abra.json``:
# 0-3 against the GloVe pair with **no assassin deaths at all**.  We were not
# killed, we were out-paced -- 2.7 words a turn against our 1.0 on seed 100,
# and an 8-8 photo finish lost by one word on seed 102.  The 90% duel record
# behind Pidgeot was measured only against the Rattata heuristics, which score
# about 1.3 words a turn.
RACE_ON_ARM = "race_on"
RACE_OFF_ARM = "race_off"
RACE_SPEC = "harness/sweeps/race_duel.json"
GLOVE_CM = "players.codemaster_glove.AICodemaster"
GLOVE_G = "players.guesser_glove.AIGuesser"

#: 12 seeds against Abra, 10 against Rattata, 4 mirror.  The Abra seeds
#: deliberately *contain* 100-102, the three recorded losses, so the arm can be
#: read against a known result before any of the new seeds are believed.
RACE_ABRA_SEEDS = "100-111"
RACE_RATTATA_SEEDS = "0-9"
RACE_MIRROR_SEEDS = "100-103"

#: Per-game duel pricing.  Only red is on the API against Abra and Rattata, so
#: those cost what the measured two-team run cost ($0.166); a mirror puts both
#: codemasters and both guessers on the API, so it is close to double.
RACE_GAME_USD = 0.17
RACE_MIRROR_GAME_USD = 0.31

#: The do-not-harm floor against the slower pair, and the recorded number it is
#: read against: ``val_c_pidgeot`` went 18/20 over seeds 0-19 and 9/10 over
#: 0-9, with 0 assassin deaths.
RACE_RATTATA_MIN_WIN_RATE = 0.80
#: Against Abra the incumbent is 0-3.  Anything at or above this is a material
#: improvement worth shipping; the target is half the field.
RACE_ABRA_MIN_WINS = 4
RACE_ABRA_TARGET_WINS = 6

RACE_BATTERY = [
    Run(
        "race_abra_duel",
        "paired duel vs the Abra GloVe pair, 12 seeds each arm: the run that "
        "decides everything. Seeds 100-102 are the recorded 0-3, so the "
        "race_off arm has a known answer to reproduce before the other nine "
        "seeds are trusted",
        ["python", "-m", "harness.sweep",
         "--spec", RACE_SPEC,
         "--seeds", RACE_ABRA_SEEDS, "--two-team", "--jobs", "4",
         "--blue-cm", GLOVE_CM, "--blue-g", GLOVE_G,
         "--out-dir", "results_local/race_abra",
         "--report", "harness/reports/race_abra_duel.txt"],
        [(RACE_ON_ARM, 12, 0.0), (RACE_OFF_ARM, 12, 0.0)],
        usd_per_game=RACE_GAME_USD,
    ),
    Run(
        "race_rattata_duel",
        "do-not-harm regression vs the Rattata heuristics, 10 seeds, race "
        "awareness ON only -- val_c_pidgeot already recorded the off arm on "
        "these exact seeds (9/10, 0 assassin), so a second arm would be "
        "paying twice for an answer we have",
        ["python", "-m", "harness.sweep",
         "--spec", "harness/sweeps/mega_c_pidgeot_default.json",
         "--seeds", RACE_RATTATA_SEEDS, "--two-team", "--jobs", "4",
         "--blue-cm", HEURISTIC_CM, "--blue-g", HEURISTIC_G,
         "--out-dir", "results_local/race_rattata",
         "--report", "harness/reports/race_rattata_duel.txt"],
        [("pidgeot_default", 10, 0.0)],
        usd_per_game=RACE_GAME_USD,
    ),
    Run(
        "race_mirror",
        "mirror sanity, 4 seeds: both sides race-aware, both sides on the "
        "API. Two escalating agents in the same game is the one configuration "
        "no other run covers, and it is the cheapest place to find out that "
        "mutual escalation is unstable",
        ["python", "-m", "harness.arena",
         "--seeds", RACE_MIRROR_SEEDS, "--jobs", "4",
         "--red-cm", OBIRDY_CM, "--red-g", OBIRDY_G,
         "--blue-cm", OBIRDY_CM, "--blue-g", OBIRDY_G,
         "--out", "results_local/race_mirror.json"],
        [("mirror", 4, 0.0)],
        usd_per_game=RACE_MIRROR_GAME_USD,
    ),
]

RACE_TITLE = "RACE AWARENESS (DUEL TRACK) VALIDATION BATTERY"
RACE_PRICING = ("priced at $%.2f per duel game (only red on the API) and "
                "$%.2f per mirror game (both sides on it)"
                % (RACE_GAME_USD, RACE_MIRROR_GAME_USD))


def duel_stats(path: str) -> Dict[str, Any]:
    """Win rate and red-side assassin deaths for one recorded duel arm."""
    empty = {"games": 0, "red_wins": 0, "win_rate": None,
             "red_assassin_deaths": 0, "illegal_clues": 0,
             "mean_red_found": None, "mean_blue_found": None}
    try:
        with open(path, "r") as handle:
            games = json.load(handle)
    except Exception:  # noqa: BLE001 - a missing arm is "not measured"
        return empty
    if isinstance(games, dict):
        games = games.get("games") or []
    games = [g for g in games if not g.get("single_team")]
    if not games:
        return empty
    deaths = sum(1 for g in games
                 if g.get("assassin_hit") and g.get("assassin_hitter") == "Red")
    return {
        "games": len(games),
        "red_wins": sum(1 for g in games if g.get("winner") == "R"),
        "win_rate": sum(1 for g in games if g.get("winner") == "R")
        / float(len(games)),
        "red_assassin_deaths": deaths,
        "illegal_clues": sum(int(g.get("illegal_clue_count") or 0)
                             for g in games),
        "mean_red_found": sum(g.get("red_found") or 0
                              for g in games) / float(len(games)),
        "mean_blue_found": sum(g.get("blue_found") or 0
                               for g in games) / float(len(games)),
    }


def read_race_arms(out_dir: str = "results_local") -> Dict[str, Any]:
    """Every arm of the race battery that is on disk."""
    from harness.sweep import result_filename

    abra = os.path.join(out_dir, "race_abra")
    rattata = os.path.join(out_dir, "race_rattata")
    return {
        "abra_on": duel_stats(os.path.join(
            abra, result_filename(RACE_ON_ARM, "default", False))),
        "abra_off": duel_stats(os.path.join(
            abra, result_filename(RACE_OFF_ARM, "default", False))),
        "rattata_on": duel_stats(os.path.join(
            rattata, result_filename("pidgeot_default", "default", False))),
        "mirror": duel_stats(os.path.join(out_dir, "race_mirror.json")),
    }


def race_death_audit(out_dir: str = "results_local") -> Dict[str, Any]:
    """Every red assassin death in the battery, judged against its projection.

    A death is *defensible* only if the race was already projected lost when
    the clue that caused it was given -- which is the whole claim this round
    rests on, and which only the recorded move history can settle.  See
    ``harness/race_audit.py``; the projection comes from the agent's own
    ``race_state``, imported rather than re-derived.
    """
    from harness import race_audit
    from harness.sweep import result_filename

    paths = [
        os.path.join(out_dir, "race_abra",
                     result_filename(RACE_ON_ARM, "default", False)),
        os.path.join(out_dir, "race_abra",
                     result_filename(RACE_OFF_ARM, "default", False)),
        os.path.join(out_dir, "race_rattata",
                     result_filename("pidgeot_default", "default", False)),
        os.path.join(out_dir, "race_mirror.json"),
    ]
    deaths, defensible, reports = 0, 0, []
    for path in paths:
        if not os.path.exists(path):
            continue
        report = race_audit.audit_file(path)
        deaths += report["deaths"]
        defensible += report["deaths_projected_lost"]
        reports.append(report)
    return {"deaths": deaths, "defensible": defensible,
            "indefensible": deaths - defensible, "reports": reports}


def race_verdict(arms: Dict[str, Any],
                 audit: Dict[str, Any] = None) -> Dict[str, Any]:
    """Does race awareness ship?

    Four conditions, no partial credit:

    * **Abra improves materially.** The race arm must beat the paired race-off
      arm outright *and* clear ``RACE_ABRA_MIN_WINS`` of 12.  Beating 0-3 by
      accident is easy; beating the arm that played the same twelve boards is
      not.
    * **Rattata does not regress.** At least ``RACE_RATTATA_MIN_WIN_RATE``,
      against the 9/10 already recorded on those seeds.
    * **No red assassin death anywhere was taken while level or ahead.** A
      death in a game we were projected to lose costs nothing the loss would
      not have cost; the same death in a game we were winning is the feature
      doing exactly what it must never do.
    * **No illegal clues anywhere**, which is an invariant and not a trade.
    """
    reasons = []
    on = arms.get("abra_on") or {}
    off = arms.get("abra_off") or {}
    rattata = arms.get("rattata_on") or {}

    if not on.get("games") or not off.get("games"):
        reasons.append("the paired Abra arms are not both measured")
    else:
        if on["red_wins"] <= off["red_wins"]:
            reasons.append("Abra: race_on %d/%d does not beat race_off %d/%d"
                           % (on["red_wins"], on["games"],
                              off["red_wins"], off["games"]))
        if on["red_wins"] < RACE_ABRA_MIN_WINS:
            reasons.append("Abra: %d wins, gate wants >= %d (target %d)"
                           % (on["red_wins"], RACE_ABRA_MIN_WINS,
                              RACE_ABRA_TARGET_WINS))

    if not rattata.get("games"):
        reasons.append("the Rattata regression arm is not measured")
    elif rattata["win_rate"] < RACE_RATTATA_MIN_WIN_RATE:
        reasons.append("Rattata: %.0f%% win rate, gate wants >= %.0f%%"
                       % (100 * rattata["win_rate"],
                          100 * RACE_RATTATA_MIN_WIN_RATE))

    illegal = sum((stats or {}).get("illegal_clues", 0)
                  for stats in arms.values())
    if illegal:
        reasons.append("%d illegal clues" % illegal)

    audit = audit or {"deaths": 0, "defensible": 0, "indefensible": 0}
    if audit["indefensible"]:
        reasons.append("%d assassin death(s) taken while level or ahead"
                       % audit["indefensible"])

    return {
        "promote": not reasons,
        "abra_wins": on.get("red_wins"),
        "abra_control_wins": off.get("red_wins"),
        "rattata_win_rate": rattata.get("win_rate"),
        "deaths": audit["deaths"],
        "indefensible_deaths": audit["indefensible"],
        "reasons": reasons,
        "verdict": ("race awareness ships as the Mega Pidgeot candidate"
                    if not reasons else "Pidgeot stands"),
    }


def format_race_verdict(arms: Dict[str, Any] = None,
                        out_dir: str = "results_local") -> str:
    arms = read_race_arms(out_dir) if arms is None else arms
    audit = race_death_audit(out_dir)
    lines = ["RACE AWARENESS GATE", ""]
    header = "%-12s %6s %9s %10s %9s %9s" % (
        "arm", "games", "red wins", "win rate", "assassin", "illegal")
    lines.extend([header, "-" * len(header)])
    for name in ("abra_on", "abra_off", "rattata_on", "mirror"):
        stats = arms.get(name) or {}
        if not stats.get("games"):
            continue
        lines.append("%-12s %6d %9d %9.0f%% %9d %9d" % (
            name, stats["games"], stats["red_wins"],
            100 * (stats["win_rate"] or 0.0),
            stats["red_assassin_deaths"], stats["illegal_clues"]))
    verdict = race_verdict(arms, audit)
    lines.append("")
    lines.append("red assassin deaths: %d, of which %d were taken in a game "
                 "already projected lost" % (audit["deaths"],
                                             audit["defensible"]))
    for report in audit["reports"]:
        for entry in report["verdicts"]:
            if entry["assassin_death"]:
                lines.append("  %s seed %s turn %s (%s): deficit %s -- %s"
                             % (os.path.basename(report["path"]),
                                entry["seed"], entry["death_turn"],
                                entry["death_clue"], entry["death_deficit"],
                                "priced" if entry["death_projected_lost"]
                                else "NOT PRICED"))
    for reason in verdict["reasons"]:
        lines.append("  blocked: %s" % reason)
    lines.append(verdict["verdict"].upper())
    return "\n".join(lines)


BATTERIES = {"pidgeot": BATTERY, "mega": MEGA_BATTERY, "race": RACE_BATTERY}


def plan(runs=None) -> Dict[str, Any]:
    runs = list(runs if runs is not None else BATTERY)
    core = [run for run in runs if not run.optional]
    return {
        "call_usd": CALL_USD,
        "runs": [{"name": run.name,
                  "purpose": run.purpose,
                  "command": run.command,
                  "optional": run.optional,
                  "games": run.games,
                  "calls": round(run.calls),
                  "usd": round(run.usd, 2)} for run in runs],
        "core_total": {"games": sum(r.games for r in core),
                       "calls": round(sum(r.calls for r in core)),
                       "usd": round(sum(r.usd for r in core), 2)},
        "grand_total": {"games": sum(r.games for r in runs),
                        "calls": round(sum(r.calls for r in runs)),
                        "usd": round(sum(r.usd for r in runs), 2)},
    }


def format_plan(runs=None, title=None, pricing=None) -> str:
    runs = list(runs if runs is not None else BATTERY)
    # A per-game battery has no call estimate to print, and a column of zeroes
    # reads like a bug rather than like "not how this one is priced".
    per_call = any(run.usd_per_game is None for run in runs)
    columns = ("%-20s %7s %7s %8s  %s" if per_call else "%-20s %7s %8s  %s")
    header = (columns % ("run", "games", "calls", "usd", "arms") if per_call
              else columns % ("run", "games", "usd", "arms"))
    lines = [(title or "PIDGEOT VALIDATION BATTERY")
             + "  (nothing below has been run)",
             pricing or ("priced at $%.4f per call, the mean of the four "
                         "measured sonnet runs" % CALL_USD),
             "",
             header, "-" * len(header)]
    for run in runs:
        arms = ", ".join("%s x%d" % (name, games)
                         for name, games, _ in run.arms)
        name = run.name + ("*" if run.optional else "")
        if per_call:
            lines.append(columns % (name, run.games, round(run.calls),
                                    "%.2f" % run.usd, arms))
        else:
            lines.append(columns % (name, run.games, "%.2f" % run.usd, arms))
    summary = plan(runs)
    lines.append("-" * len(header))
    for label, key, note in (("TOTAL", "core_total", "core battery"),
                             ("TOTAL+", "grand_total",
                              "including * optional runs")):
        totals = summary[key]
        if per_call:
            lines.append(columns % (label, totals["games"], totals["calls"],
                                    "%.2f" % totals["usd"], note))
        else:
            lines.append(columns % (label, totals["games"],
                                    "%.2f" % totals["usd"], note))
    lines.append("")
    for run in runs:
        lines.append("# %s -- %s" % (run.name, run.purpose))
        lines.append(run.command)
        lines.append("")
    lines.append("Run from the repo root with the key exported:")
    lines.append("  set -a; . ./.env; set +a")
    return "\n".join(lines)


MEGA_TITLE = "MEGA PIDGEOT PROMOTION BATTERY"
MEGA_PRICING = ("priced at $%.2f per game, the blended figure measured across "
                "the three Pidgeot validation runs (probe included)" % GAME_USD)


def _select(names: List[str], battery: List[Run] = None) -> List[Run]:
    battery = BATTERY if battery is None else battery
    if not names or "all" in names:
        return [run for run in battery if not run.optional]
    if "everything" in names:
        return list(battery)
    chosen = []
    known = dict((run.name, run) for run in battery)
    for name in names:
        if name not in known:
            raise SystemExit("unknown run %r; known: %s"
                             % (name, ", ".join(sorted(known))))
        chosen.append(known[name])
    return chosen


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Print (or run) an eval battery.")
    parser.add_argument("--battery", default="pidgeot",
                        choices=sorted(BATTERIES),
                        help="which battery (default: pidgeot)")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable plan")
    parser.add_argument("--run", nargs="*", default=None,
                        metavar="NAME",
                        help="execute these runs (or 'all' / 'everything')")
    parser.add_argument("--yes", action="store_true",
                        help="required alongside --run; this spends money")
    parser.add_argument("--verdict", action="store_true",
                        help="read the mega battery's gate off the recorded "
                             "results and stop")
    args = parser.parse_args(argv)

    battery = BATTERIES[args.battery]
    mega = args.battery == "mega"
    race = args.battery == "race"
    title = MEGA_TITLE if mega else (RACE_TITLE if race else None)
    pricing = MEGA_PRICING if mega else (RACE_PRICING if race else None)

    if args.verdict:
        print(format_race_verdict() if race else format_verdict())
        return 0

    if args.run is None:
        if args.json:
            print(json.dumps(plan(battery), indent=2))
        else:
            print(format_plan(battery, title, pricing))
        return 0

    runs = _select(args.run, battery)
    if not args.yes:
        print(format_plan(runs, title, pricing))
        sys.stderr.write(
            "\nrefusing to run without --yes (estimated $%.2f)\n"
            % sum(run.usd for run in runs))
        return 2

    for run in runs:
        run_argv = run.resolved_argv()
        sys.stderr.write("== %s (~$%.2f)\n%s\n"
                         % (run.name, run.usd, " ".join(run_argv)))
        sys.stderr.flush()
        code = subprocess.call(run_argv)
        if code != 0:
            sys.stderr.write("!! %s exited %d; stopping\n" % (run.name, code))
            return code
    if mega:
        print("")
        print(format_verdict())
    if race:
        print("")
        print(format_race_verdict())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
