"""Config sweeps over a fixed seed set.

Every config in a sweep plays the **same boards**, so the comparison is paired
and a difference of half a turn is not just seed luck.  Results are dumped to
JSON per config and summarised into one table with the numbers that decide
adoption: mean score, assassin rate (which must stay at 0), mean clue number,
and the API spend the run cost us.

Usage::

    python -m harness.sweep --spec sweeps/numbers.json --seeds 0-29 --jobs 6

The spec is a JSON object mapping a config name to the kwargs handed to the red
codemaster and guesser::

    {"baseline": {},
     "greedy":   {"cmr": {"majority_fraction": 0.34}},
     "greedier": {"cmr": {"majority_fraction": 0.34, "bonus_guess_weight": 0.5},
                  "gr":  {"sweep_confidence": 0.85}}}

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from harness import arena, stats

OBIRDY = {
    "codemaster": "players.codemaster_obirdy.AICodemaster",
    "guesser": "players.guesser_obirdy.AIGuesser",
}


def result_filename(name: str, pool: str = "default",
                    single_team: bool = True) -> str:
    """Per-arm result file, keyed by everything that changes the games.

    A whole battery runs the same config names against different pools and
    tracks, so a name-only filename lets a later arm silently overwrite an
    earlier one's raw records.  That is not a cosmetic problem: the 2026-07-31
    battery lost the per-game record of the one Pidgeot assassin death on the
    default pool that way, and the forensics afterwards had nothing on disk to
    read.
    """
    track = "solo" if single_team else "duel"
    return "sweep_%s_%s_%s.json" % (name, str(pool or "default"), track)


def run_config(name: str, config: Dict[str, Any], seeds: List[int],
               pool: str = "default", single_team: bool = True,
               n_jobs: int = 4, out_dir: str = "results_local",
               blue: Dict[str, str] = None) -> Dict[str, Any]:
    """Play ``seeds`` under one config and return its summary."""
    started = time.time()
    results = arena.run_batch(
        red_codemaster=config.get("red_cm", OBIRDY["codemaster"]),
        red_guesser=config.get("red_g", OBIRDY["guesser"]),
        blue_codemaster=(blue or {}).get("codemaster"),
        blue_guesser=(blue or {}).get("guesser"),
        seeds=seeds,
        single_team=single_team,
        pool=pool,
        n_jobs=n_jobs,
        game_name=name,
        cmr_kwargs=config.get("cmr") or {},
        gr_kwargs=config.get("gr") or {},
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, result_filename(name, pool, single_team))
        with open(path, "w") as handle:
            json.dump(results, handle, indent=2, default=str)
    summary = stats.summarize(results)
    summary["_name"] = name
    summary["_config"] = config
    summary["_wall_s"] = time.time() - started
    return summary


def format_table(summaries: List[Dict[str, Any]]) -> str:
    """One row per config: the numbers adoption actually turns on."""
    header = ("%-16s %7s %8s %9s %8s %8s %9s"
              % ("config", "games", "score", "95% CI", "assassin",
                 "clue no", "cost $"))
    lines = [header, "-" * len(header)]
    for summary in summaries:
        single = summary.get("single_team") or {}
        two = summary.get("two_team") or {}
        score = single.get("score") or {}
        numbers = (summary.get("clue_numbers") or {}).get("red_cm") or {}
        api = summary.get("api_usage") or {}
        if single:
            headline = score.get("mean")
            interval = "%s-%s" % (_short(score.get("ci_low")),
                                  _short(score.get("ci_high")))
            rate = (single.get("assassin") or {}).get("rate")
        else:
            headline = (two.get("red_win") or {}).get("rate")
            interval = "win rate"
            rate = (two.get("assassin") or {}).get("rate")
        lines.append("%-16s %7d %8s %9s %8s %8s %9s"
                     % (summary["_name"],
                        summary.get("completed", 0),
                        _short(headline),
                        interval,
                        _short(rate, 3),
                        _short(numbers.get("mean")),
                        _short(api.get("usd"), 3)))
    return "\n".join(lines)


def _short(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return ("%." + str(digits) + "f") % value


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sweep agent configs.")
    parser.add_argument("--spec", required=True, help="JSON file of configs")
    parser.add_argument("--seeds", default="0-29")
    parser.add_argument("--pool", default="default")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--two-team", action="store_true")
    parser.add_argument("--blue-cm", default=None)
    parser.add_argument("--blue-g", default=None)
    parser.add_argument("--out-dir", default="results_local")
    parser.add_argument("--report", default=None, help="write the table here")
    args = parser.parse_args(argv)

    with open(args.spec) as handle:
        spec = json.load(handle)

    seeds = arena.parse_seeds(args.seeds)
    blue = None
    if args.blue_cm or args.blue_g:
        blue = {"codemaster": args.blue_cm, "guesser": args.blue_g}

    summaries = []
    for name, config in spec.items():
        sys.stderr.write("== %s (%d seeds)\n" % (name, len(seeds)))
        sys.stderr.flush()
        summary = run_config(
            name, config, seeds, pool=args.pool,
            single_team=not args.two_team, n_jobs=args.jobs,
            out_dir=args.out_dir, blue=blue)
        summaries.append(summary)
        sys.stderr.write("   score %s  assassin %s  cost $%s\n" % (
            _short(((summary.get("single_team") or {}).get("score") or {}).get("mean")),
            _short(((summary.get("single_team") or {}).get("assassin") or {}).get("rate"), 3),
            _short((summary.get("api_usage") or {}).get("usd"), 3)))
        sys.stderr.flush()

    table = format_table(summaries)
    print(table)
    if args.report:
        with open(args.report, "w") as handle:
            handle.write(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
