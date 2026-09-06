"""Judge a recorded duel's race -- and every assassin death in it -- offline.

The race-awareness gate is not "did we die?" but "was the risk that killed us
priced correctly at the moment it was taken?".  In the two-team track the score
is binary, so an assassin death in a game we were already projected to lose
costs nothing that losing the race would not have cost anyway; the same death
in a game we were winning throws away a won game.  Those are opposite verdicts
about the same event, and only the recorded move history can tell them apart.

This module replays a recorded game turn by turn and asks the codemaster's own
:func:`race_state` -- literally the same function, imported, never re-derived --
where the race stood *before* each of our clues.  It then locates the clue that
was in play when the assassin was revealed and reports the projection at that
moment.

Nothing here calls an API or plays a game.  It reads ``results_local/*.json``
written by ``harness.arena`` / ``harness.sweep``.

Usage::

    python -m harness.race_audit results_local/gauntlet_vs_abra.json
    python -m harness.race_audit results_local/*.json --json
    python -m harness.race_audit results_local/val_c_pidgeot.json --team Red

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from players.codemaster_obirdy import (  # noqa: E402
    TEAM_TOTALS, race_state,
)

ASSASSIN_MARKER = "*Assassin*"


def _other(team: str) -> str:
    return "Blue" if team == "Red" else "Red"


def walk_game(game: Dict[str, Any], team: str = "Red") -> List[Dict[str, Any]]:
    """One entry per ``team`` codemaster turn, with the race state before it.

    ``own_left`` / ``opp_left`` are reconstructed from the reveals in the
    history rather than from the final board, because the state that matters is
    the one the agent saw at the time.  A colour is "left" until *anybody*
    reveals it -- a word we hand the opponent by guessing it for them is gone
    from their pile just as surely as one they earned.
    """
    history = game.get("move_history") or []
    opponent = _other(team)
    own_marker, opp_marker = "*%s*" % team, "*%s*" % opponent
    own_found = opp_found = 0
    prefix = "%s_Codemaster" % team
    turns: List[Dict[str, Any]] = []

    for index, move in enumerate(history):
        if not move:
            continue
        actor = str(move[0])
        if actor == prefix:
            state = race_state(history[:index], team,
                               TEAM_TOTALS.get(team, 9) - own_found,
                               TEAM_TOTALS.get(opponent, 8) - opp_found)
            turns.append({
                "turn": len(turns) + 1,
                "clue": move[1] if len(move) > 1 else None,
                "number": move[2] if len(move) > 2 else None,
                "race": state,
                "assassin": False,
            })
        elif actor.endswith("_Guesser") and len(move) >= 3:
            revealed = str(move[2])
            if revealed == own_marker:
                own_found += 1
            elif revealed == opp_marker:
                opp_found += 1
            elif revealed == ASSASSIN_MARKER and actor.startswith(team) and turns:
                # The clue in play is the most recent one we gave.
                turns[-1]["assassin"] = True
    return turns


def audit_game(game: Dict[str, Any], team: str = "Red") -> Dict[str, Any]:
    """Per-game verdict: escalation profile, and any death's justification."""
    turns = walk_game(game, team)
    scales = [t["race"]["scale"] for t in turns if t["race"] is not None]
    death = next((t for t in turns if t["assassin"]), None)
    verdict: Dict[str, Any] = {
        "seed": game.get("seed"),
        "winner": game.get("winner"),
        "red_found": game.get("red_found"),
        "blue_found": game.get("blue_found"),
        "turns": len(turns),
        "scales": [round(s, 2) for s in scales],
        "max_scale": round(max(scales), 2) if scales else 0.0,
        "escalated_turns": sum(1 for s in scales if s > 0.0),
        "assassin_death": death is not None,
        "death_turn": None,
        "death_clue": None,
        "death_deficit": None,
        "death_scale": None,
        "death_projected_lost": None,
    }
    if death is not None:
        state = death["race"]
        verdict["death_turn"] = death["turn"]
        verdict["death_clue"] = "%s %s" % (death["clue"], death["number"])
        if state is None:
            # Our opening clue, or a single-team game: there was no race to
            # read, so nothing priced this risk and it cannot be defended.
            verdict["death_deficit"] = None
            verdict["death_scale"] = 0.0
            verdict["death_projected_lost"] = False
        else:
            verdict["death_deficit"] = state["deficit"]
            verdict["death_scale"] = round(state["scale"], 2)
            verdict["death_projected_lost"] = state["deficit"] > 0
    return verdict


def audit_file(path: str, team: str = "Red") -> Dict[str, Any]:
    with open(path, "r") as handle:
        games = json.load(handle)
    if isinstance(games, dict):
        games = games.get("games") or []
    duels = [g for g in games if not g.get("single_team")]
    verdicts = [audit_game(g, team) for g in duels]
    deaths = [v for v in verdicts if v["assassin_death"]]
    escalated = [v for v in verdicts if v["escalated_turns"] > 0]
    return {
        "path": path,
        "games": len(verdicts),
        "skipped_single_team": len(games) - len(duels),
        "wins": sum(1 for v in verdicts if v["winner"] == ("R" if team == "Red"
                                                           else "B")),
        "games_with_escalation": len(escalated),
        "deaths": len(deaths),
        "deaths_projected_lost": sum(1 for v in deaths
                                     if v["death_projected_lost"]),
        "deaths_while_level_or_ahead": sum(1 for v in deaths
                                           if not v["death_projected_lost"]),
        "verdicts": verdicts,
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = ["== %s" % report["path"],
             "   %d duel games, %d wins, %d with escalation, %d assassin deaths"
             % (report["games"], report["wins"],
                report["games_with_escalation"], report["deaths"])]
    if report["skipped_single_team"]:
        lines.append("   (%d single-team games skipped -- no race to read)"
                     % report["skipped_single_team"])
    for verdict in report["verdicts"]:
        marker = " DEATH" if verdict["assassin_death"] else ""
        lines.append("   seed %-5s %s %s-%s max %.2f over %d/%d turns%s"
                     % (verdict["seed"], verdict["winner"],
                        verdict["red_found"], verdict["blue_found"],
                        verdict["max_scale"], verdict["escalated_turns"],
                        verdict["turns"], marker))
        if verdict["assassin_death"]:
            lines.append("      died on turn %s (%s): deficit %s, escalation "
                         "%.2f -- %s"
                         % (verdict["death_turn"], verdict["death_clue"],
                            verdict["death_deficit"], verdict["death_scale"],
                            "projected lost, the risk was priced"
                            if verdict["death_projected_lost"]
                            else "LEVEL OR AHEAD, the risk was not priced"))
    lines.append("   gate: %d/%d deaths were projected-lost when taken"
                 % (report["deaths_projected_lost"], report["deaths"]))
    return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the two-team race in recorded games.")
    parser.add_argument("paths", nargs="+", help="result JSON files")
    parser.add_argument("--team", default="Red", choices=("Red", "Blue"))
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    args = parser.parse_args(argv)

    reports = [audit_file(path, args.team) for path in args.paths]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            print(format_report(report))
    bad = sum(r["deaths_while_level_or_ahead"] for r in reports)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
