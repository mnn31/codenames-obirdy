"""Aggregation and reporting for arena results.

Pure standard library (no numpy/scipy) so the harness runs anywhere the
framework runs.  Python 3.9 compatible.

Metrics
-------
* single-team: mean score (turns, loss = 25) with a 95% confidence interval
* two-team: win-rate tables with Wilson 95% intervals
* assassin-hit rate, illegal-clue rate (must be 0), error rate
* per-move latency percentiles (p50/p90/p95/p99/max), overall and per role
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Two-sided t critical values at 95% for df 1..30; normal approximation beyond.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
Z95 = 1.959964

#: USD per million tokens, (input, output), for cost estimates only.  These are
#: list prices we pay for our own evaluation runs; they are not part of any
#: submitted agent.  Unknown models fall back to the priciest entry so an
#: estimate is never optimistic.
MODEL_PRICES = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
FALLBACK_PRICE = (5.00, 25.00)


def model_price(model: str):
    """(input $/Mtok, output $/Mtok) for a model id, prefix-matched."""
    name = str(model or "")
    if name in MODEL_PRICES:
        return MODEL_PRICES[name]
    for known, price in MODEL_PRICES.items():
        if name.startswith(known):
            return price
    return FALLBACK_PRICE


def clue_number_summary(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-role clue-number mean and histogram.

    How many words a codemaster asks for per clue is the single most direct
    lever on single-team score, so it is worth reading straight off every run
    rather than inferring it from turns taken.
    """
    by_role: Dict[str, List[int]] = {}
    for result in results:
        for clue in result.get("clues") or []:
            try:
                number = int(clue.get("number"))
            except (TypeError, ValueError):
                continue
            by_role.setdefault(str(clue.get("role", "?")), []).append(number)
    out: Dict[str, Any] = {}
    for role, numbers in sorted(by_role.items()):
        histogram: Dict[str, int] = {}
        for number in numbers:
            histogram[str(number)] = histogram.get(str(number), 0) + 1
        out[role] = {
            "n": len(numbers),
            "mean": (sum(numbers) / float(len(numbers))) if numbers else None,
            "histogram": histogram,
        }
    return out


def usage_summary(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-game agent usage into calls, tokens and estimated USD."""
    calls = retries = failures = 0
    input_tokens = output_tokens = 0
    cost = 0.0
    models: Dict[str, int] = {}
    games = 0
    for result in results:
        blocks = result.get("usage") or {}
        if blocks:
            games += 1
        for block in blocks.values():
            calls += int(block.get("calls") or 0)
            retries += int(block.get("retries") or 0)
            failures += int(block.get("failures") or 0)
            tin = int(block.get("input_tokens") or 0)
            tout = int(block.get("output_tokens") or 0)
            input_tokens += tin
            output_tokens += tout
            model = str(block.get("model") or "?")
            models[model] = models.get(model, 0) + int(block.get("calls") or 0)
            price_in, price_out = model_price(model)
            cost += (tin * price_in + tout * price_out) / 1e6
    return {
        "games": games,
        "calls": calls,
        "retries": retries,
        "failures": failures,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": cost,
        "usd_per_game": (cost / games) if games else None,
        "models": models,
    }


def t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T95.get(df, Z95)


# ---------------------------------------------------------------------------
# Small statistical helpers
# ---------------------------------------------------------------------------

def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Mean with a 95% t-interval (normal approximation for n > 31)."""
    data = [float(v) for v in values if v is not None]
    n = len(data)
    out = {"n": n, "mean": None, "sd": None, "sem": None,
           "ci_low": None, "ci_high": None, "margin": None,
           "min": None, "max": None}
    if n == 0:
        return out
    mean = sum(data) / n
    out["mean"] = mean
    out["min"] = min(data)
    out["max"] = max(data)
    if n == 1:
        out["sd"] = 0.0
        out["sem"] = 0.0
        out["ci_low"] = out["ci_high"] = mean
        out["margin"] = 0.0
        return out
    variance = sum((v - mean) ** 2 for v in data) / (n - 1)
    sd = math.sqrt(variance)
    sem = sd / math.sqrt(n)
    margin = t_critical(n - 1) * sem
    out.update({"sd": sd, "sem": sem, "margin": margin,
                "ci_low": mean - margin, "ci_high": mean + margin})
    return out


def wilson_interval(successes: int, total: int) -> Dict[str, Any]:
    """Wilson score 95% interval -- well behaved at 0% and 100%."""
    if total <= 0:
        return {"rate": None, "n": 0, "successes": 0,
                "ci_low": None, "ci_high": None}
    p = successes / float(total)
    z2 = Z95 * Z95
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2 * total)) / denom
    spread = (Z95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / denom
    return {
        "rate": p,
        "n": total,
        "successes": successes,
        "ci_low": max(0.0, centre - spread),
        "ci_high": min(1.0, centre + spread),
    }


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile; ``q`` in [0, 100]."""
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def latency_summary(seconds: Sequence[float]) -> Dict[str, Any]:
    data = [float(s) for s in seconds if s is not None]
    if not data:
        return {"n": 0}
    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "p50": percentile(data, 50),
        "p90": percentile(data, 90),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": max(data),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _pairing(result: Dict[str, Any], side: str) -> str:
    cm = str(result.get("%s_codemaster" % side) or "?").rsplit(".", 2)[-2:]
    g = str(result.get("%s_guesser" % side) or "?").rsplit(".", 2)[-2:]
    return "%s + %s" % (".".join(cm), ".".join(g))


def summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of arena game dicts into a report-ready summary."""
    results = list(results)
    total = len(results)
    errors = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]

    single = [r for r in ok if r.get("single_team")]
    twoteam = [r for r in ok if not r.get("single_team")]

    all_latencies: List[float] = []
    by_role: Dict[str, List[float]] = {}
    illegal_total = 0
    clue_total = 0
    illegal_examples: List[Dict[str, Any]] = []

    for r in ok:
        for entry in r.get("latencies") or []:
            seconds = entry.get("seconds")
            if seconds is None:
                continue
            all_latencies.append(seconds)
            by_role.setdefault(entry.get("role", "?"), []).append(seconds)
        illegal_total += int(r.get("illegal_clue_count") or 0)
        clue_total += int(r.get("clue_count") or 0)
        for bad in (r.get("illegal_clues") or [])[:3]:
            if len(illegal_examples) < 10:
                item = dict(bad)
                item["seed"] = r.get("seed")
                illegal_examples.append(item)

    summary: Dict[str, Any] = {
        "games": total,
        "completed": len(ok),
        "errors": len(errors),
        "error_examples": [
            {"seed": r.get("seed"), "error": r.get("error")} for r in errors[:5]
        ],
        "pools": sorted({str(r.get("pool")) for r in results}),
        "assassin": wilson_interval(
            sum(1 for r in ok if r.get("assassin_hit")), len(ok)),
        "illegal_clues": {
            "count": illegal_total,
            "clues": clue_total,
            "rate": (illegal_total / clue_total) if clue_total else None,
            "examples": illegal_examples,
        },
        "latency": {
            "overall": latency_summary(all_latencies),
            "by_role": {role: latency_summary(vals)
                        for role, vals in sorted(by_role.items())},
        },
        "game_duration_s": latency_summary(
            [r.get("duration_s") for r in ok if r.get("duration_s") is not None]),
        "single_team": None,
        "two_team": None,
        "clue_numbers": clue_number_summary(ok),
        "api_usage": usage_summary(ok),
    }

    if single:
        scores = [r["score"] for r in single if r.get("score") is not None]
        wins = sum(1 for r in single if r.get("winner") == "R")
        won_turns = [r["turns"] for r in single
                     if r.get("winner") == "R" and r.get("turns") is not None]
        summary["single_team"] = {
            "games": len(single),
            "score": mean_ci(scores),
            "win": wilson_interval(wins, len(single)),
            "turns_when_won": mean_ci(won_turns),
            "assassin": wilson_interval(
                sum(1 for r in single if r.get("assassin_hit")), len(single)),
            "red_found": mean_ci([r.get("red_found") for r in single]),
        }

    if twoteam:
        red_wins = sum(1 for r in twoteam if r.get("winner") == "R")
        blue_wins = sum(1 for r in twoteam if r.get("winner") == "B")
        table: Dict[str, Dict[str, Any]] = {}
        for r in twoteam:
            for side, code in (("red", "R"), ("blue", "B")):
                key = _pairing(r, side)
                row = table.setdefault(key, {"games": 0, "wins": 0})
                row["games"] += 1
                if r.get("winner") == code:
                    row["wins"] += 1
        for key, row in table.items():
            row.update(wilson_interval(row["wins"], row["games"]))
        summary["two_team"] = {
            "games": len(twoteam),
            "red_win": wilson_interval(red_wins, len(twoteam)),
            "blue_win": wilson_interval(blue_wins, len(twoteam)),
            "turns": mean_ci([r.get("turns") for r in twoteam
                              if r.get("turns") is not None]),
            "assassin": wilson_interval(
                sum(1 for r in twoteam if r.get("assassin_hit")), len(twoteam)),
            "by_pairing": table,
        }

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return ("%." + str(digits) + "f") % value


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "%.1f%%" % (100.0 * value)


def _rate_line(label: str, block: Dict[str, Any], width: int = 22) -> str:
    if not block or block.get("rate") is None:
        return "  %-*s n/a" % (width, label)
    return "  %-*s %s   (%d/%d, 95%% CI %s - %s)" % (
        width, label, _pct(block["rate"]), block["successes"], block["n"],
        _pct(block["ci_low"]), _pct(block["ci_high"]))


def format_report(results_or_summary, title: str = "arena results") -> str:
    """Render a clean, printable report from results or a summary dict."""
    if isinstance(results_or_summary, dict) and "games" in results_or_summary \
            and "latency" in results_or_summary:
        summary = results_or_summary
    else:
        summary = summarize(results_or_summary)

    lines: List[str] = []
    rule = "=" * 68
    lines.append(rule)
    lines.append(title)
    lines.append(rule)
    lines.append("  games                  %d  (completed %d, errored %d)"
                 % (summary["games"], summary["completed"], summary["errors"]))
    lines.append("  word pools             %s" % ", ".join(summary["pools"]))

    for example in summary.get("error_examples", []):
        lines.append("    ! seed %s: %s" % (example["seed"], example["error"]))

    single = summary.get("single_team")
    if single:
        score = single["score"]
        lines.append("")
        lines.append("SINGLE TEAM  (score = turns to find all 9 red, loss = 25)")
        lines.append("  games                  %d" % single["games"])
        lines.append("  mean score             %s  95%% CI [%s, %s]  (sd %s)"
                     % (_fmt(score["mean"]), _fmt(score["ci_low"]),
                        _fmt(score["ci_high"]), _fmt(score["sd"])))
        lines.append("  score range            %s - %s"
                     % (_fmt(score["min"]), _fmt(score["max"])))
        lines.append(_rate_line("win rate", single["win"]))
        lines.append("  turns when won         %s (n=%d)"
                     % (_fmt(single["turns_when_won"]["mean"]),
                        single["turns_when_won"]["n"]))
        lines.append("  mean red words found   %s"
                     % _fmt(single["red_found"]["mean"]))
        lines.append(_rate_line("assassin rate", single["assassin"]))

    two = summary.get("two_team")
    if two:
        lines.append("")
        lines.append("TWO TEAMS")
        lines.append("  games                  %d" % two["games"])
        lines.append(_rate_line("red win rate", two["red_win"]))
        lines.append(_rate_line("blue win rate", two["blue_win"]))
        lines.append("  mean turns/game        %s  95%% CI [%s, %s]"
                     % (_fmt(two["turns"]["mean"]), _fmt(two["turns"]["ci_low"]),
                        _fmt(two["turns"]["ci_high"])))
        lines.append(_rate_line("assassin rate", two["assassin"]))
        if two["by_pairing"]:
            lines.append("")
            lines.append("  win rate by pairing")
            width = max(24, min(72, max(len(k) for k in two["by_pairing"])))
            lines.append("    %-*s %6s %6s %-18s"
                         % (width, "codemaster + guesser", "games", "wins",
                            "win rate"))
            for key in sorted(two["by_pairing"],
                              key=lambda k: -(two["by_pairing"][k]["rate"] or 0)):
                row = two["by_pairing"][key]
                lines.append("    %-*s %6d %6d %s [%s, %s]"
                             % (width, key[:width], row["games"], row["wins"],
                                _pct(row["rate"]), _pct(row["ci_low"]),
                                _pct(row["ci_high"])))

    illegal = summary["illegal_clues"]
    lines.append("")
    lines.append("ROBUSTNESS")
    lines.append("  clues issued           %d" % illegal["clues"])
    lines.append("  illegal clues          %d  (%s)  <- must be 0"
                 % (illegal["count"], _pct(illegal["rate"])))
    for example in illegal["examples"][:5]:
        lines.append("    ! seed %s %s clue=%r: %s"
                     % (example.get("seed"), example.get("role"),
                        example.get("clue"), example.get("reason")))
    lines.append(_rate_line("assassin rate (all)", summary["assassin"]))

    for role, block in (summary.get("clue_numbers") or {}).items():
        if not block.get("n"):
            continue
        spread = " ".join("%sx%d" % (number, count) for number, count
                          in sorted(block["histogram"].items(),
                                    key=lambda item: int(item[0])))
        lines.append("  clue number %-10s mean %s   (%s)"
                     % (role, _fmt(block["mean"]), spread))

    lat = summary["latency"]["overall"]
    lines.append("")
    lines.append("LATENCY  (seconds per agent call)")
    if lat.get("n"):
        lines.append("  calls                  %d" % lat["n"])
        lines.append("  mean / p50 / p90       %s / %s / %s"
                     % (_fmt(lat["mean"], 4), _fmt(lat["p50"], 4),
                        _fmt(lat["p90"], 4)))
        lines.append("  p95 / p99 / max        %s / %s / %s"
                     % (_fmt(lat["p95"], 4), _fmt(lat["p99"], 4),
                        _fmt(lat["max"], 4)))
        for role, block in summary["latency"]["by_role"].items():
            lines.append("    %-20s n=%-6d p50=%s p95=%s max=%s"
                         % (role, block["n"], _fmt(block["p50"], 4),
                            _fmt(block["p95"], 4), _fmt(block["max"], 4)))
    else:
        lines.append("  (no calls recorded)")

    dur = summary["game_duration_s"]
    if dur.get("n"):
        lines.append("")
        lines.append("  wall clock / game      mean %s s, p95 %s s, max %s s"
                     % (_fmt(dur["mean"], 3), _fmt(dur["p95"], 3),
                        _fmt(dur["max"], 3)))

    api = summary.get("api_usage") or {}
    if api.get("calls"):
        lines.append("")
        lines.append("API USAGE  (our evaluation spend, estimated)")
        lines.append("  models                 %s"
                     % ", ".join("%s x%d" % (m, n)
                                 for m, n in sorted(api["models"].items())))
        lines.append("  calls / retries / fail %d / %d / %d"
                     % (api["calls"], api["retries"], api["failures"]))
        lines.append("  tokens in / out        %d / %d"
                     % (api["input_tokens"], api["output_tokens"]))
        lines.append("  estimated cost         $%s  ($%s per game)"
                     % (_fmt(api["usd"], 4), _fmt(api["usd_per_game"], 4)))

    lines.append(rule)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise an arena results JSON file.")
    parser.add_argument("results", help="JSON file written by arena.py --out")
    parser.add_argument("--json", action="store_true",
                        help="print the summary as JSON instead of a report")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    with open(args.results) as handle:
        results = json.load(handle)

    if args.json:
        print(json.dumps(summarize(results), indent=2, default=str))
    else:
        print(format_report(results, title=args.title or args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
