"""Turn arena result files into self-contained replay records for the viewer.

An arena result (``results_local/*.json``) records the *final* board, where
every revealed cell has been overwritten with ``*Red*`` / ``*Blue*`` /
``*Civilian*`` / ``*Assassin*`` and every unrevealed cell still holds its word.
That is enough to watch a game only if you already know the key grid, and the
key grid of the unrevealed cells is nowhere in the file.

It is recoverable, though: ``game.Game.__init__`` seeds ``random`` with the
game's seed, shuffles the word pool, takes 25, then shuffles the key grid, and
``harness.secret_pool.generate_board`` reproduces that call sequence exactly.
So we regenerate ``(words, key_grid)`` from ``(pool, seed)`` and then *check*
the reconstruction against everything the file does record:

* every unrevealed cell must still hold the regenerated word at that index;
* every revealed cell's marker must equal the regenerated key at that index;
* replaying ``move_history`` against the regenerated key must produce exactly
  the recorded final board, and the recorded winner.

A game that fails any of those is dropped (or, with ``--allow-mismatch``, kept
and flagged) rather than silently shipped -- a wrong key grid would be a
replay of a game nobody played.

Usage
-----
    python -m harness.export_replay results_local/gauntlet_mirror.json --open
    python -m harness.export_replay results_local/eval_c_default.json#14,19
    python -m harness.export_replay --starter --into-viewer tools/viewer.html

Python 3.9 compatible.  Offline: reads JSON, writes JSON/HTML, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from harness import secret_pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER_HTML = os.path.join(REPO_ROOT, "tools", "viewer.html")
REPLAY_DIR = os.path.join(REPO_ROOT, "results_local", "replays")

FORMAT_NAME = "codenames-replay"
FORMAT_VERSION = 1

EMBED_START = "/*CN_EMBED_START*/"
EMBED_END = "/*CN_EMBED_END*/"

# Marker -> single-letter key code.
COLOR_CODE = {"Red": "R", "Blue": "B", "Civilian": "C", "Assassin": "A"}
MARKERS = {"*Red*": "R", "*Blue*": "B", "*Civilian*": "C", "*Assassin*": "A"}

# Ladder names from docs/versions.md, so the viewer can say "oBirdy vs Rattata"
# instead of "players.codemaster_heuristic.AICodemaster".
AGENT_NAMES = {
    "codemaster_obirdy": "oBirdy",
    "guesser_obirdy": "oBirdy",
    "codemaster_glove": "Abra",
    "guesser_glove": "Abra",
    "codemaster_heuristic": "Rattata",
    "guesser_heuristic": "Rattata",
    "codemaster_random": "Magikarp",
    "guesser_random": "Magikarp",
    "codemaster_GPT": "GPT",
    "guesser_GPT": "GPT",
}


# ---------------------------------------------------------------------------
# The curated starter set that ships embedded in tools/viewer.html
# ---------------------------------------------------------------------------
# (source stem, game index, label, one-line blurb)
STARTER_SET: List[Tuple[str, int, str, str]] = [
    ("reeval_c_fixed", 10, "The PAGEANT death",
     "PAGEANT 2 pulls PRINCESS, then STATE -- the assassin. The board that made "
     "us build the clue-safety nets."),
    ("val_c_pidgeot", 10, "The PAGEANT dodge",
     "Same seed, same board, after the fix. The danger sensor flagged PRESENT, "
     "the turn went out as GIFT 1, and blue eventually handed us the win."),
    ("eval_c_default", 14, "The SADDLE disaster",
     "SADDLE 1 finds MOUNT, and then a bonus guess nobody needed walks into AIR."),
    ("eval_c_default", 19, "DRILL into the pilot",
     "Another bonus guess off a stale board: TRAIN was right, PILOT was the assassin."),
    ("val_x_pidgeot", 7, "Two clues and out",
     "oBirdy calling for a stranger's guesser. Six moves, then Abra takes a bonus "
     "guess and finds DECK."),
    ("val_a_pidgeot", 0, "Six clues, nine words",
     "The clean solo run: no civilians, no blues, no wasted turns."),
    ("eval_b_pidgeotto", 2, "NINTENDO 3, slang board",
     "The generalisation pool -- HOGWARTS, KIRBY, MULTIVERSE -- swept in six turns."),
    ("val_c_pidgeot", 14, "One word short",
     "A duel that goes the distance and ends 8-8 with blue closing it out first."),
    ("reeval_c_fixed", 17, "The HORN handoff",
     "VOLUME 2 gets SOUND, then hands blue a word, and blue never gives it back."),
    ("val_c_pidgeot", 18, "The long grind",
     "Seventeen turns, four civilians, and red still gets home."),
    ("gauntlet_mirror", 2, "Mirror match",
     "oBirdy on both sides of the table. Whoever moves first should win, and "
     "on this board red does, 9-6."),
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def friendly_agent(path: Optional[str]) -> str:
    """'players.codemaster_heuristic.AICodemaster' -> 'Rattata'."""
    if not path:
        return "?"
    parts = str(path).split(".")
    module = parts[-2] if len(parts) >= 2 else parts[0]
    if module in AGENT_NAMES:
        return AGENT_NAMES[module]
    stem = module.replace("codemaster_", "").replace("guesser_", "")
    return stem or module


# ---------------------------------------------------------------------------
# Reconstruction + verification
# ---------------------------------------------------------------------------

class ReplayError(Exception):
    pass


def reconstruct(game: Dict) -> Tuple[List[str], List[str]]:
    """Regenerate ``(words, key_grid)`` for a recorded game."""
    pool = game.get("pool") or "default"
    if pool == "custom":
        raise ReplayError("game used a custom word pool that is not reproducible")
    board = secret_pool.generate_board(pool, int(game["seed"]))
    return board["words"], board["key_grid"]


def verify(game: Dict, words: Sequence[str], key: Sequence[str]) -> List[str]:
    """Return a list of problems; empty means the reconstruction is sound."""
    problems: List[str] = []
    final = game.get("board") or []

    if len(words) != 25 or len(key) != 25:
        problems.append("regenerated board is not 25 cells")
        return problems
    if len(final) != 25:
        problems.append("recorded final board is not 25 cells")
        return problems
    if len(set(words)) != 25:
        problems.append("regenerated board has duplicate words")

    counts = {c: list(key).count(c) for c in ("Red", "Blue", "Civilian", "Assassin")}
    if counts != {"Red": 9, "Blue": 8, "Civilian": 7, "Assassin": 1}:
        problems.append("key grid composition is %s" % counts)

    # 1. every cell of the recorded final board agrees with the reconstruction
    for i, cell in enumerate(final):
        if cell in MARKERS:
            if COLOR_CODE[key[i]] != MARKERS[cell]:
                problems.append(
                    "cell %d revealed as %s but key says %s" % (i, cell, key[i]))
        elif cell != words[i]:
            problems.append(
                "cell %d holds %r but reconstruction says %r" % (i, cell, words[i]))

    # 2. replaying the move history against the key reproduces that final board
    index_of = {w: i for i, w in enumerate(words)}
    revealed = [False] * 25
    for move in game.get("move_history") or []:
        if len(move) < 4 or not str(move[0]).endswith("_Guesser"):
            continue
        word = str(move[1]).upper().strip()
        if word not in index_of:
            problems.append("guessed word %r is not on the reconstructed board" % word)
            continue
        i = index_of[word]
        recorded = MARKERS.get(str(move[2]))
        if recorded is None:
            problems.append("cell %d has unknown reveal marker %r" % (i, move[2]))
        elif recorded != COLOR_CODE[key[i]]:
            problems.append(
                "guess %r revealed %s but key says %s" % (word, move[2], key[i]))
        revealed[i] = True

    for i in range(25):
        was_revealed = final[i] in MARKERS
        if was_revealed != revealed[i]:
            problems.append(
                "cell %d (%s): final board says revealed=%s, move history says %s"
                % (i, words[i], was_revealed, revealed[i]))

    return problems


def _clue_seconds(game: Dict) -> List[Optional[float]]:
    """Codemaster thinking time per clue, in clue order (may be empty)."""
    out: List[Optional[float]] = []
    for entry in game.get("latencies") or []:
        if entry.get("method") == "get_clue":
            try:
                out.append(round(float(entry.get("seconds") or 0.0), 2))
            except (TypeError, ValueError):
                out.append(None)
    return out


def build_moves(game: Dict, words: Sequence[str], key: Sequence[str]) -> List[Dict]:
    """Normalise ``move_history`` into self-describing move objects."""
    index_of = {w: i for i, w in enumerate(words)}
    seconds = _clue_seconds(game)
    moves: List[Dict] = []
    clue_no = 0
    for move in game.get("move_history") or []:
        role = str(move[0])
        team = "R" if role.startswith("Red") else "B"
        if role.endswith("_Codemaster"):
            entry = {"t": "clue", "team": team, "clue": str(move[1]),
                     "n": int(move[2])}
            if clue_no < len(seconds) and seconds[clue_no] is not None:
                entry["sec"] = seconds[clue_no]
            clue_no += 1
            moves.append(entry)
        else:
            word = str(move[1]).upper().strip()
            i = index_of[word]
            moves.append({
                "t": "guess", "team": team, "word": word, "i": i,
                "color": COLOR_CODE[key[i]],
                "keep": bool(move[3]) if len(move) > 3 else False,
            })
    return moves


def to_replay(game: Dict, source: str, index: int,
              label: Optional[str] = None, note: Optional[str] = None,
              allow_mismatch: bool = False) -> Dict:
    """Build one replay record, verifying the key-grid reconstruction."""
    words, key = reconstruct(game)
    problems = verify(game, words, key)
    if problems and not allow_mismatch:
        raise ReplayError("; ".join(problems[:4]))

    usage = game.get("usage") or {}

    def model_of(role: str) -> Optional[str]:
        entry = usage.get(role)
        return entry.get("model") if isinstance(entry, dict) else None

    stem = os.path.splitext(os.path.basename(source))[0]
    agents = {
        "red_cm": friendly_agent(game.get("red_codemaster")),
        "red_g": friendly_agent(game.get("red_guesser")),
        "blue_cm": friendly_agent(game.get("blue_codemaster")),
        "blue_g": friendly_agent(game.get("blue_guesser")),
    }
    record = {
        "id": "%s#%d" % (stem, index),
        "source": os.path.basename(source),
        "index": index,
        "seed": int(game["seed"]),
        "pool": game.get("pool") or "default",
        "single_team": bool(game.get("single_team")),
        "words": list(words),
        "key": "".join(COLOR_CODE[c] for c in key),
        "agents": agents,
        "agent_paths": {
            "red_cm": game.get("red_codemaster"),
            "red_g": game.get("red_guesser"),
            "blue_cm": game.get("blue_codemaster"),
            "blue_g": game.get("blue_guesser"),
        },
        "models": {k: v for k, v in (
            ("red_cm", model_of("red_cm")), ("red_g", model_of("red_g")),
            ("blue_cm", model_of("blue_cm")), ("blue_g", model_of("blue_g")),
        ) if v},
        "winner": game.get("winner"),
        "turns": game.get("turns"),
        "red_found": game.get("red_found"),
        "blue_found": game.get("blue_found"),
        "civilians_hit": game.get("civilians_hit"),
        "assassin_hit": bool(game.get("assassin_hit")),
        "assassin_hitter": game.get("assassin_hitter"),
        "illegal_clue_count": game.get("illegal_clue_count") or 0,
        "duration_s": round(float(game["duration_s"]), 1) if game.get("duration_s") else None,
        "moves": build_moves(game, words, key),
    }
    if label:
        record["label"] = label
    if note:
        record["note"] = note
    if problems:
        record["warnings"] = problems
    return record


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_result_file(path: str) -> List[Dict]:
    with open(path, "r") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("games", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            # sweep-style container: {"arm name": [game, ...], ...}
            arms = [value for value in payload.values() if isinstance(value, list)]
            if arms:
                payload = [game for arm in arms for game in arm]
    if not isinstance(payload, list):
        raise ReplayError("%s is not an arena result list" % path)
    games = [g for g in payload if isinstance(g, dict) and "move_history" in g]
    if not games:
        raise ReplayError("%s has no games with a move_history" % path)
    return games


def parse_target(target: str) -> Tuple[str, Optional[List[int]]]:
    """``path`` or ``path#3`` or ``path#3,7`` -> (path, indices or None)."""
    if "#" in target:
        path, _, rest = target.partition("#")
        indices = [int(chunk) for chunk in rest.split(",") if chunk.strip()]
        return path, indices
    return target, None


def collect(targets: Sequence[str], allow_mismatch: bool = False,
            labels: Optional[Dict[Tuple[str, int], Tuple[str, str]]] = None,
            quiet: bool = False) -> Tuple[List[Dict], List[str]]:
    """Export every requested game.  Returns (records, failure messages)."""
    records: List[Dict] = []
    failures: List[str] = []
    labels = labels or {}
    for target in targets:
        path, indices = parse_target(target)
        if not os.path.exists(path):
            failures.append("%s: no such file" % path)
            continue
        try:
            games = load_result_file(path)
        except (ReplayError, ValueError) as exc:
            failures.append("%s: %s" % (path, exc))
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        wanted = indices if indices is not None else range(len(games))
        for index in wanted:
            if index < 0 or index >= len(games):
                failures.append("%s#%d: out of range (%d games)"
                                % (stem, index, len(games)))
                continue
            label, note = labels.get((stem, index), (None, None))
            try:
                records.append(to_replay(games[index], path, index, label, note,
                                         allow_mismatch=allow_mismatch))
            except (ReplayError, KeyError, ValueError) as exc:
                failures.append("%s#%d: %s" % (stem, index, exc))
        if not quiet:
            print("  %-40s %d game(s)" % (os.path.basename(path), len(list(wanted))))
    return records, failures


def envelope(records: Sequence[Dict]) -> Dict:
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "games": list(records),
    }


# ---------------------------------------------------------------------------
# Viewer splicing
# ---------------------------------------------------------------------------

def splice_viewer(viewer_path: str, payload: Dict, out_path: Optional[str] = None) -> str:
    """Write ``viewer_path`` (or a copy) with ``payload`` as its embedded set."""
    with open(viewer_path, "r") as handle:
        html = handle.read()
    start = html.find(EMBED_START)
    end = html.find(EMBED_END)
    if start < 0 or end < 0 or end < start:
        raise ReplayError("%s has no %s .. %s block" % (viewer_path, EMBED_START, EMBED_END))
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    # </script> inside a string literal would close the tag early.
    blob = blob.replace("</", "<\\/")
    body = "%s\nwindow.CN_EMBEDDED = %s;\n%s" % (EMBED_START, blob, EMBED_END)
    html = html[:start] + body + html[end + len(EMBED_END):]
    target = out_path or viewer_path
    directory = os.path.dirname(os.path.abspath(target))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, "w") as handle:
        handle.write(html)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _starter_targets() -> Tuple[List[str], Dict[Tuple[str, int], Tuple[str, str]]]:
    by_file: Dict[str, List[int]] = {}
    labels: Dict[Tuple[str, int], Tuple[str, str]] = {}
    order: List[str] = []
    for stem, index, label, note in STARTER_SET:
        if stem not in by_file:
            by_file[stem] = []
            order.append(stem)
        by_file[stem].append(index)
        labels[(stem, index)] = (label, note)
    targets = ["%s#%s" % (os.path.join(REPO_ROOT, "results_local", stem + ".json"),
                          ",".join(str(i) for i in by_file[stem]))
               for stem in order]
    return targets, labels


def _sort_to_starter_order(records: List[Dict]) -> List[Dict]:
    wanted = [(stem, index) for stem, index, _, _ in STARTER_SET]
    def rank(record):
        key = (os.path.splitext(record["source"])[0], record["index"])
        return wanted.index(key) if key in wanted else len(wanted)
    return sorted(records, key=rank)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export arena results as replays for tools/viewer.html.")
    parser.add_argument("targets", nargs="*",
                        help="result JSON files; append #3 or #3,7 to pick games")
    parser.add_argument("--starter", action="store_true",
                        help="export the curated starter set instead")
    parser.add_argument("-o", "--out",
                        help="replay JSON path (default: <input>.replay.json)")
    parser.add_argument("--into-viewer", nargs="?", const=VIEWER_HTML, default=None,
                        help="rewrite this viewer's embedded games (default tools/viewer.html)")
    parser.add_argument("--open", dest="do_open", action="store_true",
                        help="write a standalone viewer with these games and open it")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="keep games whose key reconstruction disagrees, flagged")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    labels: Dict[Tuple[str, int], Tuple[str, str]] = {}
    if args.starter:
        targets, labels = _starter_targets()
    else:
        targets = args.targets
    if not targets:
        parser.error("give at least one result file, or --starter")

    if not args.quiet:
        print("exporting:")
    records, failures = collect(targets, allow_mismatch=args.allow_mismatch,
                                labels=labels, quiet=args.quiet)
    if args.starter:
        records = _sort_to_starter_order(records)

    for problem in failures:
        print("  FAILED  %s" % problem, file=sys.stderr)
    if not records:
        print("nothing exported", file=sys.stderr)
        return 1

    payload = envelope(records)
    if not args.quiet:
        print("verified %d game(s), key grid reconstructed and replayed clean"
              % len(records))

    wrote: List[str] = []

    if args.out or not (args.into_viewer or args.do_open):
        first_path = parse_target(targets[0])[0]
        default = os.path.splitext(first_path)[0] + ".replay.json"
        out_path = args.out or default
        directory = os.path.dirname(os.path.abspath(out_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=1)
        wrote.append(out_path)

    if args.into_viewer:
        wrote.append(splice_viewer(args.into_viewer, payload))

    if args.do_open:
        stem = os.path.splitext(os.path.basename(parse_target(targets[0])[0]))[0]
        copy_path = os.path.join(REPLAY_DIR, "%s_viewer.html" % stem)
        wrote.append(splice_viewer(VIEWER_HTML, payload, out_path=copy_path))
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(copy_path))

    for path in wrote:
        print("wrote %s" % path)
    return 1 if (failures and not args.allow_mismatch) else 0


if __name__ == "__main__":
    raise SystemExit(main())
