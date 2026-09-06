# tools/ — the replay viewer

`viewer.html` is a spectator's view of games we have already played: a 5x5
board, the clue banner, and a play button. It is one file, it makes **zero
network requests**, and it works by double-clicking it. No server, no build
step, no install.

Eleven games ship embedded in it. Anything else you want to watch goes in by
drag-and-drop, so a new arena run never needs the HTML rebuilt.

---

## Just watch something

```
open tools/viewer.html          # macOS; or double-click it in Finder
```

| key | does |
|---|---|
| `space` | play / pause |
| `←` `→` | step one move |
| `⇧←` `⇧→` | jump a whole turn |
| `Home` `End` | start / end of the game |
| `s` | spymaster view — tints every unrevealed tile with its key colour |
| `r` | restart the game |
| `j` `k` | next / previous game |
| `[` `]` | slower / faster autoplay |

The scrubber and the commentary lines are also clickable — clicking a line
jumps the board to that move.

## Watch a run that just finished

```
python -m harness.export_replay results_local/gauntlet_mirror.json --open
```

That verifies the key grid, builds a standalone copy of the viewer with those
games already loaded at `results_local/replays/gauntlet_mirror_viewer.html`,
and opens it. The copy is self-contained, so nothing in `tools/` changes and
you can hand the file to somebody.

If you would rather keep the tab you already have open, export the JSON and
drop it on the window instead:

```
python -m harness.export_replay results_local/gauntlet_vs_abra.json
# -> results_local/gauntlet_vs_abra.replay.json ; drag it onto the viewer
```

Dropping a **raw** arena result does not work, and the viewer will say so. Raw
results do not contain the key grid — see below.

---

## The exporter

```
python -m harness.export_replay TARGET... [options]
```

A `TARGET` is a result file, optionally with the games you want:

```
results_local/eval_c_default.json          # all 20 games
results_local/eval_c_default.json#14       # just game 14
results_local/eval_c_default.json#14,19    # two of them
```

| option | does |
|---|---|
| `-o PATH` | where to write the replay JSON (default `<input>.replay.json`) |
| `--open` | build a standalone viewer with these games and open it |
| `--into-viewer [PATH]` | rewrite the embedded set inside `tools/viewer.html` |
| `--starter` | export the curated ten instead of reading targets |
| `--allow-mismatch` | keep games that fail verification, flagged, instead of dropping them |

Several files at once are fine; they land in one replay file:

```
python -m harness.export_replay results_local/gauntlet_*.json -o results_local/replays/gauntlet.replay.json
```

Both the `{"arm": [game, ...]}` sweep containers and plain game lists are
accepted.

To change which games ship inside the viewer, edit `STARTER_SET` at the top of
`harness/export_replay.py` and re-run:

```
python -m harness.export_replay --starter --into-viewer
```

---

## Where the key grid comes from

An arena result records the board *after* the game. Revealed cells have been
overwritten with `*Red*` / `*Blue*` / `*Civilian*` / `*Assassin*`; unrevealed
cells still hold their word. So the colours of everything nobody guessed are
simply not in the file, and you cannot replay a game without them.

They are recoverable because the board is a pure function of the seed.
`game.Game.__init__` seeds `random` with the game's seed, shuffles the word
pool, takes 25, then shuffles the key grid, and
`harness.secret_pool.generate_board` reproduces that call sequence exactly.

The exporter regenerates `(words, key_grid)` from `(pool, seed)` and then
checks it against everything the file *does* record:

1. every unrevealed cell still holds the regenerated word at that index;
2. every revealed cell's marker equals the regenerated key at that index;
3. replaying `move_history` against the regenerated key produces exactly the
   recorded final board, and the recorded winner.

A game that fails any of those is dropped rather than shipped. A wrong key grid
would be a replay of a game nobody played. Every result file in
`results_local/` currently reconstructs clean — 531 games across 47 files,
including the gauntlet run.

---

## Checking the viewer without a browser

`viewer.html?selftest=1` replaces the page with a pass/fail table and writes
the same result to the console and to `window.CN_SELFTEST_RESULT`. It asserts,
for every embedded game, that the board is 25 unique words, that the key is
9 red / 8 blue / 7 civilian / 1 assassin, that each guess agrees with both the
word list and the key, and that replaying the timeline reproduces the recorded
score, turn count and winner.

All the replay logic lives in the `CN` object as side-effect-free functions
that never touch the DOM, so if you ever want it in CI you can pull the second
`<script>` block out of the file and run it under `node` with a stub `document`
— `runSelfTest` is the only part that renders anything.
