# harness/ — evaluation infrastructure

Everything in this directory is **ours** and is **not submitted**. It measures agents
that run inside the vendored competition framework in `framework/`, which is the
official 2026 code and is never modified — we only *add* files under
`framework/players/`.

Requirements: Python 3.9+, `colorama` (already needed by the framework). The
`openai` / `anthropic` packages are optional and only imported when the matching
backend is actually used.

```
harness/
  llm_backend.py   pluggable chat backends (OpenAI / Anthropic / Mock)
  arena.py         batch game runner: seeds x pairings, parallel, instrumented
  stats.py         aggregation + printable report (CIs, win rates, latency, spend)
  sweep.py         paired config sweeps over a fixed seed set
  eval_battery.py  the pending eval battery: its commands and its bill
  secret_pool.py   alternative 25-word boards from custom pools (slang/pop-culture)
  glove_data.py    builds the gitignored GloVe cache for the Abra partners
  simtable.py      builds the codemaster's bundled clue/board similarity table
  package_submission.py  cuts the submission zip, key substituted in
  selftest.py      offline end-to-end smoke test (no API keys, no network)
  test_obirdy.py   unit tests for the oBirdy agents (257)
  test_partners.py unit tests for the Abra partner agents (42)
```

`python -m pytest harness/ -q` runs all 299 offline; neither file needs a key
or a network.

## `package_submission.py` — cutting the zip

The organisers will not set a per-team environment variable (two entries both
wanting `ANTHROPIC_API_KEY` would clash), so the tournament key ships *inside*
both agent files. What is committed is the placeholder
`HARDCODED_API_KEY = "OBIRDY-KEY-PLACEHOLDER"`, which the agents ignore because
it does not start with `sk-`; this script is the only thing that ever replaces
it.

```bash
python -m harness.package_submission              # the real thing, key from .env
python -m harness.package_submission --placeholder  # re-cut the testing zip
```

It reads `ANTHROPIC_API_KEY_TOURNAMENT` from `.env` (or `--env-file`), rewrites
exactly one line per agent file, and writes
`submission/obirdy_submission_keyed.zip` laid out the way the organisers unpack
it:

```
codemaster_obirdy.py
guesser_obirdy.py
oBirdy/obirdy_simtable.bin.gz
INSTRUCTIONS.md
```

Then it reopens the finished zip and checks the bytes that will actually be
sent: one `HARDCODED_API_KEY` assignment per file, the key present exactly once
and the placeholder gone. The key is never printed, not even a prefix. A zip
carrying a real key must have `keyed` in its filename — that is the pattern
`.gitignore` matches, and the script refuses any other name.

## `simtable.py` — the codemaster's bundled similarity table

Unlike everything else here, this script's *output* is submitted:
`framework/players/oBirdy/obirdy_simtable.bin.gz` ships in the agent's own
per-team subfolder (which is where the organisers want auxiliary files) and is
committed. It holds GloVe cosines between a clue vocabulary and the board word
pool, so the codemaster can see that a clue pulls harder on the assassin than
on the word its number is buying — the failure the LLM panel is structurally
blind to, because the panel only reports words it happened to rank.

```bash
python -m harness.glove_data --wide   # once: build data/glove_wide.npz
python -m harness.simtable --build --stats
```

Rebuild it whenever the clue vocabulary or the board pool changes. The
codemaster degrades to a no-op without it, so a fresh checkout that has never
run the builder still plays legally — the shipped file just means it never has
to.

**Read the `--wide` cache, not `glove_cache.npz`.** The 60k cache Abra uses
cannot see two thirds of the slang pool: `PLATYPUS` is GloVe rank 68860,
`PIKACHU` 92066, `XENOMORPH` 375895, `SPEEDRUN` 398546. Building against the
shallow cache is still supported (it is the automatic fallback) but produces a
table that covers 123 of the 232 slang words instead of 193. The other 39 are
not in GloVe 6B at all, in any casing or hyphenation — `TIKTOK`, `FORTNITE`,
`BLOCKCHAIN`, `LOOTBOX`, `YEET` and friends postdate the 2014 corpus — and
`simtable.UNREACHABLE_SLANG` names them so no future coverage number pretends
otherwise. The themed pools cost nothing there: `UNREACHABLE_THEMED` is empty,
because every themed word is in GloVe 6B. `HYRULE` (rank 195170) and `DRUMKIT`
(269862) are below the wide cache's 150k prefix, which is why
`glove_data.pool_words` names the themed pools too — rebuild the wide cache
after editing a pool, or those words come back as unpriceable.

The board vocabulary is three separately-ranked banks, because the competition
pool is secret and the pools we have are a sample of it, not the thing itself:

| bank | words | depth per clue | what it is |
|---|---|---|---|
| real pools | 588 | top 64 | the framework pool + the slang pool |
| themed pools | 176 | top 32 | every `THEMED_POOLS` word, and the organisers' board verbatim |
| generated | 949 | top 16 | `tournament_candidates`, seeded on the two above |

Each bank is ranked in its own field rather than merged into one global top-64.
That is not cosmetic: the first build that merged the generated bank cost the
sensor a recorded death (`LADDER`→`BOX`, pool rank 53) by pushing it off the
end of the list, and 176 themed words in the same field would be the same
mistake again. `--no-extra-board` builds the bundled-pools-only table, and
`--candidate-budget` sizes the generated bank.

**Format versions.** The board vocabulary outgrew the original layout, which
packed each stored pair into one 16-bit word (10-bit board index, 6-bit code)
and so could not describe more than 1024 board words. Version 2 (`OBSIM2`)
keeps the indices and the codes in parallel arrays: a full 16-bit index, and a
*smaller* file, because a run of ascending indices and a run of 6-bit codes
each compress much better apart than interleaved. The agent reads both, the
builder writes version 1 whenever the vocabulary still fits in it, and
`TestSimTableFormatVersions` pins both directions.

## Quickstart

```bash
# 30-second sanity check of the entire harness (runs ~30 offline games)
python -m harness.selftest

# 20 single-team games with the offline heuristic pair, 4 processes
python -m harness.arena --seeds 0-19 --single-team --jobs 4

# 50 two-team games on the slang/pop-culture pool, saved for later analysis
python -m harness.arena --seeds 0-49 --pool slang --jobs 8 --out results/slang.json

# Re-report a saved run
python -m harness.stats results/slang.json
python -m harness.stats results/slang.json --json   # machine-readable summary
```

All commands run from the repo root (the `harness` package must be importable).

---

## `arena.py` — batch runner

```python
from harness import arena, stats

results = arena.run_batch(
    red_codemaster="players.codemaster_ours.AICodemaster",
    red_guesser="players.guesser_ours.AIGuesser",
    blue_codemaster="players.codemaster_heuristic.AICodemaster",  # defaults to red's
    blue_guesser="players.guesser_heuristic.AIGuesser",
    seeds=range(100),
    single_team=False,     # True = single-team track (red only)
    pool="slang",          # None/"default"/"slang" | word list | wordlist path
    n_jobs=8,              # multiprocessing (spawn) fan-out
    max_turns=80,          # hard cap; a game that exceeds it is recorded as an error
    cmr_kwargs={"version": "gpt-4o-2024-05-13"},  # forwarded to the red codemaster
)
print(stats.format_report(results))
```

Agents are named by dotted import path exactly as `run_game.py` names them
(`players.<module>.<Class>`), resolved with `framework/` on `sys.path`.

### One result dict per game

| key | meaning |
|---|---|
| `seed`, `single_team`, `pool`, `game_name` | run identity |
| `red_codemaster` / `red_guesser` / `blue_*` | agent import paths |
| `winner` | `"R"` or `"B"` (as the engine reports it) |
| `turns` | the engine's own `turn_counter` (clues issued by *both* teams) |
| `score` | single-team only: `turns` on a win, `25` on a loss |
| `red_found`, `blue_found`, `civilians_hit` | words revealed by colour |
| `assassin_hit`, `assassin_hitter` | whether the assassin was picked, and by which team |
| `illegal_clues`, `illegal_clue_count`, `clue_count` | clue-legality audit |
| `latencies` | `[{role, method, seconds}, ...]` for every agent call |
| `board`, `move_history` | final board and the engine's full move log |
| `duration_s`, `error`, `traceback` | wall clock, and crash details if any |

### How results are captured (no framework edits)

`game.Game` keeps its turn counter as a **local variable** in `run()` and only
surfaces it by handing it to `write_results()`, which appends to
`results/bot_results*.txt` relative to the process cwd. Rather than parse those
files (which would race under multiprocessing), the arena subclasses `Game` and
overrides `write_results` to stash the number on the instance. Side effect: no
`results/` directory is ever created.

Everything else is read straight off the finished `Game` object
(`game_winner`, `words_on_board`, `get_move_history()`).

Per-move latency and clue legality come from generated subclasses of the agent
classes that wrap `get_clue` / `get_answer` / `keep_guessing`. The wrappers keep
the original `__name__`, so nothing downstream can tell the difference.

### Clue legality

The engine does **not** validate clues at all — the bundled `codemaster_GPT`
polices itself. The arena therefore audits every clue independently against the
competition rules: single alphabetic English word, number ≥ 1, and neither
containing nor contained by any unrevealed board word. Any violation lands in
`illegal_clues` with a reason. **The reported illegal-clue rate for our agents
must be 0** — a malformed response is a disqualification at the real event.

### Turn cap

A guesser that returns `None` makes `game.Game` break out of the inner loop
*without* passing the turn, so a buggy agent can hang the engine forever. The
instrumented codemaster raises `TurnCapExceeded` past `max_turns` (default 80);
the game is recorded with an `error` instead of hanging the batch.

---

## `secret_pool.py` — alternative word pools

The real competition pool is secret and may include slang / pop-culture words.
`SLANG_POOL` bundles ~230 such words (`HOGWARTS`, `XENOMORPH`, `TIKTOK`,
`POKEMON`, `CYBERPUNK`, `SPEEDRUN`, …) for generalisation testing.

**Injection approach.** `game.Game.__init__` hard-codes
`open("players/cm_wordlist.txt")` — a path relative to the process cwd, with no
kwarg to override it. The clean fix that leaves the framework untouched is to
build a throwaway sandbox directory containing `players/cm_wordlist.txt` and run
the game with that directory as cwd, keeping `framework/` on `sys.path` for
imports. `arena.run_batch` does this for *every* run (including the default
pool), so behaviour is uniform and `framework/players/cm_wordlist.txt` is only
ever read.

```python
from harness import secret_pool

secret_pool.resolve_pool("slang")            # -> list of words
secret_pool.mixed_pool(fraction_slang=0.3)   # partial-novelty pool
secret_pool.generate_board("slang", seed=7)  # {"words": [...25], "key_grid": [...25]}

sandbox = secret_pool.make_sandbox("slang")  # temp cwd with the pool installed
secret_pool.destroy_sandbox(sandbox)
```

`generate_board` replays the engine's own RNG sequence (seed → shuffle pool →
take 25 → shuffle key grid), so it reproduces exactly the board a given seed will
produce — handy for previewing boards without playing them:

```bash
python -m harness.secret_pool --pool slang --seeds 0-2
```

---

## `stats.py` — aggregation and reporting

```python
from harness import stats

summary = stats.summarize(results)          # nested dict, JSON-serialisable
print(stats.format_report(results, title="v1 vs baseline"))
```

Reported:

- **Single team** — mean score with a 95% t-interval, score range, win rate
  (Wilson interval), mean turns on wins, mean red words found, assassin rate.
- **Two teams** — red/blue win rates with Wilson intervals, mean turns per game,
  and a win-rate table broken down by codemaster+guesser pairing.
- **Robustness** — clues issued, illegal-clue count and rate (with examples),
  overall assassin rate, crash count with example tracebacks.
- **Latency** — p50/p90/p95/p99/max seconds per agent call, overall and per role
  (`red_cm`, `red_g`, `blue_cm`, `blue_g`), plus wall clock per game. The event's
  soft limit is 60 s per response, so watch p95/max on the LLM agents.

Pure standard library — no numpy/scipy — so it runs anywhere the framework does.

---

## `llm_backend.py` — pluggable chat models

```python
from harness.llm_backend import get_backend

llm = get_backend("anthropic", model="claude-opus-5", timeout=25.0, max_retries=3)
text = llm.chat(
    [{"role": "user", "content": "Give a one-word clue for APPLE and TREE."}],
    system="You are a Codenames codemaster.",
    max_tokens=64,
    timeout=15.0,       # per attempt; overrides the constructor default
)
print(llm.stats())      # calls, retries, failures, mean latency, token counts
```

| backend | key env var | notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | Chat Completions; `pip install openai` |
| `anthropic` | `ANTHROPIC_API_KEY` | Messages API; `pip install anthropic` |
| `mock` | — | deterministic and offline |

- **Keys come from environment variables only.** They are never accepted as
  literals, never written to disk, and never logged. Clients are constructed
  lazily, so importing the module without a key is harmless. Our funded
  Anthropic key lives in the gitignored `.env` at the repo root, so export it
  into the shell before a live run (`set -a; . ./.env; set +a`).
- **Retries** use exponential backoff with jitter (`max_retries`, `base_delay`,
  `max_delay`). The provider SDKs' own retry logic is disabled so there is a
  single retry policy. Worst-case wall clock is
  `timeout * (max_retries + 1)` plus backoff — size it against the 60 s soft
  limit.
- **`timeout`** is per attempt, settable on the constructor and per `chat()` call.
- **Temperature** is only forwarded when explicitly passed, because current
  Claude models reject non-default sampling parameters.

### Plumbing runs on a free router (`OBIRDY_PROVIDER`)

`harness/llm_backend.py` is the *harness'* client. The submitted agents carry
their own inlined one, and it now speaks two providers:

| `OBIRDY_PROVIDER` | transport | key |
|---|---|---|
| unset / `anthropic` | Anthropic Messages via the optional SDK | `ANTHROPIC_API_KEY` |
| `openai_compat` | `POST {base}/chat/completions` over stdlib `urllib` | `OBIRDY_COMPAT_KEY`, else `HF_TOKEN` |

**Anthropic stays the default and the competition configuration.** The compat
path exists so pipeline plumbing — does a full game run end to end, does the
probe fire, does the guesser parse a real model's JSON — can be exercised on
Hugging Face's free router without touching competition credits. It adds no
dependency: the `anthropic` import is lazy, so compat mode works on a machine
where the SDK is not installed.

```bash
export OBIRDY_PROVIDER=openai_compat
export OBIRDY_MODEL=Qwen/Qwen2.5-72B-Instruct   # default for this provider
export OBIRDY_BASE_URL=https://router.huggingface.co/v1   # default
# HF_TOKEN comes from the gitignored .env: set -a; . ./.env; set +a
python -m harness.arena --single-team --seeds 0-1 --jobs 1 \
  --red-cm players.codemaster_obirdy.AICodemaster \
  --red-g  players.guesser_obirdy.AIGuesser
```

Rules of thumb: **plumbing on the free tier, every number that goes in a report
on sonnet.** A score measured against a different model is not comparable to
anything in `harness/reports/`, and `stats` prices unknown model ids at the most
expensive entry, so the USD column on a compat run is meaningless. Both agents
also take `provider=` / `base_url=` kwargs, so a sweep arm can pin one
explicitly rather than relying on the environment.

`MockBackend` makes agent logic testable offline:

```python
from harness.llm_backend import MockBackend

llm = MockBackend(
    rules=[(r"provide a single word clue", "('pebble',2)"),
           (r"keep guessing", "yes")],
    responses=["APPLE", "TREE"],   # scripted queue, consumed then cycled
    default="OK",                  # else a deterministic sha256-derived reply
    fail_times=2,                  # first N calls raise, to exercise retries
)
llm.prompts   # every rendered prompt, for assertions
```

---

## Offline partner agents (`framework/players/`)

Added, never modifying anything that shipped:

| file | class | behaviour |
|---|---|---|
| `heuristic_common.py` | — | shared word-similarity + clue-legality helpers |
| `codemaster_heuristic.py` | `AICodemaster` | scores a bundled clue vocabulary by letter overlap and a small association table, penalising blue/civilian/assassin |
| `guesser_heuristic.py` | `AIGuesser` | picks the highest-similarity unrevealed word; stops at the clue number |
| `codemaster_random.py` | `AICodemaster` | uniformly random *legal* clue, number 1 (weak ladder floor) |
| `guesser_random.py` | `AIGuesser` | uniformly random unrevealed word |
| `glove_common.py` | — | shared GloVe cache loader + legality helper (Abra) |
| `codemaster_glove.py` | `AICodemaster` | **Abra**: cosine-similarity clue search with an assassin margin |
| `guesser_glove.py` | `AIGuesser` | **Abra**: ranks the board by cosine similarity to the clue |

The random (**Magikarp**) and heuristic (**Rattata**) pairs exist to finish
games legally and instantly with no network — they are not competition entries
and make no attempt to play well (heuristic beats random 10–0, and single-team
heuristic-vs-heuristic scores ~19). **Abra** is the one partner that plays for
real: see below. They subclass the
framework ABCs with the same constructor signature as `codemaster_GPT` /
`guesser_GPT`, so they also run directly:

```bash
cd framework
python run_game.py players.codemaster_heuristic.AICodemaster \
                   players.guesser_heuristic.AIGuesser \
                   players.codemaster_random.AICodemaster \
                   players.guesser_random.AIGuesser --seed 42
```

### Abra — the GloVe embedding partner

Rattata is a *legality* floor: it finishes games, but its "associations" are
letter overlap, so it says nothing about whether our clues are decodable by a
teammate who understands the words. **Abra** does: it scores clues and guesses
by cosine similarity in static GloVe space, so it is semantically real and
knows none of our conventions. That is the stranger the 2026 cross-pairing
evaluation actually puts us next to, so it is what we measure against. Keep the
Rattata/Magikarp pairs as the ladder floor.

Abra may depend on numpy and on downloaded vectors precisely because it is
**not** submitted — no submitted agent could assume either.

**Download the vectors (once, ~860 MB; `data/` is gitignored — never commit it):**

```bash
mkdir -p data
curl -L -o data/glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
python -m harness.glove_data          # -> data/glove_cache.npz (~64 MB)
python -m harness.glove_data --wide   # -> data/glove_wide.npz  (~160 MB)
```

`glove_data` keeps the most frequent 60k alphabetic words at 300 dimensions as
L2-normalised `float32`, because the raw 1 GB text file would otherwise be
re-parsed in every spawned arena worker. Override with `--dim` / `--vocab`, or
point the agents elsewhere with `ABRA_GLOVE_CACHE`.

`--wide` writes a **second, separate** 150k-deep cache (plus every pool word
wherever it sits) for `simtable.py`. It is a separate file on purpose: Abra's
clue candidates are a prefix of its cache, so deepening `glove_cache.npz` would
have changed what Abra says and made every recorded baseline incomparable.
Only the table builder reads the wide file.

**Without the cache the agents still run** — they fall back to the Rattata
letter-overlap score — so a fresh checkout passes the test suite. The
embedding-specific tests in `harness/test_partners.py` skip instead of failing.

```bash
python -m harness.arena --single-team --seeds 0-7 \
  --red-cm players.codemaster_obirdy.AICodemaster \
  --red-g  players.guesser_glove.AIGuesser
```

---

## `sweep.py` — paired config sweeps

Every config in a sweep plays the **same** seed set, so a half-turn difference
is a real difference and not seed luck. The spec is a JSON map of config name →
kwargs forwarded to the red agents:

```json
{"baseline":  {},
 "greedy":    {"cmr": {"majority_fraction": 0.34}},
 "sweep_off": {"cmr": {"sweep": false}}}
```

```bash
python -m harness.sweep --spec sweeps/numbers.json --seeds 0-29 --jobs 6 \
                        --report harness/reports/sweep_numbers.txt
```

The table reports mean score with its interval, assassin rate, **mean clue
number**, and estimated USD spent — clue number because it is the most direct
lever on single-team score, and spend because these runs are funded by us.

Cost accounting rides on the agents' own `usage_summary()`: `arena` collects it
per game into `result["usage"]`, and `stats` prices it with `MODEL_PRICES`.
Unknown model ids are priced at the most expensive entry, so an estimate is
never optimistic.

Bundled specs in `harness/sweeps/`:

| spec | arms |
|---|---|
| `numbers.json` | the clue-number knobs, individually and together |
| `panel.json` | one batched panel call vs `panel_isolated=3` |
| `sweep_clue.json` | endgame sweep clue on/off |
| `finalists.json` | Pidgey defaults vs the withdrawn clue-number push |
| `pidgeot.json` | shipped default vs `preset="pidgeot"` (push + danger probe) |
| `pidgeot_two_team.json` | the pidgeot preset alone, for a two-team run |
| `abra_cross.json` | our codemaster with Abra's guesser, and the reverse |

---

## `eval_battery.py` — the pending sonnet battery and its bill

Everything the Pidgeot decision needs, priced before a credit is spent. The
script **does not call an API**; printing the plan is the default and `--run`
refuses to execute without `--yes`.

```bash
python -m harness.eval_battery                      # plan + cost table
python -m harness.eval_battery --json               # machine-readable
python -m harness.eval_battery --run all --yes      # actually spend money
python -m harness.eval_battery --run eval_x_abra --yes
```

**Estimated cost, as of the last measured runs: ~$19.70 for the core battery
(106 games, ~4,100 calls), ~$23.70 including the optional two-team pidgeot
arm (126 games, ~4,900 calls).** Expect roughly 35–45 minutes of wall clock at
`--jobs 4`.

| run | games | calls | est. $ | what it settles |
|---|---|---|---|---|
| `eval_ab_default` | 30 | 1200 | 5.76 | eval A protocol, default vs pidgeot preset, paired seeds |
| `eval_ab_slang` | 20 | 800 | 3.84 | eval B protocol, same two arms on the slang pool |
| `eval_c_default` | 20 | 640 | 3.07 | eval C on the **shipped** config — the number the Pidgeotto tagging gate is still missing |
| `eval_x_abra` | 16 | 312 | 1.50 | cross-pairing with Abra, both directions (the recorded attempt ran on a dead API and is void) |
| `panel_isolated_ab` | 20 | 1150 | 5.52 | batched vs isolated panel calls |
| `eval_c_pidgeot`\* | 20 | 840 | 4.03 | eval C on the pidgeot preset; only worth it if the solo arms look promotable |

\* optional; excluded from the core total.

**How the estimate is built.** Not from a flat dollars-per-game figure: the
danger probe adds one call per finalist per clue and `panel_isolated` adds
three per panel sample, so a per-game average from earlier runs would
under-price exactly the arms being measured. Instead the battery is priced per
call at **$0.0048**, the mean of the four measured sonnet runs in
`harness/reports/` ($0.00466 / $0.00539 / $0.00457 / $0.00467 per call), times
a calls-per-game figure per run shape decomposed from those same runs:

| run shape | calls/game | derivation |
|---|---|---|
| solo, both agents ours, no probe | 36 | eval A measured: 540 calls / 15 games |
| solo, pidgeot preset | 44 | 3 codemaster + 3 probe calls per clue, on ~5.5 clues/game, plus ~11 guesser |
| two teams, default | 32 | only red is on the API |
| two teams, pidgeot | 42 | as above with the probe |
| our codemaster + a stranger's guesser | 22 | brainstorm + 2 panel samples per clue |
| a stranger's codemaster + our guesser | 17 | 2 guesser samples per turn |
| panel isolated (`panel_isolated=3`) | 79 | 9 codemaster calls per clue |

Sanity check: measured per-game spend across those runs was $0.120–$0.189, and
the battery works out to ~$0.19/game, which is the top of that range — the
probe arms are supposed to be the expensive ones.

---

## Framework quirks worth remembering

Discovered by reading `framework/game.py` and `framework/players/*_GPT.py`:

1. **Players are constructed positionally** — `codemaster_red("Red", **cmr_kwargs)`.
   Team is the first positional argument, not a kwarg. Match the GPT signature:
   `def __init__(self, team="Red", ...)`.
2. **The board is 9 red / 8 blue / 7 civilian / 1 assassin**, and **red moves
   first** — so red has one more word to find than blue.
3. **`turn_counter` counts clues from both teams** and is not exposed on the
   `Game` object; only `write_results()` ever sees it.
4. **Board words and log files use cwd-relative paths** (`players/cm_wordlist.txt`,
   `results/…`). Anything that runs `Game` must control the cwd.
5. **`do_print=False` swaps out `sys.stdout` and *closes* it in `Game.__del__`.**
   The arena keeps `do_print=True` and uses `contextlib.redirect_stdout` instead.
6. **A guesser returning `None` does not end the turn** — the engine breaks the
   inner loop and re-enters with the same team, so it can spin forever.
7. **Clue legality is not enforced by the engine.** Validation lives inside the
   agents; the organisers additionally have human judges reviewing clue spirit.
8. **`guesser_GPT` never resets its `self.guesses` counter between turns**, so
   after its cumulative guess count exceeds a clue number it stops guessing
   immediately for the rest of the game. Our guessers reset in `set_clue`.
9. **`run_game.py --single_team` takes a raw string**, so any value at all
   (including `--single_team False`) is truthy. Prefer `arena.run_batch` or the
   `Game(..., single_team=True)` kwarg over the CLI flag.
10. **`players/` has no `__init__.py`** — it works as a namespace package as long
    as `framework/` is on `sys.path`. Shared code between our agents can live in
    a module there and be imported as `from players import <module>`.
