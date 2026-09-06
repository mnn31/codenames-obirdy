# Team oBirdy, Codenames AI Competition 2026 submission

Contact: Manan Gupta, team oBirdy. (This is the instruction sheet as sent to the
competition organizers. In this public copy the embedded API key is a placeholder.)

## Files

Unpack this zip **into the framework's `codenames/players/` directory**, keeping
the `oBirdy/` folder:

| path (under `players/`) | what it is |
|---|---|
| `codemaster_obirdy.py` | Codemaster agent (single self-contained Python file) |
| `guesser_obirdy.py` | Guesser agent (single self-contained Python file) |
| `oBirdy/obirdy_simtable_v2.bin.gz` | ~3.4 MB data file used by the codemaster as an extra clue-safety check. The codemaster finds it relative to its own file, so nothing needs configuring. If it is absent or unreadable both agents still run normally without it. |

## Setup

Python 3.9+ with the framework's own dependencies, plus one package:

```
pip install -U anthropic colorama
```

**Please use `-U`: `anthropic >= 0.60` is required.** The `-U` matters. Releases
before 0.60 have no `thinking` parameter on `messages.create`, so the model
keeps extended thinking on, spends the whole token budget on a reasoning block
and returns no text, which turns both agents into their much weaker offline
fallback. The agents now route the parameter through `extra_body` so an older
SDK still works, and they print a warning when they see one:

```
[oBirdy] WARNING: anthropic 0.29.0 is older than the tested minimum 0.60 -- please pip install -U anthropic; continuing with compatibility mode
```

If that line appears, `pip install -U anthropic` and re-run. (Very old releases
also need `httpx < 0.28`, which is one more reason to just take the current
one.)

**Beyond that there is no setup, no environment variable is needed.** Our API key is
embedded in both agent files and its funding is ours. If you ever need to
override it, `ANTHROPIC_API_KEY` takes precedence over the embedded key:

```
export ANTHROPIC_API_KEY=<your key>          # optional, overrides ours
```

No GPU, no local models, no downloads; both agents together use well under 1 GB
RAM, so VRAM sharing is not a concern.

## Running

From the framework's `codenames/` directory, e.g. both our agents on the red team:

```
python run_game.py players.codemaster_obirdy.AICodemaster players.guesser_obirdy.AIGuesser players.codemaster_GPT.AICodemaster players.guesser_GPT.AIGuesser --seed 42
```

Single-team track: add `--single_team True`.

## Diagnostic output

Both agents print `[oBirdy]`-prefixed lines to stdout so a run is never a black
box. At startup each one reports its build, model, whether it found an API key
(**never the key itself**), whether `anthropic` imported, and where the data
file was loaded from:

```
[oBirdy] codemaster(Red) init: version=mega-pidgeot-x build 2026-08-08 model=claude-opus-5 preset=pidgeot provider=anthropic
[oBirdy] codemaster(Red) api key: found via embedded key (value never printed)
[oBirdy] codemaster(Red) anthropic package: ok (version 0.69.0)
[oBirdy] codemaster(Red) similarity table: loaded from .../players/oBirdy/obirdy_simtable_v2.bin.gz (format 2, 30391 clues, 1713 board words)
```

Then one line per move:

```
[oBirdy] codemaster(Red) clue 3: HARBOUR 2 | path=panel+probe candidates=8 | probe 1 run, 0 vetoed | sensor 0 flagged, 2 out-of-vocabulary
[oBirdy] guesser(Red) guess 1/2: WHALE (confidence high, margin 0.34) | clue=HARBOUR 2 | ranking=llm
```

**If anything goes wrong with the API the agents say so rather than degrading
quietly.** Both fall back to a deterministic offline mode that plays legally but
much worse, every clue numbered 1, drawn from a fixed vocabulary, and that
fallback always announces itself:

```
[oBirdy] WARNING: LLM call failed (APITimeoutError: request timed out) -- using offline fallback
[oBirdy] WARNING: no API key found (checked the api_key kwarg, ANTHROPIC_API_KEY, and the embedded HARDCODED_API_KEY) -- every clue will come from the offline fallback
```

If you see a `WARNING` line, the agent is **not** running as intended, please
tell us. Set `OBIRDY_QUIET=1` to silence everything except those warnings.

## Notes for the organisers

- **External service:** the agents call the Anthropic API (codemaster: `claude-opus-5`, guesser: `claude-sonnet-5`).
  The key and its funding are provided by us. `OBIRDY_MODEL` can override the
  model if ever needed.
- **Latency:** typical clue/guess responses are 3 to 14 s; worst observed 36 s. Both
  agents enforce an internal ~45 s deadline (under the 60 s soft limit) and
  degrade gracefully rather than overrun it.
- **Robustness:** if the API is unreachable mid-game, the agents still return
  legal, well-formatted moves via the offline fallback, they never raise and
  never produce a malformed response.
- **Fair play:** the agents read only the arguments the framework passes them
  (plus their own bundled data file). They do not read the framework's log files
  and keep no state across games.
