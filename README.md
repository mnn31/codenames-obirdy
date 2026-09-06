# oBirdy — 1st place, Single Team track, IEEE CoG 2026 Codenames AI Competition

Competition agents and evaluation infrastructure for the
[Codenames AI Competition](https://github.com/stepmat/Codenames_GPT) at the
IEEE Conference on Games 2026.

| Track | Result |
|---|---|
| **Single Team** (mixed partners, 75 games) | **🥇 1st — 12.27 mean score** |
| Two Teams (72 games) | 7th — 47.2% win rate |
| **Overall** (average rank across both tracks) | **🥈 2nd of 12 teams** |

Entered solo by a high-school student. The field was 12 teams (6 returning, 5 new)
plus a GPT-4.1 baseline built from the organizers' own reference agents.

---

## What the agents do

Two agents: a **codemaster** (gives clues) and a **guesser**. Both are single
self-contained Python files, plus one bundled data file.

### Codemaster: propose → simulate → verify

Rather than asking a model for a clue and trusting it, each turn runs a pipeline:

1. **Propose.** The LLM brainstorms ~8 candidate clues over subsets of our words,
   including words left over from earlier clues (read from the framework's shared
   move history).
2. **Filter.** Strict legality: a single alphabetic English word, no sub-word
   derivation in either direction against any unrevealed board word.
3. **Simulate.** For every surviving candidate, sampled LLM calls play the role of
   a *key-blind teammate* and rank the whole board. This is a Monte Carlo estimate
   of what a partner would actually do — not the proposing model's opinion of its
   own clue.
4. **Score.** Expected number of our words found before the first mistake, minus
   penalties, with the assassin weighted ~7–9× everything else. The scoring
   asymmetry is deliberate: in the single-team track a loss costs 25 against a
   mean near 7, so one assassin pick erases several games of good play.
5. **Verify, twice, independently.**
   - a **danger probe** — one more LLM call that asks point-blank how strongly the
     winning clue pulls toward the assassin and each opponent word (0–10). A high
     rating vetoes the clue and the next candidate is probed instead;
   - an **embedding sensor** — a bundled quantized GloVe similarity table
     (`framework/players/oBirdy/obirdy_simtable_v2.bin.gz`, 30k clue words ×
     1.7k board words) that flags clues geometrically close to the assassin, and
     demotes clues no embedding knows at all.

   The two checks exist because they **fail differently**: the LLM misses
   geometric proximity, embeddings miss cultural and lateral leaps.
6. **The number is computed, not claimed.** The clue number is the longest
   all-ours prefix the simulated panel actually produces, capped further if the
   probe still rates the clue mildly dangerous.

### Guesser

Ranks the remaining words from several LLM samples with the word order shuffled
per sample (position bias is real and measurable), blends model scores with a
rank-based term, and applies a calibrated stop rule: the mandatory first guess is
free, further guesses continue while the next candidate stays confident relative
to the turn's best. It tracks unexhausted clues across turns and handles clue
number `0` (unlimited guesses) as a bounded sweep.

### Race awareness (two-team track only)

The codemaster estimates each side's words-per-turn pace from the move history and
projects who reaches their last word first. When projected to *lose* the race, it
interpolates toward more ambitious clue numbers; level or ahead, it stays
conservative. Single-team behaviour is byte-identical with this disabled — in that
track a slow win still scores, so there is nothing to gamble for.

### Not losing on technicalities

A crash or malformed response is a disqualification, so: hard per-move wall
(~45–50 s against the competition's 60 s soft limit) enforced off-thread, retry
with backoff, compatibility shims for old SDK versions, and a deterministic
offline fallback that plays legally when the API is unreachable. Every fallback
announces itself in the logs rather than degrading silently.

---

## Findings other entrants may care about

**1. A framework gotcha that silently disables stop rules.**
`game.Game.run` calls `set_board()`, then `get_answer()`, then reveals the guessed
word *in its own list*, then calls `keep_guessing()` — with no `set_board()` in
between. An agent that cached the board is therefore reasoning about a board one
guess stale: the word just guessed still looks unrevealed and re-ranks first, so
every confidence ratio comes out 1.0 and the stop rule becomes dead code. Before
fixing this, our first guesses were 91% accurate while bonus guesses were 15%.

**2. Balance beat peak strength under mixed pairing.**
The 2026 single-team track paired each submission's agents with *other teams'*
agents. We won it without the best agent in either role — we were 2nd-best
codemaster (11.62) and 2nd-best guesser (12.49), and the only entry ranked top-3
in both. The team with the single best codemaster in the competition (11.46)
finished 4th because its guesser ranked 10th.

**3. Aggression failed three separate promotion gates.**
Raising clue numbers looked like a clear win on first measurement three times and
died on paired re-runs every time — small-sample noise in this benchmark is
vicious. Nothing shipped here without a paired-seed comparison on identical
boards; the harness exists for exactly that reason.

**4. Safety held, and it wasn't free.**
Zero assassin losses in 72 two-team games (one of six clean sheets in the field) —
but the same conservatism finished 7th in that track, where a race loss and an
assassin loss score identically. The winning duel entry simply played faster:
6.36 turns per win against our 9.59. Risk calibration should be *per track*, and
ours was tuned for the track we won.

---

## Repository layout

```
framework/            vendored competition framework (MIT, unmodified) + agents
  players/codemaster_obirdy.py    our codemaster  (single file)
  players/guesser_obirdy.py       our guesser     (single file)
  players/oBirdy/                 bundled similarity table
  players/*_heuristic|random|glove|kadabra|typenull|silvally|lycanroc*
                                  sparring partners and rejected challengers
harness/              evaluation infrastructure (not part of the submission)
  arena.py            batch game runner, parallel, per-move instrumentation
  stats.py            aggregation with confidence intervals
  sweep.py            paired-seed A/B comparisons
  secret_pool.py      alternative word pools (slang, themed, organiser-style)
  simtable.py         builds the bundled similarity table from GloVe
  test_*.py           ~680 offline tests (no API key needed)
tools/viewer.html     self-contained replay viewer for recorded games
docs/versions.md      full development log: every promotion, every failure
```

`tools/viewer.html` opens in any browser with no setup and replays recorded games
with a spymaster toggle, autoplay and per-move commentary.

## Running

```bash
pip install -U anthropic colorama          # anthropic >= 0.60
export ANTHROPIC_API_KEY=...               # your own key
cd framework
python run_game.py players.codemaster_obirdy.AICodemaster \
                   players.guesser_obirdy.AIGuesser \
                   players.codemaster_GPT.AICodemaster \
                   players.guesser_GPT.AIGuesser --seed 42
# single-team track: add --single_team True
```

Models are configurable (`OBIRDY_MODEL`); the submitted configuration used
`claude-opus-5` for the codemaster and `claude-sonnet-5` for the guesser. Without
a key the agents still play legally via the offline fallback, and say so loudly.

Batch evaluation:

```bash
python -m harness.arena --seeds 0-9 --single-team --jobs 4 \
  --red-cm players.codemaster_obirdy.AICodemaster \
  --red-g  players.guesser_obirdy.AIGuesser \
  --blue-cm players.codemaster_heuristic.AICodemaster \
  --blue-g  players.guesser_heuristic.AIGuesser
```

## Honest limitations

- The two-team result (7th) is the design's real weakness, not bad luck.
- A residual ~5% single-team assassin rate comes from lateral cultural
  associations that neither an LLM probe nor GloVe geometry sees. Several
  attempts to close it failed; it is documented rather than hidden.
- The bundled table is GloVe 6B, so it is blind to post-2014 coinages; 39 such
  words are listed explicitly in `harness/simtable.py`.

## Attribution

The `framework/` directory is the organizers' competition framework
([stepmat/Codenames_GPT](https://github.com/stepmat/Codenames_GPT), MIT, see
`framework/LICENSE_upstream`), vendored unmodified — all of our code is added
files. Everything else in this repository is MIT licensed (see `LICENSE`).

Built by Manan Gupta (team oBirdy).
