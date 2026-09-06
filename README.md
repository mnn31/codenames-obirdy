# oBirdy: 1st place, Single Team track, IEEE CoG 2026 Codenames AI Competition

Competition agents and evaluation tooling for the
[Codenames AI Competition](https://github.com/stepmat/Codenames_GPT) at the IEEE
Conference on Games 2026.

| Track | Result |
|---|---|
| **Single Team** (mixed partners, 75 games) | **1st place, 12.27 mean score** |
| Two Teams (72 games) | 7th place, 47.2% win rate |
| **Overall** (average rank across both tracks) | **2nd place of 12 teams** |

The field was 12 teams (6 returning, 5 new) plus a GPT-4.1 baseline built from the
organizers' reference agents. This entry was written solo by a high school student.

## What the agents do

There are two agents: a codemaster that gives clues, and a guesser that interprets
them. Each is a single self-contained Python file, plus one bundled data file.

### The codemaster: propose, simulate, then verify

Instead of asking a model for a clue and trusting the answer, every turn runs
through a pipeline.

**Propose.** The model brainstorms about 8 candidate clues over subsets of our
words. Words left over from earlier clues get fed back in, read from the
framework's shared move history.

**Filter.** Clues have to be a single alphabetic English word with no sub-word
derivation in either direction against any unrevealed board word.

**Simulate.** For each surviving candidate, extra model calls play the role of a
teammate who cannot see the key, and rank the whole board. This gives a Monte
Carlo estimate of what a partner would actually do with the clue, which is a very
different thing from asking the proposing model how good its own clue was.

**Score.** Each candidate gets an expected value: how many of our words the
simulated teammate finds before the first mistake, minus penalties. The assassin
carries roughly 7 to 9 times the weight of anything else. That asymmetry is
deliberate, because in the single team track a loss scores 25 against a mean
around 7, so one assassin pick wipes out several games of good play.

**Verify twice, independently.** The winning candidate then has to survive two
separate checks:

* a danger probe, which is one more model call that asks directly how strongly the
  clue pulls toward the assassin and each opponent word on a 0 to 10 scale. A high
  rating vetoes the clue and the next candidate gets probed instead.
* an embedding sensor, which reads a bundled quantized GloVe similarity table
  (`framework/players/oBirdy/obirdy_simtable_v2.bin.gz`, 30k clue words by 1.7k
  board words). It flags clues that sit geometrically close to the assassin, and
  demotes clues that no embedding recognizes at all.

Both exist because they fail in different ways. The model check misses geometric
proximity, and the embedding check misses cultural or lateral associations. Using
either one alone leaves a hole.

**The number is computed, not claimed.** The clue number is the longest run of our
own words that the simulated panel actually produces, capped further if the probe
still rates the clue mildly dangerous.

### The guesser

It ranks the remaining words using several model samples, with the word order
shuffled for each sample, because position bias is real and measurable. Model
scores get blended with a rank based term. The stop rule is calibrated rather than
asked for: the first guess of a turn is mandatory and free, and further guesses
continue while the next candidate stays confident relative to the best word of the
turn. It also remembers clues from earlier turns that were never fully used, and
handles a clue number of 0 (unlimited guesses) as a bounded sweep.

### Race awareness in the two team track

The codemaster estimates how many words per turn each side is finding, using the
move history, and projects who will reach their last word first. When it projects
a loss, it shifts toward more ambitious clue numbers in proportion to how far
behind it is. When level or ahead it stays conservative. Single team play is
byte identical with this turned off, since a slow win still scores there and
there is nothing to gamble for.

### Not losing on technicalities

A crash or a malformed response means disqualification, so the agents are built to
survive bad conditions. There is a hard per move wall of about 45 to 50 seconds
against the competition's 60 second soft limit, enforced on a separate thread, plus
retries with backoff, compatibility handling for old SDK versions, and a
deterministic offline fallback that plays legally when the API cannot be reached.
Every fallback prints a warning instead of degrading quietly, which is how a
funding lapse got caught during the competition window rather than after it.

## Findings that might be useful to other entrants

**A framework detail that silently disables stop rules.** `game.Game.run` calls
`set_board()`, then `get_answer()`, then reveals the guessed word in its own list,
then calls `keep_guessing()`, with no `set_board()` in between. An agent that
cached the board is therefore reasoning about a board that is one guess out of
date. The word it just guessed still looks unrevealed, so it ranks first again as
"the next candidate" and every confidence ratio comes out at 1.0. Before this was
fixed, first guesses here were 91% accurate while bonus guesses were 15%.

**Balance mattered more than peak strength.** The 2026 single team track paired
each submission's agents with other teams' agents. This entry won it without
having the best agent in either role. It was second best codemaster at 11.62 and
second best guesser at 12.49, and the only entry ranked top three in both. The
team with the strongest codemaster in the whole field, at 11.46, finished fourth
overall because their guesser ranked tenth.

**Aggression failed three separate promotion gates.** Raising clue numbers looked
like a clear improvement on first measurement three separate times, and died on
paired re-runs every time. Small sample noise in this benchmark is severe. Nothing
in this repository shipped without a paired seed comparison on identical boards,
which is the reason the harness exists at all.

**Safety worked, and it was not free.** Zero assassin losses across 72 two team
games, one of six clean sheets in the field. The same conservatism finished 7th in
that track, where losing a race and hitting the assassin score exactly the same.
The winning duel entry simply moved faster, winning in 6.36 turns on average
against 9.59 here. Risk should probably be calibrated per track, and this one was
tuned for the track it won.

## Layout

```
framework/            competition framework (MIT, unmodified) with agents added
  players/codemaster_obirdy.py   the codemaster, single file
  players/guesser_obirdy.py      the guesser, single file
  players/oBirdy/                bundled similarity table
  players/*_heuristic|random|glove|kadabra|typenull|silvally|lycanroc*
                                 sparring partners and rejected challenger designs
harness/              evaluation tooling, not part of the submission
  arena.py            batch game runner, parallel, per move instrumentation
  stats.py            aggregation with confidence intervals
  sweep.py            paired seed A/B comparisons
  secret_pool.py      alternative word pools (slang, themed, organizer style)
  simtable.py         builds the bundled similarity table from GloVe
  test_*.py           about 680 offline tests, no API key required
tools/viewer.html     self-contained replay viewer for recorded games
docs/versions.md      development log, including everything that failed
```

`tools/viewer.html` opens in a browser with no setup and replays recorded games,
with a spymaster toggle, autoplay and per move commentary.

## Running it

```bash
pip install -U anthropic colorama          # anthropic >= 0.60
export ANTHROPIC_API_KEY=...               # your own key
cd framework
python run_game.py players.codemaster_obirdy.AICodemaster \
                   players.guesser_obirdy.AIGuesser \
                   players.codemaster_GPT.AICodemaster \
                   players.guesser_GPT.AIGuesser --seed 42
# for the single team track, add --single_team True
```

Models are configurable through `OBIRDY_MODEL`. The submitted configuration used
`claude-opus-5` for the codemaster and `claude-sonnet-5` for the guesser. Without a
key the agents still play legal games through the offline fallback, and say so in
the logs.

Batch evaluation:

```bash
python -m harness.arena --seeds 0-9 --single-team --jobs 4 \
  --red-cm players.codemaster_obirdy.AICodemaster \
  --red-g  players.guesser_obirdy.AIGuesser \
  --blue-cm players.codemaster_heuristic.AICodemaster \
  --blue-g  players.guesser_heuristic.AIGuesser
```

## Known limitations

The 7th place finish in the two team track reflects a real weakness in the design,
not bad luck. There is also a residual assassin rate of roughly 5% in single team
play, caused by lateral cultural associations that neither a model probe nor GloVe
geometry can see. Several attempts to close that gap failed and are documented in
`docs/versions.md` rather than hidden. Finally, the bundled table comes from GloVe
6B, so it does not know post-2014 coinages. The 39 words in that category are
listed explicitly in `harness/simtable.py`.

## Attribution

The `framework/` directory is the organizers' competition framework from
[stepmat/Codenames_GPT](https://github.com/stepmat/Codenames_GPT), MIT licensed,
vendored without modification. See `framework/LICENSE_upstream`. Everything else
here is MIT licensed under `LICENSE`.

Written by Manan Gupta, team oBirdy.
