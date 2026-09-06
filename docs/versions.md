# Version naming: Pokémon evolution lines

## 🏆 OFFICIAL RESULT, IEEE CoG 2026 Codenames AI Competition

**Team oBirdy (Mega Pidgeot X) finished 1st place in the Single Team track and
2nd place overall**, out of 12 teams plus the GPT-4.1 baseline.

| Track | Result |
|---|---|
| **Single Team (mixed partners, 75 games)** | **1st, 12.27** (2nd Edamame 12.31, 3rd Purrfect Cat-titude 12.80) |
| Two Teams (72 games) | 7th, 47.2% win rate (1st C.O.S.M.O. 84.7%) |
| **Overall (avg rank, prize money)** | **2nd** (1st Edamame, 3rd C.O.S.M.O.) |

Why the single-team track was won: the 2026 format paired each submission's
agents with *other teams'* agents, so cross-partner generality decided it.
oBirdy was the only entry ranked top-3 in **both** roles, codemaster 11.62
(2nd best) and guesser 12.49 (2nd best). Teams with one elite role and one weak
role (e.g. Semantic Squad: best codemaster at 11.46, guesser 15.64) fell behind.

The safety thesis also held: **0 assassin losses in 72 two-team games**, one of
only six entries with a clean sheet.

Where the design was wrong, exactly as the development log predicted: the
two-team track's binary scoring rewards pace, and conservatism was structurally
mispriced there. C.O.S.M.O. won it by winning *faster*, 6.36 turns per win
against oBirdy's 9.59, 8.11 words found against 6.83.


Rule: a version = a Pokémon. Tweaks/improvements evolve it to the next stage. A really
strong final form gets a Mega. A fundamentally different architecture starts a NEW
evolution line (different Pokémon family). Git tags mark each version on main.

## Main line, the oBirdy LLM agents (bird line, obviously)

| Version | Tag | What it is | Key numbers |
|---|---|---|---|
| **Pidgey** | `pidgey` | v1: panel-simulation codemaster + sampled-ranking guesser, assassin-first risk model, sweep clue, move-history memory | solo 6.93 avg (default pool), 7.60 (slang), 70% two-team, 0 assassins, 0 illegal clues |
| **Pidgeotto** | *untagged* | tuning round: anywhere-in-ranking assassin repellence, two-team urgency fix, sweep clue retired, panel-memory bug fix. **Clue-number push tried and withdrawn.** | battery (2026-07-31): solo 7.00 (default), 7.60 (slang), 75% two-team but **2 red assassin deaths in 20 duel games**, ties Pidgey on solo, fails the 0%-assassin invariant in duels; not tagged |
| **Pidgeot** | `pidgeot`, **VERSION OF RECORD** | all the Pidgeotto fixes, plus the stale-board / urgency / no-repeat round, plus two clue-safety nets shipped on by default: the danger probe (one call for the winning clue, a second only if it vetoes) and an offline embedding danger sensor reading a bundled 1 MB similarity table. Conservative numbers throughout. | validation (2026-07-31): **duels 90% (18/20) with 0 red assassin deaths, incl. the seed-10 board that killed the previous build**; solo 7.50 (n=10, a ~0.4-turn safety tax vs Pidgey, inside noise); Abra cross 13.12 where **both remaining deaths were the partner's own stale-board bonus-guess bug** (our clue was guessed correctly first in each); 0 illegal clues anywhere; CM latency max 36 s vs 60 s limit. Promoted on tail-risk elimination: every codemaster-attributable assassin death on record is now prevented. |
| ~~*ambitious numbers*~~ | `preset="ambitious"` | the clue-number push (bonus 0.55 / civilian 0.45 / slack 1). Led two sweeps, then lost three games to the assassin on a repeat run, then failed again *with* the probe armed (solo 7.87 with a death, 65% two-team). Parked as a named preset, not deleted. | **not shipped** |
| ~~*Mega Pidgeot, take one*~~ | *superseded* | **Pidgeot + `preset="ambitious"` everywhere, both safety nets armed.** The raised clue numbers re-tried behind the shipped probe and sensor. Never run, and superseded rather than run: one number policy cannot be right for both tracks, because the single-team score prices a death at 25 against a mean near 7 while the two-team score is binary. | **not run.** Battery and gate still below; `python -m harness.eval_battery --battery mega` |
| **Mega Pidgeot** | `mega-pidgeot` | **Pidgeot + duel-only race awareness.** Pace projection from move history escalates the number knobs toward `ambitious` only when projected to lose the race; single-team byte-identical. | race battery: Abra duels 4/12 vs 2/12 control, Rattata 10/10, 0 deaths. Held the belt through organizer testing (8-0 on their machine) until the challenger championship. |
| **MEGA PIDGEOT X** | `mega-pidgeot-x`, **VERSION OF RECORD** | **Mega Pidgeot with `claude-opus-5` as the codemaster's default model** (guesser stays `claude-sonnet-5`). Found by the 2026-08-08 challenger championship: four challenger architectures (Kadabra, Type: Null, Silvally, Lycanroc) + a model-swap arm fought the champion on paired boards; the model swap won the bracket, beat the Dusk Lycanroc fusion in the finals decider (6.86 vs 7.07 combined, clearly better on slang 6.62 vs 7.12), and cost only ~1.4-2x per game. | championship + finals + coronation, 54 games total: solo default **6.60**, organiser **7.00-7.17**, slang **6.62** (vs Mega Pidgeot's 7.70/8.50/, ); clue number ~1.55; **duels 10/10 vs heuristics**; Abra cross 12.62 with both deaths partner-attributable; old-SDK compat verified with opus; **0 assassin deaths and 0 illegal clues in every arm**; CM latency max 44.5 s vs the 45 s wall (themed worst case), typical p95 ~17 s. ~1.2 turns faster than the bot the organizers tested, same spotless safety. |

## Ladder / partner zoo (not submitted)

| Pokémon | What it is |
|---|---|
| **Magikarp** | random agents (`codemaster_random` / `guesser_random`), the flailing ladder floor |
| **Rattata** | letter-overlap heuristic agents, plays legally, semantically blind |
| **Abra** | GloVe embedding partner agents, the "realistic stranger" for cross-pairing tests. Built and working; self-play scores 6.80 over 10 default-pool seeds, so it is a real opponent rather than a pushover. Needs `data/glove_cache.npz` (see harness/README.md); falls back to Rattata's letter overlap without it. |

## Possible future lines

- **Rowlet line**, a local-model (no-external-API) variant chasing the special prize, if built.

Eval report files carry the version name, e.g. `harness/reports/eval_a_pidgeotto.txt`.

## Final pre-submission verdict (2026-08-03)

Re-validation (36 games, caffeinated): **0 moves over 55 s** (the wall works), **0 illegal
clues**, organiser-pool solo 7.33 clean, duels 9/10 with 0 red deaths, but 2 assassin
deaths in 20 default solo games (SINGER→WASHER, LEGEND→BOLT). Probe rated the assassin
**2.0/10 on both fatal clues**: lateral cultural associations invisible to both the probe
and GloVe. No defensible threshold catches them. Pooled shipped-config solo death rate:
**3/55 ≈ 5%**, the earlier 0-in-25 was an undersample, and a 0-deaths-in-36 gate is
statistically unpassable at that base rate (a perfect bot fails it ~2/3 of the time).
**Decision: ship.** All fixable causes are fixed (SDK compat, hard latency wall, sensor on
all paths, probe number-cap at 4.0); the residual ~5% is the measured cost of competitive
clue pace, paid equally or worse by opponents who carry no safety nets at all.

## Why Pidgeotto is not tagged

The tuning round shipped four changes and withheld one. Pidgey stays the
version of record for submission until the gate below is met on the code
currently on `main`.

**Kept** (all on `main`, 177 offline tests green, 160 in
`harness/test_obirdy.py` plus 17 in `harness/test_partners.py`; run the whole
suite with `python -m pytest harness/ -q`):

- assassin repellence anywhere in the panel's top-6 ranking, decayed by
  position, plus a weaker version for opponent words, the cross-pairing
  change, still **unmeasured** because credits ran out before the Abra runs
- two-team urgency fix: take the bonus guess when the opponent is one or two
  words from winning instead of demanding leftover support. Every Pidgey
  two-team loss was an 8-8 photo finish; eval C went 70% → 85%
- endgame sweep clue retired by default after a paired A/B came back exactly
  level (8.92 both arms), zero measured gain is not worth the tail risk of an
  unlimited clue. The machinery and the guesser's num=0 handling both remain
- panel-memory bug fix: the codemaster was adding *lists of orderings* to a set
  of word strings, so panel-derived leftovers were silently never remembered

**Withdrawn:** the clue-number push (bonus_guess_weight 0.55, civilian_penalty
0.45, claimed_slack 1). It led the sweep on 20 haiku seeds (6.55 vs 8.35) and
on a first sonnet run (6.33, 0 assassins), then a second independent sonnet run
of the identical config on the identical seeds returned 9.80 with three
assassin losses. Two of the three died on guess 2 or 3 of a raised number
walking into an assassin the panel had never surfaced, more words per clue is
mechanically more assassin exposure, and the repellence term cannot price a
word it never sees. Held at the Pidgey values; knobs remain tunable.

**Gate for tagging Pidgeotto:** eval A < 6.93 and eval B < 7.60 and eval C >=
70%, all on sonnet, all with 0% assassin, all measured on the shipped config.
Eval A currently sits at 7.20 (the finalist control arm); eval B and eval C
have only been measured on the withdrawn variant.

## Pidgeot: what exists, and what it is waiting on *(superseded 2026-07-31, the probe now ships on; see the clue-safety round at the end)*

The push was withdrawn because the panel never surfaced the assassin, so
nothing in the scorer could price it. The **danger probe** asks instead: for
the top few finalist clues, one extra no-key-guesser call per clue rates 0-10
how hard that clue pulls on the assassin and on each opponent word. A
meaningful assassin rating vetoes the clue outright (a penalty that dwarfs any
expected value, and the number drops to 1); opponent pull is a softer,
proportional penalty. A finalist the probe could not reach, deadline gone, API
down, reply unparseable, is treated as *suspect*, not clean: it loses score
and its clue number falls back to whatever the brainstorm itself claimed, which
is exactly the optimism the probe was meant to underwrite.

`preset="pidgeot"` turns on the probe together with the withdrawn values
(bonus 0.55 / civilian 0.45 / slack 1). **Shipped defaults are untouched** , 
the default preset is empty, the probe is off, and the regression tests assert
that a probe-neutral run picks the same clue and the same number as before.

Promotion is a live-eval decision, not a code decision:
`python -m harness.eval_battery` prints the exact runs (paired eval A/B,
eval C, the two Abra cross-pairings, the panel-isolation A/B) and the ~$19.70
they cost.

## Battery results (2026-07-31, ~$18 actual)

| run | shipped default | pidgeot preset |
|---|---|---|
| solo, default pool (15 paired seeds) | **7.00** (CI 6.49 to 7.51), 0 assassin | 7.87 (CI 5.14 to 10.59), **1 assassin death** |
| solo, slang pool (10 paired seeds) | 7.60, 0 assassin | 7.20, 0 assassin |
| two-team vs heuristics (20 seeds) | **75%** win, but 2 red assassin deaths | 65% win, 1 assassin death |
| panel batched vs isolated (10+10) | 6.80 vs 7.10, no contamination effect; **keep batched** (cheaper) |, |
| cross-pair: our CM + Abra guesser (8) | 12.62, **25% assassin** |, |
| cross-pair: Abra CM + our guesser (8) | 11.00, 12.5% assassin |, |

**Verdicts.** (1) Pidgeot preset: not promoted, the probe did not stop the
assassin death it was built to stop, and both tracks came back worse. (2)
Pidgeotto: still not tagged, solo is a statistical tie with Pidgey and the
two-team urgency fix appears to buy its win-rate bump (75%) at the price of
assassin deaths that Pidgey never had. (3) **Pidgey remains the version of
record.** (4) Panel contamination is a non-issue; batched stays. (5) The
worst number on the board is cross-pairing with an embedding-only stranger , 
double-digit scores and real assassin rates in both directions. Note Abra is a
pessimistic stranger (pure GloVe); the earlier LLM-stranger test (D3) was
clean, and most 2026 entrants will be LLM-based. Forensics on all assassin
deaths is the gating next step.

## Assassin forensics (2026-07-31, offline, from the raw game records)

Every death below was reconstructed from `results_local/*.json`: the board and
key grid regenerate exactly from `seed` + `pool` via
`harness.secret_pool.generate_board`, and every reconstruction was checked
against the recorded final board before being trusted.

### The finding that dominates everything else

`game.Game.run` calls, per guess:

```
set_board(live_list)   # our set_board takes a *copy*
get_answer()
_accept_guess()        # the engine reveals the word in its own list
keep_guessing()        # <- no set_board in between
```

so the guesser's board was **one guess stale exactly when the stop rule ran**.
The word we had just guessed still looked unrevealed, so it was re-offered as
"the next candidate" and every confidence ratio came out 1.0. Both the
continue threshold and the bonus threshold were dead code; the guesser simply
always took one more guess than it believed it should. The unit tests missed
it because they hand-fed a refreshed board before calling `keep_guessing()`,
which the engine never does. It reproduces in four lines.

The corpus agrees. Across 329 recorded games our guesser played:

| guess | n | ours | civilian | opponent | assassin |
|---|---|---|---|---|---|
| 1st | 2310 | 91% | 4% | 5% | 1% |
| 2nd | 873 | 69% | 13% | 17% | 1% |
| 3rd | 134 | 69% | 15% | 14% | 2% |
| **bonus (+1)** | **33** | **15%** | **58%** | **21%** | **6%** |

A gate that was supposed to make the bonus guess rare and confident let 33
through at a 15% hit rate.

### The individual deaths

| # | run / seed | clue | what happened | diagnosis |
|---|---|---|---|---|
| 1 | `sweep_pidgeot` seed 13 (pidgeot preset, two-team) | `CLOCK 2` | TICK (ours), then PART, the assassin, as guess 2 of the raised number | **probe ran, did not veto.** `probes_run=9, probes_vetoed=0`: three clues x three finalists, all answered. Not probe-skipped. |
| 2 | `eval_c_default` seed 14 | `SADDLE 1` | MOUNT (ours), then AIR on the bonus guess | **urgency rule, not EV-rational.** Red two words short, blue one from home; the +1 buys one word and so could not win, while 2 of the 9 unrevealed words ended the game on the spot. |
| 3 | `eval_c_default` seed 19 | `DRILL 1` | TRAIN (ours), then PILOT on the bonus guess | **urgency rule firing on stale counts.** Truthfully red was one word from home and blue two, `_must_gamble` was False. The stale board inflated own_left to 2 and flipped it True. Also a genuine codemaster miss: GloVe puts DRILL closer to PILOT (0.252) than to TRAIN (0.213). |
| 4 | `sweep_obirdy_cm_abra_g` seed 2 | `LASTDRINK 1` | UNDERTAKER (ours), then NAIL on Abra's own bonus | **our clue was not an English word.** A brainstorm mashup that passes `clue_is_legal` (single token, alphabetic, no board substring) and is out of vocabulary for any embedding guesser, which then degenerates to letter overlap, LASTDRINK shares N/A/I/L with the assassin. |
| 5 | `sweep_obirdy_cm_abra_g` seed 6 | `NAIL 2` (the second time) | FILE, MOUTH (ours), then BOX on Abra's bonus | **repeated clue plus an embedding-blind panel.** NAIL had already been given on turn 2 and produced a civilian. In GloVe, NAIL-BOX (0.179) is indistinguishable from NAIL-FILE (0.193) and NAIL-MOUTH (0.190). Our LLM panel never surfaced BOX. |
| 6 | `sweep_abra_cm_obirdy_g` seed 7 | `FADE 2` (Abra's clue) | BACK (ours), then DECK as guess 2 | **their clue was weak, our stop rule was the vacuous one.** FADE-DECK is −0.006 in GloVe; a working confidence ratio would have refused guess 2. Not overconfidence, no confidence was being measured. |

Deaths 2, 3 and 6 are the stale-board bug. Deaths 4 and 5 are our codemaster
handing a stranger a clue it could not decode.

### What could not be recovered

`harness/sweep.py` wrote `results_local/sweep_<name>.json` per arm, so the
battery's three pidgeot arms overwrote each other and the raw record of death
#1 *on the default pool* was gone by the time the battery finished. The
two-team pidgeot death above is the surviving instance of the same preset on
the same pool. Filenames are now keyed by pool and track.

The probe also recorded only `probes_run` / `probes_vetoed`, which can say
that nothing was vetoed but not whether the model rated the assassin low
(probe-blind) or rated it 3 to 5 and the threshold let it through
(threshold-too-high). That distinction is unanswerable from the files on disk;
per-finalist detail is now recorded so the next battery can answer it. What is
certain is the structural half: the relative veto compared the assassin
against `max(reference)`, and reference is the clue's own top-ranked own
words, which every clue points hard at, so it could essentially never fire.

## Pidgeot forensics round: what changed

Five separately-revertable commits, each with tests (177 offline tests green:
160 in `harness/test_obirdy.py`, 17 in `harness/test_partners.py`).

1. **The guesser's board staleness** (`_pending`). The accepted-but-unseen
   guess is excluded from the options and credited to our own count. The
   colour is free: `_accept_guess` only leaves the same side to move when the
   word belonged to the guessing team, so a pending guess at `keep_guessing`
   time is always one of ours. This restores the stop rule that was already
   designed and tuned, rather than inventing a new one.
2. **The two-team bonus gamble must be able to win**, the race condition is
   unchanged and a winnability condition is added (`own_left <= 1`). The 8-8
   photo finish the rule was written for satisfies it by construction; death
   #2 does not. The test fixture claimed to be a photo finish while leaving
   red nine words short and was rebuilt to mean what it says.
3. **Never give the same clue twice in one game**, read back from
   `move_history` so it survives our own state. Worst with the API down, where
   the deterministic fallback is a pure function of the board: `TOOL` six
   times in one recorded game.
4. **The probe's relative veto compares against the marginal target**
   (`min(reference)`) instead of the headline one, plus per-finalist probe
   detail in `usage_summary`. Preset-only; the shipped default leaves the
   probe off and the regression tests still assert a probe-neutral run picks
   the same clue and number.
5. **Sweep result files keyed by pool and track**, so a battery cannot
   overwrite its own evidence again.

**Measured offline already.** On the five `eval_a2_pidgeotto` seeds where the
codemaster made zero successful API calls, the fully degraded path, which
needs no credits to re-measure, mean score 19.20 → 16.20, assassin deaths
2 → 0, repeated clues 67 → 0. Lower is better on the solo score.

### Considered and not implemented

- **A bundled GloVe subset as a second danger sensor in the codemaster.**
  *Reversed the same day, see the clue-safety round below. This judgement was
  made against six deaths of which only two were codemaster-side; the next run
  produced five more, four of them plainly visible in the embedding, and the
  cost estimate assumed a multi-MB matrix rather than a 1 MB lookup table.*
  Checked against the actual deaths rather than assumed. It would have vetoed
  `DRILL` (death #3, already fixed on the guesser side) and flagged `NAIL` as
  mush (death #5), but NAIL-BOX 0.179 against NAIL-FILE 0.193 means it cannot
  tell the assassin from our own targets there, only that the whole clue is
  weak. It would not have touched `CLOCK` (0.134 vs TICK 0.36) or `SADDLE`
  (−0.033). One outright catch out of six deaths, against a multi-MB data file
  loaded in every worker, and the scenario it targets (a pure-embedding
  partner) is the pessimistic one, 2026 opponents are mostly LLM-based, and
  the LLM-stranger test D3 was clean. Not worth it.
- **The vocabulary half of the same file, though, is worth it.** `LASTDRINK`
  is out of a 60k-word GloVe vocabulary, and *being out of vocabulary is the
  whole signal*, it needs the word list, not the vectors, at roughly 600 KB
  and no matrix loads. This is the recommended next codemaster change: reject
  a brainstormed clue that is not a real English word. It is a new bundled
  data file, so it wants its own change and its own missing-file degradation
  path.
- **A guesser confidence floor on the first guess.** No evidence for it: 12
  assassin hits on guess 1 in 2310 first guesses (0.5%), and none of the six
  deaths above is one of ours on a first guess.
- **Making the probe veto absolute.** Already effectively absolute: a vetoed
  clue loses 50 points and an unprobed one only 0.25, so a veto can never win
  the selection against any other scored candidate.

### Minimal re-eval battery to judge these fixes *(superseded, see Pidgeot below)*

Only the shipped-default runs matter, items 1 to 3 change the default agents,
item 4 is preset-only and item 5 is harness plumbing.

| run | seeds | why | cost |
|---|---|---|---|
| eval C, two-team vs heuristics | 0 to 19 | the only place the urgency fix and the 2 shipped assassin deaths live | ~$3.00 |
| eval A, solo default | 0 to 14 | the stop-rule fix removes guesses, so the solo score is the thing that could regress | ~$2.50 |
| Abra cross-pair, our CM + Abra guesser | 0 to 7 | both codemaster deaths (repeat ban) | ~$1.20 |
| Abra cross-pair, Abra CM + our guesser | 0 to 7 | death #6 (stop rule against a foreign clue) | ~$0.30 |

**~$7.00 total.** Eval B (slang) and the pidgeot arms are not needed for this
verdict: slang was already 0-assassin in both arms, and the pidgeot preset
stays unshipped until the default is clean. Gate to tag: eval A no worse than
7.00, eval C win rate >= 70%, **0% assassin on every run**.

---

# Pidgeot (2026-07-31): the clue-safety round

That battery ran. Results, then what they said, then what changed.

| run | result | verdict |
|---|---|---|
| eval A, matched solo, 15 seeds (`reeval_a_fixed`) | **7.13**, 0 assassin, 100% win | clean, the stop-rule fix cost nothing |
| eval C, two-team vs heuristics, 20 seeds (`reeval_c_fixed`) | 80% win, **1 assassin death** (seed 10) | one death left, and it is a *codemaster* death |
| Abra cross-pair, our CM + Abra guesser, 8 seeds | 19.38, **5 assassin deaths in 8** (was 2 in 8) | worse, and the reason is not what the fixes touched |
| Abra cross-pair, Abra CM + our guesser, 8 seeds | 10.00, 1 death (was 1) | the guesser side is fine |

The guesser-side fixes worked. Every remaining death is the codemaster handing
out a clue that points at the assassin, which nothing in the pipeline could
price, because the LLM panel only reports the words it happens to rank.

## Death-by-death: the five abra-direction losses

Reconstructed offline from `results_local/sweep_obirdy_cm_abra_g_default_solo.json`;
boards regenerate from `seed` + `pool` via `harness.secret_pool.generate_board`
and were checked against the recorded final board. Similarities are GloVe 6B
300d cosines from `data/glove_cache.npz`, the same vectors Abra's guesser uses,
so these are literally the numbers that decided the guess.

| # | seed | clue | fatal guess | clue↔assassin | clue↔weakest target | diagnosis |
|---|---|---|---|---|---|---|
| 1 | 1 | `EGGSHELL 1` | FLUTE | **out of vocabulary** |, | not a word GloVe knows. Abra falls back to letter overlap and guesses essentially at random |
| 2 | 3 | `SURGE 1` | LEAD | **0.281** | CHARGE 0.169 | the assassin is the #1 word on the whole board for this clue |
| 3 | 5 | `PACKING 1` | OLIVE | **0.164** | TRUNK 0.147 | assassin #1, our target #2, a 0.017 margin the wrong way |
| 4 | 6 | `LADDER 1` | BOX | **0.145** | SCALE 0.131 | same shape, and the whole clue is mush: nothing clears 0.15 |
| 5 | 7 | `WOODWIND 1` | DECK | 0.139 | FLUTE **0.554** | **not ours.** The clue was excellent; Abra took a bonus guess it should not have, its own `keep_guessing` sees a board one guess stale, the identical bug we just fixed on our side |

Four of the five are embedding-proximity pulls our LLM panel could never see,
and the fifth is our clue being unreadable to an embedding at all. Only death
5 is the partner's fault.

The eval C death has the same shape. Seed 10, `PAGEANT 2`, assassin `STATE`:
PAGEANT-PRINCESS 0.228, **PAGEANT-STATE 0.198**, PAGEANT-SPOT 0.124. STATE is
the second-strongest word on the board and the guesser's number was 2. The
panel never surfaced it.

### Did the no-repeat rule make the clues worse?

Partly, and it is worth saying out loud. Comparing the same cross-pairing
before and after the fix:

| | clues/game | repeats | out-of-GloVe clues |
|---|---|---|---|
| before (`sweep_obirdy_cm_abra_g`) | 7.6 | 6 | 3.3%, LASTDRINK, REDBREAST |
| after (`..._default_solo`) | 9.0 | **0** | 5.6%, CHIRP, EGGSHELL, HARBORMEASURE, ONRUSH |

Banning repeats does what it was asked to do, and it lengthens games, and a
longer game with a shrinking board pushes the brainstorm further down its own
list. Seed 1's clue sequence is the whole story: BURN, CANYON, DRILL, GRIZZLY,
JOB, PERCH, NEST, CHIRP, TWEET, EGG, **EGGSHELL**. Seed 3 ends
ONRUSH, ONSLAUGHT, SURGE, three synonyms in a row. Seed 5 ends LUGGAGE,
ASCENDED, PACKING.

On the solo pool the drift is much smaller (6.9 → 7.1 clues per game, 2.9% →
3.7% out-of-vocabulary), so this is a cross-pairing tail effect rather than a
reason to take the rule back out, repeats were 36.8% fatal. It is, however,
the direct justification for the out-of-vocabulary demotion below: the rule
that pushes us toward novel clues needs a counterweight that pushes us back
toward *real* ones.

## What changed

Three commits, conservative numbers throughout, the ambitious clue-number
preset stays parked (it failed its own paired eval even with the probe armed;
it is now `preset="ambitious"` and nothing points at it).

**1. A bundled clue↔board similarity table** (`harness/simtable.py` →
`framework/players/obirdy_simtable.bin.gz`).

- **1.04 MB gzipped**, 8690 clue words × 516 board words, 533k stored pairs
  (61.4 per clue), floor 0.12, depth 64.
- Coverage: **393 of the 395** framework pool words, 123 of 232 slang-pool
  words, 1081 of the 1142 distinct clues our recorded games ever produced
  (97.1% of clue *tokens*), and 100% of the codemaster's offline fallback
  vocabulary.
- Depth 64 is not decoration: the fatal word sat at pool rank 12, 13, 28 and
  **53** in the four deaths above, so a top-16 or top-32 table would have
  missed LADDER.
- Values are exact cosines on a 64-level Lloyd-max codebook (mean error 0.0016,
  p99 0.016). Reduced-dimension vectors were tried first, 32-to-128-dim SVD,
  int8, and are far too lossy: at 48 dims PACKING-TRUNK reconstructs *above*
  PACKING-OLIVE, flipping the exact decision the sensor exists to make.
- It costs 6 ms to load and ~10 µs per lookup, needs only `gzip` and `json`,
  and ships as an additional submission file (the rules allow them). Missing,
  corrupt and truncated files all degrade to a silent no-op, so the agent file
  stays self-contained *in code*.
- The competition pool is secret. If it is not the framework's 395 words the
  sensor simply goes quiet on the words it does not know, which is the
  intended failure direction.

**2. An embedding danger sensor in the codemaster.** For every scored
candidate, before anything spends an API call:

- if the clue's cosine to the assassin is within `EMBED_MARGIN` (0.02) of its
  cosine to the **weakest word the clue number is buying**, penalise 6.0 , 
  enough to dominate any expected-value difference, deliberately less than the
  probe's veto so a probe veto still wins the argument;
- a target the table does not list counts as *the floor*, not as missing: a
  word the embedding barely knows is exactly the word a stranger will not
  reach;
- if the clue is not in the table's vocabulary at all, demote by 0.05, a
  tie-break, so an in-vocabulary candidate wins at equal score. This is the
  LASTDRINK / EGGSHELL half.
- Measured over 2308 recorded in-vocabulary clue turns it fires on **5.8%**,
  against 22% of the fatal turns, a ~3.8× enrichment, and it flags PAGEANT,
  SURGE, PACKING and LADDER while correctly leaving WOODWIND alone.

**3. The danger probe on by shipped default, made cheap.** It was built as the
safety net under a raised clue number; the evidence says clue safety is the
binding failure on its own, so it ships and the raised numbers do not.

- Only the **winning** candidate is probed: one extra call per turn.
- A veto falls through to the next-best candidate and probes that, then stops.
  **Two calls per turn is the ceiling**, against nine under the old top-3
  shape.
- Past the budget the next candidate is taken *unprobed*, which already means
  the brainstorm's claimed-count ceiling plus the missing penalty, not a pass.
  A crash inside a probe lands on "no answer" for the same reason, so it gets
  the conservative handling rather than silently reverting to the optimistic
  number.
- The relative veto still compares against `min(reference)`, the marginal
  target, which is the fix that made it able to fire at all.

Known gap: the deterministic offline fallback (`_fallback_clue`, used only when
the API is unreachable) still scores by letter overlap and does not consult the
table. Its whole vocabulary is in the table, so wiring it up is easy, it is
left out of this round because the API-down path is not the competition path.

The regression tests that asserted a probe-off default have been rewritten to
assert the opposite. That assertion was the thing under review, not a fixed
point, and the commit says so.

**Test count: 203 offline** (186 in `harness/test_obirdy.py`, 17 in
`harness/test_partners.py`), `python -m pytest harness/ -q`, no key, no
network. New coverage: the table's binary format round-trip against an
independent writer, missing / corrupt / truncated files, sensor hit / miss /
out-of-vocabulary / margin-tunable, the four recorded deaths flagged and the
one non-death not flagged, the probe firing exactly once on a clean leader,
the veto fall-through chain, and the two-probe ceiling.

## The validation battery

`python -m harness.eval_battery` prints these commands and their bill. Each run
has a recorded predecessor on the same seeds, so a single arm is already a
comparison and a paired A/B would be paying twice for the same answer.

| run | seeds | why | games | cost |
|---|---|---|---|---|
| **eval A**, matched solo, default pool | 0 to 9 | the probe adds a call per clue and the sensor can reject a clue, so the solo score is what could regress. Beat: **7.13** | 10 | ~$1.50 |
| **eval C**, two-team vs heuristics | **0 to 19** | seed 10 is the PAGEANT death this round exists to stop and **must** be in the range. Beat: 80% win, 1 death | 20 | ~$3.00 |
| **Abra cross**, our CM + Abra guesser | 0 to 7 | the 5-in-8 run; both new nets aim here. Beat: 19.38, 5 deaths | 8 | ~$1.20 |

**38 games, ~$5.70** at the measured $0.15/game. Budget ~$6.50: the probe adds
roughly one call per clue, about 15% above the pre-probe per-game figure.

Optional, only if the three above come back ambiguous: eval B on the slang pool
(~$1.50, the sensor is inert on most slang words anyway and slang was already
0-assassin), and a paired probe-off/sensor-off control on 10 solo seeds
(~$3.00) to attribute a change rather than merely observe it.

**Gate to tag Pidgeot:** eval A no worse than 7.13, eval C win rate ≥ 75%, the
Abra cross-pairing at **0 assassin deaths in 8**, anything above 1 means the
sensor did not do the one job it was built for, and 0 illegal clues
everywhere. If eval C seed 10 dies again on a clue the sensor never flagged,
the table's vocabulary is the thing to look at first.

---

# Mega Pidgeot (2026-08-01): the candidate, and what the safety tax turned out to be

That battery ran and Pidgeot was tagged. Before spending anything on the next
candidate, the one number that argued against the shipped build, a solo score
of 7.50 against the pre-probe 7.13, read at the time as a ~0.4-turn "safety
tax", was taken apart offline from `results_local/val_a_pidgeot.json`,
`val_c_pidgeot.json` and `val_x_pidgeot.json`.

## The tax is noise. It is not the probe and it is not the sensor.

`val_a` and `reeval_a_fixed` share seeds 0 to 9, so the comparison is paired:

| | |
|---|---|
| val_a (probe + sensor), 10 seeds | **7.50** |
| reeval_a_fixed on the same 10 seeds | **7.10** (7.13 over its full 15) |
| paired difference | **+0.40**, sd 0.84, t = 1.50, **95% CI −0.20 to +1.00** |

The interval spans zero. That alone is only "unproven", so here is the
mechanical half, which is stronger.

**The probe cannot lengthen a game unless it vetoes or goes unanswered.**
`_probe_select_inner` probes the leader and, if the leader survives, returns
that leader with its number untouched, the sub-veto penalty is subtracted but
nothing is re-ranked afterwards. The only other channel is an *unanswered*
probe, which drops the number back to the brainstorm's claimed count.

In `val_a` neither channel fired in 8 of the 10 games: `probes_run` equals
`clue_count` exactly in those 8 (and `clue_count + 1` in the two that vetoed),
so all 77 probes came back parseable and no number was ever capped. **In eight
of ten games the probe provably could not change the output**, and those eight
games average **+0.375**, against **+0.5** for the two games that did veto. The
shift lives entirely in games where the thing being blamed did nothing.

Per seed, with the paired difference against the pre-probe run:

| seed | Δ score | probes | vetoed | sensor flags | oov demotions |
|---|---|---|---|---|---|
| 0 | 0 | 6 | 0 | 2 | 5 |
| 1 | +1 | 7 | 0 | 1 | 7 |
| 2 | 0 | 8 | 0 | 0 | 4 |
| 3 | **−1** | 7 | 0 | **6** | 3 |
| 4 | 0 | 9 | **1** | 2 | 1 |
| 5 | 0 | 7 | 0 | 3 | 2 |
| 6 | **+2** | 8 | 0 | 2 | 12 |
| 7 | +1 | 9 | 0 | 1 | 12 |
| 8 | 0 | 8 | 0 | 2 | 10 |
| 9 | +1 | 8 | **1** | 0 | 4 |

The worst seed (6, +2) has no veto. The only seed that *improved* (3, −1) has
the most sensor flags of any game in the run. Across the ten, sensor flags
correlate with the score change at **r = −0.58**, more sensor activity goes
with better games, which is the opposite sign from a tax. The one positive
association is with out-of-vocabulary demotions (r = +0.65), and that penalty
is 0.05 on a scale where candidates are separated by whole points: it can only
break a tie, and a game full of mashup clues is a game the brainstorm was
already struggling in. Reported, not believed.

For scale, this repo already records the identical config on identical seeds
returning 6.33 and then 9.80. A 0.37 gap at n = 10 is inside that.

## The 15 vetoes, judged

Across all 38 validation games the probe vetoed 15 clues over 14 turns (solo 2,
duel 9, cross-pair 4). The `probe_log` entries align to turns exactly , 
consuming one entry per turn plus one per veto reproduces every game's clue
list with nothing left over, so each veto can be paired with the clue that
replaced it.

- **12 of 15 are relative vetoes** at an assassin rating of 3 or 4, every one
  of them on a clue whose *own marginal target* the same probe rated no higher.
  That is the LADDER shape from the last forensics round: not "the assassin is
  strong" but "nothing here is strong, including what we are buying". Rejecting
  those is the rule working.
- **3 are absolute** (assassin ≥ 5). One (assassin 6, our marginal target 0) is
  unarguable. The other two rated the assassin 5 while rating our own marginal
  target **6** and **8**, a clue that pointed harder at us than at the
  assassin, killed by a ceiling rule. Those two are the only defensible
  false-positive claims in the whole battery.
- **Every single one of the 15 fell through to a clue the probe rated strictly
  lower on the assassin** (3→0/1/2, 4→1, 5→0/1, 6→0). On our own side the trade
  was a wash: the replacement's marginal-target rating was better on 5 turns,
  level on 4 and worse on 5.
- Both shipped arms finished at **0 assassin deaths**.

**What could not be recovered:** `probe_log` recorded ratings but not the clue,
so none of the 15 can be named. Fixed, entries now carry `clue` and whether
the embedding sensor independently flagged the same word.

## The sensor's flag is effectively a veto, and it has a named catch

Over the 38 games the sensor raised 126 full flags and 264 out-of-vocabulary
demotions. Of the flagged candidates **1 was still issued (0.8%)**; of the
merely-demoted ones **29 were (11%)**. The demoted population is the natural
control, same "the sensor said something" selection, negligible penalty, so
the 6.0 penalty behaves as an outright rejection and the 0.05 demotion behaves
as a tie-break. Both exactly as designed, now measured rather than assumed.

The catch worth naming is eval C **seed 10**, the board whose `PAGEANT 2` into
`STATE` killed the previous build. This time the sensor flagged `PRESENT`, whose
only above-floor pull anywhere on our words *or* the assassin is **STATE at
0.339**; the turn went out as `GIFT 1` instead (BOX 0.359, PRINCESS 0.258,
assassin below the floor). Won, no death.

## What changed in code

1. **`probe_log` names the clue** and records `embed_flagged`, so the next
   battery can judge a veto without a re-run.
2. **`probe_veto_requires_embed`** (default **False**): gates the *relative*
   veto on the embedding sensor agreeing. The absolute veto is a ceiling, not a
   comparison, and is never gated. **Default deliberately unchanged**, 15
   vetoes, every one trading down to a safer clue, 0 deaths in both shipped
   arms and a tax that is not attributable is not evidence for loosening a
   safety net. The knob exists because the two absolute vetoes above are the
   shape that would justify it, and re-arming should cost one kwarg.
3. **The ambitious preset's composition is pinned by tests.** It only names
   three number knobs, so the probe and sensor already compose with it, but
   nothing asserted that; a later edit to the preset table could have silently
   disarmed the arm under test.

**Test count: 243 offline** (201 in `harness/test_obirdy.py`, 42 in
`harness/test_partners.py`, 3 of which skip without the GloVe cache),
`python -m pytest harness/ -q`, no key, no network. New coverage: the probe log
naming its clue and its sensor agreement, the corroboration knob on and off
against both veto kinds, the ambitious preset's composition with both nets
(probe shape, fall-through ceiling, sensor pick, oov demotion, and that the
numbers really do rise), and the whole promotion gate, arm reading, the eval C
arm pick, the margin, and a death in any arm blocking.

## The promotion battery

```
python -m harness.eval_battery --battery mega            # the plan and the bill
python -m harness.eval_battery --battery mega --run all --yes
python -m harness.eval_battery --battery mega --verdict  # read the gate
```

| run | arms | seeds | games | why |
|---|---|---|---|---|
| `mega_solo_default` | `pidgeot_default` + `ambitious_nets` | 0 to 14 | 30 | the arm that decides everything; both arms play the same boards |
| `mega_solo_slang` | both | 0 to 9 | 20 | 109 of the 232 slang words are outside the similarity table, so this is raised numbers running with one net half-down |
| `mega_two_team` | the winning solo arm | 0 to 9 | 10 | eval C confirmation; the command is resolved from the solo results at run time |

**60 games, ~$10.20** at **$0.17/game**, the blended figure the three
validation runs actually measured with the probe included (solo $0.192,
two-team $0.166, cross-pair $0.176). Solo-heavy, so read it as a floor nearer
$11.

**Gate to promote `ambitious_nets` to Mega Pidgeot** (`promotion_verdict`, a
pure function, unit-tested):

- solo mean on the **default pool** beats `pidgeot_default` by **≥ 0.7 turns**;
- **0 assassin deaths in every arm that ran**, the shipped one included, a
  death anywhere means the run is not clean enough to promote anything on.

Slang is measured and reported but carries no margin: a win bought where the
sensor is half-blind is not the win being claimed. 0.7 is set deliberately
above the 0.4 that this round just showed a re-run can manufacture.

**If the gate is not met, Pidgeot stands** and remains the version of record.

---

# The slang round (2026-08-01): the sensor's blind half

The mega battery ran `pidgeot_default` on both pools and the shipped agent came
out **slower on slang than on the default pool, 8.20 turns against 7.60**.
That is backwards from what a secret, slang-heavy tournament pool wants, and
the sensor that is supposed to hold the safety floor could see 123 of the 232
slang words. This round is the diagnosis of both.

## Coverage: two causes, one of them fixable

Of the 109 slang words the old table could not price:

| cause | n | fixable? |
|---|---|---|
| in GloVe 6B, below the 60k prefix `glove_cache.npz` keeps | **70** | yes, read a deeper cache |
| not in GloVe 6B at all, in any casing or hyphenation | **39** | **no** |

The frequency misses are not marginal: `PLATYPUS` is rank 68860, `PIKACHU`
92066, `INSTAGRAM` 109262, `XENOMORPH` 375895, `SPEEDRUN` 398546. The cache was
built at 60k for Abra, whose *clue candidates* are a prefix of it, a rare word
entering that file is a rare word entering Abra's mouth. So the fix is a
**second** cache (`data/glove_wide.npz`, 150k deep plus every pool word by
name) that only the table builder reads. Abra is byte-for-byte unchanged and
every recorded baseline stays comparable.

The 39 are a hard floor. GloVe 6B is a 2014 corpus; `TIKTOK`, `FORTNITE`,
`DEEPFAKE`, `BLOCKCHAIN`, `EMOJI`, `SELFIE` and `YEET` postdate it, and
`LOOTBOX`, `MOSHPIT`, `PLOTTWIST`, `SITUATIONSHIP` are coined compounds that
never had their own vectors. The only near-miss is `tik-tok` at rank 205909,
which means a clock noise. No embedding fix exists and none was attempted:
`simtable.UNREACHABLE_SLANG` names all 39 so the coverage number is always
quoted against an achievable denominator, and the codemaster already handles
them right, a board word the table cannot price counts as the floor, not as
absent.

## The table now

| | before | after |
|---|---|---|
| framework pool covered | 393 / 395 | **395 / 395** |
| slang pool covered | 123 / 232 | **193 / 232** (= 193 / 193 reachable) |
| board words | 516 | 1008 (588 pool + 420 generated) |
| clue words | 8 690 | 30 286 |
| issued clues priceable, default pool | 92.1% | **98.2%** |
| issued clues priceable, slang pool | 80.5% | **92.7%** |
| gzipped | 1.01 MB | **3.43 MB** |

The 420 generated board words are a bet on the secret pool, made by a stated
rule rather than by taste: the top-25 GloVe neighbours of every slang seed at
cosine ≥ 0.45, restricted to rows 2 000 to 120 000 of the wide cache, with
inflected forms and derivations of already-selected words dropped, ordered by
how many distinct seeds voted. What that produces is franchise and character
names, consumer tech, internet brands, music genres and food, the register the
pool is described in.

On the exact ten slang boards the validation runs on (seeds 0-9), the change
lands where it has to: board words the sensor can price go **140/250 -> 208/250**,
and the **assassin** -- the one word the whole net exists for -- goes from
priceable on **5 of 10 boards to 9 of 10**. Before this, the sensor was a coin
flip on slang about whether it could see the thing it is for.

**The generated bank is ranked separately, and that is not cosmetic.** The
first build merged it into one global top-64 and the sensor went *silent* on
`LADDER`→`BOX` (0.145, the shallowest of the four recorded deaths): twice the
board vocabulary means twice the competition for 64 slots, and the death fell
off the end. Pool words keep their own top-64 and the generated bank gets its
own top-16. `test_the_generated_bank_never_displaces_a_pool_word` pins it.

One smaller catch, in the same family: reading a 150k cache put `EGGSHELL`
(rank 73529) into the clue vocabulary, and `EGGSHELL` is one of the two clues
the out-of-vocabulary demotion was *built from*, the Abra partner that lost a
game to it caches 60k words and still cannot read it. `CLUE_VOCAB_CEILING`
caps every clue-side source at Abra's own depth. Board depth is free
information; clue depth is a claim about what a partner can decode.

## The 0.6-turn slang gap is the clue number, and the number is not a bug

Recorded games, `pidgeot_default`, seeds 0 to 14 default / 0 to 9 slang:

| | default | slang |
|---|---|---|
| turns | 7.60 | 8.20 |
| illegal clues | 0 | 0 |
| guess accuracy | 90.0% | **90.0%** |
| first-guess accuracy | 92.1% | **93.9%** |
| civilians hit / game | 0.60 | **0.40** |
| clues delivering their full number | 81% | **88%** |
| **mean clue number** | **1.39** | **1.22** |
| multi-word clues | 38% | **17%** |

Nothing on the clue side is broken. The brainstorm is not proposing
board-adjacent words (0 illegal, either pool), the panel is not misranking (the
guesser's *first* pick is right more often on slang), and the clues that do get
issued are more reliable, not less. The agent is under-clueing, not
mis-clueing.

And the numbers account for the whole gap arithmetically. Red words per turn
predicted as `number × accuracy`: slang `1.22 × 0.90 = 1.098` against an
observed `9 / 8.20 = 1.098`, exact. Give slang the default pool's mean number
and it finishes in `9 / (1.39 × 0.90) = 7.2` turns, i.e. ahead of the default
pool.

**Why the numbers are lower is the board, read correctly.** A slang pool is
pop-culture nouns that cluster hard, gaming, anime, internet, food, so more
opponent words sit near any clue, and the danger probe says so: it vetoes
**17.1%** of slang clues against 7.0% on the default pool, rates the worst
opponent word 3.28 against 2.71, and reports a 3.04 margin against 3.43. A veto
sends the clue through `_conservative_number`. The chain is intact end to end;
it is fed a harder board.

**No agent change was made, and the reason is that the opposite has already
been run.** `ambitious_nets` is exactly "raise the numbers":

| arm | pool | turns | mean number | assassin deaths |
|---|---|---|---|---|
| `pidgeot_default` | default | 7.60 | 1.39 | 0 / 15 |
| `pidgeot_default` | slang | 8.20 | 1.22 | **0 / 10** |
| `ambitious_nets` | default | 6.33 | 1.65 | 1 / 15 |
| `ambitious_nets` | slang | **6.70** | 1.52 | **1 / 10** |

Raising the numbers on slang is worth **1.5 turns** and costs an assassin in
ten games. The zero-assassin invariant is the whole reason Pidgeot is the
version of record, so the timid number is the shipped trade, not a defect , 
and the honest read of 8.20-vs-7.60 is that the agent is paying a *larger*
safety tax on slang because slang boards are genuinely more entangled.

The interesting consequence is that this is the first round where the sensor
might change that trade rather than just observing it. On slang boards the old
table was silent because the *assassin* was usually out of vocabulary
(`embed_flagged` 11 on slang against 83 on default, with `embed_oov` running
2.40 per clue against 1.21). With 193 slang words priceable instead of 123, the
free offline net is actually load-bearing on slang for the first time, which
is the precondition an ambitious slang arm was always missing, and the reason
the mega battery's slang arm carried no margin.

## Validation spec, paired slang solo, old table vs new (~$3.50, **not run**)

The table is a data change under a live LLM, so it is measured, not assumed.
Both arms are the shipped `pidgeot_default` agent on the same 10 slang boards;
the only difference is which file the sensor reads. `OBIRDY_SIMTABLE` is the
lever rather than the `simtable_path` kwarg because arena workers run with a
sandbox directory as their cwd, so a relative path in a spec file would not
resolve; the env var is absolute and is inherited by every worker.

```bash
set -a; . ./.env; set +a

# the old table, off the last commit that shipped it
git show c7204f8:framework/players/obirdy_simtable.bin.gz > data/simtable_old.bin.gz

# arm 1 -- old table (1.01 MB, 393/395 default, 123/232 slang)
OBIRDY_SIMTABLE="$PWD/data/simtable_old.bin.gz" \
python -m harness.sweep --spec harness/sweeps/mega_c_pidgeot_default.json \
  --pool slang --seeds 0-9 --jobs 4 \
  --out-dir results_local/slang_table_old \
  --report harness/reports/slang_table_old.txt

# arm 2 -- new table (3.43 MB, 395/395 default, 193/232 slang)
OBIRDY_SIMTABLE="$PWD/framework/players/obirdy_simtable.bin.gz" \
python -m harness.sweep --spec harness/sweeps/mega_c_pidgeot_default.json \
  --pool slang --seeds 0-9 --jobs 4 \
  --out-dir results_local/slang_table_new \
  --report harness/reports/slang_table_new.txt
```

Separate `--out-dir`s because both arms run the config name `pidgeot_default`
and would otherwise write the same result filename and clobber each other --
and clobber the mega battery's slang record with it.

20 games at the measured solo rate of $0.192/game = **$3.84**; the mega
battery's slang arm actually came in at $0.245/game for 10, so read the bill as
**$3.50-$4.90**.

**Gate to promote the new table:**

- slang solo score **no worse** than the old-table arm. Equal is a pass: this
  is a coverage fix, and what it buys is a *safety* net becoming operational on
  slang for the first time, not a speed win. A regression beyond noise is the
  only failing outcome.
- **0 assassin deaths in both arms**, which is the invariant the version of
  record exists to hold.
- sensor coverage **strictly better**, read from
  `python -m harness.simtable --stats`, not from the games: 395/395 vs 393/395
  on the framework pool and 193/232 vs 123/232 on slang.

All three met: promote `framework/players/obirdy_simtable.bin.gz` as shipped
and delete `data/simtable_old.bin.gz`. If the score regresses, the lever to try
before reverting is the generated bank -- it is the speculative half of the
change, and `python -m harness.simtable --build --no-extra-board` keeps every
measured coverage win while removing every guess.

---

# The organiser round (2026-08-02): making the submission runnable, and legible

The organisers test-ran our zip and came back with three requirements. None of
them changes how the agent plays, **Pidgeot remains the version of record and
no gameplay code was touched**, but all three change whether it plays at all.

Their run is why the third one exists. They could not get a key into the
process (they tried hardcoding one into our files and it did not take), so both
agents spent the entire game on the deterministic offline fallback, every clue
numbered 1, drawn from `_FALLBACK_VOCAB`: TOOL, ROYAL, SPEECH, and *nothing in
the output said so*. It won anyway, which is the worst possible outcome: a
silent degradation that looks exactly like a working agent. Every recorded
number we have says that path is worth several turns and an assassin rate.

## 1. The data file moves to `players/oBirdy/`

Auxiliary files belong in a per-team subfolder; a flat `players/` has every
entry's data competing for the same names. Resolution order in the codemaster
is now:

1. `simtable_path` kwarg, or `OBIRDY_SIMTABLE`;
2. `<module dir>/oBirdy/obirdy_simtable.bin.gz`, the submitted layout;
3. `<module dir>/obirdy_simtable.bin.gz`, the pre-2026-08 layout, kept so an
   older checkout or an already-unpacked zip still finds its data;
4. `<repo>/data/`, development only.

The file itself moved (`git mv`), `harness/simtable.py` builds to the new path,
and `_SimTable` now records which path it loaded so the startup banner can say.

## 2. The key travels inside the agent files

The organisers will not set a per-team environment variable, two entries both
wanting `ANTHROPIC_API_KEY` would clash. Both files therefore carry
`HARDCODED_API_KEY`, and key resolution is: constructor kwarg → `ANTHROPIC_API_KEY`
(kept, because every offline battery we run uses it) → the embedded constant.

What is committed is `HARDCODED_API_KEY = "OBIRDY-KEY-PLACEHOLDER"`, and the
agents ignore it because it does not start with `sk-`: with the placeholder in
place the agent behaves exactly as it does with no key at all, so a checked-in
placeholder can never be mistaken for a credential. `harness/package_submission.py`
is the only thing that ever replaces it; it rewrites exactly one line per file
(asserted, and the substituted source is re-parsed with `ast` to prove nothing
else moved), lays the zip out as the organisers unpack it, and then reopens the
finished artefact to check the bytes that will actually be sent. A zip carrying
a real key must have `keyed` in its name, that is what `.gitignore` matches , 
and the script refuses any other filename.

## 3. Diagnostics, loud by default

`print()` (their harness captures stdout), one line, prefixed `[oBirdy]`, ASCII
only, an em dash on a Windows code page would turn a diagnostic into a
`UnicodeEncodeError` mid-game.

At init each agent reports its build, model, whether a key was found **and from
which source, never the key itself**, whether `anthropic` imported and at what
version, and where the similarity table came from (or where it looked). Per
turn: one line for the clue pipeline (chosen clue and number, which branch
produced it, candidate count, probes run and vetoed, sensor flags and
out-of-vocabulary demotions) and one per guess (the pick, a confidence bucket
taken from the margin over the runner-up rather than over the turn's best,
which is 1.0 by construction on a first guess, and whether the ranking came
from the LLM or the offline fallback).

The load-bearing half is the warnings, which quiet mode does **not** suppress:

```
[oBirdy] WARNING: LLM call failed (APITimeoutError: request timed out) -- using offline fallback
[oBirdy] WARNING: no API key found (checked the api_key kwarg, ANTHROPIC_API_KEY, and the embedded HARDCODED_API_KEY) -- every clue will come from the offline fallback
```

Anything that reaches the offline path announces it: a failed call, an
unavailable client (once per agent, not once per call), a clue pipeline that
crashed, a guess ranked by letter overlap. `OBIRDY_QUIET=1` or `quiet=True`
silences everything else. Exception text is length-capped and run through a
`sk-[A-Za-z0-9_-]+` redaction, so an SDK that ever echoed the credential back
inside an error message cannot get it printed into a shared tournament log.

**Test count: 299 offline** (257 in `harness/test_obirdy.py`, 42 in
`harness/test_partners.py`), `python -m pytest harness/ -q`, no key, no
network. New coverage: the table resolution order (subfolder over legacy, env
over both, kwarg over everything, and no table anywhere still playing), key
resolution in both files including that only `sk-` values count, the
substitution hitting exactly one site per file and changing exactly one line,
the zip's layout and contents verified from the artefact, the init banner, the
per-turn lines, the exact warning string on a simulated API failure, quiet mode
by kwarg and by env var, and a scan of everything printed during a full clue
and guess for key material.

## Cutting the real zip

```bash
python -m harness.package_submission          # reads ANTHROPIC_API_KEY_TOURNAMENT from .env
```

Writes `submission/obirdy_submission_keyed.zip` (gitignored). The committed
`submission/obirdy_submission_testing.zip` is the same build with the
placeholder (`--placeholder`) and cannot play; the loose unpacked copies that
used to sit in `submission/` are gone, because a hand-maintained second copy of
the agents is exactly how a stale file gets shipped.

---

# Mega Pidgeot, take two (2026-08-03): race awareness on the duel track

The first Mega Pidgeot candidate was "raise the clue numbers everywhere". It
was never run, and it should not be: the two tracks are scored differently and
a single number policy cannot be right for both. This round replaces it. The
candidate is now **Pidgeot plus a duel-only race reading**, and the single-team
agent is byte-identical to the version of record.

## The discovery: we are losing duels without ever meeting the assassin

`results_local/gauntlet_vs_abra.json` -- three duels against the Abra pair
(GloVe codemaster + GloVe guesser, perfectly coupled, clue numbers 2-3):

| seed | result | our pace | their pace |
|---|---|---|---|
| 100 | **lost 8-3** | 1.00 words/turn | **2.67** |
| 101 | **lost 8-5** | 1.25 | 2.00 |
| 102 | **lost 8-8**, by one word | 1.60 | 1.60 |

**0-3, and zero assassin deaths in the three games.** Across them our guesser
went 16 for 17 (94%), and on seed 100 it went 3 for 3 and still lost 8-3. Our
mean clue number was **1.50** against their **1.83**. Nothing in the pipeline
misfired. We were out-raced.

The 90% duel record that promoted Pidgeot (`val_c_pidgeot`, 18/20) was measured
entirely against the Rattata heuristic pair, which clears about 1.3 words a
turn -- slower than we do. That is not a duel record, it is a record against
one slow opponent.

**The pricing error.** Pidgeot's conservatism was bought against the
*single-team* score, where an assassin death costs 25 against a mean near 7.
In the two-team track the score is binary: losing a race 8-3 is worth exactly
what dying is worth. Risk-aversion has real value while we are winning and
**none at all** once we are losing anyway. One number policy priced for the
first track was being applied to both.

## What changed

### 1. The duel signal

`game.Game.run` only ever gives red the move when `single_team` is set, so a
`Blue_*` entry in `move_history` is impossible in a single-team game. That is
the whole detector, and it is the one the guesser already uses
(`_two_team_mode`). Two consequences worth saying out loud:

- **the key grid is not a signal.** `game.Game.__init__` builds
  `["Red"]*9 + ["Blue"]*8 + ["Civilian"]*7 + ["Assassin"]` before it ever looks
  at `single_team`, so both tracks see an identical 9/8/7/1 composition. The
  idea that the grid tells you which track you are in is simply wrong;
- **our own first clue reads as no-race**, because blue has not moved yet.
  That is the correct answer rather than a gap: there is no evidence to read on
  turn 1, and it means the opening clue is conservative in every game we ever
  play.

### 2. The pace reading (`race_state`, a pure function)

From the history alone, per side: turns taken, and that side's *own* colour
revealed on its *own* turns. A word we hand the opponent by guessing it for
them shrinks their pile but is not evidence about how fast they clue, so it
moves `opp_left` and not `opp_pace`. Each pace is shrunk toward a prior by one
pseudo-turn -- **1.5 words/turn for them, 1.2 for us** -- so one unlucky turn
cannot convince the agent it is losing. The opponent prior is deliberately a
*competent* opponent and not the Rattata pace we happen to have the most games
against.

Turns are integers, and that matters. Projected turns are **rounded up**: a
side needing 1.2 turns needs 2. Red moves first and therefore wins ties, so
`deficit = ceil(own_left/own_pace) - ceil(opp_left/opp_pace)`, with a +1 for
blue. Positive means projected to lose by that many turns.

The continuous version of this was tried first and is wrong: with 8 words left
and both paces near 1, a fractional difference multiplies into a large deficit,
and the 9-vs-8 asymmetry alone reads as "losing" from move one. Rounding up
puts an even opening at deficit 0, exactly as it should be.

### 3. Escalation, scaled rather than switched

`scale = clamp(deficit / 2, 0, 1)`, and the three knobs of `preset="ambitious"`
are interpolated from the shipped values toward the ambitious ones by that
scale -- **never past them** (the cap is structural, and tested):

| knob | shipped | ambitious | what it does |
|---|---|---|---|
| `bonus_guess_weight` | 0.35 | 0.55 | how much a bonus guess is worth |
| `civilian_penalty` | 0.70 | 0.45 | how much a civilian costs |
| `claimed_slack` | 0 | 1 | **the only knob that adds a word to the number** |

Deficits are whole turns, so the shipped scale takes exactly three values: **0**
(level or ahead -- nothing changes), **0.5** (one turn behind), **1.0** (two or
more). `claimed_slack` has its own gate at 0.5 rather than riding the same ramp,
because it is the one that mechanically increases assassin exposure. Arming it
at one turn behind is the aggressive reading, chosen because the photo-finish
losses are exactly the games one extra word would have turned. `race_slack_scale`
is the knob to raise to 0.75 if the Rattata regression comes back with a death;
that is the first lever, not turning the feature off.

Both safety nets are untouched at every level: the danger probe still runs, the
embedding sensor still prices every candidate, and a probe veto still beats
everything. The known cost of the ambitious arm is roughly one assassin death
per 10-15 solo games *with* the nets armed. In a race we are projected to lose
that trade is correct. While ahead it is not. That is what the scaling is.

### 4. Inert in single-team, provably

`race_mode` (kwarg, or `OBIRDY_RACE_MODE=0`) defaults on. The escalated values
live in per-turn attributes initialised to the base ones, so a direct
`_score_candidate` call, a single-team game and a race-mode-off duel all read
the shipped numbers. The per-turn diagnostics line gains a race segment **only**
in a duel, and `usage_summary` gains its race keys only when a race was actually
read -- a single-team summary is the same dict it was.

## What the reading says about the games already on disk

`harness/race_audit.py` replays a recorded game and asks the agent's own
`race_state` -- imported, not re-derived -- where the race stood before each of
our clues. Over four recorded runs:

| run | games | escalating turns | reading |
|---|---|---|---|
| vs Abra (0-3) | 3 | seed 100 hits full escalation at **turn 2**, seed 101 at turn 3 | the two blowouts are caught early |
| seed 102 alone | 1 | quiet through turns 2-4, **0.5 at turn 5** | the turn we clued `CIRCULAR 1` needing two words |
| mirror (1-3) | 3 | escalates in both losses, goes silent from turn 4 in the win | ahead stays conservative |
| `val_c_pidgeot` vs Rattata (18/20) | 20 | **66% of turns at scale 0** | the seven comfortable wins never escalate at all; the escalation concentrates in the two losses and the long grinds |

The mirror run is the most interesting of these. Both sides are *our* agent, so
red -- nine words, moving first -- lost 2 of 3 to itself. The problem is our
pace, not Abra.

`reeval_c_fixed` seed 10, the historical `PAGEANT 2` into `STATE`, is judged by
the auditor at **deficit 2 -- fully projected lost**. The one
codemaster-attributable assassin death on record already satisfies the new
gate's "the risk was priced" test.

## Tests

**350 offline** (from 299): 40 new in `harness/test_obirdy.py` and 11 in
`harness/test_partners.py`, `python -m pytest harness/ -q`, no key, no network.
The two decisive Abra histories are copied into the test file verbatim
(`results_local/` is gitignored) with a test that checks the copies against the
real files whenever they are on disk.

Coverage: the counts and paces read off the recorded histories; seed 100
escalating by turn 2 and seed 102 only in its endgame; a gifted word moving the
count and not the pace; the priors standing with no evidence; an even opening
reading as no deficit; blue losing the tie red wins; a scoreless team keeping a
finite pace; monotonicity of the escalation in the scale; the cap at the
ambitious values from every direction, including a scale of 99; the slack
threshold and its retune; an already-ambitious config never being pulled back;
single-team play byte-identical with race mode on and off, including the
printed output and the usage summary; the turn line gaining a race segment only
in a duel; the escalated number equalling the ambitious arm and never exceeding
it; the auditor agreeing with the agent, judging a death taken while losing as
priced and one taken while ahead as not, ignoring the opponent's own deaths,
and skipping single-team games; and all four conditions of the gate.

## The validation battery *(not run)*

```
python -m harness.eval_battery --battery race              # the plan and the bill
python -m harness.eval_battery --battery race --run all --yes
python -m harness.eval_battery --battery race --verdict    # read the gate
```

| run | arms | seeds | games | why |
|---|---|---|---|---|
| `race_abra_duel` | `race_on` + `race_off`, paired | 100-111 | 24 | the run that decides everything. Seeds 100-102 are the recorded 0-3, so the control arm has a known answer to reproduce before the other nine seeds are believed |
| `race_rattata_duel` | shipped, race on | 0-9 | 10 | do-not-harm. `val_c_pidgeot` already recorded the off arm on these exact seeds at 9/10 with 0 deaths, so a second arm would be paying twice |
| `race_mirror` | both sides ours | 100-103 | 4 | two escalating agents in one game is the configuration nothing else covers |

**38 games, ~$7.00.** $0.17 per duel game (only red is on the API against Abra
and Rattata, which is what the measured two-team runs cost) and $0.31 for a
mirror game, which puts both codemasters and both guessers on it.

**Gate to ship race awareness** (`race_verdict`, a pure function, unit-tested):

- **Abra improves materially**: the race arm beats the paired race-off arm
  outright *and* wins at least **4 of 12**, against an incumbent record of 0-3.
  The target is 6+;
- **Rattata does not regress**: at least **80%**, against the 9/10 recorded;
- **no red assassin death anywhere was taken while level or ahead.** Deaths are
  not counted, they are *judged*: `harness/race_audit.py` reads the recorded
  history back through the same pace function and reports the deficit at the
  turn the fatal clue was given. A death in a game already projected lost costs
  nothing the loss would not have cost. The same death in a game we were
  winning is the feature doing the one thing it must never do, and blocks
  outright however good the win rates are;
- **0 illegal clues**, which is an invariant rather than a trade.

If the gate is not met, **Pidgeot stands** and remains the version of record.
If only the Rattata condition fails, the retune before reverting is
`race_slack_scale=0.75`, which withdraws the number lift from the one-turn-behind
games and leaves the two-turns-behind escalation and the scoring weights alone.

---

## The organiser-machine round (2026-08-03)

The organisers ran the submission zip on their own machine and their log carried
three symptoms. Two of them turn out to be the same bug, and the third is a
missing safety net that the same bug walked us into.

### 1. `anthropic==0.29.0` silently turned both agents into their fallback

Their SDK is 0.29.0; ours is 0.120.x. Reproduced live in a throwaway venv, and
the mechanism is exact:

- `Messages.create` in 0.29.0 is explicitly typed and has **no `thinking`
  parameter**, so `thinking={"type": "disabled"}` never reaches the network , 
  Python raises `TypeError: Messages.create() got an unexpected keyword
  argument 'thinking'` first;
- our sampling-shape probe matched that on the string `"unexpected keyword"`,
  read it as *the model refuses this field*, and fell through to the bare shape;
- with the field gone, `claude-sonnet-5` runs **extended thinking on by
  default**. The request came back `stop_reason=max_tokens`, `output_tokens=900`
  of 900, and **one `thinking` block and no `text` block**;
- our extractor keeps `type == "text"` blocks, so it returned `""`, not an
  exception, so `last_error` stayed `None` and no fallback warning fired.

Everything the organisers saw follows from that. The brainstorm parsed 0
candidates, leaving only the three deterministic `_heuristic_candidates` →
`candidates=3`. The panel, on the same 900-token budget, sometimes squeaked a
short answer out and sometimes did not → `path=unsimulated`. The probe, at 400
tokens, never returned anything.

Measured on their exact fatal board:

| | brainstorm candidates | panel | shape resolved |
|---|---|---|---|
| 0.29.0, before | 0 (→ `candidates=3`) | often empty | `{}` |
| 0.29.0, after | 12 (→ 8 simulated) | all 8 ranked | `thinking` + `output_config` |
| 0.120.2 | 12 (→ 8 simulated) | all 8 ranked | `thinking` + `output_config` |

**Fix.** A `TypeError` about an unknown Python keyword is a *client-side* gap,
not the server declining the field, and the two now have different handling.
The named fields are re-sent through `extra_body`, which every release back to
0.29 copies verbatim into the request JSON, so an old SDK asks the server for
exactly what a new one asks for. A server-side refusal (`does not support`,
`extra inputs are not permitted`) still drops the field and moves to the simpler
shape, as before. On a modern SDK nothing changes: `extra_body` is never used
and the request is byte-identical.

Two more defences behind it, because the first one only works for failures we
predicted:

- **an empty reply is a failure now.** A response with no text block sets
  `EmptyResponse` with its `stop_reason` and budget, is reported like any other
  fallback, and, when `stop_reason` was `max_tokens`, retries once at 4x the
  budget. Even an SDK too old for `extra_body` would recover;
- **a version below the tested floor warns at startup**, in both agents:

```
[oBirdy] WARNING: anthropic 0.29.0 is older than the tested minimum 0.60 -- please pip install -U anthropic; continuing with compatibility mode
```

`submission/INSTRUCTIONS.md` now says `pip install -U anthropic colorama` and
explains why the `-U` matters. (Note for anyone reproducing this: 0.29.0 also
needs `httpx < 0.28`, or the client constructor itself dies on `proxies`. Our
code catches that and falls back cleanly, but it is one more reason to take the
current release.)

### 2. `path=unsimulated` ran no safety net at all

Their fatal turn: `path=unsimulated (panel returned nothing) ... probe 0 run ...
sensor 0 flagged`, clue `TOOL 1`, unrevealed assassin `KNIFE`, guesser walked
into it. The branch took `simulate[0]` and returned, and neither net had a say.

The embedding sensor did not need the API to have an opinion. On the shipped
table:

| | pull |
|---|---|
| `TOOL` → `KNIFE` (assassin) | **0.328** |
| `TOOL` → `BELT` (the only own word `TOOL` knows) | 0.211 |

0.328 ≥ 0.211 − 0.02, so **the sensor as tuned would have flagged it**, by 0.117
against a 0.02 margin, with no retune needed. It was simply never asked. And it
holds for any subset of that board's own words, because `BELT` is the only one
the table connects to `TOOL` at all, every other reference is the 0.12 floor.

**Fix.** The sensor is free, offline and blind to whether the API is up, so it
now prices *every* candidate on the no-panel branches, `unsimulated` and
`offline-fallback` alike. A flagged survivor is capped at one word. The probe
also gets a turn there when the clock and the API allow; a dead API lands on
"no answer", which is the conservative handling (claimed-count ceiling, missing
penalty) and not a pass. The incoming candidate order is the tie-break, so a
board the sensor likes throughout picks exactly what the branch picked before.

Replayed on their board, `TOOL` is flagged and `MEAL 1` is given instead.

The deterministic `_fallback_clue` was the other half of this and is fixed with
it: it scored by letter overlap alone, the "known gap" recorded in the probe
round above, and where the recorded six-`TOOL` game came from, and now walks
its ranked vocabulary past a flagged leader, falling back to the best-scoring
clue at one word if the sensor dislikes all of them.

### 3. The 18 number-1 clues were symptom (1), not the themed board

`harness/secret_pool.py` grows single-domain pools (`themed-gaming`,
`themed-space`, `themed-kitchen`, `themed-music`, `themed-sport`) plus
`organiser`, the organisers' 25 words verbatim. `harness/themed_probe.py`
measures them; the report is `harness/reports/themed_boards.txt`.

The natural reading, every word near every other word, so the panel majority
rule can only return 1, does not survive the measurement:

- **with the brainstorm dead, the number is 1 on every pool**, 100% of clues,
  the default framework pool included. The no-panel branch takes heuristic
  candidates, which claim no targets, so its number is
  `min(2, len(targets) or 1, ...)` = 1 by construction. That is the entire
  18-clue streak, and it is symptom (1);
- **give the same 25 words a working panel and it does not collapse**: mean
  number 2.00 with not one number-1 clue, against 2.00 on the default pool;
- the mock-free density agrees. Their board is 12.6% "both of a clue's two
  strongest board words are ours" against the default pool's 10.7%, marginally
  denser, nowhere near a collapse.

**No pacing change made.** Loosening the number rule would spend the risk budget
the parked `ambitious` preset already failed to earn twice, against a symptom
with a different cause that is now fixed. Race mode is untouched and remains
where clue-number escalation lives.

One genuine finding did fall out of it: the shipped table's *board* vocabulary
is the framework pool plus the slang pool, 1008 words, and only **12 of the
organisers' 25 words are in it**. On a themed tournament board the sensor is
running half-blind. Widening the table's board vocabulary is the follow-up, it
is a rebuild of a shipped data file, not a knob, so it is not being done in a
fix round three weeks before the deadline.

### Tests

**374 offline** (from 350), `python -m pytest harness/ -q`, plus
`python -m harness.selftest`. New coverage: version parsing and the floor; the
startup warning firing on 0.29.0 and staying silent on 0.120; the `extra_body`
route being taken, remembered, and never used on a modern SDK; a server refusal
still dropping the field; an empty reply counting as a failure, retrying wider
once when the budget ran out and not otherwise; the shipped table's own
`TOOL`/`KNIFE`/`BELT` numbers; `TOOL` flagged and not given on the unsimulated
path; a flagged clue with no alternative capped at one word; a clean board still
taking the leading candidate; both branch labels preserved; a whole dead-API
turn never giving `TOOL`; the fallback skipping a flagged leader and still
answering when everything is flagged; and the number-1 streak reproducing on an
ordinary board while the majority rule still counts to 3 on the themed one.

### Recommended final validation before the 18 August zip

Nothing in this round changes the competition path, every fix is on a
degraded-path branch or in the transport, so this is confirmation, not a new
promotion battery.

| run | how | games | ~cost |
|---|---|---|---|
| old-SDK smoke, both agents | one solo game in a venv pinned to `anthropic==0.29.0` + `httpx==0.27.2`, watch for `candidates=8` and `path=panel+probe` | 1 | $0.10 |
| current-SDK regression, default pool | `python -m harness.eval_battery --battery race` control arm, or 10 solo seeds | 10 | $0.90 |
| themed-pool live check | 6 solo seeds on `--pool organiser`, the board that started this | 6 | $0.55 |
| duel do-not-harm | 10 Rattata duel seeds, the `val_c_pidgeot` seeds, race arm | 10 | $1.70 |
| the race gate itself, if it is still to be decided | `--battery race --run all` | 38 | $7.00 |

**Without the race battery: 27 games, ~$3.25.** With it, 65 games and ~$10.25.
Run the old-SDK smoke first, it is a tenth of a dollar and it is the one that
answers the question the organisers actually asked.

# The themed-vocabulary round (2026-08-03): the sensor's other blind half

The organiser round ended with one finding it could not act on: the shipped
similarity table's **board** vocabulary was the framework pool plus the slang
pool, 1008 words, and only **12 of the organisers' 25 themed words were in it**.
The embedding sensor, the only safety net that costs nothing and works with the
API down, was reading half a board on exactly the kind of board the organisers
ran us on. This round rebuilds the table so it can read all of it.

## What the sensor could see, before and after

Ten boards per pool (`generate_board`, seeds 0-9). "Assassin seen" is the one
that matters: it is the word the whole net exists for.

| pool | words priced /25, before | after | assassin priceable, before | after |
|---|---|---|---|---|
| default | 25.0 | 25.0 | 10/10 | 10/10 |
| slang | 20.8 | 20.8 | 9/10 | 9/10 |
| **organiser** (their board) | **12.0** | **25.0** | **4/10** | **10/10** |
| themed-gaming | 9.9 | 25.0 | 5/10 | 10/10 |
| themed-space | 3.2 | 25.0 | 2/10 | 10/10 |
| themed-kitchen | 0.5 | 25.0 | **0/10** | 10/10 |
| themed-music | 4.6 | 25.0 | 2/10 | 10/10 |
| themed-sport | 1.8 | 25.0 | **0/10** | 10/10 |

On two of the five themed pools the sensor could not price the assassin on a
single board out of ten. It was not weak there, it was absent.

Per-pool board coverage (`python -m harness.simtable --stats`):

| pool | before | after |
|---|---|---|
| default (395) | 395 | 395 |
| slang (232) | 193 | 193 (= all 193 reachable) |
| themed-gaming (50) | 20 | **50** |
| themed-space (40) | 5 | **40** |
| themed-kitchen (40) | 1 | **40** |
| themed-music (40) | 7 | **40** |
| themed-sport (40) | 3 | **40** |
| organiser (25) | 12 | **25** |

## `HYRULE` is not out of vocabulary

The expected casualty was the coined-name half of the themed pools, on the
`UNREACHABLE_SLANG` pattern from the slang round. Checked against the full 400k
GloVe 6B vocabulary in every casing and hyphenation, the answer is that **all
208 themed-pool words exist**, `HYRULE` included. Two of them merely sit below
the wide cache's 150k prefix:

| word | GloVe rank | was |
|---|---|---|
| `HYRULE` | 195 170 | missing from `glove_wide.npz` |
| `DRUMKIT` | 269 862 | missing from `glove_wide.npz` |

`glove_data.pool_words` now names the themed pools as well, so the wide cache
keeps them by name wherever they sit, the same mechanism that already rescued
`XENOMORPH` and `SPEEDRUN`. The rebuilt cache is a strict superset of the old
one: the first 150 000 rows are byte-identical, so the clue vocabulary (a 30k
prefix) did not move and no recorded baseline shifts. `simtable.UNREACHABLE_THEMED`
is empty and is **pinned** empty by a test, an entry appearing there later
means a themed pool grew a coinage, which is a thing to notice, not to absorb.

## The board vocabulary: three banks, one selection rule

| bank | words | depth per clue | what it is |
|---|---|---|---|
| real pools | 588 | top 64 | framework pool + slang pool, as before |
| themed pools | 176 | top 32 | every `THEMED_POOLS` word plus `ORGANISER_THEMED_BOARD`, minus what the real pools already held |
| generated | 949 | top 16 | `tournament_candidates`, budget 1024 |

**Each bank is ranked in its own field.** This is the same argument the slang
round had to learn the hard way: merging the generated bank into one global
top-64 pushed `LADDER`→`BOX` (0.145, the shallowest of the four recorded
deaths) off the end of its clue's list and the sensor went silent on a death it
used to catch. 176 themed words dropped into the pool bank would be that
mistake a second time, with `BOX` already at pool rank 53 of 64. So themed
words get their own bank and their own quota, and the real-pool bank is
untouched, `test_the_later_banks_never_displace_an_earlier_one` pins the depth
and every bank's quota.

The generated bank keeps the slang round's stated rule, with one change: the
**seeds** are now every bundled non-default pool word, slang, all five themed
pools, and the organisers' board, instead of the slang pool alone. Everything
else is as it was: top-25 GloVe neighbours per seed at cosine ≥ 0.45,
restricted to rows 2 000 to 120 000 of the wide cache, inflected forms and
derivations of already-selected words dropped, ordered by how many distinct
seeds voted for a word and then by best cosine, take the budget. The default
pool is deliberately *not* seeded: its neighbours are words the table already
covers, and the budget is for the half of the vocabulary that was the problem.

The budget itself needed an argument for the first time. Under the old format
it was not a judgement at all, it was "whatever is left of the 1024-word
index", which came to 420. The new index holds 65 536, so the constraint is now
the 5 MB the file has to ship in: 1024 costs ~0.55 MB and lands the file at
4.10 MB, leaving room for the pools to grow. The filters retired 75 of the 1024,
so the bank is 949.

## The format was the binding constraint, so it moved

1008 + 176 themed words exceeds 1024 before a single generated candidate, and
1024 is a hard ceiling: version 1 packed each stored pair into one 16-bit word
as `(board index << 6) | code`, leaving 10 bits of index. The three ways out,
measured on the identical 1.74 M entries of the shipped table:

| encoding | raw | gzipped |
|---|---|---|
| v1: 16-bit packed (10-bit index) | 3.32 MB | 3.30 MB |
| 24-bit packed (18-bit index) | 4.98 MB | 3.82 MB |
| **v2: uint16 index + uint8 code, parallel arrays** | 4.98 MB | **3.01 MB** |

Splitting the index from the code is what makes the wider index free: a run of
ascending indices and a run of 64-symbol codes each compress far better apart
than interleaved, and the split layout is *smaller than the format it replaces*
while carrying a full 16-bit index (65 536 board words, 64x the old ceiling).
Quantisation is untouched, same 6-bit Lloyd-max codebook, same floor, so no
resolution was traded for the room. Dropping to a 5-bit code to buy an 11-bit
index inside the old 16 bits was the alternative, and it would have paid for
board words with sensor resolution at the 0.02 margins the danger rule turns on.

Compatibility, both directions, because an agent file and its data file can be
unpacked from different zips:

- the reader takes `OBSIM1` and `OBSIM2` and reports which it read;
- the builder **writes version 1 whenever the vocabulary still fits in it**, so
  a small or `--no-extra-board` table stays readable by a pre-2026-08-03 agent;
- `TestSimTableFormatVersions` writes both layouts with the test file's own
  independent encoder, reads them back through the agent, and asserts identical
  values; it also pins that a version-1 write of an out-of-range column raises
  instead of aliasing, that a 1499-column index round-trips under version 2,
  and that a truncated or unknown-magic file loads as `None` rather than as
  plausible numbers.

## The table now

| | before | after |
|---|---|---|
| format | `OBSIM1` | **`OBSIM2`** |
| board words | 1008 (588 pool + 420 generated) | **1713** (588 pool + 176 themed + 949 generated) |
| clue words | 30 286 | 30 391 |
| stored pairs | 1 741 874 (57.5 per clue) | 2 242 238 (73.8 per clue) |
| gzipped | 3.43 MB | **4.10 MB** (limit 5 MB) |

## The themed-board probe, re-run on a table that can see the board

`harness/themed_probe.py` measured the themed pools last round with the caveat
that two of them were almost entirely outside the table, 0 and 114 scorable
clue/board pairs, so those rows were measuring coverage, not density. Re-run
on the rebuilt table (`harness/reports/themed_boards.txt` carries both runs):

| pool | scorable pairs, before → after | top2-own, before → after | mock number, before → after |
|---|---|---|---|
| default | 32 055 → 32 240 | 10.7% → 11.2% | 2.00 → 2.05 |
| organiser | 3 440 → 25 640 | 12.6% → 12.0% | 2.00 → 2.00 |
| themed-gaming | 2 424 → 27 206 | 9.5% → 11.5% | 1.85 → 2.00 |
| themed-space | 461 → 23 589 | 19.5% → 12.6% | 1.45 → 1.90 |
| themed-kitchen | **0** → 15 613 |, → 13.0% | 1.25 → 2.05 |
| themed-music | 579 → 20 046 | 7.8% → 11.8% | 1.40 → 2.15 |
| themed-sport | 114 → 28 212 | 0.0% → 10.0% | 1.10 → 2.05 |

The verdict of the organiser round is unchanged and now rests on data rather
than on the absence of it. Every themed pool measures 10.0%-13.0% "both of a
clue's two strongest board words are ours" against the default pool's 11.2%,
and mock clue numbers of 1.90 to 2.15 against 2.05, the same band. The themed
rows that used to look like a collapse (mean 1.10, 90% of clues at 1 on
themed-sport) were the mock brainstorm having almost no board word it could
price. That was the sensor being blind, not the board being dense.

## Regressions

Both named safety cases still fire, on the rebuilt table:

- `LADDER`→`BOX` 0.145 stored and **flagged**, with two younger banks now
  competing beside it; all four recorded codemaster-side deaths (`PAGEANT`,
  `SURGE`, `PACKING`, `LADDER`) still flagged, and `WOODWIND`→`DECK`, the
  death that was not ours, still not flagged;
- `TOOL`→`KNIFE` 0.347 against `BELT` 0.210, still flagged on the no-panel
  path, still answering `MEAL 1` on the organisers' fatal board.

One test moved rather than broke: it asserted `TOOL`→`KNIFE` = 0.328, which was
the *old codebook's* nearest level. The exact GloVe cosine is 0.341 and the new
book lands on 0.347, the rebuilt codebook is closer to the truth, not further,
and the assertion now names the cosine instead of an artefact of a particular
quantisation.

## Tests

**382 offline** (from 374), `python -m pytest harness/ -q`, plus
`python -m harness.selftest`. New coverage: every themed pool and the
organisers' board covered outright; `UNREACHABLE_THEMED` pinned empty; the
themed bank derived from the pools rather than transcribed; the candidate seeds
excluding the default pool; both format versions read back identical values;
version 1 still loading; a board index past 1024 round-tripping under version 2
and refused under version 1; the builder's encoder agreeing byte-for-byte with
the test file's independent one on both layouts; truncated and unknown-magic
files refused; and every bank's per-clue quota respected.

## What this does not change

No agent behaviour, no knob, no clue policy. The sensor's rule, margin, penalty
and floor are all untouched; it can simply now see the board it is being asked
about. Nothing in the race gate or the mega candidate is affected, and the
recommended validation runs at the end of the organiser round stand as written
,  though the "themed-pool live check" line is now worth more than it was, since
the sensor will actually be awake for it.

---

# The pre-final forensics round (2026-08-03): an unbounded move, and two deaths

The final validation battery ran (`results_local/final_a_default.json`,
`final_b_organiser.json`, `final_c_duel.json`, log `final_battery.log`). The
old-SDK smoke came back clean -- `candidates=8`, `path=panel+probe` on every
turn under `anthropic==0.29.0`, which is the question the organisers actually
asked -- and the duel arm went 9/10 with **0 assassin deaths in 129 clues**.
Two things in the solo arms did not pass.

| arm | games | mean score | red assassin deaths | illegal clues |
|---|---|---|---|---|
| A, solo, default pool, seeds 0-9 | 10 | 9.20 | **1** (seed 8) | 0 / 71 |
| B, solo, `organiser` pool, seeds 0-5 | 6 | 10.33 | **1** (seed 3) | 0 / 43 |
| C, duel vs heuristics, seeds 0-9 | 10 | 90% win | 0 | 0 / 129 |

## 1. The 852-second get_clue was the laptop, and the bug is that it did not matter

Four `red_cm` `get_clue` calls in arm A came back at **850.6, 851.7, 853.6 and
854.9 s** against a 45 s budget -- one each in seeds 0-3, the four games running
concurrently under `--jobs 4`, all resolving at the same wall-clock instant.
Seeds 4-9, which ran after, have a maximum of 35.2 s. `failures=0`,
no fallback warning, and the calls **succeeded**.

`pmset -g log` names it:

```
2026-08-03 18:44:21 -0700 Sleep  Entering Sleep state due to 'Clamshell Sleep':
                                 TCPKeepAlive=active Using Batt (Charge:89%) 843 secs
2026-08-03 18:58:24 -0700 Wake   Wake from Deep Idle
```

**843 seconds of clamshell sleep**, and the four stalls are 843 s plus each
call's own honest latency:

| recorded | minus the 843 s sleep |
|---|---|
| 850.60 | 7.60 |
| 851.71 | 8.71 |
| 853.60 | 10.60 |
| 854.92 | 11.92 |

`TCPKeepAlive=active` is why they completed rather than erroring, and it is why
`failures` stayed 0. So: not socket keepalive, not an SDK retry with no
timeout, not a rate limit. The lid.

**That is the measurement explained, not the defect.** The defect is that
nothing in the agent could have stopped it. Three separate holes, all fixed:

- **`_run_parallel` joined the wedged worker anyway.** It ran the panel jobs
  under `with ThreadPoolExecutor(...) as pool:`. `future.result(timeout=...)`
  honoured the deadline correctly -- and then the context manager exited
  through `shutdown(wait=True)`, which blocks until every worker thread
  finishes. The timeout bought nothing at the closing brace. Reproduced: the
  old shape returns in **4.01 s** on a 4-second job under a 0.3 s deadline; the
  new one returns in **0.31 s**.
- **The deadline is cooperative and only cooperative.** `_Deadline` is checked
  at the top of every stage, before every attempt and inside every backoff, and
  every one of those checks was correct. None of them runs while the thread is
  inside a socket read. A budget nothing can enforce from the outside is not a
  budget.
- **The per-attempt timeout was present but had no floor under it.** Both
  providers do pass one on every path -- `client.messages.create(timeout=...)`
  on the modern route, the same value on the `extra_body` route, and
  `urlopen(timeout=...)` on the compat route, all bounded by
  `CALL_TIMEOUT_S = 22.0` and shrunk further by the remaining budget. What was
  missing was a default: the client was built with no `timeout`, so any path
  that ever dropped its own would silently inherit the SDK's **600 s**
  (`Timeout(connect=5.0, read=600, write=600, pool=600)`).

### The fixes

1. **`_run_parallel` abandons instead of joining** -- explicit executor,
   `shutdown(wait=False, cancel_futures=True)`, stragglers finish into results
   nobody reads.
2. **A hard wall, `MOVE_WALL_S = 50.0`, under `get_clue` and `get_answer`.**
   The pipeline runs on a daemon thread and past the wall we stop waiting for
   it and answer from the deterministic offline path. 45 s budget, 50 s wall,
   60 s event limit. The guess pipeline carries a generation counter so an
   abandoned worker cannot write its own pick into `_pending` afterwards and
   hide a word nobody guessed -- the staleness bug `_pending` exists to fix,
   arriving from the other side. Measured with a client that sleeps 6 s:
   **6.03 s and a legal clue without the wall, 0.63 s and a legal clue with
   it.**
3. **`anthropic.Anthropic(..., timeout=CALL_TIMEOUT_S)`**, with a
   `TypeError` fallback for an SDK too old for the keyword, so 600 s is not
   reachable from any path.
4. **`_Deadline` counts the larger elapsed of the wall and monotonic clocks**,
   so neither a sleeping machine nor a backwards NTP step can hide time from
   the budget.

The wall is a backstop and nothing else: with a healthy pipeline the clue is
byte-identical with the wall on and with `move_wall=0`, and a test asserts it.

## 2. Both deaths are the clue *number*, and the probe rated both at exactly 4

### Seed 8, `final_a_default`: `PLATE 2` -> FORK (ours), then WASHER

Board regenerated from `seed` + `pool` and checked against the recorded final
board. Reds unrevealed at clue time: SCIENTIST, GERMANY, PLASTIC, FORK, OPERA,
SINK. Assassin: WASHER.

| pair | shipped table | exact GloVe 6B 300d |
|---|---|---|
| PLATE-PLASTIC (red) | 0.309 | 0.309 |
| **PLATE-FORK (red, the marginal word the 2 was buying)** | **0.296** | **0.298** |
| PLATE-SINK (red) | 0.230 | 0.231 |
| **PLATE-WASHER (assassin)** | **not stored** | **0.146** |

**The embedding sensor did not miss this; it is right.** `0.146 >= 0.298 - 0.02`
is false, so the rule does not fire even on the exact unquantised cosine. PLATE
pulls its own targets twice as hard as it pulls the assassin. The table not
storing WASHER (it falls off PLATE's top-64 pool-bank row) changes nothing
about the verdict.

The **probe** saw it: `{"clue": "PLATE", "assassin": 4.0, "margin": 6.0,
"veto": false}`. Absolute veto needs 5.0; the relative veto needs the assassin
to out-rate our own marginal target and 4 < 6. It missed the absolute by one
point.

### Seed 3, `final_b_organiser`: `NPC 2` -> BOT (ours), then QUEST

Reds unrevealed: HYRULE, LOOT, SPAWN, BOT. Assassin: QUEST.

`table.pulls("NPC", <board>)` returns **`{}`** -- every one of the 25 words,
ours and theirs, sits below the 0.12 floor. NPC's stored row is entirely
GloVe's *National People's Congress* sense (CHINA 0.347, BEIJING 0.296, WUSHU);
the gaming sense does not exist in GloVe 6B. The exact cosines are
NPC-QUEST **0.039**, NPC-BOT 0.076, NPC-SPAWN -0.022.

Probe: `{"clue": "NPC", "assassin": 4.0, "margin": 5.0, "veto": false}`. Same
shape, one point under the absolute veto again.

Worth naming: on that board the sensor **flagged** GAMING, ONLINE, NINTENDO,
KINGDOM and PROGRAM (6.0 each, every reference at the bare 0.12 floor) and gave
NPC **0.0** -- so the unpriceable clue outscored every priceable one by six
points *because* the sensor could not see it. `_embedding_penalty` returns 0.0
the moment the assassin is absent from the table, while substituting the floor
for absent *targets*. That asymmetry is real and is named here rather than
retuned: making the unpriceable case a full flag would have flagged every
candidate on that board equally and changed the ranking not at all.

### The fix: `PROBE_NUMBER_CAP_SCORE = 4.0`

Both deaths took the assassin as **guess 2 of a 2**, on a clue the veto was
right to keep. The instrument that fits that is the number, not the veto.

A surviving clue whose probe rated the assassin at or above 4.0 keeps its place
in the selection and asks for **one word**. Measured over the **657 issued
clues in the 98 recorded games** that carry per-clue probe detail:

| probe assassin rating | issued clues |
|---|---|
| 5.0+ | 0 -- already vetoed |
| **4.0** | **8, of which 2 were fatal** |
| 3.0 | 29 |
| 2.0 | 67 |
| 0-1 | 553 |

Capping all 8 costs **8 words in 657 clues** -- 0.012 words per clue, about one
word every dozen games -- and removes the exposure that killed both games. The
eight are BATTLE 2, PLATE 2, NPC 2, SPORTS 3 (twice) and FANTASY 2.
`probe_number_capped` is now in `usage_summary` and `number_capped` on the log
entry, so the next battery can price the trade rather than assume it.

## 3. Regression or base rate? **Base rate.**

The recent rounds are exonerated on both deaths, individually:

- **the OBSIM2 rebuild did not move either fatal row.** Read back from
  `6a0d328` (the last OBSIM1 table) against the shipped one: `PLATE` gives
  `{FORK 0.2923, PLASTIC 0.30804, SINK 0.23158}` then
  `{FORK 0.29579, PLASTIC 0.30913, SINK 0.22971}` -- codebook noise, WASHER
  absent in both. `NPC` returns `{}` in both. The bank split and the rebuilt
  codebook changed neither verdict;
- **sensor-on-all-paths is not involved.** It only touches the `unsimulated`
  and `offline-fallback` branches. Both deaths are `panel+probe`
  (`probes_run == clue_count` in both games);
- **the probe-log detail is a recording change**, which is how these two are
  diagnosable at all;
- **race mode is inert**: both are single-team games.

The honest arithmetic. Shipped config, our codemaster *and* our guesser, single
team, probe armed:

| | games | red assassin deaths |
|---|---|---|
| default pool, before tonight (`sweep_pidgeot_default`, `val_a_pidgeot`) | 25 | 0 |
| slang pool, before tonight | 10 | 0 |
| **default pool, tonight** | **10** | **1** |
| **`organiser` pool, tonight (never played live before)** | **6** | **1** |

A 95% confidence bound on a rate consistent with **0 in 25** is **11.3%**. The
combined default-pool record is now **1 in 35 (2.9%)**, comfortably inside it.
The prior runs never had the power to distinguish a 0% rate from a ~5% one; the
"0 in 55" that this build was promoted on counted duel games, where our pair
still stands at **1 red assassin death in 72**. Tonight did not regress
anything -- it sampled enough to see a tail that was always there. The number
cap is a genuine improvement on that tail, not a repair.

The `organiser` pool is the honest caveat: 1 in 6 on a board type that has
never been played live, with a confidence interval wide enough to drive
through. It should be re-measured, not concluded from.

## Tests

**403 offline** (from 382), `python -m pytest harness/ -q`, plus
`python -m harness.selftest`, no key and no network. New coverage: a wedged
client still returning a legal clue and a legal guess inside the wall; the wall
warning firing in quiet mode; the abandoned guess pipeline unable to claim
`_pending`; a healthy pipeline byte-identical with the wall on and off; the
wall sitting between the budget and the event limit; `_run_parallel` abandoning
a wedged job instead of joining it; the deadline counting whichever clock ran
further, in both directions; a per-attempt timeout on the modern route, the
`extra_body` route and the compat route; the client's own default timeout and
the old-SDK constructor fallback; and the number cap -- sitting below the veto,
firing at the cap, leaving lower ratings alone, logged and counted, never
raising a number, not touching an unanswered probe, and capping both recorded
deaths replayed from the ratings actually logged.

## The minimal re-validation battery

Everything changed is either a bound on a degraded path or a one-word narrowing
of 1.2% of clues, so this confirms rather than re-decides. Run it with a
`caffeinate -dimsu` in front of the whole battery -- the lid is what produced
the 852 s number, and it will do it again.

| run | how | games | ~cost |
|---|---|---|---|
| **A. solo, default pool, seeds 0-9** | the arm that died; re-run identical so it is paired against tonight's 9.20 | 10 | $1.80 |
| **B. solo, default pool, seeds 10-19** | fresh seeds, because A alone cannot tell a fixed tail from a lucky one | 10 | $1.80 |
| **C. solo, `organiser` pool, seeds 0-5** | the other death, and the only live data that pool has | 6 | $1.15 |
| **D. duel vs heuristics, seeds 0-9** | do-no-harm on the track the cap can only slow down | 10 | $1.75 |

**36 games, ~$4.50** at the $0.176/game these three arms just measured; read the
bill as **$4.50-$5.00**. The old-SDK smoke does not need repeating -- it passed,
and nothing in this round touches the shape probe.

**Gate:**

- **0 red assassin deaths across all 36 games.** Two deaths were the finding;
  the fix has to hold on both boards that produced them and on ten seeds that
  did not;
- **no `get_clue` or `get_answer` latency above 55 s anywhere**, read from the
  `latencies` array, which is the whole point of the wall. A single one over it
  means the wall is not where it is claimed to be;
- **solo mean on the default pool no worse than 7.60** over A and B together
  (tonight's 9.20 carries the 25 from the death; the nine surviving games
  average 7.44). The cap removes at most one word from about one clue in
  eighty, so anything beyond noise here is the cap costing more than it was
  measured to cost;
- **duel win rate >= 80%** and **0 illegal clues anywhere**, both invariants
  rather than trades.

If the gate holds, Pidgeot stands as the version of record with the bounding
fixes and the number cap folded in. If a death survives on either fatal board,
the next lever is `probe_number_cap_score=3.0` -- 29 more issued clues in the
corpus, still under 6% of turns -- and not the veto threshold, which throws the
clue away instead of a word.
