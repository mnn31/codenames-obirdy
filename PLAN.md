# Codenames AI Competition 2026 — Battle Plan

Goal: **win both tracks** at the IEEE CoG 2026 Codenames AI Competition.

## The competition (facts, verified 2026-07-31)

- Official 2026 framework: https://github.com/stepmat/Codenames_GPT (vendored in `framework/`).
  The old repo (CodenamesAICompetition/Game) is the 2019-era predecessor — superseded.
- We submit **two agents**: one Codemaster and one Guesser, ideally each a single Python file
  mirroring `codemaster_GPT.py` / `guesser_GPT.py`. No modifying framework files.
- **Tracks:**
  - **Single Team:** red team alone; score = turns to find all red words (9 for red, the starting team) (lower better,
    loss = 25). Our codemaster will ALSO be paired with other entrants' guessers (and vice
    versa) — hardcoded partner strategies are explicitly neutralized.
  - **Two Teams:** full competitive Codenames vs another entrant team; ranked by win-rate.
- **Format:** round robin → top 4 → single-elim knockout (best of 3).
- **Word pool is secret** and may include slang/pop-culture ("Hogwarts", "Xenomorph") —
  static embedding lookups alone will fail; LLM-grade language coverage is required.
- **Hardware:** Threadripper 5955WX, 256 GB RAM, RTX A6000 48 GB (shared by both agents),
  100 GB storage, Windows 11 / Ubuntu 24.04. Soft 60 s per response; repeat violations = DQ.
- External APIs allowed (we fund the key). Special side prize for no-external-API entries.
- Any exception / malformed response = disqualification → robustness is a first-class feature.
- Clue legality: single English word, no board-word derivation (string sub-word check in code,
  plus HUMAN judges reviewing clue spirit), no positional/compound/misspelled tricks.

### Key dates
| Date | Milestone |
|---|---|
| **Aug 4, 2026** | Registration deadline (post team name + members on their Discord) |
| **Aug 11, 2026** | Testing submission (organizers verify code runs, give feedback) |
| **Aug 18, 2026** | Final submission |
| **Sep 1–4, 2026** | Results presented at IEEE CoG (Madrid) |

## Strategic analysis — where games are won

1. **Assassin avoidance dominates.** One assassin pick = instant loss (25 pts / lost game).
   Expected-value math says: a clue that gains +0.5 words/turn but carries 3% assassin risk
   is a losing trade. Both agents must be explicitly risk-calibrated.
2. **Generality beats cleverness — but matched play dominates.** Discord intel
   (docs/discord_intel.md): in 2025 all official games were matched pairs (our CM with our
   guesser); the 2026 README adds cross-pairing evaluation to the single-team track.
   Design point: co-optimize our CM+guesser (shared model family/conventions) for matched
   play, while keeping clues generic enough that a stranger's guesser finds them obvious.
   Our guesser must handle weird clues from strangers' codemasters gracefully.
3. **The unseen word pool kills embedding-only bots.** GPT-4o-class LLMs handle "Xenomorph"
   fine; GloVe doesn't. LLM-first, embeddings only as a fallback/sanity signal.
4. **The baseline is weak.** The provided GPT agents are single-shot prompts with no search,
   no simulation, no risk model, ad-hoc parsing. Large headroom:
   - Codemaster: generate many candidate clues → **simulate** how a *panel of diverse
     simulated guessers* would rank board words for each clue → pick the clue maximizing
     expected reds found minus λ·(blue/assassin risk). This is MCMC-adjacent policy search:
     propose, score under uncertainty, accept the best.
   - Guesser: get per-word association scores from the LLM (multiple samples → distribution),
     stop guessing when P(next word is red) drops below a threshold that depends on game
     state (e.g., gamble more when losing in two-team track).
5. **Move history matters.** Unguessed targets from earlier clues carry over; the framework
   exposes full history via `get_move_history()`. Both agents should track "leftover" clued
   words — the baseline ignores this entirely.
6. **Robustness = survival.** Validation wrappers, retry loops, deterministic fallbacks
   (embedding-based) if the API errors, and hard internal timeouts under 60 s. A DQ scores
   worse than a mediocre bot.
7. **Endgame sweep (Discord-confirmed):** clue number 0 = unlimited guesses (9999 also
   legal as pseudo-infinity). When few of our words remain and leftover associations are
   strong, a sweep clue can clear the board in one turn. Guesser must handle num=0.
8. **Between games:** agents are re-initialized every game; persistent local files for
   cross-game adaptation are permitted (clear specifics with organizers); never read the
   official game log.

## Architecture

```
framework/players/codemaster_ours.py   # submission file 1 (self-contained)
framework/players/guesser_ours.py      # submission file 2 (self-contained)
harness/                               # OUR eval infrastructure (not submitted)
  arena.py        # batch game runner: seeds × pairings, parallel
  partners.py     # zoo of teammate agents (baseline GPT, embedding bots, weak/strong LLMs)
  stats.py        # score aggregation, confidence intervals, head-to-head tables
  secret_pool.py  # simulated "unseen" word pools (slang/pop-culture) for generalization tests
```

- **LLM backend:** pluggable (`OpenAI` / Anthropic / local HF model). Decision pending:
  strongest API model for the main prize; optionally a local-model variant (A6000 fits a
  quantized 70B / solid 32B) chasing the no-external-services special prize as a second entry.
- **Codemaster pipeline:** candidate generation (LLM brainstorm, N≈15–30 clues over target
  subsets) → legality filter (sub-word check + dictionary) → simulated-guesser panel scoring
  → risk-adjusted argmax. Cache aggressively; stay well under 60 s.
- **Guesser pipeline:** clue → per-word association distribution (sampled LLM rankings +
  embedding prior) → sequential guessing with dynamic stop rule using clue number, move
  history, and track-aware risk tolerance.

## Evaluation protocol (how we know we're winning)

- Fixed seed sets (≥100 boards) for paired comparisons of every agent revision.
- Metrics: single-team avg score (with own + foreign partners), two-team win-rate vs a
  ladder of opponents, assassin-hit rate, illegal-clue rate (must be 0), p95 latency.
- Generalization suite: same metrics on synthetic slang/pop-culture word pools.
- Regression gate: no revision ships unless it beats the incumbent with statistical margin.

## Execution phases

1. **Now:** repo + plan (this commit); registration info to Manan (Discord post — human action).
2. **Phase 1 (by ~Aug 3):** harness (arena, stats, partner zoo, secret pools) + pluggable LLM
   backend + reproduce baseline GPT-vs-GPT numbers. [Opus agents build; Sonnet for mechanical]
3. **Phase 2 (by ~Aug 8):** v1 of our codemaster + guesser (simulation-scored clues, sampled
   guesser, risk model, robustness wrappers). Beat baseline decisively on all metrics.
4. **Phase 3 (by Aug 11):** freeze v1 → **testing submission** to organizers for feedback.
5. **Phase 4 (Aug 11–17):** iterate — prompt/hyperparameter tournaments, teammate
   generalization hardening, latency tuning, optional local-model second entry.
6. **Phase 5 (Aug 18):** final submission — single-file agents, pinned deps, run
   instructions for their exact hardware, funded API key.

## Status

- [x] Registered on Discord (2026-07-31). **Team name: oBirdy.**
- [x] LLM provider: Anthropic API. Key lives in the local gitignored `.env`
      (`ANTHROPIC_API_KEY`), verified working. Never commit it.
- [ ] Optional local-model side entry (no-external-services prize) if time permits.
