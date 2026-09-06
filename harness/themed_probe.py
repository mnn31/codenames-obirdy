"""Does a single-domain board actually collapse our clue numbers?

The organisers reported a fully-themed board (all 25 words gaming / internet
culture) on which our codemaster gave 18 consecutive number-1 clues and took 19
turns to win solo, while the baseline GPT team cleared it with INTERNET 4 /
COMPUTER 3 / SPACE 2.  The obvious reading is that our panel majority rule --
"the number is the longest all-own prefix a majority of samples agree on" --
cannot fire on a board where every word is near every other word.

That reading is wrong, and this module is why.  The *same* log shows
``candidates=3`` on nearly every turn, which is the signature of a dead
brainstorm (see the 0.29.0 post-mortem in ``codemaster_obirdy``).  With the
brainstorm dead, the only candidates left are ``_heuristic_candidates``, none of
which claims a target, and the no-panel branch's number is
``min(2, len(targets) or 1, ...)`` -- which is 1, always, on every board in
existence.  :func:`degraded_numbers` demonstrates that on the default pool too.

Two measurements, both offline:

``density``
    Mock-free.  Over the similarity table's clue vocabulary, how often are the
    two board words a clue pulls hardest on *both* ours?  That is the raw
    material the majority rule reads, with no brainstorm in between.

``mock_numbers``
    A full ``get_clue`` with the LLM stood in for by the table: the brainstorm
    proposes the clues that pull hardest on our own words, and the panel ranks
    the whole board by cosine, which is what an average operative who cannot see
    the key is approximating.  The two stages score different things (the
    brainstorm never sees the opponent's words), so the panel can and does
    disagree.

Caveat, now historical: when this module was written the shipped table's *board*
vocabulary was the framework pool plus the slang pool, 1008 words, and
``themed-kitchen`` and ``themed-sport`` were almost entirely outside it -- those
rows measured table coverage, not board density, and only ``organiser`` (12 of
its 25 words covered) and ``default`` meant anything.  The 2026-08-03 rebuild
puts every themed-pool word in the table, so every row here is now a density
measurement.  Re-read the numbers accordingly: a themed row that was low because
the table could not see the board is a different statement from a themed row
that is low because the board really is sparse.

Run::

    python -m harness.themed_probe
    python -m harness.themed_probe --pools default,organiser --seeds 40
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from players import codemaster_obirdy as cm  # noqa: E402

from harness.secret_pool import generate_board  # noqa: E402

DEFAULT_POOLS = ("default", "slang", "organiser", "themed-gaming",
                 "themed-space", "themed-kitchen", "themed-music",
                 "themed-sport")
#: Sample stride over the clue vocabulary.  30k clues x 20 boards x 8 pools is
#: minutes of pointless work for a statistic that is stable at a seventh of it.
VOCAB_STRIDE = 7


def _table():
    table = cm._SimTable.load()
    if table is None:
        raise SystemExit("no bundled simtable -- python -m harness.simtable")
    return table


# ---------------------------------------------------------------------------
# 1. Board density, with no mock anywhere near it
# ---------------------------------------------------------------------------

def density(pool, seeds, table=None) -> Dict[str, float]:
    """How often a clue's two strongest board words are both ours.

    Only clues reaching three or more board words above the table floor are
    counted: below that the ranking is one word and a coin toss, which says
    nothing about density in either direction.
    """
    table = table or _table()
    vocab = sorted(table.clue_words)[::VOCAB_STRIDE]
    both = 0
    total = 0
    own_in_top3 = 0.0
    for seed in seeds:
        spec = generate_board(pool, seed)
        words = spec["words"]
        own = set(w for w, kind in zip(words, spec["key_grid"])
                  if kind == "Red")
        for clue in vocab:
            if not cm.clue_is_legal(clue, words):
                continue
            pulls = table.pulls(clue, words) or {}
            if len(pulls) < 3:
                continue
            ranked = sorted(pulls, key=lambda w: -pulls[w])
            total += 1
            own_in_top3 += sum(1 for word in ranked[:3] if word in own)
            if ranked[0] in own and ranked[1] in own:
                both += 1
    return {"pool": pool, "clues": total,
            "top2_own": both / float(total or 1),
            "own_in_top3": own_in_top3 / float(total or 1)}


# ---------------------------------------------------------------------------
# 2. The number the machinery actually produces
# ---------------------------------------------------------------------------

def _mock_responder(agent, table):
    """Stand in for both LLM stages using the similarity table."""
    vocab = sorted(table.clue_words)

    def brainstorm(board, own, limit=12):
        scored = []
        for clue in vocab:
            if not cm.clue_is_legal(clue, board):
                continue
            pulls = table.pulls(clue, list(own))
            if not pulls:
                continue
            values = sorted(pulls.values(), reverse=True)
            second = values[1] if len(values) > 1 else values[0]
            scored.append((values[0] + (second if len(values) > 1 else 0.0),
                           clue,
                           [w for w in own if pulls.get(w, 0.0) >= second][:4]))
        scored.sort(reverse=True)
        return [{"clue": clue, "targets": targets}
                for _, clue, targets in scored[:limit]]

    def rank(clue, board, top_k=6):
        pulls = table.pulls(clue, list(board)) or {}
        return sorted(board, key=lambda w: -pulls.get(w, 0.0))[:top_k]

    def responder(system, user, max_tokens=600, deadline=None):
        board = agent._unrevealed()
        if "spymaster" in system:
            return json.dumps(brainstorm(board, agent._split_board()[0]))
        if "most associated" in user:
            clues = user.split("Clues: ")[1].split("\n")[0].split(", ")
            return json.dumps(dict((clue, rank(clue, board))
                                   for clue in clues))
        return None                  # the probe is an LLM-only question

    return responder


def _agent(pool, seed, **kwargs):
    spec = generate_board(pool, seed)
    agent = cm.AICodemaster("Red", quiet=True, api_key="sk-ant-offline-mock",
                            **kwargs)
    agent.set_game_state(list(spec["words"]), list(spec["key_grid"]))
    agent.set_move_history([])
    agent.llm.available = lambda: True
    return agent


def mock_numbers(pool, seeds, table=None) -> List[int]:
    """One opening clue per seed, panel and brainstorm both mocked."""
    table = table or _table()
    numbers = []
    for seed in seeds:
        agent = _agent(pool, seed, probe=False)
        agent.llm.chat = _mock_responder(agent, table)
        numbers.append(agent.get_clue()[1])
    return numbers


def degraded_numbers(pool, seeds, turns=6) -> List[int]:
    """What the 0.29.0 build emitted: every LLM call returning nothing.

    Not a themed-board statistic.  It comes out the same on every pool, which
    is the point -- the number-1 streak was never about the board.
    """
    numbers = []
    for seed in seeds:
        agent = _agent(pool, seed)
        agent.llm.chat = lambda *args, **kwargs: None
        for _ in range(turns):
            numbers.append(agent.get_clue()[1])
            for index, word in enumerate(agent.words):
                if agent.maps[index] == "Red" and not cm._is_revealed(word):
                    agent.words[index] = "*%s*" % word.lower()
                    break
    return numbers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values) or 1)


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pools", default=",".join(DEFAULT_POOLS))
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args(argv)

    table = _table()
    seeds = range(args.seeds)
    print("%-16s %9s %10s %8s | %-16s | %s"
          % ("pool", "top2-own", "own-in-top3", "clues", "mock numbers",
             "0.29.0 numbers"))
    for pool in [p.strip() for p in args.pools.split(",") if p.strip()]:
        stats = density(pool, seeds, table)
        mock = mock_numbers(pool, seeds, table)
        dead = degraded_numbers(pool, range(min(5, args.seeds)))
        print("%-16s %8.1f%% %10.2f %8d | mean %.2f, %2.0f%% 1 | mean %.2f, "
              "%.0f%% are 1"
              % (pool, 100.0 * stats["top2_own"], stats["own_in_top3"],
                 stats["clues"], _mean(mock),
                 100.0 * sum(1 for n in mock if n == 1) / len(mock),
                 _mean(dead),
                 100.0 * sum(1 for n in dead if n == 1) / len(dead)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
