"""Offline smoke test for the whole harness -- no API keys, no network.

Run it with::

    python -m harness.selftest

It exercises, in order:

1. :mod:`harness.llm_backend` -- MockBackend determinism, scripted queues,
   regex rules and the retry/backoff path.
2. :mod:`harness.secret_pool` -- pool loading and board reproduction.
3. :mod:`harness.arena` -- the illegal-clue auditor and the turn cap, using
   deliberately broken agents defined below.
4. A >=20 game batch (single-team and two-team, default and slang pools,
   sequential and parallel) fed through :mod:`harness.stats`.

Exit status is non-zero if anything crashes or an invariant fails.
"""

from __future__ import annotations

import sys

from harness import arena, secret_pool, stats
from harness.llm_backend import LLMCallError, MockBackend, get_backend

arena.ensure_framework_on_path()

from players.codemaster import Codemaster  # noqa: E402  (needs sys.path first)
from players.guesser import Guesser  # noqa: E402


# ---------------------------------------------------------------------------
# Deliberately broken agents -- used only to prove the auditor fires.
# They live here (not in framework/players/) so the submission directory only
# holds real agents.
# ---------------------------------------------------------------------------

class IllegalCodemaster(Codemaster):
    """Always clues with a word derived from a live board word."""

    def __init__(self, team="Red", **kwargs):
        super().__init__()
        self.team = team
        self.words = []
        self.maps = []

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_clue(self):
        for word in self.words:
            if word and word[0] != "*" and word.isalpha():
                return [word, 1]  # identical to a board word => illegal
        return ["SIGNAL", 1]


class StallingCodemaster(Codemaster):
    """Legal clues, but paired with a guesser that never scores -> turn cap."""

    def __init__(self, team="Red", **kwargs):
        super().__init__()
        self.team = team
        self.words = []

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_clue(self):
        return ["SIGNAL", 1]


class NullGuesser(Guesser):
    """Returns None every turn.

    game.Game breaks out of the guessing loop without switching teams when the
    guess is None, so the engine would spin forever -- this is exactly the
    hang the arena's turn cap exists to catch.
    """

    def __init__(self, team="Red", **kwargs):
        super().__init__()
        self.team = team

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        return [clue, num]

    def keep_guessing(self):
        return False

    def get_answer(self):
        return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(label, condition):
    status = "ok  " if condition else "FAIL"
    print("  [%s] %s" % (status, label))
    if not condition:
        raise AssertionError(label)


def test_llm_backend():
    print("\n1. llm_backend")

    mock = MockBackend(rules=[(r"clue", "('pebble',2)")], default="FALLBACK")
    out = mock.chat([{"role": "user", "content": "give me a clue please"}])
    check("regex rule matched", out == "('pebble',2)")
    check("default used otherwise",
          mock.chat([{"role": "user", "content": "hello"}]) == "FALLBACK")

    scripted = MockBackend(responses=["A", "B"], cycle=True)
    got = [scripted.chat([{"role": "user", "content": "x"}]) for _ in range(4)]
    check("scripted queue cycles", got == ["A", "B", "A", "B"])

    a = MockBackend().chat([{"role": "user", "content": "deterministic?"}])
    b = MockBackend().chat([{"role": "user", "content": "deterministic?"}])
    check("hash fallback is deterministic across instances", a == b)

    retrying = MockBackend(default="RECOVERED", fail_times=2, max_retries=3)
    check("retries then succeeds",
          retrying.chat([{"role": "user", "content": "x"}]) == "RECOVERED")
    check("retry counter incremented", retrying.retries == 2)

    exhausted = MockBackend(default="never", fail_times=5, max_retries=2)
    try:
        exhausted.chat([{"role": "user", "content": "x"}])
        raised = False
    except LLMCallError:
        raised = True
    check("raises LLMCallError once retries are exhausted", raised)
    check("failure counter incremented", exhausted.failures == 1)

    check("factory returns a MockBackend",
          isinstance(get_backend("mock"), MockBackend))
    check("stats block is populated", get_backend("mock").stats()["calls"] == 0)

    # The API backends must not be constructible into a live call without a key
    # -- but constructing the object itself must stay cheap and side-effect free.
    for name in ("openai", "anthropic"):
        backend = get_backend(name)
        check("%s backend constructs lazily" % name, backend._client is None)


def test_secret_pool():
    print("\n2. secret_pool")
    slang = secret_pool.resolve_pool("slang")
    default = secret_pool.resolve_pool("default")
    check("slang pool has >= 200 words", len(slang) >= 200)
    check("slang pool has no duplicates", len(slang) == len(set(slang)))
    check("default pool loaded", len(default) > 300)
    check("pools are disjoint enough",
          len(set(slang) & set(default)) < 25)

    board = secret_pool.generate_board("slang", seed=3)
    check("board is 25 words", len(board["words"]) == 25)
    check("key grid is 9/8/7/1",
          [board["key_grid"].count(k) for k in
           ("Red", "Blue", "Civilian", "Assassin")] == [9, 8, 7, 1])
    check("board words come from the pool",
          all(w in slang for w in board["words"]))

    themed = secret_pool.resolve_pool("themed-gaming")
    check("themed pool is big enough to vary by seed", len(themed) >= 40)
    check("every theme can fill a board",
          all(len(secret_pool.themed_pool(name)) >= secret_pool.BOARD_SIZE
              for name in secret_pool.THEMED_POOLS))
    organiser = secret_pool.generate_board("organiser", seed=1)
    check("the organisers' themed board is reproduced verbatim",
          sorted(organiser["words"])
          == sorted(secret_pool.ORGANISER_THEMED_BOARD))


def test_auditors():
    print("\n3. arena auditors")

    illegal = arena.run_batch(
        red_codemaster="harness.selftest.IllegalCodemaster",
        red_guesser=arena.DEFAULT_HEURISTIC["guesser"],
        seeds=[1], single_team=True, n_jobs=1)[0]
    check("illegal clues detected", illegal["illegal_clue_count"] > 0)
    check("illegal clue carries a reason",
          "reason" in illegal["illegal_clues"][0])

    capped = arena.run_batch(
        red_codemaster="harness.selftest.StallingCodemaster",
        red_guesser="harness.selftest.NullGuesser",
        seeds=[1], single_team=True, n_jobs=1, max_turns=12)[0]
    check("turn cap surfaces as an error", capped["error"] is not None)
    check("turn cap error is TurnCapExceeded",
          "TurnCapExceeded" in str(capped["error"]))

    legal = arena.run_batch(seeds=[1], single_team=True, n_jobs=1)[0]
    check("heuristic agents emit zero illegal clues",
          legal["illegal_clue_count"] == 0)


def test_batches():
    print("\n4. batches + stats")

    batches = [
        ("single-team / default pool",
         dict(seeds=range(0, 8), single_team=True, pool="default", n_jobs=4)),
        ("two-team / default pool",
         dict(seeds=range(0, 8), single_team=False, pool="default", n_jobs=4)),
        ("single-team / slang pool",
         dict(seeds=range(100, 106), single_team=True, pool="slang", n_jobs=2)),
        ("two-team / slang pool",
         dict(seeds=range(100, 106), single_team=False, pool="slang", n_jobs=1)),
    ]

    everything = []
    for label, kwargs in batches:
        results = arena.run_batch(game_name=label, **kwargs)
        everything.extend(results)
        errors = [r for r in results if r.get("error")]
        check("%s: %d games, no crashes" % (label, len(results)), not errors)
        report = stats.format_report(results, title=label)
        check("%s: report renders" % label, len(report.splitlines()) > 10)
        print(report)

    check("ran at least 20 games", len(everything) >= 20)
    check("no illegal clues anywhere",
          sum(r["illegal_clue_count"] for r in everything) == 0)

    summary = stats.summarize(everything)
    check("summary counts every game", summary["games"] == len(everything))
    check("single-team block present", summary["single_team"] is not None)
    check("two-team block present", summary["two_team"] is not None)
    check("latency percentiles computed",
          summary["latency"]["overall"]["p95"] is not None)
    print(stats.format_report(summary, title="COMBINED (%d games)" % len(everything)))


def main(argv=None) -> int:
    print("harness selftest -- offline, no API keys required")
    try:
        test_llm_backend()
        test_secret_pool()
        test_auditors()
        test_batches()
    except AssertionError as exc:
        print("\nSELFTEST FAILED: %s" % exc)
        return 1
    print("\nAll harness selftest checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
