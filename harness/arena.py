"""Batch game runner for the vendored Codenames competition framework.

Design constraints
------------------
* ``framework/`` is the official 2026 competition code and is **never** edited.
  Everything here instruments it from the outside: ``game.Game`` is subclassed
  to capture the turn count it would otherwise only write to a log file, and
  the player classes are wrapped in generated subclasses that time each call
  and audit clue legality.
* ``game.Game`` opens ``players/cm_wordlist.txt`` and writes ``results/*.txt``
  using **paths relative to the process cwd**.  Every game therefore runs with
  its cwd set to a throwaway sandbox directory (see ``harness.secret_pool``),
  which is also how alternative word pools are injected.  Logging is captured
  in-process instead of hitting disk, so no files are produced at all.
* Games are independent, so batches can fan out over a ``multiprocessing``
  pool.  Worker payloads are plain dicts of strings/ints (picklable under the
  ``spawn`` start method used on macOS).

Framework quirks worth knowing (discovered by reading ``game.py``):

* Players are constructed as ``cls(team, **kwargs)`` -- team is **positional**.
* The board is 9 red / 8 blue / 7 civilian / 1 assassin, and **red always moves
  first**.
* ``run()`` keeps ``turn_counter`` as a local and only surfaces it through
  ``write_results``; there is no attribute to read afterwards.
* A guesser returning ``None`` breaks the inner loop *without* passing the turn,
  so a guesser that always returns ``None`` hangs the engine.  We defend with a
  turn cap in the codemaster wrapper.
* Clue legality is *not* enforced by the engine at all -- ``codemaster_GPT``
  polices itself.  We audit it here so we can report an illegal-clue rate.

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import multiprocessing
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

from harness import secret_pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")

#: Hard ceiling on clues per game; guards against a pathological agent that
#: never lets the engine terminate.
DEFAULT_MAX_TURNS = 80

#: Single-team track: a loss scores as this many turns.
LOSS_SCORE = 25

DEFAULT_HEURISTIC = {
    "codemaster": "players.codemaster_heuristic.AICodemaster",
    "guesser": "players.guesser_heuristic.AIGuesser",
}


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def ensure_framework_on_path() -> None:
    if FRAMEWORK_DIR not in sys.path:
        sys.path.insert(0, FRAMEWORK_DIR)


def import_class(path: str):
    """'players.codemaster_GPT.AICodemaster' -> the class object."""
    ensure_framework_on_path()
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError("expected a dotted import path, got %r" % path)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class TurnCapExceeded(RuntimeError):
    """Raised by the instrumented codemaster when a game refuses to end."""


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

class _Recorder:
    """Per-game collector shared by the wrapped players."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        self.max_turns = max_turns
        self.latencies: List[Dict[str, Any]] = []
        self.illegal_clues: List[Dict[str, Any]] = []
        self.clues: List[Dict[str, Any]] = []
        self.turns_started = 0

    def time_call(self, role: str, method: str, fn, *args, **kwargs):
        started = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            self.latencies.append({
                "role": role,
                "method": method,
                "seconds": time.time() - started,
            })


def _audit_clue(clue: Any, number: Any, board_words: Sequence[str]) -> Optional[str]:
    """Return a reason string when the clue breaks competition rules, else None.

    Single English word, not derived from / deriving any unrevealed board
    word, number >= 0.  Note ``0`` is legal and means *unlimited guesses*
    (framework README); the bundled ``codemaster_GPT`` rejects it only because
    it polices itself more tightly than the rules require.
    """
    if clue is None:
        return "clue is None"
    text = str(clue).strip().upper()
    if not text:
        return "empty clue"
    if len(text.split()) != 1:
        return "clue is not a single word"
    if not text.isalpha():
        return "clue contains non-alphabetic characters"
    try:
        num = int(number)
    except (TypeError, ValueError):
        return "clue number is not an integer"
    if num < 0:
        return "clue number < 0"
    for word in board_words or ():
        if not word or word[0] == "*":
            continue
        norm = "".join(ch for ch in str(word).upper() if ch.isalpha())
        if not norm:
            continue
        if text in norm or norm in text:
            return "clue derives from board word %s" % norm
    return None


def wrap_codemaster(base_cls, role: str, recorder: _Recorder):
    """Generate a Codemaster subclass that times calls and audits clues."""

    class InstrumentedCodemaster(base_cls):  # type: ignore[misc, valid-type]
        def set_game_state(self, words, maps):
            self._harness_words = list(words)
            return base_cls.set_game_state(self, words, maps)

        def get_clue(self):
            recorder.turns_started += 1
            if recorder.turns_started > recorder.max_turns:
                raise TurnCapExceeded(
                    "exceeded %d clues in one game" % recorder.max_turns)

            result = recorder.time_call(role, "get_clue", base_cls.get_clue, self)

            board = getattr(self, "_harness_words", []) or []
            try:
                clue, number = result[0], result[1]
            except (TypeError, IndexError, KeyError):
                recorder.illegal_clues.append({
                    "role": role, "clue": repr(result),
                    "reason": "clue is not a (word, number) pair",
                })
                return result

            reason = _audit_clue(clue, number, board)
            entry = {"role": role, "clue": str(clue), "number": number}
            recorder.clues.append(entry)
            if reason:
                bad = dict(entry)
                bad["reason"] = reason
                recorder.illegal_clues.append(bad)
            return result

    InstrumentedCodemaster.__name__ = base_cls.__name__
    InstrumentedCodemaster.__qualname__ = base_cls.__qualname__
    return InstrumentedCodemaster


def wrap_guesser(base_cls, role: str, recorder: _Recorder):
    """Generate a Guesser subclass that times ``get_answer``/``keep_guessing``."""

    class InstrumentedGuesser(base_cls):  # type: ignore[misc, valid-type]
        def get_answer(self):
            return recorder.time_call(role, "get_answer", base_cls.get_answer, self)

        def keep_guessing(self):
            return recorder.time_call(
                role, "keep_guessing", base_cls.keep_guessing, self)

    InstrumentedGuesser.__name__ = base_cls.__name__
    InstrumentedGuesser.__qualname__ = base_cls.__qualname__
    return InstrumentedGuesser


def make_instrumented_game(game_cls):
    """Subclass ``game.Game`` so ``write_results`` captures instead of writing.

    ``run()`` only exposes its ``turn_counter`` by handing it to
    ``write_results``; overriding that is the least invasive way to capture the
    engine's own turn count, and it also stops the framework creating a
    ``results/`` directory.
    """

    class InstrumentedGame(game_cls):  # type: ignore[misc, valid-type]
        captured_turns = None

        def write_results(self, num_of_turns):
            self.captured_turns = num_of_turns

    return InstrumentedGame


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def collect_usage(game) -> Dict[str, Dict[str, Any]]:
    """Per-role LLM usage, for agents that expose ``usage_summary()``.

    Offline agents have no such method, so the dict is simply missing those
    roles.  This is how a batch's API spend is estimated (``harness.stats``).
    """
    roles = {
        "red_cm": "codemaster_red", "red_g": "guesser_red",
        "blue_cm": "codemaster_blue", "blue_g": "guesser_blue",
    }
    usage: Dict[str, Dict[str, Any]] = {}
    for role, attribute in roles.items():
        player = getattr(game, attribute, None)
        summary = getattr(player, "usage_summary", None)
        if not callable(summary):
            continue
        try:
            usage[role] = dict(summary())
        except Exception:  # noqa: BLE001 - diagnostics must never break a run
            continue
    return usage


def _blank_result(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "seed": spec.get("seed"),
        "single_team": bool(spec.get("single_team")),
        "pool": spec.get("pool_name"),
        "game_name": spec.get("game_name", "default"),
        "red_codemaster": spec.get("red_codemaster"),
        "red_guesser": spec.get("red_guesser"),
        "blue_codemaster": spec.get("blue_codemaster"),
        "blue_guesser": spec.get("blue_guesser"),
        "winner": None,
        "turns": None,
        "score": None,
        "red_found": 0,
        "blue_found": 0,
        "civilians_hit": 0,
        "assassin_hit": False,
        "assassin_hitter": None,
        "illegal_clues": [],
        "illegal_clue_count": 0,
        "clue_count": 0,
        "clues": [],
        "latencies": [],
        "usage": {},
        "duration_s": None,
        "error": None,
    }


def run_single_game(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Play one game described by ``spec`` and return a plain-dict result.

    ``spec`` keys: ``seed``, ``single_team``, ``sandbox`` (cwd containing
    ``players/cm_wordlist.txt``), the four dotted agent paths, optional
    ``*_kwargs`` dicts, ``game_name``, ``max_turns``, ``pool_name``.
    """
    result = _blank_result(spec)
    started = time.time()

    previous_cwd = os.getcwd()
    ensure_framework_on_path()

    try:
        from game import Game  # noqa: WPS433 - deliberately late (needs sys.path)

        recorder = _Recorder(int(spec.get("max_turns") or DEFAULT_MAX_TURNS))

        cmr = wrap_codemaster(import_class(spec["red_codemaster"]), "red_cm", recorder)
        gr = wrap_guesser(import_class(spec["red_guesser"]), "red_g", recorder)
        cmb = wrap_codemaster(import_class(spec["blue_codemaster"]), "blue_cm", recorder)
        gb = wrap_guesser(import_class(spec["blue_guesser"]), "blue_g", recorder)

        sandbox = spec.get("sandbox")
        if sandbox:
            os.chdir(sandbox)

        InstrumentedGame = make_instrumented_game(Game)

        # game.Game prints heavily and, with do_print=False, swaps out and later
        # *closes* sys.stdout in __del__.  Keeping do_print=True and redirecting
        # ourselves is safer and equally silent.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            game = InstrumentedGame(
                cmr, gr, cmb, gb,
                seed=int(spec["seed"]),
                do_print=True,
                do_log=True,          # routed to captured_turns, not to disk
                game_name=spec.get("game_name", "default"),
                cmr_kwargs=dict(spec.get("cmr_kwargs") or {}),
                gr_kwargs=dict(spec.get("gr_kwargs") or {}),
                cmb_kwargs=dict(spec.get("cmb_kwargs") or {}),
                gb_kwargs=dict(spec.get("gb_kwargs") or {}),
                single_team=bool(spec.get("single_team")),
            )
            game.run()

        board = list(game.words_on_board)
        result["winner"] = game.game_winner
        result["turns"] = game.captured_turns
        result["red_found"] = board.count("*Red*")
        result["blue_found"] = board.count("*Blue*")
        result["civilians_hit"] = board.count("*Civilian*")
        result["assassin_hit"] = board.count("*Assassin*") > 0
        result["board"] = board
        result["move_history"] = list(game.get_move_history())

        if result["assassin_hit"]:
            for move in reversed(result["move_history"]):
                if len(move) >= 3 and str(move[0]).endswith("_Guesser") \
                        and move[2] == "*Assassin*":
                    result["assassin_hitter"] = str(move[0]).split("_")[0]
                    break

        if bool(spec.get("single_team")):
            won = result["winner"] == "R"
            result["score"] = result["turns"] if won else LOSS_SCORE

        result["illegal_clues"] = recorder.illegal_clues
        result["illegal_clue_count"] = len(recorder.illegal_clues)
        result["clue_count"] = len(recorder.clues)
        result["clues"] = recorder.clues
        result["latencies"] = recorder.latencies
        result["usage"] = collect_usage(game)

    except BaseException as exc:  # noqa: BLE001 - a crashed game is a data point
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        result["traceback"] = traceback.format_exc()
    finally:
        os.chdir(previous_cwd)
        result["duration_s"] = time.time() - started

    return result


def _worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level entry point for multiprocessing (must be importable)."""
    return run_single_game(spec)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def build_specs(
    seeds: Sequence[int],
    red_codemaster: str,
    red_guesser: str,
    blue_codemaster: Optional[str] = None,
    blue_guesser: Optional[str] = None,
    single_team: bool = False,
    sandbox: Optional[str] = None,
    pool_name: Optional[str] = None,
    game_name: str = "default",
    max_turns: int = DEFAULT_MAX_TURNS,
    cmr_kwargs: Optional[Dict[str, Any]] = None,
    gr_kwargs: Optional[Dict[str, Any]] = None,
    cmb_kwargs: Optional[Dict[str, Any]] = None,
    gb_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    blue_codemaster = blue_codemaster or red_codemaster
    blue_guesser = blue_guesser or red_guesser
    return [
        {
            "seed": int(seed),
            "single_team": bool(single_team),
            "sandbox": sandbox,
            "pool_name": pool_name,
            "game_name": game_name,
            "max_turns": int(max_turns),
            "red_codemaster": red_codemaster,
            "red_guesser": red_guesser,
            "blue_codemaster": blue_codemaster,
            "blue_guesser": blue_guesser,
            "cmr_kwargs": cmr_kwargs or {},
            "gr_kwargs": gr_kwargs or {},
            "cmb_kwargs": cmb_kwargs or {},
            "gb_kwargs": gb_kwargs or {},
        }
        for seed in seeds
    ]


def run_batch(
    red_codemaster: str = DEFAULT_HEURISTIC["codemaster"],
    red_guesser: str = DEFAULT_HEURISTIC["guesser"],
    blue_codemaster: Optional[str] = None,
    blue_guesser: Optional[str] = None,
    seeds: Sequence[int] = range(10),
    single_team: bool = False,
    pool=None,
    n_jobs: int = 1,
    game_name: str = "default",
    max_turns: int = DEFAULT_MAX_TURNS,
    progress: bool = False,
    cmr_kwargs: Optional[Dict[str, Any]] = None,
    gr_kwargs: Optional[Dict[str, Any]] = None,
    cmb_kwargs: Optional[Dict[str, Any]] = None,
    gb_kwargs: Optional[Dict[str, Any]] = None,
    sandbox_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run ``len(seeds)`` games and return one result dict per game.

    ``pool`` is a pool name ('default' / 'slang'), an explicit word list, or a
    path to a wordlist file; it is materialised into a temporary sandbox
    directory used as the games' cwd, so ``framework/`` is left untouched and
    no ``results/`` files are created.
    """
    seeds = [int(s) for s in seeds]
    pool_name = pool if isinstance(pool, str) else ("custom" if pool else "default")

    sandbox = secret_pool.make_sandbox(pool, root=sandbox_root)
    try:
        specs = build_specs(
            seeds=seeds,
            red_codemaster=red_codemaster,
            red_guesser=red_guesser,
            blue_codemaster=blue_codemaster,
            blue_guesser=blue_guesser,
            single_team=single_team,
            sandbox=sandbox,
            pool_name=pool_name,
            game_name=game_name,
            max_turns=max_turns,
            cmr_kwargs=cmr_kwargs,
            gr_kwargs=gr_kwargs,
            cmb_kwargs=cmb_kwargs,
            gb_kwargs=gb_kwargs,
        )

        if n_jobs and n_jobs > 1 and len(specs) > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=min(int(n_jobs), len(specs))) as pool_exec:
                results = pool_exec.map(_worker, specs)
        else:
            results = []
            for index, spec in enumerate(specs, 1):
                results.append(run_single_game(spec))
                if progress:
                    sys.stderr.write("\r  game %d/%d" % (index, len(specs)))
                    sys.stderr.flush()
            if progress:
                sys.stderr.write("\n")
        return list(results)
    finally:
        secret_pool.destroy_sandbox(sandbox)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_seeds(text: str) -> List[int]:
    """'0-9', '1,2,5', '0-4,10' -> list of ints."""
    seeds: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:]:
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch of Codenames games through the vendored framework.")
    parser.add_argument("--red-cm", default=DEFAULT_HEURISTIC["codemaster"])
    parser.add_argument("--red-g", default=DEFAULT_HEURISTIC["guesser"])
    parser.add_argument("--blue-cm", default=None,
                        help="defaults to --red-cm")
    parser.add_argument("--blue-g", default=None, help="defaults to --red-g")
    parser.add_argument("--seeds", default="0-19", help="e.g. 0-19 or 1,2,3")
    parser.add_argument("--single-team", action="store_true",
                        help="single-team track (red only)")
    parser.add_argument("--pool", default="default",
                        help="word pool: default, slang, or a wordlist path")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--game-name", default="default")
    parser.add_argument("--out", default=None, help="write results JSON here")
    parser.add_argument("--quiet", action="store_true",
                        help="skip the printed report")
    args = parser.parse_args(argv)

    results = run_batch(
        red_codemaster=args.red_cm,
        red_guesser=args.red_g,
        blue_codemaster=args.blue_cm,
        blue_guesser=args.blue_g,
        seeds=parse_seeds(args.seeds),
        single_team=args.single_team,
        pool=args.pool,
        n_jobs=args.jobs,
        game_name=args.game_name,
        max_turns=args.max_turns,
        progress=not args.quiet,
    )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        sys.stderr.write("wrote %s (%d games)\n" % (args.out, len(results)))

    if not args.quiet:
        from harness import stats
        print(stats.format_report(results, title="arena: %s" % args.game_name))

    return 1 if any(r.get("error") for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
