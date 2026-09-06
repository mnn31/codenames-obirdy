"""Offline heuristic Guesser (no external API).

Sparring partner for the evaluation harness -- legal and fast, not strong.
"""

import random

from players.guesser import Guesser
from players import heuristic_common as hc


class AIGuesser(Guesser):
    """Signature mirrors players.guesser_GPT.AIGuesser (team passed positionally)."""

    def __init__(self, team="Red", seed=None, version=None):
        super().__init__()
        self.team = team
        self.version = version
        self.words = []
        self.clue = ""
        self.num = 0
        self.guesses = 0
        # Stable seed: PYTHONHASHSEED randomises hash() of str across processes.
        self.rng = random.Random(seed if seed is not None
                                 else 7 + sum(ord(c) for c in str(team)))

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = int(num) if num else 0
        # Reset the per-turn counter.  (The bundled GPT guesser never does this,
        # which silently caps its total guesses for the whole game.)
        self.guesses = 0
        return [clue, num]

    def keep_guessing(self):
        return self.guesses < self.num

    def get_answer(self):
        guess = hc.pick_guess(self.clue, self.words, rng=self.rng)
        if guess is None:
            return None
        self.guesses += 1
        return guess
