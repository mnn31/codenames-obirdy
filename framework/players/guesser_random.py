"""Offline random Guesser (no external API).

Deliberately weak floor-of-the-ladder partner for the harness.  Picks a
uniformly random unrevealed board word and stops after ``num`` guesses.
"""

import random

from players.guesser import Guesser
from players import heuristic_common as hc


class AIGuesser(Guesser):

    def __init__(self, team="Red", seed=None, version=None):
        super().__init__()
        self.team = team
        self.version = version
        self.words = []
        self.clue = ""
        self.num = 0
        self.guesses = 0
        self.rng = random.Random(seed if seed is not None
                                 else 13 + sum(ord(c) for c in str(team)))

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = int(num) if num else 0
        self.guesses = 0
        return [clue, num]

    def keep_guessing(self):
        return self.guesses < self.num

    def get_answer(self):
        options = hc.unrevealed(self.words)
        if not options:
            return None
        self.guesses += 1
        return self.rng.choice(options)
