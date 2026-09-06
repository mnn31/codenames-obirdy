"""Offline random Codemaster (no external API).

Deliberately weak floor-of-the-ladder opponent for the harness.  Emits a
uniformly random *legal* clue with number 1.
"""

import random

from players.codemaster import Codemaster
from players import heuristic_common as hc


class AICodemaster(Codemaster):

    def __init__(self, team="Red", seed=None, version=None):
        super().__init__()
        self.team = team
        self.version = version
        self.words = []
        self.maps = []
        self.rng = random.Random(seed if seed is not None
                                 else 11 + sum(ord(c) for c in str(team)))

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_clue(self):
        legal = [c for c in hc.CLUE_VOCAB if hc.clue_is_legal(c, self.words)]
        if not legal:
            return hc.pick_clue(self.words, self.maps, self.team, rng=self.rng)
        return [self.rng.choice(legal), 1]
