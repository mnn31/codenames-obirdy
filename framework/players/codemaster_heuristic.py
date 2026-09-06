"""Offline heuristic Codemaster (no external API).

Used as a sparring partner / smoke-test agent by the evaluation harness.
It plays legally and terminates quickly; it does not try to play well.

Run it exactly like the bundled GPT agents:

    python run_game.py players.codemaster_heuristic.AICodemaster \\
                       players.guesser_heuristic.AIGuesser \\
                       players.codemaster_heuristic.AICodemaster \\
                       players.guesser_heuristic.AIGuesser --seed 42
"""

import random

from players.codemaster import Codemaster
from players import heuristic_common as hc


class AICodemaster(Codemaster):
    """Signature mirrors players.codemaster_GPT.AICodemaster.

    ``game.Game`` constructs players as ``cls("Red", **kwargs)`` -- the team is
    passed positionally, and any extra kwargs (e.g. ``version``) come from the
    ``cmr_kwargs`` / ``cmb_kwargs`` dicts.
    """

    def __init__(self, team="Red", seed=None, version=None):
        super().__init__()
        self.team = team
        self.version = version
        self.words = []
        self.maps = []
        # Stable seed: PYTHONHASHSEED randomises hash() of str across processes.
        self.rng = random.Random(seed if seed is not None
                                 else sum(ord(c) for c in str(team)))

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_clue(self):
        return hc.pick_clue(self.words, self.maps, self.team, rng=self.rng)
