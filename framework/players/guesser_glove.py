"""Abra guesser -- GloVe embedding partner (partner zoo, not submitted).

Ranks the unrevealed board by cosine similarity to the clue and guesses down
that list, stopping at the clue number plus the traditional bonus guess when
the next word is still strongly associated.

Its value in cross-pairing evaluation is precisely that it has no idea what our
codemaster is thinking: it decodes a clue the way a static embedding does, so a
clue it finds obvious is one a stranger plausibly finds obvious too.

Falls back to letter-overlap similarity when the vector cache is absent.

Python 3.9 compatible.
"""

try:
    from players.guesser import Guesser
except Exception:  # pragma: no cover - keeps the file runnable standalone
    class Guesser(object):
        def __init__(self):
            self.move_history = []

        def set_move_history(self, move_history):
            self.move_history = move_history


try:
    from players import glove_common
    from players import heuristic_common
except Exception:  # pragma: no cover
    import glove_common
    import heuristic_common


class AIGuesser(Guesser):
    """Cosine-similarity guesser with a relative-confidence stop rule."""

    def __init__(self, team="Red", **kwargs):
        super(AIGuesser, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        #: Take the bonus (+1) guess only while the next word stays this close
        #: to the turn's best match.
        self.bonus_ratio = float(kwargs.get("bonus_ratio", 0.90))
        self.words = []
        self.clue = ""
        self.num = 1
        self.guesses = 0
        self._ranking = None
        self._ranking_key = None

    # -- framework hooks ---------------------------------------------------

    def set_board(self, words):
        self.words = list(words)

    def set_clue(self, clue, num):
        self.clue = str(clue or "")
        try:
            self.num = int(num)
        except (TypeError, ValueError):
            self.num = 1
        self.guesses = 0
        self._ranking = None
        self._ranking_key = None
        print("The clue is:", clue, num)
        return [clue, num]

    def get_answer(self):
        options = self._options()
        if not options:
            return None
        ranking = self._rank(options)
        self.guesses += 1
        for word, _ in ranking:
            if word in options:
                return word
        return options[0]

    def keep_guessing(self):
        try:
            return self._keep_guessing_inner()
        except Exception:  # noqa: BLE001 - a partner must never crash a game
            return False

    # -- internals ---------------------------------------------------------

    def _options(self):
        return [w for w in self.words if not glove_common._is_revealed(w)]

    def _limit(self):
        """Guess budget: clue number, with 0 (or a huge number) as unlimited."""
        if self.num <= 0 or self.num > len(self.words):
            return len(self._options())
        return max(1, self.num)

    def _keep_guessing_inner(self):
        options = self._options()
        if not options:
            return False
        limit = self._limit()
        if self.guesses < limit:
            return True
        if self.guesses >= limit + 1:
            return False
        ranking = self._rank(options)
        live = [(w, s) for w, s in ranking if w in options]
        if not live or ranking[0][1] <= 0:
            return False
        return live[0][1] >= self.bonus_ratio * ranking[0][1]

    def _rank(self, options):
        key = (self.clue, len(self.words))
        if self._ranking is not None and self._ranking_key == key:
            return self._ranking
        scores = self._embedding_scores(options)
        if scores is None:
            scores = dict(
                (word, heuristic_common.similarity(
                    glove_common._normalise(self.clue), word))
                for word in options)
        ranking = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        self._ranking = ranking
        self._ranking_key = key
        return ranking

    def _embedding_scores(self, options):
        loaded = glove_common.vectors()
        if loaded is None:
            return None
        _, matrix, index = loaded
        clue_row = index.get(glove_common._normalise(self.clue).lower())
        if clue_row is None:
            return None
        kept, rows = glove_common.board_rows(options, index)
        if not rows:
            return None
        sims = matrix[rows].dot(matrix[clue_row])
        scores = dict((word, float(value)) for word, value in zip(kept, sims))
        # Words missing from the vocabulary must still be guessable, just last.
        for word in options:
            scores.setdefault(word, -1.0)
        return scores
