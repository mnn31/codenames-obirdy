"""Abra codemaster -- GloVe embedding partner (partner zoo, not submitted).

Plays the classic static-embedding strategy: score every candidate clue word by
cosine similarity to the unrevealed board, keep the clues whose own-team words
sit clearly above everything else, and report as the number how many own words
outrank the first non-own word.

Why it exists: the 2026 single-team track pairs our codemaster with other
entrants' guessers.  The Rattata heuristic partner is a *legality* floor -- it
finishes games without crashing but its "associations" are letter overlap, so
it cannot tell us whether our clues are decodable by someone who actually
understands the words.  Abra can: it is semantically real and knows nothing of
our conventions, which is exactly the stranger we need to survive.

Falls back to a letter-overlap score when the vector cache is absent, so this
file always runs.

Python 3.9 compatible.
"""

try:
    from players.codemaster import Codemaster
except Exception:  # pragma: no cover - keeps the file runnable standalone
    class Codemaster(object):
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


class AICodemaster(Codemaster):
    """Cosine-similarity codemaster with an assassin-aware margin rule."""

    def __init__(self, team="Red", **kwargs):
        super(AICodemaster, self).__init__()
        self.team = team if team in ("Red", "Blue") else "Red"
        self.opponent = "Blue" if self.team == "Red" else "Red"
        self.clue_vocab = int(
            kwargs.get("clue_vocab", glove_common.DEFAULT_CLUE_VOCAB))
        #: Minimum cosine gap between the last own word we claim and the best
        #: non-own word.  Bigger = fewer, safer words per clue.
        self.margin = float(kwargs.get("margin", 0.03))
        self.max_number = int(kwargs.get("max_number", 4))
        self.words = []
        self.maps = []

    # -- framework hooks ---------------------------------------------------

    def set_game_state(self, words, maps):
        self.words = list(words)
        self.maps = list(maps)

    def get_clue(self):
        try:
            clue = self._embedding_clue()
            if clue is not None:
                return clue
        except Exception:  # noqa: BLE001 - a partner must never crash a game
            pass
        return self._fallback_clue()

    # -- embedding path ----------------------------------------------------

    def _split(self):
        own, opp, civ, assassin = [], [], [], []
        for i, word in enumerate(self.words):
            if glove_common._is_revealed(word):
                continue
            kind = self.maps[i] if i < len(self.maps) else "Civilian"
            if kind == self.team:
                own.append(word)
            elif kind == self.opponent:
                opp.append(word)
            elif kind == "Assassin":
                assassin.append(word)
            else:
                civ.append(word)
        return own, opp, civ, assassin

    def _embedding_clue(self):
        loaded = glove_common.vectors()
        if loaded is None:
            return None
        words, matrix, index = loaded

        own, opp, civ, assassin = self._split()
        if not own:
            return None

        import numpy as np

        board = own + opp + civ + assassin
        kept, rows = glove_common.board_rows(board, index)
        if not rows:
            return None
        own_set = set(own)
        assassin_set = set(assassin)
        opp_set = set(opp)

        limit = min(self.clue_vocab, matrix.shape[0])
        # (vocab x dim) . (dim x board) -> cosine of every clue to every word.
        sims = matrix[:limit].dot(matrix[rows].T)

        own_cols = [i for i, w in enumerate(kept) if w in own_set]
        if not own_cols:
            return None
        # Danger weighting: the assassin must outrank nothing, opponent words
        # are bad, civilians merely waste a turn.
        risk_cols = []
        risk_weight = []
        for i, word in enumerate(kept):
            if word in assassin_set:
                risk_cols.append(i)
                risk_weight.append(0.20)     # treated as 0.20 better than it is
            elif word in opp_set:
                risk_cols.append(i)
                risk_weight.append(0.06)
            elif word not in own_set:
                risk_cols.append(i)
                risk_weight.append(0.02)

        own_sims = sims[:, own_cols]
        if risk_cols:
            adjusted = sims[:, risk_cols] + np.asarray(risk_weight, dtype="float32")
            ceiling = adjusted.max(axis=1)
        else:
            ceiling = np.full(limit, -1.0, dtype="float32")

        # Words we can claim: own words beating the danger ceiling by a margin.
        claimable = (own_sims > (ceiling + self.margin)[:, None])
        counts = claimable.sum(axis=1)
        # Score: how many we can claim, tie-broken by how strongly.
        strength = (own_sims * claimable).sum(axis=1)
        score = counts.astype("float32") * 10.0 + strength

        board_words = self.words
        order = np.argsort(-score)
        for candidate in order[:400]:
            if counts[candidate] < 1:
                break
            clue = words[candidate]
            if not glove_common.clue_is_legal(clue, board_words):
                continue
            number = int(min(counts[candidate], self.max_number, len(own)))
            return [clue.upper(), max(1, number)]
        return None

    # -- offline fallback --------------------------------------------------

    def _fallback_clue(self):
        """Letter-overlap clue, so a missing cache never breaks a batch."""
        own, opp, civ, assassin = self._split()
        best_clue, best_score = None, None
        for clue in heuristic_common.CLUE_VOCAB:
            if not glove_common.clue_is_legal(clue, self.words):
                continue
            total = 0.0
            for word in own:
                total += heuristic_common.similarity(clue, word)
            for word in opp + civ:
                total -= 0.5 * heuristic_common.similarity(clue, word)
            for word in assassin:
                total -= 4.0 * heuristic_common.similarity(clue, word)
            if best_score is None or total > best_score:
                best_clue, best_score = clue, total
        if best_clue is None:
            return ["SIGNAL", 1]
        return [best_clue, 1]
