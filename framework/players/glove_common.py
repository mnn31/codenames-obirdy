"""Shared vector machinery for the Abra partner agents.

Abra is the partner-zoo entry that plays **semantically**: it scores clues and
guesses by cosine similarity in a static GloVe embedding space rather than by
the letter overlap the Rattata heuristic agents use.  It exists so the
cross-pairing evaluation has a realistic stranger -- a competent, non-LLM
teammate whose associations are genuinely its own -- instead of a bot whose
"associations" our agents could never anticipate because they are not
associations at all.

These agents are **not** a competition submission.  They are ours, they live
here only because the framework imports players by dotted path, and they are
deliberately allowed to depend on numpy and on a gitignored ``data/`` cache
that a submitted agent could never assume.

If the cache is missing the agents still run: ``vectors()`` returns ``None``
and the callers fall back to the offline letter-overlap similarity, so the test
suite and the harness selftest work on a fresh checkout with no download.

Build the cache with ``python -m harness.glove_data`` (see harness/README.md).

Python 3.9 compatible.
"""

import os
import re
import threading

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_PATH = os.environ.get(
    "ABRA_GLOVE_CACHE", os.path.join(REPO_ROOT, "data", "glove_cache.npz"))

#: How many of the (frequency-ordered) cached words may be proposed as a clue.
#: The whole cache is used for *lookups*; only this prefix is used as clue
#: candidates, because rarer words make clues no human partner would decode.
DEFAULT_CLUE_VOCAB = 20000

_LOCK = threading.Lock()
_STATE = {"loaded": False, "words": None, "vectors": None, "index": None}


def _normalise(word):
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def _is_revealed(board_word):
    text = str(board_word)
    return len(text) > 0 and text[0] == "*"


def vectors():
    """``(words, matrix, index)`` for the cache, or ``None`` if unavailable.

    ``matrix`` rows are L2-normalised ``float32``, so a cosine similarity is a
    plain dot product.  Loaded once per process and shared by both agents.
    """
    if _STATE["loaded"]:
        return _STATE["vectors_tuple"]
    with _LOCK:
        if _STATE["loaded"]:
            return _STATE["vectors_tuple"]
        result = None
        try:
            import numpy as np
            with np.load(CACHE_PATH, allow_pickle=False) as data:
                words = [str(w) for w in data["words"]]
                matrix = data["vectors"]
            index = dict((word, i) for i, word in enumerate(words))
            result = (words, matrix, index)
        except Exception:  # noqa: BLE001 - a missing cache is not an error
            result = None
        _STATE["vectors_tuple"] = result
        _STATE["loaded"] = True
        return result


def board_rows(board_words, index):
    """``(kept_words, row_numbers)`` for board words present in the vocabulary."""
    kept = []
    rows = []
    for word in board_words:
        key = _normalise(word).lower()
        row = index.get(key)
        if row is None:
            continue
        kept.append(word)
        rows.append(row)
    return kept, rows


def clue_is_legal(clue, board_words):
    """Competition legality: one alphabetic word, no board-word derivation."""
    if not clue:
        return False
    text = str(clue).strip().upper()
    if len(text.split()) != 1 or not text.isalpha() or len(text) < 3:
        return False
    for word in board_words or ():
        if _is_revealed(word):
            continue
        norm = _normalise(word)
        if norm and (text in norm or norm in text):
            return False
    return True
