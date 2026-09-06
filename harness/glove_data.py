"""Prepare the GloVe vector cache used by the Abra partner agents.

The Abra agents (``framework/players/codemaster_glove.py`` /
``guesser_glove.py``) are a *partner zoo* entry, not a competition submission:
they exist so cross-pairing evaluation has a stranger that plays
**semantically** rather than by letter overlap.

The raw GloVe release is ~860 MB zipped and ~1 GB per dimension unzipped, which
is far too heavy to load in every one of the arena's spawned worker processes.
This module turns it into a compact ``.npz`` holding the most frequent slice of
the vocabulary as ``float32`` unit vectors -- 60k x 300 lands around 70 MB and
loads in well under a second.

Nothing here is committed: ``data/`` is gitignored.  See ``harness/README.md``
for the one-line download command.

Usage::

    python -m harness.glove_data --build              # from data/glove.6B.zip
    python -m harness.glove_data --dim 300 --vocab 60000
    python -m harness.glove_data --wide               # -> data/glove_wide.npz

The **wide** cache is a second, separate file built for
``harness/simtable.py``.  It exists because the two consumers want opposite
things from the vocabulary.  Abra wants the *frequent* slice: its clue
candidates are a prefix of this file, so a rare word entering the cache is a
rare word entering its mouth.  The similarity table wants the *deep* slice: a
board word only has to be looked up, never spoken, and the words a
slang/pop-culture pool is made of sit far down the frequency list -- ``PLATYPUS``
at GloVe rank 68860, ``XENOMORPH`` at 375895, ``SPEEDRUN`` at 398546.  Widening
``glove_cache.npz`` itself would have moved Abra's board lookups and made every
recorded baseline incomparable, so the caches are kept apart.

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "glove.6B.zip")
CACHE_PATH = os.path.join(DATA_DIR, "glove_cache.npz")
#: Deep-vocabulary cache for the similarity-table build -- see the module
#: docstring for why this is a separate file from ``glove_cache.npz``.
WIDE_CACHE_PATH = os.path.join(DATA_DIR, "glove_wide.npz")

DEFAULT_DIM = 300
DEFAULT_VOCAB = 60000
#: Frequency depth of the wide cache.  150k usable words is where the
#: pop-culture nouns a tournament pool is built from stop being findable in
#: bulk (see ``harness/simtable.py``); pool words below it are pulled in by
#: name via ``extra_words`` rather than by widening this further.
WIDE_VOCAB = 150000

#: GloVe is lowercase and includes punctuation, digits and subword junk.  Only
#: plain alphabetic words of a sane length are useful either as a board word or
#: as a clue.
MIN_WORD_LEN = 3
MAX_WORD_LEN = 16


def is_usable(word: str) -> bool:
    return (word.isalpha() and word.isascii()
            and MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN)


def _open_vectors(dim: int, zip_path: str):
    """Yield lines of ``glove.6B.<dim>d.txt`` from the zip or a plain file."""
    plain = os.path.join(DATA_DIR, "glove.6B.%dd.txt" % dim)
    if os.path.exists(plain):
        with open(plain, "r", encoding="utf-8") as handle:
            for line in handle:
                yield line
        return
    if not os.path.exists(zip_path):
        raise SystemExit(
            "no vectors found: expected %s or %s\n"
            "download it first -- see harness/README.md" % (plain, zip_path))
    member = "glove.6B.%dd.txt" % dim
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as raw:
            for line in raw:
                yield line.decode("utf-8")


def build_cache(dim: int = DEFAULT_DIM, vocab: int = DEFAULT_VOCAB,
                zip_path: str = ZIP_PATH,
                out_path: str = CACHE_PATH,
                extra_words: Optional[Iterable[str]] = None) -> str:
    """Write ``out_path`` with the top ``vocab`` usable words as unit vectors.

    GloVe files are ordered by corpus frequency, so taking a prefix keeps the
    common words a Codenames board and a clue vocabulary actually need.
    ``extra_words`` (e.g. a custom slang pool) are kept wherever they appear.
    """
    import numpy as np  # local: only the cache builder needs numpy at import

    wanted = set(w.strip().lower() for w in (extra_words or ()) if w)
    words: List[str] = []
    rows: List[List[float]] = []
    seen = set()

    for line in _open_vectors(dim, zip_path):
        parts = line.rstrip().split(" ")
        if len(parts) != dim + 1:
            continue
        word = parts[0]
        if word in seen or not is_usable(word):
            continue
        if len(words) >= vocab and word not in wanted:
            if not wanted:
                break          # nothing further can qualify
            continue
        seen.add(word)
        words.append(word)
        rows.append([float(x) for x in parts[1:]])

    matrix = np.asarray(rows, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix /= norms

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, words=np.array(words), vectors=matrix)
    return out_path


def pool_words() -> List[str]:
    """Every board word any bundled pool can deal, lowercased.

    Passed as ``extra_words`` to the wide build so a pool word is kept however
    far down the frequency list it sits.  The themed pools are included for
    exactly that reason: ``HYRULE`` sits at GloVe rank 195170 and ``DRUMKIT``
    at 269862, so a 150k prefix drops them even though the embedding knows
    both -- and a board word the similarity table cannot price is a word the
    sensor is blind to on a themed tournament board.
    """
    from harness.secret_pool import (ORGANISER_THEMED_BOARD, SLANG_POOL,
                                     THEMED_POOLS, load_default_pool)

    words = list(load_default_pool()) + list(SLANG_POOL)
    for pool in THEMED_POOLS.values():
        words.extend(pool)
    words.extend(ORGANISER_THEMED_BOARD)
    return [str(w).strip().lower() for w in words if w]


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--vocab", type=int, default=None)
    parser.add_argument("--zip", default=ZIP_PATH)
    parser.add_argument("--out", default=None)
    parser.add_argument("--build", action="store_true",
                        help="(default action; kept for readability)")
    parser.add_argument("--wide", action="store_true",
                        help="build the deep cache the simtable reads")
    args = parser.parse_args(argv)

    out = args.out or (WIDE_CACHE_PATH if args.wide else CACHE_PATH)
    vocab = args.vocab if args.vocab is not None else (
        WIDE_VOCAB if args.wide else DEFAULT_VOCAB)
    path = build_cache(dim=args.dim, vocab=vocab,
                       zip_path=args.zip, out_path=out,
                       extra_words=pool_words() if args.wide else None)
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    sys.stderr.write("wrote %s (%.1f MB)\n" % (path, size_mb))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
