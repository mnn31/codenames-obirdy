"""Build the bundled clue/board similarity table the codemaster ships with.

Why this file exists
--------------------
Our codemaster's only view of a clue's danger is the LLM panel, and the panel
reports the words it *happened to rank*.  Five of the eight games in the
2026-07-31 ``obirdy_cm_abra_g`` cross-pairing died on an assassin the panel
never surfaced; four of those five are visible in a static embedding
(``SURGE``->LEAD 0.281 against its own target CHARGE 0.169, and so on).  The
danger probe closes half of the gap by asking the model directly, but it costs
an API call and it can only ask about a handful of words.  A bundled table
closes the other half for free and offline.

What it contains
----------------
For every clue word in a bundled vocabulary, the board words it pulls on
hardest -- GloVe cosine, exact (not a reduced-dimension approximation, which
was measured and is far too lossy at the 0.02-0.05 margins the decision turns
on), quantised onto a 64-entry codebook shipped in the header.

Board vocabulary, in three separately-ranked banks: the framework's 395-word
pool plus the harness slang pool; every word of every themed pool in
``harness/secret_pool.py`` (including the organisers' 25-word board verbatim);
and a generated bank of tournament-style words (see
:func:`tournament_candidates`).  All three are restricted to words the
embedding actually knows.  Clue vocabulary: a frequency prefix of GloVe, plus
the codemaster's own offline fallback vocabulary, plus every clue our recorded
games have ever produced.

Why the board vocabulary is generated rather than listed
-------------------------------------------------------
The competition pool is secret and described as slang/pop-culture heavy, so the
two pools we *have* are a sample, not the population.  Covering only them left
the sensor reading 393 of 395 default-pool words but 123 of 232 slang words --
half-blind in exactly the half the tournament lives in.  Two separate causes
were behind those 109 misses and only one of them is fixable:

* **70** were plain frequency misses.  The word is in GloVe 6B, just below the
  60k prefix ``data/glove_cache.npz`` keeps (``PIKACHU`` at rank 92066,
  ``XENOMORPH`` 375895, ``SPEEDRUN`` 398546).  Reading the deeper
  ``data/glove_wide.npz`` recovers every one of them -- and the two default-pool
  misses, ``LEPRECHAUN`` 78868 and ``PLATYPUS`` 68860, with them.
* **39** are genuinely out of vocabulary in any casing or hyphenation --
  post-2014 coinages (``TIKTOK``, ``FORTNITE``, ``DEEPFAKE``, ``BLOCKCHAIN``,
  ``YEET``) and coined compounds (``LOOTBOX``, ``MOSHPIT``, ``PLOTTWIST``,
  ``SITUATIONSHIP``).  GloVe 6B was trained on a 2014 dump; ``tik-tok`` exists
  at rank 205909 but means a clock noise, and nothing else has so much as a
  hyphenated form.  No embedding fix exists for these, and none is attempted:
  the codemaster already handles them correctly, since a board word the table
  cannot price counts as the floor rather than being dropped.

The themed pools cost nothing on that second count: **every** word of every
themed pool is reachable, ``HYRULE`` included.  It is absent from the 150k wide
cache (GloVe rank 195170, and ``DRUMKIT`` 269862), not from GloVe, so
``harness.glove_data.pool_words`` now names the themed pools too and the
rebuilt cache carries both.  ``UNREACHABLE_THEMED`` is empty, and pinned empty.

Format (after gunzip), all little-endian
----------------------------------------
::

    b"OBSIM1\\n" | b"OBSIM2\\n"
    <header JSON>\\n            counts, floor, codebook
    <board word>\\n  x n_board
    <clue word>\\n   x n_clue
    <offsets>                  (n_clue + 1) x uint32, entry index per clue
    -- version 1 --
    <entries>                  uint16 = (board index << 6) | code
    -- version 2 --
    <indices>                  uint16 x n_entries, board index
    <codes>                    uint8  x n_entries, codebook level

Version 1 packs the index and the code into one 16-bit word, which caps the
board vocabulary at ``2**10 = 1024``.  Covering the themed pools needs more
than that (1008 + 176 new themed words before a single generated candidate),
so version 2 splits the two into parallel arrays: the index gets a full 16
bits (65536 board words, 64x the old ceiling) and the code its own byte.

Splitting is what makes the wider index free.  The obvious widening -- one
24-bit packed entry -- costs 3.82 MB gzipped where the old 16-bit packing cost
3.30 MB, because interleaving a 6-bit code with an index destroys both
streams' regularity.  Split apart, the same 1.74 M entries gzip to **3.01 MB**:
the code stream is 64 symbols with a heavy skew, and the index stream is
ascending runs.  A wider index that also makes the file smaller than the one it
replaces is not a trade-off worth agonising over.

Both versions are read by ``framework/players/codemaster_obirdy.py``, and the
builder emits version 1 whenever the board vocabulary still fits in 1024 words,
so a small table built by this module stays readable by a pre-2026-08-03 agent.

The reader needs only ``gzip`` and ``json``; there is no numpy, no matrix and
no float parsing at load time.

Usage::

    python -m harness.simtable --build
    python -m harness.simtable --stats

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
#: Preferred source: the deep cache (``python -m harness.glove_data --wide``).
#: ``glove_cache.npz`` is accepted as a fallback so a checkout that only ran the
#: documented one-liner still builds a table -- it just builds a shallower one.
GLOVE_WIDE = os.path.join(DATA_DIR, "glove_wide.npz")
GLOVE_CACHE = os.path.join(DATA_DIR, "glove_cache.npz")
RESULTS_DIR = os.path.join(REPO_ROOT, "results_local")


def default_glove() -> str:
    return GLOVE_WIDE if os.path.exists(GLOVE_WIDE) else GLOVE_CACHE

#: Ships in the agent's own ``players/oBirdy/`` subfolder, not under the
#: gitignored ``data/``: it is part of the submission ("entrants may submit
#: additional files"), so it is committed.  The per-team subfolder is what the
#: organisers asked for -- a flat ``players/`` has every entry's data files
#: competing for the same names.
OUT_PATH = os.path.join(REPO_ROOT, "framework", "players", "oBirdy",
                        "obirdy_simtable.bin.gz")

#: Magic per format version; the index is the version number.
MAGIC_BY_VERSION = {1: b"OBSIM1\n", 2: b"OBSIM2\n"}
#: Kept under its old name for callers that only ever wrote version 1.
MAGIC = MAGIC_BY_VERSION[1]
DEFAULT_VERSION = 2

#: Similarities below this are not stored.  Every one of the six recorded
#: codemaster-side assassin deaths has a clue-assassin cosine at or above it
#: (the smallest is LADDER-BOX at 0.145), and the floor is what keeps the file
#: small.
DEFAULT_FLOOR = 0.12
#: Board words kept per clue, **counted over the two real pools only**.  The
#: deaths need depth: the fatal word sat at pool rank 12 (PAGEANT-STATE), 13
#: (SURGE-LEAD), 28 (PACKING-OLIVE) and 53 (LADDER-BOX), so a shallow top-16 or
#: top-32 list would have missed the last.
#:
#: The quota is per-pool rather than global because it has to be.  Merging the
#: generated candidate bank into one global top-64 doubles the field a pool word
#: competes in, and measurably pushed LADDER-BOX off the end -- the sensor went
#: silent on a death it used to catch.  Ranking the banks separately keeps every
#: pool word at exactly the depth it had before this vocabulary grew, and that
#: is why the themed pools went in as a *third* bank rather than being merged
#: into this one: 176 new words in the same field is the same displacement
#: argument again, and LADDER-BOX sits at pool rank 53 of 64.
DEFAULT_TOP_K = 64
#: Board words kept per clue from the *themed-pool* bank.  Half the pool depth:
#: a themed board deals 25 words out of a 40-50 word pool, so a clue's danger on
#: it is concentrated in far fewer columns than a 588-word pool spreads it over,
#: and the measured mean is 13 words above the floor -- the quota is headroom,
#: not a budget being spent.
DEFAULT_THEMED_TOP_K = 32
#: Board words kept per clue from the *generated* bank.  Shallower on purpose:
#: these are guesses at a pool we have never seen, so they buy breadth cheaply
#: and are not worth paying rank-53 depth for.
DEFAULT_CANDIDATE_TOP_K = 16
#: Frequency prefix of the GloVe vocabulary offered as clue words.  Raised from
#: 8000 because the 8000-word prefix could not price 19.5% of the clues the
#: recorded slang games actually issued (against 7.9% on the default pool):
#: BAGEL, BANANA, QUANTUM, DIMENSION and ADORABLE all sit between rank 9k and
#: 33k, and a clue the table cannot find takes the out-of-vocabulary demotion
#: instead of being judged.  30000 takes both figures to 0.0% and 4.9% -- the
#: four survivors are coined compounds (FACIALHAIR, GAMECHARACTER) which are
#: exactly what the demotion is for.  This is the size knob: the file grows
#: almost linearly in it, and 45000 would cost 4.9 MB for 0.2 points of clue
#: coverage.
DEFAULT_CLUE_VOCAB = 30000
#: Absolute depth limit on the clue side, applied to *every* source.
#:
#: The clue vocabulary answers a different question from the board vocabulary.
#: A board word is a card that will be dealt whether or not anyone can decode
#: it, so depth there is free information.  A clue is a word we are about to
#: say to a partner, and the whole point of the out-of-vocabulary demotion is
#: to model "a static-embedding partner will not decode this".  Reading a
#: deeper cache must not quietly answer that question with a deeper vocabulary
#: than the partner has: it made ``EGGSHELL`` (GloVe rank 73529, one of the two
#: clues this demotion was built from) suddenly look decodable, when the Abra
#: partner that lost the game to it caches 60k words and still cannot read it.
#: 60000 is that partner's depth, and the depth of the cache the documented
#: one-liner builds.
CLUE_VOCAB_CEILING = 60000
#: Codebook size, both versions.  In version 1 these 6 bits share a 16-bit word
#: with the board index; in version 2 they get a byte of their own.
CODE_BITS = 6
CODE_LEVELS = 1 << CODE_BITS
#: Hard ceiling on the board vocabulary, per format version: 10 bits of index in
#: version 1, a full 16-bit index of its own in version 2.
MAX_BOARD_WORDS_BY_VERSION = {1: 1 << (16 - CODE_BITS), 2: 1 << 16}
MAX_BOARD_WORDS = MAX_BOARD_WORDS_BY_VERSION[DEFAULT_VERSION]
LEGACY_MAX_BOARD_WORDS = MAX_BOARD_WORDS_BY_VERSION[1]

# -- tournament candidate generation ---------------------------------------
#: Neighbours considered per slang seed.
CANDIDATE_PER_SEED = 25
#: Minimum cosine for a neighbour to count as a candidate.
CANDIDATE_MIN_SIM = 0.45
#: Frequency band a candidate must sit in, as a row range of the wide cache.
#: Below ~2000 is function-word and newswire territory ("said", "government"),
#: which no pool deals; above ~120000 the neighbours stop being words a player
#: would recognise on a card.
CANDIDATE_RANK_LO = 2000
CANDIDATE_RANK_HI = 120000
#: How many generated words the bank may hold.
#:
#: Under version 1 this was not a judgement at all -- it was ``1024 - len(the
#: real pools)``, whatever the index had left over, which came to 420.  Version
#: 2's index holds 65536, so the number has to be argued rather than inherited.
#: The binding constraint is now the 5 MB the file has to fit in, and the
#: generated bank is the only part of the vocabulary that is a guess.  Measured,
#: everything else held fixed:
#:
#:     budget      0 -> 764 board words, format 1, 3.41 MB
#:     budget    512 -> 1276 board words, 3.99 MB   (+0.58)
#:     budget   1024 -> 1713 board words, 4.10 MB   (+0.11)
#:
#: The second 512 is nearly free because the bank's own top-16 quota is already
#: full for most clues by then -- more breadth, barely more stored pairs.  That
#: is the argument for 1024 and also the argument against 2048: past the quota
#: the file stops growing and so does the information, while the guesses get
#: worse the further down the vote ranking they come from.
#: (The filters retire some of the budget; 1024 yields 949 words.)
DEFAULT_CANDIDATE_BUDGET = 1024
#: Headroom left under the format's board ceiling so a later pool edit cannot
#: silently overflow the index.
BOARD_HEADROOM = 16


# ---------------------------------------------------------------------------
# Vocabulary assembly
# ---------------------------------------------------------------------------

def _fallback_vocab() -> List[str]:
    """The codemaster's own offline clue vocabulary, read from the agent."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "framework"))
    try:
        from players import codemaster_obirdy as cm  # noqa: WPS433
        return list(cm._FALLBACK_VOCAB)
    except Exception:
        return []


def recorded_clues(results_dir: str = RESULTS_DIR) -> List[str]:
    """Every clue any recorded game produced -- 'the panel's common outputs'."""
    seen = set()
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path, "r") as handle:
                games = json.load(handle)
        except Exception:
            continue
        if not isinstance(games, list):
            continue
        for game in games:
            if not isinstance(game, dict):
                continue
            for move in game.get("move_history") or ():
                if (isinstance(move, (list, tuple)) and len(move) > 1
                        and str(move[0]).endswith("Codemaster")):
                    seen.add(str(move[1]).strip().upper())
    return sorted(seen)


def _uppercase_unique(words: Iterable[str], seen=None) -> List[str]:
    out: List[str] = []
    seen = set() if seen is None else seen
    for word in words:
        upper = str(word).strip().upper()
        if upper and upper not in seen:
            seen.add(upper)
            out.append(upper)
    return out


def bundled_pools() -> List[str]:
    """The two word pools we actually have, uppercased and de-duplicated."""
    from harness.secret_pool import SLANG_POOL, load_default_pool

    return _uppercase_unique(list(load_default_pool()) + list(SLANG_POOL))


def themed_pool_words() -> List[str]:
    """Every themed-pool word, minus anything the two real pools already hold.

    Includes ``ORGANISER_THEMED_BOARD`` verbatim -- it is a subset of
    ``themed-gaming`` today, but it is the one board we know the organisers ran
    us on, and it must not depend on a themed pool staying a superset of it.
    """
    from harness.secret_pool import ORGANISER_THEMED_BOARD, THEMED_POOLS

    words: List[str] = []
    for name in sorted(THEMED_POOLS):
        words.extend(THEMED_POOLS[name])
    words.extend(ORGANISER_THEMED_BOARD)
    return _uppercase_unique(words, seen=set(bundled_pools()))


#: Slang-pool words GloVe 6B does not contain in any casing or hyphenation, so
#: no widening of the vector source can ever cover them.  Verified against the
#: full 400k vocabulary, not the cached prefix.  Kept here so the coverage
#: report can quote an achievable denominator instead of a flattering one, and
#: so a future rebuild that "fixes" one of these is treated as a surprise.
UNREACHABLE_SLANG = (
    "AIRPODS", "BINGEWATCH", "BLOCKCHAIN", "BOSSFIGHT", "CHATBOT",
    "CLICKBAIT", "DEATHSTAR", "DEEPFAKE", "DOGECOIN", "DOOMSCROLL", "EMOJI",
    "ESCAPEROOM", "FORTNITE", "GLAMPING", "GRIEFER", "HEADCANON", "HYPERLOOP",
    "ISEKAI", "LASERTAG", "LOOTBOX", "MOSHPIT", "MUKBANG", "ORIGINSTORY",
    "PLOTTWIST", "RANSOMWARE", "RAPBATTLE", "SELFIE", "SERVERFARM",
    "SIDEHUSTLE", "SITUATIONSHIP", "SLENDERMAN", "SPEEDFORCE", "STAYCATION",
    "SWEATLORD", "SYSADMIN", "TIKTOK", "TSUNDERE", "WAIFU", "YEET",
)

#: Themed-pool words no widening can reach.  There are none, and that is the
#: measurement, not an omission: all 208 words of the five themed pools and the
#: organisers' board exist in GloVe 6B.  ``HYRULE`` was the expected casualty
#: and is not one -- it sits at rank 195170, below the wide cache's 150k prefix
#: but comfortably inside the 400k vocabulary, so naming the themed pools in
#: ``harness.glove_data.pool_words`` pulls it (and ``DRUMKIT``, rank 269862) in
#: by name.  A word appearing here later means a themed pool grew a coinage.
UNREACHABLE_THEMED = ()

#: Every board word that is unreachable whatever we do, for reports that want
#: one denominator rather than two.
UNREACHABLE = tuple(sorted(set(UNREACHABLE_SLANG) | set(UNREACHABLE_THEMED)))


#: Suffix, characters to strip.  Longest first so ``ies`` beats ``es`` beats
#: ``s``.
_INFLECTIONS = (("ies", 3), ("ers", 3), ("ing", 3), ("es", 2), ("ed", 2),
                ("ly", 2), ("er", 2), ("s", 1))


def _is_inflected(word: str, vocab) -> bool:
    """True for a plural / participle / adverb whose stem is also a word.

    Codenames cards are bare singular nouns.  The nearest neighbours of a
    slang word are full of ALBUMS, UPLOADED, WIRELESSLY and MUSTACHES, which
    are the same card twice and waste index space.
    """
    lower = word.lower()
    if lower.endswith("ss"):
        return False
    for suffix, strip in _INFLECTIONS:
        if not lower.endswith(suffix) or len(lower) - strip < 3:
            continue
        stem = lower[:-strip]
        if stem in vocab:
            return True
        if suffix == "ies" and stem + "y" in vocab:
            return True
        # bike -> biker, upload -> uploading, mustache -> mustaches: the stem
        # loses its silent e before the suffix.
        if suffix in ("ed", "ing", "er", "ers", "es") and stem + "e" in vocab:
            return True
    return False


def _is_derivation(word: str, others: Iterable[str]) -> bool:
    """The framework's own illegal-clue relation: either contains the other."""
    for other in others:
        if word in other or other in word:
            return True
    return False


def candidate_seeds() -> List[str]:
    """The register the generated bank is grown from: slang + every themed pool.

    Deliberately *not* the default pool.  Its neighbours are the words the
    table already covers, so seeding on it would spend the budget re-describing
    the half of the vocabulary that was never the problem.
    """
    from harness.secret_pool import (ORGANISER_THEMED_BOARD, SLANG_POOL,
                                     THEMED_POOLS)

    words = list(SLANG_POOL)
    for name in sorted(THEMED_POOLS):
        words.extend(THEMED_POOLS[name])
    words.extend(ORGANISER_THEMED_BOARD)
    return _uppercase_unique(words)


def tournament_candidates(vocab, matrix, index, taken, budget,
                          per_seed: int = CANDIDATE_PER_SEED,
                          min_sim: float = CANDIDATE_MIN_SIM,
                          rank_lo: int = CANDIDATE_RANK_LO,
                          rank_hi: int = CANDIDATE_RANK_HI) -> List[str]:
    """Guess at the words a secret slang/pop-culture pool would deal.

    The selection rule, in full, so the list is reproducible rather than
    hand-picked:

    1. **Seeds** are the words of every bundled non-default pool the embedding
       knows: the slang pool, the five themed pools, and the organisers' own
       board.  They are the only sample of the tournament's register we have,
       so the pool we want is approximated by *what sits next to them*.  The
       themed pools joined the seed set in the same round they joined the board
       vocabulary, and they change what comes out: seeding on the slang pool
       alone produced a bank with almost nothing from the register the
       organisers actually dealt us (their board shares one word with the
       default pool and twelve with the whole 1008-word table).
    2. For each seed take its ``per_seed`` nearest GloVe neighbours at cosine
       ``>= min_sim``, restricted to rows ``[rank_lo, rank_hi)`` of the wide
       cache -- common enough to be printable on a card, rare enough not to be
       a function word.  (The cache's row order is GloVe's frequency order.)
    3. Drop inflected forms whose stem is also a word, and anything that
       contains or is contained in a word already selected -- the same
       relation the competition uses to rule a clue illegal, so no two entries
       are the same card.
    4. Order by how many distinct seeds voted for a word, then by its best
       cosine, and take ``budget``.

    What comes out is franchise and character names (SUPERMAN, DRACULA, FRODO,
    DARTH, SEGA), consumer tech (ITUNES, SMARTPHONE, USB, SKYPE, VHS), internet
    brands (TWITTER, FLICKR), music genres (ELECTRONICA, FUNK, TRANCE) and food
    (LATTE, WASABI, PEPPERONI) -- the register the pool is described in.
    """
    import numpy as np

    seeds = candidate_seeds()
    rows = sorted(set(index[w.lower()] for w in seeds if w.lower() in index))
    if not rows or budget <= 0:
        return []

    sims = matrix[rows].dot(matrix.T)
    sims[:, :max(0, rank_lo)] = -1.0
    sims[:, min(len(vocab), max(0, rank_hi)):] = -1.0
    for position, row in enumerate(rows):
        sims[position, row] = -1.0
    order = np.argsort(-sims, axis=1)[:, :max(1, per_seed)]

    votes: Dict[str, int] = {}
    best: Dict[str, float] = {}
    for position in range(len(rows)):
        for column in order[position]:
            value = float(sims[position, column])
            if value < min_sim:
                continue
            word = vocab[int(column)].upper()
            votes[word] = votes.get(word, 0) + 1
            best[word] = max(best.get(word, 0.0), value)

    known = set(vocab)
    chosen: List[str] = []
    against = list(taken)
    for word in sorted(votes, key=lambda w: (-votes[w], -best[w], w)):
        if len(chosen) >= budget:
            break
        if _is_inflected(word, known) or _is_derivation(word, against):
            continue
        chosen.append(word)
        against.append(word)
    return chosen


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _codebook(values, levels: int = CODE_LEVELS,
              iterations: int = 40) -> List[float]:
    """Lloyd-max codebook over the stored similarities.

    A uniform 6-bit scale over [floor, 1.0] would cost 0.014 per step, most of
    the 0.02 margin the danger rule turns on.  Plain quantiles overcorrect the
    other way: the distribution is so right-skewed that the whole strong tail
    collapses onto one level, and the sensor then cannot tell a 0.50 assassin
    from a 0.60 target.  Lloyd-max minimises squared error over the actual
    distribution, which keeps ~0.002 resolution near the floor *and* usable
    steps in the tail.  Runs on a histogram, so the vector count is irrelevant.
    """
    import numpy as np

    values = np.asarray(values, dtype="float64")
    lo, hi = float(values.min()), float(values.max())
    bins = 8192
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    book = np.quantile(values, np.linspace(0.0, 1.0, levels))
    for _ in range(iterations):
        assign = np.abs(centres[:, None] - book[None, :]).argmin(axis=1)
        weight = np.bincount(assign, weights=counts, minlength=levels)
        total = np.bincount(assign, weights=counts * centres, minlength=levels)
        moved = weight > 0
        book[moved] = total[moved] / weight[moved]
        book = np.sort(book)
    # Strictly increasing, so nearest-level lookup is well defined.
    for i in range(1, len(book)):
        if book[i] <= book[i - 1]:
            book[i] = book[i - 1] + 1e-4
    return [round(float(x), 5) for x in book]


def encode(header: Dict[str, object], board_words: Sequence[str],
           clue_words: Sequence[str], offsets: Sequence[int],
           columns: Sequence[int], codes: Sequence[int],
           version: int = DEFAULT_VERSION) -> bytes:
    """Serialise one table.  Both format versions, one code path.

    Split out of :func:`build` so the version-1 encoder is exercised by the
    tests without a GloVe cache in the room -- the compatibility claim is only
    worth something if something still writes the old layout.
    """
    if version not in MAGIC_BY_VERSION:
        raise ValueError("unknown format version %r" % (version,))
    ceiling = MAX_BOARD_WORDS_BY_VERSION[version]
    if len(board_words) > ceiling:
        raise ValueError("board vocabulary of %d does not fit format %d's "
                         "%d-word index" % (len(board_words), version, ceiling))

    head = dict(header)
    head["format"] = "obirdy-simtable-%d" % version
    head["version"] = version
    head["n_board"] = len(board_words)
    head["n_clue"] = len(clue_words)
    head["n_entries"] = len(columns)

    blob = bytearray()
    blob += MAGIC_BY_VERSION[version]
    blob += json.dumps(head, separators=(",", ":")).encode("utf-8") + b"\n"
    blob += ("\n".join(board_words) + "\n").encode("utf-8")
    blob += ("\n".join(clue_words) + "\n").encode("utf-8")
    for value in offsets:
        blob += int(value).to_bytes(4, "little")
    if version == 1:
        for column, code in zip(columns, codes):
            blob += (((int(column) << CODE_BITS) | int(code))
                     .to_bytes(2, "little"))
    else:
        for column in columns:
            blob += int(column).to_bytes(2, "little")
        blob += bytes(int(code) for code in codes)
    return bytes(blob)


def build(out_path: str = OUT_PATH,
          glove_cache: Optional[str] = None,
          floor: float = DEFAULT_FLOOR,
          top_k: int = DEFAULT_TOP_K,
          clue_vocab: int = DEFAULT_CLUE_VOCAB,
          extra_clues: Optional[Iterable[str]] = None,
          extra_board: bool = True,
          candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
          themed_top_k: int = DEFAULT_THEMED_TOP_K,
          candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
          version: Optional[int] = None) -> Dict[str, object]:
    """Write the table and return its stats.

    ``version`` defaults to the narrowest format the board vocabulary fits in,
    so a small table is still written in the version-1 layout an older agent
    can read.
    """
    import numpy as np

    glove_cache = glove_cache or default_glove()
    with np.load(glove_cache, allow_pickle=False) as data:
        vocab = [str(w) for w in data["words"]]
        matrix = data["vectors"]
    index = dict((word, i) for i, word in enumerate(vocab))

    pools = [w for w in bundled_pools() if w.lower() in index]
    themed = [w for w in themed_pool_words() if w.lower() in index]
    budget = min(int(candidate_budget),
                 MAX_BOARD_WORDS - BOARD_HEADROOM - len(pools) - len(themed))
    candidates = (tournament_candidates(vocab, matrix, index, pools + themed,
                                        budget)
                  if extra_board else [])
    board_words = pools + themed + candidates
    if version is None:
        version = 1 if len(board_words) <= LEGACY_MAX_BOARD_WORDS else 2
    if len(board_words) > MAX_BOARD_WORDS_BY_VERSION[version] - BOARD_HEADROOM:
        raise SystemExit("board vocabulary of %d words does not fit format "
                         "%d" % (len(board_words), version))
    board_matrix = matrix[[index[w.lower()] for w in board_words]]

    ceiling = min(len(vocab), max(int(clue_vocab), CLUE_VOCAB_CEILING))
    wanted: List[str] = []
    seen = set()
    for source in (vocab[:max(0, clue_vocab)],
                   [w.lower() for w in _fallback_vocab()],
                   [w.lower() for w in (extra_clues if extra_clues is not None
                                        else recorded_clues())],
                   [w.lower() for w in board_words]):
        for word in source:
            row = index.get(word)
            if row is None or row >= ceiling or word in seen:
                continue
            seen.add(word)
            wanted.append(word)
    clue_words = sorted(wanted)
    clue_matrix = matrix[[index[w] for w in clue_words]]

    sims = clue_matrix.dot(board_matrix.T)
    kept_mask = sims >= float(floor)

    def _rank(columns, keep):
        """Top ``keep`` of ``columns`` per clue, as (order, values, mask)."""
        if not len(columns) or keep <= 0:
            empty = np.zeros((len(clue_words), 0), dtype="int64")
            return empty, empty.astype(sims.dtype), empty.astype(bool)
        block = sims[:, columns]
        local = np.argsort(-block, axis=1)[:, :int(keep)]
        return (columns[local],
                np.take_along_axis(block, local, axis=1),
                np.take_along_axis(kept_mask[:, columns], local, axis=1))

    # One bank per vocabulary source, each ranked in its own field.  A word
    # only ever competes for depth against words of its own provenance, so
    # adding a bank cannot cost an older bank a stored pair.
    banks = [(np.arange(0, len(pools)), top_k),
             (np.arange(len(pools), len(pools) + len(themed)), themed_top_k),
             (np.arange(len(pools) + len(themed), len(board_words)),
              candidate_top_k)]
    ranked = [_rank(cols, keep) for cols, keep in banks]
    order = np.concatenate([r[0] for r in ranked], axis=1)
    kept_vals = np.concatenate([r[1] for r in ranked], axis=1)
    kept_ok = np.concatenate([r[2] for r in ranked], axis=1)

    book = _codebook(kept_vals[kept_ok])
    book_arr = np.asarray(book, dtype="float32")

    offsets = [0]
    columns: List[int] = []
    entry_codes: List[int] = []
    for row in range(len(clue_words)):
        cols = order[row][kept_ok[row]]
        vals = kept_vals[row][kept_ok[row]]
        if len(cols):
            codes = np.abs(book_arr[None, :] - vals[:, None]).argmin(axis=1)
            # Emit in board-index order rather than similarity order: the
            # reader wants values, not ranks, and an ascending index stream
            # compresses far better than a permuted one.
            for position in np.argsort(cols, kind="stable"):
                columns.append(int(cols[position]))
                entry_codes.append(int(codes[position]))
        offsets.append(len(columns))

    header = {
        "source": os.path.basename(glove_cache),
        "floor": float(floor),
        "top_k": int(top_k),
        "themed_top_k": int(themed_top_k),
        "candidate_top_k": int(candidate_top_k),
        "code_bits": CODE_BITS,
        #: How many of ``n_board`` come from the two real pools, and how many
        #: more from the themed pools; the rest are generated tournament
        #: guesses and are reported separately so a coverage number is never
        #: quietly inflated by them.
        "n_pool": len(pools),
        "n_themed": len(themed),
        "n_candidates": len(candidates),
        "levels": book,
    }
    blob = encode(header, board_words, clue_words, offsets, columns,
                  entry_codes, version=version)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wb", compresslevel=9) as handle:
        handle.write(blob)

    return {
        "path": out_path,
        "bytes": os.path.getsize(out_path),
        "raw_bytes": len(blob),
        "version": version,
        "n_board": len(board_words),
        "n_pool": len(pools),
        "n_themed": len(themed),
        "n_candidates": len(candidates),
        "n_clue": len(clue_words),
        "n_entries": len(columns),
        "mean_entries": len(columns) / float(max(1, len(clue_words))),
        "floor": floor,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def pool_coverage(table) -> Dict[str, Dict[str, object]]:
    """Per-pool board coverage: covered / total / unreachable / what is missing.

    One row per pool the harness can deal, because one aggregate number hid the
    thing that mattered: the shipped 1008-word table read 100% of the default
    pool and 12 of the organisers' 25 words, and the average of those two is a
    statistic about nothing.
    """
    from harness.secret_pool import (ORGANISER_THEMED_BOARD, SLANG_POOL,
                                     THEMED_POOLS, load_default_pool)

    pools = [("default", load_default_pool()), ("slang", SLANG_POOL)]
    pools.extend(("themed-%s" % name, THEMED_POOLS[name])
                 for name in sorted(THEMED_POOLS))
    pools.append(("organiser", ORGANISER_THEMED_BOARD))

    unreachable = set(UNREACHABLE)
    out: Dict[str, Dict[str, object]] = {}
    for name, words in pools:
        upper = _uppercase_unique(words)
        missing = [w for w in upper if w not in table.board_index]
        reachable = len(upper) - sum(1 for w in upper if w in unreachable)
        out[name] = {
            "covered": len(upper) - len(missing),
            "total": len(upper),
            "reachable": reachable,
            "pct": 100.0 * (len(upper) - len(missing)) / max(1, len(upper)),
            "pct_of_reachable": (100.0 * (len(upper) - len(missing))
                                 / max(1, reachable)),
            "missing": missing,
        }
    return out


def coverage(path: str = OUT_PATH) -> Dict[str, object]:
    """How much of the clue traffic and the board pool the table can price."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "framework"))
    from players import codemaster_obirdy as cm

    table = cm._SimTable.load(path)
    if table is None:
        raise SystemExit("no table at %s -- build it first" % path)

    from harness.secret_pool import SLANG_POOL, load_default_pool

    default_pool = [w.upper() for w in load_default_pool()]
    slang_pool = [w.upper() for w in SLANG_POOL]
    themed = themed_pool_words()

    clues = recorded_clues()
    counts: Dict[str, int] = {}
    for path_json in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(path_json, "r") as handle:
                games = json.load(handle)
        except Exception:
            continue
        if not isinstance(games, list):
            continue
        for game in games:
            if not isinstance(game, dict):
                continue
            for move in game.get("move_history") or ():
                if (isinstance(move, (list, tuple)) and len(move) > 1
                        and str(move[0]).endswith("Codemaster")):
                    key = str(move[1]).strip().upper()
                    counts[key] = counts.get(key, 0) + 1

    tokens = sum(counts.values()) or 1
    #: The slang words no embedding can reach -- reported so the slang coverage
    #: number is read against what is achievable, not against 232.
    slang_reachable = len(slang_pool) - len(UNREACHABLE_SLANG)
    return {
        "bytes": os.path.getsize(path),
        "version": int(table.version),
        "n_board": len(table.board_index),
        "n_clue": len(table.clue_words),
        "default_pool_covered": sum(1 for w in default_pool
                                    if w in table.board_index),
        "default_pool": len(default_pool),
        "slang_pool_covered": sum(1 for w in slang_pool
                                  if w in table.board_index),
        "slang_pool": len(slang_pool),
        "slang_pool_reachable": slang_reachable,
        "slang_truly_oov": len(UNREACHABLE_SLANG),
        "themed_pools_covered": sum(1 for w in themed
                                    if w in table.board_index),
        "themed_pools": len(themed),
        "themed_truly_oov": len(UNREACHABLE_THEMED),
        "per_pool": pool_coverage(table),
        "recorded_clue_types_covered": sum(1 for c in clues
                                           if table.has_clue(c)),
        "recorded_clue_types": len(clues),
        "recorded_clue_tokens_pct": 100.0 * sum(
            v for k, v in counts.items() if table.has_clue(k)) / tokens,
    }


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--glove", default=None)
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--clue-vocab", type=int, default=DEFAULT_CLUE_VOCAB)
    parser.add_argument("--candidate-budget", type=int,
                        default=DEFAULT_CANDIDATE_BUDGET)
    parser.add_argument("--version", type=int, default=None,
                        help="force a format version (default: the narrowest "
                             "one the board vocabulary fits)")
    parser.add_argument("--no-extra-board", action="store_true",
                        help="the bundled pools only, no generated candidates")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)

    if args.build or not args.stats:
        stats = build(out_path=args.out, glove_cache=args.glove,
                      floor=args.floor, top_k=args.top_k,
                      clue_vocab=args.clue_vocab,
                      candidate_budget=args.candidate_budget,
                      version=args.version,
                      extra_board=not args.no_extra_board)
        sys.stderr.write(
            "wrote %s (format %d)\n  %.0f KB gzipped (%.0f KB raw), %d clues x "
            "%d board words (%d pool + %d themed + %d generated), %d entries "
            "(%.1f per clue)\n"
            % (stats["path"], stats["version"], stats["bytes"] / 1024.0,
               stats["raw_bytes"] / 1024.0, stats["n_clue"],
               stats["n_board"], stats["n_pool"], stats["n_themed"],
               stats["n_candidates"], stats["n_entries"],
               stats["mean_entries"]))
    if args.stats:
        report = coverage(args.out)
        per_pool = report.pop("per_pool")
        for key, value in sorted(report.items()):
            sys.stderr.write("  %-28s %s\n" % (key, value))
        sys.stderr.write("\n  %-16s %9s %8s %s\n"
                         % ("pool", "covered", "of", "missing"))
        for name in sorted(per_pool):
            row = per_pool[name]
            sys.stderr.write(
                "  %-16s %9d %8d  %5.1f%%  %s\n"
                % (name, row["covered"], row["total"], row["pct"],
                   " ".join(row["missing"]) or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
