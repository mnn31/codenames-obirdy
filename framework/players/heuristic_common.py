"""Shared helpers for the offline (no-API) heuristic agents.

These agents exist purely so the evaluation harness can run complete, legal
games without touching any network service.  They are NOT competition entries
and make no attempt to play well -- they only have to finish games legally and
fast.

Importable as ``players.heuristic_common`` whenever ``framework/`` is on
``sys.path`` (which is how ``run_game.py`` and ``game.py`` are already run).

Python 3.9 compatible.
"""

import random
import re

# --------------------------------------------------------------------------
# A very small bundled association table: clue -> board words it hints at.
# Built from the shipped ``players/cm_wordlist.txt`` vocabulary.  When a board
# uses an unknown word pool (see harness/secret_pool.py) none of these fire and
# the agents fall back purely to string similarity, which is fine -- the goal
# is legality and termination, not quality.
# --------------------------------------------------------------------------
ASSOCIATIONS = {
    "CREATURE": ["BEAR", "LION", "HORSE", "MOUSE", "WHALE", "DUCK", "EAGLE",
                 "SHARK", "RABBIT", "PENGUIN", "OCTOPUS", "KANGAROO", "DOG",
                 "CAT", "BUFFALO", "SCORPION", "PLATYPUS", "DRAGON", "HAWK",
                 "ROBIN", "SLUG", "WORM", "SEAL", "CRANE", "BAT"],
    "MYTH": ["ANGEL", "GHOST", "GIANT", "DWARF", "UNICORN", "CENTAUR",
             "PHOENIX", "DRAGON", "WITCH", "LEPRECHAUN", "ATLANTIS", "GENIUS"],
    "NATION": ["AFRICA", "AMERICA", "AUSTRALIA", "CANADA", "CHINA", "EGYPT",
               "ENGLAND", "EUROPE", "FRANCE", "GERMANY", "GREECE", "INDIA",
               "MEXICO", "TURKEY", "ANTARCTICA", "CZECH"],
    "METROPOLIS": ["BEIJING", "BERLIN", "LONDON", "MOSCOW", "ROME", "TOKYO",
                   "WASHINGTON", "HOLLYWOOD", "CAPITAL"],
    "PLANET": ["MERCURY", "JUPITER", "SATURN", "MOON", "STAR", "SPACE",
               "SATELLITE", "TELESCOPE", "ALIEN"],
    "VEHICLE": ["CAR", "TRAIN", "PLANE", "JET", "SHIP", "VAN", "LIMOUSINE",
                "AMBULANCE", "HELICOPTER", "ENGINE", "TRACK", "SUB"],
    "WEAPON": ["BOMB", "MISSILE", "PISTOL", "KNIFE", "SPIKE", "WHIP", "BOW",
               "TORCH", "FIGHTER", "SOLDIER", "WAR", "STRIKE"],
    "MUSIC": ["BAND", "PIANO", "FLUTE", "BUGLE", "HORN", "OPERA", "CONCERT",
              "NOTE", "CONDUCTOR", "STRING", "ORGAN", "SOUND"],
    "SPORT": ["BALL", "COURT", "FIELD", "RACKET", "CRICKET", "PITCH", "BAT",
              "STADIUM", "GAME", "PLAY", "SWING", "MATCH"],
    "MEAL": ["APPLE", "CARROT", "CHOCOLATE", "HONEY", "KETCHUP", "LEMON",
             "OLIVE", "ORANGE", "PIE", "PUMPKIN", "BERRY", "NUT", "HAM",
             "JAM", "MINT", "KIWI", "COOK"],
    "CLOTHES": ["BELT", "BOOT", "CAP", "CLOAK", "DRESS", "GLOVE", "HOOD",
                "PANTS", "SHOE", "SOCK", "SUIT", "TIE", "COTTON"],
    "TOOL": ["DRILL", "FORK", "HOOK", "NAIL", "NEEDLE", "PIN", "BRUSH",
             "SCALE", "PIPE", "LOCK", "KEY", "SWITCH", "PLATE"],
    "JOB": ["AGENT", "DOCTOR", "LAWYER", "NURSE", "PILOT", "TEACHER", "COOK",
            "SPY", "SCIENTIST", "PIRATE", "NINJA", "THIEF", "UNDERTAKER",
            "SMUGGLER", "POLICE", "VET"],
    "ROYAL": ["KING", "QUEEN", "PRINCESS", "KNIGHT", "CROWN", "COURT",
              "RULER", "CASTLE", "TEMPLE", "PALM"],
    "BUILDING": ["CHURCH", "HOSPITAL", "HOTEL", "SCHOOL", "THEATER", "TOWER",
                 "SKYSCRAPER", "EMBASSY", "BANK", "SHOP", "LAB", "PYRAMID",
                 "BRIDGE", "WALL", "MINE"],
    "METAL": ["COPPER", "GOLD", "IRON", "LEAD", "MARBLE", "DIAMOND", "IVORY",
              "ROCK", "MINE", "BOLT", "CHAIN"],
    "WEATHER": ["ICE", "SNOW", "WIND", "COLD", "WAVE", "STREAM", "AIR",
                "FIRE", "WATER", "SNOWMAN", "CLIFF", "BEACH"],
    "PLANT": ["GRASS", "MAPLE", "ROSE", "ROOT", "FOREST", "TRUNK", "STRAW",
              "LOG", "BARK", "SPINE", "PALM"],
    "BODY": ["ARM", "EYE", "FACE", "FOOT", "HAND", "HEAD", "HEART", "MOUTH",
             "THUMB", "TOOTH", "SPINE", "CHEST", "LAP", "TAIL"],
    "PAPERWORK": ["BILL", "CARD", "CHECK", "CONTRACT", "FILE", "MAIL", "NOTE",
                  "PAPER", "POST", "PRESS", "DRAFT", "CODE", "NOVEL", "COMIC"],
    "GAMBLE": ["CASINO", "DICE", "ROULETTE", "LUCK", "CHANCE", "CLUB", "DECK",
               "JACK", "POOL", "BET", "STOCK"],
    "MACHINE": ["ROBOT", "SERVER", "SCREEN", "TABLET", "BATTERY", "LASER",
                "MICROSCOPE", "VACUUM", "WASHER", "ENGINE", "MODEL"],
    "MOTION": ["DANCE", "FALL", "FLY", "ROW", "SWING", "TRIP", "CYCLE",
               "DROP", "SLIP", "WAKE", "PASS", "CAST"],
    "SHAPE": ["CIRCLE", "SQUARE", "TRIANGLE", "ROUND", "LINE", "POINT",
              "FIGURE", "BOX", "RING", "CROSS", "BLOCK"],
    "DARK": ["SHADOW", "NIGHT", "DEATH", "POISON", "DISEASE", "GHOST",
             "SOUL", "PIT", "HOLE", "GRAVE"],
    "LIGHT": ["DAY", "SUN", "TORCH", "FIRE", "STAR", "GLASS", "MOON",
              "SPOT", "RAY", "GLOW"],
    "MONEY": ["BANK", "BILL", "GOLD", "POUND", "MILLIONAIRE", "STOCK",
              "CHANGE", "CHARGE", "MINT", "BUCK"],
    "OCEAN": ["BEACH", "SHIP", "WHALE", "SHARK", "SEAL", "PORT", "SCUBA",
              "WAVE", "FISH", "BERMUDA", "ATLANTIS", "OCTOPUS"],
    "SCHOOLROOM": ["PUPIL", "TEACHER", "BOARD", "DESK", "CHAIR", "TABLE",
                   "DEGREE", "SCHOOL", "STAFF", "CLASS"],
    "SPACEAGE": ["ALIEN", "ROBOT", "LASER", "SATELLITE", "SPACE", "MOON",
                 "STAR", "SUPERHERO", "TELESCOPE"],
}

# Reverse index: board word -> set of clue words that mention it.
WORD_TO_CLUES = {}
for _clue, _words in ASSOCIATIONS.items():
    for _w in _words:
        WORD_TO_CLUES.setdefault(_w, set()).add(_clue)

# A general purpose clue vocabulary.  Deliberately generic single English
# words so the sub-word legality filter rarely rejects them.
CLUE_VOCAB = sorted(set(list(ASSOCIATIONS.keys()) + [
    "ANIMAL", "BUILDER", "CAPTAIN", "CARGO", "CHASE", "CLIMATE", "COLOUR",
    "COMFORT", "CONFLICT", "COUNTRY", "CULTURE", "CURRENT", "DANGER",
    "DESERT", "DEVICE", "DINNER", "DISTANCE", "DRIVER", "EMPIRE", "ENERGY",
    "ESCAPE", "FABRIC", "FACTORY", "FARMER", "FESTIVAL", "FLAVOUR", "FOSSIL",
    "FRIEND", "FUTURE", "GADGET", "GARDEN", "GLACIER", "HARBOUR", "HARVEST",
    "HAZARD", "HELMET", "HERO", "HISTORY", "HOLIDAY", "HUNTER", "ISLAND",
    "JOURNEY", "JUNGLE", "KITCHEN", "LADDER", "LANTERN", "LEGEND", "LEISURE",
    "LIBRARY", "LIQUID", "MACHINE", "MAGNET", "MARKET", "MEADOW", "MEMORY",
    "MERCHANT", "MESSAGE", "MINERAL", "MIRROR", "MONSTER", "MORNING",
    "MOTOR", "MOUNTAIN", "MUSEUM", "MYSTERY", "NATURE", "NUMBER", "OFFICE",
    "ORBIT", "OUTDOORS", "PACKAGE", "PARADE", "PATTERN", "PICNIC", "PILLOW",
    "PIONEER", "PLANET", "POCKET", "POWDER", "PRISON", "PUZZLE", "QUARRY",
    "RAINBOW", "RECIPE", "RESCUE", "RHYTHM", "RIDDLE", "RIVER", "RUBBLE",
    "SAFARI", "SEASON", "SECRET", "SHELTER", "SIGNAL", "SILENCE", "SKETCH",
    "SOLDIER", "SOUVENIR", "SPEECH", "STORAGE", "STORY", "STRANGER",
    "SUMMIT", "SUPPER", "SURFACE", "SYMBOL", "TALENT", "TAVERN", "TEMPEST",
    "THUNDER", "TICKET", "TRADITION", "TRAFFIC", "TREASURE", "TROPHY",
    "TUNNEL", "VALLEY", "VICTORY", "VILLAGE", "VOYAGE", "WEALTH", "WEAPON",
    "WHISPER", "WILDLIFE", "WINTER", "WIZARD", "WONDER", "WORKSHOP",
]))

_ALPHA_RE = re.compile(r"^[A-Z]+$")


def normalise(word):
    """Uppercase, strip, keep letters only (board words may contain spaces)."""
    return re.sub(r"[^A-Z]", "", str(word).upper().strip())


def is_revealed(board_word):
    return len(board_word) > 0 and board_word[0] == "*"


def unrevealed(words):
    return [w for w in words if not is_revealed(w)]


def clue_is_legal(clue, board_words):
    """Mirror the framework's legality rule used by codemaster_GPT.

    A clue must be a single alphabetic English word and must neither contain
    nor be contained by any *unrevealed* word on the board.
    """
    if not clue or not _ALPHA_RE.match(clue):
        return False
    if len(clue) < 3:
        return False
    for w in board_words:
        if is_revealed(w):
            continue
        norm = normalise(w)
        if not norm:
            continue
        if clue in norm or norm in clue:
            return False
    return True


def _letters(word):
    return set(ch for ch in word if ch.isalpha())


def _shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def similarity(clue, board_word):
    """Cheap, deterministic word-association proxy in roughly [0, 2]."""
    word = normalise(board_word)
    if not word or not clue:
        return 0.0

    score = 0.0
    if word in ASSOCIATIONS.get(clue, ()):  # bundled association table
        score += 1.0

    a, b = _letters(clue), _letters(word)
    union = a | b
    if union:
        score += 0.6 * (len(a & b) / float(len(union)))

    score += 0.15 * min(_shared_prefix(clue, word), 3)

    # Small nudge so ties break deterministically rather than by list order.
    score += 0.001 * (len(word) % 7)
    return score


def score_clue(clue, words, maps, team, blue_penalty=0.7,
               civilian_penalty=0.3, assassin_penalty=4.0):
    """Expected-value-ish score for a clue given the codemaster's key grid."""
    own = team
    opponent = "Blue" if team == "Red" else "Red"

    total = 0.0
    hits = 0
    for i, w in enumerate(words):
        if is_revealed(w):
            continue
        sim = similarity(clue, w)
        kind = maps[i]
        if kind == own:
            total += sim
            if sim >= 0.45:
                hits += 1
        elif kind == opponent:
            total -= blue_penalty * sim
        elif kind == "Assassin":
            total -= assassin_penalty * sim
        else:
            total -= civilian_penalty * sim
    return total, max(1, hits)


def pick_clue(words, maps, team, rng=None, vocab=None):
    """Return ``[clue, number]`` -- always legal, never raises."""
    vocab = vocab or CLUE_VOCAB
    rng = rng or random

    best = None
    best_score = None
    best_num = 1
    for clue in vocab:
        if not clue_is_legal(clue, words):
            continue
        score, hits = score_clue(clue, words, maps, team)
        if best_score is None or score > best_score:
            best, best_score, best_num = clue, score, hits

    if best is None:
        # Extremely unlikely, but never emit an illegal/empty clue.
        for candidate in ("SIGNAL", "OBJECT", "SUBJECT", "TOPIC", "IDEA"):
            if clue_is_legal(candidate, words):
                return [candidate, 1]
        return ["THING", 1]

    own_left = sum(1 for i, w in enumerate(words)
                   if not is_revealed(w) and maps[i] == team)
    number = max(1, min(best_num, max(1, own_left)))
    return [best, number]


def pick_guess(clue, words, rng=None):
    """Return the unrevealed board word most associated with ``clue``."""
    rng = rng or random
    options = unrevealed(words)
    if not options:
        return None
    clue_n = normalise(clue)
    if not clue_n:
        return rng.choice(options)
    scored = [(similarity(clue_n, w), w) for w in options]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored[0][1]
