"""Alternative 25-word boards from custom word pools.

The competition's real word pool is secret and may contain slang and
pop-culture terms ("Hogwarts", "Xenomorph").  Static embedding lookups fail on
those, so we need to measure generalisation before the final submission.

How the framework picks board words
-----------------------------------
``game.Game.__init__`` does, verbatim::

    with open("players/cm_wordlist.txt", "r") as f:
        temp = f.read().splitlines()
        assert len(temp) == len(set(temp)), "game wordpool should not have duplicates"
        random.shuffle(temp)
        self.words_on_board = temp[:25]

That path is **relative to the process working directory**, and there is no
kwarg to override it.  So the clean, framework-untouched way to inject a pool
is to build a throwaway sandbox directory containing ``players/cm_wordlist.txt``
and run the game with that directory as the cwd, while ``framework/`` stays on
``sys.path`` for imports.  :func:`make_sandbox` does exactly that; the arena
uses it for every run (including the default pool) so behaviour is uniform and
the vendored framework directory is never written to.

Python 3.9 compatible.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK_DIR = os.path.join(REPO_ROOT, "framework")
DEFAULT_WORDLIST = os.path.join(FRAMEWORK_DIR, "players", "cm_wordlist.txt")

# Board composition, mirroring game.Game.
NUM_RED, NUM_BLUE, NUM_CIVILIAN, NUM_ASSASSIN = 9, 8, 7, 1
BOARD_SIZE = NUM_RED + NUM_BLUE + NUM_CIVILIAN + NUM_ASSASSIN  # 25


# ---------------------------------------------------------------------------
# Bundled generalisation pool: slang / pop-culture / internet vocabulary.
# ---------------------------------------------------------------------------
SLANG_POOL: List[str] = [
    "HOGWARTS", "XENOMORPH", "TIKTOK", "POKEMON", "MINECRAFT", "FORTNITE",
    "NETFLIX", "SPOTIFY", "YOUTUBE", "TWITCH", "DISCORD", "REDDIT",
    "INSTAGRAM", "SNAPCHAT", "PODCAST", "MEME", "EMOJI", "HASHTAG",
    "SELFIE", "VLOG", "STREAMER", "INFLUENCER", "CLICKBAIT", "PAYWALL",
    "DEEPFAKE", "CHATBOT", "ALGORITHM", "BLOCKCHAIN", "CRYPTO", "BITCOIN",
    "DOGECOIN", "METAVERSE", "AVATAR", "HEADSET", "DRONE", "SEGWAY",
    "ROOMBA", "TESLA", "SPACEX", "STARLINK", "IPHONE", "ANDROID",
    "AIRPODS", "BLUETOOTH", "WIFI", "FIREWALL", "MALWARE", "PHISHING",
    "RANSOMWARE", "HACKER", "SYSADMIN", "SERVERFARM", "MAINFRAME", "PIXEL",
    "GLITCH", "SPEEDRUN", "RESPAWN", "LOOTBOX", "BOSSFIGHT", "NOOB",
    "SWEATLORD", "GRIEFER", "MODDER", "ESPORTS", "JOYSTICK", "ARCADE",
    "PINBALL", "TETRIS", "PACMAN", "MARIO", "ZELDA", "SONIC",
    "KIRBY", "PIKACHU", "GODZILLA", "MOTHRA", "KAIJU", "MECHA",
    "GUNDAM", "ANIME", "MANGA", "COSPLAY", "OTAKU", "WAIFU",
    "SENPAI", "SHONEN", "ISEKAI", "TSUNDERE", "KAWAII", "KARAOKE",
    "MUKBANG", "BOBA", "RAMEN", "SUSHI", "BURRITO", "NACHOS",
    "GUACAMOLE", "AVOCADO", "KOMBUCHA", "MATCHA", "ESPRESSO", "MOCHA",
    "BRUNCH", "FOODIE", "VEGAN", "GLUTEN", "KETO", "SMOOTHIE",
    "SKINCARE", "SNEAKERS", "HOODIE", "BEANIE", "TATTOO", "PIERCING",
    "MULLET", "SIDEBURNS", "HIPSTER", "GOTH", "PUNK", "GRUNGE",
    "MOSHPIT", "FESTIVAL", "COACHELLA", "MIXTAPE", "REMIX", "AUTOTUNE",
    "DUBSTEP", "TECHNO", "REGGAE", "HIPHOP", "RAPBATTLE", "BEATBOX",
    "TURNTABLE", "VINYL", "BOOMBOX", "WALKMAN", "CASSETTE", "SYNTH",
    "KARATE", "PARKOUR", "SKATEBOARD", "SNOWBOARD", "SURFBOARD", "WAKEBOARD",
    "BUNGEE", "ZIPLINE", "PAINTBALL", "LASERTAG", "ESCAPEROOM", "GLAMPING",
    "ROADTRIP", "STAYCATION", "BACKPACKER", "HOSTEL", "AIRBNB", "UBER",
    "SCOOTER", "MONORAIL", "HYPERLOOP", "CYBERPUNK", "STEAMPUNK", "DYSTOPIA",
    "APOCALYPSE", "ZOMBIE", "VAMPIRE", "WEREWOLF", "MUMMY", "POLTERGEIST",
    "CTHULHU", "KRAKEN", "BIGFOOT", "CHUPACABRA", "MOTHMAN", "SLENDERMAN",
    "TARDIS", "DALEK", "JEDI", "SITH", "WOOKIEE", "EWOK",
    "LIGHTSABER", "DEATHSTAR", "MILLENNIUM", "HOBBIT", "ORC", "BALROG",
    "MORDOR", "NARNIA", "MATRIX", "NEO", "TERMINATOR", "ROBOCOP",
    "PREDATOR", "GREMLIN", "MINION", "SHREK", "PIXAR", "MARVEL",
    "AVENGERS", "THANOS", "GOTHAM", "KRYPTON", "WAKANDA", "MULTIVERSE",
    "SPEEDFORCE", "KRYPTONITE", "BATCAVE", "SIDEKICK", "SUPERVILLAIN",
    "ORIGINSTORY", "SPINOFF", "REBOOT", "CAMEO", "BINGEWATCH", "SPOILER",
    "PLOTTWIST", "CLIFFHANGER", "FANDOM", "SHIPPING", "HEADCANON", "STAN",
    "GOAT", "FLEX", "YEET", "SIMP", "CRINGE", "VIBE",
    "GHOSTING", "SIDEHUSTLE", "BURNOUT", "DOOMSCROLL", "SITUATIONSHIP",
]


# ---------------------------------------------------------------------------
# Pool utilities
# ---------------------------------------------------------------------------

def load_default_pool(path: Optional[str] = None) -> List[str]:
    """Read the framework's shipped word pool (read-only)."""
    with open(path or DEFAULT_WORDLIST, "r") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def normalise_pool(words: Sequence[str]) -> List[str]:
    """Uppercase, strip, de-duplicate (order preserved) and validate."""
    seen = set()
    out = []
    for raw in words:
        word = str(raw).strip().upper()
        if not word or word in seen:
            continue
        seen.add(word)
        out.append(word)
    if len(out) < BOARD_SIZE:
        raise ValueError(
            "word pool needs at least %d unique words, got %d" % (BOARD_SIZE, len(out))
        )
    return out


# ---------------------------------------------------------------------------
# Themed pools: every word on the board from one domain.
# ---------------------------------------------------------------------------
#
# The organisers ran our codemaster on a board where all 25 words were gaming /
# internet culture, and it gave 18 consecutive number-1 clues and needed 19
# turns to win solo while the baseline GPT team cleared it with INTERNET 4 /
# COMPUTER 3 / SPACE 2.  That is the pathology these pools exist to reproduce
# offline: when every word is near every other word, a clue strong enough to
# cover two of ours is equally strong on somebody else's, the panel's rankings
# interleave the colours, and the "longest all-own prefix a majority of samples
# agree on" rule can only ever return 1.
#
# A pool needs enough words that seeds differ; the recorded board itself is kept
# verbatim as ``ORGANISER_THEMED_BOARD`` so the exact failure is reproducible.
THEMED_POOLS: Dict[str, List[str]] = {
    "gaming": [
        "QUIDDITCH", "HYRULE", "KRYPTONITE", "WOOKIEE", "MEME", "BYTE",
        "CYBER", "PIXEL", "VIRAL", "ZOOM", "STREAM", "BOT", "SPAWN", "LOOT",
        "NINJA", "FLEX", "DROP", "JAM", "MAGIC", "STAR", "VULCAN", "GALAXY",
        "CRASH", "HERO", "QUEST", "GLITCH", "LEVEL", "BOSS", "COMBO", "RESPAWN",
        "AVATAR", "GUILD", "RAID", "PATCH", "LAG", "PING", "MOD", "SKIN",
        "EMOTE", "CONSOLE", "ARCADE", "JOYSTICK", "SPEEDRUN", "GRIND", "BUFF",
        "NERF", "META", "LOBBY", "SERVER", "STREAK",
    ],
    "space": [
        "COMET", "ORBIT", "NEBULA", "QUASAR", "PULSAR", "ECLIPSE", "CRATER",
        "ROCKET", "SHUTTLE", "LAUNCH", "GRAVITY", "SATELLITE", "ASTEROID",
        "METEOR", "TELESCOPE", "COSMOS", "SOLAR", "LUNAR", "MARS", "VENUS",
        "SATURN", "PLUTO", "GALAXY", "STAR", "MOON", "PLANET", "ROVER",
        "CAPSULE", "PROBE", "BOOSTER", "THRUSTER", "AIRLOCK", "SPACEWALK",
        "VOID", "HORIZON", "SUPERNOVA", "ASTRONAUT", "MISSION", "COUNTDOWN",
        "PAYLOAD",
    ],
    "kitchen": [
        "WHISK", "LADLE", "SKILLET", "SAUCEPAN", "KETTLE", "TOASTER", "OVEN",
        "BURNER", "SIMMER", "ROAST", "BROIL", "KNEAD", "BATTER", "DOUGH",
        "YEAST", "FLOUR", "SUGAR", "BUTTER", "PANTRY", "RECIPE", "GARNISH",
        "MARINADE", "SEASON", "SPICE", "GRATER", "COLANDER", "SPATULA",
        "CUTTING", "APRON", "PLATTER", "SAUCE", "STOCK", "BROTH", "GLAZE",
        "CARAMEL", "MERINGUE", "PASTRY", "SOUFFLE", "OMELETTE", "SKEWER",
    ],
    "music": [
        "TEMPO", "RHYTHM", "MELODY", "HARMONY", "CHORD", "SCALE", "OCTAVE",
        "TREBLE", "BASS", "TENOR", "SOPRANO", "CONCERTO", "SONATA", "SYMPHONY",
        "OVERTURE", "ENCORE", "CHORUS", "VERSE", "BRIDGE", "REFRAIN", "STANZA",
        "FRETBOARD", "STRING", "REED", "BRASS", "PERCUSSION", "DRUMKIT",
        "CYMBAL", "TAMBOURINE", "METRONOME", "CONDUCTOR", "ORCHESTRA",
        "ENSEMBLE", "SOLOIST", "AUDITION", "RECITAL", "STUDIO", "MIXTAPE",
        "REMIX", "SAMPLE",
    ],
    "sport": [
        "DRIBBLE", "TACKLE", "SPRINT", "HURDLE", "RELAY", "MARATHON", "PENALTY",
        "OFFSIDE", "REFEREE", "WHISTLE", "DUGOUT", "BULLPEN", "INNING",
        "QUARTER", "OVERTIME", "TIMEOUT", "PLAYOFF", "TROPHY", "MEDAL",
        "PODIUM", "COACH", "ROSTER", "DRAFT", "ROOKIE", "VETERAN", "STADIUM",
        "BLEACHER", "SCOREBOARD", "TOURNAMENT", "SEEDING", "RALLY", "SERVE",
        "VOLLEY", "SMASH", "PITCH", "SLUGGER", "GOALIE", "STRIKER", "MIDFIELD",
        "SCRIMMAGE",
    ],
}

#: The organisers' board, verbatim, so the recorded 19-turn game is a fixture
#: rather than an anecdote.  ``JAM`` is the only word it shares with the default
#: pool, which is exactly why the sensor's default-pool tuning does not transfer.
ORGANISER_THEMED_BOARD: List[str] = [
    "QUIDDITCH", "HYRULE", "KRYPTONITE", "WOOKIEE", "MEME",
    "BYTE", "CYBER", "PIXEL", "VIRAL", "ZOOM",
    "STREAM", "BOT", "SPAWN", "LOOT", "NINJA",
    "FLEX", "DROP", "JAM", "MAGIC", "STAR",
    "VULCAN", "GALAXY", "CRASH", "HERO", "QUEST",
]


def themed_pool(theme: str = "gaming") -> List[str]:
    """One single-domain pool by name.  Unknown names are an error, not a
    silent fall-through to the default pool -- a themed run that quietly used
    the default pool would report the opposite of what it measured."""
    key = str(theme).strip().lower()
    if key not in THEMED_POOLS:
        raise ValueError("unknown theme %r (known: %s)"
                         % (theme, ", ".join(sorted(THEMED_POOLS))))
    return list(THEMED_POOLS[key])


POOLS: Dict[str, object] = {
    "default": load_default_pool,
    "slang": lambda: list(SLANG_POOL),
    "organiser": lambda: list(ORGANISER_THEMED_BOARD),
}
POOLS.update(dict(("themed-%s" % name, (lambda n: lambda: themed_pool(n))(name))
                  for name in THEMED_POOLS))


def resolve_pool(pool) -> List[str]:
    """Accept a pool name, an explicit word list, or a path to a wordlist file."""
    if pool is None:
        return normalise_pool(load_default_pool())
    if isinstance(pool, str):
        key = pool.strip().lower()
        if key in POOLS:
            return normalise_pool(POOLS[key]())
        if os.path.exists(pool):
            return normalise_pool(load_default_pool(pool))
        raise ValueError(
            "unknown pool %r (known: %s, or pass a path / explicit word list)"
            % (pool, sorted(POOLS))
        )
    return normalise_pool(list(pool))


def mixed_pool(fraction_slang: float = 0.5, seed: int = 0) -> List[str]:
    """Blend the default pool with the slang pool (partial-novelty testing)."""
    rng = random.Random(seed)
    base = normalise_pool(load_default_pool())
    slang = normalise_pool(SLANG_POOL)
    n_slang = int(round(len(base) * max(0.0, min(1.0, fraction_slang))))
    rng.shuffle(base)
    rng.shuffle(slang)
    return normalise_pool(base[: max(0, len(base) - n_slang)] + slang[:n_slang])


# ---------------------------------------------------------------------------
# Board generation (mirrors game.Game exactly, for offline inspection)
# ---------------------------------------------------------------------------

def generate_board(pool=None, seed: int = 0) -> Dict[str, List[str]]:
    """Reproduce the board + key grid ``game.Game`` would build for ``seed``.

    Useful for previewing / analysing boards without running a game.  The
    ordering of ``random`` calls matches ``Game.__init__``: seed, shuffle the
    pool, take 25, then shuffle the key grid.
    """
    words = resolve_pool(pool)
    rng = random.Random()
    rng.seed(int(seed))

    temp = list(words)
    rng.shuffle(temp)
    board = temp[:BOARD_SIZE]

    key_grid = (["Red"] * NUM_RED + ["Blue"] * NUM_BLUE
                + ["Civilian"] * NUM_CIVILIAN + ["Assassin"] * NUM_ASSASSIN)
    rng.shuffle(key_grid)
    return {"words": board, "key_grid": key_grid}


# ---------------------------------------------------------------------------
# Sandbox: inject a pool without touching framework/
# ---------------------------------------------------------------------------

def write_wordlist(words: Sequence[str], target_dir: str) -> str:
    """Write ``target_dir/players/cm_wordlist.txt`` and return its path."""
    players_dir = os.path.join(target_dir, "players")
    os.makedirs(players_dir, exist_ok=True)
    path = os.path.join(players_dir, "cm_wordlist.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(normalise_pool(words)) + "\n")
    return path


def make_sandbox(pool=None, root: Optional[str] = None, prefix: str = "cn-pool-") -> str:
    """Create a temp cwd whose ``players/cm_wordlist.txt`` holds ``pool``.

    Run ``game.Game`` with this directory as the process cwd (and ``framework/``
    on ``sys.path``) to play on the custom pool.  Nothing under ``framework/``
    is modified.  The caller owns cleanup -- see :func:`destroy_sandbox`.
    """
    directory = tempfile.mkdtemp(prefix=prefix, dir=root)
    write_wordlist(resolve_pool(pool), directory)
    return directory


def destroy_sandbox(directory: str) -> None:
    if directory and os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI: preview boards
# ---------------------------------------------------------------------------

def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Preview alternative Codenames boards.")
    parser.add_argument("--pool", default="default",
                        help="pool name (%s), or a path to a wordlist file"
                             % ", ".join(sorted(POOLS)))
    parser.add_argument("--seeds", default="0", help="comma list or A-B range")
    args = parser.parse_args(argv)

    seeds = []
    for chunk in str(args.seeds).split(","):
        chunk = chunk.strip()
        if "-" in chunk and not chunk.startswith("-"):
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            seeds.append(int(chunk))

    words = resolve_pool(args.pool)
    print("pool=%s  size=%d" % (args.pool, len(words)))
    for seed in seeds:
        board = generate_board(words, seed)
        print("\nseed %d" % seed)
        for row in range(5):
            cells = []
            for col in range(5):
                idx = row * 5 + col
                cells.append("%-14s(%s)" % (board["words"][idx],
                                            board["key_grid"][idx][0]))
            print("  " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
