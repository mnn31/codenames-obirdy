"""Cut the submission zip, with the tournament key substituted in.

The organisers refuse per-team environment variables, so the key has to travel
inside the agent files.  It must not travel into git with them: what is
committed is ``HARDCODED_API_KEY = "OBIRDY-KEY-PLACEHOLDER"``, and this script
is the only thing that ever replaces it.  The output filename must contain
``keyed`` (``.gitignore`` has ``submission/*keyed*``), which is what stops a
zip with a live key inside it from being committed by a stray ``git add -A``.

Usage::

    python -m harness.package_submission                  # reads ./.env
    python -m harness.package_submission --env-file X.env
    python -m harness.package_submission --placeholder    # no key (testing zip)

The key is read from ``ANTHROPIC_API_KEY_TOURNAMENT`` in the env file, or from
the same variable in the environment.  It is never printed, not even a prefix.

Zip layout, which mirrors what the organisers unpack into their ``players/``::

    codemaster_obirdy.py
    guesser_obirdy.py
    oBirdy/obirdy_simtable_v2.bin.gz
    INSTRUCTIONS.md

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from typing import Dict, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYERS_DIR = os.path.join(REPO_ROOT, "framework", "players")
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submission")

#: The value that lives in git.  Must match ``HARDCODED_API_KEY`` in both agent
#: files; ``harness/test_obirdy.py`` asserts that it does.
PLACEHOLDER = "OBIRDY-KEY-PLACEHOLDER"
#: The one assignment the substitution is allowed to touch, anchored to the
#: start of a line so a mention inside a docstring can never be rewritten.
KEY_ASSIGNMENT = re.compile(
    r'^(HARDCODED_API_KEY\s*=\s*)"[^"\n]*"$', re.MULTILINE)

KEY_VAR = "ANTHROPIC_API_KEY_TOURNAMENT"

AGENT_FILES = ("codemaster_obirdy.py", "guesser_obirdy.py")
SIMTABLE_ARCNAME = os.path.join("oBirdy", "obirdy_simtable_v2.bin.gz").replace(
    os.sep, "/")
INSTRUCTIONS = "INSTRUCTIONS.md"

DEFAULT_OUT = os.path.join(SUBMISSION_DIR, "obirdy_submission_keyed.zip")
PLACEHOLDER_OUT = os.path.join(SUBMISSION_DIR, "obirdy_submission_testing.zip")


# ---------------------------------------------------------------------------
# Pieces, each independently testable
# ---------------------------------------------------------------------------

def read_env_file(path: str) -> Dict[str, str]:
    """``KEY=value`` pairs from a dotenv-style file.  Missing file -> ``{}``."""
    values: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return values
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            values[name.strip()] = value
    return values


def resolve_key(env_file: Optional[str] = None) -> str:
    """The tournament key: env file first, then the ambient environment."""
    if env_file is None:
        env_file = os.path.join(REPO_ROOT, ".env")
    key = read_env_file(env_file).get(KEY_VAR) or os.environ.get(KEY_VAR) or ""
    key = key.strip()
    if not key:
        raise SystemExit(
            "no %s in %s or the environment -- nothing to substitute"
            % (KEY_VAR, env_file))
    if not key.startswith("sk-"):
        raise SystemExit(
            "%s does not look like an Anthropic key (expected it to start "
            "with 'sk-')" % KEY_VAR)
    return key


def substitute_key(source: str, key: str) -> Tuple[str, int]:
    """Rewrite the ``HARDCODED_API_KEY`` assignment.  ``(text, n_sites)``.

    The count is returned rather than asserted so the caller can insist on
    exactly one: zero means the constant was renamed and the zip would ship
    keyless, more than one means the regex is matching something it should not.
    """
    if '"' in key or "\n" in key:
        raise SystemExit("refusing to embed a key containing a quote or newline")
    text, count = KEY_ASSIGNMENT.subn(lambda m: '%s"%s"' % (m.group(1), key),
                                      source)
    return text, count


def agent_source(name: str, key: Optional[str]) -> str:
    """One agent file, ready to ship."""
    with open(os.path.join(PLAYERS_DIR, name), "r") as handle:
        source = handle.read()
    if key is None:
        return source
    text, count = substitute_key(source, key)
    if count != 1:
        raise SystemExit(
            "%s: expected exactly 1 HARDCODED_API_KEY assignment, found %d"
            % (name, count))
    # The constant itself must be gone.  The docstrings above it may still
    # name the placeholder, and should: that is where it is explained.
    for line in text.splitlines():
        if line.startswith("HARDCODED_API_KEY") and PLACEHOLDER in line:
            raise SystemExit("%s: placeholder survived substitution" % name)
    return text


# ---------------------------------------------------------------------------
# The zip
# ---------------------------------------------------------------------------

def build(out_path: str = DEFAULT_OUT, key: Optional[str] = None) -> str:
    """Write the zip.  ``key=None`` ships the committed placeholder."""
    if key is not None and "keyed" not in os.path.basename(out_path):
        raise SystemExit(
            "a zip carrying a real key must have 'keyed' in its filename "
            "(that is what .gitignore matches) -- got %s"
            % os.path.basename(out_path))

    table = os.path.join(PLAYERS_DIR, "oBirdy", "obirdy_simtable_v2.bin.gz")
    if not os.path.exists(table):
        raise SystemExit("no similarity table at %s -- run "
                         "python -m harness.simtable --build" % table)
    instructions = os.path.join(SUBMISSION_DIR, INSTRUCTIONS)
    if not os.path.exists(instructions):
        raise SystemExit("no %s" % instructions)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in AGENT_FILES:
            archive.writestr(name, agent_source(name, key))
        archive.write(table, SIMTABLE_ARCNAME)
        archive.write(instructions, INSTRUCTIONS)
    return out_path


def verify(out_path: str, key: Optional[str] = None) -> Dict[str, object]:
    """Read the finished zip back and check it says what it should.

    Building and verifying from the same variables would prove nothing, so this
    reopens the artefact and inspects the bytes that will actually be sent.
    """
    report: Dict[str, object] = {}
    with zipfile.ZipFile(out_path) as archive:
        names = archive.namelist()
        report["names"] = names
        expected = list(AGENT_FILES) + [SIMTABLE_ARCNAME, INSTRUCTIONS]
        missing = [name for name in expected if name not in names]
        if missing:
            raise SystemExit("zip is missing %s" % ", ".join(missing))
        for name in AGENT_FILES:
            source = archive.read(name).decode("utf-8")
            constant = [line for line in source.splitlines()
                        if line.startswith("HARDCODED_API_KEY")]
            if len(constant) != 1:
                raise SystemExit("%s: %d HARDCODED_API_KEY assignments"
                                 % (name, len(constant)))
            if key is None:
                if PLACEHOLDER not in constant[0]:
                    raise SystemExit("%s: placeholder build lost its "
                                     "placeholder" % name)
            else:
                if key not in constant[0]:
                    raise SystemExit("%s: the key did not land in the "
                                     "constant" % name)
                if source.count(key) != 1:
                    raise SystemExit("%s: the key appears %d times, expected 1"
                                     % (name, source.count(key)))
                if PLACEHOLDER in constant[0]:
                    raise SystemExit("%s: placeholder survived" % name)
        report["size"] = os.path.getsize(out_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", default=None,
                        help="dotenv file holding %s (default: ./.env)"
                             % KEY_VAR)
    parser.add_argument("--out", default=None,
                        help="output zip (default: %s, or %s with "
                             "--placeholder)" % (DEFAULT_OUT, PLACEHOLDER_OUT))
    parser.add_argument("--placeholder", action="store_true",
                        help="ship the committed placeholder instead of a key")
    args = parser.parse_args(argv)

    key = None if args.placeholder else resolve_key(args.env_file)
    out = args.out or (PLACEHOLDER_OUT if args.placeholder else DEFAULT_OUT)

    build(out, key)
    report = verify(out, key)

    print("wrote %s (%.2f MB)" % (out, report["size"] / (1024.0 * 1024.0)))
    for name in report["names"]:
        print("  %s" % name)
    if key is None:
        print("key: placeholder (%s) -- this zip cannot play" % PLACEHOLDER)
    else:
        print("key: %s substituted into 1 site in each agent file "
              "(%d characters, never printed)" % (KEY_VAR, len(key)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
