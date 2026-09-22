#!/usr/bin/env python3
"""Report changes in paths protected by AutoHumanEgo V1."""

from __future__ import annotations

import argparse
import subprocess
import sys


PROTECTED_PREFIXES = ("training/", "preprocess/", "inference/", "data/", "datasets/")
PROTECTED_EXACT = (
    "cfg/training/serve_bread/HumanEgo.yaml",
    "cfg/training/open_door/OpenDoorLeftHandOnly.yaml",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing", action="store_true", help="report but do not fail on pre-existing edits")
    args = parser.parse_args()
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], check=True, text=True, capture_output=True)
    changed = []
    for line in status.stdout.splitlines():
        path = line[3:] if len(line) >= 4 else line
        if any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES) or path in PROTECTED_EXACT:
            changed.append(line)
    if changed:
        print("Protected paths already changed:")
        print("\n".join(changed))
        return 0 if args.allow_existing else 2
    print("Protected paths are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
