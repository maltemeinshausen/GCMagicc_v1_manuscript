#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone dispatcher for the three locked PET sensitivity workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PET_METHODS = ("thornthwaite", "hargreaves", "penman-monteith")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(PET_METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    requested = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    invalid = sorted(set(requested) - set(PET_METHODS))
    if invalid:
        parser.error("unknown PET method(s): " + ", ".join(invalid))
    workflow = Path(__file__).with_name("754_add_SPEI_to_ensemble_outputs.py")
    for method in requested:
        command = [sys.executable, str(workflow), "--pet-method", method, *args.arguments]
        print(" ".join(command))
        if not args.dry_run:
            result = subprocess.run(command, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
