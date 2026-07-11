# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from .release import fetch, reproduce, smoke, verify


def main() -> int:
    parser = argparse.ArgumentParser(prog="gcmagicc-repro")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    sub.add_parser("smoke")
    sub.add_parser("verify")
    repro = sub.add_parser("reproduce")
    repro.add_argument("--figure", required=True)
    repro.add_argument("--dry-run", action="store_true")
    repro.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "fetch":
        return fetch()
    if args.command == "smoke":
        return smoke()
    if args.command == "verify":
        return verify()
    return reproduce(args.figure, args.dry_run, args.args)


if __name__ == "__main__":
    raise SystemExit(main())
