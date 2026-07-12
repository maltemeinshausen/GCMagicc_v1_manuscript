#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit release validation counts from a read-only metrics SQLite file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


QUERY_DOMAIN = """SELECT version_tag, metricdomain, COUNT(*)
FROM gofnc GROUP BY version_tag, metricdomain ORDER BY version_tag, metricdomain"""
QUERY_SCENARIO = """SELECT version_tag, experiment_id, COUNT(*)
FROM gofnc GROUP BY version_tag, experiment_id ORDER BY version_tag, experiment_id"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database = args.database.resolve()
    connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    domain_rows = connection.execute(QUERY_DOMAIN).fetchall()
    scenario_rows = connection.execute(QUERY_SCENARIO).fetchall()
    connection.close()

    versions = sorted({str(row[0]) for row in domain_rows})
    payload: dict[str, object] = {
        "schema": "gcmagicc-metrics-audit/v1",
        "database": {
            "basename": database.name,
            "bytes": database.stat().st_size,
            "sha256": sha256(database),
        },
        "queries": {"gofnc_by_domain": QUERY_DOMAIN, "gofnc_by_scenario": QUERY_SCENARIO},
        "versions": {},
    }
    grand_total = 0
    grand_holdout = 0
    for version in versions:
        domains = {str(domain): int(count) for v, domain, count in domain_rows if str(v) == version}
        scenarios = {str(scenario): int(count) for v, scenario, count in scenario_rows if str(v) == version}
        total = sum(domains.values())
        holdout = scenarios.get("ssp245", 0)
        payload["versions"][version] = {
            "gofnc_records": total,
            "ssp245_holdout_records": holdout,
            "by_metricdomain": domains,
            "by_experiment_id": scenarios,
        }
        grand_total += total
        grand_holdout += holdout
    payload["all_versions"] = {
        "gofnc_records": grand_total,
        "ssp245_holdout_records": grand_holdout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
