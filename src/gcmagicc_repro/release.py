# SPDX-License-Identifier: Apache-2.0
"""Fetch, verify, smoke, and figure-dispatch operations."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .kernels import area_weighted_mean, corrected_tas_predictor, modified_hargreaves_monthly_mm, moving_block_resample


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_MANIFEST = ROOT / "data" / "external_data_manifest.json"

FIGURES = {
    "turkiye": ("1100_turkiye_regional_application.py", []),
    "drought": ("1040_Figure4_DroughtAttribution_ExampleCountry.py", []),
    "drought-common-protocol": ("1090_drought_common_protocol.py", []),
    "resolution": ("1050_figureSX_resolution.py", ["gcmagicc-checkpoints"]),
    "emergent": ("1060_figureX_emergentConstraints.py", ["gcmagicc-pm-bundle"]),
    "aerosol": ("1070_figureX_AerosolPattern.py", ["gcmagicc-pm-bundle"]),
    "xs": ("1080_Figure_GCMAGICC-XS_predictionskill.py", ["gcmagicc-xs-bundle"]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external() -> dict:
    return json.loads(EXTERNAL_MANIFEST.read_text(encoding="utf-8"))


def fetch() -> int:
    pending: list[str] = []
    for obj in load_external()["objects"]:
        if obj["status"] != "published":
            pending.append(f"{obj['id']} ({obj['status']})")
            continue
        destination = ROOT / obj["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temp = Path(handle.name)
        try:
            urllib.request.urlretrieve(obj["url"], temp)
            if temp.stat().st_size != obj["bytes"] or sha256(temp) != obj["sha256"]:
                raise RuntimeError(f"checksum or size mismatch for {obj['id']}")
            temp.replace(destination)
        finally:
            if temp.exists():
                temp.unlink()
    if pending:
        print("External release objects still pending:")
        for item in pending:
            print(f"  {item}")
        return 2
    return 0


def smoke() -> int:
    cfg = json.loads((ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))
    h = cfg["hargreaves"]
    pet = modified_hargreaves_monthly_mm(
        h["temperature_c"], h["temperature_min_c"], h["temperature_max_c"], h["rsds_mj_m2_day"], h["days"]
    )
    corrected = corrected_tas_predictor([1.0, 1.0, 1.0], cfg["two_pass_delta_c"])
    weighted = area_weighted_mean([-1.0, -2.0, -3.0], [0.0, 45.0, 70.0])
    boot = moving_block_resample(list(range(20)), block=5, seed=cfg["seed"])
    result = {
        "seed": cfg["seed"],
        "modified_hargreaves_monthly_mm": round(pet, 10),
        "corrected_tas_predictor": corrected,
        "area_weighted_mean": round(weighted, 10),
        "bootstrap_prefix": boot[:10],
        "december_samples_per_year": 1,
        "bootstrap_replicates_release_protocol": 10000,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def reproduce(figure: str, dry_run: bool, extra: list[str]) -> int:
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure!r}; choose from {', '.join(sorted(FIGURES))}")
    script_name, required = FIGURES[figure]
    objects = {item["id"]: item for item in load_external()["objects"]}
    blocked = [item for item in required if objects[item]["status"] != "published"]
    script = ROOT / "src" / "gcmagicc_eval" / "workflows" / script_name
    command = [sys.executable, str(script), *extra]
    print(" ".join(command))
    if blocked:
        print("Blocked by unpublished external objects: " + ", ".join(blocked))
        return 2
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def verify() -> int:
    failures: list[str] = []
    forbidden_machine_paths = (
        "/data/projects/",
        "/scratch2/",
        "/mnt/fressnapf/",
        "/data/scratch/",
        "/r/scratch/",
        "/home/",
    )
    placeholder_tokens = ("Review" + "Placeholder", "TO" + "DO", "TB" + "D", "10.0000/")
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.stat().st_size > 50 * 1024 * 1024:
            failures.append(f"file exceeds 50 MB: {path.relative_to(ROOT)}")
        if (
            path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix.lower() in {"", ".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
            and path != Path(__file__)
        ):
            content = path.read_text(encoding="utf-8", errors="ignore")
            for prefix in forbidden_machine_paths:
                if prefix in content:
                    failures.append(f"machine-specific path in {path.relative_to(ROOT)}: {prefix}")
            for token in placeholder_tokens:
                if token in content:
                    failures.append(f"untracked placeholder token in {path.relative_to(ROOT)}: {token}")
    for required in ("LICENSES/Apache-2.0.txt", "LICENSES/CC-BY-4.0.txt", "NOTICE", ".reuse/dep5"):
        if not (ROOT / required).is_file():
            failures.append(f"missing {required}")
    manifest = ROOT / "provenance" / "manifest.csv"
    if not manifest.is_file():
        failures.append("missing provenance/manifest.csv")
    else:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                destination = ROOT / row["destination"]
                if not destination.is_file():
                    failures.append(f"missing frozen file {row['destination']}")
                elif sha256(destination) != row["sha256"]:
                    failures.append(f"hash mismatch {row['destination']}")
    forcing = ROOT / "data" / "natural_forcing_ssp245_ar6_run0_1850-2100.csv"
    if not forcing.is_file() or sha256(forcing) != "089372873cc283e8188c94dc9818cd0b75b694e4f38457f2925d516169c3e801":
        failures.append("natural-forcing artifact missing or changed")
    external = load_external()
    if external.get("schema") != "gcmagicc-external-data/v1":
        failures.append("invalid external manifest schema")
    if failures:
        print("VERIFY: FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("VERIFY: PASS")
    return 0
