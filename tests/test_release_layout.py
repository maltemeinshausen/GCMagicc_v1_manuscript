# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_manifest_never_guesses_pm_or_xs_mapping() -> None:
    objects = {item["public_model"]: item for item in json.loads((ROOT / "data/external_data_manifest.json").read_text())["objects"]}
    assert objects["GCMagicc-PM"]["status"] == "pending-model-author-provenance"
    assert objects["GCMagicc-XS"]["status"] == "pending-model-author-provenance"


def test_repository_has_no_file_over_50_mb() -> None:
    too_large = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and path.stat().st_size > 50 * 1024 * 1024]
    assert too_large == []
