# SPDX-License-Identifier: Apache-2.0
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from gcmagicc_repro.release import FIGURES


ROOT = Path(__file__).resolve().parents[1]


def test_external_manifest_records_model_author_release_objects() -> None:
    objects = {
        item["id"]: item
        for item in json.loads((ROOT / "data/external_data_manifest.json").read_text())["objects"]
    }
    assert objects["gcmagicc-pm-bundle"]["public_model"] == "GCMagicc-PM"
    assert objects["gcmagicc-xs-bundle"]["public_model"] == "GCMagicc-XS"
    figure_source = objects["gcmagicc-xs-figure-source"]
    assert figure_source["sha256"] == (
        "9436752a1d2a0b2af0d707bab9776d4cc8f874968cc2ad472c8ad16c4a5dca68"
    )
    assert figure_source["status"] == "not-redistributed"
    assert figure_source["deposit"] is False
    assert figure_source["url"] is None


def test_repository_has_no_file_over_50_mb() -> None:
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    public_paths = [ROOT / item.decode("utf-8") for item in listed.split(b"\0") if item]
    too_large = [path for path in public_paths if path.is_file() and path.stat().st_size > 50 * 1024 * 1024]
    assert too_large == []


def test_turkiye_release_bundle_is_complete() -> None:
    summary = json.loads(
        (ROOT / "figures/main/turkiye_regional_scenarios/turkiye_regional_scenarios.json").read_text()
    )
    assert summary["baseline"] == [1995, 2014]
    assert summary["future"] == [2081, 2100]
    assert summary["schema"] == "gcmagicc-turkiye-regional-scenarios/v3"
    assert set(summary["variables"]) == {"tas", "pr", "hurs"}
    assert {key: len(value) for key, value in summary["scenario_groups"].items()} == {
        "cmip6_ssps": 8,
        "ndc_ssp2com_current_policy": 10,
        "cmip7": 7,
    }
    assert len(summary["inputs"]) == 75
    assert all(item["plotted"] for item in summary["era5"].values())
    assert all(item["end_year"] == 2025 for item in summary["era5"].values())
    assert all(item["members"] == 20 for values in summary["variables"].values() for item in values.values())


def test_workflow_schematic_records_the_locked_method() -> None:
    summary = json.loads((ROOT / "figures/main/training_inference_workflow/gcmagicc_training_inference_workflow.json").read_text())
    assert summary["schema"] == "gcmagicc-training-inference-workflow/v2"
    assert summary["inference"]["requires_gridded_esm_or_reanalysis_fields"] is False
    assert summary["inference"]["requires_prescribed_sst_or_ocean_state"] is False
    assert summary["inference"]["full_predictor_variants"] == ["GCMagicc", "GCMagicc-CE"]
    assert summary["inference"]["reduced_predictor_variants"] == ["GCMagicc-PM", "GCMagicc-XS"]
    assert summary["inference"]["correction"] == {
        "changed_predictors": ["tas_smoothed"],
        "iterative_convergence_loop": False,
        "passes": 2,
        "unchanged_seed": True,
    }
    assert set(summary["outputs"]) == {
        "gcmagicc_training_inference_workflow.pdf",
        "gcmagicc_training_inference_workflow.png",
        "gcmagicc_training_inference_workflow.svg",
    }


def test_drought_hybrid_figure_restores_context_without_old_results() -> None:
    sidecar = json.loads(
        (ROOT / "figures/archive/iran_drought_attribution/iran_drought_attribution_synthesis_common_protocol.json").read_text()
    )
    assert sidecar["schema"] == "gcmagicc-drought-attribution-synthesis/v2"
    assert list(sidecar["panels"]) == list("abcdefg")
    assert sidecar["method"]["baseline"] == [1991, 2010]
    assert sidecar["method"]["aggregation"] == "area-weighted mean of grid-cell-standardized December SPEI-48"
    assert sidecar["design_provenance"]["scientific_results"].startswith("corrected common-protocol")
    assert len(sidecar["inputs"]) == 4
    assert set(sidecar["outputs"]) == {
        "iran_drought_attribution_synthesis_common_protocol.pdf",
        "iran_drought_attribution_synthesis_common_protocol.png",
    }

    map_artifact = json.loads(
        (ROOT / "data/derived/iran_drought_attribution/era5_irn_penman_monteith_spei48_map.json").read_text()
    )
    assert map_artifact["schema"] == "gcmagicc-era5-irn-event-map/v1"
    assert map_artifact["event"] == "December 2025"
    assert map_artifact["baseline"] == [1991, 2010]
    assert any(boundary["iso3"] == "IRN" for boundary in map_artifact["boundaries"])
    assert map_artifact["source"]["sha256"] == "2c6fd00ebd257794dd1bbe17be6c5a0e24b4caa7e8234949108e1ac01dcbcef0"

    cmip6_sidecar = json.loads(
        (ROOT / "data/derived/iran_drought_attribution/cmip6_irn_penman_monteith_spei48_sidecar.json").read_text()
    )
    assert cmip6_sidecar["schema"] == "gcmagicc-cmip6-drought-sidecar/v1"
    assert len(cmip6_sidecar["factual"]) == 54
    assert len(cmip6_sidecar["natural"]) == 9
    assert "not used in corrected attribution statistics" in cmip6_sidecar["purpose"]
    assert cmip6_sidecar["source"]["sha256"] == "c9e76841922b6b552bc82c3fae70787269121d704fef09cbf47e7ae992a72b3b"


def test_selected_iran_drought_figure_and_provenance_are_frozen() -> None:
    figure = ROOT / "figures/main/iran_drought_attribution/iran_drought_attribution_gcmagicc_v100.pdf"
    provenance = json.loads(
        (ROOT / "figures/main/iran_drought_attribution/iran_drought_attribution_gcmagicc_v100_provenance.json").read_text()
    )
    assert provenance["schema"] == "gcmagicc-iran-drought-attribution/v2"
    assert provenance["source_revision"] == "2b9bd0a9dfb111a0f813c77caaa7c798fe219c2e"
    assert provenance["artifact"] == {
        "path": "figures/main/iran_drought_attribution/iran_drought_attribution_gcmagicc_v100.pdf",
        "bytes": 50_708_076,
        "sha256": "8c456857fb63f4da9d8bac5969c49be02d1394790ff49b8b2819e6e6938233c3",
        "panels": "a-m",
    }
    assert figure.stat().st_size == provenance["artifact"]["bytes"]
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == provenance["artifact"]["sha256"]
    assert provenance["smile_protocol"]["rsds_adjustment"] == "none; raw rsds"
    assert provenance["smile_protocol"]["baseline"] == {
        "start_year": 1991,
        "end_year": 2010,
        "pooling": "model-pooled across every eligible historical member",
        "calendar_month_specific": True,
    }
    assert provenance["smile_protocol"]["member_counts"] == {
        "CanESM5": {"historical": 65, "hist-nat": 50, "ssp245": 50},
        "MIROC6": {"historical": 50, "hist-nat": 50, "ssp245": 50},
        "GISS-E2-1-G": {"historical": 46, "hist-nat": 20, "ssp245": 22},
    }


def test_validation_audit_matches_frozen_database_snapshot() -> None:
    audit = json.loads((ROOT / "data/derived/validation_metrics/metrics_audit.json").read_text())
    assert audit["database"]["sha256"] == "34feda39f369142300224773efde1c5e80c32133cc4c324d20de83403239fb03"
    assert audit["all_versions"] == {
        "gofnc_records": 4_539_079,
        "ssp245_holdout_records": 1_265_222,
    }


def test_current_validation_publication_set_matches_provenance() -> None:
    provenance = json.loads(
        (ROOT / "data/derived/validation_diagnostics/provenance.json").read_text()
    )
    assert provenance["schema"] == "gcmagicc-validation-figure-provenance/v2"
    assert provenance["source_databases"]["metrics"] == {
        "source_path": "data/metric_databases/metrics.sqlite",
        "external_object": "validation-diagnostics-metrics-sqlite-v20260821",
        "bytes": 12_056_313_856,
        "sha256": "70cba2cb782e8061ebfe4e6ef9bf47cf4a6a0e7f160f91ea1851780e54036150",
        "committed": False,
    }
    assert provenance["source_databases"]["energy_distance"] == {
        "source_path": "data/edist_databases/edist.sqlite",
        "external_object": "validation-diagnostics-edist-sqlite-v20260821",
        "bytes": 2_697_842_688,
        "sha256": "0bae785e0c539dfd2a12f576a40eb8723635838a1c25ea5cf77be691bed31352",
        "committed": False,
    }

    main = provenance["semantic_roles"]["main_validation_diagnostics"]
    main_path = ROOT / main["artifact"]
    assert main_path.stat().st_size == main["bytes"]
    assert hashlib.sha256(main_path.read_bytes()).hexdigest() == main["sha256"]

    supplementary = provenance["semantic_roles"]["supplementary_validation_diagnostics"]
    supplementary_root = ROOT / supplementary["artifact_directory"]
    assert len(supplementary["artifacts"]) == 10
    for artifact in supplementary["artifacts"]:
        path = supplementary_root / artifact["file"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    for record in [provenance["publication_set"]["manifest"], *provenance["table_sources"]]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    validation = provenance["source_databases"]["validation_record"]
    validation_path = ROOT / validation["path"]
    assert validation_path.stat().st_size == validation["bytes"]
    assert hashlib.sha256(validation_path.read_bytes()).hexdigest() == validation["sha256"]

    external = {
        item["id"]: item
        for item in json.loads((ROOT / "data/external_data_manifest.json").read_text())["objects"]
    }
    for key in ("metrics", "energy_distance"):
        database = provenance["source_databases"][key]
        external_record = external[database["external_object"]]
        assert external_record["bytes"] == database["bytes"]
        assert external_record["sha256"] == database["sha256"]


def test_observational_alignment_bundle_matches_provenance() -> None:
    provenance = json.loads(
        (ROOT / "data/derived/observational_alignment/provenance.json").read_text()
    )
    assert provenance["schema"] == "gcmagicc-observational-alignment-provenance/v1"
    assert provenance["authoritative_bundle"] == "observational_alignment_v100_20260827_221422"
    assert provenance["source_revision"] == {
        "git_base": "gcmmagicc revision 8b30bcd9743f53cf7eecd79d112c64373bcd9b13",
        "containing_revision": "gcmmagicc revision c5c3d9f170f332649f81ee16f3403da6c650d599",
        "working_tree": "bundle generated immediately before commit c5c3d9f, which contains the exact generator content",
        "generator_path": "notebooks/1021_observational_alignment.py",
        "generator_sha256": "8b4de15fc90a1e8675ec6137c1f331e93ad7c1745779d4af45c2cbf6262cd238",
    }

    records = [
        provenance["normalized_json"],
        *provenance["compact_panel_tables"],
        provenance["canonical_pdf"],
        provenance["canonical_png"],
    ]
    for record in records:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    normalized = (ROOT / provenance["normalized_json"]["path"]).read_text()
    normalized_payload = json.loads(normalized)

    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    normalized_strings = list(strings(normalized_payload))
    assert all(not value.startswith("/") for value in normalized_strings)
    assert any(value.startswith("external/gcmmagicc/") for value in normalized_strings)

    panel_b = (ROOT / provenance["compact_panel_tables"][0]["path"]).read_text()
    assert "n100 CMIP7 ScenarioMIP" in panel_b


def _read_numeric_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_emergent_prepared_data_and_reconstructed_medians_are_complete() -> None:
    data_root = ROOT / "data/derived/emergent_constraints"
    model_rows = _read_numeric_csv(data_root / "model_trends.csv")
    assert len(model_rows) == 799
    assert len({row["version"] for row in model_rows}) == 19

    residuals = np.asarray(
        [float(row["trend_y_true"]) - float(row["trend_y_hat"]) for row in model_rows],
        dtype=float,
    )
    expected_counts = {
        "ssp119": 30,
        "ssp126": 39,
        "ssp370": 48,
        "ssp434": 15,
        "ssp460": 12,
        "ssp585": 57,
    }
    expected_medians = {
        "ssp119": 1.9335785,
        "ssp126": 2.33450315,
        "ssp370": 4.28726205,
        "ssp434": 2.60142518,
        "ssp460": 3.58517455,
        "ssp585": 5.67181405,
    }
    quantile_rows = {
        row["quantile"]: row
        for row in _read_numeric_csv(data_root / "quantiles_by_scenario.csv")
    }
    assert set(quantile_rows) == {"2.5%", "5%", "10%", "50%", "90%", "95%", "97.5%"}

    for scenario, count in expected_counts.items():
        era_rows = _read_numeric_csv(data_root / f"era5_conditioned_{scenario}.csv")
        assert len(era_rows) == count
        era = np.asarray([float(row["trend_yera_hat"]) for row in era_rows], dtype=float)
        rng = np.random.default_rng(0)
        draws = rng.choice(era, size=2000, replace=True) + rng.choice(
            residuals, size=2000, replace=True
        )
        median = float(np.quantile(draws, 0.5))
        np.testing.assert_allclose(median, expected_medians[scenario], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            float(quantile_rows["50%"][scenario]), median, rtol=0.0, atol=1e-12
        )


def test_figure_dependencies_match_public_model_variants() -> None:
    assert FIGURES["resolution"][1] == ["gcmagicc-checkpoints"]
    assert FIGURES["aerosol"][1] == ["gcmagicc-checkpoints"]
    assert FIGURES["emergent"][1] == []
    assert FIGURES["xs"][1] == []


def test_xs_compact_points_are_release_native_and_complete() -> None:
    provenance = json.loads(
        (ROOT / "data/derived/gcmagicc_xs/provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["raw_source"]["bytes"] == 1_528_623_568
    assert provenance["raw_source"]["sha256"] == "9436752a1d2a0b2af0d707bab9776d4cc8f874968cc2ad472c8ad16c4a5dca68"
    assert provenance["compact_plotted_points"]["rows"] == 205_407
    assert provenance["compact_plotted_points"]["groups"] == 1_440
    assert provenance["reproducibility"]["compact_summary_to_figure"] == "release-native"


def test_semantic_figure_registry_does_not_assign_numbers() -> None:
    registry = (ROOT / "provenance/figure_registry.csv").read_text(encoding="utf-8")
    assert "Figure1" not in registry
    assert "Figure 1" not in registry
    assert "figure_number" not in registry
