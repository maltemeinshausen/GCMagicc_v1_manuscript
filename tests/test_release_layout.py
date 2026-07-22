# SPDX-License-Identifier: Apache-2.0
import hashlib
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
