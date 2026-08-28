#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the frozen-source provenance manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def describe(path: Path) -> dict[str, str]:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("data/derived/emergent_constraints/"):
        is_original_script = rel.endswith("original_sample_emergent2.py")
        return {
            "source_repository": "gcmmagicc-data:nicolaiplots",
            "source_revision": "plots.tar sha256:43923f2cf9563ec19273ad98a7422737448749ed7f8d6bddb92e5e590f8ef36c",
            "source_path": rel,
            "copyright": "Nicolai Meinshausen",
            "license": "CC-BY-4.0",
            "role": (
                "original emergent-constraint sampling script"
                if is_original_script
                else "prepared GCMagicc-PM emergent-constraint data and provenance"
            ),
        }
    if rel.startswith("data/derived/resolution_sensitivity/"):
        return {
            "source_repository": "gcmmagicc-data:nicolaiplots",
            "source_revision": "plots.tar sha256:43923f2cf9563ec19273ad98a7422737448749ed7f8d6bddb92e5e590f8ef36c",
            "source_path": "data/nicolaiplots/plotsT1/plots_resolutions/RESOLUTIONPLOTS/run_meta.json",
            "copyright": "Nicolai Meinshausen",
            "license": "CC-BY-4.0",
            "role": "normalized GCMagicc resolution-sensitivity metadata",
        }
    if rel.startswith("data/derived/aerosol_sensitivity/source/"):
        return {
            "source_repository": "gcmmagicc-data:nicolaiplots",
            "source_revision": "plots.tar sha256:43923f2cf9563ec19273ad98a7422737448749ed7f8d6bddb92e5e590f8ef36c",
            "source_path": rel,
            "copyright": "Nicolai Meinshausen",
            "license": "Apache-2.0",
            "role": "normalized original GCMagicc aerosol generator or plotter",
        }
    if rel.startswith("data/derived/aerosol_sensitivity/"):
        return {
            "source_repository": "gcmmagicc-data:nicolaiplots",
            "source_revision": "plots.tar sha256:43923f2cf9563ec19273ad98a7422737448749ed7f8d6bddb92e5e590f8ef36c",
            "source_path": "data/nicolaiplots/plotsT1/plots_aerosol",
            "copyright": "Nicolai Meinshausen",
            "license": "CC-BY-4.0",
            "role": "normalized GCMagicc aerosol-sensitivity metadata",
        }
    if rel.startswith("data/derived/gcmagicc_xs/"):
        return {
            "source_repository": "gcmmagicc-data:GCMagicc-XS_data",
            "source_revision": "full CSV sha256:9436752a1d2a0b2af0d707bab9776d4cc8f874968cc2ad472c8ad16c4a5dca68",
            "source_path": "data/GCMagicc-XS_data/full_monthly_results.csv",
            "copyright": "Nicolai Meinshausen",
            "license": "CC-BY-4.0",
            "role": "compact GCMagicc-XS plotted data and provenance",
        }
    if rel.startswith("data/derived/validation_diagnostics/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "checkout 8b30bcd + generator snapshots; publication-set main and supplementary bundles 20260821_1121",
            "source_path": "data/manuscript_figures/validation_diagnostics/publication_set_v100/v100",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "publication-set validation-figure selection provenance",
        }
    if rel.startswith("data/derived/observational_alignment/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "gcmmagicc revision c5c3d9f + generator sha256:8b4de15fc90a1e8675ec6137c1f331e93ad7c1745779d4af45c2cbf6262cd238; bundle 20260827_221422 generated immediately before containing commit",
            "source_path": "data/manuscript_figures/observational_alignment/observational_alignment_v100_20260827_221422",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "normalized observational-alignment plotted data and provenance",
        }
    if rel.startswith("data/derived/spei_sensitivity/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "16c9d72b1c4e01fbadb019f8d82e23e1fe426e23",
            "source_path": "data/spei_sensitivity/iran_smile_full_20260722",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "normalized Iran SPEI-sensitivity prepared data and provenance",
        }
    if rel.startswith("figures/supplementary/emergent_constraints/"):
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-native prepared-data workflow 2026-07-26",
            "source_path": "src/gcmagicc_eval/workflows/1060_emergent_constraints.py",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "supplementary GCMagicc-PM emergent-constraints figure",
        }
    if rel.startswith("figures/supplementary/resolution_sensitivity/"):
        return {
            "source_repository": "gcmmagicc-data:nicolaiplots",
            "source_revision": "plots.tar sha256:43923f2cf9563ec19273ad98a7422737448749ed7f8d6bddb92e5e590f8ef36c",
            "source_path": "data/nicolaiplots/plotsT1/plots_resolutions/RESOLUTIONPLOTS/synthesis_figures",
            "copyright": "Nicolai Meinshausen",
            "license": "CC-BY-4.0",
            "role": "supplementary GCMagicc resolution-sensitivity synthesis",
        }
    if rel.startswith("figures/supplementary/aerosol_sensitivity/"):
        return {
            "source_repository": "gcmmagicc-data:nicolaiplots",
            "source_revision": "plots.tar sha256:43923f2cf9563ec19273ad98a7422737448749ed7f8d6bddb92e5e590f8ef36c",
            "source_path": "data/nicolaiplots/plotsT1/plots_aerosol",
            "copyright": "Nicolai Meinshausen",
            "license": "CC-BY-4.0",
            "role": "supplementary GCMagicc aerosol-sensitivity map",
        }
    if rel.startswith("figures/supplementary/gcmagicc_xs_extrapolation/"):
        return {
            "source_repository": "gcmmagicc-data:GCMagicc-XS_data",
            "source_revision": "full CSV sha256:9436752a1d2a0b2af0d707bab9776d4cc8f874968cc2ad472c8ad16c4a5dca68",
            "source_path": "data/GCMagicc-XS_data/figures",
            "copyright": "Nicolai Meinshausen",
            "license": "CC-BY-4.0",
            "role": "supplementary GCMagicc-XS prediction-skill figure",
        }
    if rel.startswith("figures/main/validation_diagnostics/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "checkout 8b30bcd + generator sha256:3bda0f95467015dc2ee4a913063e390b34ab4fd00f7771b79900210cb3acdbe9; batch sha256:39148a59207a285ef82a8b10e03b81e882f113248d064c35a88a5baf6ca4bb4b; bundle 20260821_1121",
            "source_path": "data/manuscript_figures/validation_diagnostics/publication_set_v100/v100/main",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "main validation diagnostics selected by semantic manuscript role",
        }
    if rel.startswith("figures/supplementary/validation_diagnostics/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "checkout 8b30bcd + generator sha256:3bda0f95467015dc2ee4a913063e390b34ab4fd00f7771b79900210cb3acdbe9; batch sha256:39148a59207a285ef82a8b10e03b81e882f113248d064c35a88a5baf6ca4bb4b; bundle 20260821_1121",
            "source_path": "data/manuscript_figures/validation_diagnostics/publication_set_v100/v100/supplementary",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "supplementary validation diagnostics selected by semantic manuscript role",
        }
    if rel.startswith(("figures/supplementary/scoreedistc_ssp245/", "figures/supplementary/observation_referenced_edisto/")):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "metrics database sha256:34feda39f369142300224773efde1c5e80c32133cc4c324d20de83403239fb03",
            "source_path": "data/metric_databases/metrics.sqlite",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "supplementary validation metric panel",
        }
    if rel.startswith("figures/main/observational_alignment/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "gcmmagicc revision c5c3d9f + generator sha256:8b4de15fc90a1e8675ec6137c1f331e93ad7c1745779d4af45c2cbf6262cd238; bundle 20260827_221422 generated immediately before containing commit",
            "source_path": "data/manuscript_figures/observational_alignment/observational_alignment_v100_20260827_221422",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "main observational-alignment figure",
        }
    if rel.startswith("figures/supplementary/spei_sensitivity/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "16c9d72b1c4e01fbadb019f8d82e23e1fe426e23",
            "source_path": "data/manuscript_figures/supplementary/spei_sensitivity",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "supplementary Iran SPEI-sensitivity figure",
        }
    if rel in {"provenance/source_data_audit.csv", "provenance/source_data_audit.md", "provenance/figure_registry.csv"}:
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release audit 2026-07-26",
            "source_path": rel,
            "copyright": "Nicolai Meinshausen; Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "release provenance audit or semantic figure registry",
        }
    if rel.startswith("data/derived/iran_drought_attribution/"):
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-generated common protocol v1",
            "source_path": "src/gcmagicc_eval/workflows/1090_drought_common_protocol.py",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "drought common-protocol derived data and run manifest",
        }
    if rel.startswith("data/derived/turkiye_regional_scenarios/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "2b9bd0a9dfb111a0f813c77caaa7c798fe219c2e",
            "source_path": "data/projection_plots_simple_815/versioned/v100/regional_scenario_july1_exact_20260703",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "frozen annual Türkiye regional projection percentiles",
        }
    if rel == "data/derived/validation_metrics/metrics_audit.json":
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "metrics.sqlite snapshot 2026-04-20",
            "source_path": "data/metric_databases/metrics.sqlite (read-only audit; database not copied)",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "validation record-count audit with source database checksum and SQL",
        }
    if rel.startswith("figures/main/iran_drought_attribution/"):
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "2b9bd0a9dfb111a0f813c77caaa7c798fe219c2e",
            "source_path": "data/manuscript_figures/drought_attribution/v100/IRN/20260721_001353",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "selected 13-panel Iran drought-attribution figure and provenance",
        }
    if rel.startswith("figures/supplementary/smile_common_protocol/"):
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-generated common protocol v1",
            "source_path": "src/gcmagicc_eval/workflows/1090_drought_common_protocol.py",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "supplementary drought SMILE common-protocol figure",
        }
    if rel.startswith("figures/archive/iran_drought_attribution/"):
        source = (
            "src/gcmagicc_eval/workflows/1130_drought_attribution_synthesis.py"
            if "synthesis" in rel
            else "src/gcmagicc_eval/workflows/1090_drought_common_protocol.py"
        )
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-generated common protocol v1",
            "source_path": source,
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "preserved unselected Iran drought-attribution variant",
        }
    if rel.startswith("figures/main/turkiye_regional_scenarios/"):
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-generated Türkiye application v1",
            "source_path": "src/gcmagicc_eval/workflows/1100_turkiye_regional_application.py",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "Türkiye regional application figure and summary",
        }
    if rel.startswith("figures/main/training_inference_workflow/"):
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-generated workflow schematic v1",
            "source_path": "src/gcmagicc_eval/workflows/1120_gcmagicc_workflow_schematic.py",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "GCMagicc training and inference workflow schematic",
        }
    if rel.startswith("src/gcmagicc_model/gcmagicc/"):
        name = rel.rsplit("/", 1)[-1]
        return {
            "source_repository": "unversioned:gcm_firefly_data",
            "source_revision": "content-hash snapshot 2026-07-11",
            "source_path": f"model_NxlversA5/{name}",
            "copyright": "Nicolai Meinshausen",
            "license": "Apache-2.0",
            "role": "GCMagicc core model/inference",
        }
    if rel.startswith("src/gcmagicc_model/gcmagicc_ce/"):
        name = rel.rsplit("/", 1)[-1]
        return {
            "source_repository": "unversioned:gcm_firefly_data",
            "source_revision": "content-hash snapshot 2026-07-11",
            "source_path": f"model_NthreeversT1/{name}",
            "copyright": "Nicolai Meinshausen",
            "license": "Apache-2.0",
            "role": "GCMagicc-CE core model/inference",
        }
    if rel.startswith("src/gcmagicc_eval/helpers/"):
        helper_rel = rel.removeprefix("src/gcmagicc_eval/helpers/")
        if helper_rel == "cmip6_operations.py":
            return {
                "source_repository": "cmipcruncher_firefly",
                "source_revision": "e310d238536fa136eacb5e7ace118b1ec2cc0837",
                "source_path": "src/cmip6cruncher/operations.py",
                "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
                "license": "Apache-2.0",
                "role": "training-data smoothing helper",
            }
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "2b9bd0a9dfb111a0f813c77caaa7c798fe219c2e",
            "source_path": f"scr/{helper_rel}",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "Apache-2.0",
            "role": "evaluation helper",
        }
    if rel.startswith("src/gcmagicc_eval/recipes/"):
        name = rel.rsplit("/", 1)[-1]
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "2b9bd0a9dfb111a0f813c77caaa7c798fe219c2e",
            "source_path": f"notebooks/recipes/{name}",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "Apache-2.0",
            "role": "evaluation PET and SPEI recipe adapted for standalone import",
        }
    if rel.startswith("src/gcmagicc_eval/workflows/"):
        name = rel.rsplit("/", 1)[-1]
        if name in {
            "1050_resolution_sensitivity.py",
            "1060_emergent_constraints.py",
            "1070_aerosol_pattern.py",
            "1080_gcmagicc_xs_prediction_skill.py",
        }:
            return {
                "source_repository": "GCMagicc_v1_manuscript",
                "source_revision": "release-native provenance integration 2026-07-26",
                "source_path": rel,
                "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
                "license": "Apache-2.0",
                "role": "release-native manuscript figure workflow",
            }
        if name == "1090_drought_common_protocol.py":
            return {
                "source_repository": "GCMagicc_v1_manuscript",
                "source_revision": "release-native implementation 2026-07-11",
                "source_path": rel,
                "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
                "license": "Apache-2.0",
                "role": "corrected drought and three-SMILE common-protocol workflow",
            }
        if name == "1120_gcmagicc_workflow_schematic.py":
            return {
                "source_repository": "GCMagicc_v1_manuscript",
                "source_revision": "release-native implementation 2026-07-12",
                "source_path": rel,
                "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
                "license": "Apache-2.0",
                "role": "publication workflow schematic generator",
            }
        if name == "1130_drought_attribution_synthesis.py":
            return {
                "source_repository": "GCMagicc_v1_manuscript",
                "source_revision": "release-native implementation 2026-07-12",
                "source_path": rel,
                "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
                "license": "Apache-2.0",
                "role": "corrected seven-panel drought synthesis figure generator",
            }
        if name.startswith(("320_", "321_", "331_")):
            repo, revision = "gcmagicc_ensemble_runner", "fabeb92623a82d7adc0527ded00177ba09f1d2a8"
            source = f"notebooks/{name}"
        elif name == "220_data_processing.py":
            repo, revision = "cmipcruncher_firefly", "e310d238536fa136eacb5e7ace118b1ec2cc0837"
            source = f"notebooks/{name}"
        else:
            repo, revision = "gcmmagicc", "2b9bd0a9dfb111a0f813c77caaa7c798fe219c2e"
            source = f"notebooks/{name}"
        return {
            "source_repository": repo,
            "source_revision": revision,
            "source_path": source,
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "Apache-2.0",
            "role": "evaluation/figure workflow",
        }
    return {
        "source_repository": "2025magicc",
        "source_revision": "67e3609782c400bff42a04e384fb7fd0ec12bf9a",
        "source_path": "derived from output_resampled_100/AR6/runmode_natural/ssp245_ar6_natural.parquet run_id=0",
        "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
        "license": "CC-BY-4.0",
        "role": "natural-forcing predictor data",
    }


def main() -> None:
    files = sorted((ROOT / "src/gcmagicc_model").rglob("*.py"))
    files += sorted((ROOT / "src/gcmagicc_model").rglob("*.pkl"))
    files += sorted((ROOT / "src/gcmagicc_model").rglob("*.pt"))
    files += sorted((ROOT / "src/gcmagicc_model").rglob("*.csv"))
    files += sorted((ROOT / "src/gcmagicc_eval").rglob("*.py"))
    files += [ROOT / "data/natural_forcing_ssp245_ar6_run0_1850-2100.csv"]
    files += sorted((ROOT / "data/derived/iran_drought_attribution").glob("*"))
    files += sorted((ROOT / "data/derived/turkiye_regional_scenarios").rglob("*.json"))
    files += sorted((ROOT / "data/derived/validation_metrics").glob("*.json"))
    for name in (
        "emergent_constraints",
        "resolution_sensitivity",
        "aerosol_sensitivity",
        "gcmagicc_xs",
        "validation_diagnostics",
        "observational_alignment",
        "spei_sensitivity",
    ):
        files += sorted(path for path in (ROOT / "data/derived" / name).rglob("*") if path.is_file())
    files += sorted((ROOT / "figures/main/iran_drought_attribution").glob("*"))
    files += sorted((ROOT / "figures/supplementary/smile_common_protocol").glob("*"))
    files += sorted((ROOT / "figures/archive/iran_drought_attribution").glob("*"))
    files += sorted((ROOT / "figures/main/turkiye_regional_scenarios").glob("*"))
    files += sorted((ROOT / "figures/main/training_inference_workflow").glob("*"))
    for relative in (
        "figures/supplementary/emergent_constraints",
        "figures/supplementary/resolution_sensitivity",
        "figures/supplementary/aerosol_sensitivity",
        "figures/supplementary/gcmagicc_xs_extrapolation",
        "figures/main/validation_diagnostics",
        "figures/supplementary/validation_diagnostics",
        "figures/supplementary/scoreedistc_ssp245",
        "figures/supplementary/observation_referenced_edisto",
        "figures/main/observational_alignment",
        "figures/supplementary/spei_sensitivity",
    ):
        files += sorted(path for path in (ROOT / relative).rglob("*") if path.is_file())
    files += [
        ROOT / "provenance/source_data_audit.csv",
        ROOT / "provenance/source_data_audit.md",
        ROOT / "provenance/figure_registry.csv",
    ]
    rows = []
    for path in sorted(set(files)):
        meta = describe(path)
        rows.append(
            {
                **meta,
                "destination": path.relative_to(ROOT).as_posix(),
                "sha256": digest(path),
                "bytes": str(path.stat().st_size),
            }
        )
    fields = ["source_repository", "source_revision", "source_path", "destination", "sha256", "bytes", "copyright", "license", "role"]
    with (ROOT / "provenance/manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} provenance rows")


if __name__ == "__main__":
    main()
