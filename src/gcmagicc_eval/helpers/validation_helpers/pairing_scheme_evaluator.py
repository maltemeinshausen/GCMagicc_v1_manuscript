from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

SC_VERSION = "v101"
SO_VERSION = "v101gxe"

DOMAIN_BY_VERSION = {
    SC_VERSION: "S(c)",
    SO_VERSION: "S(o)",
}

_ALLOWED_COMPARISONS_BY_DOMAIN = {
    "S(c)": {"nc", "cc", "cz", "zz", "nn"},
    "S(o)": {"oc", "cc", "on", "nn"},
}

_GROUPING_COLUMNS = ["domain", "gof_bucket", "recipe"]
_REQUIRED_RUN_COLUMNS = [
    "model_version",
    "recipe",
    "comparison",
    "source_id",
    "experiment_id",
    "member_id",
    "comp_source_id",
    "comp_member_id",
]


@dataclass(frozen=True)
class EvaluationOutputs:
    strict_domain_counts: pd.DataFrame
    reuse_aware_counts: pd.DataFrame
    cc_reuse_groups: pd.DataFrame
    option_ranking: pd.DataFrame


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _make_run_key(row: pd.Series, *, include_version: bool) -> str:
    fields = []
    if include_version:
        fields.append(_as_text(row.get("model_version", "")))
    fields.extend(
        [
            _as_text(row.get("recipe", "")),
            _as_text(row.get("comparison", "")).lower(),
            _as_text(row.get("source_id", "")),
            _as_text(row.get("experiment_id", "")),
            _as_text(row.get("member_id", "")),
            _as_text(row.get("comp_source_id", "")),
            _as_text(row.get("comp_member_id", "")),
        ]
    )
    return "|".join(fields)


def _domain_for_version(version: str) -> str:
    dom = DOMAIN_BY_VERSION.get(_as_text(version))
    if not dom:
        raise ValueError(
            f"Unsupported model_version '{version}'. Expected one of: {', '.join(sorted(DOMAIN_BY_VERSION))}."
        )
    return dom


def _gof_bucket_for(version: str, comparison: str) -> str:
    comp = _as_text(comparison).lower()
    if comp == "nn":
        if _as_text(version) == SC_VERSION:
            return "NN_SC"
        if _as_text(version) == SO_VERSION:
            return "NN_SO"
        return "NN"
    if comp == "cc":
        if _as_text(version) == SC_VERSION:
            return "CC_SC"
        if _as_text(version) == SO_VERSION:
            return "CC_SO"
        return "CC"
    mapping = {
        "nc": "NC",
        "cz": "CZ",
        "zz": "ZZ",
        "oc": "OC",
        "on": "ON",
    }
    return mapping.get(comp, comp.upper())


def _normalize_runs(
    df: pd.DataFrame,
    *,
    require_option: bool,
    option_column: str = "option",
) -> pd.DataFrame:
    missing = [c for c in _REQUIRED_RUN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required run columns: {', '.join(missing)}")
    if require_option and option_column not in df.columns:
        raise ValueError(f"Expected option column '{option_column}' in expected-runs table")

    out = df.copy()
    if option_column not in out.columns:
        out[option_column] = "baseline"

    for col in _REQUIRED_RUN_COLUMNS + [option_column]:
        out[col] = out[col].map(_as_text)

    out["comparison"] = out["comparison"].str.lower()
    out["domain"] = out["model_version"].map(_domain_for_version)

    allowed = out.apply(
        lambda r: r["comparison"] in _ALLOWED_COMPARISONS_BY_DOMAIN.get(r["domain"], set()),
        axis=1,
    )
    if not bool(allowed.all()):
        bad = out.loc[~allowed, ["model_version", "domain", "comparison"]].head(5)
        raise ValueError(
            "Found comparison/domain mismatches (first rows): "
            + bad.to_dict(orient="records").__repr__()
        )

    out["gof_bucket"] = out.apply(
        lambda r: _gof_bucket_for(r["model_version"], r["comparison"]),
        axis=1,
    )
    out["run_key"] = out.apply(lambda r: _make_run_key(r, include_version=True), axis=1)
    out["cc_reuse_group"] = out.apply(
        lambda r: _make_run_key(r, include_version=False) if r["comparison"] == "cc" else "",
        axis=1,
    )

    keep_cols = [
        option_column,
        "model_version",
        "domain",
        "recipe",
        "comparison",
        "gof_bucket",
        "source_id",
        "experiment_id",
        "member_id",
        "comp_source_id",
        "comp_member_id",
        "run_key",
        "cc_reuse_group",
    ]
    return out[keep_cols]


def _counts_from_sets(expected_keys: set[str], completed_keys: set[str]) -> dict[str, int]:
    useful = len(expected_keys & completed_keys)
    to_purge = len(completed_keys - expected_keys)
    missing = len(expected_keys - completed_keys)
    return {
        "i_completed_still_useful": useful,
        "ii_existing_to_be_purged": to_purge,
        "iii_missing": missing,
        "iv_total_necessary": useful + missing,
    }


def _evaluate_strict_for_option(
    option: str,
    expected_opt: pd.DataFrame,
    completed: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    expected_unique = expected_opt.drop_duplicates(subset=["run_key"])
    completed_unique = completed.drop_duplicates(subset=["run_key"])

    exp_keys = {
        tuple(v)
        for v in expected_unique[_GROUPING_COLUMNS].drop_duplicates().itertuples(index=False, name=None)
    }
    cmp_keys = {
        tuple(v)
        for v in completed_unique[_GROUPING_COLUMNS].drop_duplicates().itertuples(index=False, name=None)
    }

    for domain, gof_bucket, recipe in sorted(exp_keys | cmp_keys):
        e_mask = (
            (expected_unique["domain"] == domain)
            & (expected_unique["gof_bucket"] == gof_bucket)
            & (expected_unique["recipe"] == recipe)
        )
        c_mask = (
            (completed_unique["domain"] == domain)
            & (completed_unique["gof_bucket"] == gof_bucket)
            & (completed_unique["recipe"] == recipe)
        )
        e_keys_set = set(expected_unique.loc[e_mask, "run_key"].tolist())
        c_keys_set = set(completed_unique.loc[c_mask, "run_key"].tolist())
        counts = _counts_from_sets(e_keys_set, c_keys_set)
        rows.append(
            {
                "option": option,
                "accounting_mode": "strict",
                "domain": domain,
                "gof_bucket": gof_bucket,
                "recipe": recipe,
                **counts,
                "cc_reuse_shadow_count": 0,
            }
        )

    return pd.DataFrame(rows)


def _domain_for_cc_bucket(bucket: str) -> str | None:
    if bucket == "CC_SC":
        return "S(c)"
    if bucket == "CC_SO":
        return "S(o)"
    return None


def _cc_bucket_for_domain(domain: str) -> str:
    return "CC_SC" if domain == "S(c)" else "CC_SO"


def _evaluate_reuse_aware_for_option(
    option: str,
    expected_opt: pd.DataFrame,
    completed: pd.DataFrame,
    strict_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cc_rows: list[dict[str, Any]] = []

    # Non-CC rows are unchanged between strict and reuse-aware.
    non_cc = strict_rows[~strict_rows["gof_bucket"].isin(["CC_SC", "CC_SO"])].copy()
    non_cc["accounting_mode"] = "reuse_aware"
    rows.extend(non_cc.to_dict(orient="records"))

    exp_cc = expected_opt[expected_opt["comparison"] == "cc"].copy()
    cmp_cc = completed[completed["comparison"] == "cc"].copy()

    exp_cc = exp_cc.drop_duplicates(subset=["domain", "recipe", "cc_reuse_group"])
    cmp_cc = cmp_cc.drop_duplicates(subset=["domain", "recipe", "cc_reuse_group"])

    # Build reusable CC group diagnostics and owner assignments.
    recipes = sorted(set(exp_cc["recipe"]).union(set(cmp_cc["recipe"])))
    cc_domain_order = ["S(c)", "S(o)"]

    for recipe in recipes:
        exp_r = exp_cc[exp_cc["recipe"] == recipe]
        cmp_r = cmp_cc[cmp_cc["recipe"] == recipe]

        exp_domains_by_group = (
            exp_r.groupby("cc_reuse_group")["domain"].agg(lambda s: sorted(set(s))).to_dict()
            if not exp_r.empty
            else {}
        )
        cmp_domains_by_group = (
            cmp_r.groupby("cc_reuse_group")["domain"].agg(lambda s: sorted(set(s))).to_dict()
            if not cmp_r.empty
            else {}
        )

        exp_groups = set(exp_domains_by_group)
        cmp_groups = set(cmp_domains_by_group)

        group_owner: dict[str, str] = {}
        for group_key, domains in exp_domains_by_group.items():
            owner = "S(c)" if "S(c)" in domains else domains[0]
            group_owner[group_key] = owner

        for group_key in sorted(exp_groups | cmp_groups):
            exp_domains = exp_domains_by_group.get(group_key, [])
            cmp_domains = cmp_domains_by_group.get(group_key, [])
            owner = group_owner.get(group_key)
            if owner is None:
                owner = "S(c)" if "S(c)" in cmp_domains else (cmp_domains[0] if cmp_domains else "S(c)")
            completed_any = group_key in cmp_groups
            expected_any = group_key in exp_groups
            cc_rows.append(
                {
                    "option": option,
                    "recipe": recipe,
                    "cc_reuse_group": group_key,
                    "expected_domains": ",".join(exp_domains),
                    "completed_domains": ",".join(cmp_domains),
                    "shared_between_domains_expected": len(exp_domains) > 1,
                    "completed_any": bool(completed_any),
                    "owner_domain": owner,
                    "strict_required_count": len(exp_domains),
                    "reuse_required_count": 1 if expected_any else 0,
                    "strict_missing_count": len(exp_domains) if (expected_any and not completed_any) else 0,
                    "reuse_missing_count": 1 if (expected_any and not completed_any) else 0,
                }
            )

        # Accumulate reuse-aware i/ii/iii/iv for CC by owner domain.
        counts_by_domain = {
            domain: {
                "i_completed_still_useful": 0,
                "ii_existing_to_be_purged": 0,
                "iii_missing": 0,
                "iv_total_necessary": 0,
                "cc_reuse_shadow_count": 0,
            }
            for domain in cc_domain_order
        }

        for group_key, exp_domains in exp_domains_by_group.items():
            owner = group_owner[group_key]
            completed_any = group_key in cmp_groups
            cmp_domains = cmp_domains_by_group.get(group_key, [])
            if completed_any:
                counts_by_domain[owner]["i_completed_still_useful"] += 1
            else:
                counts_by_domain[owner]["iii_missing"] += 1
            counts_by_domain[owner]["iv_total_necessary"] += 1

            if len(cmp_domains) > 1:
                # Keep exactly one completed group instance globally; extras are stale.
                completed_owner = owner if owner in cmp_domains else ("S(c)" if "S(c)" in cmp_domains else cmp_domains[0])
                for dom in cmp_domains:
                    if dom != completed_owner:
                        counts_by_domain[dom]["ii_existing_to_be_purged"] += 1

            if len(exp_domains) > 1:
                for dom in exp_domains:
                    if dom != owner:
                        counts_by_domain[dom]["cc_reuse_shadow_count"] += 1

        # Existing CC runs that are not required by this option are purge candidates.
        for group_key, cmp_domains in cmp_domains_by_group.items():
            if group_key in exp_groups:
                continue
            for dom in cmp_domains:
                counts_by_domain[dom]["ii_existing_to_be_purged"] += 1

        for domain in cc_domain_order:
            # Keep rows only if this recipe touches CC in expected or completed space.
            has_any = bool(
                ((exp_r["domain"] == domain).any())
                or ((cmp_r["domain"] == domain).any())
                or counts_by_domain[domain]["cc_reuse_shadow_count"] > 0
            )
            if not has_any:
                continue
            rows.append(
                {
                    "option": option,
                    "accounting_mode": "reuse_aware",
                    "domain": domain,
                    "gof_bucket": _cc_bucket_for_domain(domain),
                    "recipe": recipe,
                    **counts_by_domain[domain],
                }
            )

    reuse_df = pd.DataFrame(rows)
    cc_groups_df = pd.DataFrame(cc_rows)
    return reuse_df, cc_groups_df


def _build_option_ranking(
    strict_df: pd.DataFrame,
    reuse_df: pd.DataFrame,
    cc_groups_df: pd.DataFrame,
) -> pd.DataFrame:
    options = sorted(set(strict_df["option"]).union(set(reuse_df["option"])))
    rows: list[dict[str, Any]] = []

    for option in options:
        strict_opt = strict_df[strict_df["option"] == option]
        reuse_opt = reuse_df[reuse_df["option"] == option]
        if "option" in cc_groups_df.columns:
            cc_opt = cc_groups_df[cc_groups_df["option"] == option]
        else:
            cc_opt = pd.DataFrame(columns=["shared_between_domains_expected", "completed_any"])

        strict_missing = int(strict_opt["iii_missing"].sum())
        reuse_missing = int(reuse_opt["iii_missing"].sum())
        strict_total = int(strict_opt["iv_total_necessary"].sum())
        reuse_total = int(reuse_opt["iv_total_necessary"].sum())

        cc_reused_completed = int(
            cc_opt[
                (cc_opt["shared_between_domains_expected"]) & (cc_opt["completed_any"])
            ].shape[0]
        )

        rows.append(
            {
                "option": option,
                "strict_missing_total": strict_missing,
                "reuse_aware_missing_total": reuse_missing,
                "strict_total_necessary": strict_total,
                "reuse_aware_total_necessary": reuse_total,
                "reuse_saved_missing_runs": strict_missing - reuse_missing,
                "cc_reused_completed_count": cc_reused_completed,
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    ranking = ranking.sort_values(
        by=[
            "reuse_aware_missing_total",
            "strict_missing_total",
            "cc_reused_completed_count",
        ],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking


def evaluate_pairing_options(
    expected_runs: pd.DataFrame,
    completed_runs: pd.DataFrame,
    *,
    option_column: str = "option",
) -> EvaluationOutputs:
    expected = _normalize_runs(expected_runs, require_option=True, option_column=option_column)
    completed = _normalize_runs(completed_runs, require_option=False, option_column=option_column)

    strict_frames: list[pd.DataFrame] = []
    reuse_frames: list[pd.DataFrame] = []
    cc_frames: list[pd.DataFrame] = []

    completed_unique = completed.drop_duplicates(subset=["run_key"])

    for option in sorted(expected[option_column].unique()):
        expected_opt = expected[expected[option_column] == option].copy()
        strict_opt = _evaluate_strict_for_option(option, expected_opt, completed_unique)
        reuse_opt, cc_opt = _evaluate_reuse_aware_for_option(
            option,
            expected_opt,
            completed_unique,
            strict_opt,
        )

        strict_frames.append(strict_opt)
        reuse_frames.append(reuse_opt)
        cc_frames.append(cc_opt)

    strict_df = (
        pd.concat(strict_frames, ignore_index=True)
        if strict_frames
        else pd.DataFrame(
            columns=[
                "option",
                "accounting_mode",
                "domain",
                "gof_bucket",
                "recipe",
                "i_completed_still_useful",
                "ii_existing_to_be_purged",
                "iii_missing",
                "iv_total_necessary",
                "cc_reuse_shadow_count",
            ]
        )
    )
    reuse_df = (
        pd.concat(reuse_frames, ignore_index=True)
        if reuse_frames
        else strict_df.copy()
    )
    cc_df = pd.concat(cc_frames, ignore_index=True) if cc_frames else pd.DataFrame()

    for df in (strict_df, reuse_df):
        if not df.empty:
            df.sort_values(
                by=["option", "domain", "gof_bucket", "recipe"],
                inplace=True,
                ignore_index=True,
            )

    if not cc_df.empty:
        cc_df.sort_values(
            by=["option", "recipe", "cc_reuse_group"],
            inplace=True,
            ignore_index=True,
        )

    ranking = _build_option_ranking(strict_df, reuse_df, cc_df)

    return EvaluationOutputs(
        strict_domain_counts=strict_df,
        reuse_aware_counts=reuse_df,
        cc_reuse_groups=cc_df,
        option_ranking=ranking,
    )


def write_evaluation_outputs(outputs: EvaluationOutputs, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "strict_domain_counts": out / "strict_domain_counts.csv",
        "reuse_aware_counts": out / "reuse_aware_counts.csv",
        "cc_reuse_groups": out / "cc_reuse_groups.csv",
        "option_ranking": out / "option_ranking.csv",
    }

    outputs.strict_domain_counts.to_csv(paths["strict_domain_counts"], index=False)
    outputs.reuse_aware_counts.to_csv(paths["reuse_aware_counts"], index=False)
    outputs.cc_reuse_groups.to_csv(paths["cc_reuse_groups"], index=False)
    outputs.option_ranking.to_csv(paths["option_ranking"], index=False)

    json_payload = {
        "strict_domain_counts": outputs.strict_domain_counts.to_dict(orient="records"),
        "reuse_aware_counts": outputs.reuse_aware_counts.to_dict(orient="records"),
        "cc_reuse_groups": outputs.cc_reuse_groups.to_dict(orient="records"),
        "option_ranking": outputs.option_ranking.to_dict(orient="records"),
    }
    json_path = out / "pairing_option_evaluation.json"
    json_path.write_text(json.dumps(json_payload, indent=2))
    paths["json"] = json_path

    return {k: str(v) for k, v in paths.items()}
