#!/usr/bin/env python3
"""
Sample a distribution by adding residuals (from an SSP CSV) to ERA-hat values (from an ERA CSV),
then save a histogram PNG and report selected quantiles.

Expected CSV columns (case-sensitive after stripping whitespace):
  - data_ssp: trend_y_true, trend_y_hat
  - data_era: trend_yera_hat

You can configure defaults below or override via CLI flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# -----------------------------
# User-configurable defaults
# -----------------------------
maxlag = 6
depend = True
dir = "Ssmall"
# ["ssp119", "ssp126", "ssp370", "ssp434", "ssp460", "ssp534", "ssp585", "ssp534-over"]

scen = "ssp585"
data_ssp: str = "figchangeera" + dir + "_1/trends_ssp_" + str(depend) + "_" + str(maxlag) + ".csv"
data_era: str = "./figchangeforceera" + dir + "_1/trends_" + str(scen) + "_" + str(depend) + "_" + str(maxlag) + ".csv"
nsim: int = 2000
seed: int | None = 0
output_png: str = "hist_sampled_trends.png"
bins: int = 50


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_finite_array(series: pd.Series, name: str) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"No finite numeric values found for column '{name}'.")
    return arr


def simulate_distribution(
    data_ssp_path: str | Path,
    data_era_path: str | Path,
    nsim: int = 300,
    seed: int | None = 0,
) -> np.ndarray:
    """Return samples: randomly chosen trend_yera_hat + randomly chosen residual."""
    if nsim <= 0:
        raise ValueError("nsim must be a positive integer.")

    df_ssp = _clean_columns(pd.read_csv(data_ssp_path))
    df_era = _clean_columns(pd.read_csv(data_era_path))

    for col in ("trend_y_true", "trend_y_hat"):
        if col not in df_ssp.columns:
            raise KeyError(f"data_ssp missing required column '{col}'. Found: {list(df_ssp.columns)}")

    if "trend_yera_hat" not in df_era.columns:
        raise KeyError(
            f"data_era missing required column 'trend_yera_hat'. Found: {list(df_era.columns)}"
        )

    # Compute residuals on the rows where both columns are finite
    y_true_full = pd.to_numeric(df_ssp["trend_y_true"], errors="coerce").to_numpy(dtype=float)
    y_hat_full = pd.to_numeric(df_ssp["trend_y_hat"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true_full) & np.isfinite(y_hat_full)
    residuals = (y_true_full[mask] - y_hat_full[mask]).astype(float)
    if residuals.size == 0:
        raise ValueError("No finite residuals could be computed from data_ssp (trend_y_true - trend_y_hat).")

    era_vals = _to_finite_array(df_era["trend_yera_hat"], "trend_yera_hat")

    rng = np.random.default_rng(seed)

    base = rng.choice(era_vals, size=nsim, replace=True)
    res = rng.choice(residuals, size=nsim, replace=True)
    samples = base + res
    return samples


def compute_quantiles(samples: np.ndarray) -> dict[str, float]:
    qs = np.array([0.025, 0.05, 0.10, 0.90, 0.95, 0.975], dtype=float)
    vals = np.quantile(samples, qs)
    return {
        "2.5%": float(vals[0]),
        "5%": float(vals[1]),
        "10%": float(vals[2]),
        "90%": float(vals[3]),
        "95%": float(vals[4]),
        "97.5%": float(vals[5]),
    }


def save_histogram(samples: np.ndarray, out_png: str | Path, bins: int = 30) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(samples, bins=bins)
    plt.xlabel("trend_yera_hat + residual(trend_y_true - trend_y_hat)")
    plt.ylabel("Count")
    plt.title("Sampled distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample trend distribution by adding SSP residuals to ERA-hat values."
    )
    p.add_argument("--data_ssp", type=str, default=data_ssp, help="Path to SSP CSV file.")
    p.add_argument("--data_era", type=str, default=data_era, help="Path to ERA CSV file.")
    p.add_argument("--nsim", type=int, default=nsim, help="Number of simulations/samples.")
    p.add_argument("--seed", type=int, default=seed, help="Random seed (set to -1 for no seed).")
    p.add_argument("--output_png", type=str, default=output_png, help="Output histogram PNG path.")
    p.add_argument("--bins", type=int, default=bins, help="Number of histogram bins.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    seed_arg: int | None
    if args.seed == -1:
        seed_arg = None
    else:
        seed_arg = int(args.seed)

    try:
        samples = simulate_distribution(
            data_ssp_path=args.data_ssp,
            data_era_path=args.data_era,
            nsim=args.nsim,
            seed=seed_arg,
        )
        q = compute_quantiles(samples)
        save_histogram(samples, args.output_png, bins=args.bins)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Print quantiles in requested set
    print("Quantiles for sampled distribution:")
    for k in ["2.5%", "5%", "10%", "90%", "95%", "97.5%"]:
        print(f"  {k:>5} : {q[k]:.6f}")
    print(f"Histogram saved to: {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
