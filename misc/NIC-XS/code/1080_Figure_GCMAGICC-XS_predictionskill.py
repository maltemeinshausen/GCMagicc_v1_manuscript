# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Nicolai Meinshausen.
"""
1080_Figure_GCMAGICC-XS_predictionskill.py
==========================================

Release figure + metrics table for the **GCMagicc-XS** (reduced-predictor,
"forcing-only") extrapolation-skill evaluation.

Consumes the machine-readable results table produced by
``eval_nside2_nosmooth_miss585.py`` (``full_monthly_results.csv``) and, per
driving CMIP6 model and variable, draws observed vs predicted global-mean
timeseries with the 5-95% prediction band, and writes a per-(model,scenario,
variable) metrics table (mean, 5-95% interval width, R2, RMSE, bias, 5-95%
coverage). Held-out (test) scenarios -- the genuine extrapolation cases
(ssp585, abrupt-4xCO2) -- are drawn with solid emphasis; training scenarios are
muted.

This script does NO model inference and imports no local model modules: it is a
pure CSV -> figure/table transform, so it reproduces in a minimal
pandas/numpy/matplotlib environment.

Input schema (columns; * = required):
  model*, scenario*, is_test*, variable*, month_idx*, obs*, pred_mean*,
  pred_p05, pred_p10, pred_p50, pred_p90, pred_p95, n_draws
If pred_p05/pred_p95 are absent the script falls back to pred_p10/pred_p90 and
labels the band accordingly (older CSVs only carried 10/90).

Units: values are already in physical units (inverse affine transform applied
upstream in the eval script). No further transform is applied before metrics.

Outlier-naming rule (documented, deterministic): for each variable, a model is
flagged a "material outlier" if its RMSE exceeds
    median(RMSE over models) + K_IQR * IQR(RMSE over models),   K_IQR = 1.5
computed over the held-out (test) rows only. Flagged models are annotated on the
figure and marked in the metrics table column ``rmse_outlier``.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K_IQR = 1.5

VAR_LONG = {
    'psl': 'Sea-level pressure', 'tas': 'Near-surface air temp.',
    'pr': 'Precipitation', 'sfcWind': 'Near-surface wind speed',
    'ts': 'Surface temperature', 'tasmin': 'Daily-min air temp.',
    'tasmax': 'Daily-max air temp.', 'rsds': 'Downwelling shortwave',
    'hurs': 'Relative humidity', 'huss': 'Specific humidity',
}


def annual_mean(series, month_idx):
    """Collapse a monthly series to annual means keyed by month_idx//12."""
    yr = month_idx // 12
    df = pd.DataFrame({'yr': yr, 'v': series})
    g = df.groupby('yr')['v'].mean()
    return g.index.to_numpy(), g.to_numpy()


def metrics(obs, pred):
    obs = np.asarray(obs, float); pred = np.asarray(pred, float)
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    if obs.size < 2:
        return dict(rmse=np.nan, bias=np.nan, r2=np.nan, n=int(obs.size))
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((obs - obs.mean())**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return dict(rmse=rmse, bias=bias, r2=r2, n=int(obs.size))


def pick_band(df):
    """Return (lo_col, hi_col, label) preferring 5/95 over 10/90."""
    if 'pred_p05' in df.columns and df['pred_p05'].notna().any():
        return 'pred_p05', 'pred_p95', '5-95%'
    return 'pred_p10', 'pred_p90', '10-90% (legacy)'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True,
                    help='full_monthly_results.csv from eval_nside2_nosmooth_miss585.py')
    ap.add_argument('--outdir', default='../outputs',
                    help='directory for figures + metrics table')
    ap.add_argument('--variables', default='',
                    help='comma-separated subset of variables (default: all present)')
    ap.add_argument('--annual', action='store_true', default=True,
                    help='plot annual means (default) rather than monthly')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    lo_col, hi_col, band_label = pick_band(df)

    variables = ([v for v in args.variables.split(',') if v]
                 or sorted(df['variable'].unique()))

    # ---- metrics table (per model x scenario x variable) --------------------
    rows = []
    for (mname, scen, vn), sub in df.groupby(['model', 'scenario', 'variable']):
        sub = sub.sort_values('month_idx')
        mt = metrics(sub['obs'], sub['pred_mean'])
        lo = sub[lo_col].to_numpy(); hi = sub[hi_col].to_numpy()
        obs = sub['obs'].to_numpy()
        cover = float(np.mean((obs >= lo) & (obs <= hi))) if lo.size else np.nan
        rows.append(dict(
            model=mname, scenario=scen, variable=vn,
            is_test=bool(sub['is_test'].iloc[0]),
            obs_mean=float(np.nanmean(obs)),
            pred_mean=float(np.nanmean(sub['pred_mean'])),
            interval_width_mean=float(np.nanmean(hi - lo)) if lo.size else np.nan,
            band=band_label, coverage=cover, **mt))
    mt_df = pd.DataFrame(rows)

    # outlier flag per variable, over held-out rows
    mt_df['rmse_outlier'] = False
    for vn, sub in mt_df[mt_df['is_test']].groupby('variable'):
        r = sub['rmse'].to_numpy()
        r = r[np.isfinite(r)]
        if r.size >= 4:
            q1, q3 = np.percentile(r, [25, 75])
            thr = np.median(r) + K_IQR * (q3 - q1)
            mask = (mt_df['variable'] == vn) & mt_df['is_test'] & (mt_df['rmse'] > thr)
            mt_df.loc[mask, 'rmse_outlier'] = True

    mt_path = os.path.join(args.outdir, 'xs_prediction_skill_metrics.csv')
    mt_df.to_csv(mt_path, index=False)
    print(f"Saved {mt_path} ({len(mt_df)} rows); band={band_label}")

    # ---- per-variable figure: one panel per driving model -------------------
    for vn in variables:
        dv = df[df['variable'] == vn]
        if dv.empty:
            continue
        models = sorted(dv['model'].unique())
        ncol = 4
        nrow = int(np.ceil(len(models) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        outliers = set(mt_df[(mt_df['variable'] == vn) & mt_df['rmse_outlier']]['model'])
        for i, mname in enumerate(models):
            ax = axes[i // ncol][i % ncol]
            ax.axis('on')
            dm = dv[dv['model'] == mname]
            for scen, ds in dm.groupby('scenario'):
                ds = ds.sort_values('month_idx')
                is_test = bool(ds['is_test'].iloc[0])
                x, obs = annual_mean(ds['obs'].to_numpy(), ds['month_idx'].to_numpy())
                _, pm = annual_mean(ds['pred_mean'].to_numpy(), ds['month_idx'].to_numpy())
                _, lo = annual_mean(ds[lo_col].to_numpy(), ds['month_idx'].to_numpy())
                _, hi = annual_mean(ds[hi_col].to_numpy(), ds['month_idx'].to_numpy())
                a = 1.0 if is_test else 0.35
                lw = 1.6 if is_test else 0.9
                ax.fill_between(x, lo, hi, alpha=0.18 * a, color='C0', lw=0)
                ax.plot(x, obs, color='k', lw=lw, alpha=a)
                ax.plot(x, pm, color='C3' if is_test else 'C0', lw=lw, ls='-', alpha=a)
            title = mname + ('  *OUTLIER*' if mname in outliers else '')
            ax.set_title(title, fontsize=8,
                         color='red' if mname in outliers else 'black')
            ax.tick_params(labelsize=6)
        # legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='k', lw=1.6, label='Observed (driving ESM)'),
            Line2D([0], [0], color='C3', lw=1.6, label='GCMagicc-XS mean (held-out)'),
            Line2D([0], [0], color='C0', lw=1.0, label='GCMagicc-XS mean (training)'),
            Line2D([0], [0], color='C0', lw=6, alpha=0.25, label=f'{band_label} band'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f'GCMagicc-XS extrapolation skill - {VAR_LONG.get(vn, vn)} ({vn})\n'
                     f'annual global mean; solid = held-out scenarios (ssp585, abrupt-4xCO2)',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_pdf = os.path.join(args.outdir, f'Figure_GCMAGICC-XS_predictionskill_{vn}.pdf')
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_pdf}")


if __name__ == '__main__':
    main()
