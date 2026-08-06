"""
Evaluate the nside=2 NOSMOOTH model (no tas_smoothed/rtmt_smoothed predictors).
Same plots as eval_nside2.py but for the nosmooth variant.
"""
import argparse, os, glob, re, pickle
SCRATCH_BASE = os.environ.get('SCRATCH_BASE', '/scratch2')
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from help_functions import Downsample, regular_grid_to_healpix
from models import LB2, L4de

_parser = argparse.ArgumentParser()
_parser.add_argument('--tag', default='', help='suffix for output dir, e.g. _tmp')
_parser.add_argument('--max-files', type=int, default=1,
                     help='max ensemble members per model/scenario (default 1)')
_parser.add_argument('--n-draws', type=int, default=200,
                     help='independent stochastic draws per file for quantile estimation '
                          '(default 200; release figure uses 200 so that p05/p95 are stable)')
_parser.add_argument('--n-draws-ar', type=int, default=5,
                     help='autoregressive draws per file (diagnostic-only, ~1000x costlier; '
                          'default 5). The release figure does not use these.')
_parser.add_argument('--seed', type=int, default=0,
                     help='global RNG seed for reproducible stochastic draws (default 0)')
_parser.add_argument('--model-base', default=os.environ.get('MODEL_BASE', ''),
                     help='override checkpoint base prefix ("..._nosmooth"); '
                          'for the release bundle point this at ../checkpoints/'
                          'modelsNfour_7Augext_miss585_nosmooth')
_parser.add_argument('--data-dir', default=os.environ.get('DATA_DIR', ''),
                     help='override normalized input data dir (contains data_DAT_*.h5); '
                          'default derives from SCRATCH_BASE')
_parser.add_argument('--save-member-draws', action='store_true', default=True,
                     help='save global-mean member draws for held-out scenarios to member_draws.npz')
_args = _parser.parse_args()

# ─── Configuration ──────────────────────────────────────────────────────────
DATE = '7Augext'
SUFFIX = '_miss585'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside_hi = 2
npix = 12 * nside_hi**2   # 48
n_features = 10

# X columns: drop tas_smoothed (4) and rtmt_smoothed (5)
X_COLS = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
x_features = len(X_COLS)  # 13

variab = 10
maxlag = 72
lags = [1,2,3,4,5,6,8,10,12,14,16,18,20,24,27,30,33,36,42,48,54,60,66,72]

normalized_dir = _args.data_dir or (SCRATCH_BASE + '/userdata/nicolai/cmip6/normalized_' + DATE + '/')
model_base = _args.model_base or (SCRATCH_BASE + '/userdata/nicolai/cmip6/modelsNfour_' + DATE + SUFFIX + '_nosmooth')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots_eval_nside2_nosmooth_miss585' + _args.tag)
os.makedirs(plot_dir, exist_ok=True)

var_names = ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"]

# meta lives in ../config/ in the release bundle; fall back to the script dir
# (original layout) or a META_PATH override.
def _find_meta():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get('META_PATH', ''),
             os.path.join(here, '..', 'config', 'meta_' + DATE + '.pkl'),
             os.path.join(here, 'meta_' + DATE + '.pkl')]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError('meta_' + DATE + '.pkl not found in ../config or script dir')

with open(_find_meta(), 'rb') as f:
    meta = pickle.load(f)
transformation_scalars = meta['transformation_scalars']

# ─── Load models ────────────────────────────────────────────────────────────
def load_model():
    bias = LB2(0, nside_hi, n_features=n_features).to(device)
    bias.load_state_dict(torch.load(
        os.path.join(model_base + f'_bS{nside_hi}', 'best_model.pt'),
        map_location=device))
    bias.eval()

    effect = L4de(nside_hi, n_features=n_features, x_features=x_features,
                  variab=variab, lags=lags, add_latent_dim=50).to(device)
    effect.load_state_dict(torch.load(
        os.path.join(model_base + f'_bdeSxlsp{nside_hi}_{maxlag}_{variab}',
                     'best_model.pt'),
        map_location=device))
    effect.eval()
    return bias, effect

# ─── Inverse transform to physical units ────────────────────────────────────
def inv_transform(y, var_idx):
    v = var_names[var_idx]
    if v in transformation_scalars:
        pre = transformation_scalars[v]['pre']
        post = transformation_scalars[v]['post']
        return (y - post) / pre
    return y

# ─── Downsample y64 to y2 ──────────────────────────────────────────────────
downs = Downsample()

def downsample_to_nside2(y64):
    y = torch.from_numpy(y64).float()
    for _ in range(5):  # 64->32->16->8->4->2
        y = downs(y)
    return y

# ─── Find files for specific model+scenario ─────────────────────────────────
def find_files(model_name, scenario, max_files=3):
    pattern = f'data_DAT_{model_name}_{scenario}_*.h5'
    files = sorted(glob.glob(os.path.join(normalized_dir, pattern)))
    return files[:max_files]

# ─── Generate predictions for one file ──────────────────────────────────────
@torch.no_grad()
def predict_file(bias_model, effect_model, fpath, n_draws=10, n_draws_ar=None):
    # The release figure + full_monthly_results.csv use only the INDEPENDENT draws
    # ('draws'). The autoregressive draws ('draws_ar') feed diagnostic-only plots and
    # are ~1000x more expensive (sequential rollout over all T months), so they are
    # generated at a small count by default (n_draws_ar) to keep runtime tractable.
    if n_draws_ar is None:
        n_draws_ar = n_draws
    with h5py.File(fpath, 'r') as h5:
        y64 = h5['y64'][:]
        X = h5['X'][:]

    y2_obs = downsample_to_nside2(y64)           # (T, F, 48)
    x_tensor = torch.from_numpy(X[:, X_COLS]).float().to(device)
    T = x_tensor.shape[0]

    # bias prediction
    y_bias = bias_model(x=x_tensor, y_low=None).detach()  # (T, F, 48)

    # stochastic draws (independent mode, no lags) — used for the release quantiles
    draws = []
    for _ in range(n_draws):
        y_pred = effect_model(x=x_tensor, y_in=y_bias, dependence=False)
        draws.append(y_pred.detach())

    # auto-regressive draws (diagnostic only; small count)
    draws_ar = []
    for _ in range(max(1, n_draws_ar)):
        y_pred_ar = effect_model(x=x_tensor, y_in=y_bias, dependence=True)
        draws_ar.append(y_pred_ar.detach())

    return {
        'obs': y2_obs,               # (T, F, 48)
        'bias': y_bias.cpu(),        # (T, F, 48)
        'draws': torch.stack(draws).cpu(), # (n_draws, T, F, 48)
        'draws_ar': torch.stack(draws_ar).cpu(),
        'X': X,                      # full X for forcing plots (use original cols)
        'T': T
    }

# ─── Global mean (uniform weighting on HEALPix) ────────────────────────────
def global_mean(y):
    return y.mean(dim=-1)  # (..., F)

# ─── Scenarios to evaluate ──────────────────────────────────────────────────
test_scenarios = {
    'ssp585': {'color': 'red', 'ls': '-'},
    'abrupt-4xCO2': {'color': 'darkred', 'ls': '--'},
}
train_scenarios = {
    'historical': {'color': 'gray', 'ls': '-'},
    'ssp119': {'color': 'darkblue', 'ls': '-'},
    'ssp126': {'color': 'blue', 'ls': '-'},
    'ssp370': {'color': 'purple', 'ls': '-'},
    'ssp434': {'color': 'orange', 'ls': '-'},
    'ssp460': {'color': 'goldenrod', 'ls': '-'},
    'hist-GHG': {'color': 'olive', 'ls': '--'},
    'hist-aer': {'color': 'cyan', 'ls': '--'},
    'hist-nat': {'color': 'lime', 'ls': '--'},
    'abrupt-2xCO2': {'color': 'salmon', 'ls': '--'},
}

# representative models (one from each major family)
eval_models = [
    'ACCESS-CM2', 'ACCESS-ESM1-5',
    'CanESM5', 'CanESM5-1', 'CanESM5-CanOE',
    'CESM2', 'CESM2-WACCM',
    'EC-Earth3', 'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR',
    'FGOALS-g3',
    'GFDL-CM4', 'GFDL-ESM4',
    'GISS-E2-1-G', 'GISS-E2-1-H', 'GISS-E2-2-G',
    'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM',
    'INM-CM4-8', 'INM-CM5-0',
    'IPSL-CM6A-LR',
    'MIROC-ES2L', 'MIROC6',
    'UKESM1-0-LL',
]

def main():
    # ─── Reproducible RNG (draws depend on torch/numpy/random state) ──────────
    import random
    random.seed(_args.seed)
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_args.seed)
    print(f"Config: n_draws={_args.n_draws} seed={_args.seed} device={device}")
    print(f"  model_base={model_base}")
    print(f"  normalized_dir={normalized_dir}")

    print("Loading nosmooth models...")
    bias_model, effect_model = load_model()

    all_results = {}

    for scenarios, label in [(test_scenarios, 'TEST'), (train_scenarios, 'TRAIN')]:
        for scenario, sty in scenarios.items():
            for model_name in eval_models:
                files = find_files(model_name, scenario, max_files=_args.max_files)
                if not files:
                    continue
                for fi, fpath in enumerate(files):
                    key = (scenario, model_name, fi)
                    print(f"[{label}] {model_name} / {scenario} [{fi+1}/{len(files)}] ({os.path.basename(fpath)})")
                    try:
                        result = predict_file(bias_model, effect_model, fpath,
                                              n_draws=_args.n_draws, n_draws_ar=_args.n_draws_ar)
                        result['scenario'] = scenario
                        result['model_name'] = model_name
                        result['is_test'] = (label == 'TEST')
                        all_results[key] = result
                    except Exception as e:
                        print(f"  Error: {e}")

    if not all_results:
        print("No results to plot.")
        return

    # ─── Export full CSV (all vars, all scenarios, monthly) ──────────
    # Emit member-level draws + 5/50/95 (and legacy 10/90) percentiles in physical
    # units. Global mean over HEALPix pixels is applied first; percentiles are then
    # taken over the n_draws stochastic realisations. member_draws.npz stores the raw
    # global-mean member draws (held-out scenarios only) so any quantile is
    # recomputable offline without re-running the GPU.
    import pandas as pd
    csv_rows_full = []
    member_store = {}          # "scen|model|var" -> (n_draws, T) physical global-mean draws
    for (scen, mname, *_), res in sorted(all_results.items()):
        is_test = res['is_test']
        T = res['T']
        for vi, vn in enumerate(var_names):
            obs_monthly = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
            draws_gm = global_mean(res['draws'])[:, :, vi].numpy()      # (n_draws, T) normalized
            draws_phys = inv_transform(draws_gm, vi)                     # (n_draws, T) physical
            pred_monthly = draws_phys.mean(axis=0)
            p05 = np.percentile(draws_phys,  5, axis=0)
            p10 = np.percentile(draws_phys, 10, axis=0)
            p50 = np.percentile(draws_phys, 50, axis=0)
            p90 = np.percentile(draws_phys, 90, axis=0)
            p95 = np.percentile(draws_phys, 95, axis=0)
            if _args.save_member_draws and is_test:
                member_store[f"{scen}|{mname}|{vn}"] = draws_phys.astype(np.float32)
            for t in range(T):
                csv_rows_full.append({
                    'model': mname, 'scenario': scen, 'is_test': is_test,
                    'variable': vn, 'month_idx': t,
                    'obs': float(obs_monthly[t]),
                    'pred_mean': float(pred_monthly[t]),
                    'pred_p05': float(p05[t]),
                    'pred_p10': float(p10[t]),
                    'pred_p50': float(p50[t]),
                    'pred_p90': float(p90[t]),
                    'pred_p95': float(p95[t]),
                    'n_draws': int(_args.n_draws),
                })
    df_full = pd.DataFrame(csv_rows_full)
    full_csv_path = os.path.join(plot_dir, 'full_monthly_results.csv')
    df_full.to_csv(full_csv_path, index=False)
    print(f"Saved {full_csv_path} ({len(df_full)} rows)")

    if _args.save_member_draws and member_store:
        npz_path = os.path.join(plot_dir, 'member_draws.npz')
        np.savez_compressed(npz_path, **member_store)
        print(f"Saved {npz_path} ({len(member_store)} series, n_draws={_args.n_draws})")

    # ─── PLOT 1: Global mean timeseries per variable ──────────────────
    for var_idx, var_name in enumerate(var_names):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

        for ax, (scenarios, title) in zip(axes, [
            (test_scenarios, f'Test scenarios (held out) — {var_name}'),
            (train_scenarios, f'Training scenarios — {var_name}')
        ]):
            for (scen, mname, *_), res in sorted(all_results.items()):
                if scen not in scenarios:
                    continue
                sty = scenarios[scen]
                obs_gm = global_mean(res['obs'])[:, var_idx].numpy()
                obs_phys = inv_transform(obs_gm, var_idx)

                draws_gm = global_mean(res['draws'])[:, :, var_idx].numpy()  # (n_draws, T)
                draws_phys = inv_transform(draws_gm, var_idx)
                pred_mean = draws_phys.mean(axis=0)
                pred_lo = np.percentile(draws_phys, 10, axis=0)
                pred_hi = np.percentile(draws_phys, 90, axis=0)

                t = np.arange(len(obs_phys))
                ax.plot(t, obs_phys, color=sty['color'], ls=sty['ls'],
                        alpha=0.8, lw=1.5, label=f'{mname} {scen} (obs)')
                ax.plot(t, pred_mean, color=sty['color'], ls=':', alpha=0.8, lw=1.5)
                ax.fill_between(t, pred_lo, pred_hi, color=sty['color'], alpha=0.12)

            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Month index')
            ax.set_ylabel(f'{var_name} (physical units)')
            ax.legend(fontsize=7, ncol=2, loc='upper left')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'[NOSMOOTH] Global mean {var_name} at nside=2: observed (solid) vs predicted (dotted, 10-90% band)',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'timeseries_{var_name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved timeseries_{var_name}.png")

    # ─── PLOT 2: Obs vs Pred scatter (annual means) ─────────────────────
    for var_idx, var_name in enumerate(var_names):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, (scenarios, title) in zip(axes, [
            (test_scenarios, f'Test scenarios — {var_name}'),
            (train_scenarios, f'Train scenarios — {var_name}')
        ]):
            for (scen, mname, *_), res in sorted(all_results.items()):
                if scen not in scenarios:
                    continue
                sty = scenarios[scen]
                obs_gm = inv_transform(global_mean(res['obs'])[:, var_idx].numpy(), var_idx)
                pred_gm = inv_transform(global_mean(res['draws'].mean(dim=0))[:, var_idx].numpy(), var_idx)
                n_years = len(obs_gm) // 12
                if n_years < 1:
                    continue
                obs_ann = obs_gm[:n_years*12].reshape(n_years, 12).mean(axis=1)
                pred_ann = pred_gm[:n_years*12].reshape(n_years, 12).mean(axis=1)
                ax.scatter(obs_ann, pred_ann, c=sty['color'], s=15, alpha=0.6,
                          label=f'{mname} {scen}')

            lims = ax.get_xlim()
            ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f'Observed annual mean {var_name}')
            ax.set_ylabel(f'Predicted annual mean {var_name}')
            ax.set_title(title)
            ax.legend(fontsize=6, ncol=2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        fig.suptitle('[NOSMOOTH]', fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'scatter_{var_name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved scatter_{var_name}.png")

    # ─── PLOTS 3/3b/3c/3d: Forcing-Response and Obs-vs-Pred for TAS & PR ──
    all_scenarios = {**test_scenarios, **train_scenarios}
    skip_abrupt = {'abrupt-4xCO2'}
    plot_vars = [(1, 'tas', 'TAS', 'K'), (2, 'pr', 'PR', 'kg/m²/s'),
                 (3, 'sfcWind', 'sfcWind', 'm/s'), (5, 'tasmin', 'Tasmin', 'K'),
                 (6, 'tasmax', 'Tasmax', 'K'), (9, 'huss', 'HUSS', 'kg/kg')]

    for vi, vshort, vlong, vunit in plot_vars:
        csv_rows = []
        for (scen, mname, *_), res in sorted(all_results.items()):
            if scen not in all_scenarios or scen in skip_abrupt:
                continue
            is_test = res['is_test']
            obs_v = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
            pred_v = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
            co2_erf = res['X'][:, -1]
            n_years = min(len(obs_v) // 12, len(co2_erf) // 12)
            if n_years < 1:
                continue
            obs_ann = obs_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            pred_ann = pred_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            co2_ann = co2_erf[:n_years*12].reshape(n_years, 12).mean(axis=1)
            for yr in range(n_years):
                csv_rows.append({
                    'model': mname, 'scenario': scen, 'is_test': is_test,
                    'year_idx': yr, 'co2_erf': float(co2_ann[yr]),
                    f'obs_{vshort}': float(obs_ann[yr]),
                    f'pred_{vshort}': float(pred_ann[yr]),
                })
        import pandas as pd
        df_fr = pd.DataFrame(csv_rows)
        csv_path = os.path.join(plot_dir, f'forcing_response_{vshort}_data.csv')
        df_fr.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

        fig, ax = plt.subplots(figsize=(10, 6))
        for (scen, mname, *_), res in sorted(all_results.items()):
            if scen not in all_scenarios or scen in skip_abrupt:
                continue
            sty = all_scenarios[scen]
            is_test = res['is_test']
            obs_v = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
            pred_v = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
            co2_erf = res['X'][:, -1]
            n_years = min(len(obs_v) // 12, len(co2_erf) // 12)
            if n_years < 1:
                continue
            obs_ann = obs_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            pred_ann = pred_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            co2_ann = co2_erf[:n_years*12].reshape(n_years, 12).mean(axis=1)
            marker = 's' if is_test else 'o'
            edge = 'black' if is_test else 'none'
            ax.scatter(co2_ann, obs_ann, c=sty['color'], marker=marker,
                      edgecolors=edge, s=20, alpha=0.5, linewidths=0.5)
            ax.scatter(co2_ann, pred_ann, c=sty['color'], marker=marker,
                      edgecolors=edge, s=20, alpha=0.5, linewidths=0.5,
                      facecolors='none')
        handles = []
        for scen, sty in all_scenarios.items():
            if scen in skip_abrupt:
                continue
            is_test = scen in test_scenarios
            handles.append(Line2D([0],[0], marker='s' if is_test else 'o',
                                 color='w', markerfacecolor=sty['color'],
                                 markeredgecolor='black' if is_test else sty['color'],
                                 markersize=8, label=scen))
        handles.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                             markersize=8, label='Observed (filled)'))
        handles.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
                             markeredgecolor='gray', markersize=8, label='Predicted (hollow)'))
        ax.legend(handles=handles, fontsize=8, ncol=2)
        ax.set_xlabel('CO2 ERF (annual mean)', fontsize=12)
        ax.set_ylabel(f'Global mean {vlong} ({vunit})', fontsize=12)
        ax.set_title(f'[NOSMOOTH] Forcing-Response: extrapolation? ({vlong})', fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'forcing_response_{vshort}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved forcing_response_{vshort}.png")

        for model_name in eval_models:
            model_data = {k: v for k, v in all_results.items() if k[1] == model_name and k[0] not in skip_abrupt}
            if not model_data:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for (scen, mn, *_), res in sorted(model_data.items()):
                if scen not in all_scenarios:
                    continue
                sty = all_scenarios[scen]
                is_test = res['is_test']
                obs_v = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
                pred_v = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
                co2_erf = res['X'][:, -1]
                n_years = min(len(obs_v) // 12, len(co2_erf) // 12)
                if n_years < 1:
                    continue
                obs_ann = obs_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
                pred_ann = pred_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
                co2_ann = co2_erf[:n_years*12].reshape(n_years, 12).mean(axis=1)
                marker = 's' if is_test else 'o'
                edge = 'black' if is_test else 'none'
                ax.scatter(co2_ann, obs_ann, c=sty['color'], marker=marker,
                          edgecolors=edge, s=30, alpha=0.6, linewidths=0.5)
                ax.scatter(co2_ann, pred_ann, c=sty['color'], marker=marker,
                          edgecolors=edge, s=30, alpha=0.6, linewidths=0.5,
                          facecolors='none')
            handles = []
            for scen, sty in all_scenarios.items():
                if scen in skip_abrupt:
                    continue
                is_test = scen in test_scenarios
                handles.append(Line2D([0],[0], marker='s' if is_test else 'o',
                                     color='w', markerfacecolor=sty['color'],
                                     markeredgecolor='black' if is_test else sty['color'],
                                     markersize=8, label=scen))
            handles.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                                 markersize=8, label='Observed (filled)'))
            handles.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
                                 markeredgecolor='gray', markersize=8, label='Predicted (hollow)'))
            ax.legend(handles=handles, fontsize=7, ncol=2)
            ax.set_xlabel('CO2 ERF (annual mean)', fontsize=11)
            ax.set_ylabel(f'Global mean {vlong} ({vunit})', fontsize=11)
            ax.set_title(f'[NOSMOOTH] Forcing-Response {vlong}: {model_name}', fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'forcing_response_{vshort}_{model_name}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f"Saved per-model forcing_response_{vshort} plots")

        fig, ax = plt.subplots(figsize=(8, 8))
        all_obs_vals, all_pred_vals = [], []
        for (scen, mname, *_), res in sorted(all_results.items()):
            if scen not in all_scenarios or scen in skip_abrupt:
                continue
            sty = all_scenarios[scen]
            is_test = res['is_test']
            obs_v = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
            pred_v = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
            n_years = len(obs_v) // 12
            if n_years < 1:
                continue
            obs_ann = obs_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            pred_ann = pred_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            all_obs_vals.extend(obs_ann)
            all_pred_vals.extend(pred_ann)
            marker = 's' if is_test else 'o'
            edge = 'black' if is_test else 'none'
            ax.scatter(obs_ann, pred_ann, c=sty['color'], marker=marker,
                      edgecolors=edge, s=20, alpha=0.5, linewidths=0.5)
        if all_obs_vals:
            vmin = min(min(all_obs_vals), min(all_pred_vals))
            vmax = max(max(all_obs_vals), max(all_pred_vals))
            pad = (vmax - vmin) * 0.05
            ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], 'k--', lw=0.8, alpha=0.5)
            ax.set_xlim(vmin-pad, vmax+pad)
            ax.set_ylim(vmin-pad, vmax+pad)
            ax.set_aspect('equal')
        handles = []
        for scen, sty in all_scenarios.items():
            if scen in skip_abrupt:
                continue
            is_test = scen in test_scenarios
            handles.append(Line2D([0],[0], marker='s' if is_test else 'o',
                                 color='w', markerfacecolor=sty['color'],
                                 markeredgecolor='black' if is_test else sty['color'],
                                 markersize=8, label=scen))
        ax.legend(handles=handles, fontsize=8, ncol=2)
        ax.set_xlabel(f'Observed global mean {vlong} ({vunit})', fontsize=12)
        ax.set_ylabel(f'Predicted global mean {vlong} ({vunit})', fontsize=12)
        ax.set_title(f'[NOSMOOTH] Obs vs Pred global {vlong} (annual mean)', fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'obs_vs_pred_{vshort}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved obs_vs_pred_{vshort}.png")

        for model_name in eval_models:
            model_data = {k: v for k, v in all_results.items() if k[1] == model_name and k[0] not in skip_abrupt}
            if not model_data:
                continue
            fig, ax = plt.subplots(figsize=(7, 7))
            m_obs, m_pred = [], []
            for (scen, mn, *_), res in sorted(model_data.items()):
                if scen not in all_scenarios:
                    continue
                sty = all_scenarios[scen]
                is_test = res['is_test']
                obs_v = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
                pred_v = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
                n_years = len(obs_v) // 12
                if n_years < 1:
                    continue
                obs_ann = obs_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
                pred_ann = pred_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
                m_obs.extend(obs_ann)
                m_pred.extend(pred_ann)
                marker = 's' if is_test else 'o'
                edge = 'black' if is_test else 'none'
                ax.scatter(obs_ann, pred_ann, c=sty['color'], marker=marker,
                          edgecolors=edge, s=30, alpha=0.6, linewidths=0.5)
            if m_obs:
                vmin = min(min(m_obs), min(m_pred))
                vmax = max(max(m_obs), max(m_pred))
                pad = (vmax - vmin) * 0.05
                ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], 'k--', lw=0.8, alpha=0.5)
                ax.set_xlim(vmin-pad, vmax+pad)
                ax.set_ylim(vmin-pad, vmax+pad)
                ax.set_aspect('equal')
            handles = []
            for scen, sty in all_scenarios.items():
                if scen in skip_abrupt:
                    continue
                is_test = scen in test_scenarios
                handles.append(Line2D([0],[0], marker='s' if is_test else 'o',
                                     color='w', markerfacecolor=sty['color'],
                                     markeredgecolor='black' if is_test else sty['color'],
                                     markersize=8, label=scen))
            ax.legend(handles=handles, fontsize=7, ncol=2)
            ax.set_xlabel(f'Observed global mean {vlong} ({vunit})', fontsize=11)
            ax.set_ylabel(f'Predicted global mean {vlong} ({vunit})', fontsize=11)
            ax.set_title(f'[NOSMOOTH] Obs vs Pred {vlong}: {model_name}', fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'obs_vs_pred_{vshort}_{model_name}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f"Saved per-model obs_vs_pred_{vshort} plots")

        # ── PLOT 3e: Joint train/test side-by-side Obs vs Pred ───────────
        fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(14, 7))
        all_vals = []
        plot_data_split = {'train': [], 'test': []}
        for (scen, mname, *_), res in sorted(all_results.items()):
            if scen not in all_scenarios or scen in skip_abrupt:
                continue
            sty = all_scenarios[scen]
            is_test = res['is_test']
            obs_v = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
            pred_v = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
            n_years = len(obs_v) // 12
            if n_years < 1:
                continue
            obs_ann = obs_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            pred_ann = pred_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
            all_vals.extend(obs_ann)
            all_vals.extend(pred_ann)
            key = 'test' if is_test else 'train'
            plot_data_split[key].append((scen, sty, obs_ann, pred_ann))
        if all_vals:
            vmin = min(all_vals)
            vmax = max(all_vals)
            pad = (vmax - vmin) * 0.05
            for ax, label, data in [(ax_tr, 'Train scenarios', plot_data_split['train']),
                                     (ax_te, 'Test scenarios', plot_data_split['test'])]:
                for scen, sty, obs_ann, pred_ann in data:
                    ax.scatter(obs_ann, pred_ann, c=sty['color'], s=20, alpha=0.5,
                              edgecolors='none', linewidths=0.5)
                ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], 'k--', lw=0.8, alpha=0.5)
                ax.set_xlim(vmin-pad, vmax+pad)
                ax.set_ylim(vmin-pad, vmax+pad)
                ax.set_aspect('equal')
                ax.set_xlabel(f'Observed global mean {vlong} ({vunit})', fontsize=11)
                ax.set_ylabel(f'Predicted global mean {vlong} ({vunit})', fontsize=11)
                ax.set_title(label, fontsize=12)
                ax.grid(True, alpha=0.3)
                handles = []
                relevant = train_scenarios if label.startswith('Train') else test_scenarios
                for scen, sty in relevant.items():
                    if scen in skip_abrupt:
                        continue
                    handles.append(Line2D([0],[0], marker='o', color='w',
                                         markerfacecolor=sty['color'], markersize=8, label=scen))
                ax.legend(handles=handles, fontsize=7, ncol=1)
        fig.suptitle(f'[NOSMOOTH] Obs vs Pred global {vlong}: Train (left) vs Test (right)', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'obs_vs_pred_{vshort}_split.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved obs_vs_pred_{vshort}_split.png")

        # ── PLOT 3f: Per-model train/test side-by-side Obs vs Pred ───────
        for model_name in eval_models:
            model_data = {k: v for k, v in all_results.items() if k[1] == model_name and k[0] not in skip_abrupt}
            if not model_data:
                continue
            fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(12, 6))
            all_vals = []
            split = {'train': [], 'test': []}
            for (scen, mn, *_), res in sorted(model_data.items()):
                if scen not in all_scenarios:
                    continue
                sty = all_scenarios[scen]
                is_test = res['is_test']
                obs_v = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
                pred_v = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
                n_years = len(obs_v) // 12
                if n_years < 1:
                    continue
                obs_ann = obs_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
                pred_ann = pred_v[:n_years*12].reshape(n_years, 12).mean(axis=1)
                all_vals.extend(obs_ann)
                all_vals.extend(pred_ann)
                key = 'test' if is_test else 'train'
                split[key].append((scen, sty, obs_ann, pred_ann))
            if all_vals:
                vmin = min(all_vals)
                vmax = max(all_vals)
                pad = (vmax - vmin) * 0.05
                for ax, label, data in [(ax_tr, 'Train', split['train']),
                                         (ax_te, 'Test', split['test'])]:
                    for scen, sty, obs_ann, pred_ann in data:
                        ax.scatter(obs_ann, pred_ann, c=sty['color'], s=25, alpha=0.6,
                                  edgecolors='none', linewidths=0.5)
                    ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], 'k--', lw=0.8, alpha=0.5)
                    ax.set_xlim(vmin-pad, vmax+pad)
                    ax.set_ylim(vmin-pad, vmax+pad)
                    ax.set_aspect('equal')
                    ax.set_xlabel(f'Observed {vlong} ({vunit})', fontsize=10)
                    ax.set_ylabel(f'Predicted {vlong} ({vunit})', fontsize=10)
                    ax.set_title(label, fontsize=11)
                    ax.grid(True, alpha=0.3)
                    handles = []
                    relevant = train_scenarios if label == 'Train' else test_scenarios
                    for scen, sty in relevant.items():
                        if scen in skip_abrupt:
                            continue
                        handles.append(Line2D([0],[0], marker='o', color='w',
                                             markerfacecolor=sty['color'], markersize=7, label=scen))
                    ax.legend(handles=handles, fontsize=7, ncol=1)
            fig.suptitle(f'[NOSMOOTH] Obs vs Pred {vlong}: {model_name}', fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'obs_vs_pred_{vshort}_split_{model_name}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f"Saved per-model obs_vs_pred_{vshort}_split plots")

    # ─── PLOT 4: Independent vs auto-regressive (temporal structure) ────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    key_to_plot = None
    for key in all_results:
        if key[0] == 'ssp585':
            key_to_plot = key
            break
    if key_to_plot is None:
        for key in all_results:
            if key[0] in test_scenarios:
                key_to_plot = key
                break

    if key_to_plot is not None:
        res = all_results[key_to_plot]
        var_idx = 1  # tas
        T = res['T']
        t = np.arange(T)
        obs = inv_transform(global_mean(res['obs'])[:, var_idx].numpy(), var_idx)

        for ax, (draws_key, title) in zip(axes, [
            ('draws', f'Independent draws — {key_to_plot[1]} {key_to_plot[0]}'),
            ('draws_ar', f'Auto-regressive draws — {key_to_plot[1]} {key_to_plot[0]}')
        ]):
            draws = inv_transform(global_mean(res[draws_key])[:, :, var_idx].numpy(), var_idx)
            ax.plot(t, obs, 'k-', lw=2, label='Observed')
            for i in range(min(5, draws.shape[0])):
                ax.plot(t, draws[i], alpha=0.4, lw=0.8)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('TAS (K)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Month index')
        fig.suptitle('[NOSMOOTH] Global mean TAS: independent vs auto-regressive stochastic draws',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'independent_vs_ar.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved independent_vs_ar.png")

    # ─── PLOT 5: RMSE summary per scenario per variable ─────────────────
    scenario_rmse = {}
    for (scen, mname, *_), res in all_results.items():
        if scen not in scenario_rmse:
            scenario_rmse[scen] = {v: [] for v in var_names}
        for vi, vn in enumerate(var_names):
            obs = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
            pred = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
            rmse = np.sqrt(np.mean((obs - pred)**2))
            scenario_rmse[scen][vn].append(rmse)

    fig, ax = plt.subplots(figsize=(12, 5))
    scenarios_ordered = sorted(scenario_rmse.keys(),
                               key=lambda s: s in test_scenarios, reverse=True)
    x_pos = np.arange(len(var_names))
    width = 0.8 / max(len(scenarios_ordered), 1)

    for i, scen in enumerate(scenarios_ordered):
        is_test = scen in test_scenarios
        vals = [np.mean(scenario_rmse[scen][v]) if scenario_rmse[scen][v] else 0
                for v in var_names]
        bars = ax.bar(x_pos + i * width, vals, width * 0.9,
                     label=scen, alpha=0.8,
                     edgecolor='black' if is_test else 'none',
                     linewidth=1.5 if is_test else 0)

    ax.set_xticks(x_pos + width * len(scenarios_ordered) / 2)
    ax.set_xticklabels(var_names, rotation=45)
    ax.set_ylabel('RMSE (physical units)')
    ax.set_title('[NOSMOOTH] Global-mean RMSE by scenario and variable\n(black-bordered = held-out test scenarios)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'rmse_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved rmse_summary.png")

    # ─── PLOT 6: Per-model-family timeseries (all variables, all scenarios) ──
    all_scenarios = {**test_scenarios, **train_scenarios}

    for model_name in eval_models:
        model_results = {k: v for k, v in all_results.items() if k[1] == model_name}
        if not model_results:
            continue

        fig, axes = plt.subplots(2, 5, figsize=(28, 10))
        axes_flat = axes.flatten()

        for var_idx, var_name in enumerate(var_names):
            ax = axes_flat[var_idx]

            for (scen, mname, *_), res in sorted(model_results.items()):
                sty = all_scenarios.get(scen, {'color': 'gray', 'ls': '-'})
                is_test = res['is_test']

                obs_gm = global_mean(res['obs'])[:, var_idx].numpy()
                obs_phys = inv_transform(obs_gm, var_idx)

                draws_gm = global_mean(res['draws'])[:, :, var_idx].numpy()
                draws_phys = inv_transform(draws_gm, var_idx)
                pred_mean = draws_phys.mean(axis=0)
                pred_lo = np.percentile(draws_phys, 10, axis=0)
                pred_hi = np.percentile(draws_phys, 90, axis=0)

                t = np.arange(len(obs_phys))
                lw = 1.8 if is_test else 1.0
                alpha_line = 0.9 if is_test else 0.5
                ax.plot(t, obs_phys, color=sty['color'], ls=sty['ls'],
                        alpha=alpha_line, lw=lw, label=f'{scen} obs')
                ax.plot(t, pred_mean, color=sty['color'], ls=':',
                        alpha=alpha_line, lw=lw)
                ax.fill_between(t, pred_lo, pred_hi, color=sty['color'],
                                alpha=0.15 if is_test else 0.07)

            ax.set_title(var_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Month')
            ax.grid(True, alpha=0.3)
            if var_idx == 0:
                ax.legend(fontsize=6, ncol=1, loc='upper left')

        fig.suptitle(f'[NOSMOOTH] {model_name} — all scenarios at nside={nside_hi}\n'
                     f'(solid=obs, dotted=pred, bands=10-90%ile; bold=test scenarios)',
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fname = f'per_model_{model_name}.png'
        fig.savefig(os.path.join(plot_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")

    # ─── PLOT 7: Per-model RMSE comparison (one subplot per model family) ─
    n_models_found = sum(1 for m in eval_models
                         if any(k[1] == m for k in all_results))
    if n_models_found > 0:
        ncols = min(3, n_models_found)
        nrows = (n_models_found + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        if n_models_found == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()
        ax_idx = 0

        for model_name in eval_models:
            model_results = {k: v for k, v in all_results.items() if k[1] == model_name}
            if not model_results:
                continue
            ax = axes_flat[ax_idx]
            ax_idx += 1

            scen_rmse = {}
            for (scen, mname, *_), res in model_results.items():
                rmses = []
                for vi, vn in enumerate(var_names):
                    obs = inv_transform(global_mean(res['obs'])[:, vi].numpy(), vi)
                    pred = inv_transform(global_mean(res['draws'].mean(dim=0))[:, vi].numpy(), vi)
                    rmses.append(np.sqrt(np.mean((obs - pred)**2)))
                scen_rmse[scen] = rmses

            x_pos = np.arange(len(var_names))
            width = 0.8 / max(len(scen_rmse), 1)
            for i, scen in enumerate(sorted(scen_rmse.keys(),
                                            key=lambda s: s in test_scenarios, reverse=True)):
                is_test = scen in test_scenarios
                sty = all_scenarios.get(scen, {'color': 'gray'})
                ax.bar(x_pos + i * width, scen_rmse[scen], width * 0.9,
                       label=scen, color=sty['color'], alpha=0.8,
                       edgecolor='black' if is_test else 'none',
                       linewidth=1.5 if is_test else 0)

            ax.set_xticks(x_pos + width * len(scen_rmse) / 2)
            ax.set_xticklabels(var_names, rotation=45, fontsize=8)
            ax.set_ylabel('RMSE')
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3, axis='y')

        for i in range(ax_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

        fig.suptitle(f'[NOSMOOTH] Per-model RMSE at nside={nside_hi}\n(black border = test scenarios)',
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'rmse_per_model.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved rmse_per_model.png")

    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == '__main__':
    main()
