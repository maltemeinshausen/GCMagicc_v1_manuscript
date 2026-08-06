"""
Energy-distance score distributions across hold-out scenarios.

Figure: 11-panel grid (3x4) showing relative energy-distance score S
distributions for each recipe family, comparing trained vs hold-out scenarios.

    S = ED(emulator, CMIP6) / ED(CMIP6, other CMIP6)

S < 1: emulator closer to truth than inter-model spread.
S > 1: emulator worse than inter-model agreement.

Usage:
    python eval_energy_distance.py --nside 2
    python eval_energy_distance.py --nside 2 --nosmooth
    python eval_energy_distance.py --nside 1
    python eval_energy_distance.py --nside 1 --nosmooth
"""

import argparse
import csv
import os
SCRATCH_BASE = os.environ.get('SCRATCH_BASE', '/scratch2')
import glob
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from help_functions import Downsample
from models import LB2, L4de

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Energy-distance score validation')
parser.add_argument('--nside', type=int, default=2, choices=[1, 2])
parser.add_argument('--nosmooth', action='store_true')
args = parser.parse_args()

nside_hi = args.nside
nosmooth = args.nosmooth

# ─── Configuration ────────────────────────────────────────────────────────
DATE = '7Augext'
SUFFIX = '_miss585'
device = torch.device('cpu')

npix = 12 * nside_hi**2
n_features = 10
variab = 10
maxlag = 72
lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
        24, 27, 30, 33, 36, 42, 48, 54, 60, 66, 72]

X_COLS = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14] if nosmooth else None
x_features = len(X_COLS) if nosmooth else 15

normalized_dir = SCRATCH_BASE + f'/userdata/nicolai/cmip6/normalized_{DATE}/'

ns_tag = '_nosmooth' if nosmooth else ''
model_base = SCRATCH_BASE + f'/userdata/nicolai/cmip6/modelsNfour_{DATE}{SUFFIX}{ns_tag}'

tag = f'nside{nside_hi}{"_nosmooth" if nosmooth else ""}'
plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f'plots_eval_{tag}')
os.makedirs(plot_dir, exist_ok=True)

var_names = ["psl", "tas", "pr", "sfcWind", "ts",
             "tasmin", "tasmax", "rsds", "hurs", "huss"]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'meta_{DATE}.pkl'), 'rb') as f:
    meta = pickle.load(f)
transformation_scalars = meta['transformation_scalars']


# ─── Load trained emulator ────────────────────────────────────────────────
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


# ─── Helpers ──────────────────────────────────────────────────────────────
downs = Downsample()

N_DOWNSAMPLE = {1: 6, 2: 5}


def downsample_to_nside(y64):
    y = torch.from_numpy(y64).float()
    for _ in range(N_DOWNSAMPLE[nside_hi]):
        y = downs(y)
    return y.numpy()


def find_file(model_name, scenario):
    pattern = f'data_DAT_{model_name}_{scenario}_*.h5'
    files = sorted(glob.glob(os.path.join(normalized_dir, pattern)))
    return files[0] if files else None


def get_pixel_latitudes(nside):
    if HAS_HEALPY:
        theta, _ = hp.pix2ang(nside, np.arange(12 * nside**2), nest=False)
        return 90.0 - np.degrees(theta)
    if nside == 1:
        lat_hi = np.degrees(np.arccos(2.0 / 3.0))
        return np.array([lat_hi] * 4 + [0.0] * 4 + [-lat_hi] * 4)
    if nside == 2:
        lats = []
        for ring in range(1, 4 * nside):
            if ring < nside:
                n_in_ring = 4 * ring
                cos_t = 1.0 - ring**2 / (3.0 * nside**2)
            elif ring < 3 * nside:
                n_in_ring = 4 * nside
                cos_t = 4.0 / 3.0 - 2.0 * ring / (3.0 * nside)
            else:
                n_in_ring = 4 * (4 * nside - ring)
                cos_t = -(1.0 - (4 * nside - ring)**2 / (3.0 * nside**2))
            lat = 90.0 - np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
            lats.extend([lat] * n_in_ring)
        return np.array(lats)
    raise ValueError(f"No healpy fallback for nside={nside}")


# ─── Recipe family metric functions ───────────────────────────────────────
#
# Convention: each function takes y of shape (T, n_features, npix) and
# returns dict {label: 1-D metric vector}.  Labels like 'var3' allow
# per-variable breakdown; labels like 'var2_var7' denote cross-variable
# metrics.  Energy distances are computed per label.

MAX_MONTHS = 840   # 70 yr cap for time-dependent metrics

RECIPE_NAMES = [
    'BiasMaps', 'GlobalTimeseries', 'Histograms', 'SeasonalCycle',
    'ZonalMeans', 'Variability', 'JointEvolution', 'HumidityCoupling',
    'IndicatorFreq', 'ENSOTeleconnections', 'ENSOCharacteristics',
]


def recipe_BiasMaps(y):
    """Time-mean spatial field per variable."""
    T, nf, _ = y.shape
    return {f'var{vi}': y[:, vi, :].mean(axis=0) for vi in range(nf)}


def recipe_GlobalTimeseries(y):
    """Global-mean monthly time series per variable (truncated)."""
    T, nf, _ = y.shape
    Tc = min(T, MAX_MONTHS)
    return {f'var{vi}': y[:Tc, vi, :].mean(axis=1) for vi in range(nf)}


def recipe_Histograms(y, n_q=50):
    """Quantile function of spatio-temporal values per variable."""
    probs = np.linspace(0.01, 0.99, n_q)
    T, nf, _ = y.shape
    return {f'var{vi}': np.quantile(y[:, vi, :].ravel(), probs)
            for vi in range(nf)}


def recipe_SeasonalCycle(y):
    """Monthly climatology of global mean per variable (12-vector)."""
    T, nf, _ = y.shape
    gm = y.mean(axis=2)
    ny = T // 12
    if ny < 1:
        return {}
    return {f'var{vi}': gm[:ny * 12, vi].reshape(ny, 12).mean(axis=0)
            for vi in range(nf)}


def recipe_ZonalMeans(y):
    """Time-mean zonal-mean profile per variable."""
    T, nf, npx = y.shape
    lats = get_pixel_latitudes(nside_hi)
    n_bands = max(3, npx // 4)
    edges = np.linspace(-90, 90, n_bands + 1)
    out = {}
    for vi in range(nf):
        field = y[:, vi, :].mean(axis=0)
        zm = np.zeros(n_bands)
        for b in range(n_bands):
            mask = (lats >= edges[b]) & (lats < edges[b + 1])
            if mask.any():
                zm[b] = field[mask].mean()
        out[f'var{vi}'] = zm
    return out


def recipe_Variability(y):
    """Deseasonalised, detrended global-mean anomaly series per variable."""
    T, nf, _ = y.shape
    gm = y.mean(axis=2)
    ny = T // 12
    if ny < 2:
        return {}
    Tc = min(ny, MAX_MONTHS // 12)
    out = {}
    for vi in range(nf):
        monthly = gm[:Tc * 12, vi].reshape(Tc, 12)
        anom = (monthly - monthly.mean(axis=0)).ravel()
        t = np.arange(len(anom), dtype=float)
        p = np.polyfit(t, anom, 1)
        anom -= np.polyval(p, t)
        out[f'var{vi}'] = anom
    return out


def recipe_JointEvolution(y):
    """Cross-variable temporal cross-correlation (12 lags) per pair."""
    T, nf, _ = y.shape
    gm = y.mean(axis=2)
    Tc = min(T, MAX_MONTHS)
    if Tc < 24:
        return {}
    out = {}
    for vi in range(nf):
        a = gm[:Tc, vi]
        a = (a - a.mean()) / (a.std() + 1e-10)
        for vj in range(vi + 1, nf):
            b = gm[:Tc, vj]
            b = (b - b.mean()) / (b.std() + 1e-10)
            n_lag = min(12, Tc - 1)
            xcorr = np.array([np.corrcoef(a[:Tc - l], b[l:Tc])[0, 1]
                              for l in range(n_lag)])
            out[f'var{vi}_var{vj}'] = xcorr
    return out


def recipe_HumidityCoupling(y):
    """Paired (humidity, temperature) global-mean series."""
    T, nf, _ = y.shape
    gm = y.mean(axis=2)
    Tc = min(T, MAX_MONTHS)
    pairs = [(8, 1, 'hurs_tas'), (8, 4, 'hurs_ts'),
             (9, 1, 'huss_tas'), (9, 4, 'huss_ts')]
    out = {}
    for hi, ti, name in pairs:
        if hi >= nf or ti >= nf:
            continue
        out[name] = np.stack([gm[:Tc, hi], gm[:Tc, ti]], axis=1).ravel()
    return out


def recipe_IndicatorFreq(y, n_q=40):
    """Tail-focused quantile function of global-mean per variable."""
    probs = np.concatenate([np.linspace(0.01, 0.10, n_q // 2),
                            np.linspace(0.90, 0.99, n_q // 2)])
    T, nf, _ = y.shape
    gm = y.mean(axis=2)
    return {f'var{vi}': np.quantile(gm[:, vi], probs)
            for vi in range(nf)}


def recipe_ENSOTeleconnections(y):
    """Correlation of tropical TAS index with every pixel, per variable."""
    T, nf, npx = y.shape
    lats = get_pixel_latitudes(nside_hi)
    trop = np.abs(lats) < 30
    if not trop.any():
        return {}
    Tc = min(T, MAX_MONTHS)
    idx = y[:Tc, 1, :][:, trop].mean(axis=1)
    idx = idx - idx.mean()
    s_idx = idx.std() + 1e-10
    out = {}
    for vi in range(nf):
        corr = np.zeros(npx)
        for p in range(npx):
            ts = y[:Tc, vi, p] - y[:Tc, vi, p].mean()
            corr[p] = np.mean(idx * ts) / (s_idx * (ts.std() + 1e-10))
        out[f'var{vi}'] = corr
    return out


def recipe_ENSOCharacteristics(y):
    """ENSO index properties: std, skewness, autocorrelation structure."""
    T, nf, _ = y.shape
    lats = get_pixel_latitudes(nside_hi)
    trop = np.abs(lats) < 30
    if not trop.any():
        return {}
    ny = T // 12
    if ny < 3:
        return {}
    out = {}
    for vi, name in [(1, 'tas'), (0, 'psl')]:
        if vi >= nf:
            continue
        raw = y[:ny * 12, vi, :][:, trop].mean(axis=1)
        monthly = raw.reshape(ny, 12)
        anom = (monthly - monthly.mean(axis=0)).ravel()
        acf_full = np.correlate(anom, anom, 'full')
        acf = acf_full[len(acf_full) // 2:]
        acf = acf / (acf[0] + 1e-10)
        n_lag = min(25, len(acf))
        sigma = anom.std() + 1e-10
        skew = np.mean((anom - anom.mean())**3) / sigma**3
        out[name] = np.concatenate([[sigma, skew], acf[1:n_lag]])
    return out


RECIPE_FUNCS = {n: globals()[f'recipe_{n}'] for n in RECIPE_NAMES}


# ─── Energy score ─────────────────────────────────────────────────────────
def energy_score(draw_vecs, truth_vec):
    """ES between K draws and a single truth vector (all 1-D, same length)."""
    K = len(draw_vecs)
    if K == 0:
        return np.nan
    cross = np.mean([np.linalg.norm(d - truth_vec) for d in draw_vecs])
    if K > 1:
        pairs = []
        for k in range(K):
            for l in range(k + 1, K):
                pairs.append(np.linalg.norm(draw_vecs[k] - draw_vecs[l]))
        self_term = np.mean(pairs)
    else:
        self_term = 0.0
    return cross - 0.5 * self_term


# ─── Scenarios & models ──────────────────────────────────────────────────
test_scenarios = ['ssp585', 'abrupt-4xCO2']
train_scenarios = ['historical', 'ssp119', 'ssp126', 'ssp370', 'ssp434', 'ssp460',
                   'hist-GHG', 'hist-aer', 'hist-nat', 'abrupt-2xCO2']
all_scenarios = train_scenarios + test_scenarios

SCENARIO_STYLES = {
    'train':         {'color': '#2166ac', 'ls': '-',
                      'label': 'Trained (hist/ssp119/126/370/434/460/hist-*/abr2x)'},
    'ssp585':        {'color': '#b2182b', 'ls': '--',
                      'label': 'SSP5-8.5 hold-out'},
    'abrupt-4xCO2':  {'color': '#8c510a', 'ls': (0, (5, 2)),
                      'label': 'abrupt-4xCO2 hold-out'},
}

eval_models = ['CanESM5', 'MIROC6', 'IPSL-CM6A-LR',
               'UKESM1-0-LL', 'GFDL-ESM4', 'ACCESS-ESM1-5']


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print(f"Energy-distance validation: nside={nside_hi}, nosmooth={nosmooth}")
    print("Loading models...")
    bias_model, effect_model = load_model()

    # ── 1. Generate truth + emulator draws for every (scenario, model) ────
    data = {}
    n_draws = 10
    for scenario in all_scenarios:
        for model_name in eval_models:
            fpath = find_file(model_name, scenario)
            if fpath is None:
                continue
            print(f"  {model_name} / {scenario}")
            try:
                with h5py.File(fpath, 'r') as h5:
                    y64 = h5['y64'][:]
                    X = h5['X'][:]

                truth = downsample_to_nside(y64)

                x_tensor = torch.from_numpy(X).float()
                if nosmooth:
                    x_tensor = x_tensor[:, X_COLS]
                x_tensor = x_tensor.to(device)

                with torch.no_grad():
                    y_bias = bias_model(x=x_tensor, y_low=None).detach()
                    draws = []
                    for _ in range(n_draws):
                        yp = effect_model(x=x_tensor, y_in=y_bias,
                                          dependence=True)
                        draws.append(yp.detach().numpy())

                data[(scenario, model_name)] = {
                    'truth': truth,
                    'draws': draws,
                }
            except Exception as e:
                print(f"    Error: {e}")

    if not data:
        print("No data loaded.")
        return

    # ── 2. Compute S scores per (recipe, scenario_class) ─────────────────
    S_scores = defaultdict(lambda: defaultdict(list))

    for recipe_name, recipe_func in RECIPE_FUNCS.items():
        print(f"  Recipe: {recipe_name}")

        # 2a. Compute metric vectors for truth and each draw
        all_metrics = {}
        for key, d in data.items():
            truth_m = recipe_func(d['truth'])
            draw_ms = [recipe_func(dr) for dr in d['draws']]
            all_metrics[key] = {'truth': truth_m, 'draws': draw_ms}

        # 2b. Reference: within-scenario inter-model distances per label
        ref_dists = defaultdict(list)
        for scenario in all_scenarios:
            models_here = [m for m in eval_models
                           if (scenario, m) in all_metrics]
            for i, m1 in enumerate(models_here):
                for m2 in models_here[i + 1:]:
                    t1 = all_metrics[(scenario, m1)]['truth']
                    t2 = all_metrics[(scenario, m2)]['truth']
                    for label in t1:
                        if label not in t2:
                            continue
                        v1, v2 = t1[label], t2[label]
                        ml = min(len(v1), len(v2))
                        d = np.linalg.norm(v1[:ml] - v2[:ml])
                        if d > 0:
                            ref_dists[label].append(d)

        # 2c. S = ES(emulator, truth) / median(ref) per (scenario, model, label)
        for scenario in all_scenarios:
            scen_class = ('train' if scenario in train_scenarios
                          else scenario)
            for model_name in eval_models:
                key = (scenario, model_name)
                if key not in all_metrics:
                    continue
                tm = all_metrics[key]['truth']
                dms = all_metrics[key]['draws']
                for label in tm:
                    if label not in ref_dists or not ref_dists[label]:
                        continue
                    tv = tm[label]
                    dvs = []
                    for dm in dms:
                        if label in dm:
                            v = dm[label]
                            ml = min(len(v), len(tv))
                            dvs.append(v[:ml])
                    if not dvs:
                        continue
                    tv_t = tv[:len(dvs[0])]
                    es = energy_score(dvs, tv_t)
                    ref = np.median(ref_dists[label])
                    if ref > 0 and np.isfinite(es) and es > 0:
                        S_scores[recipe_name][scen_class].append(es / ref)

    # ── 3. Figure ────────────────────────────────────────────────────────
    print("Generating figure...")
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
    })

    fig, axes = plt.subplots(3, 4, figsize=(11.0, 8.5))
    axes_flat = axes.flatten()

    for idx, recipe_name in enumerate(RECIPE_NAMES):
        ax = axes_flat[idx]
        plotted = False

        for scen_class in ['train'] + test_scenarios:
            sty = SCENARIO_STYLES.get(scen_class)
            if sty is None:
                continue
            vals = np.array(S_scores[recipe_name].get(scen_class, []))
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) < 3:
                continue
            plotted = True
            log_v = np.log10(vals)

            if HAS_SCIPY and len(vals) >= 5:
                kde = gaussian_kde(log_v, bw_method=0.35)
                xg = np.linspace(log_v.min() - 0.5,
                                 log_v.max() + 0.5, 300)
                dens = kde(xg)
                ax.plot(10**xg, dens, color=sty['color'], ls=sty['ls'],
                        lw=1.4, label=sty['label'])
                ax.fill_between(10**xg, dens, alpha=0.06,
                                color=sty['color'])
            else:
                ax.hist(vals, bins=15, density=True, alpha=0.35,
                        color=sty['color'], label=sty['label'],
                        histtype='stepfilled')

        ax.axvline(1.0, color='k', ls=':', lw=0.8, alpha=0.6, zorder=0)
        ax.set_xscale('symlog', linthresh=0.1)
        ax.set_title(recipe_name, fontweight='bold')
        if idx >= 8:
            ax.set_xlabel('S')
        if idx % 4 == 0:
            ax.set_ylabel('Density')
        ax.grid(True, alpha=0.15, which='both')

        if not plotted:
            ax.text(0.5, 0.5, 'Insufficient data\nat this resolution',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='gray', fontstyle='italic')

    # Last panel (12th): legend only
    ax_leg = axes_flat[11]
    ax_leg.set_visible(False)
    handles = []
    for sc in ['train'] + test_scenarios:
        sty = SCENARIO_STYLES[sc]
        handles.append(Line2D([0], [0], color=sty['color'], ls=sty['ls'],
                              lw=1.5, label=sty['label']))
    handles.append(Line2D([0], [0], color='k', ls=':', lw=0.8,
                          label='S = 1 (emulator = inter-model spread)'))
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(0.98, 0.02), fontsize=8, frameon=True,
               fancybox=False, edgecolor='0.5')

    smooth_label = ' [NOSMOOTH]' if nosmooth else ''
    fig.suptitle(
        f'Energy-distance score distributions across hold-out scenarios'
        f' — nside={nside_hi}{smooth_label}\n'
        r'$S = \mathrm{ED}(\mathrm{emulator},\,\mathrm{CMIP6})\;/\;'
        r'\mathrm{ED}(\mathrm{CMIP6},\,\mathrm{CMIP6})$',
        fontsize=11, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    date_str = '20260508'
    base = f'Figure_1_extrapolation_energy_distance_{tag}_{date_str}'
    fig.savefig(os.path.join(plot_dir, base + '.png'),
                dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, base + '.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {base}.png / .pdf")

    # ── 4. Summary CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(plot_dir, base + '_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['recipe', 'scenario_class', 'n',
                     'median_S', 'Q1', 'Q3', 'IQR'])
        for rn in RECIPE_NAMES:
            for sc in ['train'] + test_scenarios:
                vals = np.array(S_scores[rn].get(sc, []))
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals) > 0:
                    q1, med, q3 = np.percentile(vals, [25, 50, 75])
                    w.writerow([rn, sc, len(vals),
                                f'{med:.4f}', f'{q1:.4f}',
                                f'{q3:.4f}', f'{q3 - q1:.4f}'])
                else:
                    w.writerow([rn, sc, 0, '', '', '', ''])
    print(f"Saved {csv_path}")

    print(f"\nAll energy-distance outputs in {plot_dir}/")


if __name__ == '__main__':
    main()
