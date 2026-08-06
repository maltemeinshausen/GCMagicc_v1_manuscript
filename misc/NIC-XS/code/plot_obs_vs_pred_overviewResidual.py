"""
Combined multi-panel residual overview: (pred - obs) vs obs.
Reads from full_monthly_results.csv — no model inference needed.
Produces one overview PNG per variable, using monthly data.
Works for both miss585 and missSSP2 splits (auto-detects scenarios).
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='path to full_monthly_results.csv')
parser.add_argument('--variable', default='all',
                    help='variable to plot, or "all" for every variable')
parser.add_argument('--out-dir', default=None, help='output directory (default: same as csv)')
parser.add_argument('--tag', default='', help='tag appended to filename')
parser.add_argument('--title-prefix', default='', help='prefix for figure title')
args = parser.parse_args()

df_all = pd.read_csv(args.csv)

SKIP_MODELS = {'CESM2-WACCM'}
df_all = df_all[~df_all['model'].isin(SKIP_MODELS)]

skip_abrupt = {'abrupt-4xCO2', 'abrupt-2xCO2', '1pctCO2'}
df_all = df_all[~df_all['scenario'].isin(skip_abrupt)]

scenario_colors = {
    'ssp585': 'red',
    'ssp370': 'purple',
    'ssp460': 'goldenrod',
    'historical': 'gray',
    'ssp119': 'darkblue',
    'ssp126': 'blue',
    'ssp245': 'teal',
    'ssp434': 'orange',
    'hist-GHG': 'olive',
    'hist-aer': 'cyan',
    'hist-nat': 'lime',
}

variables = sorted(df_all['variable'].unique()) if args.variable == 'all' else [args.variable]
out_dir = args.out_dir or os.path.dirname(args.csv)

for var in variables:
    df = df_all[df_all['variable'] == var]
    models = sorted(df['model'].unique())
    n_models = len(models)

    ncols_models = 4
    nrows_models = int(np.ceil(n_models / ncols_models))

    fig = plt.figure(figsize=(ncols_models * 4.5, nrows_models * 2.4))
    outer = gridspec.GridSpec(nrows_models, ncols_models,
                              wspace=0.22, hspace=0.30,
                              left=0.04, right=0.98, top=0.92, bottom=0.03)

    train_scens_seen = set()
    test_scens_seen = set()

    for idx, model_name in enumerate(models):
        row = idx // ncols_models
        col = idx % ncols_models
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col],
                                                wspace=0.05)
        ax_tr = fig.add_subplot(inner[0])
        ax_te = fig.add_subplot(inner[1])

        mdf = df[df['model'] == model_name]

        all_obs = []
        all_resid = []
        split = {'train': [], 'test': []}
        for scen, sdf in mdf.groupby('scenario'):
            obs = sdf['obs'].values
            pred = sdf['pred_mean'].values
            resid = pred - obs
            all_obs.extend(obs)
            all_resid.extend(resid)
            is_test = sdf['is_test'].iloc[0]
            key = 'test' if is_test else 'train'
            split[key].append((scen, obs, resid))
            if is_test:
                test_scens_seen.add(scen)
            else:
                train_scens_seen.add(scen)

        if not all_obs:
            continue
        xmin = min(all_obs)
        xmax = max(all_obs)
        xpad = (xmax - xmin) * 0.03
        resid_abs_max = max(abs(r) for r in all_resid) if all_resid else 1.0
        ypad = resid_abs_max * 1.1

        for ax, label, data in [(ax_tr, 'Train', split['train']),
                                 (ax_te, 'Test', split['test'])]:
            for scen, obs_v, resid_v in data:
                c = scenario_colors.get(scen, 'black')
                ax.scatter(obs_v, resid_v, c=c, s=1.5, alpha=0.25,
                           edgecolors='none', rasterized=True)
            ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.4)
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(-ypad, ypad)
            ax.tick_params(labelsize=5, length=2, pad=1)
            if label == 'Test':
                ax.set_yticklabels([])
            if row == nrows_models - 1:
                ax.set_xlabel('Obs', fontsize=6)
            if col == 0 and label == 'Train':
                ax.set_ylabel('Pred − Obs', fontsize=6)
            ax.grid(True, alpha=0.2, linewidth=0.3)
            ax.set_title(label, fontsize=6, pad=1)

        ax_tr.annotate(model_name, xy=(0.5, 1.0), xycoords='axes fraction',
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=7, fontweight='bold', ha='center', va='bottom',
                       annotation_clip=False)

    legend_order = ['historical', 'ssp119', 'ssp126', 'ssp245', 'ssp370',
                    'ssp434', 'ssp460', 'ssp585',
                    'hist-GHG', 'hist-aer', 'hist-nat']
    all_scens = train_scens_seen | test_scens_seen
    handles = []
    for scen in legend_order:
        if scen in all_scens:
            c = scenario_colors.get(scen, 'black')
            suffix = ' (test)' if scen in test_scens_seen else ''
            handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=c, markersize=4,
                                  label=scen + suffix))
    prefix = f'{args.title_prefix} ' if args.title_prefix else ''
    fig.suptitle(f'{prefix}Residual (Pred − Obs) vs Obs: {var.upper()} (monthly, all models)',
                 fontsize=11, y=1.0)
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 10),
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.97))

    out_name = os.path.join(out_dir, f'obs_vs_pred_{var}_overviewResidual{args.tag}.png')
    fig.savefig(out_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_name}")
