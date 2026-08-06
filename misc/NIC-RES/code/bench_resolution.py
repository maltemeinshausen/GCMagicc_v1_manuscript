# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Nicolai Meinshausen.
"""
bench_resolution.py
===================

Compute-cost panel for the NIC-RES resolution experiment: GCMagicc inference
**wall-clock time and peak memory as a function of HEALPix output resolution
(nside)**.

The multi-resolution sampler (``run_helpers.sample_from_combined_model``) solves
the HEALPix cascade coarse-to-fine and stops at the requested ``nside``. Timing
``sample(nside=N)`` therefore measures the end-to-end cost to produce output at
resolution N (nside in {1,2,4,8,16,32,64,128,256}; npix = 12*nside^2).

Measurement boundaries (documented, matched across nside):
  * The HEALPix->lat/lon reprojection is EXCLUDED (rectangular=False) so we time
    the model path, not the (nside-independent) cartographic resampling.
  * The A5 sampler loads its per-level checkpoints INSIDE each call, so each
    timed iteration measures end-to-end sampler cost to reach the target nside
    (checkpoint load for levels 1..nside + forward cascade) -- i.e. the cost as
    invoked in production. It is NOT a forward-only microbenchmark. ``--warmup``
    untimed iterations first warm the filesystem/OS page cache (and CUDA caches
    / cuDNN autotune on GPU); then ``--reps`` timed iterations, median + IQR.
    To isolate forward-only cost, load once with a persistent-model sampler
    variant instead (see README).
  * CUDA timings use ``torch.cuda.synchronize`` around ``perf_counter``; peak
    memory from ``torch.cuda.max_memory_allocated`` (reset each rep). On CPU,
    peak RSS via resource.getrusage.
  * Fixed precision fp32 (``--dtype float32``); GPU pinned via --device.

Inputs: only model checkpoints and a forcing predictor matrix x of shape
(T, x_features). The benchmark does NOT need target data: pass a real x via
--x-file (.npy/.pt, T x 15) for a representative run, or let the script
synthesise a smooth x of length --months (default 1200 = 100 yr).

Outputs (--outdir):
  resolution_timing.csv   nside,npix,months,wall_s_median,wall_s_iqr,
                          peak_mem_bytes,reps,warmup,device,dtype
  resolution_timing.pdf   time-vs-nside and memory-vs-nside panels (log-log)
"""
import argparse
import os
import sys
import time
import resource
import numpy as np
import torch

# run_helpers.py is shipped alongside this script in the bundle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_helpers import sample_from_combined_model  # noqa: E402

ALL_NSIDE = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def synth_x(months, x_features=15, seed=0):
    """Smooth, physically-plausible-ish forcing matrix (T, x_features).

    Column 0 is the model index (int); remaining columns are smooth ramps +
    seasonal terms. Values only drive compute cost, not scientific output, so
    the exact numbers are immaterial to timing.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    x = np.zeros((months, x_features), dtype=np.float32)
    x[:, 0] = 0  # model index (ERA5 slot)
    for j in range(1, x_features):
        ramp = np.linspace(0, 1, months) * (0.5 + 0.5 * rng.random())
        seas = 0.1 * np.sin(2 * np.pi * (t % 12) / 12 + j)
        x[:, j] = ramp + seas
    return torch.from_numpy(x)


def time_one(x, nside, model_dir, date, device, dtype, warmup, reps):
    peak_bytes = np.nan
    is_cuda = device.startswith('cuda') and torch.cuda.is_available()

    def _call():
        # rectangular=False: exclude the HEALPix->lat/lon reprojection so we time
        # only the model forward cascade. asnumpy=False keeps output on device.
        return sample_from_combined_model(
            x, device=device, dirname=model_dir, DATE=date,
            dependence=False, nside=nside, rectangular=False,
            asnumpy=False, seed=0)

    for _ in range(warmup):
        _call()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(reps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _call()
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    if is_cuda:
        peak_bytes = float(torch.cuda.max_memory_allocated())
    else:
        peak_bytes = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0
    times = np.array(times)
    q1, q3 = np.percentile(times, [25, 75])
    return float(np.median(times)), float(q3 - q1), peak_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', default=os.environ.get('MODEL_DIR', '../checkpoints/modelsA/'),
                    help='dir with modelsNfour_<DATE>_* checkpoints + meta/ranges')
    ap.add_argument('--date', default='7Augext')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', default='float32', choices=['float32'])
    ap.add_argument('--months', type=int, default=1200, help='length T of x if synthesised (default 1200)')
    ap.add_argument('--x-file', default='', help='optional .npy/.pt forcing matrix (T x x_features)')
    ap.add_argument('--nside-max', type=int, default=64, help='largest nside to benchmark (default 64)')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--outdir', default='../outputs')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.x_file:
        if args.x_file.endswith('.pt'):
            x = torch.load(args.x_file, map_location='cpu').float()
        else:
            x = torch.from_numpy(np.load(args.x_file)).float()
    else:
        x = synth_x(args.months)
    months = x.shape[0]

    nsides = [n for n in ALL_NSIDE if n <= args.nside_max]
    print(f"Benchmarking nside={nsides} months={months} device={args.device} "
          f"reps={args.reps} warmup={args.warmup}")

    rows = []
    for nside in nsides:
        try:
            med, iqr, peak = time_one(x, nside, args.model_dir, args.date,
                                      args.device, args.dtype, args.warmup, args.reps)
        except Exception as e:
            print(f"  nside={nside}: FAILED ({e})")
            continue
        npix = 12 * nside * nside
        rows.append(dict(nside=nside, npix=npix, months=months,
                         wall_s_median=med, wall_s_iqr=iqr, peak_mem_bytes=peak,
                         reps=args.reps, warmup=args.warmup,
                         device=args.device, dtype=args.dtype))
        print(f"  nside={nside:4d} npix={npix:7d}  {med:8.4f}s (IQR {iqr:.4f})  "
              f"peak {peak/1e9:.3f} GB")

    if not rows:
        print("No successful benchmarks."); return

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, 'resolution_timing.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.errorbar(df['npix'], df['wall_s_median'], yerr=df['wall_s_iqr'],
                marker='o', capsize=3)
    a1.set_xscale('log'); a1.set_yscale('log')
    a1.set_xlabel('HEALPix npix (= 12 nside^2)'); a1.set_ylabel('wall time (s)')
    a1.set_title(f'Inference time vs resolution ({months//12} yr, {args.device})')
    a1.grid(True, which='both', alpha=0.3)
    a2.plot(df['npix'], df['peak_mem_bytes'] / 1e9, marker='s', color='C1')
    a2.set_xscale('log')
    a2.set_xlabel('HEALPix npix'); a2.set_ylabel('peak memory (GB)')
    a2.set_title('Peak memory vs resolution')
    a2.grid(True, which='both', alpha=0.3)
    for ax in (a1, a2):
        ax.set_xticks(df['npix'])
        ax.set_xticklabels([str(n) for n in df['nside']])
        ax.set_xlabel('nside')
    fig.tight_layout()
    pdf_path = os.path.join(args.outdir, 'resolution_timing.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved {pdf_path}")


if __name__ == '__main__':
    main()
