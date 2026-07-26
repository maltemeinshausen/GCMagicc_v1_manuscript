#!/usr/bin/env python3
import argparse
import json
import base64
import zlib
from pathlib import Path

import numpy as np
import healpy as hp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Example:
#   python plot_from_json_tasmax.py DAER3_tasmax_mean/aerosol_0_T1new_tasmax.json --field delta_mean
#   python plot_from_json_tasmax.py ... --field yh_mean
#   python plot_from_json_tasmax.py ... --field ycf_mean


def decode_f32_zlib_b64(obj: dict) -> np.ndarray:
    comp = base64.b64decode(obj["data_b64_zlib"].encode("ascii"))
    raw = zlib.decompress(comp)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(obj["shape"])
    return arr


def _get_map(payload: dict, field: str) -> np.ndarray:
    """
    New-version fields:
      - delta_mean: (P,)
      - yh_mean:    (P,)   (only present if store_counterfactuals=True)
      - ycf_mean:   (P,)   (only present if store_counterfactuals=True)

    Backward compatibility:
      - delta_std:  if present, decode as (P,)
      - yh_tasmax / ycf_tasmax: (nsim, P) (legacy); take mean over nsim
    """
    if field in ("delta_mean", "delta_std", "yh_mean", "ycf_mean"):
        if field not in payload:
            raise KeyError(
                f"Field '{field}' not found in JSON. "
                f"Available top-level keys: {sorted(payload.keys())}. "
                f"(Note: yh_mean/ycf_mean are only written when store_counterfactuals=True.)"
            )
        arr = decode_f32_zlib_b64(payload[field])
        if arr.ndim != 1:
            raise ValueError(f"Expected '{field}' to be 1D (P,), got shape {arr.shape}")
        return arr

    # Legacy keys (optional)
    if field == "yh_mean":
        if "yh_tasmax" in payload:
            yh = decode_f32_zlib_b64(payload["yh_tasmax"])
            if yh.ndim != 2:
                raise ValueError(f"Expected legacy 'yh_tasmax' to be 2D (nsim,P), got {yh.shape}")
            return yh.mean(axis=0).astype(np.float32)
        raise KeyError("Neither 'yh_mean' (new) nor 'yh_tasmax' (legacy) found in JSON.")

    if field == "ycf_mean":
        if "ycf_tasmax" in payload:
            ycf = decode_f32_zlib_b64(payload["ycf_tasmax"])
            if ycf.ndim != 2:
                raise ValueError(f"Expected legacy 'ycf_tasmax' to be 2D (nsim,P), got {ycf.shape}")
            return ycf.mean(axis=0).astype(np.float32)
        raise KeyError("Neither 'ycf_mean' (new) nor 'ycf_tasmax' (legacy) found in JSON.")

    raise ValueError(f"Unsupported field: {field}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=str, help="Path to the JSON produced by the sampler script")
    ap.add_argument(
        "--field",
        type=str,
        default="delta_mean",
        choices=["delta_mean", "yh_mean", "ycf_mean", "delta_std"],
        help="What to plot (delta_std only if present in JSON)",
    )
    ap.add_argument("--out", type=str, default=None, help="Output PDF path (default: next to JSON)")
    ap.add_argument("--xsize", type=int, default=4000, help="Healpy mollview xsize")
    ap.add_argument("--rot", type=float, default=180.0, help="Rotation for mollview")
    ap.add_argument(
        "--graticule",
        action="store_true",
        help="Overlay healpy graticule (off by default to match the sampler fallback look)",
    )
    ap.add_argument(
        "--bgcolor",
        type=str,
        default="white",
        help="Background color for healpy mollview (default: white)",
    )
    args = ap.parse_args()

    with open(args.json_path, "r") as f:
        payload = json.load(f)

    meta = payload.get("meta", {})
    var = meta.get("variable", "tasmax")

    # Matplotlib PDF quality knobs
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.compression"] = 9

    arr = _get_map(payload, args.field)
    title = ""

    v = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(v) or v == 0.0:
        v = 1.0

    #fig = plt.figure()
    #fig.patch.set_facecolor(args.bgcolor)
    fig = plt.figure(facecolor=args.bgcolor)

    # Be robust across healpy versions re: bgcolor/fig kwargs
    try:
        hp.mollview(
            arr,
            fig=fig.number,
            flip="geo",
            title=title,
            xsize=args.xsize,
            rot=args.rot,
            cmap=plt.cm.get_cmap("coolwarm", 1024),
            min=-v,
            max=+v,
            bgcolor=args.bgcolor,
        )
    except TypeError:
        # Older healpy: retry without bgcolor (and possibly fig)
        try:
            hp.mollview(
                arr,
                fig=fig.number,
                flip="geo",
                title=title,
                xsize=args.xsize,
                rot=args.rot,
                cmap=plt.cm.get_cmap("coolwarm", 1024),
                min=-v,
                max=+v,
            )
        except TypeError:
            hp.mollview(
                arr,
                flip="geo",
                title=title,
                xsize=args.xsize,
                rot=args.rot,
                cmap=plt.cm.get_cmap("coolwarm", 1024),
                min=-v,
                max=+v,
            )

    if args.graticule:
        try:
            hp.graticule(dpar=30, dmer=60, verbose=False)
        except TypeError:
            hp.graticule(dpar=30, dmer=60)

    if args.out is None:
        base = str(Path(args.json_path).with_suffix(""))
        out = f"{base}_{var}_{args.field}.pdf"
    else:
        out = args.out

    plt.savefig(out, bbox_inches="tight", facecolor=args.bgcolor)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
