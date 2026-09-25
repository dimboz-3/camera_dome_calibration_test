#!/usr/bin/env python3
"""Difference image for two renders of the same scene.

Each image is normalized INDEPENDENTLY (robust percentile stretch to [0, 1]),
then B - A is computed and amplified by a gain so that small engine-level
differences become visible.  Mid-gray 128 = "no difference".

Outputs (next to the B image unless --out is given):
    <stem>_diff_x<gain>.png    amplified grayscale difference
    <stem>_diffc_x<gain>.png   colored: red = B brighter, blue = A brighter

Usage:
    python diffimg.py A.png B.png [--gain 50] [--out DIR]
    python diffimg.py --dir-a data/series --dir-b data/series/render \
        --out data/series/diff --gain 50
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def load_normalized(path: str):
    """Read image, stretch percentiles (0.1 .. 99.9) to [0,1] independently."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image {path}")
    f = img.astype(np.float64) / 255.0
    lo = float(np.percentile(f, 0.1))
    hi = float(np.percentile(f, 99.9))
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return np.clip((f - lo) / (hi - lo), 0.0, 1.0), lo, hi


def make_diff(a_path: str, b_path: str, outdir: str, gain: float):
    a, lo_a, hi_a = load_normalized(a_path)
    b, lo_b, hi_b = load_normalized(b_path)
    if a.shape != b.shape:
        raise RuntimeError(f"size mismatch: {a.shape} vs {b.shape}")

    diff = b - a                       # signed, range about [-1, 1]
    amp = np.clip(diff * gain + 0.5, 0.0, 1.0)

    stem = os.path.splitext(os.path.basename(b_path))[0]
    gray = (amp * 255.0).round().astype(np.uint8)
    gray_path = os.path.join(outdir, stem + f"_diff_x{gain:g}.png")
    cv2.imwrite(gray_path, gray)

    # colored: red = B brighter, blue = A brighter (channel-mean signed diff)
    d = diff.mean(axis=2)
    pos = np.clip(d * gain, 0.0, 1.0)
    neg = np.clip(-d * gain, 0.0, 1.0)
    bg = 0.5 * (1.0 - pos - neg)          # mid-gray background, per pixel
    col = np.dstack([neg + bg, bg, pos + bg])
    col = (col * 255.0).round().astype(np.uint8)
    col_path = os.path.join(outdir, stem + f"_diffc_x{gain:g}.png")
    cv2.imwrite(col_path, col)

    d8 = diff * 255.0
    print(f"{stem}: gain={gain:g}  |diff| mean={np.abs(d8).mean():.2f}/255  "
          f"p99={np.percentile(np.abs(d8), 99):.2f}/255  "
          f"max={np.abs(d8).max():.2f}/255")
    print(f"  norm A [{lo_a:.4f}, {hi_a:.4f}]  norm B [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"  wrote {gray_path}")
    print(f"  wrote {col_path}")


def main():
    ap = argparse.ArgumentParser(description="amplified difference image")
    ap.add_argument("a", nargs="?", help="image A (e.g. mitsuba png)")
    ap.add_argument("b", nargs="?", help="image B (e.g. blender png)")
    ap.add_argument("--dir-a", help="folder with mitsuba pngs")
    ap.add_argument("--dir-b", help="folder with blender pngs")
    ap.add_argument("--gain", type=float, default=50.0,
                    help="difference amplifier (default 50)")
    ap.add_argument("--out", default=None, help="output folder")
    args = ap.parse_args()

    if args.dir_a and args.dir_b:
        outdir = args.out or os.path.join(args.dir_b, "diff")
        os.makedirs(outdir, exist_ok=True)
        for fa in sorted(glob.glob(os.path.join(args.dir_a, "*.png"))):
            if os.path.basename(fa).startswith("checker"):
                continue
            fb = os.path.join(args.dir_b, os.path.basename(fa))
            if not os.path.exists(fb):
                print(f"skip {fa}: no counterpart in {args.dir_b}")
                continue
            try:
                make_diff(fa, fb, outdir, args.gain)
            except RuntimeError as e:
                print(f"FAIL {os.path.basename(fa)}: {e}")
    elif args.a and args.b:
        outdir = args.out or os.path.dirname(os.path.abspath(args.b))
        os.makedirs(outdir, exist_ok=True)
        make_diff(args.a, args.b, outdir, args.gain)
    else:
        ap.error("pass two images or --dir-a/--dir-b")


if __name__ == "__main__":
    main()
