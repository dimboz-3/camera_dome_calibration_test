#!/usr/bin/env python3
"""Image comparison app (app 2).

Detects checkerboard inner corners in two images (e.g. Mitsuba render and
Blender/Cycles render of the same scene) and reports the per-corner pixel
difference. The corners must have the same inner-corner pattern (from JSON).

Usage:
    python compare.py <mitsuba.png> <blender.png> [--json scene.json] [--out OUT]
    python compare.py --dir-a data --dir-b data/render [--out data/compare]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np


def cells_arg(text: str):
    cols, rows = text.lower().split("x")
    return (int(cols), int(rows))


def json_for(png_path: str):
    j = os.path.splitext(png_path)[0] + ".json"
    return j if os.path.exists(j) else None


def pattern_from_json(jpath: str):
    with open(jpath) as f:
        cfg = json.load(f)
    cols, rows = cfg["board"]["cells"]
    return (cols - 1, rows - 1)


def detect_corners(path: str, pattern):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read image {path}")

    ok, corners = cv2.findChessboardCornersSB(img, pattern, flags=0)
    if not ok:
        ok, corners = cv2.findChessboardCornersSB(
            img, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            img, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
            corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), crit)
    if not ok:
        raise RuntimeError(f"checker corners not found in {path} (pattern {pattern})")
    return corners.reshape(-1, 2).astype(np.float64)


def dihedral_candidates(g: np.ndarray, pr: int, pc: int) -> dict:
    """All 8 row/col flips of the corner grid plus the same 8 transposed.

    cv2 returns the grid starting from an arbitrary corner and may run the
    traversal column-major (it does so for boards imaged rotated by 90 deg),
    so the raw ordering can be any of these 16 index permutations of the
    canonical row-major top-left ordering.
    """
    gg = g.reshape(pr, pc, 2)
    tg = gg.transpose(1, 0, 2)
    out = {}
    for name, m in (("id", gg), ("fx", gg[:, ::-1]), ("fy", gg[::-1]),
                    ("r180", gg[::-1, ::-1]), ("t_id", tg), ("t_fx", tg[:, ::-1]),
                    ("t_fy", tg[::-1]), ("t_r180", tg[::-1, ::-1])):
        out[name] = m.reshape(-1, 2)
    return out


def canonical_order(corners: np.ndarray, pattern, gt_proj=None):
    """Reorder detected corners to canonical row-major top-left ordering.

    With a pinhole projection of the JSON corners_gt available, the candidate
    whose corners best match it wins (exact). Otherwise a directional
    heuristic is used: image rows must run +x and columns +y.
    """
    pr, pc = pattern[1], pattern[0]
    cands = dihedral_candidates(corners, pr, pc)
    if gt_proj is not None and gt_proj.shape == corners.shape:
        scored = {k: float(np.linalg.norm(gt_proj - v, axis=1).mean())
                  for k, v in cands.items()}
        best = min(scored, key=scored.get)
        return cands[best], best
    scored = {}
    for k, v in cands.items():
        gg = v.reshape(pr, pc, 2)
        scored[k] = float(np.diff(gg, axis=1)[..., 0].sum()
                          + np.diff(gg, axis=0)[..., 1].sum())
    best = max(scored, key=scored.get)
    return cands[best], best


def gt_projection(jpath: str, pattern):
    """Pinhole projection of board.corners_gt using the JSON camera model.

    The board sits inside the dome (corner rays aside, the board centre
    region is seen directly), so the plain pinhole model is the right
    reference for ordering; returned None on any mismatch.
    """
    try:
        with open(jpath) as f:
            cfg = json.load(f)
        gt = np.asarray(cfg["board"]["corners_gt"], float)
        if gt.shape != (pattern[0] * pattern[1], 3):
            return None
        cam = cfg["camera"]
        pos = np.asarray(cam["position"], float)
        look = np.asarray(cam["look_dir"], float)
        look = look / np.linalg.norm(look)
        up = np.asarray(cam["up"], float)
        right = (np.asarray(cam["right"], float) if "right" in cam
                 else np.cross(look, up))
        right = right / np.linalg.norm(right)
        fpx = float(cam["focal_px"])
        cx, cy = (float(v) for v in cam["principal_px"])
        d = gt - pos
        z = d @ look
        return np.stack([cx + fpx * (d @ right) / z,
                         cy - fpx * (d @ up) / z], axis=1)
    except (KeyError, ValueError, OSError):
        return None

def overlay(a_path: str, corners_a, corners_b, out_path: str, errs) -> None:
    img = cv2.imread(a_path, cv2.IMREAD_COLOR)
    for (x, y), e in zip(corners_a, errs):
        color = (0, 255, 0) if e < 1.0 else (0, 165, 255) if e < 3.0 else (0, 0, 255)
        cv2.circle(img, (int(round(x)), int(round(y))), 5, color, 1, cv2.LINE_AA)
    for (x, y) in corners_b:
        cv2.drawMarker(img, (int(round(x)), int(round(y))), (0, 0, 255),
                       cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    for (xa, ya), (xb, yb) in zip(corners_a, corners_b):
        cv2.line(img, (int(round(xa)), int(round(ya))),
                 (int(round(xb)), int(round(yb))), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def compare_pair(a_path: str, b_path: str, pattern, outdir: str,
                 gt_proj=None) -> dict:
    ca, order_a = canonical_order(detect_corners(a_path, pattern), pattern, gt_proj)
    cb, order_b = canonical_order(detect_corners(b_path, pattern), pattern, gt_proj)
    print(f"  corner order: A={order_a} B={order_b}")
    if ca.shape != cb.shape:
        raise RuntimeError(
            f"corner count mismatch {a_path}: {ca.shape} vs {b_path}: {cb.shape}"
        )
    errs = np.linalg.norm(ca - cb, axis=1)
    stats = {
        "n": int(errs.size),
        "mean": float(errs.mean()),
        "median": float(np.median(errs)),
        "p95": float(np.percentile(errs, 95)),
        "max": float(errs.max()),
        "rms": float(np.sqrt((errs ** 2).mean())),
    }

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a_path))[0]
    overlay_path = os.path.join(outdir, stem + "_overlay.png")
    overlay(a_path, ca, cb, overlay_path, errs)

    csv_path = os.path.join(outdir, stem + "_corners.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "ax", "ay", "bx", "by", "err_px"])
        for i, ((ax, ay), (bx, by), e) in enumerate(zip(ca, cb, errs)):
            w.writerow([i, f"{ax:.3f}", f"{ay:.3f}", f"{bx:.3f}", f"{by:.3f}",
                        f"{e:.4f}"])

    return {"stats": stats, "overlay": overlay_path, "csv": csv_path}


def append_summary(outdir: str, row: dict) -> None:
    path = os.path.join(outdir, "summary.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["basename", "n", "mean_px", "median_px", "p95_px",
                        "max_px", "rms_px"])
        w.writerow([row["basename"], row["n"], f"{row['mean']:.4f}",
                    f"{row['median']:.4f}", f"{row['p95']:.4f}",
                    f"{row['max']:.4f}", f"{row['rms']:.4f}"])


def run_one(a_path, b_path, args, outdir):
    pattern = None
    j = None
    if args.cells:
        pattern = (args.cells[0] - 1, args.cells[1] - 1)
    else:
        j = json_for(a_path) or json_for(b_path)
        if j:
            pattern = pattern_from_json(j)
    if pattern is None:
        pattern = (27, 19)
        print("note: no JSON sidecar found, assuming cells 28x20 -> pattern (27,19)")
    gt_proj = gt_projection(j, pattern) if j else None

    base = os.path.basename(a_path)
    try:
        res = compare_pair(a_path, b_path, pattern, outdir, gt_proj)
    except RuntimeError as e:
        print(f"{base}: FAIL ({e})")
        append_summary(outdir, {"basename": base, "n": 0, "mean": -1, "median": -1,
                                "p95": -1, "max": -1, "rms": -1})
        return
    s = res["stats"]
    print(f"{base}: n={s['n']}  mean={s['mean']:.3f}px  median={s['median']:.3f}px  "
          f"p95={s['p95']:.3f}px  max={s['max']:.3f}px  rms={s['rms']:.3f}px")
    print(f"  overlay: {res['overlay']}")
    append_summary(outdir, {"basename": base, **s})


def main():
    ap = argparse.ArgumentParser(description="compare checker corners of two renders")
    ap.add_argument("a", nargs="?", help="image A (e.g. mitsuba png)")
    ap.add_argument("b", nargs="?", help="image B (e.g. blender png)")
    ap.add_argument("--dir-a", help="folder with mitsuba pngs")
    ap.add_argument("--dir-b", help="folder with blender pngs")
    ap.add_argument("--cells", type=cells_arg, default=None, metavar="COLSxROWS",
                    help="board cells; default: read from JSON sidecar")
    ap.add_argument("--out", default=None, help="output folder for overlays/csv")
    args = ap.parse_args()

    if args.dir_a and args.dir_b:
        a_files = sorted(glob.glob(os.path.join(args.dir_a, "*.png")))
        outdir = args.out or "data/compare"
        for fa in a_files:
            fb = os.path.join(args.dir_b, os.path.basename(fa))
            if not os.path.exists(fb):
                print(f"skip {fa}: no counterpart in {args.dir_b}")
                continue
            run_one(fa, fb, args, outdir)
    elif args.a and args.b:
        run_one(args.a, args.b, args, args.out or "data/compare")
    else:
        ap.error("pass two images or --dir-a/--dir-b")


if __name__ == "__main__":
    main()