#!/usr/bin/env python3
"""portview generator (app 1).

Renders what the camera sees behind a thick spherical glass viewport using
Mitsuba 3, and writes a JSON scene description (ground truth) beside the PNG.

Console mode:
    python generate.py                      # defaults = the agreed scene
    python generate.py --pitch 45 --r-inner 6.0 --thickness 0.05 --ior-glass 1.5
    python generate.py --batch 5 --pitch-min 35 --pitch-max 55
    python generate.py --dry-run            # placement only, no render

Output (flat folder):
    <out>/<prefix>_<index:04d>_<timestamp>.png     camera image (Mitsuba)
    <out>/<prefix>_<index:04d>_<timestamp>.json    full scene description (GT)
    <out>/checker_<cols>x<rows>.png                shared checker texture
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene  # noqa: E402
from scene import SceneParams, camera_pose, place_board, fovs  # noqa: E402


def cells_arg(text: str):
    cols, rows = text.lower().split("x")
    return (int(cols), int(rows))


def positive_float(x):
    x = float(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def parse_args():
    ap = argparse.ArgumentParser(
        description="generate camera view behind thick spherical viewport")
    ap.add_argument("--r-inner", type=positive_float, default=6.0)
    ap.add_argument("--thickness", type=positive_float, default=0.05)
    ap.add_argument("--ior-glass", type=positive_float, default=1.5)
    ap.add_argument("--ior-env", type=float, default=1.0)
    ap.add_argument("--pitch", type=float, default=45.0, help="deg, z=x at 45")
    ap.add_argument("--azimuth", type=float, default=0.0, help="deg")
    ap.add_argument("--depth-inside", type=positive_float, default=0.01,
                    help="camera depth inward from inner surface, m")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1535)
    ap.add_argument("--focal", type=positive_float, default=1036.0, help="px")
    ap.add_argument("--cells", type=cells_arg, default=(28, 20), metavar="COLSxROWS")
    ap.add_argument("--cell-size", type=positive_float, default=0.1, help="m")
    ap.add_argument("--coverage", type=float, default=0.8, help="target frame coverage")
    ap.add_argument("--samples", type=int, default=256, help="Mitsuba spp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data", help="flat output folder")
    ap.add_argument("--prefix", default="img")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1,
                    help="generate N images, pitch spread over [--pitch-min; --pitch-max]")
    ap.add_argument("--pitch-min", type=float, default=40.0)
    ap.add_argument("--pitch-max", type=float, default=50.0)
    ap.add_argument("--dry-run", action="store_true", help="print placement and exit")
    return ap.parse_args()

def build_mitsuba_scene(p: SceneParams, board: dict, obj_path: str, tex_path: str):
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    pos, look, up = camera_pose(p)
    target = (pos + look * 10.0).tolist()
    fov_x, _ = fovs(p)

    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 12},
        "dome_outer": {
            "type": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": p.r_outer,
            "bsdf": {"type": "dielectric", "int_ior": p.ior_glass,
                     "ext_ior": p.ior_env},
        },
        "dome_inner": {
            "type": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": p.r_inner,
            # camera-side medium is air (ior_env); the gap between the two
            # spheres IS the glass layer -> outside of this interface is glass
            "bsdf": {"type": "dielectric", "int_ior": p.ior_env,
                     "ext_ior": p.ior_glass},
        },
        "board": {
            "type": "obj",
            "filename": obj_path,
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "bitmap",
                    "filename": tex_path,
                    "filter_type": "nearest",
                },
            },
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "filename": tex_path,
                    "filter_type": "nearest",
                },
            },
        },
        "sensor": {
            "type": "perspective",
            "fov": fov_x,
            "fov_axis": "x",
            "near_clip": 0.002,
            "far_clip": 2000.0,
            "to_world": mi.ScalarTransform4f.look_at(
                [float(x) for x in pos], target, [float(x) for x in up]
            ),
            "film": {
                "type": "hdrfilm",
                "width": p.width,
                "height": p.height,
                "pixel_format": "rgb",
                "component_format": "float32",
            },
            "sampler": {"type": "independent", "sample_count": p.samples},
        },
    }


def generate_one(p: SceneParams, args, index: int, ts: str):
    board = place_board(p)

    os.makedirs(args.out, exist_ok=True)
    basename = f"{args.prefix}_{index:04d}_{ts}"
    png_path = os.path.join(args.out, basename + ".png")
    json_path = os.path.join(args.out, basename + ".json")

    texture_name = f"checker_{p.cols}x{p.rows}.png"
    tex_path = os.path.join(args.out, texture_name)
    if not os.path.exists(tex_path):
        scene.write_checker_texture(tex_path, p.cols, p.rows)

    for line in scene.summary_lines(p, board):
        print("  " + line)

    if args.dry_run:
        return

    obj_text = scene.board_obj_text(board, board["width"], board["height"])
    tmp_fd, tmp_obj = tempfile.mkstemp(suffix=".obj", dir=args.out)
    with os.fdopen(tmp_fd, "w") as f:
        f.write(obj_text)

    try:
        import mitsuba as mi

        scene_dict = build_mitsuba_scene(p, board, tmp_obj, tex_path)
        mi_scene = mi.load_dict(scene_dict)
        os.unlink(tmp_obj)

        img = mi.render(mi_scene, seed=p.seed)
        bitmap = mi.Bitmap(img, channel_names=["R", "G", "B"])
        bitmap = bitmap.convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
        bitmap.write(png_path)
    finally:
        if os.path.exists(tmp_obj):
            os.unlink(tmp_obj)

    data = scene.build_json(p, board, index, ts, texture_name)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  wrote {png_path}")
    print(f"  wrote {json_path}\n")
    return png_path, json_path


def main():
    args = parse_args()

    n = max(1, args.batch)
    lo, hi = args.pitch_min, args.pitch_max
    for k in range(n):
        pitch = args.pitch if n == 1 else lo + (hi - lo) * k / (n - 1)
        p = SceneParams(
            r_inner=args.r_inner,
            thickness=args.thickness,
            ior_glass=args.ior_glass,
            ior_env=args.ior_env,
            pitch_deg=pitch,
            azimuth_deg=args.azimuth,
            depth_inside=args.depth_inside,
            width=args.width,
            height=args.height,
            focal_px=args.focal,
            cells=args.cells,
            cell_size=args.cell_size,
            coverage=args.coverage,
            samples=args.samples,
            seed=args.seed + k,
        )
        ts = scene.timestamp()
        print(f"[{k + 1}/{n}] pitch={pitch:.3f} deg  ->  "
              f"{args.prefix}_{args.index + k:04d}_{ts}")
        generate_one(p, args, args.index + k, ts)


if __name__ == "__main__":
    main()