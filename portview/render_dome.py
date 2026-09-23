#!/usr/bin/env python3
"""Increment C: board + dome glass (two dielectric shells), real camera.

Ladder so far: cbox -> board in cbox -> board alone (winding: u x v must
face the camera; empirically fixed) -> THIS: add the dome exactly as in
generate.py, with the real camera geometry (camera_pose/place_board/fovs),
but at quarter test resolution and spp=1.

Renders TWO images for A/B comparison:
  dome_board_noglass_spp1.png - board only (no dome)
  dome_board_glass_spp1.png   - board behind the dome glass
"""
import argparse
import os

import numpy as np

import mitsuba as mi
import scene
from scene import SceneParams, camera_pose, place_board, fovs

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--scale", type=int, default=1, help="resolution multiplier (base 480x384)")
ap.add_argument("--spp", type=int, default=1, help="samples per pixel (validation: 1)")
args = ap.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(root, "data")

tex_path = os.path.join(data, "checker_28x20.png")
if not os.path.exists(tex_path):
    scene.write_checker_texture(tex_path, 28, 20)

# quarter test resolution, 1 spp per directive
p = SceneParams(width=480 * args.scale, height=384 * args.scale, samples=args.spp)

board = place_board(p)
obj_path = os.path.join(data, "board_dome.obj")
with open(obj_path, "w") as f:
    f.write(scene.board_obj_text(board, board["width"], board["height"]))

mi.set_variant("scalar_rgb")
pos, look, up = camera_pose(p)
target = (pos + look * 10.0).tolist()
fov_x, _ = fovs(p)

tex = {"type": "bitmap", "filename": tex_path, "filter_type": "nearest"}
board_shape = {
    "type": "obj",
    "filename": obj_path,
    "emitter": {"type": "area", "radiance": tex},
    "bsdf": {"type": "diffuse", "reflectance": tex},
}
sensor = {
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
}

dome = {
    "dome_outer": {
        "type": "sphere",
        "center": [0.0, 0.0, 0.0],
        "radius": p.r_outer,
        "bsdf": {"type": "dielectric", "int_ior": p.ior_glass, "ext_ior": p.ior_env},
    },
    "dome_inner": {
        "type": "sphere",
        "center": [0.0, 0.0, 0.0],
        "radius": p.r_inner,
        # camera-side medium is air (ior_env); the gap between the two
        # spheres IS the glass layer -> outside of this interface is glass
        "bsdf": {"type": "dielectric", "int_ior": p.ior_env, "ext_ior": p.ior_glass},
    },
}

for tag, extra in (("noglass", {}), ("glass", dome)):
    sc = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 12},
            "sensor": sensor,
            "board": board_shape,
            **extra,
        }
    )
    image = mi.render(sc, spp=args.spp, seed=p.seed)
    out_png = os.path.join(data, f"dome_board_{tag}_spp{args.spp}.png")
    mi.util.write_bitmap(out_png, image)
    print("wrote", out_png)