#!/usr/bin/env python3
"""Increment B: the board ALONE in an empty scene (spp=1, 256px).

Isolates the board recipe (OBJ quad + bitmap emitter + diffuse) from the
cbox environment. Renders BOTH windings (u=+X and u=-X): the correct one
shows the checker, the wrong one is culled -> pure black background.
"""
import os

import numpy as np

import scene
import mitsuba as mi

root = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(root, "data")

tex_path = os.path.join(data, "checker_28x20.png")
if not os.path.exists(tex_path):
    scene.write_checker_texture(tex_path, 28, 20)

mi.set_variant("scalar_rgb")

FOV = 39.3077  # same as cbox camera for identical framing


def build(u_sign: float):
    board = {
        "center": np.zeros(3),
        "u": np.array([u_sign, 0.0, 0.0]),
        "v": np.array([0.0, 1.0, 0.0]),
    }
    obj_path = os.path.join(data, f"board_u{'p' if u_sign > 0 else 'm'}.obj")
    with open(obj_path, "w") as f:
        f.write(scene.board_obj_text(board, 28 * 0.06, 20 * 0.06))
    tex = {"type": "bitmap", "filename": tex_path, "filter_type": "nearest"}
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        "sensor": {
            "type": "perspective",
            "fov": FOV,
            "fov_axis": "smaller",
            "near_clip": 0.001,
            "far_clip": 100.0,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]
            ),
            "sampler": {"type": "independent", "sample_count": 1},
            "film": {
                "type": "hdrfilm",
                "width": 256,
                "height": 256,
                "pixel_format": "rgb",
                "rfilter": {"type": "tent"},
                "component_format": "float32",
            },
        },
        "board": {
            "type": "obj",
            "filename": obj_path,
            "bsdf": {"type": "diffuse", "reflectance": tex},
            "emitter": {"type": "area", "radiance": tex},
        },
    }


for sign in (1.0, -1.0):
    sc = mi.load_dict(build(sign))
    image = mi.render(sc, spp=1)
    name = f"board_only_u{'p' if sign > 0 else 'm'}_spp1"
    out_png = os.path.join(data, name + ".png")
    mi.util.write_bitmap(out_png, image)
    print("wrote", out_png)