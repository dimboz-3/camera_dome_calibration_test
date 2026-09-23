#!/usr/bin/env python3
"""Increment A: our checkerboard inside the stock Cornell box (spp=1).

Ladder from the docs example toward the porthole scene:
  cbox (done, render_cbox.py) -> A: our board in cbox -> B: board alone
  -> C: board + dome glass -> ported scene.

Goal here: validate OUR board recipe (OBJ quad + bitmap texture used as
BOTH area emitter and diffuse reflectance, exactly as in generate.py)
inside a known-good scene. The board replaces the cbox back wall:
28x20 cells x 0.06 m = 1.68x1.20 m at z=-0.99, facing +Z (the cbox
camera looks along -Z from (0,0,4), up=+Y).
"""
import argparse
import os

import numpy as np

import scene  # reuse the proven checker texture + OBJ writer
import mitsuba as mi

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--res", type=int, default=256, help="film resolution (doc default 256)")
ap.add_argument("--spp", type=int, default=1, help="samples per pixel (directive: 1)")
ap.add_argument("--usign", type=float, default=1.0, help="+1/-1: board u-axis sign -> winding/normal side")
args = ap.parse_args()
import os

import numpy as np

import scene  # reuse the proven checker texture + OBJ writer
import mitsuba as mi

root = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(root, "data")

tex_path = os.path.join(data, "checker_28x20.png")
if not os.path.exists(tex_path):
    scene.write_checker_texture(tex_path, 28, 20)

# board dict compatible with scene.board_obj_text()
center = np.array([0.0, 0.0, -0.99])
u = np.array([args.usign, 0.0, 0.0])  # sign flips the OBJ winding -> emitter side
v = np.array([0.0, 1.0, 0.0])
board = {"center": center, "u": u, "v": v}
board_w, board_h = 28 * 0.06, 20 * 0.06

obj_path = os.path.join(data, "board_cbox.obj")
with open(obj_path, "w") as f:
    f.write(scene.board_obj_text(board, board_w, board_h))

board_block = f"""
    <shape type="obj" id="board">
        <string name="filename" value="{obj_path}"/>
        <bsdf type="diffuse">
            <texture type="bitmap" name="reflectance">
                <string name="filename" value="{tex_path}"/>
                <string name="filter_type" value="nearest"/>
            </texture>
        </bsdf>
        <emitter type="area">
            <texture type="bitmap" name="radiance">
                <string name="filename" value="{tex_path}"/>
                <string name="filter_type" value="nearest"/>
            </texture>
        </emitter>
    </shape>
"""

xml_src = os.path.join(root, "scenes", "cbox", "cbox.xml")
with open(xml_src) as f:
    xml_text = f.read()
assert "</scene>" in xml_text

xml_path = os.path.join(root, "scenes", "cbox", "cbox_board.xml")
# must live next to cbox.xml: cbox.xml references meshes/ via relative paths,
# which mitsuba resolves against the XML's own directory.
with open(xml_src) as f, open(xml_path, "w") as g:
    g.write(f.read().replace("</scene>", board_block + "</scene>"))

mi.set_variant("scalar_rgb")
# deviation 2 from the doc example (spp) + res override ($res placeholder in cbox.xml)
mi_scene = mi.load_file(xml_path, res=args.res)
image = mi.render(mi_scene, spp=args.spp)

side = "up" if args.usign > 0 else "um"
out = os.path.join(data, f"cbox_board_{side}_spp{args.spp}")
mi.util.write_bitmap(out + ".png", image)
mi.util.write_bitmap(out + ".exr", image)
print("wrote", out + ".png")
print("wrote", out + ".exr")