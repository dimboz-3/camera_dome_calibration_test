#!/usr/bin/env python3
"""Stock Mitsuba 3 quickstart render (Cornell box) - pipeline sanity check.

Code taken verbatim from the official docs notebook
mitsuba-tutorials/quickstart/mitsuba_quickstart.ipynb
(load_file -> mi.render -> mi.util.write_bitmap), with exactly two
deviations, required by this environment / current task:
  1. variant 'scalar_rgb' instead of ('cuda_ad_rgb', 'metal_ad_rgb',
     'llvm_ad_rgb'): no CUDA/GPU backends in this codespace; scalar_rgb
     matches portview/generate.py.
  2. spp=1 instead of 256: pipeline validation only (user directive:
     1 sample is enough until the port scene is fully ported).
"""
import os

import mitsuba as mi

# deviation 1
mi.set_variant("scalar_rgb")

root = os.path.dirname(os.path.abspath(__file__))
scene = mi.load_file(os.path.join(root, "scenes", "cbox", "cbox.xml"))

# deviation 2
image = mi.render(scene, spp=1)

out = os.path.join(root, "data", "cbox_spp1")
mi.util.write_bitmap(out + ".png", image)
mi.util.write_bitmap(out + ".exr", image)
print("wrote", out + ".png")
print("wrote", out + ".exr")
