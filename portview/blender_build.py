#!/usr/bin/env python3
"""Blender scene builder (the independent "arbiter" renderer).

Reads the JSON scene description produced by generate.py and rebuilds the exact
same scene in Blender (Cycles), then renders the camera image.

Usage:
    blender -b -P blender_build.py -- <scene.json | folder> [-o outdir] [--samples N]
    python blender_build.py <scene.json | folder> [-o outdir] [--samples N]

Batch: pass a folder -> every *.json in it is rendered in one Blender session.
Output: <outdir>/<basename>.png (same basename as the Mitsuba PNG + JSON).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args(argv):
    json_path = None
    outdir = None
    samples = 128
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-o":
            outdir = argv[i + 1]
            i += 2
        elif a == "--samples":
            samples = int(argv[i + 1])
            i += 2
        else:
            json_path = a
            i += 1
    if not json_path:
        raise SystemExit(
            "usage: blender_build.py <scene.json | folder> [-o outdir] [--samples N]"
        )
    return json_path, outdir, samples


def wipe_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def make_glass_material(ior):
    mat = bpy.data.materials.new("glass")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = ior
    nt.links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_shell(r_outer, thickness, ior):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r_outer, segments=768, ring_count=384,
                                         location=(0.0, 0.0, 0.0))
    shell = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    mod = shell.modifiers.new("shell", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = -1.0  # inward -> inner radius = r_outer - thickness (exact)
    bpy.context.view_layer.objects.active = shell
    bpy.ops.object.modifier_apply(modifier=mod.name)
    shell.data.materials.append(make_glass_material(ior))
    return shell

def make_board_material(tex_path):
    mat = bpy.data.materials.new("board")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(tex_path)
    tex.interpolation = "Closest"
    tex.extension = "CLIP"
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def add_board(cfg, tex_path):
    b = cfg["board"]
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=b["center"])
    obj = bpy.context.active_object
    # plane local X/Y spans -1..1; after ry(-90): local X -> world Z, local Y -> world Y
    obj.scale = (b["height"] / 2.0, b["width"] / 2.0, 1.0)
    obj.rotation_euler = (0.0, -math.pi / 2.0, 0.0)  # plane normal -> world -X
    obj.data.materials.append(make_board_material(tex_path))
    return obj


def add_camera(cfg):
    cam_cfg = cfg["camera"]
    cam_data = bpy.data.cameras.new("cam")
    sensor_w = 36.0
    cam_data.sensor_width = sensor_w
    cam_data.sensor_fit = "AUTO"
    cam_data.lens = cam_cfg["focal_px"] * sensor_w / cam_cfg["width"]
    cam_data.clip_start = 0.002  # camera sits ~1 cm from the glass
    cam_data.clip_end = 2000.0
    obj = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = cam_cfg["position"]
    quat = Vector(cam_cfg["look_dir"]).to_track_quat("-Z", "Y")
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = quat
    bpy.context.scene.camera = obj
    return obj


def set_world_black():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)


def configure_render(resolution, samples, out_png):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.device = "CPU"
    sc.render.resolution_x = resolution[0]
    sc.render.resolution_y = resolution[1]
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = out_png


def render_json(json_path, outdir, samples):
    with open(json_path) as f:
        cfg = json.load(f)

    tex_name = cfg["board"].get("texture", "checker.png")
    tex_path = os.path.join(os.path.dirname(json_path), tex_name)
    if not os.path.exists(tex_path):
        raise FileNotFoundError(f"checker texture not found: {tex_path}")

    out_png = os.path.join(
        outdir, os.path.splitext(os.path.basename(json_path))[0] + ".png")

    wipe_scene()
    dome = cfg["dome"]
    add_shell(dome["r_outer"], dome["thickness"], dome["ior_glass"])
    add_board(cfg, tex_path)
    add_camera(cfg)
    set_world_black()
    configure_render(
        (cfg["camera"]["width"], cfg["camera"]["height"]), samples, out_png)

    bpy.ops.render.render(write_still=True)
    print("rendered ->", out_png)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    json_path, outdir, samples = parse_args(argv)

    if os.path.isdir(json_path):
        files = sorted(glob.glob(os.path.join(json_path, "*.json")))
        if not files:
            raise SystemExit(f"no .json files in {json_path}")
        outdir = outdir or os.path.join(json_path, "render")
    else:
        files = [json_path]
        outdir = outdir or os.path.join(
            os.path.dirname(os.path.abspath(json_path)), "render")

    os.makedirs(outdir, exist_ok=True)
    for f in files:
        print("scene:", f)
        render_json(f, outdir, samples)


main()