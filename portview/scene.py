"""Scene geometry: camera behind a thick spherical glass viewport (armored dome).

Frame convention (car-like, right-handed):
  X - forward, Y - left, Z - up. Origin = car center = dome center.
The camera looks along +X (orientation fixed for now), its position is given in
polar coordinates relative to the dome (pitch, azimuth) plus a radial depth
inward from the inner glass surface.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

LOOK = np.array([1.0, 0.0, 0.0])  # camera looks along +X
UP = np.array([0.0, 0.0, 1.0])    # camera "up" is +Z


def timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


@dataclass
class SceneParams:
    # dome (glass shell)
    r_inner: float = 6.0        # inner radius, m
    thickness: float = 0.05     # glass thickness, m
    ior_glass: float = 1.5
    ior_env: float = 1.0        # air on both sides
    # camera
    pitch_deg: float = 45.0     # polar elevation (z = x at 45)
    azimuth_deg: float = 0.0
    depth_inside: float = 0.01  # m inward from the inner surface
    width: int = 1920
    height: int = 1535
    focal_px: float = 1036.0
    # checkerboard
    cells: tuple = (28, 20)     # (cols, rows)
    cell_size: float = 0.1      # m
    coverage: float = 0.8       # target frame coverage
    # render
    samples: int = 256
    seed: int = 0

    @property
    def r_outer(self) -> float:
        return self.r_inner + self.thickness

    @property
    def cols(self) -> int:
        return int(self.cells[0])

    @property
    def rows(self) -> int:
        return int(self.cells[1])


def camera_pose(p: SceneParams):
    """Camera position from polar coordinates + fixed orientation."""
    r = p.r_inner - p.depth_inside
    az = math.radians(p.azimuth_deg)
    el = math.radians(p.pitch_deg)
    radial = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ])
    pos = radial * r
    return pos, LOOK.copy(), UP.copy()


def image_right(look: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Image +u direction (to the right). For look=+X, up=+Z it is -Y."""
    right = np.cross(look, up)
    return right / np.linalg.norm(right)


def place_board(p: SceneParams) -> dict:
    """Board strictly in front of the camera, plane perpendicular to look dir.

    Pinhole 80% rule (general form): the board must subtend `coverage` of the
    frame in BOTH dims, so the governing dimension gives
        D = cell_size * f / (coverage * max(W/cols, H/rows)).
    Refraction only compresses angles -> real coverage >= pinhole estimate.
    Distance is clamped so the board stays fully outside the dome.
    """
    pos, look, up = camera_pose(p)
    u = image_right(look, up)
    v = UP.copy()
    cols, rows = p.cols, p.rows
    s = p.cell_size
    board_w = cols * s
    board_h = rows * s

    d = s * p.focal_px / (p.coverage * max(p.width / cols, p.height / rows))

    def board_points(d):
        c = pos + d * look
        pts = [c + u * (board_w / 2) + v * (board_h / 2),
               c - u * (board_w / 2) + v * (board_h / 2),
               c + u * (board_w / 2) - v * (board_h / 2),
               c - u * (board_w / 2) - v * (board_h / 2),
               c]
        return c, np.array(pts)

    margin = 0.05
    r_out = p.r_outer
    for _ in range(10000):
        c, pts = board_points(d)
        if np.min(np.linalg.norm(pts, axis=1)) > r_out + margin:
            break
        d += 0.1
    else:
        raise RuntimeError("cannot place board outside the dome, increase cell_size")

    center, pts = board_points(d)
    cov_u = (board_w / 2 / d) / (p.width / 2 / p.focal_px)
    cov_v = (board_h / 2 / d) / (p.height / 2 / p.focal_px)
    return {
        "center": center,
        "u": u,
        "v": v,
        "normal": -look,
        "distance": float(d),
        "width": board_w,
        "height": board_h,
        "coverage_u": float(cov_u),
        "coverage_v": float(cov_v),
        "min_dist_to_origin": float(np.min(np.linalg.norm(pts, axis=1))),
    }

def board_corners_gt(p: SceneParams, board: dict) -> list:
    """World coordinates of inner checker corners ((cols-1)*(rows-1) points).

    Ordered row-major: j = row index from the top, i = column index from the
    left of the image. Image +u is `board['u']`, rows go downward in the image.
    """
    cols, rows, s = p.cols, p.rows, p.cell_size
    u, v, c = board["u"], board["v"], board["center"]
    out = []
    for j in range(rows - 1):
        for i in range(cols - 1):
            du = (i + 1 - cols / 2.0) * s
            dv = (rows / 2.0 - (j + 1)) * s
            pt = c + u * du + v * dv
            out.append([float(pt[0]), float(pt[1]), float(pt[2])])
    return out


def write_checker_texture(path: str, cols: int, rows: int, cell_px: int = 32) -> None:
    """Checker PNG used by BOTH engines: border is black, cell (0,0) is white."""
    import cv2  # local import: only the generator needs this

    border = max(8, cell_px // 2)
    w = cols * cell_px + 2 * border
    h = rows * cell_px + 2 * border
    img = np.zeros((h, w), np.uint8)  # black border / background
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                y0 = border + j * cell_px
                x0 = border + i * cell_px
                img[y0:y0 + cell_px, x0:x0 + cell_px] = 255
    if not cv2.imwrite(path, img):
        raise RuntimeError(f"cannot write texture {path}")


def board_obj_text(board: dict, board_w: float, board_h: float) -> str:
    """Minimal OBJ quad for Mitsuba: exact vertices + UV 0..1, normal faces camera."""
    c, u, v = board["center"], board["u"], board["v"]
    p_tl = c - u * board_w / 2 + v * board_h / 2
    p_tr = c + u * board_w / 2 + v * board_h / 2
    p_br = c + u * board_w / 2 - v * board_h / 2
    p_bl = c - u * board_w / 2 - v * board_h / 2

    def fmt(x):
        return "%.9g" % x

    lines = []
    for pt in (p_tl, p_tr, p_br, p_bl):
        lines.append("v {} {} {}".format(fmt(pt[0]), fmt(pt[1]), fmt(pt[2])))
    for vt in ("0 1", "1 1", "1 0", "0 0"):
        lines.append("vt " + vt)
    # Face traversal tl -> bl -> br -> tr: right-hand normal = -u_axis x v_axis
    # -> faces the camera (-look). Mitsuba area emitters emit front-side only.
    lines.append("f 1/1 4/4 3/3 2/2")
    return "\n".join(lines) + "\n"


def fovs(p: SceneParams):
    fov_x = math.degrees(2 * math.atan((p.width / 2) / p.focal_px))
    fov_y = math.degrees(2 * math.atan((p.height / 2) / p.focal_px))
    return fov_x, fov_y


def build_json(p: SceneParams, board: dict, index: int, ts: str, texture_name: str) -> dict:
    pos, look, up = camera_pose(p)
    right = image_right(look, up)
    fov_x, fov_y = fovs(p)
    return {
        "meta": {"app": "portview", "version": "0.1", "index": index, "timestamp": ts},
        "units": "meters",
        "frame": {"x": "forward", "y": "left", "z": "up",
                  "origin": "car center = dome center"},
        "dome": {
            "center": [0.0, 0.0, 0.0],
            "r_inner": p.r_inner,
            "thickness": p.thickness,
            "r_outer": p.r_outer,
            "ior_glass": p.ior_glass,
            "ior_env": p.ior_env,
        },
        "camera": {
            "position": [float(x) for x in pos],
            "look_dir": [float(x) for x in look],
            "up": [float(x) for x in up],
            "right": [float(x) for x in right],
            "pitch_deg": p.pitch_deg,
            "azimuth_deg": p.azimuth_deg,
            "depth_inside": p.depth_inside,
            "width": p.width,
            "height": p.height,
            "focal_px": p.focal_px,
            "principal_px": [p.width / 2.0, p.height / 2.0],
            "fov_x_deg": fov_x,
            "fov_y_deg": fov_y,
        },
        "board": {
            "center": [float(x) for x in board["center"]],
            "normal": [float(x) for x in board["normal"]],
            "u_dir": [float(x) for x in board["u"]],
            "v_dir": [float(x) for x in board["v"]],
            "cells": [p.cols, p.rows],
            "cell_size": p.cell_size,
            "width": board["width"],
            "height": board["height"],
            "distance": board["distance"],
            "coverage": {"u": board["coverage_u"], "v": board["coverage_v"]},
            "min_dist_to_origin": board["min_dist_to_origin"],
            "texture": texture_name,
            "corners_gt": board_corners_gt(p, board),
        },
        "render": {"engine": "mitsuba3", "variant": "scalar_rgb",
                   "samples": p.samples, "seed": p.seed},
    }


def summary_lines(p: SceneParams, board: dict) -> list:
    pos, _, _ = camera_pose(p)
    fov_x, fov_y = fovs(p)
    return [
        f"camera position : ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}) m  "
        f"(pitch {p.pitch_deg:.3f} deg)",
        f"camera distance to dome center: {np.linalg.norm(pos):.4f} m",
        f"resolution      : {p.width}x{p.height}, focal {p.focal_px} px, "
        f"fov {fov_x:.2f}/{fov_y:.2f} deg",
        f"dome            : r_inner={p.r_inner} m, thickness={p.thickness} m, "
        f"ior={p.ior_glass}",
        f"board           : {p.cols}x{p.rows} cells, cell {p.cell_size} m, "
        f"size {board['width']:.3f}x{board['height']:.3f} m",
        f"board distance  : {board['distance']:.3f} m from camera, coverage "
        f"u/v = {board['coverage_u']*100:.1f}%/{board['coverage_v']*100:.1f}%",
        f"board min |r|   : {board['min_dist_to_origin']:.3f} m "
        f"(dome r_outer {p.r_outer} m)",
    ]