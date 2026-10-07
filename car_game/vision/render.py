"""Legacy top-down software renderer used as an alternative AI "camera".

The default AI view is now the 3D first-person renderer in
:mod:`car_game.vision.fp_render`.  This module keeps the old car-centric
top-down frame (used when ``vision_view == "topdown"`` and by the debug
tools); it is also a useful compact ground-truth map for tests.

Original frame description follows.

The frame is car-centric: the own car sits near the bottom, its heading points
up, and the visible strip is biased forward so the model can read the upcoming
corners and obstacles.  Drawing goes through Pillow (already a project
dependency) so it works exactly the same in windowed and head-less runs.

The returned :class:`FrameTransform` is the affine mapping between world
ground coordinates (x, z) and image pixels, which lets the detector project
what it finds on the pixels back into the race world.

NOTE: this module must never import ``car_game.render`` -- that package pulls
in OpenGL at import time, which would break head-less operation.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

# --- palette (kept in sync with car_game/render/scene.py OBSTACLE_COLORS) ---
OFFTRACK = (46, 92, 52)
ASPHALT = (58, 58, 66)
ASPHALT_EDGE = (92, 92, 100)
LINE = (225, 225, 215)
START_LINE = (235, 235, 235)
BUILDING = (120, 124, 134)
TREE = (30, 84, 40)
HAZARD = (54, 44, 34)

OBSTACLE_PALETTE = {
    "cone": (242, 92, 36),
    "barrel": (235, 189, 46),
    "crate": (148, 102, 59),
}
# the two cars use colours that never collide with the obstacle palette
OWN_CAR = (255, 0, 255)
RIVAL_CAR = (0, 235, 255)


@dataclass
class CarView:
    """A car to paint into the frame (decoupled from the physics ``Car``)."""

    x: float
    z: float
    heading: float
    color: tuple            # (r, g, b) 0..255 -- what the *detector* keys on
    is_self: bool = False
    name: str = ""
    # Body colour the human actually sees (0..255), used by the shaded
    # presentation channel only.  ``None`` falls back to ``color``.
    real_color: tuple = None


class FrameTransform:
    """Affine world (x, z) <-> image pixel mapping for one rendered frame."""

    ground_anchor = False         # bbox -> use the centroid as the anchor

    def __init__(self, cx: float, cz: float, heading: float, scale: float,
                 width: int, height: int, behind_m: float):
        self.cx = float(cx)
        self.cz = float(cz)
        self.heading = float(heading)
        self.cos = math.cos(self.heading)
        self.sin = math.sin(self.heading)
        self.scale = float(scale)          # pixels per metre
        self.width = int(width)
        self.height = int(height)
        self.behind_m = float(behind_m)

    # world -> local (forward, left) --------------------------------
    def world_to_local(self, x: float, z: float):
        dx = float(x) - self.cx
        dz = float(z) - self.cz
        fwd = dx * self.cos + dz * self.sin
        left = -dx * self.sin + dz * self.cos
        return fwd, left

    def local_to_world(self, fwd: float, left: float):
        x = self.cx + fwd * self.cos - left * self.sin
        z = self.cz + fwd * self.sin + left * self.cos
        return x, z

    # world <-> pixel ----------------------------------------------
    def world_to_pixel(self, x: float, z: float):
        fwd, left = self.world_to_local(x, z)
        px = self.width * 0.5 + left * self.scale
        py = self.height - (fwd + self.behind_m) * self.scale
        return px, py

    def pixel_to_local(self, px: float, py: float):
        left = (float(px) - self.width * 0.5) / self.scale
        fwd = (self.height - float(py)) / self.scale - self.behind_m
        return fwd, left

    def pixel_to_world(self, px: float, py: float):
        fwd, left = self.pixel_to_local(px, py)
        return self.local_to_world(fwd, left)

    def pixels_per_metre(self, fwd: float | None = None) -> float:
        """Affine top-down scale is depth independent."""
        return self.scale


# ---------------------------------------------------------------------------
def render_topdown(track, cars, hazards=(), self_index: int = 0, *,
                   ahead_m: float = 80.0, size=(256, 256),
                   behind_ratio: float = 0.25) -> tuple[Image.Image, FrameTransform]:
    """Render one frame.  Returns ``(PIL.Image RGB, FrameTransform)``.

    ``cars`` is a list of :class:`CarView`; ``self_index`` marks which one is
    the driver (always painted at the bottom centre, pointing up).
    """
    w, h = int(size[0]), int(size[1])
    ahead_m = max(20.0, float(ahead_m))
    behind_m = ahead_m * max(0.05, float(behind_ratio))
    total_m = ahead_m + behind_m
    scale = h / total_m

    own = cars[self_index] if 0 <= self_index < len(cars) else None
    cx = own.x if own is not None else 0.0
    cz = own.z if own is not None else 0.0
    heading = own.heading if own is not None else 0.0
    tf = FrameTransform(cx, cz, heading, scale, w, h, behind_m)

    img = Image.new("RGB", (w, h), OFFTRACK)
    draw = ImageDraw.Draw(img)

    # --- local strip of the track corridor ------------------------
    if track is not None:
        _draw_corridor(draw, track, tf, ahead_m, behind_m, w, h)

    # --- context props (faint, purely so the model sees surroundings)
    if track is not None:
        _draw_props(draw, track, tf, w, h)

    # --- hazards ---------------------------------------------------
    for hz in hazards or ():
        if isinstance(hz, dict):
            x, z, r = hz.get("x"), hz.get("z"), float(hz.get("radius", 2.5))
        elif isinstance(hz, (tuple, list)) and len(hz) >= 2:
            x, z = hz[0], hz[1]
            r = float(hz[2]) if len(hz) > 2 else 2.5
        else:
            continue
        if x is None or z is None:
            continue
        px, py = tf.world_to_pixel(x, z)
        if _in_view(px, py, w, h, r * scale + 4):
            _disc(draw, px, py, r * scale, HAZARD, (30, 24, 18))

    # --- obstacles -------------------------------------------------
    if track is not None:
        for obs in track.obstacles:
            px, py = tf.world_to_pixel(obs.x, obs.z)
            rpx = float(obs.radius) * scale
            if not _in_view(px, py, w, h, rpx + 4):
                continue
            rgb = OBSTACLE_PALETTE.get(obs.kind, OBSTACLE_PALETTE["crate"])
            _disc(draw, px, py, rpx, rgb, _shade(rgb, 0.6))

    # --- cars ------------------------------------------------------
    for i, car in enumerate(cars):
        px, py = tf.world_to_pixel(car.x, car.z)
        if not _in_view(px, py, w, h, 16 * scale + 4):
            continue
        _draw_car(draw, car, tf, px, py)

    return img, tf


# ---------------------------------------------------------------------------
def _shade(rgb, f):
    return tuple(int(max(0, min(255, c * f))) for c in rgb)


def _in_view(px, py, w, h, margin=0.0):
    return (-margin <= px <= w + margin) and (-margin <= py <= h + margin)


def _disc(draw, px, py, r, fill, outline):
    r = max(1.5, float(r))
    draw.ellipse([px - r, py - r, px + r, py + r], fill=fill, outline=outline,
                 width=1)


def _draw_car(draw, car: CarView, tf: FrameTransform, px, py):
    """An oriented rectangle so the model can read the car's heading."""
    fwd = (math.cos(car.heading), math.sin(car.heading))
    left = (-math.sin(car.heading), math.cos(car.heading))
    half_len = 2.1 * tf.scale
    half_wid = 1.0 * tf.scale
    pts = []
    for sf, sl in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        fx = car.x + fwd[0] * sf * 2.1 + left[0] * sl * 1.0
        fz = car.z + fwd[1] * sf * 2.1 + left[1] * sl * 1.0
        pts.append(tf.world_to_pixel(fx, fz))
    draw.polygon(pts, fill=car.color, outline=(255, 255, 255))
    # a small nose marker at the front so heading is unambiguous
    nx, ny = tf.world_to_pixel(car.x + fwd[0] * 2.6, car.z + fwd[1] * 2.6)
    draw.ellipse([nx - 1.6, ny - 1.6, nx + 1.6, ny + 1.6], fill=(20, 20, 20))


def _corridor_samples(track, tf: FrameTransform, ahead_m, behind_m):
    """Centre/left/right pixels for the visible strip of track."""
    try:
        i0, t0, _, _, _, _ = track.nearest((tf.cx, tf.cz), None)
    except Exception:
        return [], [], []
    base_arc = track.arc_at(i0, t0)
    half = track.half_width
    ds = max(1.5, ahead_m / 48.0)
    s = base_arc - behind_m - ahead_m * 0.15
    end = base_arc + ahead_m + ahead_m * 0.15
    left_pts, right_pts, center_pts = [], [], []
    while s <= end:
        i, _, p = track.point_at_arc(s)
        nrm = track.normal[i]
        lx, lz = float(p[0] + nrm[0] * half), float(p[1] + nrm[1] * half)
        rx, rz = float(p[0] - nrm[0] * half), float(p[1] - nrm[1] * half)
        left_pts.append(tf.world_to_pixel(lx, lz))
        right_pts.append(tf.world_to_pixel(rx, rz))
        center_pts.append(tf.world_to_pixel(float(p[0]), float(p[1])))
        s += ds
    return left_pts, right_pts, center_pts


def _draw_corridor(draw, track, tf, ahead_m, behind_m, w, h):
    left_pts, right_pts, center_pts = _corridor_samples(track, tf, ahead_m, behind_m)
    if len(left_pts) < 2:
        return
    poly = list(left_pts) + list(reversed(right_pts))
    draw.polygon(poly, fill=ASPHALT)
    # kerb edges
    draw.line(left_pts, fill=ASPHALT_EDGE, width=2)
    draw.line(right_pts, fill=ASPHALT_EDGE, width=2)
    # dashed centre line
    for k in range(0, len(center_pts) - 1, 2):
        draw.line([center_pts[k], center_pts[k + 1]], fill=LINE, width=1)
    _draw_start_line(draw, track, tf, ahead_m, behind_m)


def _draw_start_line(draw, track, tf, ahead_m, behind_m):
    try:
        i0, t0, _, _, _, _ = track.nearest((tf.cx, tf.cz), None)
        base_arc = track.arc_at(i0, t0)
    except Exception:
        return
    n = track.n
    for lap in range(-1, 3):
        s = lap * track.length
        if not (-behind_m - 5 <= s - base_arc <= ahead_m + 5):
            continue
        lx = tf.world_to_pixel(float(track.left[0][0]), float(track.left[0][1]))
        rx = tf.world_to_pixel(float(track.right[0][0]), float(track.right[0][1]))
        draw.line([lx, rx], fill=START_LINE, width=3)
        break


def _draw_props(draw, track, tf, w, h):
    for b in getattr(track, "buildings", []) or ():
        px, py = tf.world_to_pixel(b.x, b.z)
        if not _in_view(px, py, w, h, max(b.sx, b.sz) * tf.scale):
            continue
        sx = b.sx * tf.scale * 0.5
        sz = b.sz * tf.scale * 0.5
        draw.rectangle([px - sx, py - sz, px + sx, py + sz], fill=BUILDING)
    for t in getattr(track, "trees", []) or ():
        px, py = tf.world_to_pixel(t.x, t.z)
        r = t.size * tf.scale * 0.35
        if _in_view(px, py, w, h, r + 2):
            _disc(draw, px, py, r, TREE, _shade(TREE, 0.7))


# ---------------------------------------------------------------------------
def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_numpy(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.int16)
