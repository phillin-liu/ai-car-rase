"""Build the static world mesh for a generated track.

Everything the camera sees outside the two cars is baked into a handful of
``GeoBuffer``s once per match:

* ``mesh``   -- opaque scenery (ground, road, curbs, guardrails, buildings,
  trees, obstacles, start gantry, distant hills),
* ``gloss``  -- parts that want a specular highlight (window bands, lane
  markings, gantry lights),
* ``shadow`` -- soft contact shadows drawn translucent under the props.

All heights come from :meth:`car_game.track.Track.terrain_height_many`, which
batches the cKDTree queries that used to be issued one vertex at a time.
"""
from __future__ import annotations

import math

import numpy as np

from ..palette import (BUILDINGS, CURB_A, CURB_B, FOLIAGE, GROUND, HILL, LINE,
                       OBSTACLE_COLORS, POST, RAIL, ROAD, TRUNK, WINDOW)
from .geometry import GeoBuffer

# The scene colours live in car_game.palette so the software renderer that
# produces the AI's first-person frame can share them; they are re-exported
# here under the names this module has always used.


GROUND_GRID = 56          # samples per side of the terrain height field
GROUND_DROP = 0.9         # metres the ground sits below the track elevation


class Scene:
    def __init__(self):
        self.mesh = GeoBuffer()
        self.gloss = GeoBuffer()
        self.shadow = GeoBuffer()
        self.center = np.zeros(3)
        self.extent = 250.0
        self.vertex_count = 0


def map_extent(track):
    """``(cx, cz, ext)``: map centre and half-size, shared by the builders."""
    cx = float(track.center[:, 0].mean())
    cz = float(track.center[:, 2].mean())
    ext = float(np.max(np.linalg.norm(track.center[:, [0, 2]] -
                                      np.array([cx, cz]), axis=1))) + 150.0
    return cx, cz, ext


def ground_grid(track):
    """The ground height field, exactly as :func:`_build_ground` samples it.

    Returns ``(xs, zs, H)`` where ``H[a, b]`` is the ground height at
    ``(xs[a], zs[b])``.
    """
    cx, cz, ext = map_extent(track)
    xs = np.linspace(cx - ext, cx + ext, GROUND_GRID)
    zs = np.linspace(cz - ext, cz + ext, GROUND_GRID)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return xs, zs, track.terrain_height_many(X, Z) - GROUND_DROP


def _jit(seed: float, amount: float) -> float:
    """Deterministic small colour jitter in [-amount, amount]."""
    h = math.sin(seed * 12.9898) * 43758.5453
    return (h - math.floor(h) - 0.5) * 2.0 * amount


def _tint(color, d):
    return (max(0.0, min(1.0, color[0] + d)),
            max(0.0, min(1.0, color[1] + d)),
            max(0.0, min(1.0, color[2] + d)))


def build_scene(track) -> Scene:
    sc = Scene()
    g = sc.mesh
    n = track.n
    hw = track.half_width
    cx, cz, ext = map_extent(track)
    sc.center = np.array([cx, 0.0, cz])
    sc.extent = ext

    _build_ground(g, track)
    _build_hills(g, track, cx, cz, ext)
    _build_road(g, sc.gloss, track, hw)
    _build_roadside(g, track, hw)
    _build_start(g, sc.gloss, track, hw)
    _build_buildings(g, sc.gloss, sc.shadow, track)
    _build_trees(g, sc.shadow, track)
    _build_obstacles(g, sc.shadow, track)

    sc.vertex_count = len(g.verts) + len(sc.gloss.verts) + len(sc.shadow.verts)
    return sc


# ---------------------------------------------------------------------------
def _build_ground(g: GeoBuffer, track):
    grid = GROUND_GRID
    xs, zs, H = ground_grid(track)
    for a in range(grid - 1):
        for b in range(grid - 1):
            # gentle two-scale variation; a strong per-quad jitter reads as a
            # checkerboard because the grid cells are ~25 m across
            col = _tint(GROUND, _jit(a * 0.37 + b * 0.61, 0.015))
            col = _tint(col, _jit(a * 3.1 - b * 7.7, 0.008))
            p00 = (float(xs[a]), float(H[a, b]), float(zs[b]))
            p01 = (float(xs[a]), float(H[a, b + 1]), float(zs[b + 1]))
            p11 = (float(xs[a + 1]), float(H[a + 1, b + 1]), float(zs[b + 1]))
            p10 = (float(xs[a + 1]), float(H[a + 1, b]), float(zs[b]))
            g.add_quad(p00, p01, p11, p10, col, (0.0, 1.0, 0.0))


def _build_hills(g: GeoBuffer, track, cx, cz, ext):
    rng = np.random.default_rng(991)
    for k in range(40):
        a = 2.0 * math.pi * k / 40.0 + rng.uniform(-0.04, 0.04)
        r = ext * rng.uniform(1.10, 1.55)
        x = cx + r * math.cos(a)
        z = cz + r * math.sin(a)
        h = rng.uniform(22.0, 60.0)
        rad = rng.uniform(34.0, 90.0)
        col = _tint(HILL, rng.uniform(-0.03, 0.03))
        base = track.terrain_height(x, z) - 20.0
        # a squashed cone reads as a distant ridge and fog softens it
        g.add_cylinder(x, z, base, rad, h, col, segments=16, top_radius=rad * 0.05)


def _build_road(g: GeoBuffer, gloss: GeoBuffer, track, hw):
    n = track.n
    # asphalt ribbon with per-segment tonality
    for i in range(n):
        j = (i + 1) % n
        y0 = track.center[i, 1] + 0.05
        y1 = track.center[j, 1] + 0.05
        l0 = (track.left[i, 0], y0, track.left[i, 1])
        r0 = (track.right[i, 0], y0, track.right[i, 1])
        l1 = (track.left[j, 0], y1, track.left[j, 1])
        r1 = (track.right[j, 0], y1, track.right[j, 1])
        col = _tint(ROAD, _jit(i * 17.0, 0.022))
        g.add_quad(r0, l0, l1, r1, col, (0.0, 1.0, 0.0))

    # continuous edge lines just inside the curbs
    for side in (1.0, -1.0):
        off = side * (hw - 0.30)
        left, right = [], []
        for i in range(n + 1):
            k = i % n
            p = track.center_xz[k]
            nrm = track.normal[k]
            y = track.center[k, 1] + 0.065
            left.append((float(p[0] + nrm[0] * (off + 0.13)), y,
                         float(p[1] + nrm[1] * (off + 0.13))))
            right.append((float(p[0] + nrm[0] * (off - 0.13)), y,
                          float(p[1] + nrm[1] * (off - 0.13))))
        gloss.add_ribbon(left, right, LINE, (0.0, 1.0, 0.0))

    # centre dashes
    for i in range(0, n, 10):
        j = (i + 4) % n
        y0 = track.center[i, 1] + 0.07
        y1 = track.center[j, 1] + 0.07
        c0 = (float(track.center[i, 0]), y0, float(track.center[i, 2]))
        c1 = (float(track.center[j, 0]), y1, float(track.center[j, 2]))
        nrm = track.normal[i]
        a = (c0[0] + nrm[0] * 0.20, y0, c0[2] + nrm[1] * 0.20)
        b = (c0[0] - nrm[0] * 0.20, y0, c0[2] - nrm[1] * 0.20)
        c = (c1[0] - nrm[0] * 0.20, y1, c1[2] - nrm[1] * 0.20)
        d = (c1[0] + nrm[0] * 0.20, y1, c1[2] + nrm[1] * 0.20)
        gloss.add_quad(b, a, d, c, (0.86, 0.86, 0.45), (0.0, 1.0, 0.0))


def _build_roadside(g: GeoBuffer, track, hw):
    n = track.n
    # curbs: alternating red/white segments just outside the racing surface
    for side in (1.0, -1.0):
        for i in range(n):
            j = (i + 1) % n
            col = CURB_A if (i // 3) % 2 == 0 else CURB_B
            base = track.left if side > 0 else track.right
            inner = []
            outer = []
            for k in (i, j):
                p = base[k]
                nrm = track.normal[k]
                y = track.center[k, 1] + 0.06
                inner.append((float(p[0]), y, float(p[1])))
                outer.append((float(p[0] + nrm[0] * side * 0.95),
                              y + 0.02,
                              float(p[1] + nrm[1] * side * 0.95)))
            g.add_quad(outer[0], inner[0], inner[1], outer[1], col, (0.0, 1.0, 0.0))

    # guardrails: posts + two horizontal beams, instead of a tall flat wall
    for side in (1.0, -1.0):
        base = track.left if side > 0 else track.right
        off = side * 0.95
        step = max(1, int(round(6.0 / max(track.length / n, 0.5))))
        for i in range(0, n, step):
            p = base[i]
            nrm = track.normal[i]
            x = float(p[0] + nrm[0] * off)
            z = float(p[1] + nrm[1] * off)
            y = track.center[i, 1] + 0.10
            tang = track.tangent[i]
            yaw = math.atan2(float(tang[1]), float(tang[0]))
            g.add_box(x, z, y, 0.16, 0.16, 1.02, yaw, POST)
        for lo, hi in ((0.52, 0.64), (0.94, 1.04)):
            for i in range(n):
                j = (i + 1) % n
                p0, p1 = base[i], base[j]
                n0, n1 = track.normal[i], track.normal[j]
                y0 = track.center[i, 1] + 0.10
                y1 = track.center[j, 1] + 0.10
                a0 = (float(p0[0] + n0[0] * off), y0 + lo, float(p0[1] + n0[1] * off))
                a1 = (float(p1[0] + n1[0] * off), y1 + lo, float(p1[1] + n1[1] * off))
                b1 = (float(p1[0] + n1[0] * off), y1 + hi, float(p1[1] + n1[1] * off))
                b0 = (float(p0[0] + n0[0] * off), y0 + hi, float(p0[1] + n0[1] * off))
                g.add_quad(a0, a1, b1, b0, RAIL, (-n0[0] * side, 0.0, -n0[1] * side))


def _build_start(g: GeoBuffer, gloss: GeoBuffer, track, hw):
    y0 = track.center[0, 1] + 0.085
    nrm = track.normal[0]
    tang = track.tangent[0]
    cols = 8
    step = 2.0 * hw / cols
    for k in range(cols):
        base = -hw + k * step
        for m in range(2):
            t0 = (m - 1) * 1.1
            a0 = (track.center[0, 0] + nrm[0] * base + tang[0] * t0,
                  y0, track.center[0, 2] + nrm[1] * base + tang[1] * t0)
            a1 = (track.center[0, 0] + nrm[0] * (base + step) + tang[0] * t0,
                  y0, track.center[0, 2] + nrm[1] * (base + step) + tang[1] * t0)
            a2 = (a1[0] + tang[0] * 1.1, y0, a1[2] + tang[1] * 1.1)
            a3 = (a0[0] + tang[0] * 1.1, y0, a0[2] + tang[1] * 1.1)
            col = (0.06, 0.06, 0.07) if (k + m) % 2 == 0 else (0.92, 0.92, 0.90)
            gloss.add_quad(a0, a1, a2, a3, col, (0.0, 1.0, 0.0))

    # gantry over the line
    yaw = math.atan2(float(tang[1]), float(tang[0]))
    for s in (1.0, -1.0):
        x = float(track.center[0, 0] + nrm[0] * (hw + 1.1) * s)
        z = float(track.center[0, 2] + nrm[1] * (hw + 1.1) * s)
        g.add_box(x, z, track.center[0, 1], 0.34, 0.34, 5.6, yaw, POST)
    mx = float(track.center[0, 0])
    mz = float(track.center[0, 2])
    y = track.center[0, 1]
    g.add_box(mx, mz, y + 5.6, 0.5, (hw + 1.4) * 2.0, 0.5, yaw, (0.85, 0.14, 0.12))
    g.add_box(mx, mz, y + 4.9, 0.35, (hw + 1.0) * 2.0, 0.7, yaw, (0.10, 0.12, 0.16))


def _build_buildings(g: GeoBuffer, gloss: GeoBuffer, shadow: GeoBuffer, track):
    palette = BUILDINGS
    for idx, b in enumerate(track.buildings):
        base = track.terrain_height(b.x, b.z) - GROUND_DROP
        col = palette[idx % len(palette)]
        g.add_rounded_box(b.x, b.z, base, b.sx, b.sz, b.height, b.yaw, col,
                          radius=min(b.sx, b.sz) * 0.12, corner_seg=2)
        # parapet
        g.add_box(b.x, b.z, base + b.height, b.sx * 1.03, b.sz * 1.03, 0.7,
                  b.yaw, _tint(col, -0.12))
        # window band around the middle of the facade
        h = b.height
        for level in range(1, max(2, int(h / 7.0))):
            y = base + level * 7.0
            if y > base + h - 1.0:
                break
            gloss.add_box(b.x, b.z, y, b.sx * 1.015, b.sz * 1.015, 1.9,
                          b.yaw, WINDOW)
        shadow.add_disc_y(b.x, base + 0.04, b.z,
                          max(b.sx, b.sz) * 0.62, (0.03, 0.04, 0.05), segments=16)


def _build_trees(g: GeoBuffer, shadow: GeoBuffer, track):
    for idx, t in enumerate(track.trees):
        base = track.terrain_height(t.x, t.z) - GROUND_DROP
        size = t.size
        g.add_cylinder(t.x, t.z, base, 0.22, size * 0.55, TRUNK, segments=7)
        foliage = FOLIAGE[idx % len(FOLIAGE)]
        if idx % 3 == 0:                       # round canopy
            g.add_sphere(t.x, base + size * 0.85, t.z, size * 0.42, foliage,
                         segments=10, rings=6)
        else:                                  # layered conifer
            g.add_cylinder(t.x, t.z, base + size * 0.40, size * 0.42, size * 0.62,
                           foliage, segments=9, top_radius=0.04)
            g.add_cylinder(t.x, t.z, base + size * 0.72, size * 0.30, size * 0.52,
                           _tint(foliage, 0.03), segments=9, top_radius=0.03)
        shadow.add_disc_y(t.x, base + 0.03, t.z, size * 0.36,
                          (0.03, 0.04, 0.05), segments=12)


def _build_obstacles(g: GeoBuffer, shadow: GeoBuffer, track):
    for o in track.obstacles:
        oy = track.terrain_height(o.x, o.z) + 0.05
        col = OBSTACLE_COLORS.get(o.kind, o.color)
        if o.kind == "cone":
            g.add_cylinder(o.x, o.z, oy, o.radius * 1.25, 0.10, (0.85, 0.85, 0.86),
                           segments=10)
            g.add_cylinder(o.x, o.z, oy + 0.10, o.radius, o.height - 0.10, col,
                           segments=10, top_radius=0.05)
            g.add_cylinder(o.x, o.z, oy + 0.34, o.radius * 0.72, 0.12,
                           (0.95, 0.95, 0.95), segments=10)
        elif o.kind == "barrel":
            g.add_cylinder(o.x, o.z, oy, o.radius, o.height, col, segments=12)
            for frac in (0.30, 0.68):
                g.add_cylinder(o.x, o.z, oy + o.height * frac, o.radius * 1.03,
                               0.13, (0.95, 0.95, 0.94), segments=12)
            g.add_cylinder(o.x, o.z, oy + o.height, o.radius * 1.02, 0.06,
                           _tint(col, -0.12), segments=12)
        else:  # crate
            g.add_rounded_box(o.x, o.z, oy, o.radius * 2.0, o.radius * 2.0,
                              o.height, 0.4, col, radius=0.09, corner_seg=2)
            g.add_box(o.x, o.z, oy + o.height * 0.42, o.radius * 2.02,
                      o.radius * 2.02, 0.06, 0.4, _tint(col, -0.14))
        shadow.add_disc_y(o.x, oy + 0.01, o.z, o.radius * 1.5,
                          (0.03, 0.04, 0.05), segments=10)
