"""Procedural meshes for the cars and small props.

The car is lofted from superellipse cross-sections so the body is smooth
rather than a stack of axis-aligned boxes.  Paint (tinted per driver) and
trim (fixed colours: glass, lights, wheels) are separate buffers because the
renderer applies a per-car tint to the paint pass only.

Local frame: +X forward, +Y up, +Z to the left.
"""
from __future__ import annotations

import math

from .geometry import GeoBuffer, loft_stations

# wheel geometry, shared with the renderer for placement/animation
WHEEL_R = 0.42
WHEEL_W = 0.30
WHEEL_X_FRONT = 1.32
WHEEL_X_REAR = -1.30
WHEEL_Z = 0.84
WHEEL_Y = WHEEL_R
# (x, y, z, steerable) per wheel
WHEELS = [
    (WHEEL_X_FRONT, WHEEL_Y, WHEEL_Z, True),
    (WHEEL_X_FRONT, WHEEL_Y, -WHEEL_Z, True),
    (WHEEL_X_REAR, WHEEL_Y, WHEEL_Z, False),
    (WHEEL_X_REAR, WHEEL_Y, -WHEEL_Z, False),
]

PAINT = (0.92, 0.93, 0.95)          # tinted by the driver colour
GLASS = (0.05, 0.07, 0.10)
DARK = (0.09, 0.09, 0.11)
HEADLIGHT = (1.00, 0.96, 0.78)
TAILLIGHT = (0.86, 0.10, 0.08)
EXHAUST = (0.34, 0.34, 0.38)
TIRE = (0.055, 0.055, 0.065)
RIM = (0.74, 0.76, 0.80)

# (x, half_width, y_bottom, y_top)
BODY_STATIONS = [
    (-2.05, 0.55, 0.36, 0.78),
    (-1.75, 0.74, 0.30, 0.92),
    (-1.35, 0.86, 0.26, 1.02),
    (-0.90, 0.92, 0.24, 1.08),
    (-0.40, 0.94, 0.24, 1.10),
    (0.10, 0.94, 0.24, 1.10),
    (0.60, 0.92, 0.24, 1.06),
    (1.10, 0.86, 0.24, 0.98),
    (1.55, 0.78, 0.26, 0.88),
    (1.90, 0.66, 0.30, 0.78),
    (2.05, 0.54, 0.34, 0.70),
]

GLASS_STATIONS = [
    (0.58, 0.60, 0.98, 1.12),
    (0.32, 0.70, 1.00, 1.32),
    (-0.08, 0.74, 1.00, 1.46),
    (-0.52, 0.72, 1.00, 1.44),
    (-0.88, 0.66, 0.98, 1.28),
    (-1.14, 0.56, 0.96, 1.06),
]

ROOF_STATIONS = [
    (0.40, 0.44, 1.20, 1.32),
    (0.05, 0.55, 1.38, 1.49),
    (-0.35, 0.57, 1.39, 1.50),
    (-0.70, 0.51, 1.30, 1.41),
    (-0.96, 0.42, 1.14, 1.24),
]


def build_paint() -> GeoBuffer:
    """Body panels tinted by the driver colour."""
    g = GeoBuffer()
    g.add_loft(loft_stations(BODY_STATIONS, n=4.0, segments=18), PAINT)
    g.add_loft(loft_stations(ROOF_STATIONS, n=3.0, segments=14), PAINT)
    # side mirrors
    for z in (0.98, -0.98):
        g.add_rounded_box(0.50, z, 0.94, 0.20, 0.16, 0.11, 0.0, PAINT, radius=0.04,
                          corner_seg=2)
        g.add_box(0.42, z * 0.86, 0.92, 0.16, 0.08, 0.06, 0.0, DARK)
    return g


def build_trim() -> GeoBuffer:
    """Fixed-colour parts: glass, aero, lights, exhausts."""
    g = GeoBuffer()
    # greenhouse
    g.add_loft(loft_stations(GLASS_STATIONS, n=3.2, segments=16), GLASS)
    # rear wing: two struts + a plate
    for z in (0.52, -0.52):
        g.add_box(-1.98, z, 0.72, 0.12, 0.10, 0.36, 0.0, DARK)
    g.add_box(-2.02, 0.0, 1.06, 0.42, 1.56, 0.08, 0.0, DARK)   # wing plate
    for z in (0.76, -0.76):                                     # end plates
        g.add_box(-2.02, z, 1.00, 0.44, 0.08, 0.20, 0.0, DARK)
    # front splitter + rear diffuser
    g.add_box(2.00, 0.0, 0.18, 0.44, 1.62, 0.10, 0.0, DARK)
    g.add_box(-2.02, 0.0, 0.22, 0.40, 1.52, 0.16, 0.0, DARK)
    # headlights / taillights
    for z in (0.44, -0.44):
        g.add_box(2.02, z, 0.60, 0.10, 0.34, 0.14, 0.0, HEADLIGHT)
    for z in (0.50, -0.50):
        g.add_box(-2.06, z, 0.66, 0.10, 0.36, 0.16, 0.0, TAILLIGHT)
    # exhausts
    for z in (0.30, -0.30):
        g.add_box(-2.06, z, 0.34, 0.18, 0.14, 0.13, 0.0, EXHAUST)
    return g


def build_wheel() -> GeoBuffer:
    """One wheel centred on the origin, axle along Z."""
    g = GeoBuffer()
    r, w = WHEEL_R, WHEEL_W
    profile = [
        (r * 0.52, -w * 0.5),
        (r * 0.93, -w * 0.5),
        (r, -w * 0.5 + 0.06),
        (r, w * 0.5 - 0.06),
        (r * 0.93, w * 0.5),
        (r * 0.52, w * 0.5),
    ]
    g.add_revolve_z(profile, TIRE, segments=22)
    # rim face + spokes + hub on both sides so it reads from any angle
    for sgn in (1.0, -1.0):
        z = sgn * (w * 0.5 - 0.015)
        g.add_disc_z(0.0, 0.0, z, r * 0.62, RIM, segments=20, forward=sgn > 0)
        for k in range(5):
            a = 2.0 * math.pi * k / 5.0 + 0.3
            ca, sa = math.cos(a), math.sin(a)
            tx, ty = -sa * 0.045, ca * 0.045
            rr = r * 0.56
            pts = [(ca * 0.10 + tx, sa * 0.10 + ty, z),
                   (ca * 0.10 - tx, sa * 0.10 - ty, z),
                   (ca * rr - tx, sa * rr - ty, z),
                   (ca * rr + tx, sa * rr + ty, z)]
            g.add_quad(pts[0], pts[1], pts[2], pts[3], RIM,
                       (0.0, 0.0, 1.0 if sgn > 0 else -1.0))
        g.add_disc_z(0.0, 0.0, z, 0.085, (0.85, 0.86, 0.90), segments=12,
                     forward=sgn > 0)
    return g


def build_blob_shadow(radius: float = 1.0) -> GeoBuffer:
    """Soft-ish contact shadow disc (alpha comes from the blend colour)."""
    g = GeoBuffer()
    g.add_disc_y(0.0, 0.0, 0.0, radius, (0.03, 0.04, 0.05), segments=24)
    return g
