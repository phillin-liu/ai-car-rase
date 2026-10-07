"""Small row-major matrix/vector helpers built on numpy.

Matrices are 4x4 ``numpy.float32`` arrays in *row-major* (mathematical)
layout.  When uploaded to OpenGL we pass ``transpose=True`` so the driver
converts them to the column-major layout GL expects.
"""
from __future__ import annotations

import math
import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scaling(x: float, y: float | None = None, z: float | None = None) -> np.ndarray:
    if y is None:
        y = x
    if z is None:
        z = x
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotation_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotation_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotation_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def multiply(*mats: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    for m in mats:
        out = out @ np.asarray(m, dtype=np.float32)
    return out.astype(np.float32)


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    n = np.linalg.norm(f)
    f = f / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    s = np.cross(f, up)
    ns = np.linalg.norm(s)
    s = s / ns if ns > 1e-9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


def forward_xz(heading: float) -> np.ndarray:
    """Unit forward vector in the X/Z ground plane."""
    return np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)


def left_normal_xz(heading: float) -> np.ndarray:
    """Unit left-hand normal for a heading (rotate forward +90 deg about Y)."""
    return np.array([-math.sin(heading), math.cos(heading)], dtype=np.float64)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def car_model(x: float, y: float, z: float, heading: float, scale=1.0) -> np.ndarray:
    """Model matrix placing a +X facing car at (x,y,z) with ``heading``."""
    return multiply(translation(x, y, z), rotation_y(-heading), scaling(scale))
