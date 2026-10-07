"""Camera rig: chase / first-person / overhead / free orbit."""
from __future__ import annotations

import math

import numpy as np

from ..mathutil import look_at, perspective


class Camera:
    MODES = ("chase", "first", "overhead", "free")

    def __init__(self, fov: float = 58.0):
        self.eye = np.array([0.0, 60.0, 120.0])
        self.target = np.zeros(3, dtype=np.float64)
        self._free_angle = 0.0
        self._fov = fov
        self._center = np.zeros(3, dtype=np.float64)
        self._extent = 250.0

    def set_bounds(self, center: np.ndarray, extent: float) -> None:
        self._center = np.asarray(center, dtype=np.float64).reshape(3)
        self._extent = float(extent)

    def update(self, game, mode: str, watch: int, dt: float,
               width: int, height: int):
        cars = game.cars
        idx = watch if 0 <= watch < len(cars) else 0
        car = cars[idx]
        pos = np.array([float(car.pos[0]), float(car.y), float(car.pos[1])])
        fwd = np.array([math.cos(car.heading), 0.0, math.sin(car.heading)])

        if mode == "first":
            eye = pos + fwd * 0.35 + np.array([0.0, 1.62, 0.0])
            tgt = pos + fwd * 14.0 + np.array([0.0, 0.95, 0.0])
        elif mode == "overhead":
            eye = pos + np.array([0.0, 46.0, 0.0]) - fwd * 6.0
            tgt = pos
        elif mode == "free":
            self._free_angle += dt * 0.15
            r = self._extent
            eye = self._center + np.array([r * math.cos(self._free_angle),
                                           r * 0.85,
                                           r * math.sin(self._free_angle)])
            tgt = self._center + np.array([0.0, 5.0, 0.0])
        else:  # chase
            eye = pos - fwd * 10.0 + np.array([0.0, 4.6, 0.0])
            tgt = pos + fwd * 7.0 + np.array([0.0, 1.3, 0.0])

        lerp = 1.0 if mode == "first" else 1.0 - math.exp(-dt * 8.0)
        self.eye = self.eye + (eye - self.eye) * lerp
        self.target = self.target + (tgt - self.target) * lerp

        up = (0.0, 1.0, 0.0)
        if mode == "overhead":
            up = (float(fwd[0]), 0.0, float(fwd[2]))
        view = look_at(self.eye, self.target, up)

        # a touch of extra FOV at speed for a sense of velocity
        want = 56.0 + min(9.0, float(car.speed) * 0.20)
        self._fov += (want - self._fov) * min(1.0, dt * 3.0)
        proj = perspective(self._fov, width / max(1, height), 0.4, 1400.0)
        return proj, view
