"""Car physics: arcade driving model, wall/obstacle collisions, telemetry."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .core_types import Action
from .items import ItemController, ITEM_KEYS
from .mathutil import clamp, forward_xz, left_normal_xz, wrap_angle


@dataclass
class World:
    """Lightweight view of the world handed to a car each tick."""
    time: float = 0.0
    hazards: list = field(default_factory=list)
    opponent: Optional["Car"] = None


class Car:
    def __init__(self, name: str, color, track, spawn_arc: float,
                 lateral: float, is_human: bool = False,
                 max_speed: float = 42.0, accel: float = 18.0):
        self.name = name
        self.color = color
        self.is_human = is_human
        self.track = track

        # --- tuning ---
        self.base_max_speed = max_speed
        self.base_accel = accel
        self.brake_decel = 36.0
        self.steer_rate = 2.45
        self.drag_lin = 0.015
        self.drag_quad = 0.0006
        self.car_radius = 1.15          # circle used against obstacles/hazards
        # nitro: top speed and acceleration multiplier (anti-cheat's speed cap
        # is derived from this too, in game.py)
        self.BOOST = 1.55
        self.car_half_width = 1.0
        # capsule used for car-vs-car: a 4.1 x 1.8 m body is badly approximated
        # by a 1.15 m circle, which let the two cars visibly interpenetrate
        self.car_half_len = 1.15        # half length of the capsule core
        self.car_half_wid = 0.90        # capsule radius
        self._dt = 1.0 / 60.0
        self._wall_scrape = 0.0

        self.items = ItemController()

        self.reset(spawn_arc, lateral)

    # ------------------------------------------------------------------
    def reset(self, spawn_arc: float, lateral: float) -> None:
        x, z, heading = self.track.spawn(spawn_arc, lateral)
        self.pos = np.array([x, z], dtype=np.float64)
        self.prev_pos = self.pos.copy()
        self.heading = heading
        self.speed = 0.0
        self.y = 0.0
        self.finished = False
        self.finish_time = 0.0

        # effect timers
        self.boost_until = -999.0
        self.shield_until = -999.0
        self.slow_until = -999.0
        self.slow_factor = 0.45

        # race progress (continuity tracked along the center line)
        self._hint = None
        self.start_arc = float(spawn_arc)
        self.arc_unwrapped = float(spawn_arc)
        self.laps = 0
        self.lateral = float(lateral)
        self.center_i = 0
        self.center_t = 0.0
        self.center_proj = self.pos.copy()
        self.center_dist = 0.0
        self.center_tang = np.array([1.0, 0.0])
        self._project()

        # telemetry
        self.wall_hits = 0
        self.obstacle_hits = 0
        self.hazard_hits = 0
        self.last_wall_hit = -999.0
        self.last_obs_hit = -999.0
        self.last_hazard_hit = -999.0
        self.distance = 0.0
        self.max_speed_seen = 0.0
        self.speed_sum = 0.0
        self.speed_n = 0
        self.lateral_abs_sum = 0.0
        self.lateral_n = 0
        self.off_center_time = 0.0
        self.item_uses = {k: 0 for k in ITEM_KEYS}
        self.wall_log = []
        self.obstacle_log = []

    # ------------------------------------------------------------------
    def effective_max_speed(self, now: float) -> float:
        """This car's top speed *right now*.

        A nitro multiplies it by :attr:`BOOST` and an oil/EMP hit scales it
        down, so ``base_max_speed`` is not the answer while either is live.
        Drivers are told this figure, not the base one: a driver that keeps
        targeting the unboosted speed will brake itself back down in the
        middle of its own nitro and the item is spent for nothing.
        """
        boost = self.BOOST if self.is_boosted(now) else 1.0
        slow = self.slow_factor if self.is_slowed(now) else 1.0
        return self.base_max_speed * boost * slow

    def is_shielded(self, now: float) -> bool:
        return now < self.shield_until

    def is_boosted(self, now: float) -> bool:
        return now < self.boost_until

    def is_slowed(self, now: float) -> bool:
        return now < self.slow_until

    def effect_name(self, now: float) -> str:
        if self.is_boosted(now):
            return "boost"
        if self.is_shielded(now):
            return "shield"
        if self.is_slowed(now):
            return "slow"
        return ""

    def apply_boost(self, now: float, duration: float) -> None:
        self.boost_until = max(self.boost_until, now + duration)

    def apply_shield(self, now: float, duration: float) -> None:
        self.shield_until = max(self.shield_until, now + duration)

    def apply_slow(self, now: float, duration: float, factor: float = 0.45) -> None:
        self.slow_until = max(self.slow_until, now + duration)
        self.slow_factor = min(self.slow_factor, factor)

    # ------------------------------------------------------------------
    @property
    def forward(self) -> np.ndarray:
        return forward_xz(self.heading)

    @property
    def velocity(self) -> np.ndarray:
        return self.forward * self.speed

    def _project(self) -> None:
        """Continuity-aware projection onto the center line + lap tracking.

        The nearest-point query is spatially exact; lap progress is then
        unwrapped modulo the track length so crossing the start/finish seam
        is handled without any branch disambiguation.  If the car is ever
        far from the corridor we fall back to a global search so we can
        never latch onto a distant branch.
        """
        i, t, p, d, lat, tang = self.track.nearest(
            self.pos, self._hint, window=60)
        if d > self.track.half_width + 6.0:
            i, t, p, d, lat, tang = self.track.nearest(self.pos, None, window=90)
        self._hint = i
        self.center_i = i
        self.center_t = t
        self.center_proj = p
        self.center_dist = d
        self.lateral = lat
        self.center_tang = tang
        arc = self.track.arc_at(i, t)
        length = self.track.length
        delta = (arc - self.arc_unwrapped + length * 0.5) % length - length * 0.5
        self.arc_unwrapped += delta
        self.laps = max(0, int(math.floor(
            (self.arc_unwrapped - self.start_arc) / length + 1e-9)))

    def _capsule(self):
        """Core segment of the car's collision capsule (2D, ground plane)."""
        f = self.forward
        L = self.car_half_len
        return ((float(self.pos[0] - f[0] * L), float(self.pos[1] - f[1] * L)),
                (float(self.pos[0] + f[0] * L), float(self.pos[1] + f[1] * L)))

    def lateral_offset(self):
        return self.lateral, self.center_dist, self.center_i, self.center_proj, self.center_tang

    # ------------------------------------------------------------------
    def update(self, dt: float, action: Action, world: World) -> None:
        now = world.time
        action = action.clamped()
        self._dt = max(dt, 1e-6)

        boost = self.BOOST if self.is_boosted(now) else 1.0
        slow = self.slow_factor if self.is_slowed(now) else 1.0
        max_speed = self.effective_max_speed(now)
        accel = self.base_accel * boost * slow

        # --- longitudinal --------------------------------------------
        if action.brake > 0.0 and self.speed > 0.5:
            self.speed -= action.brake * self.brake_decel * dt
        else:
            drag = self.drag_lin * self.speed + self.drag_quad * self.speed * self.speed
            self.speed += (accel * action.throttle - drag) * dt
        if self.speed < 0.0:
            self.speed = 0.0
        if self.speed > max_speed:
            # let it decay smoothly once a boost wears off
            self.speed = max(max_speed, self.speed - self.brake_decel * 0.6 * dt)

        # --- steering -------------------------------------------------
        speed_grip = min(1.0, 0.25 + self.speed / 4.0)
        high_speed_damp = 1.0 / (1.0 + self.speed * 0.028)
        omega = action.steer * self.steer_rate * speed_grip * high_speed_damp
        self.heading = wrap_angle(self.heading + omega * dt)

        # --- integrate ------------------------------------------------
        self.prev_pos = self.pos.copy()
        self.pos = self.pos + self.forward * self.speed * dt
        self.distance += float(np.linalg.norm(self.pos - self.prev_pos))

        self._project()
        self._resolve_wall(now)
        self._resolve_obstacles(now)
        self._resolve_wall(now)
        self._resolve_hazards(now, world)
        self._project()

        # sample elevation from the center line
        self.y = self.track.height_at_index(self.center_i, self.center_t)

        # --- telemetry ------------------------------------------------
        # Freeze once the car has crossed the line: it is only coasting to a
        # stop at that point, and counting those metres would drag its average
        # speed down while it waits for the other car to finish.
        if not self.finished:
            self.max_speed_seen = max(self.max_speed_seen, self.speed)
            self.speed_sum += self.speed
            self.speed_n += 1
            norm_lat = abs(self.lateral) / max(self.track.half_width, 1e-6)
            self.lateral_abs_sum += norm_lat
            self.lateral_n += 1
            if norm_lat > 0.5:
                self.off_center_time += dt

    # ------------------------------------------------------------------
    def _resolve_wall(self, now: float) -> None:
        limit = self.track.half_width - self.car_half_width
        lat = self.lateral
        if abs(lat) <= limit:
            return
        nrm = self.track.normal[self.center_i]
        clamped = math.copysign(limit, lat)
        self.pos = self.center_proj + nrm * clamped
        # impact severity: how head-on the wall contact is
        impact = abs(float(np.dot(self.forward, nrm)))
        # strongly re-align the car along the wall so it cannot ride the barrier
        tang = self.track.tangent[self.center_i]
        tang_h = math.atan2(float(tang[1]), float(tang[0]))
        self.heading = wrap_angle(
            self.heading + 0.55 * wrap_angle(tang_h - self.heading))
        if self.is_shielded(now):
            return
        # head-on impacts are brutal, but even a light scrape bleeds speed
        headon = min(1.0, impact * 2.0)
        retain = 1.0 - (1.0 - 0.28) * headon
        retain = min(retain, 0.88)
        self.speed *= retain
        # continuous scrape friction while pressed against the wall
        self.speed = max(0.0, self.speed * (1.0 - 0.30 * self._dt))
        if now - self.last_wall_hit > 0.30:
            self.wall_hits += 1
            try:
                self.wall_log.append(
                    (round(now, 1), int(self.center_i),
                     round(float(self.track.curvature[self.center_i]), 4),
                     round(float(lat), 2), round(float(self.speed), 1)))
                if len(self.wall_log) > 80:
                    self.wall_log.pop(0)
            except Exception:
                pass
            self.last_wall_hit = now

    def contain(self, now: float) -> None:
        """Re-project and pull the car back inside the corridor if needed."""
        self._project()
        limit = self.track.half_width - self.car_half_width
        if abs(self.lateral) > limit:
            self._resolve_wall(now)
            self._project()

    def _resolve_obstacles(self, now: float) -> None:
        for obs in self.track.obstacles:
            dx = self.pos[0] - obs.x
            dz = self.pos[1] - obs.z
            dist = math.hypot(dx, dz)
            min_d = obs.radius + self.car_radius
            if dist < min_d and dist > 1e-6:
                nx, nz = dx / dist, dz / dist
                self.pos[0] = obs.x + nx * (min_d + 0.05)
                self.pos[1] = obs.z + nz * (min_d + 0.05)
                # slide the car around the obstacle instead of head-butting it
                if self.speed > 1.0:
                    tx, tz = -nz, nx
                    fwd = self.forward
                    if tx * fwd[0] + tz * fwd[1] < 0.0:
                        tx, tz = -tx, -tz
                    new_h = math.atan2(tz, tx)
                    self.heading = wrap_angle(
                        self.heading + 0.35 * wrap_angle(new_h - self.heading))
                # only apply the slowdown once per contact window
                if not self.is_shielded(now) and now - self.last_obs_hit > 0.5:
                    self.speed *= obs.slow
                    self.obstacle_hits += 1
                    self.last_obs_hit = now
                    try:
                        self.obstacle_log.append(
                            (round(now, 1), obs.kind,
                             round(float(self.lateral), 2),
                             round(float(self.speed), 1)))
                        if len(self.obstacle_log) > 60:
                            self.obstacle_log.pop(0)
                    except Exception:
                        pass

    def _resolve_hazards(self, now: float, world: World) -> None:
        for hz in world.hazards:
            if hz.get("owner") == self.name:
                continue
            if now > hz.get("expire", 0.0):
                continue
            dx = self.pos[0] - hz["x"]
            dz = self.pos[1] - hz["z"]
            dist = math.hypot(dx, dz)
            if dist < hz.get("radius", 2.5) + self.car_radius:
                if not self.is_shielded(now):
                    # keep the slow refreshed while inside, but count the hit
                    # once per contact window (it used to tick every frame,
                    # inflating the metric to ~60/s)
                    self.apply_slow(now, 1.5, 0.4)
                    if now - self.last_hazard_hit > 0.5:
                        self.last_hazard_hit = now
                        self.hazard_hits += 1

    # ------------------------------------------------------------------
    @property
    def race_progress(self) -> float:
        total = self.track.length * max(1, getattr(self, "laps_total", 1))
        return (self.arc_unwrapped - self.start_arc) / max(total, 1e-6)

    # ------------------------------------------------------------------
    def telemetry(self) -> dict:
        avg_speed = self.speed_sum / max(1, self.speed_n)
        avg_lat = self.lateral_abs_sum / max(1, self.lateral_n)
        return {
            "name": self.name,
            "finished": self.finished,
            "finish_time": round(self.finish_time, 3),
            "laps": self.laps,
            "distance": round(self.distance, 1),
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(self.max_speed_seen, 2),
            "wall_hits": self.wall_hits,
            "obstacle_hits": self.obstacle_hits,
            "hazard_hits": self.hazard_hits,
            "item_uses": dict(self.item_uses),
            "item_use_total": int(sum(self.item_uses.values())),
            "avg_lateral_ratio": round(avg_lat, 4),
            "off_center_time": round(self.off_center_time, 2),
        }
