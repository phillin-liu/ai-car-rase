"""Rule-based racing AI.

**No longer part of the race.**  Both raceable sides are vision models now:
the remote ones decide in the LLM driver, and the local one decides in its
imported policy model.  Nothing selects this driver as a provider any more --
it survives because the test suite uses it as a deterministic stand-in to
exercise the simulation and the report, and because it is a useful reference
for what a hand-written racing line looks like.

It plans a lateral offset each tick by scoring candidate racing lines
against the upcoming corners and obstacles, brakes according to a safe
speed profile and uses items with simple heuristics.

The driver is intentionally *strong*: it uses a proper racing-line apex
offset, a curvature-aware speed profile with braking distance, obstacle
avoidance with hysteresis and stuck-car recovery.  ``difficulty`` scales
its cornering limit, steering precision and aggression.
"""
from __future__ import annotations

import math

from ..config import ModelConfig
from ..core_types import Action, Observation
from ..mathutil import clamp, wrap_angle
from .base import BaseDriver


class HeuristicDriver(BaseDriver):
    kind = "rule"

    def __init__(self, cfg: ModelConfig, track, name: str, difficulty: float = 1.0):
        super().__init__(cfg, track, name)
        self.difficulty = clamp(difficulty, 0.4, 1.4)
        self._prev_err = 0.0
        self._stuck_time = 0.0
        self._progress_ref = 0.0
        self._progress_time = 0.0
        self._need_recover = False
        self._chosen_off = 0.0
        self._last_corner_sign = 0.0

    def reset(self, track) -> None:
        super().reset(track)
        self._prev_err = 0.0
        self._stuck_time = 0.0
        self._progress_ref = 0.0
        self._progress_time = 0.0
        self._need_recover = False
        self._chosen_off = 0.0
        self._last_corner_sign = 0.0

    # ------------------------------------------------------------------
    def act(self, obs: Observation) -> Action:
        policy = self.policy or {}
        lateral_bias = float(policy.get("lateral_bias", 0.0))
        throttle_scale = float(policy.get("throttle_scale", 1.0))
        aggression = float(policy.get("aggression", 0.5))

        pos = obs.pos
        half = max(obs.half_width, 1e-6)
        heading = obs.heading
        fwd = (math.cos(heading), math.sin(heading))
        left = (-math.sin(heading), math.cos(heading))

        # --- competitive drive ---------------------------------------
        # The driver knows exactly one thing about the rival: the race gap.
        # It is never allowed to settle -- the target speed always encodes
        # either "catch him" or "don't let him catch me".
        gap_m = float(getattr(obs, "rival_gap_m", 0.0))
        # 1.0 when 60 m behind .. 0.0 when 60 m ahead
        chase = clamp((60.0 - gap_m) / 120.0, 0.0, 1.0)

        # --- racing line ----------------------------------------------
        dists = list(obs.lookahead_distances or [8.0, 18.0, 30.0, 45.0, 65.0, 90.0, 125.0])
        pts = list(obs.lookahead_points or [])
        curvs = list(obs.lookahead_curvatures or [0.0] * len(dists))

        look = clamp(7.0 + obs.speed * 0.75, 9.0, 42.0)
        k = min(range(len(dists)), key=lambda i: abs(dists[i] - look))
        tx, tz = pts[k] if k < len(pts) else (pos[0], pos[1])
        curv = curvs[k] if k < len(curvs) else 0.0

        # Apex line: aim wide before a corner, clip the apex, wide on exit.
        # The corner direction is inferred from the sign of nearby curvature.
        sign = 1.0 if curv > 0 else -1.0
        corner_mag = min(1.0, abs(curv) * 14.0)
        # weight recent corner sign for stability
        self._last_corner_sign = 0.7 * self._last_corner_sign + 0.3 * sign
        sign = self._last_corner_sign
        # apex offset grows with curvature and speed
        apex = -sign * corner_mag * half * clamp(0.35 + obs.speed * 0.010, 0.35, 0.78)
        line_off = clamp(apex + lateral_bias * half, -0.78 * half, 0.78 * half)

        # candidate set: dense grid around the racing line
        candidates = [line_off]
        for r in (-0.75, -0.55, -0.35, -0.2, -0.08, 0.0, 0.08, 0.2, 0.35, 0.55, 0.75):
            candidates.append(r * half)
        candidates.append(lateral_bias * half)

        obs_ahead = []
        for ox, oz, orad in obs.obstacles_ahead:
            relx, relz = ox - pos[0], oz - pos[1]
            proj = relx * fwd[0] + relz * fwd[1]
            if -5.0 <= proj <= 50.0:
                lat_track = obs.lateral + (relx * left[0] + relz * left[1])
                obs_ahead.append((proj, lat_track, orad))

        # opponent in track coordinates: without this both cars aim at the
        # same racing line and can lock together nose-to-tail
        opp_dx = obs.opp_pos[0] - pos[0]
        opp_dz = obs.opp_pos[1] - pos[1]
        opp_proj = opp_dx * fwd[0] + opp_dz * fwd[1]
        opp_lat = obs.lateral + (opp_dx * left[0] + opp_dz * left[1])
        opp_near = -4.0 < opp_proj < 18.0

        prev_off = self._chosen_off
        best_off, best_score = line_off, -1e18
        for cand in candidates:
            score = -abs(cand - line_off) / half * 1.15
            # wall margin: penalise running wide near the barrier
            edge = (half - 1.8) - abs(cand)
            if edge < 0:
                score -= abs(edge) * 4.0
            for proj, lat, rad in obs_ahead:
                gap = abs(cand - lat) - (rad + 1.75)
                urgency = 1.0 / (1.0 + proj / 9.0)
                if gap < 0.0:
                    score -= abs(gap) * 9.0 * urgency
                else:
                    score += min(gap, 2.2) * 0.18 * urgency
            if opp_near:
                gap_o = abs(cand - opp_lat) - 2.6
                urg_o = 1.0 / (1.0 + abs(opp_proj) / 8.0)
                if gap_o < 0.0:
                    score -= abs(gap_o) * 7.0 * urg_o
                else:
                    score += min(gap_o, 2.0) * 0.12 * urg_o
            if score > best_score:
                best_score, best_off = score, cand

        # hysteresis smoothing so the chosen line doesn't jitter
        self._chosen_off = 0.55 * prev_off + 0.45 * best_off
        best_off = self._chosen_off
        self.last_off = best_off
        self.last_obs = obs_ahead

        target = [tx + left[0] * best_off, tz + left[1] * best_off]

        # --- steering (PD controller) ---------------------------------
        desired = math.atan2(target[1] - pos[1], target[0] - pos[0])
        err = wrap_angle(desired - heading)
        deriv = wrap_angle(err - self._prev_err)
        self._prev_err = err
        steer = clamp(err * (1.9 + 0.35 * self.difficulty) + deriv * 0.7, -1.0, 1.0)

        # push away from the wall when running wide
        ratio = obs.lateral / half
        if abs(ratio) > 0.70:
            away = -math.copysign(1.0, obs.lateral)
            steer = clamp(steer + away * (abs(ratio) - 0.70) * 3.0, -1.0, 1.0)

        # last-resort separation nudge so a side-by-side pair never locks up
        if obs.opp_distance < 3.6:
            delta = obs.lateral - obs.opp_lateral
            steer = clamp(steer - math.copysign(0.55, delta if abs(delta) > 1e-3 else 1.0),
                          -1.0, 1.0)

        # --- safe speed profile --------------------------------------
        # Cornering grip is a property of the car, NOT of motivation: raising
        # it while chasing made the AI clip obstacles at speed and wedge
        # itself.  Ambition shows up in the straight-line target and items.
        lat_accel = (17.0 + 8.0 * self.difficulty)
        brake_accel = 22.0
        car_max = obs.max_speed if obs.max_speed > 1 else 42.0
        cruise = 0.90 + 0.04 * self.difficulty + 0.10 * chase
        v_target = car_max * min(cruise, 1.0)
        for dd, cc in zip(dists, curvs):
            vc = math.sqrt(lat_accel / max(abs(cc), 1e-4))
            v_target = min(v_target, math.sqrt(vc * vc + 2.0 * brake_accel * max(dd, 0.0)))
        v_target = min(v_target, car_max) * clamp(throttle_scale, 0.3, 1.25)

        # obstacles in the chosen path force extra braking
        block_brake = 0.0
        for proj, lat, rad in obs_ahead:
            if proj < 20.0 and abs(lat - best_off) < rad + 1.6:
                block_brake = max(block_brake, 0.38 * (1.0 - proj / 20.0))
        if obs.speed < 12.0:
            block_brake = 0.0

        err_v = v_target - obs.speed
        if err_v >= 0:
            throttle = clamp(err_v / 6.0, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-err_v / 8.0, 0.0, 1.0)

        brake = max(brake, block_brake)
        if block_brake > 0:
            throttle = min(throttle, 0.40)

        # --- recovery when stuck -------------------------------------
        self._stuck_time = self._stuck_time + obs.dt if obs.speed < 3.0 else 0.0
        if obs.race_progress > self._progress_ref + 0.004:
            self._progress_ref = obs.race_progress
            self._progress_time = obs.time
        self._need_recover = (self._stuck_time > 1.1 or
                              (obs.time - self._progress_time) > 2.4)
        if self._need_recover:
            # No progress for a while: stop ramming at speed. Bleed down to a
            # crawl and aim back at the centre line so the obstacle push can
            # slide the car free. A car that is merely stationary would never
            # resume progressing, so we never brake to a full stop.
            v_escape = 9.0
            if obs.speed > v_escape:
                throttle = 0.0
                brake = clamp((obs.speed - v_escape) / 8.0, 0.0, 1.0)
            else:
                throttle = 0.35
                brake = 0.0
            if abs(obs.lateral) > 0.4:
                steer = clamp(steer - math.copysign(0.8, obs.lateral), -1.0, 1.0)

        use_item = self._choose_item(obs, aggression, policy, best_off, obs_ahead)

        return Action(throttle=throttle, brake=brake, steer=steer,
                      use_item=use_item, reason=self.raw_reason or "rule")

    # ------------------------------------------------------------------
    def _choose_item(self, obs, aggression, policy, best_off, obs_ahead):
        explicit = policy.get("use_item")
        avail = {k: v for k, v in obs.items.items() if v.get("available")}

        def can(k):
            return k in avail

        if explicit and explicit in avail:
            return explicit
        if can("RECOVER") and self._need_recover:
            return "RECOVER"
        if can("RECOVER") and abs(obs.lateral) > 0.94 * obs.half_width and obs.speed < 8:
            return "RECOVER"

        curvs = obs.lookahead_curvatures or [0.0]
        straight = all(abs(c) < 0.010 for c in curvs[:3])
        car_max = obs.max_speed if obs.max_speed > 1 else 42.0

        # attack when behind, defend when the rival is on my tail
        gap_m = float(getattr(obs, "rival_gap_m", 0.0))
        behind = gap_m < -2.0
        leading = gap_m > 2.0

        # attack first: EMP directly pressures the car ahead of me
        if can("EMP") and obs.opp_ahead and obs.opp_distance < 40 and (behind or aggression > 0.3):
            return "EMP"
        if can("NITRO") and straight and (behind or obs.speed > 0.6 * car_max):
            return "NITRO"
        if can("OIL") and (not obs.opp_ahead) and obs.opp_distance < 30 and (leading or aggression > 0.4):
            return "OIL"
        if can("SHIELD"):
            for proj, lat, rad in obs_ahead:
                if proj < 12.0 and abs(lat - best_off) < rad + 1.4:
                    return "SHIELD"
            if abs(obs.lateral) > 0.86 * obs.half_width:
                return "SHIELD"
        return None
