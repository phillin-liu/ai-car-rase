"""Race engine: runs a single match between two cars."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .config import MatchConfig
from .core_types import Action, Observation
from .items import ITEM_MAP
from .palette import CAR_COLORS
from .cars import Car, World
from .track import Track
from .anticheat import AntiCheat


# ---------------------------------------------------------------------------
def _segment_closest(seg_a, seg_b):
    """Closest points between two 2D segments (capsule collision core)."""
    (p1, q1), (p2, q2) = seg_a, seg_b
    d1 = (q1[0] - p1[0], q1[1] - p1[1])
    d2 = (q2[0] - p2[0], q2[1] - p2[1])
    r = (p1[0] - p2[0], p1[1] - p2[1])
    a = d1[0] * d1[0] + d1[1] * d1[1]
    e = d2[0] * d2[0] + d2[1] * d2[1]
    f = d2[0] * r[0] + d2[1] * r[1]
    eps = 1e-12
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = d1[0] * r[0] + d1[1] * r[1]
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = d1[0] * d2[0] + d1[1] * d2[1]
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    c1 = (p1[0] + d1[0] * s, p1[1] + d1[1] * s)
    c2 = (p2[0] + d2[0] * t, p2[1] + d2[1] * t)
    return c1, c2


@dataclass
class MatchResult:
    match_id: int
    seed: int
    laps: int
    mode: str
    duration: float
    winner: Optional[int]
    loser: Optional[int]
    finished_by: Optional[int]
    end_reason: str = ""
    cars: list = field(default_factory=list)          # telemetry dicts
    driver_stats: list = field(default_factory=list)  # per-side driver stats
    models: list = field(default_factory=list)        # model labels
    config: dict = field(default_factory=dict)
    timestamp: str = ""
    series: dict = field(default_factory=dict)        # speed/lateral time series
    events: list = field(default_factory=list)        # in-race event log

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "seed": self.seed,
            "laps": self.laps,
            "mode": self.mode,
            "duration": round(self.duration, 3),
            "winner": self.winner,
            "loser": self.loser,
            "finished_by": self.finished_by,
            "end_reason": self.end_reason,
            "cars": self.cars,
            "driver_stats": self.driver_stats,
            "models": self.models,
            "config": self.config,
            "timestamp": self.timestamp,
            "series": self.series,
            "events": self.events,
        }


# ---------------------------------------------------------------------------
class RaceGame:
    LOOKAHEAD = [8.0, 18.0, 30.0, 45.0, 65.0, 90.0, 125.0]
    SERIES_HZ = 8.0          # telemetry sampling rate for the report curves

    def __init__(self, track: Track, cfg: MatchConfig,
                 drivers: list, model_labels: list,
                 colors=CAR_COLORS,
                 match_id: int = 0):
        self.track = track
        self.cfg = cfg
        self.drivers = drivers            # list[BaseDriver | None]
        self.model_labels = model_labels
        self.match_id = match_id

        self.cars = [
            Car("P1", colors[0], track, spawn_arc=6.0, lateral=-3.2,
                is_human=(drivers[0] is None),
                max_speed=42.0, accel=18.0),
            Car("P2", colors[1], track, spawn_arc=6.0, lateral=3.2,
                is_human=(drivers[1] is None),
                max_speed=42.0, accel=18.0),
        ]
        for c in self.cars:
            c.laps_total = cfg.laps

        self.time = 0.0
        self.hazards: list = []
        self.events: list = []
        # sampled each SERIES_HZ for the training report's speed curves
        self.series = {"t": [], "speed": [[], []], "lateral": [[], []]}
        self._series_next = 0.0
        self.state = "racing"             # racing | finished
        self.winner: Optional[int] = None
        self.loser: Optional[int] = None
        self.finished_by: Optional[int] = None
        self.result: Optional[MatchResult] = None
        self.max_time = max(120.0, cfg.laps * 70.0 + 60.0)
        self.finish_grace = 20.0
        self.winner_time = 0.0
        self.launch_time = 0.7          # equal full-throttle launch window
        self.disqualified: Optional[int] = None
        self.end_reason = ""
        self._last_obs = [None, None]
        self._last_car_contact = -999.0
        self._prev_progress = [0.0, 0.0]
        # automatic un-stick bookkeeping (see _check_stuck)
        self._progress_ref = [-1.0, -1.0]
        self._progress_seen_at = [0.0, 0.0]
        self._unstuck_at = [-999.0, -999.0]
        self.unstick_cooldown = 4.0
        self.last_action = [Action(), Action()]
        self.last_reason = ["", ""]
        self._finish_order = []
        self.anticheat = AntiCheat(track, enabled=True)

    # ------------------------------------------------------------------
    def prime_drivers(self) -> None:
        """Let every AI side take its opening look while held at the line.

        Called before the countdown, so the first frame render and the first
        model round-trip happen off the race clock.  Each side is primed from
        the same tick with the same kind of observation, so neither car gains
        an advantage -- both simply arrive at the green with a decision.
        """
        for idx, driver in enumerate(self.drivers):
            if driver is None or not hasattr(driver, "prime"):
                continue
            try:
                driver.prime(self.create_observation(idx))
            except Exception:  # noqa: BLE001 - a prime is best-effort
                pass

    # ------------------------------------------------------------------
    def _race_progress(self, car: Car, idx: int) -> float:
        total = self.track.length * max(1, self.cfg.laps)
        return (car.arc_unwrapped - car.start_arc) / max(total, 1e-6)

    def create_observation(self, idx: int) -> Observation:
        car = self.cars[idx]
        opp = self.cars[1 - idx]
        i = car.center_i
        t = car.center_t
        p = car.center_proj
        d = car.center_dist
        lat = car.lateral
        tang = car.center_tang
        base_arc = self.track.arc_at(i, t)

        # lookahead set shortened when slow for accuracy
        dists = self.LOOKAHEAD
        pts = self.track.points_ahead(base_arc, dists)
        curvs = self.track.curvature_ahead(base_arc, dists)

        # obstacles in front (within 45 m)
        fwd = (math.cos(car.heading), math.sin(car.heading))
        left = (-math.sin(car.heading), math.cos(car.heading))
        obstacles = []
        for obs in self.track.obstacles:
            relx, relz = obs.x - car.pos[0], obs.z - car.pos[1]
            proj = relx * fwd[0] + relz * fwd[1]
            # include obstacles the car is alongside / on top of (proj < 0):
            # excluding them is what let a wedged car keep ramming blind
            if -5.0 < proj < 48.0:
                obstacles.append((float(obs.x), float(obs.z), float(obs.radius)))

        # opponent relations
        dx = opp.pos[0] - car.pos[0]
        dz = opp.pos[1] - car.pos[1]
        opp_dist = math.hypot(dx, dz)
        opp_ahead = (dx * fwd[0] + dz * fwd[1]) > 0
        oi, ot, op, od, olat, otang = self.track.nearest(opp.pos, opp._hint)
        opp._hint = oi

        # race-relative position: the only rival information a driver gets
        my_prog = self._race_progress(car, idx)
        opp_prog = self._race_progress(opp, 1 - idx)
        gap = my_prog - opp_prog
        total_m = self.track.length * max(1, self.cfg.laps)

        obs = Observation(
            time=self.time,
            dt=1.0 / max(1, self.cfg.fps),
            pos=(float(car.pos[0]), float(car.pos[1])),
            heading=float(car.heading),
            speed=float(car.speed),
            # the *effective* top speed: it is higher while a nitro is live and
            # lower while slowed, so drivers can price the boost correctly
            max_speed=float(car.effective_max_speed(self.time)),
            lateral=float(lat),
            half_width=float(self.track.half_width),
            track_index=int(i),
            lap=int(car.laps),
            laps_total=int(self.cfg.laps),
            race_progress=float(my_prog),
            wall_hits=int(car.wall_hits),
            obstacle_hits=int(car.obstacle_hits),
            effect=car.effect_name(self.time),
            items=car.items.state(self.time),
            opp_pos=(float(opp.pos[0]), float(opp.pos[1])),
            opp_speed=float(opp.speed),
            opp_lateral=float(olat),
            opp_ahead=bool(opp_ahead),
            opp_distance=float(opp_dist),
            opp_progress=float(opp_prog),
            rival_gap=float(gap),
            rival_gap_m=float(gap * total_m),
            lookahead_distances=list(dists),
            lookahead_points=[(float(a), float(b)) for a, b in pts],
            lookahead_curvatures=[float(c) for c in curvs],
            hazards=[(h["x"], h["z"], h["radius"], h["owner"])
                     for h in self.hazards if h["expire"] > self.time],
            obstacles_ahead=obstacles,
        )
        return obs

    # ------------------------------------------------------------------
    def _sample_series(self) -> None:
        """Append one telemetry sample per car (used by the PDF report)."""
        self.series["t"].append(round(self.time, 2))
        for idx, car in enumerate(self.cars):
            self.series["speed"][idx].append(round(float(car.speed), 2))
            self.series["lateral"][idx].append(round(float(car.lateral), 2))

    # ------------------------------------------------------------------
    def _log(self, text: str) -> None:
        self.events.append((round(self.time, 2), text))
        if len(self.events) > 40:
            self.events.pop(0)

    def apply_item(self, idx: int, key: str) -> bool:
        car = self.cars[idx]
        if not self.cfg.items_enabled:
            return False
        if key not in ITEM_MAP or not car.items.available(key, self.time):
            return False
        now = self.time
        other = self.cars[1 - idx]
        label = ITEM_MAP[key].name_cn

        if key == "NITRO":
            car.apply_boost(now, ITEM_MAP[key].duration)
        elif key == "SHIELD":
            car.apply_shield(now, ITEM_MAP[key].duration)
        elif key == "EMP":
            dx = other.pos[0] - car.pos[0]
            dz = other.pos[1] - car.pos[1]
            fwd = (math.cos(car.heading), math.sin(car.heading))
            ahead = (dx * fwd[0] + dz * fwd[1]) > 0
            if ahead and math.hypot(dx, dz) < 42.0:
                other.apply_slow(now, ITEM_MAP[key].duration, 0.42)
            else:
                car.items.fail_lockout(key, now, 0.8)
                return False
        elif key == "OIL":
            fwd = (math.cos(car.heading), math.sin(car.heading))
            x = car.pos[0] - fwd[0] * 3.5
            z = car.pos[1] - fwd[1] * 3.5
            self.hazards.append({"x": float(x), "z": float(z), "radius": 2.8,
                                 "owner": car.name, "y": float(car.y),
                                 "expire": now + ITEM_MAP[key].duration})
        elif key == "RECOVER":
            i, t, p, d, lat, tang = self.track.nearest(car.pos, car._hint)
            car.pos = p.copy()
            car.heading = math.atan2(float(tang[1]), float(tang[0]))
            car.speed = min(car.speed, 14.0)
            # Legitimate engine-side teleport: tell the anti-cheat so the
            # repositioning is not flagged as a cheat and ends the match.
            self.anticheat.exempt_next_teleport(f"P{idx + 1}")
        car.items.use(key, now)
        car.item_uses[key] = car.item_uses.get(key, 0) + 1
        self._log(f"{car.name} 使用 {label}")
        return True

    # ------------------------------------------------------------------
    def step(self, dt: float, human_actions: Optional[dict] = None):
        if self.state != "racing":
            return
        self.time += dt
        world = World(time=self.time, hazards=self.hazards, opponent=None)

        # gather observations + actions
        for idx, car in enumerate(self.cars):
            obs = self.create_observation(idx)
            self._last_obs[idx] = obs
            driver = self.drivers[idx]
            if car.finished:
                action = Action(throttle=0.0, brake=0.4)
                reason = "已完赛"
            elif driver is not None:
                raw = driver.act(obs)
                reason = getattr(raw, "reason", "") or getattr(driver, "raw_reason", "") or driver.kind
                action = self.anticheat.validate_action(f"P{idx + 1}", raw, obs)
            else:
                action = (human_actions or {}).get(idx, Action())
                reason = "human"
            self.last_action[idx] = action
            self.last_reason[idx] = reason
            # Fair start: for a short window both cars get full throttle so a
            # human's reaction time cannot hand the AI a free head start.
            if self.time <= self.launch_time and not car.finished:
                action = Action(throttle=1.0, brake=0.0, steer=action.steer,
                                use_item=action.use_item, reason=action.reason)
                self.last_action[idx] = action
            if action.use_item:
                available = (self.cfg.items_enabled and
                             action.use_item in ITEM_MAP and
                             car.items.available(action.use_item, self.time))
                # Human key presses are trusted: requesting an item while it is
                # cooling down is normal, not cheating.
                if driver is not None:
                    self.anticheat.validate_item_use(f"P{idx + 1}",
                                                     action.use_item, available)
                self.apply_item(idx, action.use_item)
            car.update(dt, action, world)
            self.anticheat.validate_telemetry(f"P{idx + 1}", car, dt,
                                              car.base_max_speed * 1.55)
            self.anticheat.validate_wrong_way(f"P{idx + 1}", car, dt)

        # A serious rule violation (anti-cheat) immediately ends the race and
        # the offending side is disqualified.  Benign input issues (cooldown
        # press, out-of-range value) are only logged, never fatal.
        if self.anticheat.has_serious():
            v = self.anticheat.violations[-1]
            bad = 1 if v.side == "P2" else 0
            self.disqualified = bad
            self.end_reason = f"{v.side} 违规：{v.detail} — 判负"
            self._log(f"反作弊: {v.side} {v.detail} — 比赛结束")
            self.state = "finished"
            self._finalize()
            return

        self._resolve_car_collision()
        self.hazards = [h for h in self.hazards if h["expire"] > self.time]

        if self.cfg.auto_unstick:
            self._check_stuck()

        if self.time >= self._series_next:
            self._series_next = self.time + 1.0 / self.SERIES_HZ
            self._sample_series()

        # finish detection: the race only ends once every car has finished;
        # the first to cross wins, the last to cross loses.
        for idx, car in enumerate(self.cars):
            if not car.finished and car.laps >= self.cfg.laps:
                car.finished = True
                car.finish_time = self.time
                self._log(f"{car.name} 冲过终点!")
                if self.winner is None:
                    self.winner = idx
                    self.winner_time = self.time
                self.finished_by = idx
                self._finish_order.append(idx)

        # The match ends only once BOTH cars have finished: the loser's finish
        # time and completion status are part of the result, so the race must
        # not stop the moment the winner crosses the line.
        if all(c.finished for c in self.cars):
            self.state = "finished"
            self._finalize()
            return

        if self.time >= self.max_time:
            self._log("比赛超时结束")
            self.state = "finished"
            self._finalize()

    # ------------------------------------------------------------------
    def _resolve_car_collision(self) -> None:
        a, b = self.cars
        c1, c2 = _segment_closest(a._capsule(), b._capsule())
        dx = c2[0] - c1[0]
        dz = c2[1] - c1[1]
        dist = math.hypot(dx, dz)
        min_d = a.car_half_wid + b.car_half_wid
        if dist < min_d:
            if dist > 1e-6:
                nx, nz = dx / dist, dz / dist
            else:                       # coincident capsules: split sideways
                f = a.forward
                nx, nz = -f[1], f[0]
            # push them a hair past contact so they don't re-lock next tick
            overlap = (min_d - dist) * 0.5 + 0.02
            a.pos[0] -= nx * overlap
            a.pos[1] -= nz * overlap
            b.pos[0] += nx * overlap
            b.pos[1] += nz * overlap
            # Only penalise *new* contact: applying the damping every frame
            # while two cars touch used to pin both at a ~5 m/s equilibrium
            # (dist stuck exactly at the collision radius), so neither car
            # could ever pull away.
            if self.time - self._last_car_contact > 0.35:
                self._last_car_contact = self.time
                a.speed *= 0.90
                b.speed *= 0.90
                self._log("两车相撞")
            a.contain(self.time)
            b.contain(self.time)

    # ------------------------------------------------------------------
    def _check_stuck(self) -> None:
        """Return any car that stopped making progress to the centre line.

        A car can end up wedged against an obstacle or a barrier and grind
        there indefinitely -- the engine cannot always slide it free, and the
        drivers' vision-based avoidance is a reflex that can still lose a
        wedged car.  If the race position
        has not advanced for ``auto_unstick_delay`` seconds the car is placed
        back on the middle of the road, which the anti-cheat is told about so
        it is not mistaken for a teleport.
        """
        for idx, car in enumerate(self.cars):
            if car.finished:
                continue
            prog = self._race_progress(car, idx)
            if prog > self._progress_ref[idx] + 0.004:
                self._progress_ref[idx] = prog
                self._progress_seen_at[idx] = self.time
                continue
            stalled = self.time - self._progress_seen_at[idx]
            if (stalled > self.cfg.auto_unstick_delay and
                    self.time - self._unstuck_at[idx] > self.unstick_cooldown):
                self._reposition_to_centre(idx)

    def _centre_spot(self, car):
        """A clear point on the centre line, preferring the nearest one."""
        i, t, p, d, lat, tang = self.track.nearest(car.pos, car._hint)
        base_arc = self.track.arc_at(i, t)
        for ahead in (0.0, 4.0, 8.0, 12.0):
            ii, tt, pp = self.track.point_at_arc(base_arc + ahead)
            if not self._spot_blocked(float(pp[0]), float(pp[1]), car.car_radius):
                tg = self.track.tangent[ii]
                return (np.asarray(pp, dtype=np.float64), ii,
                        math.atan2(float(tg[1]), float(tg[0])))
        # everything nearby is occupied: take the nearest centre point anyway
        return (np.asarray(p, dtype=np.float64), i,
                math.atan2(float(tang[1]), float(tang[0])))

    def _spot_blocked(self, x: float, z: float, radius: float) -> bool:
        for o in self.track.obstacles:
            if math.hypot(o.x - x, o.z - z) < o.radius + radius + 0.6:
                return True
        return False

    def _reposition_to_centre(self, idx: int) -> None:
        car = self.cars[idx]
        pos, hint, heading = self._centre_spot(car)
        car.pos = pos
        car.heading = heading
        car.speed = min(car.speed, 10.0)
        car._hint = hint
        car._project()
        car.y = self.track.height_at_index(car.center_i, car.center_t)
        # a legitimate engine-side reposition, not a cheat
        self.anticheat.exempt_next_teleport(f"P{idx + 1}")
        self._unstuck_at[idx] = self.time
        self._progress_ref[idx] = self._race_progress(car, idx)
        self._progress_seen_at[idx] = self.time
        self._log(f"{car.name} 卡住，已自动回到赛道中线")

    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        car_stats = []
        for idx, car in enumerate(self.cars):
            t = car.telemetry()
            t["progress"] = round(self._race_progress(car, idx), 4)
            t["is_human"] = car.is_human
            t["model"] = self.model_labels[idx]
            car_stats.append(t)

        driver_stats = []
        for idx, d in enumerate(self.drivers):
            if d is None:
                driver_stats.append({"driver": "human", "model": self.model_labels[idx]})
            else:
                st = dict(d.stats())
                # keep the AI's own trace so the training report can show logs
                st["last_reason"] = str(getattr(d, "last_reason", ""))[:160]
                log = getattr(d, "call_log", None)
                if log:
                    st["log"] = [dict(x) for x in log[-12:]]
                driver_stats.append(st)

        # last to finish loses; a disqualified car always loses
        if self.disqualified is not None:
            self.loser = self.disqualified
            self.winner = 1 - self.disqualified
        elif len(self._finish_order) == len(self.cars) and self._finish_order:
            self.loser = self._finish_order[-1]
        elif unfinished := [i for i, c in enumerate(self.cars) if not c.finished]:
            # on a timeout the loser is whoever got *least* far, not always P1
            self.loser = min(unfinished,
                             key=lambda i: self._race_progress(self.cars[i], i))
        else:
            self.loser = None

        self.result = MatchResult(
            match_id=self.match_id,
            seed=self.track.seed,
            laps=self.cfg.laps,
            mode=self.cfg.mode,
            duration=self.time,
            winner=self.winner,
            loser=self.loser,
            finished_by=self.finished_by,
            end_reason=self.end_reason,
            cars=car_stats,
            driver_stats=driver_stats,
            models=list(self.model_labels),
            config=self.cfg.to_dict(),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            series={
                "t": list(self.series["t"]),
                "speed": [list(self.series["speed"][0]),
                          list(self.series["speed"][1])],
                "lateral": [list(self.series["lateral"][0]),
                            list(self.series["lateral"][1])],
                "names": [str(x) for x in self.model_labels],
            },
            events=[[round(float(t), 2), str(s)] for t, s in self.events],
        )
        if self.anticheat.has_cheated():
            self.result.config = dict(self.result.config)
            self.result.config["anticheat"] = self.anticheat.summary()

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Compact state for the renderer / status reporting."""
        return {
            "time": self.time,
            "state": self.state,
            "laps": self.cfg.laps,
            "winner": self.winner,
            "loser": self.loser,
            "debug": bool(getattr(self.cfg, "debug", False)),
            "cars": [
                {
                    "pos": (float(c.pos[0]), float(c.pos[1])),
                    "y": float(c.y),
                    "heading": float(c.heading),
                    "speed": float(c.speed),
                    "color": c.color,
                    "name": c.name,
                    "model": (self.model_labels[i]
                              if i < len(self.model_labels) else c.name),
                    "laps": c.laps,
                    "finished": c.finished,
                    "finish_time": c.finish_time,
                    "effect": c.effect_name(self.time),
                    "wall_hits": c.wall_hits,
                    "obstacle_hits": c.obstacle_hits,
                    "hazard_hits": c.hazard_hits,
                    "item_uses": int(sum(c.item_uses.values())),
                    "progress": round(self._race_progress(c, i), 4),
                    "action": {
                        "throttle": round(self.last_action[i].throttle, 2),
                        "brake": round(self.last_action[i].brake, 2),
                        "steer": round(self.last_action[i].steer, 2),
                        "use_item": self.last_action[i].use_item,
                    },
                    "reason": self.last_reason[i],
                }
                for i, c in enumerate(self.cars)
            ],
            "drivers": [self._driver_debug(i) for i in range(len(self.cars))],
            "hazards": list(self.hazards),
            "events": list(self.events[-6:]),
        }

    def _driver_debug(self, idx: int) -> dict:
        d = self.drivers[idx]
        if d is None:
            return {"driver": "human", "model": "human"}
        info = {
            "driver": getattr(d, "kind", "?"),
            "model": self.model_labels[idx] if idx < len(self.model_labels) else "?",
        }
        try:
            info.update(d.stats())
        except Exception:
            pass
        if hasattr(d, "debug_info"):
            try:
                info.update(d.debug_info())
            except Exception:
                pass
        return info
