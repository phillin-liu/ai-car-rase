"""Shared dataclasses passed between the engine and drivers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Action:
    """A control command produced by a driver for a single simulation tick."""

    throttle: float = 0.0   # 0..1
    brake: float = 0.0      # 0..1
    steer: float = 0.0      # -1 (left) .. 1 (right)
    use_item: Optional[str] = None
    reason: str = ""

    def clamped(self) -> "Action":
        return Action(
            throttle=max(0.0, min(1.0, float(self.throttle))),
            brake=max(0.0, min(1.0, float(self.brake))),
            steer=max(-1.0, min(1.0, float(self.steer))),
            use_item=self.use_item,
            reason=self.reason,
        )


@dataclass
class Observation:
    """Everything a driver is allowed to know about the current world state."""

    time: float = 0.0
    dt: float = 0.0

    # --- own car -------------------------------------------------------
    pos: tuple = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0
    max_speed: float = 1.0
    lateral: float = 0.0            # signed offset from centerline (+ = left)
    half_width: float = 8.0
    track_index: int = 0
    lap: int = 0
    laps_total: int = 1
    race_progress: float = 0.0      # 0..1 across the whole race
    wall_hits: int = 0
    obstacle_hits: int = 0
    effect: str = ""                # "", "boost", "shield", "slow"

    # --- items ---------------------------------------------------------
    items: dict = field(default_factory=dict)   # key -> {available, ready_in}

    # --- opponent ------------------------------------------------------
    opp_pos: tuple = (0.0, 0.0)
    opp_speed: float = 0.0
    opp_lateral: float = 0.0
    opp_ahead: bool = True
    opp_distance: float = 0.0
    opp_progress: float = 0.0

    # --- rival race position (the only thing a driver "knows" about the
    #     other car: who is in front, and by how much) -------------------
    rival_gap: float = 0.0        # race_progress - opponent's, + = I lead
    rival_gap_m: float = 0.0      # same, approximated in metres

    # --- track lookahead ----------------------------------------------
    lookahead_distances: list = field(default_factory=list)
    lookahead_points: list = field(default_factory=list)
    lookahead_curvatures: list = field(default_factory=list)

    # --- dynamic hazards / on-track objects ---------------------------
    hazards: list = field(default_factory=list)      # (x, z, radius, owner)
    obstacles_ahead: list = field(default_factory=list)  # (x, z, radius)

    def to_dict(self) -> dict:
        """Textual telemetry -- for local tools/tests ONLY.

        The AI drivers no longer receive this (nor any other textual scene
        description): vision models get the rendered frame, and program-side
        perception happens through :mod:`car_game.vision`.  Never send this to
        a model.
        """
        d = {
            "time": round(self.time, 2),
            "pos": [round(self.pos[0], 1), round(self.pos[1], 1)],
            "heading": round(self.heading, 3),
            "speed": round(self.speed, 2),
            "max_speed": round(self.max_speed, 2),
            "lateral": round(self.lateral, 2),
            "half_width": round(self.half_width, 2),
            "lap": self.lap,
            "laps_total": self.laps_total,
            "wall_hits": self.wall_hits,
            "obstacle_hits": self.obstacle_hits,
            "effect": self.effect,
            "items": {k: round(v.get("ready_in", 0.0), 1)
                      for k, v in self.items.items() if v.get("available")},
            "opponent": {
                "distance": round(self.opp_distance, 1),
                "ahead": self.opp_ahead,
                "speed": round(self.opp_speed, 2),
                "lateral": round(self.opp_lateral, 2),
            },
            # Deliberately thin: the driver is told the race gap and nothing
            # else about the rival, which is what makes "chase or defend"
            # the decision it has to make rather than a telemetry readout.
            "rival": {
                "status": ("领先" if self.rival_gap_m > 2.0 else
                           "落后" if self.rival_gap_m < -2.0 else "并排"),
                "gap_m": round(self.rival_gap_m, 1),
            },
            "curvatures": [round(c, 5) for c in self.lookahead_curvatures],
        }
        return d
