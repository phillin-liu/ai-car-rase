"""Race items and per-car cooldown handling.

Every item has an independent cooldown.  While an item is cooling down it
cannot be used again, which is the core balance rule requested.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ItemDef:
    key: str
    name_cn: str
    name_en: str
    cooldown: float
    duration: float
    kind: str        # self | target | hazard | recover
    color: tuple
    desc: str


ITEM_DEFS = [
    ItemDef("NITRO", "氮气加速", "Nitro", 6.0, 2.5, "self",
            (1.0, 0.55, 0.10), "短时间内大幅提升极速与加速度"),
    ItemDef("SHIELD", "能量护盾", "Shield", 10.0, 3.0, "self",
            (0.25, 0.75, 1.0), "期间免疫墙壁与障碍物造成的减速"),
    ItemDef("EMP", "电磁干扰", "EMP", 12.0, 2.0, "target",
            (0.85, 0.25, 1.0), "干扰前方一定范围内的对手，使其减速"),
    ItemDef("OIL", "油污陷阱", "Oil Slick", 8.0, 8.0, "hazard",
            (0.18, 0.18, 0.20), "在身后留下油污，对手驶过会被减速"),
    ItemDef("RECOVER", "紧急复位", "Recover", 7.0, 0.0, "recover",
            (0.35, 1.0, 0.45), "将赛车复位到赛道中心线并恢复朝向"),
]

ITEM_MAP = {d.key: d for d in ITEM_DEFS}
ITEM_KEYS = [d.key for d in ITEM_DEFS]


class ItemController:
    """Tracks cooldowns for a single car."""

    def __init__(self) -> None:
        self.ready_at = {k: 0.0 for k in ITEM_KEYS}
        self.use_count = {k: 0 for k in ITEM_KEYS}
        self.last_used_at = {k: -999.0 for k in ITEM_KEYS}

    def available(self, key: str, now: float) -> bool:
        return now >= self.ready_at.get(key, 0.0)

    def ready_in(self, key: str, now: float) -> float:
        return max(0.0, self.ready_at.get(key, 0.0) - now)

    def use(self, key: str, now: float) -> None:
        d = ITEM_MAP[key]
        self.ready_at[key] = now + d.cooldown
        self.use_count[key] = self.use_count.get(key, 0) + 1
        self.last_used_at[key] = now

    def fail_lockout(self, key: str, now: float, t: float = 0.6) -> None:
        """Small lockout when an item could not be applied (e.g. EMP out of range)."""
        self.ready_at[key] = max(self.ready_at.get(key, 0.0), now + t)

    def state(self, now: float) -> dict:
        return {
            k: {"available": self.available(k, now),
                "ready_in": round(self.ready_in(k, now), 2)}
            for k in ITEM_KEYS
        }

    def available_keys(self, now: float):
        return [k for k in ITEM_KEYS if self.available(k, now)]
