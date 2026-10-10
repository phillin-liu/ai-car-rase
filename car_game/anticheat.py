"""Anti-cheat validation for AI drivers.

The game engine already clamps every action, but a cheating driver could
still abuse the interface in subtler ways (request items that are on
cooldown, teleport, exceed the physical speed cap, or submit non-finite
values).  This module validates the *raw* driver output and the resulting
telemetry and records violations.

A Rust implementation of the same checks lives in ``anticheat/`` so the
heavy batch validation can run out-of-process.  It is run straight from
source through ``cargo run`` -- there is no prebuilt binary to ship, so a
fresh clone works as long as the Rust toolchain is installed.  When cargo or
the source is missing we fall back to the pure-Python validator below.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from .config import ROOT
from .core_types import Action
from .items import ITEM_MAP

RUST_DIR = os.path.join(ROOT, "anticheat")
RUST_SRC = os.path.join(RUST_DIR, "src", "main.rs")
RUST_MANIFEST = os.path.join(RUST_DIR, "Cargo.toml")


@dataclass
class Violation:
    side: str
    code: str
    detail: str


# Only these codes represent real cheating and end the match.  Input mistakes
# such as pressing an item during cooldown or a slightly out-of-range value
# are logged but must NOT reset the race.
SERIOUS_CODES = {
    "null_action", "non_finite", "bad_item", "non_finite_pos", "bad_speed",
    "teleport", "speed_cap", "wrong_way",
}


class AntiCheat:
    """Validates driver output and car telemetry for impossible behaviour."""

    MAX_SPEED_FACTOR = 1.12       # small tolerance over the theoretical max
    MAX_POS_DELTA_FACTOR = 4.0    # generous: collision resolution can push a car
    MAX_POS_DELTA_SLACK = 4.0     # extra metres of tolerance per tick

    def __init__(self, track=None, enabled: bool = True):
        self.enabled = enabled
        self.track = track
        self.violations: list = []
        self._last_pos = {}
        self._last_speed = {}
        self._wrong_time = {}
        self._wrong_flag = {}
        self._reported = {}
        self._tp_exempt: set = set()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.violations.clear()
        self._last_pos.clear()
        self._last_speed.clear()
        self._wrong_time.clear()
        self._wrong_flag.clear()
        self._reported.clear()
        self._tp_exempt.clear()

    def exempt_next_teleport(self, side: str) -> None:
        """Whitelist one legitimate repositioning (RECOVER item).

        The engine itself moves the car instantly back onto the centre
        line, which is otherwise indistinguishable from a teleport cheat.
        The next telemetry check for ``side`` skips the jump test.
        """
        self._tp_exempt.add(side)

    def record(self, side: str, code: str, detail: str) -> None:
        if self.enabled:
            self.violations.append(Violation(side, code, detail))
            if len(self.violations) > 400:
                del self.violations[:100]

    # ------------------------------------------------------------------
    def validate_action(self, side: str, action: Action, obs) -> Action:
        """Return a sanitized Action; log anything the driver got wrong."""
        a = action
        if a is None:
            self.record(side, "null_action", "返回了空动作(None)")
            return Action()

        def bad(v):
            return v is None or (isinstance(v, (int, float)) and not math.isfinite(float(v)))

        if bad(a.throttle) or bad(a.brake) or bad(a.steer):
            self.record(side, "non_finite", "动作包含非法数值(NaN/Inf)")
        else:
            if not (0.0 <= float(a.throttle) <= 1.0):
                self.record(side, "throttle_range", f"油门超出 0~1：{a.throttle}")
            if not (0.0 <= float(a.brake) <= 1.0):
                self.record(side, "brake_range", f"刹车超出 0~1：{a.brake}")
            if not (-1.0 <= float(a.steer) <= 1.0):
                self.record(side, "steer_range", f"转向超出 -1~1：{a.steer}")
        if a.use_item is not None and a.use_item not in ITEM_MAP:
            self.record(side, "bad_item", f"请求了不存在的道具：{a.use_item!r}")
        return a.clamped()

    def validate_item_use(self, side: str, key: str, available: bool) -> None:
        if key is not None and not available:
            self.record(side, "item_cooldown",
                        f"冷却未结束时强行使用道具 {key}")

    # ------------------------------------------------------------------
    def validate_telemetry(self, side: str, car, dt: float, max_speed: float) -> None:
        if not self.enabled or car is None:
            return
        try:
            pos = (float(car.pos[0]), float(car.pos[1]))
            speed = float(car.speed)

            if not (math.isfinite(pos[0]) and math.isfinite(pos[1])):
                self.record(side, "non_finite_pos", "车辆坐标变为非法值(NaN/Inf)")
            if speed < -1e-3 or not math.isfinite(speed):
                self.record(side, "bad_speed", f"出现非法速度：{speed}")

            # no teleporting: only flag grossly impossible jumps.  The physics
            # engine legitimately nudges cars (wall clamp / car separation), so
            # the tolerance is deliberately generous to avoid false positives.
            # A pending RECOVER exemption skips the jump test for this frame.
            if side in self._tp_exempt:
                self._tp_exempt.discard(side)
                self._last_pos[side] = pos
                self._last_speed[side] = speed
                return
            last = self._last_pos.get(side)
            if last is not None:
                dx = pos[0] - last[0]
                dz = pos[1] - last[1]
                moved = math.hypot(dx, dz)
                max_moved = (max(abs(speed), abs(self._last_speed.get(side, 0.0)),
                                 1.0) * dt * self.MAX_POS_DELTA_FACTOR
                             + self.MAX_POS_DELTA_SLACK)
                if moved > max_moved and moved > 6.0:
                    self.record(side, "teleport",
                                f"瞬移：{dt:.3f}s 内移动 {moved:.2f}m（上限 {max_moved:.2f}m）")

            # no exceeding the physical speed cap (accounting for nitro)
            if speed > max_speed * self.MAX_SPEED_FACTOR:
                self.record(side, "speed_cap",
                            f"超过速度上限：{speed:.2f} > {max_speed:.2f}")

            self._last_pos[side] = pos
            self._last_speed[side] = speed
        except Exception:
            pass

    # ------------------------------------------------------------------
    def validate_wrong_way(self, side: str, car, dt: float,
                           min_speed: float = 5.0, hold: float = 1.5) -> None:
        """Flag a car that is driving against the track direction (逆行)."""
        if not self.enabled or car is None or self.track is None:
            return
        try:
            fwd = car.forward
            tang = car.center_tang
            dot = float(fwd[0] * tang[0] + fwd[1] * tang[1])
            if float(car.speed) > min_speed and dot < -0.35:
                self._wrong_time[side] = self._wrong_time.get(side, 0.0) + dt
                if self._wrong_time[side] >= hold and not self._wrong_flag.get(side):
                    self._wrong_flag[side] = True
                    self.record(side, "wrong_way", "逆行：车头方向与赛道方向相反(在反着跑)")
            else:
                self._wrong_time[side] = 0.0
                self._wrong_flag[side] = False
        except Exception:
            pass

    # ------------------------------------------------------------------
    def live_reasons(self, limit: int = 3) -> list:
        """Human readable anti-cheat messages for the HUD (most recent)."""
        out = []
        for v in self.violations[-limit:]:
            out.append(f"{v.side} {v.detail}")
        return out

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        codes = {}
        for v in self.violations:
            codes[v.code] = codes.get(v.code, 0) + 1
        return {
            "violations": len(self.violations),
            "by_code": codes,
            "details": [{"side": v.side, "code": v.code, "detail": v.detail}
                        for v in self.violations[-20:]],
        }

    def has_cheated(self) -> bool:
        return len(self.violations) > 0

    def has_serious(self) -> bool:
        """True only for violations serious enough to end the match."""
        return any(v.code in SERIOUS_CODES for v in self.violations)


# ---------------------------------------------------------------------------
# Optional Rust-backed batch validator (compiled and run from source)
#
# ``cargo run`` reuses the build in ``anticheat/target/`` when the source has
# not changed, so repeat calls only pay cargo's startup.  The first call on a
# cold clone compiles serde + serde_json once, hence the generous timeout.
# ---------------------------------------------------------------------------
def cargo_available() -> bool:
    return shutil.which("cargo") is not None


def rust_source_available() -> bool:
    """True when the Rust validator can be compiled and run from source."""
    return os.path.exists(RUST_SRC) and cargo_available()


def build_rust() -> tuple[bool, str]:
    """Pre-compile the Rust validator (requires cargo).

    Optional -- ``validate_results_rust`` compiles on demand anyway; this just
    keeps the first real verification from paying for the build.
    """
    if not os.path.isdir(RUST_DIR):
        return False, f"找不到 Rust 源码目录 {RUST_DIR}"
    try:
        proc = subprocess.run(
            ["cargo", "build", "--release", "--quiet"],
            cwd=RUST_DIR, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return False, proc.stderr.strip()[-800:]
        return True, os.path.join(RUST_DIR, "target", "release", "ac-validate")
    except FileNotFoundError:
        return False, "未安装 cargo / Rust 工具链"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def validate_results_rust(matches: list) -> dict:
    """Validate a list of raw match result dicts with the Rust source.

    The validator is compiled and executed by cargo on demand, so a fresh
    checkout needs only the Rust toolchain -- nothing prebuilt.  Falls back to
    the pure-Python validator when cargo or the source is unavailable.
    """
    if not rust_source_available():
        return validate_results_python(matches)

    payload = {"matches": _json_safe(matches)}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        path = fh.name
    try:
        proc = subprocess.run(
            ["cargo", "run", "--release", "--quiet",
             "--manifest-path", RUST_MANIFEST, "--", path],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600)
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout)
            except ValueError:
                return {"engine": "rust", "error": proc.stdout[-500:]}
        return {"engine": "rust", "error": proc.stderr.strip()[-500:]}
    except Exception as exc:  # noqa: BLE001
        return {"engine": "rust", "error": str(exc)}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def validate_results_python(matches: list) -> dict:
    """Pure-Python batch validator with the same rules as the Rust binary."""
    ac = AntiCheat()
    for m in matches:
        cars = m.get("cars", []) or []
        for idx, car in enumerate(cars):
            side = car.get("name") or f"P{idx + 1}"
            avg = _f(car.get("avg_speed"))
            maxs = _f(car.get("max_speed"))
            dist = _f(car.get("distance"))
            wall = _f(car.get("wall_hits"))
            obs = _f(car.get("obstacle_hits"))
            haz = _f(car.get("hazard_hits"))
            prog = _f(car.get("progress"))
            ftime = _f(car.get("finish_time"))

            if not math.isfinite(avg) or avg < 0.0:
                ac.record(side, "non_finite", "avg_speed invalid")
            if not math.isfinite(maxs) or maxs > 300.0:
                ac.record(side, "speed_cap", f"max_speed={maxs}")
            if not math.isfinite(dist) or dist < 0.0:
                ac.record(side, "non_finite", "distance invalid")
            if (not math.isfinite(wall) or wall < 0.0
                    or not math.isfinite(obs) or obs < 0.0
                    or not math.isfinite(haz) or haz < 0.0):
                ac.record(side, "negative_hits", "negative collision counters")
            if not math.isfinite(prog) or prog < 0.0:
                ac.record(side, "bad_progress", f"progress={prog}")
            if car.get("finished") is True and (not math.isfinite(ftime) or ftime < 0.0):
                ac.record(side, "bad_finish", "finished but finish_time invalid")
    out = ac.summary()
    out["engine"] = "python"
    return out


def verify_matches(matches: list) -> dict:
    """Verify raw match result dicts, preferring the Rust engine.

    When cargo is present but the Rust run fails (compile error, crash) we
    transparently fall back to the pure-Python validator so verification never
    crashes a caller.
    """
    if rust_source_available():
        result = validate_results_rust(matches)
        if "error" not in result:
            result.setdefault("checked", len(matches))
            return result
        fallback = validate_results_python(matches)
        fallback["rust_error"] = result.get("error")
        fallback["checked"] = len(matches)
        return fallback
    result = validate_results_python(matches)
    result["checked"] = len(matches)
    return result


def format_verdict(verdict: dict) -> str:
    """Render a verification verdict (from verify_matches) as text."""
    lines = []
    engine = verdict.get("engine", "?")
    checked = verdict.get("checked", 0)
    total = verdict.get("violations", 0)
    lines.append(f"引擎: {engine}    已校验对局: {checked}")
    if verdict.get("rust_error"):
        lines.append(f"Rust 运行失败，已回退 Python: {verdict['rust_error']}")
    if total == 0:
        lines.append("结论: 未发现异常，全部数据通过物理合理性校验。")
        return "\n".join(lines)
    lines.append(f"结论: 发现 {total} 处可疑数据。")
    by_code = verdict.get("by_code") or {}
    if by_code:
        lines.append("分类: " + "  ".join(f"{k}={v}" for k, v in by_code.items()))
    for d in verdict.get("details", []):
        lines.append(f"  - [{d.get('side')}] {d.get('code')}: {d.get('detail')}")
    return "\n".join(lines)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _json_safe(value):
    """Recursively replace non-finite floats with None (valid JSON).

    Python's json encoder happily writes bare ``NaN``/``Infinity`` but those
    are not valid JSON and make serde_json reject the whole payload.  The
    Rust validator treats a missing value as ``NaN`` anyway, so ``None``
    preserves the intended verdict.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
