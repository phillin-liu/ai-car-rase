"""Driver backed by a locally served `sidecar` policy (``sidecar-serve``).

This provider races a policy that was trained in the *sidecar* simulator
(``configs/ai_car_race.toml``), not a vision model.  ``sidecar-serve`` exposes
the signed ``.sidecar`` file over HTTP; ``GET /info`` describes the observation
vector (LIDAR ray count/FOV, the ordered scalar names and the normalisation
constants), and ``POST /act`` maps one observation to one discrete action.

The adapter here is the missing half: it builds that observation from the
game's own state and turns the returned action back into a continuous
``Action``.  Keeping the layout in ``/info`` (rather than hard-coding it here)
means a model trained with a different item list, LIDAR count or mode still
works as long as the game can supply the named scalars.

Mismatches that are worth knowing about:

* The trained policy races alone on a random track.  The opponent, the game's
  item traps and the "AI vision" philosophy do not exist in its training
  world, so it ignores them.
* The sidecar corridor is ``wall_offset`` metres from the centre line while
  this game clamps the car centre at ``half_width - car_half_width``.  The
  LIDAR therefore reports the *effective* wall where this game actually stops
  the car, so the policy keeps its learned margin instead of hugging the
  barrier.  Lateral offsets are scaled by the same ratio so the whole
  corridor is presented in the simulator's units.
* The game's steering fades with speed while the simulator's kinematic bicycle
  does not, so the policy's steer *angle* is tracked here and converted to the
  game input that produces the same yaw rate (saturating when the game cannot
  match it).
* The policy decides at 30 Hz (the simulator's ``dt``).  The game runs at
  60 Hz, so every decision is held for about two physics steps, matching the
  training cadence instead of the UI's ``decision_interval`` knob.
"""
from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from ..config import ModelConfig
from ..core_types import Action, Observation
from ..mathutil import clamp, wrap_angle
from .base import BaseDriver

#: The simulator trained one decision per ``dt = 1 / 30``.  Queries are paced
#: at that rate regardless of ``ModelConfig.decision_interval`` (the UI's
#: minimum of 0.1 s would be three times coarser).
DEFAULT_INTERVAL = 1.0 / 30.0

#: Longitudinal / lateral codes used by sidecar's ``prev_*`` observation
#: scalars (``src/env/action.rs``).
_LONG_CODE = {"throttle": 0.0, "brake": 1.0, "coast": 2.0}
_LAT_CODE = {"left": 0.0, "right": 1.0, "straight": 2.0}

#: HTTP statuses that retrying cannot fix (wrong model, wrong observation).
_FATAL_STATUS = frozenset({400, 404, 405, 422})

#: Item keys this game knows; the policy emits the lower-case names.
_ITEM_KEYS = frozenset({"NITRO", "SHIELD", "EMP", "OIL", "RECOVER"})

#: The game stops the car centre this far inside the painted edge
#: (``Car.car_half_width``; note ``car_radius`` is the separate obstacle
#: radius), which is the wall the policy should perceive.
_CAR_HALF_WIDTH = 1.0

#: The game's own steering law (``Car.update``).  The policy was trained on a
#: kinematic bicycle with a *constant* turning radius, while the game fades
#: steering out with speed; these constants let the adapter ask for the yaw
#: rate the policy expects instead of the raw ``-1/0/+1`` steer.
_GAME_STEER_RATE = 2.45
_GAME_GRIP_BASE = 0.25
_GAME_GRIP_SPEED = 4.0
_GAME_DAMP = 0.028

#: Where an unconfigured sidecar-serve is expected to listen.
DEFAULT_URL = "http://127.0.0.1:7879"

#: Model names that mean "whatever this server loaded first", not a route.
#: ``ai-car-race`` is the placeholder this program shipped as the provider
#: default before sidecar-serve exposed one route per policy; configs written
#: back then must keep working (they raced the default policy).
_SERVER_DEFAULT_NAMES = frozenset({"", "ai-car-race", "default", "model"})


class SidecarHTTPError(RuntimeError):
    """A ``sidecar-serve`` reply with an error status, body kept for the UI."""

    def __init__(self, url: str, status: int, body: str = ""):
        self.url = url
        self.status = int(status)
        self.body = str(body or "")
        detail = " ".join(self.body.split())[:200]
        super().__init__(f"{url} 返回 HTTP {self.status}"
                         + (f"：{detail}" if detail else ""))

    def server_error(self) -> str:
        """The server's own explanation, when the body is JSON."""
        try:
            body = json.loads(self.body)
        except ValueError:
            return " ".join(self.body.split())
        if isinstance(body, dict):
            return str(body.get("error") or body.get("message") or "")
        return ""

    def available(self) -> list[str]:
        """Model names named after ``available:`` in the server's reply."""
        text = self.server_error()
        low = text.lower()
        if "available:" not in low:
            return []
        rest = text[low.index("available:") + len("available:"):]
        return [part.strip(" `'") for part in rest.split(",") if part.strip(" `'")]


def model_slug(raw: str) -> str:
    """The URL slug ``sidecar-serve`` derives from a model/file name.

    Mirrors ``slug_from`` in ``src/bin/sidecar_serve.rs``: lower-case ASCII
    alphanumerics, single dashes, no leading/trailing dash.  Accepting the
    ``.sidecar`` file name as well means a user can paste what ``ls`` shows.
    """
    name = str(raw or "").strip()
    if name.lower().endswith(".sidecar"):
        name = name[: -len(".sidecar")]
    out: list[str] = []
    dash = False
    for ch in name.lower():
        if ch.isascii() and ch.isalnum():
            out.append(ch)
            dash = False
        elif out and not dash:
            out.append("-")
            dash = True
    while out and out[-1] == "-":
        out.pop()
    return "".join(out)


def uses_server_default(model: str | None) -> bool:
    """Does ``model`` mean "the server's default policy" rather than a route?"""
    return model_slug(model or "") in _SERVER_DEFAULT_NAMES


def _parts(cfg: ModelConfig) -> tuple[str, str, list[str]]:
    """``(scheme, netloc, path segments)`` of the configured base URL."""
    raw = cfg.resolved_base_url() or DEFAULT_URL
    split = urlsplit(raw.rstrip("/"))
    return split.scheme, split.netloc, [s for s in split.path.split("/") if s]


def _rebuild(scheme: str, netloc: str, segments: list[str]) -> str:
    path = "/" + "/".join(segments) if segments else ""
    return urlunsplit((scheme, netloc, path, "", ""))


def server_url(cfg: ModelConfig) -> str:
    """The server root -- no model slug -- for ``/models`` and error hints.

    Configs written before the per-model routes put the slug in ``base_url``
    (``http://127.0.0.1:7879/ai-car-race-v2-best``); strip it so ``/models``
    still resolves.
    """
    scheme, netloc, segments = _parts(cfg)
    slug = model_slug(cfg.model)
    if slug and segments and segments[-1] == slug:
        segments = segments[:-1]
    return _rebuild(scheme, netloc, segments)


def endpoint(cfg: ModelConfig) -> str:
    """The URL of ``cfg``'s policy (never empty).

    A ``sidecar-serve`` can host several policies at once, one per
    ``/<slug>`` route (``GET /models`` lists them).  The configured ``model``
    therefore selects the route; a bare base URL -- or a legacy name such as
    the old default ``ai-car-race`` -- keeps talking to the server's default.
    """
    scheme, netloc, segments = _parts(cfg)
    slug = model_slug(cfg.model)
    if slug in _SERVER_DEFAULT_NAMES:
        return _rebuild(scheme, netloc, segments)
    if not segments or segments[-1] != slug:
        segments = segments + [slug]
    return _rebuild(scheme, netloc, segments)


def _get(url: str, timeout: float) -> dict:
    """GET + JSON, with the two failures the UI has to explain separately."""
    import requests

    try:
        resp = requests.get(url, timeout=float(timeout))
    except Exception as exc:  # noqa: BLE001 - the caller renders the message
        raise RuntimeError(
            f"无法连接 {url}：请先运行 sidecar-serve（{str(exc)[:100]}）"
        ) from exc
    if resp.status_code >= 400:
        raise SidecarHTTPError(url, resp.status_code, resp.text[:400])
    try:
        body = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"{url} 返回的不是 JSON") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"{url} 返回的不是 JSON 对象")
    return body


def fetch_info(cfg: ModelConfig, timeout: float | None = None) -> dict:
    """``GET /info`` -- model, action table and observation layout."""
    if timeout is None:
        timeout = clamp(float(cfg.timeout or 10.0), 0.5, 30.0)
    return _get(endpoint(cfg) + "/info", float(timeout))


def fetch_models(cfg: ModelConfig, timeout: float | None = None) -> list[dict]:
    """``GET /models`` -- every policy this ``sidecar-serve`` hosts.

    Older servers have no ``/models`` route; for those the single policy at
    ``/info`` is reported instead, so the list is never empty just because the
    server is old.
    """
    if timeout is None:
        timeout = clamp(float(cfg.timeout or 10.0), 0.5, 30.0)
    timeout = float(timeout)
    try:
        body = _get(server_url(cfg) + "/models", timeout)
    except SidecarHTTPError as exc:
        if exc.status != 404:
            raise
        info = _get(server_url(cfg) + "/info", timeout)
        name = policy_name(cfg, info)
        if name.startswith("http"):      # older /info had no name at all
            name = ""
        return [{"slug": model_slug(name) or "default",
                 "name": name or "default", "default": True,
                 "input_dim": int(info.get("input_dim") or 0),
                 "action_size": len(info.get("actions") or [])}]
    entries = body.get("models")
    out: list[dict] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug") or entry.get("name") or "")
        if slug:
            out.append({**entry, "slug": model_slug(slug)})
    return out


def policy_names(cfg: ModelConfig, timeout: float | None = None) -> list[str]:
    """Display names of the hosted policies, default first."""
    models = fetch_models(cfg, timeout)
    models.sort(key=lambda m: (not m.get("default"), str(m.get("slug"))))
    names = [str(m.get("name") or m.get("slug") or "") for m in models]
    return [n for n in names if n]


def default_policy_name(cfg: ModelConfig, timeout: float | None = None) -> str:
    """The routed policy's own name (``""`` when the server cannot say).

    ``/info`` deliberately carries no name; ``/health`` does.  The console uses
    this so a side left on the server default is labelled with the policy it is
    actually racing instead of with the URL.
    """
    if timeout is None:
        timeout = clamp(float(cfg.timeout or 10.0), 0.5, 30.0)
    try:
        body = _get(endpoint(cfg) + "/health", float(timeout))
    except Exception:  # noqa: BLE001 - a label is never worth failing over
        return ""
    return str(body.get("name") or body.get("model") or body.get("file") or "")


def policy_name(cfg: ModelConfig, info: dict) -> str:
    """Label for the served policy.

    Newer ``sidecar-serve`` builds keep ``/info`` minimal and publish the model
    name only in ``/health``; older ones also carried ``model_info.name`` (and
    briefly ``model_file``/``model_path``).  Fall back to the name the user
    configured so the UI is never blank.
    """
    name = ((info.get("model_info") or {}).get("name")
            or info.get("model_name")
            or info.get("model_file")
            or info.get("model_path")
            or ("" if uses_server_default(getattr(cfg, "model", ""))
                else getattr(cfg, "model", "")))
    return str(name or endpoint(cfg))


class SidecarDriver(BaseDriver):
    """Drive from a ``sidecar-serve`` policy."""

    kind = "sidecar"

    def __init__(self, cfg: ModelConfig, track, name: str,
                 difficulty: float = 1.0, stagger: float = 0.0):
        super().__init__(cfg, track, name)
        self.pool = ThreadPoolExecutor(max_workers=1,
                                       thread_name_prefix=f"sidecar-{name}")
        self._lock = threading.Lock()
        self._future = None
        self._decision = None
        self._next_query_at = 0.0
        self._interval = DEFAULT_INTERVAL
        # Same reasoning as the LLM driver: the opening decision is not
        # delayed (both cars must get it on the same tick), every later one
        # inherits the offset so the two sides do not query in lock-step.
        self._stagger = max(0.0, min(0.9, float(stagger))) * self._interval

        self._url = endpoint(cfg)
        self._timeout = clamp(float(cfg.timeout or 10.0), 0.5, 30.0)
        # temperature <= 0 means "greedy": exactly what a race wants.
        self._sample = float(cfg.temperature or 0.0) > 0.0
        self._temperature = max(1e-3, float(cfg.temperature or 0.0))

        self._info = None
        self._input_dim = 0
        self._lidar_rays = 24
        self._lidar_fov = 1.5 * math.pi
        self._lidar_range_mult = 6.0
        self._lidar_range_min = 30.0
        self._lidar_max = 30.0
        # Map the game's (narrower) car-centre corridor onto the simulator's
        # wall offset so lateral offsets and LIDAR distances stay in the units
        # the policy was trained on.
        self._corridor_scale = 1.0
        self._checkpoints = 16
        self._scalar_names: list[str] = []
        self._norm: dict = {}
        self._car_cfg: dict = {}
        self._powerup_items: list[str] = []
        self._actions: dict[int, dict] = {}
        self._action_count = 0
        self._ready = False

        # geometry cache: (track identity, segment starts, segment ends)
        self._geom = None
        self._corridor = 7.0
        self._wall_offset = 0.0
        self._prev_drive = [2.0, 2.0]
        self._prev_heading = None
        self._prev_time = None
        self._pending_item = None
        self._last_submit_time = 0.0
        # Sidecar steer *angle* (radians) state, slew-limited exactly like the
        # simulator; converted to the game's steer input every tick.
        self._steer = 0.0

        self.api_requests = 0
        self.api_errors = 0
        self.latency_sum = 0.0
        self.warmup_requests = 0
        self.warmup_errors = 0
        self.last_reason = ""
        self.last_error = ""
        self.last_action_name = ""
        self.call_log: list[dict] = []
        self.fatal_error = ""
        if track is not None:
            self.reset(track)

    # ------------------------------------------------------------------
    def reset(self, track) -> None:
        super().reset(track)
        with self._lock:
            self._future = None
            self._decision = None
            self._next_query_at = 0.0
        self._pending_item = None
        self._prev_drive = [2.0, 2.0]
        self._prev_heading = None
        self._prev_time = None
        self._steer = 0.0
        self._geom = None
        self.last_error = ""
        self.fatal_error = ""

    def close(self) -> None:
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover
            self.pool.shutdown(wait=False)

    @property
    def has_decision(self) -> bool:
        with self._lock:
            self._harvest()
            return self._decision is not None

    def prime(self, obs: Observation) -> None:
        """Take the opening look while the field is still held at the line."""
        with self._lock:
            if self._future is not None or self._decision is not None:
                return
            self._submit(obs)

    # ------------------------------------------------------------------
    # model handshake
    # ------------------------------------------------------------------
    def _adopt(self, info: dict) -> None:
        """Validate ``/info`` and keep the observation layout it describes."""
        obs = info.get("observation")
        if not isinstance(obs, dict):
            raise ValueError(
                "sidecar-serve 没有返回 observation 描述：请升级 sidecar "
                "（旧版只给 input_dim，无法得知雷达/标量布局）")
        mode = str(obs.get("mode") or "race")
        if mode not in ("race", "stunt"):
            raise ValueError(f"该模型的环境模式 mode={mode} 暂不支持（仅 race / stunt）")

        names = [str(v) for v in (obs.get("scalar_names") or [])]
        lidar_rays = int(obs.get("lidar_rays") or 0)
        frame_stack = int(obs.get("frame_stack") or 1)
        if not names or lidar_rays <= 0:
            raise ValueError("observation 描述缺少 scalar_names / lidar_rays")
        if frame_stack != 1:
            raise ValueError(f"frame_stack={frame_stack} 暂不支持（仅 1）")

        expected = lidar_rays + len(names)
        input_dim = int(info.get("input_dim") or 0)
        if expected != input_dim:
            raise ValueError(
                f"观测维数不一致：描述算出 {expected}，模型要求 {input_dim}")

        actions = {}
        for entry in info.get("actions") or []:
            if isinstance(entry, dict):
                actions[int(entry.get("index", len(actions)))] = entry
        if not actions:
            raise ValueError("sidecar-serve 没有返回动作表")

        lidar = obs.get("lidar") or {}
        norm = dict(obs.get("normalisation") or {})
        car = dict(obs.get("car") or {})
        powerups = obs.get("powerups") or {}

        self._info = info
        self._input_dim = input_dim
        self._lidar_rays = lidar_rays
        self._lidar_fov = float(obs.get("lidar_fov") or 1.5 * math.pi)
        self._lidar_range_mult = float(lidar.get("range_mult") or 6.0)
        self._lidar_range_min = float(lidar.get("range_min") or 30.0)
        self._wall_offset = float(lidar.get("wall_offset") or 0.0)
        self._checkpoints = int(obs.get("checkpoints") or 16)
        self._scalar_names = names
        self._norm = norm
        self._car_cfg = car
        self._powerup_items = [str(v) for v in (powerups.get("items") or [])]
        self._actions = actions
        self._action_count = len(actions)
        self._ready = True
        # The LIDAR saturation distance depends on the corridor width, which
        # the model file does not carry.  Derive it from this game's corridor
        # with the model's own range knobs.
        self._prepare_geometry(self.track)

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._adopt(fetch_info(self.cfg, self._timeout))

    def _fatal_from_http(self, exc: SidecarHTTPError) -> str:
        """Turn a refused ``sidecar-serve`` reply into an actionable message.

        A missing model and an old server that has no per-model routes both
        answer 404; they need different fixes, so they are told apart by the
        server's own error text (``unknown model `x`; available: a, b``).
        """
        available = exc.available()
        name = str(getattr(self.cfg, "model", "") or "")
        if exc.status == 404 and not available \
                and "unknown model" not in exc.server_error().lower():
            return ("sidecar-serve 不支持按模型路由（旧版只服务单个策略）："
                    "请清空「模型」或升级 sidecar")[:160]
        hint = f"（可用：{'、'.join(available)}）" if available else ""
        return f"sidecar 模型 {name or '?'} 不存在{hint}"[:160]

    def handshake(self, timeout: float) -> tuple[bool, str]:
        """Probe ``/info`` and validate the observation/action layout."""
        t0 = time.time()
        try:
            info = fetch_info(self.cfg, timeout)
            with self._lock:
                self._adopt(info)
        except SidecarHTTPError as exc:
            # the server answered: this is a configuration error, not a slow
            # connection, so retrying it forever would only waste the user's
            # time -- the runner ends the session with this text on screen
            self.last_error = str(exc)[:300]
            self.fatal_error = self._fatal_from_http(exc)
            self.warmup_errors += 1
            self._log_call(0.0, False, self.fatal_error)
            return False, self.last_error
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            self.last_error = str(exc)[:300]
            msg = str(exc)
            if ("维数" in msg or "模式" in msg or "observation 描述" in msg
                    or "动作表" in msg or "frame_stack" in msg):
                self.fatal_error = f"sidecar 模型不兼容: {msg[:120]}"
            self.warmup_errors += 1
            return False, self.last_error
        self.warmup_requests += 1
        name = policy_name(self.cfg, info)
        detail = (f"sidecar 策略模型 {name}（{self._input_dim} 维观测 / "
                  f"{self._action_count} 个动作，{time.time() - t0:.2f}s）")
        self.last_reason = detail
        self._log_call(0.0, True, detail)
        return True, detail

    # ------------------------------------------------------------------
    # query loop
    # ------------------------------------------------------------------
    def _submit(self, obs: Observation) -> None:
        if self.fatal_error:
            # the model was refused outright (wrong name, incompatible
            # layout): every further call would fail the same way
            return
        try:
            self._ensure_ready()
        except Exception as exc:  # noqa: BLE001 - server not up yet: retry soon
            self.last_error = str(exc)[:300]
            self._next_query_at = float(obs.time) + 0.5
            return
        try:
            vector = self._build_obs(obs)
        except Exception as exc:  # noqa: BLE001 - a bad observation is fatal
            self.fatal_error = f"sidecar 观测构造失败: {exc}"
            self.last_error = str(exc)[:300]
            return
        self._next_query_at = float(obs.time) + self._interval + self._stagger
        self._stagger = 0.0
        self._last_submit_time = float(obs.time)
        self._future = self.pool.submit(self._request, vector, float(obs.time))
        self.api_requests += 1

    def _request(self, vector: list[float], race_time: float) -> dict:
        import requests

        t0 = time.time()
        url = self._url + "/act"
        payload = {"obs": vector, "sample": self._sample}
        if self._sample:
            payload["temperature"] = self._temperature
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            if resp.status_code >= 400:
                text = resp.text[:200]
                if resp.status_code in _FATAL_STATUS:
                    self.fatal_error = f"sidecar /act HTTP {resp.status_code}: {text[:120]}"
                raise RuntimeError(f"HTTP {resp.status_code}: {text}")
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - harvested by the caller
            dt_ms = (time.time() - t0) * 1000.0
            raise RuntimeError(f"{str(exc)[:160]} ({dt_ms:.0f}ms)") from exc
        dt_ms = (time.time() - t0) * 1000.0
        self.latency_sum += dt_ms
        index = int(body.get("action", 0))
        entry = self._actions.get(index)
        if entry is None:
            raise RuntimeError(f"策略返回了未知动作下标 {index}")
        decision = {
            "index": index,
            "name": str(entry.get("name") or body.get("name") or f"action_{index}"),
            "longitudinal": str(entry.get("longitudinal") or ""),
            "lateral": str(entry.get("lateral") or ""),
            "item": entry.get("item"),
            "value": float(body.get("value", 0.0) or 0.0),
        }
        if not decision["longitudinal"] or not decision["lateral"]:
            decision = self._parse_name(decision)
        self._log_call(race_time, True,
                       f"{decision['name']} v={decision['value']:+.2f}")
        return decision

    def _parse_name(self, decision: dict) -> dict:
        """Fallback for models whose action metadata lacks the drive fields."""
        name = decision["name"].lower()
        if not decision["longitudinal"]:
            for label in ("throttle", "brake", "coast"):
                if label in name:
                    decision["longitudinal"] = label
                    break
        if not decision["lateral"]:
            for label in ("left", "right", "straight"):
                if label in name:
                    decision["lateral"] = label
                    break
        if name.startswith("use_"):
            # Every item action in the simulator is throttle + straight.
            decision["longitudinal"] = decision["longitudinal"] or "throttle"
            decision["lateral"] = decision["lateral"] or "straight"
            if decision["item"] is None:
                decision["item"] = name[4:]
        return decision

    def _harvest(self) -> None:
        if self._future is None or not self._future.done():
            return
        try:
            self._decision = self._future.result()
            self.last_action_name = self._decision["name"]
            self.last_reason = (
                f"sidecar {self._decision['name']} "
                f"(v={self._decision['value']:+.2f})")
            # An item action fires the item exactly once, on the tick its
            # decision is adopted -- exactly like one simulator step.
            item = self._decision.get("item")
            if item:
                key = str(item).upper()
                if key in _ITEM_KEYS:
                    self._pending_item = key
        except Exception as exc:  # noqa: BLE001 - keep the last good policy
            self.api_errors += 1
            self.last_error = str(exc)[:300]
            self.last_reason = f"error: {exc}"[:80]
            self._log_call(self._last_submit_time, False, str(exc)[:120])
        finally:
            self._future = None

    def _log_call(self, t: float, ok: bool, text: str) -> None:
        """Append a console-visible call log entry (``t`` is race time)."""
        self.call_log.append({
            "t": round(max(0.0, float(t)), 2),
            "ok": bool(ok),
            "text": str(text)[:160],
        })
        if len(self.call_log) > 8:
            self.call_log.pop(0)

    # ------------------------------------------------------------------
    def act(self, obs: Observation) -> Action:
        now = float(obs.time)
        with self._lock:
            if self._future is None and now >= self._next_query_at:
                self._submit(obs)
            self._harvest()

        if self._decision is None:
            return Action(throttle=0.0, brake=0.45, steer=0.0,
                          reason="等待 sidecar 策略模型决策")

        d = self._decision
        self._prev_drive = [
            _LONG_CODE.get(d["longitudinal"], 2.0),
            _LAT_CODE.get(d["lateral"], 2.0),
        ]
        longitudinal = d["longitudinal"]
        item = self._pending_item
        self._pending_item = None
        if item and not (obs.items.get(item) or {}).get("available", False):
            # The policy can only fire an item when its own observation said
            # it was ready; if the item went on cooldown in the meantime,
            # drop it instead of tripping the anti-cheat's cooldown check.
            item = None
        return Action(
            throttle=1.0 if longitudinal == "throttle" else 0.0,
            brake=1.0 if longitudinal == "brake" else 0.0,
            steer=self._drive_steer(d["lateral"], obs),
            use_item=item,
            reason=self.last_reason,
        )

    def _drive_steer(self, lateral: str, obs: Observation) -> float:
        """Turn the policy's left/right/straight into the game's steer input.

        The simulator steers a kinematic bicycle towards +/-``max_steer`` at
        ``steer_rate`` rad/s, so its yaw rate is ``v / wheelbase * tan(steer)``.
        The game instead multiplies a raw ``steer`` by ``steer_rate *
        speed_grip * high_speed_damp``.  Feeding the policy's command straight
        through would oversteer at walking pace (the game's grip factor is
        near-quadratic there) and understeer flat out, which is exactly how a
        policy that is clean in its own simulator ends up sawing at the wall.
        So keep the simulator's steer *angle* here and ask for the game input
        that produces the same yaw rate.  Where the game physically cannot
        (high speed), the result saturates at 1.0 -- the least it can do.
        """
        car = self._car_cfg
        max_steer = abs(float(car.get("max_steer") or 0.2))
        rate = abs(float(car.get("steer_rate") or _GAME_STEER_RATE))
        target = 0.0
        if lateral == "left":
            target = max_steer
        elif lateral == "right":
            target = -max_steer
        dt = clamp(float(getattr(obs, "dt", 0.0) or 0.0), 1e-4, 0.25)
        self._steer += clamp(target - self._steer, -rate * dt, rate * dt)

        speed = max(0.0, float(obs.speed))
        wheelbase = abs(float(car.get("wheelbase") or 2.6))
        if wheelbase <= 1e-6:
            return 0.0
        omega = speed / wheelbase * math.tan(self._steer)
        grip = min(1.0, _GAME_GRIP_BASE + speed / _GAME_GRIP_SPEED)
        damp = 1.0 / (1.0 + speed * _GAME_DAMP)
        unit = _GAME_STEER_RATE * grip * damp
        if unit <= 1e-6:
            return 0.0
        return clamp(omega / unit, -1.0, 1.0)

    # ------------------------------------------------------------------
    # observation construction
    # ------------------------------------------------------------------
    def _prepare_geometry(self, track) -> None:
        """Cache boundary segments where this game actually stops the car."""
        if track is None:
            return
        if self._geom is not None and self._geom[0] is track:
            return
        xz = getattr(track, "center_xz", None)
        if xz is None:
            xz = np.asarray(track.center)[:, [0, 2]]
        normal = np.asarray(track.normal, dtype=np.float64)
        half = float(track.half_width)
        corridor = max(1.0, half - _CAR_HALF_WIDTH)
        # A tiny outward margin keeps the origin *inside* the boundary even
        # when the game has clamped the car centre exactly onto the limit;
        # sitting exactly on it makes the near wall invisible to a t > 0 ray
        # cast, and a policy that cannot see the wall it is scraping against
        # keeps the throttle pinned.
        wall = corridor + 0.02
        left = xz + normal * wall
        right = xz - normal * wall
        starts = np.concatenate([left, right])
        ends = np.concatenate([np.roll(left, -1, axis=0),
                               np.roll(right, -1, axis=0)])
        self._corridor = corridor
        self._corridor_scale = (self._wall_offset / corridor
                                if self._wall_offset > corridor else 1.0)
        self._lidar_max = max(corridor * self._lidar_range_mult,
                              self._lidar_range_min)
        self._geom = (track, starts, ends)

    def _lidar(self, origin, heading: float) -> list[float]:
        """Ray-cast the same fan the simulator uses (``1 - d / max``)."""
        self._prepare_geometry(self.track)
        if self._geom is None:
            return [0.0] * self._lidar_rays
        _, starts, ends = self._geom
        rays = self._lidar_rays
        if rays == 1:
            frac = np.zeros(1)
        else:
            frac = np.linspace(-0.5, 0.5, rays)
        angles = heading + frac * self._lidar_fov
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        edges = ends - starts
        origin = np.asarray(origin, dtype=np.float64)
        # (rays, segments) segments-vs-ray intersection, vectorised.
        rel = starts[None, :, :] - origin[None, None, :]
        denom = (dirs[:, None, 0] * edges[None, :, 1]
                 - dirs[:, None, 1] * edges[None, :, 0])
        safe = np.abs(denom) > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (rel[..., 0] * edges[None, :, 1]
                 - rel[..., 1] * edges[None, :, 0]) / denom
            # u = -cross(a - o, d) / cross(d, e); the sign of the latter
            # cross product is easy to flip, and a flipped u silently turns
            # real hits into misses (and segment *extensions* into hits).
            u = (rel[..., 1] * dirs[:, None, 0]
                 - rel[..., 0] * dirs[:, None, 1]) / denom
        valid = safe & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)
        best = np.minimum(t.min(axis=1), self._lidar_max)
        best = np.clip(best, 0.0, self._lidar_max)
        return (1.0 - best / self._lidar_max).tolist()

    def _scalar_values(self, obs: Observation) -> dict:
        track = self.track
        n = max(1, int(getattr(track, "n", len(track.center))))
        i = int(obs.track_index) % n
        half_width = max(1e-6, float(obs.half_width))
        x, z = float(obs.pos[0]), float(obs.pos[1])
        heading = float(obs.heading)

        norm = self._norm
        speed_ref = float(norm.get("speed_ref") or 0.0)
        if speed_ref <= 0.0:
            speed_ref = float(self._car_cfg.get("max_speed") or 42.0)

        tangent = np.asarray(track.tangent[i], dtype=np.float64)
        track_heading = math.atan2(float(tangent[1]), float(tangent[0]))
        # Pacing is 30 Hz, so this is a two-step average of the game's yaw
        # rate; the simulator reads its own instantaneous omega.
        yaw_rate = 0.0
        if self._prev_heading is not None and self._prev_time is not None:
            dt = float(obs.time) - self._prev_time
            if dt > 1e-6:
                yaw_rate = wrap_angle(heading - self._prev_heading) / dt
        self._prev_heading = heading
        self._prev_time = float(obs.time)

        # Checkpoint distance exactly as the env tracks it: evenly spaced
        # checkpoints measured from the lap start, next one ahead.
        total = max(1e-6, float(track.length))
        laps = max(1, int(getattr(obs, "laps_total", 1) or 1))
        progress = clamp(float(obs.race_progress), 0.0, 1.0) * total * laps
        lap_s = progress % total
        count = max(1, self._checkpoints)
        step = 1.0 / count
        x_frac = lap_s / total
        index = int(math.floor(x_frac / step - 0.5)) + 1
        index = max(0, min(count, index))
        if index < count:
            cp_s = total * (index + 0.5) * step
        else:
            cp_s = total * (0.5 * step) + total
        cp_dist = ((cp_s - lap_s + total * 0.5) % total) - total * 0.5
        cp_dist /= total

        # Elevation: the game stores it on the centre line only.
        heights = np.asarray(track.center, dtype=np.float64)[:, 1]
        seg = np.asarray(track.seg_len, dtype=np.float64)
        h = float(heights[i])
        ds = float(seg[i - 1] + seg[i]) if n > 2 else 1.0
        grade = float(heights[(i + 1) % n] - heights[i - 1]) / max(ds, 1e-6)

        # Nearest obstacle ahead (sidecar measures to the obstacle centre).
        best = self._lidar_max
        fx, fz = math.cos(heading), math.sin(heading)
        for obstacle in getattr(track, "obstacles", []) or []:
            dx, dz = float(obstacle.x) - x, float(obstacle.z) - z
            dist = math.hypot(dx, dz)
            if dist > best:
                continue
            radius = float(getattr(obstacle, "radius", 0.0))
            if dx * fx + dz * fz < -2.0 * radius:
                continue
            best = dist
        obstacle_dist = clamp(1.0 - best / max(self._lidar_max, 1e-3), 0.0, 1.0)

        boosted = float(getattr(obs, "effect", "") == "boost")
        shielded = float(getattr(obs, "effect", "") == "shield")
        slowed = float(getattr(obs, "effect", "") == "slow")
        items = getattr(obs, "items", {}) or {}

        values = {
            "speed": clamp(float(obs.speed) / speed_ref, 0.0, 1.0),
            "lateral": clamp(float(obs.lateral) * self._corridor_scale
                             / half_width,
                             -float(norm.get("lateral_clamp", 2.0)),
                             float(norm.get("lateral_clamp", 2.0))),
            "heading": clamp(wrap_angle(heading - track_heading)
                             / float(norm.get("heading_ref", math.pi)),
                             -float(norm.get("heading_clamp", 1.0)),
                             float(norm.get("heading_clamp", 1.0))),
            "curvature": math.tanh(float(track.curvature[i])
                                   * float(norm.get("curvature_gain", 8.0))),
            "prev_longitudinal": self._prev_drive[0] / 2.0,
            "prev_lateral": self._prev_drive[1] / 2.0,
            "checkpoint_dist": clamp(cp_dist,
                                     -float(norm.get("checkpoint_clamp", 1.0)),
                                     float(norm.get("checkpoint_clamp", 1.0))),
            "height": clamp(h / float(norm.get("height_ref", 20.0)),
                            -float(norm.get("height_clamp", 2.0)),
                            float(norm.get("height_clamp", 2.0))),
            "grade": clamp(grade, -float(norm.get("actor_clamp", 1.0)),
                           float(norm.get("actor_clamp", 1.0))),
            "airborne": 0.0,          # the game never leaves the ground
            "wetness": 0.0,
            "tyre_wear": 0.0,
            "fuel": 1.0,
            "obstacle_dist": obstacle_dist,
            "grip": 1.0,
            "slip": 0.0,              # arcade model: no body sideslip
            "yaw_rate": clamp(yaw_rate / float(norm.get("yaw_rate_ref", 3.0)),
                              -float(norm.get("yaw_rate_clamp", 1.0)),
                              float(norm.get("yaw_rate_clamp", 1.0))),
            "shielded": shielded,
            "boosted": boosted,
            "slowed": slowed,
        }
        for name in self._powerup_items:
            key = name.upper()
            entry = items.get(key) if key in _ITEM_KEYS else None
            ready = bool((entry or {}).get("available", False))
            if name.lower() == "nitro":
                active = bool(boosted)
            elif name.lower() == "shield":
                active = bool(shielded)
            elif name.lower() in ("emp", "oil"):
                active = bool(slowed)
            else:
                active = False
            values[f"{name}_ready"] = 1.0 if ready else 0.0
            values[f"{name}_active"] = 1.0 if active else 0.0
        return values

    def _build_obs(self, obs: Observation) -> list[float]:
        vector = self._lidar(obs.pos, float(obs.heading))
        values = self._scalar_values(obs)
        vector.extend(float(values.get(name, 0.0)) for name in self._scalar_names)
        if len(vector) != self._input_dim:
            raise ValueError(
                f"观测长度 {len(vector)} != 模型要求的 {self._input_dim}")
        return vector

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        ok = max(0, self.api_requests - self.api_errors)
        return {
            "driver": self.kind,
            "model": self.cfg.model,
            "provider": self.cfg.provider,
            "endpoint": self._url,
            "api_requests": self.api_requests,
            "api_errors": self.api_errors,
            "avg_latency_ms": round(self.latency_sum / max(1, ok), 1),
            "success_rate": round(ok / max(1, self.api_requests), 3),
            "last_reason": self.last_reason,
            "last_error": self.last_error,
        }

    def debug_info(self) -> dict:
        return {
            "provider": self.cfg.provider,
            "last_reason": self.last_reason,
            "last_response": self.last_action_name,
            "last_error": self.last_error,
            "call_log": list(self.call_log[-4:]),
            "sidecar": {
                "endpoint": self._url,
                "input_dim": self._input_dim,
                "actions": self._action_count,
                "model": policy_name(self.cfg, self._info or {}),
            },
        }
