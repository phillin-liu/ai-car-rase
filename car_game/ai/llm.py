"""Vision-only LLM backed drivers.

The model owns the driving strategy, but it perceives the world the same way
a driver does: through an **image**.  Every query renders the current scene to
a **first-person 3D frame** (the same camera the human sees) and sends that raw
picture to the model -- never a textual description of the track, the
obstacles or the rival's position.  A companion detector reads the same
frame and turns what it finds on the pixels back into geometry, which is what
the low-level executor uses to avoid obstacles.

A transient API error does *not* discard the policy: the car keeps executing
the last good decision until a new one arrives.  Only a driver that has never
received a single decision coasts and waits.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..config import ModelConfig
from ..core_types import Action, Observation
from ..mathutil import clamp, wrap_angle
from .base import BaseDriver


SYSTEM_PROMPT = """你是一名赛车策略 AI，控制一辆赛车在封闭赛道上与另一辆车竞速。
你只能通过**图像**感知世界：每轮你会收到一张**第一人称 3D 驾驶视角**的画面
（从你车内驾驶员的位置、朝车头方向看出去的 3D 世界：天空、地面、向远处延伸的赛道、
路面的障碍物（橙色锥桶/黄色油桶/棕色木箱）以及对手车辆（青色）），
以及一个只含你本车信息的仪表盘。

请**完全依靠图像**判断：赛道走向与弯道、路面上的障碍物、赛道边界、
以及对手车辆（青色）在你前方还是后方、偏左还是偏右。程序不会再给你任何文字形式的
场景描述，不要索要或依赖文字遥测。

## 看图方法（第一人称透视）
* 画面**底部中央的正下方**就是你车头正前方的路面；画面越往上越远，地平线处最远。
* 同一个物体：越靠画面底部越近，在画面上越大也越近。远处的障碍物只是路面上的
  一小块色斑，不要因为它小就忽略。
* 判断障碍物在你的左边还是右边，要看它**底座（与地面接触的那条边）**在画面水平
  方向的位置，而不是它的顶部：底座在画面中央左边 = 在你的左侧，在右边 = 在你的
  右侧；偏离画面中央越远，横向偏移越大。
* 底座已经贴到画面最左/最右边缘的障碍物，就压在赛道左/右边界上。
* 赛道两侧的护栏/边墙同样要躲，别把车开到画面两侧。

## 避障（每一帧先做，精度优先）
每一轮先扫一遍前方路面，逐一确认看到的每个锥桶/油桶/木箱：
1. **多远**——越靠画面底部越近（通常是画面下半部里底座最低的那几个）；
2. **在左还是右**——看底座偏离画面中央的方向和幅度；
3. **哪一侧空**——它离赛道左边界近，右边就是空当；反之亦然。
然后把 lateral_bias 明确指向你选定的那条空当，注意：
* **只有一个障碍物**：从空间更大的一侧绕过去（它偏左就从右侧过，偏右就从左侧过），
  绝不从它贴着赛道边缘的那一侧硬挤。
* **障碍物正在正前方中央**：果断选一侧，lateral_bias 至少偏到 ±0.4，
  不要犹豫着直冲过去。
* **连续或左右交错的障碍物**：提前在第一个障碍物之前就开始横移，为下一个留好位置，
  而不是到跟前才打方向。
* **横向留足余量**：lateral_bias 要指向空当的**中间**，不要刚好擦着障碍物过去；
  宁可多让 0.2，也不要撞上——撞一次的损失远大于多绕的那点距离。
* 这一帧看到了障碍物，就必须在这一帧的决策里避开。若这一帧撞上了，说明上一帧没
  避开，必须在下一次决策里立刻纠正，而不是继续直行。
* 绝不允许撞上：锥桶/油桶/木箱、赛道护栏、对手车辆。

**速度是这局比赛的主战场。** 仪表盘上的 speed_kmh / top_speed_kmh / speed_percent
就是你的实时车速、本车极速和已用掉的比例。油门只有在你主动给的时候才有——
decision_interval 秒内你只能改一次决策，所以：

* **直道和缓弯要把 throttle_scale 顶到 1.0~1.2**，让 speed_percent 一直贴着 100%；
  在直道上留有余量，等于把整条直道送给对手。
* 只有在图像里出现**明显急弯、障碍物、或者赛道边界收窄**时才收油或刹车，
  而且收油要短、要果断，过了这一段立刻顶回去。
* 不要为了"稳妥"长期压低 throttle_scale——巡航输掉比赛，比冒一次风险更糟。

只输出一个 JSON 决策，不要输出多余文本，字段：
{
  "lateral_bias": -0.75 到 0.75 之间的小数,  // 目标横向位置(-向赛道左侧, +向右侧, 0=居中)
  "throttle_scale": 0.5 到 1.2 之间的小数,   // 目标车速 = 本车极速 × 该系数。1.0 = 用尽极速，
                                            // 再往上只是维持全油门；这是你唯一的速度旋钮
  "aggression": 0 到 1 之间的小数,           // 进攻倾向: 超车 / 施压 / 干扰对手。
                                            // 同时会小幅推高目标车速(0.5 为中性, 越敢打越快)
  "use_item": null 或道具名,                 // 从仪表盘 items_ready 中选择, 可选 NITRO/SHIELD/EMP/OIL/RECOVER
  "reason": "简短中文理由：先写前方最近障碍物在左/右/无、你从哪一侧绕，再写油门"
}

你的目标，按优先级：
1. 避障优先：先扫描并绕开路面障碍物与赛道边界，lateral_bias 要明确指向障碍物
   旁边的空当。宁可偏离走线、宁可减速，也不要撞上锥桶/油桶/木箱——
   撞一次损失的速度，远超过你省下的那点刹车。
2. 全速推进：把每一段路都跑到本车极限。对手在直道上会一直顶着极速，
   你只要在直道上松了油门就会被拉开，所以除进弯绕障外**一律不要保守**。
3. 盯住对手并压制：图像中若能看到对手在前方，全力追赶；若对手在后方，保持高速并守住走线。
   任何情况下都不要巡航；但追赶也不能以撞障碍物、撞墙为代价。
4. 合理使用道具（NITRO 在直道拉开距离最有效）。
"""


HANDSHAKE_SYSTEM_PROMPT = """你是一名赛车 AI，即将参加一场比赛。程序正在确认你已经就位。
请用一句话回复确认（例如「就绪」）。不要输出 JSON，不要做出驾驶决策，不要索要画面。"""

HANDSHAKE_TEXT = "比赛即将开始，请回复确认你已就绪。"


# A provider error that retrying cannot fix: the endpoint answered, and it will
# answer the same way every time.  404 is the common one (the configured model
# does not exist on that provider); 400/422 are the same failure wearing a
# different number, and 401/403 mean the key is wrong.  Waiting on any of them
# only wastes the user's time -- the session is ended instead.  Deliberately
# *not* included: 408/429/5xx, which are transient and worth retrying.
FATAL_STATUS = frozenset({400, 401, 403, 404, 422})
_FATAL_TEXT = ("model not found", "does not exist", "not_found",
               "no such model", "模型不存在")


def fatal_reason(exc: BaseException) -> str:
    """Why ``exc`` can never succeed on retry, or ``""`` if it might.

    Prefers the HTTP status the SDK attached to the exception; falls back to
    the message, because some OpenAI-compatible gateways answer a 404 with a
    200-shaped body or lose the status on the way through.
    """
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None
    if code in FATAL_STATUS:
        return f"HTTP {code}"
    text = str(exc).lower()
    for needle in _FATAL_TEXT:
        if needle in text:
            return "模型不存在"
    return ""


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    # strip markdown fences
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


class LLMDriver(BaseDriver):
    kind = "llm"

    def __init__(self, cfg: ModelConfig, track, name: str, difficulty: float = 1.0,
                 stagger: float = 0.0):
        super().__init__(cfg, track, name)
        self.pool = ThreadPoolExecutor(max_workers=1,
                                       thread_name_prefix=f"llm-{name}")
        self._lock = threading.Lock()
        self._future = None
        self._decision = None
        self._analysis = None
        self._next_query_at = 0.0
        self._interval = max(0.3, float(cfg.decision_interval))
        # Delay the *second* query by this fraction of an interval and every
        # later one inherits the offset.  The first query is deliberately not
        # delayed: both cars must get their first decision on the same tick or
        # the start is unfair.  After that, staggering the two sides' frame
        # rendering halves the peak load -- both renders are pure-Python and
        # hold the GIL, so landing them together is what makes the 3D window
        # hitch once per decision.
        self._stagger = max(0.0, min(0.9, float(stagger))) * self._interval
        self._client = None
        self._no_temperature = False
        self._history = []
        self._prev_err = 0.0
        self.api_requests = 0
        self.api_errors = 0
        self.latency_sum = 0.0
        # Warm-up ("are you ready?") calls are counted apart from race calls.
        # The first call to a provider is the cold one -- tens of seconds of
        # DNS/TLS/model load -- and folding that into api_requests/latency_sum
        # would report it as a slow, error-prone *race* and skew the analysis.
        self.warmup_requests = 0
        self.warmup_errors = 0
        self.last_reason = ""
        # --- vision state --------------------------------------------
        self._local_model = None
        self.last_image = None
        self.last_detections = None
        # --- debug / proof-of-call -----------------------------------
        self.last_prompt = ""
        self.last_response = ""
        self.last_error = ""
        # non-empty when the provider said something retrying cannot fix (404
        # model missing, bad key): the runner ends the session on this
        self.fatal_error = ""
        self.call_log = []          # recent (t, ok, summary) entries

    # ------------------------------------------------------------------
    def reset(self, track) -> None:
        super().reset(track)
        self._decision = None
        self._analysis = None
        self._next_query_at = 0.0
        self._prev_err = 0.0
        self._history.clear()
        self.warmup_requests = 0
        self.warmup_errors = 0
        self.fatal_error = ""
        self.last_image = None
        self.last_detections = None

    def close(self) -> None:
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover
            self.pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    def _submit(self, obs: Observation) -> None:
        """Fire one query for this observation (lock held by the caller).

        The first query of a match is deliberately **not** staggered: both cars
        must get their opening decision on the same tick or the start is
        unfair.  Every later one inherits the offset, so the two sides' pure
        Python renders never land together and stall the 3D window.
        """
        self._next_query_at = float(obs.time) + self._interval + self._stagger
        self._stagger = 0.0             # one-off: the offset is now built in
        self._future = self.pool.submit(self._query, self._capture(obs))
        self.api_requests += 1

    def _harvest(self) -> None:
        """Adopt a finished query's decision, if there is one (lock held)."""
        if self._future is None or not self._future.done():
            return
        try:
            self._decision, self._analysis = self._future.result()
            self.last_detections = self._analysis
            self.last_reason = str(self._decision.pop("reason", ""))[:80]
        except Exception as exc:  # noqa: BLE001
            self.api_errors += 1
            # keep the last good policy: a transient error must not blank the
            # decision and stall the car mid-race
            self.last_reason = f"error: {exc}"[:80]
        finally:
            self._future = None

    @property
    def has_decision(self) -> bool:
        """True once this side holds a decision it can actually drive on."""
        with self._lock:
            self._harvest()
            return self._decision is not None

    def prime(self, obs: Observation) -> None:
        """Take the opening look *before* the race clock starts.

        Rendering the first frame and waiting for the model to answer it costs
        seconds, and every one of them used to tick off the race clock while
        the car sat on the line braking: the other car was already 100 m down
        the road by the first decision.  Starting the query while the field is
        still held at the line moves that whole cost before the start, so both
        cars launch on the green with a real decision in hand.

        Idempotent, and a no-op once anything has been decided: calling it
        again cannot cost a second request.
        """
        with self._lock:
            if self._future is not None or self._decision is not None:
                return
            self._submit(obs)

    def act(self, obs: Observation) -> Action:
        now = obs.time
        with self._lock:
            if self._future is None and now >= self._next_query_at:
                self._submit(obs)
            self._harvest()

        # NO local assistance: without a model decision the car coasts to
        # a stop and waits.  It NEVER falls back to the rule strategy.
        if self._decision is None:
            return Action(throttle=0.0, brake=0.45, steer=0.0, use_item=None,
                          reason="等待视觉模型决策")

        d = self._decision
        self._history.append({
            "t": round(now, 1),
            "lateral_bias": d.get("lateral_bias"),
            "throttle_scale": d.get("throttle_scale"),
            "use_item": d.get("use_item"),
        })
        if len(self._history) > 8:
            self._history.pop(0)
        return self._direct_action(obs, d, self._analysis)

    # ------------------------------------------------------------------
    def _capture(self, obs: Observation) -> dict:
        """Immutable per-tick snapshot handed to the worker thread.

        It carries geometry so the worker thread can *render* the frame and
        hand it to the local detector; none of the scene fields are ever put
        in the request sent to a remote model (see :meth:`_dashboard`).
        """
        return {
            "time": round(float(obs.time), 3),
            "pos": (float(obs.pos[0]), float(obs.pos[1])),
            "heading": float(obs.heading),
            "speed": float(obs.speed),
            "max_speed": float(obs.max_speed),
            "lateral": float(obs.lateral),
            "half_width": float(obs.half_width),
            "lap": int(obs.lap),
            "laps_total": int(obs.laps_total),
            "items": {k: {"available": bool(v.get("available"))}
                      for k, v in (obs.items or {}).items()},
            "opp_pos": (float(obs.opp_pos[0]), float(obs.opp_pos[1])),
            "opp_speed": float(obs.opp_speed),
            "opp_lateral": float(obs.opp_lateral),
            "opp_ahead": bool(obs.opp_ahead),
            "opp_distance": float(obs.opp_distance),
            "rival_gap_m": float(obs.rival_gap_m),
            "hazards": list(obs.hazards or []),
            "curvatures": list(obs.lookahead_curvatures or []),
        }

    def _dashboard(self, ctx: dict) -> dict:
        """The ONLY text a remote model gets: this car's own instruments.

        Deliberately still *only this car* -- no track geometry, no obstacle
        coordinates, no rival, no gap.  ``top_speed_kmh``/``speed_percent`` are
        instruments, not scene data: without them ``throttle_scale`` is a
        number with no unit and a cautious model will hedge it low and hand
        every straight to the other car.
        """
        ready = [k for k, v in ctx["items"].items() if v.get("available")]
        top = max(1e-6, float(ctx["max_speed"]))
        return {
            "speed_kmh": round(ctx["speed"] * 3.6, 1),
            "top_speed_kmh": round(top * 3.6, 1),
            "speed_percent": round(100.0 * float(ctx["speed"]) / top),
            "lap": ctx["lap"],
            "laps_total": ctx["laps_total"],
            "items_ready": ready,
            "objective": "依据图像避障并压制对手，尽快完赛",
        }

    # ------------------------------------------------------------------
    def _vision(self, ctx: dict):
        """Render this driver's frame and detect on it (worker thread)."""
        from ..vision.local_model import ensure_local_vision_model
        from ..vision.render import RIVAL_CAR, CarView
        if self._local_model is None:
            self._local_model = ensure_local_vision_model(self.cfg)
        if self.track is None:
            from PIL import Image
            size = int(getattr(self.cfg, "vision_image_size", 256) or 256)
            img = Image.new("RGB", (size, size), (0, 0, 0))
            from ..vision.detector import SceneAnalysis
            return img, SceneAnalysis()
        views = [
            CarView(ctx["pos"][0], ctx["pos"][1], ctx["heading"],
                    (255, 0, 255), is_self=True, name=self.name),
            CarView(ctx["opp_pos"][0], ctx["opp_pos"][1], ctx["heading"],
                    RIVAL_CAR, is_self=False, name="rival"),
        ]
        return self._local_model.analyze(self.track, views, ctx["hazards"], 0)

    # ------------------------------------------------------------------
    @staticmethod
    def _clear_lane(hazards, base: float, half: float) -> float:
        """The offset, in metres from the centre line, with the most room.

        ``hazards`` is ``[(forward_m, lateral_from_centreline_m, radius_m)]``
        already shifted into the centre-line frame, and ``base`` is the line
        the model asked for.  Every candidate in a fan spanning the road is
        scored on real clearance against *all* the hazards at once -- and
        clearance outweighs loyalty to ``base`` by roughly an order of
        magnitude -- so two obstacles close together cannot push the answer
        back into each other, and the widest genuinely free gap wins.
        """
        step = 0.05 * half
        limit = 0.78 * half
        best, best_score = base, None
        for k in range(-16, 17):
            cand = clamp(base + k * step, -limit, limit)
            # holding the model's line is worth something -- but never more
            # than driving into an obstacle costs
            score = -abs(cand - base) / half * 1.1
            edge = (half - 1.8) - abs(cand)          # keep off the barrier
            if edge < 0.0:
                score -= abs(edge) * 4.0
            for fwd, lat, rad in hazards:
                urgency = 1.0 / (1.0 + max(fwd, 0.0) / 9.0)
                # the margin never shrinks with distance: the line starts
                # easing away well before the obstacle instead of swerving at
                # the last moment, which is what clips a cone
                gap = abs(cand - lat) - (rad + 1.9)
                if gap < 0.0:
                    score -= abs(gap) * 9.0 * urgency
                else:
                    score += min(gap, 2.0) * 0.18 * urgency
            if best_score is None or score > best_score:
                best_score, best = score, cand
        return best

    # ------------------------------------------------------------------
    def _direct_action(self, obs: Observation, d: dict, analysis=None) -> Action:
        """Convert the model policy into controls.

        Steering follows the model's target line; obstacle avoidance comes
        from the local detector's readings of the rendered frame (never from
        the track's ground-truth obstacle list).
        """
        half = max(obs.half_width, 1e-6)
        bias = clamp(float(d.get("lateral_bias", 0.0)), -0.75, 0.75)
        tscale = clamp(float(d.get("throttle_scale", 1.0)), 0.3, 1.2)
        # ``aggression`` used to be parsed and then dropped on the floor -- the
        # model was asked for an appetite for a fight and nothing consumed it.
        # It now biases the target speed by +/-8%, neutral at the 0.5 default,
        # so the model's own answer to "how hard am I pushing" reaches the car.
        aggr = clamp(float(d.get("aggression", 0.5)), 0.0, 1.0)
        pace = tscale * (0.92 + 0.16 * aggr)

        # aim at a point on the model's line one lookahead ahead
        dists = list(obs.lookahead_distances or [8.0])
        pts = list(obs.lookahead_points or [(obs.pos[0], obs.pos[1])])
        look = clamp(9.0 + obs.speed * 0.6, 10.0, 40.0)
        k = min(range(len(dists)), key=lambda i: abs(dists[i] - look))
        px, pz = pts[k] if k < len(pts) else (obs.pos[0], obs.pos[1])

        heading = obs.heading
        left = (-math.sin(heading), math.cos(heading))

        # --- vision-based obstacle avoidance -------------------------
        # A reflex layered on the model's line.  The model chooses the lane;
        # the detector's reading of this frame then scores a fan of candidate
        # offsets around that lane and takes the one with the most real
        # clearance.  Scoring the whole fan at once, rather than nudging away
        # from one obstacle at a time, is what makes two obstacles close
        # together work: a second nudge used to push the line straight back
        # into the first one.
        #
        # The detections are in *this car's* frame (metres left of the car)
        # while an offset is measured from the centre line, so each one is
        # shifted by the car's own lateral position before it is compared --
        # without that the reflex aims at the wrong place whenever the car is
        # off the centre of the road, which is exactly when it has to work.
        base = clamp(bias * half, -0.78 * half, 0.78 * half)
        # look further ahead the faster we go, so avoidance starts early
        horizon = min(70.0, 24.0 + obs.speed * 1.5)
        # A detection is held from the frame the last decision was taken on --
        # a whole decision interval ago, which at racing speed is tens of
        # metres of road.  Its *car-relative* readings are stale the instant
        # they arrive, and adding a stale "metres left of the car" to the car's
        # *current* lateral makes every obstacle appear to travel sideways with
        # the car: the reflex then never commits, oscillating between lanes
        # while the real cone goes straight into the nose.  The detector maps
        # its pixels back to absolute world coordinates precisely so this can
        # be re-projected against where the car is *now*.
        cos_h, sin_h = math.cos(obs.heading), math.sin(obs.heading)
        hazards = []
        for det in (getattr(analysis, "obstacles", None) or []):
            dx = float(det.world[0]) - obs.pos[0]
            dz = float(det.world[1]) - obs.pos[1]
            fwd = dx * cos_h + dz * sin_h
            if fwd < -2.5 or fwd > horizon:
                continue
            lat = obs.lateral + (-dx * sin_h + dz * cos_h)
            hazards.append((fwd, lat, max(0.2, float(det.radius))))
        off = self._clear_lane(hazards, base, half) if hazards else base
        # Brake for anything sitting on the line the model asked for, even if
        # the reflex chose to steer around it: up close there is not enough
        # road left to sidestep, and a cone taken at 30 m/s costs far more
        # than the throttle it saves.  A hazard that is clear of both the
        # model's line and the chosen one is simply overtaken.
        block_brake = 0.0
        for fwd, lat, rad in hazards:
            if fwd >= 22.0 or obs.speed <= 7.0:
                continue
            if min(abs(lat - base), abs(lat - off)) < rad + 1.6:
                block_brake = max(block_brake, 0.5 * (1.0 - fwd / 22.0))

        tx = px + left[0] * off
        tz = pz + left[1] * off
        desired = math.atan2(tz - obs.pos[1], tx - obs.pos[0])
        err = wrap_angle(desired - heading)
        deriv = wrap_angle(err - self._prev_err)
        self._prev_err = err
        steer = clamp(err * 2.2 + deriv * 0.6, -1.0, 1.0)

        v_target = obs.max_speed * pace
        err_v = v_target - obs.speed
        if err_v >= 0:
            throttle = clamp(err_v / 6.0, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-err_v / 8.0, 0.0, 1.0)

        brake = max(brake, block_brake)
        if block_brake > 0:
            throttle = min(throttle, 0.45)

        # items: only what the model asked for, never a local substitute
        use_item = d.get("use_item")
        if use_item is not None:
            avail = {key: v for key, v in (obs.items or {}).items()
                     if v.get("available")}
            if use_item not in avail:
                use_item = None
        return Action(throttle=throttle, brake=brake, steer=steer,
                      use_item=use_item, reason=self.last_reason or "llm")

    # ------------------------------------------------------------------
    def _query(self, ctx: dict) -> tuple:
        t0 = time.time()
        try:
            img, analysis = self._vision(ctx)
            self.last_image = img
            self.last_detections = analysis
            dashboard = self._dashboard(ctx)
            self.last_prompt = (f"vision frame {img.width}x{img.height}"
                                f" + dashboard (无场景文字描述)")
            raw = self._call_provider(ctx, img, dashboard)
            self.last_response = (raw or "").strip()[:800]
            self.last_error = ""
            self.fatal_error = ""
            data = _extract_json(raw)
            data.setdefault("lateral_bias", 0.0)
            data.setdefault("throttle_scale", 1.0)
            data.setdefault("aggression", 0.5)
            data.setdefault("use_item", None)
            self._log_call(ctx, True, self.last_response)
            return data, analysis
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)[:300]
            # a 404 mid-race means the car will coast to the end: say so, so
            # the runner can end the match instead of storing a fake result
            self.fatal_error = fatal_reason(exc)
            self._log_call(ctx, False, self.last_error)
            raise
        finally:
            self.latency_sum += time.time() - t0

    def _log_call(self, ctx, ok: bool, text: str) -> None:
        self.call_log.append({
            "t": round(float(ctx.get("time", 0.0)), 1),
            "ok": ok,
            "text": text[:160],
        })
        if len(self.call_log) > 12:
            self.call_log.pop(0)

    def debug_info(self) -> dict:
        """Rich per-call details for the in-game debug overlay."""
        det = self.last_detections
        return {
            "provider": self.cfg.provider,
            "last_reason": self.last_reason,
            "last_response": self.last_response,
            "last_error": self.last_error,
            "call_log": list(self.call_log[-4:]),
            "vision": {
                "detector": getattr(det, "detector", None) if det else None,
                "obstacles": len(det.obstacles) if det else 0,
                "rival": bool(det.rival) if det else False,
                "image": (f"{self.last_image.width}x{self.last_image.height}"
                          if self.last_image is not None else None),
            },
        }

    # ------------------------------------------------------------------
    def handshake(self, timeout: float) -> tuple[bool, str]:
        """Ask the model to confirm it is ready, once, before the race starts.

        Blocking and **text-only**: no frame is rendered and nothing is put in
        ``_future``/``_decision``/``_analysis``, so this can never disturb the
        driving loop that starts afterwards.  Returns ``(replied, detail)``;
        ``detail`` carries the model's answer or the failure reason so the
        console can show which side is late and why.

        The call is billed to the warm-up counters, not to the race ones: it
        opens the connection and loads the model so the *first race decision*
        is fast, and that cold cost must not be reported as race latency or as
        a race error.  It is still written to ``call_log`` -- it is a real
        call and the console shows it as proof.
        """
        try:
            raw = self._call_provider_text(timeout)
            self.last_response = (raw or "").strip()[:800]
            self.last_error = ""
            self.fatal_error = ""
            self.warmup_requests += 1
            self._log_call({"time": 0.0}, True, self.last_response or "(空回复)")
            return True, self.last_response
        except Exception as exc:  # noqa: BLE001 - a late model is not fatal
            self.last_error = str(exc)[:300]
            # ...unless the provider said the model is not there at all.  That
            # is not "late", it is unfixable, and the runner must not start.
            self.fatal_error = fatal_reason(exc)
            self.warmup_errors += 1
            self._log_call({"time": 0.0}, False, self.last_error)
            return False, self.last_error

    def _call_provider_text(self, timeout: float) -> str:
        if self.cfg.spec().kind == "anthropic":
            return self._call_anthropic_text(timeout)
        return self._call_openai_text(timeout)

    def _call_openai_text(self, timeout: float) -> str:
        client = self._openai_client()
        resp = client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=64,
            timeout=timeout,
            messages=[
                {"role": "system", "content": HANDSHAKE_SYSTEM_PROMPT},
                {"role": "user", "content": HANDSHAKE_TEXT},
            ],
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic_text(self, timeout: float) -> str:
        client = self._anthropic_client()
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=64,
            timeout=timeout,
            system=HANDSHAKE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": HANDSHAKE_TEXT}],
        )
        # same temperature fallback as the driving call
        if not self._no_temperature:
            kwargs["temperature"] = self.cfg.temperature
        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if not self._no_temperature and "temperature" in str(exc).lower():
                self._no_temperature = True
                kwargs.pop("temperature", None)
                resp = client.messages.create(**kwargs)
            else:
                raise
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts)

    # ------------------------------------------------------------------
    def _call_provider(self, ctx: dict, img, dashboard: dict) -> str:
        """Dispatch on the provider's wire protocol.

        Anthropic has its own SDK; everything else (Ollama, OpenAI and every
        domestic vendor) is spoken over the OpenAI-compatible protocol.
        """
        if self.cfg.spec().kind == "anthropic":
            return self._call_anthropic(img, dashboard)
        return self._call_openai(img, dashboard)

    # ------------------------------------------------------------------
    def _openai_client(self):
        if self._client is None:
            from ..vision.client import openai_client
            self._client = openai_client(self.cfg)
        return self._client

    def _anthropic_client(self):
        if self._client is None:
            from ..vision.client import anthropic_client
            self._client = anthropic_client(self.cfg)
        return self._client

    @staticmethod
    def _frame_text(dashboard: dict) -> str:
        """The caption that rides along with every frame.

        The reminder is deliberate: the obstacle scan is the one step a model
        skips when it is in a hurry, and a skipped scan is a hit.
        """
        return ("第一人称3D驾驶视角画面（从你车内向前看）。先逐个扫描画面中的障碍物"
                "（远近、在左还是在右、从哪一侧绕），再输出 JSON 决策。仪表盘: "
                + json.dumps(dashboard, ensure_ascii=False))

    def _call_openai(self, img, dashboard: dict) -> str:
        from ..vision.render import encode_png_b64
        client = self._openai_client()
        b64 = encode_png_b64(img)
        resp = client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": self._frame_text(dashboard)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, img, dashboard: dict) -> str:
        from ..vision.render import encode_png_b64
        client = self._anthropic_client()
        b64 = encode_png_b64(img)
        content = [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": b64}},
            {"type": "text", "text": self._frame_text(dashboard)},
        ]
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        # the newest Claude models reject sampling parameters; try with the
        # configured temperature first and remember to omit it if refused
        if not self._no_temperature:
            kwargs["temperature"] = self.cfg.temperature
        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if not self._no_temperature and "temperature" in str(exc).lower():
                self._no_temperature = True
                kwargs.pop("temperature", None)
                resp = client.messages.create(**kwargs)
            else:
                raise
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        avg_lat = self.latency_sum / max(1, self.api_requests)
        return {
            "driver": self.kind,
            "model": self.cfg.model,
            "provider": self.cfg.provider,
            "api_requests": self.api_requests,
            "api_errors": self.api_errors,
            "warmup_requests": self.warmup_requests,
            "warmup_errors": self.warmup_errors,
            "avg_latency_ms": round(avg_lat * 1000.0, 1),
            "success_rate": round(1.0 - self.api_errors / max(1, self.api_requests), 3),
            "vision": (getattr(self.last_detections, "detector", None)
                       if self.last_detections else None),
        }
