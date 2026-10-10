"""In-game HUD.

Design rules:

* **minimal by default** -- two one-line car chips, a race pill, the item bar,
  a short event ticker and the centre overlays; ``H`` adds telemetry,
  ``--debug`` adds the AI decision panel.
* **transparent** -- panels are not filled. Text carries a dark outline so it
  stays readable over the road without a solid block covering the view.
* **no hard-coded positions** -- every element is placed from
  :meth:`Hud._metrics`, so nothing overlaps from 540p to 1080p.
* **cheap** -- one reusable surface plus a text cache; no per-frame surface
  allocation and no full-window copy that changes nothing.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import pygame

from ..fonts import CJK_FAMILIES, find_font_file
from ..items import ITEM_DEFS

# colours ---------------------------------------------------------------
CARD = (10, 13, 20, 110)          # only used by the few elements that need a fill
OUTLINE = (0, 0, 0, 190)
TEXT = (240, 245, 252)
DIM = (196, 206, 222)
ACCENT = (255, 226, 130)
GOOD = (140, 232, 170)
BAD = (250, 150, 150)

MAX_TEXT_CACHE = 640

# resolved once per process by :func:`_cjk_font_path`
_FONT_PATH = ""
_FONT_RESOLVED = False


def countdown_label(countdown: float) -> str:
    """What the centre overlay shows for this countdown value.

    One function rather than a literal in the paint path, because the renderer
    uses it as the *redraw key* for the HUD: the caption only changes when this
    string does, and rebuilding a full-window HUD texture (plus its 8 MB
    upload) for a caption that has not changed is pure waste.
    """
    if countdown > 0.01:
        return str(min(3, max(1, int(math.ceil(countdown)))))
    if countdown < -0.01:
        return "GO!"
    return ""


def _cjk_font_path():
    """Locate a CJK-capable font once, then remember it.

    ``pygame.font.match_font`` uses fontconfig on Linux and the registry on
    Windows, so an installed CJK face is found by name even when it lives
    somewhere our candidate list does not know; :func:`find_font_file` is the
    explicit-path / ``fc-list`` fallback (see ``car_game/fonts.py``).
    """
    global _FONT_PATH, _FONT_RESOLVED
    if not _FONT_RESOLVED:
        _FONT_RESOLVED = True
        for name in CJK_FAMILIES:
            try:
                _FONT_PATH = pygame.font.match_font(name)
            except Exception:
                _FONT_PATH = None
            if _FONT_PATH:
                break
        if not _FONT_PATH:
            _FONT_PATH = find_font_file() or ""
    return _FONT_PATH or None


def _font(size: int):
    path = _cjk_font_path()
    if path:
        try:
            return pygame.font.Font(path, size)
        except Exception:
            pass
    return pygame.font.Font(None, size)


class Hud:
    def __init__(self, width: int, height: int):
        pygame.font.init()
        self.w = int(width)
        self.h = int(height)
        self.surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self._cache: OrderedDict = OrderedDict()
        self._fonts: dict = {}
        self._veil = None
        self.detail = False
        self._debug = False
        self.frame_ms = 0.0
        self._build_fonts()

    # -- fonts / text ---------------------------------------------------
    def _build_fonts(self):
        s = self.scale
        self._fonts = {
            "tiny": _font(max(11, int(11 * s))),
            "small": _font(max(12, int(13 * s))),
            "base": _font(max(13, int(15 * s))),
            "big": _font(max(18, int(22 * s))),
            "huge": _font(max(46, int(80 * s))),
        }

    def resize(self, width: int, height: int):
        self.w, self.h = int(width), int(height)
        self.surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self._veil = None
        self._cache.clear()
        self._build_fonts()

    def _paused_veil(self):
        """The dim sheet drawn over a paused race, built once per size.

        Allocating a full-window surface every frame was pure garbage: the
        veil is constant, so it is kept and re-blitted.
        """
        if self._veil is None:
            self._veil = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            self._veil.fill((0, 0, 0, 110))
        return self._veil

    @property
    def scale(self) -> float:
        return max(0.72, min(1.35, self.h / 720.0))

    def text(self, font: str, msg, color):
        key = (font, str(msg), tuple(color))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        img = self._fonts[font].render(str(msg), True, tuple(color))
        self._cache[key] = img
        if len(self._cache) > MAX_TEXT_CACHE:
            self._cache.popitem(last=False)
        return img

    def _fit(self, font: str, msg, max_w: int) -> str:
        f = self._fonts[font]
        msg = str(msg)
        if f.size(msg)[0] <= max_w:
            return msg
        while msg and f.size(msg + "…")[0] > max_w:
            msg = msg[:-1]
        return msg + "…" if msg else "…"

    # -- drawing primitives ---------------------------------------------
    def _blit(self, img, xy):
        self.surface.blit(img, xy)

    def _label(self, font: str, msg, xy, color=TEXT, outline=True):
        """Text with a 1px dark outline, so no background panel is needed."""
        img = self.text(font, msg, color)
        x, y = int(xy[0]), int(xy[1])
        if outline:
            o = self.text(font, msg, OUTLINE)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self.surface.blit(o, (x + dx, y + dy))
        self.surface.blit(img, (x, y))
        return img

    def _label_centred(self, font: str, msg, cx, y, color=TEXT, outline=True):
        img = self.text(font, msg, color)
        return self._label(font, msg, (cx - img.get_width() / 2, y), color, outline)

    def _bar(self, rect, frac, color, bg=(30, 36, 48, 170)):
        pygame.draw.rect(self.surface, bg, rect, border_radius=3)
        w = max(0, int(rect[2] * max(0.0, min(1.0, frac))))
        if w:
            pygame.draw.rect(self.surface, color, (rect[0], rect[1], w, rect[3]),
                             border_radius=3)

    # -- layout ---------------------------------------------------------
    def _metrics(self):
        """All geometry derived from one place so nothing can collide."""
        s = self.scale
        m = max(7, int(10 * s))
        chip_w = max(180, int(250 * s))
        chip_h = max(34, int(50 * s))
        if self.detail or self._debug:
            chip_h = max(62, int(78 * s))
        ch = max(30, int(38 * s))
        if self.detail:
            ch = max(44, int(54 * s))
        item_top = self.h - ch - m
        item_bottom = self.h - m
        return {
            "s": s, "m": m, "chip_w": chip_w, "chip_h": chip_h,
            "item_h": ch, "item_top": item_top, "item_bottom": item_bottom,
            "hint_y": item_top - int(20 * s),
            "body_bottom": item_top - int(26 * s),
        }

    def _chip_rect(self, i: int, mt: dict):
        x = mt["m"] if i == 0 else self.w - mt["m"] - mt["chip_w"]
        return (x, mt["m"], mt["chip_w"], mt["chip_h"])

    # ==================================================================
    def paint(self, game, camera="chase", watch=0, show_watch=True,
              countdown=0.0, paused=False, debug=False):
        self._debug = bool(debug)
        self.surface.fill((0, 0, 0, 0))
        mt = self._metrics()
        self._paint_chips(game, watch, mt)
        self._paint_race_pill(game, watch, mt)
        self._paint_item_bar(game, watch, mt)
        self._paint_events(game, mt)
        self._paint_anticheat(game, mt)
        if self.detail or self._debug:
            self._paint_camera_hint(camera, show_watch, mt)
        if self._debug:
            self._paint_debug(game, mt)
        self._paint_overlays(game, countdown, paused, mt)
        return self.surface

    # -- car chips ------------------------------------------------------
    def _paint_chips(self, game, watch, mt):
        s = mt["s"]
        for i, car in enumerate(game.cars[:2]):
            x, y, w, h = self._chip_rect(i, mt)
            accent = tuple(int(c * 255) for c in car.color)
            active = (i == watch)
            pad = int(14 * s)
            # thin accent bar instead of a translucent card
            pygame.draw.rect(self.surface, accent + (215 if active else 130,),
                             (x, y + int(3 * s), int(3 * s), h - int(8 * s)),
                             border_radius=2)
            pygame.draw.rect(self.surface, accent + (235,),
                             (x + pad, y + int(5 * s),
                              int(9 * s), int(9 * s)), border_radius=3)
            label = (game.model_labels[i] if i < len(game.model_labels)
                     else car.name)
            lap = min(car.laps + (0 if car.finished else 1), game.cfg.laps)
            tag = {"boost": "氮气", "shield": "护盾",
                   "slow": "减速"}.get(car.effect_name(game.time), "")
            # plain text only: the HUD font stack (Microsoft YaHei / SimHei /
            # SimSun) has no glyph for check-mark symbols, and a missing glyph
            # renders as a blank box -- "✔ 完赛" came out as "□ 完赛"
            info = "已完赛" if car.finished else f"圈 {lap}/{game.cfg.laps}"
            if tag:
                info += f"  {tag}"
            info_img = self.text("tiny", info, GOOD if car.finished else DIM)
            name_room = w - pad * 2 - info_img.get_width() - int(16 * s)
            name = self._fit("small", f"{car.name} · {label}", max(60, name_room))
            self._label("small", name, (x + pad + int(14 * s), y + int(2 * s)),
                        TEXT if active else DIM)
            self._label("tiny", info, (x + w - pad - info_img.get_width(),
                                       y + int(4 * s)),
                        GOOD if car.finished else DIM)
            # speed + unit on one baseline
            spd_y = y + int(18 * s)
            spd = self._label("big", f"{car.speed * 3.6:.0f}", (x + pad, spd_y),
                              ACCENT if active else TEXT)
            self._label("tiny", "km/h",
                        (x + pad + spd.get_width() + int(4 * s),
                         spd_y + spd.get_height() - int(15 * s)), DIM)
            if h > int(58 * s):                       # detailed rows
                yy = y + int(50 * s)
                line = (f"撞墙 {car.wall_hits}   撞障碍 {car.obstacle_hits}"
                        f"   道具 {int(sum(car.item_uses.values()))}")
                self._label("tiny", line, (x + pad, yy), DIM)
                self._bar((x + pad, yy + int(14 * s), w - pad * 2, int(4 * s)),
                          float(getattr(car, "race_progress", 0.0)), accent)

    # -- race pill ------------------------------------------------------
    def _paint_race_pill(self, game, watch, mt):
        s, m = mt["s"], mt["m"]
        t = game.time
        score = getattr(game, "session_score", None)
        drivers = getattr(game, "drivers", []) or []
        ai_vs_ai = bool(score) and len(drivers) >= 2 and all(
            d is not None for d in drivers)
        if ai_vs_ai:
            txt = (f"{game.cars[0].name} {score[0]} : {score[1]} "
                   f"{game.cars[1].name}   第 {getattr(game, 'session_match_no', 0)} 局"
                   f"   {t:5.1f}s")
        else:
            lead = game.cars[watch] if 0 <= watch < len(game.cars) else game.cars[0]
            lap = min(lead.laps + 1, game.cfg.laps)
            txt = f"圈 {lap}/{game.cfg.laps}   {t:5.1f}s"
        # keep the pill strictly between the two chips
        room = self.w - 2 * (mt["chip_w"] + 2 * m)
        txt = self._fit("base", txt, max(80, room))
        self._label_centred("base", txt, self.w // 2, m, ACCENT)

    # -- item bar -------------------------------------------------------
    def _paint_item_bar(self, game, watch, mt):
        s = mt["s"]
        player = game.cars[watch] if 0 <= watch < len(game.cars) else game.cars[0]
        n = len(ITEM_DEFS)
        cw = max(52, int(86 * s))
        ch = mt["item_h"]
        gap = max(4, int(6 * s))
        total = n * cw + (n - 1) * gap
        x0 = self.w // 2 - total // 2
        y0 = mt["item_top"]
        for i, d in enumerate(ITEM_DEFS):
            ready_in = player.items.ready_in(d.key, game.time)
            avail = ready_in <= 0.0
            bx = x0 + i * (cw + gap)
            base = tuple(int(v * 255) for v in d.color)
            if avail:
                fill = base + (196,)
                border = (255, 255, 255, 185)
            else:
                fill = (base[0] // 3 + 22, base[1] // 3 + 22,
                        base[2] // 3 + 22, 132)
                border = (110, 118, 134, 165)
            pygame.draw.rect(self.surface, fill, (bx, y0, cw, ch),
                             border_radius=int(7 * s))
            pygame.draw.rect(self.surface, border, (bx, y0, cw, ch), 1,
                             border_radius=int(7 * s))
            key = self.text("tiny", str(i + 1), (16, 16, 20))
            self._blit(key, (bx + int(6 * s), y0 + int(5 * s)))
            if self.detail:
                secs = "" if avail else f"{ready_in:.0f}s"
                secs_img = self.text("tiny", secs, DIM) if secs else None
                room = cw - int(24 * s)
                if secs_img is not None:
                    room -= secs_img.get_width() + int(4 * s)
                name = self._fit("tiny", d.name_cn, max(28, room))
                self._label("tiny", name, (bx + int(22 * s), y0 + int(4 * s)),
                            (255, 255, 255) if avail else DIM, outline=False)
                if secs_img is not None:
                    self._blit(secs_img,
                               (bx + cw - int(6 * s) - secs_img.get_width(),
                                y0 + int(4 * s)))
            bar_y = y0 + ch - int(8 * s)
            if avail:
                self._bar((bx + int(6 * s), bar_y, cw - int(12 * s), int(3 * s)),
                          1.0, (190, 250, 200, 220))
            else:
                frac = 1.0 - min(ready_in / max(d.cooldown, 1e-3), 1.0)
                self._bar((bx + int(6 * s), bar_y, cw - int(12 * s), int(3 * s)),
                          frac, (130, 205, 255, 230))

    # -- events / hint / anticheat --------------------------------------
    def _paint_events(self, game, mt):
        events = getattr(game, "events", None)
        if not events:
            return
        s = mt["s"]
        shown = events[-4:] if self.detail else events[-3:]
        lh = max(12, int(15 * s))
        y = mt["body_bottom"] - lh * len(shown)
        max_w = max(120, self.w // 2 - mt["m"] - 40)
        for ts, ev in shown:
            msg = self._fit("tiny", f"[{ts:5.1f}] {ev}", max_w)
            self._label("tiny", msg, (mt["m"], y), DIM)
            y += lh

    def _paint_camera_hint(self, camera, show_watch, mt):
        hint = f"镜头 {camera}   F1 追尾  F2 第一人称  F3 俯视  F4 自由"
        if show_watch:
            hint += "   Tab 切换车辆"
        hint += "   H 详细"
        self._label_centred("tiny", hint, self.w // 2, mt["hint_y"], DIM)

    def _paint_anticheat(self, game, mt):
        ac = getattr(game, "anticheat", None)
        if ac is None or not getattr(ac, "enabled", False) or not ac.has_serious():
            return
        reasons = ac.live_reasons(2)
        if not reasons:
            return
        s = mt["s"]
        lines = ["反作弊判罚"] + reasons
        w = min(int(460 * s), self.w - 2 * mt["m"])
        lh = max(13, int(16 * s))
        h = int(8 * s) + lh * len(lines)
        x = self.w // 2 - w // 2
        y = max(mt["m"], self.h // 2 - int(150 * s) - h)
        pygame.draw.rect(self.surface, (150, 22, 22, 190), (x, y, w, h),
                         border_radius=int(7 * s))
        pygame.draw.rect(self.surface, (255, 130, 130, 225), (x, y, w, h), 1,
                         border_radius=int(7 * s))
        yy = y + int(4 * s)
        for i, ln in enumerate(lines):
            self._label_centred("small" if i == 0 else "tiny", ln,
                                self.w // 2, yy,
                                (255, 228, 180) if i == 0 else (255, 244, 244),
                                outline=False)
            yy += lh

    # -- debug ----------------------------------------------------------
    def _debug_lines(self, game):
        lines = []
        if self.frame_ms > 0.01:
            lines.append(f"帧率 {1000.0 / self.frame_ms:4.1f} fps   "
                         f"{self.frame_ms:5.2f} ms/帧")
        drivers = getattr(game, "drivers", []) or []
        actions = getattr(game, "last_action", []) or []
        reasons = getattr(game, "last_reason", []) or []
        labels = getattr(game, "model_labels", []) or []
        for i, car in enumerate(game.cars):
            d = drivers[i] if i < len(drivers) else None
            lines.append(f"[{car.name}] {labels[i] if i < len(labels) else '?'}")
            act = actions[i] if i < len(actions) else None
            if act is not None:
                lines.append(f"  油门 {act.throttle:.2f}  刹车 {act.brake:.2f}  "
                             f"转向 {act.steer:+.2f}  道具 {act.use_item or '-'}")
            lines.append(f"  理由 {(reasons[i] if i < len(reasons) else '')[:40]}")
            if d is not None and getattr(d, "kind", "") == "llm":
                req = getattr(d, "api_requests", 0)
                err = getattr(d, "api_errors", 0)
                lat = getattr(d, "latency_sum", 0.0) / max(1, req) * 1000.0
                lines.append(f"  视觉LLM {getattr(d.cfg, 'model', '?')}  请求 {req}  "
                             f"错误 {err}  平均 {lat:.0f}ms")
                resp = getattr(d, "last_response", "")
                if resp:
                    lines.append(f"  响应 {resp[:42]}")
                errt = getattr(d, "last_error", "")
                if errt:
                    lines.append(f"  错误 {errt[:42]}")
            if d is not None:
                det = getattr(d, "last_detections", None)
                if det is not None:
                    lines.append(f"  视觉检测: 障碍 {len(det.obstacles)}  "
                                 f"对手 {'可见' if det.rival else '不可见'}")
            elif d is None:
                lines.append("  人类玩家（键盘输入）")
        return lines

    def _paint_debug(self, game, mt):
        s = mt["s"]
        lines = self._debug_lines(game)
        lh = max(12, int(14 * s))
        w = min(int(340 * s), self.w // 2 - 2 * mt["m"])
        # sit BELOW the right-hand chip and stop above the item bar
        x = self.w - mt["m"] - w
        y = mt["m"] + mt["chip_h"] + int(10 * s)
        room = max(0, mt["body_bottom"] - y)
        visible = max(1, min(len(lines), room // lh))
        yy = y
        for ln in lines[:visible]:
            col = ACCENT if ln.startswith("[") else TEXT
            msg = self._fit("tiny", ln, w)
            self._label("tiny", msg, (x, yy), col)
            yy += lh

    # -- overlays -------------------------------------------------------
    def _paint_overlays(self, game, countdown, paused, mt):
        s = mt["s"]
        label = countdown_label(countdown)
        if label:
            img = self.text("huge", label, ACCENT)
            cx = self.w // 2 - img.get_width() // 2
            cy = self.h // 2 - img.get_height() // 2
            o = self.text("huge", label, OUTLINE)
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                self._blit(o, (cx + dx, cy + dy))
            self._blit(img, (cx, cy))

        if game.state == "finished":
            self._paint_finished(game, mt)

        if paused and game.state == "racing":
            self._blit(self._paused_veil(), (0, 0))
            self._label_centred("huge", "已暂停", self.w // 2,
                                self.h // 2 - int(40 * s), ACCENT)

    def _paint_finished(self, game, mt):
        s = mt["s"]
        w = min(int(560 * s), self.w - 2 * mt["m"])
        h = int(210 * s)
        x = self.w // 2 - w // 2
        y = self.h // 2 - h // 2
        pygame.draw.rect(self.surface, (10, 13, 20, 175), (x, y, w, h),
                         border_radius=int(10 * s))
        pygame.draw.rect(self.surface, (180, 195, 215, 150), (x, y, w, h), 1,
                         border_radius=int(10 * s))
        reason = getattr(game, "end_reason", "") or ""
        if reason:
            head, col = "反作弊判负", BAD
        elif game.winner is not None:
            win = game.cars[game.winner]
            head = f"{win.name} 获胜 · {win.finish_time:.2f}s"
            col = ACCENT
        else:
            head, col = "本局结束", ACCENT
        self._label_centred("big", head, self.w // 2, y + int(14 * s), col)
        if reason:
            self._label_centred("tiny", self._fit("tiny", reason, w - int(24 * s)),
                                self.w // 2, y + int(46 * s), BAD)
        yy = y + int(74 * s)
        for i, car in enumerate(game.cars):
            accent = tuple(int(c * 255) for c in car.color)
            pygame.draw.rect(self.surface, accent, (x + int(18 * s), yy + 2,
                                                    int(11 * s), int(11 * s)),
                             border_radius=3)
            label = (game.model_labels[i] if i < len(game.model_labels) else car.name)
            status = f"完成 {car.finish_time:.2f}s" if car.finished else "未完成"
            self._label("base", self._fit("base", f"{car.name}  {label}",
                                          w - int(150 * s)),
                        (x + int(36 * s), yy - int(3 * s)), TEXT, outline=False)
            st = self.text("tiny", status, GOOD if car.finished else BAD)
            self._label("tiny", status,
                        (x + w - st.get_width() - int(18 * s), yy),
                        GOOD if car.finished else BAD, outline=False)
            yy += int(28 * s)
        score = getattr(game, "session_score", None)
        if score:
            names = getattr(game, "session_names", None) or [c.name for c in game.cars]
            self._label_centred(
                "base",
                f"总比分 {names[0]} {score[0]} : {score[1]} {names[1]}"
                f"   平局 {getattr(game, 'session_draws', 0)}",
                self.w // 2, yy + int(4 * s), ACCENT, outline=False)
        self._label_centred("tiny", "即将进入下一局…（Esc 退出）", self.w // 2,
                            y + h - int(20 * s), DIM, outline=False)
