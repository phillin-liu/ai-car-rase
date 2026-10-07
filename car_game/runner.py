"""Multi-match session runner.

Runs a configured session either head-less (as fast as possible) or in a
pygame/OpenGL window with optional human control.  Results are analysed and
stored in the transient session store after every match.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from datetime import datetime

from . import analysis as analysis_mod
from .ai import build_driver
from .config import GameConfig, RUNTIME_DIR, ensure_dirs
from .core_types import Action
from .game import RaceGame
from .items import ITEM_KEYS
from .store import MatchStore, RuntimeHistoryStore
from .track import generate_track


STATUS_FILE = os.path.join(RUNTIME_DIR, "status.json")
STOP_FILE = os.path.join(RUNTIME_DIR, "stop.flag")
PAUSE_FILE = os.path.join(RUNTIME_DIR, "pause.flag")
# "a console is watching me, so it can answer questions I cannot" -- written by
# the console when it spawns this process, removed when it exits
GUI_FILE = os.path.join(RUNTIME_DIR, "console.flag")
# the console's answer to "keep waiting for the models?" ("wait" / "go")
HANDSHAKE_FILE = os.path.join(RUNTIME_DIR, "handshake.flag")

# How long one "are you ready?" call may take.  This is a safety net for a dead
# network, *not* a waiting period: the first call to a provider is the expensive
# one (DNS + TLS + a cold model can run to tens of seconds) and every later call
# on the same client is quick.  A live-but-slow model is waited for, because
# starting the race without it is what makes the two cars unequal.
WARMUP_TIMEOUT = 180.0
# how long a console gets to answer our question before we fall back to the
# terminal / give up waiting
CONSOLE_ANSWER_GRACE = 8.0

# How long the field is held on the line while the opening decision is fetched.
# The connection is already warm (the handshake just opened it), so this only
# bounds a genuinely slow model, and the race starts either way once it runs
# out.  It exists because that first round-trip used to be paid *after* the
# green: the car braked on the line for seconds while the other car drove away.
PRIME_TIMEOUT = 12.0


# ---------------------------------------------------------------------------
# ``status.json`` is written here and read by the console in a *different*
# process, so the atomic swap below races that reader.  Windows gives no share
# mode that lets ``os.replace`` replace a file someone else has open: the call
# fails with a sharing violation (WinError 5/32) for as long as the reader --
# or a virus scanner that just picked the file up -- holds it.  That used to
# raise straight out of the render loop and kill the session.  The status file
# is a best-effort UI hint that nothing else depends on, so a blocked swap is
# retried briefly and then written in place instead: it must never be able to
# end a race.
STATUS_REPLACE_TRIES = 3
STATUS_REPLACE_DELAY = 0.03     # seconds between retries (~90 ms worst case)


def write_status(data: dict) -> None:
    ensure_dirs()
    data = dict(data)
    data["updated"] = datetime.now().isoformat(timespec="seconds")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = STATUS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        return                    # nowhere to stage it: skip this update
    swapped = False
    for _ in range(STATUS_REPLACE_TRIES):
        try:
            os.replace(tmp, STATUS_FILE)
            swapped = True
            break
        except PermissionError:
            # the console poll or a scanner has it open right now -- back off
            # and try again so the reader keeps getting whole files
            time.sleep(STATUS_REPLACE_DELAY)
        except OSError:
            break
    if not swapped:
        # Still locked.  Writing over the old file in place opens with
        # share-write (only the *replace* needs share-delete), so it gets
        # through -- and ``read_status`` already treats a half-written file as
        # "no status", which costs the console one poll and nothing else.
        try:
            with open(STATUS_FILE, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            pass
        try:
            os.remove(tmp)
        except OSError:
            pass


def read_status() -> dict:
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def request_stop() -> None:
    ensure_dirs()
    with open(STOP_FILE, "w", encoding="utf-8") as fh:
        fh.write("stop")


def clear_stop() -> None:
    try:
        os.remove(STOP_FILE)
    except OSError:
        pass


def stop_requested() -> bool:
    return os.path.exists(STOP_FILE)


# --- pre-race handshake signalling -----------------------------------------
def mark_gui_present() -> None:
    """Called by the console: 'a window is watching, ask me questions'."""
    ensure_dirs()
    with open(GUI_FILE, "w", encoding="utf-8") as fh:
        fh.write("console")


def clear_gui_present() -> None:
    try:
        os.remove(GUI_FILE)
    except OSError:
        pass


def gui_present() -> bool:
    return os.path.exists(GUI_FILE)


def answer_handshake(decision: str) -> None:
    """The console's answer: ``"wait"`` to keep waiting, ``"go"`` to start."""

    ensure_dirs()
    with open(HANDSHAKE_FILE, "w", encoding="utf-8") as fh:
        fh.write(str(decision))


def handshake_answer() -> str:
    try:
        with open(HANDSHAKE_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def clear_handshake() -> None:
    try:
        os.remove(HANDSHAKE_FILE)
    except OSError:
        pass


def request_pause() -> None:
    ensure_dirs()
    with open(PAUSE_FILE, "w", encoding="utf-8") as fh:
        fh.write("pause")


def clear_pause() -> None:
    try:
        os.remove(PAUSE_FILE)
    except OSError:
        pass


def pause_requested() -> bool:
    return os.path.exists(PAUSE_FILE)


def toggle_pause() -> bool:
    """Toggle pause state; returns True if the race is now paused."""
    if pause_requested():
        clear_pause()
        return False
    request_pause()
    return True


# ---------------------------------------------------------------------------
def _fresh_seed(used: set) -> int:
    """A random track seed not yet used in this session."""
    rng = random.SystemRandom()
    for _ in range(128):
        s = rng.randint(1, 2_000_000_000)
        if s not in used:
            return s
    return rng.randint(1, 2_000_000_000)


def _make_track(cfg: GameConfig, used: set):
    """Generate the next map (a bad random draw retries)."""
    for _ in range(8):
        seed = _fresh_seed(used)
        used.add(seed)
        try:
            return generate_track(seed, cfg.match.track_width,
                                  cfg.match.items_enabled)
        except Exception:
            continue
    raise RuntimeError("无法生成赛道")


def _tty_available() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def _ask_keep_waiting(remaining, labels, errors, on_tick=None) -> bool:
    """Ask whether to keep waiting for the late models.  True = keep waiting.

    Three channels, in order: the console (it can raise a dialog), the
    terminal, and -- if there is neither -- nobody, in which case the caller
    grants one extra bounded retry instead of asking.
    """
    who = "、".join(labels[i] for i in remaining)
    why = "；".join(f"{labels[i]}: {errors.get(i, '未回复')[:60]}"
                    for i in remaining)
    if gui_present():
        deadline = time.time() + CONSOLE_ANSWER_GRACE
        while time.time() < deadline:
            if stop_requested():
                return False
            if on_tick is not None and not on_tick():
                return False
            answer = handshake_answer()
            if answer in ("wait", "go"):
                # consume it: one answer authorises exactly one more round, so
                # a stale "wait" can never turn the retry loop into a spin
                clear_handshake()
                return answer == "wait"
            time.sleep(0.2)
        return False                      # no answer in time: start anyway
    if _tty_available():
        print(f"[握手] {who} 尚未回复：{why}")
        try:
            answer = input("是否继续等待？(y = 继续等，其他 = 直接开始) ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower().startswith("y")
    return False


def _unusable_model(drivers, labels) -> str:
    """A line naming every side whose model the provider refused outright.

    "模型不存在" is not a slow connection and not a transient error -- retrying
    it forever just wastes the user's time, so the caller ends the session
    rather than racing a car that can never be driven.
    """
    bad = []
    for i, d in enumerate(drivers):
        if d is None:
            continue
        why = getattr(d, "fatal_error", "")
        if why:
            bad.append(f"{labels[i]}（{why}）")
    return "、".join(bad)


def _end_unusable(base_status: dict, who: str) -> None:
    """Stop the session because a model cannot be called at all."""
    data = dict(base_status)
    data["state"] = "error"
    data["error"] = f"{who} 无法调用（模型不存在或无权限），比赛已结束"
    write_status(data)
    print(f"[模型] {who} 无法调用（模型不存在或无权限），比赛结束")
    request_stop()


def _handshake(drivers, labels, cfg, base_status: dict, on_tick=None) -> list:
    """Ask every AI side to confirm it is ready before the race starts.

    Every side is asked **at the same time**, and the race waits for all of
    them.  There is deliberately no short fixed deadline: the first call to a
    provider costs the same for both cars (DNS + TLS + a cold model), so the
    wait is symmetric.  What is *not* symmetric is starting anyway -- the warm
    car drives off while the cold one is still connecting and coasts, and that
    is what corrupts the match data.  So the start is never given away on a
    timer; only an explicit "start now" from the user skips the wait.

    Returns the sides that never answered (empty when everyone is ready).
    """
    sides = [i for i, d in enumerate(drivers)
             if d is not None and hasattr(d, "handshake")]
    if not sides:
        return []

    clear_handshake()
    ready: dict = {}
    errors: dict = {}
    rounds = 0

    def broadcast(note: str = "") -> None:
        data = dict(base_status)
        data["state"] = "waiting_models"
        data["handshake"] = {
            "answered": [labels[i] for i in sorted(ready)],
            "pending": [labels[i] for i in sides if i not in ready],
            # "still connecting" and "gave up" are different situations: only
            # the second one is worth interrupting the user about
            "failed": [labels[i] for i in sorted(errors) if i not in ready],
            "round": rounds,
            "note": note,
        }
        write_status(data)

    def attempt(idx: int) -> None:
        ok, detail = drivers[idx].handshake(WARMUP_TIMEOUT)
        if ok:
            ready[idx] = detail
            errors.pop(idx, None)
        else:
            # a side that already confirmed stays ready: the re-warm round at
            # the end must not be able to push it back into the pending list
            errors[idx] = detail

    def run_round(indices) -> bool:
        """Fire the handshakes at once and wait, keeping the window alive."""
        nonlocal rounds
        rounds += 1
        threads = [threading.Thread(target=attempt, args=(i,), daemon=True)
                   for i in indices]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            if stop_requested():
                return False
            if on_tick is not None and not on_tick():
                return False
            for t in threads:
                t.join(0.05)
        return True

    def unusable() -> bool:
        """End the session the moment a model is refused outright."""
        who = _unusable_model(drivers, labels)
        if not who:
            return False
        _end_unusable(base_status, who)
        clear_handshake()
        return True

    broadcast("等待模型确认就绪")
    if not run_round(sides):
        clear_handshake()
        return [i for i in sides if i not in ready]
    if unusable():
        return [i for i in sides if i not in ready]
    broadcast("等待模型确认就绪")

    # With no console and no terminal there is nobody to ask, so grant exactly
    # one extra retry rather than stalling the whole session on a dead model.
    extra_rounds = 0 if (gui_present() or _tty_available()) else 1
    while True:
        remaining = [i for i in sides if i not in ready]
        if not remaining or stop_requested():
            break
        if extra_rounds > 0:
            extra_rounds -= 1
            broadcast("无控制台/终端可询问，再宽限一轮")
            if not run_round(remaining):
                break
            if unusable():
                return [i for i in sides if i not in ready]
            broadcast("等待模型确认就绪")
            continue
        if not _ask_keep_waiting(remaining, labels, errors, on_tick):
            break
        broadcast("继续等待模型")
        if not run_round(remaining):
            break
        if unusable():
            return [i for i in sides if i not in ready]
        broadcast("等待模型确认就绪")

    # If one side answered long before the other, the quick side's connection
    # has been idle ever since -- exactly the cold start we just waited out.
    # Ping everyone once more, together, so both clients are warm and both cars
    # leave the line on the same tick.
    if rounds > 1 and not stop_requested() and not [i for i in sides
                                                    if i not in ready]:
        broadcast("重新预热全部模型，保证同时起步")
        run_round(sides)

    clear_handshake()
    return [i for i in sides if i not in ready]


def _prime_drivers(game, drivers, labels, base_status, on_tick=None) -> None:
    """Fetch the opening decision before the race clock starts.

    Every side is primed from the same tick with the same kind of observation,
    and the countdown only begins once they have answered (or ``PRIME_TIMEOUT``
    runs out).  Rendering the first frame and waiting for the model to read it
    is the single most expensive call of a match; doing it here makes it cost
    pit-lane time instead of race distance, so both cars leave the line on the
    green with a real decision instead of coasting while they wait.
    """
    sides = [i for i, d in enumerate(drivers)
             if d is not None and hasattr(d, "prime")]
    if not sides:
        return
    game.prime_drivers()

    def pending() -> list:
        return [i for i in sides
                if not getattr(drivers[i], "has_decision", True)]

    def broadcast() -> None:
        data = dict(base_status)
        data["state"] = "waiting_models"
        data["handshake"] = {
            "answered": [],
            "pending": [labels[i] for i in pending()],
            "failed": [],
            "round": 0,
            "note": "等待模型首次决策（开局画面渲染中）",
        }
        write_status(data)

    if pending():
        broadcast()
    deadline = time.time() + PRIME_TIMEOUT
    while pending() and time.time() < deadline and not stop_requested():
        if on_tick is not None:
            # keep the window drawing while we wait, exactly like the handshake
            if not on_tick():
                return
        else:
            time.sleep(0.05)


def _driver_status(game) -> list:
    """Compact per-driver info (API request/error counts, last decision)."""
    out = []
    for d in game.drivers:
        if d is None:
            out.append({"driver": "human"})
            continue
        try:
            info = dict(d.stats())
        except Exception:
            info = {"driver": getattr(d, "kind", "?")}
        if hasattr(d, "debug_info"):
            try:
                info.update(d.debug_info())
            except Exception:
                pass
        out.append(info)
    return out


_history_store = None


def _history() -> RuntimeHistoryStore:
    """The cross-session history store, opened once per process."""
    global _history_store
    if _history_store is None:
        _history_store = RuntimeHistoryStore()
    return _history_store


def _archive_match(analysis, result, cfg) -> None:
    """Copy a finished match into the persistent history.

    Deliberately swallowed: a locked file or a full disk must never take down a
    session that is otherwise running fine.
    """
    try:
        _history().save_match(analysis, result, cfg.to_public_dict())
    except Exception:
        pass


def _track_outline(track, points: int = 72):
    """A small closed polyline of the centre line, for the console's map."""
    try:
        xz = (track.center_xz if track.center_xz is not None
              else track.center[:, [0, 2]])
        n = len(xz)
        if n < 2:
            return []
        step = max(1, n // max(1, int(points)))
        return [[round(float(xz[i][0]), 2), round(float(xz[i][1]), 2)]
                for i in range(0, n, step)]
    except Exception:
        return []


def _live_status(game, match_id, total, infinite, labels,
                 score=None, draws=0) -> dict:
    """Compact in-race state written to runtime/status.json for the GUI."""
    snap = game.snapshot()
    data = {
        "state": game.state,
        "match": match_id,
        "total": total,
        "infinite": infinite,
        "models": labels,
        "laps": game.cfg.laps,
        "live": {
            "time": round(game.time, 1),
            "cars": [
                {
                    "name": c["name"],
                    "model": c.get("model", ""),
                    "speed": c["speed"],
                    "laps": c["laps"],
                    "wall_hits": c["wall_hits"],
                    "obstacle_hits": c["obstacle_hits"],
                    "item_uses": c.get("item_uses", 0),
                    "effect": c.get("effect", ""),
                    "finished": c["finished"],
                    # the console draws the cars on its own track map, so it
                    # needs where they are, not just how fast
                    "pos": [round(float(c["pos"][0]), 2),
                            round(float(c["pos"][1]), 2)],
                    "heading": round(float(c["heading"]), 4),
                    "progress": c.get("progress", 0.0),
                }
                for c in snap["cars"]
            ],
            "track": {"outline": _track_outline(getattr(game, "track", None))},
            "drivers": snap["drivers"],
        },
        "anticheat": {
            **game.anticheat.summary(),
            "live_reasons": game.anticheat.live_reasons(4),
        },
    }
    if score is not None:
        data["score"] = {"p1": int(score[0]), "p2": int(score[1]),
                         "draws": int(draws)}
    return data


def _build_drivers(track, cfg: GameConfig, allow_human: bool):
    drivers = []
    labels = []
    for idx, side in enumerate((cfg.p1, cfg.p2)):
        human = (cfg.match.mode == "human_vs_ai" and cfg.match.human_player == idx)
        if human and allow_human:
            drivers.append(None)
            labels.append("人类玩家" if not side.display_name else side.display_name)
        else:
            # Either an AI side, or a human side nobody is actually driving
            # (unseen in practice: head-less runs downgrade human-vs-AI to
            # AI-vs-AI first).  There is no offline stand-in any more -- the
            # side is driven by whatever model it is configured with.
            # P2's queries are offset by half an interval: rendering a frame is
            # a pure-Python job that holds the GIL, so two sides deciding on
            # the same tick collide and stall the 3D window.
            drivers.append(build_driver(side, track, f"P{idx+1}",
                                        cfg.match.difficulty, stagger=0.5 * idx))
            labels.append(side.label())
    return drivers, labels


# ---------------------------------------------------------------------------
# Keyboard: Windows virtual-key codes for a global (focus-independent) read
_VK = {
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
}
_ITEM_VKS = [0x31, 0x32, 0x33, 0x34, 0x35]      # 1..5


def _async_down(vk: int) -> bool:
    """Windows global key state – works even if the window is unfocused."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        return False


class HumanController:
    """Reads keyboard input and produces Actions (windowed mode only).

    Three redundant sources are combined so keys always work: the pygame
    event queue, ``pygame.key.get_pressed`` and (on Windows) the global
    ``GetAsyncKeyState`` API.  This survives a window that never received
    focus, lost focus, or missed KEYUP events.

    Steering: positive steer turns the car right on screen, negative left.
    Arrow keys and WASD both work.
    """

    def __init__(self):
        self._pressed = set()
        self._item_queue = []
        self._async_prev = set()

    def feed(self, events) -> None:
        """Update key state / item queue from pygame events."""
        import pygame
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                self._pressed.add(ev.key)
                for i, k in enumerate((pygame.K_1, pygame.K_2, pygame.K_3,
                                       pygame.K_4, pygame.K_5)):
                    if ev.key == k and i < len(ITEM_KEYS):
                        self._item_queue.append(ITEM_KEYS[i])
            elif ev.type == pygame.KEYUP:
                self._pressed.discard(ev.key)

    def get_action(self, events=None) -> Action:
        import pygame
        if events:
            self.feed(events)
        try:
            polled = pygame.key.get_pressed()
        except Exception:
            polled = None

        def down(*pairs) -> bool:
            for pk, vk in pairs:
                if pk in self._pressed:
                    return True
                if polled is not None:
                    try:
                        if polled[pk]:
                            return True
                    except Exception:
                        pass
                if _async_down(vk):
                    return True
            return False

        throttle = 1.0 if down((pygame.K_UP, _VK["up"]),
                               (pygame.K_w, _VK["w"])) else 0.0
        brake = 1.0 if down((pygame.K_DOWN, _VK["down"]),
                            (pygame.K_s, _VK["s"])) else 0.0

        # Screen-left -> negative steer, screen-right -> positive steer.
        steer = 0.0
        if down((pygame.K_LEFT, _VK["left"]), (pygame.K_a, _VK["a"])):
            steer -= 1.0
        if down((pygame.K_RIGHT, _VK["right"]), (pygame.K_d, _VK["d"])):
            steer += 1.0

        # item keys: edge-triggered from the global keys (focus independent)
        for i, vk in enumerate(_ITEM_VKS):
            now_down = _async_down(vk)
            if now_down and vk not in self._async_prev and i < len(ITEM_KEYS):
                self._item_queue.append(ITEM_KEYS[i])
            if now_down:
                self._async_prev.add(vk)
            else:
                self._async_prev.discard(vk)

        use_item = self._item_queue.pop(0) if self._item_queue else None
        return Action(throttle=throttle, brake=brake, steer=steer,
                      use_item=use_item, reason="human")


def _focus_game_window() -> None:
    """Best-effort: bring the pygame window to the foreground (Windows)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        import pygame
        hwnd = pygame.display.get_wm_info().get("window")
        if hwnd:
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 5)          # SW_SHOW
            user32.SetForegroundWindow(hwnd)
            user32.SetFocus(hwnd)
    except Exception:
        pass


# ---------------------------------------------------------------------------
def run_headless(cfg: GameConfig, on_match=None, verbose: bool = True) -> dict:
    """Run a full session without opening a window.

    Head-less mode is intended for server training, therefore human
    controls are never supported here: a human_vs_ai request is silently
    downgraded to ai_vs_ai.
    """
    ensure_dirs()
    clear_stop()
    clear_pause()
    if cfg.match.mode == "human_vs_ai":
        cfg.match.mode = "ai_vs_ai"
    store = MatchStore()
    store.clear()

    total = int(cfg.match.matches)
    infinite = total <= 0
    index = 0
    session_start = time.time()
    last_analysis = None
    score = [0, 0]
    draws = 0
    # One random map per match, generated on demand.
    used_seeds: set = set()

    while infinite or index < total:
        if stop_requested():
            break
        match_id = index + 1
        track = _make_track(cfg, used_seeds)
        drivers, labels = _build_drivers(track, cfg, allow_human=False)
        game = RaceGame(track, cfg.match, drivers, labels, match_id=match_id)
        game.session_score = list(score)
        game.session_draws = draws
        game.session_match_no = match_id
        game.session_total = total
        game.session_names = [str(labels[0])[:12], str(labels[1])[:12]]

        # keep real-time pace when real LLM providers are in use so the async
        # decision threads can actually keep up; otherwise run flat out.
        llm_active = any(getattr(d, "kind", None) == "llm"
                         for d in drivers if d is not None)
        pacing = bool(cfg.match.realtime and llm_active)

        # nobody starts until the models say they are ready
        late = _handshake(drivers, labels, cfg, {
            "match": match_id, "total": total, "infinite": infinite,
            "laps": game.cfg.laps, "models": labels,
            "score": {"p1": score[0], "p2": score[1], "draws": draws},
        })
        if late:
            print(f"[握手] {', '.join(labels[i] for i in late)} 未确认，仍然开始")
        if stop_requested():
            break           # a model was refused outright: no race to run

        # both sides take their opening look before the clock starts
        _prime_drivers(game, drivers, labels, {
            "match": match_id, "total": total, "infinite": infinite,
            "laps": game.cfg.laps, "models": labels,
            "score": {"p1": score[0], "p2": score[1], "draws": draws},
        })

        dt = 1.0 / max(30, cfg.match.fps)
        t0 = time.time()
        last_live = 0.0
        while game.state == "racing":
            if pause_requested():
                time.sleep(0.05)
                continue
            why = _unusable_model(drivers, labels)
            if why:
                _end_unusable(_live_status(game, match_id, total, infinite,
                                           labels, score, draws), why)
                break
            game.step(dt)
            if pacing:
                lag = (t0 + game.time) - time.time()
                if lag > 0:
                    time.sleep(min(lag, 0.05))
            now_wall = time.time()
            if now_wall - last_live > 0.5:
                last_live = now_wall
                write_status(_live_status(game, match_id, total,
                                          infinite, labels, score, draws))
            if stop_requested():
                break
        result = game.result
        if result is None:
            break

        if result.winner is not None:
            score[result.winner] += 1
        else:
            draws += 1

        analysis = analysis_mod.analyze_match(result.to_dict())
        store.save_match(analysis, result.to_dict(), cfg.to_public_dict())
        _archive_match(analysis, result.to_dict(), cfg)
        last_analysis = analysis

        write_status({
            "state": "running" if (infinite or index + 1 < total) else "done",
            "match": match_id,
            "laps": game.cfg.laps,
            "total": total,
            "infinite": infinite,
            "last": analysis,
            "models": labels,
            "anticheat": game.anticheat.summary(),
            "drivers": _driver_status(game),
            "score": {"p1": score[0], "p2": score[1], "draws": draws},
            "elapsed": round(time.time() - session_start, 1),
        })
        if on_match:
            on_match(analysis)
        if verbose:
            _print_match(analysis, match_id, total, len(labels))
        for d in drivers:
            if d is not None:
                d.close()
        index += 1

    final = read_status()
    # "error" outranks "stopped": the console must keep showing *why* the run
    # ended, not just that the stop flag was set
    if stop_requested() and final.get("state") != "error":
        final["state"] = "stopped"
    write_status(final)
    clear_stop()
    return {"matches": index, "last": last_analysis}


def _print_match(analysis: dict, match_id: int, total: int, n: int) -> None:
    tag = "AI对AI" if n == 2 else ""
    print(f"\n===== 第 {match_id}" + (f"/{total}" if total > 0 else "") + f" 局 {tag} =====")
    print(analysis_mod.format_comparison_table(analysis))
    for a in analysis.get("analyses", []):
        ds = a.get("driver_stats", {}) or {}
        if ds.get("driver") == "llm":
            print(f"  [AI 调用] {a['name']} {a['model']}: "
                  f"请求 {ds.get('api_requests')}  错误 {ds.get('api_errors')}  "
                  f"平均延迟 {ds.get('avg_latency_ms')}ms  "
                  f"成功率 {ds.get('success_rate')}")
    import sys
    sys.stdout.flush()


# ---------------------------------------------------------------------------
def run_windowed(cfg: GameConfig, on_match=None) -> dict:
    """Run a full session in a pygame/OpenGL window."""
    import pygame
    from OpenGL.GL import GL_SAMPLES, glGetIntegerv
    from .render import Renderer

    ensure_dirs()
    clear_stop()
    clear_pause()
    store = MatchStore()
    store.clear()

    pygame.init()
    # Ask for a real depth buffer and 4x MSAA *before* the GL context exists.
    # The old code never requested a depth buffer at all, which is a latent
    # z-ordering bug on its own.  If the driver refuses MSAA we retry without.
    msaa = bool(getattr(cfg.match, "msaa", True))
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    size = (cfg.match.window_width, cfg.match.window_height)
    # MSAA multiplies the fragment work, and it is the *pixel count* that makes
    # it hurt: at 1080p on integrated graphics 4x multisampling alone can cost
    # more than the entire rest of the frame, while at 720p it is nearly free.
    # So the sample count follows the resolution; the checkbox still turns it
    # off entirely for anyone who wants the last few frames.
    samples = 4 if size[0] * size[1] <= 1600 * 900 else 2
    try:
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        if msaa:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, samples)
        pygame.display.set_mode(size, flags)
    except pygame.error:
        msaa = False
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.set_mode(size, flags)
    try:
        if int(glGetIntegerv(GL_SAMPLES)) <= 0:
            msaa = False
    except Exception:
        pass
    if msaa:
        print(f"[画面] {size[0]}x{size[1]} 抗锯齿 {samples}x")
    pygame.display.set_caption("AI Car Race - 3D")
    renderer = Renderer(cfg.match.window_width, cfg.match.window_height, msaa=msaa)
    _focus_game_window()

    total = int(cfg.match.matches)
    infinite = total <= 0
    index = 0
    human = HumanController() if cfg.match.mode == "human_vs_ai" else None
    camera = cfg.match.view
    can_switch = cfg.match.spectate and cfg.match.mode == "ai_vs_ai"
    watch = cfg.match.human_player if cfg.match.mode == "human_vs_ai" else 0
    running = True
    session_start = time.time()
    score = [0, 0]
    draws = 0
    used_seeds: set = set()

    while running and (infinite or index < total):
        if stop_requested():
            break
        match_id = index + 1
        track = _make_track(cfg, used_seeds)
        drivers, labels = _build_drivers(track, cfg, allow_human=True)
        game = RaceGame(track, cfg.match, drivers, labels, match_id=match_id)
        game.session_score = list(score)
        game.session_draws = draws
        game.session_match_no = match_id
        game.session_total = total
        game.session_names = [str(labels[0])[:12], str(labels[1])[:12]]
        renderer.set_track(track)

        clock = pygame.time.Clock()
        dt = 1.0 / max(30, cfg.match.fps)
        # Pay the cold-start cost of the new scene here, with the cars held on
        # the line, rather than inside the first measured frame of the race
        # (see Renderer.warmup).
        renderer.warmup(game, camera=camera, watch=watch, show_watch=can_switch,
                        dt=dt, detail=renderer.hud.detail,
                        debug=bool(getattr(cfg.match, "debug", False)))

        # nobody starts until the models say they are ready; this runs before
        # the countdown, so the window is up but the cars are held on the line
        def _keep_window_alive() -> bool:
            """Pump events + redraw while we wait, so the window is not frozen."""
            nonlocal running
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
                    return False
            renderer.draw(game, camera=camera, watch=watch,
                          show_watch=can_switch, dt=dt, countdown=0.0,
                          paused=False, detail=renderer.hud.detail,
                          debug=bool(getattr(cfg.match, "debug", False)))
            pygame.display.flip()
            clock.tick(cfg.match.fps)
            return True

        late = _handshake(drivers, labels, cfg, {
            "match": match_id, "total": total, "infinite": infinite,
            "laps": game.cfg.laps, "models": labels,
            "score": {"p1": score[0], "p2": score[1], "draws": draws},
        }, on_tick=_keep_window_alive)
        if late:
            print(f"[握手] {', '.join(labels[i] for i in late)} 未确认，仍然开始")
        if not running:
            break

        # Held on the line while the opening decision is fetched -- the
        # countdown below only starts once both cars can actually drive.
        _prime_drivers(game, drivers, labels, {
            "match": match_id, "total": total, "infinite": infinite,
            "laps": game.cfg.laps, "models": labels,
            "score": {"p1": score[0], "p2": score[1], "draws": draws},
        }, on_tick=_keep_window_alive)
        if not running:
            break

        intro = 3.0
        go_flash = 0.0
        overlay_time = 0.0
        focus_frames = 90
        last_live = 0.0
        scored = False

        while running:
            if stop_requested():
                running = False
                break
            why = _unusable_model(drivers, labels)
            if why:
                _end_unusable(_live_status(game, match_id, total, infinite,
                                           labels, score, draws), why)
                running = False
                break
            events = pygame.event.get()
            for ev in events:
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_F1:
                        camera = "chase"
                    elif ev.key == pygame.K_F2:
                        camera = "first"
                    elif ev.key == pygame.K_F3:
                        camera = "overhead"
                    elif ev.key == pygame.K_F4:
                        camera = "free"
                    elif ev.key in (pygame.K_TAB, pygame.K_SPACE) and can_switch:
                        watch = 1 - watch
                    elif ev.key == pygame.K_h:
                        renderer.toggle_detail()
                    elif ev.key == pygame.K_p:
                        toggle_pause()
            # Keep feeding the human controller even during the countdown so
            # held keys are never lost, and re-assert window focus early on.
            if human is not None:
                human.feed(events)
            if focus_frames > 0:
                focus_frames -= 1
                if focus_frames % 30 == 0:
                    _focus_game_window()

            clock.tick(cfg.match.fps)
            paused = pause_requested()
            if game.state == "racing" and not paused:
                if intro > 0:
                    intro -= dt
                    if intro <= 0:
                        go_flash = 0.9
                else:
                    actions = {}
                    if human is not None:
                        hidx = cfg.match.human_player
                        actions[hidx] = human.get_action()
                    game.step(dt, actions)
                if go_flash > 0:
                    go_flash = max(0.0, go_flash - dt)
                now_wall = time.time()
                if now_wall - last_live > 0.5:
                    last_live = now_wall
                    write_status(_live_status(game, match_id, total,
                                              infinite, labels, score, draws))
            elif game.state == "racing":
                pass  # paused: freeze the world
            else:
                overlay_time += clock.get_time() / 1000.0
                if not scored and game.result is not None:
                    scored = True
                    if game.result.winner is not None:
                        score[game.result.winner] += 1
                    else:
                        draws += 1
                    game.session_score = list(score)
                    game.session_draws = draws

            countdown = intro if intro > 0 else -go_flash
            renderer.draw(game, camera=camera, watch=watch,
                          show_watch=can_switch, dt=dt, countdown=countdown,
                          paused=paused, detail=renderer.hud.detail,
                          debug=bool(getattr(cfg.match, "debug", False)))
            pygame.display.flip()

            if game.state == "finished" and overlay_time > 2.0:
                break

        if game.result is not None:
            analysis = analysis_mod.analyze_match(game.result.to_dict())
            store.save_match(analysis, game.result.to_dict(),
                             cfg.to_public_dict())
            _archive_match(analysis, game.result.to_dict(), cfg)
            write_status({
                "state": "running" if (infinite or index + 1 < total) else "done",
                "match": match_id,
                "laps": game.cfg.laps,
                "total": total,
                "infinite": infinite,
                "last": analysis,
                "models": labels,
                "anticheat": game.anticheat.summary(),
                "drivers": _driver_status(game),
                "score": {"p1": score[0], "p2": score[1], "draws": draws},
                "elapsed": round(time.time() - session_start, 1),
            })
            if on_match:
                on_match(analysis)
            print(analysis_mod.format_comparison_table(analysis))
        for d in drivers:
            if d is not None:
                d.close()
        index += 1

    renderer.close()
    pygame.quit()
    final = read_status()
    # "error" outranks "stopped": the console must keep showing *why* the run
    # ended, not just that the stop flag was set
    if stop_requested() and final.get("state") != "error":
        final["state"] = "stopped"
    write_status(final)
    clear_stop()
    return {"matches": index}


def run_session(cfg: GameConfig, on_match=None) -> dict:
    if cfg.match.headless or not cfg.match.render:
        return run_headless(cfg, on_match=on_match)
    return run_windowed(cfg, on_match=on_match)
