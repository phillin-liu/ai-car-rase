"""Terminal command mode.

When the game is launched from a terminal (stdin/stdout are TTYs) we enter
this text-only shell instead of the graphical config window.  Head-less
operation is aimed at server training, so every output is plain text and
the interactive shell provides the commands needed to drive it.
"""
from __future__ import annotations

import os
import shlex
import sys

from .config import (DEFAULT_MODELS, GameConfig, ensure_dirs, load_settings,
                     save_settings)
from .providers import is_known, normalize_provider, provider_ids
from .runner import run_headless, run_session
from .store import MatchStore


HELP = f"""可用命令:
  run  [key=value ...]    运行对局 (无窗口)
       可选: matches= 局数(0=无限)  laps= 圈数
             p1= / p2= 提供方({"/".join(provider_ids())})
             ollama 为本机 Ollama（无需密钥），
             其余为云端视觉模型，需要对应 API 密钥（只填密钥即可）
             model1= / model2= 模型名
             difficulty= 难度  fps= 帧率  items=0/1  debug=0/1
  set  key=value          修改并保存设置 (如 set laps=3)
  show                    显示当前设置
  export-report           导出当前会话训练报告 (PDF)
  anticheat build|check   编译/检查 Rust 反作弊模块
  quit / exit / q         退出
"""


def _print_config(cfg: GameConfig) -> None:
    m = cfg.match
    print("── 当前设置 ──────────────────────────────")
    print(f"  模式: {m.mode}  人类车辆: P{m.human_player + 1}")
    print(f"  圈数: {m.laps}  局数: {m.matches if m.matches > 0 else '无限'}")
    print(f"  难度: {m.difficulty}")
    print(f"  道具: {'开' if m.items_enabled else '关'}  无窗口: {'是' if m.headless else '否'}"
          f"  调试: {'开' if getattr(m, 'debug', False) else '关'}")
    print(f"  P1: {cfg.p1.provider}:{cfg.p1.model}  ({cfg.p1.display_name})")
    print(f"  P2: {cfg.p2.provider}:{cfg.p2.model}  ({cfg.p2.display_name})")


def _parse_kv(args: list) -> dict:
    out = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip().lower()] = v.strip()
        else:
            out[a.strip().lower()] = True
    return out


def _apply_kv(cfg: GameConfig, kv: dict) -> None:
    m = cfg.match
    if "matches" in kv:
        m.matches = int(kv["matches"])
    if "laps" in kv:
        m.laps = int(kv["laps"])
    if "difficulty" in kv:
        m.difficulty = float(kv["difficulty"])
    if "fps" in kv:
        m.fps = int(kv["fps"])
    if "items" in kv:
        m.items_enabled = str(kv["items"]) not in ("0", "false", "no")
    if "debug" in kv:
        m.debug = str(kv["debug"]) not in ("0", "false", "no", "off")
    if "mode" in kv:
        m.mode = kv["mode"] if kv["mode"] in ("ai_vs_ai", "human_vs_ai") else "ai_vs_ai"
    for side_key, side in (("p1", cfg.p1), ("p2", cfg.p2)):
        if side_key in kv:
            requested = str(kv[side_key]).strip().lower()
            side.provider = normalize_provider(requested)
            if not is_known(requested):
                print(f"[!] 未知提供方 {requested!r}，已改用 "
                      f"{side.provider}。可选：{'/'.join(provider_ids())}")
            side.model = DEFAULT_MODELS.get(side.provider, side.model)
            side.display_name = f"{side.provider}:{side.model}"
    if "model1" in kv:
        cfg.p1.model = str(kv["model1"])
    if "model2" in kv:
        cfg.p2.model = str(kv["model2"])


def _cmd_run(cfg: GameConfig, kv: dict) -> None:
    _apply_kv(cfg, kv)
    # Headless sessions are for training: never allow human controls there.
    if cfg.match.mode == "human_vs_ai":
        print("[!] 无窗口模式不支持 人类对AI，已切换为 AI对AI。")
        cfg.match.mode = "ai_vs_ai"
    cfg.match.headless = True
    cfg.match.render = False
    save_settings(cfg)
    print(f"开始运行: {cfg.match.matches if cfg.match.matches > 0 else '无限'} 局")
    try:
        run_session(cfg)
    except KeyboardInterrupt:
        from .runner import request_stop
        request_stop()
        print("\n已停止")


def _cmd_export_report() -> None:
    store = MatchStore()
    if store.count() <= 0:
        print("[!] 当前会话还没有对局数据，无法导出训练报告")
        return
    out = os.path.join("runtime", "training_report.pdf")
    os.makedirs("runtime", exist_ok=True)
    try:
        store.export_report(out)
    except ImportError:
        print("[!] 生成 PDF 需要 Pillow")
        return
    print(f"训练报告已导出到 {out}")


def _cmd_anticheat(action: str) -> None:
    from . import anticheat
    if action == "build":
        ok, msg = anticheat.build_rust()
        print(f"构建{'成功' if ok else '失败'}: {msg}")
        return
    print(f"Rust 源码: {anticheat.RUST_SRC}")
    if not os.path.exists(anticheat.RUST_SRC):
        print("状态: 找不到源码，将使用 Python 校验")
    elif not anticheat.cargo_available():
        print("状态: 未安装 cargo，将使用 Python 校验")
    else:
        print("状态: 可用（cargo run 从源码现场编译运行）")
        print("编译产物缓存在 anticheat/target/，首次运行较慢。")


def run_cli(cfg: GameConfig) -> int:
    ensure_dirs()
    print("=" * 56)
    print("xinke studio")
    print("(c) xinke studio保留所有权利。")
    print(" AI Car Race · 终端指令模式")
    print(" 输入 help 查看命令，quit 退出")
    print("=" * 56)
    _print_config(cfg)

    while True:
        try:
            line = input("\ncar> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            return 0
        if not line:
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            print("[!] 引号不匹配")
            continue
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            print("再见")
            return 0
        if cmd in ("help", "h", "?"):
            print(HELP)
        elif cmd == "run":
            _cmd_run(cfg, _parse_kv(parts[1:]))
        elif cmd == "set":
            _apply_kv(cfg, _parse_kv(parts[1:]))
            save_settings(cfg)
            print("已保存")
            _print_config(cfg)
        elif cmd == "show":
            _print_config(cfg)
        elif cmd in ("export-report", "report", "pdf"):
            _cmd_export_report()
        elif cmd == "anticheat":
            _cmd_anticheat(parts[1] if len(parts) > 1 else "check")
        else:
            print(f"[!] 未知命令: {cmd}，输入 help 查看帮助")
    return 0
