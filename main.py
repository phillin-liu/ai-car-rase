"""Entry point for the AI Car Race project.

Usage
-----
  python main.py                    # 打开 PyQt5 配置窗口 (默认)
  python main.py --shell            # 进入文本指令模式
  python main.py --run --headless --matches 5 --p1 ollama --p2 ollama
  python main.py --export-report
"""
from __future__ import annotations

import argparse
import os
import sys


def _utf8():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    from car_game.providers import provider_ids
    ap = argparse.ArgumentParser(description='AI 3D 双人赛车')
    ap.add_argument('--shell', action='store_true',
                    help='进入文本指令模式 (默认打开图形配置窗口)')
    ap.add_argument('--run', action='store_true', help='直接运行对局(不打开配置窗口)')
    ap.add_argument('--config', help='会话配置 JSON 文件')
    ap.add_argument('--headless', action='store_true', help='无窗口运行')
    ap.add_argument('--windowed', action='store_true', help='强制窗口运行')
    ap.add_argument('--matches', type=int, help='对局局数(0 表示无限)')
    ap.add_argument('--laps', type=int, help='每局圈数')
    ap.add_argument('--p1', help=f'P1 提供方: {"/".join(provider_ids())}')
    ap.add_argument('--p2', help='P2 提供方 (同上)')
    ap.add_argument('--model1', help='P1 模型名称')
    ap.add_argument('--model2', help='P2 模型名称')
    ap.add_argument('--mode', choices=['ai_vs_ai', 'human_vs_ai'], help='对战模式')
    ap.add_argument('--difficulty', type=float, help='AI 难度')
    ap.add_argument('--debug', action='store_true', help='显示 AI 决策 / 遥测调试叠加层')
    ap.add_argument('--no-msaa', action='store_true', help='关闭 4x 多重采样抗锯齿')
    ap.add_argument('--export-report', action='store_true',
                    help='导出当前会话的训练报告 (PDF) 后退出')
    ap.add_argument('--anticheat', nargs='?', const='check', choices=['check', 'build'],
                    help='检查或构建 Rust 反作弊模块')
    ap.add_argument('--lang', choices=['zh', 'en'],
                    help='控制台界面语言 / console UI language (zh, en)')
    return ap


def apply_overrides(cfg, args) -> None:
    from car_game.config import DEFAULT_MODELS
    m = cfg.match
    if args.headless:
        m.headless = True
        m.render = False
    if args.windowed:
        m.headless = False
        m.render = True
    if args.matches is not None:
        m.matches = args.matches
    if args.laps is not None:
        m.laps = args.laps
    if args.mode:
        m.mode = args.mode
    if args.difficulty is not None:
        m.difficulty = args.difficulty
    if args.debug:
        m.debug = True
    if args.no_msaa:
        m.msaa = False
    for arg, side, model_arg, tag in ((args.p1, cfg.p1, args.model1, 'P1'),
                                      (args.p2, cfg.p2, args.model2, 'P2')):
        if not arg:
            continue
        from car_game.providers import is_known, normalize_provider
        provider = normalize_provider(arg)
        if not is_known(arg):
            print(f'[!] 未知提供方 {arg!r}，{tag} 已改用 {provider}')
        side.provider = provider
        side.model = model_arg or DEFAULT_MODELS.get(provider, side.model)
        side.display_name = f'{provider}:{side.model}'


def _handle_export_commands(args) -> bool:
    """Run one-shot export/anti-cheat commands and return True if handled."""
    from car_game.config import RUNTIME_DIR, ensure_dirs
    from car_game.store import MatchStore
    ensure_dirs()
    store = MatchStore()

    if args.export_report:
        if store.count() <= 0:
            print('[!] 当前会话还没有对局数据，无法导出训练报告')
            return True
        out = os.path.join(RUNTIME_DIR, 'training_report.pdf')
        try:
            store.export_report(out)
        except ImportError:
            print('[!] 生成 PDF 需要 Pillow')
            return True
        print(f'训练报告已导出到 {out}')
        return True
    if args.anticheat:
        from car_game import anticheat
        if args.anticheat == 'build':
            ok, msg = anticheat.build_rust()
            print(f'Rust 反作弊构建{"成功" if ok else "失败"}: {msg}')
        else:
            print(f'Rust 反作弊二进制: {"可用" if anticheat.rust_binary_available() else "未编译"}'
                  '  (用 --anticheat build 编译)')
        return True
    return False


def main(argv=None) -> int:
    _utf8()
    args = build_parser().parse_args(argv)

    from car_game.config import GameConfig, ensure_dirs, load_settings
    ensure_dirs()

    # 一次性命令 (--export-report / --anticheat) 处理完即退出
    if _handle_export_commands(args):
        return 0

    if args.run:
        if args.config:
            # 读取 --config 指定的会话配置 JSON，
            # 并补上 config/keys.json 里已保存的 API 密钥
            from car_game.config import apply_saved_keys
            cfg = apply_saved_keys(GameConfig.load(args.config))
        else:
            cfg = load_settings()
        apply_overrides(cfg, args)

        if cfg.match.mode == 'human_vs_ai' and (cfg.match.headless or not cfg.match.render):
            print('[!] 无窗口模式不支持 人类对AI，已切换为 AI对AI。')
            cfg.match.mode = 'ai_vs_ai'
        from car_game.runner import run_session
        try:
            result = run_session(cfg)
        except KeyboardInterrupt:
            from car_game.runner import request_stop
            request_stop()
            print('\n已停止')
            return 0
        print(f'\n会话结束: 共 {result.get("matches", 0)} 局')
        return 0

    # 语言覆盖 (--lang)，需要在启动界面之前生效
    if args.lang:
        # 覆盖设置文件里保存的语言
        # (zh / en)
        from car_game.ui.i18n import set_language
        set_language(args.lang)

    if args.shell:
        cfg = load_settings()
        apply_overrides(cfg, args)
        from car_game.cli import run_cli
        return run_cli(cfg)

    # 默认：打开 PyQt5 图形配置窗口
    try:
        from car_game.ui.console import launch
    except Exception as exc:
        print(f'无法启动配置窗口: {exc}\n`，或使用 --shell '
              f'进入文本模式、'
              f'--run 命令行运行。')
        return 1
    launch()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
