"""Cross-platform CJK font discovery.

The in-game HUD (pygame) and the PDF training report (Pillow) both need a
font that can actually draw Chinese.  Neither library ships one, and no CJK
face has the same name or path on Windows, macOS and Linux, so the search
list lives here once instead of being duplicated (and half-forgotten) in the
renderers.

``find_font_file`` is deliberately dependency-free: it only touches ``os``,
``subprocess`` and ``sys`` so it can be imported from any layer.
"""
from __future__ import annotations

import os
import subprocess
import sys

# Family names most likely to exist on a CJK-capable system, tried in
# priority order by pygame's font matcher on every platform.
CJK_FAMILIES = (
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans CN",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Droid Sans Fallback",
    "Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "PingFang SC",
    "Heiti SC", "Arial Unicode MS",
)


def _windows_files() -> list:
    windir = (os.environ.get("WINDIR") or os.environ.get("SystemRoot")
              or r"C:\Windows")
    return [os.path.join(windir, "Fonts", name) for name in (
        "msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc",
        "msjh.ttc", "msjhbd.ttc",
    )]


def _macos_files() -> list:
    return [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]


_LINUX_DIRS = (
    "/usr/share/fonts/noto-cjk",
    "/usr/share/fonts/google-noto-cjk",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/noto",
    "/usr/share/fonts/opentype/source-han-sans",
    "/usr/share/fonts/adobe-source-han-sans",
    "/usr/share/fonts/adobe-source-han-sans-otf",
    "/usr/share/fonts/truetype/wqy",
    "/usr/share/fonts/wqy-microhei",
    "/usr/share/fonts/wqy-zenhei",
    "/usr/share/fonts/wenquanyi/wqy-microhei",
    "/usr/share/fonts/wenquanyi/wqy-zenhei",
    "/usr/share/fonts/truetype/arphic",
    "/usr/share/fonts/truetype/droid",
    "/usr/share/fonts/droid",
    "/usr/share/fonts/truetype/liberation",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.local/share/fonts"),
    os.path.expanduser("~/.fonts"),
)

_LINUX_NAMES = (
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKsc-Regular.otf",
    "NotoSansSC-Regular.otf",
    "SourceHanSansCN-Regular.otf",
    "SourceHanSansSC-Regular.otf",
    "wqy-microhei.ttc",
    "wqy-zenhei.ttc",
    "DroidSansFallbackFull.ttf",
    "DroidSansFallback.ttf",
    "uming.ttc",
    "ukai.ttc",
    "Arial Unicode.ttf",
)

_BOLD_NAMES = (
    "msyhbd.ttc",
    "msjhbd.ttc",
    "NotoSansCJK-Bold.ttc",
    "NotoSansCJKsc-Bold.otf",
    "NotoSansSC-Bold.otf",
    "SourceHanSansCN-Bold.otf",
    "SourceHanSansSC-Bold.otf",
    "wqy-microhei.ttc",          # no separate bold face; regular is fine
)


def _linux_files() -> list:
    out = []
    names = tuple(dict.fromkeys(_LINUX_NAMES + _BOLD_NAMES))
    for d in _LINUX_DIRS:
        for name in names:
            out.append(os.path.join(d, name))
    return out


def _all_files() -> list:
    return _windows_files() + _macos_files() + _linux_files()


def _first_existing(paths) -> str | None:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _fontconfig_files():
    """Ask fontconfig for fonts that actually support Chinese (Linux)."""
    try:
        proc = subprocess.run(
            ["fc-list", ":lang=zh", "-f", "%{file}\n"],
            capture_output=True, text=True, timeout=5)
    except Exception:                       # fc-list missing / not Linux
        return None
    for line in (proc.stdout or "").splitlines():
        path = line.strip()
        if path and os.path.exists(path):
            return path
    return None


def find_font_file(files=None) -> str | None:
    """Return the path of the first available CJK-capable font, or None.

    Order: explicit cross-platform candidates, then fontconfig on Linux
    (``fc-list :lang=zh``), so a distro with an unusual font layout is still
    covered even when it is not one of the hard-coded paths.
    """
    path = _first_existing(files if files is not None else _all_files())
    if path:
        return path
    if sys.platform.startswith("linux"):
        return _fontconfig_files()
    return None


def find_bold_font_file() -> str | None:
    """Like :func:`find_font_file` but prefers an explicit bold face."""
    paths = []
    for base in _windows_files() + _linux_files():
        name = os.path.basename(base)
        if name in _BOLD_NAMES:
            paths.append(base)
    return _first_existing(paths) or find_font_file()
