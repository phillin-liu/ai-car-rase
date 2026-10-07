"""Fonts and the few bits of colour the console needs.

The console deliberately uses the platform's default look -- on Windows that
is the native "windowsvista" style -- so there is **no global stylesheet and
no colour-palette override**.  Only two things carry colour, because the
colour *means* something: the status indicator and the driver dots.

Fonts are resolved in one place for a reason.  ``QFont("Microsoft YaHei UI",
n)`` does not fail when that exact name is missing -- every widget silently
falls back on its own, so two components side by side can end up in different
faces and different metrics.  :func:`ui_font` therefore resolves the family
once, from a preference list, and every component asks for a *role* rather
than picking its own point size.
"""
from __future__ import annotations

from PyQt5.QtGui import QFont, QFontDatabase

# first installed family wins; the CJK faces come first so Chinese and Latin
# are drawn from one font instead of being mixed per glyph
FONT_STACK = ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
              "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft JhengHei UI",
              "SimHei", "Segoe UI", "Arial")

FONT_MONO_STACK = ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Courier New")

# named sizes, so no component invents its own and drifts out of proportion
FONT_SIZES = {
    "small": 9,      # table cells, hints, footnote text
    "body": 10,      # the default: labels, buttons, inputs
    "subtitle": 11,
    "title": 12,     # window / section headings
    "stat": 14,      # a number in a stat tile
    "hero": 20,      # the big speed readout
}

# 4 / 8 / 12 / 16 / 24 spacing scale
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}

# readable on the default light Windows theme
STATUS_COLORS = {
    "idle": "#1a7f37",
    "running": "#9a6700",
    "paused": "#0b5cad",
    "error": "#b42318",
}

_family_cache: dict = {}


def _resolve(stack: tuple) -> str:
    """First installed family from ``stack`` (cached; needs a QApplication)."""
    key = stack[0]
    if key in _family_cache:
        return _family_cache[key]
    name = stack[0]
    try:
        installed = set(QFontDatabase().families())
    except Exception:                       # no QApplication yet
        return name
    for candidate in stack:
        if candidate in installed:
            name = candidate
            break
    _family_cache[key] = name
    return name


def ui_font(size: int = FONT_SIZES["body"], bold: bool = False) -> QFont:
    f = QFont(_resolve(FONT_STACK), int(size))
    f.setBold(bool(bold))
    return f


def role_font(role: str = "body", bold: bool = False) -> QFont:
    """:func:`ui_font` by name -- prefer this over a bare point size."""
    return ui_font(FONT_SIZES.get(role, FONT_SIZES["body"]), bold)


def mono_font(size: int = FONT_SIZES["small"]) -> QFont:
    f = QFont(_resolve(FONT_MONO_STACK), int(size))
    f.setStyleHint(QFont.Monospace)
    return f
