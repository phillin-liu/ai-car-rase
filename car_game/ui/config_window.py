"""Backwards-compatible import path.

The console was split into :mod:`car_game.ui.theme`, :mod:`car_game.ui.widgets`
and :mod:`car_game.ui.console`; this shim keeps older imports working.
"""
from .console import ConfigWindow, launch

__all__ = ["ConfigWindow", "launch"]
