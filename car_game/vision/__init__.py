"""Program-side vision: render the world to an image, then detect on pixels.

Nothing in this package is sent to a language model as text.  The AI drivers
render a **first-person 3D "driver's eye" frame** (the same camera the human
player sees -- see :mod:`car_game.vision.fp_render`), run a detector over it,
and act on the detections.  A legacy top-down frame is still available by
setting ``vision_view = "topdown"``.  Vision-capable LLMs receive only the raw
image (plus a tiny self-dashboard); text-only models are refused.

The package is deliberately free of any OpenGL import so it works headless.
"""
from __future__ import annotations

__all__ = ["render", "detector", "local_model", "check", "client",
           "models_scan"]
