"""Render a driver's frame, then detect on it.

``LocalVisionModel`` is the program's own eyes: one render step plus the
built-in detector, behind a single convenient ``analyze`` call.  Every driver
uses it -- a remote model is shown the shaded frame it returns, and the
detections it produces are what keep the car off the obstacles without ever
reading the track's ground-truth list.

The name is historical: it dates from when a local side drove from these
detections.  That side is gone; the renderer and detector are not.
"""
from __future__ import annotations

from ..palette import rgb255
from .detector import SceneAnalysis, VisionDetector, build_detector
from .fp_render import render_first_person
from .render import OWN_CAR, RIVAL_CAR, CarView, render_topdown


def car_views(cars, self_index: int = 0):
    """Build :class:`CarView`s from physics ``Car`` objects (or dicts)."""
    views = []
    for i, car in enumerate(cars):
        if isinstance(car, dict):
            x, z = car["pos"][0], car["pos"][1]
            heading = car.get("heading", 0.0)
            color = car.get("color")
        else:
            x, z = float(car.pos[0]), float(car.pos[1])
            heading = float(car.heading)
            color = getattr(car, "color", None)
        views.append(CarView(float(x), float(z), float(heading),
                             OWN_CAR if i == self_index else RIVAL_CAR,
                             is_self=(i == self_index),
                             real_color=rgb255(color) if color else None))
    return views


class LocalVisionModel:
    """Render + detect, with no network and no text."""

    def __init__(self, cfg=None, detector: VisionDetector | None = None):
        self.cfg = cfg
        self.detector = detector if detector is not None else build_detector(cfg)
        self.image_size = int(getattr(cfg, "vision_image_size", 256) or 256)
        self.ahead_m = float(getattr(cfg, "vision_ahead_m", 80.0) or 80.0)
        self.view = str(getattr(cfg, "vision_view", "first") or "first").lower()
        self.fov = float(getattr(cfg, "vision_fov", 58.0) or 58.0)
        self.supersample = max(1, int(getattr(cfg, "vision_supersample", 2) or 1))

    # ------------------------------------------------------------------
    def render(self, track, views, hazards=(), self_index: int = 0, *,
               shaded: bool = False):
        """Render one driver's viewpoint.

        ``vision_view == "first"`` (the default) renders the same 3D
        first-person camera the human sees; ``"topdown"`` keeps the legacy
        car-centric map view.  ``shaded`` picks the realistic channel (what the
        model is shown) rather than the flat one the detector reads; it only
        applies to the first-person view.
        """
        if self.view not in ("topdown", "top", "map", "2d"):
            return render_first_person(
                track, views, hazards, self_index,
                ahead_m=self.ahead_m, size=(self.image_size, self.image_size),
                fov=self.fov, shaded=shaded,
                supersample=self.supersample if shaded else 1)
        return render_topdown(
            track, views, hazards, self_index,
            ahead_m=self.ahead_m, size=(self.image_size, self.image_size))

    def detect(self, image, transform) -> SceneAnalysis:
        return self.detector.analyze(image, transform)

    def analyze(self, track, views, hazards=(), self_index: int = 0):
        """Return ``(PIL.Image, SceneAnalysis)`` for one driver's viewpoint.

        Two frames are rendered from the same camera: the realistic one the
        model is shown, and the flat high-contrast one the built-in detector
        segments on.  Feeding the detector the shaded frame instead would break
        it -- its colour thresholds assume these exact flat values.
        """
        img, tf = self.render(track, views, hazards, self_index, shaded=True)
        flat, _ = self.render(track, views, hazards, self_index, shaded=False)
        return img, self.detect(flat, tf)

    def describe(self) -> str:
        return getattr(self.detector, "name", "builtin")


def ensure_local_vision_model(cfg=None) -> LocalVisionModel:
    """Always succeeds: built-in detector, plus optional imported weights."""
    return LocalVisionModel(cfg)
