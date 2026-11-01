"""Object detection over the rendered vision frame.

Two backends are available:

* :class:`BuiltinDetector` -- deterministic colour segmentation with
  connected components.  It needs no extra dependency and no weight file, so
  it always works; this is the default.
* :class:`YoloDetector` -- an optional learned model loaded from an imported
  ``.pt`` / ``.pth`` / ``.onnx`` file via ``ultralytics``.  It falls back to
  the built-in detector whenever the library or the file is unavailable.

Both map what they find on the pixels back into world coordinates through the
frame transform (affine for the top-down view, perspective for the 3D
first-person view), so the driving layer never touches the track's
ground-truth obstacle list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .render import OBSTACLE_PALETTE, RIVAL_CAR, to_numpy


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple                 # (x0, y0, x1, y1) in pixels
    world: tuple                # (x, z) race-world ground coordinates
    forward_m: float            # metres ahead of the own car (negative = behind)
    lateral_m: float            # metres left of the own car (+ = left)
    radius: float               # metres

    def as_obstacle_tuple(self):
        return (float(self.world[0]), float(self.world[1]), float(self.radius))


@dataclass
class SceneAnalysis:
    obstacles: list = field(default_factory=list)
    rival: Detection | None = None
    detector: str = "builtin"

    def ahead(self, max_forward: float = 50.0, min_forward: float = -6.0):
        return [d for d in self.obstacles
                if min_forward <= d.forward_m <= max_forward]

    def summary(self) -> dict:
        return {
            "detector": self.detector,
            "obstacles": len(self.obstacles),
            "rival": self.rival is not None,
        }


# ---------------------------------------------------------------------------
class VisionDetector:
    name = "base"

    def detect(self, image, transform, include=("obstacle", "rival")):
        raise NotImplementedError

    def analyze(self, image, transform) -> SceneAnalysis:
        dets = self.detect(image, transform)
        return SceneAnalysis(
            obstacles=[d for d in dets if d.label != "rival"],
            rival=next((d for d in dets if d.label == "rival"), None),
            detector=self.name,
        )


# ---------------------------------------------------------------------------
class BuiltinDetector(VisionDetector):
    """Colour segmentation + connected components on the rendered frame."""

    name = "builtin"

    def __init__(self, tol: int = 26, min_area: int = 4):
        self.tol = int(tol)
        self.min_area = int(min_area)

    def detect(self, image, transform, include=("obstacle", "rival")):
        arr = to_numpy(image)
        out = []
        if "obstacle" in include:
            for label, rgb in OBSTACLE_PALETTE.items():
                out.extend(self._components(arr, rgb, label, transform))
        if "rival" in include:
            out.extend(self._components(arr, RIVAL_CAR, "rival", transform))
        return out

    # ------------------------------------------------------------------
    def _components(self, arr, rgb, label, transform):
        diff = np.abs(arr - np.array(rgb, dtype=np.int16)).max(axis=2)
        mask = diff <= self.tol
        if not mask.any():
            return []
        try:
            from scipy import ndimage
        except Exception:                       # pragma: no cover - scipy absent
            return _fallback_components(mask, diff, label, transform, self.min_area)
        labels, n = ndimage.label(mask)
        if n == 0:
            return []
        ys, xs = np.nonzero(mask)
        lab = labels[ys, xs]
        dists = diff[ys, xs]
        out = []
        for k in range(1, n + 1):
            sel = lab == k
            area = int(sel.sum())
            if area < self.min_area:
                continue
            cy = float(ys[sel].mean())
            cx = float(xs[sel].mean())
            px = xs[sel]
            py = ys[sel]
            bbox = (float(px.min()), float(py.min()),
                    float(px.max() + 1), float(py.max() + 1))
            # Perspective frames put the object's ground contact at the
            # bottom of its box, so anchor there; top-down frames use the
            # centroid (the disc centre).
            ground = getattr(transform, "ground_anchor", False)
            ax = 0.5 * (bbox[0] + bbox[2]) if ground else cx
            ay = bbox[3] if ground else cy
            loc = transform.pixel_to_local(ax, ay)
            if loc is None:
                continue
            fwd, lat = loc
            wx, wz = transform.local_to_world(fwd, lat)
            if ground:
                ppm = transform.pixels_per_metre(fwd)
                radius = max(0.15, (bbox[2] - bbox[0]) / (2.0 * max(ppm, 1e-6)))
            else:
                radius = math.sqrt(area / math.pi) / max(transform.scale, 1e-6)
            conf = max(0.5, 1.0 - float(dists[sel].mean()) / 64.0)
            out.append(Detection(label, round(conf, 3), bbox, (wx, wz),
                                 fwd, lat, radius))
        return out


def _fallback_components(mask, diff, label, transform, min_area):
    """Tiny flood-fill fallback if scipy is unavailable (never expected)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) < min_area:
                continue
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            cy, cx = sum(ys) / len(ys), sum(xs) / len(xs)
            # exclusive right/bottom edges, exactly like the scipy path below:
            # the perspective anchor reads bbox[3] as the ground contact, and
            # a pixel index instead of an edge puts it 1 px high -- which at
            # 40 m is ~5 m of error in the reported obstacle distance
            bbox = (float(min(xs)), float(min(ys)),
                    float(max(xs) + 1), float(max(ys) + 1))
            ground = getattr(transform, "ground_anchor", False)
            ax = 0.5 * (bbox[0] + bbox[2]) if ground else cx
            ay = bbox[3] if ground else cy
            loc = transform.pixel_to_local(ax, ay)
            if loc is None:
                continue
            fwd, lat = loc
            wx, wz = transform.local_to_world(fwd, lat)
            if ground:
                ppm = transform.pixels_per_metre(fwd)
                radius = max(0.15, (bbox[2] - bbox[0]) / (2.0 * max(ppm, 1e-6)))
            else:
                radius = math.sqrt(len(pts) / math.pi) / max(transform.scale, 1e-6)
            out.append(Detection(label, 0.6, bbox, (wx, wz), fwd, lat, radius))
    return out


# ---------------------------------------------------------------------------
class YoloDetector(VisionDetector):
    """Optional learned detector over an imported ``.pt`` / ``.pth`` model.

    Loaded lazily on the first frame through ``ultralytics``; if the library
    is missing or the file cannot be loaded the detector silently falls back
    to :class:`BuiltinDetector`, so a bad weights path can never stop the
    race.  Class names are mapped onto the game's own labels.
    """

    name = "yolo"

    # YOLO class name (lowercased) -> game label
    _LABEL_MAP = {
        "cone": "cone", "traffic cone": "cone", "锥": "cone",
        "锥桶": "cone", "traffic_cone": "cone",
        "barrel": "barrel", "oil barrel": "barrel", "drum": "barrel",
        "油桶": "barrel", "桶": "barrel",
        "crate": "crate", "box": "crate", "wooden box": "crate",
        "木箱": "crate", "箱子": "crate",
        "car": "rival", "rival": "rival", "vehicle": "rival",
        "车": "rival", "对手": "rival",
    }

    def __init__(self, weights: str, conf: float = 0.25):
        self.weights = str(weights)
        self.conf = float(conf)
        self._model = None
        self._failed = False
        self._names: dict = {}
        self._fallback = BuiltinDetector()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, dict):
            self._names = {int(k): str(v) for k, v in names.items()}
        else:
            self._names = {i: str(v) for i, v in enumerate(names)}

    def _label(self, cls_id: int) -> str:
        raw = self._names.get(int(cls_id), str(cls_id))
        return self._LABEL_MAP.get(raw.lower(), "obstacle")

    # ------------------------------------------------------------------
    def detect(self, image, transform, include=("obstacle", "rival")):
        if self._model is None and not self._failed:
            try:
                self._load()
            except Exception:  # noqa: BLE001 - optional dependency / bad file
                self._failed = True
                self.name = "builtin"
        if self._failed or self._model is None:
            return self._fallback.detect(image, transform, include)

        arr = to_numpy(image).astype(np.uint8)
        out = []
        try:
            results = self._model.predict(arr, conf=self.conf, verbose=False)
        except Exception:  # noqa: BLE001
            return self._fallback.detect(image, transform, include)
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            for (x0, y0, x1, y1), cf, cl in zip(xyxy, confs, clss):
                label = self._label(int(cl))
                if label == "rival" and "rival" not in include:
                    continue
                if label != "rival" and "obstacle" not in include:
                    continue
                det = self._to_detection(label, float(x0), float(y0),
                                         float(x1), float(y1), float(cf),
                                         transform)
                if det is not None:
                    out.append(det)
        return out

    @staticmethod
    def _to_detection(label, x0, y0, x1, y1, conf, transform):
        ground = getattr(transform, "ground_anchor", False)
        ax = 0.5 * (x0 + x1) if ground else 0.5 * (x0 + x1)
        ay = y1 if ground else 0.5 * (y0 + y1)
        loc = transform.pixel_to_local(ax, ay)
        if loc is None:
            return None
        fwd, lat = loc
        wx, wz = transform.local_to_world(fwd, lat)
        if ground:
            ppm = transform.pixels_per_metre(fwd)
            radius = max(0.2, (x1 - x0) / (2.0 * max(ppm, 1e-6)))
        else:
            radius = max(0.2, 0.5 * (x1 - x0) / max(transform.scale, 1e-6))
        return Detection(label, round(conf, 3), (x0, y0, x1, y1),
                         (wx, wz), fwd, lat, radius)


# ---------------------------------------------------------------------------
def build_detector(cfg=None) -> VisionDetector:
    """Pick the companion detector for a side.

    When the user imported a ``.pt`` / ``.pth`` / ``.onnx`` weights file it is
    used through :class:`YoloDetector`; otherwise (and whenever the file
    cannot be loaded) the built-in colour detector runs.
    """
    weights = str(getattr(cfg, "vision_weights", "") or "").strip()
    if weights:
        return YoloDetector(weights)
    return BuiltinDetector()
