"""Software first-person 3D renderer for the AI's "camera".

The human player watches the race through the OpenGL renderer in first-person
mode (see :class:`car_game.render.camera.Camera`, ``mode == "first"``).  This
module renders the *same viewpoint* -- same eye/target rig, same projection
orientation -- but with a pure-Pillow painter's-algorithm rasteriser, so it
also works in a head-less server process with no GL context at all.

Two channels
------------

``shaded=False`` (the **default**) is the *detection* channel: flat, unlit,
high-contrast colours that
:class:`car_game.vision.detector.BuiltinDetector` segments on.  Its output is
frozen -- changing a pixel of it changes what the AI perceives -- so
``tests/test_fp_render.py`` locks it down.

``shaded=True`` is the *presentation* channel: the same geometry, lit with the
same sun / hemisphere-ambient / distance-fog terms the GL shader uses and drawn
with the shared palette from :mod:`car_game.palette`, so the frame the model is
shown looks like what the player sees in the 3D window.  It also adds the
scenery the flat channel omits for speed: guardrails,
the start/finish gantry, window bands, tree trunks, volumetric obstacles,
fuller car bodies and contact shadows.

:meth:`car_game.vision.local_model.LocalVisionModel.analyze` renders both -- the
model gets the shaded frame, the detector keeps reading the flat one.

The returned :class:`FirstPersonTransform` mirrors the perspective projection
of the 3D view and can unproject a pixel onto the ground plane, which lets
:mod:`car_game.vision.detector` keep turning coloured pixels back into world
geometry exactly like it does for the old top-down frame.

NOTE: this module must never import ``car_game.render`` -- that package pulls
in OpenGL at import time.  Only ``car_game.vision.render`` (pure Pillow) and
``car_game.palette`` (standard library only) are imported.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .. import palette as _pal
from .render import OBSTACLE_PALETTE, CarView

# --- detector (flat) palette ------------------------------------------------
# Frozen on purpose.  These are *not* the scene colours: they are the
# high-contrast proxies BuiltinDetector segments on, and the whole detection
# pipeline is calibrated against these exact bytes.  The realistic colours the
# presentation channel uses come from car_game.palette instead.
SKY_TOP = (66, 117, 199)
SKY_HORIZON = (194, 214, 235)
GROUND_FAR = (66, 77, 66)
GROUND_NEAR = (46, 92, 52)
HILL = (92, 116, 96)
ASPHALT = (58, 58, 66)
LINE = (225, 225, 215)
KERB_A = (199, 41, 36)
KERB_B = (228, 228, 222)
BUILDING = (120, 124, 134)
TREE = (30, 84, 40)
TRUNK = (77, 54, 33)
HAZARD = (54, 44, 34)

# --- presentation (shaded) palette ------------------------------------------
# The very values the GL renderer feeds its shaders, converted once for Pillow.
S_SKY_TOP = _pal.rgb255(_pal.SKY_TOP)
S_SKY_HORIZON = _pal.rgb255(_pal.SKY_HORIZON)
S_GROUND = _pal.rgb255(_pal.GROUND)
S_HILL = _pal.rgb255(_pal.HILL)
S_ROAD = _pal.rgb255(_pal.ROAD)
S_LINE = _pal.rgb255(_pal.LINE)
S_KERB_A = _pal.rgb255(_pal.CURB_A)
S_KERB_B = _pal.rgb255(_pal.CURB_B)
S_RAIL = _pal.rgb255(_pal.RAIL)
S_POST = _pal.rgb255(_pal.POST)
S_WINDOW = _pal.rgb255(_pal.WINDOW)
S_TRUNK = _pal.rgb255(_pal.TRUNK)
S_HAZARD = _pal.rgb255(_pal.OBSTACLE_COLORS["crate"])
S_FOLIAGE = [_pal.rgb255(c) for c in _pal.FOLIAGE]
S_BUILDINGS = [_pal.rgb255(c) for c in _pal.BUILDINGS]
S_OBSTACLE = {k: _pal.rgb255(v) for k, v in _pal.OBSTACLE_COLORS.items()}
S_GANTRY = (216, 36, 31)
S_GANTRY_BEAM = (26, 31, 41)

_LIGHT = (0.0, 0.0, 0.0)
_nl = math.sqrt(sum(c * c for c in _pal.LIGHT_DIR))
_LIGHT = tuple(c / _nl for c in _pal.LIGHT_DIR)
_FOG_A, _FOG_B = _pal.FOG_RANGE

_EPS = 1e-6


def _mix(a, b, t: float):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def _normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _jit(seed: float, amount: float) -> float:
    """Deterministic brightness jitter in [-amount, amount] (as scene.py)."""
    h = math.sin(seed * 12.9898) * 43758.5453
    return (h - math.floor(h) - 0.5) * 2.0 * amount


def _tint(color, d: float):
    """Brighten/darken an 8-bit colour by the fraction ``d``."""
    return (max(0, min(255, int(color[0] * (1.0 + d)))),
            max(0, min(255, int(color[1] * (1.0 + d)))),
            max(0, min(255, int(color[2] * (1.0 + d)))))


# ---------------------------------------------------------------------------
class FirstPersonTransform:
    """Perspective world <-> pixel mapping for one first-person frame.

    The camera basis matches ``mathutil.look_at`` (row 0 = ``cross(forward,
    up)``), so screen-left/right agrees with the human's 3D view.  Ground
    pixels can be ray-cast back to the world X/Z plane, which is what the
    detector consumes.
    """

    ground_anchor = True          # bbox -> use the bottom-centre as the anchor

    def __init__(self, eye, forward, right, up, focal, cx, cz, heading,
                 ground_y, width, height, near=0.4, terrain=None):
        self.eye = np.asarray(eye, dtype=np.float64).reshape(3)
        self.forward = _normalize(np.asarray(forward, dtype=np.float64).reshape(3))
        self.right = _normalize(np.asarray(right, dtype=np.float64).reshape(3))
        self.up = np.asarray(up, dtype=np.float64).reshape(3)
        self.focal = float(focal)
        self.cx = float(cx)
        self.cz = float(cz)
        self.heading = float(heading)
        self.cos = math.cos(self.heading)
        self.sin = math.sin(self.heading)
        self.ground_y = float(ground_y)
        self.width = int(width)
        self.height = int(height)
        self.near = float(near)
        self.terrain = terrain      # optional (x, z) -> ground height

    # -- nominal scale (kept for interface parity with FrameTransform) -----
    @property
    def scale(self) -> float:
        """Pixels per metre for an object sitting 1 m in front of the camera."""
        return self.focal

    def pixels_per_metre(self, fwd: float) -> float:
        """Pixels per metre at ``fwd`` metres ahead (perspective foreshortening)."""
        return self.focal / max(float(fwd), 0.25)

    # -- projections -------------------------------------------------------
    def to_camera(self, p):
        d = np.asarray(p, dtype=np.float64).reshape(3) - self.eye
        return (float(np.dot(self.right, d)),
                float(np.dot(self.up, d)),
                float(np.dot(self.forward, d)))

    def camera_to_pixel(self, c):
        inv = self.focal / max(c[2], _EPS)
        return (self.width * 0.5 + c[0] * inv,
                self.height * 0.5 - c[1] * inv)

    def project(self, p):
        c = self.to_camera(p)
        if c[2] <= self.near:
            return None
        px, py = self.camera_to_pixel(c)
        return (px, py, c[2])

    def horizon_y(self) -> float:
        hx, hz = float(self.forward[0]), float(self.forward[2])
        n = math.hypot(hx, hz)
        if n < _EPS:
            return self.height * 0.5
        p = self.eye + np.array([hx / n, 0.0, hz / n]) * 1.0e6
        return self.project(p)[1]

    # -- interface used by render_topdown callers / debug tools -----------
    def world_to_pixel(self, x: float, z: float, y: float | None = None):
        p = (float(x), self.ground_y if y is None else float(y), float(z))
        res = self.project(p)
        return (res[0], res[1]) if res is not None else (float("nan"),
                                                         float("nan"))

    def pixel_ray(self, px: float, py: float):
        """Unit world-space ray through the given pixel (from the eye)."""
        dx = (float(px) - self.width * 0.5) / self.focal
        dy = -(float(py) - self.height * 0.5) / self.focal
        d = self.right * dx + self.up * dy + self.forward
        return _normalize(d)

    def pixel_to_ground(self, px: float, py: float):
        """Hit the pixel ray against the ground; ``(x, z)`` or ``None``.

        With a flat reference plane (the default; ``terrain is None``) this is
        an exact plane intersection.  If a terrain sampler is supplied the ray
        is marched against it instead.
        """
        d = self.pixel_ray(px, py)
        if d[1] > -1e-4:
            return None
        ox, oy, oz = float(self.eye[0]), float(self.eye[1]), float(self.eye[2])
        if self.terrain is None:
            t = (self.ground_y - oy) / d[1]
            if t <= 0.0:
                return None
            return (ox + d[0] * t, oz + d[2] * t)
        # march the ray until it drops below the terrain, then bisect.  A
        # single flat plane cannot describe a track that climbs and falls.
        t = 1.0
        step = 4.0
        prev_t, prev_f = None, None
        while t <= 300.0:
            x = ox + d[0] * t
            y = oy + d[1] * t
            z = oz + d[2] * t
            f = y - float(self.terrain(x, z))
            if f <= 0.0:
                if prev_f is None or prev_f <= 0.0:
                    return (float(x), float(z))
                lo, hi = prev_t, t
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    xm = ox + d[0] * mid
                    ym = oy + d[1] * mid
                    zm = oz + d[2] * mid
                    if ym - float(self.terrain(xm, zm)) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                return (float(ox + d[0] * hi), float(oz + d[2] * hi))
            prev_t, prev_f = t, f
            t += step
        return None

    def pixel_to_local(self, px: float, py: float):
        """Pixel -> (forward, lateral) metres relative to the own car."""
        hit = self.pixel_to_ground(px, py)
        if hit is None:
            return None
        dx = hit[0] - self.cx
        dz = hit[1] - self.cz
        return (dx * self.cos + dz * self.sin,
                -dx * self.sin + dz * self.cos)

    def local_to_world(self, fwd: float, left: float):
        x = self.cx + fwd * self.cos - left * self.sin
        z = self.cz + fwd * self.sin + left * self.cos
        return (float(x), float(z))


# ---------------------------------------------------------------------------
def render_first_person(track, cars, hazards=(), self_index: int = 0, *,
                        ahead_m: float = 80.0, size=(256, 256),
                        fov: float = 58.0,
                        behind_ratio: float = 0.16,
                        shaded: bool = False,
                        supersample: int = 1):
    """Render one first-person frame.  Returns ``(PIL.Image, transform)``.

    ``cars`` is a list of :class:`~car_game.vision.render.CarView`; the car at
    ``self_index`` is the driver (its camera defines the view).

    ``shaded`` selects the presentation channel (lit + fogged, with the full
    scenery) rather than the flat detection channel.  ``supersample`` renders
    the shaded channel at ``N x`` and downsamples it for anti-aliasing; it is
    ignored for the flat channel, which must keep hard colour edges.
    """
    w, h = int(size[0]), int(size[1])
    ahead_m = max(20.0, float(ahead_m))
    behind_m = ahead_m * max(0.02, float(behind_ratio))

    own = cars[self_index] if 0 <= self_index < len(cars) else None
    cx = float(own.x) if own is not None else 0.0
    cz = float(own.z) if own is not None else 0.0
    heading = float(own.heading) if own is not None else 0.0

    ground_y = 0.0
    base_arc = 0.0
    if track is not None:
        try:
            i0, t0, _, _, _, _ = track.nearest((cx, cz), None)
            base_arc = track.arc_at(i0, t0)
            ground_y = float(track.height_at_index(i0, t0))
        except Exception:
            pass

    # --- the exact first-person rig used by render/camera.py -------------
    fwd = np.array([math.cos(heading), 0.0, math.sin(heading)], dtype=np.float64)
    pos = np.array([cx, ground_y, cz], dtype=np.float64)
    eye = pos + fwd * 0.35 + np.array([0.0, 1.62, 0.0])
    target = pos + fwd * 14.0 + np.array([0.0, 0.95, 0.0])
    view = _normalize(target - eye)
    right = _normalize(np.cross(view, np.array([0.0, 1.0, 0.0])))
    up = np.cross(right, view)
    fov = max(25.0, min(90.0, float(fov)))
    focal = (h * 0.5) / math.tan(math.radians(fov) * 0.5)

    tf = FirstPersonTransform(eye, view, right, up, focal, cx, cz, heading,
                              ground_y, w, h, near=0.4, terrain=None)

    # The shaded channel may render large and shrink; the transform stays in
    # the caller's pixel space, so the returned image and the returned
    # transform always agree.
    ss = max(1, int(supersample)) if shaded else 1
    if ss > 1:
        img = _background_shaded(w * ss, h * ss, tf, ss)
    else:
        img = (_background_shaded(w, h, tf, 1) if shaded
               else _background(w, h, tf))
    draw = ImageDraw.Draw(img)

    faces: list = []
    if track is not None:
        samples = _road_samples(track, base_arc, ahead_m, behind_m, ground_y)
        _build_road(faces, track, samples, shaded)
        if shaded:
            _build_roadside(faces, samples, float(track.half_width))
            _build_start(faces, track, cx, cz, heading, ground_y,
                         ahead_m, behind_m)
        _build_props(faces, track, cx, cz, heading, ground_y,
                     ahead_m, behind_m, terrain=None, shaded=shaded)
    _build_hazards(faces, hazards, ground_y, shaded)
    _build_cars(faces, cars, self_index, ground_y, right, up,
                terrain=None, shaded=shaded)

    # contact shadows ride in the same list as everything else so the depth
    # sort can hide them behind nearer geometry; a separate earlier pass would
    # let even the distant road paint straight over them
    _paint_faces(draw, faces, tf, shaded=shaded, pixel_scale=ss)

    if ss > 1:
        img = img.resize((w, h), Image.LANCZOS)
    return img, tf


# ---------------------------------------------------------------------------
def _background(w: int, h: int, tf: FirstPersonTransform) -> Image.Image:
    horizon = tf.horizon_y()
    hy = int(round(min(max(horizon, 0.0), float(h))))
    arr = np.empty((h, w, 3), dtype=np.uint8)
    for y in range(h):
        if y < hy:
            arr[y, :, :] = _mix(SKY_TOP, SKY_HORIZON, y / max(hy - 1, 1))
        else:
            arr[y, :, :] = _mix(GROUND_FAR, GROUND_NEAR,
                                (y - hy) / max(h - hy - 1, 1))
    img = Image.fromarray(arr, "RGB")
    # a soft distant ridge so the horizon is not a flat line
    pts = []
    for x in range(0, w + 8, 8):
        n = math.sin(x * 0.055 + 1.3) * 0.55 + math.sin(x * 0.11 + 0.4) * 0.45
        pts.append((float(x), horizon - 8.0 - 16.0 * (n * 0.5 + 0.5)))
    pts.append((float(w), horizon))
    pts.append((0.0, horizon))
    ImageDraw.Draw(img).polygon(pts, fill=HILL)
    return img


def _smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 0.0
    t = (x - a) / (b - a)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def _background_shaded(w: int, h: int, tf: FirstPersonTransform,
                       scale: int) -> Image.Image:
    """Sky + fogged distance, matching the GL sky shader and fog range."""
    horizon = tf.horizon_y() * scale
    hy = int(round(min(max(horizon, 0.0), float(h))))
    top, horizon_col = S_SKY_TOP, S_SKY_HORIZON
    arr = np.empty((h, w, 3), dtype=np.uint8)
    for y in range(h):
        if y < hy:
            # 0 at the horizon, 1 at the top of the frame: the GL sky shader
            # mixes horizon -> top by uv.y ** SKY_POW
            v = 1.0 - y / max(hy - 1, 1)
            arr[y, :, :] = _mix(horizon_col, top, v ** _pal.SKY_POW)
        else:
            d = (y - hy) / max(h - hy - 1, 1)
            # the far ground dissolves into the horizon colour (fog), the
            # near ground is the real grass tone
            arr[y, :, :] = _mix(horizon_col, S_GROUND, _smoothstep(0.0, 0.34, d))
    img = Image.fromarray(arr, "RGB")
    pts = []
    for x in range(0, w + 8 * scale, 8 * scale):
        n = math.sin(x * 0.055 / scale + 1.3) * 0.55 + \
            math.sin(x * 0.11 / scale + 0.4) * 0.45
        pts.append((float(x),
                    horizon - (8.0 + 16.0 * (n * 0.5 + 0.5)) * scale))
    pts.append((float(w), horizon))
    pts.append((0.0, horizon))
    ImageDraw.Draw(img).polygon(pts, fill=_mix(S_HILL, S_SKY_HORIZON, 0.45))
    return img


# ---------------------------------------------------------------------------
def _edge(sample, side: float, off: float):
    x, z, y, nx, nz = sample
    return (x + nx * off * side, y, z + nz * off * side)


def _road_samples(track, base_arc: float, ahead_m: float, behind_m: float,
                  ground_y: float):
    """Centre-line samples across the visible window (shared by the builders).

    A monocular camera cannot disambiguate an object on a local rise from one
    in a dip, so the AI's frame keeps the ground level at the driver's own
    elevation rather than following the terrain.
    """
    total = ahead_m + behind_m
    step = max(1.5, min(5.0, total / 44.0))
    samples = []
    s = base_arc - behind_m
    end = base_arc + ahead_m
    while s <= end + 1e-6:
        i, _frac, p = track.point_at_arc(s)
        nrm = track.normal[i]
        samples.append((float(p[0]), float(p[1]), ground_y + 0.02,
                        float(nrm[0]), float(nrm[1])))
        s += step
    return samples


def _build_road(faces, track, samples, shaded: bool) -> None:
    if len(samples) < 2:
        return
    hw = float(track.half_width)

    for k in range(len(samples) - 1):
        a, b = samples[k], samples[k + 1]
        road = S_ROAD if shaded else ASPHALT
        if shaded:
            road = _tint(road, _jit(k * 17.0, 0.022))
        faces.append(([_edge(a, 1.0, hw), _edge(a, -1.0, hw),
                       _edge(b, -1.0, hw), _edge(b, 1.0, hw)], road))
        # kerbs (alternating red / white, like the 3D scene)
        ka, kb = (S_KERB_A, S_KERB_B) if shaded else (KERB_A, KERB_B)
        kerb = ka if (k // 3) % 2 == 0 else kb
        line = S_LINE if shaded else LINE
        for side in (1.0, -1.0):
            faces.append(([_edge(a, side, hw), _edge(a, side, hw + 0.85),
                           _edge(b, side, hw + 0.85), _edge(b, side, hw)], kerb))
            # continuous edge line just inside the kerb
            faces.append(([_edge(a, side, hw - 0.10), _edge(a, side, hw - 0.40),
                           _edge(b, side, hw - 0.40),
                           _edge(b, side, hw - 0.10)], line))

    # centre dashes
    line = S_LINE if shaded else LINE
    for k in range(0, len(samples) - 1, 3):
        a, b = samples[k], samples[k + 1]
        faces.append(([_edge(a, 1.0, 0.18), _edge(a, -1.0, 0.18),
                       _edge(b, -1.0, 0.18), _edge(b, 1.0, 0.18)], line))


def _rail_pt(sample, hw: float, side: float, dy: float):
    x, z, y, nx, nz = sample
    off = (hw + 0.95) * side
    return (x + nx * off, y + 0.08 + dy, z + nz * off)


def _build_roadside(faces, samples, hw: float) -> None:
    """Guardrails: posts plus two horizontal beams, as in render/scene.py."""
    n = len(samples)
    if n < 2:
        return
    # spacing of posts: every ~6 m of centre line
    step = max(1, int(round(6.0 / max(_sample_spacing(samples), 0.5))))
    for k in range(0, n - 1, step):
        a = samples[k]
        b = samples[k + 1]
        tx, tz = b[0] - a[0], b[2] - a[2]
        if abs(tx) + abs(tz) < _EPS:
            continue
        yaw = math.atan2(tz, tx)
        for side in (1.0, -1.0):
            p = _rail_pt(a, hw, side, 0.0)
            faces.extend(_box(p[0], p[2], p[1], 0.16, 0.16, 1.02, yaw, S_POST))
    for lo, hi in ((0.52, 0.64), (0.94, 1.04)):
        for k in range(n - 1):
            a, b = samples[k], samples[k + 1]
            for side in (1.0, -1.0):
                a0 = _rail_pt(a, hw, side, lo)
                a1 = _rail_pt(b, hw, side, lo)
                b1 = _rail_pt(b, hw, side, hi)
                b0 = _rail_pt(a, hw, side, hi)
                faces.append(([a0, a1, b1, b0], S_RAIL))


def _sample_spacing(samples) -> float:
    a, b = samples[0], samples[1]
    return math.hypot(b[0] - a[0], b[2] - a[2])


def _build_start(faces, track, cx: float, cz: float, heading: float,
                 ground_y: float, ahead_m: float, behind_m: float) -> None:
    """Checker line and gantry at the start, if it is inside the view."""
    try:
        c0 = track.center[0]
        nrm = track.normal[0]
        tang = track.tangent[0]
        hw = float(track.half_width)
    except Exception:
        return
    sx, sz = float(c0[0]), float(c0[2])
    dx, dz = sx - cx, sz - cz
    c, s = math.cos(heading), math.sin(heading)
    f = dx * c + dz * s
    lat = -dx * s + dz * c
    if not (-behind_m - 12.0 <= f <= ahead_m + 12.0):
        return
    if abs(lat) > ahead_m * 0.95 + 12.0:
        return

    y0 = ground_y + 0.085
    cols = 8
    step = 2.0 * hw / cols
    for k in range(cols):
        base = -hw + k * step
        for m in range(2):
            t0 = (m - 1) * 1.1
            a0 = (sx + nrm[0] * base + tang[0] * t0, y0,
                  sz + nrm[1] * base + tang[1] * t0)
            a1 = (sx + nrm[0] * (base + step) + tang[0] * t0, y0,
                  sz + nrm[1] * (base + step) + tang[1] * t0)
            a2 = (a1[0] + tang[0] * 1.1, y0, a1[2] + tang[1] * 1.1)
            a3 = (a0[0] + tang[0] * 1.1, y0, a0[2] + tang[1] * 1.1)
            col = (26, 26, 30) if (k + m) % 2 == 0 else (235, 235, 230)
            faces.append(([a0, a1, a2, a3], col))

    yaw = math.atan2(float(tang[1]), float(tang[0]))
    for side in (1.0, -1.0):
        x = sx + nrm[0] * (hw + 1.1) * side
        z = sz + nrm[1] * (hw + 1.1) * side
        faces.extend(_box(x, z, ground_y, 0.34, 0.34, 5.6, yaw, S_POST))
    faces.extend(_box(sx, sz, ground_y + 5.6, 0.5, (hw + 1.4) * 2.0, 0.5,
                      yaw, S_GANTRY))
    faces.extend(_box(sx, sz, ground_y + 4.9, 0.35, (hw + 1.0) * 2.0, 0.7,
                      yaw, S_GANTRY_BEAM))


# ---------------------------------------------------------------------------
def _box(cx: float, cz: float, y0: float, sx: float, sz: float, h: float,
         yaw: float, color):
    hx, hz = sx * 0.5, sz * 0.5
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for lx, lz in ((-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)):
        corners.append((cx + lx * c - lz * s, cz + lx * s + lz * c))
    y1 = y0 + h
    top = [(x, y1, z) for x, z in corners]
    bot = [(x, y0, z) for x, z in corners]
    faces = [(top, color), (bot[::-1], color)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append(([bot[i], bot[j], top[j], top[i]], color))
    return faces


def _prism(cx: float, cz: float, y0: float, radius: float, h: float, color,
           segments: int = 8):
    """An upright n-gon prism, used for barrels (scene.py uses a cylinder)."""
    top, bot = [], []
    for k in range(segments):
        a = 2.0 * math.pi * k / segments
        x = cx + radius * math.cos(a)
        z = cz + radius * math.sin(a)
        bot.append((x, y0, z))
        top.append((x, y0 + h, z))
    faces = [(top, color), (bot[::-1], color)]
    for i in range(segments):
        j = (i + 1) % segments
        faces.append(([bot[i], bot[j], top[j], top[i]], color))
    return faces


def _pyramid(cx: float, cz: float, y0: float, radius: float, h: float, color,
             segments: int = 6):
    """A low-poly cone, used for traffic cones."""
    apex = (cx, y0 + h, cz)
    ring = []
    for k in range(segments):
        a = 2.0 * math.pi * k / segments
        ring.append((cx + radius * math.cos(a), y0, cz + radius * math.sin(a)))
    faces = [(ring[::-1], color)]
    for i in range(segments):
        j = (i + 1) % segments
        faces.append(([ring[i], ring[j], apex], color))
    return faces


def _billboard(cx: float, cy: float, cz: float, half_w: float, height: float,
               right, up, color, shape: str = "rect"):
    center = np.array([cx, cy, cz], dtype=np.float64)
    r = right * half_w
    u = up * (height * 0.5)
    bl = center - r - u
    br = center + r - u
    tl = center - r + u
    tr = center + r + u
    if shape == "cone":
        apex = center + up * (height * 0.5)
        pts = [bl, br, apex]
    elif shape == "octagon":
        r2 = right * half_w
        u2 = up * (height * 0.5)
        k = 0.62
        pts = [center - r2, center - r2 * k + u2 * k, center + u2,
               center + r2 * k + u2 * k, center + r2,
               center + r2 * k - u2 * k, center - u2,
               center - r2 * k - u2 * k]
    else:
        pts = [bl, br, tr, tl]
    return [([tuple(float(v) for v in p) for p in pts], color)]


def _disc_y(cx: float, cy: float, cz: float, radius: float, color,
            segments: int = 12, depth_bias: float = 0.0):
    """A flat disc lying on the ground (contact shadow / hazard pool)."""
    pts = []
    for k in range(segments):
        a = 2.0 * math.pi * k / segments
        pts.append((cx + radius * math.cos(a), cy, cz + radius * math.sin(a)))
    return [(pts, color, depth_bias)]


# ---------------------------------------------------------------------------
def _build_props(faces, track, cx: float, cz: float, heading: float,
                 ground_y: float, ahead_m: float, behind_m: float,
                 terrain=None, shaded: bool = False) -> None:
    c, s = math.cos(heading), math.sin(heading)
    # camera right/up are recomputed here so billboards face the driver
    view = np.array([c, 0.0, s], dtype=np.float64)
    right = _normalize(np.cross(view, np.array([0.0, 1.0, 0.0])))
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def height_at(x, z):
        if terrain is None:
            return ground_y
        try:
            return float(terrain(x, z))
        except Exception:
            return ground_y

    def visible(x: float, z: float, margin: float = 6.0) -> bool:
        dx, dz = x - cx, z - cz
        f = dx * c + dz * s
        lat = -dx * s + dz * c
        return (-behind_m - margin <= f <= ahead_m + margin
                and abs(lat) <= ahead_m * 0.95 + margin)

    # obstacles: flat palette billboards so the detector can read them, or
    # small volumes when the frame is only ever seen by a model / the human
    for o in getattr(track, "obstacles", []) or ():
        if not visible(o.x, o.z):
            continue
        base = height_at(o.x, o.z) + 0.02
        if shaded:
            rgb = S_OBSTACLE.get(o.kind, S_OBSTACLE["crate"])
            if o.kind == "cone":
                faces.extend(_pyramid(o.x, o.z, base, o.radius * 0.9,
                                      o.height, rgb))
            elif o.kind == "barrel":
                faces.extend(_prism(o.x, o.z, base, o.radius * 0.8,
                                    o.height, rgb))
            else:
                faces.extend(_box(o.x, o.z, base, o.radius * 1.6,
                                  o.radius * 1.6, o.height, 0.0, rgb))
            continue
        rgb = OBSTACLE_PALETTE.get(o.kind, OBSTACLE_PALETTE["crate"])
        shape = "cone" if o.kind == "cone" else "rect"
        faces.extend(_billboard(o.x, base + o.height * 0.5, o.z,
                                o.radius, o.height, right, up, rgb, shape))

    # buildings
    for idx, b in enumerate(getattr(track, "buildings", []) or ()):
        if not visible(b.x, b.z, margin=max(b.sx, b.sz)):
            continue
        base = height_at(b.x, b.z) - 0.9
        if not shaded:
            faces.extend(_box(b.x, b.z, base, b.sx, b.sz, b.height, b.yaw,
                              BUILDING))
            continue
        body = S_BUILDINGS[idx % len(S_BUILDINGS)]
        faces.extend(_box(b.x, b.z, base, b.sx, b.sz, b.height, b.yaw, body))
        # parapet
        faces.extend(_box(b.x, b.z, base + b.height, b.sx * 1.03,
                          b.sz * 1.03, 0.7, b.yaw, _tint(body, -0.12)))
        # one window band per ~7 m of facade (scene.py does the same)
        for level in range(1, max(2, int(b.height / 7.0))):
            y = base + level * 7.0
            if y > base + b.height - 1.0:
                break
            faces.extend(_box(b.x, b.z, y, b.sx * 1.015, b.sz * 1.015, 1.9,
                              b.yaw, S_WINDOW))

    # trees
    for idx, t in enumerate(getattr(track, "trees", []) or ()):
        if not visible(t.x, t.z, margin=t.size):
            continue
        base = height_at(t.x, t.z)
        if not shaded:
            faces.extend(_billboard(t.x, base + t.size * 0.85, t.z,
                                    t.size * 0.40, t.size * 0.80, right, up,
                                    TREE, "octagon"))
            continue
        faces.extend(_prism(t.x, t.z, base, 0.22, t.size * 0.55, S_TRUNK,
                            segments=7))
        foliage = S_FOLIAGE[idx % len(S_FOLIAGE)]
        if idx % 3 == 0:                       # round canopy
            faces.extend(_pyramid(t.x, t.z, base + t.size * 0.62,
                                  t.size * 0.52, t.size * 0.62, foliage,
                                  segments=8))
        else:                                  # taller conifer
            faces.extend(_pyramid(t.x, t.z, base + t.size * 0.40,
                                  t.size * 0.52, t.size * 1.05, foliage,
                                  segments=8))


def _build_hazards(faces, hazards, ground_y: float, shaded: bool = False) -> None:
    color = S_HAZARD if shaded else HAZARD
    for hz in hazards or ():
        if isinstance(hz, dict):
            x, z = hz.get("x"), hz.get("z")
            r = float(hz.get("radius", 2.5))
        elif isinstance(hz, (tuple, list)) and len(hz) >= 2:
            x, z = hz[0], hz[1]
            r = float(hz[2]) if len(hz) > 2 else 2.5
        else:
            continue
        if x is None or z is None:
            continue
        pts = []
        for k in range(12):
            a = 2.0 * math.pi * k / 12.0
            pts.append((float(x) + r * math.cos(a), ground_y + 0.03,
                        float(z) + r * math.sin(a)))
        faces.append((pts, color))


# ---------------------------------------------------------------------------
def _rgb(color):
    return tuple(int(max(0, min(255, round(v)))) for v in color)


def _body_color(car: CarView):
    """The shaded channel shows the real livery; the flat one keeps the
    detector's high-contrast colours."""
    return _rgb(car.real_color) if car.real_color is not None else _rgb(car.color)


def _build_cars(faces, cars, self_index: int, ground_y: float,
                right, up, terrain=None, shaded: bool = False) -> None:
    def height_at(x, z):
        if terrain is None:
            return ground_y
        try:
            return float(terrain(x, z))
        except Exception:
            return ground_y

    for idx, car in enumerate(cars):
        if idx == self_index:
            _own_hood(faces, car, ground_y, shaded)
            continue
        base = height_at(car.x, car.z)
        color = _body_color(car) if shaded else _rgb(car.color)
        if shaded:
            # body + cabin + four wheels, approximating models.build_paint
            faces.extend(_box(car.x, car.z, base + 0.30, 4.1, 1.8, 0.72,
                              car.heading, color))
            faces.extend(_box(car.x, car.z, base + 1.02, 1.9, 1.55, 0.62,
                              car.heading, _tint(color, -0.18)))
            glass = _tint(color, -0.55)
            faces.extend(_box(car.x, car.z, base + 1.30, 1.55, 1.45, 0.34,
                              car.heading, glass))
            _wheels(faces, car, base)
            # contact shadow: biased a little further away so the body above
            # it always wins the depth sort
            faces.extend(_disc_y(car.x, base + 0.03, car.z, 1.45,
                                 (16, 16, 18), segments=10, depth_bias=0.5))
            continue
        faces.extend(_box(car.x, car.z, base + 0.05, 4.1, 1.8, 1.05,
                          car.heading, color))
        faces.extend(_billboard(car.x, base + 1.35, car.z, 0.75, 0.55,
                                right, up, color, "octagon"))


def _wheels(faces, car, base: float) -> None:
    c, s = math.cos(car.heading), math.sin(car.heading)

    def world(lx, lz):
        return (car.x + lx * c - lz * s, car.z + lx * s + lz * c)

    for lx in (1.35, -1.35):
        for lz in (0.86, -0.86):
            x, z = world(lx, lz)
            faces.extend(_box(x, z, base + 0.02, 0.62, 0.30, 0.62,
                              car.heading, (26, 26, 28)))


def _own_hood(faces, car, ground_y: float, shaded: bool = False) -> None:
    c, s = math.cos(car.heading), math.sin(car.heading)
    color = _body_color(car) if shaded else _rgb(car.color)

    def world(lx, ly, lz):
        return (car.x + lx * c - lz * s, ground_y + ly, car.z + lx * s + lz * c)

    hood = [world(0.55, 0.98, -0.90), world(0.55, 0.98, 0.90),
            world(2.02, 0.74, 0.50), world(2.02, 0.74, -0.50)]
    faces.append((hood, color))
    if not shaded:
        return
    # the front wheels poke into the bottom of the frame
    for lz in (0.86, -0.86):
        faces.extend(_box(world(1.35, 0.0, lz)[0], world(1.35, 0.0, lz)[2],
                          ground_y + 0.02, 0.62, 0.30, 0.62, car.heading,
                          (26, 26, 28)))


# ---------------------------------------------------------------------------
def _clip_near(cam_pts, near: float):
    """Sutherland-Hodgman clip of a camera-space polygon against ``z >= near``."""
    out = []
    n = len(cam_pts)
    for i in range(n):
        a = cam_pts[i]
        b = cam_pts[(i + 1) % n]
        ain = a[2] >= near
        bin_ = b[2] >= near
        if ain:
            out.append(a)
        if ain != bin_:
            dz = b[2] - a[2]
            t = (near - a[2]) / dz if abs(dz) > 1e-9 else 0.0
            out.append((a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        near))
    return out


def _face_normal(pts, eye):
    """World-space unit normal of a polygon, oriented towards the camera."""
    n = len(pts)
    nx = ny = nz = 0.0
    for i in range(1, n - 1):
        ux = pts[i][0] - pts[0][0]
        uy = pts[i][1] - pts[0][1]
        uz = pts[i][2] - pts[0][2]
        vx = pts[i + 1][0] - pts[0][0]
        vy = pts[i + 1][1] - pts[0][1]
        vz = pts[i + 1][2] - pts[0][2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        if nx * nx + ny * ny + nz * nz > 1e-16:
            break
    else:
        return None
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n
    if nx * (eye[0] - cx) + ny * (eye[1] - cy) + nz * (eye[2] - cz) < 0.0:
        nx, ny, nz = -nx, -ny, -nz
    return (nx, ny, nz, cx, cy, cz)


def _shade(color, nrm, eye):
    """Blinn-Phong-lite: hemisphere ambient + Lambert sun + distance fog.

    Mirrors the GL LIT_FS fragment shader so the AI's frame and the window
    agree on tone.
    """
    nx, ny, nz, cx, cy, cz = nrm
    hemi = 0.5 + 0.5 * ny
    ar = _pal.SKY_GROUND[0] + (_pal.SKY_TOP[0] - _pal.SKY_GROUND[0]) * hemi
    ag = _pal.SKY_GROUND[1] + (_pal.SKY_TOP[1] - _pal.SKY_GROUND[1]) * hemi
    ab = _pal.SKY_GROUND[2] + (_pal.SKY_TOP[2] - _pal.SKY_GROUND[2]) * hemi
    diff = nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2]
    if diff < 0.0:
        diff = 0.0
    base_r = color[0] / 255.0
    base_g = color[1] / 255.0
    base_b = color[2] / 255.0
    r = base_r * ar * _pal.AMBIENT + base_r * diff * _pal.DIFFUSE
    g = base_g * ag * _pal.AMBIENT + base_g * diff * _pal.DIFFUSE
    b = base_b * ab * _pal.AMBIENT + base_b * diff * _pal.DIFFUSE
    dx = cx - eye[0]
    dy = cy - eye[1]
    dz = cz - eye[2]
    t = _smoothstep(_FOG_A, _FOG_B, math.sqrt(dx * dx + dy * dy + dz * dz))
    r += (_pal.SKY_HORIZON[0] - r) * t
    g += (_pal.SKY_HORIZON[1] - g) * t
    b += (_pal.SKY_HORIZON[2] - b) * t
    return (int(max(0, min(255, round(r * 255.0)))),
            int(max(0, min(255, round(g * 255.0)))),
            int(max(0, min(255, round(b * 255.0)))))


def _paint_faces(draw: ImageDraw.ImageDraw, faces, tf: FirstPersonTransform,
                 shaded: bool = False, pixel_scale: int = 1):
    eye = tf.eye
    scale = float(max(1, int(pixel_scale)))
    projected = []
    for face in faces:
        # a face may carry a depth bias as its third element, which lets a
        # decal (a contact shadow) lose to the body sitting on top of it
        pts, color = face[0], face[1]
        bias = float(face[2]) if len(face) > 2 else 0.0
        cam = _clip_near([tf.to_camera(p) for p in pts], tf.near)
        if len(cam) < 3:
            continue
        depth = sum(c[2] for c in cam) / len(cam) + bias
        scr = [tf.camera_to_pixel(c) for c in cam]
        if scale != 1.0:
            scr = [(px * scale, py * scale) for px, py in scr]
        projected.append((depth, scr, color, pts))
    # painter's algorithm: draw the farthest polygons first
    projected.sort(key=lambda item: -item[0])
    for _depth, scr, color, pts in projected:
        if shaded:
            nrm = _face_normal(pts, eye)
            if nrm is not None:
                color = _shade(color, nrm, eye)
        draw.polygon(scr, fill=color)
