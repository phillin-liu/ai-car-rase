"""Procedural race track generation and spatial queries.

The track is a closed, non self-intersecting loop built from smoothed
Catmull-Rom control points.  Around the center line we build:

* two boundary walls (the car is slowed when it touches them),
* decorative / structural buildings placed just outside the corridor,
* obstacles distributed *on* the racing surface.

All queries are done in the X/Z ground plane; the center line also carries
a (small) Y elevation so the 3D view has some relief.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Obstacle:
    kind: str          # cone | barrel | crate | barrier
    x: float
    z: float
    radius: float
    height: float
    slow: float        # speed multiplier applied on impact
    color: tuple


@dataclass
class Building:
    x: float
    z: float
    sx: float          # footprint along local x
    sz: float          # footprint along local z
    height: float
    yaw: float
    color: tuple


@dataclass
class Tree:
    x: float
    z: float
    size: float


class _NearestCentre:
    """Nearest point of a set of ``(x, z)`` points, numpy only.

    A stand-in for ``scipy.spatial.cKDTree`` that implements just the
    ``query`` call :class:`Track` makes.  ``scipy`` is an optional dependency
    in practice -- nothing pins it at run time -- and the terrain code used to
    let the resulting ``ImportError`` fall through to a bare ``return hills``.
    That silently produced a map whose ground was raw Perlin noise while the
    road, the obstacles and the trees were all placed at the track's own
    elevation, i.e. grass several metres *above* the road and objects buried
    in it.  A brute force scan of a few hundred points is cheap next to that.
    """

    # Distance matrices are built in blocks so a big query grid cannot
    # allocate an (all points x all query points) array at once.
    BLOCK = 512

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)

    def query(self, q):
        single = np.asarray(q, dtype=np.float64).ndim == 1
        pts = np.atleast_2d(np.asarray(q, dtype=np.float64))
        dist = np.empty(len(pts), dtype=np.float64)
        idx = np.empty(len(pts), dtype=np.int64)
        for k in range(0, len(pts), self.BLOCK):
            block = pts[k:k + self.BLOCK]
            d2 = ((block[:, None, :] - self.points[None, :, :]) ** 2).sum(axis=2)
            best = d2.argmin(axis=1)
            dist[k:k + len(block)] = np.sqrt(d2[np.arange(len(block)), best])
            idx[k:k + len(block)] = best
        if single:
            return float(dist[0]), int(idx[0])
        return dist, idx


@dataclass
class Track:
    seed: int
    center: np.ndarray            # (N, 3) = (x, y_elevation, z)
    width: float
    left: np.ndarray              # (N, 2) ground x/z
    right: np.ndarray             # (N, 2) ground x/z
    tangent: np.ndarray           # (N, 2) unit tangents (x/z)
    normal: np.ndarray            # (N, 2) unit left normals (x/z)
    curvature: np.ndarray         # (N,) signed curvature
    arc: np.ndarray               # (N,) cumulative arc length
    seg_len: np.ndarray           # (N,) segment lengths to next point
    length: float
    center_xz: np.ndarray = None  # (N, 2) ground plane center line
    obstacles: list = field(default_factory=list)
    buildings: list = field(default_factory=list)
    trees: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # basic geometry helpers
    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.center)

    @property
    def half_width(self) -> float:
        return self.width * 0.5

    def height_at_index(self, i: int, t: float) -> float:
        j = (i + 1) % self.n
        return float(self.center[i, 1] * (1.0 - t) + self.center[j, 1] * t)

    def point_at_arc(self, s: float):
        """Return (index, frac, point_xy) for an arc length (wraps)."""
        s = s % self.length
        i = int(np.searchsorted(self.arc, s, side="right") - 1)
        i = max(0, min(self.n - 1, i))
        seg = self.seg_len[i]
        t = 0.0 if seg <= 1e-9 else (s - self.arc[i]) / seg
        t = max(0.0, min(1.0, t))
        xz = self.center_xz if self.center_xz is not None else self.center[:, [0, 2]]
        p = xz[i] * (1.0 - t) + xz[(i + 1) % self.n] * t
        return i, float(t), p

    def forward_at_index(self, i: int) -> np.ndarray:
        return self.tangent[i]

    def start_line(self):
        return self.left[0], self.right[0]

    def spawn(self, arc_offset: float, lateral: float):
        """Return (x, z, heading) for a grid slot."""
        i, t, p = self.point_at_arc(arc_offset)
        nrm = self.normal[i]
        tang = self.tangent[i]
        x = float(p[0] + nrm[0] * lateral)
        z = float(p[1] + nrm[1] * lateral)
        heading = math.atan2(float(tang[1]), float(tang[0]))
        return x, z, heading

    # ------------------------------------------------------------------
    # nearest point query
    # ------------------------------------------------------------------
    def nearest(self, pos, hint: int | None = None, expected_arc: float | None = None,
                window: int = 45, tol: float = 3.0):
        """Find the closest point on the center line.

        Returns ``(index, t, proj_xy, distance, lateral, tangent)`` where
        ``lateral`` is signed (+ = left of travel direction).  When
        ``expected_arc`` is given the result is disambiguated near the
        start/finish seam by preferring the candidate whose arc length is
        closest to ``expected_arc`` (within ``tol`` metres of the best).
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(2)
        n = self.n
        if hint is None:
            idx = np.arange(n)
        else:
            idx = (np.arange(hint - window, hint + window + 1)) % n
        xz = self.center_xz if self.center_xz is not None else self.center[:, [0, 2]]
        a = xz[idx]
        b = xz[(idx + 1) % n]
        ab = b - a
        ab2 = np.einsum("ij,ij->i", ab, ab) + 1e-9
        ap = pos - a
        t = np.clip(np.einsum("ij,ij->i", ap, ab) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        diff = pos - proj
        d2 = np.einsum("ij,ij->i", diff, diff)
        k = int(np.argmin(d2))
        if expected_arc is not None:
            arcs = self.arc[idx] + t * self.seg_len[idx]
            dmin = math.sqrt(max(d2[k], 0.0))
            near = np.where(np.sqrt(d2) <= dmin + tol)[0]
            if len(near) > 1:
                dd = np.abs(arcs[near] - expected_arc)
                dd = np.minimum(dd, self.length - dd)
                k = int(near[int(np.argmin(dd))])
        i = int(idx[k])
        ti = float(t[k])
        p = proj[k]
        nrm = self.normal[i]
        tangent = self.tangent[i]
        lateral = float(np.dot(diff[k], nrm))
        dist = float(math.sqrt(max(d2[k], 0.0)))
        return i, ti, p, dist, lateral, tangent

    def arc_at(self, i: int, t: float) -> float:
        return float(self.arc[i] + t * self.seg_len[i])

    # ------------------------------------------------------------------
    # terrain
    # ------------------------------------------------------------------
    def _kdtree(self):
        """Nearest centre-line point index (built once, cached per track)."""
        tree = getattr(self, "_kdtree_cache", None)
        if tree is None:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(self.center_xz)
            except Exception:
                tree = _NearestCentre(self.center_xz)
            self._kdtree_cache = tree
        return tree

    # How far outside the racing surface the ground is pinned to the track's
    # own elevation, and how far past that it takes to become open hills.
    #
    # The ground grid is ~12 m across per cell while the road is only ~16 m
    # wide, so one quad can straddle the entire racing surface.  With a plain
    # ``exp(-d / 70)`` falloff a vertex even 10 m off the road already carried
    # enough of the fBm hills to sit a couple of metres above it -- the quad
    # then painted grass *over* the track and buried the obstacles standing on
    # it, because obstacles are placed at the analytic height.  A 32 m apron is
    # wider than a grid diagonal, so every quad that overlaps the road has all
    # four corners pinned to the track elevation; linear interpolation between
    # pinned corners stays pinned, which makes the poke-through impossible
    # instead of merely unlikely.
    APRON = 32.0
    HILL_FALLOFF = 140.0

    def _hills_weight(self, dist: float) -> float:
        """Blend weight of the open hills ``dist`` metres from the centre line.

        Zero across the apron (the ground follows the track there), then eased
        in with a smoothstep over :attr:`HILL_FALLOFF` metres so the hills
        start without a crease.
        """
        d = max(float(dist) - self.half_width, 0.0)
        t = min(1.0, max(0.0, (d - self.APRON) / self.HILL_FALLOFF))
        return t * t * (3.0 - 2.0 * t)

    def _hills_weight_many(self, dist) -> np.ndarray:
        """Vectorised :meth:`_hills_weight` (kept in step with it exactly)."""
        d = np.maximum(np.asarray(dist, dtype=np.float64) - self.half_width, 0.0)
        t = np.clip((d - self.APRON) / self.HILL_FALLOFF, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def terrain_height(self, x: float, z: float) -> float:
        """Perlin-noise terrain blended into the track elevation.

        Near the racing surface the ground follows the track's own
        elevation (so nothing floats or clips); farther away it rolls
        into fBm Perlin hills, giving the map natural-looking relief.

        The nearest-point query is not optional: without it the caller gets
        pure hills, i.e. a ground plane that cuts through the road and buries
        everything standing on it.  ``_NearestCentre`` makes it numpy-only so
        that can no longer happen through a missing ``scipy``.
        """
        pn = getattr(self, "_terrain_noise", None)
        if pn is None:
            pn = PerlinNoise(self.seed)
            self._terrain_noise = pn
        hills = pn.fbm(x * 0.006, z * 0.006, octaves=4) * 10.0
        dd, idx = self._kdtree().query([x, z])
        h = float(self.center[int(idx), 1])
        w = self._hills_weight(float(dd))
        return h * (1.0 - w) + hills * w

    def terrain_height_many(self, xs, zs) -> np.ndarray:
        """Vectorised :meth:`terrain_height` over arrays of x/z.

        The mesh builder used to call the scalar version once per grid vertex,
        which issued one nearest-point query per vertex (4096 of them).  This
        does a single batched query and a vectorised fBm evaluation instead.
        """
        xs = np.asarray(xs, dtype=np.float64)
        zs = np.asarray(zs, dtype=np.float64)
        pn = getattr(self, "_terrain_noise", None)
        if pn is None:
            pn = PerlinNoise(self.seed)
            self._terrain_noise = pn
        hills = pn.fbm_vec(xs * 0.006, zs * 0.006, octaves=4) * 10.0
        dd, idx = self._kdtree().query(np.column_stack([xs.ravel(), zs.ravel()]))
        dd = np.asarray(dd, dtype=np.float64).ravel()
        idx = np.asarray(idx, dtype=np.int64).ravel()
        h = self.center[idx, 1].astype(np.float64)
        w = self._hills_weight_many(dd)
        out = h * (1.0 - w) + hills.ravel() * w
        return out.reshape(xs.shape)

    def curvature_ahead(self, base_arc: float, distances):
        """Signed curvature sampled ``distances`` metres ahead of base_arc."""
        out = []
        for d in distances:
            idx, _, _ = self.point_at_arc(base_arc + d)
            out.append(float(self.curvature[idx]))
        return out

    def points_ahead(self, base_arc: float, distances):
        out = []
        for d in distances:
            _, _, p = self.point_at_arc(base_arc + d)
            out.append((float(p[0]), float(p[1])))
        return out


# ---------------------------------------------------------------------------
# Perlin noise
# ---------------------------------------------------------------------------
class PerlinNoise:
    """Classic 2D Perlin gradient noise, deterministic per seed, plus fBm."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        perm = np.arange(256, dtype=np.int64)
        rng.shuffle(perm)
        self._perm = np.concatenate([perm, perm])

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _grad(h: int, x: float, y: float) -> float:
        h &= 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        return ((u if (h & 1) == 0 else -u) +
                (v if (h & 2) == 0 else -v))

    def noise2(self, x: float, y: float) -> float:
        """Single-octave Perlin noise, roughly in [-1, 1]."""
        x0 = math.floor(x)
        y0 = math.floor(y)
        xf = x - x0
        yf = y - y0
        xi = int(x0) & 255
        yi = int(y0) & 255
        perm = self._perm
        aa = int(perm[int(perm[xi]) + yi])
        ab = int(perm[int(perm[xi]) + yi + 1])
        ba = int(perm[int(perm[xi + 1]) + yi])
        bb = int(perm[int(perm[xi + 1]) + yi + 1])
        u = self._fade(xf)
        v = self._fade(yf)
        g_aa = self._grad(aa, xf, yf)
        g_ba = self._grad(ba, xf - 1.0, yf)
        g_ab = self._grad(ab, xf, yf - 1.0)
        g_bb = self._grad(bb, xf - 1.0, yf - 1.0)
        x1 = g_aa + u * (g_ba - g_aa)
        x2 = g_ab + u * (g_bb - g_ab)
        return x1 + v * (x2 - x1)

    def fbm(self, x: float, y: float, octaves: int = 4,
            lacunarity: float = 2.0, gain: float = 0.5) -> float:
        """Fractal Brownian motion (summed octaves), normalised to ~[-1, 1]."""
        amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
        for _ in range(octaves):
            total += amp * self.noise2(x * freq, y * freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm if norm else 0.0

    # -- vectorised variants (used for the terrain grid) -----------------
    @staticmethod
    def _grad_v(h, x, y):
        h = h & 7
        u = np.where(h < 4, x, y)
        v = np.where(h < 4, y, x)
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def noise2_vec(self, x, y):
        """Vectorised single-octave Perlin noise (numpy broadcasting)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x0 = np.floor(x)
        y0 = np.floor(y)
        xf = x - x0
        yf = y - y0
        xi = x0.astype(np.int64) & 255
        yi = y0.astype(np.int64) & 255
        perm = self._perm
        aa = perm[perm[xi] + yi]
        ab = perm[perm[xi] + yi + 1]
        ba = perm[perm[xi + 1] + yi]
        bb = perm[perm[xi + 1] + yi + 1]
        u = self._fade(xf)
        v = self._fade(yf)
        x1 = self._grad_v(aa, xf, yf) + u * (self._grad_v(ba, xf - 1.0, yf)
                                             - self._grad_v(aa, xf, yf))
        x2 = self._grad_v(ab, xf, yf - 1.0) + u * (
            self._grad_v(bb, xf - 1.0, yf - 1.0) - self._grad_v(ab, xf, yf - 1.0))
        return x1 + v * (x2 - x1)

    def fbm_vec(self, x, y, octaves: int = 4,
                lacunarity: float = 2.0, gain: float = 0.5):
        amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
        for _ in range(octaves):
            total = total + amp * self.noise2_vec(x * freq, y * freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm if norm else total


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def _catmull_rom_closed(ctrl: np.ndarray, samples_per_seg: int) -> np.ndarray:
    n = len(ctrl)
    out = []
    ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    for i in range(n):
        p0 = ctrl[(i - 1) % n]
        p1 = ctrl[i]
        p2 = ctrl[(i + 1) % n]
        p3 = ctrl[(i + 2) % n]
        a = 2.0 * p1
        b = -p0 + p2
        c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
        d = -p0 + 3.0 * p1 - 3.0 * p2 + p3
        pts = 0.5 * (a + b * ts + c * ts * ts + d * ts ** 3)
        out.append(pts)
    return np.vstack(out)


def _smooth_closed(p: np.ndarray, iters: int, alpha: float) -> np.ndarray:
    for _ in range(iters):
        p = (1.0 - alpha) * p + alpha * 0.5 * (
            np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
    return p


def _resample_closed(p: np.ndarray, ds: float) -> np.ndarray:
    d = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(d)])
    total = float(arc[-1])
    m = max(24, int(round(total / ds)))
    targets = np.linspace(0.0, total, m, endpoint=False)
    xp = np.concatenate([p[:, 0], p[:1, 0]])
    zp = np.concatenate([p[:, 1], p[:1, 1]])
    x = np.interp(targets, arc, xp)
    z = np.interp(targets, arc, zp)
    return np.stack([x, z], axis=1)


def _min_self_distance(pts: np.ndarray, gap: int) -> float:
    """Minimum distance between points far apart along the loop (index gap)."""
    n = len(pts)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        k = min(n, 48)
        dists, idxs = tree.query(pts, k=k)
        rows = np.repeat(np.arange(n), k - 1)
        cols = idxs[:, 1:].reshape(-1)
        dd = dists[:, 1:].reshape(-1)
        circ = np.abs(rows - cols)
        circ = np.minimum(circ, n - circ)
        mask = circ >= gap
        if not np.any(mask):
            return float("inf")
        return float(dd[mask].min())
    except Exception:  # pragma: no cover - fallback
        step = max(1, n // 200)
        q = pts[::step]
        qidx = np.arange(0, n, step)
        diff = q[:, None, :] - pts[None, :, :]
        d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        idx = np.abs(qidx[:, None] - np.arange(n)[None, :])
        idx = np.minimum(idx, n - idx)
        d[idx < gap] = np.inf
        return float(d.min())


def _elevation(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Track elevation from Perlin noise sampled on a circle.

    Sampling the noise on a circle in the noise domain makes the profile
    perfectly periodic, so the closed loop never has an elevation seam at
    the start/finish line.
    """
    n = len(pts)
    pn = PerlinNoise(int(rng.integers(0, 2 ** 31)))
    s = np.arange(n, dtype=np.float64) / n
    ang = 2.0 * math.pi * s
    phase = float(rng.uniform(0.0, 100.0))
    amp = float(rng.uniform(3.0, 6.5))
    freq = float(rng.uniform(1.8, 3.2))
    y = np.array([amp * pn.fbm(freq * math.cos(a) + phase,
                               freq * math.sin(a) + phase, octaves=3)
                  for a in ang])
    # a second, larger-scale swell for long hills along the lap
    swell_amp = float(rng.uniform(2.0, 4.5))
    y += swell_amp * np.array([pn.fbm(0.9 * math.cos(a) + 50.0,
                                      0.9 * math.sin(a) + 50.0, octaves=1)
                               for a in ang])
    return y


def _curvature(pts: np.ndarray) -> np.ndarray:
    tang = np.roll(pts, -1, axis=0) - pts
    norm = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    tang = tang / norm
    cross = tang[:, 0] * np.roll(tang[:, 1], -1) - tang[:, 1] * np.roll(tang[:, 0], -1)
    dot = np.clip(np.einsum("ij,ij->i", tang, np.roll(tang, -1, axis=0)), -1, 1)
    ang = np.arccos(dot)
    ds = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1) + 1e-9
    signed = np.sign(cross) * ang / ds
    # smooth curvature a touch
    for _ in range(3):
        signed = 0.5 * signed + 0.25 * (np.roll(signed, 1) + np.roll(signed, -1))
    return signed


def _distance_to_polyline(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    a = points
    b = np.roll(points, -1, axis=0)
    ab = b - a
    ab2 = np.einsum("ij,ij->i", ab, ab) + 1e-9
    ap = q[None, :] - a
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / ab2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = q[None, :] - proj
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def _build_one(seed: int, width: float):
    rng = np.random.default_rng(seed)
    n_ctrl = int(rng.integers(9, 14))
    base_r = float(rng.uniform(150.0, 210.0))
    # ordered angular gaps -> strictly increasing angles in [0, 2pi)
    gaps = rng.uniform(0.7, 1.5, n_ctrl)
    gaps = gaps / gaps.sum() * 2.0 * math.pi
    angles = np.concatenate([[0.0], np.cumsum(gaps)[:-1]])
    radii = base_r * rng.uniform(0.82, 1.18, n_ctrl)
    ctrl = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    samples = _catmull_rom_closed(ctrl, 60)
    gap = int(width / 1.6) + 3
    max_curv = 0.050
    center = _resample_closed(_smooth_closed(samples, 30, 0.25), 1.6)
    curv = _curvature(center)
    for _ in range(45):
        center = _resample_closed(_smooth_closed(center, 3, 0.18), 1.6)
        curv = _curvature(center)
        md = _min_self_distance(center, gap)
        if abs(curv).max() <= max_curv and md > width * 1.25:
            return center, md, True
    return center, float(md), False


def generate_track(seed: int, width: float = 16.0, items_enabled: bool = True) -> Track:
    """Generate a random, valid closed race track."""
    base_seed = int(seed) & 0x7FFFFFFF
    best = None
    for attempt in range(48):
        s = base_seed + attempt * 7919
        center2d, min_dist, ok = _build_one(s, width)
        if best is None or min_dist > best[1]:
            best = (center2d, min_dist, ok, s)
        if ok:
            best = (center2d, min_dist, ok, s)
            break
    center2d, min_dist, ok, used_seed = best
    rng = np.random.default_rng(used_seed ^ 0x5DEECE66D)

    n = len(center2d)
    y = _elevation(center2d, rng) * 0.6
    center = np.stack([center2d[:, 0], y, center2d[:, 1]], axis=1)

    # tangents / normals / curvature in X/Z
    tang = np.roll(center2d, -1, axis=0) - center2d
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    curv = _curvature(center2d)

    left = center2d + normal * (width * 0.5)
    right = center2d - normal * (width * 0.5)
    seg_len = np.linalg.norm(np.roll(center2d, -1, axis=0) - center2d, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    total = float(np.sum(seg_len))

    track = Track(
        seed=used_seed, center=center, width=width,
        left=left, right=right, tangent=tang, normal=normal,
        curvature=curv, arc=arc, seg_len=seg_len, length=total,
        center_xz=center2d,
    )

    # --- on-track obstacles ------------------------------------------
    # Clusters of one to three obstacles, spaced along the lap.  Every cluster
    # hugs one side of the racing surface (or is split into a "gate" with an
    # open middle), so a lane wide enough to drive through always remains: a
    # denser field has to read as a slalom, never as a wall the car cannot get
    # past.  Each entry is ``(side, |lateral| as a fraction of half width,
    # how many obstacles that side gets)``.
    obstacles = []
    colors = [(0.95, 0.35, 0.15), (0.95, 0.75, 0.15), (0.45, 0.45, 0.5),
              (0.85, 0.2, 0.2), (0.2, 0.6, 0.9)]
    s = 42.0
    while s < total - 24.0:
        roll = rng.random()
        if roll < 0.50:                       # one obstacle, off to one side
            plan = [(rng.choice((-1.0, 1.0)), rng.uniform(0.16, 0.58), 1)]
        elif roll < 0.80:                     # two stacked down one side
            plan = [(rng.choice((-1.0, 1.0)), rng.uniform(0.16, 0.58), 2)]
        elif roll < 0.92:                     # three down one side
            plan = [(rng.choice((-1.0, 1.0)), rng.uniform(0.22, 0.58), 3)]
        else:                                 # a gate: one each side, middle open
            side = rng.choice((-1.0, 1.0))
            plan = [(side, rng.uniform(0.50, 0.66), 1),
                    (-side, rng.uniform(0.50, 0.66), 1)]
        for side, lat_ratio, count in plan:
            for k in range(count):
                # staggering the arcs turns a same-side cluster into a chicane
                idx, _, _ = track.point_at_arc(s + k * rng.uniform(4.0, 9.0))
                nrm = normal[idx]
                lat = side * lat_ratio * (width * 0.5)
                x = float(center2d[idx, 0] + nrm[0] * lat)
                z = float(center2d[idx, 1] + nrm[1] * lat)
                kind_roll = rng.random()
                if kind_roll < 0.6:
                    obs = Obstacle("cone", x, z, 0.6, 1.3, 0.75, colors[0])
                elif kind_roll < 0.85:
                    obs = Obstacle("barrel", x, z, 0.8, 1.5, 0.65, colors[1])
                else:
                    obs = Obstacle("crate", x, z, 1.1, 1.5, 0.6, colors[3])
                obstacles.append(obs)
        s += float(rng.uniform(48.0, 105.0))
    track.obstacles = obstacles

    # --- buildings ----------------------------------------------------
    buildings = []
    palette = [
        (0.55, 0.60, 0.68), (0.72, 0.55, 0.45), (0.45, 0.55, 0.62),
        (0.80, 0.70, 0.55), (0.38, 0.42, 0.50), (0.66, 0.48, 0.52),
    ]
    extent = float(np.max(np.linalg.norm(center2d, axis=1))) + 40.0
    tries = 0
    while len(buildings) < 26 and tries < 4000:
        tries += 1
        x = float(rng.uniform(-extent, extent))
        z = float(rng.uniform(-extent, extent))
        d = float(_distance_to_polyline(center2d, np.array([x, z])).min())
        if d > width * 0.5 + 75.0:
            continue
        sx = float(rng.uniform(8.0, 22.0))
        sz = float(rng.uniform(8.0, 22.0))
        h = float(rng.uniform(8.0, 42.0))
        yaw = float(rng.uniform(0, math.pi))
        # keep the whole footprint clear of the corridor (no clipping)
        if d < width * 0.5 + max(sx, sz) * 0.5 + 3.0:
            continue
        # avoid overlaps
        bad = False
        for b in buildings:
            if math.hypot(b.x - x, b.z - z) < (max(b.sx, b.sz) + max(sx, sz)) * 0.7:
                bad = True
                break
        if bad:
            continue
        buildings.append(Building(x, z, sx, sz, h, yaw,
                                  palette[int(rng.integers(0, len(palette)))]))
    track.buildings = buildings

    # --- trees / decoration ------------------------------------------
    trees = []
    tries = 0
    while len(trees) < 90 and tries < 4000:
        tries += 1
        x = float(rng.uniform(-extent, extent))
        z = float(rng.uniform(-extent, extent))
        d = float(_distance_to_polyline(center2d, np.array([x, z])).min())
        size = float(rng.uniform(2.5, 6.0))
        if d > width * 0.5 + 60.0:
            continue
        if d < width * 0.5 + size * 0.6 + 2.0:
            continue
        trees.append(Tree(x, z, size))
    track.trees = trees

    return track
