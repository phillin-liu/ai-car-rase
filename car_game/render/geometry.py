"""Procedural geometry builders.

Winding convention
------------------
Every face is emitted **counter-clockwise when seen from outside**, matching
``glFrontFace(GL_CCW)`` + ``glCullFace(GL_BACK)``.  Callers pass the *outward*
normal explicitly; :func:`GeoBuffer.add_quad` / :func:`add_triangle` flip the
vertex order whenever the supplied order disagrees with that normal.

This matters: the previous renderer derived normals with ``cross(p1-p0,
p2-p0)`` and therefore never noticed that ``add_box`` / cylinder sides /
walls were wound *inward*.  With back-face culling on, the surfaces facing
the camera were deleted and you saw the inside of the far faces -- the real
reason the cars, buildings and obstacles looked broken rather than merely
blocky.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Vector helpers -- plain tuples of floats, deliberately.
#
# This module emits tens of thousands of vertices when a track is meshed, and
# every vertex needs a cross product and a normalisation.  ``np.cross`` costs
# ~30 us per call on 3-vectors (it goes through moveaxis/normalize_axis_tuple
# to support broadcasting), so building one track spent ~2.3 s inside it and
# froze the window for most of a second between races.  Three multiplies and a
# sqrt each in plain Python is ~1 us, which is why the mesh build is now a
# small fraction of what it was.  Nothing here needs broadcasting.
# ---------------------------------------------------------------------------
def _v(p):
    """A point as ``(x, y, z)`` floats (accepts tuples, lists, ndarrays)."""
    return (float(p[0]), float(p[1]), float(p[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _n(v):
    v = _v(v)
    ln = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if ln < 1e-12:
        return (0.0, 1.0, 0.0)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def rotate_xz(dx: float, dz: float, yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c - dz * s, dx * s + dz * c


class GeoBuffer:
    """Accumulates interleaved ``pos(3) normal(3) color(3)`` vertices."""

    def __init__(self):
        self.verts: list = []
        self.indices: list = []

    # -- low level ------------------------------------------------------
    def _vertex(self, p, n, c) -> int:
        self.verts.append((p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2]))
        return len(self.verts) - 1

    def add_triangle(self, p0, p1, p2, color, normal=None):
        p0, p1, p2 = _v(p0), _v(p1), _v(p2)
        cr = _cross(_sub(p1, p0), _sub(p2, p0))
        if normal is None:
            n = _n(cr)
        else:
            n = _n(normal)
            if _dot(cr, n) < 0.0:
                p1, p2 = p2, p1
        base = len(self.verts)
        for p in (p0, p1, p2):
            self._vertex(p, n, color)
        self.indices += [base, base + 1, base + 2]

    def add_quad(self, p0, p1, p2, p3, color, normal=None):
        p0, p1, p2, p3 = _v(p0), _v(p1), _v(p2), _v(p3)
        cr = _cross(_sub(p1, p0), _sub(p2, p0))
        if normal is None:
            n = _n(cr)
        else:
            n = _n(normal)
            if _dot(cr, n) < 0.0:
                # reverse the cycle while keeping the vertex order p0..p3
                p1, p3 = p3, p1
        base = len(self.verts)
        for p in (p0, p1, p2, p3):
            self._vertex(p, n, color)
        self.indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    def add_quad_normals(self, pts, norms, color):
        """Quad with a distinct normal per vertex (smooth shading)."""
        pts = [_v(p) for p in pts]
        norms = [_n(n) for n in norms]
        avg = _n((sum(n[0] for n in norms),
                  sum(n[1] for n in norms),
                  sum(n[2] for n in norms)))
        cr = _cross(_sub(pts[1], pts[0]), _sub(pts[2], pts[0]))
        if _dot(cr, avg) < 0.0:
            pts = [pts[0], pts[3], pts[2], pts[1]]
            norms = [norms[0], norms[3], norms[2], norms[1]]
        base = len(self.verts)
        for p, n in zip(pts, norms):
            self._vertex(p, n, color)
        self.indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    # -- solids ---------------------------------------------------------
    def add_box(self, cx, cz, y0, sx, sz, h, yaw, color):
        """Axis-aligned (yaw-rotated) box with all six faces, wound outward."""
        hx, hz = sx * 0.5, sz * 0.5
        corners = [(cx + rx, cz + rz)
                   for dx, dz in ((-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz))
                   for rx, rz in (rotate_xz(dx, dz, yaw),)]
        y1 = y0 + h
        top = [(x, y1, z) for x, z in corners]
        bot = [(x, y0, z) for x, z in corners]
        self.add_quad(top[0], top[1], top[2], top[3], color, (0.0, 1.0, 0.0))
        self.add_quad(bot[0], bot[3], bot[2], bot[1], color, (0.0, -1.0, 0.0))
        for i in range(4):
            j = (i + 1) % 4
            dx = corners[j][0] - corners[i][0]
            dz = corners[j][1] - corners[i][1]
            out = _n((dz, 0.0, -dx))
            self.add_quad(bot[i], bot[j], top[j], top[i], color, out)

    def add_extruded_polygon(self, poly_xz, y0, y1, color, cap=True):
        """Extrude a simple polygon (list of ``(x, z)``) from ``y0`` to ``y1``."""
        poly = [(float(x), float(z)) for x, z in poly_xz]
        if _signed_area(poly) < 0.0:
            poly.reverse()
        n = len(poly)
        bot = [(x, y0, z) for x, z in poly]
        top = [(x, y1, z) for x, z in poly]
        for i in range(n):
            j = (i + 1) % n
            dx = poly[j][0] - poly[i][0]
            dz = poly[j][1] - poly[i][1]
            out = _n((dz, 0.0, -dx))
            self.add_quad(bot[i], bot[j], top[j], top[i], color, out)
        if cap:
            cx = sum(p[0] for p in poly) / n
            cz = sum(p[1] for p in poly) / n
            ctop = (cx, y1, cz)
            cbot = (cx, y0, cz)
            for i in range(n):
                j = (i + 1) % n
                self.add_triangle(ctop, top[i], top[j], color, (0.0, 1.0, 0.0))
                self.add_triangle(cbot, bot[j], bot[i], color, (0.0, -1.0, 0.0))

    def add_rounded_box(self, cx, cz, y0, sx, sz, h, yaw, color, radius=0.35,
                        corner_seg=3):
        """Box with beveled vertical edges (good for crates and buildings)."""
        profile = rounded_rect_profile(sx, sz, radius, corner_seg)
        c, s = math.cos(yaw), math.sin(yaw)
        poly = [(cx + x * c - z * s, cz + x * s + z * c) for x, z in profile]
        self.add_extruded_polygon(poly, y0, y0 + h, color)

    def add_cylinder(self, cx, cz, y0, radius, h, color, segments=16,
                     top_radius=None, cap_top=True, cap_bottom=True):
        tr = radius if top_radius is None else top_radius
        y1 = y0 + h
        rb, rt = [], []
        for k in range(segments):
            a = 2.0 * math.pi * k / segments
            ca, sa = math.cos(a), math.sin(a)
            rb.append((cx + radius * ca, y0, cz + radius * sa))
            rt.append((cx + tr * ca, y1, cz + tr * sa))
        apex = tr < 1e-6
        for k in range(segments):
            k2 = (k + 1) % segments
            mid = 2.0 * math.pi * (k + 0.5) / segments
            nrm = (math.cos(mid), 0.0, math.sin(mid))
            if apex:
                self.add_triangle(rb[k], rb[k2], (cx, y1, cz), color, nrm)
            else:
                self.add_quad(rb[k], rb[k2], rt[k2], rt[k], color, nrm)
        if cap_top and not apex:
            c = (cx, y1, cz)
            for k in range(segments):
                k2 = (k + 1) % segments
                self.add_triangle(c, rt[k], rt[k2], color, (0.0, 1.0, 0.0))
        if cap_bottom and radius > 1e-6:
            c = (cx, y0, cz)
            for k in range(segments):
                k2 = (k + 1) % segments
                self.add_triangle(c, rb[k2], rb[k], color, (0.0, -1.0, 0.0))

    def add_cone(self, cx, cz, y0, radius, h, color, segments=16, cap_bottom=True):
        self.add_cylinder(cx, cz, y0, radius, h, color, segments,
                          top_radius=0.0, cap_bottom=cap_bottom)

    def add_sphere(self, cx, cy, cz, r, color, segments=16, rings=10):
        pts = []
        for i in range(rings + 1):
            phi = math.pi * i / rings
            pts.append((math.sin(phi), math.cos(phi)))
        for i in range(rings):
            sp, cp = pts[i]
            sq, cq = pts[i + 1]
            for k in range(segments):
                a0 = 2.0 * math.pi * k / segments
                a1 = 2.0 * math.pi * (k + 1) / segments
                p00 = (cx + r * sp * math.cos(a0), cy + r * cp, cz + r * sp * math.sin(a0))
                p01 = (cx + r * sp * math.cos(a1), cy + r * cp, cz + r * sp * math.sin(a1))
                p11 = (cx + r * sq * math.cos(a1), cy + r * cq, cz + r * sq * math.sin(a1))
                p10 = (cx + r * sq * math.cos(a0), cy + r * cq, cz + r * sq * math.sin(a0))
                mid = 0.5 * (a0 + a1)
                sm, cm2 = 0.5 * (sp + sq), 0.5 * (cp + cq)
                nrm = (sm * math.cos(mid), cm2, sm * math.sin(mid))
                self.add_quad(p00, p01, p11, p10, color, nrm)

    def add_revolve_z(self, profile, color, segments=20, center=(0.0, 0.0),
                      close=True):
        """Revolve a ``(radius, z)`` cross-section around the Z axis.

        Used for wheels (tyres, rims, brake discs).  Outward normals are
        derived from the profile centroid, so any convex closed profile works.
        """
        prof = [(float(r), float(z)) for r, z in profile]
        if close:
            prof = prof + [prof[0]]
        m = len(prof)
        rc = sum(p[0] for p in prof) / m
        zc = sum(p[1] for p in prof) / m
        cx, cy = float(center[0]), float(center[1])
        for i in range(m - 1):
            r0, z0 = prof[i]
            r1, z1 = prof[i + 1]
            rm, zm = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
            for k in range(segments):
                a0 = 2.0 * math.pi * k / segments
                a1 = 2.0 * math.pi * (k + 1) / segments
                p00 = (cx + r0 * math.cos(a0), cy + r0 * math.sin(a0), z0)
                p01 = (cx + r1 * math.cos(a0), cy + r1 * math.sin(a0), z1)
                p11 = (cx + r1 * math.cos(a1), cy + r1 * math.sin(a1), z1)
                p10 = (cx + r0 * math.cos(a1), cy + r0 * math.sin(a1), z0)
                mid = 0.5 * (a0 + a1)
                nr = rm - rc
                nrm = (nr * math.cos(mid), nr * math.sin(mid), zm - zc)
                if abs(nrm[0]) + abs(nrm[1]) + abs(nrm[2]) < 1e-9:
                    continue
                self.add_quad(p00, p01, p11, p10, color, nrm)

    # -- swept surfaces -------------------------------------------------
    def add_loft(self, rings, color, closed_profile=True):
        """Skin a list of equally-sized rings with smooth per-vertex normals.

        ``rings`` is ``list[list[(x, y, z)]]``; ``rings[k][i]`` is vertex *i*
        of station *k*.  Normals are computed from the two surface tangents
        and oriented away from the ring centroid.
        """
        rings = [[_v(p) for p in ring] for ring in rings]
        ns = len(rings)
        if ns < 2:
            return
        m = len(rings[0])
        centroids = [(sum(p[0] for p in r) / len(r),
                      sum(p[1] for p in r) / len(r),
                      sum(p[2] for p in r) / len(r)) for r in rings]
        count = m if closed_profile else m - 1
        normals = [[None] * m for _ in range(ns)]
        for k in range(ns):
            kp = min(max(k - 1, 0), ns - 1)
            kn = min(max(k + 1, 0), ns - 1)
            for i in range(m):
                ip = (i - 1) % m
                inx = (i + 1) % m
                t_c = _sub(rings[k][inx], rings[k][ip])
                t_p = _sub(rings[kn][i], rings[kp][i])
                outward = _sub(rings[k][i], centroids[k])
                nrm = _cross(t_p, t_c)
                if _dot(nrm, nrm) < 1e-18:      # degenerate: use the centroid
                    nrm = outward
                if _dot(nrm, outward) < 0.0:
                    nrm = (-nrm[0], -nrm[1], -nrm[2])
                normals[k][i] = _n(nrm)
        for k in range(ns - 1):
            for i in range(count):
                j = (i + 1) % m
                self.add_quad_normals(
                    [rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]],
                    [normals[k][i], normals[k][j], normals[k + 1][j], normals[k + 1][i]],
                    color)

    def add_ribbon(self, left, right, color, normal=(0.0, 1.0, 0.0)):
        """Quad strip between two equal-length polylines (ground decals)."""
        for i in range(len(left) - 1):
            self.add_quad(right[i], left[i], left[i + 1], right[i + 1], color, normal)

    def add_disc_y(self, cx, cy, cz, radius, color, segments=20, up=True):
        """Horizontal disc (blob shadows, start line dots)."""
        n = (0.0, 1.0, 0.0) if up else (0.0, -1.0, 0.0)
        c = (cx, cy, cz)
        for k in range(segments):
            a0 = 2.0 * math.pi * k / segments
            a1 = 2.0 * math.pi * (k + 1) / segments
            p0 = (cx + radius * math.cos(a0), cy, cz + radius * math.sin(a0))
            p1 = (cx + radius * math.cos(a1), cy, cz + radius * math.sin(a1))
            self.add_triangle(c, p0, p1, color, n)

    def add_disc_z(self, cx, cy, cz, radius, color, segments=20, forward=True):
        """Disc in the XY plane (wheel rims/caps)."""
        n = (0.0, 0.0, 1.0) if forward else (0.0, 0.0, -1.0)
        c = (cx, cy, cz)
        for k in range(segments):
            a0 = 2.0 * math.pi * k / segments
            a1 = 2.0 * math.pi * (k + 1) / segments
            p0 = (cx + radius * math.cos(a0), cy + radius * math.sin(a0), cz)
            p1 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1), cz)
            self.add_triangle(c, p0, p1, color, n)


# ---------------------------------------------------------------------------
def _signed_area(poly) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        a += x0 * z1 - x1 * z0
    return 0.5 * a


def rounded_rect_profile(sx: float, sz: float, radius: float, corner_seg: int = 3):
    """Closed ``(x, z)`` polygon of a rounded rectangle centred on the origin."""
    hx, hz = sx * 0.5, sz * 0.5
    r = max(0.0, min(radius, hx - 1e-4, hz - 1e-4))
    if r <= 1e-6:
        return [(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)]
    pts = []
    corners = [(hx - r, hz - r, 0.0), (-hx + r, hz - r, math.pi / 2),
               (-hx + r, -hz + r, math.pi), (hx - r, -hz + r, 1.5 * math.pi)]
    for ccx, ccz, a0 in corners:
        for s in range(corner_seg + 1):
            a = a0 + (math.pi / 2) * s / corner_seg
            pts.append((ccx + r * math.cos(a), ccz + r * math.sin(a)))
    return pts


def superellipse_profile(half_width: float, y_bottom: float, y_top: float,
                         n: float = 4.0, segments: int = 16):
    """Closed ``(y, z)`` cross-section: a rounded-rectangle superellipse."""
    cy = 0.5 * (y_bottom + y_top)
    hy = 0.5 * (y_top - y_bottom)
    hw = max(half_width, 1e-4)
    pts = []
    for k in range(segments):
        a = 2.0 * math.pi * k / segments
        ca, sa = math.cos(a), math.sin(a)
        z = hw * math.copysign(abs(ca) ** (2.0 / n), ca)
        y = cy + hy * math.copysign(abs(sa) ** (2.0 / n), sa)
        pts.append((y, z))
    return pts


def loft_stations(stations, n: float = 4.0, segments: int = 16):
    """Build ``rings`` for :meth:`GeoBuffer.add_loft` from a station list.

    Each station is ``(x, half_width, y_bottom, y_top)``; the car faces +X,
    so rings advance along X and each profile lives in the (Y, Z) plane.
    """
    rings = []
    for x, hw, yb, yt in stations:
        ring = [(x, y, z) for y, z in superellipse_profile(hw, yb, yt, n, segments)]
        rings.append(ring)
    return rings
