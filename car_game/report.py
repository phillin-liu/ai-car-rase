"""PDF training report.

Charts are drawn with **Pillow only** (no matplotlib) and every finished page
is embedded losslessly into a minimal hand-written PDF.  All text -- including
Chinese -- is drawn into the page image with a CJK system font, so the PDF
needs no embedded font and renders identically everywhere.

Contents of a report:

* overview: models, W/L, cross-match summary table, average-speed bars
* speed curves: one chart per match (both drivers), with a small stats table
* AI log: per-match request/error/latency/success + recent decisions + events
* capability share: stacked bars of the five subscores per side
* score breakdown: pie of the weighted score contributions per side
"""
from __future__ import annotations

import math
import os
import pathlib
import zlib
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

# A4 at 150 dpi
PW, PH = 1240, 1754
MARGIN = 84
CONTENT_W = PW - 2 * MARGIN

C_INK = (28, 32, 38)
C_DIM = (110, 118, 130)
C_GRID = (216, 220, 226)
C_BORDER = (188, 194, 202)
C_P1 = (198, 58, 52)
C_P2 = (46, 104, 190)
C_BG = (255, 255, 255)
C_HEAD = (240, 242, 246)

CAP_WEIGHTS = {"完成度": 0.35, "驾驶质量": 0.25, "平均速度": 0.15,
               "道具运用": 0.13, "稳定性": 0.12}
CAP_COLORS = {
    "完成度": (69, 133, 200),
    "驾驶质量": (232, 138, 62),
    "平均速度": (96, 178, 108),
    "道具运用": (196, 96, 178),
    "稳定性": (222, 196, 74),
}

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _font_path() -> str | None:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


_FONT_CACHE: dict = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _font_path()
    f = None
    if path:
        for idx in ((1,) if bold and path.endswith(".ttc") else (0,)) + (0,):
            try:
                f = ImageFont.truetype(path, size, index=idx)
                break
            except Exception:
                continue
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


# ---------------------------------------------------------------------------
class Page:
    """A single A4 page with a flowing layout cursor."""

    def __init__(self, title: str, subtitle: str = ""):
        self.img = Image.new("RGB", (PW, PH), C_BG)
        self.d = ImageDraw.Draw(self.img)
        self.y = MARGIN
        self.title = title
        self.subtitle = subtitle
        self._header()

    def _header(self):
        self.d.text((MARGIN, self.y), self.title, font=font(30, True), fill=C_INK)
        self.y += 42
        if self.subtitle:
            self.d.text((MARGIN, self.y), self.subtitle, font=font(15), fill=C_DIM)
            self.y += 26
        self.d.line([(MARGIN, self.y), (PW - MARGIN, self.y)], fill=C_BORDER, width=2)
        self.y += 22

    # -- text -----------------------------------------------------------
    def heading(self, text: str):
        self._ensure(46)
        self.y += 6
        self.d.text((MARGIN, self.y), text, font=font(21, True), fill=C_INK)
        self.y += 34

    def para(self, text: str, size: int = 16, color=C_INK, gap: int = 6):
        # wrap first and check the *whole* block fits, otherwise a paragraph
        # that overflows would be redrawn from the top on the next page
        lines = _wrap(self.d, text, font(size), CONTENT_W)
        self._ensure(len(lines) * (size + 8) + gap)
        for line in lines:
            self.d.text((MARGIN, self.y), line, font=font(size), fill=color)
            self.y += size + 8
        self.y += gap

    def kv(self, pairs, size: int = 16, cols: int = 2):
        col_w = CONTENT_W // cols
        row_h = size + 12
        rows = math.ceil(len(pairs) / cols)
        self._ensure(rows * row_h + 8)
        for i, (k, v) in enumerate(pairs):
            x = MARGIN + (i % cols) * col_w
            self.d.text((x, self.y), f"{k}", font=font(size), fill=C_DIM)
            self.d.text((x + 150, self.y), f"{v}", font=font(size, True), fill=C_INK)
            if i % cols == cols - 1 or i == len(pairs) - 1:
                self.y += row_h
        self.y += 8

    # -- tables ---------------------------------------------------------
    def table(self, headers, rows, widths=None, size: int = 15):
        if not rows:
            self.para("（无数据）", color=C_DIM)
            return
        widths = widths or _even_widths(len(headers), CONTENT_W)
        row_h = size + 16
        self._ensure(row_h * (len(rows) + 1) + 8)
        self.d.rectangle([MARGIN, self.y, MARGIN + CONTENT_W, self.y + row_h],
                         fill=C_HEAD, outline=C_BORDER)
        x = MARGIN
        for h, w in zip(headers, widths):
            self.d.text((x + 8, self.y + 8), str(h), font=font(size, True), fill=C_INK)
            x += w
        self.y += row_h
        for r, row in enumerate(rows):
            if r % 2 == 1:
                self.d.rectangle([MARGIN, self.y, MARGIN + CONTENT_W, self.y + row_h],
                                 fill=(248, 249, 251))
            x = MARGIN
            for val, w in zip(row, widths):
                self.d.text((x + 8, self.y + 8), _ellipsis(self.d, val, font(size), w - 16),
                            font=font(size), fill=C_INK)
                x += w
            self.d.line([(MARGIN, self.y + row_h), (MARGIN + CONTENT_W, self.y + row_h)],
                        fill=C_GRID, width=1)
            self.y += row_h
        self.y += 12

    # -- charts ---------------------------------------------------------
    def chart(self, height: int, draw_fn):
        """Reserve a box and let ``draw_fn(draw, rect)`` paint it."""
        self._ensure(height + 16)
        rect = (MARGIN, self.y, MARGIN + CONTENT_W, self.y + height)
        draw_fn(self.d, rect)
        self.y += height + 16

    def _ensure(self, needed: int):
        if self.y + needed > PH - MARGIN:
            raise _PageFull()


class _PageFull(Exception):
    def __init__(self, title=None):
        super().__init__("page full")
        self.title = title


# ---------------------------------------------------------------------------
def _even_widths(n: int, total: int):
    w = total // n
    return [w] * (n - 1) + [total - w * (n - 1)]


def _ellipsis(draw, text, f, max_w: int) -> str:
    text = str(text)
    if draw.textlength(text, font=f) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=f) > max_w:
        text = text[:-1]
    return text + "…"


def _wrap(draw, text, f, max_w: int):
    """Character-wise wrap (CJK has no spaces to break on)."""
    text = str(text)
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if draw.textlength(cur + ch, font=f) > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines or [""]


# ---------------------------------------------------------------------------
def _legend(draw, x, y, entries, size=14, gap=170):
    for i, (label, color) in enumerate(entries):
        cx = x + i * gap
        draw.rectangle([cx, y + 2, cx + 14, y + 16], fill=color, outline=C_BORDER)
        draw.text((cx + 20, y), label, font=font(size), fill=C_INK)


def draw_speed_curves(draw, rect, series, names, colors, title=""):
    """Line chart of speed (km/h) over race time for both drivers."""
    x0, y0, x1, y1 = rect
    pad_l, pad_b, pad_t = 62, 34, 30
    plot = (x0 + pad_l, y0 + pad_t, x1 - 12, y1 - pad_b)
    times = series.get("t") or []
    speeds = series.get("speed") or []
    draw.rectangle([x0, y0, x1, y1], outline=C_BORDER)
    if title:
        draw.text((x0 + 10, y0 + 6), title, font=font(15, True), fill=C_INK)
    if not times or len(speeds) < 2:
        draw.text((plot[0] + 10, plot[1] + 10), "（本局没有记录车速序列）",
                  font=font(14), fill=C_DIM)
        return
    kmh = [[s * 3.6 for s in side] for side in speeds]
    vmax = max([max(v) for v in kmh if v] or [1.0])
    vmax = max(20.0, math.ceil(vmax / 20.0) * 20.0)
    tmax = max(times) or 1.0

    for k in range(5):                       # horizontal grid + y labels
        gy = plot[1] + (plot[3] - plot[1]) * k / 4
        draw.line([(plot[0], gy), (plot[2], gy)], fill=C_GRID, width=1)
        draw.text((x0 + 8, gy - 8), f"{vmax * (4 - k) / 4:.0f}", font=font(12), fill=C_DIM)
    draw.text((x0 + 8, plot[1] - 22), "km/h", font=font(12), fill=C_DIM)
    draw.line([(plot[0], plot[3]), (plot[2], plot[3])], fill=C_BORDER, width=1)
    draw.line([(plot[0], plot[1]), (plot[0], plot[3])], fill=C_BORDER, width=1)
    draw.text((plot[2] - 40, plot[3] + 10), f"{tmax:.0f}s", font=font(12), fill=C_DIM)

    def px(t, v):
        return (plot[0] + (t / tmax) * (plot[2] - plot[0]),
                plot[3] - (v / vmax) * (plot[3] - plot[1]))

    step = max(1, len(times) // 600)
    for i, side in enumerate(kmh[:2]):
        pts = [px(times[j], side[j]) for j in range(0, min(len(times), len(side)), step)]
        if len(pts) > 1:
            draw.line(pts, fill=colors[i], width=2, joint="curve")
    _legend(draw, plot[0], y1 - 24,
            [(names[i] if i < len(names) else f"P{i+1}", colors[i]) for i in range(2)],
            size=13, gap=180)


def draw_speed_bars(draw, rect, labels, values, colors, unit="km/h"):
    """Simple two-bar comparison (average speed)."""
    x0, y0, x1, y1 = rect
    draw.rectangle([x0, y0, x1, y1], outline=C_BORDER)
    draw.text((x0 + 10, y0 + 6), f"平均车速 ({unit})", font=font(15, True), fill=C_INK)
    if not values:
        return
    base = y1 - 34
    top = y0 + 40
    vmax = max(values + [1.0]) * 1.2
    n = len(values)
    bw = min(110, (x1 - x0 - 80) // max(1, n) - 30)
    for i, v in enumerate(values):
        cx = x0 + 70 + i * ((x1 - x0 - 120) // max(1, n))
        h = int((v / vmax) * (base - top))
        draw.rectangle([cx, base - h, cx + bw, base], fill=colors[i % len(colors)],
                       outline=C_BORDER)
        draw.text((cx, base - h - 22), f"{v:.1f}", font=font(14, True), fill=C_INK)
        draw.text((cx, base + 6), _ellipsis(draw, labels[i], font(13), bw + 60),
                  font=font(13), fill=C_DIM)


def draw_capability_bars(draw, rect, entries, names):
    """Stacked horizontal bars: each side's five capabilities, proportional."""
    x0, y0, x1, y1 = rect
    draw.rectangle([x0, y0, x1, y1], outline=C_BORDER)
    draw.text((x0 + 10, y0 + 6), "能力占比（分项得分构成）", font=font(15, True), fill=C_INK)
    caps = list(CAP_WEIGHTS)
    bar_x, bar_w = x0 + 140, x1 - x0 - 180
    for i, subs in enumerate(entries[:2]):
        by = y0 + 52 + i * 78
        label = names[i] if i < len(names) else f"P{i+1}"
        draw.text((x0 + 12, by + 14),
                  _ellipsis(draw, label, font(14, True), bar_x - x0 - 24),
                  font=font(14, True), fill=C_INK)
        vals = [max(0.0, float(subs.get(c, 0.0))) for c in caps]
        total = sum(vals) or 1.0
        cx = bar_x
        for c, v in zip(caps, vals):
            w = int(bar_w * v / total)
            if w <= 0:
                continue
            draw.rectangle([cx, by, cx + w, by + 34], fill=CAP_COLORS[c], outline=C_BORDER)
            if w > 52:
                draw.text((cx + 6, by + 8), f"{v / total * 100:.0f}%", font=font(13),
                          fill=(255, 255, 255))
            cx += w
        draw.rectangle([bar_x, by, bar_x + bar_w, by + 34], outline=C_BORDER)
    _legend(draw, x0 + 12, y1 - 34,
            [(c, CAP_COLORS[c]) for c in caps], size=13, gap=118)
    return caps


def draw_score_pies(draw, rect, entries, names):
    """One pie per side: the weighted contribution of each capability."""
    x0, y0, x1, y1 = rect
    draw.rectangle([x0, y0, x1, y1], outline=C_BORDER)
    draw.text((x0 + 10, y0 + 6), "得分点构成（分项 × 权重）", font=font(15, True), fill=C_INK)
    caps = list(CAP_WEIGHTS)
    # leave room for the legend row *and* the side label under each pie
    size = min(y1 - y0 - 120, (x1 - x0) // 2 - 40)
    for i, subs in enumerate(entries[:2]):
        cx = x0 + 40 + i * ((x1 - x0) // 2) + size / 2
        cy = y0 + 46 + size / 2
        box = [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]
        contrib = [max(0.0, float(subs.get(c, 0.0))) * CAP_WEIGHTS[c] for c in caps]
        total = sum(contrib) or 1.0
        start = -90.0
        for c, v in zip(caps, contrib):
            end = start + 360.0 * v / total
            draw.pieslice(box, start, end, fill=CAP_COLORS[c], outline=(255, 255, 255),
                          width=2)
            mid = math.radians((start + end) / 2)
            lx = cx + math.cos(mid) * size * 0.33
            ly = cy + math.sin(mid) * size * 0.33
            pct = v / total * 100
            if pct >= 7:
                txt = f"{pct:.0f}%"
                w = draw.textlength(txt, font=font(13, True))
                draw.text((lx - w / 2, ly - 8), txt, font=font(13, True), fill=(255, 255, 255))
            start = end
        label = names[i] if i < len(names) else f"P{i+1}"
        lw = draw.textlength(label, font=font(14, True))
        draw.text((cx - lw / 2, cy + size / 2 + 8), label, font=font(14, True), fill=C_INK)
    _legend(draw, x0 + 12, y1 - 30, [(c, CAP_COLORS[c]) for c in caps], size=13, gap=118)


# ---------------------------------------------------------------------------
class PdfWriter:
    """Minimal PDF 1.4 writer with one lossless RGB image per page."""

    def __init__(self):
        self.pages = []

    def add_page(self, img: Image.Image):
        img = img.convert("RGB")
        self.pages.append((zlib.compress(img.tobytes(), 6), img.width, img.height))

    def save(self, path: str):
        pw, ph = 595.276, 841.89        # A4 points
        n = len(self.pages)
        objs: dict[int, bytes] = {}
        page_ids = [3 + 3 * i for i in range(n)]
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
        for i, (data, w, h) in enumerate(self.pages):
            pid, cid, iid = 3 + 3 * i, 4 + 3 * i, 5 + 3 * i
            content = f"q {pw:.2f} 0 0 {ph:.2f} 0 0 cm /Im0 Do Q".encode()
            objs[pid] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
                         f"/Resources << /XObject << /Im0 {iid} 0 R >> >> "
                         f"/Contents {cid} 0 R >>").encode()
            objs[cid] = (b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
                         + content + b"\nendstream")
            objs[iid] = (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
                         f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
                         f"/Filter /FlateDecode /Length {len(data)} >>\nstream\n").encode() \
                + data + b"\nendstream"
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        for oid in sorted(objs):
            offsets[oid] = len(out)
            out += f"{oid} 0 obj\n".encode() + objs[oid] + b"\nendobj\n"
        xref = len(out)
        top = max(objs)
        out += f"xref\n0 {top + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for oid in range(1, top + 1):
            out += (f"{offsets[oid]:010d} 00000 n \n".encode() if oid in offsets
                    else b"0000000000 65535 f \n")
        out += (f"trailer\n<< /Size {top + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode()
        pathlib.Path(path).write_bytes(bytes(out))


# ---------------------------------------------------------------------------
class _Doc:
    """Page-flow helper: appends pages, rolling over when one fills up."""

    def __init__(self, title: str, subtitle: str, writer: PdfWriter):
        self.title = title
        self.subtitle = subtitle
        self.writer = writer
        self.page = Page(title, subtitle)

    def guard(self, method: str, *a, **kw):
        """Call ``page.<method>(*a)``; on overflow roll to a fresh page.

        The method name is looked up on the *current* page each attempt --
        passing a bound method here would keep drawing into the page that was
        already flushed, silently losing the content and emitting a blank page.
        """
        for _ in range(4):
            try:
                return getattr(self.page, method)(*a, **kw)
            except _PageFull:
                self.flush()
                self.page = Page(f"{self.title}（续）", self.subtitle)
        return None

    def flush(self):
        self.writer.add_page(self.page.img)


# ---------------------------------------------------------------------------
def build_report(store, path: str) -> str:
    """Build a PDF training report for the current session and return ``path``."""
    matches = store.list_matches()
    agg = store.aggregate()
    writer = PdfWriter()
    title = "训练报告"
    doc = _Doc(title, f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}", writer)

    names = _side_names(matches, agg)
    # ---------------- overview ----------------
    counts = _win_counts(matches)
    doc.guard("heading", "总览")
    doc.guard("kv", [
        ("对局场次", f"{len(matches)}"),
        ("P1", names[0]),
        ("P2", names[1]),
        ("P1 胜 / 平 / P2 胜", f"{counts[0]} / {counts[2]} / {counts[1]}"),
        ("统计口径", "跨局平均"),
    ], cols=2)

    doc.guard("heading", "车型/模型总体表现")
    headers = ["模型", "场次", "胜率%", "完赛率%", "平均分", "平均速度", "撞墙", "撞障碍"]
    rows = [[r.get("model"), r.get("matches"), r.get("win_rate"),
             r.get("finish_rate"), r.get("avg_score"), r.get("avg_speed"),
             r.get("avg_wall"), r.get("avg_obs")] for r in agg]
    doc.guard("table", headers, rows)

    doc.guard("chart", 250, lambda d, r: draw_speed_bars(
        d, r, [str(x.get("model")) for x in agg[:2]],
        [float(x.get("avg_speed") or 0.0) for x in agg[:2]], [C_P1, C_P2]))

    # ---------------- capability + score pies ----------------
    subs = [x.get("subscores") or {} for x in agg[:2]]
    while len(subs) < 2:
        subs.append({})
    doc.guard("heading", "能力与得分构成")
    doc.guard("chart", 240,
              lambda d, r: draw_capability_bars(d, r, subs, names))
    doc.guard("chart", 330,
              lambda d, r: draw_score_pies(d, r, subs, names))

    # ---------------- per-match detail ----------------
    for m in matches:
        try:
            payload = store.load_match(int(m["match_id"]))
        except Exception:
            continue
        analysis = payload.get("analysis") or {}
        result = payload.get("result") or {}
        series = result.get("series") or {}
        exams = analysis.get("analyses") or []
        doc.guard("heading",
                  f"第 {m.get('match_id')} 局 · "
                  f"{analysis.get('duration', '-')}s")

        def _chart(d, r, series=series, exams=exams):
            draw_speed_curves(d, r, series,
                              [a.get("name", "") for a in exams] or ["P1", "P2"],
                              [C_P1, C_P2])
        doc.guard("chart", 260, _chart)

        rows = []
        for a in exams[:2]:
            mt = a.get("metrics", {})
            rows.append([a.get("name"), a.get("model"),
                         mt.get("平均速度"), mt.get("最高速度"),
                         mt.get("撞墙次数"), mt.get("撞障碍物次数"),
                         f"{a.get('total')} ({a.get('grade')})"])
        doc.guard("table",
                  ["", "模型", "平均车速", "最高车速", "撞墙", "撞障碍", "评分"],
                  rows, widths=[70, 300, 120, 120, 90, 100, 272])

    # ---------------- AI log ----------------
    doc.guard("heading", "AI 日志")
    log_rows = []
    event_lines = []
    for m in matches:
        try:
            payload = store.load_match(int(m["match_id"]))
        except Exception:
            continue
        result = payload.get("result") or {}
        for i, d in enumerate(result.get("driver_stats") or []):
            # "local" is kept so reports of matches recorded before the
            # built-in side was removed still render their rows
            if not isinstance(d, dict) or d.get("driver") not in ("llm", "local"):
                continue
            kind = d.get("driver")
            model_cell = d.get("model")
            if kind == "local":
                det = (d.get("vision") or {})
                det_name = det.get("detector") if isinstance(det, dict) else None
                model_cell = f"{d.get('model')} (本地视觉{':' + det_name if det_name else ''})"
            log_rows.append([
                f"第{m.get('match_id')}局 P{i+1}", model_cell,
                d.get("api_requests") if kind == "llm" else "—",
                d.get("api_errors") if kind == "llm" else "—",
                f"{d.get('avg_latency_ms', 0)}ms" if kind == "llm" else "—",
                (f"{float(d.get('success_rate', 1.0)) * 100:.0f}%"
                 if kind == "llm" else "—"),
                str(d.get("last_reason", ""))[:46],
            ])
        for t, text in (result.get("events") or [])[-8:]:
            event_lines.append(f"[第{m.get('match_id')}局 {t:>5.1f}s] {text}")
    doc.guard("para",
              "模型调用统计（每位玩家每局一行）。标注为「本地视觉」的历史记录来自"
              "程序还自带离线一侧的版本，不产生外部调用，因此请求/延迟一栏为空。",
              size=15, color=C_DIM)
    doc.guard("table",
              ["", "模型", "请求", "错误", "平均延迟", "成功率", "最近决策"],
              log_rows or [], widths=[120, 240, 80, 80, 130, 110, 312])
    if event_lines:
        doc.guard("heading", "事件日志（各局最近若干条）")
        doc.guard("para", "\n".join(event_lines[-40:]), size=14)

    doc.flush()
    writer.save(path)
    return path


def _side_names(matches, agg):
    for m in reversed(matches):
        models = m.get("models") or []
        if len(models) >= 2:
            return [str(models[0]), str(models[1])]
    names = [str(r.get("model")) for r in agg[:2]]
    while len(names) < 2:
        names.append("P" + str(len(names) + 1))
    return names


def _win_counts(matches):
    p1 = p2 = draw = 0
    for m in matches:
        w = m.get("winner")
        if w == 0:
            p1 += 1
        elif w == 1:
            p2 += 1
        else:
            draw += 1
    return p1, p2, draw
