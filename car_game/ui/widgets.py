"""Reusable console widgets.

Everything here is built from stock Qt widgets (``QGroupBox``, ``QLabel``,
``QProgressBar``, ``QTableWidget``) with no custom stylesheet, so the console
follows the Windows default theme.  The widgets exist to structure the data
-- cards, stat tiles, meters, tables -- not to re-skin it.
"""
from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QAbstractItemView, QFrame, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QListWidget, QProgressBar,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ..palette import CAR_COLORS
from .i18n import t
from .theme import SPACE, STATUS_COLORS, role_font


def card(title: str | None = None) -> QGroupBox:
    """A titled native group panel; the child layout is attached as ``.body``."""
    box = QGroupBox(title or "")
    if not title:
        box.setTitle("")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
    lay.setSpacing(SPACE["sm"])
    box.body = lay
    return box


def divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def collapsible(title: str, checked: bool = False):
    """A group box whose contents can be folded away.

    Returns ``(box, body_widget)``; lay the contents out on ``body_widget``.
    Used to keep the crowded tabs to a readable handful of controls by default
    -- the fields are still there, just one click away.
    """
    box = QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(checked)
    outer = QVBoxLayout(box)
    outer.setContentsMargins(SPACE["md"], SPACE["sm"], SPACE["md"], SPACE["md"])
    body = QWidget()
    inner = QVBoxLayout(body)
    inner.setContentsMargins(0, 0, 0, 0)
    inner.setSpacing(SPACE["sm"])
    outer.addWidget(body)
    box.body = inner

    def _sync(on: bool):
        body.setVisible(on)

    box.toggled.connect(_sync)
    _sync(checked)
    return box, body


def _qcolor(rgb) -> QColor:
    """A 0..1 float triple (car_game.palette) as a QColor."""
    return QColor(int(max(0.0, min(1.0, rgb[0])) * 255),
                  int(max(0.0, min(1.0, rgb[1])) * 255),
                  int(max(0.0, min(1.0, rgb[2])) * 255))


class StatTile(QGroupBox):
    """Small labelled number (used in the live scoreboard strip)."""

    def __init__(self, title: str, value: str = "—", unit: str = ""):
        super().__init__(title)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(4)
        self._value = QLabel(value)
        self._value.setFont(role_font("stat", bold=True))
        self._unit = QLabel(unit)
        row.addWidget(self._value)
        row.addWidget(self._unit, 0, Qt.AlignBottom)
        row.addStretch(1)
        lay.addLayout(row)

    def set_value(self, value: str, unit: str | None = None):
        self._value.setText(str(value))
        if unit is not None:
            self._unit.setText(unit)


class MetricChip(QWidget):
    """Inline ``label  value``; the value is bold.

    Two labels with real fonts rather than one rich-text label: an HTML ``<b>``
    is rendered through the rich-text engine, which picks its own face for CJK
    and its own bold synthesis, so a chip would not match the stat tiles and
    pills sitting next to it.
    """

    def __init__(self, label: str, value: str = "—"):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._label = QLabel(label)
        self._label.setFont(role_font("body"))
        self._value = QLabel(value)
        self._value.setFont(role_font("body", bold=True))
        row.addWidget(self._label)
        row.addWidget(self._value)
        row.addStretch(1)

    def set_value(self, value: str):
        self._value.setText(str(value))


class BarMeter(QProgressBar):
    """Plain native progress bar (0..1)."""

    def __init__(self):
        super().__init__()
        self.setRange(0, 1000)
        self.setTextVisible(False)
        self.setFixedHeight(16)

    def set_fraction(self, frac: float):
        self.setValue(int(max(0.0, min(1.0, frac)) * 1000))


class StatusPill(QLabel):
    """Bold coloured status text (no custom background)."""

    def __init__(self, text: str = "就绪", state: str = "idle"):
        text = t(text)
        super().__init__(text)
        self.setFont(role_font("body", bold=True))
        self.set_state(state)

    def set_state(self, state: str, text: str | None = None):
        self._state = state
        if text is not None:
            self.setText(text)
        self.setStyleSheet(f"color:{STATUS_COLORS.get(state, '#000000')};")


class TelemetryCard(QGroupBox):
    """One driver's live card: colour dot, speed, lap, collisions, meter."""

    def __init__(self, side: str, color: tuple):
        super().__init__(side)
        self._rgb = tuple(int(c * 255) for c in color)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        lay.setSpacing(SPACE["sm"])

        head = QHBoxLayout()
        head.setSpacing(SPACE["sm"])
        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        self._dot.setStyleSheet(
            f"background-color:rgb{self._rgb}; border:1px solid #555;")
        self._name = QLabel(side)
        self._name.setFont(role_font("body", bold=True))
        self._model = QLabel("")
        head.addWidget(self._dot)
        head.addWidget(self._name)
        head.addWidget(self._model, 1)
        self._tag = QLabel("")
        head.addWidget(self._tag)
        lay.addLayout(head)

        value_row = QHBoxLayout()
        value_row.setSpacing(6)
        self._speed = QLabel("0")
        self._speed.setFont(role_font("hero", bold=True))
        self._kmh = QLabel("km/h")
        value_row.addWidget(self._speed)
        value_row.addWidget(self._kmh, 0, Qt.AlignBottom)
        value_row.addStretch(1)
        self._laps = QLabel(t("圈 —"))
        value_row.addWidget(self._laps, 0, Qt.AlignBottom)
        lay.addLayout(value_row)

        self._meter = BarMeter()
        lay.addWidget(self._meter)

        chips = QHBoxLayout()
        chips.setSpacing(SPACE["lg"])
        self._wall = MetricChip(t("撞墙"), "0")
        self._obs = MetricChip(t("撞障碍"), "0")
        self._items = MetricChip(t("道具"), "0")
        for c in (self._wall, self._obs, self._items):
            chips.addWidget(c)
        chips.addStretch(1)
        lay.addLayout(chips)

    def set_title(self, name: str, model: str):
        self._name.setText(name)
        self._model.setText(model)

    def update_from(self, car: dict, laps: int):
        self._speed.setText(f"{car.get('speed', 0.0) * 3.6:.0f}")
        lap_now = min(int(car.get("laps", 0)) + (0 if car.get("finished") else 1), laps)
        self._laps.setText(t("✔ 完赛") if car.get("finished")
                           else t("圈 {lap}/{laps}", lap=lap_now, laps=laps))
        self._wall.set_value(str(car.get("wall_hits", 0)))
        self._obs.set_value(str(car.get("obstacle_hits", 0)))
        self._items.set_value(str(car.get("item_uses", 0)))
        self._meter.set_fraction(float(car.get("speed", 0.0)) / 45.0)
        effect = str(car.get("effect") or "")
        tag = {"boost": t("氮气"), "shield": t("护盾"),
               "slow": t("减速")}.get(effect, "")
        if car.get("finished"):
            tag = t("✔ 完赛")
        if not tag:
            tag = str(car.get("model", ""))[:22]
        self._tag.setText(tag)


class EmptyState(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)


class ComparisonTable(QTableWidget):
    """Renders a match's comparison rows as a proper table."""

    def __init__(self):
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels([t("指标"), "P1", "P2"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

    def load(self, analysis: dict):
        exams = analysis.get("analyses", [])
        self.setRowCount(0)
        if len(exams) < 2:
            return
        self.setHorizontalHeaderLabels(
            [t("指标"), f"{exams[0]['name']} {exams[0]['model']}",
             f"{exams[1]['name']} {exams[1]['model']}"])
        rows = list(analysis.get("table", []))
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c in range(3):
                val = row[c] if c < len(row) else ""
                # the metric names come out of the stored analysis, so they are
                # translated for display only -- the JSON keys stay Chinese
                item = QTableWidgetItem(t(str(val)) if c == 0 else str(val))
                if c > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.setItem(r, c, item)
        self.resizeRowsToContents()


def fill_advice(lst: QListWidget, items) -> None:
    """Replace an advice list's contents and re-fit its height."""
    lst.clear()
    for s in items:
        lst.addItem(f"· {s}")
    # the row height is the widget's, not a guessed 26 px: with a taller font a
    # fixed cap squeezes the rows together instead of letting the list scroll
    row = max(18, lst.fontMetrics().height() + 6)
    lst.setMaximumHeight(min(row * 6, row * max(1, len(items)) + 8))


def advice_list(items) -> QListWidget:
    lst = QListWidget()
    lst.setEditTriggers(QAbstractItemView.NoEditTriggers)
    lst.setSelectionMode(QAbstractItemView.NoSelection)
    lst.setFocusPolicy(Qt.NoFocus)
    fill_advice(lst, items)
    return lst


# ---------------------------------------------------------------------------
# charts
#
# Hand-painted with QPainter instead of pulling in matplotlib/pyqtgraph: the
# console ships with PyQt5 only, and everything here is a few polylines and
# bars.  Colours are the cars' own liveries, so the console, the 3D window and
# the AI's frame all agree on which car is which.
# ---------------------------------------------------------------------------
class SparkChart(QWidget):
    """Speed against race time for both cars.

    The console appends one sample per status poll; the buffer is bounded so a
    long session cannot grow without limit.
    """

    MAX_POINTS = 600

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self._time: list = []
        self._series = [[], []]          # km/h, one list per car
        self._names = ["P1", "P2"]

    def clear(self):
        self._time = []
        self._series = [[], []]
        self.update()

    @property
    def last_time(self):
        """Race time of the newest sample, or ``None`` when empty."""
        return self._time[-1] if self._time else None

    def set_names(self, names):
        self._names = list(names)[:2] or self._names
        self.update()

    def add_sample(self, t: float, speeds):
        self._time.append(float(t))
        for i in range(2):
            value = float(speeds[i]) if i < len(speeds) else 0.0
            self._series[i].append(value)
        if len(self._time) > self.MAX_POINTS:
            drop = len(self._time) - self.MAX_POINTS
            self._time = self._time[drop:]
            for s in self._series:
                del s[:drop]
        self.update()

    # -- painting ------------------------------------------------------
    def paintEvent(self, _event):        # noqa: N802 - Qt naming
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # margins follow the font, so the axis numbers cannot run into the plot
        # or into each other when the UI font is not the one this was tuned on
        fm = p.fontMetrics()
        rect = self.rect().adjusted(fm.horizontalAdvance("0000") + 12, 10, -10,
                                    -fm.height() - 8)
        p.fillRect(self.rect(), self.palette().base())
        if rect.width() <= 8 or rect.height() <= 8:
            return
        if len(self._time) < 2:
            p.setPen(self.palette().text().color())
            p.drawText(self.rect(), Qt.AlignCenter, t("等待比赛数据…"))
            return

        top = max(180.0, max((max(s) for s in self._series if s), default=0.0))
        t0, t1 = self._time[0], self._time[-1]
        span = max(1e-6, t1 - t0)

        # grid + axis labels
        grid = QPen(self.palette().mid().color(), 1, Qt.DotLine)
        for k in range(5):
            y = rect.top() + rect.height() * k / 4.0
            p.setPen(grid)
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(self.palette().text().color())
            p.drawText(QRectF(0, y - fm.height() / 2.0, rect.left() - 6,
                              fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter,
                       f"{top * (1 - k / 4.0):.0f}")

        for i, series in enumerate(self._series):
            if len(series) < 2:
                continue
            pen = QPen(_qcolor(CAR_COLORS[i]), 2)
            p.setPen(pen)
            poly = QPolygonF([
                QPointF(rect.left() + rect.width() * (self._time[j] - t0) / span,
                        rect.bottom() - rect.height() * min(series[j] / top, 1.0))
                for j in range(len(series))])
            p.drawPolyline(poly)

        p.setPen(self.palette().text().color())
        axis = QRectF(rect.left(), rect.bottom() + 2, rect.width(), fm.height())
        p.drawText(axis, Qt.AlignLeft, f"{t0:.0f}s")
        p.drawText(axis, Qt.AlignRight, f"{t1:.0f}s  ·  km/h")


class TrackMap(QWidget):
    """Top-down track outline with both cars and the live running order."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(170)
        self._outline: list = []
        self._cars: list = []

    def set_track(self, outline):
        self._outline = [(float(x), float(z)) for x, z in (outline or [])]
        self.update()

    def set_cars(self, cars):
        self._cars = list(cars or [])
        self.update()

    def paintEvent(self, _event):        # noqa: N802 - Qt naming
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), self.palette().base())
        if len(self._outline) < 2:
            p.setPen(self.palette().text().color())
            p.drawText(self.rect(), Qt.AlignCenter, t("开始对局后显示赛道"))
            return

        xs = [x for x, _z in self._outline]
        zs = [z for _x, z in self._outline]
        minx, maxx = min(xs), max(xs)
        minz, maxz = min(zs), max(zs)
        span_x = max(1e-6, maxx - minx)
        span_z = max(1e-6, maxz - minz)
        pad = 12
        area = self.rect().adjusted(pad, pad, -pad, -pad)
        scale = min(area.width() / span_x, area.height() / span_z)
        # centre the smaller axis so the map keeps the track's real proportions
        ox = area.left() + (area.width() - span_x * scale) * 0.5
        oy = area.top() + (area.height() - span_z * scale) * 0.5

        def to_screen(x, z):
            return QPointF(ox + (x - minx) * scale, oy + (z - minz) * scale)

        p.setPen(QPen(self.palette().mid().color(), 3))
        p.drawPolyline(QPolygonF([to_screen(x, z) for x, z in self._outline]))

        for car in self._cars:
            pos = car.get("pos")
            if not pos:
                continue
            idx = int(car.get("index", 0))
            centre = to_screen(float(pos[0]), float(pos[1]))
            color = _qcolor(CAR_COLORS[idx]) if idx < len(CAR_COLORS) else QColor(0, 0, 0)
            p.setBrush(color)
            p.setPen(QPen(self.palette().text().color(), 1))
            p.drawEllipse(centre, 5, 5)
            heading = float(car.get("heading", 0.0))
            p.drawLine(centre, QPointF(centre.x() + 12 * math.cos(heading),
                                       centre.y() + 12 * math.sin(heading)))

        order = sorted(self._cars, key=lambda c: -float(c.get("progress", 0.0)))
        fm = p.fontMetrics()
        p.setPen(self.palette().text().color())
        text = "   ".join(f"{i + 1}. {c.get('name', '?')}"
                          for i, c in enumerate(order))
        p.drawText(QRectF(6, 4, self.rect().width() - 12, fm.height()),
                   Qt.AlignLeft,
                   fm.elidedText(text, Qt.ElideRight, self.rect().width() - 14))


class BarChart(QWidget):
    """Grouped bars comparing models across matches.

    Each metric is scaled to its own maximum (they have wildly different
    units); the value is printed above the bar so the normalisation is only a
    shape cue, never the number the user reads.
    """

    METRICS = [("win_rate", "胜率%"), ("finish_rate", "完赛率%"),
               ("avg_score", "平均分"), ("avg_speed", "平均速度")]
    _COLORS = [(0.20, 0.45, 0.75), (0.25, 0.62, 0.40),
               (0.80, 0.55, 0.15), (0.60, 0.33, 0.62)]

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self._rows: list = []

    def set_rows(self, rows):
        self._rows = list(rows or [])
        self.update()

    def paintEvent(self, _event):        # noqa: N802 - Qt naming
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), self.palette().base())
        if not self._rows:
            p.setPen(self.palette().text().color())
            p.drawText(self.rect(), Qt.AlignCenter, t("暂无跨对局数据"))
            return

        names = [str(r.get("model", "?")) for r in self._rows]
        maxima = {key: max((float(r.get(key, 0.0) or 0.0) for r in self._rows),
                           default=0.0) or 1.0
                  for key, _label in self.METRICS}

        # The legend is laid out from the font metrics, not a fixed 84 px pitch:
        # a hard-coded pitch with a fixed 78 px text box overlaps itself as soon
        # as the font is wider than the one it was tuned against -- which is
        # exactly what Windows display scaling and a different UI font do.
        fm = p.fontMetrics()
        swatch, gap, pad = 10.0, 4.0, 16.0
        legend_h = max(16, fm.height())
        x = 12.0
        p.setPen(self.palette().text().color())
        for m, (key, label) in enumerate(self.METRICS):
            label = t(label)
            w = fm.horizontalAdvance(label)
            p.fillRect(QRectF(x, 4 + (legend_h - swatch) / 2, swatch, swatch),
                       _qcolor(self._COLORS[m]))
            p.drawText(QRectF(x + swatch + gap, 4, w, legend_h),
                       Qt.AlignLeft | Qt.AlignVCenter, label)
            x += swatch + gap + w + pad

        rect = self.rect().adjusted(12, 8 + legend_h + 8, -12, -8 - legend_h)
        if rect.height() <= 8 or rect.width() <= 8:
            return
        group_w = rect.width() / max(1, len(names))
        bar_w = max(4.0, (group_w - 12) / len(self.METRICS))

        for g, row in enumerate(self._rows):
            base_x = rect.left() + g * group_w
            for m, (key, _label) in enumerate(self.METRICS):
                value = float(row.get(key, 0.0) or 0.0)
                h = rect.height() * min(value / maxima[key], 1.0)
                bar = QRectF(base_x + 6 + m * bar_w, rect.bottom() - h,
                             bar_w - 2, max(1.0, h))
                p.fillRect(bar, _qcolor(self._COLORS[m]))
                if bar_w >= 18:
                    p.setPen(self.palette().text().color())
                    p.drawText(QRectF(bar.left() - 6, bar.top() - 15,
                                      bar.width() + 12, 14),
                               Qt.AlignCenter, f"{value:.0f}")
            p.setPen(self.palette().text().color())
            p.drawText(QRectF(base_x, rect.bottom() + 2, group_w, legend_h),
                       Qt.AlignCenter,
                       fm.elidedText(names[g], Qt.ElideRight, int(group_w)))
