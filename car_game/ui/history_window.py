"""The history window: every match ever run, in its own resizable window.

It lives beside :mod:`car_game.ui.console` rather than inside it because the
console was already the largest module in the project, and because this is a
self-contained view over one store (``RuntimeHistoryStore``) with its own
delete / clear / export actions.

Opened non-modally: the race can keep running while it is on screen.
"""
from __future__ import annotations

import os
import subprocess
import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QAbstractItemView, QDialog, QFileDialog,
                             QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                             QPushButton, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from ..config import external_env
from .i18n import t
from .theme import SPACE
from .widgets import (BarChart, ComparisonTable, advice_list, card,
                      fill_advice)

# the aggregate table's columns, in the order the store returns them
AGG_KEYS = ["model", "matches", "win_rate", "finish_rate", "avg_score",
            "avg_speed", "avg_wall", "avg_obs"]

# How often the window looks for newly finished matches.  The check is a
# directory scan against the store's cache, so it is cheap enough to run this
# often while a race is going on behind the window.
REFRESH_MS = 1000


class HistoryDialog(QDialog):
    """Browse, inspect and prune the cross-session match history."""

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle(t("历史数据 · 跨会话保留"))
        self.resize(1020, 660)
        self.setMinimumSize(760, 480)
        self._rows = 0          # rows already filled into the table
        self._sig = None
        self._build()
        self.load()
        # A race can keep running behind this window, so it keeps itself
        # current instead of waiting to be reopened.
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    # ------------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"],
                                 SPACE["md"])
        outer.setSpacing(SPACE["md"])

        head = QHBoxLayout()
        head.setSpacing(SPACE["sm"])
        self.count_label = QLabel(t("历史记录 0 条"))
        head.addWidget(self.count_label)
        head.addStretch(1)
        self.delete_btn = QPushButton(t("删除选中"))
        self.delete_btn.clicked.connect(self.on_delete)
        self.clear_btn = QPushButton(t("清空全部"))
        self.clear_btn.clicked.connect(self.on_clear)
        self.export_btn = QPushButton(t("导出历史报告"))
        self.export_btn.clicked.connect(self.on_export)
        self.open_btn = QPushButton(t("打开历史文件夹"))
        self.open_btn.clicked.connect(self.on_open_folder)
        for b in (self.delete_btn, self.clear_btn, self.export_btn,
                  self.open_btn):
            head.addWidget(b)
        outer.addLayout(head)

        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(3)

        m_card = card(t("全部历史对局"))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [t("编号"), t("时间"), t("模式"), t("P1 模型"), t("P2 模型"),
             t("获胜"), t("P1 评分"), t("P2 评分")])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(130)
        # let Qt track the column widths: re-measuring every row on each live
        # refresh is what would make a long history crawl
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_selected)
        m_card.body.addWidget(self.table)
        split.addWidget(m_card)

        right = QSplitter(Qt.Vertical)
        right.setHandleWidth(3)
        a_card = card(t("历史汇总（按模型，跨全部对局）"))
        self.agg_table = QTableWidget(0, 8)
        self.agg_table.setHorizontalHeaderLabels(
            [t("模型"), t("场次"), t("胜率%"), t("完赛率%"), t("平均分"),
             t("平均速度"), t("平均撞墙"), t("平均撞障碍")])
        self.agg_table.verticalHeader().setVisible(False)
        self.agg_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.agg_table.setAlternatingRowColors(True)
        self.agg_table.horizontalHeader().setStretchLastSection(True)
        self.agg_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.agg_table.setMinimumHeight(84)
        a_card.body.addWidget(self.agg_table)
        self.chart = BarChart()
        a_card.body.addWidget(self.chart)
        right.addWidget(a_card)

        d_card = card(t("赛后分析详情"))
        self.detail_head = QLabel(t("选择左侧一条历史记录查看详情"))
        d_card.body.addWidget(self.detail_head)
        self.detail_table = ComparisonTable()
        self.detail_table.setMinimumHeight(110)
        d_card.body.addWidget(self.detail_table)
        self.advice = advice_list([])
        d_card.body.addWidget(self.advice)
        right.addWidget(d_card)
        right.setSizes([300, 320])
        split.addWidget(right)
        split.setSizes([520, 500])
        outer.addWidget(split, 1)

    # ------------------------------------------------------------------
    def load(self):
        """Rebuild the whole window from the store."""
        matches = self.store.list_matches()
        self._fill_table(matches)
        self._reload_aggregate()
        has = bool(matches)
        for b in (self.delete_btn, self.clear_btn, self.export_btn):
            b.setEnabled(has)
        self._set_detail(None)
        self._sig = self.store.signature()

    def refresh(self):
        """Pick up matches finished since the last look.

        Called on a timer while a race may be running behind the window.
        Comparing fingerprints is a directory scan and the parsed rows come
        from the store's cache, so this costs nothing when nothing changed --
        and when something did, only the new rows are added, so the user's
        selection and the detail pane survive.
        """
        if not self.isVisible():
            return                      # nothing to keep current while hidden
        try:
            sig = self.store.signature()
        except OSError:
            return
        if sig == self._sig:
            return
        try:
            matches = self.store.list_matches()
        except OSError:
            return
        if len(matches) < self._rows:
            self.load()                     # records were deleted: rebuild
            return
        self.table.setRowCount(len(matches))
        for r in range(self._rows, len(matches)):
            self._set_row(r, matches[r])
        self._rows = len(matches)
        self._sig = sig
        self.count_label.setText(
            t("历史记录 1 条") if len(matches) == 1
            else t("历史记录 {n} 条", n=len(matches)))
        self._reload_aggregate()
        for b in (self.delete_btn, self.clear_btn, self.export_btn):
            b.setEnabled(True)

    # ------------------------------------------------------------------
    @staticmethod
    def _cells(m) -> list:
        models = m.get("models") or ["", ""]
        scores = m.get("scores") or [None, None]
        winner = m.get("winner")
        mode = {"human_vs_ai": t("人类对AI"),
                "ai_vs_ai": t("AI对AI")}.get(str(m.get("mode") or ""), "")
        return [str(m.get("match_id", "")),
                str(m.get("timestamp", ""))[5:16],
                mode,
                models[0] if models else "",
                models[1] if len(models) > 1 else "",
                "-" if winner is None else f"P{winner + 1}",
                "" if scores[0] is None else str(scores[0]),
                "" if scores[1] is None else str(scores[1])]

    def _set_row(self, r: int, m) -> None:
        for c, val in enumerate(self._cells(m)):
            self.table.setItem(r, c, QTableWidgetItem(val))

    def _fill_table(self, matches) -> None:
        self.count_label.setText(
            t("历史记录 1 条") if len(matches) == 1
            else t("历史记录 {n} 条", n=len(matches)))
        self.table.setRowCount(len(matches))
        for r, m in enumerate(matches):
            self._set_row(r, m)
        self._rows = len(matches)

    def _reload_aggregate(self) -> None:
        agg = self.store.aggregate()
        self.agg_table.setRowCount(len(agg))
        for r, row in enumerate(agg):
            for c, key in enumerate(AGG_KEYS):
                self.agg_table.setItem(
                    r, c, QTableWidgetItem(str(row.get(key, ""))))
        self.chart.set_rows(agg)

    def select(self, match_id: int):
        """Highlight one record (used by the sidebar's double-click)."""
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.text() == str(match_id):
                self.table.selectRow(r)
                return

    # ------------------------------------------------------------------
    def _selected_id(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        try:
            return int(self.table.item(rows[0].row(), 0).text())
        except (AttributeError, TypeError, ValueError):
            return None

    def _set_detail(self, analysis):
        if not analysis:
            self.detail_head.setText(t("选择左侧一条历史记录查看详情"))
            self.detail_table.setRowCount(0)
            fill_advice(self.advice, [])
            return
        self.detail_head.setText(
            t("历史第 {mid} 条   ", mid=analysis.get("match_id"))
            + t("（本次会话第 {no} 局）   ",
                no=analysis.get("session_match_id", "?"))
            + t("圈数 {laps}   用时 {duration}s   ",
                laps=analysis.get("laps"), duration=analysis.get("duration"))
            + str(analysis.get("timestamp", "")))
        self.detail_table.load(analysis)
        lines = []
        for a in analysis.get("analyses", []):
            lines.append(f"[{a['name']}] {a['model']}  "
                         f"{a['total']} ({a['grade']})")
            lines.extend(a.get("advice", []))
        fill_advice(self.advice, lines)

    def show_selected(self):
        mid = self._selected_id()
        if mid is None:
            return
        try:
            payload = self.store.load_match(mid)
        except Exception as exc:  # noqa: BLE001
            self.detail_head.setText(t("加载失败: {err}", err=exc))
            return
        self._set_detail(payload.get("analysis", {}))

    # ------------------------------------------------------------------
    def on_delete(self):
        mid = self._selected_id()
        if mid is None:
            QMessageBox.information(self, t("删除历史"), t("请先选中一条历史记录"))
            return
        if QMessageBox.question(
                self, t("删除历史"),
                t("确定删除历史记录 #{mid} 吗？该操作不可撤销。", mid=mid),
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.store.delete(mid)
        self.load()

    def on_clear(self):
        total = self.store.count()
        if total <= 0:
            QMessageBox.information(self, t("清空历史"), t("历史数据已经是空的"))
            return
        if QMessageBox.question(
                self, t("清空历史"),
                t("确定清空全部 {total} 条历史记录吗？该操作不可撤销。\n",
                  total=total)
                + t("（本次会话的数据不受影响，仍可在「数据分析」页导出）"),
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.store.clear_all()
        self.load()

    def on_export(self):
        if self.store.count() <= 0:
            QMessageBox.information(self, t("导出历史报告"), t("历史数据是空的"))
            return
        path, _ = QFileDialog.getSaveFileName(self, t("导出历史报告"),
                                              "history_report.pdf",
                                              t("PDF 文件 (*.pdf)"))
        if not path:
            return
        try:
            self.store.export_report(path)
            QMessageBox.information(self, t("导出历史报告"),
                                    t("已导出到\n{path}", path=path))
        except ImportError:
            QMessageBox.critical(self, t("导出失败"),
                                 t("生成 PDF 需要 Pillow"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, t("导出失败"), str(exc))

    def on_open_folder(self):
        path = self.store.matches_dir
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path], env=external_env())
            else:
                subprocess.Popen(["xdg-open", path], env=external_env())
        except Exception:
            QMessageBox.information(self, t("数据路径"), path)
