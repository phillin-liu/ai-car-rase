"""Race control console (PyQt5).

A single dashboard: a slim top bar (start / pause / stop / save + status), a
session sidebar and four tabs -- 对局设置 / AI 模型 / 运行状态 / 数据分析.

Unlike the previous console, live telemetry and post-match analysis are
rendered as cards, meters and tables instead of two ``QPlainTextEdit`` walls
of text, and the help text lives behind a single "?" button.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton,
    QRadioButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..config import (DEFAULT_MODELS, PROVIDER_LABELS, PROVIDERS, ROOT,
                      RUNTIME_DIR, GameConfig, MatchConfig, ModelConfig,
                      ensure_dirs, load_keys, load_settings, save_keys,
                      save_settings)
from ..items import ITEM_DEFS
from ..providers import get as get_provider, normalize_provider
from .i18n import t
from ..runner import (answer_handshake, clear_gui_present,
                      clear_handshake, clear_pause, mark_gui_present,
                      pause_requested, read_status, request_stop,
                      toggle_pause)
from ..store import MatchStore, RuntimeHistoryStore
from .history_window import HistoryDialog
from .theme import SPACE, role_font
from .widgets import (BarChart, ComparisonTable, MetricChip, SparkChart,
                      StatTile, StatusPill, TelemetryCard, TrackMap,
                      advice_list, card, collapsible, divider, fill_advice)

SESSION_FILE = os.path.join(RUNTIME_DIR, "session.json")
MAIN = os.path.join(ROOT, "main.py")

SIZE_PRESETS = ["960x540", "1280x720", "1600x900", "1920x1080"]


class _Bridge(QObject):
    test_done = pyqtSignal(int, str)
    scan_done = pyqtSignal(int, list, str)   # side, model ids, error


class ConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle(t("AI 赛车 · 控制台"))
        self.resize(1360, 860)
        self.setMinimumSize(1120, 700)

        self.cfg = load_settings()
        self.proc: subprocess.Popen | None = None
        self.store = MatchStore()
        # separate root: the session store is wiped on every start, this one
        # is only ever emptied by the user
        self.history = RuntimeHistoryStore()

        self.history_dialog = None      # created on first use
        # pre-race handshake prompt bookkeeping (see _maybe_ask_keep_waiting)
        self._handshake_prompted = False
        self._handshake_key = None
        # the user asked for a stop, but the runner has not acted on it yet
        self._stopping = False
        # directory fingerprints, so the live lists only rebuild on a change
        self._sess_sig = None
        self._hist_sig = None

        self.bridge = _Bridge()
        self.bridge.test_done.connect(self._on_test_done)
        self.bridge.scan_done.connect(self._on_scan_done)

        self._build_ui()
        self._load_into_ui()
        self.refresh_session()
        self.load_history()

        self.timer = QTimer(self)
        self.timer.setInterval(800)
        self.timer.timeout.connect(self._poll_status)
        self.timer.start()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._build_toolbar(root)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(3)
        split.addWidget(self._build_sidebar())
        split.addWidget(self._build_tabs())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([290, 1060])
        bl.addWidget(split, 1)
        root.addWidget(body, 1)

    def _build_toolbar(self, root: QVBoxLayout):
        bar_host = QWidget()
        bar = QHBoxLayout(bar_host)
        bar.setContentsMargins(SPACE["md"], SPACE["sm"], SPACE["md"], SPACE["sm"])
        bar.setSpacing(SPACE["sm"])

        title = QLabel(t("AI 赛车 · 对局控制台"))
        title.setFont(role_font("title", bold=True))
        bar.addWidget(title)
        bar.addSpacing(SPACE["xl"])

        self.start_btn = QPushButton(t("开始对局"))
        self.start_btn.setFont(role_font("body", bold=True))
        self.start_btn.clicked.connect(self.on_start)
        bar.addWidget(self.start_btn)

        self.pause_btn = QPushButton(t("暂停"))
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.on_pause)
        bar.addWidget(self.pause_btn)

        self.stop_btn = QPushButton(t("停止"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop)
        bar.addWidget(self.stop_btn)

        save_btn = QPushButton(t("保存设置"))
        save_btn.clicked.connect(self.on_save)
        bar.addWidget(save_btn)

        bar.addStretch(1)
        self.status_pill = StatusPill(t("就绪"), "idle")
        bar.addWidget(self.status_pill)
        root.addWidget(bar_host)
        root.addWidget(divider())

    # ---- sidebar ------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACE["md"])

        sess = card(t("本次会话"))
        # the tip used to be a word-wrapped label that made this card tall for
        # no reason; it is a tooltip now
        sess.setToolTip(t("每局随机生成一张全新赛道（互不重复）。\n"
                          "开始新对局会清除本次会话数据；每局都已归档进下方历史，\n"
                          "不会被清除。"))
        chips = QHBoxLayout()
        chips.setSpacing(SPACE["sm"])
        self.chip_matches = MetricChip(t("局数"), "0")
        self.chip_dirty = MetricChip(t("更新"), "-")
        for c in (self.chip_matches, self.chip_dirty):
            chips.addWidget(c)
        chips.addStretch(1)
        sess.body.addLayout(chips)

        sess.body.addWidget(divider())
        acts = QHBoxLayout()
        acts.setSpacing(SPACE["sm"])
        export_btn = QPushButton(t("导出训练报告"))
        export_btn.clicked.connect(self.on_export_report)
        open_btn = QPushButton(t("打开数据文件夹"))
        open_btn.clicked.connect(self.on_open_folder)
        acts.addWidget(export_btn, 1)
        acts.addWidget(open_btn, 1)
        sess.body.addLayout(acts)
        lay.addWidget(sess, 0)

        # --- history: a short list here, the full picture in its own window
        hist = card(t("历史记录"))
        hist.setToolTip(t("跨会话保留：开新对局不会清除。\n"
                          "点「历史汇总」打开完整窗口（含删除 / 清空 / 导出）。"))
        self.history_list = QListWidget()
        self.history_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.history_list.setMinimumHeight(90)
        self.history_list.itemDoubleClicked.connect(self.on_history_item)
        hist.body.addWidget(self.history_list, 1)
        hist_btn = QPushButton(t("历史汇总"))
        hist_btn.clicked.connect(self.on_open_history)
        hist.body.addWidget(hist_btn)
        lay.addWidget(hist, 1)
        return panel

    def on_history_item(self, item):
        """Double-clicking a record opens the full window on it."""
        mid = item.data(Qt.UserRole)
        self.on_open_history()
        if mid is not None and self.history_dialog is not None:
            self.history_dialog.select(int(mid))

    def on_open_history(self):
        """Show the history window, reusing it if it is already open."""
        if self.history_dialog is None:
            self.history_dialog = HistoryDialog(self.history, self)
        self.history_dialog.load()
        self.history_dialog.show()
        self.history_dialog.raise_()
        self.history_dialog.activateWindow()

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self._build_match_tab()
        self._build_models_tab()
        self._build_live_tab()
        self._build_data_tab()
        # looked up rather than hard-coded: moving a tab used to mean hunting
        # down magic indices, which is exactly how the history tab broke
        self.live_tab_index = self.tabs.indexOf(self.live_tab)
        self.data_tab_index = self.tabs.indexOf(self.data_tab)
        self.tabs.setCurrentIndex(0)
        return self.tabs

    # ---- match settings ------------------------------------------------
    def _build_match_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, t("对局设置"))
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        outer.setSpacing(SPACE["md"])

        mode = card(t("对战模式"))
        mh = QHBoxLayout()
        mh.setSpacing(SPACE["lg"])
        self.mode_ai = QRadioButton(t("AI 对 AI"))
        self.mode_human = QRadioButton(t("人类 对 AI"))
        for b in (self.mode_ai, self.mode_human):
            mh.addWidget(b)
        mh.addSpacing(SPACE["md"])
        lbl = QLabel(t("人类控制车辆"))
        mh.addWidget(lbl)
        self.human_combo = QComboBox()
        self.human_combo.addItems(["P1", "P2"])
        self.human_combo.setFixedWidth(80)
        mh.addWidget(self.human_combo)
        mh.addStretch(1)
        mode.body.addLayout(mh)
        self.mode_ai.toggled.connect(self._on_mode)
        outer.addWidget(mode)

        row = QHBoxLayout()
        row.setSpacing(SPACE["md"])

        race = card(t("赛制"))
        rf = QGridLayout()
        rf.setHorizontalSpacing(SPACE["md"])
        rf.setVerticalSpacing(SPACE["sm"])
        self.laps_spin = QSpinBox()
        self.laps_spin.setRange(1, 50)
        self.matches_spin = QSpinBox()
        self.matches_spin.setRange(1, 999)
        self.infinite_check = QCheckBox(t("无限连续对局"))
        self.infinite_check.toggled.connect(self._on_infinite)
        self.items_check = QCheckBox(t("启用道具系统 (按 1-5 使用)"))

        rf.addWidget(QLabel(t("每局圈数")), 0, 0)
        rf.addWidget(self.laps_spin, 0, 1)
        rf.addWidget(QLabel(t("对局局数")), 1, 0)
        mrow = QHBoxLayout()
        mrow.setSpacing(SPACE["sm"])
        mrow.addWidget(self.matches_spin)
        mrow.addWidget(self.infinite_check)
        mrow.addStretch(1)
        rf.addLayout(mrow, 1, 1)
        rf.addWidget(self.items_check, 2, 0, 1, 2)
        rf.setColumnStretch(1, 1)
        race.body.addLayout(rf)
        row.addWidget(race, 1)

        run = card(t("运行方式"))
        gf = QGridLayout()
        gf.setHorizontalSpacing(SPACE["md"])
        gf.setVerticalSpacing(SPACE["sm"])
        self.size_combo = QComboBox()
        self.size_combo.addItems(SIZE_PRESETS)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["chase", "first", "overhead", "free"])
        self.diff_spin = QDoubleSpinBox()
        self.diff_spin.setRange(0.6, 1.3)
        self.diff_spin.setSingleStep(0.05)
        self.headless_check = QCheckBox(t("无窗口运行 (无界面模式)"))
        self.debug_check = QCheckBox(t("调试模式 (显示 AI 决策/遥测)"))
        self.msaa_check = QCheckBox(t("4x 抗锯齿 (MSAA)"))
        gf.addWidget(QLabel(t("窗口尺寸")), 0, 0)
        gf.addWidget(self.size_combo, 0, 1)
        gf.addWidget(QLabel(t("初始视角")), 1, 0)
        gf.addWidget(self.view_combo, 1, 1)
        gf.addWidget(QLabel(t("AI 难度")), 2, 0)
        gf.addWidget(self.diff_spin, 2, 1)
        gf.addWidget(self.headless_check, 3, 0, 1, 2)
        gf.addWidget(self.debug_check, 4, 0, 1, 2)
        gf.addWidget(self.msaa_check, 5, 0, 1, 2)
        gf.setColumnStretch(1, 1)
        run.body.addLayout(gf)
        row.addWidget(run, 1)

        outer.addLayout(row)

        tip = QLabel(t("每局开赛时随机生成一张全新且互不重复的赛道。\n"
                       "游戏中按 H 可切换详细 HUD，调试模式下会另开窗口显示 AI 识别视角。"))
        tip.setWordWrap(True)
        outer.addWidget(tip)
        outer.addStretch(1)

    def _on_infinite(self, checked: bool):
        self.matches_spin.setEnabled(not checked)
        self.laps_spin.setEnabled(not checked)

    def _on_mode(self):
        self.human_combo.setEnabled(self.mode_human.isChecked())

    # ---- models --------------------------------------------------------
    def _build_models_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, t("AI 模型"))
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        outer.setSpacing(SPACE["md"])

        row = QHBoxLayout()
        row.setSpacing(SPACE["md"])
        self.model_widgets: dict = {}
        for i in (0, 1):
            col = card(t("玩家 P{n}", n=i + 1))
            form = QGridLayout()
            form.setHorizontalSpacing(SPACE["sm"])
            form.setVerticalSpacing(SPACE["sm"])
            self._build_model_form(form, i)
            col.body.addLayout(form)
            self._build_model_extra(col, i)
            btns = QHBoxLayout()
            test = QPushButton(t("测试 P{n} 连接 (含视觉检测)", n=i + 1))
            test.clicked.connect(lambda _c, idx=i: self.on_test(idx))
            btns.addWidget(test)
            btns.addStretch(1)
            col.body.addLayout(btns)
            row.addWidget(col, 1)
        outer.addLayout(row)

        bottom = QHBoxLayout()
        self.test_label = QLabel("")
        save_keys_btn = QPushButton(t("保存 API 密钥"))
        save_keys_btn.clicked.connect(self.on_save_keys)
        bottom.addWidget(self.test_label, 1)
        bottom.addWidget(save_keys_btn)
        outer.addLayout(bottom)
        outer.addStretch(1)

    def _build_model_form(self, form: QGridLayout, i: int):
        """The handful of fields every side needs; the rest folds away."""
        w = {}
        self.model_widgets[i] = w

        def add(row, label, widget, extra=None):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
            if extra is not None:
                form.addWidget(extra, row, 2)

        w["provider"] = QComboBox()
        for spec in PROVIDERS:
            w["provider"].addItem(t(spec.label), spec.id)
        w["provider"].currentIndexChanged.connect(lambda _t, idx=i: self._on_provider(idx))
        add(0, t("提供方"), w["provider"])

        # an editable combo: the scan fills it, but any model id can be typed
        w["model"] = QComboBox()
        w["model"].setEditable(True)
        w["model"].setInsertPolicy(QComboBox.NoInsert)
        w["model"].lineEdit().setPlaceholderText(t("选择或输入模型名称"))
        scan = QPushButton(t("扫描模型"))
        scan.setToolTip(t("联网拉取该提供方可用的模型列表"))
        scan.clicked.connect(lambda _c, idx=i: self.on_scan_models(idx))
        w["scan_btn"] = scan
        add(1, t("模型"), w["model"], scan)

        w["display_name"] = QLineEdit()
        add(2, t("显示名称"), w["display_name"])

        w["api_key"] = QLineEdit()
        w["api_key"].setEchoMode(QLineEdit.Password)
        show = QCheckBox(t("显示"))
        show.toggled.connect(lambda on, e=w["api_key"]:
                             e.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        add(3, t("API 密钥"), w["api_key"], show)
        form.setColumnStretch(1, 1)

    def _build_model_extra(self, col, i: int):
        """Connection tuning, folded away by default.

        The tab used to show all ~32 controls at once; keeping only the fields
        that decide *which model is used* on the surface makes it readable.
        """
        w = self.model_widgets[i]

        def grid():
            g = QGridLayout()
            g.setHorizontalSpacing(SPACE["md"])
            g.setVerticalSpacing(SPACE["sm"])
            g.setColumnStretch(1, 1)
            return g

        adv, _body = collapsible(t("高级 · 连接与采样"))
        ag = grid()
        w["base_url"] = QLineEdit()
        w["base_url"].setPlaceholderText(t("留空则使用内置地址 / 环境变量"))
        ag.addWidget(QLabel("API Base URL"), 0, 0)
        ag.addWidget(w["base_url"], 0, 1)
        w["temperature"] = QDoubleSpinBox()
        w["temperature"].setRange(0.0, 2.0)
        w["temperature"].setSingleStep(0.1)
        ag.addWidget(QLabel(t("温度")), 1, 0)
        ag.addWidget(w["temperature"], 1, 1)
        w["decision_interval"] = QDoubleSpinBox()
        w["decision_interval"].setRange(0.1, 10.0)
        w["decision_interval"].setSingleStep(0.1)
        ag.addWidget(QLabel(t("决策间隔(秒)")), 2, 0)
        ag.addWidget(w["decision_interval"], 2, 1)
        w["timeout"] = QDoubleSpinBox()
        w["timeout"].setRange(1.0, 300.0)
        w["timeout"].setSingleStep(1.0)
        ag.addWidget(QLabel(t("超时(秒)")), 3, 0)
        ag.addWidget(w["timeout"], 3, 1)
        adv.body.addLayout(ag)
        col.body.addWidget(adv)

    def _select_provider(self, i: int, provider_id: str) -> None:
        """Set the provider combo by id (fires _on_provider)."""
        w = self.model_widgets[i]
        idx = w["provider"].findData(provider_id)
        if idx >= 0:
            w["provider"].setCurrentIndex(idx)

    def _on_provider(self, i: int):
        w = self.model_widgets[i]
        spec = get_provider(w["provider"].currentData())
        w["base_url"].setText(spec.base_url)
        w["base_url"].setPlaceholderText(
            spec.base_url or t("留空则使用内置地址 / 环境变量"))
        w["api_key"].setEnabled(spec.key_required)
        w["api_key"].setToolTip(t(spec.signup_hint))
        w["model"].lineEdit().setPlaceholderText(
            spec.default_model or t("选择或输入模型名称"))
        # switching vendors must not leave the previous vendor's model behind
        if spec.default_model:
            w["model"].setCurrentText(spec.default_model)
        w["scan_btn"].setEnabled(spec.listed)
        w["scan_btn"].setToolTip(t("联网拉取该提供方可用的模型列表"))
        hint = t(spec.note or spec.signup_hint)
        if hint:
            self.test_label.setText(f"P{i + 1} {t(spec.label)}: {hint}")

    def _side_config(self, i: int) -> ModelConfig:
        """A throwaway ModelConfig for the widgets of side ``i``."""
        w = self.model_widgets[i]
        return ModelConfig(
            provider=w["provider"].currentData(),
            model=w["model"].currentText().strip(),
            base_url=w["base_url"].text().strip(),
            api_key=w["api_key"].text().strip(),
            timeout=float(w["timeout"].value()),
        )

    def _fill_model_choices(self, i: int, models: list[str]) -> None:
        w = self.model_widgets[i]
        current = w["model"].currentText().strip()
        w["model"].blockSignals(True)
        w["model"].clear()
        w["model"].addItems(models)
        if current:
            w["model"].setCurrentText(current)
        elif models:
            w["model"].setCurrentText(models[0])
        w["model"].blockSignals(False)

    def on_scan_models(self, i: int):
        """Fetch the provider's model list in the background."""
        w = self.model_widgets[i]
        cfg = self._side_config(i)
        self.test_label.setText(t("P{n}: 正在扫描模型 ...", n=i + 1))
        w["scan_btn"].setEnabled(False)

        def worker():
            try:
                from ..vision.models_scan import list_models
                models = list_models(cfg)
                self.bridge.scan_done.emit(i, models, "")
            except Exception as exc:  # noqa: BLE001
                self.bridge.scan_done.emit(i, [], str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_done(self, i: int, models: list, err: str):
        w = self.model_widgets[i]
        w["scan_btn"].setEnabled(True)
        if err or not models:
            self.test_label.setText(
                t("P{n}: 扫描失败 — {err}", n=i + 1,
                  err=err or t("没有返回模型")))
            return
        self._fill_model_choices(i, [str(m) for m in models])
        self.test_label.setText(
            t("P{n}: 扫描到 {count} 个模型，已填入下拉框", n=i + 1,
              count=len(models)))

    # ---- live status ---------------------------------------------------
    def _build_live_tab(self):
        tab = QWidget()
        self.live_tab = tab
        self.tabs.addTab(tab, t("运行状态"))
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        outer.setSpacing(SPACE["md"])

        board = QHBoxLayout()
        board.setSpacing(SPACE["sm"])
        self.live_state = StatTile(t("状态"), t("空闲"))
        self.live_match = StatTile(t("对局"), "—")
        self.live_score = StatTile(t("比分"), "—")
        self.live_time = StatTile(t("赛道时间"), "—", "s")
        # NB: not `t` -- that is the translation function
        for tile in (self.live_state, self.live_match, self.live_score,
                     self.live_time):
            board.addWidget(tile, 1)
        outer.addLayout(board)

        cars = QHBoxLayout()
        cars.setSpacing(SPACE["md"])
        self.car_cards = [TelemetryCard("P1", (0.95, 0.25, 0.2)),
                          TelemetryCard("P2", (0.25, 0.55, 0.95))]
        for c in self.car_cards:
            cars.addWidget(c, 1)
        outer.addLayout(cars)

        # the map and the speed curves share the middle band and grow with
        # the window instead of being pinned to a fixed height
        mid = QSplitter(Qt.Horizontal)
        mid.setHandleWidth(3)
        map_card = card(t("赛道与实时位次"))
        self.track_map = TrackMap()
        map_card.body.addWidget(self.track_map)
        mid.addWidget(map_card)

        chart_card = card(t("车速 / 赛道时间"))
        self.speed_chart = SparkChart()
        chart_card.body.addWidget(self.speed_chart)
        mid.addWidget(chart_card)
        mid.setSizes([420, 640])
        outer.addWidget(mid, 3)

        # side by side rather than stacked: five full-width bands do not fit
        # the tab at the console's minimum height, and both cards are mostly
        # empty space at full width anyway
        lower = QSplitter(Qt.Horizontal)
        lower.setHandleWidth(3)

        log_card = card(t("AI 调用日志"))
        head = QHBoxLayout()
        head.setSpacing(SPACE["sm"])
        self.log_filter_errors = QCheckBox(t("只看失败与错误"))
        head.addWidget(self.log_filter_errors)
        head.addStretch(1)
        self.log_hint = QLabel("")
        head.addWidget(self.log_hint)
        log_card.body.addLayout(head)
        # the per-driver totals used to be a whole table row; one compact line
        # is enough now that the per-call detail has its own table
        self.log_summary = QLabel(t("尚无调用记录。"))
        self.log_summary.setWordWrap(True)
        log_card.body.addWidget(self.log_summary)
        self.driver_table = QTableWidget(0, 5)
        self.driver_table.setHorizontalHeaderLabels(
            [t("时间"), t("方向"), t("模型"), t("结果"), t("摘要")])
        self.driver_table.verticalHeader().setVisible(False)
        self.driver_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.driver_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.driver_table.setAlternatingRowColors(True)
        self.driver_table.horizontalHeader().setStretchLastSection(True)
        self.driver_table.setMinimumHeight(70)
        log_card.body.addWidget(self.driver_table)
        lower.addWidget(log_card)

        ac_card = card(t("反作弊"))
        self.ac_summary = QLabel(t("实时检测中：每局校验动作 / 遥测 / 逆行，违规会立即判负。"))
        self.ac_summary.setWordWrap(True)
        ac_card.body.addWidget(self.ac_summary)
        self.ac_list = QListWidget()
        self.ac_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ac_list.setMinimumHeight(56)
        ac_card.body.addWidget(self.ac_list)
        lower.addWidget(ac_card)
        lower.setSizes([720, 340])
        outer.addWidget(lower, 4)

    # ---- data analysis -------------------------------------------------
    def _build_data_tab(self):
        tab = QWidget()
        self.data_tab = tab
        self.tabs.addTab(tab, t("数据分析"))
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        outer.setSpacing(SPACE["sm"])

        # list on the left, everything about the selected run on the right:
        # the old single vertical stack squeezed all three cards together
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(3)

        m_card = card(t("比赛记录"))
        self.match_table = QTableWidget(0, 7)
        self.match_table.setHorizontalHeaderLabels(
            [t("局"), t("时间"), t("P1 模型"), t("P2 模型"), t("获胜"),
             t("P1 评分"), t("P2 评分")])
        self.match_table.verticalHeader().setVisible(False)
        self.match_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.match_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.match_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.match_table.setAlternatingRowColors(True)
        self.match_table.horizontalHeader().setStretchLastSection(True)
        self.match_table.setMinimumHeight(120)
        self.match_table.itemSelectionChanged.connect(self.show_selected_match)
        m_card.body.addWidget(self.match_table)
        split.addWidget(m_card)

        a_card = card(t("模型总体表现（跨对局汇总）"))
        self.agg_table = QTableWidget(0, 8)
        self.agg_table.setHorizontalHeaderLabels(
            [t("模型"), t("场次"), t("胜率%"), t("完赛率%"), t("平均分"),
             t("平均速度"), t("平均撞墙"), t("平均撞障碍")])
        self.agg_table.verticalHeader().setVisible(False)
        self.agg_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.agg_table.setAlternatingRowColors(True)
        self.agg_table.horizontalHeader().setStretchLastSection(True)
        self.agg_table.setMinimumHeight(84)
        a_card.body.addWidget(self.agg_table)
        self.agg_chart = BarChart()
        a_card.body.addWidget(self.agg_chart)

        right = QSplitter(Qt.Vertical)
        right.setHandleWidth(3)
        right.addWidget(a_card)

        d_card = card(t("赛后分析详情"))
        self.detail_head = QLabel(t("选择上方一条比赛记录查看详情"))
        d_card.body.addWidget(self.detail_head)
        self.detail_table = ComparisonTable()
        self.detail_table.setMinimumHeight(110)
        d_card.body.addWidget(self.detail_table)
        self.detail_advice = advice_list([])
        d_card.body.addWidget(self.detail_advice)
        right.addWidget(d_card)
        right.setSizes([300, 320])
        split.addWidget(right)
        split.setSizes([420, 640])
        outer.addWidget(split, 1)


    # ==================================================================
    # settings <-> UI
    # ==================================================================
    def _load_into_ui(self):
        m = self.cfg.match
        (self.mode_ai if m.mode == "ai_vs_ai" else self.mode_human).setChecked(True)
        self.human_combo.setCurrentText("P1" if m.human_player == 0 else "P2")
        self.laps_spin.setValue(m.laps)
        self.infinite_check.setChecked(m.matches <= 0)
        self.matches_spin.setValue(m.matches if m.matches > 0 else 5)
        self.matches_spin.setEnabled(m.matches > 0)
        self.laps_spin.setEnabled(m.matches > 0)
        self.items_check.setChecked(m.items_enabled)
        self.headless_check.setChecked(m.headless)
        self.debug_check.setChecked(bool(getattr(m, "debug", False)))
        self.msaa_check.setChecked(bool(getattr(m, "msaa", True)))
        self.size_combo.setCurrentText(f"{m.window_width}x{m.window_height}")
        self.view_combo.setCurrentText(m.view)
        self.diff_spin.setValue(m.difficulty)
        for i, side in ((0, self.cfg.p1), (1, self.cfg.p2)):
            w = self.model_widgets[i]
            self._select_provider(i, normalize_provider(side.provider))
            self._on_provider(i)      # firing is skipped when the index is unchanged
            w["model"].setCurrentText(side.model)
            w["display_name"].setText(side.display_name)
            w["base_url"].setText(side.base_url)
            w["api_key"].setText(side.api_key)
            w["temperature"].setValue(side.temperature)
            w["decision_interval"].setValue(side.decision_interval)
            w["timeout"].setValue(side.timeout)
        self._on_mode()

    def collect_config(self) -> GameConfig:
        width, _, height = self.size_combo.currentText().partition("x")
        match = MatchConfig(
            mode="human_vs_ai" if self.mode_human.isChecked() else "ai_vs_ai",
            human_player=0 if self.human_combo.currentText() == "P1" else 1,
            laps=self.laps_spin.value(),
            matches=0 if self.infinite_check.isChecked() else self.matches_spin.value(),
            render=not self.headless_check.isChecked(),
            headless=self.headless_check.isChecked(),
            view=self.view_combo.currentText(),
            spectate=True,
            items_enabled=self.items_check.isChecked(),
            window_width=int(width) if width.isdigit() else 1280,
            window_height=int(height) if height.isdigit() else 720,
            difficulty=float(self.diff_spin.value()),
            debug=self.debug_check.isChecked(),
            msaa=self.msaa_check.isChecked(),
        )
        cfg = GameConfig(match=match)
        for i, key in ((0, "p1"), (1, "p2")):
            wd = self.model_widgets[i]
            # the vision_* render settings are no longer on the panel; carry
            # whatever the loaded config had so a hand-edited settings.json is
            # not silently reset to the defaults on every save
            prev = getattr(self.cfg, key)
            mc = ModelConfig(
                provider=normalize_provider(wd["provider"].currentData()),
                model=wd["model"].currentText().strip(),
                display_name=wd["display_name"].text().strip(),
                api_key=wd["api_key"].text().strip(),
                base_url=wd["base_url"].text().strip(),
                temperature=float(wd["temperature"].value()),
                decision_interval=float(wd["decision_interval"].value()),
                timeout=float(wd["timeout"].value()),
                vision_view=getattr(prev, "vision_view", "first"),
                vision_image_size=int(getattr(prev, "vision_image_size", 256)
                                      or 256),
                vision_fov=float(getattr(prev, "vision_fov", 58.0) or 58.0),
                vision_ahead_m=float(getattr(prev, "vision_ahead_m", 80.0)
                                     or 80.0),
                vision_supersample=int(getattr(prev, "vision_supersample", 2)
                                       or 1),
            )
            if not mc.model:
                mc.model = DEFAULT_MODELS.get(mc.provider, "")
            if not mc.display_name:
                mc.display_name = f"{get_provider(mc.provider).label} P{i + 1}"
            setattr(cfg, key, mc)
        return cfg

    # ==================================================================
    # actions
    # ==================================================================
    def on_save(self):
        self.cfg = self.collect_config()
        save_settings(self.cfg)
        self.status_pill.set_state("idle", t("设置已保存"))

    def on_save_keys(self):
        cfg = self.collect_config()
        keys = load_keys()
        for side in (cfg.p1, cfg.p2):
            if side.api_key:
                keys[side.provider] = side.api_key
        save_keys(keys)
        save_settings(cfg)
        self.status_pill.set_state("idle", t("密钥已保存"))

    def on_test(self, i: int):
        cfg = self.collect_config()
        side = cfg.p1 if i == 0 else cfg.p2
        self.test_label.setText(t("正在测试 P{n} 的视觉能力 ...", n=i + 1))

        def worker():
            try:
                from ..vision.check import probe_model
                ok, is_vision, msg = probe_model(side)
            except Exception as exc:  # noqa: BLE001
                ok, is_vision, msg = False, None, t("失败: {err}", err=exc)
            if ok:
                tag = t("✓ 视觉模型")
            elif is_vision is False:
                tag = t("✗ 非视觉模型")
            else:
                tag = t("✗ 未通过")
            self.bridge.test_done.emit(i, f"{tag} — {msg}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_test_done(self, i: int, msg: str):
        self.test_label.setText(f"P{i + 1}: {msg}")

    # ---- session control ----------------------------------------------
    def on_start(self):
        if self.proc and self.proc.poll() is None:
            QMessageBox.information(self, t("提示"), t("已有对局在运行中"))
            return
        self.cfg = self.collect_config()

        # every provider in the registry is a vision model; the only way a side
        # cannot run is a cloud provider with no key (or no model picked)
        blocked = []
        for tag, side in (("P1", self.cfg.p1), ("P2", self.cfg.p2)):
            if side.needs_key():
                blocked.append((tag, side, t("缺少 API 密钥")))
            elif not side.model:
                blocked.append((tag, side, t("没有选择模型")))
        if blocked:
            detail = "、".join(
                f"{tag} ({PROVIDER_LABELS.get(s.provider, s.provider)}, {why})"
                for tag, s, why in blocked)
            QMessageBox.critical(
                self, t("无法开始：AI 还没配置好"),
                f"{detail}\n\n"
                + t("请在「AI 模型」页填写 API 密钥（点「扫描模型」可拉取可用模型）。")
                + t("想完全离线可改用 Ollama：先在本机运行 ollama serve 并 pull 一个视觉模型。"))
            self.tabs.setCurrentIndex(1)
            return

        ensure_dirs()
        # No reminder before clearing any more: every match is archived into
        # runtime/history/ as it finishes, so the session store is no longer
        # the only copy of anything.  Export it if you want the per-session
        # report; the history keeps the records either way.
        self.store.clear()
        # a new session starts a new map: the old curves and outline are stale
        self.speed_chart.clear()
        self.track_map.set_track([])
        self.track_map.set_cars([])

        save_settings(self.cfg)
        clear_pause()
        # written without API keys; the subprocess rehydrates from keys.json
        tmp = SESSION_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.cfg.to_public_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, SESSION_FILE)
        try:
            os.remove(os.path.join(RUNTIME_DIR, "stop.flag"))
        except OSError:
            pass

        # tell the runner a window is watching, so it can ask us questions it
        # cannot put on screen itself (the pre-race model handshake)
        clear_handshake()
        mark_gui_present()

        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                [sys.executable, MAIN, "--run", "--config", SESSION_FILE],
                cwd=ROOT, creationflags=flags)
        except Exception as exc:  # noqa: BLE001
            clear_gui_present()
            QMessageBox.critical(self, t("启动失败"), str(exc))
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText(t("⏸  暂停"))
        self.status_pill.set_state("running", t("对局运行中"))
        self.tabs.setCurrentIndex(self.live_tab_index)
        self._stopping = False
        self.refresh_session()

    def on_pause(self):
        if not (self.proc and self.proc.poll() is None):
            return
        paused = toggle_pause()
        self.pause_btn.setText(t("▶  继续") if paused else t("⏸  暂停"))
        self.status_pill.set_state("paused" if paused else "running",
                                   t("已暂停") if paused else t("对局运行中"))

    def on_stop(self):
        clear_pause()
        request_stop()
        # the runner may take a moment to notice the flag, and status.json will
        # still say "racing" until it does -- so remember that the user asked
        # for this, and let that outrank the stale file (see _poll_status)
        self._stopping = True
        if self.proc and self.proc.poll() is None:
            QTimer.singleShot(1500, self._force_kill)
        self.status_pill.set_state("paused", t("正在停止…"))

    def _force_kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self._reset_buttons()

    def _reset_buttons(self):
        clear_pause()
        # the run is over: no window is watching any more, and a handshake
        # answer left behind would apply to the next session
        clear_gui_present()
        clear_handshake()
        self._handshake_prompted = False
        self._stopping = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸  暂停")

    def closeEvent(self, event):  # noqa: N802
        if self.proc and self.proc.poll() is None:
            r = QMessageBox.question(self, t("退出"), t("对局仍在运行，确定退出并停止吗?"),
                                     QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                event.ignore()
                return
            clear_pause()
            request_stop()
            try:
                self.proc.terminate()
            except Exception:
                pass
        clear_gui_present()
        clear_handshake()
        event.accept()

    # ==================================================================
    # polling / live status
    # ==================================================================
    def _poll_status(self):
        try:
            if self.proc is not None and self.proc.poll() is not None:
                self.proc = None
                self._reset_buttons()
                self.status_pill.set_state("idle", t("对局已结束"))
                self._show_data_tab()
            self._refresh_stores()
            st = read_status()
            if st:
                # once the runner is gone -- stopped, finished or force-killed
                # -- status.json is a leftover that still says "racing".  The
                # process is the truth, so tell _render_live not to believe it.
                self._render_live(st, ended=self.proc is None)
                self._maybe_ask_keep_waiting(st)
                # an "error" status is a verdict, not a progress label: do not
                # let the running/paused pill paint over it
                if self.proc is not None and st.get("state") != "error":
                    if self._stopping:
                        self.pause_btn.setText("⏸  暂停")
                        self.status_pill.set_state("paused", t("正在停止…"))
                    elif pause_requested():
                        self.pause_btn.setText("▶  继续")
                        self.status_pill.set_state("paused", t("已暂停"))
                    else:
                        self.pause_btn.setText("⏸  暂停")
                        self.status_pill.set_state("running", t("对局运行中"))
        except Exception:
            pass

    def _refresh_stores(self):
        """Repaint the session/history lists the moment a match lands.

        Both stores cache their parsed rows against a directory fingerprint, so
        this is a scan plus a comparison -- cheap enough to run on every poll
        (0.8 s), which is what keeps the counters and the history list live
        without re-parsing every stored match.
        """
        try:
            sig = self.store.signature()
        except OSError:
            sig = None
        if sig != self._sess_sig:
            self._sess_sig = sig
            self.refresh_session()
        try:
            sig = self.history.signature()
        except OSError:
            sig = None
        if sig != self._hist_sig:
            self._hist_sig = sig
            self.load_history()

    def _render_live(self, st: dict, ended: bool = False):
        live = st.get("live") or {}
        last = st.get("last") or {}
        state = st.get("state", "-")
        # status.json is written by the runner; when the runner is no longer
        # running, anything it last wrote is history -- never show a dead
        # session as still racing
        if ended and state in ("racing", "running", "waiting_models", "paused"):
            state = "stopped"
        match_no = st.get("match", 0)
        total = st.get("total", 0)
        score = st.get("score") or {}

        names = {"racing": t("比赛中"), "running": t("运行中"),
                 "finished": t("本局结束"), "done": t("已完成"),
                 "stopped": t("已停止"), "waiting_models": t("等待模型就绪"),
                 "error": t("模型不可用")}
        self.live_state.set_value(names.get(state, str(state)))
        if state == "error" and st.get("error"):
            # the run was ended for a reason the user has to act on
            self.status_pill.set_state("error", str(st["error"])[:60])
        self.live_match.set_value(
            f"{match_no}/{'∞' if not total else total}")
        if score:
            self.live_score.set_value(f"{score.get('p1', 0)} : {score.get('p2', 0)}",
                                      t("平局 {n}", n=score.get("draws", 0)))
        else:
            self.live_score.set_value("—", "")
        self.live_time.set_value(f"{live.get('time', 0.0):.1f}")

        models = st.get("models") or []
        laps = int(st.get("laps") or 0) or 99
        cars = live.get("cars") or []
        if cars:
            for i, c in enumerate(cars[:2]):
                self.car_cards[i].set_title(c.get("name", f"P{i+1}"),
                                            str(c.get("model", models[i] if i < len(models) else "")))
                self.car_cards[i].update_from(c, laps)
        elif last.get("analyses"):
            for i, a in enumerate(last["analyses"][:2]):
                mt = a.get("metrics", {})
                self.car_cards[i].set_title(a.get("name", f"P{i+1}"), a.get("model", ""))
                self.car_cards[i].update_from({
                    "speed": float(mt.get("平均速度") or 0.0) / 3.6,
                    "laps": int(st.get("laps", 0) or 0),
                    "wall_hits": mt.get("撞墙次数", 0),
                    "obstacle_hits": mt.get("撞障碍物次数", 0),
                    "item_uses": mt.get("道具使用次数", 0),
                    "finished": bool(mt.get("完赛")),
                }, laps)

        # --- live charts ---------------------------------------------
        if cars:
            # NB: not `t` -- that is the translation function
            race_t = float(live.get("time", 0.0))
            if race_t != self.speed_chart.last_time:
                self.speed_chart.add_sample(
                    race_t, [float(c.get("speed", 0.0)) * 3.6 for c in cars[:2]])
            self.speed_chart.set_names([c.get("name", f"P{i + 1}")
                                        for i, c in enumerate(cars[:2])])
            outline = (live.get("track") or {}).get("outline") or []
            if outline:
                self.track_map.set_track(outline)
            self.track_map.set_cars([
                {"name": c.get("name", f"P{i + 1}"), "index": i,
                 "pos": c.get("pos"), "heading": c.get("heading", 0.0),
                 "progress": c.get("progress", 0.0)}
                for i, c in enumerate(cars[:2])
            ])

        drivers = st.get("drivers") or live.get("drivers") or []
        only_errors = self.log_filter_errors.isChecked()
        self.log_hint.setText(t("仅显示失败的调用") if only_errors
                              else t("每次请求的成败与回复摘要"))

        # pre-race handshake: say who has not confirmed yet
        shake = st.get("handshake") or {}
        if shake:
            pending = "、".join(shake.get("pending") or []) or t("无")
            answered = "、".join(shake.get("answered") or []) or t("无")
            self.log_hint.setText(
                t("发车前握手：已就绪 {answered} ｜ 未回复 {pending}",
                  answered=answered, pending=pending)
                + (f" ｜ {shake.get('note')}" if shake.get("note") else ""))

        # one compact totals line per side, plus the per-call detail below
        totals = []
        rows = []
        for i, d in enumerate(drivers):
            kind = d.get("driver", "?")
            kind_label = {"llm": t("视觉LLM"), "local": t("本地视觉"),
                          "human": t("人类")}.get(kind, kind)
            model = str(d.get("model", ""))
            err = int(d.get("api_errors", 0) or 0)
            if kind != "human":
                totals.append(
                    f"P{i + 1} {kind_label} {model} · req "
                    f"{int(d.get('api_requests', 0) or 0)} · err {err} · "
                    f"avg {float(d.get('avg_latency_ms', 0.0) or 0.0):.0f}ms · "
                    f"ok {float(d.get('success_rate', 1.0) or 0.0) * 100:.0f}%")
            calls = d.get("call_log") or []
            if calls:
                for entry in calls:
                    ok = bool(entry.get("ok"))
                    if only_errors and ok:
                        continue
                    rows.append(("err" if not ok else "", [
                        f"{float(entry.get('t', 0.0)):.1f}s",
                        f"P{i + 1} {kind_label}",
                        model,
                        t("成功") if ok else t("失败"),
                        str(entry.get("text", ""))[:80]]))
            elif err and only_errors:
                rows.append(("err", [f"{float(live.get('time', 0.0)):.1f}s",
                                     f"P{i + 1} {kind_label}", model, t("失败"),
                                     str(d.get("last_error", ""))[:80]]))
        self.log_summary.setText("　|　".join(totals) if totals
                                 else t("尚无调用记录。"))

        self.driver_table.setRowCount(len(rows))
        for r, (flag, cells) in enumerate(rows):
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if flag == "err" and c == 3:
                    item.setForeground(Qt.red)
                self.driver_table.setItem(r, c, item)
        self.driver_table.resizeColumnsToContents()
        self.driver_table.horizontalHeader().setStretchLastSection(True)

        ac = st.get("anticheat") or {}
        violations = ac.get("violations", 0)
        if violations:
            self.ac_summary.setText(t("⚠ 本局记录 {count} 处违规", count=violations))
            self.ac_summary.setStyleSheet("color:#b42318;")
        else:
            self.ac_summary.setText(t("实时检测中：未发现违规。"))
            self.ac_summary.setStyleSheet("")
        reasons = ac.get("live_reasons") or []
        self.ac_list.clear()
        for r in reasons:
            self.ac_list.addItem(f"⚠ {r}")

    def _maybe_ask_keep_waiting(self, st: dict):
        """Ask the user only when a model has actually *failed* to connect.

        A slow first connection is expected -- it is exactly what the runner is
        waiting for -- so elapsed time never raises this box, only a real
        failure does.  Raised from a timer callback, so the guard flag matters:
        a modal box opens a nested event loop and this poll *will* be delivered
        inside it.
        """
        shake = st.get("handshake") or {}
        pending = tuple(shake.get("pending") or ())
        failed = tuple(shake.get("failed") or ())
        if st.get("state") != "waiting_models" or not pending or not failed:
            self._handshake_prompted = False
            self._handshake_key = None
            return
        # keyed on the round too: a repeated failure is a new question, and the
        # user must be asked it again rather than silently timed out
        key = (shake.get("round"), pending, failed)
        if key != self._handshake_key:
            # first sight of this round: let it finish before asking about it
            self._handshake_key = key
            self._handshake_prompted = False
            return
        if self._handshake_prompted:
            return
        self._handshake_prompted = True
        # hand off to a fresh event-loop turn so this poll returns immediately
        QTimer.singleShot(0, lambda s=dict(shake): self._ask_keep_waiting(s))

    def _ask_keep_waiting(self, shake: dict):
        who = "、".join(shake.get("failed") or shake.get("pending") or [])
        answered = "、".join(shake.get("answered") or []) or t("无")
        box = QMessageBox(self)
        box.setWindowTitle(t("模型未就绪"))
        box.setIcon(QMessageBox.Question)
        # stay on top of whatever the user has focused: this question blocks the
        # race, so it must not be able to hide behind another window
        box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        box.setText(t("{who} 连接失败，尚未就绪。\n\n已就绪：{answered}",
                      who=who, answered=answered))
        box.setInformativeText(
            t("要继续等下去吗？选「直接开始」将不等回复立刻倒计时")
            + t("（两车起步时间会不一致，本局数据不可比）。"))
        wait_btn = box.addButton(t("继续等待"), QMessageBox.AcceptRole)
        box.addButton(t("直接开始"), QMessageBox.RejectRole)
        box.raise_()
        box.activateWindow()
        box.exec_()
        answer_handshake("wait" if box.clickedButton() is wait_btn else "go")


    def _show_data_tab(self):
        self.tabs.setCurrentIndex(self.data_tab_index)
        self.load_session()
        if self.match_table.rowCount():
            self.match_table.selectRow(self.match_table.rowCount() - 1)

    # ==================================================================
    # session data
    # ==================================================================
    def refresh_session(self):
        matches = self.store.list_matches()
        self.chip_matches.set_value(str(len(matches)))
        updated = "-"
        if matches:
            updated = str(matches[-1].get("timestamp", "-"))[5:16] or "-"
        self.chip_dirty.set_value(updated)

    def load_session(self):
        matches = self.store.list_matches()
        self.match_table.setRowCount(len(matches))
        for r, m in enumerate(matches):
            models = m.get("models") or ["", ""]
            winner = m.get("winner")
            scores = m.get("scores") or [None, None]
            cells = [str(m.get("match_id", "")),
                     str(m.get("timestamp", ""))[5:16],
                     models[0] if models else "",
                     models[1] if len(models) > 1 else "",
                     "-" if winner is None else f"P{winner + 1}",
                     "" if scores[0] is None else str(scores[0]),
                     "" if scores[1] is None else str(scores[1])]
            for c, val in enumerate(cells):
                self.match_table.setItem(r, c, QTableWidgetItem(val))
        self.match_table.resizeColumnsToContents()
        self.match_table.horizontalHeader().setStretchLastSection(True)
        self._refresh_aggregate()
        self.refresh_session()
        self._set_detail(None)

    def _refresh_aggregate(self):
        agg = self.store.aggregate()
        self.agg_table.setRowCount(len(agg))
        keys = ["model", "matches", "win_rate", "finish_rate", "avg_score",
                "avg_speed", "avg_wall", "avg_obs"]
        for r, row in enumerate(agg):
            for c, k in enumerate(keys):
                self.agg_table.setItem(r, c, QTableWidgetItem(str(row.get(k, ""))))
        self.agg_table.resizeColumnsToContents()
        self.agg_table.horizontalHeader().setStretchLastSection(True)
        self.agg_chart.set_rows(agg)

    def _set_detail(self, analysis):
        if not analysis:
            self.detail_head.setText(t("选择左侧一条比赛记录查看详情"))
            self.detail_table.setRowCount(0)
            fill_advice(self.detail_advice, [])
            return
        self.detail_head.setText(
            t("第 {mid} 局   ", mid=analysis.get("match_id"))
            + t("圈数 {laps}   用时 {duration}s   ",
                laps=analysis.get("laps"), duration=analysis.get("duration"))
            + str(analysis.get("timestamp", "")))
        self.detail_table.load(analysis)
        lines = []
        for a in analysis.get("analyses", []):
            lines.append(f"[{a['name']}] {a['model']}  "
                         f"{a['total']} ({a['grade']})")
            lines.extend(a.get("advice", []))
        fill_advice(self.detail_advice, lines)

    def show_selected_match(self):
        rows = self.match_table.selectionModel().selectedRows()
        if not rows:
            return
        try:
            mid = int(self.match_table.item(rows[0].row(), 0).text())
            payload = self.store.load_match(mid)
        except Exception as exc:  # noqa: BLE001
            self.detail_head.setText(t("加载失败: {err}", err=exc))
            return
        self._set_detail(payload.get("analysis", {}))

    # ---- session actions ----------------------------------------------
    def on_export_report(self) -> bool:
        if self.store.count() <= 0:
            QMessageBox.information(self, t("导出训练报告"), t("当前会话还没有对局数据"))
            return False
        path, _ = QFileDialog.getSaveFileName(self, t("导出训练报告"),
                                              "training_report.pdf",
                                              t("PDF 文件 (*.pdf)"))
        if not path:
            return False
        try:
            self.store.export_report(path)
            self.status_pill.set_state("idle", t("报告已导出"))
            return True
        except ImportError:
            QMessageBox.critical(self, t("导出失败"),
                                 t("生成 PDF 需要 Pillow"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, t("导出失败"), str(exc))
        return False

    def on_open_folder(self):
        self._open_dir(self.store.matches_dir)

    def _open_dir(self, path: str):
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            QMessageBox.information(self, t("数据路径"), path)

    # ==================================================================
    # history (the records that outlive a session)
    # ==================================================================
    def load_history(self):
        """Refresh the sidebar's short list and the window, if it is open."""
        matches = self.history.list_matches()
        self.history_list.clear()
        for m in matches:
            models = m.get("models") or ["", ""]
            winner = m.get("winner")
            who = "-" if winner is None else f"P{winner + 1}"
            text = (f"#{m.get('match_id')}  "
                    f"{str(m.get('timestamp', ''))[5:16]}  "
                    f"{str(models[0])[:10]} vs {str(models[1])[:10]}  "
                    + t("胜 {who}", who=who))
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, m.get("match_id"))
            self.history_list.addItem(item)
        if not matches:
            self.history_list.addItem(t("（暂无历史记录）"))
        # the window keeps itself current on its own timer, and its incremental
        # refresh keeps the user's selection -- a full load() here would clear
        # the detail pane under them once a second
        if self.history_dialog is not None:
            self.history_dialog.refresh()


def launch():
    # no setStyle / setStyleSheet: use the platform's native widgets
    app = QApplication.instance() or QApplication(sys.argv)
    # On every platform, not just Windows: the widgets that set an explicit
    # font would otherwise use the resolved family while everything around
    # them used the platform default -- the mismatch that read as "conflict".
    app.setFont(role_font("body"))
    win = ConfigWindow()
    win.show()
    app.exec_()
