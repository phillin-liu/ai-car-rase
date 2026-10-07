"""Console translations (Chinese / English).

Scope
-----
**The console only** -- the main window, the history window and the shared
widgets.  The in-game HUD, the debug vision window and the PDF report keep
their own text: they are separate surfaces with separate audiences, and
translating them was explicitly out of scope.

Keys are the Chinese source strings themselves.  That keeps the call sites
readable (``t("开始对局")`` says what it is), needs no invented symbol table,
and makes an untranslated string simply fall through unchanged instead of
raising -- which matters when the table inevitably drifts behind the code.

Strings that arrive *out of the data* -- the analysis' own metric names, which
are written into every stored match JSON -- are translated here too, at display
time, so no stored record ever has to be migrated.

Usage
-----
    from .i18n import t, set_language

    button = QPushButton(t("开始对局"))
    label.setText(t("历史记录 {n} 条", n=len(matches)))

Switched once at start-up from ``--lang``; nothing in the console changes
language at runtime.
"""
from __future__ import annotations

LANGUAGES = ("zh", "en")
DEFAULT_LANGUAGE = "zh"

_lang = DEFAULT_LANGUAGE


def set_language(code: str) -> str:
    """Select the console language; unknown codes fall back to Chinese."""
    global _lang
    code = str(code or "").strip().lower()
    if code.startswith("en"):
        code = "en"
    elif code.startswith("zh") or code in ("cn", "chs"):
        code = "zh"
    else:
        code = DEFAULT_LANGUAGE
    _lang = code
    return _lang


def language() -> str:
    return _lang


def t(text, **kwargs) -> str:
    """The translation of ``text``, with ``{name}`` placeholders filled in."""
    out = _EN.get(text, text) if _lang == "en" else text
    if not kwargs:
        return out
    try:
        return out.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        # a stray brace in a translation must not take down the UI thread
        return out


# ---------------------------------------------------------------------------
# Metric names that come back out of the stored analysis.  Display-time only:
# the keys in the JSON stay Chinese, so old and new records stay readable.
METRIC_ZH_TO_KEY = {
    "平均速度": "avg_speed",
    "撞墙次数": "wall_hits",
    "撞障碍物次数": "obstacle_hits",
    "道具使用次数": "item_uses",
    "完赛": "finished",
}

_EN: dict = {
    # -- main window / toolbar ------------------------------------------
    "AI 赛车 · 控制台": "AI Racing · Console",
    "AI 赛车 · 对局控制台": "AI Racing · Race Console",
    "开始对局": "Start race",
    "暂停": "Pause",
    "停止": "Stop",
    "保存设置": "Save settings",
    "帮助": "Help",
    "操作说明 / 道具 / 反作弊": "Controls / items / anti-cheat",
    "就绪": "Ready",

    # -- sidebar ---------------------------------------------------------
    "本次会话": "This session",
    "每局随机生成一张全新赛道（互不重复）。\n"
    "开始新对局会清除本次会话数据；每局都已归档进下方历史，\n"
    "不会被清除。":
        "Each race gets a brand-new random track.\n"
        "Starting a new race clears this session's data; every race is\n"
        "archived into the history below and is never cleared.",
    "局数": "Races",
    "更新": "Updated",
    "导出训练报告": "Export training report",
    "打开数据文件夹": "Open data folder",
    "历史记录": "History",
    "跨会话保留：开新对局不会清除。\n"
    "点「历史汇总」打开完整窗口（含删除 / 清空 / 导出）。":
        "Kept across sessions; starting a new race does not clear it.\n"
        "Click History summary for the full window (delete / clear / export).",
    "历史汇总": "History summary",

    # -- settings tab ----------------------------------------------------
    "对局设置": "Race settings",
    "对战模式": "Mode",
    "AI 对 AI": "AI vs AI",
    "人类 对 AI": "Human vs AI",
    "人类控制车辆": "Human-controlled car",
    "赛制": "Format",
    "无限连续对局": "Endless",
    "启用道具系统 (按 1-5 使用)": "Items (press 1-5)",
    "每局圈数": "Laps per race",
    "对局局数": "Number of races",
    "运行方式": "Run mode",
    "无窗口运行 (无界面模式)": "Headless (no window)",
    "调试模式 (显示 AI 决策/遥测)": "Debug (AI decisions / telemetry)",
    "4x 抗锯齿 (MSAA)": "Anti-aliasing (MSAA)",
    "窗口尺寸": "Window size",
    "初始视角": "Camera",
    "AI 难度": "AI difficulty",
    "每局开赛时随机生成一张全新且互不重复的赛道。\n"
    "游戏中按 H 可切换详细 HUD，调试模式下会另开窗口显示 AI 识别视角。":
        "A brand-new random track is generated for every race.\n"
        "Press H in game for the detailed HUD; debug mode opens a second\n"
        "window showing what each AI sees.",

    # -- models tab ------------------------------------------------------
    "AI 模型": "AI models",
    "玩家 P{n}": "Player P{n}",
    "测试 P{n} 连接 (含视觉检测)": "Test P{n} (with vision check)",
    "保存 API 密钥": "Save API keys",
    "提供方": "Provider",
    "选择或输入模型名称": "Pick or type a model name",
    "扫描模型": "Scan models",
    "联网拉取该提供方可用的模型列表":
        "Fetch this provider's available model list",
    "模型": "Model",
    "显示名称": "Display name",
    "显示": "Show",
    "API 密钥": "API key",
    "高级 · 连接与采样": "Advanced · connection & sampling",
    "留空则使用内置地址 / 环境变量": "Blank uses the built-in URL / env var",
    "温度": "Temperature",
    "决策间隔(秒)": "Decision interval (s)",
    "超时(秒)": "Timeout (s)",
    "P{n}: 正在扫描模型 ...": "P{n}: scanning models ...",
    "P{n}: 扫描失败 — {err}": "P{n}: scan failed — {err}",
    "没有返回模型": "no models returned",
    "P{n}: 扫描到 {count} 个模型，已填入下拉框":
        "P{n}: {count} models found, filled into the list",
    "P{n}: 正在测试视觉能力 ...": "P{n}: testing vision ...",
    "正在测试 P{n} 的视觉能力 ...": "Testing P{n} vision ...",
    "失败: {err}": "Failed: {err}",
    "✓ 视觉模型": "✓ vision model",
    "✗ 非视觉模型": "✗ not a vision model",
    "✗ 未通过": "✗ failed",

    # -- live tab --------------------------------------------------------
    "运行状态": "Live status",
    "状态": "State",
    "空闲": "Idle",
    "对局": "Race",
    "比分": "Score",
    "赛道时间": "Track time",
    "赛道与实时位次": "Track & running order",
    "车速 / 赛道时间": "Speed / race time",
    "AI 调用日志": "AI call log",
    "只看失败与错误": "Failures only",
    "仅显示失败的调用": "Failures only",
    "尚无调用记录。": "No calls recorded yet.",
    "每次请求的成败与回复摘要": "Outcome and reply summary per request",
    "时间": "Time",
    "方向": "Side",
    "结果": "Result",
    "摘要": "Detail",
    "反作弊": "Anti-cheat",
    "实时检测中：每局校验动作 / 遥测 / 逆行，违规会立即判负。":
        "Live: every race checks actions / telemetry / wrong-way; a "
        "violation loses the race immediately.",

    # -- data tab --------------------------------------------------------
    "数据分析": "Data",
    "比赛记录": "Matches",
    "局": "#",
    "P1 模型": "P1 model",
    "P2 模型": "P2 model",
    "获胜": "Winner",
    "P1 评分": "P1 score",
    "P2 评分": "P2 score",
    "模型总体表现（跨对局汇总）": "Model summary (all races)",
    "场次": "Races",
    "胜率%": "Win %",
    "完赛率%": "Finish %",
    "平均分": "Avg score",
    "平均速度": "Avg speed",
    "平均撞墙": "Avg wall hits",
    "平均撞障碍": "Avg obstacle hits",
    "赛后分析详情": "Race analysis",
    "选择左侧一条比赛记录查看详情": "Pick a race on the left for details",
    "选择左侧一条历史记录查看详情": "Pick a record on the left for details",
    "选择上方一条比赛记录查看详情": "Pick a race above for details",
    "设置已保存": "Settings saved",
    "密钥已保存": "Keys saved",
    "指标": "Metric",
    "平局": "Ties",
    "平局 {n}": "Ties {n}",

    # -- notices / dialogs -----------------------------------------------
    "提示": "Notice",
    "已有对局在运行中": "A race is already running",
    "缺少 API 密钥": "Missing API key",
    "没有选择模型": "No model selected",
    "无法开始：AI 还没配置好": "Cannot start: the AI is not configured yet",
    "请在「AI 模型」页填写 API 密钥（点「扫描模型」可拉取可用模型）。":
        "Fill in the API key on the AI models tab (Scan models fetches the "
        "available ones).",
    "想完全离线可改用 Ollama：先在本机运行 ollama serve 并 pull 一个视觉模型。":
        "For a fully offline setup use Ollama: run `ollama serve` locally and "
        "pull a vision model.",
    "启动失败": "Failed to start",
    "退出": "Quit",
    "对局仍在运行，确定退出并停止吗?": "A race is still running. Quit and stop it?",
    "对局已结束": "Race finished",
    "操作说明": "Controls",
    "游戏按键": "Game keys",
    "  {n}. {name}（冷却 {cd:.0f}s）— {desc}":
        "  {n}. {name} (cooldown {cd:.0f}s) — {desc}",
    "游戏按键\n"
    "  ↑/W 加速    ↓/S 刹车    ←/A →/D 转向\n"
    "  1-5 使用道具    P 暂停    Esc 退出本局\n"
    "  F1 追尾  F2 第一人称  F3 俯视  F4 自由视角\n"
    "  Tab/Space AI 对 AI 时切换观察车辆\n"
    "  H 切换详细 HUD（默认极简）\n":
        "Game keys\n"
        "  ↑/W accel    ↓/S brake    ←/A →/D steer\n"
        "  1-5 use item    P pause    Esc end this heat\n"
        "  F1 chase  F2 first person  F3 overhead  F4 free camera\n"
        "  Tab/Space switch the watched car in AI vs AI\n"
        "  H detailed HUD (minimal by default)\n",
    "反作弊（始终开启）\n"
    "  逆行、瞬移、超速、非法数值、不存在的道具会立即判负并结束本局；\n"
    "  冷却期重复按键等普通操作只记录，不影响比赛。":
        "Anti-cheat (always on)\n"
        "  Wrong way, teleporting, speeding, illegal values and unknown items\n"
        "  lose the race immediately; ordinary input slips such as pressing an\n"
        "  item while it is cooling down are only logged.",

    # -- status pill / live state -----------------------------------------
    "⏸  暂停": "⏸  Pause",
    "▶  继续": "▶  Resume",
    "对局运行中": "Race running",
    "已暂停": "Paused",
    "正在停止…": "Stopping…",
    "比赛中": "Racing",
    "运行中": "Running",
    "本局结束": "Heat over",
    "已完成": "Done",
    "已停止": "Stopped",
    "等待模型就绪": "Waiting for models",
    "模型不可用": "Model unavailable",
    "等待画面…": "Waiting for a frame…",

    # -- handshake dialog --------------------------------------------------
    "模型未就绪": "Model not ready",
    "{who} 连接失败，尚未就绪。\n\n已就绪：{answered}":
        "{who} failed to connect and is not ready.\n\nReady: {answered}",
    "要继续等下去吗？选「直接开始」将不等回复立刻倒计时":
        "Keep waiting? \"Start now\" begins the countdown without waiting",
    "（两车起步时间会不一致，本局数据不可比）。":
        " (the two cars will start at different times, so this race is not "
        "comparable).",
    "继续等待": "Keep waiting",
    "直接开始": "Start now",
    "发车前握手：已就绪 {answered} ｜ 未回复 {pending}":
        "Pre-race handshake: ready {answered} | pending {pending}",
    "无": "None",
    "人类": "Human",
    "视觉LLM": "Vision LLM",
    "本地视觉": "Local vision",
    "成功": "OK",
    "失败": "Failed",
    "⚠ 本局记录 {count} 处违规": "⚠ {count} violation(s) this race",
    "实时检测中：未发现违规。": "Live check: no violations found.",

    # -- match detail ------------------------------------------------------
    "第 {mid} 局   ": "Race {mid}   ",
    "圈数 {laps}   用时 {duration}s   ": "Laps {laps}   Time {duration}s   ",
    "加载失败: {err}": "Load failed: {err}",
    "当前会话还没有对局数据": "No race data in this session yet",
    "PDF 文件 (*.pdf)": "PDF files (*.pdf)",
    "报告已导出": "Report exported",
    "导出失败": "Export failed",
    "生成 PDF 需要 Pillow":
        "PDF export needs Pillow",
    "数据路径": "Data path",
    "胜 {who}": "Winner {who}",
    "（暂无历史记录）": "(no history yet)",

    # -- history window ------------------------------------------------------
    "历史数据 · 跨会话保留": "History · kept across sessions",
    "历史记录 0 条": "0 records",
    "历史记录 1 条": "1 record",
    "历史记录 {n} 条": "{n} records",
    "删除选中": "Delete selected",
    "清空全部": "Clear all",
    "导出历史报告": "Export history report",
    "打开历史文件夹": "Open history folder",
    "全部历史对局": "All races",
    "编号": "#",
    "模式": "Mode",
    "人类对AI": "Human vs AI",
    "AI对AI": "AI vs AI",
    "历史汇总（按模型，跨全部对局）": "Summary by model (all races)",
    "历史第 {mid} 条   ": "Record {mid}   ",
    "（本次会话第 {no} 局）   ": "(session race {no})   ",
    "删除历史": "Delete record",
    "请先选中一条历史记录": "Select a record first",
    "确定删除历史记录 #{mid} 吗？该操作不可撤销。":
        "Delete record #{mid}? This cannot be undone.",
    "清空历史": "Clear history",
    "历史数据已经是空的": "History is already empty",
    "确定清空全部 {total} 条历史记录吗？该操作不可撤销。\n":
        "Clear all {total} records? This cannot be undone.\n",
    "（本次会话的数据不受影响，仍可在「数据分析」页导出）":
        "(This session's data is untouched and can still be exported from "
        "the Data tab)",
    "历史数据是空的": "History is empty",
    "已导出到\n{path}": "Exported to\n{path}",

    # -- provider registry --------------------------------------------------
    # The labels, signup hints and notes live in providers.py, which is shared
    # with the CLI and the stored config; they are translated here at display
    # time rather than in the registry itself.
    "Ollama（本地服务）": "Ollama (local)",
    "无需密钥；先在本机运行 ollama serve 并 pull 一个视觉模型":
        "No key needed; run `ollama serve` locally and pull a vision model",
    "需要本机安装并启动 Ollama。": "Requires Ollama installed and running.",
    "OpenAI（视觉）": "OpenAI (vision)",
    "platform.openai.com 申请，或设置 OPENAI_API_KEY":
        "Get one at platform.openai.com, or set OPENAI_API_KEY",
    "Anthropic（视觉）": "Anthropic (vision)",
    "console.anthropic.com 申请，或设置 ANTHROPIC_API_KEY":
        "Get one at console.anthropic.com, or set ANTHROPIC_API_KEY",
    "DeepSeek（深度求索）": "DeepSeek",
    "platform.deepseek.com 申请密钥": "Get a key at platform.deepseek.com",
    "智谱 GLM": "Zhipu GLM",
    "open.bigmodel.cn 申请密钥": "Get a key at open.bigmodel.cn",
    "通义千问（DashScope）": "Qwen (DashScope)",
    "dashscope.console.aliyun.com 申请密钥":
        "Get a key at dashscope.console.aliyun.com",
    "Kimi（月之暗面）": "Kimi (Moonshot)",
    "platform.moonshot.cn 申请密钥": "Get a key at platform.moonshot.cn",
    "豆包（火山方舟）": "Doubao (Volcengine Ark)",
    "console.volcengine.com/ark 申请密钥":
        "Get a key at console.volcengine.com/ark",
    "模型名称要填方舟的「接入点 ID」（ep-…），不是模型显示名。":
        "The model id must be the Ark endpoint id (ep-...), not the display "
        "name.",
    "硅基流动 SiliconFlow": "SiliconFlow",
    "siliconflow.cn 申请密钥": "Get a key at siliconflow.cn",
    "MiniMax": "MiniMax",
    "platform.minimaxi.com 申请密钥": "Get a key at platform.minimaxi.com",
    "零一万物 Yi": "01.AI Yi",
    "platform.lingyiwanwu.com 申请密钥":
        "Get a key at platform.lingyiwanwu.com",
    "自定义（兼容 OpenAI 接口）": "Custom (OpenAI-compatible)",
    "填写任意 OpenAI 兼容服务的地址与密钥":
        "Any OpenAI-compatible endpoint and key",
    "自建 vLLM / LM Studio / 中转站请用这一项。":
        "Use this for self-hosted vLLM / LM Studio / proxies.",

    # -- metric names that live inside the stored analysis --------------------
    # Display-only: the JSON keeps its Chinese keys, so old records stay valid.
    "完赛": "Finished",
    "用时(s)": "Time (s)",
    "最高速度": "Top speed",
    "撞墙次数": "Wall hits",
    "撞障碍物次数": "Obstacle hits",
    "道具使用次数": "Items used",
    "平均横向偏移": "Avg lateral offset",
    "行驶距离(m)": "Distance (m)",
    "进度": "Progress",
    "总分": "Total score",
    "完成度": "Completion",
    "驾驶质量": "Driving quality",
    "道具运用": "Item usage",
    "稳定性": "Consistency",

    # -- shared widgets ------------------------------------------------------
    "圈 —": "Lap —",
    "圈 {lap}/{laps}": "Lap {lap}/{laps}",
    "撞墙": "Wall",
    "撞障碍": "Obstacle",
    "道具": "Items",
    "✔ 完赛": "✔ Finished",
    "氮气": "Boost",
    "护盾": "Shield",
    "减速": "Slow",
    "等待比赛数据…": "Waiting for race data…",
    "开始对局后显示赛道": "The track appears once a race starts",
    "暂无跨对局数据": "No cross-race data yet",
}
