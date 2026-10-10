# AI Car Race — 3D 双人赛车

一个用 **Python + OpenGL** 写的 3D 双人赛车。每局随机生成赛道，支持
**AI 对 AI** 和 **人类 对 AI** 两种模式，双方各自配置一个视觉大模型来开车。
自带独立的 PyQt5 配置窗口、无窗口运行模式、赛后评分与 PDF 训练报告导出。

---

## 1. 功能

- **随机赛道**：每局开赛时生成一张新赛道。赛道两侧程序化摆放带屋顶的立方体建筑，
  路面分布锥桶 / 油桶 / 木箱，撞到会减速。
- **碰墙减速**：车辆越界会被夹回并减速，碰撞角度越正面减速越多（护盾可免疫）。
- **道具系统**：5 种道具，各自独立冷却，冷却期间不可再次使用。
- **卡住脱困**：连续 2.5s 没有沿赛道推进即判定卡住，自动放回赛道中线并对准方向。
  这是引擎机制，不占道具，不会被判违规。
- **AI 视觉驾驶**：每一方都是一个视觉大模型。程序按 `decision_interval`（默认 1s）
  把当前第一人称 3D 驾驶画面发给模型，模型返回驾驶策略。
- **双方独立模型**：P1 / P2 各自选择提供方、模型、密钥、温度。
- **多厂商支持**：`ollama`（本机服务，无需密钥）、`openai`、`anthropic`，
  以及 DeepSeek / 智谱 GLM / 通义千问 / Kimi / 豆包 / 硅基流动 / MiniMax / 零一万物、
  任意 OpenAI 兼容接口。国内厂商的接口地址已内置，只需填 API 密钥。
- **模型扫描**：联网拉取所选厂商的可用模型列表，直接填进下拉框。
- **发车前握手**：开赛前同时给双方模型发一条「即将开始」的消息，双方都确认后
  才开始 3-2-1 倒计时，避免赛车在起跑线上空等。
- **视角**：AI 对 AI 可切换观察任一 AI（含第一人称）；人类对 AI 仅玩家视角。
- **3D 画面**：MSAA 4x 抗锯齿、方向光 + 高光 + 半球环境光、天空渐变与距离雾、
  路肩 / 护栏 / 起终点门架、蒙皮车身与圆角车模、贴地接触阴影。
- **HUD**：默认极简——左右单行车卡（时速 / 圈数 / 状态）+ 顶部圈数计时与比分
  + 底部道具条 + 事件滚动。按 `H` 展开详细遥测，`--debug` 叠加 AI 决策面板。
- **赛后分析**：每局生成数据对比表 + 总体评分 + 改进建议；跨局按模型汇总。
- **训练报告**：一键导出 PDF 训练报告。
- **历史数据**：每局自动存一份到 `runtime/history/`，控制台里可浏览、删除、导出。
- **反作弊**：实时校验动作 / 遥测 / 逆行，严重违规立即判负并结束本局。
- **英文控制台**：`--lang en` 把控制台界面切成英文（游戏内 HUD 与 PDF 报告保持中文）。
- **暂停**：游戏中按 `P`，或在控制台点「⏸ 暂停」。

---

## 2. 环境要求

- Python 3.10+（开发环境 3.12）
- 支持 OpenGL 3.3 及以上的显卡（项目在 NVIDIA RTX 4060 / OpenGL 4.6 上验证）
- Windows / Linux / macOS 均可

安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖：`numpy`、`scipy`、`pygame`、`PyOpenGL`、`PyQt5`、`Pillow`（PDF 训练报告）；
`openai` 一个包即可覆盖 OpenAI、Ollama 与全部国内 OpenAI 兼容厂商，只有 Anthropic
需要额外的 `anthropic`。

> 配置窗口使用 **PyQt5**。如果不想或无法安装 PyQt5，可以用
> `python main.py --shell` 进入文本指令模式，或用 `--run` 直接跑对局。

### Linux 说明

代码本身跨平台，Linux 下额外注意两点：

- **中文字体**：游戏内 HUD 与 PDF 训练报告需要一套中文字体。程序会按
  fontconfig（`fc-list :lang=zh`）和各发行版的常见路径自动查找；若一个都没找到，
  请安装一套 CJK 字体，例如 Debian/Ubuntu `sudo apt install fonts-noto-cjk`、
  Arch `sudo pacman -S noto-fonts-cjk`、Fedora `sudo dnf install google-noto-sans-cjk-fonts`，
  或 `wqy-microhei` / `wqy-zenhei`。缺少字体时中文会显示为方框。
- **桌面环境**：图形配置窗口 / 3D 窗口需要 X11 或 Wayland 会话；纯服务器请用
  `--shell` 或 `--run --headless`。Rust 反作弊二进制在 Linux 下为不带 `.exe` 的
  `anticheat/target/release/ac-validate`，`python main.py --anticheat build` 会自动编译。

---

## 3. 快速开始

### 图形界面（默认）

```bash
python main.py
```

会打开独立的 **PyQt5 配置窗口**。设置完成后点击 **▶ 开始对局**，游戏以子进程方式
在新窗口启动；随时点击 **■ 停止** 结束（无限模式也适用）。

### 文本指令模式

```bash
python main.py --shell
```

在终端进入交互式文本 shell（适合服务器 / 无图形环境），支持 `run` / `set` /
`show` / `export-report` / `anticheat` 等命令，输入 `help` 查看。

### 命令行 / 无窗口

```bash
# 对 Ollama 本地服务（先 ollama serve 并 pull 一个视觉模型），5 局
python main.py --run --headless --matches 5 --laps 2 --p1 ollama --p2 ollama

# 云端模型对云端模型（无限对局，Ctrl+C 停止）
# 密钥从 config/keys.json 或环境变量读取，地址已内置
python main.py --run --headless --matches 0 --p1 deepseek --p2 qwen

# 不指定局数，跑到手动停止
python main.py --run --headless --matches 0 --p1 ollama --p2 deepseek

# 带调试叠加层运行（窗口模式，显示 AI 决策与模型调用）
python main.py --run --windowed --debug --laps 1 --matches 1 --p1 ollama --p2 ollama

# 编译 / 检查 Rust 反作弊二进制
python main.py --anticheat build
python main.py --anticheat check

# 使用保存的会话配置
python main.py --run --config runtime/session.json
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--shell` | 文本指令模式 |
| `--headless` / `--windowed` | 无窗口 / 有窗口运行 |
| `--matches N` | 对局局数，`0` 为无限 |
| `--laps N` | 每局圈数 |
| `--p1` / `--p2 PROVIDER` | 双方的模型提供方 |
| `--model1` / `--model2 NAME` | 双方的模型名 |
| `--mode ai_vs_ai\|human_vs_ai` | 对战模式 |
| `--debug` | 叠加 AI 决策与模型调用 |
| `--no-msaa` | 关闭 4x 抗锯齿 |
| `--export-report` | 导出当前会话 PDF 训练报告后退出 |
| `--lang en` | 英文控制台 |

---

## 4. 配置窗口说明

窗口是一体化「控制台」，使用 **Windows 原生界面风格**。顶部工具栏（开始对局 /
暂停 / 停止 / 保存设置 / 帮助 + 状态指示），左侧为**本次会话**（局数 / 更新时间 /
导出训练报告 / 打开数据文件夹），右侧为四个标签页：

- **对局设置**：对战模式、人类控制车辆、圈数、对局局数（或无限）、道具开关、
  无窗口运行、调试模式、4x 抗锯齿、窗口尺寸、初始视角、AI 难度。
  勾选「无限连续对局」后局数与圈数输入框会禁用。
- **AI 模型**：为 P1 / P2 分别设置提供方、模型、显示名称、API Base URL、API 密钥、
  温度、决策间隔、超时（连接相关的几项收在「高级」折叠组里）。
  切换提供方会自动填入内置地址与默认模型名；点「扫描模型」可联网拉取该厂商的
  可用模型。「测试连接」会验证模型是否支持图像输入。
  密钥只保存在 `config/keys.json`（已被 `.gitignore` 忽略），设置文件、会话文件与
  对局结果里都不会写入密钥，启动时自动回填。未填写密钥时自动读取环境变量
  （`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`ZHIPU_API_KEY`、
  `MOONSHOT_API_KEY`、`ARK_API_KEY` …）；Anthropic 还支持 `ANTHROPIC_AUTH_TOKEN`
  与 `ANTHROPIC_BASE_URL`。
- **运行状态**：顶部指标块（状态 / 对局 / 比分 / 赛道时间），两张车辆卡
  （大号时速 + 速度条 + 圈数 + 撞墙 / 撞障碍 / 道具），一张 **AI 决策与模型调用表**
  （驱动 / 模型 / 请求 / 错误 / 平均延迟 / 成功率 / 最近决策），以及反作弊卡片。
  停止后自动切到「数据分析」。
- **数据分析**：比赛记录表（局 / 时间 / 双方模型 / 获胜 / 评分）、跨局模型汇总表、
  单局详情（对比表 + 改进建议）；支持导出 PDF 训练报告、打开数据文件夹。

其他：游戏内 HUD 默认保持极简且**没有底板**——文字直接叠在画面上并带深色描边，
左上 / 右上各一张车卡，顶部中间是圈数 / 计时（AI 对 AI 时显示实时总比分），
底部中间是道具条，左下角是事件滚动。所有位置都由分辨率推导，540p 到 1080p
都不会互相重叠。人类对 AI 时左侧为人类、右侧为 AI。

### 游戏窗口按键

| 按键 | 作用 |
| --- | --- |
| ↑ / W | 加速 |
| ↓ / S | 刹车 |
| ← / A、→ / D | 左转 / 右转 |
| P | 暂停 / 继续 |
| 1 – 5 | 使用对应道具 |
| F1 / F2 / F3 / F4 | 追尾 / 第一人称 / 俯视 / 自由视角 |
| Tab / Space | AI 对 AI 时切换观察的 AI 车辆 |
| H | 切换 HUD 详细模式（默认极简） |
| Esc | 退出当前对局 |

---

## 5. 道具系统

| 道具 | 冷却 | 效果 |
| --- | --- | --- |
| 氮气加速 NITRO | 6s | 2.5s 内大幅提升极速与加速度 |
| 能量护盾 SHIELD | 10s | 3s 内免疫撞墙 / 撞障碍减速 |
| 电磁干扰 EMP | 12s | 令前方 42m 内对手减速 2s（超范围则失败并短锁） |
| 油污陷阱 OIL | 8s | 身后留下油污，对手驶过减速 |
| 紧急复位 RECOVER | 7s | 复位到赛道中线并恢复朝向，用于脱困 |

冷却由 `ItemController` 统一管理，冷却期间 `available()` 返回 `False`。

### 卡住自动脱困

除了 `RECOVER` 道具，引擎本身还有自动脱困：连续 `auto_unstick_delay`（默认 2.5s）
没有沿赛道推进就判定卡住，自动把它放回赛道中线、朝向对准赛道方向、速度压到 10 m/s，
并记一条事件「`P1 卡住，已自动回到赛道中线`」。

- 位置会优先挑选中线上不在障碍物里的点（必要时沿中线向前 4/8/12m 找空位），
  避免刚放回去又被同一个障碍弹住。
- 这是引擎侧的合法复位，会通知反作弊豁免一次「瞬移」判定，不会被判违规。
- 同一辆车两次脱困之间有 4s 冷却，不会反复传送。
- 对人类玩家同样生效（正常比赛几乎不会触发）。
- 可在配置里关闭：`MatchConfig.auto_unstick = False`（`auto_unstick_delay` 调整触发时间）。

---

## 6. AI 模型接口

### 支持的提供方

| 提供方 | 说明 |
| --- | --- |
| `ollama` | 本机 Ollama 服务，走它的 OpenAI 兼容接口（`http://localhost:11434/v1`），无需密钥 |
| `openai` / `anthropic` | 官方接口 |
| `deepseek` / `zhipu` / `qwen` / `moonshot` / `doubao` / `siliconflow` / `minimax` / `yi` | 国内厂商，接口地址已内置 |
| `compatible` | 自定义 OpenAI 兼容接口 |

全部提供方都是视觉模型。驱动以异步线程按 `decision_interval`（默认 1s）轮询。

### 观测与动作

每次请求发送两部分：

1. **一张第一人称 3D 驾驶视角的画面**（从自车车内向前看，能看到前方赛道、
   障碍物与对手青色车）；
2. 一个只含**本车仪表盘**的极小 JSON：当前速度、本车极速、已用比例、圈数、可用道具。

对对手的了解限制为一件事——相对位置差。观测里的 `rival` 只给出 `status`
（领先 / 落后 / 并排）与 `gap_m`（正数=我领先，负数=我落后），没有对手的车速、
走线或道具。

模型返回 JSON 策略：

```json
{ "lateral_bias": 0.0, "throttle_scale": 1.0, "aggression": 0.5,
  "use_item": "NITRO", "reason": "前方直线，准备加速" }
```

- 场景判断（弯道、障碍物、赛道边界、对手位置）来自图像。
- 落地方向与油门由程序做机械换算；障碍规避使用程序内置检测器对同一帧的检测结果。
- 收到新策略前沿用最近一次有效策略；若从未拿到任何策略，车辆会温和滑行等待。
- API 调用失败会记录 `api_errors`，成功率与错误数计入赛后评分。
- 使用真实在线模型时，无窗口模式会自动按实时速度运行，以便决策线程跟上。

### 调试

`--debug` 或勾选「调试模式」后，画面会显示每个 AI 的当前决策、决策理由、
提供方与模型名、API 请求次数 / 错误次数 / 平均延迟，以及最近一次模型原始响应。
「运行状态」页与每局后的控制台输出同样显示这些数据，`runtime/status.json`
的 `drivers` 字段实时写入。

---

## 7. 赛后分析与评分

每局结束后，程序根据遥测数据为双方生成：

- **对比数据表**：完赛、用时、平均 / 最高速度、撞墙次数、撞障碍次数、道具使用、
  平均横向偏移、进度、各分项与总分。
- **总体评分**（0–100，加权）：
  完成度 35% + 驾驶质量 25% + 平均速度 15% + 道具运用 13% + 稳定性 12%。
  等级：S ≥90，A ≥80，B ≥70，C ≥60，D <60。
- **改进建议**：依据撞墙 / 撞障碍 / 横向偏移 / 道具使用 / API 错误等阈值自动生成。

跨对局在「数据分析」页按模型汇总：场次、胜率、完赛率、平均分、平均速度、
平均撞墙 / 撞障碍、平均道具使用。

### 训练报告（PDF）

在左侧「本次会话」栏点 **导出训练报告**（或 `python main.py --export-report`），
会为当前会话生成一份 A4 PDF，包含：

- **总览**：场次、双方模型、胜平负、跨局汇总表；
- **平均车速对比柱状图**；
- **车速曲线**：每一局一张图，双方车速（km/h）随比赛时间变化，附该局统计表；
- **AI 日志**：每个 LLM 玩家每局的请求数 / 错误数 / 平均延迟 / 成功率 / 最近决策，
  以及各局的事件日志（道具使用、碰撞、冲线等）；
- **能力占比图**：双方 5 项能力（完成度 / 驾驶质量 / 平均速度 / 道具运用 / 稳定性）
  的堆叠条形对比；
- **得分点圆饼图**：每方最终得分由哪些分项构成。

车速曲线需要逐帧采样，因此对局结果中带 `series`（车速 / 横向偏移时间序列）与
`events`（事件日志）字段；更早的对局没有它们，报告会自动跳过曲线。

> 生成 PDF 只用 Pillow（无 matplotlib / reportlab）。图表用 Pillow 绘制，每页以
> 无损方式嵌入自写的极简 PDF 容器；中文用系统中文字体绘制，不依赖 PDF 内嵌字体。

---

## 8. 反作弊

反作弊在每一局中始终开启，无需手动操作。每帧对两位 AI 检查，**严重违规**会立即
弹出红色告警、判违规方负并结束本局（`end_reason`），同时写入 `runtime/status.json`
与该局 `result.config.anticheat`。普通操作（如道具冷却期间重复按键、数值轻微越界）
只记录，不会结束比赛。

检查项：

1. **动作校验**（`validate_action`）：`null_action`（空动作）、`non_finite`（NaN/Inf）、
   `throttle_range` / `brake_range` / `steer_range`（油门 / 刹车 / 转向越界）、
   `bad_item`（不存在的道具）。
2. **道具与遥测**（`validate_item_use` / `validate_telemetry`）：`item_cooldown`
   （冷却期强行用道具）、`teleport`（瞬移）、`speed_cap`（超过理论极速）、
   `bad_speed` / `non_finite_pos`（非法速度 / 坐标）。
3. **逆行检测**（`validate_wrong_way`）：`wrong_way`——车头方向与赛道方向相反
   且持续超过 1s。

显示位置：游戏画面顶部 / 底部的红色告警条（实时）、结算面板中的 `反作弊: ...` 行、
GUI「运行状态」页的 AI 面板、单局详情。

```bash
python main.py --anticheat build   # 可选：编译 Rust 批量审计模块（需要 cargo）
python main.py --anticheat check
```

> Rust 模块 `anticheat/` 仅用于可选的赛后批量审计，实时反作弊不依赖它。

---

## 9. 目录结构

```
ai-car-rase/
├── main.py                     # 入口：控制台 / 命令行运行
├── conftest.py                 # pytest 路径设置
├── requirements.txt
├── car_game/
│   ├── config.py               # 配置 dataclass + 持久化 + 密钥
│   ├── fonts.py                # 跨平台中文字体查找（HUD / PDF 共用）
│   ├── providers.py            # 视觉模型 provider 注册表
│   ├── mathutil.py             # 矩阵 / 向量、相机矩阵
│   ├── core_types.py           # Action / Observation
│   ├── track.py                # 随机赛道生成与空间查询
│   ├── cars.py                 # 车辆物理、胶囊碰撞、遥测
│   ├── items.py                # 道具定义与冷却控制器
│   ├── game.py                 # 单局比赛引擎 + 实时反作弊
│   ├── anticheat.py            # 反作弊校验（Python 实现 + Rust 调用封装）
│   ├── analysis.py             # 赛后评分 / 建议 / 汇总
│   ├── store.py                # 会话比赛存储 + 汇总 / 报告导出入口
│   ├── report.py               # PDF 训练报告（Pillow 自绘 + 极简 PDF 写出）
│   ├── runner.py               # 多局会话（窗口 / 无窗口）
│   ├── cli.py                  # 终端指令模式
│   ├── ai/
│   │   ├── base.py             # 驱动基类
│   │   ├── heuristic.py        # 规则 AI（保留供参考）
│   │   ├── llm.py              # 视觉 LLM 驱动（异步）
│   │   └── registry.py         # 驱动工厂
│   ├── vision/                 # 程序侧视觉（软件渲染 + 检测，无 OpenGL 依赖）
│   │   ├── fp_render.py        # 第一人称 3D 画面渲染
│   │   ├── render.py           # 俯视画面渲染 + world↔pixel 变换
│   │   ├── detector.py         # 内置 CV 检测器（可选导入 .pt/.pth 权重）
│   │   ├── local_model.py      # 渲染 + 检测
│   │   ├── client.py           # OpenAI / Anthropic 客户端工厂
│   │   ├── models_scan.py      # 「扫描模型」：拉取云端模型列表
│   │   └── check.py            # 连接时的视觉能力探测
│   ├── render/                 # OpenGL 渲染
│   │   ├── geometry.py         # 程序化几何
│   │   ├── gl.py               # 着色器 / Mesh / GL 状态
│   │   ├── models.py           # 车模与贴地阴影
│   │   ├── scene.py            # 赛道与场景静态网格
│   │   ├── camera.py           # 相机（追尾 / 第一人称 / 俯视 / 自由）
│   │   ├── hud.py              # 游戏内 HUD（极简 / 详细 / 调试三态）
│   │   └── renderer.py         # 帧编排
│   └── ui/                     # PyQt5 控制台
│       ├── theme.py            # 调色板 / 间距 / QSS
│       ├── widgets.py          # 卡片 / 指标块 / 进度条 / 对比表
│       ├── console.py          # 控制台主窗口
│       ├── history_window.py   # 「历史汇总」弹窗
│       └── config_window.py    # 兼容旧导入路径的 shim
├── tests/                      # pytest：几何绕序 / 赛道 / 场景 / 模拟
├── anticheat/                  # Rust 批量反作弊校验器 (ac-validate)
│   ├── Cargo.toml
│   └── src/main.rs
├── config/                     # settings.json, keys.json（运行时生成）
└── runtime/                    # session.json, status.json, session/
```

---

## 10. 注意事项

- 无窗口模式默认按实时节奏运行，好让异步的模型决策线程跟得上；想尽快批量跑完
  可以关掉它（配置文件里的 `match.realtime`）。
- 人类对 AI 模式下不提供「仅观看 AI 视角」的切换。
- `runtime/`（含比赛数据）与 `config/keys.json` 均为运行时数据，不纳入版本控制。
- **API 密钥只写入 `config/keys.json`**：设置文件、会话文件与对局结果里都是空字符串，
  启动时自动回填。
- 3D 窗口默认申请 4x MSAA 与 24 位深度缓冲；驱动不支持时自动回退，也可用
  `--no-msaa` 关闭抗锯齿。
- 若显卡不支持 OpenGL 3.3，请更新驱动或改用无窗口模式跑分析。
- 自测（无需显卡）：`python -m pytest tests/`。

---

## 11. 打包发布

仓库里有两套 Arch 打包配置，产物都是 `.pkg.tar.zst`：

### Nuitka 免依赖自包含包（推荐）

`arch/nuitka/` 用 [Nuitka](https://nuitka.net/) 把整个程序编译成
**standalone 发行树**：自带 Python 解释器、全部 `python-*` 模块（numpy /
pygame / PyOpenGL / PyQt5 / Pillow / openai …）、以及 Qt5 / SDL2 / X11
等原生库。安装机 `pacman -U` 时**不会**拉取任何 python 依赖。

```bash
cd arch/nuitka
./build.sh          # Nuitka 编译 + 打包原生库 + 生成 makepkg 源 tar 包
makepkg -f          # 产出 ai-car-rase-0.1.0-1-x86_64.pkg.tar.zst
sudo pacman -U ai-car-rase-0.1.0-1-x86_64.pkg.tar.zst
```

`build.sh` 会：

1. `nuitka --standalone --enable-plugin=pyqt5` 编译 `main.py`；
2. 运行 `bundle_libs.py`，把 `ldd` 闭包里缺失的 Qt5 / SDL2 / X11 / ICU /
   OpenBLAS 等约 120 个原生库复制进 `<dist>/lib/`；
3. 生成 `ai-car-rase-<ver>-nuitka.tar.zst` 供 `PKGBUILD` 打包。

启动器 `/usr/bin/ai-car-rase` 把 `LD_LIBRARY_PATH` 指向 `/usr/lib/ai-car-rase/lib`，
用户数据仍写到 `~/.local/share/ai-car-rase/`（可用 `AI_CAR_RASE_HOME` 覆盖）。

> 只有两样东西刻意不打包、仍由系统提供：**glibc / 动态加载器**，以及
> **OpenGL-GPU 驱动栈**（libGL、libEGL、libgbm、libdrm 与厂商驱动）——
> 后者必须与机器硬件匹配，冻结 Mesa 会破坏 NVIDIA/AMD 驱动。因此
> `PKGBUILD` 里的 `depends` 为空，任何能跑 3D 游戏的 Arch 系统都能直接安装。
>
> 已用「只有 glibc + 本包」的最小 chroot 验证：无窗口整局对局可正常跑完。

### 源码包（依赖发行版 python 包）

`arch/aur/` 是纯 Python 源码包，依赖 `python-numpy`、`python-pygame`、
`python-opengl`、`python-pyqt5` 等；体积小，适合已经装好 Python 生态的系统。
`cd arch/aur && ./build.sh -i` 编译并安装。
