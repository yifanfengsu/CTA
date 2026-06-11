# 项目全面盘点报告 — cta_strategy（2026-06-11）

> **读者**：未读过本仓库任何文件的外部战略顾问。本报告自包含，用于决定新研究的方向与优先级。
> **性质**：纯只读盘点。未改任何文件（本报告是唯一新增文件）、未跑回测、未做数据库写操作、未碰 VPS。
> **标签约定**：
> - **[事实]** — 有据可查（附出处路径）；
> - **[作废结论]** — 基于已确认污染的 OKX DEMO 行情数据得出，证据基础已失效；
> - **[待重验假设]** — 机制性直觉，在干净数据上未验证。

## 项目一句话背景

OKX 永续合约 CTA 策略研发项目（vnpy 框架，本地 WSL + VPS 实盘）。13 个月研究（2024 Q4 → 2026 Q2，趋势跟踪 → 4h 均值回归 → 5m 均值回归）于 2026-06-10 被确认**全部建立在 OKX DEMO 模拟盘行情上**（`.env` 配置错误导致所有历史 K 线下载带 `x-simulated-trading: 1` 头）。用真实 mainnet 数据重建数据库并重验后，主力策略 MR-5m（5 分钟 Donchian 通道反向交易）确认**无 edge**（毛利≈0，净亏 100% 为手续费），项目于 2026-06-11 正式关闭。[事实：`reports/MR5M_postmortem.md`]

当前状态：持有一份干净的 3.4 年 mainnet 1 分钟数据库、一套经实战验证的执行工程栈、一套被事故锤炼过的研究方法论，和 70+ 个绩效数字全部作废的历史研究档案。新策略研究即将从零开始。

---

# 第 1 部分：基础设施盘点

## 1a. 数据资产

### 主数据库 `.vntrader/database_mainnet.db`（唯一可信回测数据源）

| 项 | 值 | 出处 |
|---|---|---|
| 币种 | BTC / ETH / SOL / LINK / DOGE（USDT 永续，OKX） | 库内 `dbbardata`（本次只读查询核实） |
| 周期 | 仅 1m（5m 等高周期由研究脚本内存聚合） | 同上 |
| 跨度 | 2023-01-01 00:00 → 2026-05-28 23:59（Asia/Shanghai naive，bar open 时刻） | 同上 |
| bar 数 | 每币 1,791,360 根，5 币共 8,956,800 根 | 同上 |
| 完整性 | 分钟网格**零缺口**（5 币 gap_count=0 / missing_minutes=0，未做任何填补/插值） | `reports/regime/mainnet_rebuild_20260610/integrity_summary.json` |
| 来源元数据 | 库内 `download_meta` 表 5 行，全部 `server=MAINNET / source=okx-public-rest`，含脚本版本与起止时间；另有 5 份 manifest 显式 `"server": "MAINNET"`、`"demo_header": false` | 本次只读查询 + `reports/regime/mainnet_rebuild_20260610/download_manifests/` |
| 下载方式 | OKX 公开 REST `history-candles`，硬编码 mainnet，不读 `.env`、无鉴权头 | `scripts/download_mainnet_history.py` 头部注释 |
| 体积 | ~1.41 GiB | 本次 `ls` 核实 |

**重要缺口 [事实]**：该库**尚未与任何外部独立源做抽样交叉验证**。重建任务的可行性探测确认了 Binance vision 月度 1m zip 可作为交叉验证源（`reports/regime/mainnet_rebuild_20260610/feasibility_probe.json`），但"≥3 个随机日逐 bar 对比"这一步**从未执行**——全仓库 grep 无任何 Binance 对比脚本或报告。CLAUDE.md 数据铁律要求"新数据与外部独立源抽样交叉验证后才可用于研究"，严格地说 mainnet 库本身还没过这一关（mainnet 基线重验是在未做此项的情况下跑的）。间接佐证存在：探测时 5 币起点价格与真实市场吻合、demo-vs-mainnet 对比中 mainnet 侧行为符合真实市场预期，但正式核查未做。

### 旧库 `.vntrader/database.db`（已确认污染，研究禁用）

- 同样 5 币 × 1,791,360 根 1m（区间与新库完全相同），**100% 经 vnpy_okx gateway-DEMO 路径写入**（29/29 connect 会话均 DEMO、~2090 chunks 均 gateway、2026-05-07~09 的全量重下覆盖了全部历史）。[事实：`reports/regime/data_contamination_forensics_20260610/`]
- 仅保留两个用途：取证基准、demo-vs-mainnet 对比。CLAUDE.md 铁律：**严禁用于任何回测或研究**。
- 库表 `dbbardata` 无来源列——这正是污染 13 个月不显形的结构原因之一。

### 资金费率数据 `data/funding/`

| 项 | 值 |
|---|---|
| 覆盖 | 5 币，2023-01-01 → **2026-03-31**（每币 3,558 条记录 + 表头），8h 结算粒度 |
| 形态 | `data/funding/okx/` 整合 CSV（含 raw_json 原文）+ `data/funding/okx_historical_raw/` 每币 39 个月度原始包 |
| 缺口 | **2026-04 → 2026-05 缺失**（K 线已到 2026-05-28，资金费率落后约 2 个月） |

**环境核查状态（任务点名的未检查问题）[事实 + 诚实标注]**：本次盘点做了代码检视——`scripts/download_okx_funding_history.py` 用裸 `urllib` 请求硬编码的公开 URL `https://www.okx.com/api/v5/public/funding-rate-history`，**全文无 `x-simulated-trading` 头、不读 `OKX_SERVER`**；`download_okx_historical_funding_files.py` 下载的是 OKX 官网历史数据页的 zip 文件，同样无鉴权头。因此**初步判断资金费率数据走的是 mainnet 公开路径，不太可能经过 DEMO 污染通道**。但必须如实标注：(1) 这只是本次盘点的快速代码检视，不是 K 线那种 H1∧H2∧H3 级别的正式取证；(2) 该数据从未与外部源交叉验证；(3) 取证报告只显式排除了 `import_okx_funding_csv.py` 写 K 线表的嫌疑，未对资金费率数据本身的环境做过结论。新研究若依赖 funding（例如 carry 类策略），应先补一次正式核查。

### 其他数据

- `data/history_manifests/`：28 份旧 K 线下载 manifest。**无 server/环境字段**（事故的结构根因之一，仅记 `source_used=gateway`）——只剩取证价值。[事实：本次抽查 + 取证报告]
- `data/raw/`：仅 3 个 BTC 1m CSV 导出（2025 年起的片段），来源为旧库时代，研究价值约等于零。
- `config/instruments/`：7 个币种的 OKX 合约规格 JSON（5 个主力 + BNB、XRP）——注意 BNB/XRP **只有合约规格、没有任何 K 线数据**。

### 数据缺口清单（新研究可能需要但现在没有的，只列事实）

| 缺口类型 | 现状 |
|---|---|
| 更多币种 | 只有 5 币；OKX 有 200+ USDT 永续。横截面类策略（动量排序、配对等）数据不足 |
| 更高频数据 | 只有 1m K 线；无 tick、无订单簿（L2）、无逐笔成交。旧库有空的 `dbtickdata` 表 |
| 衍生品确认数据 | 无历史 OI、taker buy/sell ratio、long-short ratio（2025 年的 readiness audit 已确认"覆盖未证明"，`reports/research/derivatives_data_readiness/`） |
| 现货数据 | 无（期现基差类策略不可做） |
| 链上数据 | 无 |
| 其他交易所行情 | 无（跨所价差/外部源校验需另行下载；Binance vision 已知可用） |
| 资金费率近 2 个月 | 缺 2026-04 之后 |
| 成交量可信度 | mainnet 库 volume 字段未单独验证过（demo 库 volume 已确认失真 6.6-20×，mainnet 的 volume 理应真实，但同样没做过外部核查） |

## 1b. 代码资产

### 主力回测引擎 `scripts/backtest_mr_5m_compare.py`（586 行）[事实：本次通读全文]

**定位**：MR-5m 实盘 runner 的 1:1 保真复刻。地位经 CLAUDE.md 确认不变（"引擎地位不变"）。

**能力边界（读代码后的判断）**：

| 维度 | 支持 | 不支持 / 硬编码 |
|---|---|---|
| 数据 | 1m 加载 + 内存聚合到 5m（聚合函数 `r5` 可聚合任意 N 分钟，改周期成本低） | `--database-path` **默认指向污染库 `database.db`**，用 mainnet 必须显式传参或外层注入 |
| 策略形态 | 单一形态：Donchian fade（突破上轨做空/下轨做多）+ 三优先级出场（midline 止盈 → 当根浮动 ATR 止损 → max_hold） | 入场/出场逻辑**内联在 ~100 行的 `backtest_symbol` 循环里**，无策略抽象层；策略参数（LB=24/ATR=14/stop 1.0×/MH=48）是模块级常量 |
| 仓位 | 单仓（flat→long/short→flat），固定名义 $500×5x，整数张数（ctVal 取整、[1,1000] 截断） | 不支持加仓/金字塔/对冲双向/组合层保证金占用 |
| 成交与费用 | 入场=信号 bar 收盘价 maker 限价（假设 100% 成交，返佣 −0.002%）；出场=收盘价 ∓1 tick taker（0.05%）；止损用当根 low/high 触发判定 | 无成交率模型、无逆向选择建模、**无资金费率**、无滑点压力测试参数 |
| 指标 | Wilder ATR（与实盘 BarAggregator 逐 bar 一致）、Donchian（严格排除当根） | 其他指标自己加 |
| 输出 | 每笔交易 jsonl（与实盘 trade_log 同 schema）+ 组合指标（PF/胜率/DD/连亏/分年）+ markdown 报告 | `--report` 路径**硬编码了 demo 时代的结论文案**（"保留 DOGE""2024 逆风年"等），用 mainnet 数据重跑会再生成已作废的叙述 |

**改造成趋势策略的工作量判断**：可复用的是数据加载、聚合、Wilder ATR、费用/成交模型、整数张数、metrics 与 jsonl 输出（约占代码 60%）；需要重写的是 `backtest_symbol` 的入场/出场状态机（~100 行）和场景/阈值常量。**量级是"写一个新的 300-400 行脚本并 import 其工具函数"，不是"改几个参数"**——已有先例：`research_mr5m_mainnet_baseline.py`（166 行）演示了外层注入数据源与阈值、引擎零修改的复用模式。趋势策略持仓远长于 48 根 bar，max_hold/浮动止损语义需重新设计；若需要资金费率（趋势持仓跨多个 8h 结算），引擎完全没有这部分，要新建。

### 其他回测脚本（均基于旧库时代，数字作废，代码可参考）

| 脚本 | 状态 |
|---|---|
| `backtest_mr_5m_dynsize.py` / `_dynsize_c2c3.py` | 动态仓位 + walk-forward 框架，**walk-forward 代码结构可复用** |
| `backtest_mr_5m_breaker.py` | 熔断器仿真（含虚拟 PnL 误伤分析实现，方法可复用） |
| `backtest_mr_5m.py` / `_v2.py` / `backtest_mr_v1.py` / `backtest_okx_mhf.py` | 早期版本/已停用策略，仅档案价值 |

### 数据工具

| 工具 | 可复用性 |
|---|---|
| `download_mainnet_history.py`（309 行） | **新标准**：硬编码 mainnet、不读 .env、断点续传、manifest 带 server 字段、缺口如实记录不填补、限频 8 req/s。新增币种/区间可直接扩展 SYMBOLS/RANGE 常量 |
| `download_okx_history.py`（2455 行，最复杂） | **污染通道本体**（gateway 主路径继承 .env 的 DEMO）。其 `--source rest` 备胎路径恒 mainnet 但从未被触发过。不建议继续使用；保留为取证对象 |
| `download_history.py` / `merge_recent.py` | **硬编码 demo header**（取证确认），未修复。禁用状态 |
| `verify_okx_history.py` / `verify_okx_funding_history.py` | 完整性校验（网格缺口/重复），与环境无关，可复用 |
| 资金费率三件套（download/import/verify） | 走公开 mainnet URL（见 1a），可复用，但建议先补环境核查 |
| `forensics_data_contamination.py` / `research_demo_vs_mainnet.py` | 只读取证 + 逐 bar 对比/可信度地图工具，未来数据质检可复用 |

### 执行栈 `scripts/run_mr_5m_direct.py`（1269 行，已关停归档，禁止随意修改）

经 VPS 模拟盘实战验证（2026-06-01 → 06-08 密集修复期，git log 可查 7+ 次 bug fix），**与行情数据污染无关（在真实 OKX 模拟撮合上验证），未来任何策略上实盘均可复用**的组件清单 [事实：本次通读函数清单 + `MR5M_postmortem.md` 第 6/7 节]：

| 组件 | 内容 |
|---|---|
| `BarAggregator` | 1m→5m 聚合 + Wilder ATR + Donchian + 仓位状态（信号部分策略专用，聚合/ATR 通用） |
| IOC 下单 | `place_order`：限价单 IOC，杜绝残留挂单导致本地/交易所状态脱钩（GTC→IOC 是实战教训） |
| 订单生命周期 | `check_order_filled` / `cancel_order` / 残单 3 次重试 / exit latch（每 bar 单发、不阻塞） |
| 仓位对账 | `sync_positions_from_okx`（启动/重连同步）+ `pos_verify`（每 5 分钟本地 vs OKX 比对，drift 即平仓）+ force_close |
| 费用对账 | `get_fills_fee`（fills API 实际手续费）+ OKX bills 对账体系（修复过 fee 串号、入场费双计两个真实 bug） |
| 通知 | `Notifier`：PushPlus 微信异步队列、3 次重试、HTTPS |
| WebSocket | `OKXTickerFeed` 连接管理 + ping/重连 |
| 运维 | trade_log 月度轮转 jsonl、6 小时摘要、优雅 shutdown、`restart_mr5m.sh` |

### 研究脚本沉淀的可复用模式

- **数据加载**：`research_mr_5m.load_1m`（旧库路径）/ `research_demo_vs_mainnet.load_1m_ro`（**mode=ro 只读打开，新研究标准做法**）。
- **外层注入**：`research_mr5m_mainnet_baseline.py` 的模式——import 引擎、注入数据源/阈值、引擎文件零修改。
- **train/test 切分**：全项目统一切点 2025-04-09 08:00 UTC（2/3-1/3 时间序），`parse_history_range` 时区工具。
- **walk-forward / 敏感性曲线 / 二维 pivot**：分别在 `backtest_mr_5m_dynsize_c2c3.py`、`research_atr_filter_oos.py`、`research_atr_ratio_decomp.py` 有现成实现。
- **集成测试惯例**：`tests/` 49 个文件，风格是"跑脚本 + 验输出文件"。

## 1c. 流程资产

### CLAUDE.md 当前硬约束（数据环境铁律，最高优先级，违反即停）[事实：`CLAUDE.md`]

1. 唯一可信回测数据源 = `database_mainnet.db`；旧 `database.db` 严禁用于任何回测或研究。
2. 任何写库脚本必须：启动时 stdout 打印数据环境（MAINNET/DEMO）；manifest 写 `server` 字段；新数据与外部独立源抽样交叉验证（≥3 随机日逐 bar）后才可用于研究。
3. `.env` 的 `OKX_SERVER` 不得被任何数据下载脚本隐式继承——数据环境必须显式命令行指定。
4. 新研究启动门槛：复盘第 8 节检查清单逐项过一遍。
5. 不修改归档实盘脚本与基准引擎；研究产出 = 独立脚本 + markdown 报告 + 中间 jsonl。

### 复盘第 8 节强制检查清单（新研究启动前逐项确认）[事实：`reports/MR5M_postmortem.md` §8]

1. **数据源验证**（同上铁律 2，含 Binance vision 抽样对比的具体做法）。
2. **回测假设显式化**：费率/滑点/成交假设写进每份报告头部，不许埋在代码里。
3. **基线优先**：任何想法先跑"无过滤、固定仓位"裸基线；**毛利≈0 或为负直接放弃**——优化器只能放大 edge，不能创造 edge。
4. **gate 不可在结果出来后修改**：阈值跑之前写死；差 0.023 也不放宽。
5. **否定性结论与肯定性结论同等对待**：同等证据标准、同等文档固化。

### PROJECT_GUIDE 维护规范 [事实：`CLAUDE.md`]

PROJECT_GUIDE.md 是反映"当前最佳认知"的活文档，维护方式是**外科手术式**：找到被推翻的具体行 → 修订正文（不是末尾追加）→ 被推翻处加日期注释 → 在产出报告里列"文档更新清单"和"未改动文档及原因"。必须维护两个章节："已验证的核心事实"与"已知未验证的假设"。单次研究典型改动 <30 行。

### 方法论工具箱（详见 2b）

五项诚实性检验、虚拟 PnL 误伤分析、DD 元凶切中率检验、取证三段论、敏感性曲线读法等——定义与出处见第 2b 节。

---

# 第 2 部分：研究历史盘点

> **总框架**：13 个月共 5 大阶段、150+ 个参数/配置组合被测试，结论是"只有 MR-5m Donchian fade 有 edge"。2026-06 确认全部数字基于 DEMO 污染数据；mainnet 重验后连这唯一的"有 edge"也被证伪。**全部历史绩效数字（含否定性结论）作废。**
> demo 污染的方向性（理解下表"先验重置"列的钥匙）[事实：`demo_vs_mainnet_comparison_20260610/` + `mr5m_mainnet_baseline_20260611/`]：demo 行情是"同步但带独立微观结构噪声"的镜像——价格中位偏差仅 0.003-0.03%，但 ~20% 的 1m bar 偏差 >0.1%；噪声形态以"假突破 + V 形瞬间复原"为主。这种噪声**系统性美化均值回归**（假突破立即回归 → midline 止盈白送，单笔虚增 +$9.98、胜率 48.3%→36.1%），并**系统性摧毁趋势跟踪**（同样的假反转打断趋势持仓）；volume 失真 6.6-20× 使一切成交量信号无效；ATR 系统性偏高（BTC 中位 +35.6%）使一切 ATR 基量级失真；2025-07→2026-03 持续脱锚使"近期表现"判断受害最深。

## 2a. 已探索的策略形态地图

### ① 趋势跟踪族（2024 Q4 → 2025 Q1，10+ 个家族）

- **思路**：加密货币存在持续趋势，用 Donchian/EMA 突破捕获。版本演进：1m Donchian breakout → HTF 多时间框架（1h regime + 15m 结构 + 5m 入场）→ V2 单品种 1h/4h → V3 五品种 4h/1d（Donchian/EMA/Ensemble 策略族）→ V3 扩展（2023-2026 全区间重测 + 真实资金费率调整）。
- **[作废结论]**：全部未通过 gate。无 stable candidate；亏损集中在 choppy/high_vol_choppy（占时间 38.7%）；"strong trend 仅占 4.79% 时间，且 V3 利润不来自 strong trend"——结论是"趋势策略在赚随机的钱"。唯一弱线索 1d EMA 50/200 金叉在无成本口径全正，但被 top-trade 集中度（OOS top 5% 贡献 198%）、funding 压力测试拒绝。[出处：`reports/CTA_strategy_failure_postmortem.md` §2、`reports/research/trend_following_v3*/`]
- **作废原因（方向性分析）**：demo 的假反转噪声**正面打击**趋势持仓——突破后被不存在的 V 形复原扫掉止损或回吐。趋势策略是 demo 污染的**最大受害者**。
- **先验重置**：**完全未知，且相对历史认知应上调**。"趋势跟踪全败"这一封路结论的证据是假的；干净数据上从未测过。注意两个仍可能成立的独立疑点：strong trend 占比是否真的只有 ~5%（regime 标签也是在 demo 数据上打的）需要重测；高 taker 成本结构对低频趋势相对友好（持仓长、笔数少）。

### ② 跨币种信号族（2025 Q2，5 个家族）

- **思路**：用多币种横截面信息择时——CSRB（跨币种反转广度）、VSVCB（突破 + squeeze + volume 确认）、早期趋势分类器（正向 + 反向）、breadth phase15、外部 regime 分类器（含可行性与 gate 审计）。
- **[作废结论]**：全部 train/validation/OOS 失败，全部未通过 gate。[出处：`reports/research/csrb_v1*/`、`vsvcb_v1*/`、`early_trend_classifier_v1*/`、`external_regime_classifier*/`]
- **作废原因**：双重污染——价格信号本身的微观结构是假的；**VSVCB 等依赖 volume 确认的策略用的成交量失真 6.6-20×**，这类研究在 demo 数据上压根没有被真正测过。
- **先验重置**：完全未知。volume 类信号在干净数据上等于从未研究过。

### ③ MR-v1（4h 均值回归，2025 Q2-Q3）

- **思路**：fade 4h Donchian 突破（LB=8、1.0×ATR 止损、midline 止盈）。三阶段研究全过 gate，曾报 Sharpe 3.82 / PF 5.38 / 36 个月全盈利——后发现 stop-fill look-ahead bug，修正后 PnL 从 +$28k → −$2.6k（**12× 利润虚增是回测 bug，与数据污染无关**，这条教训仍有效）。修 bug 后仍通过门控进入模拟盘，实盘不达预期 → 转向 5m。
- **[作废结论]**：4h fade 有弱 edge、5m 比 4h 好 35×（"5min 突破反转率 87-95% vs 4h 78-82%"）。
- **作废原因**：反转率本身就是 demo 噪声的直接产物——假突破 + 瞬间复原把"反转率"灌高，且周期越短噪声占比越大，**"5m 优于 4h"极可能是污染放大效应的体现**。
- **先验重置**：变坏。MR-5m 已在 mainnet 证伪（见④）；4h 版未在 mainnet 重测，但其立项依据（反转率）已不可信。

### ④ MR-5m（主力策略，2025 Q3 → 2026 Q2）— 唯一在干净数据上重验过的方向

- **思路**：fade 5m Donchian 突破（LB=24 / Wilder ATR 14 / 止损 1.0×ATR / midline 止盈 / max_hold 48 / 5 币 $500×5x）。
- **demo 时代结论 [作废]**：test PF 2.06 / +$189k、胜率 48.3%、5/5 币全周期 PF>1、Sharpe 5.83。
- **mainnet 重验结论 [事实，这是干净数据上唯一的已验证结论]**：**无 edge**。三配置（无过滤 C1 / 现行阈值 C2 / mainnet 重导出 p40 C3）全亏：全期 PF 0.81-0.85，test PF 0.81-0.83；毛利≈0（全期 +$6k~+$10k，发生在 $50M+ 成交额上，是噪声）；**净亏 100% 是手续费**（C2 全期：毛利 +$9,194，费用 −$76,088，净 −$66,894）；5 币 × 4 年 20 格无一 PF>1（最高 SOL 2023 = 0.99）；胜率 36%。且这是乐观口径（maker 100% 成交、1 tick 滑点、无 funding）。demo 毛利 edge 的 ~99% 是数据假象。[出处：`reports/regime/mr5m_mainnet_baseline_20260611/`]
- **先验**：**已关闭，不是先验问题**。信号无方向性信息，无"降本拯救"空间。

### ⑤ 出场优化族（MR-5m 之上，20+ 变体）

- **思路与 [作废结论]**：Chandelier exit（6 mult × 5 max_hold）全败；追踪止损全败；"midline 止盈是唯一有效出场（+$6.3k）"。
- **作废原因**：midline 的优越性正是 demo V 形复原直接喂出来的——假突破瞬间回中轨 = midline 止盈白送。**"midline 唯一有效"很可能是污染信号最浓的一条结论**。
- **先验重置**：随 MR-5m 关闭失去载体。机制教训"追踪止损截断 fade 策略的利润"降级为 [待重验假设]。

### ⑥ 入场过滤器族（36 组合 + 替代通道 12 组合）

- **[作废结论]**：ADX+EMA slope 24 组合全部劣于基线（"过滤器同时筛掉赢家和输家"）；Keltner 替代 Donchian 12 组合全负（"固定通道的笨拙是优势"）；ATR p30 regime filter 是唯一过 OOS 的过滤器（+1.9% 组合净利 / +0.08 PF，"是质量精修不是风控"——其 "−72% DD" 宣称已在 demo 时代被 OOS 检验撤回）。
- **mainnet 补充 [事实]**：ATR filter 在真实数据上只能"少交易 → 少交手续费 → 少亏"（C2 比 C1 少亏 $37k），不能创造 edge。[出处：`mr5m_mainnet_baseline_20260611/` Q3]
- **先验重置**："entry filter 对 MR 无效"的机制论降级为 [待重验假设]；但"过滤器不能把毛利≈0 变成正期望"在 mainnet 上获得了一次真实印证，可视为半个干净证据。

### ⑦ 反应式熔断器 + 组合层风控（74 + 若干配置）

- **[作废结论]**：连续止损/滚动 PF 熔断 74 配置全部负收益（误伤健康利润 > 省下的亏损，"靠少交易让指标变好是陷阱"）；组合层 R1 日亏损熔断同机制再证（误伤 +$33k-$45k）；R2 同向仓位上限 / R3 横截面降档安全阈值下对 DD 无效。[出处：`reports/regime/portfolio_risk_phase_2a_20260609/`]
- **先验重置**：载体已关闭。"反应式风控误伤大于节省"的机制逻辑有一般性，但量化依据全部来自污染数据，[待重验假设]。

### ⑧ 动态仓位族（DYN，2026 Q1-Q2 重心）

- **[作废结论]**：atr_ratio 百分位三档加注（C2-1：$250/$500/$750）过完整 walk-forward 与五项诚实性检验；atr_ratio 拆解发现真正的 regime 因子是 **ATR 绝对分位**（5/5 币 PF 0.49→2.84 单调），atr_ratio 只是代理变量（"押对结果用错原因"）；v2B 候选 +$290k 但 max DD 1.523× 微超 1.5× gate；size cap V1 15/15 失败（"高 ATR 笔是利润引擎不是 DD 元凶"）。
- **作废原因**：所有 ATR 分位/分层在 demo ATR 系统性偏高（+15~36%、p99 偏差可达数倍）之上构建；且 v2B 的 DD 元凶最终被诊断为 2025-05-29 SOL/DOGE 合成数据 ramp（数据异常，非市场模式）——这正是揭开整个事故的线头。
- **先验重置**：载体已关闭。"波动率绝对水平分层有信息量"是其中机制感最强的一条，[待重验假设]；但教训"仓位放大器不能改变毛利符号"（基线无 edge 时 DYN 无意义）在 mainnet 基线报告中被明确陈述（推断而非实测）。

### ⑨ 其他探索（一次性/支线）

信号特征研究（`research_signal_features.py`）、入场政策对比（`research_entry_policies.py`）、exit-only B/C 方案、5min 反转统计（`research_5min_reversal.py`）、alpha sweep（10 候选参数扫描）、消融实验（9 候选 × 两档成本）、市场画像（`market_profile_l4.py`）等——全部隶属上述框架的支线，数字一律作废，无独立机制结论需要单列。

## 2b. 方法论遗产（与结论无关、被实战验证有效的方法本身）

| 方法 | 一句话 | 出处 |
|---|---|---|
| train/test 时间切分纪律 | 全项目统一切点 2025-04-09 08:00 UTC（2/3-1/3），杜绝窗口挑选；OOS 评估用 entry 归属 | `reports/regime/atr_filter_oos_validation_20260609/` |
| 敏感性曲线读法 | 结论必须在阈值邻域（p20-p40）平坦才算稳健，尖峰即过拟合 | 同上 |
| 五项诚实性检验 | ①参数稳健（±1pct 净利变化<10%）②子周期（前后两半均胜基准）③单币种（5/5 胜基准）④平均名义 ≤1.1× 基准（防"靠加杠杆变好"）⑤避开已知负期望格 | `reports/regime/dyn_v2_design_20260609/` |
| walk-forward | 滚动训练-验证推进，防单窗口运气 | `backtest_mr_5m_dynsize_c2c3.py`、`dyn_v2_design_20260609/walk_forward/` |
| 二维 pivot 分解 | 把一个"有效因子"拆成两个候选维度的交叉表，识别代理变量（atr_ratio vs 绝对分位） | `reports/regime/atr_ratio_decomposition_20260609/` |
| 虚拟 PnL 误伤分析 | 风控暂停期间继续记虚拟交易，量化"省下的亏损 vs 误伤的利润" | 熔断器研究、`portfolio_risk_phase_2a_20260609/` |
| DD 元凶切中率检验 | 风控方案必须证明它切中的交易就是历史 DD 的构成交易（v1 cap 切中率仅 12.5-18.8% → 否决） | `reports/regime/dyn_v2b_size_cap_v1_20260609/` |
| gate 预先写死、事后不改 | 1.5× DD gate 差 0.023 拒绝放宽 → 触发 DD 诊断 → 揭开整条事故链 | `reports/regime/v2b_dd_diagnosis_20260610/` |
| 诊断允许"D 选项" | 问题设计预设 A/B/C 三种市场解释，但允许"以上都不是"——数据 artifact 才得以被命名 | 同上 |
| 取证三段论 H1∧H2∧H3 | 代码路径存在 ∧ 历史上被触发 ∧ 与污染特征吻合，三项逐一核实才定罪 | `reports/regime/data_contamination_forensics_20260610/` |
| 逐 bar 对比 + 可信度地图 | 两数据源 1:1 对齐后按 币种×月 打格子（green/yellow/red），含滞后检验排除技术假象 | `reports/regime/demo_vs_mainnet_comparison_20260610/` |
| 回测 bug 五清单（数据无关，仍有效） | ①stop-fill look-ahead（止损必须按止损价成交，曾虚增 12× 利润）②DD 分母用 peak 不用 notional ③intrabar 先查止损再更新极值 ④midline 禁用全局预计算 ⑤Calmar 分子分母同基 | `reports/CTA_strategy_failure_postmortem.md` §7 |
| 双账号对照实验 + bills 对账 | 实盘 A/B 对照设计、以 OKX bills 为唯一真相源对账 | git log 2026-06-01→08、`run_mr_5m_direct.py` |
| 否定性结论同等固化 | 死胡同与发现同等写进"已验证的核心事实"，防止重走也防止假封路 | `CLAUDE.md` 工作原则 |

## 2c. 数据事故链完整摘要（新研究铁律的来历，一页版）

1. **种子（≤2026-04-17）**：`.env` 写入 `OKX_SERVER=DEMO`（`.env.example` 模板默认值即 DEMO）。`download_okx_history.py` 主路径走 vnpy_okx gateway，gateway 在 DEMO 模式给**所有**请求（含公共历史 K 线）加 `x-simulated-trading: 1` → 返回 OKX 模拟盘行情。另两个增量脚本干脆硬编码了该 header。库表无来源列、manifest 无环境字段、脚本不打印 server——三层记录设施全部失明。2026-05-07~09 的 5 币全量重下覆盖整库 → **100% demo**。
2. **潜伏 13 个月**：demo 是"同步带噪"镜像（价格中位偏差 0.003-0.03%），月线/日线肉眼检查全过。期间"赚不赚""过拟合吗""执行偏差多大"都被反复挑战，唯独"数据是真的吗"从未被问。
3. **线头（2026-06-10）**：v2B 候选 max DD 1.523× 超 1.5× gate 仅 0.023，**拒绝放宽** → 立项 DD 诊断 → 把单日回撤拆到逐 bar → 发现 2025-05-29 SOL/DOGE 的数学上不可能形态（每分钟等差 ±10.3 阶梯、单根 K 线 ±91% 瞬间复原、同时刻 BTC/ETH/LINK 纹丝不动）→ 全库扫描检出 **598 个合成 ramp**（SOL 344 / DOGE 216 / ETH 23 / LINK 15 / BTC 0）。
4. **取证（同日）**：H1∧H2∧H3 三段论坐实污染通道与覆盖范围（29/29 下载会话均 DEMO）。598 个 ramp 只是可检出的子集，问题是全库性的。
5. **重建（同日）**：`download_mainnet_history.py`（硬编码 mainnet、带 `download_meta`/manifest 环境字段）从 OKX 公开 REST 重下 3.4 年 × 5 币，零缺口、零填补。
6. **定损（2026-06-10）**：demo vs mainnet 逐 bar 对比——205 个币种×月格子 199 red（97%）；信号 Jaccard 仅 0.34-0.55（一半"研究过的信号"在真实市场不存在）；demo ATR 系统性 +36%（BTC 中位）；volume 失真 6.6-20×；2025-07→2026-03 持续脱锚（2026 年 Jaccard 仅 0.17-0.31，越近越假）。
7. **裁决（2026-06-11）**：mainnet 基线重验 → FLAT 无 edge（毛利≈0）→ **项目关闭**。从第一个异常到关闭：**2 天**。真金从未入场，资金损失为零（不计 13 个月时间）。
8. **结论**：研究纪律没能保证结论正确（每一步 OOS/walk-forward 都按规矩做了，数据是假的照样全错）；**纪律的真正作用是保证错误结论活不到真金那天**——这次它做到了。铁律（外部源交叉验证、环境显式声明、server 字段、毛利基线门槛、gate 不可后改）每一条都对应链条上的一个失效点。

[出处：`reports/MR5M_postmortem.md` 全文 + `reports/regime/{v2b_dd_diagnosis,data_contamination_forensics,mainnet_rebuild,demo_vs_mainnet_comparison}_20260610/` + `mr5m_mainnet_baseline_20260611/`]

---

# 第 3 部分：当前状态快照（2026-06-11）

## 3a. VPS / 实盘状态

- **文档记载**：CLAUDE.md（2026-06-11 版）称 `run_mr_5m_direct.py` "已关停归档"。
- **本地可见证据**：本地无 systemd unit（`deploy/systemd/` 目录**为空**）；`restart_mr5m.sh` 指向 VPS 路径 `/run-project/vnpy_strategy_test/CTA`，说明实盘进程在另一台机器上由该路径管理。
- **结论**：**实盘进程是否真正已停止，本地无法确认，需在 VPS 上确认**（`ps`/`systemctl`/screen 会话 + 最后一条 trade_log 时间）。双账号（DYN/FLAT）模拟盘 2026-06-01 启动，若未手动停仍可能在跑。

## 3b. 本地环境健康度

| 项 | 状态 |
|---|---|
| venv | `.venv` Python 3.12.3，vnpy 4.3.0 / vnpy_ctastrategy 1.4.1 / vnpy_okx 2026.4.15 / vnpy_sqlite 1.1.3 / pandas 3.0.2 / numpy 2.4.4 |
| doctor.py | 本次实际运行：**PASS**（依赖、gateway import、运行时文件全部 OK） |
| Makefile | 1137 行、100+ target，覆盖环境/下载/验证/回测/研究/审计全流程；**绝大多数研究类 target 指向旧库时代脚本，直接跑会读污染库**（历史档案价值 > 直接复用价值） |
| tests/ | 49 个集成测试文件（本次未运行——只读盘点不跑回测类负载） |
| git | 远程 `github.com:yifanfengsu/CTA.git`；本地 main 与 origin/main 同步；当前分支 `research/atr-filter-oos` 与 main 同指针 |

## 3c. 悬而未决事项清单（负债，按风险排序）

1. **`.env` 的 `OKX_SERVER` 仍是 `DEMO`**（本次核实）。事故根因配置原封未动。新铁律靠"脚本不读 .env"防御，但任何旧脚本/gateway 路径被无意识跑起来，污染通道立即复活。`download_history.py` / `merge_recent.py` 的硬编码 demo header 也**未修复**（取证报告 Q4 明确"本次未动"）。
2. **mainnet 库未做外部源交叉验证**（详见 1a）。检查清单第 1 项对当前主数据库本身尚未闭环——这是新研究启动门槛的第一道未完成项。
3. **大量关键资产未纳入版本控制**：`git status` 显示 19 个未跟踪文件，包括 **CLAUDE.md、PROJECT_GUIDE.md、主力回测引擎 `backtest_mr_5m_compare.py`、mainnet 下载器、全部事故链研究脚本**；且 `.gitignore` 整体排除 `reports/*`（70+ 报告目录、两份 postmortem 均不在 git 里）和 `*.db`（1.41 GiB mainnet 库无任何备份）。**当前项目的核心知识资产只存在于这台 WSL 的工作区里，一次误操作或磁盘故障即可灭失。**
4. **实盘脚本的工作区未提交修改**：`run_mr_5m_direct.py` 有 +6/−1 的未提交 diff——内容正是 IOC 关键修复（`body["ordType"] = "ioc"`）。即 git HEAD 上的版本**不含**这个实战 bug fix；若 VPS 部署的是工作区拷贝，则"代码真相"只存在于未提交状态。
5. **vnpy 全局设置仍指向污染库**：doctor 输出显示 vn.py `database.database: database.db`——任何通过 vnpy 引擎路径（CTA 回测 GUI、`download_history` make target 等）的读写都会默认落在旧库。
6. **回测引擎默认参数指向污染库**：`backtest_mr_5m_compare.py` 的 `--database-path` 默认 `database.db`（用对数据全靠调用方记得注入）；其 `--report` 路径还会再生成 demo 时代的结论文案。
7. **资金费率数据环境未正式取证**（初判干净，见 1a），且缺 2026-04 之后。
8. **CLAUDE.md 的"关键研究发现"遗留**：mainnet 基线报告记载，CLAUDE.md 中 demo 数字的修订被有意推迟"等用户对实盘何去何从做决策后一并修订"——该决策与修订动作截至盘点时点未见记录闭环（现行 CLAUDE.md 已声明项目关闭，但此项以用户确认为准）。
9. **旧 demo 库的处置未决**：1.5 GB 旧库仍在原位（取证基准用途），与新库同目录仅一字之差（`database.db` vs `database_mainnet.db`），长期保留即长期误用风险。
10. 小项：`deploy/systemd/` 空目录与 PROJECT_GUIDE 描述（"systemd 部署配置"）不符；`task.txt` 是早期任务清单遗留（已过时）；策略配置 `.vntrader/cta_strategy_setting.json` 仍是停用的 MR-v1 参数。

---

# 第 4 部分：面向新研究的客观约束（只列事实，不做方向建议）

## 4a. 数据约束

**现有数据能支持**：5 币（BTC/ETH/SOL/LINK/DOGE）USDT 永续、1m 及以上任意聚合周期、2023-01 → 2026-05 共 3.4 年的价格类研究（趋势/MR/波动率/跨 5 币的小横截面信号）；叠加 8h 资金费率（到 2026-03）的成本修正或 carry 分析（环境核查补完后）。

**现有数据不能支持**：微观结构研究（无 tick/订单簿）；大横截面研究（仅 5 币）；衍生品确认信号（无 OI/taker ratio/long-short）；期现/跨所价差（无现货、无他所数据）；成交量信号（mainnet volume 未验证；如要用需先核查）；2026-04 之后的 funding；以及**严格按铁律，任何研究在 mainnet 库完成外部源交叉验证之前都不应启动**。

## 4b. 成本结构对策略毛利的隐含要求

OKX 永续费率（本项目实际口径）：**maker −0.002%（返佣）/ taker 0.05%**；资金费率 8h 一次（典型 ±0.01%，方向不定）。

mainnet 基线给出的实际量纲 [事实：`mr5m_mainnet_baseline_20260611/`]：
- C2 配置 63,404 笔，总费用 $76,088 → **约 $1.20/笔**，对应每笔 ~$2,500 实际名义的 **~0.048%/回合**（maker 进 + taker 出的乐观口径）。
- MR-5m 的毛利只有 +$9,194 / 63,404 笔 ≈ **$0.145/笔（~0.006% of notional）**——毛利只有成本的 1/8，这就是"无 edge"在成本维度的样子。
- 隐含门槛：在"maker 进 + taker 出"口径下，策略**平均每回合毛利必须显著超过 ~0.05% 名义**才可能净正；若两边都 taker（趋势类追价入场常见），门槛升到 **~0.10%+**；持仓跨多个 8h 结算的低频策略还要叠加资金费率敞口（fade 类常逆费率方向，趋势类视方向而定）。
- 复盘检查清单将其制度化为：**裸基线毛利≈0 或为负的信号直接放弃**——费用优化、过滤、仓位都救不了（mainnet 基线对此给过实证：ATR filter 只能把净亏从 −$104k 降到 −$67k，方向不变）。
- 频率含义（事实陈述）：信号越高频，单笔毛利门槛相对固定成本越难达到——MR-5m 全期 9 万笔的体量把 0.048% 的成本放大成了 $110k 量级的确定性损耗。

## 4c. 工程约束：研究 → 模拟盘 → 实盘路径

| 环节 | 现成 | 要新建 |
|---|---|---|
| 数据获取 | mainnet 下载器（5 币 1m，可扩区间/币种）、funding 三件套、完整性校验 | 外部源交叉验证脚本（铁律要求但还没有）；新数据类型（tick/OI/横截面）的全套管线 |
| 回测 | 费用/成交/整数张数/指标的工具函数、metrics、jsonl 口径、walk-forward/敏感性/pivot 框架 | 策略逻辑层（现引擎只有 Donchian fade 状态机）；资金费率计入；非 5m 周期/多仓位形态的回测语义 |
| 研究纪律设施 | train/test 切点惯例、五项诚实性检验、误伤分析、gate 流程、报告模板、PROJECT_GUIDE 维护规范 | （无——这是最完整的一块） |
| 模拟盘 | 完整 runner 骨架：WebSocket、聚合、IOC 下单、对账、pos_verify、通知、日志轮转（策略信号部分需替换） | 新策略的信号模块；若多策略并行需调度层 |
| 实盘 | OKX bills 对账、force_close、双账号 A/B 实验设计经验 | VPS 部署现状需先确认（3a）；systemd 单元文件实际不在仓库 |

## 4d. 历史教训形成的硬门槛（全部已制度化为 CLAUDE.md 铁律 / 复盘 §8 清单）

1. **数据环境显式化**：写库脚本打印环境、manifest 带 `server`、禁止隐式继承 `.env`。
2. **外部源交叉验证**：新数据入库后、用于研究前，≥3 随机日与独立源逐 bar 对比。
3. **裸基线毛利门槛**：先跑无过滤固定仓位基线，毛利≈0 即放弃整个方向。
4. **gate 预先写死，结果出来后不可修改**（差 0.023 也不放宽）。
5. **否定性结论与肯定性结论同等证据标准、同等固化**。
6. **回测假设写进报告头部**（费率/滑点/成交）。
7. **统计显著性优先**，不基于直觉/短期实盘样本拍板；保守优于激进。
8. **不修改归档实盘脚本与基准引擎**；研究产出独立脚本 + 报告 + jsonl。
9. **历史"死胡同"清单不再是硬边界**（其证据基于污染数据），但重启任何方向都要按上述 1-7 重走完整流程。

---

# 盘点过程中发现的意外

（与现有文档记载不符、或此前无人注意的事项；与 3c 部分重叠者此处只列"意外"属性）

1. **核心知识资产几乎全部游离在版本控制之外**（意外程度最高）：`.gitignore` 排除整个 `reports/`，两份 postmortem、全部 70+ 研究报告、事故链档案均不在 git；CLAUDE.md、PROJECT_GUIDE.md、主力回测引擎、mainnet 下载器等 19 个文件未跟踪；mainnet 数据库无备份。文档反复强调"同等固化"的研究结论，物理上只有一份拷贝。
2. **实盘脚本的 IOC 关键修复只存在于未提交的工作区 diff**：git 历史有 7+ 次执行 bug 修复提交，但 2026-06-07 的 GTC→IOC 切换（复盘时间线明确记载的节点）在 HEAD 上不存在，仅以未提交修改形式存在——仓库历史与"实战验证过的代码"不一致。
3. **`.env` 的 `OKX_SERVER=DEMO` 原封未动**：事故复盘已完成、铁律已立，但根因配置本身仍在原位，且两个硬编码 demo header 的脚本未修。
4. **vnpy 全局设置与回测引擎默认参数双双仍指向污染库**：`database.name=database.db`（doctor 输出）+ `--database-path` 默认值。"唯一可信数据源"目前靠每个调用方自觉注入，没有任何默认值层面的防呆。
5. **回测引擎 `--report` 会再生成已作废的结论**：`generate_report()` 硬编码了 demo 时代的叙述（"保留 DOGE""2024 全市场逆风年"——后者已被 mainnet 重验明确证伪）。
6. **`deploy/systemd/` 是空目录**：PROJECT_GUIDE 称"systemd 部署配置"，实际部署配置不在仓库（推测在 VPS 本机）。
7. **资金费率数据走的是干净路径（初判）**——这是个**正面**意外：任务预设它"可能同样经过 DEMO 路径"，代码检视显示其用裸 urllib 请求硬编码公开 mainnet URL，无模拟盘头。但正式取证与外部验证仍未做（见 1a）。
8. **mainnet 库自身的外部交叉验证缺位**：铁律和检查清单都要求，可行性探测也找好了 Binance vision 源，但这一步在重建-对比-重验的两天冲刺中没有被执行——新研究启动门槛的第一项实际上是悬空的。
9. **BNB/XRP 的合约规格文件存在但无任何数据**：`config/instruments/` 有 7 币规格，数据只有 5 币——疑似曾计划扩币种的半成品。
10. **funding 数据与 K 线数据存在 2 个月的覆盖错位**（funding 到 2026-03-31，K 线到 2026-05-28），任何 funding-aware 回测在 2026-04 之后无法做。

---

项目盘点完成于 2026-06-11 12:16 UTC，纯只读 / 未改任何文件 / 未跑回测 / 未碰 VPS：[已确认]
