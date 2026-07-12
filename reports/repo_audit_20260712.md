# cta_strategy 仓库现状全面盘点

> **性质**：重构前的**只读盘点**。本报告为唯一新增文件，未移动/删除/重命名/修改任何其他文件。
> **生成于**：2026-07-12T09:58Z (UTC)
> **目的**：交给"设计 Claude"作为结构重构的决策依据。
> **最高约束**：本仓库有正在 VPS 跑的 B2_4h 前向系统（依赖内部相对路径）、一个污染库（绝不能碰）。见第 4 部分。

---

## 摘要（先读这段）

- 仓库 = 一条围绕**加密永续 CTA 研究**生长了 ~3 个月的单体研究仓库，git 远程 `git@github.com:yifanfengsu/CTA.git`。
- 磁盘 ~8.8G，其中数据/DB/产物占绝大部分：`.vntrader/` 2.9G（两个 1.5G 的 sqlite 库）、`data/` 2.5G（Binance Vision 交叉验证数据 2.4G）、`reports/` 1.6G、`.venv/` 1.4G、`.git/` 303M。
- **代码量不大**：`scripts/` 126 个 .py（4.7M），`strategies/` 4 个文件，`tests/` 40 个测试。真正的"活代码"很少，大部分是一次性研究脚本。
- **研究产物极多且已归档**：`reports/` 下 33 个子目录 + 10 个顶层文档，1821 个 git-tracked 文件。绝大多数是**已完成、不再改**的研究存档。
- **~12 条研究线全部收敛**（详见第 5 部分）：除 B2_4h 在做零成本前向观察外，其余全部 CLOSED/FAIL/NOT-VIABLE。**全部是加密标的**（OKX 永续 + Deribit 期权），没有任何传统市场（美股/期货）的容纳结构。
- **核心痛点**：① 加密永续假设焊死在目录/路径/命名里；② 活文档与一次性产物混杂；③ `scripts/` 平铺 126 个脚本无分层；④ 命名规律不统一（带日期/不带日期、中英混杂）；⑤ VRP 独立成 `vrp/` 而其余研究线散在 `reports/`——结构不对称。

---

## 第 1 部分：完整目录结构

### 1a. 目录树（2–3 层，大数据目录汇总标注）

```
cta_strategy/                          # git repo, remote = github.com/yifanfengsu/CTA
├── CLAUDE.md                          # AI 助手项目约束（铁律/工作原则/数据环境铁律）
├── PROJECT_GUIDE.md                   # 项目活文档（57KB，当前最佳认知 + 已验证/未验证事实两节）
├── README.md                          # 项目说明（93KB，1521 行，大量 Makefile 命令文档 + 研究线索引）
├── README_BOOTSTRAP.md               # 环境自举说明（147 行）
├── Makefile                          # 统一入口（46KB, 1137 行, 100+ target）
├── restart_mr5m.sh                    # ⚠️ MR-5m 实盘重启脚本（已归档策略，硬编码 VPS 路径 /run-project/…/CTA）
├── .env                              # 敏感凭证（gitignored；OKX_SERVER 等）—— 未打印内容
├── .env.example                      # 凭证模板（安全）
├── .gitignore                        # 忽略规则（DB/venv/raw data/forward 运行态/binance zip 等）
├── .codex                            # 空文件（0 字节，用途不明）
│
├── .agents/                          # 空目录（无内容）
├── .claude/                          # Claude Code 配置：settings.local.json（7KB, 权限白名单等）
├── .obsidian_vault/                  # 空目录（Obsidian 笔记库占位，无内容）
├── .codex / .venv/ / .git/           # 工具/环境目录（.venv 1.4G, .git 303M）
│
├── config/                           # 策略/运行配置（64K）
│   ├── runtime.json                  #   运行时配置
│   ├── strategy_default.json         #   默认策略参数
│   ├── strategy_mr_v1.json           #   MR-v1 参数
│   ├── strategy_sanity_min_size.json #   最小仓位 sanity 配置
│   └── instruments/                  #   7 个 OKX 永续合约规格 json（btc/eth/sol/link/doge/bnb/xrp _swap_okx.json）
│
├── strategies/                       # vnpy CTA 策略模板（240K，回测/实盘用）
│   ├── mr_5m_strategy.py             #   MR-5m 均值回归（核心历史策略）
│   ├── mr_v1_strategy.py             #   MR-v1
│   ├── okx_adaptive_mhf_strategy.py  #   自适应 MHF 策略（早期）
│   └── __init__.py
│
├── scripts/                          # ⚠️ 126 个 .py 平铺（4.7M）—— 研究/回测/审计/分析/下载/运维全混在一层
│   ├── forward_b2_4h.py              #   ⚠️ 前向系统主脚本（VPS 正在跑，见第 4 部分）
│   ├── research_trend_baseline.py    #   ⚠️ 前向系统依赖（引擎，被 forward 逐字 import）
│   ├── research_trend_validation_r2.py #  ⚠️ 前向系统依赖（引擎，被 forward 逐字 import）
│   ├── run_mr_5m_direct.py           #   MR-5m 实盘 runner（已关停归档，engineering 参考）
│   ├── backtest_mr_5m_compare.py     #   基准回测引擎（保真复刻实盘，费用/成交口径 1:1）
│   ├── research_*.py (~40 个)        #   各研究线脚本
│   ├── audit_*.py / analyze_*.py     #   审计/分析
│   ├── postmortem_*.py (4 个)        #   事后归因（单文件 60–90KB）
│   ├── download_*.py / verify_*.py   #   数据工程
│   ├── run_*.py / compare_*.py       #   实验运行/对比
│   ├── common_runtime.py, history_utils.py, history_time_utils.py, doctor.py, init_db.py  # 少数共享工具
│   └── README_DATA_SOURCE.md
│
├── tests/                            # 40 个单测（1.4M），命名 test_<对应脚本>.py，主要覆盖 research/postmortem/audit 脚本
│
├── forward/                          # ⚠️ B2_4h 前向观察系统（正在 VPS 跑）—— 见第 4a
│   ├── config_frozen.json            #   ⚠️ 冻结配置（SHA256 001b0d9e…；改动=新系统）
│   ├── baseline_distribution.json    #   ⚠️ gate 阈值基线
│   ├── gates_preregistered.md        #   预注册 gate（K1/K2/K3 KILL, U1/U2 UPGRADE）
│   ├── VPS_DEPLOYMENT_MANUAL.md       #   VPS 手动部署手册（用户执行，Claude 不碰 VPS）
│   ├── README.md / REVIEW_template.md / dry_run_validation.json / notify.conf.template / last_run_log.txt
│   └── data/                         #   本地为空；运行态 store 在 VPS（gitignored）
│                                     #   （运行时另有 forward/state/ + forward/notify.conf，均 gitignored）
│
├── data/                            # 数据资产（2.5G）—— 见 1b / 第 2e
│   ├── binance_vision/              #   2.4G, 11354 文件：~105 币种子目录（BTCUSDT…）+ funding/ + manifest + README
│   │                                #   （逐个 zip gitignored；交叉验证/因子研究数据源）
│   ├── funding/                     #   8.7M, 400 文件：okx/ + okx_historical_raw/（资金费率 CSV，gitignored）
│   ├── basis/                       #   304K：futures/ + index/（期现套利原始数据，csv gitignored, manifest tracked）
│   ├── history_manifests/           #   1.5M, 28 文件：OKX 1m K线下载清单 json
│   └── raw/                         #   85M, 4 文件：OKX 原始 K线 CSV（gitignored）
│
├── reports/                         # 研究产物（1.6G, 1821 tracked 文件）—— 见 1b / 第 2d
│   ├── (33 个研究子目录)             #   见下方清单
│   └── (10 个顶层文档)               #   见下方清单
│
├── vrp/                             # ⚠️ 独立的 VRP（期权波动率风险溢价）研究线（4.1M）—— 结构不对称，见痛点 3b
│   ├── scripts/                     #   research_atm_vrp_stageA.py / _stageB.py
│   ├── reports/                     #   stageA_data / stageB_premium_truth（README + PREREGISTRATION）
│   └── data/                        #   cycles jsonl + summary + manifest + cache/（Deribit tradingview 缓存 ~70 json）
│
├── deploy/                          # 部署目录（12K）
│   └── systemd/.gitkeep             #   空（systemd 配置在 VPS 本机维护，仓库内仅占位）
│
├── logs/                            # 脚本运行日志（4.4M；除 .gitkeep 外全 gitignored）
│
└── .vntrader/                       # ⚠️ vnpy 引擎数据（2.9G）—— 见 1b / 第 4b
    ├── database_mainnet.db          #   ⚠️ 唯一可信回测数据源（1.5G, mainnet 干净数据）+ -shm/-wal
    ├── database_DEMO_CONTAMINATED.db #  ⚠️⚠️ 污染库（1.5G, DEMO 行情，绝不能用于任何研究）
    ├── cta_strategy_data.json / cta_strategy_setting.json / vt_setting.json / log/
```

**顶层计数**：14 个内容目录（另 `.git` `.venv` 2 个工具目录）+ 10 个顶层文件。

### 1b. 顶层每项一句话职责

| 项 | 类型 | 职责 |
|----|------|------|
| `CLAUDE.md` | 活文档/规则 | AI 助手的项目约束：数据环境铁律、工作原则、铁律 A/B/C、PROJECT_GUIDE 维护流程 |
| `PROJECT_GUIDE.md` | 活文档 | 项目活文档，反映当前最佳认知；含"已验证核心事实" + "已知未验证假设"两节 |
| `README.md` | 半活文档 | 项目说明 + Makefile 命令全文档 + 早期研究线索引（93KB，多数为历史命令说明） |
| `README_BOOTSTRAP.md` | 活文档 | 新环境自举步骤 |
| `Makefile` | 配置/入口 | 100+ target 的统一命令入口（1137 行）—— 高度加密/OKX 特定 |
| `restart_mr5m.sh` | ⚠️ 遗留脚本 | MR-5m 实盘重启（策略已归档），硬编码 VPS 路径，散在根目录 |
| `.env` / `.env.example` | 配置 | OKX 凭证（.env gitignored，未打印）；example 为模板 |
| `.gitignore` | 配置 | 忽略 DB/venv/raw/forward 运行态/binance zip 等 |
| `.codex` | 未知 | 0 字节空文件，用途不明（可能是早期工具占位） |
| `.agents/` | 空 | 空目录，无内容 |
| `.claude/` | 配置 | Claude Code `settings.local.json`（权限白名单等） |
| `.obsidian_vault/` | 空 | Obsidian 笔记库占位，空 |
| `config/` | 配置 | 策略参数 + OKX 合约规格（`instruments/`）+ 运行时配置 |
| `strategies/` | 代码 | 4 个 vnpy CTA 策略模板 |
| `scripts/` | 代码 | 126 个 .py：研究/回测/审计/分析/下载/运维/实盘 runner **全平铺一层** |
| `tests/` | 代码 | 40 个单测，映射到研究/审计脚本 |
| `forward/` | ⚠️ 运行中系统 | B2_4h 零成本前向观察配置与文档（VPS 正在跑） |
| `data/` | 数据 | 行情/资金费率/期现/清单原始数据（大数据 gitignored） |
| `reports/` | 研究产物 | 全部研究输出（markdown + json + jsonl + csv） |
| `vrp/` | 研究线 | 独立的期权 VRP 研究线（自带 scripts/reports/data） |
| `deploy/` | 部署 | 仅占位（systemd 在 VPS 本机维护） |
| `logs/` | 运行态 | 脚本日志（gitignored） |
| `.vntrader/` | ⚠️ 数据/DB | vnpy 数据库（mainnet 干净库 + DEMO 污染库） |

---

## 第 2 部分：分类 —— 活文档/配置/代码 vs 历史产物

### 2a. 活文档与规则（长期维护，会持续引用）

| 文件 | 作用 | CLAUDE.md 内的核心铁律（摘） |
|------|------|------|
| `CLAUDE.md` | 项目最高约束 | **数据环境铁律**（唯一可信源 = `database_mainnet.db`；污染库严禁用）；**工作原则**（统计显著性优先、数据不支持就诚实说不、不过拟合、gate 阈值跑前写死不可事后改）；**铁律 A** gate 事后改门根治；**铁律 B** 多重检验打折 + 立项算术前置（deflated Sharpe）；**铁律 C** 正偏策略不用 Sharpe 做主要生死判据；**PROJECT_GUIDE 外科手术式维护流程** |
| `PROJECT_GUIDE.md` | 活文档 | 结构：一句话定位 → 新研究篇章 → 目录结构 → 核心策略 MR-5m → 研究历程 → **已验证核心事实** → **已知未验证假设** → 研究产出索引 → 脚本分类 → Makefile 体系 → 数据库配置 → 约束 |
| `README.md` | 半活/半历史 | 项目简介 + 大量早期研究模块说明 + Makefile 命令全文档；**多数内容是历史命令说明，偏"流水账"** |
| `README_BOOTSTRAP.md` | 活文档 | 环境自举 |
| `forward/README.md` + `gates_preregistered.md` + `VPS_DEPLOYMENT_MANUAL.md` | 活文档（运行中系统） | 前向系统定位声明、预注册 gate、部署手册 |
| `.claude/settings.local.json` | 配置 | Claude Code 权限/设置 |

### 2b. 配置

- **`config/`**：`runtime.json`、3 个 `strategy_*.json`、`instruments/` 下 7 个 OKX 永续合约规格。**全部 OKX 永续特定**。
- **`forward/config_frozen.json`** ⚠️：B2_4h 冻结配置，SHA256 `001b0d9e…`。**绝不能动**——改动即新系统、gate 作废。
- **`.env` / `.env.example`**：OKX 凭证。`.env` gitignored 未打印。
- **`.vntrader/`** ⚠️：
  - `database_mainnet.db`（1.5G）= **唯一可信回测源**，只读。
  - `database_DEMO_CONTAMINATED.db`（1.5G）= **已确认污染的 DEMO 行情，严禁用于任何回测/研究，仅作取证基准**。
  - `cta_strategy_*.json` / `vt_setting.json` = vnpy 引擎设置。

### 2c. 代码

- **`scripts/` 126 个 .py**，可分（当前**无目录分层，全平铺**）：
  - **实盘 runner**（VPS 专用）：`run_mr_5m_direct.py`（已归档）、`run_mr_v1_direct.py`、`run_*_demo.py`
  - **回测引擎/脚本**：`backtest_mr_5m_compare.py`（**基准引擎**）、`backtest_*.py` 系列
  - **研究脚本**：~40 个 `research_*.py`（每条研究线一个/多个）
  - **事后归因**：4 个 `postmortem_*.py`（单文件 60–90KB）
  - **审计/分析**：`audit_*.py` / `analyze_*.py` / `diagnose_*.py` / `compare_*.py`
  - **数据工程**：`download_*.py` / `verify_*.py` / `import_*.py` / `refresh_*.py` / `probe_*.py`
  - **共享工具/库**：`common_runtime.py`、`history_utils.py`、`history_time_utils.py`、`doctor.py`、`init_db.py`
- **共享引擎/复用点**（重构必须保持可 import）：
  - `backtest_mr_5m_compare.py` —— 基准回测引擎，多处复用其数据加载与指标。
  - `research_trend_baseline.py` + `research_trend_validation_r2.py` —— ⚠️ **被 `forward_b2_4h.py` 逐字 import**（前向系统引擎）。
  - `common_runtime.py` / `history_*_utils.py` —— 数据加载/时间工具。
- **`strategies/`**：4 个 vnpy 策略模板。
- **`tests/`**：40 个单测，映射到研究/审计脚本（覆盖偏研究脚本，非策略本体）。
- **`deploy/`**：空占位。

### 2d. 历史研究产物（一次性、已完成）

**`reports/` 33 个子目录**（按命名可见的规律）：

*带日期后缀（新研究篇章，2026-06 起，clean mainnet 数据）：*
```
b2_4h_pnl_audit_20260628      b2_4h_vol_targeting_20260628
basis_arbitrage_feasibility_20260615   breakout_pullback_15m_20260613
cross_sectional_carry_holding_20260613 cross_sectional_ic_20260613
factor_scale_feasibility_20260628      flow_vs_price_trend_20260628
forward_provenance_check_20260622      forward_restart_20260622
funding_structure_20260612             mr_timescale_structure_20260612
order_flow_exhaustion_feasibility_20260628  pairs_cointegration_20260613
trend_adx_filter_20260616     trend_baseline_20260611    trend_dualcycle_20260611
trend_faster_entry_20260616   trend_funding_confirm_20260613
trend_methodology_hardening_20260622   trend_portfolio_20260611
trend_validation_20260611     trend_validation_r2_20260611   volatility_event_20260613
```
*不带日期（早期/大宗产物目录，多为 demo 时代或聚合）：*
```
ablation/ (309M, 606f)   alpha_sweep/ (138M, 175f)   archive/ (task.txt)
backtest/ (35M)   backtest_compare/ (38M)   history_verify/ (25f)
mr_v1_demo_prep/   regime/ (3.9M, 17 子项)   research/ (1.0G, 47 子目录)
```

*`reports/research/` 下 47 个子目录*（早期趋势/MR/跨币种研究，demo 时代占多数）：`trend_following_v2/v3`、`early_trend_classifier_v1(_inverse)`、`csrb_v1(_postmortem)`、`vsvcb`、`external_regime_classifier`、`mr_v1*`、`htf_signals`、`trace_*` 等。

*`reports/regime/` 17 子项*：数据事故取证链（`data_contamination_forensics_20260610`、`demo_vs_mainnet_comparison_20260610`、`mainnet_rebuild_20260610`、`data_trust_closure_20260611`、`vulnerability_patch_20260611`）+ MR 动态仓位/ATR filter OOS 等。

**`reports/` 10 个顶层文档**（跨研究线的关键结论/复盘，偏活/半活）：
```
CTA_strategy_failure_postmortem.md    MR5M_postmortem.md
PROJECT_FINAL_SUMMARY_20260614.md     perpetual_signal_space_closure_20260613.md
trend_line_closure_20260612.md        project_inventory_20260611.md
research_catalog_trend_mr_20260622.md vps_shutdown_checklist.md
history_verify_latest.json            (+ .gitkeep)
```

**性质判断**：
- **已完成存档（不再改）**：`reports/research/*`、`ablation/`、`alpha_sweep/`、`backtest*/`、`regime/` 下取证链、以及所有带日期的研究子目录。这些是**知识资产**，2026-06-11 起纳入 git（`.gitignore` 注释明确说明）。
- **可能仍活跃引用**：`reports/` 顶层的 postmortem / closure / final_summary / catalog / vps_shutdown_checklist —— 会被后续研究和记忆引用，属"半活"。

### 2e. 数据

| 目录 | 组织方式 | 类型 | git |
|------|---------|------|-----|
| `data/binance_vision/` | **按币种**（~105 个 `<SYM>USDT/`）+ `funding/` | 交叉验证/因子研究原始行情（klines + fundingRate zip） | zip gitignored；README/manifest tracked |
| `data/funding/` | 按交易所（`okx/`、`okx_historical_raw/`） | 资金费率 CSV | csv gitignored |
| `data/basis/` | `futures/` + `index/` | 期现套利原始数据 | csv gitignored；manifest tracked |
| `data/history_manifests/` | 按 symbol-timeframe-daterange | OKX 1m 下载清单 json | tracked（小） |
| `data/raw/` | 按 symbol-timeframe-window | OKX 原始 K线 CSV | gitignored（85M） |

- **原始行情**：`binance_vision`、`raw`、`basis`。**中间产物/分析就绪序列**大多落在 `reports/<line>/` 内。
- **大文件全部 gitignored**（zip/csv/db/raw）；**小 manifest/README 入 git**（保证可复现）。**组织方式全部按加密交易所/币种**。

---

## 第 3 部分：当前结构的痛点（重构的靶子，不粉饰）

### 3a. 命名不一致
- **带日期 vs 不带日期混用**：新研究线目录带 `_YYYYMMDD`（`flow_vs_price_trend_20260628`），但早期大宗目录不带（`ablation/`、`alpha_sweep/`、`research/`）。同一个 `reports/` 下两套命名规律。
- **中英混杂**：文档正文大量中文，目录/文件名英文；`CLAUDE.md`/`PROJECT_GUIDE.md` 中文，`README.md` 中英混排。
- **策略代号不自解释**：`B2_4h`、`D2`、`csrb_v1`、`vsvcb_v1`、`mhf` 等代号散落，无集中图例（需查记忆/报告才知含义）。
- **顶层脚本命名前缀多样**：`research_` / `backtest_` / `audit_` / `analyze_` / `postmortem_` / `compare_` / `diagnose_` / `download_` / `verify_` / `run_` —— 前缀即隐含分类，却没被目录化。

### 3b. 职责混淆 / 同类散多处
- **`scripts/` 126 个 .py 全平铺一层**：实盘 runner、基准引擎、研究脚本、数据下载、运维工具、共享库全混在一起，无法一眼区分"活代码/引擎" vs "一次性研究脚本"。
- **VRP 独立成 `vrp/`（自带 scripts/reports/data），其余 ~11 条研究线却散在 `reports/` + `scripts/`** —— **结构严重不对称**。要么所有研究线都像 VRP 一样自成模块，要么都摊平，现在是两种范式并存。
- **`reports/research/` 是一个"目录中的目录"黑洞**（1.0G, 47 子目录），把早期趋势/MR/跨币种研究全塞进去，与外层带日期的研究子目录并列——同类研究产物散在两个层级。
- **前向系统的引擎依赖藏在 `scripts/` 里**：`forward_b2_4h.py` 依赖 `scripts/research_trend_baseline.py` + `research_trend_validation_r2.py`，但这三者与其他 120+ 研究脚本平铺混放，"运行中系统的关键代码"没有任何视觉/目录隔离。

### 3c. 加密特定 vs 通用（对未来最致命）
- **`config/instruments/` 全是 `*_swap_okx.json`**（OKX 永续合约规格）——目录名/文件名焊死永续。
- **`data/` 按加密交易所组织**（`binance_vision/`、`funding/okx/`、`basis/`）——没有"市场/资产类别"这一层，美股/期货数据无处安放。
- **`Makefile`（1137 行）+ `README.md`（1521 行）高度 OKX/永续特定**：download-okx-history、funding、instrument-metadata 等 target 假设永续。
- **`.vntrader/`** 是 vnpy 加密引擎的固定目录名；`.env` 只有 OKX 凭证字段。
- **回测引擎口径写死永续**：`backtest_mr_5m_compare.py` 的费用/成交/funding/ctVal 口径是 OKX 永续专用，传统市场（无 funding、有隔夜、不同费率/tick）无法直接复用。
- 一句话：**当前几乎每一层都假设"加密永续 + OKX/Binance"**，没有任何"多市场"的抽象层。

### 3d. 活文档 vs 产物混杂
- `reports/` 顶层同时放**一次性研究子目录**和**半活的复盘/closure/catalog 文档**（`MR5M_postmortem.md`、`trend_line_closure_20260612.md`、`PROJECT_FINAL_SUMMARY_20260614.md`）——长期要引用的结论和一次性产物同层。
- `README.md` 把**活的自举/命令说明**和**历史研究模块流水账**混在一个 93KB 文件里，难以维护（真正的活认知在 `PROJECT_GUIDE.md`，`README.md` 大量内容已过时）。

### 3e. 根目录杂乱
- **`restart_mr5m.sh` 散在根目录**：单个 shell 脚本，属已归档策略，硬编码 VPS 路径，应归入 `scripts/` 或 `deploy/` 或 `archive/`。
- **`.codex`（0 字节空文件）、`.agents/`（空）、`.obsidian_vault/`（空）** 三个空占位散在顶层，无说明。
- 顶层同时有 `README.md` + `README_BOOTSTRAP.md` + `PROJECT_GUIDE.md` + `CLAUDE.md` 四个说明类文件，职责边界对新读者不清晰。

### 3f. 其他观察
- **`reports/` 1.6G 且 1821 文件已入 git**（含 `ablation/` 606 文件、`alpha_sweep/` 175 文件的大宗 jsonl/csv）——git 仓库被研究产物撑大（`.git/` 已 303M），clone 慢；VPS 部署要 clone 整个含 1.6G 产物的仓库才能跑一个前向脚本。
- **`tests/` 覆盖偏研究脚本而非策略本体**——重构若移动 `scripts/`，40 个测试的 import 路径会同步失效。
- **两个 1.5G 的 DB 并存于 `.vntrader/`**（一个可信、一个污染），仅靠文件名区分，物理同目录，有误用风险（虽已改名防呆）。

---

## 第 4 部分：不可动的约束 ⚠️（重构的硬边界）

### 4a. ⚠️ 前向系统（正在 VPS 跑，B2_4h 零成本前向观察）

`scripts/forward_b2_4h.py` 用 `ROOT = Path(__file__).resolve().parents[1]` 推导路径，即**依赖脚本相对仓库根的位置**。VPS 从 `git@github.com:yifanfengsu/CTA.git` clone 到 `/opt/forward_b2` 运行。**以下相对路径关系一旦破坏，前向系统失效**：

| 依赖 | 路径 | 说明 |
|------|------|------|
| 主脚本 | `scripts/forward_b2_4h.py` | cron 直接调用；移动即断 |
| 引擎依赖① | `scripts/research_trend_baseline.py` | 被逐字 import（`sys.path.insert(parent=scripts/)`） |
| 引擎依赖② | `scripts/research_trend_validation_r2.py` | 被逐字 import |
| 冻结配置 | `forward/config_frozen.json` | `FWD = ROOT/forward`；SHA256 校验，改路径/内容都触发 selfcheck 失败 |
| 基线 | `forward/baseline_distribution.json` | gate 数字 |
| 运行态 store | `forward/data/`（gitignored，VPS 本地） | append-only 1m+funding |
| 运行态 state | `forward/state/`（gitignored，VPS 本地）：`deploy.json`、`ledger_trades.jsonl`、`ledger_daily_m2m.jsonl`、`gap_log.jsonl`、`positions`、`heartbeat` | **绝不能删/动**——含前向账本与部署锚点 |
| seed 源（只读一次） | `.vntrader/database_mainnet.db` + `data/funding/okx/*.csv` | seed 时只读；funding CSV 随 git clone 到位 |
| 推送配置 | `forward/notify.conf`（gitignored，VPS 本地） | PushPlus token |

**VPS cron（用户手动装，Claude 不碰）**：
```
5 0,4,8,12,16,20 * * *  …/scripts/forward_b2_4h.py --cron-4h
20 0 * * *              …/scripts/forward_b2_4h.py --cron-daily
0 1 1 * *               …/scripts/forward_b2_4h.py --reconcile
```
- **三角色分离铁律**：开发（Claude 本地）/ 部署（用户手动 VPS）/ 运行（VPS cron 自主）。**Claude 绝不 SSH/操作 VPS**。
- **重构含义**：任何移动 `scripts/forward_b2_4h.py`、两个引擎脚本、`forward/` 内容、或它们的相对关系，都需**同步更新 VPS 部署手册并由用户重新部署**。config_frozen.json **内容一字不能改**（改路径也会让 selfcheck 的 SHA256 与内容比对逻辑需要重新固化）。

### 4b. ⚠️⚠️ 污染库（永久隔离，绝不能碰）
- **`.vntrader/database_DEMO_CONTAMINATED.db`（1.5G）** = 已确认污染的 OKX DEMO 行情。**严禁用于任何回测/研究**，仅作取证/对比基准。**不移动、不删除、不重命名、不读入任何研究流程**。
- 同目录的 **`.vntrader/database_mainnet.db`（1.5G）= 唯一可信源**，只读。备份在 `~/backups/database_mainnet_20260611.db.gz`（SHA256 `a6d6928d…495d`）。**两库都是 gitignored**（`.gitignore: *.db`），不在 git 内，重构移动会脱离前向 seed 依赖（4a）。

### 4c. ⚠️ B2_4h 冻结资产
- **策略实现**：B2_4h 不是独立策略文件，而是 `research_trend_baseline.py`(tb) + `research_trend_validation_r2.py`(r2) 的组合，被 `forward_b2_4h.py` 逐字复用。**这两个脚本对前向系统等同"已冻结引擎"，不能改**。
- **冻结数字**：$68,194.82 (OKX) / $300,752.78 (Binance) —— 已审计 PASS，写死在报告与 `forward/baseline_distribution.json` 的 gate 反推里。
- **config_frozen.json** 全参数永久冻结（EMA20/100、5 币、sizing、costs、forward_start_utc）。

### 4d. VPS/部署依赖
- **`restart_mr5m.sh`** 硬编码 `/run-project/vnpy_strategy_test/CTA`（MR-5m 旧实盘 VPS 路径，策略已关停）——移动它不破坏当前前向系统，但它反映"曾有另一套 VPS 路径约定"。
- **`deploy/systemd/`** 为空占位（真实 systemd 配置在 VPS 本机维护，仓库内无）。
- **git 远程名为 `CTA`**（`github.com/yifanfengsu/CTA`）——VPS clone 依赖此远程；重构不应改远程 URL，否则用户需重配。

### 4e. 其他"移动会破坏运行中东西"的约束
- `tests/` 40 个测试 import `scripts/` 脚本——移动 scripts 会连锁破坏测试（非运行中系统，但会红）。
- `Makefile` 100+ target 硬编码 `scripts/xxx.py` 路径——移动脚本会破坏所有 `make` 命令。
- `.gitignore` 精确列举 `forward/data/`、`forward/state/`、`forward/notify.conf`、`data/*` 等路径——重构须同步。

---

## 第 5 部分：面向未来（多市场系统化研究）

### 5a. 哪些结构最不适应"加入美股/期货"的未来

按"改造成本 × 阻碍程度"排序：

1. **`data/` 缺"市场/资产类别"层**（最致命）：现在直接 `data/binance_vision/`、`data/funding/okx/`，美股日线、期货连续合约、期权链无处安放。未来需要 `data/<market>/<venue>/<symbol>/…` 或 `data/crypto/…` + `data/us_equity/…` + `data/futures/…` 的顶层分市场结构。
2. **回测引擎焊死永续口径**：`backtest_mr_5m_compare.py` 假设 funding/ctVal/taker-maker/24h 连续交易。美股有隔夜跳空、交易时段、分红、不同费率；期货有到期/展期/保证金。需要**市场无关的引擎核心 + 每市场的成本/日历/成交适配层**。
3. **`config/instruments/` 焊死 OKX 永续**：需要泛化为按市场的合约/标的规格（tick、乘数、交易时段、结算方式）。
4. **`Makefile` + `README.md` 高度 OKX 特定**：download-okx-*、funding target 无法容纳新市场；命令入口需按市场重组或换成子命令式 CLI。
5. **`scripts/` 平铺无法容纳多市场研究线**：126 个已难找，再叠加多市场会失控。需按"市场 / 研究线 / 阶段"分层，或每条研究线自成模块（见 5b）。
6. **`.vntrader/`（vnpy 专属目录名）**：vnpy 偏加密/期货 CTA；美股研究可能不走 vnpy，这个目录名会误导。

### 5b. 现有研究线的组织方式，能否扩展到"多市场多研究线"

**当前 ~12 条研究线（全部加密，全部收敛）**：

| # | 研究线 | 载体/市场 | 状态 | 产物位置 |
|---|--------|-----------|------|---------|
| 1 | MR-5m 均值回归 | 加密永续 | CLOSED（mainnet 无 edge） | `reports/regime/`, `reports/research/mr_*` |
| 2 | MR 时间尺度（15m–4h） | 加密永续 | CLOSED | `reports/mr_timescale_structure_20260612` |
| 3 | Funding 结构反转 | 加密永续 | CLOSED | `reports/funding_structure_20260612` |
| 4 | 趋势跟踪 B2_4h | 加密永续 | 资源性关闭 + **前向观察中** | `reports/trend_*`, `forward/` |
| 5 | Breakout pullback 15m | 加密永续 | CLOSED | `reports/breakout_pullback_15m_20260613` |
| 6 | Pairs 协整 | 加密永续 | CLOSED | `reports/pairs_cointegration_20260613` |
| 7 | 横截面 IC / carry | 加密永续 | CLOSED | `reports/cross_sectional_*` |
| 8 | 波动率事件 | 加密永续 | CLOSED（需期权载体） | `reports/volatility_event_20260613` |
| 9 | Order-flow exhaustion | 加密（tick） | CLOSED | `reports/order_flow_exhaustion_feasibility_20260628` |
| 10 | Factor scale（大规模因子） | 加密（100 币） | NOT VIABLE | `reports/factor_scale_feasibility_20260628` |
| 11 | Flow vs price trend | 加密 | FAIL | `reports/flow_vs_price_trend_20260628` |
| 12 | VRP ATM（期权） | Deribit 期权 | CLOSED | `vrp/`（**独立模块**） |

**组织方式诊断**：
- **两种不一致的范式并存**：VRP（#12）自成 `vrp/{scripts,reports,data}` 模块；其余 11 条摊在共享 `scripts/` + `reports/<line>/`。
- **共享 `scripts/` 范式不可扩展**：再加"美股动量""期货 carry""期权 XX"等新市场研究线，`scripts/` 会从 126 涨到几百，`reports/` 顶层会更乱。
- **VRP 的"研究线自成模块"范式更适合未来**：每条线 `<line>/{scripts,data,reports,config}` 自包含，跨市场天然隔离。但需要一个**共享 `core/`（引擎/数据加载/统计工具/gate 框架）** 供所有线复用，避免重复造轮子（当前 `backtest_mr_5m_compare.py`、`common_runtime.py`、统计工具就是隐性的 core，只是没独立出来）。
- **结论**：现有组织**不能**平滑扩展到多市场；未来结构大概率需要 `markets/<market>/` × `research_lines/<line>/`（模块化，仿 VRP）+ `core/`（市场无关的引擎与方法学）+ `data/<market>/` 的三支重组。同时把"活文档（CLAUDE/PROJECT_GUIDE/方法学铁律）"与"研究产物存档"彻底分层。

---

## 给设计 Claude 的重构约束速查（TL;DR）

- ⚠️ **不可动**：`scripts/forward_b2_4h.py` + `scripts/research_trend_baseline.py` + `scripts/research_trend_validation_r2.py` + 整个 `forward/` 的相对关系（VPS 正在跑，动了要重发部署手册 + 用户重部署）。
- ⚠️ **绝不能碰**：`.vntrader/database_DEMO_CONTAMINATED.db`（污染库）。
- ⚠️ **只读、勿脱依赖**：`.vntrader/database_mainnet.db`（前向 seed 源）+ `data/funding/okx/*.csv`。
- 🔒 **内容冻结**：`forward/config_frozen.json`（SHA256 001b0d9e…）、B2_4h 冻结数字与 gate。
- 🔗 **移动连锁失效**：`Makefile`（100+ target 硬编码 scripts 路径）、`tests/`（import scripts）、`.gitignore`（精确路径）。
- 🎯 **重构真正的靶子**：`data/` 加"市场层"、`scripts/` 分层或研究线模块化（仿 `vrp/`）、活文档与产物分离、`reports/research/` 黑洞拆解、根目录空占位/遗留脚本归位、回测引擎抽出市场无关 core。

---

仓库盘点完成于 2026-07-12T09:58Z (UTC) / 顶层项:[14 个内容目录 + 10 个文件（另 .git/.venv 2 个工具目录）] / 研究报告:[reports/ 33 子目录 + 10 顶层文档，另 vrp/ 独立] / 研究线:[12 条，全加密，仅 B2_4h 前向观察中，余全部 CLOSED/FAIL] / 活文档:[CLAUDE.md、PROJECT_GUIDE.md、README(_BOOTSTRAP).md、forward/README+gates+VPS手册、.claude/settings] / 不可动约束:[前向系统 scripts/forward_b2_4h.py+两引擎+forward/、污染库 database_DEMO_CONTAMINATED.db、mainnet seed 源、config_frozen SHA256、VPS cron/远程 CTA 已标注] / 痛点:[命名不一致(带/不带日期·中英混)、职责混淆(scripts 平铺126·VRP独立而余散置·reports/research 黑洞)、加密永续焊死(data/config/引擎/Makefile 无多市场层)、活文档与产物混杂、根目录空占位+遗留脚本已列] / 未动任何文件:[确认，仅新增本报告 reports/repo_audit_20260712.md]
