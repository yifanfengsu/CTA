# CTA

## 项目简介

这是一个面向 OKX `BTCUSDT` 永续合约的 vn.py headless CTA 项目，核心依赖是 `vn.py`、`vnpy_okx`、`vnpy_ctastrategy` 和 `vnpy_sqlite`。项目只面向 WSL、Linux、VPS 等命令行环境，不引入 GUI/Qt 流程。

当前优先目标是：历史数据下载、完整性验证、回测、alpha 诊断、保守参数 sweep、OKX 模拟盘前置检查。

重要限制：当前标准 vn.py `BacktestingEngine` 结果未自动计入 OKX perpetual funding fee。回测报告只包含价格盈亏、手续费、滑点等常规项，不能把它直接视为完整实盘收益。

当前仓库没有 `scripts/run_cta.py`。Makefile 只覆盖模拟盘前置检查，真正 demo runner 需要后续补 `scripts/run_cta.py` 后再加入运行命令。

## Trend Regime Diagnostics：趋势环境诊断

`make diagnose-trend-regimes` 是 Trend Following V3.0 失败后的研究诊断入口，不是策略、不是参数搜索、不是 demo/live runner。

诊断目标是判断 2023-2026 五品种样本里是否存在可观测趋势 regime，并检查 V3.0 失败是否主要因为没有过滤 `choppy` / `high_vol_choppy` 环境。脚本会从本地 vn.py sqlite 读取 1m bar，重采样到 closed-bar `4h` / `1d`，输出 regime 分布、趋势评分和 V3 extended trades 的 regime 归因。

只有当 `reports/research/trend_regime_diagnostics/v3_1_regime_recommendations.json` 明确支持 `proceed_to_v3_1_research=true` 时，才允许进入 research-only V3.1。即使允许 V3.1 research，也仍然必须保持 `strategy_development_allowed=false` 和 `demo_live_allowed=false`。

不允许把本诊断结果直接转成 Strategy V3、demo runner 或 live runner；funding fee 与收益集中度风险必须继续作为硬约束处理。

## Research Decision Dossier

`make research-dossier` 用于归档当前趋势跟踪研究结论和最终决策，不开发策略、不新增参数搜索、不进入 demo/live。

该 dossier 汇总原始 backtest、Signal Lab、HTF、Trend V2、Trend V3、Extended V3、Postmortem、Regime Diagnostics、数据准备、actual OKX funding、External Regime Classifier Gate Audit 和 Derivatives Data Readiness Audit 报告，用来明确哪些策略 family 已失败、为什么当前 Strategy V3 被阻断，以及下一阶段是否只允许维护研究工具、可选但不推荐地扩大品种，或暂停开发。

Funding-aware final gate 已纳入 dossier：当前五品种 universe 的 OKX Historical Market Data funding 已完整导入并通过 `verify-funding`，但 `funding_adjusted_stable_candidate_exists=false`、`can_enter_funding_aware_v3_1_research=false`。因此当前仍不允许 Strategy V3、V3.1、demo 或 live。

External Regime Classifier Gate Audit 已完成：旧 `stable_candidate_like` 口径不再作为通过依据，修正后的 strict gate 没有 stable candidate，`can_enter_research_only_v3_1_classifier_experiment=false`。当前五品种趋势跟踪 family 已最终封档，不能继续包装为 Strategy V3、V3.1、demo 或 live。

Derivatives Data Readiness Audit 已完成：funding 和 mark/index candle 可用，但 OI/taker/long-short 等关键历史衍生品特征未证明能覆盖 `2023-01-01` 到 `2026-03-31`，因此 `can_enter_derivatives_confirmed_trend_research=false`。Funding alone 和 mark/index basis proxy 不能视为 derivatives confirmation 已满足。

输出目录为 `reports/research/research_decision_dossier/`，核心决策必须保持 `final_current_trend_family_archived=true`、`final_current_research_archived=true`、`final_strategy_development_allowed=false`、`final_demo_live_allowed=false`、`strategy_development_allowed=false`、`demo_live_allowed=false`、`proceed_to_v3_1_research=false`、`can_enter_derivatives_confirmed_trend_research=false`。后续默认暂停策略开发，只维护数据与研究工具，除非先提出全新的研究前提和验收标准。

## VSVCB-v1：低波动率挤压后的成交量确认突破

`make research-vsvcb-v1` 是全新的 Research-Only Phase 1 研究入口，用于检验 Volatility Squeeze with Volume Confirmation Breakout 假设：普通突破如果同时发生在低 Bollinger Band Width 挤压之后，并伴随突破 K 线成交量显著放大，是否比普通突破更容易延续。

本阶段只做事件研究和固定持有 T 根 K 线基准回测，输出 A/B/C/D/E 消融对照、no-cost / cost-aware / funding-adjusted 结果、集中度、MFE/MAE、反向测试和 Phase 1 裁决。脚本读取本地 vn.py sqlite 1m OHLCV，并使用 `data/funding/okx` 下的 OKX funding CSV；不会连接交易接口，不会下单。

VSVCB-v1 不修改 `OkxAdaptiveMhfStrategy`，不新增 Strategy class，不新增 demo/live runner，也不允许把 Phase 1 结果直接标记为可交易。无论 Phase 1 是否通过，`strategy_development_allowed=false` 和 `demo_live_allowed=false` 都必须保持不变。

如果 Phase 1 不通过，必须进入 postmortem，不允许根据 OOS 调参重试，不允许执行参数高原、随机化测试或任何“救回”VSVCB-v1 的参数搜索。如果 Phase 1 通过，也只能进入 Phase 2 参数高原和随机化研究，仍不能进入正式策略开发、demo 或 live。

`make postmortem-vsvcb-v1` 只读取 `reports/research/vsvcb_v1/` 下已经完成的 Phase 1 输出，生成 `reports/research/vsvcb_v1_postmortem/`。Postmortem 用来确认失败不是数据缺失或明显实现错误，并解释 D 组失败、E 组反向为正、symbol / direction / timeframe / horizon / feature-bin 和 conflict filter 影响；它不是调参工具。

如果 E 组明显优于 D，只能标记 `possible_false_breakout_research_hypothesis=true`，不能把 E 组反向收益解释为趋势跟踪 edge，也不能把 Phase 1 失败结果标记为可交易。VSVCB-v1 不允许直接修改正式策略，不允许新增 Strategy class，不允许进入 demo/live。

## CSRB-v1：Crypto Session Range Breakout

`make research-csrb-v1` 是新的 Research-Only Phase 1 研究入口，用于检验 Crypto Session Range Breakout 假设：虽然 crypto 24/7 交易，但 Asia / Europe / US 时段切换仍可能带来流动性和参与者结构变化；当低活跃时段形成清晰区间后，高活跃时段开始的突破可能代表新的方向性订单流。

CSRB-v1 借鉴 opening range breakout / London Breakout / Dual Thrust 的 session range breakout 思路，但实现为适配 OKX crypto perpetual 的独立离线研究框架。第一阶段只做事件研究和固定持有基准回测，默认从本地 vn.py sqlite 读取 1m OHLCV，按 closed-bar 重采样到 `15m,30m,1h`，并使用 `data/funding/okx` 下的 actual OKX funding CSV 输出 no-cost / cost-aware / funding-adjusted 三层结果。

本研究输出 Asia range → Europe breakout、Europe range → US breakout、session-agnostic ordinary breakout、randomized session time control 和 reverse test。它必须保留 random breakout 对照组，不使用 EMA / MACD / ADX 作为第一版过滤器，不用未来收益、MFE 或 MAE 作为入场特征，不根据 OOS 结果调参。

CSRB-v1 不修改 `OkxAdaptiveMhfStrategy`，不新增 Strategy class，不新增 demo/live runner，不连接真实交易，也不写 API key。无论 Phase 1 是否通过，`strategy_development_allowed=false` 和 `demo_live_allowed=false` 都必须保持不变；若全部 gate 通过，也只允许进入 Phase 2 research，不允许直接标记为可交易。

## Derivatives-confirmed Trend Research

当前 V3 family 已在 Research Decision Dossier 中失败，且 actual OKX funding 纳入后仍不能进入 Strategy V3、V3.1、demo 或 live。VSVCB-v1 Phase 1 的正向趋势延续假设也已失败，不能通过反向测试或 OOS 调参把它包装成趋势跟踪策略。

新的研究前提是 `price trend + derivatives confirmation`：价格趋势本身不够，只有当趋势同时得到 open interest、taker flow、long/short ratio、basis/premium、funding 等衍生品参与度确认时，才可能继续研究其稳定性。

`make audit-derivatives-data` 只做 OKX public/no-key derivatives data readiness audit。它会对 Open Interest、Funding Rate、Mark Price、Index Price、Taker Buy/Sell Volume、Long/Short Account Ratio、Contracts Open Interest and Volume、Premium/Basis proxy 等数据源做小样本 endpoint probe，输出可用特征、不可用特征、分段下载计划和 research-only gate。

Derivatives Data Readiness Audit 已完成，最终 gate 为 `can_enter_derivatives_confirmed_trend_research=false`。原因是 current open interest 只能作为 snapshot，taker buy/sell volume、long/short account ratio、contracts OI/volume、OI history、premium history 都没有证明能覆盖 `2023-01-01` 到 `2026-03-31`；funding 完整和 mark/index candle 可用仍不足以替代 OI/taker/long-short confirmation。

本阶段不下载多年大数据，不修改 `OkxAdaptiveMhfStrategy`，不新增 Strategy class，不新增 demo/live runner，不连接真实交易，也不引入均值回归。当前推荐暂停策略开发，只维护数据与研究工具；Strategy V3、V3.1、demo 和 live 仍然禁止。

## External Regime Classifier Feasibility

`make audit-external-regime` 用于审计是否具备 external regime classifier 的 research-only 前提，不是策略开发、不修改交易逻辑、不新增 demo/live runner。

当前 V3 family 已在 Research Decision Dossier 中失败，且 funding-aware final gate 仍保持关闭。本审计只检查五品种不扩容前提下，已有 1m 行情、OKX funding CSV 和可无密钥获取但尚未落地的 OKX public/trading statistics 外部特征，是否足够支持新的趋势 regime classifier 研究。

允许的下一步仅限 classifier research，例如构造 trend breadth、cross-symbol correlation/dispersion、volatility regime、funding regime 等 regime 特征，并单独审计 open interest、long/short ratio、taker volume、basis/premium、mark/index divergence 的下载可行性。即使 `external_regime_classifier_research_allowed=true`，也必须继续保持 `strategy_development_allowed=false` 和 `demo_live_allowed=false`，不得进入 demo/live。

输出目录为 `reports/research/external_regime_feasibility/`，包含 Markdown、JSON 和 CSV 特征清单。

## External Regime Classifier Research

`make research-external-regime-classifier` 是 research-only 的外部 regime classifier 研究入口，不是策略开发，不修改 `OkxAdaptiveMhfStrategy`，不新增 demo/live runner。

该研究只使用已有五品种 market data 和 OKX Historical Market Data funding，构造日线级 market-wide trend、cross-symbol、volatility、drawdown/rebound 和 funding regime 特征。classifier 特征禁止使用未来收益、未来 regime 信息、V3 policy PnL 或任何 post-trade 结果。

阈值只能从 `train_ext` 学习：`2023-01-01` 到 `2024-06-30`。`validation_ext`（`2024-07-01` 到 `2025-06-30`）和 `oos_ext`（`2025-07-01` 到 `2026-03-31`）只能用于检验，不允许反向调阈值或选择规则。

脚本会对 V3 extended trades 做 post-trade regime attribution，并输出离线 classifier filter experiment。即使结果满足 gate，也只允许 `can_enter_research_only_v3_1_classifier_experiment=true`，仍必须保持 `strategy_development_allowed=false` 和 `demo_live_allowed=false`，不得标记为 tradable。

classifier filter result 还必须通过 `make audit-external-regime-gates` 的 gate consistency audit。该审计使用 Dossier / Extended V3 compare 一致的 strict gate：`original_all` 不能绕过 strict gate，OOS top 5% trade contribution、largest symbol PnL share、actual funding-adjusted PnL、三段 trade count 和 OOS trade-set 是否被 filter 真正改变都必须重新检查。只有 gate audit 也通过时，才允许进入 research-only V3.1 classifier-filtered experiment；Strategy V3、demo 和 live 仍然禁止。

输出目录为 `reports/research/external_regime_classifier/`。

## 目录结构

- `config/`：运行配置和策略配置。
- `config/instruments/`：本地合约元数据，例如 `btcusdt_swap_okx.json`。
- `data/raw/`：可选原始行情/CSV 导出目录，已在 `.gitignore` 中保留 `.gitkeep`。
- `data/history_manifests/`：历史下载断点续传 manifest。
- `scripts/`：所有 headless 命令行脚本。
- `strategies/`：CTA 策略类。
- `tests/`：`unittest` 测试。
- `reports/`：回测、验证、诊断、sweep 输出目录，生成内容不提交。
- `logs/`：脚本日志目录，生成 `.log` 不提交。
- `deploy/systemd/`：后续 VPS/systemd 部署占位。

## 快速开始

```bash
git clone git@github.com:yifanfengsu/CTA.git
cd CTA
make venv
source .venv/bin/activate
make install
make env
```

然后编辑 `.env`，填入 OKX DEMO 或 REAL 对应的 API Key、Secret、Passphrase，再执行：

```bash
make doctor
```

## .env 配置说明

`.env` 从 `.env.example` 复制生成：

```dotenv
OKX_API_KEY=
OKX_SECRET_KEY=
OKX_PASSPHRASE=
OKX_SERVER=DEMO
OKX_PROXY_HOST=
OKX_PROXY_PORT=0
```

- `OKX_API_KEY`：OKX API Key。
- `OKX_SECRET_KEY`：OKX Secret Key。
- `OKX_PASSPHRASE`：OKX API Passphrase。
- `OKX_SERVER`：`DEMO` 或 `REAL`，要和密钥环境匹配。
- `OKX_PROXY_HOST`：代理主机，留空表示不用代理。
- `OKX_PROXY_PORT`：代理端口；不用代理时保持 `0`。

不要提交 `.env`。本仓库也忽略本地 SQLite 数据库、日志、报告和大体积行情输出。

## 推荐执行顺序

1. `make doctor`
2. `make inspect-okx`
3. `make check-okx SERVER=DEMO`
4. `make download-history-dry-run`
5. `make download-history`
6. `make verify-history`
7. `make backtest`
8. `make backtest-no-cost`
9. `make analyze-alpha REPORT_DIR=... COMPARE_REPORT_DIR=...`
10. `make analyze-trades REPORT_DIR=...`
11. `make backtest-trace START=... END=... OUTPUT_DIR=...`
12. `make analyze-signals REPORT_DIR=...`
13. `make research-entry REPORT_DIR=reports/research/trace_train`
14. `make research-entry REPORT_DIR=reports/research/trace_validation`
15. `make research-entry REPORT_DIR=reports/research/trace_oos`
16. `make research-features REPORT_DIR=reports/research/trace_train`
17. `make research-features REPORT_DIR=reports/research/trace_validation`
18. `make research-features REPORT_DIR=reports/research/trace_oos`
19. `make compare-features TRAIN_DIR=... VALIDATION_DIR=... OOS_DIR=...`
20. `make research-htf SPLIT=train`
21. `make research-htf SPLIT=validation`
22. `make research-htf SPLIT=oos`
23. `make compare-htf`
24. `make ablation SPLIT=train|validation|oos OUTPUT_DIR=...`
25. `make postmortem-trend-v3`
26. `make audit-extended-history`
27. `make research-trend-v3-extended SPLIT=train_ext`
28. `make research-trend-v3-extended SPLIT=validation_ext`
29. `make research-trend-v3-extended SPLIT=oos_ext`
30. `make compare-trend-v3-extended`
31. `make diagnose-trend-regimes`
32. `make research-dossier`
33. `make audit-external-regime`
34. `make research-external-regime-classifier`
35. `make audit-external-regime-gates`
36. `make research-vsvcb-v1`
37. `make postmortem-vsvcb-v1`
38. `make audit-derivatives-data`
39. `make research-csrb-v1`
40. `make alpha-sweep`
41. 满足条件后再考虑补 demo runner/模拟盘。

## Makefile 变量

所有变量都可以用 `make target VAR=value` 覆盖：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `PYTHON` | `.venv/bin/python` | Python 解释器路径 |
| `PIP` | `$(PYTHON) -m pip` | pip 调用方式 |
| `VT_SYMBOL` | `BTCUSDT_SWAP_OKX.GLOBAL` | 回测/下载/验证标的 |
| `SYMBOLS` | `BTCUSDT_SWAP_OKX.GLOBAL ETHUSDT_SWAP_OKX.GLOBAL SOLUSDT_SWAP_OKX.GLOBAL LINKUSDT_SWAP_OKX.GLOBAL DOGEUSDT_SWAP_OKX.GLOBAL` | 批量下载/批量验证的第一批多品种标的 |
| `INST_IDS` | `BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP,LINK-USDT-SWAP,DOGE-USDT-SWAP` | OKX metadata 刷新的第一批合约 ID |
| `INTERVAL` | `1m` | 历史 K 线周期 |
| `START` | `2023-01-01` | 起始日期，包含当天；funding research 默认使用 2023-2026 扩展窗口 |
| `END` | `2026-03-31` | 结束日期，包含当天 |
| `TIMEZONE` | `Asia/Shanghai` | 日期解释时区 |
| `CHUNK_DAYS` | `3` | 历史下载分块天数 |
| `SERVER` | `DEMO` | OKX 环境，`DEMO` 或 `REAL` |
| `CAPITAL` | `5000` | 回测初始资金 |
| `RATE` | `0.0005` | 手续费率 |
| `SLIPPAGE_MODE` | `ticks` | 滑点模式：`ticks` 或 `absolute` |
| `SLIPPAGE` | `2` | 滑点数值 |
| `REPORT_DIR` | 空 | alpha 诊断主报告目录，必填 |
| `COMPARE_REPORT_DIR` | 空 | alpha 诊断对照报告目录 |
| `OUTPUT_DIR` | 空 | 指定输出目录；为空时脚本自动生成 |
| `SIGNAL_TRACE_PATH` | 空 | signal trace CSV 路径；为空时使用 `REPORT_DIR/signal_trace.csv` 或 `OUTPUT_DIR/signal_trace.csv` |
| `FORMAT` | 空 | `analyze-trades` 输出格式，支持 `json`、`csv`、`md`，为空时全部输出 |
| `SPLIT` | `train` | 样本切分：`full`、`train`、`validation`、`oos`；HTF research 默认用 `train` |
| `HORIZONS` | `5,15,30,60,120` | signal outcome 分析的未来分钟窗口 |
| `ENTRY_HORIZONS` | `15,30,60,120` | entry policy bracket 研究的未来分钟窗口 |
| `FEATURE_HORIZONS` | `15,30,60,120` | Signal Lab 特征研究的未来收益窗口 |
| `HTF_HORIZONS` | `60,120,240,480` | HTF Signal Research 的未来分钟窗口 |
| `FEATURE_BINS` | `5` | Signal Lab 数值特征 quantile 分箱数 |
| `FEATURE_MIN_COUNT` | `50` | Signal Lab 报告判定候选特征时使用的最小样本数 |
| `FEATURE_LIST` | 空 | Signal Lab 指定特征列表；为空时分析全部特征 |
| `ENTRY_MAX_WAIT_BARS` | `10` | delayed/pullback/followthrough policy 最多等待的 1m bar 数 |
| `STOP_ATR_GRID` | `1.0,1.5,2.0,2.5,3.0,4.0` | entry policy 虚拟 bracket 的止损 ATR 网格 |
| `TP_ATR_GRID` | `1.5,2.0,2.5,3.0,4.0,5.0` | entry policy 虚拟 bracket 的止盈 ATR 网格 |
| `HTF_STOP_ATR_GRID` | `1.5,2.0,2.5,3.0,4.0` | HTF no-cost bracket 的止损 ATR 网格 |
| `HTF_TP_ATR_GRID` | `2.0,3.0,4.0,5.0,6.0` | HTF no-cost bracket 的止盈 ATR 网格 |
| `HTF_COOLDOWN_BARS_5M` | `6` | HTF 同 policy/同方向信号冷却 5m bar 数 |
| `HTF_OUTPUT_DIR` | `reports/research/htf_signals/$(SPLIT)` | `research-htf` 输出目录 |
| `HTF_TRAIN_DIR` | `reports/research/htf_signals/train` | `compare-htf` 默认 train 目录 |
| `HTF_VALIDATION_DIR` | `reports/research/htf_signals/validation` | `compare-htf` 默认 validation 目录 |
| `HTF_OOS_DIR` | `reports/research/htf_signals/oos` | `compare-htf` 默认 oos 目录 |
| `HTF_COMPARE_OUTPUT_DIR` | `reports/research/htf_compare` | `compare-htf` 输出目录 |
| `TREND_V3_OUTPUT_DIR` | `reports/research/trend_following_v3/$(SPLIT)` | `research-trend-v3` 输出目录 |
| `TREND_V3_TRAIN_DIR` | `reports/research/trend_following_v3/train` | `compare-trend-v3` 默认 train 目录 |
| `TREND_V3_VALIDATION_DIR` | `reports/research/trend_following_v3/validation` | `compare-trend-v3` 默认 validation 目录 |
| `TREND_V3_OOS_DIR` | `reports/research/trend_following_v3/oos` | `compare-trend-v3` 默认 oos 目录 |
| `TREND_V3_COMPARE_OUTPUT_DIR` | `reports/research/trend_following_v3_compare` | `compare-trend-v3` 输出目录 |
| `TREND_V3_POSTMORTEM_OUTPUT_DIR` | `reports/research/trend_following_v3_postmortem` | `postmortem-trend-v3` 输出目录 |
| `TREND_V3_EXT_OUTPUT_DIR` | `reports/research/trend_following_v3_extended/$(EXT_SPLIT)` | `research-trend-v3-extended` 输出目录 |
| `TREND_V3_EXT_TRAIN_DIR` | `reports/research/trend_following_v3_extended/train_ext` | `compare-trend-v3-extended` 默认 train_ext 目录 |
| `TREND_V3_EXT_VALIDATION_DIR` | `reports/research/trend_following_v3_extended/validation_ext` | `compare-trend-v3-extended` 默认 validation_ext 目录 |
| `TREND_V3_EXT_OOS_DIR` | `reports/research/trend_following_v3_extended/oos_ext` | `compare-trend-v3-extended` 默认 oos_ext 目录 |
| `TREND_V3_EXT_COMPARE_OUTPUT_DIR` | `reports/research/trend_following_v3_extended_compare` | `compare-trend-v3-extended` 输出目录 |
| `FUNDING_OUTPUT_DIR` | `data/funding/okx` | OKX funding CSV 输出目录，CSV 被 `.gitignore` 忽略 |
| `FUNDING_REPORTS_DIR` | `reports/research/funding` | OKX funding 下载和验证报告目录 |
| `TREND_V3_ACTUAL_FUNDING_OUTPUT_DIR` | `reports/research/trend_following_v3_actual_funding` | Trend V3 actual funding 分析输出目录 |
| `EXTENDED_HISTORY_OUTPUT_DIR` | `reports/research/extended_history_availability` | `audit-extended-history` 输出目录 |
| `RESEARCH_DOSSIER_OUTPUT_DIR` | `reports/research/research_decision_dossier` | `research-dossier` 输出目录 |
| `EXTERNAL_REGIME_OUTPUT_DIR` | `reports/research/external_regime_feasibility` | `audit-external-regime` 输出目录 |
| `EXTERNAL_REGIME_CLASSIFIER_OUTPUT_DIR` | `reports/research/external_regime_classifier` | `research-external-regime-classifier` 输出目录 |
| `EXTERNAL_REGIME_GATE_AUDIT_OUTPUT_DIR` | `reports/research/external_regime_classifier_gate_audit` | `audit-external-regime-gates` 输出目录 |
| `VSVCB_OUTPUT_DIR` | `reports/research/vsvcb_v1` | `research-vsvcb-v1` 输出目录 |
| `VSVCB_POSTMORTEM_OUTPUT_DIR` | `reports/research/vsvcb_v1_postmortem` | `postmortem-vsvcb-v1` 输出目录 |
| `DERIVATIVES_DATA_READINESS_OUTPUT_DIR` | `reports/research/derivatives_data_readiness` | `audit-derivatives-data` 输出目录 |
| `CSRB_OUTPUT_DIR` | `reports/research/csrb_v1` | `research-csrb-v1` 输出目录 |
| `TRAIN_DIR` | 空 | `compare-features` 的 train `signal_feature_research` 目录 |
| `VALIDATION_DIR` | 空 | `compare-features` 的 validation `signal_feature_research` 目录 |
| `OOS_DIR` | 空 | `compare-features` 的 oos `signal_feature_research` 目录 |
| `STRATEGY_CONFIG` | `config/strategy_default.json` | 默认策略配置 |
| `SANITY_CONFIG` | `config/strategy_sanity_min_size.json` | 保守 sanity 配置 |
| `MAX_RUNS` | `100` | ablation 最多候选数 |
| `MAX_RETRIES` | `8` | 历史下载每源重试次数 |
| `THROTTLE_SECONDS` | `0.35` | 历史下载请求间隔 |

## Makefile 命令总览

| 命令 | 作用 | 是否联网 | 是否写数据库 | 主要输出 | 常用示例 |
| --- | --- | --- | --- | --- | --- |
| `make help` | 打印命令和变量示例 | 否 | 否 | 终端输出 | `make` |
| `make venv` | 创建 `.venv` | 否 | 否 | `.venv/` | `make venv` |
| `make install` | 安装运行依赖 | 是，pip | 否 | `.venv/` 包 | `make install` |
| `make env` | 创建 `.env`，不覆盖已有文件 | 否 | 否 | `.env` | `make env` |
| `make doctor` | 本地依赖和 vn.py sqlite 自检 | 否 | 否 | `logs/doctor.log` | `make doctor` |
| `make inspect-okx` | 本地 OKX gateway 字段检查 | 否 | 否 | `logs/inspect_okx_gateway.log` | `make inspect-okx` |
| `make check-okx` | OKX 登录和合约元数据检查，不下单 | 是 | 否 | `config/instruments/*.json`、日志 | `make check-okx SERVER=DEMO` |
| `make refresh-okx-metadata-dry-run` | 读取 OKX public metadata 并生成报告，不写 instrument | 是，public REST | 否 | `reports/research/multisymbol_readiness/okx_metadata_refresh*` | `make refresh-okx-metadata-dry-run` |
| `make refresh-okx-metadata` | 读取 OKX public metadata 并更新 instrument JSON | 是，public REST | 否 | `config/instruments/*.json`、metadata refresh 报告 | `make refresh-okx-metadata` |
| `make download-history-dry-run` | 生成下载计划，不下载、不写 bar | 否 | 否 | 终端计划、日志 | `make download-history-dry-run START=2025-01-01 END=2025-01-02 CHUNK_DAYS=1` |
| `make download-history` | 下载历史数据、逐块保存、校验完整性 | 是 | 是 | sqlite、`data/history_manifests/`、日志 | `make download-history CHUNK_DAYS=3` |
| `make download-history-batch-dry-run` | 对 `SYMBOLS` 逐个生成下载计划，不写数据库 | 否 | 否 | 终端计划、日志 | `make download-history-batch-dry-run START=2025-01-01 END=2025-01-07` |
| `make download-history-batch` | 对 `SYMBOLS` 逐个下载 1m 历史数据 | 是 | 是 | sqlite、manifest、日志 | `make download-history-batch START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3` |
| `make repair-history` | 按本地缺口修复历史数据 | 是 | 是 | sqlite、manifest、日志 | `make repair-history START=2025-01-01 END=2025-01-31` |
| `make verify-history` | 独立验证本地历史完整性 | 否 | 否 | `reports/history_verify_latest.json` | `make verify-history` |
| `make verify-history-batch` | 对 `SYMBOLS` 逐个验证历史覆盖 | 否 | 否 | `reports/history_verify/*` | `make verify-history-batch START=2025-01-01 END=2026-03-31` |
| `make audit-multisymbol` | 审计多品种 metadata、sqlite 覆盖和 Makefile readiness | 否 | 否 | `reports/research/multisymbol_readiness/` | `make audit-multisymbol` |
| `make backtest` | 成本版回测 | 否 | 否 | `reports/backtest/YYYYMMDD_HHMMSS/` 或 `OUTPUT_DIR` | `make backtest OUTPUT_DIR=reports/backtest/manual_cost` |
| `make backtest-no-cost` | 无成本回测，用于判断毛 alpha | 否 | 否 | 回测报告目录 | `make backtest-no-cost OUTPUT_DIR=reports/backtest/manual_no_cost` |
| `make backtest-trace` | 无成本回测并导出 `signal_trace.csv` | 否 | 否 | 回测报告目录和 `signal_trace.csv` | `make backtest-trace START=2025-01-01 END=2025-03-31 OUTPUT_DIR=reports/research/trace_2025q1` |
| `make backtest-sanity` | 使用保守最小手数配置回测 | 否 | 否 | 回测报告目录 | `make backtest-sanity` |
| `make analyze-alpha` | 分析一个或两个回测报告 | 否 | 否 | `REPORT_DIR/alpha_diagnostics/` 或 `OUTPUT_DIR` | `make analyze-alpha REPORT_DIR=... COMPARE_REPORT_DIR=...` |
| `make analyze-trades` | 交易归因诊断，按方向、时段、星期、月份和频率拆解亏损 | 否 | 否 | `REPORT_DIR/trade_attribution/` 或 `OUTPUT_DIR` | `make analyze-trades REPORT_DIR=reports/backtest/main_no_cost_20250101_20260331` |
| `make analyze-signals` | 分析 entry signal 后的 MFE/MAE 和突破延续性 | 否 | 否 | `REPORT_DIR/signal_outcomes/` 或 `OUTPUT_DIR` | `make analyze-signals REPORT_DIR=reports/research/trace_2025q1` |
| `make research-entry` | 用 signal trace 和未来 1m bar 离线研究 delayed confirm、pullback 和 breakout distance filter | 否 | 否 | `REPORT_DIR/entry_policy_research/` 或 `OUTPUT_DIR` | `make research-entry REPORT_DIR=reports/research/trace_train` |
| `make research-features` | 用 signal trace 和 1m bar 研究特征对未来收益/MFE/MAE 的预测力 | 否 | 否 | `REPORT_DIR/signal_feature_research/` 或 `OUTPUT_DIR` | `make research-features REPORT_DIR=reports/research/trace_train` |
| `make compare-features` | 比较 train/validation/oos 的 Signal Lab 特征稳定性 | 否 | 否 | `reports/research/feature_compare/` 或 `OUTPUT_DIR` | `make compare-features TRAIN_DIR=... VALIDATION_DIR=... OOS_DIR=...` |
| `make research-htf` | 离线研究 1h regime + 15m structure + 5m pullback/reclaim 信号质量 | 否 | 否 | `reports/research/htf_signals/$(SPLIT)` | `make research-htf SPLIT=train` |
| `make compare-htf` | 比较 train/validation/oos 的 HTF policy 稳定性 | 否 | 否 | `reports/research/htf_compare/` | `make compare-htf` |
| `make research-trend-v3` | 多品种组合级趋势跟踪研究 | 否 | 否 | `reports/research/trend_following_v3/$(SPLIT)` | `make research-trend-v3 SPLIT=train` |
| `make compare-trend-v3` | 比较 Trend V3 train/validation/oos 稳定性 | 否 | 否 | `reports/research/trend_following_v3_compare/` | `make compare-trend-v3` |
| `make research-trend-v3-extended` | 2023-2026 长样本复测同一 Trend V3.0 policy set | 否 | 否 | `reports/research/trend_following_v3_extended/$(EXT_SPLIT)` | `make research-trend-v3-extended SPLIT=train_ext` |
| `make compare-trend-v3-extended` | 比较 train_ext/validation_ext/oos_ext 并输出 funding stress | 否 | 否 | `reports/research/trend_following_v3_extended_compare/` | `make compare-trend-v3-extended` |
| `make download-funding-dry-run` | 生成 OKX public funding history 下载计划，不写 funding CSV | 否 | 否 | `reports/research/funding/okx_funding_download_*` | `make download-funding-dry-run` |
| `make download-funding` | 下载 OKX public funding history CSV | 是，public REST | 否 | `data/funding/okx/*.csv`、download report | `make download-funding` |
| `make verify-funding` | 验证 funding CSV timestamp、重复和大缺口 | 否 | 否 | `reports/research/funding/okx_funding_verify_*` | `make verify-funding` |
| `make verify-funding-allow-partial` | 允许 partial funding 以便审计报告继续生成，但不能用于策略决策 | 否 | 否 | `reports/research/funding/okx_funding_verify_*` | `make verify-funding-allow-partial` |
| `make import-funding-csv` | 导入用户手动下载的 OKX historical funding rate CSV | 否 | 否 | `data/funding/okx/*.csv` | `make import-funding-csv INPUT=/path/funding.csv INST_ID=BTC-USDT-SWAP` |
| `make probe-funding-source` | 探测 OKX Historical Market Data query endpoint 是否可自动提供 funding 文件 | 是，public docs/query | 否 | `reports/research/funding_endpoint_probe/` | `make probe-funding-source` |
| `make download-funding-historical-dry-run` | 生成 OKX historical funding file 下载计划，不下载大文件 | 否 | 否 | `reports/research/funding_historical_download/` | `make download-funding-historical-dry-run` |
| `make download-funding-historical` | endpoint probe 确认可用后自动下载并导入 OKX historical funding 文件 | 是，public query/file URLs | 否 | `data/funding/okx_historical_raw/`、`data/funding/okx/` | `make download-funding-historical` |
| `make analyze-trend-v3-funding` | 将 Trend V3 extended trades 对齐真实 OKX funding 并重算 PnL | 否 | 否 | `reports/research/trend_following_v3_actual_funding/` | `make analyze-trend-v3-funding` |
| `make postmortem-trend-v3` | Trend V3.0 失败归因复盘 | 否 | 否 | `reports/research/trend_following_v3_postmortem/` | `make postmortem-trend-v3` |
| `make audit-extended-history` | 审计 2025/2023/2021 长历史可用性和下载计划 | 否 | 否 | `reports/research/extended_history_availability/` | `make audit-extended-history` |
| `make research-dossier` | 归档当前研究结论、失败方向和下一步研究选项 | 否 | 否 | `reports/research/research_decision_dossier/` | `make research-dossier` |
| `make audit-external-regime` | 审计 external regime classifier research-only 可行性 | 否 | 否 | `reports/research/external_regime_feasibility/` | `make audit-external-regime` |
| `make research-external-regime-classifier` | 基于 train_ext 阈值研究 external regime classifier 离线过滤实验 | 否 | 否 | `reports/research/external_regime_classifier/` | `make research-external-regime-classifier` |
| `make audit-external-regime-gates` | 审计 classifier strict gate 是否与 Dossier / Extended V3 compare 一致 | 否 | 否 | `reports/research/external_regime_classifier_gate_audit/` | `make audit-external-regime-gates` |
| `make research-vsvcb-v1` | VSVCB-v1 Research-Only Phase 1 事件研究和固定持有基准 | 否 | 否 | `reports/research/vsvcb_v1/` | `make research-vsvcb-v1` |
| `make postmortem-vsvcb-v1` | VSVCB-v1 Phase 1 失败复盘，不调参、不开发策略 | 否 | 否 | `reports/research/vsvcb_v1_postmortem/` | `make postmortem-vsvcb-v1` |
| `make audit-derivatives-data` | OKX public/no-key 衍生品数据可得性审计，不下载多年数据 | 否 | 否 | `reports/research/derivatives_data_readiness/` | `make audit-derivatives-data` |
| `make research-csrb-v1` | CSRB-v1 Research-Only Phase 1 session range breakout 事件研究和固定持有基准 | 否 | 否 | `reports/research/csrb_v1/` | `make research-csrb-v1` |
| `make alpha-sweep` | 保守参数 shortlist sweep | 否 | 否 | `reports/alpha_sweep/YYYYMMDD_HHMMSS/` 或 `OUTPUT_DIR` | `make alpha-sweep OUTPUT_DIR=reports/alpha_sweep/manual_001` |
| `make ablation` | 方向、周末、小时过滤诊断实验 | 否 | 否 | `reports/ablation/main_20250101_20260331/` 或 `OUTPUT_DIR` | `make ablation SPLIT=oos OUTPUT_DIR=reports/ablation/oos` |
| `make test` | 运行全部单元测试 | 否 | 否 | 终端输出 | `make test` |
| `make test-one` | 运行单个测试文件 | 否 | 否 | 终端输出 | `make test-one TEST=tests/test_history_time_utils.py` |
| `make compile` | 编译检查脚本、策略、测试 | 否 | 否 | `__pycache__/` | `make compile` |
| `make clean-cache` | 删除缓存目录 | 否 | 否 | 删除缓存 | `make clean-cache` |
| `make clean-logs` | 删除 `logs/*.log`，保留 `.gitkeep` | 否 | 否 | 清理日志 | `make clean-logs` |
| `make clean-reports` | 删除报告，必须确认 | 否 | 否 | 清理 `reports/` | `make clean-reports CONFIRM=1` |
| `make tail-log` | 查看日志尾部 | 否 | 否 | 终端输出 | `make tail-log LOG_FILE=logs/download_okx_history.log` |

## 每个命令的详细用法

### `make help`

默认目标，`make` 等同于 `make help`。用于快速查看当前支持的命令、常用变量和覆盖示例。

### `make venv`

创建 `.venv`：

```bash
make venv
```

如果 `.venv` 已存在，目标会提示并跳过创建。

### `make install`

升级 pip 并安装项目运行依赖：

- `vnpy`
- `vnpy_ctastrategy`
- `vnpy_okx`
- `vnpy_sqlite`
- `python-dotenv`
- `pandas`
- `numpy`

仓库当前没有 `requirements.txt`，Makefile 内部保留最小安装列表。

### `make env`

当 `.env` 不存在时，从 `.env.example` 复制：

```bash
make env
```

如果 `.env` 已存在，不会覆盖，避免误删本地密钥。

### `make doctor`

运行：

```bash
make doctor
```

实际调用 `scripts/doctor.py`，检查 Python、依赖包、OKX gateway 导入、vn.py sqlite 设置和 `.env` 是否存在。主要日志在 `logs/doctor.log`。

### `make inspect-okx`

运行：

```bash
make inspect-okx
```

实际调用 `scripts/inspect_okx_gateway.py`。这是本地 gateway 字段检查，不连接交易所，适合在填 API Key 前确认本地 `vnpy_okx` 版本的字段名称。

### `make check-okx`

运行：

```bash
make check-okx SERVER=DEMO
make check-okx SERVER=REAL
```

实际调用 `scripts/check_okx_connection.py --vt-symbol $(VT_SYMBOL) --server $(SERVER) --timeout 30`。它会连接 OKX、等待私有登录和目标合约元数据，但不会下单。成功后会更新 `config/instruments/` 中的合约元数据文件。

只有 `.env` 已填完整并且 `OKX_SERVER` 与密钥环境匹配时才执行该命令。

### `make refresh-okx-metadata-dry-run` / `make refresh-okx-metadata`

通过 OKX public instruments endpoint 刷新 SWAP 合约元数据；不需要 API key，不连接真实交易，不下单。

```bash
make refresh-okx-metadata-dry-run
make refresh-okx-metadata
```

默认读取 `INST_IDS`，输出：

- `reports/research/multisymbol_readiness/okx_metadata_refresh.json`
- `reports/research/multisymbol_readiness/okx_metadata_refresh_report.md`

`dry-run` 只写报告，不写 `config/instruments/`。`write` 成功时会把 OKX 返回的 `instId/instType/ctVal/tickSz/minSz` 映射为 canonical `okx_inst_id/product/size/pricetick/min_volume`，保留 `name`，并把 `needs_okx_contract_metadata_refresh` 改为 `false`；失败时不会填假值，placeholder 会继续保持 `needs_okx_contract_metadata_refresh=true`。

### `make download-history-dry-run`

生成历史下载计划：

```bash
make download-history-dry-run START=2025-01-01 END=2025-01-02 CHUNK_DAYS=1
```

实际调用 `scripts/download_okx_history.py`，固定带 `--source auto --resume --dry-run`。该命令不联系 OKX、不保存 bar、不写 manifest；它会读取本地 sqlite 覆盖情况并打印计划。

### `make download-history`

下载历史数据并写入本地 vn.py sqlite：

```bash
make download-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
```

固定带：

- `--source auto`
- `--resume`
- `--save-per-chunk`
- `--verify-db`
- `--strict-completeness`
- `--max-retries $(MAX_RETRIES)`
- `--throttle-seconds $(THROTTLE_SECONDS)`

主要输出是本地 sqlite 数据库、`data/history_manifests/` 断点续传文件、`logs/download_okx_history.log`。

### `make download-history-batch-dry-run` / `make download-history-batch`

对 `SYMBOLS` 逐个调用 `scripts/download_okx_history.py`。每个 symbol 开始前会打印当前 symbol，任一 symbol 失败时整个 Make target 返回非 0。

```bash
make download-history-batch-dry-run START=2025-01-01 END=2025-01-07
make download-history-batch START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
```

`download-history-batch-dry-run` 固定带 `--dry-run`，不下载、不写数据库。真实批量下载固定带 `--source auto --resume --save-per-chunk --verify-db --strict-completeness`，不会被默认目标自动执行，必须手动运行。

### `make repair-history`

按本地 sqlite 缺口进行修复：

```bash
make repair-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
```

固定带 `--repair-missing --source auto --resume --verify-db --strict-completeness`。当 `verify-history` 或 backtest preflight 报缺口时，优先运行这个目标。

### `make verify-history`

独立验证本地历史覆盖：

```bash
make verify-history START=2025-01-01 END=2026-03-31
```

输出固定为 `reports/history_verify_latest.json`，避免把 `VT_SYMBOL` 中的特殊字符拼进文件名。严格模式下发现缺口会返回非零退出码，并打印建议的 repair 命令。

### `make verify-history-batch`

对 `SYMBOLS` 逐个验证本地历史覆盖：

```bash
make verify-history-batch START=2025-01-01 END=2026-03-31
```

每个 symbol 输出单独 JSON 到 `reports/history_verify/`，文件名会把 `.` 和 `/` 替换成 `_`。

### `make audit-multisymbol`

运行多品种数据准备能力审计：

```bash
make audit-multisymbol
make audit-multisymbol START=2025-01-01 END=2025-01-07
```

该命令只读取本地文件和 sqlite，不下载真实数据、不连接 OKX、不修改策略交易逻辑。输出：

- `reports/research/multisymbol_readiness/multisymbol_readiness.json`
- `reports/research/multisymbol_readiness/multisymbol_readiness_report.md`

审计内容包括：

- 扫描 `config/instruments/*.json`；
- 检查每个 instrument 是否包含 canonical `okx_inst_id/product/size/pricetick/min_volume`；
- 检查本地 sqlite 是否完整覆盖 audit window 内对应 `symbol/exchange/interval` 历史数据；
- 检查 Makefile 是否存在多品种批量下载和批量 verify 目标；
- 判断是否可以进入 Trend V3 多品种研究设计。

`metadata_complete` 和 `history_ready` 是两件事：前者只表示 instrument JSON 的 canonical metadata 完整且不需要刷新；后者表示当前 audit window 的 `expected_count/total_count/missing_count/gap_count` 完整通过。默认 audit window 是 `START=2025-01-01 END=2026-03-31 INTERVAL=1m TIMEZONE=Asia/Shanghai`。7 天短区间验证只用于验收下载链路，不代表可以进入完整 Trend V3。

凡是带有 `needs_okx_contract_metadata_refresh=true`、缺少 `okx_inst_id/product` 或 `size/pricetick/min_volume` 非正数的 instrument，都不能直接用于正式回测。`ready_symbols` 只统计 metadata 完整且当前 audit window 完整覆盖的 symbol；`can_enter_trend_v3` 至少要求 3 个 ready symbol，并包含 BTC 和 ETH，同时 Makefile 具备 batch download / batch verify 目标。使用短区间 `START/END` 覆盖 audit window 时，报告中的 ready 只代表该短窗口 ready，不代表完整 Trend V3 readiness。

### `make audit-extended-history`

运行趋势跟踪长样本可用性审计：

```bash
make audit-extended-history
```

该命令只读取本地 instrument JSON 和 vn.py sqlite，不下载数据、不连接私有交易、不下单、不修改策略逻辑。默认审计 BTC / ETH / SOL / LINK / DOGE 的三个窗口：

- `2025-01-01:2026-03-31`
- `2023-01-01:2026-03-31`
- `2021-01-01:2026-03-31`

输出：

- `reports/research/extended_history_availability/extended_history_availability.json`
- `reports/research/extended_history_availability/extended_history_availability_report.md`
- `reports/research/extended_history_availability/extended_history_missing_ranges.csv`
- `reports/research/extended_history_availability/extended_history_download_plan.csv`

报告会回答当前 2025 window 是否完整、扩展到 2023 或 2021 需要补哪些 symbol 和区间、是否存在 listing time unknown、推荐下一步下载窗口，以及是否可以进入 Extended Trend Research。`--check-okx-listing-metadata` 是可选 public metadata 检查；如果 OKX metadata 没有 `listTime` 或网络不可用，报告记录 `unknown`/warning，不会伪造上市时间，也不会让审计失败。

### `make backtest`

运行成本版回测：

```bash
make backtest
make backtest OUTPUT_DIR=reports/backtest/manual_cost
```

默认使用 `config/strategy_default.json`，带手续费 `RATE=0.0005`、滑点 `SLIPPAGE_MODE=ticks`、`SLIPPAGE=2`，并启用 `--data-check-strict`。如果 `OUTPUT_DIR` 为空，脚本自动生成 `reports/backtest/YYYYMMDD_HHMMSS/`。

### `make backtest-no-cost`

运行无成本回测：

```bash
make backtest-no-cost OUTPUT_DIR=reports/backtest/manual_no_cost
```

固定 `--rate 0 --slippage-mode absolute --slippage 0`，其他参数同 `make backtest`。这个目标只用于判断毛 alpha，不用于模拟实盘收益判断。

### `make backtest-trace`

运行无成本回测并导出信号级 trace：

```bash
make backtest-trace START=2025-01-01 END=2025-03-31 OUTPUT_DIR=reports/research/trace_2025q1
```

这个目标等价于 no-cost 回测加 `--export-signal-trace`。默认输出 `OUTPUT_DIR/signal_trace.csv`；如需覆盖路径，可设置 `SIGNAL_TRACE_PATH=...`。trace 默认关闭，普通 `make backtest` 和 `make backtest-no-cost` 不会导出该文件。

### `make backtest-sanity`

使用 `config/strategy_sanity_min_size.json` 运行保守最小手数回测：

```bash
make backtest-sanity
```

该配置使用 `OkxAdaptiveMhfStrategy`、`fixed_size=0.01`、`risk_per_trade=0.0005`，并包含保守的 `max_leverage`、`max_notional_ratio`、`max_trades_per_day`。这个目标用于确认链路和报告是否正常，不用于证明策略已经可上线。

### `make analyze-alpha`

分析回测报告：

```bash
make analyze-alpha REPORT_DIR=reports/backtest/manual_cost
make analyze-alpha REPORT_DIR=reports/backtest/manual_cost COMPARE_REPORT_DIR=reports/backtest/manual_no_cost
```

`REPORT_DIR` 必填。`COMPARE_REPORT_DIR` 为空时只分析单个报告；非空时传入 `--compare-report-dir`，通常用于成本版 vs 无成本版。默认输出到 `REPORT_DIR/alpha_diagnostics/`，也可以用 `OUTPUT_DIR` 覆盖。

### `make analyze-trades`

对某个回测报告做交易归因诊断：

```bash
make analyze-trades REPORT_DIR=reports/backtest/main_no_cost_20250101_20260331
make analyze-trades REPORT_DIR=reports/backtest/main_no_cost_20250101_20260331 FORMAT=json
```

`REPORT_DIR` 必填。默认输出到 `REPORT_DIR/trade_attribution/`，也可以用 `OUTPUT_DIR` 覆盖。脚本读取 `stats.json`、`diagnostics.json`、`trades.csv`、`orders.csv`、`daily_pnl.csv`；缺少非关键文件时只写 warning，不直接崩溃。

### `make analyze-signals`

分析带 `signal_trace.csv` 的回测报告目录：

```bash
make analyze-signals REPORT_DIR=reports/research/trace_2025q1
make analyze-signals REPORT_DIR=reports/research/trace_2025q1 HORIZONS=5,15,30,60
```

`REPORT_DIR` 必填。默认读取 `REPORT_DIR/signal_trace.csv`，从本地 vn.py sqlite 读取 1m bar，并输出到 `REPORT_DIR/signal_outcomes/`。如果 signal trace 放在其他位置，可用 `SIGNAL_TRACE_PATH=...` 覆盖；如果要改输出目录，可用 `OUTPUT_DIR=...`。

### `make research-entry`

用已经导出的 `signal_trace.csv` 做离线入场时机研究：

```bash
make research-entry REPORT_DIR=reports/research/trace_train
make research-entry REPORT_DIR=reports/research/trace_validation
make research-entry REPORT_DIR=reports/research/trace_oos
```

默认读取 `REPORT_DIR/signal_trace.csv`，从本地 vn.py sqlite 读取未来 1m bar，输出到 `REPORT_DIR/entry_policy_research/`。可用 `SIGNAL_TRACE_PATH=...` 指定 trace，用 `OUTPUT_DIR=...` 改输出目录，用 `ENTRY_HORIZONS`、`ENTRY_MAX_WAIT_BARS`、`STOP_ATR_GRID`、`TP_ATR_GRID` 覆盖 bracket 研究参数。

输出文件：

- `entry_policy_summary.json`
- `entry_policy_leaderboard.csv`
- `bracket_grid.csv`
- `policy_by_side.csv`
- `policy_by_hour.csv`
- `policy_report.md`

### `make research-features`

用 `signal_trace.csv` 和本地 vn.py sqlite 中的 1m bar 生成 Signal Lab 特征数据集：

```bash
make research-features REPORT_DIR=reports/research/trace_train
make research-features REPORT_DIR=reports/research/trace_validation
make research-features REPORT_DIR=reports/research/trace_oos
```

默认读取 `REPORT_DIR/signal_trace.csv`，优先合并 `REPORT_DIR/signal_outcomes/signal_outcomes.csv`；如果没有已有 outcome，则脚本内部用 1m bar 计算 `future_return_15m/30m/60m/120m`、`mfe_60m`、`mae_60m`、`mfe_atr`、`mae_atr`、`stop_first`、`tp_first`、`good_signal_60m` 和 `bad_signal_60m`。输出到 `REPORT_DIR/signal_feature_research/`，可用 `SIGNAL_TRACE_PATH`、`OUTPUT_DIR`、`FEATURE_HORIZONS`、`FEATURE_BINS`、`FEATURE_MIN_COUNT`、`FEATURE_LIST` 覆盖。

输出文件：

- `feature_dataset.csv`
- `feature_summary.json`
- `feature_ic.csv`
- `feature_bins.csv`
- `categorical_feature_bins.csv`
- `feature_report.md`

### `make compare-features`

比较三段 Signal Lab 结果：

```bash
make compare-features \
  TRAIN_DIR=reports/research/trace_train/signal_feature_research \
  VALIDATION_DIR=reports/research/trace_validation/signal_feature_research \
  OOS_DIR=reports/research/trace_oos/signal_feature_research
```

输出到 `reports/research/feature_compare/`，可用 `OUTPUT_DIR` 覆盖。报告会标记哪些特征在 train / validation / oos 方向一致，哪些只在单段有效疑似过拟合，以及是否存在可进入策略候选的稳定特征。

### `make research-htf`

离线研究 HTF signal candidates，不修改现有策略交易逻辑，也不进入 demo：

```bash
make research-htf SPLIT=train
make research-htf SPLIT=validation
make research-htf SPLIT=oos
```

默认 split 范围：

- `train`: 2025-01-01 到 2025-09-30
- `validation`: 2025-10-01 到 2025-12-31
- `oos`: 2026-01-01 到 2026-03-31
- `full`: 2025-01-01 到 2026-03-31

脚本从本地 vn.py sqlite 读取 1m bar，内部 resample 出 5m、15m、1h，并输出 `data_quality.json`。默认输出到 `reports/research/htf_signals/$(SPLIT)`，可用 `HTF_OUTPUT_DIR` 覆盖。

输出文件：

- `htf_signal_dataset.csv`
- `htf_policy_summary.json`
- `htf_policy_leaderboard.csv`
- `htf_bracket_grid.csv`
- `htf_policy_by_side.csv`
- `htf_policy_by_hour.csv`
- `htf_policy_by_weekday.csv`
- `htf_research_report.md`
- `data_quality.json`

### `make compare-htf`

比较三段 HTF Signal Research 结果：

```bash
make compare-htf
```

默认读取：

- `reports/research/htf_signals/train`
- `reports/research/htf_signals/validation`
- `reports/research/htf_signals/oos`

输出到 `reports/research/htf_compare/`。报告会判断哪些 policy 在三段方向一致，哪些只在单一 split 有效，是否存在稳定候选进入 Strategy V2；如果没有，`htf_compare_report.md` 会输出 `no_stable_htf_policy=true`。

### Trend V3 Data Preparation：多品种数据准备

Trend V3 仍然是趋势跟踪方向。readiness 未通过前，不开发 Trend V3 多品种研究器，也不把 metadata 不完整的 instrument 用于正式回测。第一批建议 symbols 是 BTC / ETH / SOL / LINK / DOGE；BNB / XRP 作为第二批，占位文件可以存在，但未刷新成功时必须保持 `needs_okx_contract_metadata_refresh=true`。

标准流程：

```bash
make refresh-okx-metadata-dry-run
make refresh-okx-metadata
make download-history-batch-dry-run START=2025-01-01 END=2025-01-07
make download-history-batch START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
make verify-history-batch START=2025-01-01 END=2026-03-31
make audit-multisymbol START=2025-01-01 END=2026-03-31
```

流程说明：

- 先刷新 OKX public contract metadata，确认 `okx_inst_id/product/size/pricetick/min_volume` 来自 OKX 返回值；
- 再用 batch dry-run 检查每个 symbol 的下载计划；
- 再手动运行真实 batch download，写入本地 sqlite；
- 再运行 batch verify，保存每个 symbol 的单独 verify JSON；
- 最后按完整窗口运行 audit readiness，确认 `ready_symbols`、`can_enter_trend_v3` 和 blocking reasons。

完整进入 Trend V3 前必须执行：

```bash
make refresh-okx-metadata
make download-history-batch START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
make verify-history-batch START=2025-01-01 END=2026-03-31
make audit-multisymbol START=2025-01-01 END=2026-03-31
```

`audit-multisymbol` 只读本地文件和 sqlite，不下载数据、不连接 OKX、不修改策略交易逻辑。`ready_symbols` 只统计 metadata 完整且 audit window 内 1m history 完整覆盖的 symbol；`can_enter_trend_v3` 至少要求 3 个 ready symbol，并包含 BTC 和 ETH，同时 Makefile 具备 batch download / batch verify 目标。短区间 audit 例如 `make audit-multisymbol START=2025-01-01 END=2025-01-07` 可以验证 7 天数据链路，但不能替代默认完整窗口 `2025-01-01` 到 `2026-03-31`。

重要限制：当前研究结果暂未纳入 OKX perpetual funding fee，后续进入 Trend V3 后必须单独做 funding fee 敏感性分析。

### Trend Following V2：低频趋势跟踪研究

Trend Following V2 是离线研究框架，不修改现有策略交易逻辑，不进入 demo/live。它不是均值回归，目标是研究真正低频、长持仓的趋势跟踪，而不是 1m 短线追突破或 signal 后固定 horizon return。

```bash
make research-trend-v2 SPLIT=train
make research-trend-v2 SPLIT=validation
make research-trend-v2 SPLIT=oos
make compare-trend-v2
```

默认 split 范围：

- `train`: 2025-01-01 到 2025-09-30
- `validation`: 2025-10-01 到 2025-12-31
- `oos`: 2026-01-01 到 2026-03-31
- `full`: 2025-01-01 到 2026-03-31

脚本从本地 vn.py sqlite 读取 1m bar，内部 resample 出 `15m`、`1h`、`4h`。高周期信号只使用已完成 bar，并用下一根 `15m` close 近似执行价格。默认输出到 `reports/research/trend_following_v2/$(SPLIT)`，跨样本比较输出到 `reports/research/trend_following_v2_compare`。

成本和仓位口径：`fixed_size` / trade `volume` 按合约张数解释，OKX `contract_size` 来自 `config/instruments/*.json` 的 `size` 字段。PnL、手续费、滑点和 R 风险距离均使用 `price_diff * volume * contract_size` 口径；滑点不直接改写 `entry_price` / `exit_price`，而是作为双边不利成交的独立成本项记录到 `slippage`。

默认研究 policy：

- `tf_1h_donchian_20_10`
- `tf_1h_donchian_55_20`
- `tf_4h_donchian_20_10`
- `tf_1h_ema_cross_atr_trail`
- `tf_4h_ema_cross_atr_trail`
- `tf_1h_vol_compression_breakout`
- `tf_1h_donchian_55_with_risk_filters`
- `tf_4h_donchian_20_with_risk_filters`

每个基础 policy 默认测试 ATR trailing multiplier `3.0`、`4.0`、`5.0`，并完整模拟持仓、出场、MFE/MAE、手续费和滑点。趋势跟踪允许低胜率，也允许中位数单笔收益为负；判断重点是总收益、尾部收益、成本后表现、回撤、交易集中度和 OOS。只有跨 train / validation / oos 稳定，才可以进入 Strategy V2；当前不会进入 demo。

输出文件：

- `trend_policy_summary.json`
- `trend_policy_leaderboard.csv`
- `trend_trades.csv`
- `trend_daily_pnl.csv`
- `trend_equity_curve.csv`
- `trend_policy_by_side.csv`
- `trend_policy_by_month.csv`
- `trend_report.md`
- `data_quality.json`
- `trend_research_audit.json`

跨样本比较读取三段 `trend_policy_leaderboard.csv`，输出：

- `trend_compare_summary.json`
- `trend_compare_leaderboard.csv`
- `trend_compare_report.md`

### Trend Following V3：多品种组合级趋势跟踪研究

Trend Following V3 不是实盘策略，而是离线研究框架；它不修改 `OkxAdaptiveMhfStrategy`，不新增 demo/live runner，也不允许直接进入模拟盘或实盘。目标是验证多品种趋势跟踪是否比 BTC 单品种 Trend V2 更稳定。

```bash
make research-trend-v3 SPLIT=train
make research-trend-v3 SPLIT=validation
make research-trend-v3 SPLIT=oos
make compare-trend-v3
```

当前默认 symbols：

- `BTCUSDT_SWAP_OKX.GLOBAL`
- `ETHUSDT_SWAP_OKX.GLOBAL`
- `SOLUSDT_SWAP_OKX.GLOBAL`
- `LINKUSDT_SWAP_OKX.GLOBAL`
- `DOGEUSDT_SWAP_OKX.GLOBAL`

研究使用 `4h` / `1d` Donchian、EMA 50/200、volatility compression breakout，并在组合层面模拟持仓、权益曲线、回撤、symbol contribution、top trade concentration 和最大并发仓位。每个 policy 都输出 no-cost 与 cost-aware 两套结果；cost-aware 包含手续费和滑点，但当前不计 OKX perpetual funding fee，报告会固定写出 funding fee warning。

默认 split 范围：

- `train`: 2025-01-01 到 2025-09-30
- `validation`: 2025-10-01 到 2025-12-31
- `oos`: 2026-01-01 到 2026-03-31
- `full`: 2025-01-01 到 2026-03-31

核心输出：

- `trend_v3_summary.json`
- `trend_v3_policy_leaderboard.csv`
- `trend_v3_portfolio_equity_curve.csv`
- `trend_v3_portfolio_daily_pnl.csv`
- `trend_v3_trades.csv`
- `trend_v3_policy_by_symbol.csv`
- `trend_v3_policy_by_month.csv`
- `trend_v3_symbol_contribution.csv`
- `trend_v3_drawdown.csv`
- `trend_v3_report.md`
- `trend_v3_research_audit.json`
- `data_quality.json`

跨样本比较输出：

- `trend_v3_compare_summary.json`
- `trend_v3_compare_leaderboard.csv`
- `trend_v3_compare_report.md`

只有 `trend_v3_compare_summary.json` 中 `stable_candidate_exists=true` 时，才允许进入 Strategy V3 原型开发；否则保持研究失败结论，不能直接进入 demo/live。

### Trend V3 Postmortem：趋势跟踪失败归因

Trend V3 Postmortem 不是策略开发，也不是参数优化。它只读取已经生成的 Trend V3 train / validation / oos 和 compare 报告，用于解释 V3.0 为什么没有 stable candidate。

```bash
make postmortem-trend-v3
```

默认读取：

- `reports/research/trend_following_v3/train`
- `reports/research/trend_following_v3/validation`
- `reports/research/trend_following_v3/oos`
- `reports/research/trend_following_v3_compare`

Postmortem 会检查：

- policy family：Donchian、EMA、vol compression、ensemble、risk filtered；
- symbol contribution：是否过度依赖单一 symbol，是否去掉某个 symbol 后更好；
- monthly / quarterly regime：validation 和 OOS 的月份结构是否切换；
- tail concentration：top 1 / top 5% / top 10% 盈利交易贡献；
- funding stress：在没有真实 funding 数据时，只做明确标记的 synthetic funding stress。

输出目录默认是 `reports/research/trend_following_v3_postmortem/`，核心文件包括：

- `trend_v3_postmortem_summary.json`
- `trend_v3_postmortem_report.md`
- `policy_family_analysis.csv`
- `symbol_contribution_postmortem.csv`
- `by_month.csv`
- `by_quarter.csv`
- `by_symbol_month.csv`
- `by_policy_month.csv`
- `top_trade_concentration.csv`
- `funding_sensitivity.csv`
- `rejected_candidate_reasons.csv`
- `v3_1_recommendations.json`

只有 postmortem 建议 `proceed_to_v3_1=true` 时，才允许进入 V3.1 研究设计；即使进入 V3.1，也仍然是离线研究，不允许直接进入 Strategy V3 原型、demo 或 live。Postmortem 不会修改 `OkxAdaptiveMhfStrategy`，不新增 demo/live runner，也不会连接真实交易或下单。

### Extended History Availability：趋势跟踪长样本准备

Trend V3.0 已经完成多品种组合级研究和 postmortem；当 `stable_candidate_exists=false` 时，不直接开发 V3.1，也不进入正式策略、demo 或 live。下一步先判断多品种趋势跟踪是否具备更长历史样本，而不是继续扩大当前 V3.0 参数搜索。

```bash
make audit-extended-history
```

`audit-extended-history` 不下载数据，只生成本地覆盖审计、缺口区间和下载计划。长历史下载前必须确认：

- 目标 symbol 的 `okx_inst_id/product/size/pricetick/min_volume` metadata 完整；
- 本地 sqlite 对 `2023-01-01:2026-03-31` 或 `2021-01-01:2026-03-31` 的缺口范围明确；
- OKX listing metadata 已确认，或者报告明确标记 `listing_before_window_start=unknown`，不能猜测合约上市时间；
- 下载命令经过人工确认，不能由 audit 自动执行。

只有扩展历史数据准备完成，且 report 中 `can_enter_extended_trend_research=true`，才进入 Extended Trend Research。availability audit 只是数据准备结论，不是策略收益结论。

### Extended Trend Research：2023-2026 长样本趋势跟踪复测

Extended Trend Research 是离线研究，不是正式策略开发，不修改 `OkxAdaptiveMhfStrategy`，不新增 demo/live runner，也不连接真实交易或下单。目的只是检查 V3.0 失败是否由 2025-2026 短样本导致。

复测必须复用同一组 Trend V3.0 policy set，不新增参数网格，不调整 policy 逻辑，不引入均值回归。长样本 split 使用闭区间日期语义：

| split | start | end |
| --- | --- | --- |
| `train_ext` | 2023-01-01 | 2024-06-30 |
| `validation_ext` | 2024-07-01 | 2025-06-30 |
| `oos_ext` | 2025-07-01 | 2026-03-31 |
| `full_ext` | 2023-01-01 | 2026-03-31 |

运行：

```bash
make research-trend-v3-extended SPLIT=train_ext
make research-trend-v3-extended SPLIT=validation_ext
make research-trend-v3-extended SPLIT=oos_ext
make research-trend-v3-extended SPLIT=full_ext
make compare-trend-v3-extended
```

research 输出到 `reports/research/trend_following_v3_extended/<split>/`，仍生成 `trend_v3_summary.json`、`trend_v3_policy_leaderboard.csv`、`trend_v3_trades.csv`、`trend_v3_report.md`、`data_quality.json` 和 `trend_v3_research_audit.json` 等 V3.0 同结构文件。

compare 输出到 `reports/research/trend_following_v3_extended_compare/`：

- `trend_v3_extended_compare_summary.json`
- `trend_v3_extended_compare_leaderboard.csv`
- `trend_v3_extended_compare_report.md`

compare 只用 `train_ext` / `validation_ext` / `oos_ext` 判定 `stable_candidate`。规则保持保守：三段 no-cost 都必须为正，`oos_ext` cost-aware 不亏，`oos_ext` 回撤不超过 30%，三段交易次数都 >=10，且 OOS 不依赖单一 symbol 或极少数交易。报告还会输出 synthetic funding stress（1/3/5/10 bps per 8h），并明确它不是实际 OKX funding fee。

只有 `stable_candidate_exists=true` 且集中度和 funding 风险可控时，才允许进入 V3.1 research audit；即使出现候选，也不能直接进入 Strategy V3 原型、demo 或 live。若 extended compare 仍无 stable candidate，则停止当前 V3.0 family，不继续扩大参数搜索。

### Funding-aware Trend Research

Funding-aware Trend Research 是离线研究，不是策略开发，不修改 `OkxAdaptiveMhfStrategy`，不新增 demo/live runner，也不连接真实交易或下单。目标是用真实 OKX public funding rate history 评估 Trend V3 extended trades 在永续 funding fee 后的表现。

OKX perpetual funding fee 会显著影响低频趋势跟踪，尤其是多日持仓和跨 funding timestamp 的仓位。REST 数据来源是 OKX public `GET /api/v5/public/funding-rate-history`，本阶段不需要 API key。OKX 文档分页语义是：`before` 返回 requested `fundingTime` 之后的更新记录，`after` 返回 requested `fundingTime` 之前的更旧记录，`limit` 最大 400；downloader 因此使用 `after=<oldest fundingTime>` 做 backward pagination。验证脚本按返回 timestamp 计算实际间隔，不假设固定 8h funding interval，因为部分合约 funding 频率可能调整。

重要限制：OKX REST `funding-rate-history` 只能作为近期 public REST 来源使用，当前审计结果显示 BTC / ETH / SOL / LINK / DOGE 只覆盖 2026-02-06 到 2026-03-31 的 partial 数据，不能覆盖 2023-2026。REST 下载如果输出 `funding_data_complete=false`、`partial_endpoint_limited` 或 `partial_pagination_failed`，这些 actual funding 结果只能作为 available_data_only 审计输出，不能用于策略恢复、V3.1 判断、Strategy V3 开发、demo 或 live。

优先数据源：OKX changelog 记录 2025-09-02 新增 public Historical Market Data query endpoint，用于 batch historical market data，并支持 funding rate module 与 daily/monthly aggregation。由于当前可访问文档未必总能确认正式 endpoint path，本仓库先用 `scripts/probe_okx_historical_market_data.py` 探测；如果 probe 输出 `endpoint_discovery_failed=true` 或 `can_auto_download=false`，downloader 不会猜 URL，也不会伪造 funding 数据。

fallback：OKX Historical market data 页面提供 historical perpetual funding rates from March 2022 onwards。如果 public REST 无法覆盖 2023-2026，且 Historical Market Data query endpoint 不可用或无法确认，才回退到手动 CSV 导入、经过审计的免费外部数据源，或暂停 funding-aware research。手动 CSV 用 `scripts/import_okx_funding_csv.py` 导入到 `data/funding/okx`。导入后必须重新运行 `verify-funding`，只有 funding verify 完整通过后，actual funding analysis 才能用于判断策略是否有研究资格。

运行顺序：

```bash
make download-funding-dry-run
make download-funding
make verify-funding
make analyze-trend-v3-funding
```

REST 不完整时的审计顺序：

```bash
make download-funding
make verify-funding-allow-partial
make analyze-trend-v3-funding
```

优先尝试 historical query endpoint：

```bash
make probe-funding-source
make download-funding-historical-dry-run
make download-funding-historical
make verify-funding
make analyze-trend-v3-funding
```

如果 `probe-funding-source` 报告 `endpoint_available=false` 或 `can_auto_download=false`，不要运行实际 historical download；保持 gates 关闭，并使用下面的手动 CSV 导入路径或暂停研究。

手动 historical CSV 导入示例：

```bash
make import-funding-csv INPUT=/path/to/okx_funding.csv INST_ID=BTC-USDT-SWAP
make import-funding-csv INPUTS=/path/part1.csv,/path/part2.csv INST_ID=BTC-USDT-SWAP
make import-funding-csv INPUT_DIR=/path/okx_funding_downloads INST_ID=BTC-USDT-SWAP
make verify-funding
make analyze-trend-v3-funding
```

`import_okx_funding_csv.py` 支持单文件、多文件、目录匹配同一 `inst_id` 的 CSV，并默认 overwrite 统一输出；需要保留已有 output CSV 时可用 `APPEND=1`，脚本会按 `funding_time` 合并去重并升序排序。导入报告写入 `reports/research/funding/okx_funding_import_report.md` 和 `okx_funding_import_summary.json`。

输出：

- `data/funding/okx/*_funding_2023-01-01_2026-03-31.csv`：原始 funding CSV，已被 `.gitignore` 忽略。
- `reports/research/funding/okx_funding_download_report.md`
- `reports/research/funding/okx_funding_download_requests.csv`
- `reports/research/funding/okx_funding_verify_report.md`
- `reports/research/funding_endpoint_probe/okx_historical_market_data_probe.json`
- `reports/research/funding_endpoint_probe/okx_historical_market_data_probe_report.md`
- `reports/research/funding_historical_download/okx_historical_funding_download_summary.json`
- `reports/research/funding_historical_download/okx_historical_funding_download_report.md`
- `reports/research/funding_historical_download/okx_historical_funding_files.csv`
- `reports/research/trend_following_v3_actual_funding/actual_funding_report.md`
- `reports/research/trend_following_v3_actual_funding/actual_funding_summary.json`
- `reports/research/trend_following_v3_actual_funding/actual_funding_policy_summary.csv`

actual funding 分析同时输出两种口径：`conservative` 将 `abs(notional) * abs(funding_rate)` 全部视作成本；`signed` 按 long/short 与 funding rate 符号估算支付或收取。当前 V3 trade 文件没有 mark price，脚本会使用 `entry_price * volume * contract_size` 近似 funding notional，并在报告中固定写出 warning，不会伪造 mark price。

Funding-adjusted 结果仍不能直接进入 demo/live。只有 funding_data_complete=true、funding-aware 结果在 train_ext / validation_ext / oos_ext 稳定，且原有 V3 extended stable candidate 约束也成立时，才允许进入 research-only V3.1；funding_data_complete=false 时，Strategy V3 / V3.1 / demo / live 全部禁止。

### `make alpha-sweep`

运行保守参数 shortlist sweep：

```bash
make alpha-sweep
make alpha-sweep OUTPUT_DIR=reports/alpha_sweep/manual_001
```

默认以 `config/strategy_sanity_min_size.json` 为 base config，固定 `--max-runs 100 --data-check-strict`。脚本内部会强制保守风控上限，例如 `fixed_size=0.01`、`risk_per_trade<=0.0005`、`max_leverage<=0.5`、`max_notional_ratio<=0.5`、`max_trades_per_day<=10`。

### `make ablation`

运行策略入场过滤 ablation 诊断：

```bash
make ablation START=2025-01-01 END=2026-03-31 OUTPUT_DIR=reports/ablation/main_full
make ablation SPLIT=train OUTPUT_DIR=reports/ablation/train
make ablation SPLIT=validation OUTPUT_DIR=reports/ablation/validation
make ablation SPLIT=oos OUTPUT_DIR=reports/ablation/oos
```

默认使用 `config/strategy_sanity_min_size.json`，固定 `--data-check-strict`，并对每个候选分别运行无成本和成本版回测。`SPLIT=full` 使用 `START/END`；`SPLIT=train|validation|oos` 默认使用脚本内置区间，除非显式传入 `START` 或 `END`。

输出文件：

- `ablation_summary.json`
- `ablation_leaderboard.csv`
- `ablation_report.md`
- `candidates/*/{no_cost,cost}/`：每个候选对应的回测报告目录。

### `make test`

运行全部单元测试：

```bash
make test
```

实际调用 `python -m unittest discover -s tests -p "test_*.py"`。

### `make test-one`

运行单个测试文件：

```bash
make test-one TEST=tests/test_history_time_utils.py
```

`TEST` 为空会报错。Makefile 会把 `tests/test_history_time_utils.py` 转为 `tests.test_history_time_utils` 后交给 `unittest`。

### `make compile`

编译检查：

```bash
make compile
```

实际调用 `python -m compileall scripts strategies tests`。该命令会产生 `__pycache__/`，可用 `make clean-cache` 删除。

### `make clean-cache`

删除 `__pycache__`、`.pytest_cache`、`.mypy_cache`、`.ruff_cache`。

### `make clean-logs`

删除 `logs/` 下的 `.log` 文件，保留 `logs/.gitkeep`。

### `make clean-reports`

默认不会删除：

```bash
make clean-reports
```

必须显式确认：

```bash
make clean-reports CONFIRM=1
```

确认后删除 `reports/` 下生成内容，保留 `reports/.gitkeep`。

### `make tail-log`

查看日志尾部：

```bash
make tail-log
make tail-log LOG_FILE=logs/download_okx_history.log
```

默认查看 `logs/backtest_okx_mhf.log`。可用 `TAIL_LINES=200` 覆盖初始显示行数。

## 参数覆盖示例

```bash
make download-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
make verify-history START=2025-01-01 END=2026-03-31
make backtest START=2025-01-01 END=2026-03-31 OUTPUT_DIR=reports/backtest/manual_cost
make backtest-no-cost OUTPUT_DIR=reports/backtest/manual_no_cost
make analyze-alpha REPORT_DIR=reports/backtest/manual_cost COMPARE_REPORT_DIR=reports/backtest/manual_no_cost
make analyze-trades REPORT_DIR=reports/backtest/manual_no_cost
make backtest-trace START=2025-01-01 END=2025-03-31 OUTPUT_DIR=reports/research/trace_2025q1
make analyze-signals REPORT_DIR=reports/research/trace_2025q1
make research-entry REPORT_DIR=reports/research/trace_train
make research-entry REPORT_DIR=reports/research/trace_validation
make research-entry REPORT_DIR=reports/research/trace_oos
make alpha-sweep OUTPUT_DIR=reports/alpha_sweep/manual_001
make ablation SPLIT=train OUTPUT_DIR=reports/ablation/train
```

覆盖方式统一是 `make 目标 变量=值`。例如要切换 REAL 连接检查：

```bash
make check-okx SERVER=REAL
```

## 回测报告说明

`make backtest`、`make backtest-no-cost`、`make backtest-sanity` 会生成同一套报告文件：

- `warning.txt`：说明 vn.py 标准回测未自动计入 OKX 永续 funding fee。
- `run_config.json`：本次运行参数、策略配置、合约元数据、输出文件路径、数据检查摘要。
- `stats.json`：vn.py 统计结果和脚本补充的交易次数、胜率、破产标记等。
- `diagnostics.json`：资金曲线、破产检测、日级汇总等诊断信息。
- `daily_pnl.csv`：每日盈亏、余额、手续费、滑点、成交数等。
- `trades.csv`：成交明细。
- `orders.csv`：委托明细。
- `chart.html`：回测图表 HTML。headless 环境不会自动打开浏览器，可把文件路径交给已有浏览器打开。
- `signal_trace.csv`：仅在 `--export-signal-trace` 或 `make backtest-trace` 开启时生成，记录候选/entry 信号快照。

## Alpha 诊断说明

`make analyze-alpha` 的核心用途是把成本版和无成本版拆开看：

- 成本版：包含手续费和滑点，更接近常规回测成本。
- 无成本版：`rate=0` 且 `slippage=0`，用于判断是否存在毛 alpha。
- 毛 alpha：无成本版仍无法盈利时，不能把亏损归因于成本。
- 成本拖累：无成本版和成本版的差额，用来衡量手续费、滑点对策略的压制。
- 交易频率：成交密度过高时，成本会吞掉弱 alpha。
- 最差小时/最差日期：用于定位时段过滤或风险控制的优先级。
- 不能只看收益曲线：收益曲线可能掩盖交易频率、成本拖累、局部大亏、样本偶然性和破产风险。

诊断输出默认在 `REPORT_DIR/alpha_diagnostics/`，包括 `alpha_summary.json`、`alpha_diagnostics.md`、月/周/日/时段/方向/持仓时长等 CSV。

## 交易归因诊断

`make analyze-trades` 用在参数 sweep 已经证明没有正收益候选、继续调参没有意义时。它不修改策略交易逻辑，只把一个回测目录里的成交、委托、日级盈亏和统计结果拆开看，定位亏损来自方向、小时、星期、月份、交易频率，还是无成本版本本身就没有毛 alpha。

它和 `make analyze-alpha` 的区别：

- `analyze-alpha` 重点比较成本版和无成本版，回答“有没有毛 alpha、成本拖累多大”。
- `analyze-trades` 重点解释单个报告内部的交易归因，回答“哪些方向、时段、日期、频率桶在亏钱”。
- `analyze-trades` 会在默认命名约定下自动寻找 sibling 成本/无成本报告辅助判断，但主分析仍以 `REPORT_DIR` 为准。

默认输出在 `REPORT_DIR/trade_attribution/`：

- `attribution_summary.json`
- `attribution_by_side.csv`
- `attribution_by_hour.csv`
- `attribution_by_weekday.csv`
- `attribution_by_month.csv`
- `attribution_daily_worst.csv`
- `attribution_frequency_distribution.csv`
- `attribution_report.md`

如果无成本版 `total_net_pnl` 仍为负，报告会标记 `gross_alpha_negative=true`。这种情况下当前策略不能进入 OKX DEMO，也不应继续做扩大仓位或更激进的参数优化；应先回到信号归因、降频过滤和策略假设复核。

## 信号级研究：MFE/MAE 与突破延续性诊断

这个阶段不是为了赚钱，而是验证入场信号是否有预测力。`make backtest-trace` 只开启信号快照导出，不改变策略入场、出场、止损或止盈逻辑；`make analyze-signals` 再用 entry signal 之后的 1m bar 计算未来收益、MFE 和 MAE。

推荐流程：

```bash
make backtest-trace START=2025-01-01 END=2025-03-31 OUTPUT_DIR=reports/research/trace_2025q1
make analyze-signals REPORT_DIR=reports/research/trace_2025q1
```

`signal_trace.csv` 字段包括 `signal_id`、`datetime`、`vt_symbol`、`direction`、`action`、`price`、`close_1m`、`donchian_high`、`donchian_low`、`breakout_distance`、`breakout_distance_atr`、`atr_1m`、`atr_pct`、`rsi`、`fast_ema_5m`、`slow_ema_5m`、`ema_spread`、`ema_spread_pct`、`regime`、`regime_persistence_count`、`hour`、`weekday`、`is_weekend`、`filter_reject_reason`、`position_before`、`volume`、`stop_price`、`take_profit_price`、`trail_stop_price`。当前策略没有的字段会留空，不硬凑。

`make analyze-signals` 默认输出：

- `signal_outcomes.csv`
- `outcome_summary.json`
- `outcome_by_side.csv`
- `outcome_by_hour.csv`
- `outcome_by_weekday.csv`
- `outcome_by_regime.csv`
- `outcome_by_breakout_distance_bucket.csv`
- `outcome_report.md`

解读规则：

- 如果未来 15/30/60m 的 MFE/MAE 没有优势，应重构入场逻辑，而不是继续盲目参数优化。
- 如果信号有 MFE 但最终亏，优先检查出场、止损、止盈和持仓时间逻辑。
- 如果信号没有 MFE，说明入场逻辑本身无效。
- 如果 `future_return_60m` 中位数仍为负，报告会提示 `breakout continuation hypothesis failed`。

## 入场时机研究：Delayed Confirm / Pullback / Breakout Distance Filter

`make research-entry` 和 `scripts/research_entry_policies.py` 用 `signal_trace.csv` 加未来 1m bar 做虚拟 bracket 回放，只输出 no-cost 离线研究结果。它不修改现有策略交易逻辑，也不等于生产策略。

推荐必须按 train / validation / oos 三段分别运行：

```bash
make research-entry REPORT_DIR=reports/research/trace_train
make research-entry REPORT_DIR=reports/research/trace_validation
make research-entry REPORT_DIR=reports/research/trace_oos
```

默认研究这些 policy：

- `immediate_baseline`：按 trace price 立即入场，作为基准。
- `skip_large_breakout_gt_1atr`：跳过 `breakout_distance_atr > 1` 的立即入场。
- `skip_large_breakout_gt_2atr`：跳过 `breakout_distance_atr > 2` 的立即入场。
- `small_to_mid_breakout_0_25_to_1atr`：只保留 `0.25 <= breakout_distance_atr <= 1` 的立即入场。
- `delayed_confirm_1bar`：等待 1 根 1m bar，收盘仍确认突破方向才入场。
- `delayed_confirm_3bar`：等待 3 根 1m bar，第三根收盘仍确认突破方向才入场。
- `pullback_to_breakout_level_5bar`：最多等 5 根 1m bar 回踩原突破位，触及后虚拟入场。
- `pullback_to_breakout_level_10bar`：最多等 10 根 1m bar 回踩原突破位，触及后虚拟入场。
- `momentum_followthrough_3bar`：等待 3 根 1m bar，至少出现 0.25 ATR 顺向推进才入场。
- `avoid_stop_first_profile`：用前 3 根 1m bar 过滤早期 1 ATR adverse-first 的信号。

研究逻辑：

- long / short 分别按方向计算 stop、take profit 和 horizon close 出场。
- 同一根 1m bar 同时触发 stop 和 take profit 时，保守按 stop first。
- 如果 horizon 内没有触发 stop / take profit，则按 horizon 最后一根 close 出场。
- `bracket_grid.csv` 会遍历 `ENTRY_HORIZONS`、`STOP_ATR_GRID`、`TP_ATR_GRID`；`entry_policy_leaderboard.csv` 对每个 policy 取 no-cost `expectancy_r` 最好的组合。

解读规则：

- 必须跨 train / validation / oos 稳定，不能只看单段样本。
- 只有 OOS 仍为正 expectancy，才有资格转成策略逻辑。
- 如果三段没有任何 policy 稳定为正，报告会输出 `entry_policy_hypothesis_failed=true`，应继续复核入场假设，而不是直接把离线 policy 搬进生产。

## Signal Lab：特征研究与跨样本稳定性

Signal Lab 是研究工具，不是生产策略。它只读取 `signal_trace.csv` 和 1m bar，生成特征、标签和统计报告，不修改现有策略交易逻辑。

当前默认特征包括：

- trace 快照特征：`breakout_distance_atr`、`atr_pct`、`ema_spread_pct`、`rsi`、`hour`、`weekday`、`is_weekend`、`direction`、`regime`
- Donchian 结构：`donchian_width_atr`、`close_location_in_donchian`
- 信号前行情：`recent_return_5m`、`recent_return_15m`、`recent_return_30m`、`recent_volatility_30m`、`volume_zscore_30m`
- 信号 bar 形态：`upper_wick_ratio`、`lower_wick_ratio`、`body_ratio`、`range_atr`

推荐必须按 train / validation / oos 三段分别运行，再做跨样本比较：

```bash
make research-features REPORT_DIR=reports/research/trace_train
make research-features REPORT_DIR=reports/research/trace_validation
make research-features REPORT_DIR=reports/research/trace_oos

make compare-features \
  TRAIN_DIR=reports/research/trace_train/signal_feature_research \
  VALIDATION_DIR=reports/research/trace_validation/signal_feature_research \
  OOS_DIR=reports/research/trace_oos/signal_feature_research
```

解读规则：

- 只有跨 train / validation / oos 方向一致的特征，才可以进入下一步策略设计。
- 单一 split 有效不能用，尤其不能用单段 OOS 大尾部结果直接转策略。
- `feature_report.md` 如果输出 `signal_feature_hypothesis_failed=true`，说明单段内没有足够强的特征证据。
- `feature_compare_report.md` 如果输出 `no_stable_feature_edge=true`，当前信号体系应放弃，而不是继续优化 Donchian breakout 参数。

## HTF Signal Research：1h regime + 15m structure + 5m pullback/reclaim

HTF Signal Research 是研究框架，不是生产策略。它用于验证方案 A：1h 判断大方向，15m 确认趋势结构，5m 等待回踩后重新转强，1m 只用于下单价格、止损跟踪和未来 bar outcome 计算，不再产生方向。

Signal Lab 第一阶段发现，高波动、大突破距离、过度延伸、放量和大实体 bar 更像风险特征，而不是可追的毛 alpha。因此 HTF research 默认加入 vol cap 和 no overextension 候选，用来检验过滤“过热/衰竭风险”后，信号质量是否改善。

默认 policy candidates：

- `htf_1h_ema_regime_only`: 1h close/EMA50/EMA200 regime baseline。
- `htf_1h_ema_15m_ema_structure`: 1h regime + 15m close > EMA21 且 EMA21 > EMA55。
- `htf_1h_ema_15m_vwap_structure`: 1h regime + 15m close > rolling VWAP 且 EMA21 > EMA55。
- `htf_1h_ema_15m_donchian_structure`: 1h regime + 15m close > Donchian mid 且 Donchian high slope 非负。
- `htf_1h_15m_structure_with_vol_cap`: EMA structure + 15m ATR_pct/recent volatility percentile <= 0.8。
- `htf_1h_15m_structure_strict_vol_cap`: EMA structure + 15m ATR_pct/recent volatility percentile <= 0.6。
- `htf_1h_15m_structure_no_overextension`: EMA structure + directional recent return、volume z-score、body ratio percentile <= 0.8。
- `htf_1h_15m_structure_5m_pullback_reclaim`: 1h/15m 顺势后，5m 曾回踩接近 15m EMA21 或 VWAP，再重新站上 5m EMA21。
- `htf_1h_15m_structure_5m_pullback_reclaim_vol_cap`: pullback/reclaim + loose vol cap。
- `htf_1h_15m_structure_5m_pullback_reclaim_strict`: pullback/reclaim + strict vol cap + no overextension。

推荐流程：

```bash
make research-htf SPLIT=train
make research-htf SPLIT=validation
make research-htf SPLIT=oos
make compare-htf
```

解读规则：

- 只有 train / validation / oos 都稳定的 policy，才可以进入 `OkxHtfPullbackStrategy` 或 Strategy V2 设计。
- 单一 split 有效必须标记为 overfit risk，不能直接转生产。
- `htf_research_report.md` 如果输出 `htf_signal_hypothesis_failed=true`，说明当前 HTF 假设在该证据标准下仍不成立。
- `htf_compare_report.md` 如果输出 `no_stable_htf_policy=true`，不要进入 demo，也不要把单段有效 policy 写成生产策略。

## 策略 Ablation 实验

`make ablation` 和 `scripts/run_ablation_experiments.py` 用于诊断方向过滤、周末过滤、小时过滤和样本切分后的稳定性，不等于参数优化，也不应直接产出生产参数。

默认候选：

- `baseline`：原始 sanity 配置。
- `long_only`：禁止新开空，只测试多头侧。
- `short_only`：禁止新开多，只测试空头侧。
- `no_weekend`：禁止周六/周日新开仓。
- `weekdays_only`：只允许周一到周五新开仓。
- `no_worst_hours_from_current_report`：屏蔽当前 full sample 交易归因里的最差小时，属于 in-sample diagnostic，不能直接用于实盘。
- `no_weekend_no_worst_hours`：同时禁止周末和当前归因最差小时，仍然属于 in-sample diagnostic。
- `thursday_only`：只允许周四新开仓，属于 sample-mined / high overfit risk。
- `weekday_no_worst_hours`：工作日过滤叠加当前归因最差小时过滤，仍需样本外验证。

解读规则：

- 从 full sample 得到的 `no_weekend`、`no_worst_hours_from_current_report`、`no_weekend_no_worst_hours`、`weekday_no_worst_hours` 不能直接用于实盘。
- 必须同时检查 `train`、`validation`、`oos` 是否方向一致；只看 full sample 容易把噪声当规律。
- 只要 no-cost 仍然为负，就不能进入 OKX DEMO。
- 如果 no-cost 为正但 cost 为负，说明成本拖累或交易频率仍然不可接受。
- 如果 full 为正但 oos 为负，说明过拟合风险很高。
- ablation 结论只能决定下一轮研究重点，不能替代正式训练/验证/样本外流程。

## 防止过拟合的工作流

- 不要直接大规模参数搜索。
- 先固定 `2025-01-01` 到 `2026-03-31` 为主样本。
- 做成本版和无成本版对照，先确认是否有毛 alpha。
- 优先减少交易频率，不要扩大仓位掩盖问题。
- `alpha-sweep` 只做保守 shortlist，不把 sweep 结果直接当最终参数。
- 后续正式优化应再拆 train、validation、out-of-sample。
- 模拟盘前仍要保持最小仓位、保守风控和可回滚流程。

## OKX 模拟盘前置条件

进入 OKX DEMO 模拟盘前，至少满足：

- `make doctor` 通过。
- `make check-okx SERVER=DEMO` 通过。
- 目标区间历史数据完整，`make verify-history` 通过。
- 成本版和无成本版报告都完成。
- Alpha 诊断证明至少存在毛 alpha，并且成本拖累可解释。
- 策略没有用扩大仓位掩盖亏损。

当前如果没有 `scripts/run_cta.py`，不要声称已经支持模拟盘自动运行。下一步需要补 demo runner，再加入 systemd、日志轮转、异常退出重启、只读配置检查和最小下单保护。

## 常见问题

### `.env` 不存在

运行：

```bash
make env
```

然后编辑 `.env`。不要把 `.env` 提交到 Git。

### OKX DEMO/REAL 不匹配

`OKX_SERVER=DEMO` 要配 DEMO API Key；`OKX_SERVER=REAL` 要配 REAL API Key。也可以临时覆盖：

```bash
make check-okx SERVER=DEMO
```

### 代理错误

如果设置了 `OKX_PROXY_HOST`，必须设置正整数 `OKX_PROXY_PORT`。如果不用代理，保持：

```dotenv
OKX_PROXY_HOST=
OKX_PROXY_PORT=0
```

### 历史数据缺口

先验证：

```bash
make verify-history START=2025-01-01 END=2026-03-31
```

再按提示修复，或直接运行：

```bash
make repair-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
```

### `vnpy_sqlite` 未安装

运行：

```bash
make install
make doctor
```

如果仍失败，检查当前 shell 是否已激活 `.venv`，或确认 `PYTHON=.venv/bin/python` 指向正确环境。

### `BacktestingEngine` 没有 funding fee

这是当前回测口径限制。`warning.txt` 和 `run_config.json` 会记录该提醒。正式评估 OKX 永续策略前，需要后续补资金费率数据和资金费成本模型。

### `chart.html` 如何打开

Makefile 不启动 GUI。可以在有浏览器的环境中打开生成的 `chart.html` 文件；在 VPS/headless 上可下载该 HTML 后本地打开。

### `reports/` 和 `logs/` 如何清理

```bash
make clean-logs
make clean-reports CONFIRM=1
make clean-cache
```

`clean-reports` 必须带 `CONFIRM=1`，避免误删回测结果。
