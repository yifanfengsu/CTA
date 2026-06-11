# MR-5m 项目复盘：一个建立在模拟盘行情上的假 edge

> 2026-06-11 · 数据污染 → 取证 → mainnet 重验 → 项目关闭的完整记录
> 姊妹篇：`CTA_strategy_failure_postmortem.md`（策略方向层面的 13 个月复盘，2026-06-07，其全部数字同样基于污染数据，见本文第 7 节）

---

## 1. 一句话结论

MR-5m（5m Donchian fade）在 OKX 真实数据上无 edge（test PF 0.826、毛利≈0、净亏 100% 为手续费）；13 个月研究的全部绩效结论建立在 OKX DEMO 模拟盘行情上，demo 微观结构噪声把一个无信息的信号美化成了 PF 2.06 的假 edge。

---

## 2. 时间线（关键节点）

| 时间 | 节点 | 证据 |
|------|------|------|
| 2026-04-17 15:36Z | `.env` 设 `OKX_SERVER=DEMO`（最早可证时点；`.env.example` 模板默认值即 DEMO，更早即如此的可能性高） | .env mtime |
| 2026-04-17 15:39Z | 首次有日志的历史数据下载会话，`Server: DEMO` | `logs/download_okx_history.log` |
| 2026-05-07 → 05-09 | 5 币 2023-2024 全量重下（每币 244 chunks），覆盖库内全部历史——**自此整库 100% demo** | manifests + 取证 |
| 2026-06-01 | 双账号模拟盘对照实验启动（VPS，DYN vs FLAT） | PROJECT_GUIDE 时间线 |
| 2026-06-01 → 06-08 | 执行异常修复（GTC→IOC、force_close、fee 串号、exit latch 等 7+ bug）+ OKX bills 对账体系建立 | git log |
| 2026-06-09 | ATR filter OOS 验证（撤回"−72% DD"叙述） | `reports/regime/atr_filter_oos_validation_20260609/` |
| 2026-06-09 | atr_ratio 拆解（DYN 押对结果用错原因） | `reports/regime/atr_ratio_decomposition_20260609/` |
| 2026-06-09 | DYN-v2 设计 + 组合层风控 R1/R2/R3 证否 + size cap V1 证否 | `dyn_v2_design / portfolio_risk_phase_2a / dyn_v2b_size_cap_v1` |
| 2026-06-10 | **v2B DD 诊断：1.5× DD gate 拒绝放宽 → 追到单 bar → 发现 598 个数学上不可能的合成 ramp** | `reports/regime/v2b_dd_diagnosis_20260610/` |
| 2026-06-10 | 取证确认：整库（5×1,791,360 根）100% 经 gateway-DEMO 写入，H1∧H2∧H3 全成立 | `reports/regime/data_contamination_forensics_20260610/` |
| 2026-06-10 | mainnet 重建：3.4 年 1m 数据零缺口，带来源元数据 | `reports/regime/mainnet_rebuild_20260610/` |
| 2026-06-10 | DEMO vs mainnet 逐 bar 对比：205 个币种-月格子 197+ red（97%），信号 Jaccard 0.34-0.55 | `reports/regime/demo_vs_mainnet_comparison_20260610/` |
| 2026-06-11 | **基线重验：FLAT 在 mainnet 无 edge（三配置全亏，毛利≈0）→ 项目关闭** | `reports/regime/mr5m_mainnet_baseline_20260611/` |

从发现第一个异常（DD 诊断）到项目关闭：**2 天**。

---

## 3. 根因分析（分层）

**直接根因**：`.env` 中 `OKX_SERVER=DEMO`；`download_okx_history.py` 的默认主路径（vnpy_okx gateway）继承该配置；vnpy_okx 在 DEMO 模式下给**所有**请求（含公共行情 history-candles）加 `x-simulated-trading: 1`——返回的是 OKX 模拟盘环境行情。另两个增量写库脚本（`download_history.py` / `merge_recent.py`）则直接**硬编码**了该 header。仓库内不存在任何确定走 mainnet 的写库路径曾被执行过（唯一干净的 REST 备胎从未触发）。

**结构根因**：配置错误发生后，没有任何一层能让它显形——
- 数据库 `dbbardata` 无来源列（`gateway_name` 不落库）；
- manifest 记录了 `source_used=gateway`，但**没有 DEMO/REAL 字段**；
- 下载脚本不向用户打印数据环境（server 值只埋在结构化日志的 connect 事件里，事后取证才翻出来）。
一个单点配置错误穿透了三层本应拦住它的记录设施。

**流程根因**：历史数据从未与任何外部源做过抽样交叉验证。13 个月里，"回测赚钱吗""过拟合了吗""执行偏差多大"都被反复挑战过，唯独"**数据是真的吗**"这个更基础的问题从未被问出口。数据被当成了公理。

**为什么 13 个月才发现**：demo 行情不是乱码，是"同步但带噪"的镜像——价格中位偏差仅 0.003-0.03%，月度走势同向，宏观形态足够像真的。任何月线/日线级别的肉眼检查都会通过。它只在两处露馅：单 bar 级别的数学不可能形态（等差阶梯、单根 ±91% 复原），和策略敏感量级的系统性偏置（ATR、信号微观结构）。直到 v2B 的 DD 诊断把单日回撤拆到逐 bar，第一处才暴露；直到逐 bar 对比，第二处才量化。

---

## 4. demo 假象的机制（供未来识别同类问题）

来源：`demo_vs_mainnet_comparison_20260610/` 与 `mr5m_mainnet_baseline_20260611/demo_attribution.md`。

| 机制 | 量级 | 对策略的作用 |
|------|------|------|
| 单笔质量虚增 | 单笔均值 demo +$8.88 vs mainnet −$1.09；胜率 48.3% vs 36.1% | 把毛利≈0 的信号变成 PF 2.06 |
| ATR 系统性偏高 | BTC 中位 1.356×（90% 的 bar demo>main） | 止损更宽、ATR 过滤通过集不同、atr_ratio 分档失真 |
| 信号集偏离 | Jaccard 0.34-0.55；mainnet 信号召回仅 43-66% | 一半"研究过的信号"在真实市场不存在 |
| 出场系统性美化 | 重合信号抽样：5-15% 出场原因不同，PnL 差符号均值 5/5 币为正 | demo 的 V 形瞬间复原送 midline 止盈 |
| 成交量虚构 | demo/main 量比中位 6.6-20× | 任何 volume 相关研究无效 |
| 极端合成 ramp | 598 个事件（SOL 344/DOGE 216/BTC 0），70% 4h 内完全复原 | 双向污染 net 与 DD（通常送钱，2025-05-29 例外打穿） |
| 时变性 | 2025-07→2026-03 持续价位脱锚（周级 ±1%+）；2026 年 Jaccard 仅 0.17-0.31 | 越近的数据越假——"近期表现"判断最受害 |

识别特征（任何一条命中即应停下查数据）：胜率/单笔均值在新数据源上不连续地跳变；ATR 类指标与外部源系统性偏离；同时刻其他标的纹丝不动的单标的极端行情；等差步长价格序列；完美往返。

---

## 5. 什么救了我们（与失败同等重要）

1. **1.5× DD 硬 gate 拒绝放宽。** v2B 的 DD 比值 1.523× 离门槛只差 0.023，"放宽到 1.6"在当时看起来完全合理。没放。由此触发的 DD 诊断是整条发现链的第一张多米诺。
2. **诊断任务允许 A/B/C 之外的"D 选项"。** DD 诊断的问题设计预设了三种市场解释，但任务允许"以上都不是"——数据 artifact 因此得以被命名，而不是被硬塞进最接近的市场叙事。
3. **取证任务只认证据。** H1（代码路径存在）∧ H2（历史上被触发）∧ H3（与污染特征吻合）三段论逐项核实，根因坐实为可复查的事实链，而非"大概率是 demo"的猜测。后续 mainnet 重建的全部投入建立在这个确定性上。
4. **全部发现发生在 DEMO 阶段、真金入场之前。** 双账号实验、执行修复、对账体系——所有"为真金做准备"的工程都完成了，但真金始终没进场。损失为零（不计时间）。

结论：研究纪律的作用不是保证结论正确——13 个月里每一步的 train/test 切分、OOS 验证、walk-forward 都按规矩做了，结论照样全错，因为数据是假的。**纪律的真正作用是保证错误结论活不到真金那天。** 这一次它做到了。

---

## 6. 资产清单（项目留下了什么）

| 类别 | 资产 | 位置 |
|------|------|------|
| 数据 | 干净的 3.4 年 mainnet 1m 库（5 币零缺口，带 `download_meta` 来源元数据 + server=MAINNET manifest） | `.vntrader/database_mainnet.db` |
| 数据 | 已确认污染的 demo 库（取证基准，研究禁用） | `.vntrader/database.db` |
| 工程 | mainnet 下载器（硬编码环境、断点续传、manifest 带环境字段） | `scripts/download_mainnet_history.py` |
| 工程 | 实战验证过的执行栈：IOC 下单、force_close、position verify、OKX bills 对账、exit latch | `scripts/run_mr_5m_direct.py`（归档） |
| 工程 | 回测引擎（费用/成交/指标口径与实盘 1:1） | `scripts/backtest_mr_5m_compare.py` |
| 方法论 | train/test 切分纪律、五项诚实性检验、walk-forward 流程、误伤（虚拟 PnL）分析、gate 不事后修改、PROJECT_GUIDE 外科手术维护规范 | 全部报告中反复执行 |
| 方法论 | 数据取证三段论（H1∧H2∧H3）与逐 bar 对比/可信度地图工具 | `scripts/forensics_data_contamination.py` / `research_demo_vs_mainnet.py` |
| 档案 | 70+ 研究报告目录 + 本次 5 份数据事故档案 | `reports/` |

---

## 7. 作废范围声明

1. **全部绩效数字作废**：所有基于 `database.db` 的 PF / 净利 / DD / 胜率 / 阈值 / 分位，无一例外。包括 FLAT PF 2.06、v2B +$290k、DOGE PF 1.67、ATR p30/p40 阈值。
2. **全部"死胡同"结论同样作废**：这是容易被忽略的一半。趋势跟踪 10+ 族全败、入场过滤器 36 组合全败、Keltner 12 组合全负、熔断器 74 配置全负——这些**否定性结论也是在污染数据上得出的**。demo 噪声在美化均值回归的同时（假 V 形复原送 midline），恰恰系统性摧毁趋势策略（假反转打断趋势持仓）。这些方向在干净数据上的真实表现未知。`CTA_strategy_failure_postmortem.md` 全文数字随之作废，仅其方法论 Bug 清单（第 7 节）仍然有效——那些是回测代码缺陷，与数据无关。
3. **机制性直觉降级为假设**："whipsaw 危害 MR""midline 是唯一有效出场""Fixed > Adaptive"等，从"已确认模式"降级为"待干净数据重验的假设"。它们可能仍然成立（部分有市场结构逻辑支撑），但目前没有任何干净证据。
4. **不作废的**：执行层工程（IOC/对账/force_close 是在真实 OKX 模拟撮合上验证的，与行情数据污染无关）、方法论流程、本次事故链的全部取证结论。

---

## 8. 新研究启动前的强制检查清单（本复盘最重要的输出）

任何基于 mainnet 数据的新策略研究，启动前逐项确认：

- [ ] **数据源验证**：任何写库脚本启动时必须向 stdout 打印数据环境（MAINNET/DEMO）；manifest 必须含 `server` 字段；新数据入库后、用于研究前，与至少一个独立外部源抽样交叉验证（如 Binance vision 月度文件，抽 ≥3 个随机日逐 bar 对比方向与量级——注意 Binance 是不同交易所价格，验的是"同一个市场"，不是逐 tick 相等）。
- [ ] **回测假设显式化**：手续费率、滑点假设、成交假设（maker/taker、成交率）写进每份报告头部，不许埋在代码里。
- [ ] **基线优先**：任何策略想法先跑"无过滤、固定仓位"裸基线；**毛利为零或为负的信号直接放弃**，不进入任何过滤/仓位/出场优化——优化器只能放大 edge，不能创造 edge（本项目用 13 个月验证了这句话的代价）。
- [ ] **gate 不可在结果出来后修改**：阈值在跑之前写下；差 0.023 也不放宽（1.5× DD gate 的 0.023 就是本次发现链的起点）。
- [ ] **否定性结论与肯定性结论同等对待**：同等的证据标准、同等的文档固化（进 PROJECT_GUIDE"已验证的核心事实"），防止死胡同被反复重走，也防止假死胡同永久封路（见第 7 节第 2 条的教训：封路的证据也可能是假的）。

---

*生成时间：2026-06-11 · 事故档案：reports/regime/{v2b_dd_diagnosis,data_contamination_forensics,mainnet_rebuild,demo_vs_mainnet_comparison}_20260610/ + mr5m_mainnet_baseline_20260611/*
