# trend_b2_4h — 趋势跟踪 B2_4h 线（资源性关闭 + 前向观察中）

**状态**：2026-06-12 资源性关闭（非证伪，验证周期 18mo+ 不可负担，保留重开条件）；
B2_4h 本体在 `forward/` 做零成本前向观察（VPS 运行中）。
关闭文档：`research/_closed/_synthesis/trend_line_closure_20260612.md`；
方法论加固（铁律 A/B/C 来源）：`research/_closed/_synthesis/trend_methodology_hardening_20260622/`。

## ⚠️ 引擎脚本不在本目录（前向冻结区）

B2_4h 不是独立策略文件，而是引擎组合，被 `scripts/forward_b2_4h.py` 逐字 import。
以下文件因前向依赖**冻结于仓库根 `scripts/`**，不得移动/修改；
待前向观察结束、前向系统退役后方可随后续批次迁移入位：

| 冻结文件（在 `scripts/`） | 角色 |
|---|---|
| `research_trend_baseline.py` | tb 引擎（被 forward 逐字 import） |
| `research_trend_validation_r2.py` | r2 引擎（被 forward 逐字 import） |
| `research_trend_validation.py` | tv（r2 的传递依赖） |
| `research_trend_dualcycle.py` | dc（forward 报告分支 lazy import） |
| `research_okx_vs_binance.py` | ramp 检测器（dc 的传递依赖） |
| `backtest_mr_5m_compare.py` | 基准引擎（tb 引用 CONTRACT_SPECS） |
| `binance_funding.py` | Binance funding 解析（forward/dc lazy import） |

本目录 `scripts/` 只收非冻结的增强/审计研究脚本（adx_filter / faster_entry /
funding_confirm / portfolio / vol_targeting / deflated_sharpe——五次增强全部
被同一防御栈判死，结论见各 reports/ 子目录）。

## reports/ 索引

baseline / validation / validation_r2 / dualcycle / portfolio（2026-06-11 主链）、
funding_confirm（INVALID）、adx_filter（INVALID & HARMFUL）、faster_entry（NO
ADVANTAGE）、b2_4h_pnl_audit（PASS）、b2_4h_vol_targeting（NOT ADOPTED）。
