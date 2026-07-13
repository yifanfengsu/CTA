# DIAGNOSTIC — 浮点误差量级一次性诊断（2026-07-13 UTC）

**脚本**: `diagnostic_float_error.py`（本目录，只读诊断）·
**复现**: `.venv/bin/python .claude/skills/financial-precision/diagnostic_float_error.py`

## 目的与纪律

把"研究/回测阶段 float64 可用"的依据从**间接证明**（b2_4h_pnl_audit 独立重推
与引擎逐分一致）升级为**显式量级确认**：用 `precision_toolkit.py` 的 Decimal
路径重算 B2_4h（OKX）全期净 P&L，与冻结数字对比，报告差额量级。

纪律（预先写死）：只读已有研究产出；**不修改冻结数字**；若 Decimal 重算与
冻结数字差额 > $0.01 → 停下如实报告，那是需要单独调查的发现。

## 输入（全部只读）

| 文件 | 内容 |
|---|---|
| `trend_baseline_20260611/configs/B2_4h_trades.jsonl` | 470 笔成交明细（entry/exit/size/gross/fee/funding，引擎产物） |
| `b2_4h_pnl_audit_20260628/audit_summary.json` | 冻结数字 `frozen_net = 68194.8186` |
| `b2_4h_pnl_audit_20260628/okx_b2_4h_funding_ledger.jsonl` | 470 笔持仓 / **18,410 次** funding 结算（引擎 + 独立路径双列） |

JSON 数字一律 `parse_float=Decimal`（按文件中十进制字面量精确进入 Decimal 域，
不经 float 中转）。ctVal 取自冻结引擎 `backtest_mr_5m_compare.CONTRACT_SPECS`
（BTC 0.01 / ETH 0.1 / SOL 1 / LINK 1 / DOGE 1000）；fee 口径 taker 0.05%/边双腿。

## 结果

| 路径 | 净 P&L（Decimal） | vs 冻结 $68,194.8186 |
|---|---|---|
| **A** 逐笔 net 列 Decimal 求和（冻结数字同构成，聚合层换 Decimal） | 68,194.8186 | **$0.0000（精确一致）** |
| **B** 三分量恒等式 Σgross+Σfee+Σfunding | 68,194.8183 | −$0.0003（= 分量各自 4 位小数存档舍入，与审计 `identity_residual 0.0003` 一致） |
| **C** toolkit 全重算（gross/fee 从 entry/exit/size/ctVal 用 `pnl()`/`fee()` 重推 + funding 列） | 68,194.8177 | −$0.0009 |

路径 C 的逐笔核对：**470 笔 gross 重推与存档逐笔零差**（max|Δ|=0——公式、
方向符号、ctVal 映射全部验证）；fee 逐笔 max|Δ| = $0.00005（= 存档 4 位小数
舍入地板，与审计报告的同一地板值一致）。

**float64 误差隔离**（同一 1,410 个金额分量，float 裸累加 vs Decimal 精确和）：

| 量 | 值 |
|---|---|
| 纯 float64 累加误差（naive sum） | **$1.4×10⁻¹⁰** |
| `math.fsum` 误差 | $0.0000 |
| funding 独立路径 vs 引擎（18,410 次结算聚合） | $0.00114 |
| Decimal vs float 同一算术回路耗时比（470 笔 gross+fee，200 遍均值） | **7.0×**（0.320ms vs 0.046ms） |

## 结论（写死）

1. **研究阶段 float64 可用的依据 = 本诊断**：本项目量级下（470 笔 / 18,410 次
   结算 / 净额 ~$68k），纯 float64 累加误差 ~1.4×10⁻¹⁰ 美元——比 $0.01 小
   **8 个数量级**，比项目最小金额判定阈值（forward K1 gate $1,296）小
   **~13 个数量级**。上表全部差额（0.0003/0.0009）来自**存档 4 位小数舍入**，
   不是 float 运算误差。
2. 分界的性能面实证：纯算术回路 Decimal 慢 7×；数值密集回测整链（含指标/
   数组运算被迫逐元素化）代价约一个量级或更高——研究阶段强制全链 Decimal
   是纯摩擦、无判定收益。
3. **重跑条件**：未来标的量级/笔数增长 100 倍以上（或出现单笔名义额放大到
   使 ~1e-10 相对误差接近分级阈值的场景），重跑本诊断。实盘/对账域不适用
   本结论——那里强制 Decimal（见 SKILL.md 第 1 段分界）。
4. 冻结数字 **$68,194.82 未被修改**；差额未超 $0.01 停下条款未触发。
