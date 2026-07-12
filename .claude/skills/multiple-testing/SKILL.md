---
name: multiple-testing
description: 多重检验打折——deflated Sharpe（Bailey-LdP）+ N 三口径（名义/有效 ENB/扩展下界）+ 验证周期从 bootstrap SE 反推 + FDR。凡"从 N 个配置里选最优"的研究出数字时必须使用（铁律 B）。
---

# multiple-testing — 多重检验打折（deflated Sharpe / N 三口径 / FDR）

## 0. 流水线位置

**第 6 站**（见 `.claude/skills/PIPELINE.md`）。
上游输入：第 2 站开、第 5 站封版的 **trial-ledger**（全部被测配置 + 各自
Sharpe/日收益序列）+ 第 4 站的 bootstrap SE。
下游产出：deflated Sharpe（三 N 口径）+ 打折后验证周期 → 交第 7 站判决。

## 1. 何时用

- 凡"从 N 配置选最优"的研究报数字前（铁律 B：报告必须含名义 N + 有效 N +
  deflated Sharpe）。
- 立项算术（第 4 站）也用本 skill 的验证周期公式——铁律 B 要求打折**前置**到
  开题，不是事后补。
- 多假设并行判定（多因子/多币/多 gate）→ FDR（Benjamini-Hochberg）。

**不用会犯什么错（真实案例）**：
- **幸存者 Sharpe 未打折**：B2_4h 的 0.655 是 15 配置里**最高**的一条——正是
  "best-of-N"选出来的那个。原"15 年验证周期"用组合 Sharpe 0.5 + iid + 未打折，
  数值碰巧落在合理区间但方法是错的；诚实重算：deflated 0.510（有效 N 2.35）/
  0.267（名义 15）/ 0.162（扩展 ≥47），周期 14/51/138 年
  （出处：`reports/trend_methodology_hardening_20260622/` Q2–Q4）。
- **双样本当打折用**：双样本答"另一样本是否成立"，打折答"选择偏差抬高了多少"，
  **互不可替代**（铁律 B 原文；B2_4h 双周期 Binance 复测缓解但不消除选择偏差，
  因 gate 体系在 OKX 上成形）。

## 2. 怎么用

代码入口：本目录 `deflated_sharpe.py`（从 `scripts/research_deflated_sharpe.py`
提炼，Bailey & López de Prado 2014 按原文实现，无自创变体）。

1. **收集 trial-ledger**：全部 N 个配置的日收益序列（缺 → 用冻结引擎重算并做
   保真校验，如原研究的 FIDELITY PASS：重建 vs 冻结 max|差|=$0.000000）。
2. **N 三口径全报**（`n_calibres()`）：
   - 名义 N = trial-ledger 行数；
   - **有效 N（主算）** = ENB = (Σλ)²/Σλ² —— N×N 日收益相关矩阵特征值
     （15 个趋势配置 PC1 独占 61% 方差 → 实为 ≈2.35 个独立试验；主算用有效 N
     防过度惩罚相关试验）；
   - 扩展 N = 下界（"至少测过这么多"，含邻域变体/事后增强），明确标"≥"。
3. **DSR**（`dsr_report()`，全部 DAILY 单位）：
   ```
   SR*₀ = √V·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]    V = Var(全配置日Sharpe, ddof=1)
   PSR(SR*) = Φ[(SR̂−SR*)·√(n−1) / √(1−skew·SR̂+((kurt−1)/4)·SR̂²)]
   deflated Sharpe = SR̂ − SR*₀ ；DSR = PSR(SR*₀)
   ```
   **偏度/峰度用实际分布**（非正态假设；kurt 为非超额口径，正态=3）。
4. **验证周期**（`years_to_sig()`）：T* = T₀·(1.96/t₀)²，t₀ = SR_年化/SE_年化，
   SE 用第 4 站对自相关诚实的 bootstrap SE——**不用 iid 闭式**（与 bootstrap
   自相矛盾）、不用幸存者 Sharpe。
5. **sanity check（唯一"反向即错"）**：N>1 ⇒ SR*₀≥0 ⇒ deflated ≤ 观测。
   违反 = 计算 bug，其余结果方向开放（不预设打折结论）。
6. FDR：多假设 p 值列表 → `fdr_bh()`，报调整后判定。

## 3. 怎么失效

- **N 漏报（trial-ledger 缺失）→ DSR 假**：为什么——SR*₀ 随 N 单调升，漏报 N
  等于少扣"best-of-N 的运气门槛"；中途试过又没入账的配置最常漏。
- **Var(SR) 需要全配置 Sharpe，原始研究未存**：为什么——DSR 的 V 是"试验间
  Sharpe 方差"，只存赢家就算不出；预注册阶段必须落盘全配置产物
  （B2_4h 被迫两年后重算的教训）。
- **名义 N 过罚（配置相关时）**：为什么——15 个高相关趋势配置只等价 2.35 次
  独立尝试，用名义 15 会把门槛抬到 best-of-15 的极值期望，过度惩罚；主算用
  有效 N、名义/扩展作单调敏感性并列。
- **偏度峰度套正态**：为什么——PSR 分母含 skew/kurt 修正项，趋势策略肥尾
  （B2_4h 超额峰度 +4.5）下正态假设会高估 DSR 置信。
- **验证周期用 iid 闭式或幸存者 Sharpe**：为什么——iid 闭式与 bootstrap SE
  自相矛盾；幸存者 Sharpe 未扣选择偏差，周期系统性偏短（9y vs 诚实 14–138y）。
- **"纠错任务自身藏确认偏误"（元教训）**：为什么——做打折的人往往预设"结论
  应该更差"；方法/参数/口径必须在看结果前定死且对反向结论开放（原研究预注册
  "可能强化、可能不变、可能温和松动，数据说什么写什么"，实测有效 N 下 haircut
  仅 −22%，如实报告）。

## 素材出处

- `scripts/research_deflated_sharpe.py`（DSR/ENB/stationary bootstrap/验证周期全实现 + 预注册 docstring）
- `reports/trend_methodology_hardening_20260622/README.md`（N 三口径、DSR 手验明细、Q4 验证周期分解、Q7 观察）
- CLAUDE.md 铁律 B
