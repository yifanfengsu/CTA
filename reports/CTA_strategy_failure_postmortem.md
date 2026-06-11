# CTA 策略失败全史：13 个月教训总结

> 2024-2026 · 趋势跟踪 → 均值回归 · 方法论、策略方向、回测陷阱的完整复盘

---

## 目录

1. [失败全景图](#1-失败全景图)
2. [第一阶段：13 个月趋势跟踪全败（2024-2025）](#2-第一阶段13-个月趋势跟踪全败)
3. [第二阶段：MR-v1 均值回归的转折（4h → 5min）](#3-第二阶段mr-v1-均值回归的转折)
4. [第三阶段：MR-5m 出口优化全败](#4-第三阶段mr-5m-出口优化全败)
5. [第四阶段：入场过滤器全败](#5-第四阶段入场过滤器全败)
6. [第五阶段：状态熔断器全败](#6-第五阶段状态熔断器全败)
7. [方法论 Bug 清单](#7-方法论-bug-清单)
8. [跨策略模式总结](#8-跨策略模式总结)
9. [四个关键问题回答](#9-四个关键问题回答)

---

## 1. 失败全景图

| 类别 | 总尝试数 | 失败数 | 通过数 |
|------|---------|--------|--------|
| 趋势跟踪策略族 | 10+ 个 | **全部** | 0 |
| 入场过滤器 | 36 组合 | **全部** | 0 |
| 状态熔断器 | 74 配置 | **全部** | 0 |
| 出场优化（Chandelier/追踪止损） | 20+ 变体 | **全部** | 0 |
| 替代通道（Keltner） | 12 组合 | **全部** | 0 |
| **唯一有效**：Donchian fade + Wilder ATR 止损 + midline 止盈 | — | — | ✓ |

---

## 2. 第一阶段：13 个月趋势跟踪全败

**根本假设**：加密货币存在持续性趋势，可以用 Donchian/EMA 突破捕获。

**失败 timeline**：

### 2.1 1m Donchian Breakout
- **尝试**：短周期 Donchian 突破做趋势跟踪
- **结果**：无稳定正收益
- **Signal Lab 诊断**：找到 8 个**稳定负向**风险特征
  - high volatility、high ATR、large breakout distance、high recent return、volume spike、large body ratio
- **根因**：短周期突破不是趋势信号，而是**过热/衰竭信号**
- **教训**：在 5min/15min/30min 上，"突破"更可能被反转

### 2.2 HTF Signal Research（多时间框架）
- **假设**：1h 确定 regime → 15m 结构信号 → 5m 回踩/收复入场
- **结果**：Train/Validation/OOS 全部负收益
- **教训**：多 TF 叠加不增加稳定性，只增加自由度

### 2.3 Trend V2（单品种趋势跟踪）
- **尝试**：单品种 1h/4h 趋势跟踪 family
- **结果**：无稳定候选策略
- **教训**：单品种风险过于集中

### 2.4 Trend V3（多品种组合趋势跟踪）
- **尝试**：5 品种 × 4h/1d × Donchian/EMA/Ensemble policies
- **结果**：无 stable candidate。问题：
  - Top trade concentration：少数交易贡献大部分收益
  - Symbol concentration：单一品种主导
  - OOS fragility：样本外不稳定
  - Choppy/high_vol_choppy 环境亏损严重（占时间 38.7%）

### 2.5 Extended V3（延长数据区间）
- **尝试**：V3 policy set 在 2023-2026 完整数据上重测
- **结果**：stable_candidate_exists=false
- **唯一弱线索**：1d EMA 50/200 crossover 在所有 no-cost 口径上为正
  - 但被 top trade concentration、funding stress、regime diagnostics 拒绝

### 2.6 Regime Diagnostics（趋势 regime 归因）
- **关键发现**：
  - **Strong trend 仅占时间 4.79%**
  - V3 策略利润**不来自** strong trend regime
  - 1d EMA 的 strong-trend 无成本 PnL = **负值**
  - Donchian 亏损集中在 choppy/high_vol_choppy
- **含义**：如果"趋势跟踪"策略不靠"趋势"赚钱，那它在赚什么？答案是随机。

### 2.7 资金费率实际数据验证
- 下载了 OKX 实际历史 funding rate（2023-2026，3558 条/品种）
- **结果**：funding-aware 分析完成后，gates 仍然关闭
- funding_adjusted_stable_candidate_exists=false

### 2.8 外部 Regime Classifier Gate Audit
- 检查分类器过滤是否能拯救 V3.1
- **结果**：无 strict stable candidate
- v3_1d_ema_50_200_atr5 的 OOS top 5% trade contribution = 1.9818（门控阈值 0.8）
- exclude_hostile_chop_overheated 和 exclude_funding_overheated **没有改变 OOS trade set**

### 2.9 Derivatives Data Readiness Audit
- **结果**：数据就绪门控被阻塞
- Funding 数据完整，但 funding alone ≠ derivatives confirmation
- 历史 OI/taker buy-sell/long-short ratio 覆盖未证明
- **无法开展** derivatives-confirmed trend research

### 2.10 VSVCB-v1（正突破假设）
- **假设**：Breakout + squeeze + volume confirmation
- **结果**：Train/Validation/OOS 失败
- 反向测试比正突破假设更强——这是非常糟糕的信号

### 趋势跟踪阶段最终裁决

```
strategy_development_allowed = false
demo_live_allowed = false
current_five_symbol_trend_following_family = ARCHIVED
```

**推荐的下一步**：暂停策略开发，只维护数据和研究工具。

---

## 3. 第二阶段：MR-v1 均值回归的转折

### MR-v1（4h）
- **逻辑**：fade Donchian 突破 → 突破上轨做空 / 下轨做多
- **参数**：LB=8, ATR=1.0x(Wilder), MH=60, 1x 杠杆
- **Phase 1/2/3 结果**（修正前）：
  - Sharpe 3.82, PF 5.38, 36/36 月盈利
- **Stop-Fill Bug 修正后**：
  - PnL 从 +$28,022 → -$2,564
  - 利润几乎全部来自回测定价错误
- **仍通过门控**：中轨止盈提供正收益，逻辑未被推翻

### MR-5m（当前）
- **转向 5min 的原因**：5min 突破反转率 87-95% vs 4h 78-82%
  - 同一逻辑在更细粒度上产生 **35× 更好 PnL**（maker 费率）
- **当前回测**（2023-2026，maker 费率）：
  - Sharpe 5.83, PF 1.74, OOS > Train, Max DD -36.3%
  - 中轨止盈 +$334k（37% 交易） vs 止损 -$192k（63% 交易）→ PF=1.74 完全由 midline 贡献
- **实盘**（5/31-6/6，683 笔）：净 +$63，midline 唯一正收益

---

## 4. 第三阶段：MR-5m 出口优化全败

**研究脚本**：`scripts/research_mr_v1_chandelier.py`

### 4.1 Chandelier Exit（烛台止损）
- **变体数**：20+（Chandelier mult × MH 组合）
- **参数空间**：Chandelier mult [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] × max_hold [12, 20, 30, 40, 48]
- **结果**：**全部失败**
- **根因**：追踪止损过早退出，截断 midline 利润但不减少 stop 亏损
- **证据脚本**：`scripts/research_mr_v1_chandelier.py`

### 4.2 Trailing Stop（移动止损）
- **结果**：全部失败
- **教训**：midline 止盈是**唯一有效的出场机制**

### 结论（写死在 memory 里）：
> "中轨止盈唯一有效(+$6.3k), Chandelier/追踪止损全部失败, 不再碰 exit 优化"

---

## 5. 第四阶段：入场过滤器全败

### 5.1 ADX + EMA Slope 双重过滤器
**研究脚本**：`scripts/research_mr_v1_filter.py`

| 参数 | 值 |
|------|-----|
| adx_period | [10, 14] |
| adx_threshold | [25, 28, 32] |
| ema_period | [120, 168] |
| slope_floor | [0.0005, 0.001] |
| 组合数 | 24 |

**结果**：全部劣于基线。

| 排名 | PnL | Trades | 说明 |
|------|-----|--------|------|
| 基线无过滤 | +$4,048 | 2,216 | Stop -$36k, Mid +$40k |
| 最优过滤器 | +$533 | 346 | Stop -$5k, Mid +$5.5k |
| 最差过滤器 | -$1,094 | 549 | |

**根因**：杀敌一千自损八百。
- Stop 损失 -$36k → -$5k（降 86%） ✓
- Midline 利润 +$40k → +$5.5k（也降 86%） ✗
- 斜率过滤器无法区分"该过滤的坏信号"和"不该过滤的好信号"

**教训**：**Entry filters don't work for mean reversion。** MR 利润来自大量交易中的少数大赢家——过滤器同时筛掉赢家和输家。

### 5.2 Keltner Channel 替代 Donchian
**研究脚本**：`scripts/research_mr_v2_keltner.py`

| 参数 | 值 |
|------|-----|
| ema_period | [20, 30] |
| atr_period | [14, 20] |
| atr_mult | [1.5, 2.0, 2.5] |
| 组合数 | 12 |

**结果**：**全部 PnL 为负。最优 -$992（vs Donchian 基线 +$4,048）。**

**根因**：KC 自适应波动率在 fade 策略中是劣势。
- 暴跌时通道扩张 → 入场点太远
- 横盘时通道收缩 → 假信号太多
- Donchian 的"笨拙"是优势

---

## 6. 第五阶段：状态熔断器全败

**假设**：连续止损后暂停交易，或滚动 PF 低于阈值暂停。

**配置数**：74 个组合。

**结果**：**全部负收益。**

**根因**：
- 失效月总亏损仅 -$10,690（占总盈利 ~3%）
- 失效月与健康月共用信号
- 事后暂停误伤的健康利润 > 省下的亏损

**关键教训**：
> "靠少交易让指标变好"(looking better by trading less) 是陷阱。

---

## 7. 方法论 Bug 清单

### Bug 1: Stop-Fill Look-Ahead
- **错误**：`exit_price = bar["open"]` 作为止损成交价
- **修正**：`min(bar["open"], stop_price)` for longs
- **影响**：MR-v1 PnL 从 +$28,022 → -$2,564（**12× 利润虚增**）
- **教训**：止损必须 fill at stop_price

### Bug 2: Max DD 分母错误
- **错误**：`(equity - peak) / NOTIONAL` 用 $1,000 notional
- **修正**：`(equity - peak) / peak × 100`
- **影响**：MR-5m DD 修正前 -737% → 修正后 -36.3%

### Bug 3: Intrabar Path Dependency
- **错误**：先更新极值再检查止损 → 前视偏差
- **修正**：先检查止损（用前一极值），再更新极值

### Bug 4: Global vs Local Midline
- **错误**：Midline 预计算用全局 DataFrame → 前视泄露
- **修正**：在每个 bar 时间点只使用截至该时刻的数据

### Bug 5: Calmar Ratio 单位不匹配
- numerator（年化收益率/$notional）和 denominator（max DD%）必须同基

---

## 8. 跨策略模式总结

| 模式 | 含义 | 确认次数 |
|------|------|---------|
| Fixed > Adaptive | 固定参数优于自适应参数 | 5× |
| Entry Filters Don't Work For MR | 过滤同时筛掉赢家和输家 | ADX + EMA 24 组合 |
| "Looking better by trading less" is a trap | 通过少交易改善指标是假象 | 熔断器 74 配置 |
| OOS > Train is anomalous | 不是过拟合信号，说明 OOS 窗口特殊 | MR-5m |
| 5min Alpha > 4h Alpha | 均值回归在高频上更有效 | 35× better PnL |
| Midline止盈唯一有效 | 所有替代出场方式全败 | 20+ 变体 |
| Donchian > Keltner for fade | 固定通道的"笨拙"是优势 | 12 组合全负 |
| SOL 是 episodic 非结构性 | 早期集中但 2024+ 分散 | 诊断结论 |
| Underwater Concentrated | 水下 60.8% 但 70% 浅水 | MR-5m |

---

## 9. 四个关键问题回答

### Q1: "失败"具体是什么样？

**不是一种失败，是五种不同的失败模式：**

| 失败模式 | 具体表现 | 出现位置 | 对应修法 |
|----------|---------|---------|---------|
| **回测就亏** | PnL 为负，无正收益 | Keltner(12/12)、VSVCB-v1、HTF Research | 放弃方向 |
| **回测赚但 unstable** | Train 正 OOS 负，no stable candidate | Trend V3 全部 family | 不是参数问题是假设问题 |
| **回测赚但 concentration kill** | Top 5% trades > 80% PnL，单一品种 > 50% | V3 1d EMA | 不能实盘，尾部风险不可控 |
| **回测赚但方法论 bug** | Stop-fill bug 虚增 12× 利润 | MR-v1 Phase 3 | 修正后重新评估 |
| **回测赚实盘也赚但幅度不匹配** | 实盘 +$63 vs 回测预期 $142k | 当前 MR-5m | ① maker vs taker 费率（回测 maker +$142k，实盘 taker）② 采样率不足（683 笔 vs ~50k 笔）|

**当前 MR-5m 处于哪种状态？**
- 回测（maker）：Sharpe 5.83, PF 1.74 → **回测过关**
- 实盘（taker）：+$63 / 7 天 → 样本太小，**不能下结论**
- 但实盘已验证核心结构：midline 是唯一赚钱出口，和回测完全一致
- 最大不确定性：taker 费率下长期表现如何（maker → taker 差异 ~0.07%/笔）

### Q2: 还做 BTCUSDT 5m 吗？换标的/周期？

**数据说话：**

| 品种 | MR-5m 全周期 PF | 利润贡献 | 判断 |
|------|----------------|---------|------|
| SOL | **最强** | 54.4% | 保留，但 episodic（2024+ 降到 23-31%） |
| DOGE | **1.67（第二强）** | — | 保留，近年走强，不能因为一周亏损踢掉 |
| BTC | 中等 | — | 保留，虽然不是最强但也不拖后腿 |
| ETH | 中等 | — | 保留 |
| LINK | 中等 | — | 保留 |

**结论**：5 品种全保留。踢掉 DOGE 会损失 $78k 历史利润而 PF 几乎不变（1.68→1.69）。

**周期**：5min 确实是更好选择。5min 突破反转率 87-95%，4h 只有 78-82%。同一逻辑在 5min 产生 35× 更好 PnL。

**是否换标的/周期？不换。** 当前配置经过数据验证，没有换的理由。BTC 虽然不是最强，但也不该踢掉——多品种分散本身就是风控。

### Q3: 你的 edge 假设是什么？

**当前 edge 假设（MR-5m）：**
> 加密货币 5min 级别的 Donchian 通道突破后，价格倾向于均值回归（回归中轨），而非继续趋势。

**为什么这个假设可能成立：**
1. 5min 级别是噪声交易者的主战场 → 突破往往是过度反应
2. 加密货币缺乏基本面锚 → 短期价格运动过度后回归更显著
3. 5min 突破反转率 87-95% → 频率上确实在回归
4. 跨品种有效（5/5 品种全周期 PF > 1）
5. 跨时间有效（OOS 不差于 Train）

**弱点和风险：**
1. "什么在回归"不够精确——是回归中轨（Donchian midline），中轨本身在移动
2. 没有微观结构支撑（无 OI/taker/long-short 验证）
3. Taker 费率下 maker 返佣优势消失 → 盈亏比从 1.74 可能降到接近 1.0
4. 2024 是全市场逆风年（所有币种 PF 最弱）→ 存在系统性失效周期

**之前趋势跟踪为什么失败的假设诊断：**
- Strong trend 仅占 4.79% 时间 → 如果 edge 是"捕获趋势"，那 95% 时间在亏
- Choppy/high_vol_choppy 占 38.7% → Donchian 突破在这些环境全是假信号
- 趋势跟踪策略不靠趋势赚钱（regime 归因证明）→ 策略在随机游走

### Q4: 你能接受的真实预期是什么？

**诚实的答案：**

当前回测 **Sharpe 5.83** 是**不可持续的**。原因：
1. Maker 费率优势（-0.02% vs taker 0.05%）→ 实盘正在用 taker
2. OOS 窗口只有 9 个月（2025H2-2026Q1）→ 太短
3. Sharpe 5.83 在传统 CTA 中极其罕见 → 大概率有未被发现的回测偏差

**你应该预期的现实区间：**

| 指标 | 当前回测 | 保守估计 | 说明 |
|------|---------|---------|------|
| 年化收益 | ~8,500% | 15-30% | $500 起点太小，% 无意义；用 $5k 本金的绝对收益算 |
| Sharpe | 5.83 | **0.5-1.5** | 1 以上就是很好的 CTA |
| Max DD | -36.3% | **-20% 到 -40%** | 当前已在这个区间 |
| PF | 1.74 | **1.1-1.3** | taker 费率会吃掉大量利润 |
| 月胜率 | ~60% | **50-65%** | 别期望 100% |

**关键提醒**：
- 月化 10% 不回调 = 年化 214% + 零回撤 = **不存在**
- 夏普 1 + 年化 15% + 回撤 20% = 已经是非常好的 CTA
- 如果你心里的目标是"每月赚钱、不亏钱"，这个目标本身会逼你过拟合
- 接受回撤不是弱点，是策略的呼吸。MR-5m 水下 60.8% 天数，但 70% 是浅水 (0-5%)
- **最重要的**：失效月只占总盈利 3%，空间在健康月份的 97%——不要为了抹平 3% 的亏损而破坏 97% 的利润

---

## 附录：所有失败证据索引

| 失败方向 | 研究脚本 | 日志/报告 |
|----------|---------|----------|
| ADX + EMA 过滤 | `scripts/research_mr_v1_filter.py` | `logs/research_mr_v1_filter.log` |
| Keltner Channel | `scripts/research_mr_v2_keltner.py` | `logs/research_mr_v2_keltner.log` |
| Chandelier Exit | `scripts/research_mr_v1_chandelier.py` | `logs/research_mr_v1_chandelier.log` |
| Midline Exit 研究 | `scripts/research_mr_v1_midline.py` | `logs/research_mr_v1_midline.log` |
| Phase 2 验证 | `scripts/research_mr_v1_phase2.py` | `logs/research_mr_v1_phase2.log` |
| Phase 3 验证 | — | `logs/research_mr_v1_phase3.log` |
| MR-5m 深度研究 | `scripts/research_mr_5m.py` | `logs/research_mr_5m.log` |
| MR-5m 深度 v3 | `scripts/research_mr_5m_deep_v3.py` | — |
| 趋势跟踪研究决策档案 | — | `reports/research/research_decision_dossier/research_decision_dossier.md` |

---

*生成时间：2026-06-07 · 数据截至 OKX 2023-01 ~ 2026-05 1m K线 · 5 品种 (BTC/ETH/SOL/LINK/DOGE)*
