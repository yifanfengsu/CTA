---
name: preregistration
description: 量化研究预注册纪律——gate/阈值/宇宙/OOS 看结果前写死 + trial-ledger 配置入账。任何新研究开题、任何涉及剔除/筛选/过滤的 gate 设计、任何"测多个配置选最优"的实验设计时必须使用。
---

# preregistration — 预注册 + trial-ledger

## 0. 流水线位置

**第 2 站**（见 `.claude/skills/PIPELINE.md`）。
上游输入：第 1 站的机制登记卡（机制/先验/样本偏差三段声明）。
下游产出：预注册文档（gate 全部写死）+ trial-ledger 初版 → 交第 3 站开跑；
trial-ledger 终版（含中途新增配置）→ 交第 6 站 `multiple-testing/` 作 N。

## 1. 何时用

- **任何新研究开题时**（无例外）：判定线/阈值/币种宇宙/成本口径/OOS 纪律，
  全部在看任何结果之前写死并 commit。
- **任何要测 ≥2 个配置的实验**：每个被测配置登记入 trial-ledger。
- **gate 判死后想"复活"死者时**：本 skill 的铁律 A 条款直接适用——禁止。

**不用会犯什么错（真实案例）**：
- **V1→V1′ 事后移动球门**：预注册集中度 gate V1 判死 15/15 后，重定义 gate 为 V1′
  复活 5 个幸存者。无论 V1′ 机制论证多有道理，这是**程序污点而非自我修正**，幸存
  证据等级永久降低（出处：`reports/trend_methodology_hardening_20260622/` Q6①）。
- **15 配置 Sharpe 未落盘**：趋势基线当年只存了净利/PF，没存 15 配置的 Sharpe 与
  日收益序列——两年后做 DSR 打折时 Var(SR) 这个硬前提缺失，被迫用冻结引擎全量重算
  （字节级保真校验后才敢用）。trial-ledger 的"可还原性"要求即由此而来
  （出处：同上 Q0）。
- **差 0.023 不放宽**：1.5× DD gate 差 0.023 未放宽 → 顺藤摸出整条数据污染事故链。
  预注册的价值恰恰在"想放宽的那一刻"（出处：`reports/MR5M_postmortem.md` §8）。

## 2. 怎么用

1. **写预注册文档**：用本目录 `gate_template.md`（从 vrp StageB 的
   `PREREGISTRATION.md` 范式提炼）。必填：口径/gate 数值阈值/**结构要求**
   （如"尾部交易须跨 ≥K 币、双样本均复现"——结构要求与数值阈值同等必须先写死）/
   情景集/裁定规则/样本方向中性核对（gate 0）。
2. **先于结果 commit**：预注册文档单独 commit（如 vrp StageB 的 LOCKED d4597b2
   先于结果），commit hash 写进最终报告——这是"先写死"的可验证证据。
3. **开 trial-ledger**：逐配置一行（配置 id/参数/立项时间），**中途新增的配置也
   逐条追加**，并为每个配置**落盘足够还原 Sharpe/日收益的产物**（不是只存赢家）。
4. **跑完后**：gate 判定照预注册字面执行。字面判定与综合判定分歧时，两者**并列
   报告**、不改门（factor_scale 范式：coded gate PASS 原样报告 + 预注册 FAIL 条款
   综合判死，方向朝保守、全透明）。
5. 预注册 gate 判死后要复活，唯一合法途径 = **另一个事先写好的、不同的 gate**。

## 3. 怎么失效

- **gate 死后重定义复活**：为什么——重定义发生在看到结果之后，选择自由度已泄漏进
  新定义，新 gate 的通过概率被结果污染（V1′ 案例，铁律 A 的由来）。
- **阈值看数据后定**：为什么——阈值成为拟合参数，gate 的假阳性率不再是名义值
  （"看一眼再定门"等价于对着答案画线）。
- **trial 未入账 → N 漏报**：为什么——第 6 站 DSR 的 SR*₀ 随 N 单调上升，N 少报
  则打折不足，deflated Sharpe 系统性虚高（B2_4h 名义 15 vs 扩展 ≥47，deflated
  0.267 vs 0.162——N 直接改变结论量级）。
- **OOS 被迭代 → 失去 OOS 身份**：为什么——每次对 OOS 样本看一眼结果再改配置，
  OOS 就变成第二个训练集；选择偏差无法在同一样本上消除（B2_4h 的 3.4 年 OKX 样本
  被筛查+验证两轮用尽后，只有真前向能产生干净证据）。
- **只预注册数值、不预注册结构要求**：为什么——结构要求（跨币数/双样本复现/
  同号稳定）是防"单点侥幸过门"的主力，漏写则事后补写等价于改门。
- **预注册但不单独 commit**：为什么——没有先于结果的时间戳，"先写死"不可验证，
  纪律退化为口头声明。

## 素材出处

- `vrp/reports/atm_vrp_stageB_premium_truth_20260628/PREREGISTRATION.md`（成熟范式全文）
- `reports/trend_methodology_hardening_20260622/README.md`（V1′ 定性 + Q0 Sharpe 未存教训 + 铁律 A/B 全文）
- `reports/factor_scale_feasibility_20260628/README.md`（字面 gate 与综合判定分歧的透明处理）
- `reports/MR5M_postmortem.md` §8（gate 不可事后修改条款）
- CLAUDE.md 铁律 A
