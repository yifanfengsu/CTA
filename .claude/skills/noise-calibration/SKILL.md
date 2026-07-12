---
name: noise-calibration
description: 噪声标定——用 block-shuffle/安慰剂建立假阳性基线，real 指标必须超噪声基线而非超 0。凡研究结论含"检测到 X 个信号/对子/事件/IC 显著"的计数或显著性声明时必须使用。
---

# noise-calibration — 噪声标定（假阳性基线）

## 0. 流水线位置

**第 3–5 站之间**（见 `.claude/skills/PIPELINE.md`）：拟合出"检测到 X 个信号"
之后、单次检验判定之前。
上游输入：第 3 站的拟合产物（信号计数/IC/事件集）。
下游产出：real vs null 的分位与单侧 p → 交第 5 站 gate 判定。

## 1. 何时用

- 凡结论形如"检测到 X 个协整对 / IC = +0.05 显著 / 冲击后回弹 Y%"——**任何计数类
  或显著性类中间结论**，判定前必过噪声标定。
- 大宇宙 × 多因子/多对子的筛查（数百次隐式检验，假阳性是**保证**出现的）。
- 事件研究（冲击/偏离/信号触发后的条件统计）——配 matched 随机时点安慰剂。

**不用会犯什么错（真实案例）**：
- **协整 231 对 ≈ 纯噪声**：pairs 研究 C1 报"平均 12.48 对/窗协整显著"，当时无噪声
  标定；事后算术：231 对 × p<0.05 = **11.55 对/窗的期望假阳性**——"存在性 PASS"
  实际只高出噪声底 8%。这是本项目漏装噪声标定的唯一一次，被 factor_scale 作为
  教训显式修复（出处：`reports/factor_scale_feasibility_20260628/` POSITIONING 段、
  `research/_closed/crypto_perp/pairs_cointegration/reports/pairs_cointegration_20260613/`）。
- **正 IC ≠ 信号**：factor_scale 三因子 real IC 全为"非零"，但噪声标定后只有 REV
  超 NULL 基线（p=0.000），CAR 在 K=50/100 与噪声不可区分——超 0 不是检验，
  超噪声基线才是（出处：factor_scale Q2）。
- **冲击"回弹"= bid-ask bounce**：order_flow 的冲击后回弹曲线与随机时点安慰剂
  几乎重合（39.9% vs 40.2%@5s），半价差地板 0.024bps——没有安慰剂就会把付出的
  点差当成 edge（出处：`reports/order_flow_exhaustion_feasibility_20260628/` Q2）。

## 2. 怎么用

代码入口：本目录 `null_calibration.py`（从 `scripts/research_factor_scale.py` 的
NULL-A/NULL-B 实现提炼；独立可 import）。

1. **选 null 设计**（双 null 是成熟范式）：
   - **NULL-A（主）**：逐序列独立 moving-block bootstrap 打乱收益列、真实信号列
     固定 → 破坏"信号→收益"对齐，保留每列的边际分布 + 短程串行结构。
     入口 `null_ic_panel(score, ret, n_shuffle, block)`。
   - **NULL-B（副）**：整行共享 block bootstrap → 额外保留横截面同期相关。
   - 事件研究用 **matched 随机时点安慰剂**：随机非事件 bin 跑与 real **完全相同**
     的测量管线（order_flow 范式）。
2. **次数**：≥200（factor_scale 用 200；分位数在 p95/p99 层面稳定的最低量级）。
3. **判定读法**：real 超 null 的 **p95**（单侧 p<0.05）才算"超噪声"；报
   `real_percentile`、`p_one_sided`、null 的 p95/p99。
4. **在合成数据上先验证 null 管线本身**：纯噪声 → real 落在 null 内（p 大）；
   植入信号 → p≈0（factor_scale 的 synthetic validation 步骤；
   `null_calibration.py --selftest` 复现）。

## 3. 怎么失效

- **朴素 shuffle 破坏自相关 → 基线本身错**：为什么——逐点打乱把序列打成 iid，
  null 分布过窄，真实的序列相关性被误判为信号；必须用 block-shuffle 保留短程
  结构（factor_scale 用 block=5d 的理由）。
- **null 与 real 的检验管线不完全一致 → 标定失真**：为什么——null 的意义是"同一
  台机器喂噪声输出什么"；管线任何分叉（对齐方式/剔除规则/统计量）都让比较失去
  对照性（order_flow 安慰剂用与冲击完全相同的多偏移测量）。
- **null 次数不足 → 分位不稳**：为什么——p95/p99 是尾分位，200 次下 p99 仅由
  2 个样本决定；对 p99 级判定须加倍次数或只用 p95。
- **只对最终赢家做标定**：为什么——筛查阶段的多重比较已经发生，只标定赢家等于
  把"从 N 个里挑出来"的选择偏差漏掉（那是第 6 站 `multiple-testing/` 的事，
  两者都要做、不可互替）。
- **拿 t 值代替噪声标定**：为什么——t 随样本量机械上升（factor_scale 明确用 IC
  点估计而非 t 判规模效应），且 t 的原假设是"=0"，不是"=同管线噪声输出"。

## 素材出处

- `scripts/research_factor_scale.py`（NULL-A/NULL-B 实现、synthetic validation、block_idx_1d/2d）
- `reports/factor_scale_feasibility_20260628/README.md` Q2
- `research/_closed/crypto_perp/pairs_cointegration/reports/pairs_cointegration_20260613/`（漏装噪声标定的教训，11.55 噪声底算术）
- `reports/order_flow_exhaustion_feasibility_20260628/README.md` Q2（随机时点安慰剂 + 半价差地板）
