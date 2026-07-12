---
name: bootstrap-inference
description: block/stationary bootstrap 误差棒与下界——先测 ACF 再选 iid/block、块长多档披露、下界稳定性判 gate 稳不稳。任何误差棒/置信区间/5% 下界/验证周期计算时必须使用。
---

# bootstrap-inference — 诚实误差棒（block bootstrap）

## 0. 流水线位置

**第 4 站（误差棒与可行性）与第 6 站（多重检验的 SE 输入）**
（见 `.claude/skills/PIPELINE.md`）。
上游输入：第 3 站的收益/PnL 序列。
下游产出：SE / CI / 多档块长 5% 下界 → 交第 4 站验证周期算术、第 6 站 DSR、
第 7 站生存判定。

## 1. 何时用

- 任何 Sharpe/均值 的 SE 与置信区间（验证周期反推的分母）。
- 任何"多年平均 P&L 的 5% 下界"类生存判定（VRP StageB 的 B-gate 主件）。
- 决定 iid 还是 block **之前**——先跑本 skill 的 ACF 检查，**该测不该假设**。

**不用会犯什么错（真实案例）**：
- **iid 直接套自相关序列**：低估 SE → 下界过窄 → 假显著。反向教训同样真实：
  B2_4h 任务预设"196h 持仓 → 日收益自相关 → block 会拉长周期"，实测日 M2M
  ACF lag1-5 全 ≈0（持仓期内恒定仓位不制造收益自相关），block/iid SE 比 0.979
  ——**方向由数据裁决，预设两个方向都可能错**
  （出处：`reports/trend_methodology_hardening_20260622/` Q4）。
- **单一块长出下界**：VRP StageB 预注册 block∈{1,3,6 月}三档披露，规定"下界
  正负若依赖块长 → 判 gate 不稳"。实测三档全负（−0.61/−0.52/−0.48）= 稳定地
  "区分不出零"——若只跑一档就没有这层稳定性证据
  （出处：`vrp/reports/atm_vrp_stageB_premium_truth_20260628/`）。

## 2. 怎么用

代码入口：本目录 `block_bootstrap.py`（stationary bootstrap 从
`scripts/research_deflated_sharpe.py` 提炼，moving-block 从
`scripts/research_factor_scale.py` 提炼，多档下界范式从 vrp StageB 提炼）。

1. **先测 ACF**：`acf_check(series, nlags=5)`。全部 |ACF|≈0（如 <0.05 量级且
   无系统符号）⇒ iid 够用，block 修正 immaterial；否则 block 必须。
   把 ACF 数字写进报告（这是"测了"的证据）。
2. **选块长（经济理由，不对样本内挑）**：主档 = 经济尺度（如
   round(中位持仓小时/24)=8d；月度周期数据用 1 月）；披露档 = {短/主/长}
   （如 {4,8,16}d 或 {1,3,6} 月）。
3. **stationary bootstrap**（Politis-Romano，几何块长）：
   `stationary_bootstrap_stat(series, stat_fn, L, B=10000, seed)` → SE/CI/分位。
   L=1 即 iid 参照，inflation = SE_block/SE_iid 必报。
4. **多档下界披露**：`multi_block_lower_bound(series, stat_fn, Ls, q=0.05)` →
   每档 5%（可加 1%）下界 + **同号稳定性**布尔。gate 用下界时三档必须同号，
   异号 = gate 不稳（预注册裁定，非事后解释）。
5. **验证周期**：SE 交给 `multiple-testing/deflated_sharpe.years_to_sig`
   （打折后 Sharpe + 本 SE）。

## 3. 怎么失效

- **块长不足 → 低估自相关 → 下界过乐观**：为什么——短块斩断长程依赖，重排样本
  比真实历史"更独立"，方差被低估；块长须 ≥ 依赖的特征尺度（持仓期/结算周期）。
- **iid 套自相关序列**：为什么——正自相关下有效样本量 < 名义 n，iid SE ∝ 1/√n
  偏小，CI 假窄（经典失效；本项目实测 B2_4h 恰好不踩，但那是测出来的，不是
  假设出来的）。
- **bootstrap 只能重组历史 → 抽不出比样本更坏的尾部（peso）**：为什么——重采样
  的支撑集 = 已实现样本；样本恰好缺牙（无 312 级尾部）时任何分位都过乐观。
  **必须配前瞻情景注入**（历史最坏×2 / 反向极端 / whipsaw——见
  `honest-verdict/` 与 vrp StageB 情景集），bootstrap 下界与情景注入是互补件
  不是替代件。
- **对块长挑最优**：为什么——块长成为拟合参数，等价于对样本内挑"下界最好看"
  的档（与 vol-targeting lookback 描线同构）；主档按经济理由预注册，其余档
  只作稳定性披露。
- **B 次数不足**：为什么——5%/1% 下界是尾分位，B=1000 时 1% 分位仅 10 个点
  支撑；下界类判定用 B≥10,000。

## 素材出处

- `scripts/research_deflated_sharpe.py`（stationary bootstrap 实现、ACF 直接验证、inflation 口径）
- `reports/trend_methodology_hardening_20260622/README.md` Q4（"该测不该假设"的反预设发现）
- `vrp/reports/atm_vrp_stageB_premium_truth_20260628/`（block 三档下界 + 同号稳定性 + 情景互补）
- `scripts/research_factor_scale.py`（moving-block 索引实现）
