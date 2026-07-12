---
name: cointegration
description: 配对/价差协整研究工序——滚动 formation/trading 防前视、EG 非对称性、log 价差、半衰期、distance-matched 基线、必配噪声标定。任何配对交易/价差回归/统计套利方向的拟合与筛查时必须使用。
---

# cointegration — 配对/价差协整工序

## 0. 流水线位置

**第 3–5 站**（方向区-配对分支，见 `.claude/skills/PIPELINE.md`）。
上游输入：第 2 站预注册（formation/trading 窗长、p 阈值、z 阈值、成本、C1-C5
判定线全部先写死）。
下游产出：滚动筛查结果 + 事件统计（**全事件**，不只回归事件）→ 交第 5 站
gate 判定（配 `noise-calibration/`）→ 第 6 站。

## 1. 何时用

- 任何配对交易/价差回归/统计套利方向的前置研究与拟合。
- 任何"两资产长期均衡关系"类的主张（对冲组合、beta 中性腿）。

**不用会犯什么错（真实案例，全部出自 `research/_closed/crypto_perp/pairs_cointegration/reports/pairs_cointegration_20260613/`）**：
- **把假阳性当机制**：C1 报 12.48 对/窗协整显著——事后算术显示 231 对 × p<0.05
  的噪声底就是 11.55 对/窗，"存在性"仅高出噪声 8%。**没有噪声标定的协整计数
  没有意义**（该研究漏装，由 factor_scale 修复入流程）。
- **条件厚度陷阱**：C4"回归很肥（+13.8%）"是**只对 17.9% 成功回归事件**的条件
  统计；68.5% 发散事件把事件加权净期望吃成 −1.50%。只报回归成功事件 = 幸存者
  偏差的微观版——**必须报全部事件**。
- **回归率不比无条件基线**：C3 的关键设计——|z|≥2 后回归率 17.9% 必须对比
  "随机时点到达 |z|<0.5"的无条件基线 42.4%（two-prop z=−13.1，反向显著）。
  不设基线会把 17.9% 误读成"有回归"。

## 2. 怎么用

代码入口：本目录 `coint_toolkit.py`（从 `research/_closed/crypto_perp/pairs_cointegration/scripts/research_pairs_cointegration.py`
提炼；EG 检验经 statsmodels 惰性 import，其余纯 numpy）。

1. **价差口径**：log 价差 = `log(A) − β·log(B) − α`，β/α 由 **formation 窗 OLS**
   估计（`ols_spread()`）——不用价格比值（β=1 无依据）。
2. **EG 方向性**：`coint(A,B)` 与 `coint(B,A)` **不是同一检验**（OLS 残差不同）。
   预注册必须写死方向约定（如按固定字典序 + 双向都测取并集/交集，选一种写死），
   否则 C1 计数含糊（`eg_pvalue()` 强制显式传方向）。宇宙全集 C(n,2)，不做
   全样本挑对。Johansen（多资产/无方向）作为并列口径可加，属另一预注册检验。
3. **滚动 formation/trading 分离（防前视的生死线）**：formation 窗（90d）估
   β/μ/σ + 跑 EG；trading 窗（30d）**只用 formation 参数，不重估**；月度前移
   （`rolling_windows()`）。在 formation 窗内选对再在同窗内测回归 = 前视。
4. **半衰期必报**：`half_life()`（AR(1)：Δs_t = a + b·s_{t−1}，HL = −ln2/ln(1+b)）。
   半衰期 > trading 窗 ⇒ 该窗内根本等不到回归，厚度统计无意义。
5. **稳定性指标**：持续窗数（中位）、churn 率（相邻窗可交易集换血率）——加密
   教训：中位持续 1 窗、churn 91%，协整瞬时不可前向交易（C2 是真正的生死门）。
6. **事件统计**：首次 |z|≥2 触发，解析至回归(|z|<0.5)/发散(|z|>3)/未决；
   **全事件加权**净期望（C5）> 0 才算破裂可控。回归率对比基线用
   **distance-matched**：2σ 事件的对照是"随机时点同窗剩余时间到达 |z|<0.5"，
   不是无条件全时点混入均值附近 trivial 点的口径（`baseline_reach_rate()`）。
7. **必配 `noise-calibration/`**：协整对计数过噪声底（`placebo_expected_count`）。

## 3. 怎么失效

- **EG 方向未定 → 计数含糊**：为什么——EG 对 (A,B) 与 (B,A) 的 ADF 统计量不同，
  临界情形一向显著一向不显著；方向不写死则"多少对协整"依赖实现细节，不可复现。
- **无噪声标定 → 假阳性当机制**：为什么——n 对 × p 阈值给出确定的期望假阳性数，
  计数不超它就与"零真协整"不可区分（231×0.05=11.55 vs 实测 12.48 的教训）。
- **formation 窗内选对 → 前视**：为什么——同窗内"协整显著"与"价差回归"在数学上
  相关（残差平稳就是会回归），同窗测回归率必然虚高；必须 out-of-formation 的
  trading 窗验证。
- **回归率对比无条件基线 → 不公**：为什么——无条件基线混入大量已在均值附近的
  trivial 时点，抬高对照组；须 distance-matched（同为极端偏离状态起算）……
  注意本项目实测中即便用了偏"宽松"的无条件基线（42.4%），条件回归率（17.9%）
  仍反向更低——真实失败可以比不公平的对照更惨。
- **半衰期不报 → 窗长与机制错配**：为什么——AR(1) 半衰期是回归速度的物理量，
  trading 窗 < 半衰期时"未决"事件堆积，把机制失败与窗口太短混在一起。
- **幸存者样本高估稳定性**：为什么——退市币（LUNA 类）的"一腿归零"是协整破裂
  最极端形态，幸存者样本天然缺失 → 破裂尾部系统性低估；结论须标注方向
  （偏乐观样本仍判死 ⇒ 否定更稳）。

## 素材出处

- `research/_closed/crypto_perp/pairs_cointegration/scripts/research_pairs_cointegration.py`（滚动筛查/事件解析/基线口径全实现）
- `research/_closed/crypto_perp/pairs_cointegration/reports/pairs_cointegration_20260613/README.md`（C1-C5 判定、噪声教训、条件厚度陷阱、Q4 机制分析）
- `research/_closed/crypto_tick/factor_scale/reports/factor_scale_feasibility_20260628/README.md`（噪声底算术的显式修复记录）
