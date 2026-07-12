---
name: ic-analysis
description: 横截面因子 IC 分析工序——IC≠可交易 alpha 的三道墙（单调性/成本后价差/流动性分层）、看 IC 点估计非 t 值、幸存者偏差方向标注。任何横截面因子/排序信号/多币宇宙研究时必须使用。
---

# ic-analysis — 横截面 IC 工序（IC ≠ 可交易 alpha）

## 0. 流水线位置

**第 3–6 站**（横截面分支，见 `.claude/skills/PIPELINE.md`）。
上游输入：第 2 站预注册（因子集冻结、宇宙嵌套规则、分层与费率口径先写死）。
下游产出：IC + 噪声标定 + 分层可交易价差表 → 交第 5/6 站 gate 判定与打折
→ 第 7 站判决。

## 1. 何时用

- 任何横截面因子（动量/carry/反转/…）、任何"排序选币"信号。
- 任何"扩大宇宙规模能否改善 alpha"类问题（22→50→100 币）。

**不用会犯什么错（真实案例）**：
- **IC 显著 ≠ 能赚钱**（本项目三次实证）：F-MOM 真 alpha t=3.86 但日 IC≈0 已衰减；
  carry IC 随持有期单调增却死于参数尖峰；factor_scale REV 毛 IC 随规模升
  （+0.030→+0.056，超噪声 p=0.000）但 pool100 **净 −23%/yr**、流动层毛 −62%/yr
  （出处：`research/_closed/crypto_perp/cross_sectional/reports/cross_sectional_ic_20260613/`、
  `reports/factor_scale_feasibility_20260628/`）。
- **流动性伪装漏检**：REV 的净正 alpha 全部集中在不可交易的 illiquid 底层
  （+22%/yr @ 乐观 8bps，真实小币冲击成本 30–100bps+ 会抹掉）——**edge 与
  拿不到它是同一事实**。IC-based 伪装 gate（G2 用 IC 比例）太弱没咬住，
  可交易价差分层检验才咬住（出处：factor_scale Q4/Q5 + 方法论教训）。
- **用 t 判规模效应**：t 随横截面 N 机械上升（更多币 → 更小 SE → t 升），
  IC 点估计不动时 t 的上升不是信息增加（factor_scale Q3 显式用点估计）。

## 2. 怎么用

代码入口：本目录 `ic_toolkit.py`（从 `scripts/research_factor_scale.py` 的
IC/分层/价差机件提炼）。

1. **因子集与口径冻结**（第 2 站）：因子公式逐字预注册、参数不重搜、零变体
   ——这是横截面研究反 p-hack 的核心防线（factor_scale："factor set
   pre-registered & fixed"）。
2. **IC 口径**：逐日 Spearman(score, fwd-1d ret)（`daily_ic()`），信息截止
   前一收盘（shift(1)，无前视）。
3. **必过噪声标定**：real mean-IC vs `noise-calibration/` 的 NULL-A/B p95
   （超 0 不算数）。
4. **三道墙逐一检验（IC → 可交易 alpha 的距离）**：
   - **单调性**：分位组收益须随因子分位单调（`quintile_monotonicity()`）；
   - **成本后价差**：quintile long-short 日换手 × 费率 → 净年化
     （`ls_spread()`——换手 1.8×/日的反转类信号成本墙是结构性的）；
   - **流动性分层**：按成交额 top/mid/bottom 分层报 IC **和** 净价差
     （`tier_split()`）——净正只在底层 = 流动性伪装，预注册 FAIL 条款。
5. **规模梯度看 IC 点估计**：IC(K) 随 K 的变化用点估计 ± null SE，不用 t。
6. **幸存者方向标注**：宇宙无退市币 ⇒ alpha 偏乐观；对"买输家"类因子
   （REV）偏得最狠（死币 = 从未反弹的输家）。判死更稳 / 判活须标"乐观上界"。

## 3. 怎么失效

- **IC 正即当信号**：为什么——大宇宙 × 多因子下"IC 显著"是多重检验的保证产物，
  必须超同管线噪声基线（`noise-calibration/`），且过了噪声也只是毛信息。
- **用 t 值判规模/强度**：为什么——t = IC_mean/IC_std·√T 且横截面 N 增大直接
  压 IC_std，t 上升与"信号变强"无关；判强度用 IC 点估计与净价差。
- **IC-based 伪装 gate 太弱**：为什么——IC 是排序相关，对"alpha 集中在哪个
  流动性层"不敏感（top IC 0.0375 ≥ 0.5×bottom 就"过"了 G2）；**必须配可交易
  价差分层**——毛/净价差按层报，净正层是否可交易一目了然。
- **quintile 不看单调性**：为什么——非单调（中间组最好）时 long-short 价差是
  两个尾组的偶然差，排序信息并不存在；单调性是"IC 反映真实排序"的必要证据。
- **换手成本用一刀切低费率**：为什么——反转类信号日换手 ~2×，成本 = 换手 ×
  费率 × 365 是年化几十个点的结构墙；小币还有冲击成本（8bps 假设对 illiquid
  层"故意乐观"，判死才稳）。
- **忽略幸存者方向**：为什么——横截面宇宙按"现在还活着"选币，对买输家因子
  系统性抬高 alpha；不标注方向会把乐观上界当点估计用。

## 素材出处

- `scripts/research_factor_scale.py`（daily IC / ls_spread / 分层 / 规模梯度实现）
- `reports/factor_scale_feasibility_20260628/README.md`（三道墙、流动性伪装、t vs 点估计、幸存者方向）
- `research/_closed/crypto_perp/cross_sectional/reports/cross_sectional_ic_20260613/`（IC≠可交易 alpha 第一/二次实证：MOM/CAR/REV 22 币）
