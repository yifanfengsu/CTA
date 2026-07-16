---
name: volatility-modeling
description: GARCH 族波动率建模工序——ARCH 效应前置检验、Student-t innovation、persistence/半衰期/无条件方差量纲显式、多步均值回归预测与 EWMA(IGARCH) 对照。任何波动率预测/时变误差棒/波动 regime 刻画（第 0 站波动区测绘、期权类 RV 腿、风控 σ 估计）时必须使用。
---

# volatility-modeling — GARCH 族波动率建模

## 0. 流水线位置

**横跨第 0 站与第 3 站**（见 `.claude/skills/PIPELINE.md` §2 挂载表）：
- **第 0 站（出生地检查）**：市场波动结构测绘——ARCH 效应检验回答"这个市场的
  波动可不可预测"，与地图的"方向区/波动区"划分直接对应。
- **第 3 站（拟合与名单审查）**：收益分布的条件异方差诊断（与分布检查同批）。

下游交接：**第 4 站误差棒**（GARCH 的条件方差是时变的 → 时变的诚实误差棒，
替代"全样本一个 σ"的粗口径）、**honest-verdict**（判决的不确定性量化输入）。

与本仓库研究史的锚点（这些不是外部教科书例，是本仓库已付学费的事实）：
- **波动率可预测，方向不可预测——同一事实的两面**。方向区已探明右偏延续主导、
  无回归 edge（`research/_closed/_synthesis/PROJECT_FINAL_SUMMARY_20260614.md` §3、
  `research/_closed/crypto_tick/flow_vs_price/reports/flow_vs_price_trend_20260628/`）；
  波动区的聚集性是本市场少数真实的可预测结构（5/5 币 ×2 样本，
  `research/_closed/crypto_perp/volatility_event/reports/volatility_event_20260613/`）。
  GARCH 是波动区那一面的标准模型语言——它建模的正是聚集本身。
- **VRP 线的 RV 口径升级路径**。atm_vrp 研究的 IV−RV 缝中，RV 用的是粗口径
  （永续日线 close-to-close、年化 ×365，
  `research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageA_data_20260628/`）。
  GARCH 条件预测是该口径的升级路径——对应 vrp 线关闭时预留的窄复活条件
  （`research/_closed/crypto_options/vrp_atm/`）。升级 RV 预测≠复活该线：复活须走
  预注册的复活 gate，本 skill 只提供工具。
- **vol-targeting 否决案例的再解读**。B2_4h-VT 的 σ_t 用 EWMA（半衰期 48 bars），
  EWMA 是 RiskMetrics/IGARCH 特例（β 固定 λ、ω=0、**无均值回归项**）。该案例死因
  是"削右尾"（铁律 C），**不是 σ 估计不准**——换 GARCH 估 σ 不改判决；但 EWMA vs
  GARCH 的结构差异（失效模式⑦）在长 horizon 预测里是实质的
  （`research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_vol_targeting_20260628/`）。

## 1. 何时用

触发场景：任何需要**波动率预测 / 时变误差棒 / 波动 regime 刻画**的研究——
期权类研究的 RV 腿、风控类的 σ 估计、第 0 站新市场波动结构测绘。

**前置检验（必做，按序）**：
1. **收益平稳性**（ADF）——对收益（不是价格）检验；不平稳则先差分/查数据。
2. **ARCH 效应检验**（`arch_effect_test`：ARCH-LM + Ljung-Box on r²，双检验合取）
   ——**无 ARCH 效应则 GARCH 无意义**，到此为止，不硬拟合。
3. **分布形态检查**（偏度/峰度）→ 决定 innovation 分布假设。加密峰度 ≫3
   （BTC 日收益全样本峰度 31，见本目录 DIAGNOSTIC），**禁用正态 innovation**，
   Student-t 起步（skew-t 备选）——"正态套肥尾"是本仓库既有教训
   （PIPELINE 第 3 站分布检查、DSR 显式用实际矩的同源纪律）。

**不用/慎用**：
- **预测收益方向**：GARCH 预测的是分布参数（σ_t）不是方向；均值方程 μ≈0
  （BTC 日频 |μ/SE|≈1.75，不显著）——这是本仓库"方向区不可预测"结论的模型层
  版本，企图从 GARCH 的 μ 里找方向 = 换个语言重犯已探明的错。
- **超短样本**：GARCH(1,1)+t 有 4–5 个参数，几百个观测下估计不稳（β SE 爆炸、
  优化器贴边界）；日频经验底线千级观测。
- **跨结构断点直接拟合**：312/LUNA/FTX 类断点会把 persistence 推向 1
  （失效模式③；本目录 BTC 诊断的全样本拟合就撞了 1 边界——活例）。

## 2. 怎么用

流程（每步对应 `garch_toolkit.py` 一个入口；工具自检
`python garch_toolkit.py` 冒烟 19/19 PASS 才可用）：

1. **数据准备**：对数收益、**小数标度**（0.01=1%）；显式声明频率与年化因子
   ——加密 **365**（24/7，与 vrp_atm RV 口径一致），传统市场 252；日界/时钟
   对齐显式声明（时钟错配 = 虚假缝的既有教训，vrp_atm 的 8h 网格同源）。
2. **ARCH 效应**：`arch_effect_test(returns)`——不过则停。
3. **模型选择**：GARCH(1,1) 起步；杠杆效应显著才上 EGARCH/GJR（`o=1`）。
   注意**加密的杠杆效应与股票不同**（双向大波动，负收益未必系统性推高后续
   波动），是否需要不对称项**须实测**（比较 AIC/BIC + γ 显著性），不套股票先验。
4. **innovation 分布**：`dist='t'` 起步（前置检验③的结论落地）。
5. **拟合**：`fit_garch(returns, dist='t', model='GARCH')`——内部 ×100 喂优化器、
   ω 自动 /100² 还原，调用方只见小数标度（percent/decimal 混用从接口上封死）。
6. **诊断**：`residual_diagnostics(fit)`——标准化残差 Ljung-Box（水平+平方）+
   剩余 ARCH-LM 全部 p>0.05 才算吃干净；t 设定下 std resid 峰度 >3 属预期。
7. **预测**：`forecast_vol(fit, horizon)`——多步条件方差路径 + 向无条件方差的
   均值回归形态；半衰期 = ln(0.5)/ln(α+β)（`persistence_report`）。

| 函数 | 用途 |
|---|---|
| `arch_effect_test(returns)` | ARCH-LM + Ljung-Box(r²) 双检验合取（前置 gate） |
| `fit_garch(returns, dist='t', model='GARCH')` | 封装 arch 包拟合；参数还原小数标度 |
| `persistence_report(fit, periods_per_year)` | α+β 点估计+delta-method SE、半衰期、无条件方差/日 vol/年化 vol **三量纲同报** |
| `forecast_vol(fit, horizon)` | 多步条件 vol 路径（cond_var/cond_vol/ann_vol 三列显式） |
| `residual_diagnostics(fit)` | 标准化残差检验套件（合格线显式） |
| `ewma_sigma(returns, lam)` | EWMA 对照（RiskMetrics=IGARCH 特例，注释见函数 docstring） |
| `simulate_garch(ω, α, β, n)` | 已知参数模拟（自检/功效分析用） |

依赖：`arch` 包（本 skill 落地时装入 `.venv`：`.venv/bin/pip install arch`，
冒烟基准 arch 8.0.0；toolkit 内惰性 import——与 cointegration skill 的
statsmodels 同款登记方式）+ 既有 numpy/pandas/scipy/statsmodels。

**输出解读规范（本技能第一大错误源）**：一切波动率数字必须标注三件事——
**方差还是 vol / 日还是年化 / 年化因子几**。自证基准（冒烟 dim-1..3，代码现算
非引用）：ω=1e-5, α=0.1, β=0.85 → 无条件**日方差** 2×10⁻⁴ → **日 vol** 1.41%
→ **年化 vol** ≈27%（×√365）。同一组参数，三个量纲三个数字——漏标注哪层，
读者就错哪层。

## 3. 怎么失效

- **量纲混淆**：为什么——ω/(1−α−β) 给出的是**方差**不是波动率，日/年化再错配
  一层（√365 vs √252），错一层就是数量级级别的虚假数字；本 skill 的素材笔记
  原稿就把该式的结果当 vol 读（勘误见冒烟 dim-1..3），量纲错误连教材级素材
  都躲不过，故 toolkit 三量纲强制同报。
- **正态 innovation 套肥尾**：为什么——加密日收益峰度 ≫3（BTC 全样本 31），
  正态假设系统性低估尾部概率 → VaR/误差棒过窄 → 下游判决虚假乐观；t/skew-t
  是起步配置不是可选项（既有"正态套肥尖"教训的 GARCH 层版本）。
- **persistence 边界误判**：为什么——α+β 接近 1 **不等于**模型失效：估计有
  标准误，且结构断点（312/LUNA/FTX）会伪装成高 persistence——本目录 BTC 诊断
  全样本拟合 α+β 撞 1 边界、分段后降到 0.989（62 天半衰期）即活例。正确动作是
  **看 SE、分段拟合/断点检验、评估半衰期实用性**，不是机械 0.99 红线一刀切。
- **均值方程误用**：为什么——GARCH 输出分布参数非收益方向，μ 在加密日频不显著
  （方向区结论的模型层版本）；把 μ 或"波动率预测"当方向信号用 = 已探明死路
  （volatility_event：捕波动率在线性载体上退化为方向）。
- **样本外幻觉**：为什么——样本内拟合优度（loglik/AIC）≠ 预测力；vol 预测须
  OOS 评估（QLIKE 损失 / Mincer-Zarnowitz 回归），且预测靶（RV proxy）本身有
  测量误差——拿平方收益当"真 RV"评估会系统性低估可预测性。
- **频率错配**：为什么——GARCH 参数不跨频率通用（时间聚合改变 α、β；日频
  α+β≈0.99 不能推小时频同值），用日收益拟合回答小时级问题 = 用错参数集；
  换频率必须重拟合 + 重报年化因子。
- **EWMA 当 GARCH 用**：为什么——EWMA 是 IGARCH 特例（ω=0、α+β≡1、无均值
  回归项），多步预测走平、**永不**回归长期水平，长 horizon 与 GARCH 系统性
  分叉（冒烟 ewma-2 数字自证）；B2_4h-VT 用的正是 EWMA（其死因是削右尾而非
  σ 估计，见第 0 段锚点），风控短窗 EWMA 够用，**预测**场景必须带回归项。

## 素材出处

- 用户系统化学习阶段 1.3 笔记（GARCH 族要点；其中"无条件方差 ≈0.15"的量纲
  错误已由冒烟 dim-1..3 现算修正，本 skill 不引用笔记示例数值）。
- 本目录 `garch_toolkit.py`（冒烟 19/19 PASS：参数回收/双向报警/量纲换算/
  半衰期/预测收敛/EWMA 对照全部模拟自证）。
- 本目录 `DIAGNOSTIC_BTC_20260716.md`（BTC 日收益 GARCH 测绘：ARCH 效应显著、
  全样本撞 1 边界的断点活例、近段 α+β=0.9890/半衰期 62 天/年化无条件 vol 63%；
  第 0 站波动区参考件，非研究结论）。
- `research/_closed/crypto_perp/volatility_event/reports/volatility_event_20260613/`
  （波动可预测/不可货币化于线性载体——本 skill 的第 0 站地图出处）。
- `research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageA_data_20260628/`
  （RV 粗口径与 365 年化约定——GARCH 升级路径的基线）。
- `research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_vol_targeting_20260628/`
  （EWMA σ 的实战案例与其真实死因——失效模式⑦的边界）。
