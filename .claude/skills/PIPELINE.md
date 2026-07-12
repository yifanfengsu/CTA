# PIPELINE — 策略的一生（研究流水线装配图）

> 本文档是 `.claude/skills/` 技能库的总装配图：任何新策略研究**必须**按下面九站走完
> 一生（第 0 站出生 → 第 8 站尸检入库），每站挂载对应 skill。skill 不是可选参考，
> 是**到站必用**的工序（见 `research/CLAUDE.md` 的 skill 调用义务）。
>
> 内容全部提炼自本仓库 2024–2026 两年实战（MR-5m 13 个月 + 干净数据阶段 13+ 份研究
> + VRP 线 + 5 次 B2_4h 增强否决），**不是新发明**。每站标注出处。
> 方法论的"为什么"层见 `docs/METHODOLOGY.md`；本文档是它的流程化。

---

## 1. 流水线全景 — 九站

```
第0站 出生地检查 ──► 第1站 机制登记 ──► 第2站 预注册+trial-ledger ──► 第3站 拟合与名单审查
                                                                            │
第8站 尸检入库 ◄── 第7站 判决 ◄── 第6站 多重检验 ◄── 第5站 单次检验 ◄── 第4站 误差棒与可行性
   │                                                                        
   └────────死因喂回──────► 第0站的地图 + 第1站的先验（闭环，见 §4）
```

### 第 0 站 · 出生地检查（市场结构地图）

**问题**：本市场的结构性事实允许什么类型的策略？新想法与已探明的地图冲突吗？

本项目已探明的地图（加密永续 + Deribit 期权，双样本验证，出处
`reports/PROJECT_FINAL_SUMMARY_20260614.md` §3 + 后续研究）：

- **方向区**：**右偏延续主导，从 100ms tick 到 4h bar 全尺度确认**（单序列/横截面/
  价差/订单流四种正交构造 + tick 级 order-flow exhaustion，全部死于延续 >回归）。
  → 允许趋势（顺延续），否定一切回归式（逆延续）。
  出处：`PROJECT_FINAL_SUMMARY` §3、`reports/order_flow_exhaustion_feasibility_20260628/`。
- **波动率区**：RV 高度可预测（聚集），但线性载体（永续）不可货币化（捕波动率退化为
  方向）；期权载体上 ATM IV≈RV 无溢价（ETH 无缝、BTC 剥离方向后 net≈0 且 whipsaw 尾）。
  出处：`reports/volatility_event_20260613/`、`vrp/reports/atm_vrp_stage{A,B}_*/`。
- **横截面区**：真 alpha 存在但太弱/衰减/不可交易（流动性伪装）；规模只改善毛 IC
  不改善可交易 alpha。出处：`reports/cross_sectional_ic_20260613/`、
  `reports/factor_scale_feasibility_20260628/`。
- **微观结构区**：成本/速度墙 ~50×，零售延迟下无可收割残余（用户已排除 HFT 基建）。
  出处：`reports/order_flow_exhaustion_feasibility_20260628/`。

**通过标准**：新想法要么与地图相容（落在"允许"区），要么**显式声明挑战地图的哪一条
+ 机制上为什么这次不同**（如 order-flow exhaustion 曾正当挑战"延续主导"——赌 impact
消散而非价格回归，机制真正新颖，故认真测了；结果延续在 tick 级第五次显形）。
与地图冲突又说不出"为什么这次不同"的想法，在本站死，零成本。

**交接物 →1**：出生证（想法一句话 + 与地图的关系声明）。

### 第 1 站 · 机制登记与先验

**问题**：为什么这里该有钱？**谁在付钱、为什么钱还在**（没被套利掉）？
说不出机制 = data mining，不得进入第 2 站。

范式（出处 `reports/pairs_cointegration_20260613/README.md` 置顶三段声明）：
- **机制依据声明**：一段话说清利润来源的经济机制。
- **先验声明**：基于死因库（§4）登记先验——这个想法最可能死于哪条已知死因？
  先验中性/偏负都行，但必须写。
- **样本偏差声明**：幸存者偏差方向标注（偏乐观→判死更稳/判活须打折；
  出处：pairs/factor_scale 的 SURVIVORSHIP frozen 声明）。

**交接物 →2**：机制登记卡（三段声明）。

### 第 2 站 · 预注册 + trial-ledger　🔧 `preregistration/`

gate/阈值/币种宇宙/OOS 纪律**看结果前写死**（铁律 A）；同时登记本次要测多少配置
→ 开 trial-ledger（第 6 站的 N 从这里来，漏记 = 毒害第 6 站）。
成熟范式：`vrp/reports/atm_vrp_stageB_premium_truth_20260628/PREREGISTRATION.md`
（先于结果单独 commit）。

**交接物 →3**：预注册文档 + trial-ledger 初版。

### 第 3 站 · 拟合与名单审查

跑描述统计/拟合。必做：
- **分布检查**：偏度/峰度/尾部计数——正态假设套肥尾是系统性错误
  （B2_4h 日收益超额峰度 +4.5、逐笔偏度 9.13；DSR 因此显式用实际矩，
  出处 `reports/trend_methodology_hardening_20260622/`）。
- **数据纯净核验**：数据环境打印/manifest server 字段/独立锚点抽验
  （出处 `reports/MR5M_postmortem.md` §8 + `data_engineering/CLAUDE.md`）。
- 领域工序：配对方向 🔧 `cointegration/`；横截面方向 🔧 `ic-analysis/`。

**交接物 →4**：拟合产物 + 分布诊断表 + 数据纯净声明。

### 第 4 站 · 误差棒与可行性　🔧 `bootstrap-inference/`

诚实的标准误：先测 ACF 再决定 iid 还是 block（**该测不该假设**——B2_4h 日 M2M
ACF≈0 是实测出的反预设发现）。由 SE 反推验证周期，做**立项算术**（铁律 B 前置）：
验证周期超出可接受资源窗口的方向，无论形态多有吸引力，**不立项**
（出处 `reports/trend_line_closure_20260612.md`）。

**交接物 →5**：误差棒 + 验证周期预算（立项/不立项算术）。

### 第 5 站 · 单次检验　🔧 `noise-calibration/`

该配置在预注册 gate 下过不过。**凡"检测到 X 个信号/对子/事件"的计数类结论，
必过噪声标定**：real 指标必须超 shuffle/placebo 噪声基线，**超 0 不算数**
（231 对协整 ≈ 噪声底 11.55 的教训；factor_scale NULL-A/B；order_flow 随机时点
安慰剂——三次实战，一次漏装两次拦截）。

**交接物 →6**：单次检验结果 + trial-ledger 终版（含中途新增的一切配置——
中途加的也要入账）。

### 第 6 站 · 多重检验　🔧 `multiple-testing/`

"从 N 配置选最优"的研究必含：搜索配置数 N（名义/有效 ENB/扩展下界三口径）
+ deflated Sharpe（铁律 B）。验证周期用打折后 Sharpe 且从对自相关诚实的
bootstrap SE 反推。**双样本不能替代打折**（双样本答"另一样本是否成立"，
打折答"选择偏差抬高了多少"）。多假设场景配 FDR。
出处：`reports/trend_methodology_hardening_20260622/`（B2_4h 0.655→0.510/0.267/0.162）。

**交接物 →7**：打折后数字 + N 三口径表。

### 第 7 站 · 判决　🔧 `honest-verdict/`

- **双重门**：edge 为正 ∧ 尾部可生存，二者独立判，缺一判死。
- **偏态各自的纪律**（铁律 C 及其左偏镜像）：右偏不用 Sharpe 主判
  （vol-targeting 双样本 Sharpe 都升仍该否决）；左偏不用 mean 判可行性
  （mean 就是尾部——VRP Stage A 教训）。
- **答案条件于假设**：对冲频率/尾部假设/样本方向（gate 0 方向中性核对）。
- **判死与判活同证据标准**；资源关闭 ≠ 证伪，措辞纪律必守。

**交接物 →8**：判决书（判活/判死 + 全部 conditional 条款）。
判活的策略若产出"冻结数字"，其核算另过 🔧 `audit-independent/`（独立重推，
非复跑引擎；出处 `reports/b2_4h_pnl_audit_20260628/`）。

### 第 8 站 · 尸检入库

死因写进死因库——**机制层归因，不是参数层**（"没调好"不是死因；
"延续尾部吃成负期望"才是）。死因喂回第 1 站的先验、必要时更新第 0 站的地图。
文档动作：README 结论行 + `PROJECT_GUIDE.md` 外科手术式更新（含"未改动文档
及原因"显式列出）+ 记忆文件。否定性结论与肯定性结论**同等固化**。

**交接物 →0/1**：尸检报告（死因归类 → 死因库 → 地图/先验更新）。

---

## 2. 每站挂载的 skill（速查表）

| 站 | skill | 何时强制 |
|---|---|---|
| 2 | `preregistration/` | 一切研究开题时 |
| 3–5 | `noise-calibration/` | 凡"检测到 X 个信号/对子/事件"的计数类结论 |
| 3–5 | `cointegration/` | 配对/价差方向的拟合与筛查 |
| 3–6 | `ic-analysis/` | 横截面因子方向 |
| 4, 6 | `bootstrap-inference/` | 一切误差棒/下界/验证周期 |
| 6 | `multiple-testing/` | 凡"从 N 配置选最优" |
| 7 | `honest-verdict/` | 一切生死判定 |
| 7 后 | `audit-independent/` | 判活策略的冻结数字核算；重大否定结论复核 |

## 3. 站间交接物（上一站的产出 = 下一站的输入）

| 交接 | 交接物 |
|---|---|
| 0→1 | 出生证（与地图的关系声明） |
| 1→2 | 机制登记卡（机制/先验/样本偏差三段声明） |
| 2→3 | 预注册文档 + **trial-ledger 初版** |
| 3→4 | 拟合产物 + 分布诊断表 + 数据纯净声明 |
| 4→5 | 误差棒 + 验证周期预算（立项算术） |
| 5→6 | 单次检验结果 + **trial-ledger 终版**（第 2 站产出 → 第 6 站输入） |
| 6→7 | 打折后数字 + N 三口径 |
| 7→8 | 判决书（含 conditional 条款） |
| 8→0/1 | 尸检报告 → 死因库 → 地图/先验更新 |

## 4. 闭环声明 — 死因库如何复利

**死因库**（机制层归因清单，本项目已探明，出处逐条标注）：

| 死因 | 定义 | 实战出处 |
|---|---|---|
| 右偏延续 | 逆势赌回归被延续尾部吃成负期望（100ms→4h 全尺度） | 5m/15m-4h MR、funding 回归、配对、REV 因子、order-flow exhaustion |
| 成本墙 | 毛 edge 为正但薄于成本（数量级差） | breakout_pullback（1/4 成本）、REV pool100（换手 1.8×/日→40%/yr） |
| 速度墙 | edge 存活时间 < 可达延迟 | order_flow（净 −9.8bps 全延迟）、链上 front-running |
| 数据墙 | 干净测该机制所需数据不可得/太贵 | 微观结构需 L2（免费不可得）、survivor-free 需 Tardis $1-6k |
| 市场效率定价 | 确定性收益被定价到无风险利率 | basis carry（净年化 <5% ≈ rf） |
| 流动性伪装 | alpha 集中在不可交易层（edge 与拿不到它是同一事实） | factor_scale REV（净正仅不可交易底层） |
| peso / 尾部补偿 | "溢价"只是尾部风险的公平保费，样本恰好缺牙 | VRP StageB（剥方向后 net≈0 + whipsaw 尾） |
| 参数尖峰 | edge 只在参数点上成立、邻域即死 | carry 持有期（3 日过/5 日死） |
| 验证周期不可承受 | Sharpe 量级决定统计判别需十年计 | 趋势线关闭（14–138 年） |
| 描线（已知靶子单样本改善） | 对已知答案的样本画线，独立样本即反转 | funding-confirm、ADX、faster-entry、vol-targeting（5 次） |

**复利机制**：每份尸检的死因进上表 → 下一个想法在第 0/1 站先对着这张表核对先验
→ 越来越多的坏想法在**零成本的第 0–1 站**死掉，而不是在跑完回测之后。
实证：pairs 漏装噪声标定（231 对 ≈ 噪声）→ factor_scale 把噪声标定装在第一步
→ order_flow 开题即带随机时点安慰剂——同一个教训三站复利，一次漏装、两次拦截。
**这是研究能力的复利：结论会过时，死因库和流水线不会。**

---

*出处总表：CLAUDE.md（铁律 A/B/C、数据环境铁律）、`reports/MR5M_postmortem.md` §8、
`reports/PROJECT_FINAL_SUMMARY_20260614.md`、`reports/trend_methodology_hardening_20260622/`、
`reports/pairs_cointegration_20260613/`、`reports/factor_scale_feasibility_20260628/`、
`reports/order_flow_exhaustion_feasibility_20260628/`、`reports/b2_4h_pnl_audit_20260628/`、
`reports/b2_4h_vol_targeting_20260628/`、`vrp/reports/atm_vrp_stage{A,B}_*/`。*
