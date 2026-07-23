# PIPELINE — 策略的一生（研究流水线装配图）

> 本文档是 `.claude/skills/` 技能库的总装配图：任何新策略研究**必须**按下面九站走完
> 一生（第 0 站出生 → 第 8 站尸检入库），每站挂载对应 skill。skill 不是可选参考，
> 是**到站必用**的工序（见 `research/CLAUDE.md` 的 skill 调用义务）。
> 进九站之前先过 **第 -1 站·快速死亡测试**（48h 筛选层，不属九站，见 §1）——它只回答
> "值不值得进入九站"，把坏想法挡在启动成本之外；配套工具 🔧 `prior-registry/`。
>
> 内容全部提炼自本仓库 2024–2026 两年实战（MR-5m 13 个月 + 干净数据阶段 13+ 份研究
> + VRP 线 + 5 次 B2_4h 增强否决），**不是新发明**。每站标注出处。
> 方法论的"为什么"层见 `docs/METHODOLOGY.md`；本文档是它的流程化。

---

## 1. 流水线全景 — 快速筛选（第 -1 站）+ 九站

```
第-1站 快速死亡测试（48h 筛选层，不属九站）──值得深入──► 第0站 ／ 判死·够不着──► 登记入 registry
   │
   ▼
第0站 出生地检查 ──► 第1站 机制登记 ──► 第2站 预注册+trial-ledger ──► 第3站 拟合与名单审查
                                                                            │
第8站 尸检入库 ◄── 第7站 判决 ◄── 第6站 多重检验 ◄── 第5站 单次检验 ◄── 第4站 误差棒与可行性
   │                                                                        
   └──死因喂回──► 第-1/0 站地图 + 第1站先验（闭环，见 §4；死因/基因登记见 🔧 prior-registry）
```

### 第 -1 站 · 快速死亡测试（48h 筛选层，不属九站）　🔧 `prior-registry/`

**定位**：九站之前的筛选层。目标 **48 小时内**回答"**值不值得进入九站**"，**不**回答
"能不能赚钱"。作用是**降低每个想法的启动成本**——让坏想法在零成本处死掉，而不是跑完
回测之后。宽松证据标准（粗糙数据/近似盘口/免费替代数据）在此站**允许**，红线见 §5。

**三问（顺序执行，任一判死即停）**：
1. **机制是否存在且能说清？** 查 🔧 `prior-registry/`（`query_death` / `check_conflict`）。
   说不出机制 = data mining，直接死；与 **Level 1**（机制失败）冲突 → 须走第 0 站
   **Challenge Path**（答"为什么这次失效"或"我的机制如何不同"）。
2. **数据是否够得着？** 测这个想法需要什么数据、我能否以可接受成本拿到？拿不到 =
   **数据墙**，记 **Level 2** 并写明所需条件（付费数据源 / 更细粒度 / survivor-free）。
3. **edge 量级 vs 成本/速度墙？** 粗估收益量级与成本量级，**数量级对比即可**
   （如 2bps 收益 vs 10bps 成本 → 直接死）。含**速度维度**：edge 的存活时间 vs 我
   可达延迟（价差/冲击类必问）。
4. **（可选）粗糙数据是否支持方向？** 不是证明，只排除明显错误。

**输出三选一**：
- ① **进入九站**：三问皆过，值得严格验证 → 交第 0 站出生证。
- ② **判死**：某问判死 → 按三级分类学**登记入 registry**（Stage -1 判死同样是知识，
  入库动作在此处也适用）。
- ③ **够不着**：需付费数据 / 大投入 / 更低延迟才能判 → 记 **Level 2** 并写明
  `wall_condition`，**不投入**（区别于判死：机制可能真、只是当前条件够不着，条件
  变化时可用 `list_reopenable()` 复查重开）。

**实战模板（从既有五个前置门提炼的共同形态——"用最便宜的检查回答够不够得着"）**：

| 前置门 | 最便宜的检查 | 判定形态 | 报告路径 |
|---|---|---|---|
| 期现基差可行性 | 数据勘察 + 收敛/超额粗算 | 效率定价判死（超额≈rf） | `research/_closed/basis/basis_arbitrage/reports/basis_arbitrage_feasibility_20260615/` |
| ATM VRP 阶段 A | 数据地基 + 净缝门（毛缝 vs 摩擦） | 摩擦**条件于对冲频率**；ETH 毛缝≈0 即死 | `research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageA_data_20260628/` |
| 大规模因子前置门 | 免费 100 币 + 噪声标定 + 规模梯度 + 流动性分层 | 毛 IC 增 ≠ 可交易 alpha 增（流动性伪装） | `research/_closed/crypto_tick/factor_scale/reports/factor_scale_feasibility_20260628/` |
| order-flow exhaustion | 数据门先行 + 回弹存活曲线 vs 延迟 + 随机时点安慰剂 | 回弹完成 < 可达延迟即速度墙 | `research/_closed/crypto_tick/order_flow_exhaustion/reports/order_flow_exhaustion_feasibility_20260628/` |
| 跨所套利延迟前置 | 价差存活时间 vs 可达延迟 | **设计未执行**（与 order-flow 延迟门同构）；先验速度墙 | 待核实（无独立报告） |

共同形态：**先问最便宜的那个"够不够得着"（数据 / 成本 / 速度 / 效率），一票否决即止**，
不进九站的重工序。五门中四门有完整报告、一门（跨所）系设计未执行标"待核实"（如实记录）。

**交接物 →0**：Stage -1 结论（进九站 / 判死入库 / 够不着记 Level 2）。

### 第 0 站 · 出生地检查（市场结构地图）

**问题**：本市场的结构性事实允许什么类型的策略？新想法与已探明的地图冲突吗？

**地图 = posterior belief，不是市场真理。** 每条结论都是"在某样本区间、以某证据强度
成立、在某条件下可能失效"的**后验信念**，随新证据可更新（下方每条的出处链接即其证据
基础与样本区间；Level 1 死因是它的数据来源，见 🔧 `prior-registry/`，地图是其摘要
视图）。**据此，冲突 ≠ 禁止**——见下方 Challenge Path。

本项目已探明的地图（加密永续 + Deribit 期权，双样本验证，出处
`research/_closed/_synthesis/PROJECT_FINAL_SUMMARY_20260614.md` §3 + 后续研究）：

- **方向区**：**右偏延续主导，从 100ms tick 到 4h bar 全尺度确认**（单序列/横截面/
  价差/订单流四种正交构造 + tick 级 order-flow exhaustion，全部死于延续 >回归）。
  → 允许趋势（顺延续），否定一切回归式（逆延续）。
  出处：`PROJECT_FINAL_SUMMARY` §3、`research/_closed/crypto_tick/order_flow_exhaustion/reports/order_flow_exhaustion_feasibility_20260628/`。
- **波动率区**：RV 高度可预测（聚集），但线性载体（永续）不可货币化（捕波动率退化为
  方向）；期权载体上 ATM IV≈RV 无溢价（ETH 无缝、BTC 剥离方向后 net≈0 且 whipsaw 尾）。
  出处：`research/_closed/crypto_perp/volatility_event/reports/volatility_event_20260613/`、`research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stage{A,B}_*/`。
- **横截面区**：真 alpha 存在但太弱/衰减/不可交易（流动性伪装）；规模只改善毛 IC
  不改善可交易 alpha。出处：`research/_closed/crypto_perp/cross_sectional/reports/cross_sectional_ic_20260613/`、
  `research/_closed/crypto_tick/factor_scale/reports/factor_scale_feasibility_20260628/`。
- **微观结构区**：成本/速度墙 ~50×，零售延迟下无可收割残余（用户已排除 HFT 基建）。
  出处：`research/_closed/crypto_tick/order_flow_exhaustion/reports/order_flow_exhaustion_feasibility_20260628/`。

**通过标准 · Challenge Path**：与地图冲突的想法**不自动过滤**，须回答二者之一——
① **为什么历史结论这次可能失效？**（给市场结构变化的**具体证据**，不是"也许不一样"）；
② **我的机制如何不同于已死机制？**（说清与命中的 Level 1 条目的机制差异）。
**答得出 → 走 Stage -1；两个都答不出 → 在本站过滤，零成本。**

- **成功范例（写入）**：order-flow exhaustion 正当挑战"延续主导"——地图说回归已死，
  但它赌 **impact 消散而非价格回归**，机制真正不同，故获准认真测试（结果三重判死：
  延续 + 安慰剂不可区分 + 速度墙，**流程正当**）。**若被地图直接过滤，就是地图变
  教条**——Challenge Path 正是防教条的阀门（与 prior-registry 失效模式"先验变教条"配套）。
- 反面：与地图冲突又说不出"为什么这次不同"的想法 = 换个壳的回归式赌注，本站死。

**交接物 →1**：出生证（想法一句话 + 与地图的关系声明：相容 / 走通 Challenge Path）。

### 第 1 站 · 机制登记与先验

**问题**：为什么这里该有钱？**谁在付钱、为什么钱还在**（没被套利掉）？
说不出机制 = data mining，不得进入第 2 站。

范式（出处 `research/_closed/crypto_perp/pairs_cointegration/reports/pairs_cointegration_20260613/README.md` 置顶三段声明）：
- **机制依据声明**：一段话说清利润来源的经济机制。
- **先验声明**：基于死因库（§4）登记先验——这个想法最可能死于哪条已知死因？
  先验中性/偏负都行，但必须写。
- **样本偏差声明**：幸存者偏差方向标注（偏乐观→判死更稳/判活须打折；
  出处：pairs/factor_scale 的 SURVIVORSHIP frozen 声明）。

**交接物 →2**：机制登记卡（三段声明）。

### 第 2 站 · 预注册 + trial-ledger　🔧 `preregistration/`

gate/阈值/币种宇宙/OOS 纪律**看结果前写死**（铁律 A）；同时登记本次要测多少配置
→ 开 trial-ledger（第 6 站的 N 从这里来，漏记 = 毒害第 6 站）。
成熟范式：`research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageB_premium_truth_20260628/PREREGISTRATION.md`
（先于结果单独 commit）。

**交接物 →3**：预注册文档 + trial-ledger 初版。

### 第 3 站 · 拟合与名单审查

跑描述统计/拟合。必做：
- **分布检查**：偏度/峰度/尾部计数——正态假设套肥尾是系统性错误
  （B2_4h 日收益超额峰度 +4.5、逐笔偏度 9.13；DSR 因此显式用实际矩，
  出处 `research/_closed/_synthesis/trend_methodology_hardening_20260622/`）。
- **数据纯净核验**：数据环境打印/manifest server 字段/独立锚点抽验
  （出处 `research/_closed/_synthesis/MR5M_postmortem.md` §8 + `data_engineering/CLAUDE.md`）。
- 领域工序：配对方向 🔧 `cointegration/`；横截面方向 🔧 `ic-analysis/`。

**交接物 →4**：拟合产物 + 分布诊断表 + 数据纯净声明。

### 第 4 站 · 误差棒与可行性　🔧 `bootstrap-inference/`

诚实的标准误：先测 ACF 再决定 iid 还是 block（**该测不该假设**——B2_4h 日 M2M
ACF≈0 是实测出的反预设发现）。由 SE 反推验证周期，做**立项算术**（铁律 B 前置）：
验证周期超出可接受资源窗口的方向，无论形态多有吸引力，**不立项**
（出处 `research/_closed/_synthesis/trend_line_closure_20260612.md`）。

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
出处：`research/_closed/_synthesis/trend_methodology_hardening_20260622/`（B2_4h 0.655→0.510/0.267/0.162）。

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
非复跑引擎；出处 `research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_pnl_audit_20260628/`）。

### 第 8 站 · 尸检入库

死因写进死因库——**机制层归因，不是参数层**（"没调好"不是死因；
"延续尾部吃成负期望"才是）。死因喂回第 1 站的先验、必要时更新第 0 站的地图。
文档动作：README 结论行 + `docs/PROJECT_GUIDE.md` 外科手术式更新（含"未改动文档
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
| 横切（3 站拟合 ↔ 未来 live 域） | `financial-precision/` | 横切技能：凡涉及金额计算处适用——研究记账 float 分界（经一次性诊断确认）/审计对账核对侧/live 激活时强制 Decimal |
| 0, 3（横跨） | `volatility-modeling/` | 凡波动率预测/时变误差棒/波动 regime 刻画（第 0 站波动区测绘、期权类 RV 腿、风控 σ 估计）；下游交接第 4 站时变误差棒与 honest-verdict 不确定性输入 |
| −1, 1, 8（承载闭环） | `prior-registry/` | 第 -1 站查"机制是否存在"（三问之一）+ 第 1 站取先验（`query_death`/`check_conflict` 判是否须走 Challenge Path）+ 第 8 站尸检入库（死因三级分类 L1/L2/L3 + 策略基因，记机制不记参数） |

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

## 5. 探索与验证的标准分界（防"统计完美主义"，亦防放松纪律）

Stage -1 与九站用**两套证据标准**，分界必须显式，双向都要守：

- **Stage -1 / 探索阶段——允许宽松**：粗糙数据 / 近似盘口 / 免费替代数据 / 弱证据都
  可以。因为该阶段的输出是"**值不值得深入**"，不是"能不能上真金"。既有实践即如此：
  ATM VRP 阶段 A 用近期盘口冻结快照代理历史摩擦（`data/live_friction.json`）、大规模
  因子前置门用**免费 100 币** Binance Vision 而非付费 survivor-free 数据。宽松是为了
  **让筛选便宜**（否则"统计完美主义"会让每个想法的启动成本高到无法探索）。
- **PIPELINE 第 0–8 站 / 验证阶段——严格不放松**：预注册（铁律 A）/ 双样本 / 噪声
  标定 / 多重检验打折（铁律 B）/ 判决纪律（双重门、铁律 C）**一个不少**。进了九站就是
  全套证据标准，没有"探索版"的折扣。
- **红线（不可弱化）**：**绝不允许以"探索阶段"名义，把弱证据的东西推向真金。** 探索的
  宽松是**筛选阶段**的宽松，**不是证据标准的宽松**。上真金的门槛由**预注册 gate**
  决定（如 B2_4h 的前向 U1 UPGRADE gate + 用户人工确认），与"探索时用了粗糙数据"
  完全无关——探索用近似盘口，不代表真金可以用弱证据。

> 一句话：Stage -1 便宜地问"要不要认真查"，九站严格地查"是真的吗"，预注册 gate 硬性地
> 定"能不能上真金"。三道门各管一段，混用即失守——尤其第三道，永不因探索宽松而降低。

---

*出处总表：CLAUDE.md（铁律 A/B/C、数据环境铁律）、`research/_closed/_synthesis/MR5M_postmortem.md` §8、
`research/_closed/_synthesis/PROJECT_FINAL_SUMMARY_20260614.md`、`research/_closed/_synthesis/trend_methodology_hardening_20260622/`、
`research/_closed/crypto_perp/pairs_cointegration/reports/pairs_cointegration_20260613/`、`research/_closed/crypto_tick/factor_scale/reports/factor_scale_feasibility_20260628/`、
`research/_closed/crypto_tick/order_flow_exhaustion/reports/order_flow_exhaustion_feasibility_20260628/`、`research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_pnl_audit_20260628/`、
`research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_vol_targeting_20260628/`、`research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stage{A,B}_*/`。*
