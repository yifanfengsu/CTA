# FLOW 信号 vs 价格信号 —— 趋势捕获对照研究（预注册，对照 B2_4h）

**判定：FAIL（不过）。** 用 taker order-flow imbalance（趋势的"原因"）做信号，在零售
够得着的任何尺度上，**都不比** B2_4h 的价格信号（趋势的"结果"）更早或更好地捕获已被
证明存在的右偏延续。结论 = **"用 flow 代替价格无额外 edge，B2_4h 价格信号已足够。"**
**不退化为"把 flow 加到 B2_4h 当过滤器"（第五次增强=描线）——守边界。**

运行 UTC：2026-06-29T12:58:50Z · seed 20260628 · 引擎 `tb`/`r2`/`binance_funding`/
`order_flow_exhaustion` 全部 verbatim 导入、零修改 · `tb.SYMBOLS` 外层限定为 {BTC,ETH}。

---

## 定位与纪律（frozen — 全文不删改）

> 本研究的核心问题：趋势 ≈ ∫(Net Order Flow × Impact) dt——price 是结果、flow 是原因。
> B2_4h 用价格（结果）捕获趋势。问：用 taker flow imbalance（原因）做信号，能否比价格
> 信号（结果，即 B2_4h）更早/更好地捕获延续？
>
> 三个已验证的背景事实（决定本研究的边界与先验）：
> ① 右偏延续是端到端结构性事实（4h bar→100ms tick 全部确认）——所以"flow 持续→价格
>    延续"大概率成立，本研究不是验证延续存不存在（已知存在），而是验证"flow 信号 vs
>    价格信号谁更好捕获它"；
> ② order-flow exhaustion 已三重判死——所以本研究只测 flow persistence（跟随 flow），
>    绝不测 flow exhaustion reversal（fade flow，已判死）；
> ③ flow 的优势若存在，大概率在"比价格更早一点"的窗口，而该窗口是 HFT 争夺、零售够不着
>    ——故速度检验是核心判生死项，非可选。
>
> 【最硬的边界——防第五次描线】本研究的对象是"flow 信号能否**替代或超越**价格信号
> (B2_4h)"，绝对**不是**"在 B2_4h 上加 flow 过滤/增强"。B2_4h 的四次增强
> (funding/ADX/faster/V1) 已全部双样本判死，证明它是干净提取、无过拟合空间。若本研究
> 发现 flow 信号不如或约等于价格信号 → 结论是"不用 flow，B2_4h 的价格信号已足够"，
> **严禁**退化为"那把 flow 加到 B2_4h 当过滤器"(那是第五次增强=描线)。flow 与 price 是
> **对照的两个独立信号**，不是"基底+增强"。这条违反即研究失效。
>
> 判定哲学：以 B2_4h 为对照基准；噪声标定（flow 信号须超过 shuffle 基线）；速度检验
> （flow 优势在零售延迟下是否还剩）；不算 Sharpe 主判（正偏，用整个分布/捕获效率，
> 铁律 C）；预注册 gate 先于结果（铁律 A）。

## 成本口径（写进报告头部）

入场与出场均按信号 bar 收盘价 ±1 tick 以 taker 成交（费率 0.05%/边）；不假设 maker；
计入真实 Binance 8h funding；保留 OKX `ctVal`/`tickSz` 使冻结引擎逐字节复现（2020 年
低价处 tick 占比偏大，偏保守，与 dual-cycle 同口径简化）；无滑点压力测试。正偏策略
**不以 Sharpe 为主判**（铁律 C）。

## 预注册（看结果前写死，铁律 A）

- **数据**：Binance Vision UM-perp **1m klines**，BTCUSDT+ETHUSDT，2020-01..2026-05
  （77 月/币，磁盘已 sha256 核验）。标准 klines CSV 含每根 `volume`(col5) 与
  `taker_buy_volume`(col9) ⇒ taker 失衡可在**任意 bar 尺度**算出，**无需订单簿深度、
  无需下载海量 tick**。**仅 Flow 层**（明确声明）：无订单簿深度（Liquidity/Impact 层=
  数据墙，不做）。亚秒速度子检验复用 exhaustion 研究的 aggTrades（9 个季度日，磁盘已有）。
  **双样本 = {BTC, ETH}**——这是两个免费、含 taker 拆分的标的；flow edge 须在两者**都**
  成立（机制独立于币种）。价格基准 B2_4h 跑在**同一份 Binance bar 上**（数据保持恒定——
  对照两个**信号**就不能用两个数据源混淆）。
- **Flow imbalance 定义（frozen）**：`OFI_norm = (taker_buy − taker_sell)/total_vol
  = (2·taker_buy − volume)/volume ∈ [−1,+1]`（归一化净主动 taker 失衡=趋势的"原因"）。
- **Flow 信号（frozen，3 组，不搜参）**：`signal = sign(EMA_span(OFI_norm))`，always-in，
  反向出场（`positions_flip`，与 B2_4h **完全同一机制**；唯一差别是输入序列=平滑后的
  **flow** 而非 **price**）。这是 flow **persistence**（持续买压→做多/持续卖压→做空），
  **不是** exhaustion。`F1_4h_20`(4h,EMA20)、`F2_4h_50`(4h,EMA50)、`F3_1h_50`(1h,EMA50)。
  诊断（非配置）：累计 flow EMA 交叉 `sign(EMA20(ΣOFI)−EMA100(ΣOFI))`——验"积分 flow ≈ 价格"。
- **价格基准（frozen = B2_4h）**：`signal_emax(bars_4h,20,100)`，`positions_flip`，
  config_frozen 原样、不改动。同标的/区间/窗口/成本/m2m 记账。
- **预注册判生死**：见下"第 4 部分"。所有阈值在跑之前写死。

---

## 第 1 部分：Flow persistence 是否成立（机制核对，噪声标定）

| | ACF(lag1) | shuffle-block p95 | 超噪声? | IC(OFI→ret₊₁) | shuffle p95 | 超噪声? | hit-rate | 同期 corr(OFI,ret) |
|---|---|---|---|---|---|---|---|---|
| BTC | **+0.1039** | 0.0959 | ✅ | **+0.0094** | 0.0166 | ❌ | 0.476 | **+0.510** |
| ETH | **+0.1028** | 0.1021 | ✅(勉强) | **+0.0063** | 0.0153 | ❌ | 0.475 | **+0.515** |

**读法（关键）**：flow imbalance **作为一个过程是真有记忆的**——ACF(lag1)≈0.10，超过
block-shuffle 噪声基线（订单拆分/羊群的已知特征）。但这个记忆**不转化为对下一根的
收益预测**：IC(OFI(t)→ret(t+1))≈+0.008，**低于** shuffle 噪声地板，hit-rate < 0.5。
而同期 corr(OFI(t),ret(t))≈+0.51——**flow 与价格在同一根内同步发生**（两者是同一事件
的两个视角），flow **并不领先**价格。⇒ **持续性真实、但前瞻预测性区分不出噪声**：
预注册 gate ①（要求记忆能前瞻预测=可货币化）**不过**。

## 第 2 部分：Flow 信号 vs 价格信号 对照（核心，同一引擎/同窗/双样本）

> 两个**独立信号**并列对照（不是基底+增强）。各自 spans → 同一 `r2.m2m_pnl` 记账。
> 起点对齐 = 数据起点 +20 天（两信号均已预热）。正偏 ⇒ 看 net / net‑maxDD / 尾部 / 笔数，
> 不看 Sharpe。

| 信号 | BTC net | BTC net/maxDD | BTC 笔数 | ETH net | ETH net/maxDD | ETH 笔数 | shuffle‑flow p95 net (BTC/ETH) |
|---|---:|---:|---:|---:|---:|---:|---|
| **价格 B2_4h** | **$20,727** | **2.28** | 165 | **$40,506** | **3.33** | 173 | — |
| F1_4h_20 | $22,686 | 1.90 | 1266 | **−$17,340** | −0.49 | 1138 | −$401 / −$4,261 |
| F2_4h_50 | $30,886 | 4.38 | 658 | **−$31,627** | −0.35 | 562 | $1,524 / −$6,931 |
| F3_1h_50 | −$858 | −0.04 | 3585 | **−$33,178** | −0.84 | 3545 | −$24,451 / −$28,445 |

**读法**：
- **没有任何一组 flow 配置在两个币上都赢价格。** F1/F2 在 **BTC 为正**（F2 甚至 net 与
  net/DD 都超过 B2_4h），但在 **ETH 深度为负**（−$17k / −$32k）。F3 两边皆负。
- **F2 的 BTC 胜利是单币海市蜃楼**：ETH 上 F2(−$31,627) **比它自己的 block-shuffle p95
  (−$6,931) 还差**——即 ETH 上 flow 比随机还烂。一个真 edge 不会在两个最流动的币上变号。
- **flow 信号成交 4–21× 于价格信号**（F1 1266 vs 165；F3 3585 vs 165）：flow 是**高频
  whipsaw** 信号——因为 flow 这个"率"快速均值回归（ACF≈0.10 很快衰减），跟随它=被锯。
  价格"水位"才趋势（右偏延续）；B2_4h 收割的是水位的趋势。

### 2c 信号重合度（SIGNED 持仓方向一致率）

> 注：`r2.m2m_pnl` 的 `pos_by` 是 0/1 在场指示（非方向），本研究另建 SIGNED 持仓
> （±1，按 m2m 归属窗口 ei+1..xi）算方向一致率——否则两个 always-in 策略会假性 100%。

| | F1_4h_20 | F2_4h_50 | F3_1h_50 | 累计flow-cross |
|---|---|---|---|---|
| BTC 方向一致率 | 0.497 | 0.529 | 0.462 | 0.528 |
| ETH 方向一致率 | 0.450 | 0.488 | 0.420 | 0.513 |

**重合度 ≈ 0.42–0.53（≈掷硬币）**：flow 信号与价格信号方向只有约一半时间相同——
flow 是一个**确实不同**的信号，**不是**"原因=结果在你尺度上是同一持仓"。但（接第 2 部分）
**不同≠更好**：这"不同"是 whipsaw 噪声，不是信息（different AND worse）。连"先积分再
交叉"的累计-flow 构造也只有 ~0.51 一致率——归一化失衡丢掉了真正推动价格的 volume×impact
量级，**积分 flow 也复现不出价格信号**。

### 2b 时机（flow 是否更早？）

flow 在匹配的翻转点上比价格**早 ~2–4 根 4h（8–16h）、约 58% 的翻转更早**（F1 BTC
0.602/16h、ETH 0.58/8h；F2 BTC 0.59/16h、ETH 0.557/8h）。**但"更早"不带 edge**——见第 3 部分。

## 第 3 部分：速度检验（核心判生死——flow 优势在零售延迟下还剩不剩）

**3a 多分辨率 IC（OFI(t)→ret(t+1)，定位 flow 预测力的尺度）**

| 分辨率 | 1m | 5m | 15m | 1h | 4h |
|---|---|---|---|---|---|
| BTC IC | −0.0055 | −0.0146 | −0.0066 | +0.0006 | +0.0094 |
| ETH IC | −0.0064 | −0.0099 | −0.0017 | +0.0007 | +0.0063 |

前瞻 IC 在**所有**尺度都微乎其微，细尺度（1m/5m）甚至**为负**；4h 处最高也仅 +0.006~0.009，
且（第 1 部分）**低于 shuffle 噪声地板**。⇒ **不存在任何 bar 尺度上 flow→未来收益是可交易
信号**。

**3b +1 根执行延迟 haircut（晚一根进场）**

| 配置 | BTC: 无延迟 → +1根 | ETH: 无延迟 → +1根 |
|---|---|---|
| F1_4h_20 | $22,686 → **$35,211** | −$17,340 → −$11,685 |
| F2_4h_50 | $30,886 → $27,419 | −$31,627 → −$22,177 |
| F3_1h_50 | −$858 → **$2,221** | −$33,178 → −$25,933 |

**晚一根不但不伤、常常更好。** 若 flow 真有"更早=抢先捕获"的 edge，延迟一根应当摧毁它；
它没有 ⇒ **bar 尺度的"更早"是噪声、不携带信息**（与 `trend_faster_entry` 同教训：
更早≠更好）。

**3c 亚秒 flow→价格 lead-lag（aggTrades，18 币·日）**

| horizon | 1s | 2s | 5s | 10s | 30s | 60s |
|---|---|---|---|---|---|---|
| corr(flow, **未来**ret) | +0.056 | +0.053 | +0.044 | +0.037 | +0.021 | +0.012 |
| corr(flow, **过去**ret) | +0.307 | +0.238 | +0.200 | +0.187 | +0.162 | +0.137 |

flow 与**过去**价格的相关 ≈ 与**未来**价格相关的 **5×**——flow 主要是价格的**同期/滞后影子**，
不是领先指标。仅有的那点前瞻相关（+0.056@1s）**集中在亚秒并迅速衰减**——正是 impact 尚未
传导完的微观窗口=**HFT 域**，也正是 order-flow exhaustion 已判死的同一堵墙
（净 **−9.78 bps**/每个延迟档）。**flow 唯一真正"领先"价格的地方，零售够不着。**

## 第 4 部分：判生死（预注册）

| gate | 要求 | 结果 |
|---|---|---|
| ① persistence 超噪声 | ACF 超 shuffle **且** OFI→ret₊₁ IC 超 shuffle（两币、同号） | **❌**（ACF 过、前瞻 IC 不过） |
| ② dominance | 存在一组 flow 配置，两币都在 net **且** net/maxDD 上超 B2_4h，且 > shuffle p95 | **❌**（无；BTC 海市蜃楼，ETH<shuffle） |
| ③ reachable | 胜者扛得住 +1 根延迟、≥1h 采样、非仅 HFT 带 | **❌**（无胜者；且唯一领先在亚秒 HFT 带） |
| ④ not redundant | 两币方向一致率 < 0.75 | **✅**（~0.45–0.53，flow 确实"不同"——但不同≠更好） |

**FINAL = FAIL**（需 ①∧②∧③∧④ 全过才 PASS）。④ 单独为真（flow 是不同信号），但这恰恰
排除了"flow 与价格在你尺度上是同一东西"的简单解释，落到更强的结论：**flow 是一个不同、
更高频、更嘈杂的信号，且更差**。不以 Sharpe 主判。

---

## 第 5 部分：诚实结论（Q1–Q6）

**Q1 flow persistence**：**部分成立但不可货币化**。flow imbalance 作为过程有真实记忆
（ACF≈0.10>shuffle），但该记忆**不转化为前瞻收益预测**（IC<噪声地板），同期 corr≈0.51
（flow 与价格同步、不领先）。

**Q2 flow vs price（核心）**：**flow 不如价格**。无任何配置在两币上都胜；F1/F2 在 BTC 为正
但 ETH 深亏（ETH 上比随机还差），F3 两边皆负；价格 B2_4h 两币都强（net/DD 2.28 / 3.33）。
**信号重合度低（~0.45–0.53）**——flow 确实是不同的信号，但其"不同"是 whipsaw 噪声
（flow 成交 4–21× 价格），不是价格之外的额外信息。

**Q3 速度检验**：**flow 的优势全在 HFT 窗口、零售够不着**。bar 尺度上 flow 的"早 2–4 根"
不带 edge（+1 根延迟不伤反好；多分辨率 IC 全部 ≤ 噪声）；flow 唯一真正领先价格是在亚秒
（corr 未来 +0.056@1s→0@60s，过去相关是其 5×），那是 impact 未传导完的微观窗口=HFT，
=order-flow exhaustion 已判死的同一堵墙（−9.78bps/每延迟）。

**Q4 判生死**：**不过（FAIL）**，死因 = ①前瞻预测不超噪声 + ②无配置双样本超价格（BTC
单币海市蜃楼 / ETH 比 shuffle 还差） + ③唯一领先在亚秒 HFT 带。

**Q5 结论**：**用 flow 代替价格无额外 edge，B2_4h 价格信号已足够捕获延续。**
**明确：不退化为 flow 过滤 B2_4h（描线）**——flow 与 price 是对照的两个独立信号，本研究
判 flow 不如 price，结论就是"用价格即可"，绝不是"把 flow 加进 B2_4h"。
你 flow 框架里的"flow persistence alpha"，**在你够得着的范围里=价格信号已捕获**（且价格
更优）。

> **更深的"为什么"（回到 trend ≈ ∫(flow×impact)dt 的身份）**：你以为读"被积函数"(flow)
> 能比"积分"(price) 更早拿到趋势。但实证上——(a) impact 在任何可达尺度上近乎瞬时（flow
> 与价格在同一根/同一秒内同步，同期 corr 0.51），积分实时跟上被积函数；(b) 被积函数
> （flow 率）比积分（price 水位）**更嘈杂、更快均值回归**——**积分本身**正是把嘈杂的
> flow 转成可收割趋势的操作，所以"结果"(price) 反而是更好的信号，不是可抢先的滞后影子；
> (c) flow 唯一真正领先 price 的地方是亚秒微观结构(HFT)，已被墙死。**"用原因代替结果"
> 颠倒了正确的结构：你要的恰恰是那个累计的积分(price)。**

**Q6 观察节（flow 框架其余部分状态确认）**：
- **exhaustion reversal（fade flow）**：已三重判死（`order_flow_exhaustion_feasibility_20260628`，
  机制 continuation、净 −9.8bps/每延迟）——本研究遵守边界，只测 persistence，未触 reversal。
- **persistence（follow flow）**：本研究判死（≈价格的更差版本，唯一领先在 HFT）。
  ⇒ **order-flow 方向性使用的两侧（fade / follow）现已全部判死**。
- **Liquidity / Impact 层**：数据墙（需 L2/L3 订单簿，免费不可得），不做，状态不变。
- **regime transition**：不可操作（vol 可预测但方向相反、perp 线性退化为方向，见
  `volatility_event`），状态不变。
- **右偏延续**：再获一证——"原因"(flow) 也不能比"结果"(price) 更早够到它；right-skew
  continuation 作为唯一结构性事实，现已从 4h bar→100ms tick→**flow-vs-price 对照**三向确认。

---

## 与项目主线的关系 / 未改动文档

- 本研究为**否定性结论**，与肯定性结论同等固化。它**不**翻转 PROJECT_GUIDE 任何现有
  叙述，而是**新增一条策略层认知**（"flow 信号不优于价格信号；趋势的可收割载体是
  积分后的价格水位，不是 flow 率"）——按 CLAUDE.md 流程在 PROJECT_GUIDE 增一行。
- **未改动**：B2_4h `config_frozen`（原样作对照基准，零触碰）；前向系统；vrp 线；
  污染库；exhaustion 报告（仅引用其 −9.8bps 结论）。**未退化为给 B2_4h 加 flow 过滤器**。

### 由本次研究产生的文档更新

- `PROJECT_GUIDE.md`「已验证的核心事实」在 order-flow exhaustion 条目后**新增一条**：
  "Flow 信号 vs 价格信号（B2_4h）趋势捕获对照——flow（原因）不优于价格（结果），用价格即可"
  （新增认知，非翻转既有叙述，故无 historical 注释；含订单流方向性两侧全死 + 右偏延续再获一证
  + 可收割趋势是积分后的价格水位而非 flow 率）。
- **考虑过但未改动**：`reports/perpetual_signal_space_closure_20260613.md`（永续信号空间收口图）——
  本研究是已闭环主线下的一次"原因 vs 结果"补强对照，未新增/关闭任何出路，不改；exhaustion 报告
  与 `volatility_event` 记忆——仅被本研究引用，结论未变，不改。

## 复现 / 文件

```
./.venv/bin/python scripts/research_flow_vs_price.py          # 全量(~3.5min)
./.venv/bin/python scripts/research_flow_vs_price.py --smoke  # 截断数据接线自检
```
- `scripts/research_flow_vs_price.py` —— flow 信号定义、对照协议、速度检验、判定线写死 docstring。
- `results.json` —— 全部数值（Part1-3 + verdict + notes）。`manifest.json` —— 154 klines +
  18 aggTrades zip 的 sha256（入 git；zip 本体 gitignore）。`figures/` —— 3 图。
  `run_console.txt` / `run_log.txt` —— 运行日志。

**数据环境**：data.binance.vision 公共静态 CDN = Binance 生产/mainnet（构造上无 demo）。
无凭证、无 .env、无 OKX、无 VPS、无污染库、无 vrp 线、无前向系统、未碰 B2_4h config_frozen。

---

Flow vs价格趋势对照完成于2026-06-29T12:58:50Z/flow persistence:前瞻预测区分不出噪声(ACF记忆真实但不前瞻)/flow vs price:不如/信号重合度:低(~0.45-0.53,flow确为不同信号但更差)/速度检验:flow优势全在HFT窗口(亚秒,corr未来+0.056@1s→0@60s)/判定:不过-用价格即可/是否退化为flow过滤B2_4h:否,严守边界/已push:已确认(commit 0aa123a → origin/main,经 SSH-over-443)
