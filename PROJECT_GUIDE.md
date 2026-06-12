# cta_strategy/ — 项目完整说明

## 一句话定位

OKX 永续合约 CTA 策略研发项目，基于 vnpy 框架。经过 13 个月趋势跟踪探索 → MR-v1 (4h) → **MR-5m 均值回归策略**的演进后，2026-06 确认全部历史研究建立在 OKX DEMO 污染行情上，mainnet 重验显示 MR-5m 无 edge，**项目已关闭**（完整复盘：`reports/MR5M_postmortem.md`）。当前阶段：基于 `.vntrader/database_mainnet.db` 干净数据的新策略研究。70+ 个历史报告目录保留为档案；其中全部绩效数字与"死胡同"结论的证据基础已失效（见复盘第 7 节）。<!-- 2026-06-11 更新：原"当前主力，正在 VPS 模拟盘运行/重心收敛到动态仓位对照实验"被 mainnet 基线重验推翻，详见 reports/regime/mr5m_mainnet_baseline_20260611/ -->

## 新策略研究篇章（2026-06 起，报告在 reports/ 根目录新开）

- **第一研究：经典趋势形态裸基线筛查**（2026-06-11，`reports/trend_baseline_20260611/`）：
  15 个预注册经典配置（Donchian / EMA 交叉 / TSMOM × 4h/1d）在 mainnet 全样本上
  **15/15 通过毛利 gate**（taker 双边 + 真实 funding 的保守口径下净利亦全正）。
  性质为**筛查级**证据（零调参全样本），三族均获下一阶段（OOS/邻域/集中度）资格；
  下一阶段必测清单见报告 Q6。
- **第二研究：验证阶段（贝塔分离 + 5 项稳健性 gate）**（2026-06-11，
  `reports/trend_validation_20260611/`）：**VALIDATED 0 / FAIL 15**——V1 单笔集中度
  通杀（剔除 top 5% 交易后毛利全转负）；11/15 同时死于时间稳健性；bootstrap 无一
  显著。但贝塔判定为"混合"而非纯贝塔：B 族与 TSMOM-90d 仅死 V1、风险调整后
  10–17× 优于买入持有（B&H net/DD 0.21 vs 最高 3.54）、在 B&H 亏钱的 DOGE/LINK
  上仍盈利。最接近幸存：B1_4h / B2_4h / C2_4h / C2_1d（中速 4h、年份均衡）。
  下一阶段开放问题：集中度 gate 对趋势形态的"脆弱集中 vs 结构性尾部收割"区分
  （须先预注册）；short 侧 15/15 全亏的 long-only 假设。
- **第三研究：验证第二轮（V1' 集中度质量重审 + long/flat 族 D）**（2026-06-11，
  `reports/trend_validation_r2_20260611/`）：**VALIDATED\* 5 / FAIL 3**。
  V1' 三项预注册检验（跨币分散/滚动可重复/尾部效率）证明四个母配置的集中是
  **结构性尾部收割**（top5% 跨 5 币 3-4 年、100% 窗口可重复），非脆弱集中——
  B1_4h/B2_4h/C2_4h/C2_1d + D2（B2 long/flat）过堂。D1/D3/D4 死于 V5 年份集中
  （去 short 腿后 2023 占比 62-86%）。关键机制发现：**short 腿全期净亏但在 2025
  下跌年贡献 ~$16k，是时间分散器**——保险费买到了赔付。星号规则：V1' 系事后
  设计的修正检验，证据等级永久低于一次通过；原 V1 判定（0/15）永久存档。
  下一阶段（已完成，见第四研究）：组合构建 + 组合层 bootstrap 预注册。
- **第四研究：组合构建（相关结构 + 4 个预注册组合过堂）**（2026-06-11，
  `reports/trend_portfolio_20260611/`）：**PORTFOLIO-CANDIDATE\* 0 / 4——全部死于
  P1 显著性**（日收益 bootstrap CI 全含 0，t 0.89-1.03）。相关结构诊断：幸存池
  5 配置实质 **≈2 个独立信号**（PC1+PC2 = 92.4% 方差，ENB 1.72；C2_4h/C2_1d 日相关
  0.993 实为同一信号；D2 持仓系 B2_4h 多头腿严格子集，B2+D2 同收是 β 倾斜非分散）。
  P2/P3 形态层全部轻松通过（maxDD 仅为 0.5×B&H 上限的 21-23%、滚动 12 月 87-93% 正），
  但年化 Sharpe ~0.5 在 3.4 年样本下数学上不可能过 95% 显著性（需 ~15-19 年）。
  2025 至今近 18 个月四组合三负。剩余选项（用户决策）：接受终局 / 零成本前向观察
  （本地信号 + 模拟记账，不用已知失真的 OKX demo 撮合）。
- **第五研究：双周期扩展验证（Binance 2020-2026 独立样本全量复测）**（2026-06-12，
  `reports/trend_dualcycle_20260611/`）：**DUAL-VALIDATED\* 2（B2_4h、D2）/
  R-VALIDATED\* 0 / FAIL 6（全部死于 V5）**。数据快检 PASS（47 起 ramp 事件
  全部对照公开记录确认真实，vs demo 库 598 起；funding 手算自检 0 差）。
  慢配置 V5 死因从"2023 占比"换成"2020 占比 0.83-0.88"——B1_4h 单笔 DOGE 多单
  净 $1.01M 占其总净利 81%，**利润前载到最大趋势事件是慢配置的结构属性**，
  D1/D3/D4 维持死刑。2022 深熊：short 腿四配置全部正赔付（+$10k~+$21.5k），
  净亏压到 ≈0 vs B&H −$1,014k。重叠期两所一致性 +1%~+7%（近乎逐笔复现，
  前四阶段结论对交易所选择不敏感）。最终候选池收敛为**一个信号结构
  （4h EMA20/100）的两种形态**，首选含 short 腿的 B2_4h。
- **趋势线终局：资源决策关闭**（2026-06-12，`reports/trend_line_closure_20260612.md`）：
  B2_4h 的前向验证周期（升级 gate 需满 18 个月起步，完整统计证明 ~15-19 年）
  超出可接受的资源配置，趋势研究线就此关闭。**性质为资源决策，非信号证伪**
  ——支持面（双周期双所全 gate 存活、2022 熊市 short 腿正赔付、6y bootstrap
  排除 0）与反对面（3.4y 日收益不显著、近 18 个月组合层无利润、V1' 系事后修正）
  在关闭日同时为真，证据现状与重启条件（第二信号族低相关幸存者使 ENB 提升 /
  项目进入多线并行阶段）见归档文档。前向观察系统不再开发部署（gate 设计模板
  已归档备用）。研究主线转向**第二信号族选题**，选题标准新增"验证周期"维度
  （已入 CLAUDE.md 工作原则）。
- **第六研究：价格回归的时间尺度结构（纯描述性前置研究，立项判定）**
  （2026-06-12，`reports/mr_timescale_structure_20260612/`）：**不立项，全尺度**。
  预注册口径（E1 通道突破 N=20 / E2 ±2σ，k=1..16 bar，厚度线 = median ≥ 2×全成本，
  ≥3/5 币支撑，12/24 个月验证周期线）下度量 15m/30m/1h/2h/4h：回归**率**全尺度
  显著高于基线（+1~7pp，随尺度衰减、平滑无尖峰），但 15m–2h 全部 40 格 median
  幅度不过成本线；4h 唯一双样本厚度可行格（E1·k=8）**期望为负**（mean −0.07%
  OKX / −0.11% Binance，成本前）——分布左偏，少数延续尾部吃掉多数小回归。
  验证周期铁律首次执行：开题前一天的描述性统计终结一整类纠结。5m 终审未重测。

---

## 目录结构

```
cta_strategy/
├── strategies/          # vnpy CTA 策略模板（回测用）
├── scripts/             # 研究脚本、回测脚本、实盘 runner、数据工具
├── config/              # 合约规格、运行时配置、策略参数
├── data/                # 资金费率历史 + K线下载清单 + 原始数据
├── reports/             # 所有研究产出（markdown + JSON + CSV）
├── tests/               # 单元测试
├── logs/                # 脚本运行日志
├── deploy/              # 部署目录（systemd 配置在 VPS 本机维护，仓库内为空目录）
├── .vntrader/           # vnpy 引擎数据（1.4GB sqlite 数据库 + 策略设置）
├── Makefile             # 统一入口（1137 行，100+ 个 target）
├── CLAUDE.md            # AI 助手的项目约束和记忆
└── README.md            # 项目说明
<!-- 2026-06-11 更新：task.txt（早期开发任务清单，已过时）归档至 reports/archive/task.txt -->
```

---

## 核心策略：MR-5m 均值回归

### 策略逻辑

**一句话**：fade Donchian 突破。价格突破上轨做空，突破下轨做多，赌价格回归中轨。

**参数**：
| 参数 | 值 | 含义 |
|------|-----|------|
| LB | 24 | Donchian 通道 lookback（2 小时，5m×24） |
| ATR_WINDOW | 14 | Wilder ATR 周期 |
| ATR_STOP | 1.0 | 止损 = 入场价 ± 1.0×ATR |
| MAX_HOLD | 48 | 最大持仓 48 根 bar（4 小时） |
| NOTIONAL_PER_TRADE | $500 | 每笔名义金额 |
| LEVERAGE | 5x | 杠杆 |
| 币种 | BTC/ETH/SOL/LINK/DOGE | 5 个，方案 A |

**出场优先级**（严格顺序，不可调换）：
1. **midline**：价格回归 Donchian 中轨 → 止盈
2. **stop**：触发 ATR 止损（当根 bar 触发即出）
3. **max_hold**：持仓超 48 根 bar → 强制平仓

**入场方向反转**：
- 价格突破 Donchian 上轨 → **做空**（赌回落）
- 价格突破 Donchian 下轨 → **做多**（赌反弹）
- 有一个 ATR regime filter：ATR 低于阈值时不入场，过滤掉死的低波动 chop 信号。它是 **whipsaw 过滤器 / 温和质量精修**（OOS +1.9% 组合净利 / +0.08 PF），**不是风控、不降回撤**——filter on/off 组合 Max DD 完全相同（$2,250）。<!-- 2026-06-09 更新：原 backtest_mr_5m_v2.py 把它定位为"风控/降回撤（−72% Max DD）"，该数字是 in-sample artifact，OOS 不复现已撤回；详见 reports/regime/atr_filter_oos_validation_20260609/ -->

### 两个实现

| 文件 | 用途 | 行数 |
|------|------|------|
| `strategies/mr_5m_strategy.py` | vnpy CTA 模板，用于回测引擎 | 405 |
| `scripts/run_mr_5m_direct.py` | 独立 runner，直连 OKX WebSocket，**实盘/模拟盘用** | 1269 |

**两者的关系**：
- `mr_5m_strategy.py` 继承 `CtaTemplate`，在 vnpy 回测引擎内运行，使用 `ArrayManager` 管理 K 线
- `run_mr_5m_direct.py` 完全绕开 vnpy GUI，自己连接 OKX WebSocket、自己聚合 1m→5m bar、自己算 Wilder ATR、自己通过 REST API 下单、自己做仓位校验（pos_verify）、自己写 trade_log
- 两者的信号逻辑完全一致（同一套入场/出场规则），但 runner 多了大量工程化代码

### runner 的工程架构

`run_mr_5m_direct.py` 不是一个简单的脚本，是一个完整的交易系统：

```
核心类：
  BarAggregator     — 1m→5m K线聚合 + Wilder ATR + Donchian + 仓位追踪
  Notifier          — PushPlus 微信通知（异步队列，3 次重试，HTTPS）
  OkxWsClient       — OKX WebSocket 连接管理

关键函数：
  okx_sign/okx_headers/okx_get/okx_post  — OKX REST API（DEMO，x-simulated-trading: 1）
  place_order()     — REST 下单（IOC 限价单，2026-06-07 起）
  calc_size()       — 整数张数计算（ctVal 取整）
  get_fills_fee()   — 查询实际手续费（从 OKX fills API）
  sync_positions_from_okx()  — 启动/重连时从 OKX 同步仓位
  pos_verify()      — 每 5 分钟对比本地仓位 vs OKX 实际仓位，发现 drift 则平仓

关键机制：
  - pos_verify：每 5 分钟跑一次。如果 OKX 有仓位但本地没有 → 市价平掉（exit_reason="pos_verify"）
  - 如果 OKX 仓位 ≠ 本地仓位 → 市价平掉（exit_reason="pos_verify"）
  - 这是为了防止本地状态与交易所不同步导致的风险
  - trade_log 月度轮转：trade_log_5m_YYYYMM.jsonl
  - 每 6 小时输出一次摘要统计
```

---

## 研究历程：从趋势跟踪到均值回归

### 时间线

```
2024 Q4 — 2025 Q1    趋势跟踪 V2 → V3 → V3 扩展
                     ├─ 大量参数扫描、多币种测试
                     ├─ 全部事后分析
                     └─ 结论：趋势策略不可行，全部未通过 gate

2025 Q2              跨币种信号探索
                     ├─ CSRB-v1 / VSVCB-v1 / 早期趋势分类器
                     ├─ 外部 regime 分类器
                     └─ 结论：全部未通过 gate

2025 Q2 — Q3         MR-v1 (4h 均值回归)
                     ├─ Phase 1: 基础研究 → 通过
                     ├─ Phase 2: 参数扫描 → 通过
                     ├─ Phase 3: 稳健性验证 → 通过
                     ├─ 回测 Sharpe 3.82, PF 5.38, 36/36 月盈利
                     ├─ 进入模拟盘
                     └─ 结论：实盘表现不达预期 → 转向 5m

2025 Q3 — 2026 Q1    MR-5m 深度研究
                     ├─ 方案选择（A/B/C/D）
                     ├─ 出场优化（Chandelier/追踪止损/熔断器）
                     ├─ 过滤器研究
                     ├─ regime 分析
                     └─ 最终收敛到 C2-1 动态仓位

2026-06-01           双账号模拟盘对照实验启动（VPS）
2026-06-07           从 GTC limit 切为 IOC
当前                 持续运行中
```

### 关键研究发现（不可遗忘）

> ⚠️ **本节全部数字来自 DEMO 污染库**（取证：reports/regime/data_contamination_forensics_20260610/）。
> mainnet 基线重验显示 FLAT 在真实数据上**无 edge**（毛利≈0，net PF 0.83-0.85，详见
> reports/regime/mr5m_mainnet_baseline_20260611/）——以下各条的**机制叙述**逐条待用户裁定，
> **金额/PF 数字一律不可引用**。

1. **whipsaw 才是 MR 的敌人，不是 trend**。BTC 涨幅最大的月份反而是 MR 最赚钱的月。失效月是低效率/低波动/高 stop% 的震荡月。"趋势强就停做"的过滤思路是错的，因为强趋势 = 大振幅 = 中轨止盈多。

2. **反应式状态熔断器无效**（74 个配置全部负收益）。失效月总亏损仅 -$10,690（占总盈利 ~3%），事后暂停误伤的健康利润 > 省下的亏损。"靠少交易让指标变好"是陷阱。

3. **中轨止盈是唯一有效的出场优化**（+$6.3k）。Chandelier/追踪止损全部失败。**不再碰 exit 优化**。

4. **保留全部 5 币种（方案 A），DOGE 不剔除**。移除 DOGE 抹掉 ~$78k 利润而 PF 几乎不变。DOGE 全周期 PF 1.67（第二强）。

5. **「2024 是全市场 MR 逆风年」是 DEMO 数据伪象，mainnet 不重现**：真实数据上各年 PF 均匀地差（0.83-0.89），2023 反而最高，无 2024 特异性。<!-- 2026-06-11 更新：原"2024 全市场逆风年"叙述基于 DEMO 库，被 mainnet 基线重验推翻，详见 reports/regime/mr5m_mainnet_baseline_20260611/ -->

6. **失效月亏损仅占总盈利 ~3%**，优化空间在健康月份的 97%——已落实到 C2-1 动态仓位（按 atr_ratio 三档加注；atr_ratio 是波动率**动量**，OOS 验证显示它只是 ATR **绝对水平**的代理变量，真正的 regime 因子是绝对水平，详见 `reports/regime/atr_ratio_decomposition_20260609/`）。<!-- 2026-06-09 更新：DYN 押对了结果但用错了原因——atr_ratio 三档在 test 单调成立，但靠代理变量间接吃 edge，详见 reports/regime/atr_ratio_decomposition_20260609/ -->

### 已验证的核心事实（OOS/回测充分支持）

> 补充 `关键研究发现`，专列被 OOS/回测充分支持的策略层结论。

- **MR-5m FLAT 在 mainnet 真实数据上无 edge**：C1（无过滤）/C2（现行阈值）/C3（mainnet 重导出 p40）全亏——全期 PF 0.81-0.85、test PF 0.81-0.83；**毛利≈0，净亏 100% 是手续费**；5 币 × 4 年共 20 格无一 PF>1（最高 SOL 2023 = 0.99）；胜率 36%（DEMO 假象为 48%）。DEMO 的 FLAT test +$189k / PF 2.06 中**毛利 edge ~99% 为数据假象**（笔数反而 +19%，虚增的是单笔质量 −$9.98/笔）。ATR filter 只能少亏（少交易→少手续费），不能创造 edge。[`reports/regime/mr5m_mainnet_baseline_20260611/`]
- **ATR filter 是 whipsaw 过滤器**（OOS +1.9% PF），**不是风控/降回撤**。[`reports/regime/atr_filter_oos_validation_20260609/`]
- **真正的 regime 因子是 ATR 绝对分位（波动率水平）**：5/5 币种一致，PF 跨度 0.49→2.84、胜率 33.5%→60.3%，每个动量档下都稳健单调。[`reports/regime/atr_ratio_decomposition_20260609/`]
- **DYN 的 atr_ratio 是绝对水平的代理变量**：C2-1 三档单调在 test 成立但有隐患——"低绝对水平 + 高 atr_ratio"格子 PF 仅 0.24（负期望），而 DYN 恰在该区触发 large 档 $750。[同上]
- **v2B 的 max DD 元凶已识别：2025-05-29 SOL/DOGE 合成数据 ramp（数据异常，非市场模式）**。窗口 16 笔中前 12 笔仅 −$10，99.7% 的 DD 来自 15:54–16:14 双向 fade 合成直线 ramp 的 4 笔（每分钟等步长 ±10.3 阶梯、单根 K 线 ±91% 瞬间复原、同时刻 BTC/ETH/LINK 无波动、真实市场无对应记录）。此前排除的两个假设仍成立：(1) **非协同行情**（5/5 同向并发 1,005 笔净正 +$3,009 [`reports/regime/portfolio_risk_phase_2a_20260609/`]）；(2) **非极端 ATR 单笔**（size cap V1 切中率仅 12.5–18.8% [`reports/regime/dyn_v2b_size_cap_v1_20260609/`]）。详见 [`reports/regime/v2b_dd_diagnosis_20260610/`]。<!-- 2026-06-10 更新：原"元凶未识别、16 笔中等 ATR 集群亏损"叙述被诊断推翻——是数据异常，详见 reports/regime/v2b_dd_diagnosis_20260610/ -->
- **回测数据库含高频合成 ramp 异常，净利与 max DD 数字双向污染**：全期（2023-01→2026-05）检出 598 个 ≥10% 直线 ramp/单根跳变事件——SOL 344 / DOGE 216 / ETH 23 / LINK 15 / **BTC 0**，70% 在 4h 内完全复原。与异常窗口重叠的交易贡献 v2B test 净利的 **11.5%（仅等差阶梯签名）～31.6%（全部检出）**，FLAT 同量级（32.4%）；异常通常**送钱**（V 形瞬间复原 → midline 止盈），2025-05-29 是唯一一次双向打穿止损的最不利实现。**v2B 1.523× DD 的分子分母均不可信；以 SOL/DOGE 为主要利润源的历史结论效应量需在数据修复后重审。** [`reports/regime/v2b_dd_diagnosis_20260610/`]

- **经典趋势原型在 mainnet 真实数据上普遍有毛利（筛查级，2026-06-11）**：15 个预注册
  零调参配置（Donchian 20/10、55/20、100/50 / EMA 50/200、20/100 / TSMOM 30/90/180d
  × 4h/1d）全样本 2023-01→2026-05 毛利全正、保守成本口径（taker 双边 + 真实 funding）
  下净利亦全正（PF净 1.03–1.89）；demo 时代"趋势跟踪全败"被方向性证伪。
  [`reports/trend_baseline_20260611/`]
- **15/15 未通过含原 V1 的首轮验证；V1' 重审后 5 配置 VALIDATED\*（2026-06-11）**：
  原 V1（剔除 top5% 后毛利>0）通杀 15/15，但被论证为对趋势形态的 category error
  （该类策略的利润分布设计上即为尾部收割）；预注册的 V1'（分散性/可重复性/尾部
  效率）证明 B1_4h、B2_4h、C2_4h、C2_1d 的集中为**结构性**（top5% 跨 5 币 3-4 年、
  100% 滚动窗口可重复）——这四个 + D2（B2_4h long/flat）= **VALIDATED\***。
  星号永久标注：V1' 系事后设计的修正检验，证据等级低于一次通过；原 V1 判定
  永久存档不回改。其余事实不变：bootstrap 单笔均值无一显著；short 腿全期净亏
  但为 2025 时间分散器；B&H 基准净 $64k / maxDD $313k / funding 吃毛利 46%。
  [`reports/trend_validation_20260611/` + `trend_validation_r2_20260611/`]
- **趋势幸存池 ≈2 个独立信号；组合层无统计支持（2026-06-11）**：5 个 VALIDATED*
  配置的日收益相关结构显示真实分散度仅 ~2 个信号（EMA-cross 簇 + TSMOM-90d 簇）；
  4 个预注册组合 0/4 过日收益 bootstrap 显著性 gate（t ≤1.03），盈利全集中
  2023-2024、2025 至今≈0。形态层（回撤/时间稳健）全过，但显著性缺口是样本长度
  对 Sharpe ~0.5 的数学约束，换组合方式解决不了。D2 与 B2_4h 不应同收
  （前者持仓为后者多头腿子集）。[`reports/trend_portfolio_20260611/`]
- **B2_4h 与 D2 通过 Binance 2020-2026 双周期独立复测（2026-06-12）**：
  全 gate（冻结）过堂，bootstrap CI 排除 0（B2_4h [139,623] n=836 最可信）；
  其余 6 配置（含 TSMOM 簇全部）死于 V5——慢趋势配置利润前载到单一最大
  趋势事件（B1_4h 单笔 DOGE $1.01M 占 81%）是**结构属性而非样本巧合**。
  2022 真实熊市中 short 腿正赔付、whipsaw 让 long/flat 多亏 ~$19k →
  含 short 腿的 B2_4h 为首选形态。两所重叠期偏差仅 +1~7%。注意：此结论
  回答"信号在加密市场稳健"，不推翻组合阶段对 2023+ 日收益显著性的否定。
  [`reports/trend_dualcycle_20260611/`]
- **趋势线终局记录（2026-06-12）**：趋势线以**资源决策**关闭，非信号证伪——
  双周期双所证据存活（B2_4h DUAL-VALIDATED\*）与 3.4 年显著性缺口并存，
  验证周期（前向 18 个月起步 / 完整证明 ~15-19 年）与项目资源配置不匹配；
  重启条件已预留。[`reports/trend_line_closure_20260612.md`]
- **15m–4h 价格回归无可收割期望（双样本，2026-06-12）**：回归作为频率现象
  真实存在（回归率全尺度显著 > 基线，平滑衰减），但回归幅度 median 在
  15m–2h 全部不过 2×成本线；4h 唯一双样本过线格（E1 通道突破·k=8）的
  事件级 mean 为负（OKX −0.071% / Binance −0.110%，成本前）——左偏分布，
  延续尾部毁灭期望（恰是趋势策略存活的同一结构）。不存在"5m 不行但更高
  尺度行"的中间地带；中频 MR 线不立项。[`reports/mr_timescale_structure_20260612/`]

### 已知未验证的假设（明确标注：未验证）

> 关键假设，尚无回测/OOS 支持；做完后从此处移入"已验证"或撤回。

- **DYN-v2**：按 ATR 绝对分位（或二维 atr_ratio×绝对分位）重新分档是否优于当前 C2-1？[未验证，待 walk-forward]
<!-- 2026-06-10 更新："v2B 的 max DD 真实元凶是什么？[待诊断]"已验证完毕并移入上「已验证的核心事实」——元凶为 2025-05-29 SOL/DOGE 合成数据 ramp，详见 reports/regime/v2b_dd_diagnosis_20260610/（该条目历史注释：原"组合层风控回撤抑制？[未验证]"经 phase_2a 测得 R1/R2/R3 无效后转为"元凶未识别"，再经本次诊断闭环） -->
- **v2B 的 ≤1.5× 绝对 DD 硬门槛是否过严？** 放宽为"DD/net 不恶化"可能更合理（v2B 的 DD/net 与 FLAT 相同 ~0.0118，回撤是收益等比放大）——[待方法论审查，非回测]
- **实盘可分辨性**：A/B 在执行污染下能否分辨 DYN vs FLAT 的策略差异？[DEMO 数据已部分证否，待真金小额复测]

---

## 研究产出（reports/ 目录）

### 趋势跟踪系列（全部未通过 gate）

| 目录 | 内容 |
|------|------|
| `trend_following_v2/` | V2 趋势策略（train/val/oos 三 split） |
| `trend_following_v3/` | V3 趋势策略（同上） |
| `trend_following_v3_extended/` | V3 扩展历史（full/oos/train/val 四 split） |
| `trend_following_v3_actual_funding/` | V3 真实资金费率调整分析 |
| `trend_following_v3_postmortem/` | V3 失败事后分析（含 v3.1 建议） |
| `trend_following_v2_compare/` | V2 多方案对比 |
| `trend_following_v3_compare/` | V3 多方案对比 |
| `trend_following_v3_extended_compare/` | V3 扩展版对比 + 资金费率压力测试 |

### 趋势诊断系列

| 目录 | 内容 |
|------|------|
| `trend_regime_diagnostics/` | 趋势环境诊断（ex-post regime 标签） |
| `trend_opportunity_map/` | 趋势机会地图（旧策略为何没抓住） |
| `trend_entry_timing/` | 入场时机研究 |
| `trend_entry_timing_postmortem/` | 入场时机事后分析 |
| `trend_capture_exit_convexity/` | 出场凸性研究（oracle hold 上限对比） |
| `trend_health_state_exit/` | 四维健康状态出场（效率+能量+回撤+时间） |

### 跨币种信号系列（全部失败）

| 目录 | 内容 |
|------|------|
| `csrb_v1/` + `csrb_v1_postmortem/` | Cross-Symbol Reversal Breadth |
| `vsvcb_v1/` + `vsvcb_v1_postmortem/` | VSVCB |
| `early_trend_classifier_v1/` + `_inverse/` | 早期趋势分类器 |
| `cross_symbol_breadth_phase15/` | Phase 15 跨币种广度 |
| `external_regime_classifier/` + `_feasibility/` + `_gate_audit/` | 外部 regime 分类器 |
| `htf_signals/` + `htf_compare/` | 高时间框架信号 |

### MR 策略专项研究

| 目录 | 内容 |
|------|------|
| `mr_v1/` + `_phase2/` + `_phase3/` | MR-v1 三阶段研究 |
| `mr_v1_filter/` + `_midline/` + `_chandelier/` | 过滤器/中轨/Chandelier |
| `mr_v2_keltner/` | MR-v2 Keltner 通道 |
| `mr_5m_diagnostics/` | MR-5m 诊断图表 |
| `regime/` | 动态仓位研究 + regime 分析 + 信号质量 |

### 回测与实验

| 目录 | 内容 |
|------|------|
| `backtest/main_cost_20250101_20260331/` | 主力回测（含归因分析） |
| `backtest/main_no_cost_20250101_20260331/` | 无成本回测 |
| `backtest/sanity_20250101_20250102/` | 健全性检查 |
| `alpha_sweep/main_20250101_20260331/` | 10 个候选参数扫描 |
| `ablation/oos/` + `train/` + `validation/` | 消融实验（9 个候选 × 两档成本） |
| `backtest_compare/` | 多方案对比数据 (jsonl) |

### 数据工程

| 目录 | 内容 |
|------|------|
| `multisymbol_readiness/` | 多币种数据就绪度 |
| `extended_history_availability/` | 扩展历史可用性 |
| `derivatives_data_readiness/` | 衍生品数据就绪度 |
| `funding/` + `_endpoint_probe/` + `_historical_download/` | 资金费率全流程 |

---

## 脚本分类（scripts/）

### 实盘 Runner（只在 VPS 运行）
```
run_mr_5m_direct.py       — MR-5m 直连 OKX WebSocket（★ 核心实盘脚本）
run_mr_v1_direct.py       — MR-v1 直连（已停用）
run_mr_5m_demo.py         — 模拟盘入口
run_mr_v1_demo.py         — MR-v1 模拟盘（已停用）
```

### 回测脚本
```
backtest_mr_5m.py              — 基础回测
backtest_mr_5m_compare.py      — 多方案对比（★ 主力回测引擎）
backtest_mr_5m_dynsize.py      — 动态仓位回测
backtest_mr_5m_dynsize_c2c3.py — C2/C3 walk-forward
backtest_mr_5m_breaker.py      — 熔断器回测
backtest_mr_5m_v2.py           — V2 版回测
backtest_mr_v1.py              — MR-v1 回测（已停用）
backtest_okx_mhf.py            — OKX MHF 回测
```

### 研究脚本（26 个 research_*.py）
按研究主题分：
- MR 策略：`research_mr_5m.py` / `_deep.py` / `_deep_v3.py` / `research_mr_v1*.py` / `research_mr_v2_keltner.py`
- 趋势跟踪：`research_trend_following_v2/v3.py` / `research_trend_opportunity_map.py` / `research_trend_entry_timing.py` / `research_trend_capture_exit_convexity.py` / `research_trend_health_state_exit.py`
- 跨币种：`research_csrb_v1.py` / `research_vsvcb_v1.py` / `research_early_trend_classifier_v1.py`
- 其他：`research_cross_symbol_breadth_phase15.py` / `research_htf_signals.py` / `research_entry_policies.py` / `research_exit_only_bc.py` / `research_signal_features.py` / `research_external_regime_classifier.py` / `research_5min_reversal.py`

### 事后分析脚本（4 个 postmortem_*.py）
```
postmortem_csrb_v1.py
postmortem_vsvcb_v1.py
postmortem_trend_following_v3.py
postmortem_trend_entry_timing.py
```

### 分析/审计脚本
```
analyze_alpha_diagnostics.py
analyze_backtest_report.py
analyze_mr_regime_2024.py
analyze_mr_signal_quality.py
analyze_signal_outcomes.py
analyze_trade_attribution.py
analyze_trend_v3_actual_funding.py
audit_mr_5m.py
audit_multisymbol_readiness.py
audit_extended_history_availability.py
audit_external_regime_classifier_feasibility.py
audit_external_regime_classifier_gates.py
audit_okx_derivatives_data_readiness.py
diagnose_mr_5m.py
diagnose_trend_regimes.py
```

### 数据工具
```
download_history.py              — vnpy 历史数据下载
download_okx_history.py          — OKX 历史 K 线下载（2455 行，最复杂的下载脚本）
download_okx_funding_history.py  — 资金费率下载
download_okx_historical_funding_files.py — 历史资金费率批量下载
import_okx_funding_csv.py        — 资金费率导入
merge_recent.py                  — 增量合并
verify_okx_history.py            — 数据完整性验证
verify_okx_funding_history.py    — 资金费率数据验证
```

### 实验运行
```
run_ablation_experiments.py      — 消融实验批量运行
run_alpha_sweep.py               — 参数扫描批量运行
```

### 对比脚本
```
compare_trend_following_v2.py
compare_trend_following_v3.py
compare_htf_signal_research.py
compare_signal_feature_research.py
```

### 运维工具
```
doctor.py           — 项目健康检查
check_okx_connection.py — OKX 连接测试
close_positions.py  — 紧急平仓
inspect_okx_gateway.py — OKX 网关检查
build_research_decision_dossier.py — 研究决策汇总
```

---

## Makefile 体系（1137 行，100+ targets）

按功能域分组：

| 功能域 | 代表 target | 用途 |
|--------|-----------|------|
| 环境 | `venv` `install` `doctor` | 虚拟环境、依赖安装、健康检查 |
| 数据下载 | `download-history` `download-funding` | K线 / 资金费率下载（含 dry-run） |
| 数据验证 | `verify-history` `verify-funding` | 完整性校验 |
| 回测 | `backtest` `backtest-no-cost` `backtest-trace` | 主力回测 / 无成本回测 / 信号追踪 |
| 研究 | `research-mr-*` `research-trend-*` `research-htf` 等 | 各研究方向入口 |
| 事后分析 | `postmortem-trend-v3` `postmortem-csrb-v1` 等 | 失败策略的事后分析 |
| 对比 | `compare-trend-v2` `compare-trend-v3` | 多方案对比 |
| 分析 | `analyze-alpha` `analyze-signals` `analyze-trades` | 诊断分析 |
| 消融 | `ablation` `alpha-sweep` | 参数稳健性实验 |
| 审计 | `audit-multisymbol` `audit-extended-history` 等 | 数据/环境审计 |

---

## 数据库和配置

### 数据库与数据资产
<!-- 2026-06-11 更新：原"database.db 为主数据库"叙述随污染确认与修补改名而重写 -->
- **`.vntrader/database_mainnet.db`（唯一可信回测源，1.41 GB）**：5 币 × 1m，
  2023-01 → 2026-05，每币 1,791,360 根零缺口，带 `download_meta` 来源元数据；
  已通过 Binance 全量交叉验证（`reports/regime/data_trust_closure_20260611/` PASS）；
  备份与 SHA256 见 CLAUDE.md 数据铁律节。
- `.vntrader/database_DEMO_CONTAMINATED.db`（原 database.db）：已确认 OKX DEMO 污染，
  仅作取证基准，严禁研究使用。
- `data/funding/okx/`：5 币 8h 资金费率 2023-01 → 2026-06-11，已正式核查（同上报告）。
- `data/binance_vision/`：Binance UM 永续 1m 月度镜像（205 文件，sha256 全验），
  仅作交叉验证源，不可作 OKX 回测价格源。

### 合约规格
`config/instruments/` — 7 个币种的 OKX 合约参数（JSON），每个包含：
- `vt_symbol`、`size`（合约面值 ctVal）、`pricetick`（最小价格变动）、`min_volume`

### 策略配置
`.vntrader/cta_strategy_setting.json` — MR-v1 的策略参数设置（已不用）
`.vntrader/strategy_default.json` / `strategy_mr_v1.json` — 策略模板配置

### 历史数据清单
`data/history_manifests/` — 30 个 JSON 文件，记录每段历史数据的下载参数和状态

### 资金费率数据
`data/funding/okx/` — 5 个币种的资金费率 CSV（2023-01 ~ 2026-03）
`data/funding/okx_historical_raw/` — 原始下载的 zip 包（逐月）

---

## 测试

`tests/` 目录包含 45 个测试文件，覆盖几乎所有研究脚本。测试风格是集成测试（运行脚本 + 检查输出文件），不是单元测试。

---

## 项目约束（CLAUDE.md 核心原则）

研究工作中必须遵守的原则：
- **不基于直觉/短期实盘样本拍板**。统计显著性优先
- **数据不支持就诚实说不支持**，不强行给积极结论
- **不做参数过拟合**。所有方案用同一段完整历史
- **保守优于激进**
- **不修改实盘脚本**（`scripts/run_mr_5m_direct.py` 禁止修改）。研究产出独立脚本 + markdown 报告 + 中间 jsonl
- **回测引擎 `backtest_mr_5m_compare.py` 是保真复刻实盘的基准**，复用其数据加载与指标

已确认的边界：
> ⚠️ 2026-06-11：以下"边界"全部基于已确认污染的 DEMO 数据得出（含否定性结论），
> 在 mainnet 干净数据上的有效性未知，**不再作为新研究的硬边界**，保留为历史记录。
> 详见 `reports/MR5M_postmortem.md` 第 7 节。
- **不再碰 exit 优化**（Chandelier/追踪止损全部失败）
- **不再碰反应式熔断器**（74 个配置全部负收益；组合层 R1 日亏损熔断同机制再证，误伤 +$33k–$45k）[`reports/regime/portfolio_risk_phase_2a_20260609/`]
- **组合层 R2 同向仓位上限 / R3 横截面降档对 v2B max DD 无效**：非反应式陷阱（误伤≈0），但安全阈值下 DD 不动——因 v2B 的 DD 非协同造成，与 R2/R3 适用场景不匹配 [同上]
- **不再研究趋势跟踪**（V2/V3 全部未通过 gate）<!-- 2026-06-11：该 demo 时代结论已被干净数据方向性证伪——经典趋势原型 15/15 过毛利 gate（筛查级），见 reports/trend_baseline_20260611/ -->
- **不再研究跨币种信号**（CSRB/VSVCB/ETC 全部未通过 gate）
- **不再碰"极端 ATR 单笔 size cap"降 v2B 回撤**（V1 硬截断 15/15 配置 PnL 全部不达标；高 ATR 笔是利润引擎非 DD 元凶，DD 切中率仅 12.5–18.8%；详见 `reports/regime/dyn_v2b_size_cap_v1_20260609/`）
- 当前研究状态：**MR-5m 项目已关闭**；**趋势线已关闭**（2026-06-12 资源决策，非证伪，重启条件见 `reports/trend_line_closure_20260612.md`）；研究主线 = **第二信号族选题（待开题）**，选题标准含验证周期维度（启动门槛：复盘第 8 节检查清单 + CLAUDE.md 验证周期原则）。<!-- 2026-06-11 更新：原"唯一在跑方向：动态仓位（双账号对照实验）"随项目关闭而终止 --><!-- 2026-06-12 更新：原"新阶段为新策略研究"细化——第一条线（趋势）已走完五阶段，以资源决策关闭 -->

撤回的说法（曾被采信，已被 OOS 推翻）：
- ~~"ATR filter 是 −72% Max DD 风控机制"~~ —— in-sample artifact，OOS 不复现，已撤回（`reports/regime/atr_filter_oos_validation_20260609/`）。
- 注：本次 atr_ratio 拆解未推翻任何研究**方向**，无新增死胡同（DYN 是 rework 候选，非死路）。
