# cta_strategy/ — 项目完整说明

## 一句话定位

> **项目已收口（2026-06-14）**：盒内信号空间系统探明完毕，无可立项策略；完整项目总结见
> [`reports/PROJECT_FINAL_SUMMARY_20260614.md`](reports/PROJECT_FINAL_SUMMARY_20260614.md)
> （阶段一+二历程、探明地图、方法论资产、盒外出路与留门、数据资产清单）。本说明继续作为
> 活文档维护。

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
  项目进入多线并行阶段）见归档文档。前向观察系统**已于 2026-06-16 开发完成、
  待用户手动部署 VPS**（B2_4h 原样零优化/零过滤/零参数变体、gate 路径预注册冻结、
  每日 PushPlus、伪前向演习逐分钟差额 $0、引擎零修改外层包装、数据源恒 mainnet 公开
  行情无 demo/无凭证/无下单、三角色分离开发/部署/运行），见 `forward/`（README + config_frozen
  + gates_preregistered + VPS_DEPLOYMENT_MANUAL + dry_run_validation）。研究主线转向
  **期权 VRP（学习中）**，选题标准新增"验证周期"维度（已入 CLAUDE.md 工作原则）。
  <!-- 2026-06-16 更新：原"前向观察系统不再开发部署"被推翻——已开发完成待手动部署；
       原"研究主线转向第二信号族选题"细化为转向期权 VRP -->`forward/` 详见 README 顶部 SHA256。
  <!-- 2026-06-22 校正（方法论加固，reports/trend_methodology_hardening_20260622/）：
       本条两处表述经诚实重算精修——① "V1' 系事后修正"更准确的定性是"事后移动球门 /
       程序污点"（预注册 V1 判死 15/15 后重定义 gate 复活幸存者），不是"自我修正"，
       幸存证据等级因此降低；② "完整证明 ~15-19 年"系**未打折 + iid**的乐观估计：对
       B2_4h 单配置做 Bailey-LdP 多重检验打折（观测年化 Sharpe 0.655 = 15 配置最高，
       去偏后 0.510 有效N / 0.267 名义N / 0.162 扩展N下界），从对自相关诚实的 bootstrap
       SE 反推的诚实验证周期为 14 年（有效N）/ 51 年（名义）/ 138 年（扩展下界）。
       自相关修正实测≈0（日 M2M 收益 ACF≈0），周期拉长几乎全部来自选择偏差打折。
       关闭的核心算术（十年级周期）被证实，部署现状不变；详见报告。 -->
- **第六研究：价格回归的时间尺度结构（纯描述性前置研究，立项判定）**
  （2026-06-12，`reports/mr_timescale_structure_20260612/`）：**不立项，全尺度**。
  预注册口径（E1 通道突破 N=20 / E2 ±2σ，k=1..16 bar，厚度线 = median ≥ 2×全成本，
  ≥3/5 币支撑，12/24 个月验证周期线）下度量 15m/30m/1h/2h/4h：回归**率**全尺度
  显著高于基线（+1~7pp，随尺度衰减、平滑无尖峰），但 15m–2h 全部 40 格 median
  幅度不过成本线；4h 唯一双样本厚度可行格（E1·k=8）**期望为负**（mean −0.07%
  OKX / −0.11% Binance，成本前）——分布左偏，少数延续尾部吃掉多数小回归。
  验证周期铁律首次执行：开题前一天的描述性统计终结一整类纠结。5m 终审未重测。
- **第七研究：funding 结构前置研究（纯描述性，立项判定）**（2026-06-12，
  `reports/funding_structure_20260612/`）：**不立项**。极端费率事件（F1 首破
  p95/p5、F2 持续 ≥3 结算于 p90/p10，180d 滚动无前视）两腿合算：正极端側
  现金流被价格腿延续尾部以 ~1:10 吃穿（2021 牛市 1:12，费率腿最肥年 = 价格腿
  最深年），0/5 币存活；负极端側现金流 ≈0（占合计 1-6%），过厚度线的格靠价格
  反弹且**条件均值低于无条件漂移**（4/5 币）——漂移性质非事件 edge。双样本
  幸存两格的验证周期 144/895 年 ≫ 24 个月铁律。**版图结论：5 币数据上的
  回归式信号（价格 MR 5m / 15m-4h / funding 极端回归）已全部检验完毕，
  全部不立项**；剩余信号空间在回归式之外。
- **第八研究：波动率事件前置研究（纯描述性，立项判定）**（2026-06-13，
  `reports/volatility_event_20260613/`）：**不立项，死因 b（精确形态：增量线+永续
  线性壁）**。非方向命题——波动率本身可否预测并用永续货币化。物理约束：无期权数据，
  只能测永续合成做多波动率。第 1 部分（可预测性）双样本 PASS：波动率聚集极强
  （低波后 RV 全 < 无条件、高波后全 > 无条件，5/5 币 ×2 样本），但方向与直觉**相反**
  ——ΔP(高波|低波) 大幅为负（k=6h：1.1% vs 无条件 24.8%，塌降 24pp），低波预测的是
  平静延续、非爆发临近。第 2 部分（可货币化，方向中性 CAP_net=|净位移|）**死于增量线**
  （非厚度线）：厚度过（CAP_net 1.08–3.53% ≫ 0.20% 成本），但低波条件化幅度系统性
  _低于_ 无条件随机时点（增量 −0.22~−0.42% OKX / −0.31~−0.62% Binance，5/5 币双样本
  全负）——"等低波"携带负信息。深层死因=永续线性壁（静态 straddle 毛利恒零，捕波动率
  必退化为方向，方向已空）。edge 真实但属期权玩家；接入期权数据为唯一重启线索
  （记录不行动）。**永续侧 ≤4h 频段：方向式 + 回归式 + 波动率式信号已成闭环检验。**

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
  <!-- 2026-06-22 校正（reports/trend_methodology_hardening_20260622/）：本条"~15-19 年"
       系**未打折 + iid**的乐观估计。对 B2_4h 做 Bailey-LdP 多重检验打折后：观测年化
       Sharpe 0.655（=15 配置最高，正是 best-of-15 被选出者）→ deflated 0.510（有效 N
       2.35）/ 0.267（名义 15）/ 0.162（扩展 N 下界 47）；从对自相关诚实的 bootstrap SE
       反推的诚实验证周期 14 / 51 / 138 年。自相关实测≈0（日 M2M 收益 ACF≈0），拉长几乎
       全部来自选择偏差打折非序列依赖。另：本条引用的"V1' 系事后修正"准确定性为"事后
       移动球门 / 程序污点"，幸存证据等级因此降低。关闭核心算术被诚实重算证实；部署
       现状（趋势线关闭、B2_4h 前向观察运行）不变。 -->
- **B2_4h funding 同向确认无效，趋势线维持资源关闭（双样本，2026-06-14）**：
  零阈值单规则（入场时点 funding 符号确认，V+ ⊆ V0），在已知靶子 OKX 上全面改善
  （净利 $68k→$86k、Sharpe 0.65→0.99、验证周期 9.1y→3.9y），但 6 年确认样本 Binance
  上**全面变差**（净利 $301k→$199k、Sharpe 0.94→0.74、周期 4.3y→7.0y）。机制不成立：
  被否决的逆向 leg 在 Binance 平均**赚 $266/笔**（不更差，p0.49），且"逆向更差"在
  样本间、bar/trade 粒度间反复翻号——非稳定因果。OKX 单边改善 = 已知靶子描线产物。
  E1-E4 全 FAIL（均因 Binance 反向）。佐证 funding 在 ≤4h 尺度对方向性收益无可货币化
  预测力（趋势确认与 MR 极值反转两路均否）。[`reports/trend_funding_confirm_20260613/`]
- **B2_4h ADX(14)>25 趋势强度过滤无效且有害、机制反向（双样本，2026-06-16）**：与 funding
  确认同防线栈（零变体单规则、入场时点过滤、V+⊆V0、OKX 定义/Binance 6 年确认）。ADX 过滤砍
  56% 笔数，**双样本同向恶化**：净 Sharpe OKX 0.65→**−0.055**（转亏）、Binance 0.94→**0.28**；
  净利 OKX 转负、Binance 砍 86%（2021 大牛 $164k→$24k）；验证周期 OKX 9.1y→**∞**、Binance
  4.3y→**49.9y**（减法陷阱）。**机制双样本一致反向**：被过滤的"弱趋势"信号毛收益反而 OKX **28×**
  / Binance **4×** 高于被保留的"强趋势"信号；全期 top-10% 大赢家中 **~70% 的头部利润落在被
  过滤的早期低 ADX 入场上**（OKX 70.0%/Binance 70.7%）。机制：ADX 滞后，趋势策略靠早期(低 ADX)
  入场吃趋势，"趋势强度确认"系统删除利润来源——**趋势策略加趋势过滤是范畴错误**（同 V1"剔除
  top5%"）。E1-E4 全 FAIL 双样本。比 funding 死得更干净（连 OKX 描线假象都没有）。**趋势强度
  过滤线索终结，B2_4h 维持资源关闭**；两次已知靶子指标过滤实验（funding+ADX）双败，确认该防线栈
  在"OKX 单边描线"与"双样本一致反向"两种失败模式下都正确判死。[`reports/trend_adx_filter_20260616/`]
- **更早入场 EMA10/30 双样本每轴劣于 B2_4h，机制=快的对称性（2026-06-16）**：由 ADX"70% 利润在
  早期入场"引出的单点检验（文献标准快对 EMA10/30 vs 20/100，零搜索零变体，同防线栈）。V_fast 笔数
  +180%，但**双样本每一轴皆劣**：毛利 ↓31/33%、净利 ↓46/37%、Sharpe OKX 0.65→0.38 / Binance
  0.94→0.77、验证周期 9.1→26.8y / 4.3→6.5y（加法陷阱）；最大趋势年 Binance 2021 砍 56%。机制三者
  皆非：**早期捕获为真**（V_fast 在 93% 的 V0 大赢家里早进 ~30h）**但毛利反降**（非撞成本墙——撞墙
  需毛利升）——根因是**"快"的对称性**：更快均线早进也早出（中位持仓 196h→76h）+ whipsaw×3（净亏翻
  6-10 倍、whipsaw 胜率 0%），早进好处被对称早出+churn 抵消并反超。**"70% 早期利润"是 B2_4h 自身哪些
  交叉最好，不等于"用更快均线"**；B2_4h 慢均线骑趋势更久正是优势来源，与裸基线"慢参数胜出"一致。
  F1-F4 全 FAIL 双样本。**至此已知靶子 B2_4h 三次增强（funding 描线 / ADX 反向 / faster-entry 对称劣化）
  全败，"更早入场"悬念终结，趋势线维持资源关闭**；防线栈三种失败模式下均正确判死。[`reports/trend_faster_entry_20260616/`]
- **15m–4h 价格回归无可收割期望（双样本，2026-06-12）**：回归作为频率现象
  真实存在（回归率全尺度显著 > 基线，平滑衰减），但回归幅度 median 在
  15m–2h 全部不过 2×成本线；4h 唯一双样本过线格（E1 通道突破·k=8）的
  事件级 mean 为负（OKX −0.071% / Binance −0.110%，成本前）——左偏分布，
  延续尾部毁灭期望（恰是趋势策略存活的同一结构）。不存在"5m 不行但更高
  尺度行"的中间地带；中频 MR 线不立项。[`reports/mr_timescale_structure_20260612/`]
- **极端 funding 费率不构成正期望回归结构（双样本，2026-06-12）**：正极端
  （收费=逆拥挤做空）现金流与价格腿损耗比 ~1:10，全 k 全币负；负极端现金流
  可忽略（1-6%），价格反弹系 β 漂移（条件均值 4/5 币低于无条件漂移）；
  双样本幸存格粗估 Sharpe 0.07-0.16 → 验证周期 144/895 年。**回归式信号
  在本市场版图上已检验完毕（5m/15m-4h 价格 MR + funding），全部无可收割
  期望**——右偏延续结构系统性不利逆势信号；极端正费率若有用途是趋势同向
  确认因子（未检验，属新开题）。[`reports/funding_structure_20260612/`]
- **15m 突破后"50% 回调接延续"是民俗，无结构（双样本，2026-06-12）**：全部
  20-bar 突破簇为分母时，max 回调深度中位 0.82、41% 直接打穿整段、触及
  [0.4,0.6] 带率 87%——50% 条件几乎不筛选任何事件，0.5 处无聚集无拐点。
  触带后顺势毛期望为正且 k=4 双样本显著（+0.022~0.028%，t 3.5/4.3）但仅约
  全成本 1/4、距 1.5×成本厚度线 7 倍，随持有期衰减至 0；**"突破确认即追"
  （B1）在全 k 双样本 mean 为负**——等回调优于追突破，但只是"比负数好"。
  不立项，死因 c（毛正薄于成本；maker 执行方向按预注册条款记录不行动）。
  [`reports/breakout_pullback_15m_20260613/`]
- **波动率可预测但永续不可货币化，需期权载体（双样本，2026-06-13）**：波动率聚集
  在本市场极强且双样本逐格复现（低波后 24h-RV 显著低于无条件、高波后显著高于，
  5/5 币各 ×2 样本；ΔP(高波|低波) 在 k=6h 达 −0.237，低波后高波概率从 24.8% 塌到
  1.1%）——但**可预测性指向"平静/动荡各自延续"，对线性永续不可收割**。方向中性
  代理 CAP_net=|k 期净位移| 厚度过（1.08–3.53% ≫ 0.20% 双边成本）却**死于增量线**：
  低波条件化幅度系统性低于无条件随机时点（增量 5/5 币双样本全负）。深层=永续线性
  （静态 straddle 毛利恒零，捕波动率必退化为方向，方向频段已空）。**可预测性 ≠
  可货币化**；该 vol edge 真实存在但需期权凸性载体，永续不可得（接入期权数据为
  唯一重启线索，记录不行动）。[`reports/volatility_event_20260613/`]
- **加密横截面信号（动量/carry/反转）在幸存者样本上不立项（2026-06-13）**：22 币纯加密
  集合、日频、三因子参数写死。先验担心的"伪装 BTC-beta"被**证否**——三因子多空两腿
  beta 近对称(1.1-1.2)、价差 beta≈0，MOM/CAR 剥离 beta 后 alpha 显著(t3.86/2.08)=**真
  横截面 alpha**。但全灭于 G1-G5：**F-MOM** 日频 IC 恒为零(t−0.04)+近年反号塌缩(2026
  IC−0.035)，看似 1.44 净 Sharpe 的价差是幸存者+极端尾部+2020 早期集中的镜像；**F-CAR**
  唯一有显著全样本 IC(t3.11)+真 alpha，但 quintile 不单调+IC 2024 后衰减至不显著(2025+
  t0.70)；**F-REV** 正 IC 是假象，可交易价差强负(净−159%/alpha t−2.89，尾部"赢家继续赢")。
  加密横截面 alpha 近年衰减经实证（MOM 2020 见顶、CAR 2021 见顶后单调走弱），印证先验。
  **真 alpha 是必要非充分——横截面成功对冲方向但本市场该 alpha 太弱+太不稳。** 幸存者
  偏乐观，补退市币只会更差→否定更稳。carry 是唯一留痕线索（记录不行动）。
  [`reports/cross_sectional_ic_20260613/`]
  <!-- 2026-06-13 carry 持有期重看（reports/cross_sectional_carry_holding_20260613/）：
  仅改持有期一个变量、G1-G5 零放宽。3 日持有全过 G1-G5（GO-CANDIDATE*，事后持有期星号）、
  5 日仅 G4 死（alpha t1.90<2）→ 一过一败 = 参数尖峰 → 不立项。机制假设部分证实：拉长
  持有解决了 k=1 的两个死因（G2 单调 viol2→1、G5 近年 2025+IC t0.70→3.55），IC 随 horizon
  单调增（+0.019→+0.035）；但 IC 强化未转化为稳健可交易 alpha（alpha_t 2.13→2.06→1.90 反向
  走弱）。carry 仍是横截面唯一两次留痕信号，重启须用退市币 OOS 杠杆 + 单一预注册持有期，
  非第三次参数探查。 -->
- **配对/价差回归在幸存者样本上不立项；均值回归家族整体收口（2026-06-13）**：22 币纯加密、
  日频、滚动 formation(90d)/trading(30d) 协整筛选（Engle-Granger p<0.05，零前视、零全样本挑对）。
  C1 存在过（均值 12.48 对/窗）、C4 厚度过（回归毛 mean +13.8%）；但 **C2 稳定 FAIL**（协整中位
  持续仅 **1 窗**、churn **91%**——加密协整瞬时）、**C3 回归 FAIL 且反向**（|z|≥2 后回归率 17.9%
  < 无条件基线 42.4%，two-prop z−13.1，偏离更多延续）、**C5 破裂 FAIL**（715 事件 128 回归/490
  发散/97 未决，事件加权净 −1.50%——肥但稀的回归盖不过破裂尾部）。**机制核心：价差未逃脱
  右偏延续结构**——加密协整非平稳，"偏离均衡"多是关系断裂 regime，价差延续而非回归。幸存者
  无退市币→破裂尾部低估→否定更稳。**MR 家族三支（单序列价格 5m/15m-4h + 横截面 carry +
  配对价差）+ funding 极端回归全部探明，回归式信号在永续 5/22 币数据上无一立项。**
  [`reports/pairs_cointegration_20260613/`]
- **OKX 期现基差（cash-and-carry）不立项——数据可得但肉≈无风险、超额≈0（2026-06-16）**：首个
  离盒（非永续、非方向）方向。**数据可得 = 绿灯**：REST 不给已交割合约 K 线（`50047`），但**数据
  下载 CDN 逐日交易文件**可完整重建反向季度合约全生命周期（HKT 日界，免费，回溯 2019；可用
  14 季/币 = 28 轮 2022Q1–2025Q2；指数 REST 直取；Binance 交叉验证 0.067%）。USDT 本位季度合约
  约 2025 底被 OKX 下架，反向 BTC-USD/ETH-USD 是仍挂牌载体。**但肉太薄**：28 轮全 contango，扣
  0.30% taker 后净基差年化 deployable 均值 **4.62%**、全样本 **3.84%**，**均低于 5% 无风险**，相对
  T-bill 超额 **−1.16%~+0.23%（≈0 至负）**；强 regime 依赖（熊 0.4%/牛 7.5%）。**B3 风险不可控**：
  空头交割腿孤立保证金最坏逆向 **+80.4%**（牛冲）、5/28 轮 >49%、安全杠杆仅 1.24×，须 cross-margin
  占满资本→资本回报压回无风险下；且肥基差与大保证金摆动同向（corr +0.43）。**唯一完美工作的是
  edge 机制本身（强制收敛 28/28 = 100%，B4 过）——机制可靠 ≠ 有肉**。判定：加密版固收，天花板在
  无风险利率附近，不值得双腿工程。[`reports/basis_arbitrage_feasibility_20260615/`]
- **"大规模因子"路线零成本前置门——规模不带来可交易 edge，不立项（2026-06-28）**：免费
  Binance Vision 100 币日线（2022-07→2026-05，3.9yr，幸存者集合，top-K-by-vol 全历史平衡面板）
  测"把横截面三因子从 22 币扩到 100 币 alpha 是否随规模改善"。先装**噪声标定**（per-coin
  block-shuffle×200 建假阳性基线，补 MR 协整漏掉的那道关；real IC 须过噪声基线非过 0）。
  **唯一随规模改善的是短期反转 (REV) 的毛 IC**（22→50→100 = +0.030→+0.043→+0.056，强过
  噪声 p0.000）；MOM 日频为负(反转)且随规模更负；CAR 仅 K22 过、加币即 washout
  (+0.019→−0.005，carry 是大币效应不吃规模)。**但 REV 的规模红利是 IC≠可交易 alpha 的
  教科书案例**：可交易顶层分位多空毛利 −62%/yr（液体大币极端尾部延续，呼应右偏延续
  capstone），pool100 净 −23%/yr（换手 1.8×/日→成本墙 40%/yr），**唯一净正在不可交易底层**
  (+22%/yr 且已用乐观 8bps，真实小币冲击会抹平)= 流动性伪装；且 REV=买近期输家、最受
  幸存者偏乐观撑（底层小老币最甚）。**IC-coded 门技术性 PASS(REV 过 G1-G3)，综合判生死
  =不过**——命中预注册"alpha 全来自不可交易小币(流动性伪装)"死因（用预注册 Part4c 可交易
  价差证据揭示，全程透明、未事后改门，方向偏保守非复活）。机构数据门槛勘察：survivor-free
  起步 Tardis ~$1–6k、机构级 Kaiko/CoinMetrics ~$10–55k/yr，判死→省下订阅 + 大工程。
  教训：**IC 基的流动性伪装门(G2)太弱须配可交易价差分层门；规模只改善毛 IC 不改善可交易
  alpha；反转未逃右偏延续只是迁到不可交易的小币。** [`reports/factor_scale_feasibility_20260628/`]
- **订单流耗尽（order-flow exhaustion）MR——前置判决性测试不过，微观结构/订单流类 MR 方向探明、
  按用户预承诺不再考虑（2026-06-28）**：项目首个机制上真正新颖的 MR——不赌"价格回归均值"，
  赌"大单冲击耗尽一侧流动性后价格从真空回弹（冲击消散）"，本应**逃出延续主导死因**。用逐笔
  数据（Binance Vision aggTrades，毫秒级含 taker 方向；**无订单簿深度**故为下界近似，K 线测不了）
  在 BTC+ETH 9 个季度日上测 **8,521 次一边倒 1s 冲击**（中位冲击 +1.88bps，冲击真实存在）。
  **机制仍是延续不是回弹**：0.1–10s 仅 20–43% 事件回弹、均值回弹为负（价格续走），30s+ 收敛
  到~50% 硬币翻转（无记忆随机游走）；**与随机 bin 安慰剂几乎重合**（%回弹@5s 39.9% vs 40.2%），
  回弹不含冲击特异信息、仅 bid-ask bounce 量级（半价差 floor 0.024bps）。**成本/速度墙 ~50× 且
  对延迟近乎不变**：可捕获毛回弹 L=0（理想 HFT）即仅 +0.09bps、median 0.000，至 L=1s 也只 +0.23bps，
  扣 10bps taker 后**全延迟 net ≈ −9.8bps**——比"HFT 抢跑吃掉"更彻底：逐笔价格上根本没有可收割的
  瞬态回弹。σ∈{3,4,5} 不改变结论。**至此右偏延续结构从 4h bar 一路证到 100ms tick**：项目所有
  回归路线（单序列 5m/15m–4h + funding + 横截面 carry + 配对协整 + 订单流耗尽）同死于此一结构事实，
  强证其为**结构性、非任何单一时间尺度/信号构造的产物**。[`reports/order_flow_exhaustion_feasibility_20260628/`]
- **Flow 信号 vs 价格信号（B2_4h）趋势捕获对照——flow（原因）不优于价格（结果），用价格即可
  （2026-06-29）**：检验 trend≈∫(flow×impact)dt 中"读被积函数(flow)能否比读积分(price)更早/更好
  捕获右偏延续"。Binance BTC+ETH 1m klines 2020-2026（taker_buy_volume 免费给出 bar 级 taker 失衡，
  无需订单簿/逐笔），flow 信号 `sign(EMA(OFI_norm))` 与 B2_4h 同引擎/同窗/双样本并列对照。
  **结论 = flow 是一个不同、更高频、更嘈杂、且更差的信号**：① flow 作为过程有真记忆
  (ACF≈0.10>shuffle) 但**不前瞻预测收益**(IC<噪声地板)，同期 corr(OFI,ret)≈0.51（flow 与价格
  同步发生、不领先）；② **无任何 flow 配置在两币都胜 B2_4h**——F1/F2 在 BTC 为正、ETH 深亏
  （ETH 上 flow 比自身 block-shuffle 还差），F2 的 BTC 胜利是单币海市蜃楼；flow 成交 4–21× 于价格
  （flow 率快速均值回归→跟随=whipsaw）；③ 方向重合度仅 ~0.45–0.53（flow **确为不同信号**，但
  "不同≠信息"，是噪声）；④ **flow 唯一真正领先价格在亚秒**(corr 未来 +0.056@1s→0@60s，过去相关
  是其 5×)=HFT 带、=exhaustion 已判死的 −9.8bps 同墙；bar 尺度"早 2–4 根"+1 根延迟不伤反好=不带 edge。
  **更深的为什么：可收割的趋势是积分后的价格水位，不是 flow 率本身——积分正是把嘈杂 flow 转成
  趋势的操作，故"结果"(price) 反而是更好的信号。** 至此**订单流方向性使用两侧全死**（fade=exhaustion /
  follow=persistence）；右偏延续再获一证（原因也不能比结果更早够到它）。**守边界：判 flow 不如 price
  ⇒"用价格即可"，未退化为给 B2_4h 加 flow 过滤器（第五次增强=描线）。** [`reports/flow_vs_price_trend_20260628/`]

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
- **横截面研究原材料已下载，待开题**（2026-06-13）：22 个主流 USDT 永续（前 40 成交额
  ∩ ≥18 月上线，17 新 + 5 复用）的 1m K 线 + funding，sha256 全验、0 失败，~2.03 GB；
  编目/清单/局限见 `data/binance_vision/cross_sectional_{manifest,README}`。**是当前主流币
  快照、含幸存者偏差、不含退市币——不可直接用于横截面收益估计，防御属开题阶段。**

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
- 当前研究状态：**永续信号空间已闭环**——"永续(线性)×≤4h×零售 taker×5 币价格+funding"约束盒内，方向式/回归式/波动率式三大类单序列信号机制已全部双样本证否或证明验证不起。**完整地图、五条出路、重启索引见 `reports/perpetual_signal_space_closure_20260613.md`（阶段性收口文档，第一参照）。** 其下：MR-5m 项目已关闭；趋势线已关闭（2026-06-12 资源决策，非证伪，重启条件见 `reports/trend_line_closure_20260612.md`）；已检验未立项候选 = MR 时间尺度结构 / funding 结构 / 15m 突破回调（死因 c）/ 波动率事件（死因 b/增量线+永续线性壁，需期权载体，`reports/volatility_event_20260613/`）。离盒出路中横截面/相对价值曾是唯一"约束集内、无需新载体/身份"的通顺主线，但其 IC 前置研究（2026-06-13，`reports/cross_sectional_ic_20260613/`）三因子全灭、**在幸存者样本上即不立项**——横截面方向就此关闭。剩余离盒出路均需改约束盒（期现基差/对冲 funding 需现货数据；做市需 maker 身份+微观结构；期权需新载体），见 `reports/perpetual_signal_space_closure_20260613.md` §5。**其中期现基差已执行前置研究并不立项（2026-06-16）：数据可得但扣费后净基差年化≈无风险、超额≈0，且空头交割腿最坏逆向+80%需 cross-margin，加密版固收薄到不值得双腿工程，见 `reports/basis_arbitrage_feasibility_20260615/` 与上「已验证的核心事实」。**<!-- 2026-06-16 更新：原"期现基差需现货数据"为未测出路，现已测：数据绿灯但无超额，不立项 -->**均值回归家族（单序列价格 + 横截面 carry + 配对价差 + funding 极端）三支/四线全部探明完毕、整体收口（2026-06-13，`reports/pairs_cointegration_20260613/`）——回归式信号在永续数据上无一立项。**<!-- 2026-06-11 更新：原"唯一在跑方向：动态仓位（双账号对照实验）"随项目关闭而终止 --><!-- 2026-06-12 更新：原"新阶段为新策略研究"细化——第一条线（趋势）已走完五阶段，以资源决策关闭 -->

撤回的说法（曾被采信，已被 OOS 推翻）：
- ~~"ATR filter 是 −72% Max DD 风控机制"~~ —— in-sample artifact，OOS 不复现，已撤回（`reports/regime/atr_filter_oos_validation_20260609/`）。
- 注：本次 atr_ratio 拆解未推翻任何研究**方向**，无新增死胡同（DYN 是 rework 候选，非死路）。
