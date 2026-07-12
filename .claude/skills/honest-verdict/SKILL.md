---
name: honest-verdict
description: 生死判决哲学——双重门（edge∧尾部生存）、左偏/右偏各自的判定纪律（铁律C及其左偏镜像）、答案条件于假设、判死判活同证据标准。任何策略/研究线的最终生死判定、任何含偏态分布的评价时必须使用。
---

# honest-verdict — 判决纪律（第 7 站）

## 0. 流水线位置

**第 7 站**（见 `.claude/skills/PIPELINE.md`）。
上游输入：第 6 站打折后数字 + 第 4/5 站误差棒与单次检验 + 预注册判定线。
下游产出：判决书（判活/判死 + 全部 conditional 条款）→ 交第 8 站尸检入库。
本 skill 偏文档（判定哲学），代码少——工具在上游各 skill。

## 1. 何时用

- 任何策略/研究线的最终生死判定。
- 任何含偏态 P&L 的评价（趋势=右偏、卖保险=左偏）——**Sharpe/mean 主判前必读**。
- 任何"指标改善了"的声明（回撤降了/Sharpe 升了/收益+40%）出现时。

**不用会犯什么错（真实案例）**：
- **Sharpe 判右偏 → 采纳削右尾的变体**：vol-targeting 双样本 Sharpe 都升
  （0.65→0.90 / 0.94→1.13），但 Binance top1% 赢家被砍到 60%、2021 牛市利润
  −52%——Sharpe 给削右尾打高分，按 Sharpe 判会采纳一个砍掉趋势本体的变体
  （出处：`research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_vol_targeting_20260628/` Q5，铁律 C 的干净反例样本）。
- **mean 判左偏可行性 → 越权裁了被推迟的尾部**：VRP StageA 预注册 mean 门，
  实测 BTC mean 为负纯因 312 单月 −110vp——mean 就是尾部，剔最坏 1 月即翻正。
  左偏可行性门应拆开："溢价扛过摩擦"（中位/典型月）与"均值扛过尾部"（尾部
  阶段专裁）分开预注册（出处：`research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageA_data_20260628/` Q5 元教训）。
- **平时稳赚 + 尾部巨亏的伪装**：端点 1 不对冲口径胜率 53%、median +0.81 但
  min −32%spot、skew −0.86——最危险的合成误读是把高胜率当安全
  （出处：vrp StageB B1 表）。

## 2. 怎么用

判决书按以下清单逐项过（每项写进报告，缺项 = 判决不完整）：

1. **双重门**：edge 为正（打折后）∧ 尾部可生存，**独立判定，缺一判死**。
   edge 门用第 6 站 deflated 数字；生存门用 bootstrap 下界（多档同号）
   **加**前瞻情景（bootstrap 抽不出比样本更坏的尾部——peso，二者互补）。
2. **偏态先声明，再选尺子**：
   - **右偏纪律（铁律 C）**：Sharpe 惩罚上行波动而上行波动是趋势本体；主判用
     整个 P&L 分布 + 尾部捕获效率（top5%/top1% 双层捕获，B2_4h-VT 诊断模板）
     + 下行调整指标（Sortino 并列），Sharpe 仅参考。右尾捕获 <0.70 类红线预注册。
   - **左偏纪律（铁律 C 的镜像）**：mean 被尾部主导 → mean/median **分列** +
     最坏单月 + 剔尾敏感性（剔最坏 1/3 月的 mean 漂移，量化 peso 暴露）+
     **前瞻情景 > 经验 CVaR**（历史最坏×2 / 反向极端 / whipsaw 注入）；
     Sharpe 高估左偏（平滑的负偏序列 Sharpe 好看），不算。
   - **"剥离方向 ≠ 剥离尾部"**：delta-hedge 剥掉方向尾后 whipsaw/方差尾仍在
     ——换尾不等于去尾（vrp StageB −67%spot whipsaw 情景）。
3. **答案条件于假设，逐条列出**：对冲频率（净缝必须条件于它）、尾部假设、
   **样本方向**——gate 0 方向中性核对（净 Σlog、单向月占比；超阈 → 全部结论
   标 conditional，含方向的口径降级）。牛漂移会把方向收益伪装成策略收益。
4. **判死与判活同证据标准**：否定性结论同等固化（进 PROJECT_GUIDE 已验证
   事实）；偏乐观样本上判死 ⇒ 否定更稳（方向逻辑显式写出）。
5. **措辞纪律**：资源关闭 ≠ 证伪（B2_4h 不得称"证伪"或"差一点成功"）；
   字面 gate 与综合判定分歧 → 并列报告、分歧方向必须朝保守、绝不改门；
   事后移动球门写"程序污点"，不写"自我修正"。
6. **"总收益提升"自动触发最高警惕**：已被骗 5 次（funding/ADX/faster/V1/VT）
   ——单样本改善 + 已知靶子 = 描线先验；独立样本同向确认前不采信。

## 3. 怎么失效

- **用 mean 判左偏可行性**：为什么——左偏分布的 mean 由极少数尾月主导，
  mean 的正负 = "样本里尾部发生了几次"的函数，不是溢价存在性的函数。
- **用 Sharpe 判任一偏态**：为什么——Sharpe 假设对称罚波动：右偏被罚上行
  （低估），左偏因平时平滑被抬高（高估）；同一把尺子在两个方向都系统性错。
- **经验 CVaR 当尾部上界**：为什么——CVaR 从已实现样本算，样本缺牙（恰好没有
  312 级事件）时 CVaR 是下界不是上界；前瞻情景注入是唯一能"抽出比样本更坏"
  的工具。
- **胜率 + 偏度合成误读**：为什么——高胜率与深左尾完全兼容（卖保险的定义），
  "平时稳赚"是左偏尾部风险的表象而非反证；胜率永远与最坏单月并列读。
- **Calmar/回撤类指标被收益伪迹污染**：为什么——Calmar 分子是收益，样本依赖的
  收益"提升"（描线）直接抬 Calmar，与风控无关（VT 的 OKX Calmar 5.02 由 +40%
  伪迹驱动）。
- **对"改善"只看单样本**：为什么——已知靶子上单样本改善的证据权重 ≈ 0
  （能看见答案的样本里画线）；VT 若只跑 OKX 四个指标全利好、必误判。

## 素材出处

- `research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_vol_targeting_20260628/README.md`（右尾捕获双层诊断、Sharpe 双升仍否决、敞口归一模板）
- `research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageA_data_20260628/README.md`（左偏 mean 门元教训、剔尾敏感性表）
- `research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageB_premium_truth_20260628/`（双重门、gate 0 方向核对、前瞻情景三件套、剥方向≠剥尾）
- `research/_closed/_synthesis/trend_methodology_hardening_20260622/README.md`（铁律 C 全文、措辞纪律、Sortino 并列画像）
- `research/_closed/_synthesis/PROJECT_FINAL_SUMMARY_20260614.md` §4（陷阱反面模式库）
- CLAUDE.md 铁律 C + 工作原则（判死判活同标准、资源关闭≠证伪）
