# VRP 研究线 — ATM 波动率风险溢价（独立于 cta_strategy 永续研究）

> 本研究线与 cta_strategy 的永续（OKX/vnpy）研究**逻辑隔离**：独立 `vrp/` 工作区、
> 独立数据源（Deribit 期权）、独立 `data/`，**不读写** cta_strategy 的
> `database_mainnet.db` / 污染库 / 前向系统 / VPS。仅借用同一 git 仓库做版本管理。

## 标的与动机
- 标的：**Deribit BTC / ETH ATM 期权**（生产环境，`www.deribit.com`）。
- 动机：卖出 ATM 波动率，赚取 IV − RV 的波动率风险溢价（VRP）。

## VRP 特殊纪律（与本仓库过往右偏/中性研究风险结构相反，全文不删改）
> VRP 是**左偏卖保险型**策略：平时小赚、尾部巨亏。回测会**系统性高估**它
> （peso problem——尾部没在样本里发生，均值被借来的好运抬高）。判定哲学：
> 1. **双重门**：edge 为正 ∧ 尾部可生存，两者都满足才算活。
> 2. **Sharpe 是危险信号非目标**：负偏被 Sharpe 高估，本项目早期阶段不算 Sharpe。
> 3. **答案条件于假设**：尾部假设 + 对冲频率假设。净缝必须**条件于对冲频率**——
>    对冲腿摩擦是 `频率 × 单次成本` 的累积，一次性粗估会低估摩擦栈、把净缝算成
>    虚假为正（peso 的成本版）。

## 阶段划分与现状
> **ATM VRP（月度，BTC/ETH）线已全线关闭 2026-06-28**：ETH 阶段 A 判死（无 meat，IV≈RV
> 中位无溢价）；BTC 阶段 B 判死（剥离方向后 VRP 真伪不成立，净 edge≈0、摩擦吃掉 ~85% 的
> direction-stripped 毛缝、且带 whipsaw 灾难尾）。未进阶段 C/D。

- **阶段 A（DONE）**：数据地基 + 净缝生死门。数据可干净测（IV-DVOL corr 0.988、Coinbase 锚点）、
  摩擦看似非主凶（基于 σ_IV−σ_RV 粗毛缝）。报告 `reports/atm_vrp_stageA_data_20260628/`。
- **阶段 B（DONE，判死）**：用 delta-hedged（剥离方向）裁 VRP 真伪——净 edge 与零不可区分、
  5% bootstrap 下界三档全负、whipsaw 情景 −67%spot。报告 `reports/atm_vrp_stageB_premium_truth_20260628/`。
- 阶段 C 尾部压测 / 阶段 D 精确对冲路径：**未达到**（真伪已不成立，无需测尾部生存）。
- 窄复活条件（记录）：maker 对冲执行降摩擦、周 ATM、skew/wing premium（数据更难）——见阶段 B 报告。

## 目录
- `scripts/research_atm_vrp_stageA.py` — 阶段 A 主脚本（口径与判定线写死 docstring）。
- `data/` — Deribit 拉取落盘。`manifest.json` / 处理后 `cycles_*.jsonl` / `*_summary.json`
  入 git；原始 API 缓存 `cache/` 与大文件 gitignore。
- `reports/` — 每阶段一份 markdown 报告。

## 数据真实性红线（接 cta_strategy demo 污染惨案的教训）
当年 OKX demo 污染事故的根因之一是"以为"拉的是真实数据、只做了"确认"而无可证伪核对。
本线**强制**：(1) 端点 URL 写进 manifest 且确认 `testnet=false`；(2) 用独立来源
（Coinbase 现货）对关键尾部日逐日交叉验证 Deribit 标的价。详见阶段 A 报告。
