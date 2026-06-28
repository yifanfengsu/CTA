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

## 阶段划分
- **阶段 A（本阶段）**：数据地基 + 净缝生死门。只判"能否干净测 ATM VRP、毛缝扣掉
  交易摩擦（含反复对冲）后净缝还剩不剩"。**不测尾部、不测 edge 大小、不判立项、
  不算 Sharpe。** 报告：`reports/atm_vrp_stageA_data_20260628/`。
- 阶段 B：VRP 是否只是尾部补偿（前瞻情景）。— 未启动
- 阶段 C：尾部压测（真正生死）。— 未启动
- 阶段 D：精确对冲路径 P&L。— 未启动

## 目录
- `scripts/research_atm_vrp_stageA.py` — 阶段 A 主脚本（口径与判定线写死 docstring）。
- `data/` — Deribit 拉取落盘。`manifest.json` / 处理后 `cycles_*.jsonl` / `*_summary.json`
  入 git；原始 API 缓存 `cache/` 与大文件 gitignore。
- `reports/` — 每阶段一份 markdown 报告。

## 数据真实性红线（接 cta_strategy demo 污染惨案的教训）
当年 OKX demo 污染事故的根因之一是"以为"拉的是真实数据、只做了"确认"而无可证伪核对。
本线**强制**：(1) 端点 URL 写进 manifest 且确认 `testnet=false`；(2) 用独立来源
（Coinbase 现货）对关键尾部日逐日交叉验证 Deribit 标的价。详见阶段 A 报告。
