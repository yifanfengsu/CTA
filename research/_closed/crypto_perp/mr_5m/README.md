# mr_5m — MR-5m 均值回归线（CLOSED，项目级结论）

**状态**：2026-06-11 mainnet 重验确认真实数据上无 edge（FLAT 毛利≈0，PF 0.83–0.85，
亏损=费用；demo PF 2.06 系 99% 数据伪影），线关闭。

本目录只收**事故后 mainnet 干净数据上的两个决定性脚本**：
- `scripts/research_demo_vs_mainnet.py` — demo vs mainnet 全量对比（199/205 币月 RED）
- `scripts/research_mr5m_mainnet_baseline.py` — mainnet 裸基线重验（判死依据）

## 报告与其余资产的位置

- 复盘总文档：`research/_closed/_synthesis/MR5M_postmortem.md`（§8 = 新研究启动门槛检查清单）
- 数据事故取证链 + mainnet 重建/重验报告：`reports/regime/`（原位保留，事故证据链不迁）
- demo 时代研究产物（证据基础已失效）：`_archive/legacy_reports/`、`_archive/legacy_scripts/`
- 实盘 runner（已关停，工程参考）：`_archive/mr5m_runner/`
- 回测基准引擎 `backtest_mr_5m_compare.py`：冻结于 `scripts/`（前向依赖，见
  `research/_closed/crypto_perp/trend_b2_4h/README.md`）
