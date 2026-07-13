# STRATEGY_CODEX — 策略与代号图例

> 解决"代号不自解释"：本仓库报告/记忆/脚本里出现的代号，每个给
> 全称 / 一句话定义 / 状态 / 关键数字 / 出处指针。按研究线分组。
> 活文档：新增代号随研究产出补一行；状态翻转时修订对应行。

## 均值回归线（全部 CLOSED）

| 代号 | 全称/定义 | 状态 | 关键数字 | 出处 |
|---|---|---|---|---|
| **MR-5m** | 5m fade Donchian 均值回归（BTC/ETH/SOL/LINK/DOGE，LB=24/ATR14 Wilder/1.0×ATR 止损/midline 止盈） | **CLOSED**（mainnet 无 edge） | mainnet 全期 PF 0.81–0.85、毛利≈0；demo PF 2.06 系数据假象 | `research/_closed/_synthesis/MR5M_postmortem.md` |
| **MR-v1** | MR-5m 的 vnpy 策略化版本（含 chandelier/filter/midline 变体） | 归档（随 MR-5m） | — | `_archive/legacy_reports/mr_v1*/` |

## 趋势线（资源关闭，B2_4h 前向观察中）

15 个基线配置命名 = 族（A1-A3 Donchian / B1-B2 EMA 金死叉 / C1-C3 TSMOM）× 周期（4h/1d）。

| 代号 | 全称/定义 | 状态 | 关键数字 | 出处 |
|---|---|---|---|---|
| **B2_4h** | 4h EMA20/100 金死叉，always-in long/short，5 币各 $10k 名义 | **资源关闭（非证伪）+ 前向观察中**（VPS，零成本 sim） | 冻结净利 OKX $68,194.82 / Binance $300,752.78（含 funding，审计 PASS）；Sharpe 0.655 → deflated 0.510/0.267/0.162；验证周期 14/51/138 年 | `research/_closed/crypto_perp/trend_b2_4h/reports/trend_dualcycle_20260611/`、`research/_closed/_synthesis/trend_methodology_hardening_20260622/`、`research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_pnl_audit_20260628/`、`forward/` |
| **D2** | B2_4h 同信号的 long/flat 版（D 族=方向变体，非 15 基线之内） | DUAL-VALIDATED\* 后随线关闭 | 2021 牛市捕获 58%/74%；无 short 腿 | `research/_closed/crypto_perp/trend_b2_4h/reports/trend_dualcycle_20260611/` |
| **B1_4h / C1_4h / C2_4h / C2_1d …** | 其余基线幸存者（EMA10/50、TSMOM 变体等） | 随线关闭 | 15 配置 Sharpe 表见方法论加固报告 Q0 | `research/_closed/crypto_perp/trend_b2_4h/reports/trend_baseline_20260611/` |
| **V1 / V1′** | 趋势验证 r1 的单笔集中度 gate / 其事后重定义版 | V1′ = **程序污点**（事后移动球门），幸存证据等级降低 | V1 通杀 15/15；V1′ 复活 5 | `research/_closed/_synthesis/trend_methodology_hardening_20260622/` Q6① |
| **B2_4h-VT** | B2_4h 的 vol-targeting 风控变体（仓位∝1/σ，EWMA hl=48bars） | **NOT ADOPTED** | 双样本相反：DD −12% 仅 Binance、收益 +40% 仅 OKX（描线）；Binance top1% 捕获 0.60 | `research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_vol_targeting_20260628/` |
| **K1/K2/K3, U1/U2** | 前向系统预注册 KILL/UPGRADE gate | 冻结运行中 | K1 roll12 净<$1,296 / K2 maxDD>$32,483 / K3 7d 心跳 | `forward/gates_preregistered.md` |

## 横截面/因子线（CLOSED / NOT VIABLE）

| 代号 | 全称/定义 | 状态 | 关键数字 | 出处 |
|---|---|---|---|---|
| **F-MOM / F-CAR / F-REV** | 横截面动量(+ret30) / carry(−funding7) / 短期反转(−ret3) | 22 币 CLOSED；100 币规模路线 NOT VIABLE | REV 毛 IC 随规模升但净 −23%/yr（流动性伪装）；CAR 规模下 washout | `research/_closed/crypto_perp/cross_sectional/reports/cross_sectional_ic_20260613/`、`factor_scale_feasibility_20260628/` |

## 期权/波动率线（CLOSED）

| 代号 | 全称/定义 | 状态 | 关键数字 | 出处 |
|---|---|---|---|---|
| **VRP ATM（StageA/B）** | Deribit BTC/ETH 月度 ATM 短跨式波动率风险溢价；端点1=不对冲、端点2=delta-hedged（剥方向） | **全线 CLOSED**（ETH StageA 无缝；BTC StageB 剥方向后 net≈0） | BTC 端点2 净 +0.36vp（≈0）、摩擦吃 ~85%、bootstrap 5% 下界三档全负、whipsaw 情景 −67%spot | `research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stage{A,B}_*/` |

## 早期研究代号（demo 时代，证据基础已作废，仅索引）

| 代号 | 全称/定义 | 状态 | 出处 |
|---|---|---|---|
| **csrb_v1** | Crypto Session Range Breakout——亚欧美 session 切换的区间突破假设 | demo 时代归档 | `_archive/legacy_reports/csrb_v1/` |
| **vsvcb_v1** | Volatility-Squeeze Volume-Confirmed Breakout——布林带宽挤压+突破 bar 放量 | demo 时代归档 | `_archive/legacy_reports/vsvcb_v1/` |
| **mhf** | Adaptive Multi-Horizon CTA（`OkxAdaptiveMhfStrategy`，BTC 单标的） | demo 时代归档 | `strategies/okx_adaptive_mhf_strategy.py`、`_archive/legacy_reports/trend_*` |

## 通用口径缩写

| 缩写 | 含义 |
|---|---|
| **DSR** | Deflated Sharpe Ratio（Bailey-LdP 2014；skill `multiple-testing/`） |
| **ENB** | 有效独立试验数 (Σλ)²/Σλ²（有效 N） |
| **M2M** | 逐 bar mark-to-market 记账（日收益 = ceil('D') 桶） |
| **NULL-A/B** | 噪声标定双 null（逐列独立 / 整行共享 block bootstrap；skill `noise-calibration/`） |
| **gate 0** | 样本方向中性核对（净 Σlog / 单向月占比） |
| **312 / 519** | 2020-03-12 COVID 崩盘 / 2021-05-19 崩盘（尾部事件速记） |
