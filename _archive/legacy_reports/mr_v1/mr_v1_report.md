# MR-v1 Mean Reversion: Fade 20-bar Breakouts with Exit Mechanisms

## 1. 研究假设
MR-v1 是对趋势跟踪失败后「突破失败 = 均值回归」假设的 Phase 1 验证。核心思路：
- **入场**：fade 20-bar breakout（长突破→做空，短突破→做多）
- **出场**：比较 A/B/C/BC 四种出场机制的 OOS 效果
- **假设**：breakout 在 crypto 4h 级别倾向于失败并回归

## 2. 入场规则
- 4h closed bar close > 20-bar high → **做空**（fade breakout）
- 4h closed bar close < 20-bar low → **做多**（fade breakdown）
- 一次只持有一个方向
- **不调参，不优化**

## 3. 出场机制
| Variant | 描述 | Max Hold |
|---|---|---|
| A (Baseline) | 2× ATR 止损 | 60 bars |
| B (Drawdown) | drawdown from peak > 2× entry_ATR 退出 | 60 bars |
| C (1d Lock) | A 的止损被 1d EMA 确认抑制 | 120 bars |
| BC (B+C) | B 的 drawdown 被 1d EMA 确认抑制 | 120 bars |
| D (Random) | 随机入场时间 + A exit | — |
| E (Reverse) | 反向信号 + A exit | — |

## 4. 各 Variant 汇总
| variant | split | trade_count | no_cost_pnl | cost_aware_pnl | funding_adjusted_pnl | winning_rate |
|---|---|---|---|---|---|---|
| A | oos_ext | 118 | 3518.4948 | 3282.4948 | 3268.8037 | 0.5085 |
| A | train_ext | 231 | 9991.8236 | 9529.8236 | 9466.1948 | 0.4329 |
| A | validation_ext | 159 | 4044.7912 | 3726.7912 | 3714.9390 | 0.3459 |
| B | oos_ext | 118 | 1575.0533 | 1339.0533 | 1340.8245 | 0.5678 |
| B | train_ext | 231 | 2855.0926 | 2393.0926 | 2404.5433 | 0.5411 |
| B | validation_ext | 159 | 1232.7683 | 914.7683 | 918.0035 | 0.4843 |
| BC | oos_ext | 118 | 3222.5668 | 2986.5668 | 2986.7419 | 0.5932 |
| BC | train_ext | 231 | 9723.9715 | 9261.9715 | 9147.9159 | 0.5325 |
| BC | validation_ext | 159 | 10207.8169 | 9889.8169 | 9842.5418 | 0.5220 |
| C | oos_ext | 118 | 2921.7714 | 2685.7714 | 2657.6790 | 0.4322 |
| C | train_ext | 231 | 10034.0914 | 9572.0914 | 9418.0950 | 0.3723 |
| C | validation_ext | 159 | 7232.2861 | 6914.2861 | 6862.8561 | 0.3019 |
| D | oos_ext | 150 | 2521.4795 | 2221.4795 | 2196.0483 | 0.4800 |
| D | train_ext | 233 | 8103.9753 | 7637.9753 | 7405.2738 | 0.4292 |
| D | validation_ext | 125 | 2540.3177 | 2290.3177 | 2258.7966 | 0.3760 |
| E | oos_ext | 118 | 178.1135 | -57.8865 | -91.6748 | 0.3220 |
| E | train_ext | 231 | 2234.8399 | 1772.8399 | 1652.4610 | 0.3290 |
| E | validation_ext | 159 | 4406.0750 | 4088.0750 | 4049.3170 | 0.3836 |

## 5. Reverse Test
正向 (A) vs 反向 (E) 对比：
| variant | split | no_cost_pnl |
|---|---|---|
| A | oos_ext | 3518.4948 |
| A | train_ext | 9991.8236 |
| A | validation_ext | 4044.7912 |
| E | oos_ext | 178.1135 |
| E | train_ext | 2234.8399 |
| E | validation_ext | 4406.0750 |

## 6. 集中度
| variant | split | symbol | trade_count | total_no_cost_pnl | top_5pct_contribution | concentration_pass |
|---|---|---|---|---|---|---|
| A | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | 387.3550 | 0.3457 | True |
| A | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 551.1238 | 0.2315 | True |
| A | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 372.9221 | 0.4151 | True |
| B | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | 284.5608 | 0.2737 | True |
| B | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 230.6891 | 0.2865 | True |
| B | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 241.7037 | 0.4018 | True |
| BC | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | 425.6476 | 0.3450 | True |
| BC | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 381.5777 | 0.4031 | True |
| BC | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 416.0730 | 0.3587 | True |
| C | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | 112.5971 | 1.3043 | True |
| C | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 598.2219 | 0.4455 | True |
| C | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 422.8207 | 0.4342 | True |
| D | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 27 | 371.4263 | 0.2848 | True |
| D | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 42 | 1015.7821 | 0.4878 | True |
| D | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 21 | 97.9811 | 1.0617 | True |
| E | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | -67.3473 | -1.5328 | True |
| E | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 549.0215 | 0.4932 | True |
| E | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 205.6092 | 0.5251 | True |
| A | oos_ext | ETHUSDT_SWAP_OKX.GLOBAL | 23 | 866.3023 | 0.6487 | True |
| A | train_ext | ETHUSDT_SWAP_OKX.GLOBAL | 37 | 435.9452 | 0.3651 | True |

## 7. Gate 裁决
- cost_aware_pass=True
- funding_adjusted_pass=True
- reverse_test_pass=True
- random_control_pass=True
- concentration_pass=True
- can_enter_phase2=True
- final_decision=proceed_to_phase2

## 8. 限制
- strategy_development_allowed=false
- demo_live_allowed=false
- 不修改 OkxAdaptiveMhfStrategy