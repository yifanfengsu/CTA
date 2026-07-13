# Exit-Only Research: Trend Drawdown Tolerance (B) + 1d Trend Health Lock (C)

## 1. 研究假设
固定 20-bar breakout 入场，比较 5 种出场机制的组合效果。核心假设：
- **B (趋势回撤容忍)**：用 drawdown-from-peak 替代固定 ATR 止损，可以在趋势中呆更久
- **C (1d 趋势健康锁定)**：当 1d close 在 20-EMA 的有利一侧时，抑制 4h 出场信号
- **B+C**：两者结合效果最好

## 2. 入场规则
- 4h closed bar close > 20-bar high → 做多
- 4h closed bar close < 20-bar low → 做空
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
| A | oos_ext | 118 | 178.1135 | -57.8865 | -91.6748 | 0.3220 |
| A | train_ext | 231 | 2234.8399 | 1772.8399 | 1652.4610 | 0.3290 |
| A | validation_ext | 159 | 4406.0750 | 4088.0750 | 4049.3170 | 0.3836 |
| B | oos_ext | 118 | -819.2823 | -1055.2823 | -1060.3100 | 0.3559 |
| B | train_ext | 231 | -926.4910 | -1388.4910 | -1394.3516 | 0.4199 |
| B | validation_ext | 159 | 758.9753 | 440.9753 | 434.4556 | 0.4969 |
| BC | oos_ext | 118 | 981.6694 | 745.6694 | 715.5210 | 0.3559 |
| BC | train_ext | 231 | 3138.6255 | 2676.6255 | 2425.3726 | 0.4545 |
| BC | validation_ext | 159 | 6966.7890 | 6648.7890 | 6549.9116 | 0.4654 |
| C | oos_ext | 118 | 77.0378 | -158.9622 | -185.5645 | 0.2542 |
| C | train_ext | 231 | 2940.4265 | 2478.4265 | 2218.5098 | 0.3160 |
| C | validation_ext | 159 | 5606.2694 | 5288.2694 | 5204.5328 | 0.2767 |
| D | oos_ext | 150 | 2521.4795 | 2221.4795 | 2196.0483 | 0.4800 |
| D | train_ext | 233 | 8103.9753 | 7637.9753 | 7405.2738 | 0.4292 |
| D | validation_ext | 125 | 2540.3177 | 2290.3177 | 2258.7966 | 0.3760 |
| E | oos_ext | 118 | 3518.4948 | 3282.4948 | 3268.8037 | 0.5085 |
| E | train_ext | 231 | 9991.8236 | 9529.8236 | 9466.1948 | 0.4329 |
| E | validation_ext | 159 | 4044.7912 | 3726.7912 | 3714.9390 | 0.3459 |

## 5. Reverse Test
正向 (A) vs 反向 (E) 对比：
| variant | split | no_cost_pnl |
|---|---|---|
| A | oos_ext | 178.1135 |
| A | train_ext | 2234.8399 |
| A | validation_ext | 4406.0750 |
| E | oos_ext | 3518.4948 |
| E | train_ext | 9991.8236 |
| E | validation_ext | 4044.7912 |

## 6. 集中度
| variant | split | symbol | trade_count | total_no_cost_pnl | top_5pct_contribution | concentration_pass |
|---|---|---|---|---|---|---|
| A | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | -67.3473 | -1.5328 | True |
| A | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 549.0215 | 0.4932 | True |
| A | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 205.6092 | 0.5251 | True |
| B | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | -183.8720 | -0.0851 | True |
| B | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | -73.6122 | -1.1470 | True |
| B | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | -47.2569 | -0.6053 | True |
| BC | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | -31.1175 | -5.0986 | False |
| BC | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 1127.1980 | 0.3270 | True |
| BC | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 34.8135 | 5.5199 | False |
| C | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | 139.2982 | 1.7739 | True |
| C | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 1267.6787 | 0.2908 | True |
| C | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 79.3094 | 2.4230 | False |
| D | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 27 | 371.4263 | 0.2848 | True |
| D | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 42 | 1015.7821 | 0.4878 | True |
| D | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 21 | 97.9811 | 1.0617 | True |
| E | oos_ext | BTCUSDT_SWAP_OKX.GLOBAL | 23 | 387.3550 | 0.3457 | True |
| E | train_ext | BTCUSDT_SWAP_OKX.GLOBAL | 34 | 551.1238 | 0.2315 | True |
| E | validation_ext | BTCUSDT_SWAP_OKX.GLOBAL | 33 | 372.9221 | 0.4151 | True |
| A | oos_ext | ETHUSDT_SWAP_OKX.GLOBAL | 23 | 244.8485 | 0.6431 | True |
| A | train_ext | ETHUSDT_SWAP_OKX.GLOBAL | 37 | 269.7598 | 1.0163 | True |

## 7. Gate 裁决
- cost_aware_pass=True
- funding_adjusted_pass=True
- reverse_test_pass=False
- random_control_pass=False
- concentration_pass=False
- can_enter_phase2=False
- final_decision=postmortem_or_pause

## 8. 限制
- strategy_development_allowed=false
- demo_live_allowed=false
- 不修改 OkxAdaptiveMhfStrategy