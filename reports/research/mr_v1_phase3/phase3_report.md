# MR-v1 Phase 3: Formal 4h Backtest

## Parameters
- lookback=8, atr_stop=1.0×ATR, max_hold=60 bars
- Notional: $1000 per trade, Fees: 5.0bps/side, Slippage: 5.0bps/side

## Overall Metrics (funding-adjusted)
| Metric | Value |
|---|---|
| Total Trades | 2216 |
| Total PnL | $4,048.07 |
| Win Rate | 43.3% |
| Avg Trade | $1.83 |
| Sharpe (annualized) | 0.51 |
| Max Drawdown | -408.47% |
| Profit Factor | 1.11 |

## Per-Split Metrics
| Split | Trades | PnL | Win% |
|---|---|---|---|
| train_ext | 1071 | $6,007 | 45.5% |
| validation_ext | 670 | $-958 | 37.5% |
| oos_ext | 475 | $-1,002 | 46.7% |

## OOS Per-Symbol
| Symbol | Trades | PnL | Win% |
|---|---|---|---|
| BTCUSDT | 107 | $166 | 57.0% |
| ETHUSDT | 97 | $-38 | 43.3% |
| SOLUSDT | 80 | $-931 | 36.2% |
| LINKUSDT | 99 | $79 | 49.5% |
| DOGEUSDT | 92 | $-277 | 44.6% |

## Exit Reason Breakdown
| Reason | Count | PnL |
|---|---|---|
| end_of_data | 1 | $-1 |
| midline | 1007 | $40,262 |
| stop | 1208 | $-36,213 |

## Monthly PnL
| Month | PnL |
|---|---|
| 2023-01 | $4,520 |
| 2023-02 | $-295 |
| 2023-03 | $95 |
| 2023-04 | $1,130 |
| 2023-05 | $9 |
| 2023-06 | $301 |
| 2023-07 | $576 |
| 2023-08 | $-615 |
| 2023-09 | $-73 |
| 2023-10 | $-380 |
| 2023-11 | $179 |
| 2023-12 | $-42 |
| 2024-01 | $-572 |
| 2024-02 | $-267 |
| 2024-03 | $892 |
| 2024-04 | $515 |
| 2024-05 | $-390 |
| 2024-06 | $438 |
| 2024-07 | $535 |
| 2024-08 | $430 |
| 2024-09 | $-565 |
| 2024-10 | $-685 |
| 2024-11 | $-378 |
| 2024-12 | $-430 |
| 2025-01 | $-404 |
| 2025-02 | $242 |
| 2025-03 | $-94 |
| 2025-04 | $74 |
| 2025-05 | $945 |
| 2025-06 | $-663 |
| 2025-07 | $364 |
| 2025-08 | $-753 |
| 2025-09 | $-228 |
| 2025-10 | $-699 |
| 2025-11 | $283 |
| 2025-12 | $336 |
| 2026-01 | $-503 |
| 2026-02 | $-277 |
| 2026-03 | $500 |
