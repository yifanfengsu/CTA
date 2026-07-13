# MR-v1 Phase 2: Parameter Sweep & Robustness

## 1. Best Combination
- lookback=8, atr_mult=1.0, max_hold=60
- OOS funding-adjusted PnL: 5856.37
- Train: 19354.45 | Validation: 7741.92

## 2. Parameter Plateau
- Plateau found: True
- 26 combinations within 10% of best OOS
- Lookback range: [8, 15]
- ATR mult range: [1.0, 3.0]
- Max hold range: [50, 80]

## 3. Multi-Seed Random Control
- All 5 seeds pass: True
- Worst seed OOS: 2033.89
  - seed=42: OOS=2033.89, pass=True
  - seed=99: OOS=2566.57, pass=True
  - seed=137: OOS=3173.06, pass=True
  - seed=256: OOS=3493.89, pass=True
  - seed=512: OOS=3182.60, pass=True

## 4. Per-Symbol OOS PnL
| Symbol | OOS PnL | Train PnL | Val PnL | Trades |
|---|---|---|---|---|
| BTCUSDT_SWAP_OKX.GLOBAL | 906.43 | 1174.55 | 888.85 | 58 |
| ETHUSDT_SWAP_OKX.GLOBAL | 2457.52 | 946.24 | 1728.62 | 45 |
| SOLUSDT_SWAP_OKX.GLOBAL | 738.30 | 11406.34 | 1346.02 | 47 |
| LINKUSDT_SWAP_OKX.GLOBAL | 1174.14 | 2692.11 | 2375.45 | 46 |
| DOGEUSDT_SWAP_OKX.GLOBAL | 579.98 | 3135.21 | 1402.98 | 52 |

## 5. Gates
- plateau_found=True
- random_control_all_pass=True
- symbol_diversification=True
- can_enter_phase3=True

## 6. Top 10 Sweep Results (by OOS)
| lookback | atr_mult | max_hold | train | validation | oos |
|---|---|---|---|---|---|
| 8 | 1.0 | 60 | 19354 | 7742 | 5856 |
| 8 | 1.5 | 60 | 21383 | 6889 | 5762 |
| 8 | 1.0 | 65 | 21344 | 7846 | 5735 |
| 8 | 2.5 | 60 | 19518 | 5338 | 5728 |
| 8 | 1.0 | 80 | 22072 | 7874 | 5632 |
| 8 | 1.0 | 55 | 19174 | 7760 | 5620 |
| 15 | 3.0 | 60 | 10326 | 3241 | 5581 |
| 8 | 1.5 | 55 | 21462 | 6887 | 5546 |
| 8 | 1.0 | 70 | 22031 | 7448 | 5533 |
| 8 | 2.5 | 55 | 20112 | 5884 | 5484 |