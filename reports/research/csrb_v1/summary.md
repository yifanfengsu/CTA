# CSRB-v1 Phase 1 Research

## 1. CSRB-v1 是什么假设？
CSRB-v1 tests whether crypto session transitions create trend-following edge: a clean range in a lower-activity session may break when Europe or US participation arrives, and the breakout may carry directional order flow. This is research-only and uses next-bar-open fixed-hold benchmarks.

## 2. Session 定义是什么？
- Asia range: 00:00-07:59 UTC
- Europe breakout window: 08:00-11:59 UTC
- Europe range: 08:00-12:59 UTC
- US breakout window: 13:00-17:59 UTC

## 3. 数据范围是什么？
- data_start_report=2023-01-01T00:00:00+08:00
- data_end_exclusive_report=2026-04-01T00:00:00+08:00
- data_start_session=2022-12-31T16:00:00+00:00
- data_end_exclusive_session=2026-03-31T16:00:00+00:00
- session_timezone=UTC
- report_timezone=Asia/Shanghai
- funding_data_complete=true

## 4. Train / Validation / OOS 日期？
- train=2022-12-31T16:00:00+00:00 (2023-01-01T00:00:00+08:00) to 2024-12-12T06:24:00+00:00 (2024-12-12T14:24:00+08:00)
- validation=2024-12-12T06:24:00+00:00 (2024-12-12T14:24:00+08:00) to 2025-08-06T11:12:00+00:00 (2025-08-06T19:12:00+08:00)
- oos=2025-08-06T11:12:00+00:00 (2025-08-06T19:12:00+08:00) to 2026-03-31T16:00:00+00:00 (2026-04-01T00:00:00+08:00)

## 5. A/B/C/D/E 事件数量？
- A=19762
- B=2806
- C=4101
- D=6905
- E=6907

## 6. Event Group Summary
| group | session_type | timeframe | event_count | mean_future_return_16 |
|---|---|---|---|---|
| A | ordinary_rolling | 15m | 8736 | -0.0108 |
| A | ordinary_rolling | 1h | 4579 | -0.0071 |
| A | ordinary_rolling | 30m | 6447 | -0.0096 |
| B | asia_to_europe | 15m | 1132 | -0.0097 |
| B | asia_to_europe | 1h | 731 | -0.0079 |
| B | asia_to_europe | 30m | 943 | -0.0089 |
| C | europe_to_us | 15m | 2160 | -0.0049 |
| C | europe_to_us | 30m | 1941 | -0.0049 |
| D | random_time_control | 15m | 3292 | 0.0036 |
| D | random_time_control | 1h | 729 | 0.0096 |
| D | random_time_control | 30m | 2884 | 0.0071 |
| E | reverse_test | 15m | 3292 | 0.0066 |
| E | reverse_test | 1h | 731 | 0.0079 |
| E | reverse_test | 30m | 2884 | 0.0062 |

## 7. Core 15m Trade Result
| group | session_type | timeframe | split | trade_count | no_cost_pnl | cost_aware_pnl | funding_adjusted_pnl | max_drawdown_pct |
|---|---|---|---|---|---|---|---|---|
| B | asia_to_europe | 15m | oos | 239 | -2753.6872 | -3231.6872 | -3231.9012 | 3.4324 |
| B | asia_to_europe | 15m | train | 710 | -5177.3336 | -6597.3336 | -6597.0831 | 8.9498 |
| B | asia_to_europe | 15m | validation | 183 | -2974.3197 | -3340.3197 | -3340.2376 | 3.3637 |
| C | europe_to_us | 15m | oos | 458 | -2364.7192 | -3280.7192 | -3278.7909 | 3.0056 |
| C | europe_to_us | 15m | train | 1349 | -4295.5152 | -6993.5152 | -6988.6060 | 7.0272 |
| C | europe_to_us | 15m | validation | 351 | -3874.1239 | -4576.1239 | -4574.0106 | 4.1580 |

## 8. Required Answers
1. Asia→Europe 是否有效？false
2. Europe→US 是否有效？false
3. Session breakout 是否优于普通 breakout？ordinary_oos_no_cost_pnl=-9020.4801, session_oos_no_cost_pnl=-5118.4064
4. Session breakout 是否优于 random time control？[{'timeframe': '15m', 'source_session_type': 'core_session_breakout', 'split': 'oos', 'forward_trade_count': 697, 'random_trade_count': 646, 'forward_no_cost_pnl': -5118.4064257029, 'random_no_cost_pnl': 2768.5314259278, 'forward_cost_aware_pnl': -6512.4064257029, 'random_cost_aware_pnl': 1476.5314259278, 'forward_funding_adjusted_pnl': -6510.6921008783, 'random_funding_adjusted_pnl': 1474.9974813091, 'session_better': False}]
5. Reverse test 是否明显更差？false [{'timeframe': '15m', 'source_session_type': 'core_session_breakout', 'split': 'oos', 'forward_trade_count': 697, 'reverse_trade_count': 662, 'forward_no_cost_pnl': -5118.4064257029, 'reverse_no_cost_pnl': 4979.8245276681, 'forward_cost_aware_pnl': -6512.4064257029, 'reverse_cost_aware_pnl': 3655.8245276681, 'forward_funding_adjusted_pnl': -6510.6921008783, 'reverse_funding_adjusted_pnl': 3653.2979297835, 'reverse_weaker': False}]
6. no-cost 是否通过？train=false, validation=false, oos=false
7. cost-aware 是否通过？false
8. funding-adjusted 是否通过？false
9. 收益是否集中？concentration_pass=false [{'group': 'B', 'timeframe': '15m', 'split': 'oos', 'trade_count': 239, 'total_funding_adjusted_pnl': -3231.9011908008, 'largest_symbol_pnl_share': 0.4292039304, 'largest_symbol_pnl_symbol': 'ETHUSDT_SWAP_OKX.GLOBAL', 'largest_symbol_trade_count_share': 0.3849372385, 'largest_symbol_trade_count_symbol': 'BTCUSDT_SWAP_OKX.GLOBAL', 'top_5pct_trade_pnl_contribution': None, 'top_5pct_trade_pnl': 499.870478323, 'top_5pct_trade_count': 12, 'max_drawdown': 3461.6288350129, 'max_drawdown_pct': 3.4323612553}, {'group': 'C', 'timeframe': '15m', 'split': 'oos', 'trade_count': 458, 'total_funding_adjusted_pnl': -3278.7909100775, 'largest_symbol_pnl_share': 0.4443982138, 'largest_symbol_pnl_symbol': 'ETHUSDT_SWAP_OKX.GLOBAL', 'largest_symbol_trade_count_share': 0.3580786026, 'largest_symbol_trade_count_symbol': 'BTCUSDT_SWAP_OKX.GLOBAL', 'top_5pct_trade_pnl_contribution': None, 'top_5pct_trade_pnl': 1022.2822070502, 'top_5pct_trade_count': 23, 'max_drawdown': 3414.994879924, 'max_drawdown_pct': 3.0056178033}]
10. 是否允许 Phase 2？false
11. 是否允许修改正式策略？false
12. 是否允许 demo/live？false

## 9. Final Decision
- final_decision=postmortem
- continue_to_phase2=false
- strategy_development_allowed=false
- demo_live_allowed=false
