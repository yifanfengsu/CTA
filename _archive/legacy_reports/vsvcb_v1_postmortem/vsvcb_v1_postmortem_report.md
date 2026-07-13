# VSVCB-v1 Postmortem Report

## Executive Decision
- vsvcb_v1_failed=true
- trend_following_hypothesis_failed=true
- continue_to_phase2=false
- parameter_plateau_allowed=false
- randomization_allowed=false
- strategy_development_allowed=false
- demo_live_allowed=false

## Required Answers
1. VSVCB-v1 Phase 1 是否失败？是，vsvcb_v1_failed=true。
2. 是否发现实现或数据问题？possible_implementation_issue=false，market_data_complete=True，funding_data_complete=True。
3. D 组为什么失败？primary 15m train/validation/oos no-cost 分别为 -4030.6580 / -6083.4356 / -2496.6260，cost-aware 和 funding-adjusted OOS 仍为负。
4. D 组是否只是少亏，而不是有 edge？是，ablation_pass_but_no_edge=true，D 相对 A/B/C 多数只是亏得更少，未转正。
5. 反向 E 组为什么优于 D？E 是 D 的机械反向；no-cost 通常接近 -D，成本后 primary OOS 仍优于 D，reverse_test_failure=true。
6. 这是否说明 VSVCB 趋势延续假设失败？是，trend_following_hypothesis_failed=true。
7. 是否存在 false-breakout research 线索？possible_false_breakout_research_hypothesis=true，但不能把 E 标记为趋势跟踪 edge。
8. 是否允许 Phase 2？否，continue_to_phase2=false，parameter_plateau_allowed=false。
9. 是否允许修改正式策略？否，strategy_development_allowed=false，official_strategy_modification_allowed=false。
10. 是否允许 demo/live？否，demo_live_allowed=false。
11. 下一步建议是什么？Archive VSVCB-v1 trend-following. A separate pre-registered false-breakout research hypothesis may be proposed, but E is not a tradable strategy and cannot rescue Phase 1.

## Data / Implementation Sanity
| check_name | status | value |
|---|---|---|
| market_data_complete | pass | True |
| funding_data_complete | pass | True |
| D_event_count_nonzero | pass | 4928 |
| D_trade_count_enough | pass | {"oos": 508, "train": 1651, "validation": 387} |
| entry_uses_next_open | pass | 0 |
| funding_adjusted_available | pass | True |
| reverse_E_strictly_from_D | pass | True |
| same_bar_long_short_base_anomaly | pass | 0 |
| skipped_events_due_to_single_position_filter | warning | 24650 |
| possible_implementation_issue | pass | False |

## D vs A/B/C Ablation
| baseline_group | d_trade_count | baseline_trade_count | d_no_cost_pnl | baseline_no_cost_pnl | d_better_than_baseline | improvement_source |
|---|---|---|---|---|---|---|
| A | 508 | 2211 | -2496.6260 | -13396.7123 | True | signal_quality_or_payoff_improvement |
| B | 508 | 807 | -2496.6260 | -2844.4550 | True | fewer_trades_less_loss |
| C | 508 | 1355 | -2496.6260 | -9754.5801 | True | signal_quality_or_payoff_improvement |

## Reverse E
| timeframe | split | d_no_cost_pnl | e_no_cost_pnl | d_cost_aware_pnl | e_cost_aware_pnl | reverse_test_failure |
|---|---|---|---|---|---|---|
| 15m | oos | -2496.6260 | 2496.6260 | -3512.6260 | 1480.6260 | True |

## Symbol / Direction / Timeframe Notes
- BTC/ETH/SOL 拖累最大：SOLUSDT_SWAP_OKX.GLOBAL
- 多头/空头拖累最大：short
- 正收益 timeframe：[]
- 局部正收益 symbol/timeframe 样本需独立验证，不能据此调参。

## Horizon Path
- best_horizon=3
- horizon_3_better_than_10=True
- horizon_5_better_than_10=False
- horizon_20_worse_than_10=True
- early_reversal_likely=True
- mfe_exists_but_fixed_hold_wasted=True
- mae_dominates_mfe=True

## Conflict Filter
- d_event_count=4928
- d_trade_count=4220
- untraded_event_count=708
- untraded_event_rate=0.1437
- single_position_filter_may_distort_result=True

## Guardrails
- no_parameter_tuning_allowed=true
- parameter_plateau_allowed=false
- randomization_allowed=false
- strategy_development_allowed=false
- demo_live_allowed=false
- E group reverse result is not a trend-following edge and is not tradable from this postmortem.
