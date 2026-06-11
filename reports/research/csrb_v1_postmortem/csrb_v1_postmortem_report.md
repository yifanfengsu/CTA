# CSRB-v1 Postmortem Report

## Executive Decision
- csrb_v1_failed=true
- csrb_trend_following_hypothesis_failed=true
- possible_false_breakout_research_hypothesis=true
- random_control_requires_review=true
- continue_to_phase2=false
- parameter_plateau_allowed=false
- randomization_allowed=false
- strategy_development_allowed=false
- demo_live_allowed=false

## Required Answers
1. CSRB-v1 Phase 1 是否失败？是，csrb_v1_failed=true。
2. 是否发现实现或数据问题？possible_implementation_issue=false，market_data_complete=True，funding_data_complete=True。
3. B/C/core 为什么失败？B、C、core 在 primary 15m 的 train/validation/oos 均为负；core_no_cost_pnl=-21439.6988。
4. Session breakout 是否只是比 ordinary breakout 少亏，而不是有 edge？true；即使少亏，core 仍为负，不能构成 edge。
5. Random time control 为什么为正？当前 D primary no-cost=11823.4135，说明本次 session timing 不优于同结构随机时间，且需要 robustness 审计。
6. 多 seed random control 是否仍为正？mean_no_cost=-15188.8741，positive_seed_share=0.0000，multi_seed_mean_positive=false。
7. Reverse E 是否明显优于正向？true，reverse_E_oos_funding_adjusted_positive=true。
8. 这是否说明 CSRB 趋势延续假设失败？是，csrb_trend_following_hypothesis_failed=true。
9. 是否存在 false-breakout research 线索？possible_false_breakout_research_hypothesis=true，但 E 不能作为趋势跟踪通过或可交易策略。
10. 是否允许 Phase 2？否，continue_to_phase2=false。
11. 是否允许修改正式策略？否，strategy_development_allowed=false。
12. 是否允许 demo/live？否，demo_live_allowed=false。
13. 下一步建议是什么？Archive CSRB-v1 trend-following. A separate pre-registered false-breakout research hypothesis may be proposed, but E is not a tradable strategy and cannot rescue CSRB-v1.

## Data / Implementation Sanity
| check_name | status | value |
|---|---|---|
| market_data_complete | pass | True |
| funding_data_complete | pass | True |
| core_event_count_nonzero | pass | 3292 |
| core_trade_count_enough | pass | {"oos": 697, "train": 2059, "validation": 534} |
| entry_uses_next_open | pass | 0 |
| exit_uses_open_t_plus_hold_plus_one | pass | 0 |
| funding_adjusted_available | pass | True |
| reverse_E_strictly_from_BC | pass | True |
| random_D_avoids_breakout_window | pass | 0 |
| random_D_control_key_available | pass | True |
| same_bar_long_short_base_anomaly | pass | 0 |
| skipped_events_due_to_single_position_filter | warning | 9695 |
| possible_implementation_issue | pass | False |

## Session Failure Decomposition
| scope | split | trade_count | no_cost_pnl | cost_aware_pnl | funding_adjusted_pnl | has_positive_edge |
|---|---|---|---|---|---|---|
| core_session_breakout | train | 2059 | -9472.8488 | -13590.8488 | -13585.6891 | False |
| core_session_breakout | validation | 534 | -6848.4436 | -7916.4436 | -7914.2482 | False |
| core_session_breakout | oos | 697 | -5118.4064 | -6512.4064 | -6510.6921 | False |
| core_session_breakout | all | 3290 | -21439.6988 | -28019.6988 | -28010.6293 | False |

## Random Control Audit
| check_name | status | core_value | random_value | details |
|---|---|---|---|---|
| event_count_match | pass | 3292.0000 | 3292.0000 | D should preserve B+C primary event count |
| long_ratio_match | pass | 0.5058 | 0.5058 | D should preserve aggregate direction ratio |
| symbol_distribution | pass | 0.0000 | 0.0000 | max absolute bucket event-count difference |
| timeframe_distribution | pass | 0.0000 | 0.0000 | max absolute bucket event-count difference |
| split_distribution | warning | 0.0000 | 3.0000 | max absolute bucket event-count difference |
| symbol_timeframe_direction_distribution | pass | 0.0000 | 0.0000 | max absolute bucket event-count difference |
| random_avoids_breakout_window | pass | 0.0000 | 0.0000 | D timestamps must be outside source breakout windows |
| random_control_no_cost_positive | warning | -21439.6988 | 11823.4135 | positive random control requires robustness review |

## Random Control Seed Robustness
| index | seed | sample_count | no_cost_pnl | funding_adjusted_pnl |
|---|---|---|---|---|
| count | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| unique | None | None | None | None |
| top | None | None | None | None |
| freq | None | None | None | None |
| mean | 49.5000 | 3290.0000 | -15188.8741 | -21776.5977 |
| std | 29.0115 | 0.0000 | 2869.1988 | 2868.3698 |
| min | 0.0000 | 3290.0000 | -23226.9290 | -29810.8913 |
| 25% | 24.7500 | 3290.0000 | -16779.9378 | -23363.1542 |
| 50% | 49.5000 | 3290.0000 | -15076.4456 | -21668.2232 |
| 75% | 74.2500 | 3290.0000 | -13026.6153 | -19611.4805 |
| max | 99.0000 | 3290.0000 | -8861.0484 | -15444.1442 |

## Reverse Directionality
| timeframe | split | forward_no_cost_pnl | reverse_no_cost_pnl | forward_funding_adjusted_pnl | reverse_funding_adjusted_pnl | reverse_better_funding_adjusted |
|---|---|---|---|---|---|---|
| 15m | train | -9472.8488 | 9414.4677 | -13585.6891 | 5528.8364 | True |
| 15m | validation | -6848.4436 | 6790.0425 | -7914.2482 | 5751.6762 | True |
| 15m | oos | -5118.4064 | 4979.8245 | -6510.6921 | 3653.2979 | True |
| 15m | all | -21439.6988 | 21184.3347 | -28010.6293 | 14933.8106 | True |

## Symbol / Direction / Timeframe / Session Notes
- BTC/ETH/SOL 拖累最大：SOLUSDT_SWAP_OKX.GLOBAL
- long/short 拖累最大：short
- 正收益 timeframe：[]
- Asia→Europe / Europe→US 哪个更差：asia_to_europe
- 局部正收益 symbol+session：[]

## Conflict Filter
- core_event_count=6907
- core_trade_count=6899
- untraded_event_count=8
- untraded_event_rate=0.0012
- single_position_filter_may_distort_result=False

## Guardrails
- no_parameter_tuning_allowed=true
- parameter_plateau_allowed=false
- randomization_allowed=false
- strategy_development_allowed=false
- demo_live_allowed=false
- D random control is not tradable from this postmortem.
- E reverse test is not a trend-following edge and is not tradable from this postmortem.
