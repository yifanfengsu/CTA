# Trend Following V3 Extended 跨样本比较

## 核心结论
- split_scheme=extended, split_labels={'train': 'train_ext', 'validation': 'validation_ext', 'oos': 'oos_ext'}
- stable_candidate_exists=false
- stable_candidates=[]
- all_no_cost_positive_policies=['v3_1d_ema_50_200_atr5']
- oos_cost_aware_positive_policies=['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters']
- oos_cost_aware_nonnegative_policies=['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters']
- overfit_risk=true
- trend_following_v3_failed=true
- trend_following_v3_extended_failed=True
- can_enter_v3_1_research=false
- can_enter_strategy_v3_prototype=false
- demo_live_allowed=false
- stable_candidate 需要 train_ext/validation_ext/oos_ext no-cost 均为正、OOS 成本后不亏、OOS 回撤不超过 30%、三段交易次数均 >=10、OOS 不依赖单一 symbol 或极少数交易。
- Funding 部分是 synthetic funding stress，不是真实 OKX funding fee。

## 稳定候选
- 无

## 被拒候选与原因
- v3_1d_ema_50_200_atr5: oos_top_5pct_trade_pnl_contribution_over_0p8
- v3_4h_donchian_55_with_risk_filters: validation_no_cost_net_pnl_not_positive;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8
- v3_1d_donchian_20_10_atr4: oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative;oos_largest_symbol_pnl_share_over_0p7
- v3_1d_donchian_55_20_atr5: train_no_cost_net_pnl_not_positive;validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative;oos_trade_count_under_10;oos_largest_symbol_pnl_share_over_0p7
- v3_4h_vol_compression_donchian_breakout: train_no_cost_net_pnl_not_positive;validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative;oos_largest_symbol_pnl_share_over_0p7
- v3_4h_donchian_100_30_atr5: validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative;oos_largest_symbol_pnl_share_over_0p7
- v3_4h_ema_50_200_atr4: oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative
- v3_ensemble_core: validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative
- v3_4h_donchian_20_10_atr4: train_no_cost_net_pnl_not_positive;validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative;oos_largest_symbol_pnl_share_over_0p7
- v3_4h_donchian_55_20_atr4: validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;oos_cost_aware_net_pnl_negative;oos_largest_symbol_pnl_share_over_0p7

## 风险集中度
- high_largest_symbol_pnl_share_policies=['v3_4h_donchian_55_with_risk_filters', 'v3_1d_donchian_20_10_atr4', 'v3_1d_donchian_55_20_atr5', 'v3_4h_vol_compression_donchian_breakout', 'v3_4h_donchian_100_30_atr5', 'v3_4h_donchian_20_10_atr4', 'v3_4h_donchian_55_20_atr4']
- high_top_5pct_trade_pnl_contribution_policies=['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters']
- oos_single_symbol_or_tail_trade_dependency_risk=true

## Funding Stress
| policy_name | funding_bps_per_8h | original_net_pnl | funding_adjusted_net_pnl | remains_positive_after_funding |
|---|---|---|---|---|
| v3_1d_ema_50_200_atr5 | 1.0000 | 1.5524 | 0.9726 | true |
| v3_1d_ema_50_200_atr5 | 3.0000 | 1.5524 | -0.1870 | false |
| v3_1d_ema_50_200_atr5 | 5.0000 | 1.5524 | -1.3466 | false |
| v3_1d_ema_50_200_atr5 | 10.0000 | 1.5524 | -4.2455 | false |

## 输出文件
- trend_v3_extended_compare_summary.json
- trend_v3_extended_compare_leaderboard.csv
- trend_v3_extended_compare_report.md
