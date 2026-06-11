# Trend V3.0 Postmortem：趋势跟踪失败归因

## 核心结论
- trend_following_v3_failed=true
- stable_candidate_exists=false
- proceed_to_v3_1=false
- 本报告只做 V3.0 复盘，不开发 Strategy V3，不改正式策略，不进入 demo/live。
- Funding 部分是 synthetic funding stress，不是真实 OKX funding fee。

## 1. V3.0 失败的主要原因是什么？
没有任何 policy 同时通过 train/validation/oos：OOS best policy 在 validation 为负，多数 Donchian、4h、ensemble family 跨样本为负，OOS 正收益又集中在单一 symbol 和极少数尾部交易。

## 2. 是 Donchian 失败，还是 EMA 失败，还是组合风控失败？
- Donchian overall weak=true。
- EMA better than Donchian=true，但 EMA 也没有跨 validation/OOS 稳定通过。
- 组合风控没有发现持仓上限或重复持仓类 audit 问题，但无法解决行情/品种收益集中与尾部依赖问题。

## 3. 1d EMA 是否值得进入 V3.1 继续研究？
- OOS best policy=v3_1d_ema_50_200_atr5，train no-cost=1.7866，validation no-cost=-0.8521，OOS no-cost=0.9277。
- 建议：proceed_to_v3_1=false；方向=['stop_current_v3_family', 'extend_history_before_v3_1', 'add_stronger_trend_regime_filter', 'keep_1d_ema_but_fix_concentration']。

## 4. OOS best policy 是否过度依赖单一 symbol？
- largest_symbol_pnl_share=0.845702，largest_symbol=BTCUSDT_SWAP_OKX.GLOBAL，worst_symbol=ETHUSDT_SWAP_OKX.GLOBAL。
- over_depends_single_symbol=true。

## 5. 去掉 top 1 / top 5% 盈利交易后是否仍为正？
- remove_top_1_pnl=-0.912748，still_positive=false。
- remove_top_5pct_pnl=-0.912748，still_positive=false。

## 6. Funding fee stress 后是否仍为正？
| funding_bps_per_8h | funding_adjusted_net_pnl | remains_positive_after_funding |
|---|---|---|
| 1.0000 | 0.6561 | true |
| 3.0000 | 0.2215 | true |
| 5.0000 | -0.2130 | false |
| 10.0000 | -1.2994 | false |

## 7. Validation 失败来自哪些 policy / symbol / month？
### Worst validation policies
| policy_name | policy_family | month | net_pnl | trade_count |
|---|---|---|---|---|
| v3_ensemble_core | ensemble | 2025-12 | -3.0301 | 49.0000 |
| v3_4h_donchian_20_10_atr4 | 4h_donchian | 2025-12 | -2.7377 | 16.0000 |
| v3_4h_ema_50_200_atr4 | 4h_ema | 2025-10 | -2.5372 | 41.0000 |
| v3_4h_ema_50_200_atr4 | 4h_ema | 2025-12 | -1.9979 | 44.0000 |
| v3_ensemble_core | ensemble | 2025-10 | -1.9655 | 40.0000 |

### Worst validation symbols
| policy_name | symbol | net_pnl | trade_count |
|---|---|---|---|
| v3_4h_donchian_100_30_atr5 | ETHUSDT_SWAP_OKX.GLOBAL | -2.0794 | 5.0000 |
| v3_4h_donchian_20_10_atr4 | BTCUSDT_SWAP_OKX.GLOBAL | -1.5696 | 9.0000 |
| v3_1d_ema_50_200_atr5 | BTCUSDT_SWAP_OKX.GLOBAL | -1.4367 | 3.0000 |
| v3_ensemble_core | BTCUSDT_SWAP_OKX.GLOBAL | -1.1867 | 19.0000 |
| v3_4h_ema_50_200_atr4 | ETHUSDT_SWAP_OKX.GLOBAL | -1.1749 | 20.0000 |

### Worst validation months
| month | net_pnl | trade_count | win_rate |
|---|---|---|---|
| 2025-10 | -11.0718 | 133.0000 | 0.1880 |
| 2025-12 | -8.0258 | 145.0000 | 0.2000 |
| 2025-11 | 3.0845 | 82.0000 | 0.4390 |

## 8. 是否建议进入 V3.1？
- proceed_to_v3_1=false。
- red_flags=['no_stable_candidate', 'validation_no_cost_negative', 'symbol_concentration_high', 'top_trade_concentration_high', 'remove_top_1_turns_nonpositive', 'remove_top_5pct_turns_nonpositive', 'funding_5bps_turns_nonpositive', 'funding_10bps_turns_nonpositive']。
- 即使未来允许 V3.1，也只能是研究设计，不能进入 Strategy V3 原型、demo 或 live。

## 9. 如果进入 V3.1，应该研究什么？
- recommended_research_direction=['stop_current_v3_family', 'extend_history_before_v3_1', 'add_stronger_trend_regime_filter', 'keep_1d_ema_but_fix_concentration']。

## 10. 如果不进入 V3.1，应该停止哪些 policy family？
- not_recommended_direction=['reduce_4h_donchian_family', 'do_not_expand_parameter_search', 'do_not_enter_demo_live', 'do_not_develop_strategy_v3_from_current_results']。
- 4h Donchian、vol compression、ensemble 当前不应继续扩大参数搜索；1d EMA 只能作为集中度/行情过滤约束下的研究假设保留。

## Policy Family OOS Snapshot
| policy_family | trade_count | no_cost_net_pnl | net_pnl | win_rate | profit_factor | stable_direction |
|---|---|---|---|---|---|---|
| 1d_ema | 18.0000 | 0.9277 | 0.8734 | 0.2778 | 1.5535 | oos_rebound_validation_failed |
| risk_filtered | 9.0000 | 0.2873 | 0.2559 | 0.1111 | 1.2805 | oos_rebound_validation_failed |
| ensemble | 99.0000 | -0.8628 | -1.1529 | 0.2323 | 0.7101 | negative_all_splits |
| 4h_ema | 99.0000 | -1.1879 | -1.4897 | 0.2121 | 0.6497 | negative_all_splits |
| vol_compression | 31.0000 | -2.1897 | -2.2916 | 0.2258 | 0.0866 | negative_all_splits |
| 1d_donchian | 13.0000 | -4.6566 | -4.7075 | 0.3846 | 0.2986 | validation_only_reversal |
| 4h_donchian | 88.0000 | -8.3680 | -8.6576 | 0.2045 | 0.3907 | negative_all_splits |

## 输出文件
- trend_v3_postmortem_summary.json
- trend_v3_postmortem_report.md
- policy_family_analysis.csv
- symbol_contribution_postmortem.csv
- by_month.csv
- by_quarter.csv
- by_symbol_month.csv
- by_policy_month.csv
- top_trade_concentration.csv
- funding_sensitivity.csv
- rejected_candidate_reasons.csv
- v3_1_recommendations.json

## Warnings
- 无
