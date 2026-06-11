# External Regime Classifier Gate Consistency Audit

## Guardrails
- strategy_development_allowed=false
- demo_live_allowed=false
- not_tradable=true
- strict gate uses Dossier / Extended V3 concentration semantics.

## Required Questions
1. original_all 为什么会 stable_candidate_like=true？
   - 旧 classifier gate 的 top_5pct_trade_pnl_contribution 使用正收益合计作分母，低估了 top-trade concentration；Extended V3 使用 top 5% net_pnl / total net_pnl。
2. 这个判断是否与 Dossier / Extended V3 compare 一致？
   - false。strict original_all_pass=false；old_original_all_pass=false。
3. stable_candidate_like 是否漏掉 top trade concentration？
   - 旧版本漏掉。v3_1d_ema_50_200_atr5 original_all oos_top_5pct_trade_pnl_contribution=1.9818。
4. stable_candidate_like 是否漏掉 largest symbol concentration？
   - strict gate 已检查。v3_1d_ema_50_200_atr5 original_all oos_largest_symbol_pnl_share=0.3462。
5. classifier filters 是否真正改变了 OOS trade set？
   - changed_oos_count=33，unchanged_oos_count=27。
6. trend_friendly 在 OOS 是否太稀少？
   - 是。最近 research report 显示 oos_ext trend_friendly 约 0.4%，不足以支撑强结论。
7. 是否允许进入 research-only V3.1 classifier experiment？
   - can_enter_research_only_v3_1_classifier_experiment=false。
8. 是否允许 Strategy V3 / demo / live？
   - strategy_development_allowed=false
   - demo_live_allowed=false

## Original All Detail
- original_all_strict_rejected_reasons=oos_top_5pct_trade_pnl_contribution_over_0p8

## Gate Comparison Sample
| filter_name | policy_name | old | strict | rejected_reasons |
|---|---|---|---|---|
| original_all | v3_1d_ema_50_200_atr5 | false | false | oos_top_5pct_trade_pnl_contribution_over_0p8 |
| exclude_hostile_chop_overheated | v3_1d_ema_50_200_atr5 | false | false | oos_top_5pct_trade_pnl_contribution_over_0p8 |
| exclude_funding_overheated | v3_1d_ema_50_200_atr5 | false | false | oos_top_5pct_trade_pnl_contribution_over_0p8 |
| original_all | v3_4h_donchian_55_with_risk_filters | false | false | validation_no_cost_net_pnl_not_positive;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8 |
| exclude_hostile_chop_overheated | v3_4h_donchian_55_with_risk_filters | false | false | validation_no_cost_net_pnl_not_positive;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8 |
| exclude_funding_overheated | v3_4h_donchian_55_with_risk_filters | false | false | validation_no_cost_net_pnl_not_positive;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8 |
| keep_trend_friendly | v3_1d_donchian_20_10_atr4 | false | false | train_no_cost_net_pnl_not_positive;validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;train_ext_trade_count_under_10;validation_ext_trade_count_under_10;oos_ext_trade_count_under_10;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8;oos_not_enough_active_symbols |
| keep_trend_friendly_exclude_funding_overheated | v3_1d_donchian_20_10_atr4 | false | false | train_no_cost_net_pnl_not_positive;validation_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;train_ext_trade_count_under_10;validation_ext_trade_count_under_10;oos_ext_trade_count_under_10;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8;oos_not_enough_active_symbols |
| keep_trend_friendly | v3_1d_donchian_55_20_atr5 | false | false | train_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;train_ext_trade_count_under_10;validation_ext_trade_count_under_10;oos_ext_trade_count_under_10;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8;oos_not_enough_active_symbols |
| keep_trend_friendly_exclude_funding_overheated | v3_1d_donchian_55_20_atr5 | false | false | train_no_cost_net_pnl_not_positive;oos_no_cost_net_pnl_not_positive;train_ext_trade_count_under_10;validation_ext_trade_count_under_10;oos_ext_trade_count_under_10;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8;oos_not_enough_active_symbols |
| keep_trend_friendly | v3_1d_ema_50_200_atr5 | false | false | oos_no_cost_net_pnl_not_positive;train_ext_trade_count_under_10;validation_ext_trade_count_under_10;oos_ext_trade_count_under_10;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8;oos_not_enough_active_symbols |
| keep_trend_friendly_exclude_funding_overheated | v3_1d_ema_50_200_atr5 | false | false | oos_no_cost_net_pnl_not_positive;train_ext_trade_count_under_10;validation_ext_trade_count_under_10;oos_ext_trade_count_under_10;oos_largest_symbol_pnl_share_over_0p7;oos_top_5pct_trade_pnl_contribution_over_0p8;oos_not_enough_active_symbols |

## OOS Filters That Did Not Change Trade Set
- original_all / v3_1d_donchian_20_10_atr4: filter_did_not_affect_oos=true
- exclude_hostile_chop_overheated / v3_1d_donchian_20_10_atr4: filter_did_not_affect_oos=true
- exclude_funding_overheated / v3_1d_donchian_20_10_atr4: filter_did_not_affect_oos=true
- original_all / v3_1d_donchian_55_20_atr5: filter_did_not_affect_oos=true
- exclude_hostile_chop_overheated / v3_1d_donchian_55_20_atr5: filter_did_not_affect_oos=true
- exclude_funding_overheated / v3_1d_donchian_55_20_atr5: filter_did_not_affect_oos=true
- original_all / v3_1d_ema_50_200_atr5: filter_did_not_affect_oos=true
- exclude_hostile_chop_overheated / v3_1d_ema_50_200_atr5: filter_did_not_affect_oos=true
- exclude_funding_overheated / v3_1d_ema_50_200_atr5: filter_did_not_affect_oos=true
- original_all / v3_4h_donchian_100_30_atr5: filter_did_not_affect_oos=true
- exclude_hostile_chop_overheated / v3_4h_donchian_100_30_atr5: filter_did_not_affect_oos=true
- exclude_funding_overheated / v3_4h_donchian_100_30_atr5: filter_did_not_affect_oos=true
- original_all / v3_4h_donchian_20_10_atr4: filter_did_not_affect_oos=true
- exclude_hostile_chop_overheated / v3_4h_donchian_20_10_atr4: filter_did_not_affect_oos=true
- exclude_funding_overheated / v3_4h_donchian_20_10_atr4: filter_did_not_affect_oos=true
- original_all / v3_4h_donchian_55_20_atr4: filter_did_not_affect_oos=true
- exclude_funding_overheated / v3_4h_donchian_55_20_atr4: filter_did_not_affect_oos=true
- original_all / v3_4h_donchian_55_with_risk_filters: filter_did_not_affect_oos=true
- exclude_hostile_chop_overheated / v3_4h_donchian_55_with_risk_filters: filter_did_not_affect_oos=true
- exclude_funding_overheated / v3_4h_donchian_55_with_risk_filters: filter_did_not_affect_oos=true

## Decision
- can_enter_research_only_v3_1_classifier_experiment=false
- strategy_development_allowed=false
- demo_live_allowed=false
- reason=No non-original classifier filter passed strict gates after Dossier-consistent concentration checks.
