# External Regime Classifier Research

## Guardrails
- research_only=true
- not_tradable=true
- strategy_development_allowed=false
- demo_live_allowed=false
- classifier features exclude policy PnL, future returns, and future regime information.
- thresholds_source_split=train_ext
- validation_ext and oos_ext are validation only.

## Regime Distribution
- trend_friendly: train_ext=0.104, validation_ext=0.019, oos_ext=0.004, stable=true
- trend_hostile: train_ext=0.005, validation_ext=0.003, oos_ext=0.000, stable=true
- funding_overheated: train_ext=0.031, validation_ext=0.011, oos_ext=0.000, stable=true

## Required Questions
1. 是否存在独立于 V3 PnL 的 trend-friendly regime？
   - 是。label 由 market/funding 特征和 train_ext 分位数生成，不使用 V3 PnL。
2. trend-friendly regime 在 train / validation / oos 的分布是否稳定？
   - true。
3. 当前 V3 trades 是否主要亏在 trend_hostile / high_vol_chop？
   - false，loss_share=0.093。
4. v3_1d_ema_50_200_atr5 在 trend-friendly regime 下是否改善？
   - false。
5. 过滤 funding_overheated 后是否改善？
   - improved_policy_count=0。
6. 是否有任何 filtered result 在 train/validation/oos 都为正？
   - false。
7. 过滤后是否仍有 top trade concentration？
   - true。
8. 过滤后是否仍有 symbol concentration？
   - true。
9. 是否允许进入 research-only V3.1 classifier-filtered experiment？
   - can_enter_research_only_v3_1_classifier_experiment=false。
10. 是否允许 Strategy V3 / demo / live？
   - strategy_development_allowed=false
   - demo_live_allowed=false

## Filter Experiment Top Rows
| filter_name | policy_name | train no-cost | validation no-cost | oos no-cost | oos cost | oos funding-adjusted | stable_candidate_like |
|---|---|---:|---:|---:|---:|---:|---|
| original_all | v3_1d_ema_50_200_atr5 | 3.0200 | 2.0248 | 1.7022 | 1.5524 | 1.1777 | false |
| exclude_hostile_chop_overheated | v3_1d_ema_50_200_atr5 | 3.4942 | 2.6991 | 1.7022 | 1.5524 | 1.1777 | false |
| exclude_funding_overheated | v3_1d_ema_50_200_atr5 | 3.0354 | 2.0248 | 1.7022 | 1.5524 | 1.1777 | false |
| original_all | v3_4h_donchian_55_with_risk_filters | 3.2237 | -0.3198 | 1.2400 | 1.1679 | 1.0557 | false |
| exclude_hostile_chop_overheated | v3_4h_donchian_55_with_risk_filters | 3.2790 | -0.3198 | 1.2400 | 1.1679 | 1.0557 | false |
| exclude_funding_overheated | v3_4h_donchian_55_with_risk_filters | 3.2548 | -0.3198 | 1.2400 | 1.1679 | 1.0557 | false |
| keep_trend_friendly | v3_1d_donchian_20_10_atr4 | -0.8978 | -0.0159 | 0.0000 | 0.0000 |  | false |
| keep_trend_friendly_exclude_funding_overheated | v3_1d_donchian_20_10_atr4 | -0.8978 | -0.0159 | 0.0000 | 0.0000 |  | false |
| keep_trend_friendly | v3_1d_donchian_55_20_atr5 | -1.2406 | 0.0467 | 0.0000 | 0.0000 |  | false |
| keep_trend_friendly_exclude_funding_overheated | v3_1d_donchian_55_20_atr5 | -1.2406 | 0.0467 | 0.0000 | 0.0000 |  | false |

## Decision
- can_enter_research_only_v3_1_classifier_experiment=false
- strategy_development_allowed=false
- demo_live_allowed=false
- reason=No filtered policy passed all research-only gates across train_ext, validation_ext, and oos_ext.
