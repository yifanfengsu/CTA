# Trend V3 Actual Funding Analysis

## Core Conclusion
- available_data_only=false
- funding_data_complete=true
- verify_summary_missing=false
- verify_incomplete_reason=[]
- missing_before_first_available=false
- funding_event_coverage_warning=
- decision_rule=available_data_only results are not decision-grade and cannot unlock Strategy V3, V3.1, demo, or live.

## Scope
- This is funding-aware trend-following research only; it is not Strategy V3 development.
- mode=conservative
- alignment_rule=inclusive trade holding interval: entry_time <= funding_time <= exit_time
- notional_rule=entry_price * abs(volume) * abs(contract_size); fallback turnover/2; mark price is not fabricated
- signed_mode_assumption=OKX fundingRate sign research convention: positive rate means long pays short; negative rate means short pays long. This is a research assumption and should be rechecked before any production use.
- warning=mark price is unavailable in the current V3 trade files, so funding notional uses entry price approximation.

## Required Answers
1. Funding 数据是否完整？true
2. 哪些 symbol funding 数据有缺口？[]
3. V3 extended OOS best policy 在 actual funding 后是否仍为正？conservative=1.177729, signed=1.401369, positive=true
4. v3_1d_ema_50_200_atr5 在 conservative mode 后是否仍为正？true
5. v3_1d_ema_50_200_atr5 在 signed mode 后是否仍为正？true
6. 是否有任何 policy 在 train_ext / validation_ext / oos_ext funding-adjusted 后全部为正？conservative=['v3_1d_ema_50_200_atr5'], signed=['v3_1d_ema_50_200_atr5']
7. Funding 是否会使当前唯一弱线索彻底失效？false
8. 是否允许进入 funding-aware V3.1 research？false
9. 是否仍禁止 Strategy V3 / demo / live？true

## Target And OOS Best Policy Rows
| policy_name | split | original_net_pnl | funding_adjusted_net_pnl_conservative | funding_adjusted_net_pnl_signed | funding_events_count |
|---|---|---|---|---|---|
| v3_1d_ema_50_200_atr5 | oos_ext | 1.552369 | 1.177729 | 1.401369 | 2103.000000 |
| v3_1d_ema_50_200_atr5 | train_ext | 2.867049 | 2.068762 | 2.217137 | 4022.000000 |
| v3_1d_ema_50_200_atr5 | validation_ext | 1.824511 | 1.146581 | 1.337242 | 2703.000000 |

## Funding Coverage Warning
- zero_funding_event_splits_when_incomplete=[]
- funding_events_count=0 under incomplete funding coverage is interpreted as likely_due_to_missing_funding_coverage, not as evidence that no funding occurred.

## Target Policy Across Splits
| policy_name | split | original_net_pnl | funding_adjusted_net_pnl_conservative | funding_adjusted_net_pnl_signed | funding_events_count |
|---|---|---|---|---|---|
| v3_1d_ema_50_200_atr5 | train_ext | 2.867049 | 2.068762 | 2.217137 | 4022.000000 |
| v3_1d_ema_50_200_atr5 | validation_ext | 1.824511 | 1.146581 | 1.337242 | 2703.000000 |
| v3_1d_ema_50_200_atr5 | oos_ext | 1.552369 | 1.177729 | 1.401369 | 2103.000000 |

## Final Gates
- funding_adjusted_stable_candidate_exists=false
- can_enter_funding_aware_v3_1_research=false
- strategy_development_allowed=false
- demo_live_allowed=false
