# Trend Opportunity Map

This report is a research-only diagnostic. Trend labels are ex-post labels and must not be used as entry signals.

## Decision Flags
- enough_trend_opportunities=true
- trend_opportunities_are_diversified=true
- legacy_strategies_failed_to_capture_trends=true
- legacy_strategies_trade_too_much_in_nontrend=false
- pre_trend_features_exist=false
- recommended_next_research_direction=trend_exit_convexity_research
- strategy_development_allowed=false
- demo_live_allowed=false
- tradable=false

## Opportunity Summary
- trend_segment_count=1581
- uptrend_count=734
- downtrend_count=847
- trend_opportunity_days_ratio=100.00%
- strongest_symbol=DOGEUSDT_SWAP_OKX.GLOBAL count=350
- strongest_timeframe=4h count=1357
- strongest_month=2025-05 weakest_month=2026-01
- strongest_quarter=2023Q1 weakest_quarter=2025Q3

## Legacy Coverage
- real_trade_count=78672
- entered_trend_trade_share=99.91%
- avg_captured_fraction=6.89%
- missed_major_trend_count=13619
- nontrend_loss_share=0.78%
- late_entry_share=64.51%
- early_exit_share=98.40%
- main_failure_mode=entered_middle_or_late

## Required Answers
1. Did 2023-2026 five-symbol data contain enough trend opportunities? true.
2. Which symbol had the most opportunities? DOGEUSDT_SWAP_OKX.GLOBAL.
3. Which timeframe had the most opportunities? 4h.
4. Main trend months/quarters: 2025-05 / 2023Q1.
5. More uptrends or downtrends? down.
6. Did old V3/VSVCB/CSRB capture these trends? false.
7. Main old-strategy failure mode: entered_middle_or_late.
8. Were old-strategy losses mainly in non-trend periods? false.
9. Are there observable common pre-trend features? false; strongest=none.
10. Recommended next research direction: trend_exit_convexity_research.

## Output Files
- trend_opportunity_summary.json
- trend_opportunity_report.md
- trend_segments.csv
- trend_opportunity_by_symbol.csv
- trend_opportunity_by_timeframe.csv
- trend_opportunity_by_month.csv
- trend_opportunity_by_quarter.csv
- legacy_strategy_trend_coverage.csv
- pre_trend_feature_comparison.csv
- data_quality.json
