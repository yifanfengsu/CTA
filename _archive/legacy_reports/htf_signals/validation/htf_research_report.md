# HTF Signal Research 报告

## 核心结论
- split=validation, signal_count=11481
- htf_signal_hypothesis_failed=true
- overfit_risk=false

## Policy Leaderboard
- htf_1h_15m_structure_5m_pullback_reclaim_vol_cap: signals=316, median_120m=-0.000410, best_expectancy_r=0.108392, notes=positive_bracket_expectancy
- htf_1h_15m_structure_5m_pullback_reclaim: signals=483, median_120m=-0.000604, best_expectancy_r=0.028413, notes=positive_bracket_expectancy
- htf_1h_ema_regime_only: signals=2919, median_120m=-0.000448, best_expectancy_r=0.022732, notes=positive_bracket_expectancy
- htf_1h_15m_structure_5m_pullback_reclaim_strict: signals=159, median_120m=-0.000273, best_expectancy_r=0.015532, notes=positive_bracket_expectancy
- htf_1h_15m_structure_with_vol_cap: signals=1023, median_120m=-0.000488, best_expectancy_r=0.002409, notes=positive_bracket_expectancy
- htf_1h_15m_structure_strict_vol_cap: signals=661, median_120m=-0.000496, best_expectancy_r=-0.004238, notes=no_positive_edge_flag
- htf_1h_15m_structure_no_overextension: signals=1012, median_120m=-0.000613, best_expectancy_r=-0.016811, notes=no_positive_edge_flag
- htf_1h_ema_15m_vwap_structure: signals=1720, median_120m=-0.000704, best_expectancy_r=-0.017873, notes=no_positive_edge_flag
- htf_1h_ema_15m_ema_structure: signals=1635, median_120m=-0.000752, best_expectancy_r=-0.018478, notes=no_positive_edge_flag
- htf_1h_ema_15m_donchian_structure: signals=1553, median_120m=-0.000799, best_expectancy_r=-0.036234, notes=no_positive_edge_flag

## 必答问题
1. 1h regime only 是否比原 1m breakout 更好？本脚本不读取原 1m breakout 基准报告，不能单独证明比原策略更好；当前 1h regime only 的 median_future_return_120m=-0.00044762757385852225，best_expectancy_r=0.022731789694238162。
2. 15m EMA structure 是否改善？15m EMA structure 是否改善: 否，htf_1h_ema_15m_ema_structure 的 median_future_return_120m 未高于 htf_1h_ema_regime_only
3. 15m VWAP structure 是否改善？15m VWAP structure 是否改善: 否，htf_1h_ema_15m_vwap_structure 的 median_future_return_120m 未高于 htf_1h_ema_regime_only
4. 15m Donchian structure 是否改善？15m Donchian structure 是否改善: 否，htf_1h_ema_15m_donchian_structure 的 median_future_return_120m 未高于 htf_1h_ema_regime_only
5. vol cap 是否改善？vol cap 是否改善: 是，htf_1h_15m_structure_with_vol_cap 的 median_future_return_120m 高于 htf_1h_ema_15m_ema_structure；strict vol cap 是否改善: 是，htf_1h_15m_structure_strict_vol_cap 的 median_future_return_120m 高于 htf_1h_ema_15m_ema_structure
6. no overextension 是否改善？no overextension 是否改善: 是，htf_1h_15m_structure_no_overextension 的 median_future_return_120m 高于 htf_1h_ema_15m_ema_structure
7. 5m pullback reclaim 是否改善？5m pullback reclaim 是否改善: 是，htf_1h_15m_structure_5m_pullback_reclaim 的 median_future_return_120m 高于 htf_1h_ema_15m_ema_structure
8. 是否存在 train / validation / oos 都稳定为正的 policy？没有发现 train/validation/oos 都稳定为正的 policy
9. htf_signal_hypothesis_failed=true
10. overfit_risk=false；single_split_only=[]

## 输出文件
- htf_signal_dataset.csv
- htf_policy_summary.json
- htf_research_audit.json
- htf_policy_leaderboard.csv
- htf_bracket_grid.csv
- htf_policy_by_side.csv
- htf_policy_by_hour.csv
- htf_policy_by_weekday.csv
- htf_research_report.md
- data_quality.json

## Warning
- 无
