# HTF Signal Research 报告

## 核心结论
- split=oos, signal_count=11404
- htf_signal_hypothesis_failed=true
- overfit_risk=false

## Policy Leaderboard
- htf_1h_15m_structure_5m_pullback_reclaim_strict: signals=145, median_120m=-0.000227, best_expectancy_r=-0.001399, notes=no_positive_edge_flag
- htf_1h_15m_structure_5m_pullback_reclaim_vol_cap: signals=304, median_120m=-0.000481, best_expectancy_r=-0.007898, notes=no_positive_edge_flag
- htf_1h_ema_regime_only: signals=2847, median_120m=-0.000361, best_expectancy_r=-0.008738, notes=no_positive_edge_flag
- htf_1h_15m_structure_5m_pullback_reclaim: signals=456, median_120m=-0.000682, best_expectancy_r=-0.009617, notes=no_positive_edge_flag
- htf_1h_15m_structure_no_overextension: signals=999, median_120m=-0.000492, best_expectancy_r=-0.012607, notes=no_positive_edge_flag
- htf_1h_15m_structure_with_vol_cap: signals=1049, median_120m=-0.000604, best_expectancy_r=-0.017876, notes=no_positive_edge_flag
- htf_1h_ema_15m_vwap_structure: signals=1691, median_120m=-0.000671, best_expectancy_r=-0.023836, notes=no_positive_edge_flag
- htf_1h_ema_15m_ema_structure: signals=1597, median_120m=-0.000764, best_expectancy_r=-0.027283, notes=no_positive_edge_flag
- htf_1h_15m_structure_strict_vol_cap: signals=683, median_120m=-0.000604, best_expectancy_r=-0.033392, notes=no_positive_edge_flag
- htf_1h_ema_15m_donchian_structure: signals=1633, median_120m=-0.000905, best_expectancy_r=-0.054644, notes=no_positive_edge_flag

## 必答问题
1. 1h regime only 是否比原 1m breakout 更好？本脚本不读取原 1m breakout 基准报告，不能单独证明比原策略更好；当前 1h regime only 的 median_future_return_120m=-0.0003606733672817941，best_expectancy_r=-0.0087375796481309。
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
- horizon 60m 超过可用 1m bar 数据范围的 HTF signal 数: 9
- horizon 120m 超过可用 1m bar 数据范围的 HTF signal 数: 12
- horizon 240m 超过可用 1m bar 数据范围的 HTF signal 数: 24
- horizon 480m 超过可用 1m bar 数据范围的 HTF signal 数: 55
