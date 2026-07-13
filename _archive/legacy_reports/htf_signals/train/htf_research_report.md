# HTF Signal Research 报告

## 核心结论
- split=train, signal_count=32499
- htf_signal_hypothesis_failed=true
- overfit_risk=false

## Policy Leaderboard
- htf_1h_15m_structure_5m_pullback_reclaim_strict: signals=412, median_120m=-0.000180, best_expectancy_r=0.150717, notes=positive_bracket_expectancy
- htf_1h_15m_structure_5m_pullback_reclaim_vol_cap: signals=836, median_120m=-0.000264, best_expectancy_r=0.125168, notes=positive_bracket_expectancy
- htf_1h_15m_structure_with_vol_cap: signals=2938, median_120m=-0.000286, best_expectancy_r=0.088392, notes=positive_bracket_expectancy
- htf_1h_15m_structure_strict_vol_cap: signals=1863, median_120m=-0.000364, best_expectancy_r=0.076172, notes=positive_bracket_expectancy
- htf_1h_15m_structure_5m_pullback_reclaim: signals=1352, median_120m=-0.000297, best_expectancy_r=0.069063, notes=positive_bracket_expectancy
- htf_1h_15m_structure_no_overextension: signals=2881, median_120m=-0.000257, best_expectancy_r=0.061544, notes=positive_bracket_expectancy
- htf_1h_ema_15m_ema_structure: signals=4726, median_120m=-0.000451, best_expectancy_r=0.053780, notes=positive_bracket_expectancy
- htf_1h_ema_15m_vwap_structure: signals=5076, median_120m=-0.000390, best_expectancy_r=0.044511, notes=positive_bracket_expectancy
- htf_1h_ema_regime_only: signals=8261, median_120m=-0.000316, best_expectancy_r=0.028345, notes=positive_bracket_expectancy
- htf_1h_ema_15m_donchian_structure: signals=4154, median_120m=-0.000591, best_expectancy_r=0.024831, notes=positive_bracket_expectancy

## 必答问题
1. 1h regime only 是否比原 1m breakout 更好？本脚本不读取原 1m breakout 基准报告，不能单独证明比原策略更好；当前 1h regime only 的 median_future_return_120m=-0.0003156708004509179，best_expectancy_r=0.0283451366545181。
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
