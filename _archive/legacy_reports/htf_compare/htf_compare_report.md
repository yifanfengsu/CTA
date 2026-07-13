# HTF Signal Research 跨样本比较

## 核心结论
- no_stable_htf_policy=true
- overfit_risk=false
- 没有稳定候选进入 Strategy V2

## 哪些 policy 在三段方向一致？
- htf_1h_15m_structure_5m_pullback_reclaim: return_direction=negative, train_120m=-0.000297, validation_120m=-0.000604, oos_120m=-0.000682, positive_splits=
- htf_1h_15m_structure_5m_pullback_reclaim_strict: return_direction=negative, train_120m=-0.000180, validation_120m=-0.000273, oos_120m=-0.000227, positive_splits=
- htf_1h_15m_structure_5m_pullback_reclaim_vol_cap: return_direction=negative, train_120m=-0.000264, validation_120m=-0.000410, oos_120m=-0.000481, positive_splits=
- htf_1h_15m_structure_no_overextension: return_direction=negative, train_120m=-0.000257, validation_120m=-0.000613, oos_120m=-0.000492, positive_splits=
- htf_1h_15m_structure_strict_vol_cap: return_direction=negative, train_120m=-0.000364, validation_120m=-0.000496, oos_120m=-0.000604, positive_splits=
- htf_1h_15m_structure_with_vol_cap: return_direction=negative, train_120m=-0.000286, validation_120m=-0.000488, oos_120m=-0.000604, positive_splits=
- htf_1h_ema_15m_donchian_structure: return_direction=negative, train_120m=-0.000591, validation_120m=-0.000799, oos_120m=-0.000905, positive_splits=
- htf_1h_ema_15m_ema_structure: return_direction=negative, train_120m=-0.000451, validation_120m=-0.000752, oos_120m=-0.000764, positive_splits=
- htf_1h_ema_15m_vwap_structure: return_direction=negative, train_120m=-0.000390, validation_120m=-0.000704, oos_120m=-0.000671, positive_splits=
- htf_1h_ema_regime_only: return_direction=negative, train_120m=-0.000316, validation_120m=-0.000448, oos_120m=-0.000361, positive_splits=

## 哪些 policy 只在某一段有效？
- 无

## 是否有稳定候选进入 Strategy V2？
- 无

## 输出文件
- htf_compare_summary.json
- htf_compare_leaderboard.csv
- htf_compare_report.md
