# 入场政策离线研究报告

## 结论问题
- 跳过 `breakout_distance_atr > 1` 是否改善：是
- 跳过 `breakout_distance_atr > 2` 是否改善：是
- delayed confirm 是否优于 immediate entry：是
- pullback entry 是否优于 immediate entry：否
- momentum followthrough 是否优于 immediate entry：是
- 是否有 policy 在 train / validation / oos 都可能为正 expectancy：否
- entry_policy_hypothesis_failed=True

## 当前样本
- report_dir: `/home/yiast/vnpy_projects/cta_strategy/reports/research/trace_oos`
- signal_trace: `/home/yiast/vnpy_projects/cta_strategy/reports/research/trace_oos/signal_trace.csv`
- entry signal 数: 1716
- horizons: [15, 30, 60, 120]
- stop_atr_grid: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
- tp_atr_grid: [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
- 当前样本正 expectancy policies: ['momentum_followthrough_3bar', 'avoid_stop_first_profile', 'delayed_confirm_3bar', 'delayed_confirm_1bar']
- 跨样本正 expectancy policies: []

## Leaderboard Top 10
| policy | entries | exp_r | median_r | win_rate | horizon | stop | tp | stop_first | horizon_exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| momentum_followthrough_3bar | 548 | 0.1315 | -1.0000 | 0.1953 | 120 | 1.00 | 5.00 | 0.8029 | 0.0128 |
| avoid_stop_first_profile | 675 | 0.0717 | -1.0000 | 0.2978 | 15 | 1.00 | 5.00 | 0.6459 | 0.2696 |
| delayed_confirm_3bar | 702 | 0.0631 | -1.0000 | 0.2707 | 15 | 1.00 | 5.00 | 0.6738 | 0.2279 |
| delayed_confirm_1bar | 736 | 0.0241 | -1.0000 | 0.2772 | 15 | 1.00 | 5.00 | 0.6766 | 0.2351 |
| skip_large_breakout_gt_1atr | 1412 | -0.0333 | -0.0844 | 0.4348 | 15 | 4.00 | 5.00 | 0.1190 | 0.7890 |
| skip_large_breakout_gt_2atr | 1577 | -0.0386 | -0.0914 | 0.4318 | 15 | 4.00 | 5.00 | 0.1300 | 0.7692 |
| small_to_mid_breakout_0_25_to_1atr | 598 | -0.0437 | -0.1079 | 0.4314 | 15 | 4.00 | 5.00 | 0.1371 | 0.7575 |
| immediate_baseline | 1716 | -0.0577 | -0.1136 | 0.4202 | 15 | 4.00 | 5.00 | 0.1503 | 0.7430 |
| pullback_to_breakout_level_10bar | 1405 | -0.0896 | -0.1117 | 0.3858 | 15 | 4.00 | 5.00 | 0.1381 | 0.7936 |
| pullback_to_breakout_level_5bar | 1321 | -0.0947 | -0.1185 | 0.3785 | 15 | 4.00 | 5.00 | 0.1370 | 0.7949 |

## Policy 定义
- `immediate_baseline`: Signal time immediate entry at trace price.
- `skip_large_breakout_gt_1atr`: Immediate entry, but skip breakout_distance_atr > 1.
- `skip_large_breakout_gt_2atr`: Immediate entry, but skip breakout_distance_atr > 2.
- `small_to_mid_breakout_0_25_to_1atr`: Immediate entry only when 0.25 <= breakout_distance_atr <= 1.
- `delayed_confirm_1bar`: Wait one 1m bar; enter at that bar close only if close still confirms the breakout side.
- `delayed_confirm_3bar`: Wait three 1m bars; enter at the third bar close only if close still confirms the breakout side.
- `pullback_to_breakout_level_5bar`: Wait up to five 1m bars for a pullback touch of the original breakout level.
- `pullback_to_breakout_level_10bar`: Wait up to ten 1m bars for a pullback touch of the original breakout level.
- `momentum_followthrough_3bar`: Wait three 1m bars; enter only after at least 0.25 ATR favorable follow-through.
- `avoid_stop_first_profile`: Use the first three 1m bars as a gate; skip early 1 ATR adverse-first profiles.

## 输出文件
- entry_policy_summary.json
- entry_policy_leaderboard.csv
- bracket_grid.csv
- policy_by_side.csv
- policy_by_hour.csv
- policy_report.md

## Warning
- 同一根 1m bar 同时触发 stop/take-profit，保守按 stop first 处理
- entry policy hypothesis failed
