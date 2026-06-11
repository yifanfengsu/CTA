# 入场政策离线研究报告

## 结论问题
- 跳过 `breakout_distance_atr > 1` 是否改善：是
- 跳过 `breakout_distance_atr > 2` 是否改善：是
- delayed confirm 是否优于 immediate entry：否
- pullback entry 是否优于 immediate entry：是
- momentum followthrough 是否优于 immediate entry：否
- 是否有 policy 在 train / validation / oos 都可能为正 expectancy：否
- entry_policy_hypothesis_failed=True

## 当前样本
- report_dir: `/home/yiast/vnpy_projects/cta_strategy/reports/research/trace_train`
- signal_trace: `/home/yiast/vnpy_projects/cta_strategy/reports/research/trace_train/signal_trace.csv`
- entry signal 数: 5295
- horizons: [15, 30, 60, 120]
- stop_atr_grid: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
- tp_atr_grid: [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
- 当前样本正 expectancy policies: ['skip_large_breakout_gt_1atr', 'small_to_mid_breakout_0_25_to_1atr']
- 跨样本正 expectancy policies: []

## Leaderboard Top 10
| policy | entries | exp_r | median_r | win_rate | horizon | stop | tp | stop_first | horizon_exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skip_large_breakout_gt_1atr | 4658 | 0.0130 | -0.1878 | 0.4584 | 120 | 4.00 | 5.00 | 0.4079 | 0.2535 |
| small_to_mid_breakout_0_25_to_1atr | 1973 | 0.0094 | -0.1216 | 0.4607 | 60 | 4.00 | 5.00 | 0.3492 | 0.3639 |
| skip_large_breakout_gt_2atr | 4965 | -0.0022 | -0.2273 | 0.4504 | 120 | 4.00 | 5.00 | 0.4173 | 0.2473 |
| pullback_to_breakout_level_10bar | 4762 | -0.0124 | -0.2169 | 0.4492 | 120 | 4.00 | 5.00 | 0.4105 | 0.2701 |
| pullback_to_breakout_level_5bar | 4510 | -0.0147 | -0.2265 | 0.4481 | 120 | 4.00 | 5.00 | 0.4093 | 0.2736 |
| immediate_baseline | 5295 | -0.0347 | -0.3227 | 0.4357 | 120 | 4.00 | 5.00 | 0.4389 | 0.2347 |
| delayed_confirm_3bar | 2264 | -0.0377 | -0.1476 | 0.4390 | 60 | 4.00 | 5.00 | 0.3573 | 0.3887 |
| delayed_confirm_1bar | 2379 | -0.0503 | -0.3963 | 0.4266 | 120 | 4.00 | 5.00 | 0.4540 | 0.2190 |
| momentum_followthrough_3bar | 1750 | -0.0588 | 0.0841 | 0.5429 | 15 | 4.00 | 1.50 | 0.1240 | 0.4543 |
| avoid_stop_first_profile | 2113 | -0.0616 | 0.1219 | 0.5177 | 120 | 4.00 | 3.00 | 0.3914 | 0.1429 |

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
