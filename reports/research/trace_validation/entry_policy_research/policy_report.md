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
- report_dir: `/home/yiast/vnpy_projects/cta_strategy/reports/research/trace_validation`
- signal_trace: `/home/yiast/vnpy_projects/cta_strategy/reports/research/trace_validation/signal_trace.csv`
- entry signal 数: 1807
- horizons: [15, 30, 60, 120]
- stop_atr_grid: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
- tp_atr_grid: [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
- 当前样本正 expectancy policies: ['avoid_stop_first_profile', 'skip_large_breakout_gt_1atr']
- 跨样本正 expectancy policies: []

## Leaderboard Top 10
| policy | entries | exp_r | median_r | win_rate | horizon | stop | tp | stop_first | horizon_exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avoid_stop_first_profile | 764 | 0.0505 | -1.0000 | 0.2461 | 30 | 1.00 | 5.00 | 0.7369 | 0.1505 |
| skip_large_breakout_gt_1atr | 1484 | 0.0018 | -0.0810 | 0.4596 | 15 | 2.50 | 5.00 | 0.2372 | 0.6840 |
| small_to_mid_breakout_0_25_to_1atr | 581 | -0.0045 | -1.0000 | 0.3150 | 15 | 1.00 | 4.00 | 0.6472 | 0.2565 |
| skip_large_breakout_gt_2atr | 1625 | -0.0138 | -0.0395 | 0.4640 | 15 | 4.00 | 5.00 | 0.1095 | 0.8006 |
| pullback_to_breakout_level_5bar | 1421 | -0.0475 | -0.0965 | 0.4419 | 30 | 4.00 | 5.00 | 0.1999 | 0.6784 |
| pullback_to_breakout_level_10bar | 1516 | -0.0487 | -0.0980 | 0.4413 | 30 | 4.00 | 5.00 | 0.2012 | 0.6768 |
| immediate_baseline | 1807 | -0.0545 | -0.0854 | 0.4388 | 15 | 4.00 | 5.00 | 0.1339 | 0.7742 |
| delayed_confirm_3bar | 814 | -0.0698 | -1.0000 | 0.2740 | 15 | 1.00 | 5.00 | 0.6769 | 0.2531 |
| delayed_confirm_1bar | 821 | -0.0769 | -0.1681 | 0.4141 | 15 | 2.50 | 5.00 | 0.3021 | 0.6188 |
| momentum_followthrough_3bar | 651 | -0.0819 | -0.0914 | 0.4086 | 15 | 4.00 | 5.00 | 0.1567 | 0.7435 |

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
