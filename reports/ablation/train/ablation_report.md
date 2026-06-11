# 策略 Ablation 实验报告

本报告用于诊断方向、周末、小时和样本切分过滤是否稳定，不等于参数优化结论。

- split: train
- start: 2025-01-01
- end: 2025-09-30
- explicit_start_end: False

关键约束：从 full sample 归因得到的周末/小时过滤只能视为 in-sample diagnostic；必须再看 train、validation、oos 是否一致。

## Leaderboard

| rank | candidate | no-cost pnl | cost pnl | no-cost max dd% | cost max dd% | no-cost sharpe | cost sharpe | no-cost trades | cost trades | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | thursday_only | -0.43105 | -4.07966 | -0.02069 | -0.0815931 | -0.499428 | -3.78192 | 706 | 706 | sample-mined / high overfit risk |
| 2 | no_weekend_no_worst_hours | -1.55728 | -16.4223 | -0.0712725 | -0.32747 | -0.726263 | -7.18522 | 2886 | 2886 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 3 | weekday_no_worst_hours | -1.55728 | -16.4223 | -0.0712725 | -0.32747 | -0.726263 | -7.18522 | 2886 | 2886 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 4 | no_worst_hours_from_current_report | -5.3277 | -23.849 | -0.120742 | -0.476006 | -2.18189 | -9.59345 | 3580 | 3580 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 5 | no_weekend | -6.90985 | -25.3849 | -0.149428 | -0.506724 | -2.92046 | -9.90061 | 3594 | 3594 |  |
| 6 | weekdays_only | -6.90985 | -25.3849 | -0.149428 | -0.506724 | -2.92046 | -9.90061 | 3594 | 3594 |  |
| 7 | long_only | -7.17883 | -23.3859 | -0.143558 | -0.467159 | -3.23076 | -9.98662 | 3104 | 3104 |  |
| 8 | short_only | -9.62383 | -23.2662 | -0.204065 | -0.465318 | -4.67481 | -10.5084 | 2676 | 2676 |  |
| 9 | baseline | -14.2223 | -38.3366 | -0.290086 | -0.76576 | -5.25374 | -13.9722 | 4674 | 4674 |  |

## 解释规则

- no-cost 仍为负：没有毛 alpha，不能进入 OKX DEMO。
- no-cost 为正但 cost 为负：成本拖累或交易频率仍不可接受。
- full 为正但 oos 为负：高度疑似过拟合。
