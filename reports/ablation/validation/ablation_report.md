# 策略 Ablation 实验报告

本报告用于诊断方向、周末、小时和样本切分过滤是否稳定，不等于参数优化结论。

- split: validation
- start: 2025-10-01
- end: 2025-12-31
- explicit_start_end: False

关键约束：从 full sample 归因得到的周末/小时过滤只能视为 in-sample diagnostic；必须再看 train、validation、oos 是否一致。

## Leaderboard

| rank | candidate | no-cost pnl | cost pnl | no-cost max dd% | cost max dd% | no-cost sharpe | cost sharpe | no-cost trades | cost trades | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | no_weekend_no_worst_hours | 1.69926 | -3.7798 | -0.0188823 | -0.0799648 | 2.697 | -5.6146 | 1078 | 1078 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 2 | weekday_no_worst_hours | 1.69926 | -3.7798 | -0.0188823 | -0.0799648 | 2.697 | -5.6146 | 1078 | 1078 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 3 | no_worst_hours_from_current_report | 0.79268 | -6.3001 | -0.02023 | -0.130371 | 1.10052 | -8.50137 | 1398 | 1398 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 4 | no_weekend | 0.33153 | -6.05252 | -0.0301372 | -0.122012 | 0.547221 | -8.97316 | 1260 | 1260 |  |
| 5 | weekdays_only | 0.33153 | -6.05252 | -0.0301372 | -0.122012 | 0.547221 | -8.97316 | 1260 | 1260 |  |
| 6 | thursday_only | 0.01156 | -1.24535 | -0.00771251 | -0.0250046 | 0.0362413 | -3.26383 | 246 | 246 | sample-mined / high overfit risk |
| 7 | short_only | -0.3124 | -5.92313 | -0.0326219 | -0.117951 | -0.516499 | -9.75694 | 1118 | 1118 |  |
| 8 | baseline | -0.39081 | -8.9826 | -0.0449672 | -0.182004 | -0.510345 | -11.4802 | 1702 | 1702 |  |
| 9 | long_only | -1.66039 | -7.51222 | -0.0495682 | -0.152582 | -2.78384 | -11.6309 | 1146 | 1146 |  |

## 解释规则

- no-cost 仍为负：没有毛 alpha，不能进入 OKX DEMO。
- no-cost 为正但 cost 为负：成本拖累或交易频率仍不可接受。
- full 为正但 oos 为负：高度疑似过拟合。
