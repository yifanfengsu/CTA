# 策略 Ablation 实验报告

本报告用于诊断方向、周末、小时和样本切分过滤是否稳定，不等于参数优化结论。

- split: oos
- start: 2026-01-01
- end: 2026-03-31
- explicit_start_end: False

关键约束：从 full sample 归因得到的周末/小时过滤只能视为 in-sample diagnostic；必须再看 train、validation、oos 是否一致。

## Leaderboard

| rank | candidate | no-cost pnl | cost pnl | no-cost max dd% | cost max dd% | no-cost sharpe | cost sharpe | no-cost trades | cost trades | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | thursday_only | -1.17074 | -2.00088 | -0.0211728 | -0.0355719 | -3.21737 | -4.31896 | 212 | 212 | sample-mined / high overfit risk |
| 2 | short_only | -3.23901 | -7.05814 | -0.0649151 | -0.13976 | -6.38075 | -12.9332 | 998 | 998 |  |
| 3 | no_weekend_no_worst_hours | -3.35573 | -6.75208 | -0.0656959 | -0.132403 | -5.6512 | -10.1798 | 892 | 892 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 4 | weekday_no_worst_hours | -3.35573 | -6.75208 | -0.0656959 | -0.132403 | -5.6512 | -10.1798 | 892 | 892 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 5 | no_weekend | -4.02978 | -8.44707 | -0.0780773 | -0.164502 | -6.26317 | -11.5609 | 1144 | 1144 |  |
| 6 | weekdays_only | -4.02978 | -8.44707 | -0.0780773 | -0.164502 | -6.26317 | -11.5609 | 1144 | 1144 |  |
| 7 | no_worst_hours_from_current_report | -5.01999 | -9.37815 | -0.0989816 | -0.184926 | -7.80168 | -13.8351 | 1144 | 1144 | in-sample diagnostic from current full-sample attribution; not a production conclusion |
| 8 | long_only | -5.86178 | -9.25474 | -0.115247 | -0.181793 | -8.62731 | -12.5585 | 888 | 888 |  |
| 9 | baseline | -6.59134 | -12.4309 | -0.12931 | -0.244181 | -9.10162 | -16.2066 | 1512 | 1512 |  |

## 解释规则

- no-cost 仍为负：没有毛 alpha，不能进入 OKX DEMO。
- no-cost 为正但 cost 为负：成本拖累或交易频率仍不可接受。
- full 为正但 oos 为负：高度疑似过拟合。
