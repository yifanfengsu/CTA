# 信号级 MFE/MAE 与突破延续性诊断

## 核心结论
- entry signal 数：1777
- 多空 MFE：long 的 120m MFE 中位数更低
- stop/take-profit：当前 take_profit_atr 在 MFE 中位数范围内，止盈距离并非主要障碍
- 止损 vs 入场：信号存在一定 MFE，但 stop-first 比例偏高，止损距离可能过近
- 突破距离：没有证据表明 breakout_distance_atr 越大越好
- 周末：周末信号没有同时呈现 MFE 更低和 MAE 更高
- 最差小时：最差小时的未来收益确实低于总体中位数

- breakout continuation hypothesis failed

## 5/15/30/60/120m 延续性
- 5m: median_future_return=-0.000158, positive_rate=0.4429, has_positive_continuation=False
- 15m: median_future_return=-0.000370, positive_rate=0.4299, has_positive_continuation=False
- 30m: median_future_return=-0.000446, positive_rate=0.4395, has_positive_continuation=False
- 60m: median_future_return=-0.000524, positive_rate=0.4401, has_positive_continuation=False
- 120m: median_future_return=-0.000647, positive_rate=0.4536, has_positive_continuation=False

## 当前 stop_atr / take_profit_atr 诊断
- inferred_stop_atr=1.399263
- inferred_take_profit_atr=2.400000
- median_mfe_atr=5.236744
- median_mae_atr=5.022251
- take_profit_first_rate=0.3551
- stop_first_rate=0.6427

## 输出文件
- signal_outcomes.csv
- outcome_summary.json
- outcome_by_side.csv
- outcome_by_hour.csv
- outcome_by_weekday.csv
- outcome_by_regime.csv
- outcome_by_breakout_distance_bucket.csv
- outcome_report.md

## Warning
- 同一根 1m bar 同时触发 stop/take-profit，保守按 stop first 处理
- breakout continuation hypothesis failed
