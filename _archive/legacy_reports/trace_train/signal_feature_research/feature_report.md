# Signal Lab 特征研究报告

## 核心结论
- entry signal 数：5295
- outcome 来源：existing_signal_outcomes
- signal_feature_hypothesis_failed=false

## 哪些特征和 future_return_60m 关系最强？
- recent_volatility_30m: future_return_60m IC=-0.2181, count=5295
- atr_pct: future_return_60m IC=-0.1863, count=5295
- breakout_distance_atr: future_return_60m IC=-0.1464, count=5295
- recent_return_30m: future_return_60m IC=-0.1158, count=5295
- range_atr: future_return_60m IC=-0.1060, count=5295

## 哪些特征和 stop_first 最相关？
- atr_pct: stop_first IC=0.2303, count=5295
- recent_volatility_30m: stop_first IC=0.1776, count=5295
- donchian_width_atr: stop_first IC=-0.0846, count=5295
- recent_return_5m: stop_first IC=-0.0777, count=5295
- lower_wick_ratio: stop_first IC=0.0595, count=5286

## 必答诊断
- breakout_distance_atr 是否越大越差：breakout_distance_atr 越大，future_return_60m 越差的证据成立
- rsi 是否有筛选价值：rsi 没有达到稳定筛选阈值
- ema_spread_pct 是否有筛选价值：ema_spread_pct 没有达到稳定筛选阈值
- atr_pct 是否存在过低/过高风险区间：atr_pct 低分位和高分位都弱于中间区间，存在两端风险
- volume_zscore 是否能改善信号：volume_zscore_30m 没有达到稳定筛选阈值
- wick/body 结构是否能识别假突破：wick/body 结构未达到假突破识别阈值

## 输出文件
- feature_dataset.csv
- feature_summary.json
- feature_ic.csv
- feature_bins.csv
- categorical_feature_bins.csv
- feature_report.md

## Warning
- feature close_location_in_donchian 缺失值数量: 6/5295
- feature upper_wick_ratio 缺失值数量: 9/5295
- feature lower_wick_ratio 缺失值数量: 9/5295
- feature body_ratio 缺失值数量: 9/5295
- 已优先使用已有 outcome 文件: /home/yiast/vnpy_projects/cta_strategy/reports/research/trace_train/signal_outcomes/signal_outcomes.csv
