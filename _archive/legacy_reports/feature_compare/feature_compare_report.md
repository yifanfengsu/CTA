# Signal Lab 跨样本稳定性比较

## 核心结论
- no_stable_feature_edge=false
- 存在跨 train/validation/oos 方向一致且达到 IC 阈值的候选特征

## 哪些特征在 train / validation / oos 方向一致？
- recent_volatility_30m: direction=negative, train=-0.2181, validation=-0.2252, oos=-0.2934, min_abs_ic=0.2181
- atr_pct: direction=negative, train=-0.1863, validation=-0.2400, oos=-0.2748, min_abs_ic=0.1863
- breakout_distance_atr: direction=negative, train=-0.1464, validation=-0.2390, oos=-0.1193, min_abs_ic=0.1193
- recent_return_30m: direction=negative, train=-0.1158, validation=-0.0881, oos=-0.1002, min_abs_ic=0.0881
- recent_return_15m: direction=negative, train=-0.0723, validation=-0.1347, oos=-0.1069, min_abs_ic=0.0723
- recent_return_5m: direction=negative, train=-0.0598, validation=-0.1589, oos=-0.1444, min_abs_ic=0.0598
- volume_zscore_30m: direction=negative, train=-0.0892, validation=-0.0737, oos=-0.0556, min_abs_ic=0.0556
- body_ratio: direction=negative, train=-0.0783, validation=-0.1001, oos=-0.0536, min_abs_ic=0.0536

## 哪些特征只在某一段有效，疑似过拟合？
- range_atr: train=-0.1060, validation=-0.0428, oos=-0.0115, max_abs_ic=0.1060

## 是否存在可进入策略候选的稳定特征？
- recent_volatility_30m: direction=negative, train=-0.2181, validation=-0.2252, oos=-0.2934, min_abs_ic=0.2181
- atr_pct: direction=negative, train=-0.1863, validation=-0.2400, oos=-0.2748, min_abs_ic=0.1863
- breakout_distance_atr: direction=negative, train=-0.1464, validation=-0.2390, oos=-0.1193, min_abs_ic=0.1193
- recent_return_30m: direction=negative, train=-0.1158, validation=-0.0881, oos=-0.1002, min_abs_ic=0.0881
- recent_return_15m: direction=negative, train=-0.0723, validation=-0.1347, oos=-0.1069, min_abs_ic=0.0723
- recent_return_5m: direction=negative, train=-0.0598, validation=-0.1589, oos=-0.1444, min_abs_ic=0.0598
- volume_zscore_30m: direction=negative, train=-0.0892, validation=-0.0737, oos=-0.0556, min_abs_ic=0.0556
- body_ratio: direction=negative, train=-0.0783, validation=-0.1001, oos=-0.0536, min_abs_ic=0.0536

## 输出文件
- feature_compare_summary.json
- feature_compare_ic.csv
- feature_compare_report.md
