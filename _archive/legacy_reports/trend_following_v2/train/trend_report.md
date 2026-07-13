# Trend Following V2 研究报告

## 核心结论
- split=train, trade_count=2039, policy_runs=24
- no_cost_positive_policy_count=0
- cost_aware_positive_policy_count=0
- trend_following_v2_failed=false
- 该研究未计入 perpetual funding fee，成本后结论仍需在进入 Strategy V2 前复核 funding 敏感性。

## Policy Leaderboard
- tf_1h_donchian_55_with_risk_filters_atr4: trades=2, no_cost=-0.2949, net=-0.3161, max_dd%=0.01, top5pct_contrib=0.073
- tf_1h_donchian_55_with_risk_filters_atr5: trades=2, no_cost=-0.2949, net=-0.3161, max_dd%=0.01, top5pct_contrib=0.073
- tf_1h_donchian_55_with_risk_filters_atr3: trades=2, no_cost=-0.3345, net=-0.3557, max_dd%=0.01, top5pct_contrib=0.135
- tf_1h_vol_compression_breakout_atr5: trades=52, no_cost=-0.8869, net=-1.4195, max_dd%=0.04, top5pct_contrib=-1.081
- tf_1h_vol_compression_breakout_atr4: trades=52, no_cost=-0.8994, net=-1.4320, max_dd%=0.04, top5pct_contrib=-1.072
- tf_4h_donchian_20_with_risk_filters_atr3: trades=4, no_cost=-1.4162, net=-1.4570, max_dd%=0.03, top5pct_contrib=0.182
- tf_1h_vol_compression_breakout_atr3: trades=53, no_cost=-1.1056, net=-1.6465, max_dd%=0.04, top5pct_contrib=-0.932
- tf_4h_donchian_20_with_risk_filters_atr4: trades=4, no_cost=-1.6115, net=-1.6522, max_dd%=0.03, top5pct_contrib=0.194
- tf_4h_donchian_20_with_risk_filters_atr5: trades=4, no_cost=-1.6273, net=-1.6681, max_dd%=0.03, top5pct_contrib=0.192
- tf_4h_ema_cross_atr_trail_atr3: trades=74, no_cost=-1.6105, net=-2.3729, max_dd%=0.05, top5pct_contrib=-1.369
- tf_4h_ema_cross_atr_trail_atr4: trades=74, no_cost=-2.1944, net=-2.9571, max_dd%=0.06, top5pct_contrib=-1.042
- tf_4h_ema_cross_atr_trail_atr5: trades=74, no_cost=-2.1944, net=-2.9571, max_dd%=0.06, top5pct_contrib=-1.042

## 趋势跟踪判定
1. no-cost 是否为正：0 个 policy run 为正。
2. cost-aware 是否仍为正：0 个 policy run 为正。
3. OOS 是否为正：仅在 split=oos 或 compare 报告中做正式判断；当前 split=train。
4. 交易次数是否显著低于旧 1m 策略：当前 trade_count_range={'min': 2, 'max': 358}；旧策略基准未作为输入读取。
5. 是否靠极少数交易贡献收益：concentrated_profit_policies=['tf_1h_vol_compression_breakout_atr5', 'tf_1h_vol_compression_breakout_atr4', 'tf_1h_vol_compression_breakout_atr3', 'tf_4h_ema_cross_atr_trail_atr3', 'tf_4h_ema_cross_atr_trail_atr4', 'tf_4h_ema_cross_atr_trail_atr5', 'tf_1h_donchian_20_10_atr3', 'tf_1h_donchian_20_10_atr4', 'tf_1h_donchian_20_10_atr5', 'tf_1h_ema_cross_atr_trail_atr5', 'tf_1h_ema_cross_atr_trail_atr4', 'tf_1h_ema_cross_atr_trail_atr3']。
6. 最大回撤是否可接受：high_drawdown_policies_over_30pct=[]。
7. train/validation/oos 是否稳定：同级 split 文件已存在；建议运行 make compare-trend-v2 生成正式跨样本结论。
8. no-cost 为正但 cost 为负时，说明成本拖累仍不可接受，需要优先看 cost_drag。
9. train 正但 validation/oos 负时，按过拟合处理。
10. 若所有 OOS no-cost 和 cost-aware 都为负，trend_following_v2_failed=true；当前值=false。

## Policy 定义
- `tf_1h_donchian_20_10`: 1h close breaks previous Donchian 20; exit on previous Donchian 10 or ATR trail.
- `tf_1h_donchian_55_20`: 1h close breaks previous Donchian 55; exit on previous Donchian 20 or ATR trail.
- `tf_4h_donchian_20_10`: 4h close breaks previous Donchian 20; 15m bars approximate execution.
- `tf_1h_ema_cross_atr_trail`: 1h EMA50/EMA200 trend with close on the EMA50 side; exit on EMA50 loss or ATR trail.
- `tf_4h_ema_cross_atr_trail`: 4h EMA50/EMA200 trend with ATR trailing exit.
- `tf_1h_vol_compression_breakout`: 1h low-volatility and narrow-channel compression followed by Donchian breakout.
- `tf_1h_donchian_55_with_risk_filters`: 1h Donchian 55/20 with Signal Lab risk percentiles capped at 0.8.
- `tf_4h_donchian_20_with_risk_filters`: 4h Donchian 20/10 with latest completed 15m Signal Lab risk filters.

## 执行与成本假设
- sizing_mode=fixed, fixed_size=0.01, capital=5000.0
- contract_size=0.01；fixed_size/volume 按合约张数解释，PnL/fee/slippage 均乘以 contract_size。
- rate=0.0005, slippage_mode=ticks, absolute_slippage=0.2
- 滑点采用独立成本扣减口径：entry/exit price 保留 15m close 近似，slippage 单独记录为双边不利成交成本。
- HTF 信号在 1h/4h bar 收盘后才可见，下一根 15m bar close 作为执行价格近似。

## 输出文件
- trend_policy_summary.json
- trend_policy_leaderboard.csv
- trend_trades.csv
- trend_daily_pnl.csv
- trend_equity_curve.csv
- trend_policy_by_side.csv
- trend_policy_by_month.csv
- trend_report.md
- data_quality.json
- trend_research_audit.json

## Warning
- 未计入 perpetual funding fee
