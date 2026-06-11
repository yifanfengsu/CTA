# Trend Following V3 多品种组合级趋势跟踪研究

## 核心结论
- split_scheme=extended, split=validation_ext, symbols=['BTCUSDT_SWAP_OKX.GLOBAL', 'ETHUSDT_SWAP_OKX.GLOBAL', 'SOLUSDT_SWAP_OKX.GLOBAL', 'LINKUSDT_SWAP_OKX.GLOBAL', 'DOGEUSDT_SWAP_OKX.GLOBAL'], trade_count=1159
- start=2024-07-01T00:00:00+08:00, end=2025-06-30T23:59:00+08:00, end_exclusive=2025-07-01T00:00:00+08:00
- no_cost_positive_policy_count=3
- cost_aware_positive_policy_count=3
- trend_following_v3_failed=false
- funding_fee_warning=OKX perpetual funding fee is not included in Trend Following V3 research outputs.

## Policy Leaderboard
- v3_1d_donchian_20_10_atr4: trades=30, symbols=5, no_cost=2.6087, net=2.5158, max_dd%=0.07, largest_symbol_share=0.615, top5pct_contrib=1.582
- v3_4h_ema_50_200_atr4: trades=312, symbols=5, no_cost=2.9426, net=1.9558, max_dd%=0.08, largest_symbol_share=0.520, top5pct_contrib=5.735
- v3_1d_ema_50_200_atr5: trades=57, symbols=5, no_cost=2.0248, net=1.8245, max_dd%=0.05, largest_symbol_share=0.769, top5pct_contrib=2.708
- v3_4h_donchian_55_with_risk_filters: trades=24, symbols=4, no_cost=-0.3198, net=-0.3558, max_dd%=0.01, largest_symbol_share=0.529, top5pct_contrib=-2.000
- v3_1d_donchian_55_20_atr5: trades=16, symbols=5, no_cost=-1.2218, net=-1.2647, max_dd%=0.04, largest_symbol_share=0.739, top5pct_contrib=-0.691
- v3_ensemble_core: trades=332, symbols=5, no_cost=-0.5689, net=-1.6081, max_dd%=0.13, largest_symbol_share=0.428, top5pct_contrib=-7.119
- v3_4h_donchian_100_30_atr5: trades=57, symbols=5, no_cost=-4.4988, net=-4.6619, max_dd%=0.11, largest_symbol_share=0.979, top5pct_contrib=-0.788
- v3_4h_donchian_20_10_atr4: trades=152, symbols=5, no_cost=-5.1680, net=-5.6480, max_dd%=0.13, largest_symbol_share=1.000, top5pct_contrib=-1.340
- v3_4h_vol_compression_donchian_breakout: trades=94, symbols=5, no_cost=-5.4626, net=-5.7712, max_dd%=0.12, largest_symbol_share=0.465, top5pct_contrib=-0.410
- v3_4h_donchian_55_20_atr4: trades=85, symbols=5, no_cost=-6.0016, net=-6.2657, max_dd%=0.13, largest_symbol_share=1.000, top5pct_contrib=-0.695

## 必答问题
1. 多品种组合是否优于 BTC 单品种 Trend V2：{'btc_v2_reference': {'available': False, 'path': '/home/yiast/vnpy_projects/cta_strategy/reports/research/trend_following_v2/validation_ext/trend_policy_leaderboard.csv'}, 'best_v3_no_cost_net_pnl': 2.942640000000004, 'multi_symbol_no_cost_better_than_btc_v2': None}。
2. 组合层面 no-cost 为正的 policy：['v3_1d_donchian_20_10_atr4', 'v3_4h_ema_50_200_atr4', 'v3_1d_ema_50_200_atr5']。
3. 成本后仍为正的 policy：['v3_1d_donchian_20_10_atr4', 'v3_4h_ema_50_200_atr4', 'v3_1d_ema_50_200_atr5']。
4. 是否存在单一 symbol 贡献过度：['v3_1d_ema_50_200_atr5', 'v3_1d_donchian_55_20_atr5', 'v3_4h_donchian_100_30_atr5', 'v3_4h_donchian_20_10_atr4', 'v3_4h_donchian_55_20_atr4']。
5. top 5% trades 是否贡献过度：['v3_1d_donchian_20_10_atr4', 'v3_4h_ema_50_200_atr4', 'v3_1d_ema_50_200_atr5']。
6. OOS 是否稳定：Formal OOS stability is decided by scripts/compare_trend_following_v3.py across train/validation/oos or train_ext/validation_ext/oos_ext.。
7. 回撤是否可接受：max_ddpercent>30% policies=[]。
8. 交易次数是否足够：trade_count<10 policies=[]。
9. 是否可能进入 Strategy V3 原型开发：Only stable_candidate_exists=true in the compare summary can enter further research audit; extended research does not directly allow Strategy V3 or demo/live.。
10. 如果没有稳定候选，trend_following_v3_failed=false。

## V3 Policy 定义
| policy_name | timeframe | entry | exit | risk/filter |
|---|---|---|---|---|
| v3_4h_donchian_20_10_atr4 | 4h | close > previous Donchian high 20 / close < previous low 20 | Donchian 10 or ATR4 trail | portfolio caps |
| v3_4h_donchian_55_20_atr4 | 4h | Donchian 55 breakout | Donchian 20 or ATR4 trail | portfolio caps |
| v3_4h_donchian_100_30_atr5 | 4h | Donchian 100 breakout | Donchian 30 or ATR5 trail | portfolio caps |
| v3_1d_donchian_20_10_atr4 | 1d | Donchian 20 breakout | Donchian 10 or ATR4 trail | portfolio caps |
| v3_1d_donchian_55_20_atr5 | 1d | Donchian 55 breakout | Donchian 20 or ATR5 trail | portfolio caps |
| v3_4h_ema_50_200_atr4 | 4h | EMA50/EMA200 trend and close on EMA50 side | EMA50 loss or ATR4 trail | portfolio caps |
| v3_1d_ema_50_200_atr5 | 1d | EMA50/EMA200 trend and close on EMA50 side | EMA50 loss or ATR5 trail | portfolio caps |
| v3_4h_vol_compression_donchian_breakout | 4h | ATR and Donchian width percentile <= 0.4 then breakout | Donchian 10 or ATR4 trail | compression only, no mean reversion |
| v3_4h_donchian_55_with_risk_filters | 4h | Donchian 55 breakout | Donchian 20 or ATR4 trail | Signal Lab risk percentiles <= 0.8 |
| v3_ensemble_core | mixed | 4h Donchian 55/20 + 1d Donchian 20/10 + 4h EMA50/200 | component exits or ATR trail | same symbol/direction merged |

## 组合和成本假设
- capital=5000.0, capital_mode=portfolio_fixed, portfolio_capital=5000.0
- position_sizing=fixed_contract, fixed_size=0.01
- max_symbol_weight=0.35, max_portfolio_positions=3
- rate=0.0005, slippage_mode=ticks, slippage=2.0
- cost-aware net_pnl subtracts fee and slippage; no_cost_net_pnl equals gross price PnL.
- OKX perpetual funding fee is not included in Trend Following V3 research outputs.

## 输出文件
- trend_v3_summary.json
- trend_v3_policy_leaderboard.csv
- trend_v3_portfolio_equity_curve.csv
- trend_v3_portfolio_daily_pnl.csv
- trend_v3_trades.csv
- trend_v3_policy_by_symbol.csv
- trend_v3_policy_by_month.csv
- trend_v3_symbol_contribution.csv
- trend_v3_drawdown.csv
- trend_v3_report.md
- trend_v3_research_audit.json
- data_quality.json

## Warning
- OKX perpetual funding fee is not included in Trend Following V3 research outputs.
- v3_1d_donchian_20_10_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_ema_50_200_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_1d_ema_50_200_atr5: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_55_with_risk_filters: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_donchian_55_with_risk_filters: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_1d_donchian_55_20_atr5: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_1d_donchian_55_20_atr5: top_5pct_trade_pnl_contribution is based on a small trade sample.
- v3_1d_donchian_55_20_atr5: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_ensemble_core: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_ensemble_core: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_100_30_atr5: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_donchian_100_30_atr5: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_20_10_atr4: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_donchian_20_10_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_vol_compression_donchian_breakout: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_vol_compression_donchian_breakout: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_55_20_atr4: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_donchian_55_20_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
