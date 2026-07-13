# Trend Following V3 多品种组合级趋势跟踪研究

## 核心结论
- split_scheme=extended, split=full_ext, symbols=['BTCUSDT_SWAP_OKX.GLOBAL', 'ETHUSDT_SWAP_OKX.GLOBAL', 'SOLUSDT_SWAP_OKX.GLOBAL', 'LINKUSDT_SWAP_OKX.GLOBAL', 'DOGEUSDT_SWAP_OKX.GLOBAL'], trade_count=3942
- start=2023-01-01T00:00:00+08:00, end=2026-03-31T23:59:00+08:00, end_exclusive=2026-04-01T00:00:00+08:00
- no_cost_positive_policy_count=3
- cost_aware_positive_policy_count=3
- trend_following_v3_failed=false
- funding_fee_warning=OKX perpetual funding fee is not included in Trend Following V3 research outputs.

## Policy Leaderboard
- v3_1d_ema_50_200_atr5: trades=158, symbols=5, no_cost=6.5608, net=6.0724, max_dd%=0.09, largest_symbol_share=0.782, top5pct_contrib=1.950
- v3_4h_donchian_55_with_risk_filters: trades=73, symbols=5, no_cost=4.1439, net=3.9948, max_dd%=0.02, largest_symbol_share=0.747, top5pct_contrib=1.232
- v3_1d_donchian_20_10_atr4: trades=86, symbols=5, no_cost=2.4991, net=2.2723, max_dd%=0.08, largest_symbol_share=0.476, top5pct_contrib=3.619
- v3_1d_donchian_55_20_atr5: trades=43, symbols=5, no_cost=-3.5699, net=-3.6820, max_dd%=0.11, largest_symbol_share=0.941, top5pct_contrib=-0.825
- v3_4h_ema_50_200_atr4: trades=1115, symbols=5, no_cost=-0.5893, net=-3.8404, max_dd%=0.15, largest_symbol_share=1.000, top5pct_contrib=-7.784
- v3_ensemble_core: trades=1153, symbols=5, no_cost=-4.5556, net=-7.8518, max_dd%=0.23, largest_symbol_share=1.000, top5pct_contrib=-3.940
- v3_4h_donchian_100_30_atr5: trades=170, symbols=5, no_cost=-9.4852, net=-9.9408, max_dd%=0.24, largest_symbol_share=1.000, top5pct_contrib=-1.072
- v3_4h_vol_compression_donchian_breakout: trades=350, symbols=5, no_cost=-10.0485, net=-10.9478, max_dd%=0.24, largest_symbol_share=1.000, top5pct_contrib=-0.770
- v3_4h_donchian_55_20_atr4: trades=278, symbols=5, no_cost=-12.1536, net=-12.9060, max_dd%=0.34, largest_symbol_share=0.368, top5pct_contrib=-1.015
- v3_4h_donchian_20_10_atr4: trades=516, symbols=5, no_cost=-13.3497, net=-14.6577, max_dd%=0.34, largest_symbol_share=0.448, top5pct_contrib=-1.186

## 必答问题
1. 多品种组合是否优于 BTC 单品种 Trend V2：{'btc_v2_reference': {'available': False, 'path': '/home/yiast/vnpy_projects/cta_strategy/reports/research/trend_following_v2/full_ext/trend_policy_leaderboard.csv'}, 'best_v3_no_cost_net_pnl': 6.560810000000002, 'multi_symbol_no_cost_better_than_btc_v2': None}。
2. 组合层面 no-cost 为正的 policy：['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters', 'v3_1d_donchian_20_10_atr4']。
3. 成本后仍为正的 policy：['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters', 'v3_1d_donchian_20_10_atr4']。
4. 是否存在单一 symbol 贡献过度：['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters', 'v3_1d_donchian_55_20_atr5', 'v3_4h_ema_50_200_atr4', 'v3_ensemble_core', 'v3_4h_donchian_100_30_atr5', 'v3_4h_vol_compression_donchian_breakout']。
5. top 5% trades 是否贡献过度：['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters', 'v3_1d_donchian_20_10_atr4']。
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
- v3_1d_ema_50_200_atr5: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_55_with_risk_filters: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_1d_donchian_20_10_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_1d_donchian_55_20_atr5: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_1d_donchian_55_20_atr5: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_ema_50_200_atr4: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_ema_50_200_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_ensemble_core: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_ensemble_core: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_100_30_atr5: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_donchian_100_30_atr5: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_vol_compression_donchian_breakout: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_vol_compression_donchian_breakout: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_55_20_atr4: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_donchian_55_20_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
- v3_4h_donchian_20_10_atr4: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.
- v3_4h_donchian_20_10_atr4: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.
