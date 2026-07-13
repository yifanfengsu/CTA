# Trend Regime Diagnostics

## 诊断边界
- 本报告是趋势环境诊断，不是策略开发、参数搜索、demo runner 或 live runner。
- Regime 只使用已完成的 4h/1d bar；V3 trade 归因用 entry_time 向后对齐到最近一个已完成 regime bar。
- Funding fee 未进入 V3 trade PnL；下方只引用已有 synthetic funding stress 作为风险约束。

## Regime 分类规则
- strong_uptrend/strong_downtrend：EMA50/EMA200 spread 达到对应 timeframe 阈值，EMA50 与 EMA200 slope 同向，trend_efficiency >= 0.35，且 ADX14 或 proxy >= 20。
- weak_uptrend/weak_downtrend：EMA 结构同向，但 trend_efficiency 或 EMA spread 未达到 strong 条件。
- compression：ATR percentile、realized volatility percentile、Donchian width percentile 均处低位，且 trend_efficiency < 0.25。
- high_vol_choppy：未形成趋势结构，但 ATR 或 realized volatility percentile >= 0.80。
- choppy：不满足趋势、压缩或高波动震荡条件的剩余样本。

## Regime 分布
- strong trend 占比：4.79%
- choppy/high_vol_choppy 占比：38.70%
- strong_uptrend：3.67%
- strong_downtrend：1.12%
- compression：1.67%

## 核心排序
- 趋势性最好 symbol：SOLUSDT_SWAP_OKX.GLOBAL，avg_trend_score=0.4413
- 最震荡 symbol：BTCUSDT_SWAP_OKX.GLOBAL，avg_trend_score=0.4133
- 趋势性最强月份：2026-02，avg_trend_score=0.5100
- 最不适合趋势跟踪月份：2023-07，avg_trend_score=0.3767
- 趋势性最强季度：2023Q4，avg_trend_score=0.4713
- 最不适合趋势跟踪季度：2023Q2，avg_trend_score=0.3983
- 1d 是否优于 4h：true

## V3.0 Trade Regime 归因
- 盈利交易主要 regime：weak_uptrend，share=33.45%
- 亏损交易主要 regime：weak_uptrend，share=30.45%
- 盈利 PnL 来自 strong trend 的占比：2.78%
- 亏损来自 choppy/high_vol_choppy 的占比：41.24%
- V3.0 是否在 choppy/high_vol_choppy 中交易过多：false
- 1d EMA strong_no_cost_pnl=-0.4382，nonstrong_no_cost_pnl=7.1852
- 4h Donchian choppy/high_vol loss share=50.87%
- 1d Donchian choppy/high_vol loss share=57.31%

## 必答问题
1. 2023-2026 是否存在足够趋势 regime？false。
2. 哪些 symbol 趋势性最好？SOLUSDT_SWAP_OKX.GLOBAL。
3. 哪些 symbol 最震荡？BTCUSDT_SWAP_OKX.GLOBAL。
4. 哪些月份/季度趋势性最强？2026-02 / 2023Q4。
5. 哪些月份/季度最不适合趋势跟踪？2023-07 / 2023Q2。
6. 1d timeframe 是否优于 4h timeframe？true。
7. EMA regime 是否比 Donchian breakout 更符合趋势结构？false。
8. V3.0 亏损是否主要来自 choppy/high_vol_choppy？false。
9. OOS best policy 是否只是在某些 strong trend regime 中有效？false。
10. 是否建议进入 V3.1 research？false。
11. 如果建议 V3.1，应保留哪些方向？none；filters={'min_trend_efficiency': 0.35, 'min_adx_or_proxy': 20.0, 'allowed_regimes': ['strong_uptrend', 'strong_downtrend'], 'blocked_regimes': ['choppy', 'high_vol_choppy', 'compression'], 'max_atr_percentile': 0.8, 'min_ema_spread_pct': {'4h': 0.004, '1d': 0.01}}
12. 如果不建议 V3.1，应停止哪些方向？1d_ema, 1d_donchian, 4h_ema, 4h_donchian, ensemble。

## Funding 与集中度约束
- stable_candidate_exists=false
- high_top_5pct_trade_pnl_contribution_policies=['v3_1d_ema_50_200_atr5', 'v3_4h_donchian_55_with_risk_filters']
- high_largest_symbol_pnl_share_policies=['v3_4h_donchian_55_with_risk_filters', 'v3_1d_donchian_20_10_atr4', 'v3_1d_donchian_55_20_atr5', 'v3_4h_vol_compression_donchian_breakout', 'v3_4h_donchian_100_30_atr5', 'v3_4h_donchian_20_10_atr4', 'v3_4h_donchian_55_20_atr4']
- funding_stress_negative_at_3bps_or_more=true

## V3.1 Research-only 建议
- proceed_to_v3_1_research=false
- allowed_research_only=false
- strategy_development_allowed=false
- demo_live_allowed=false
- recommended_filters={"min_trend_efficiency": 0.35, "min_adx_or_proxy": 20.0, "allowed_regimes": ["strong_uptrend", "strong_downtrend"], "blocked_regimes": ["choppy", "high_vol_choppy", "compression"], "max_atr_percentile": 0.8, "min_ema_spread_pct": {"4h": 0.004, "1d": 0.01}}
- recommended_policy_families=[]
- rejected_policy_families=['1d_ema', '1d_donchian', '4h_ema', '4h_donchian', 'ensemble']

## 输出文件
- trend_regime_summary.json
- trend_regime_report.md
- data_quality.json
- regime_dataset.csv
- regime_by_symbol.csv
- regime_by_month.csv
- regime_by_quarter.csv
- regime_by_timeframe.csv
- trend_score_by_symbol.csv
- trend_score_by_month.csv
- trade_regime_attribution.csv
- policy_regime_performance.csv
- v3_1_regime_recommendations.json
