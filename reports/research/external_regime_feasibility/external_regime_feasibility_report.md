# External Regime Classifier Feasibility Audit

## Scope
- symbols=BTCUSDT_SWAP_OKX.GLOBAL, ETHUSDT_SWAP_OKX.GLOBAL, SOLUSDT_SWAP_OKX.GLOBAL, LINKUSDT_SWAP_OKX.GLOBAL, DOGEUSDT_SWAP_OKX.GLOBAL
- start=2023-01-01
- end=2026-03-31
- timezone=Asia/Shanghai
- audit_type=research_only_feasibility
- strategy_development_allowed=false
- demo_live_allowed=false

## Data Status
- market_data_complete=true
- funding_data_complete=true
- coverage_window=2023-01-01 to 2026-03-31
- funding_source=OKX Historical Market Data
- local_open_interest_nonzero_available=false

## Required Questions
1. 当前不扩大币种的前提下，是否还能做新的趋势研究？
   - 可以，但仅限 external regime classifier research，不是策略开发。
2. 当前已有数据能构造哪些 regime 特征？
   - internal market features:
- trend breadth
- cross-symbol correlation
- cross-symbol dispersion
- realized volatility regime
- market-wide ATR percentile
- number of symbols above EMA50/EMA200
- number of symbols in strong trend
- market-wide drawdown / rebound state
   - funding features:
- average funding rate across symbols
- funding dispersion
- extreme funding count
- funding trend
- funding sign breadth
3. 哪些特征需要额外下载？
- open interest
- long/short ratio
- taker buy/sell volume
- premium index / basis
- mark/index price divergence
4. 哪些特征需要 API key？
- none
5. 哪些特征可以无密钥从 OKX public endpoint 获取？
- open interest
- long/short ratio
- taker buy/sell volume
- premium index / basis
- mark/index price divergence
6. 哪些特征完全缺失？
- open interest
- long/short ratio
- taker buy/sell volume
- premium index / basis
- mark/index price divergence
7. 是否建议进入 External Regime Classifier Research？
   - external_regime_classifier_research_allowed=true
8. 如果建议，下一步研究范围是什么？
- Build a research-only dataset of classifier features from existing OHLCV and funding data.
- Define labels independently from failed policy entry/exit outcomes.
- Audit train/validation/oos stability before any strategy proposal.
- Keep strategy_development_allowed=false and demo_live_allowed=false.
9. 如果不建议，是否应暂停策略开发？
   - 否；但仍禁止策略开发和 demo/live，只允许 classifier research。

## Missing External Feature Detail
| feature | local status | key required | public endpoint |
|---|---|---|---|
| open interest | missing_locally_public_no_key_download_required | false | /api/v5/public/open-interest; /api/v5/rubik/stat/contracts/open-interest-history |
| long/short ratio | missing_locally_public_no_key_download_required | false | /api/v5/rubik/stat/contracts/long-short-account-ratio |
| taker buy/sell volume | missing_locally_public_no_key_download_required | false | /api/v5/rubik/stat/taker-volume; /api/v5/rubik/stat/contracts/taker-volume |
| premium index / basis | missing_locally_public_no_key_download_required | false | /api/v5/public/premium-history; OKX basis statistics endpoint |
| mark/index price divergence | missing_locally_public_no_key_download_required | false | /api/v5/public/mark-price; /api/v5/public/index-tickers; mark/index candlestick endpoints |

## Decision Gate
- computable_regime_feature_count=13
- required_computable_regime_feature_count=8
- no_failed_policy_entry_exit_reuse=true
- no_private_api_key_required_for_computable_features=true
- data_coverage_ok=true
- external_regime_classifier_research_allowed=true
- strategy_development_allowed=false
- demo_live_allowed=false
- recommended_next_step=Proceed to research-only external regime classifier feature design and label audit; do not build a strategy.

## Blockers
- none

## Source Notes
- OKX public API documentation: https://www.okx.com/docs-v5/en/
- Public endpoint availability here is a feasibility classification only; historical backfill depth still needs a separate downloader/probe before model training.
