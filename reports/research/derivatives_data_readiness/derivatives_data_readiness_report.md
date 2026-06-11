# OKX Derivatives Data Readiness Audit

## Scope
- inst_ids=BTC-USDT-SWAP, ETH-USDT-SWAP, SOL-USDT-SWAP, LINK-USDT-SWAP, DOGE-USDT-SWAP
- ccys=BTC, ETH, SOL, LINK, DOGE
- start=2023-01-01
- end=2026-03-31
- timezone=Asia/Shanghai
- probe_date=2026-03-01
- audit_only=true
- endpoint_probe_result_is_not_strategy_conclusion=true
- strategy_development_allowed=false
- demo_live_allowed=false

## Local Funding Status
- funding_source=OKX Historical Market Data
- funding_data_available=true
- funding_data_complete=true

## Required Questions
1. 当前不扩大币种的前提下，是否有足够衍生品数据继续趋势确认研究？
   - can_enter_derivatives_confirmed_trend_research=false。
2. 哪些数据可无密钥获取？
   - Open Interest, Funding Rate History, Mark Price, Mark Price Candles History, Index Ticker, Index Candles History, Taker Buy/Sell Volume, Contract Long/Short Account Ratio, Contracts Open Interest and Volume, Contract Open Interest History, Premium History
3. 哪些数据只支持近期，不能覆盖 2023-2026？
   - Open Interest, Mark Price, Index Ticker, Taker Buy/Sell Volume, Contract Long/Short Account Ratio, Contracts Open Interest and Volume, Contract Open Interest History, Premium History
4. 哪些数据可以通过分段 API 获取？
   - Funding Rate History, Mark Price Candles History, Index Candles History, Taker Buy/Sell Volume, Contract Long/Short Account Ratio, Contracts Open Interest and Volume, Contract Open Interest History, Premium History
5. 哪些数据需要 OKX Historical Market Data 文件？
   - taker buy/sell volume, contract long/short account ratio, contracts open interest and volume, open interest history, detailed liquidation data
6. 哪些数据完全不可用？
   - detailed liquidation data, private account level features
7. 推荐下一步是 download derivatives metrics / import historical files / pause research？
   - recommended_next_step=pause research
8. 是否允许进入 Derivatives-confirmed Trend Research？
   - can_enter_derivatives_confirmed_trend_research=false
9. 是否允许 Strategy V3 / demo / live？
   - strategy_development_allowed=false
   - demo_live_allowed=false

## Endpoint Probe Results
| endpoint | available | auth_required | period | can_cover_2023_2026 | usable | warning |
|---|---:|---:|---|---:|---:|---|
| Open Interest | true | false | current | false | false | current_snapshot_only |
| Funding Rate History | true | false | 8h | true | true | local_actual_funding_complete_from_okx_historical_market_data |
| Mark Price | true | false | current | false | false | current_snapshot_only |
| Mark Price Candles History | true | false | 1D | true | true |  |
| Index Ticker | true | false | current | false | false | current_snapshot_only |
| Index Candles History | true | false | 1D | true | true |  |
| Taker Buy/Sell Volume | true | false | 1D | false | false | start_boundary_probe_failed_or_empty |
| Contract Long/Short Account Ratio | true | false | 1D | false | false | start_boundary_probe_failed_or_empty; docs_added_after_requested_start:2024-06-13 |
| Contracts Open Interest and Volume | true | false | 1D | false | false | start_boundary_probe_failed_or_empty |
| Contract Open Interest History | true | false | 1D | false | false | start_boundary_probe_failed_or_empty; docs_added_after_requested_start:2024-06-13 |
| Premium History | true | false | historical probe | false | false | start_boundary_probe_failed_or_empty |

## Proposed Feature Tiers
### Tier 1
- actual funding rate (available_2023_2026)
- funding dispersion (available_2023_2026)
- funding sign breadth (available_2023_2026)
- funding trend (available_2023_2026)
- mark price (segment_download_ready)
- index price (segment_download_ready)

### Tier 2
- taker buy/sell volume (recent_available_full_window_unconfirmed)
- contract long/short account ratio (recent_available_full_window_unconfirmed)
- contracts open interest and volume (recent_available_full_window_unconfirmed)
- open interest history (recent_available_full_window_unconfirmed)
- mark price candle history (segment_download_ready)
- index price candle history (segment_download_ready)

### Tier 3
- premium/basis historical (derive_from_mark_index)
- detailed liquidation data (unavailable_or_out_of_scope)
- private account level features (forbidden_private_key_required)

## Decision Gate
- non_price_derivatives_feature_category_count=2
- non_price_derivatives_feature_categories_available=funding, basis_premium
- required_feature_mix_ok=false
- data_download_plan_executable=false
- no_private_api_key_required=true
- can_enter_derivatives_confirmed_trend_research=false
- strategy_development_allowed=false
- demo_live_allowed=false
- recommended_next_step=pause research

## Blockers
- non_price_derivatives_feature_categories_below_3:2
- open_interest_feature_missing_or_not_2023_2026
- taker_flow_or_long_short_ratio_missing_or_not_2023_2026
- derivatives_data_download_plan_not_executable

## Source Notes
- OKX API docs: https://www.okx.com/docs-v5/en/
- OKX changelog: https://www.okx.com/docs-v5/log_en/
- OKX changelog notes contract derivatives statistics endpoints were added on 2024-06-13; start-boundary probes decide whether 2023 coverage is actually usable.
 