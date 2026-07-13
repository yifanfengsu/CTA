# OKX Historical Market Data Probe

## Changelog Audit
- 2025-09-02: OKX changelog says a new public endpoint was added for batch retrieving historical market data; it supports trade history, candlestick, funding rate, and 50/400/5000-level orderbook modules with daily/monthly aggregation.
- 2026-04-10: OKX changelog says module 11 (Borrowing rate) was added for request parameter module.
- changelog=https://www.okx.com/docs-v5/log_en/

## Official Endpoint Semantics
- endpoint_path=/api/v5/public/market-data-history
- request_parameters=module, instType, instIdList/instFamilyList, dateAggrType, begin, end
- funding_module=3
- aggregation=daily/monthly via dateAggrType; daily has module-specific limitations in OKX docs
- response=JSON response with groupDetails[].url download links
- auth_required=false
- rate_limit=5 requests per 2 seconds by IP
- timestamp_timezone=UTC+8 for modules 1, 2, 3, and 11
- max_query_range=20 days for daily, 20 months for monthly

## Probe Result
- endpoint_available=true
- endpoint_discovery_failed=false
- endpoint_path=/api/v5/public/market-data-history
- funding_module_available=true
- aggregation_supported=true
- auth_required=false
- can_auto_download=true
- can_cover_2023_2026=true
- probe_start=2026-03-01
- probe_end=2026-03-31
- response_kind=download_link
- recommended_next_step=run make download-funding-historical

## Notes
- If the official endpoint path cannot be confirmed, this probe does not guess candidate API URLs.
- No API key is written or required by this script.
- `funding_data_complete=false` must keep V3.1, Strategy V3, demo, and live gates closed.
