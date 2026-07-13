# OKX Historical Funding Download Report

## Source Audit
- changelog_2025_09_02=OKX changelog says a new public endpoint was added for batch retrieving historical market data; it supports trade history, candlestick, funding rate, and 50/400/5000-level orderbook modules with daily/monthly aggregation.
- endpoint_url=https://www.okx.com/api/v5/public/market-data-history
- endpoint_available=true
- funding_module_available=true
- aggregation_supported=true
- auth_required=false
- can_auto_download=true
- request_parameters=module, instType, instIdList/instFamilyList, dateAggrType, begin, end
- funding_module=3
- response=JSON response with groupDetails[].url download links
- timestamp_timezone=UTC+8 for modules 1, 2, 3, and 11
- max_query_range=20 days for daily, 20 months for monthly
- rate_limit=5 requests per 2 seconds by IP

## Result
- status=downloaded
- dry_run=false
- failure_reason=
- plan_request_count=195
- downloaded_file_count=195
- extracted_csv_count=195

## Instrument Coverage
| inst_id | files_downloaded | row_count | first_time | last_time | complete |
|---|---:|---:|---|---|---|
| BTC-USDT-SWAP | 39 | 3558 | 2022-12-31T16:00:00+00:00 | 2026-03-31T08:00:00+00:00 | false |
| ETH-USDT-SWAP | 39 | 3558 | 2022-12-31T16:00:01+00:00 | 2026-03-31T08:00:00+00:00 | false |
| SOL-USDT-SWAP | 39 | 3558 | 2022-12-31T16:00:12+00:00 | 2026-03-31T08:00:00+00:00 | false |
| LINK-USDT-SWAP | 39 | 3558 | 2022-12-31T16:00:05+00:00 | 2026-03-31T08:00:00+00:00 | false |
| DOGE-USDT-SWAP | 39 | 3558 | 2022-12-31T16:00:11+00:00 | 2026-03-31T08:00:00+00:00 | false |

## Fallback
- recommended_next_step=run make verify-funding
- fallback=

## Gate Note
- Downloader output does not certify completeness; `make verify-funding` is the source of truth for `funding_data_complete`.
- If funding_data_complete=false, V3.1, Strategy V3, demo, and live remain forbidden.
