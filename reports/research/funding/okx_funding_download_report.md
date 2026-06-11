# OKX Funding Download Report

## Source Audit
- endpoint=https://www.okx.com/api/v5/public/funding-rate-history
- docs=https://app.okx.com/docs-v5/zh/#public-data-rest-api-get-funding-rate-history
- OKX docs semantics: `before` returns records newer than the requested `fundingTime`; `after` returns records older than the requested `fundingTime`; `limit` maximum is 400.
- Endpoint history may be bounded (for example recent three months); the downloader treats it as incomplete unless pagination reaches the requested start.
- fallback=https://www.okx.com/en-ar/historical-data provides historical perpetual funding rates from March 2022 onwards.
- Funding intervals are inferred from returned `fundingTime` differences; fixed 8h intervals are not assumed.

## Scope
- dry_run=false
- endpoint=https://www.okx.com/api/v5/public/funding-rate-history
- API key required=false
- timezone=Asia/Shanghai
- start=2023-01-01
- end=2026-03-31
- pagination_mode=backward
- limit=400
- max_pages=500
- stop_on_short_page=false
- pagination=backward pages using `after=<oldest fundingTime>`; short pages are confirmed with one more older request unless configured otherwise.

## Coverage Summary
- funding_data_complete=false
- endpoint_history_limit_suspected=true
- partial_pagination_failed=false
- request_trace=/home/yiast/vnpy_projects/cta_strategy/reports/research/funding/okx_funding_download_requests.csv

## Download Results
| inst_id | status | row_count | requested_start | first_available | last_available | reached_start | endpoint_limit_suspected | requests | warnings |
|---|---|---:|---|---|---|---|---|---:|---|
| BTC-USDT-SWAP | partial_endpoint_limited | 159 | 2022-12-31T16:00:00+00:00 | 2026-02-06T16:00:00+00:00 | 2026-03-31T08:00:00+00:00 | false | true | 2 | OKX returned a short page before start was reached: page_size=159, oldest_ms=1770393600000; funding data does not fully cover requested_start=2022-12-31T16:00:00+00:00; REST endpoint history lower bound appears newer than requested start |
| ETH-USDT-SWAP | partial_endpoint_limited | 159 | 2022-12-31T16:00:00+00:00 | 2026-02-06T16:00:00+00:00 | 2026-03-31T08:00:00+00:00 | false | true | 2 | OKX returned a short page before start was reached: page_size=159, oldest_ms=1770393600000; funding data does not fully cover requested_start=2022-12-31T16:00:00+00:00; REST endpoint history lower bound appears newer than requested start |
| SOL-USDT-SWAP | partial_endpoint_limited | 159 | 2022-12-31T16:00:00+00:00 | 2026-02-06T16:00:00+00:00 | 2026-03-31T08:00:00+00:00 | false | true | 2 | OKX returned a short page before start was reached: page_size=159, oldest_ms=1770393600000; funding data does not fully cover requested_start=2022-12-31T16:00:00+00:00; REST endpoint history lower bound appears newer than requested start |
| LINK-USDT-SWAP | partial_endpoint_limited | 159 | 2022-12-31T16:00:00+00:00 | 2026-02-06T16:00:00+00:00 | 2026-03-31T08:00:00+00:00 | false | true | 2 | OKX returned a short page before start was reached: page_size=159, oldest_ms=1770393600000; funding data does not fully cover requested_start=2022-12-31T16:00:00+00:00; REST endpoint history lower bound appears newer than requested start |
| DOGE-USDT-SWAP | partial_endpoint_limited | 159 | 2022-12-31T16:00:00+00:00 | 2026-02-06T16:00:00+00:00 | 2026-03-31T08:00:00+00:00 | false | true | 2 | OKX returned a short page before start was reached: page_size=159, oldest_ms=1770393600000; funding data does not fully cover requested_start=2022-12-31T16:00:00+00:00; REST endpoint history lower bound appears newer than requested start |

## Notes
- Raw funding CSV files are research inputs and are ignored by git.
- `funding_rate` uses OKX `realizedRate` when present, otherwise `fundingRate`.
- Download success does not imply data completeness; partial endpoint coverage cannot be used for strategy decisions.
- If the REST endpoint does not cover the requested start, manually import OKX Historical Data funding CSVs and rerun verification.
