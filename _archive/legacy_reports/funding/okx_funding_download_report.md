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
- start=2026-04-01
- end=2026-06-11
- pagination_mode=backward
- limit=400
- max_pages=500
- stop_on_short_page=false
- pagination=backward pages using `after=<oldest fundingTime>`; short pages are confirmed with one more older request unless configured otherwise.

## Coverage Summary
- funding_data_complete=true
- endpoint_history_limit_suspected=false
- partial_pagination_failed=false
- request_trace=/home/yiast/vnpy_projects/cta_strategy/reports/research/funding/okx_funding_download_requests.csv

## Download Results
| inst_id | status | row_count | requested_start | first_available | last_available | reached_start | endpoint_limit_suspected | requests | warnings |
|---|---|---:|---|---|---|---|---|---:|---|
| BTC-USDT-SWAP | complete | 216 | 2026-03-31T16:00:00+00:00 | 2026-03-31T16:00:00+00:00 | 2026-06-11T08:00:00+00:00 | true | false | 1 |  |
| ETH-USDT-SWAP | complete | 216 | 2026-03-31T16:00:00+00:00 | 2026-03-31T16:00:00+00:00 | 2026-06-11T08:00:00+00:00 | true | false | 1 |  |
| SOL-USDT-SWAP | complete | 216 | 2026-03-31T16:00:00+00:00 | 2026-03-31T16:00:00+00:00 | 2026-06-11T08:00:00+00:00 | true | false | 1 |  |
| LINK-USDT-SWAP | complete | 216 | 2026-03-31T16:00:00+00:00 | 2026-03-31T16:00:00+00:00 | 2026-06-11T08:00:00+00:00 | true | false | 1 |  |
| DOGE-USDT-SWAP | complete | 216 | 2026-03-31T16:00:00+00:00 | 2026-03-31T16:00:00+00:00 | 2026-06-11T08:00:00+00:00 | true | false | 1 |  |

## Notes
- Raw funding CSV files are research inputs and are ignored by git.
- `funding_rate` uses OKX `realizedRate` when present, otherwise `fundingRate`.
- Download success does not imply data completeness; partial endpoint coverage cannot be used for strategy decisions.
- If the REST endpoint does not cover the requested start, manually import OKX Historical Data funding CSVs and rerun verification.
