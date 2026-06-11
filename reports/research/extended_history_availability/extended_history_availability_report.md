# Extended History Availability Audit

## Required Questions
1. Current 2025-01-01 to 2026-03-31 complete: true.
2. 2023-01-01 to 2026-03-31 missing symbols: none.
3. 2021-01-01 to 2026-03-31 feasible with current local data: false.
4. Suspected listing later than 2021/2023: none confirmed; listing_time_unknown=BTCUSDT_SWAP_OKX.GLOBAL, DOGEUSDT_SWAP_OKX.GLOBAL, ETHUSDT_SWAP_OKX.GLOBAL, LINKUSDT_SWAP_OKX.GLOBAL, SOLUSDT_SWAP_OKX.GLOBAL.
5. Recommended next download window: 2021-01-01:2026-03-31.
6. Can enter Extended Trend Research: true.
7. Blocking reasons: none.

## Summary
- symbols=BTCUSDT_SWAP_OKX.GLOBAL, ETHUSDT_SWAP_OKX.GLOBAL, SOLUSDT_SWAP_OKX.GLOBAL, LINKUSDT_SWAP_OKX.GLOBAL, DOGEUSDT_SWAP_OKX.GLOBAL
- interval=1m
- timezone=Asia/Shanghai
- database_path=/home/yiast/vnpy_projects/cta_strategy/.vntrader/database.db
- database_exists=true
- dbbardata_table_exists=true
- matches_verify_default_database_path=true
- check_okx_listing_metadata=false
- recommended_next_download_window=2021-01-01:2026-03-31
- recommendation_reason=2021_window_is_next_missing_window_after_2023_ready
- can_enter_extended_trend_research=true

## Window Readiness
| window | ready_symbols | missing_symbols | missing_count | recommended |
|---|---:|---|---:|---|
| 2025-01-01:2026-03-31 | 5 | - | 0 | false |
| 2023-01-01:2026-03-31 | 5 | - | 0 | false |
| 2021-01-01:2026-03-31 | 0 | BTCUSDT_SWAP_OKX.GLOBAL, ETHUSDT_SWAP_OKX.GLOBAL, SOLUSDT_SWAP_OKX.GLOBAL, LINKUSDT_SWAP_OKX.GLOBAL, DOGEUSDT_SWAP_OKX.GLOBAL | 5256000 | true |

## Main Missing Ranges
| symbol | window | missing_count | suggested_command |
|---|---|---:|---|
| BTCUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1051200 | `python scripts/download_okx_history.py --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL --interval 1m --start 2021-01-01 --end 2022-12-31 --chunk-days 3 --timezone Asia/Shanghai --resume --repair-missing --source auto` |
| ETHUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1051200 | `python scripts/download_okx_history.py --vt-symbol ETHUSDT_SWAP_OKX.GLOBAL --interval 1m --start 2021-01-01 --end 2022-12-31 --chunk-days 3 --timezone Asia/Shanghai --resume --repair-missing --source auto` |
| SOLUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1051200 | `python scripts/download_okx_history.py --vt-symbol SOLUSDT_SWAP_OKX.GLOBAL --interval 1m --start 2021-01-01 --end 2022-12-31 --chunk-days 3 --timezone Asia/Shanghai --resume --repair-missing --source auto` |
| LINKUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1051200 | `python scripts/download_okx_history.py --vt-symbol LINKUSDT_SWAP_OKX.GLOBAL --interval 1m --start 2021-01-01 --end 2022-12-31 --chunk-days 3 --timezone Asia/Shanghai --resume --repair-missing --source auto` |
| DOGEUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1051200 | `python scripts/download_okx_history.py --vt-symbol DOGEUSDT_SWAP_OKX.GLOBAL --interval 1m --start 2021-01-01 --end 2022-12-31 --chunk-days 3 --timezone Asia/Shanghai --resume --repair-missing --source auto` |

## Symbol Coverage
| symbol | window | total_count | expected_count | missing_count | gap_count | coverage_ratio | history_ready | listing_before_start | warning |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| BTCUSDT_SWAP_OKX.GLOBAL | 2025-01-01:2026-03-31 | 655200 | 655200 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| BTCUSDT_SWAP_OKX.GLOBAL | 2023-01-01:2026-03-31 | 1707840 | 1707840 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| BTCUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1707840 | 2759040 | 1051200 | 1 | 0.618998 | false | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| ETHUSDT_SWAP_OKX.GLOBAL | 2025-01-01:2026-03-31 | 655200 | 655200 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| ETHUSDT_SWAP_OKX.GLOBAL | 2023-01-01:2026-03-31 | 1707840 | 1707840 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| ETHUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1707840 | 2759040 | 1051200 | 1 | 0.618998 | false | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| SOLUSDT_SWAP_OKX.GLOBAL | 2025-01-01:2026-03-31 | 655200 | 655200 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| SOLUSDT_SWAP_OKX.GLOBAL | 2023-01-01:2026-03-31 | 1707840 | 1707840 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| SOLUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1707840 | 2759040 | 1051200 | 1 | 0.618998 | false | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| LINKUSDT_SWAP_OKX.GLOBAL | 2025-01-01:2026-03-31 | 655200 | 655200 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| LINKUSDT_SWAP_OKX.GLOBAL | 2023-01-01:2026-03-31 | 1707840 | 1707840 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| LINKUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1707840 | 2759040 | 1051200 | 1 | 0.618998 | false | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| DOGEUSDT_SWAP_OKX.GLOBAL | 2025-01-01:2026-03-31 | 655200 | 655200 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| DOGEUSDT_SWAP_OKX.GLOBAL | 2023-01-01:2026-03-31 | 1707840 | 1707840 | 0 | 0 | 1.000000 | true | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |
| DOGEUSDT_SWAP_OKX.GLOBAL | 2021-01-01:2026-03-31 | 1707840 | 2759040 | 1051200 | 1 | 0.618998 | false | unknown | metadata_incomplete: okx_inst_id, product; okx_listing_metadata_not_checked; listing_before_window_start_unknown |

## Blocking Reasons
- none

## Optional Warnings
- listing_time_unknown: BTCUSDT_SWAP_OKX.GLOBAL,DOGEUSDT_SWAP_OKX.GLOBAL,ETHUSDT_SWAP_OKX.GLOBAL,LINKUSDT_SWAP_OKX.GLOBAL,SOLUSDT_SWAP_OKX.GLOBAL

## Decision Notes
- This audit is data availability only; it is not a strategy return conclusion.
- It does not download data, place orders, connect private trading, write API keys, or modify strategy logic.
- If OKX listing metadata is not checked or lacks listTime, listing_before_window_start remains unknown as a warning.
- Extended Trend Research requires the selected extended window to be fully covered locally before research starts.
