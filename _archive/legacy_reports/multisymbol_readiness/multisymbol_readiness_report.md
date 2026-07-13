# Multi-symbol Data Readiness Audit

## Summary
- configured_instruments=7
- ready_symbols=0
- can_enter_trend_v3=false
- required_symbols=BTCUSDT_SWAP_OKX.GLOBAL, ETHUSDT_SWAP_OKX.GLOBAL, SOLUSDT_SWAP_OKX.GLOBAL, LINKUSDT_SWAP_OKX.GLOBAL, DOGEUSDT_SWAP_OKX.GLOBAL
- optional_symbols=BNBUSDT_SWAP_OKX.GLOBAL, XRPUSDT_SWAP_OKX.GLOBAL
- min_ready_symbols=5
- coverage_window=2023-01-01 to 2026-03-31 (Asia/Shanghai, 1m)
- full_trend_v3_window=false
- database_path=/home/yiast/vnpy_projects/cta_strategy/.vntrader/database.db

## Coverage Window Note
The readiness counts below apply to this audit coverage window. A short-window audit validates the download chain for that window only and does not prove full Trend V3 readiness unless `full_trend_v3_window=true`.

## Required Instruments
| vt_symbol | role | okx_inst_id | okx_inst_id_source | product | metadata_complete | has_any_history | required_coverage_ready | total_count | expected_count | missing_count | gap_count | can_backtest_for_window | warning |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| BTCUSDT_SWAP_OKX.GLOBAL | required |  | fallback_available_but_not_canonical |  | false | true | true | 1707840 | 1707840 | 0 | 0 | false | missing_canonical_field: okx_inst_id; missing_canonical_field: product; missing_canonical_field: needs_okx_contract_metadata_refresh |
| ETHUSDT_SWAP_OKX.GLOBAL | required |  | fallback_available_but_not_canonical |  | false | true | true | 1707840 | 1707840 | 0 | 0 | false | missing_canonical_field: okx_inst_id; missing_canonical_field: product; missing_canonical_field: needs_okx_contract_metadata_refresh |
| SOLUSDT_SWAP_OKX.GLOBAL | required |  | fallback_available_but_not_canonical |  | false | true | true | 1707840 | 1707840 | 0 | 0 | false | missing_canonical_field: okx_inst_id; missing_canonical_field: product; missing_canonical_field: needs_okx_contract_metadata_refresh |
| LINKUSDT_SWAP_OKX.GLOBAL | required |  | fallback_available_but_not_canonical |  | false | true | true | 1707840 | 1707840 | 0 | 0 | false | missing_canonical_field: okx_inst_id; missing_canonical_field: product; missing_canonical_field: needs_okx_contract_metadata_refresh |
| DOGEUSDT_SWAP_OKX.GLOBAL | required |  | fallback_available_but_not_canonical |  | false | true | true | 1707840 | 1707840 | 0 | 0 | false | missing_canonical_field: okx_inst_id; missing_canonical_field: product; missing_canonical_field: needs_okx_contract_metadata_refresh |

## Optional Instruments
| vt_symbol | role | okx_inst_id | okx_inst_id_source | product | metadata_complete | has_any_history | required_coverage_ready | total_count | expected_count | missing_count | gap_count | can_backtest_for_window | warning |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| BNBUSDT_SWAP_OKX.GLOBAL | optional | BNB-USDT-SWAP | okx_inst_id | SWAP | false | false | false | 0 | 1707840 | 1707840 | 1 | false | invalid_canonical_field: size; invalid_canonical_field: pricetick; invalid_canonical_field: min_volume; needs_okx_contract_metadata_refresh; no_local_sqlite_history_for_BNBUSDT_SWAP_OKX.GLOBAL_1m |
| XRPUSDT_SWAP_OKX.GLOBAL | optional | XRP-USDT-SWAP | okx_inst_id | SWAP | false | false | false | 0 | 1707840 | 1707840 | 1 | false | invalid_canonical_field: size; invalid_canonical_field: pricetick; invalid_canonical_field: min_volume; needs_okx_contract_metadata_refresh; no_local_sqlite_history_for_XRPUSDT_SWAP_OKX.GLOBAL_1m |

## Makefile
- audit_multisymbol_target_exists=true
- batch_download_target_exists=true
- batch_verify_target_exists=true

## Source Capability
- download_supports_vt_symbol=true
- verify_supports_vt_symbol=true
- research_trend_v2_single_symbol=true
- research_trend_v2_filters_by_vt_symbol=true
- trend_v2_default_output_has_symbol_token=false
- tests_reference_non_btc_symbol=true

## Blocking Reasons
- ready_symbols_below_minimum: ready=0 minimum=5
- required_ready_symbols_missing: BTCUSDT_SWAP_OKX.GLOBAL, ETHUSDT_SWAP_OKX.GLOBAL, SOLUSDT_SWAP_OKX.GLOBAL, LINKUSDT_SWAP_OKX.GLOBAL, DOGEUSDT_SWAP_OKX.GLOBAL
- incomplete_required_instrument_metadata: BTCUSDT_SWAP_OKX.GLOBAL, ETHUSDT_SWAP_OKX.GLOBAL, SOLUSDT_SWAP_OKX.GLOBAL, LINKUSDT_SWAP_OKX.GLOBAL, DOGEUSDT_SWAP_OKX.GLOBAL

## Optional Warnings
- incomplete_optional_instrument_metadata: BNBUSDT_SWAP_OKX.GLOBAL, XRPUSDT_SWAP_OKX.GLOBAL
- missing_optional_local_sqlite_history: BNBUSDT_SWAP_OKX.GLOBAL, XRPUSDT_SWAP_OKX.GLOBAL

## Decision
Do not enter full Trend V3 until metadata is canonical, the default full coverage window is complete for the required symbols, at least the minimum number of symbols are ready, and batch download plus batch verify helpers exist. Optional instruments are reported as warnings and do not block first-batch Trend V3 readiness. The current audit does not download data and does not change strategy trading logic.
