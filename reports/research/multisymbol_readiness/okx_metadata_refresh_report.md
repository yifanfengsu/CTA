# OKX Instrument Metadata Refresh

## Summary
- dry_run=false
- write=true
- server=DEMO
- requested=5
- refreshed=5
- metadata_complete=5
- needs_refresh=0

## Instruments
| inst_id | vt_symbol | product | size | pricetick | min_volume | dry_run/write | canonical_schema_complete | needs_refresh | warning |
|---|---|---:|---:|---:|---:|---|---|---|---|
| BTC-USDT-SWAP | BTCUSDT_SWAP_OKX.GLOBAL | SWAP | 0.01 | 0.1 | 0.01 | write | true | false | - |
| ETH-USDT-SWAP | ETHUSDT_SWAP_OKX.GLOBAL | SWAP | 0.1 | 0.01 | 0.01 | write | true | false | - |
| SOL-USDT-SWAP | SOLUSDT_SWAP_OKX.GLOBAL | SWAP | 1.0 | 0.01 | 0.01 | write | true | false | - |
| LINK-USDT-SWAP | LINKUSDT_SWAP_OKX.GLOBAL | SWAP | 1.0 | 0.001 | 0.1 | write | true | false | - |
| DOGE-USDT-SWAP | DOGEUSDT_SWAP_OKX.GLOBAL | SWAP | 1000.0 | 1e-05 | 0.01 | write | true | false | - |

## Safety
- This script only uses OKX public contract metadata.
- It does not place orders, connect private trading, or write API keys.
- Failed refreshes keep placeholder metadata marked as needing refresh.
