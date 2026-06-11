# Binance vision 1m K 线（外部交叉验证源，永久资产）

| 项 | 值 |
|---|---|
| 来源 | `https://data.binance.vision/data/futures/um/monthly/klines/<SYM>/1m/`（Binance 官方公开静态归档，生产行情，无 demo 变体） |
| 口径 | **Binance UM USDT 本位永续 1m last-price klines**（bar open 时刻，UTC；2025-01 起文件时间戳为微秒，此前为毫秒） |
| 交易对 | BTCUSDT / ETHUSDT / SOLUSDT / LINKUSDT / DOGEUSDT |
| 区间 | 2023-01 → 2026-05（41 个月 × 5 币 = 205 个 zip） |
| 校验 | 每文件经 vision 官方 `.CHECKSUM`（sha256）验证，205/205 通过 |
| 下载时间 | 2026-06-11（UTC），脚本 `scripts/download_binance_vision.py`，详单 `manifest.json` |
| 体积 | ~344 MB（zip 不入 git，本 README 与 manifest.json 入库） |

## 用途与注意

- 首要用途：OKX mainnet 库的外部交叉验证
  （`reports/regime/data_trust_closure_20260611/`）。
- **这是另一个交易所的价格**：OKX vs Binance 存在正常永续基差，
  验证目标是"同一个真实市场"（无系统性脱锚/无合成形态），不是逐 tick 相等。
- 与 OKX 库对齐口径：OKX 库 datetime 为 Asia/Shanghai naive（bar open）→
  UTC = naive − 8h；Binance open_time 为 UTC 毫秒（≥2025-01 为微秒）。
- CSV 列：open_time, open, high, low, close, volume, close_time,
  quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
  （部分新文件首行带表头，解析时需检测）。
