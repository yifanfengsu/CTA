# mainnet 历史 K 线重建（第 1 步：重建到新库）

日期：2026-06-10（UTC）
状态：**完成** —— 5 币 × 1,791,360 根全部下载入库，零缺口

## 背景

取证（reports/regime/data_contamination_forensics_20260610/）确认旧库
`.vntrader/database.db` 全部 1m 数据来自 OKX DEMO 环境。本任务从 OKX
mainnet 公开 REST 重建同区间数据到**独立新库**，为下一个任务（DEMO vs
mainnet 逐 bar 对比）提供基准。本任务只重建、不对比、不回测。

## 第 0 步：可行性探测（feasibility_probe.json）

- **(0a) OKX mainnet REST 深度**：`history-candles`（无 demo header）BTC 1m
  至少回溯到 2020-01-01；5 币在需求起点 2022-12-31T16:00Z 全部有数据，价格与
  真实市场吻合（BTC 16593.8 / ETH 1203.15 / SOL 10.162 / LINK 5.592 /
  DOGE 0.07071）。**REST 方案可行。**
- **(0b) 替代源**：Binance vision 月度 1m zip 存在（仅可作交叉验证源，价格
  非 OKX）；OKX 静态归档只有逐笔成交，不需要。
- **(0c) 量与限频**：8,956,800 bar ≈ 89,568 请求；限频 20 req/2s，按 8 req/s
  （留 20% 余量）约 3.1 小时。
- **(0d) 方案**：A（纯 OKX mainnet REST 一次全量）—— **用户已确认**。

## 执行方案（已确认的方案 A）

| 项 | 值 |
|---|---|
| 脚本 | `scripts/download_mainnet_history.py`（硬编码 mainnet，无 demo header，不读 OKX_SERVER/.env） |
| 新库 | `.vntrader/database_mainnet.db`（与旧库物理隔离；旧库零接触） |
| schema | 与旧库 dbbardata/dbbaroverview 完全一致 + 新增 `download_meta` 元数据表（source/server/script/时间/bar数/缺口数） |
| 数据约定 | symbol/exchange/interval/datetime（上海 naive、bar open）与旧库逐字符一致，便于对比脚本复用加载代码 |
| manifest | `download_manifests/<symbol>_1m_mainnet.json`，**显式 `"server": "MAINNET"` + `"demo_header": false` 字段**（吸取旧 manifest 无环境字段的教训） |
| 区间 | 2023-01-01 00:00 → 2026-05-28 23:59（Asia/Shanghai），与旧库完全相同 |
| 限频 | 全局 0.125s/请求（8 req/s）；指数退避重试 ≤8 次 |
| 断点续传 | chunk（5 天）粒度，重跑自动跳过 status=done |
| 顺序 | BTC → ETH → SOL → LINK → DOGE（保真度高的先下，对比任务可尽早开始） |
| 完整性 | 每币种完成后统计 bar 数 / 分钟网格缺口清单 / 与旧库 1,791,360 对比；**缺口如实记录，绝不填补/插值**（confirm≠1 的 bar 丢弃） |

冒烟测试（写入前验证）：BTC 2023-01-01 单日 1440/1440 bar，datetime 格式与
旧库一致，OHLC 为真实行情。

## 执行记录

- 2026-06-10T13:09:18Z 探测完成（方案 A 经用户确认）
- 2026-06-10T13:14:25Z 全量下载启动（单进程串行）
- 2026-06-10T14:34:13Z **改为 5 币并行重启**：实测单请求往返 ~0.8s 使串行吞吐仅
  ~1.26 req/s（瓶颈是网络延迟而非限频），串行全程需 ~20h；改为每币一个进程
  （断点续传保留 BTC 已完成的 42 chunk），聚合速率仍低于 8 req/s 预算。
  脚本相应增加：CLI 按币选择、SQLite WAL + busy_timeout（并发写保护）、
  每币独立完整性文件。
- 2026-06-10T20:19:34Z BTC 完成 → 21:20:55 LINK → 21:21:04 SOL →
  21:22:08 ETH → 21:22:09 DOGE 完成。全程无错误、无 chunk 失败、无锁冲突。
- 2026-06-10T23:30:16Z WAL checkpoint + 汇总收尾。

## 完整性统计（integrity_summary.json）

| 币种 | bar 数 | 分钟网格缺口 | 缺失分钟 | 与旧库(1,791,360)差值 | 区间 |
|---|---|---|---|---|---|
| BTC | 1,791,360 | 0 | 0 | 0 | 2023-01-01 00:00 → 2026-05-28 23:59 |
| ETH | 1,791,360 | 0 | 0 | 0 | 同上 |
| SOL | 1,791,360 | 0 | 0 | 0 | 同上 |
| LINK | 1,791,360 | 0 | 0 | 0 | 同上 |
| DOGE | 1,791,360 | 0 | 0 | 0 | 同上 |

- 总计 8,956,800 行；**缺口清单为空**（mainnet 该区间分钟网格完整，无需也未做任何填补）。
- 新库体积 1,515,925,504 字节（~1.41 GiB），与旧库（1.497 GB）同量级。
- `download_meta` 表 5 行均为 `server=MAINNET / source=okx-public-rest`；
  manifest 5 份均含 `"server": "MAINNET"`、`"demo_header": false`。

## 产物清单

- `.vntrader/database_mainnet.db` —— 新库（与旧库物理隔离）
- `scripts/download_mainnet_history.py` —— 下载脚本（mainnet 硬编码）
- `feasibility_probe.json` / `download_manifests/`（5 份）/
  `integrity_summary.json`（+ 每币 `integrity_<symbol>.json`）/ `run_log.txt`

## 铁律遵守情况

- 旧库 `database.db`：未打开、未写入（schema 参考来自取证阶段的只读查询）。
- `.env`：未读取、未修改（新脚本不依赖任何环境变量/凭证）。
- VPS：未触碰。
- 接口：仅 OKX 公开行情端点 `GET /api/v5/market/history-candles`，无鉴权 header。
- 未做 DEMO vs mainnet 对比，未跑任何回测。
