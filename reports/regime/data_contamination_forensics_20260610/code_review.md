# 代码审查记录（第 0-2 步）

取证对象：`scripts/download_okx_history.py`（2455 行，全文已逐行通读）
辅助对象：`.venv/.../vnpy_okx/okx_gateway.py`、`scripts/download_history.py`、`scripts/merge_recent.py`

## 第 0 步：代码结构

### (0a) 主要函数

| 函数 | 行号 | 作用 |
|---|---|---|
| `ManifestManager` | 178-344 | manifest（checkpoint）JSON 的原子读写、chunk 状态机 |
| `GatewayObserver` | 347-410 | 捕获 vnpy 网关日志与合约元数据 |
| `parse_args` | 413-546 | CLI 参数（含 `--server {DEMO,REAL}`、`--source {auto,gateway,rest}`、`--repair-missing`） |
| `configure_sqlite_settings` | 555-570 | 强制 vnpy 数据库为本地 sqlite `database.db` |
| `read_env_config` | 573-615 | 读 `.env` 的 OKX_API_KEY/SECRET/PASSPHRASE/**OKX_SERVER**/代理 |
| `build_connection_setting` | 685-705 | 把 env 值（含 server=DEMO/REAL）装配进 vnpy 网关 connect 字典 |
| `initialize_gateway_context` | 830-874 | 连接 vnpy_okx 网关、等合约元数据 |
| `query_gateway_chunk` | 1077-1111 | 经 `MainEngine.query_history()`（即 vnpy_okx 网关）拉一个 chunk |
| `request_okx_history_page` | 1219-1281 | 直连 `https://www.okx.com/api/v5/market/history-candles`（公共 REST） |
| `query_rest_chunk` | 1284-1348 | REST 翻页拉一个 chunk |
| `query_chunk_with_retry` | 1408-1502 | 源选择 + 回退：gateway 失败 → 回退 rest |
| `save_bars_to_database` | 1528-1578 | `database.save_bar_data()` 写 sqlite（带锁重试） |
| `verify_chunk_in_database` | 1617-1640 | 写后逐 chunk 覆盖率校验 |
| `main` | 1844-2451 | 主流程：规划 chunk → 逐 chunk 下载/保存/校验/修复 → 最终覆盖率 |

### (0b) 入口流程

CLI → `parse_args` → `configure_sqlite_settings`（锁定 database.db）→ 覆盖率规划
（`--repair-missing` 时只下载缺口，否则全区间）→ `ManifestManager.sync_chunks`
→（source=gateway/auto 时）`initialize_gateway_context`：**用 .env 的 OKX_SERVER 连 vnpy_okx 网关**
→ 逐 chunk：`query_chunk_with_retry`（gateway 优先，败退 rest）→ `normalize_chunk_bars`（去重/裁剪，**不造数据**）
→ `save_bars_to_database` → `verify_chunk_in_database` →（仍有缺口且 `--repair-missing`）对缺口区间**用同样的源**重新下载再存
→ manifest 记 `source_used`（gateway/rest/mixed）。

### (0c) OKX_SERVER 环境切换 —— 有

- `read_env_config`（573-615 行）：必填字段含 `OKX_SERVER`（587 行）；`--server` CLI 可覆盖（451-455、589-590、607 行）。
- `build_connection_setting`（698 行）：`setting[field_map.server] = env_config.server` → 传给 `main_engine.connect()`（865 行）。
- 读取后改变什么：**vnpy_okx 网关整体切到 DEMO 或 REAL**。本仓库 `.env` 第 4 行实际值：`OKX_SERVER=DEMO`（文件 mtime 2026-04-17 08:36，早于全部已记录下载会话且此后未改动）。

### (0d) x-simulated-trading header

- `download_okx_history.py` 自身**从不**设置该 header；其 REST 路径只设 `User-Agent`（1234-1237 行）。
- 但 gateway 路径委托给 vnpy_okx，而 `vnpy_okx/okx_gateway.py` 中：
  - 585-586 行：`if server == "DEMO": self.simulated = True`
  - 515-520 行（`sign()` 回调，对**所有** REST 请求生效，含公共行情）：`request.headers["x-simulated-trading"] = "1"`
  - 551-552 行：签名请求同样追加该 header
  - 968-1004 行：`query_history()` 走 `/api/v5/market/history-candles` —— 同受上述 header 影响
  - 53 行：`DEMO_REST_HOST = "https://www.okx.com"`（同域名，靠 header 区分 demo 环境）
- 结论：**OKX_SERVER=DEMO 时，gateway 路径的历史 K 线请求带 `x-simulated-trading: 1`，返回的是 OKX 模拟盘环境行情。**

### (0e) 写库路径与来源标签

写库共 3 处，全部经 `save_bars_to_database`：
1. 逐 chunk 保存（2090 行）
2. `--repair-missing` 缺口修复保存（2151 行）
3. 非逐块模式的缓冲批量保存（2270 行）

来源标签：
- **数据库 `dbbardata` 无任何来源列**（vnpy schema：symbol/exchange/datetime/interval/OHLCV...）；BarData 的 `gateway_name`（"OKX" vs "OKX_REST"）**不落库**。
- manifest 记 `source_used`（gateway/rest/mixed），**但不记 DEMO/REAL**——manifest schema（sync_chunks，279-298 行）无 server 字段。
- server 值仅出现在运行日志 `logs/download_okx_history.log` 的 `okx.connect` 事件（863 行 `sanitize_setting` 只脱敏 key/secret/passphrase，Server 明文）。

## 第 1 步：gap 修复 / 合成相关路径

关键字逐项核查（gap/fill/missing/interpolate/synthetic/synth/fake/DEMO/demo/simulated/sandbox/fallback/retry/alternate/linear/extrapolate/manifest/merge）：

- `interpolate`、`synthetic`、`synth`、`fake`、`sandbox`、`linear`、`extrapolate`：**0 命中**。脚本没有任何本地造数逻辑。
- `gap`/`missing`/`fill`：全部命中在覆盖率统计与 `--repair-missing` 修复路径（1785-1796、2107-2165 行）。修复逻辑逐行确认：对 `verify_chunk_in_database` 报告的缺口区间构造 `repair_chunk`，再次调用 `query_chunk_with_retry`（**同样的 gateway→rest 源**）→ `normalize_chunk_bars`（仅去重裁剪）→ 写库。**不插值、不外推、不复制邻近 bar。**
- `DEMO`/`demo`/`simulated`：命中只在 `--server` choices（453 行）、错误提示文案（747/768/795/969 行）。脚本无"切到 DEMO 补数据"的主动逻辑——DEMO 与否完全由连接配置决定，且**一次执行内不可切换**（server 在 connect 时固定）。
- `fallback`：唯一回退方向是 **gateway → 公共 REST（mainnet）**（1456-1464 行）。
- `merge`/`manifest`：均为 checkpoint 管理，无数据合成。

结论：原诊断推测的"**mainnet 失败 → 切 DEMO 补 gap**"路径**不存在**；真实情况相反——**DEMO（gateway）是主路径，mainnet（REST）只是从未触发的备胎**。

## 第 2 步：OKX 调用方式汇总

### (2a) 全部 OKX 调用点

| 调用点 | URL | header | 环境 |
|---|---|---|---|
| gateway `query_history`（经 vnpy_okx） | `https://www.okx.com/api/v5/market/history-candles` | `x-simulated-trading: 1`（当 Server=DEMO） | **由 OKX_SERVER 决定** |
| `request_okx_history_page`（1219 行） | `https://www.okx.com/api/v5/market/history-candles`（70 行常量） | 仅 `User-Agent: cta-history-downloader/1.0` | **恒 mainnet** |

### (2b) 双源共存规则

- `--source auto`（默认）：每个 chunk 先 gateway（重试 ≤8 次），全败才 REST。
- 同一次执行中 gateway 的 DEMO/REAL **不可切换**；REST 恒 mainnet。
- 两源数据落**同一个** `database.db` 同一张表，无来源区分。

### (2c) 重点场景判定

"mainnet 请求 → gap/失败 → 切 DEMO 补齐 → 写主库"：**不存在**。
实际存在且已发生的是其镜像："**DEMO 网关请求（主路径）→ 写主库；mainnet REST 备胎从未触发**"
（全部日志中 `history.fallback_rest` 出现 0 次；全部 manifest 中 `source_used="rest"` 出现 0 次）。

## 附：另两个写库脚本（第 6 步线索，纯代码事实）

| 脚本 | DEMO 证据 | 写库方式 |
|---|---|---|
| `scripts/download_history.py`（196 行） | 第 66 行**硬编码** `"x-simulated-trading": "1"`，请求 `/api/v5/market/candles` | `database.save_bar_data()`（169 行）写同一 database.db |
| `scripts/merge_recent.py`（88 行） | 第 38 行**硬编码** `"x-simulated-trading": "1"`，请求 `/api/v5/market/candles` | 裸 sqlite3 `INSERT OR IGNORE INTO dbbardata`（72 行） |

即：本仓库**所有三条已知写库路径在配置/代码层面全部指向 OKX DEMO 环境**，不存在任何确定走 mainnet 的写库路径（download_okx_history.py 的 REST 备胎除外，但它从未执行过）。
