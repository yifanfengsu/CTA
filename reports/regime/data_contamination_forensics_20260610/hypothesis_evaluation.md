# 第 5 步：三项判断的证据评估

## H1：download_okx_history.py 含有"调用 DEMO 接口"的代码路径 —— **成立**

- 调用条件：`--source auto|gateway`（auto 为默认）且 `.env OKX_SERVER=DEMO`（或 `--server DEMO`）。
- 具体路径：
  - `scripts/download_okx_history.py:587-607` 读 OKX_SERVER → `:698` 写入网关 connect 字典 → `:865` `main_engine.connect()`
  - `scripts/download_okx_history.py:1097` `main_engine.query_history()` → vnpy_okx
  - `vnpy_okx/okx_gateway.py:585-586` Server=DEMO → `self.simulated=True`
  - `vnpy_okx/okx_gateway.py:515-520, 551-552` 所有 REST 请求加 `x-simulated-trading: 1`
  - `vnpy_okx/okx_gateway.py:999-1004` history-candles 请求同受该 header 影响
- 本仓库 `.env` 第 4 行实际值 `OKX_SERVER=DEMO`（mtime 2026-04-17，早于全部下载会话，此后未改）。
- 修正原诊断推测：**不存在** "gap 修复时切 DEMO" 的路径；DEMO 是**主路径**，mainnet REST 只是从未触发的回退。

## H2：该代码路径历史上确实被触发过 —— **成立**

- `logs/download_okx_history.log` + `logs/download_okx_1m_*.log` 共 29 条 `okx.connect` 事件，
  connect_setting 中 `"Server": "DEMO"` **29/29，REAL 0 次**（时间跨度 2026-04-17 → 2026-05-29，覆盖 5 个币种）。
- 28 份 manifest、~2090 个 chunk：`source_used` 全部为 `"gateway"`（少量 `skipped_existing` 无源标记，
  系数据先前已存在被跳过）；`"rest"` 0 个，`"mixed"` 0 个。
- 全部日志中 `history.fallback_rest` 0 次 → mainnet REST 备胎从未执行。
- 量级：仅 2023-2024 全区间 manifest 就是每币 244 chunks / 1,052,640 bars 全部 gateway；
  数据库 5 币各 1,791,360 行（2023-01-01 → 2026-05-28），区间与 manifest 拼接吻合。
- 即：**当前 database.db 中可追溯的全部 1m 数据都经由 gateway-DEMO 会话写入**（后写覆盖先写，
  2026-05-07~09 的全量重下覆盖了更早的任何数据）。

## H3：触发该路径产生的数据与 598 事件吻合 —— **成立（基于 4 个抽样 + BTC 反向证据）**

抽样反向追溯（详见 sample_traceback/）：

| 样本 | 事件 | DB 复核 | 溯源 chunk | source_used | 会话日志 |
|---|---|---|---|---|---|
| 1 | SOL 2023-01-02 05:50 (+19.5%) | 复现：9.75→13.71 阶梯，同窗 BTC/ETH/LINK/DOGE 波动 ≤0.24% | SOL 2023-2024 manifest idx=1，saved_at 2026-05-09T14:28:30Z | gateway | SOL DEMO connect 2026-05-09T14:27:59Z（差 31 秒）✓ |
| 2 | SOL 2025-05-29 15:51 (±50%) | 复现：172→259→90→172 完美往返，BTC/ETH/LINK ≤1.5%（DOGE 同窗也异常——与 DOGE 自身事件一致） | SOL 2025-2026 manifest idx=50，saved_at 2026-05-07T12:56:43Z | gateway | SOL DEMO connect 2026-05-07T12:28:35Z ✓ |
| 3 | DOGE 2024-07-25 11:41 (+86.4%) | 复现：0.086→0.16 未回落 | DOGE 2023-2024 manifest idx=191，saved_at 2026-05-09T19:07:15Z | gateway | DOGE DEMO connect 2026-05-09T18:19:57Z ✓ |
| 4 | ETH 2026-01-01 05:09 (+51.2%) | 复现：2802→3876→2453，其余 4 币 ≤1.8% | 两个 chunk 覆盖：saved_at 2026-05-07T12:15:03Z 与 2026-05-27T15:58:47Z | gateway×2 | 05-07 会话为 DEMO ✓；**05-27 会话无本地日志**（见局限性） |
| 0 | BTC | 598 事件中 BTC = 0 | —— | —— | 反向证据：BTC 同样全程 gateway-DEMO 下载，但无 ramp——说明污染特征来自 **demo 环境对各币种行情保真度差异**（BTC demo 行情紧贴真实盘，SOL/DOGE demo 盘存在合成 ramp），而非下载器对个别币种做了特殊处理 |

吻合度：598 事件分布 SOL 344 / DOGE 216 / ETH 23 / LINK 15 / BTC 0，与"全库同为 DEMO 来源、
demo 环境按标的流动性保真度递减"的机制一致；抽样 4/4 溯源到 gateway-DEMO chunk，0 个溯源到 REST/mainnet。

## 5(d) 综合结论：H1 ∧ H2 ∧ H3 = **成立 → 确认 download_okx_history.py 是污染写入通道**

机制修正：污染不是"gap 修复混入 DEMO 片段"，而是**整个主下载路径就是 DEMO 行情**——
当前 database.db 的全部 1m 历史均来自 OKX 模拟盘环境。598 个 ramp 事件是 demo 行情与真实行情
偏离最剧烈、可被数学特征检出的子集；**全库与 mainnet 的逐 bar 一致性未经验证**（本任务禁止下载，未验证）。
