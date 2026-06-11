# 数据污染源取证报告 —— download_okx_history.py

日期：2026-06-10（UTC）
性质：纯只读取证。未修改任何现有代码/数据，未调用 OKX，未触碰 VPS。
新增文件仅限本报告目录 + 只读分析脚本 `scripts/forensics_data_contamination.py`。

## TL;DR

**确认：download_okx_history.py 是污染数据的写入通道，但机制与原推测不同。**
不是"gap 修复路径偶发混入 DEMO 数据"——而是 `.env` 中 `OKX_SERVER=DEMO`，导致脚本的
**主下载路径（vnpy_okx gateway）全程带 `x-simulated-trading: 1` 请求 OKX 模拟盘行情**，
当前 `database.db` 中 5 币 × 1,791,360 根 1m K 线**全部**经此路径写入。598 个合成 ramp
事件只是 demo 行情与真实行情偏离最剧烈、可被检出的子集。

## Q1：download_okx_history.py 是污染源吗？

**是**（作为污染的写入通道；污染数据的产生方是 OKX DEMO 环境行情）。
H1 / H2 / H3 三项判断全部成立，证据见 `hypothesis_evaluation.md`。

## Q2：污染机制

1. `.env` 第 4 行：`OKX_SERVER=DEMO`（mtime 2026-04-17，早于全部下载会话，期间未改动；
   `.env.example` 模板默认值也是 DEMO）。
2. `download_okx_history.py` 默认 `--source auto`：**gateway 优先**。
   `read_env_config`（:587-607）读 OKX_SERVER → `:698` 注入网关 connect 配置 → `:1097`
   `main_engine.query_history()`。
3. vnpy_okx（`okx_gateway.py:585-586`）Server=DEMO → `simulated=True`；
   `:515-520, 551-552` 给**所有** REST 请求（含 `/api/v5/market/history-candles`，:999-1004）
   追加 `x-simulated-trading: 1` → OKX 返回**模拟盘环境**K线。
4. 返回数据经 `save_bars_to_database` 写入 `database.db`，**库表无来源标签**，manifest 只记
   `source_used=gateway` 不记 DEMO/REAL（server 值仅存在于运行日志的 connect 事件中）。
5. 脚本自带的 mainnet 公共 REST 备胎（`request_okx_history_page`，恒 mainnet、无 demo header）
   **从未触发**：全部日志 `history.fallback_rest` 0 次，全部 manifest `source_used="rest"` 0 个。
   —— 讽刺的是，唯一干净的代码路径恰恰是从没跑过的那条。
6. 脚本**没有任何**本地合成/插值/外推代码；`--repair-missing` 修复路径只是用同样的源重新下载。
   ramp 的"等差步长/完美往返"等数学特征是 **OKX demo 行情本身**的产物（demo 对低流动性标的
   的行情保真度差：SOL 344 / DOGE 216 / ETH 23 / LINK 15 / BTC 0 的分布与此一致；BTC demo
   行情紧贴真实盘所以干净）。
7. 另两个写库脚本 `download_history.py`（:66）与 `merge_recent.py`（:38）**硬编码**
   `x-simulated-trading: "1"`——增量数据同源污染。本仓库不存在任何确定走 mainnet 的写库路径。

## Q3：其他候选污染源

主通道已确认，其余候选评估（详见 code_review.md 附录）：

| 候选 | 可疑度 | 依据 |
|---|---|---|
| `download_history.py` | **确认同源（DEMO）** | 硬编码 demo header + `save_bar_data` 写同一库 |
| `merge_recent.py` | **确认同源（DEMO）** | 硬编码 demo header + 裸 INSERT 写 dbbardata |
| `init_db.py` | 中（无法溯源） | 从 `/tmp/vnpy_src.db`（上传文件）整库导入；但 2026-05-07~09 的全量 gateway 重下已覆盖其影响范围 |
| `import_okx_funding_csv.py` | 排除 | 不写 dbbardata |
| 外部工具手动写库 | 无法完全排除 | 无痕迹；且无此必要——已知通道足以解释全库特征 |

## Q4：现有证据是否足以决定下一步行动？

**足以。** 事实依据：
- 全库 100% 1m 数据来自 DEMO 环境（29/29 connect 均 DEMO、~2090 chunks 均 gateway、
  后写覆盖先写的全量重下发生在 2026-05-07~09）。
- 因此问题不是"剔除 598 个事件窗口"能解决的——**全库与 mainnet 的逐 bar 偏差程度未知**。
- 顺理成章的下一步（本任务不执行）：用 mainnet 来源重建数据（download_okx_history.py 本身
  的 `--source rest` 即恒 mainnet，可直接复用；或 `--server REAL` + 修正 .env），重下后与
  现库对比逐 bar 偏差，再决定历史回测结论中哪些需要重检。
- 同时 `download_history.py` / `merge_recent.py` 的硬编码 demo header 需要修（本次未动）。

## Q5：本次取证的局限性（诚实记录）

1. **无网络层证据**：没有下载时的抓包，无法直接观测 OKX 对带 demo header 的请求返回了什么。
   "header → demo 环境行情"的链条由 vnpy_okx 源码 + OKX API 语义 + 数据特征间接支撑。
2. **一次会话无本地日志**：抽样 4（ETH）的第二个覆盖 chunk saved_at=2026-05-27T15:58:47Z，
   本地日志无对应 connect 事件（本地日志最近会话为 05-29）。manifest 是 git 跟踪文件且该
   manifest 唯一提交在 2026-05-29（"update"），**可能由另一台机器（如 VPS）执行后经 git 同步**，
   该会话的 server 值未经日志直接验证。但同窗口另一 chunk（05-07 会话）已验证为 DEMO。
3. **demo 行情与真实行情的总体偏差未量化**：598 事件之外的"安静时段" demo 数据多大程度镜像
   mainnet，需要重新下载 mainnet 数据对比才能回答（本任务禁止下载，未做）。
4. **manifest 无 DEMO/REAL 字段**：事后溯源完全依赖运行日志恰好明文记录了 Server 字段；
   若日志被轮转/删除，将无法溯源——这是流程缺陷本身的证据（3b 判定属实，但本次被日志补救）。
5. 无法完全排除历史上存在未留痕的手动写库；但已知三条写库路径已足以解释全部观测特征，
   无需引入额外假设。

## 产物清单

- `README.md`（本文件）
- `code_review.md` —— 第 0/1/2 步代码审查逐项记录
- `manifest_audit.json` —— 第 3 步：28 份 manifest × source_used 统计
- `sample_traceback/` —— 第 4 步：5 个抽样（含 BTC 反向证据）的 DB 复核 + manifest/日志溯源
- `hypothesis_evaluation.md` —— 第 5 步：H1/H2/H3 评估
- `run_log.txt` —— 时间戳
- 分析脚本：`scripts/forensics_data_contamination.py`（只读）

## 未改动文档及原因

- `PROJECT_GUIDE.md` / `CLAUDE.md`：本次为取证任务而非研究结论产出；且"全库 DEMO 来源"的
  策略层影响要等 mainnet 对比量化后才能写成准确叙述（现在写会把"偏差程度未知"误写成定论）。
  待 mainnet 重下对比完成后，应一并更新"已验证的核心事实/已知未验证的假设"两节并修订
  数据来源叙述。
- `reports/regime/v2b_dd_diagnosis_20260610/README.md`：其"下载器存在 DEMO/gap 修复路径"
  的推测已被本报告修正（机制是整库 DEMO 而非 gap 混入），但按只读原则不改原报告，
  以本报告为准。
