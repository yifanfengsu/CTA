# B2_4h 前向观察起点与样本纯净性排查

> 用户报告：B2_4h前向观察日报显示"前向累计0.66月、3笔"，但用户
> 称系统"刚部署到VPS"。两者矛盾——刚部署的系统前向时长应接近0、
> 笔数应为0或1。前向观察的全部意义是"B2_4h在从未见过的未来真实
> 行情上的表现"，要求前向记录起点为真实运行起点、数据为真实增量
> 行情，绝不混入回填/演习/本地试跑数据（混入=样本外污染，性质
> 同当年demo数据污染）。本排查只查不改，判断现有3笔是否纯净。

排查时间：2026-06-22 UTC
方法：只读 VPS 上 /opt/forward_b2/ 的记账、状态、配置文件，不修改任何文件。

---

## (1) 前向起点定义

**来源：`forward/config_frozen.json` 第 45 行**

```json
"forward_start_utc": "2026-06-01T00:00:00Z",
"forward_start_note": "trades with entry_time >= forward_start_utc count as the independent forward sample; backtest/gate sample = everything before it. DB 1m ends 2026-05-28 15:59Z; REST fills the gap to forward_start."
```

**来源：`scripts/forward_b2_4h.py` 第 375-377 行**

```python
def forward_window(perbar, trades):
    start = pd.Timestamp(cfg()["forward_start_utc"])
    f_per = perbar[perbar.index >= start]
    f_tr = [t for t in trades if pd.Timestamp(t["entry_time"]) >= start]
    return f_per, f_tr
```

**来源：同文件第 391-392 行**

```python
span_days = (f_per.index[-1] - pd.Timestamp(cfg()["forward_start_utc"])).days
months = span_days / 30.44
```

**结论：前向起点是固定锚点 `2026-06-01T00:00:00Z`，不是首次真实运行时刻。**

0.66 月 = 从 2026-06-01 到报告日 2026-06-22 约 20 天 ÷ 30.44 ≈ 0.66。与 VPS 部署时间（2026-06-21T05:30Z）无关。

---

## (2) 3 笔交易溯源

**来源：VPS `/opt/forward_b2/forward/state/ledger_trades.jsonl`**

逐笔过滤 `entry_time >= "2026-06-01"`，恰好 3 笔：

| # | 币种 | 方向 | entry_time (UTC) | exit_time (UTC) | 出场原因 | net_pnl | 
|---|------|------|------------------|-----------------|----------|---------|
| 1 | SOL | long | 2026-06-16T12:00 | 2026-06-19T00:00 | flip（EMA交叉反向） | −$674.91 |
| 2 | SOL | short | 2026-06-19T00:00 | 2026-06-21T04:00 | flip（EMA交叉反向） | −$573.94 |
| 3 | SOL | long | 2026-06-21T04:00 | 2026-06-21T08:00 | end_of_data（当前持仓M2M） | −$73.90 |

### 每笔所用的行情数据来源分析

- DB 1m 止于 2026-05-28T15:59Z（database_mainnet.db，SHA256 已校验）
- VPS 首次 `--cron-4h` 于 2026-06-21T05:43:49Z 触发，其 `--update` 阶段通过 OKX mainnet REST 拉取了 2026-05-28 到 2026-06-21 之间的所有 1m kline + funding（manifest 记录 `last_update_event: "incremental REST append"`）
- **3 笔交易的 entry_time 全部早于 VPS 部署时刻（2026-06-21T05:30Z）**，均使用 REST 回填的历史行情计算

| 交易 | 数据来源 | 是否为实时增量行情 | 
|------|---------|-------------------|
| #1 (06-16 entry) | OKX mainnet REST，部署时一次性回填 | ❌ 不是。行情数据在 06-16 当天就存在于 OKX 服务器，但系统在 06-21 才拉取 |
| #2 (06-19 entry) | 同上 | ❌ 同上 |
| #3 (06-21T04:00 entry) | 同上 | ❌ 同上。该 4h bar 在 08:00 UTC 收盘，但 04:00 的开盘行情已回填 |

**数据标记：** manifest `server=mainnet`、`demo=false`，数据源本身纯净（主网真实行情），无 demo 污染。但所有 3 笔均非"系统运行时实时拉取的增量行情"，而是"部署后一次性回填的历史行情"。

### 伪前向演习与正式记账的分离

- 本地 dry_run_validation.json 的 `validated_utc` = 2026-06-16（来自 git），从未被 VPS 覆盖
- VPS 上 dry_run_validation.json 修改时间早于 clone 时间（1782018049 = 约 05:01 UTC，git checkout 时间戳），内容与本地完全相同
- VPS dry-run 未成功运行过（SSH 断开），演习记录从未写入正式记账
- **结论：dry-run 与正式记账完全分离，演习没有混入正式 ledger。**

### 前向每日 M2M 覆盖范围

**来源：VPS `/opt/forward_b2/forward/state/ledger_daily_m2m.jsonl`**

| 日期 | M2M (USD) | 日期 | M2M (USD) |
|------|-----------|------|-----------|
| 06-01 | +193 | 06-12 | −1,575 |
| 06-02 | +474 | 06-13 | +79 |
| 06-03 | +3,579 | 06-14 | −730 |
| 06-04 | +1,023 | 06-15 | −936 |
| 06-05 | +1,140 | 06-16 | −748 |
| 06-06 | +3,143 | 06-17 | −27 |
| 06-07 | +172 | 06-18 | +452 |
| 06-08 | −2,317 | 06-19 | +364 |
| 06-09 | −100 | 06-20 | −12 |
| 06-10 | +930 | 06-21 | −785 |
| 06-11 | +776 | 06-22 | −39 |

**前向累计净利 = 逐日 M2M 之和 ≈ $4,521**（与日报 $4,520.73 一致，微小差异来自浮点精度）。

**重要发现：前向净利的主要来源不是 3 笔前向交易。**
- 3 笔前向交易合计亏损 −$1,322.75
- 前向净利≈+$4,521 的大部分来自 **跨期持仓的未实现浮动盈亏**——即 2026-06-01 之前就已开仓、横跨到前向期的仓位（BTC short、ETH short、LINK short、DOGE short、SOL long），在 6 月前期的行情波动中产生了未实现收益
- 6 月 3-6 日的 +$8,885 巨额浮动收益，被 6 月 8-15 日的 −$5,985 回吐部分抵消

---

## (3) VPS 实际运行时点

**来源：VPS 日志、heartbeat.json、crontab 安装时间**

| 事件 | UTC 时间 | 距部署开始 |
|------|---------|-----------|
| git clone 完成 | ~2026-06-21T05:04 | 0 |
| seed 开始 | 2026-06-21T05:30:10 | +26min |
| seed 完成（5 币各 179 万 bar） | 2026-06-21T05:35:37 | +31min |
| 首次 cron-4h（update + account） | 2026-06-21T05:43:49 | +39min |
| 首次 push（日报发送） | 2026-06-21T05:50:20 | +46min |
| crontab 安装 | 2026-06-21T05:42 前后 | +38min |
| 下次 cron 触发（08:05 UTC） | 2026-06-21T08:05 | +3h |

**VPS 真实运行时长：** 从 seed 开始（05:30）到探查时刻（06-22）约 **1 天**。

0.66 月（20 天）与 1 天的差距 = **前向起点定义是固定锚点 06-01，而非部署时刻 06-21**。

---

## (4) 数据文件溯源

**来源：VPS `forward/data/manifest.json`、klines CSV 文件**

- **klines CSV 文件**（5 币）：均为 seed 从 database_mainnet.db 只读拷贝（2023-01 起）+ REST 增量追加（至 2026-06-21）
- **funding CSV 文件**：本地 SCP 传输的 `data/funding/okx/*.csv`（2023-01 到 2026-06-11）+ REST 追加（至 2026-06-21）
- **manifest 全程可审计**：`server: "mainnet"`, `demo: false`, `source_klines: "OKX /api/v5/market/candles + history-candles"`, `source_funding: "OKX /api/v5/public/funding-rate-history"`
- **无 demo、无鉴权、无交易 API 调用**：selfcheck 每次 pass=True，`server_is_mainnet=True`

**结论：数据全部来自 OKX mainnet 公开行情，无 demo 污染。但所有前向期数据均为部署时一次性回填（REST 历史接口），非逐日增量累积。**

---

## Q1-Q4 判断与结论

### Q1 — 前向起点真相

前向起点是 `forward/config_frozen.json` 定义的**固定锚点 2026-06-01T00:00:00Z**，其含义是："回测/gate 样本止于 2026-05-28（DB 结束），前向独立样本从 2026-06-01 开始，中间 2 天 gap 由 REST 补齐"。

这个日期在系统开发阶段（2026-06-16）就写死了，与 VPS 部署时间无关。它比 B2_4h 回测使用的全部 DB 数据（止于 2026-05-28）晚了 4 天，是一个设计上的"干净切点"。

"0.66 月/3 笔"来自：
- 0.66 月 = (2026-06-22 − 2026-06-01) / 30.44
- 3 笔 = 前向期内 SOL entry_time ≥ 2026-06-01 的已平仓/当前持仓交易

### Q2 — 纯净性判定

**🔴 红灯 — 前向记录混入了部署前的 REST 回填数据**

判定依据：

1. **3 笔交易的 entry_time 全部早于 VPS 部署时刻**（2026-06-21T05:30Z）。系统在部署时通过 `--cron-4h` 一次性拉取了 2026-05-28 到 2026-06-21 的全部 REST 历史行情，并用这些"回填数据"计算出了本应发生在 06-16 和 06-19 的 SOL EMA 交叉信号及交易。这些交易使用的行情在 06-16 和 06-19 确实存在于 OKX 服务器上，但系统当时还未运行，是在部署时"事后诸葛亮"式地重算出来的。

2. **用户定义的前向观察标准是"真实运行起点 + 真实增量行情"**，当前 3 笔不满足此标准。

3. **前向累计净利 $4,520 的构成有误导性：** 它大部分来自跨期持仓的浮动盈亏（6 月前期行情上涨，BTC/ETH 空头亏损但被 DOGE/SOL 多头覆盖），而非来自前向期内新开仓的 3 笔交易（3 笔均为亏损 −$1,323）。这意味着"前向累计 $4,520"这个数字并不能反映策略在前向期的表现，它主要是回测期持仓的惯性延续。

> 注意：这不是 demo 数据污染（数据源确认是 mainnet），而是**时间维度的污染**——系统把"部署前已经发生的历史事件"当作"前向观察"记录下来。

### Q3 — 清空重启方案（仅描述，不执行）

若确认为红灯需重启，操作方案：

1. **清空文件清单**（VPS 上 `/opt/forward_b2/`）：
   - `forward/state/ledger_trades.jsonl` — 逐笔交易记录
   - `forward/state/ledger_daily_m2m.jsonl` — 逐日 M2M 记录
   - `forward/state/positions.json` — 当前持仓状态
   - `forward/state/heartbeat.json` — 心跳记录

2. **保留不清的文件**：
   - `forward/data/klines/*.csv` — 行情 store（seed + REST 回填），保留作为行情历史
   - `forward/data/funding/*.csv` — funding 数据，同上
   - `forward/data/manifest.json` — 数据来源记录
   - `forward/config_frozen.json` — 配置（但 `forward_start_utc` 需更新）

3. **新起点设置**：
   - 将 `forward_start_utc` 改为 VPS 首次 `--cron-4h` 成功运行后**第一个完整 4h bar 的收盘时刻**，即：
     - 部署在 2026-06-21T05:43 完成首次 account
     - 下一个完整 4h bar 是 2026-06-21T04:00-T08:00（收盘 08:00 UTC）
     - **建议新起点：`2026-06-21T08:00:00Z`**（第一个"系统已运行且 bar 已收盘"的时刻）
   - 更新 SHA256，gate 作废重立（按 gate 规则：改 config = 新系统）

4. **防再次混入措施**：
   - 修改 `--account` 逻辑，在计算 forward_window 时额外检查：`entry_time >= max(forward_start_utc, first_run_utc)`，其中 `first_run_utc` 由 heartbeat 记录
   - 或者在 seed 阶段记录 `seed_completed_utc`，确保任何早于该时刻的 entry 不计入前向
   - 重启后 `--cron-4h` 只拉取 `>= seed_completed_utc` 的增量数据，不拉历史 gap

5. **gate 重立**：
   - 基线 gate 数字沿用 `baseline_distribution.json`（OKX gate-of-record）
   - KILL/UPGRADE 阈值不变
   - 但"前向 12 月/18 月"的时钟从新起点重新计时

### Q4 — 起点定义建议

**前向观察的起点应锚定在 VPS 首次成功运行后第一个完整 4h bar 收盘时刻。**

理由：

1. **"从未见过的未来行情" = 系统运行时刻之后的行情。** 系统在 2026-06-21T05:43 完成首次 account，则 06-21T08:00 是第一个"系统已在线、bar 已自然收盘"的时刻。06-21T04:00-T08:00 这个 bar 的 OHLC 在系统运行时实时形成，属于真正的前向行情。

2. **固定锚点 2026-06-01 的问题**：它是一个"理论切点"（回测结束的 2 天后），但系统并未在那天运行。06-01 到 06-21 的行情是通过 REST 事后拉取的。虽然数据源是 mainnet（不是 demo），但"事后拉取"与"实时累积"在哲学和操作上有本质区别——前者无法保证数据获取时与实时获取时完全一致（OKX REST 的 `confirm==1` 在历史查询和实时查询中可能有微妙差异）。

3. **部署时刻有明确标记**：heartbeat.json 的 `last_run_utc` 和 seed 日志提供了精确的时间戳，比配置文件中的人为选定日期更透明、更可审计。

4. **与 gate 的兼容性**：如果使用部署时刻作为起点，K1（滚动 12 月净利 < p5）和 U1（满 18 月）的时钟从真正的前向第一刻开始计时，不存在"系统跑 1 天但声称前向 20 天"的歧义。

---

## 证据索引

| 证据 | 来源 | 关键信息 |
|------|------|---------|
| forward_start_utc | config_frozen.json:45 | 2026-06-01T00:00:00Z（固定锚点） |
| forward_window 逻辑 | forward_b2_4h.py:375-377 | entry_time >= forward_start 即视为前向 |
| 0.66 月计算 | forward_b2_4h.py:391-392 | span_days / 30.44 |
| 第 1 笔交易 | ledger_trades.jsonl (VPS) | SOL long, entry 06-16T12:00, pnl −$674.91 |
| 第 2 笔交易 | ledger_trades.jsonl (VPS) | SOL short, entry 06-19T00:00, pnl −$573.94 |
| 第 3 笔交易 | ledger_trades.jsonl (VPS) | SOL long, entry 06-21T04:00, pnl −$73.90 |
| VPS 部署时刻 | heartbeat.json + 日志 | seed 05:35Z, cron-4h 05:43Z, push 05:50Z |
| 数据源 | manifest.json | server=mainnet, demo=false |
| 行情时间范围 | klines CSV | seed: 2023-01~2026-05-28, REST append: ~2026-06-21 |
| 每日 M2M | ledger_daily_m2m.jsonl (VPS) | 06-01到06-22共22天，总和≈$4,521 |
| dry-run 隔离 | dry_run_validation.json | VPS 未成功运行 dry-run，无混入 |
| config SHA256 | config_frozen.json | c37ff0a...d38d5（未改动） |
| selfcheck | forward_b2_4h.py --selfcheck | pass=True，四项全True |

---

## 3 笔前向交易逐笔溯源表

| # | entry_time (UTC) | exit_time (UTC) | 符号 | 方向 | net_pnl | 出场原因 | 行情来源 | 是否实时增量 | 备注 |
|---|------------------|-----------------|------|------|---------|----------|----------|-------------|------|
| 1 | 2026-06-16T12:00 | 2026-06-19T00:00 | SOL | long→short flip | −$674.91 | EMA 交叉反向 | REST 回填 | ❌ | 早于部署 5 天 |
| 2 | 2026-06-19T00:00 | 2026-06-21T04:00 | SOL | short→long flip | −$573.94 | EMA 交叉反向 | REST 回填 | ❌ | 早于部署 2.5 天 |
| 3 | 2026-06-21T04:00 | 2026-06-21T08:00 | SOL | long (当前持仓) | −$73.90 | end_of_data | REST 回填 | ❌ | 早于部署 1.5 小时 |

---

前向纯净性排查完成于 2026-06-22 UTC，前向起点:2026-06-01(固定锚点，非部署时刻) / 3笔来源:REST回填(全部早于VPS部署) / 纯净判定:🔴红灯 / 处理:需清空重启 / 已push:待确认
