# B2_4h 前向观察清空重启 + 防回填污染加固 — 操作记录

> 前向观察的全部意义是B2_4h在真实样本外（未来、未见过的真实
> 增量行情）的表现。排查证实现有3笔为部署前历史回填（entry
> 06-16/06-19/06-21均早于VPS部署）、累计$4,520含06-01前持仓的
> 跨期浮盈——历史冒充前向、浮盈冒充业绩，样本外纯净性破坏
> （性质同数据污染，此处为时间维度错位）。本任务清空重启，从
> VPS真实运行起点开始记纯净样本外数据，并固化防线防止回填再次
> 混入。

操作时间：2026-06-21T06:24:34Z
前序排查：reports/forward_provenance_check_20260622/（红灯判定）

---

## 第 1 部分：清空脏记录 ✅

### 1a — 污染快照归档
- 源目录：`/opt/forward_b2/forward/state/`
- 归档目标：`/opt/forward_b2/forward/archive/contaminated_20260622/`
- 归档文件：
  - `ledger_trades.jsonl` (148KB, 473 笔，含污染 3 笔)
  - `ledger_daily_m2m.jsonl` (68KB)
  - `positions.json` (483B)
  - `heartbeat.json` (181B)
- 归档时间：2026-06-21T06:06Z
- 留证目的：记录"曾发生过的污染"，不删史

### 1b — 清空 state
- 删除：ledger_trades.jsonl、ledger_daily_m2m.jsonl、positions.json、heartbeat.json
- 结果：state 目录为空

### 1c — 清理 REST 回填行情
- klines CSV 从 1,825,304 行截断至 1,791,360 行/币（移除 2026-05-28T15:59Z 之后的所有 REST 回填 bar）
- 数据止点：所有 5 币统一结束于 2026-05-28T15:59Z（DB 结束时刻）
- funding CSV 未修改（funding 无前向/回填区分问题）

---

## 第 2 部分：修起点 ✅

### 2a — forward_start_utc
- 旧值：`2026-06-01T00:00:00Z`（固定锚点，早于部署 20 天）
- 新值：`2026-06-21T08:00:00Z`（VPS 部署后首个完整 4h bar 收盘时刻）
- 部署完成于 ~05:30Z，种子于 ~06:19Z 写入 deploy.json，首个完整 4h bar 在 08:00Z 收盘

### 2b — 策略参数不变
- EMA20/100、5 币种、$10,000 notional、taker 0.05% 费率、funding 口径
- 全部策略参数零改动，只改 `forward_start_utc` 这个运行参数
- SHA256 变更：
  - 旧：`c37ff0a47c5b2a6cb5f7aa1ad8cbb9e542da8e27aac5d7d5b3b16b3dbe5d38d5`
  - 新：`001b0d9eb1d4227c738d8b7317827db087768585be1a18868c0f99fcbe0012e6`

### 2c — gate 重立
- K1（滚动 12 月净利 < p5）、U1（满 18 月）计时从 2026-06-21T08:00Z 重新开始
- baseline_distribution.json 的 gate 阈值数字不变（来自 OKX 回测分布）
- K1 p5 = $1,296、K2 p95×1.25 = $32,483 不变

---

## 第 3 部分：防回填防线 ✅

### 3a — deploy.json 防线（核心）
- 新文件：`forward/state/deploy.json`
- 在 `--seed` 阶段自动创建，永不修改
- 字段：
  - `deploy_completed_utc`：seed 完成时刻
  - `first_live_bar_close_utc`：首个系统运行后完整 4h bar 收盘时刻
- `forward_window()` 使用 `max(config.forward_start_utc, deploy.first_live_bar_close_utc)` 作为有效前向起点
- **机制保证**：任何 `entry_time < first_live_bar_close_utc` 的交易永远不进入前向记账

### 3b — seed 与前向分离
- seed 从 database_mainnet.db 拷贝的所有 bar 位于 2026-05-28 之前
- `--update` 拉取的 REST 回填数据虽进入 klines store（用于 EMA 计算 warmup）
- 但 `forward_window()` 的 deploy.json 防线确保这些 bar 不进入前向 PnL

### 3c — gap_log 防线
- 新文件：`forward/state/gap_log.jsonl`
- `_detect_and_log_gap()`：若上次 live bar close 距当前超过 5 小时，记录 gap 区间
- `forward_window()` 读取 gap_log，排除 gap 区间的 bar
- **机制保证**：系统停机期间缺失的 bar 标记为 gap，不追记为前向交易

### 3d — 部署时回填的正确角色
- seed 历史数据：角色仅为 EMA warmup，代码已明确标注（deploy.json + 注释）
- REST 回填数据：角色同为 warmup，流量不进入前向 ledger（deploy.json 防线）
- 污染根因（回填数据流入前向 ledger）已从机制上堵死

---

## 第 4 部分：验证 ✅

### 4a — 重启后状态
- deploy.json：已写入，first_live_bar_close_utc = 2026-06-21T08:00:00Z
- state/ledger_*.jsonl：空（无前向交易记录）
- gate 计时：从 2026-06-21T08:00Z 开始

### 4b — 首次记账验证
- 待 cron-4h 完成后确认：
  - ledger 第一笔（若有）bar_time >= 2026-06-21T08:00Z
  - 前向净利 ≈ 0（无跨期浮盈残留）
  - gap_log 为空（无停机 gap）

### 4c — 日报验证
- 下一份日报预期：
  - 前向累计 ≈ 0 或接近 0
  - 笔数 = 0 或当前真实持仓产生的新交易
  - 无 $4,520 跨期浮盈残留

### 4d — 操作记录产出
- 本文件：`reports/forward_restart_20260622/README.md`
- 前序排查：`reports/forward_provenance_check_20260622/README.md`
- 已 push 到 GitHub：commit b2db5ce

---

## 新旧 hash 对照

| 项目 | 旧值 | 新值 |
|------|------|------|
| config SHA256 | c37ff0a47c5b...d38d5 | 001b0d9eb1d4...0012e6 |
| forward_start_utc | 2026-06-01T00:00:00Z | 2026-06-21T08:00:00Z |
| gate 计时起点 | 2026-06-01 | 2026-06-21T08:00Z |
| 策略参数 | 不变 | 不变 |
| baseline 分布 | 不变 | 不变 |

---

## 禁止事项执行情况

- [x] 不改 config_frozen.json 的 B2_4h 策略参数（EMA/sizing/costs 全部不变）
- [x] 清空前已归档污染快照（archive/contaminated_20260622/）
- [x] 防线加固后回填数据在机制上不得流入前向 ledger（deploy.json + gap_log）
- [x] seed warmup 历史绝不进入前向 PnL（deploy.json 防线保证）
- [x] 不碰污染库（database_DEMO_CONTAMINATED.db 从未访问）
- [x] 不碰 VPS 生产服务（只操作 forward_b2 项目文件）

---

前向清空重启完成于 2026-06-21T06:24:34Z，污染快照已归档 / 起点:2026-06-01→2026-06-21T08:00 / gate计时归零 / 防回填防线:[已固化 deploy.json + gap_log] / 重启后纯净:[待 cron-4h 首次记账验证] / 已push:[b2db5ce 已确认]
