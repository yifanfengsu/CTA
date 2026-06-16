# B2_4h 前向观察系统（阶段 1：零成本模拟记账 + 每日 PushPlus）

**config_frozen.json SHA256:**
`c37ff0a47c5b2a6cb5f7aa1ad8cbb9e542da8e27aac5d7d5b3b16b3dbe5d38d5`
（此后任何改动 = 新系统、新 hash；gate 作废重立。）

---

## 定位声明（全文不删改）

> 本系统对 B2_4h（4h EMA20/100 交叉, long/short, always-in-market, 信号反向出场, 参数永久冻结）
> 做零成本前向观察：拉 OKX 真实 mainnet 公开行情 + 本地模拟成交记账，**不触碰任何真实或模拟
> 账户、不读任何 API 凭证、不下任何单、绝不使用 OKX demo**（demo 撮合失真是 MR-5m 事故根源）。
> B2_4h 历史 Sharpe~0.5 系在已知数据上验证；前向样本与全部历史独立，是当前唯一能增加证据
> 等级的途径。这是三阶段验证的阶段 1（模拟记账 → 满 18 月看 UPGRADE → 阶段 2 小额真实 mainnet
> 真金 → 阶段 3 正式资金），阶段 1 未过不上真金。B2_4h 经四次增强（funding/ADX/faster-entry/V1）
> 双样本全判死，确认当前形态即最优——本系统部署原样，零优化/过滤/参数变体。
> 运行架构：开发由 Claude 在本地完成；部署由用户手动在 VPS 执行；运行由 VPS 的 cron 自主触发
> （无 AI 参与）。三角色分离。

## 三角色分离

| 角色 | 谁 | 做什么 | 不做什么 |
|------|----|--------|----------|
| 开发 | Claude（本地 WSL） | 写脚本/config/gate/推送逻辑、跑伪前向演习、写部署手册、commit+push | **绝不 SSH/操作 VPS、不装远程 cron** |
| 部署 | 用户（手动在 VPS） | 按 `VPS_DEPLOYMENT_MANUAL.md` 逐步执行 clone/venv/cron 安装/核对 | — |
| 运行 | VPS cron（自主） | 每 4h 触发 `--cron-4h`（拉数+记账）、每日触发 `--cron-daily`（推送） | 无 AI 参与、不下单、不碰账户 |

## 架构

```
OKX mainnet 公开 REST (market/candles + history-candles 1m, public/funding-rate-history)
        │  (只读公开行情；无 demo / 无鉴权头 / 无下单)
        ▼
forward/data/  ← append-only 本地 store（1m + funding；独立于主库，主库只读）
        │   seed 一次性从 database_mainnet.db 只读拷贝（让 EMA20/100 带满历史），其后 REST 增量
        ▼
引擎（research_trend_baseline + validation_r2，逐字复用零修改）
   aggregate 4h → EMA20/100 → positions_flip → build_trades + m2m_pnl
        ▼
forward/state/  ← ledger_trades.jsonl（逐笔）+ ledger_daily_m2m.jsonl（逐日 M2M）+ positions/heartbeat
        ▼
每日 PushPlus 日报 + 异常即时告警
```

记账每次 `--account` 做**全量确定性重算**（store 即真相），因此与回测引擎按构造完全一致——
伪前向演习证明逐分钟差额 $0（见 `dry_run_validation.json`）。

## 数据铁律遵守
- 数据源恒为 **OKX mainnet 公开行情**；`forward/data/manifest.json` 的 `server` 字段恒为 `mainnet`、`demo=false`、可审计。
- **绝不使用 OKX demo**（任何端点/header）；**不读任何交易凭证**；**不下任何单**；**不碰任何账户**。
- 主库 `.vntrader/database_mainnet.db` 只读（仅 seed 时只读拷贝）；污染库严禁触碰；不写主库。
- 脚本启动自检（`--selfcheck`，每次运行前自动跑）：config SHA256、server=mainnet、session 无鉴权头、
  源码无 demo/鉴权/下单 token（敏感 token 以拆分字面量内置，故源码 grep 任一返回空 = 安全）。

## gate（详见 `gates_preregistered.md`，数字冻结）
- **FW-KILL**：K1 前向滚动 12 月净利 < $1,296（OKX p5）；K2 前向 maxDD > $32,483（p95×1.25）；K3 对账漂移 7 天未修复。
- **FW-UPGRADE**：U1 满 18 月 ∧ net/maxDD ≥ 1.31；U2 月度胜率 ≥ 35.0%；U3 无 KILL。
- **FW-REVIEW**：每 6 月一份中期报告，无判定动作。
- 前向样本边界 `forward_start_utc = 2026-06-01T00:00:00Z`；之前属回测/gate 样本，独立。

## 每日 PushPlus 日报字段
- **日期(UTC)** + **各币持仓方向**（BTC/ETH/SOL/LINK/DOGE long/short/flat）
- **当日新信号翻转**（币/方向/时刻）
- **当日模拟 PnL(M2M)** + **前向累计净利**（月数/笔数）+ **前向当前回撤**
- **距各 KILL 阈值余量**（K1 滚动 12 月净利距 $1,296；K2 maxDD 距 $32,483；满 12 月才评 K1）
- **系统健康**：1m 数据缺口数、config hash 前缀、数据源声明
- **异常**（KILL 触发 / 数据缺口 / 对账失配）走**独立即时告警**，不混入日报。

## 模式（脚本 `scripts/forward_b2_4h.py`）
| 模式 | 作用 | cron |
|------|------|------|
| `--selfcheck` | 环境/凭证/端点/config-hash 自检（每次运行前自动） | — |
| `--build-baseline` | 生成 `baseline_distribution.json`（gate 数字） | 一次性 |
| `--seed` | 一次性：只读拷贝主库 1m+funding 入 store | 部署时一次 |
| `--update` | 增量拉 mainnet REST 确认 1m + funding 入 store | （含于 cron-4h） |
| `--account` | 全量重算 ledger（逐笔+逐日 M2M）+ 持仓/心跳 | （含于 cron-4h） |
| `--cron-4h` | = update + account | **每 4h** |
| `--cron-daily` / `--push` | 组合并发送每日 PushPlus 日报 | **每日** |
| `--reconcile` | 月度：重拉上月 REST 比对 store + 全量重算，差额须 ~0 | 每月初 |
| `--dry-run` | 部署前 $0 演习（留出 DB 最后一个完整月，逐日喂入比对回测） | 部署前一次 |

## 部署
见 `VPS_DEPLOYMENT_MANUAL.md`（用户手动逐步执行；Claude 不操作 VPS）。

---

B2_4h前向观察系统本地开发完成于 2026-06-16 (UTC)，配置SHA256:c37ff0a47c5b2a6cb5f7aa1ad8cbb9e542da8e27aac5d7d5b3b16b3dbe5d38d5 / gate已预注册冻结(K1净利<$1,296·K2 maxDD>$32,483·U1 net/maxDD≥1.31·U2月胜率≥35%) / 本地伪前向演习:PASS差额$0(逐笔+逐日M2M+留出月全部$0,store逐bar等于DB) / 数据源:mainnet公开行情(无demo/无凭证/无下单) / 每日PushPlus逻辑:已实现 / VPS部署手册:已交付(5a-5g逐步,Claude不操作VPS) / 已push:已确认 a5b23cf
