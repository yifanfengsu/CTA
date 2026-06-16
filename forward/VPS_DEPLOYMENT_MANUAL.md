# B2_4h 前向观察 — VPS 部署手册（用户手动逐步执行）

> **角色边界**：本手册由 Claude 在本地编写；**所有 VPS 操作由你本人手动执行**。Claude 不
> SSH、不远程部署、不远程装 cron。每步给出可直接复制的命令 + 预期输出 + 出错怎么办。
> 全程只用 OKX **mainnet 公开行情**：不下单、不碰账户、不读凭证、绝不用 demo。
>
> 约定：部署目录用 `/opt/forward_b2`（无 root 权限则换 `$HOME/forward_b2`，后续命令同步替换）。
> 本地仓库 SHA256（config 冻结值）应为
> `c37ff0a47c5b2a6cb5f7aa1ad8cbb9e542da8e27aac5d7d5b3b16b3dbe5d38d5`。

---

## 0. 前置
- VPS 已装 `git`、`python3`（≥3.10）、`python3-venv`、`gzip`、`sha256sum`、`cron`。
- 确认 VPS 时区（cron 用）：
  ```bash
  timedatectl | grep "Time zone"
  ```
  预期最好是 `UTC`。**若不是 UTC**，本手册第 6 部分的 crontab 用 `CRON_TZ=UTC` 行强制 UTC，
  无需改系统时区。

---

## 5a. 确认旧环境（MR-5m 已关停，无残留）

MR-5m 实盘脚本已关停归档。部署前确认 VPS 上无其残留进程 / cron / screen：

```bash
ps aux | grep -iE "run_mr_5m_direct|mr_5m|restart_mr5m" | grep -v grep
crontab -l 2>/dev/null | grep -iE "mr_5m|mr5m" 
screen -ls 2>/dev/null | grep -i mr
```
- **预期输出**：三条命令都**无任何行**（空）。
- **若有残留**：先停掉再继续——
  ```bash
  # 进程：kill <PID>     cron：crontab -e 删除对应行     screen：screen -X -S <name> quit
  ```
- 旧 trade_log 归档提示：若旧目录有 `trade_log_5m*.jsonl` / `mr_5m_state.json`，移到
  `~/archive_mr5m/` 留存，不要删（实盘对账历史）：
  ```bash
  mkdir -p ~/archive_mr5m && mv <旧目录>/trade_log_5m*.jsonl <旧目录>/mr_5m_state.json ~/archive_mr5m/ 2>/dev/null; ls -la ~/archive_mr5m/
  ```

---

## 5b. 物理隔离部署（全新目录 + 独立 venv + 最小依赖）

```bash
sudo mkdir -p /opt/forward_b2 && sudo chown $USER:$USER /opt/forward_b2
git clone git@github.com:yifanfengsu/CTA.git /opt/forward_b2
cd /opt/forward_b2
git log --oneline -1          # 预期：看到最新提交（含 forward 系统）
```
建独立 venv，**只装 3 个依赖**（不装 vnpy / 不装任何交易网关库）：
```bash
python3 -m venv /opt/forward_b2/.venv
/opt/forward_b2/.venv/bin/pip install --upgrade pip
/opt/forward_b2/.venv/bin/pip install numpy pandas requests
/opt/forward_b2/.venv/bin/pip list | grep -iE "numpy|pandas|requests"   # 预期：三个都在
/opt/forward_b2/.venv/bin/pip list | grep -iE "vnpy|ccxt|okx" && echo "!! 不该出现交易库" || echo "OK: 无交易/网关库"
```
- **预期**：最后一行打印 `OK: 无交易/网关库`。

**传入 mainnet 行情库（一次性，用于 seed 全历史让 EMA20/100 带满记忆）**：
mainnet DB 体积大、不入 git。从你本地把 gzip 备份传到 VPS 并校验：
```bash
# 在你本地机器执行（把备份传到 VPS）：
scp ~/backups/database_mainnet_20260611.db.gz <user>@<vps>:/opt/forward_b2/.vntrader/
# 回到 VPS：
cd /opt/forward_b2 && mkdir -p .vntrader
gunzip -c .vntrader/database_mainnet_20260611.db.gz > .vntrader/database_mainnet.db
sha256sum .vntrader/database_mainnet.db
# 预期 SHA256 == a6d6928dbdec108f54ebc413ec84344d3e9cde5f4f54dd07b5adec36f573495d
chmod 0444 .vntrader/database_mainnet.db      # 只读，防呆
```
- **若 SHA256 不符**：传输损坏，重传，**不要继续**。
- funding 历史 CSV（`data/funding/okx/*.csv`）已随 git 仓库克隆到位，无需额外传输。

> 注：DB 仅 `--seed` 时只读读取一次；之后前向数据写入 `forward/data/`，DB 不再被读写。

---

## 5c. 凭证隔离核对（脚本已内置自检，这里你再手动核对一次）

```bash
cd /opt/forward_b2
# 1) 部署目录内无任何 .env / 凭证文件
find . -name ".env" -o -name "*.key" -o -name "*credential*" -o -name "*secret*" | grep -v ".git/"
# 2) 源码无 demo / 鉴权 / 下单字符串（敏感 token 在脚本里以拆分字面量内置，故应全部返回空）
grep -rnE "x-simulated-trading|OK-ACCESS-(KEY|SIGN|PASSPHRASE)|/api/v5/trade/order|/api/v5/account/" scripts/forward_b2_4h.py
# 3) 脚本内置启动自检
/opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --selfcheck
```
- **预期**：(1)(2) **无任何输出**（空）；(3) 打印 `selfcheck: … pass=True {... all True}`。
- **若 (1) 或 (2) 有输出，或 (3) pass=False**：**立即停止部署**，不要安装 cron——系统可能被改动，
  联系开发端核对。

---

## 5d. VPS 端伪前向演习核对（与本地 $0 结果逐字段一致）

本地已通过 `--dry-run`（差额 $0，见 `forward/dry_run_validation.json`）。在 VPS 上跑同一演习并比对：

```bash
cd /opt/forward_b2
/opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --dry-run
cat forward/dry_run_validation.json
```
- **预期关键字段**：`"trade_net_abs_diff": 0.0`、`"m2m_abs_diff": 0.0`、`"store_bars_exactly_equal_DB": true`、
  `"PASS": true`，且 `"config_sha256"` ==
  `c37ff0a47c5b2a6cb5f7aa1ad8cbb9e542da8e27aac5d7d5b3b16b3dbe5d38d5`。
- 与本地仓库 `forward/dry_run_validation.json`（git 内）对照，`PASS`/两个 `abs_diff`/`config_sha256`
  应一致（`validated_utc` 不同是正常的）。比对：
  ```bash
  git show HEAD:forward/dry_run_validation.json | grep -E "PASS|abs_diff|config_sha256"
  ```
- **若 PASS≠true 或差额≠0**：**停止部署**，系统/数据有偏差，联系开发端。

---

## 5e. 网络端点核对（只碰 OKX 公开行情 + PushPlus）

```bash
cd /opt/forward_b2
grep -nE "https?://[a-zA-Z0-9./_-]+" scripts/forward_b2_4h.py | grep -viE "pushplus.plus"
```
- **预期**：只出现 `https://www.okx.com`（公开行情域名），且所有 path 为
  `/api/v5/market/candles`、`/api/v5/market/history-candles`、`/api/v5/public/funding-rate-history`。
  另有 `https://www.pushplus.plus/send`（推送）。**不应出现任何 demo 域名、`/trade/`、`/account/` path。**
- 连通性测试（只读公开行情）：
  ```bash
  curl -s --max-time 15 "https://www.okx.com/api/v5/public/time"   # 预期 {"code":"0",...}
  ```

---

## 配置 PushPlus token（每日推送）

```bash
cd /opt/forward_b2/forward
cp notify.conf.template notify.conf
nano notify.conf      # 把 YOUR_TOKEN_HERE 换成你的 PushPlus token（https://www.pushplus.plus/ 微信扫码获取）
chmod 0600 notify.conf
```
- `notify.conf` 已被 gitignore，token 不入 git。脚本只从此文件读 token，绝不读 `.env` 或任何交易凭证。

---

## seed + 首次回填 + 首次记账

```bash
cd /opt/forward_b2
/opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --seed       # 一次性：DB 全历史 -> forward store
/opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --cron-4h    # 回填 DB 之后的 REST 数据 + 首次记账
```
- **预期**：`--seed` 打印每币 seeded 行数；`--cron-4h` 打印各币 `+N 1m, +M funding, gaps=0` 与
  `account: … positions={ BTC:…, … }`。
- **若 `gaps≠0`**：记录区间；通常是 OKX REST 偶发缺页，再跑一次 `--update` 多半补齐；持续缺口联系开发端。

---

## 5f. 安装 cron（你手动 `crontab -e` 粘贴；脚本本身绝不改 crontab）

```bash
crontab -e
```
粘贴以下行（**已写死 UTC**，与策略 4h UTC 边界对齐；记账在每个 4h bar 收盘后 5 分钟触发，
日报在 UTC 日切后触发，对账每月初）：

```cron
CRON_TZ=UTC
# B2_4h 前向观察 —— 每 4h 拉数+记账（bar 收盘后 5 分钟）
5 0,4,8,12,16,20 * * * cd /opt/forward_b2 && /opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --cron-4h >> /opt/forward_b2/forward/cron.log 2>&1
# 每日 PushPlus 日报（UTC 00:20，在 00:05 记账之后）
20 0 * * * cd /opt/forward_b2 && /opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --cron-daily >> /opt/forward_b2/forward/cron.log 2>&1
# 月度对账（每月 1 日 01:00 UTC）
0 1 1 * * cd /opt/forward_b2 && /opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --reconcile >> /opt/forward_b2/forward/cron.log 2>&1
```
确认 cron 已生效：
```bash
crontab -l | grep forward_b2          # 预期：看到上面三行
grep -c "" /opt/forward_b2/forward/cron.log   # 触发后 log 会增长
```
**安全停止系统**（不删除部署，仅停 cron）：
```bash
crontab -e        # 在三行前各加 '#' 注释掉，或删除这三行；保存即停。系统无其他常驻进程。
```

---

## 5g. 部署后验证清单（你手动执行一遍）

```bash
cd /opt/forward_b2
# 1) 手动触发一次完整循环
/opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --cron-4h
# 2) 手动发一条测试日报，确认手机收到
/opt/forward_b2/.venv/bin/python scripts/forward_b2_4h.py --cron-daily
# 3) 确认记账文件正常生成
ls -la forward/state/ledger_trades.jsonl forward/state/ledger_daily_m2m.jsonl forward/state/positions.json forward/state/heartbeat.json
tail -2 forward/state/ledger_daily_m2m.jsonl
cat forward/state/positions.json
# 4) 确认数据环境可审计（恒为 mainnet）
cat forward/data/manifest.json | grep -E "server|demo"     # 预期 "server":"mainnet" "demo":false
```
- ✅ 收到一条 PushPlus 日报（各币持仓 / 当日 PnL / 距 gate 余量 / 系统健康）。
- ✅ `heartbeat.json` 的 `last_run_utc` 是刚才时间。
- ✅ `manifest.json` 的 `server=mainnet`、`demo=false`。
- ✅ `crontab -l` 三行在位。

完成后系统进入自主运行：每 4h 记账、每日推送、每月对账。**阶段 1 未过（gate 见
`gates_preregistered.md`）不上任何真金**；KILL/UPGRADE 由脚本 gate 逻辑自动判定并即时告警。

---

## 出错速查
| 现象 | 处理 |
|------|------|
| `--selfcheck` pass=False | 停止；core 文件被改/config hash 变了；联系开发端 |
| `--dry-run` PASS≠true | 停止；数据或引擎偏差；不要上 cron |
| `gaps≠0` 持续 | 多跑 `--update`；仍缺联系开发端（OKX REST 缺页 vs 真缺口）|
| 收不到 PushPlus | 查 `notify.conf` token；`forward/state/pending_push/` 有留存，下次自动补发 |
| 对账 `store_matches_rest=false` | K3 计时起点；7 天内排查（REST 修订 vs store 损坏），按 gate 处理 |
| 心跳 >8h 未更新 | cron 未触发/脚本异常；查 `forward/cron.log` |
