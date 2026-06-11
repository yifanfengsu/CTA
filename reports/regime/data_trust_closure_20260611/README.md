# 数据信任闭环：mainnet 库 Binance 全量交叉验证 + funding 正式核查

日期：2026-06-11（UTC）
性质：两个本地数据库只读（mode=ro）；新下载数据写入独立目录；未跑回测；未碰 VPS。

## ⬛ 判定结果（置顶）

| 项 | 结果 |
|---|---|
| **OKX mainnet vs Binance 交叉验证（2d 预注册三 gate）** | **PASS**（G1/G2/G3 全过，gate 在跑前写死于脚本头部，未事后修改） |
| **funding 数据正式核查** | **通过**（代码审计干净 + 10/10 抽样与 OKX 当前返回精确一致 + 网格完整） |
| **funding 缺口补齐** | 至 **2026-06-11 08:00 UTC**（5 币各 +213 条） |

CLAUDE.md 数据铁律的"外部独立源交叉验证"一项，对 `database_mainnet.db` **就此闭环**；
复盘 §8 检查清单第 1 项对当前库标记已闭环（文档更新清单见文末）。

## 对齐口径（任务要求显式写明）

- OKX 库 `datetime` = **Asia/Shanghai naive，bar OPEN 时刻** → UTC 分钟 = naive − 480min（无 DST）。
- Binance vision `open_time` = **UTC 毫秒（≥2025-01 文件为微秒），bar OPEN 时刻**，统一归一到 UTC 分钟。
- 按 UTC epoch-minute 内连接；两侧均为 last-price 1m K 线。
- **OKX 与 Binance 是两个独立交易所**：存在正常永续基差，验证目标是"同一个真实市场"
  （无系统性脱锚、无合成形态），不是逐 tick 相等。

## 第 1 部分：Binance 数据获取（永久资产）

- 5 交易对（UM 永续）× 41 个月（2023-01 → 2026-05）= **205 个 zip，205/205 下载成功，
  全部通过官方 .CHECKSUM（sha256）校验**，共 344MB。脚本 `scripts/download_binance_vision.py`
  （重试 ≤4、断点续传、manifest 带显式 source/server 字段）。
- 资产文档：`data/binance_vision/README.md` + `manifest.json`（入 git）；zip 本体不入 git。
- **完整性（1b）**：重叠区间（2023-01-01 00:00 UTC → 2026-05-28 15:59 UTC）内
  **5 币全部 1,790,880/1,790,880 根 1:1 对齐，双侧零缺口**。两个边缘差异均为区间
  错位而非缺口：okx_only 480 根/币 = OKX 起点（2022-12-31 16:00 UTC = 2023-01-01 00:00 上海）
  早于 Binance 月度文件起点 8 小时；binance_only 4,800 根/币 = Binance 5 月文件
  延伸到月末（OKX 库止于 2026-05-28）。Binance 历史停机缺口：本区间未检出。

## 第 2 部分：OKX mainnet vs Binance 全量逐 bar 对比

### 2a. bar 级统计（每币 1,790,880 根对齐 bar，全期 2023-01 → 2026-05）

| 币 | close 偏差 median | p90 | p99 | p99.9 | max | >0.1% 占比 | >0.5% | >1% | >5% |
|---|---|---|---|---|---|---|---|---|---|
| BTC | **0.0343%** | 0.059% | 0.076% | 0.094% | 2.43% | 0.06% | 0.004% | 0.0005% | 0% |
| ETH | **0.0343%** | 0.061% | 0.081% | 0.131% | 2.84% | 0.27% | 0.005% | 0.0023% | 0% |
| SOL | **0.0324%** | 0.068% | 0.109% | 0.306% | 5.51% | 1.46% | 0.014% | 0.0039% | 0.00011% |
| LINK | **0.0323%** | 0.075% | 0.107% | 0.264% | 4.77% | 1.71% | 0.006% | 0.0033% | 0% |
| DOGE | **0.0342%** | 0.071% | 0.103% | 0.140% | 12.04% | 1.22% | 0.010% | 0.0050% | 0.00151% |

- **滞后检验**：5/5 币 1m 收益互相关峰值均在 **k=0**（corr 0.986–0.995）——同步、
  无错位、无滞后镜像。（对照：demo 库当年 k=0 峰值 corr 仅 0.374。）
- 尾部极值（如 DOGE max 12%）集中在闪崩分钟内两所成交路径瞬时差异，
  占比 ≤0.0015%，属跨所正常微观结构差异。

### 2b. 合成形态扫描（关键步骤）

探测器**逐字移植** `research_v2b_dd_diagnosis.py` step-4（参数不变：步长 ≥1.5%/min、
连续 ≥4 步、累计 ≥10%、单根 ≥15%、阶梯 cv≤0.30）——与 demo 库的 598 起直接可比。

| 库 | BTC | ETH | SOL | LINK | DOGE | 合计 |
|---|---|---|---|---|---|---|
| demo（污染库，历史参考） | 0 | 23 | 344 | 15 | 216 | **598** |
| **mainnet（本次）** | **0** | **1** | **3** | **3** | **4** | **11** |

**11/11 全部被 Binance 同向运动证实为真实市场事件**（证实规则预注册：净向同号 ∧
Binance 振幅 ≥0.5× OKX 振幅；实际全部落在 0.79–1.06× 区间，即两所几乎等幅同动）：

| 事件窗口（UTC） | 币 | OKX 振幅 | Binance 振幅 | 对应真实事件 |
|---|---|---|---|---|
| 2023-01-02 07:13 | SOL | +19.7% | +20.5% | SOL 年初挤压反弹 |
| 2023-06-10 04:20 | SOL/LINK | −19.3%/−18.3% | −18.3%/−17.9% | 美 SEC 诉讼期山寨币抛售 |
| 2023-08-17 21:40 | ETH/LINK/DOGE | −12.8~−13.8% | −12.1~−13.1% | 8·17 全市场闪崩 |
| 2024-03-05 19:51 | DOGE | 19.3% | 15.3% | meme 行情剧烈回撤 |
| 2025-02-03 02:03 | DOGE | −9.9% | −9.3% | 2 月初急跌 |
| 2025-10-10 21:11–21:20 | SOL/LINK/DOGE | −17~−60% | −18~−59% | 10·10 大清算级闪崩 |

2 起带 staircase 标志（LINK，清算瀑布的台阶式成交），但均被 Binance 等幅同向证实——
与 demo 库"同时刻其他所纹丝不动 + 单根 ±91% 完美复原"的合成签名有本质区别。
明细：`synthetic_scan_mainnet.json`。

### 2c. 基差合理性

- 全期签名基差（OKX−Binance）均值 +0.012% ~ +0.018%（OKX 系统性略高于 Binance
  约 1.5 bp，量级远小于正常永续基差带 ±0.05~0.2%，方向稳定、可解释为两所资金
  费/标记价机制差）。
- **逐月检查（205 个币-月）**：|月度中位签名基差| 最大值仅 **0.072%**（LINK 2024-03），
  距 0.5% gate 有 7 倍裕量；**无任何月份出现 demo 式持续单边脱锚**
  （demo 库当年 2025-07→2026-03 周级 ±1%+ 脱锚形态在 mainnet 库完全不存在）。
  每币逐月表：`okx_vs_binance_stats/<SYM>.json` 的 `monthly` 节。

### 2d. 判定（gate 预注册于脚本头部，未事后修改）

| Gate | 标准 | 实际 | 结果 |
|---|---|---|---|
| G1 | 合成形态 0 起，或全部被 Binance 同向证实 | 11 起，11/11 证实 | ✅ |
| G2 | 无月份 \|月中位签名基差\| > 0.5% | 最大 0.072% | ✅ |
| G3 | 每币全期 median \|close 偏差\| < 0.1% | 0.0323–0.0343% | ✅ |

**VERDICT: PASS**（机器判定 `verdict.json`，运行日志 `run_log.txt`）。

## 第 3 部分：funding 数据正式核查 + 补缺

### 3a. 环境取证（简化 H1/H2）

- **H1 代码审计**（程序化扫描，结果存 `funding_verification.json`）：
  `download_okx_funding_history.py` 与 `download_okx_historical_funding_files.py`
  均为 **0 处** `x-simulated-trading`、**0 处** `OKX_SERVER` 引用；仅含硬编码公开
  mainnet URL（`www.okx.com/api/v5/public/funding-rate-history`、
  `static.okx.com/cdn/okex/traderecords/...`）。盘点时的"初判干净"就此正式坐实。
- **H2 抽样重验**（seed=20260611，10 条跨 5 币跨 2023–2026）：**10/10 与 OKX 今日
  返回值精确一致（浮点误差 <1e-12）**。近端 2 条走公开 API；远端 8 条对官方历史
  CDN 月度 zip 现场重新下载比对（证据存 `funding_refetch/`）。funding 历史是不可变
  结算记录，精确一致 = 本地数据未被篡改且来源为真实结算值。
- **量级合理性**：18,870 条记录 max |rate| = 1.079%（SOL 挤压日，OKX 上限内）；
  |rate| > 0.3% 占比仅 0.026%（全部集中在 SOL/DOGE 的极端行情日）——量级合理。

### 3b. 补缺 + 完整性

- 用既有 `download_okx_funding_history.py`（API 深度实测回溯至 2026-03-11，覆盖缺口）
  补齐 **2026-04-01 → 2026-06-11**：5 币各 213 条新 CSV（`data/funding/okx/
  <INST>_funding_2026-04-01_2026-06-11.csv`，与旧 CSV 在 2026-03-31 16:00 UTC 无缝衔接、无重叠）。
- 按铁律的补缺 manifest（source/server=MAINNET/endpoint/区间/文件清单）写入
  `funding_verification.json` 的 `backfill_manifest` 节。
- **8h 网格完整性（新旧合并 2022-12-31 16:00 → 2026-06-11 08:00 UTC）**：5 币各
  3,774 条 = 理论网格数**精确相符**，零缺口、零重复。
- **诚实记录一处检查口径修正**：首版检查用"间隔 ≠ 8h（毫秒级精确）"判缺口，把旧 CDN
  记录结算时间戳的秒级抖动（如 16:00:11，最大 ±12s，BTC 无、其余 4 币 388–692 个间隔
  受影响）误报为缺口。修正为"偏离 8h > 60s 才算缺口 + 总数须等于理论网格数"——这是
  检查实现 bug 的修正（抖动显然不是缺失结算），不是放宽判定；修正前后原始数据零变化，
  抖动统计完整保留在 `funding_verification.json`。

## 回测口径备注

- Binance 为不同交易所价格，**不可直接用作 OKX 策略回测数据源**，仅作校验与异常对照。
- funding 数据现覆盖 2023-01-01 → 2026-06-11；时间戳抖动 ≤12s 对 8h 粒度的回测使用无影响。

## 产物清单

```
reports/regime/data_trust_closure_20260611/
├── README.md                      本文件
├── verdict.json                   2d 机器判定（PASS + gate 明细）
├── okx_vs_binance_stats/<SYM>.json  每币全套统计（含逐月表、滞后检验）
├── synthetic_scan_mainnet.json    合成扫描 11 事件 + Binance 对照 + demo 参考计数
├── funding_verification.json      H1/H2/量级/网格/补缺 manifest
├── funding_refetch/               H2 重验证据（OKX 官方 CDN 月度 zip 原件）
└── run_log.txt                    对比脚本运行日志
data/binance_vision/               205 个 zip（不入 git）+ README + manifest（入 git）
data/funding/okx/<INST>_funding_2026-04-01_2026-06-11.csv  补缺数据（5 份，数据文件不入 git）
scripts/download_binance_vision.py / research_okx_vs_binance.py / research_funding_trust_check.py
```

## 由本次验证产生的文档更新

- `CLAUDE.md` 数据铁律节：追加"mainnet 库已于 2026-06-11 通过 Binance 全量交叉验证"一行。
- `reports/MR5M_postmortem.md` §8 检查清单第 1 项：加注"对 database_mainnet.db 已闭环"。
- `.gitignore`：排除 `data/binance_vision/**/*.zip`（344MB 不入 git；README/manifest 入库）。
- 副作用如实记录：运行补缺时 `download_okx_funding_history.py` 按其默认行为更新了
  `reports/research/funding/` 下 3 个下载报告文件（追加 2026-04→06 区间的请求记录），一并提交。

## 未改动文档及原因

- `PROJECT_GUIDE.md`：数据资产叙述待新研究启动时一并修订（本次仅闭环验证，不新增研究结论）。
- 基准引擎与归档实盘脚本：零修改（铁律）。
- 旧 funding CSV 与两个数据库：零写入（只读校验）。
