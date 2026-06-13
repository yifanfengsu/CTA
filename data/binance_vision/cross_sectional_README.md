# 横截面研究数据资产（原材料，非研究）

生成于 2026-06-13（UTC）· 脚本 `scripts/build_cross_sectional_list.py`（清单）
+ `scripts/download_binance_vision.py --symbols ...`（下载）
+ `scripts/catalog_cross_sectional.py`（编目）。

> **定位声明**：本目录只做一件事——把一批主流永续合约的历史 K 线与 funding 下载为
> 带 sha256 校验的本地静态资产。**不定义横截面宇宙、不处理幸存者偏差、不做任何分析、
> 不做立项判定**——这些全部留给后续正式开题。这是"原材料"，不是"研究"。

---

## ⚠️ 已知局限（开题前必读，本任务显式不处理）

1. **幸存者偏差**：下载清单是**"今天（2026-06）的主流币快照"**——按 2026-05 成交额
   从当前在市合约中排名选取。**不含任何历史退市/下架合约**（如已消失的山寨永续）。
   直接用这批数据估计"过去做横截面能赚多少"会系统性高估收益（赢家留存、输家消失）。
   **幸存者偏差的防御（退市合约纳入、point-in-time universe 重建）属开题阶段决策，
   本任务不处理。**
2. **快照非 point-in-time universe**：排名用的是最新月成交额，不是各历史时点的当时
   排名。截面成分随时间变化（新币上线、旧币起落）未被建模。
3. **排名口径替代**：`fapi.binance.com` 在本环境被地域封锁（HTTP 451），无法取官方
   ticker/24hr。按任务预留的备选路径，改用 **vision 上 2026-05 月度 1d K 线的 USDT
   quote volume** 作为"近 30 天成交额"代理排名（spec 明确允许）。这是机械规则，
   不是主观挑选。
4. **gaps 不填补**：Binance 历史停机/维护缺口如实记录、不插值（沿用既有 Binance-vision
   处理惯例）。见下表 `missing_minutes`。

---

## 下载清单的机械规则（预注册，无主观增删）

- **宇宙**：data.binance.vision 上全部 UM 永续、symbol 以 `USDT` 结尾（S3 listing，
  732 个）。
- **排名**：2026-05 月度 1d K 线的 USDT quote volume 降序（588 个有 2026-05 文件）。
- **选取**：取前 40。
- **18 月门槛**：前 40 中，最早可得 1m 月份晚于 **2024-11**（即不足 18 个月）者，
  列入"观察名单"记录在案、**本次不下载**。
- **复用**：已有 5 币（BTC/ETH/SOL/LINK/DOGE）若在清单内则复用已下载部分（下载器
  断点续传、sha256 通过即跳过）。

---

## 编目结果（描述性，不判定可用性）

- **下载币种**：**22**（17 新下载 + 5 复用）。**观察名单（<18 月，未下载）**：18。
- **数据量**：K 线 ~2027.7 MB，funding ~1.13 MB。**失败下载：0**（全部 sha256 校验通过；
  首轮 16 个文件因瞬时 SSL 错误失败，第二轮断点续传全部恢复）。
- **完整性**：多数币 100% 分钟网格连续；6 个币各有 2 处缺口、共 7200 缺失分钟
  （= 5 天整块，Binance 历史维护缺口，**记录不填补**），连续率 99.75–99.79%。
- 逐币明细（排名 / 最早月 / 总 bar / 缺口 / funding 结算数 / 校验状态）见
  `cross_sectional_manifest.json`。

| 类别 | symbol（含 vision 最早 1m 月份）|
|---|---|
| 复用 5 币 | BTCUSDT ETHUSDT(2020-01) SOLUSDT(2020-09) LINKUSDT(2020-01) DOGEUSDT(2020-07) |
| 新下载 17 币 | ZEC XRP BNB SUI NEAR TON 1000PEPE XLM WLD ONDO TAO ADA FIL AVAX BCH ENA INJ |
| 观察名单 18 币（<18 月，未下载）| HYPE(2025-05) XAU(2025-12) XAG(2026-01) CL(2026-04) LAB(2025-10) BZ(2026-04) BSB(2026-03) BILL(2026-05) MU(2026-04) SKYAI(2025-05) SNDK(2026-04) UB(2025-09) B(2025-05) EDEN(2025-09) CRCL(2026-02) PLAY(2025-07) ALLO(2025-11) VVV(2025-01) |

注：XAU/XAG/CL（黄金/白银/原油代币化永续）、CRCL/SNDK（股票代币）等出现在成交额
前 40 但因上线不足 18 月进观察名单——本任务机械执行门槛，不对品种性质做判断。

---

## 目录结构

```
data/binance_vision/
├── <SYMBOL>/                         1m K 线月度 zip（不入 git；.gitignore 覆盖）
├── funding/<SYMBOL>/                 fundingRate 月度 zip（不入 git）
├── cross_sectional_download_list.json  机械清单中间产物（入 git）
├── cross_sectional_manifest.json     编目（入 git）
├── cross_sectional_README.md         本文件（入 git）
└── failed_downloads.txt              失败清单（入 git；本次为 none）
```

数据来源 `binance_vision`（公开静态，production 行情，无 demo 变体）；`server=N/A`。
两个现有数据库（mainnet / 污染库）本任务**完全未触碰**，新数据**不并入任何现有库**。

---

横截面数据下载完成于 2026-06-13（UTC），币种数:22（17 新 + 5 复用）/ 1m+funding 全校验
通过:1856/1856（928 真实 K 线月文件 + 928 funding 月文件，sha256 全通过；另各 381 个
上线前月份 404 跳过）/ 失败:无（首轮 16 个瞬时 SSL 失败经断点续传第二轮全恢复）/
manifest 已编目 / 幸存者局限已标注 / 已 push：已确认（1e75b31）
