# data_engineering/ — 数据域宪法

> 本目录为数据工程域（下载/校验/入库/manifest）。宪法先立、脚本后迁
> （重构步骤 2 迁入 `download_*` / `verify_*` / `import_*` 等脚本；本步骤目录暂空）。
> 顶层 CLAUDE.md 的"数据环境铁律"是本宪法的上位法，冲突时以顶层为准。

## 1. 唯一可信源纪律

- **唯一可信回测数据源：`.vntrader/database_mainnet.db`**（OKX mainnet 公开 REST
  重建，经 Binance 全量交叉验证 PASS，见 `reports/regime/data_trust_closure_20260611/`）。
  研究只读（`mode=ro`）。
- 备份 `~/backups/database_mainnet_20260611.db.gz`，原库 SHA256
  `a6d6928dbdec108f54ebc413ec84344d3e9cde5f4f54dd07b5adec36f573495d`；
  还原校验 `gunzip -c <备份> | sha256sum` 须等于该值。
- Binance 侧可信源：`data/binance_vision/`（生产 CDN，逐 zip 对 `.CHECKSUM`
  sha256 校验；本项目唯一覆盖完整牛熊周期的干净样本）。

## 2. 污染库绝对红线

- `.vntrader/database_DEMO_CONTAMINATED.db` = 已确认污染的 OKX DEMO 行情。
  **不读入任何研究流程、不移动、不删除、不重命名**；仅作取证/对比基准
  （取证档案：`reports/regime/data_contamination_forensics_20260610/`）。

## 3. testnet/demo 红线 —— 可证伪核对，非态度要求

"以为拉的是真实数据"挡不住污染（demo 惨案当时也"确认过"）。新数据源必须做
**可证伪**的核对，两件都做：

1. **端点 URL 审计**：打印实际请求域名/端点，验证非 demo/testnet 变体
   （范式：Deribit `www.deribit.com` + `get_index_price` 返回 `testnet=false`，
   见 `vrp/reports/atm_vrp_stageA_data_20260628/` Q1b）。
2. **独立锚点交叉验证**：与独立第三方源逐日/逐 bar 抽验（≥3 随机日，含尾部日）
   ——锚点须独立于被验源（范式：Deribit 用 Coinbase 现货锚点，平静日 median
   偏差 0.025%；mainnet 库用 Binance 全量 1:1）。注意锚点是"同一个市场"级
   验证，不要求逐 tick 相等。

## 4. 下载与入库规范

- 任何写库脚本启动时向 stdout **打印数据环境（MAINNET/DEMO）**。
- **manifest 必须写 `server` 字段** + 文件清单 + 逐文件 sha256 + 窗口/宇宙参数。
- **数据环境必须显式命令行指定**；`.env` 的 `OKX_SERVER` **不得被任何下载脚本
  隐式继承**（demo 事故根因即隐式继承，见 `reports/MR5M_postmortem.md`）。
- 新数据入库后、用于研究前，先过第 3 节两项可证伪核对。
- 已知坑（复用时核对）：OKX 库时间戳为 naive-Shanghai（UTC−8h）；pandas 3.0
  `to_datetime` 默认 µs 精度（大整数日期用 `datetime64[D]`）；Deribit 1D bar
  时间戳=开盘时点（close 为标签+24h）；已到期期权 flat-pad 假报价须按
  `t≤expiry 且 volume>0` 清洗。

## 5. git 纪律

- **大文件（zip/csv/db/raw）一律 gitignore**；**manifest/README/校验结果入 git**
  （可复现性靠 manifest，不靠大文件入库）。
- 移动/新增数据目录须同步 `.gitignore` 精确路径。

## 6. 边界

- 本域不做研究判定（那是 research 域），只保证"喂给研究的数据是真的、可复现的"。
- 前向系统 seed 依赖 `.vntrader/database_mainnet.db` + `data/funding/okx/*.csv`
  的路径——**不得移动或脱依赖**（见 `docs/AGENTS.md` forward 域规则）。
