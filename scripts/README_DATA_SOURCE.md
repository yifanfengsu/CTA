# ⚠️ 数据源警示（scripts/ 目录必读）

**唯一可信回测数据源：`.vntrader/database_mainnet.db`**（CLAUDE.md 数据环境铁律）。

## 为什么有这个文件

2026-06 确认：旧库（原 `.vntrader/database.db`）是 OKX DEMO 模拟盘行情，13 个月研究
全部建立在污染数据上（完整复盘：`research/_closed/_synthesis/MR5M_postmortem.md`；修补记录：
`reports/regime/vulnerability_patch_20260611/`）。该库已改名为
**`.vntrader/database_DEMO_CONTAMINATED.db`**，仅作取证/对比基准，严禁用于回测或研究。

## 对本目录脚本的影响

- **`backtest_mr_5m_compare.py`（基准引擎，按 CLAUDE.md 不修改引擎本身）**：
  其 `--database-path` 默认值仍是旧名 `.vntrader/database.db`——该路径现已不存在。
  **直接裸跑引擎会立即报错（fail loud），而不是静默使用污染数据——这是预期的防呆行为。**
  正确用法：显式传 `--database-path .vntrader/database_mainnet.db`，或外层注入
  （参考 `research_mr5m_mainnet_baseline.py` / `research_demo_vs_mainnet.py` 的
  `mode=ro` 只读加载模式）。
- **全部旧研究脚本（`research_*.py` / `backtest_*.py` / `analyze_*.py` 历史档案）**：
  默认路径同样指向旧名，裸跑会报错。这是期望行为，不要"修复"它们去指向 mainnet 库
  ——它们的结论基于污染数据已整体作废，仅作代码参考。
- **已禁用脚本（顶部硬退出，禁用≠删除）**：`download_okx_history.py` /
  `download_history.py` / `merge_recent.py`（三条污染写入通道）。
  下载历史数据一律用 **`download_mainnet_history.py`**（硬编码 mainnet、不读 .env、
  manifest 带 server 字段）。

## 新研究的数据使用规则（摘自 CLAUDE.md / 复盘第 8 节）

1. 只读打开：`sqlite3.connect("file:...database_mainnet.db?mode=ro", uri=True)`。
2. 写库脚本必须 stdout 打印数据环境 + manifest 写 `server` 字段。
3. 新数据入库后、用于研究前，与外部独立源抽样交叉验证（≥3 随机日逐 bar）。
4. 数据环境必须显式命令行指定，禁止隐式继承 `.env` 的 `OKX_SERVER`。
