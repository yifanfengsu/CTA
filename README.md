# CTA

## 项目简介

这是一个面向 OKX `BTCUSDT` 永续合约的 vn.py headless CTA 项目，核心依赖是 `vn.py`、`vnpy_okx`、`vnpy_ctastrategy` 和 `vnpy_sqlite`。项目只面向 WSL、Linux、VPS 等命令行环境，不引入 GUI/Qt 流程。

当前优先目标是：历史数据下载、完整性验证、回测、alpha 诊断、保守参数 sweep、OKX 模拟盘前置检查。

重要限制：当前标准 vn.py `BacktestingEngine` 结果未自动计入 OKX perpetual funding fee。回测报告只包含价格盈亏、手续费、滑点等常规项，不能把它直接视为完整实盘收益。

当前仓库没有 `scripts/run_cta.py`。Makefile 只覆盖模拟盘前置检查，真正 demo runner 需要后续补 `scripts/run_cta.py` 后再加入运行命令。

## 目录结构

- `config/`：运行配置和策略配置。
- `config/instruments/`：本地合约元数据，例如 `btcusdt_swap_okx.json`。
- `data/raw/`：可选原始行情/CSV 导出目录，已在 `.gitignore` 中保留 `.gitkeep`。
- `data/history_manifests/`：历史下载断点续传 manifest。
- `scripts/`：所有 headless 命令行脚本。
- `strategies/`：CTA 策略类。
- `tests/`：`unittest` 测试。
- `reports/`：回测、验证、诊断、sweep 输出目录，生成内容不提交。
- `logs/`：脚本日志目录，生成 `.log` 不提交。
- `deploy/systemd/`：后续 VPS/systemd 部署占位。

## 快速开始

```bash
git clone git@github.com:yifanfengsu/CTA.git
cd CTA
make venv
source .venv/bin/activate
make install
make env
```

然后编辑 `.env`，填入 OKX DEMO 或 REAL 对应的 API Key、Secret、Passphrase，再执行：

```bash
make doctor
```

## .env 配置说明

`.env` 从 `.env.example` 复制生成：

```dotenv
OKX_API_KEY=
OKX_SECRET_KEY=
OKX_PASSPHRASE=
OKX_SERVER=DEMO
OKX_PROXY_HOST=
OKX_PROXY_PORT=0
```

- `OKX_API_KEY`：OKX API Key。
- `OKX_SECRET_KEY`：OKX Secret Key。
- `OKX_PASSPHRASE`：OKX API Passphrase。
- `OKX_SERVER`：`DEMO` 或 `REAL`，要和密钥环境匹配。
- `OKX_PROXY_HOST`：代理主机，留空表示不用代理。
- `OKX_PROXY_PORT`：代理端口；不用代理时保持 `0`。

不要提交 `.env`。本仓库也忽略本地 SQLite 数据库、日志、报告和大体积行情输出。

## 推荐执行顺序

1. `make doctor`
2. `make inspect-okx`
3. `make check-okx SERVER=DEMO`
4. `make download-history-dry-run`
5. `make download-history`
6. `make verify-history`
7. `make backtest`
8. `make backtest-no-cost`
9. `make analyze-alpha REPORT_DIR=... COMPARE_REPORT_DIR=...`
10. `make alpha-sweep`
11. 满足条件后再考虑补 demo runner/模拟盘。

## Makefile 变量

所有变量都可以用 `make target VAR=value` 覆盖：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `PYTHON` | `.venv/bin/python` | Python 解释器路径 |
| `PIP` | `$(PYTHON) -m pip` | pip 调用方式 |
| `VT_SYMBOL` | `BTCUSDT_SWAP_OKX.GLOBAL` | 回测/下载/验证标的 |
| `INTERVAL` | `1m` | 历史 K 线周期 |
| `START` | `2025-01-01` | 起始日期，包含当天 |
| `END` | `2026-03-31` | 结束日期，包含当天 |
| `TIMEZONE` | `Asia/Shanghai` | 日期解释时区 |
| `CHUNK_DAYS` | `5` | 历史下载分块天数 |
| `SERVER` | `DEMO` | OKX 环境，`DEMO` 或 `REAL` |
| `CAPITAL` | `5000` | 回测初始资金 |
| `RATE` | `0.0005` | 手续费率 |
| `SLIPPAGE_MODE` | `ticks` | 滑点模式：`ticks` 或 `absolute` |
| `SLIPPAGE` | `2` | 滑点数值 |
| `REPORT_DIR` | 空 | alpha 诊断主报告目录，必填 |
| `COMPARE_REPORT_DIR` | 空 | alpha 诊断对照报告目录 |
| `OUTPUT_DIR` | 空 | 指定输出目录；为空时脚本自动生成 |
| `STRATEGY_CONFIG` | `config/strategy_default.json` | 默认策略配置 |
| `SANITY_CONFIG` | `config/strategy_sanity_min_size.json` | 保守 sanity 配置 |
| `MAX_RETRIES` | `8` | 历史下载每源重试次数 |
| `THROTTLE_SECONDS` | `0.35` | 历史下载请求间隔 |

## Makefile 命令总览

| 命令 | 作用 | 是否联网 | 是否写数据库 | 主要输出 | 常用示例 |
| --- | --- | --- | --- | --- | --- |
| `make help` | 打印命令和变量示例 | 否 | 否 | 终端输出 | `make` |
| `make venv` | 创建 `.venv` | 否 | 否 | `.venv/` | `make venv` |
| `make install` | 安装运行依赖 | 是，pip | 否 | `.venv/` 包 | `make install` |
| `make env` | 创建 `.env`，不覆盖已有文件 | 否 | 否 | `.env` | `make env` |
| `make doctor` | 本地依赖和 vn.py sqlite 自检 | 否 | 否 | `logs/doctor.log` | `make doctor` |
| `make inspect-okx` | 本地 OKX gateway 字段检查 | 否 | 否 | `logs/inspect_okx_gateway.log` | `make inspect-okx` |
| `make check-okx` | OKX 登录和合约元数据检查，不下单 | 是 | 否 | `config/instruments/*.json`、日志 | `make check-okx SERVER=DEMO` |
| `make download-history-dry-run` | 生成下载计划，不下载、不写 bar | 否 | 否 | 终端计划、日志 | `make download-history-dry-run START=2025-01-01 END=2025-01-02 CHUNK_DAYS=1` |
| `make download-history` | 下载历史数据、逐块保存、校验完整性 | 是 | 是 | sqlite、`data/history_manifests/`、日志 | `make download-history CHUNK_DAYS=3` |
| `make repair-history` | 按本地缺口修复历史数据 | 是 | 是 | sqlite、manifest、日志 | `make repair-history START=2025-01-01 END=2025-01-31` |
| `make verify-history` | 独立验证本地历史完整性 | 否 | 否 | `reports/history_verify_latest.json` | `make verify-history` |
| `make backtest` | 成本版回测 | 否 | 否 | `reports/backtest/YYYYMMDD_HHMMSS/` 或 `OUTPUT_DIR` | `make backtest OUTPUT_DIR=reports/backtest/manual_cost` |
| `make backtest-no-cost` | 无成本回测，用于判断毛 alpha | 否 | 否 | 回测报告目录 | `make backtest-no-cost OUTPUT_DIR=reports/backtest/manual_no_cost` |
| `make backtest-sanity` | 使用保守最小手数配置回测 | 否 | 否 | 回测报告目录 | `make backtest-sanity` |
| `make analyze-alpha` | 分析一个或两个回测报告 | 否 | 否 | `REPORT_DIR/alpha_diagnostics/` 或 `OUTPUT_DIR` | `make analyze-alpha REPORT_DIR=... COMPARE_REPORT_DIR=...` |
| `make alpha-sweep` | 保守参数 shortlist sweep | 否 | 否 | `reports/alpha_sweep/YYYYMMDD_HHMMSS/` 或 `OUTPUT_DIR` | `make alpha-sweep OUTPUT_DIR=reports/alpha_sweep/manual_001` |
| `make test` | 运行全部单元测试 | 否 | 否 | 终端输出 | `make test` |
| `make test-one` | 运行单个测试文件 | 否 | 否 | 终端输出 | `make test-one TEST=tests/test_history_time_utils.py` |
| `make compile` | 编译检查脚本、策略、测试 | 否 | 否 | `__pycache__/` | `make compile` |
| `make clean-cache` | 删除缓存目录 | 否 | 否 | 删除缓存 | `make clean-cache` |
| `make clean-logs` | 删除 `logs/*.log`，保留 `.gitkeep` | 否 | 否 | 清理日志 | `make clean-logs` |
| `make clean-reports` | 删除报告，必须确认 | 否 | 否 | 清理 `reports/` | `make clean-reports CONFIRM=1` |
| `make tail-log` | 查看日志尾部 | 否 | 否 | 终端输出 | `make tail-log LOG_FILE=logs/download_okx_history.log` |

## 每个命令的详细用法

### `make help`

默认目标，`make` 等同于 `make help`。用于快速查看当前支持的命令、常用变量和覆盖示例。

### `make venv`

创建 `.venv`：

```bash
make venv
```

如果 `.venv` 已存在，目标会提示并跳过创建。

### `make install`

升级 pip 并安装项目运行依赖：

- `vnpy`
- `vnpy_ctastrategy`
- `vnpy_okx`
- `vnpy_sqlite`
- `python-dotenv`
- `pandas`
- `numpy`

仓库当前没有 `requirements.txt`，Makefile 内部保留最小安装列表。

### `make env`

当 `.env` 不存在时，从 `.env.example` 复制：

```bash
make env
```

如果 `.env` 已存在，不会覆盖，避免误删本地密钥。

### `make doctor`

运行：

```bash
make doctor
```

实际调用 `scripts/doctor.py`，检查 Python、依赖包、OKX gateway 导入、vn.py sqlite 设置和 `.env` 是否存在。主要日志在 `logs/doctor.log`。

### `make inspect-okx`

运行：

```bash
make inspect-okx
```

实际调用 `scripts/inspect_okx_gateway.py`。这是本地 gateway 字段检查，不连接交易所，适合在填 API Key 前确认本地 `vnpy_okx` 版本的字段名称。

### `make check-okx`

运行：

```bash
make check-okx SERVER=DEMO
make check-okx SERVER=REAL
```

实际调用 `scripts/check_okx_connection.py --vt-symbol $(VT_SYMBOL) --server $(SERVER) --timeout 30`。它会连接 OKX、等待私有登录和目标合约元数据，但不会下单。成功后会更新 `config/instruments/` 中的合约元数据文件。

只有 `.env` 已填完整并且 `OKX_SERVER` 与密钥环境匹配时才执行该命令。

### `make download-history-dry-run`

生成历史下载计划：

```bash
make download-history-dry-run START=2025-01-01 END=2025-01-02 CHUNK_DAYS=1
```

实际调用 `scripts/download_okx_history.py`，固定带 `--source auto --resume --dry-run`。该命令不联系 OKX、不保存 bar、不写 manifest；它会读取本地 sqlite 覆盖情况并打印计划。

### `make download-history`

下载历史数据并写入本地 vn.py sqlite：

```bash
make download-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
```

固定带：

- `--source auto`
- `--resume`
- `--save-per-chunk`
- `--verify-db`
- `--strict-completeness`
- `--max-retries $(MAX_RETRIES)`
- `--throttle-seconds $(THROTTLE_SECONDS)`

主要输出是本地 sqlite 数据库、`data/history_manifests/` 断点续传文件、`logs/download_okx_history.log`。

### `make repair-history`

按本地 sqlite 缺口进行修复：

```bash
make repair-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
```

固定带 `--repair-missing --source auto --resume --verify-db --strict-completeness`。当 `verify-history` 或 backtest preflight 报缺口时，优先运行这个目标。

### `make verify-history`

独立验证本地历史覆盖：

```bash
make verify-history START=2025-01-01 END=2026-03-31
```

输出固定为 `reports/history_verify_latest.json`，避免把 `VT_SYMBOL` 中的特殊字符拼进文件名。严格模式下发现缺口会返回非零退出码，并打印建议的 repair 命令。

### `make backtest`

运行成本版回测：

```bash
make backtest
make backtest OUTPUT_DIR=reports/backtest/manual_cost
```

默认使用 `config/strategy_default.json`，带手续费 `RATE=0.0005`、滑点 `SLIPPAGE_MODE=ticks`、`SLIPPAGE=2`，并启用 `--data-check-strict`。如果 `OUTPUT_DIR` 为空，脚本自动生成 `reports/backtest/YYYYMMDD_HHMMSS/`。

### `make backtest-no-cost`

运行无成本回测：

```bash
make backtest-no-cost OUTPUT_DIR=reports/backtest/manual_no_cost
```

固定 `--rate 0 --slippage-mode absolute --slippage 0`，其他参数同 `make backtest`。这个目标只用于判断毛 alpha，不用于模拟实盘收益判断。

### `make backtest-sanity`

使用 `config/strategy_sanity_min_size.json` 运行保守最小手数回测：

```bash
make backtest-sanity
```

该配置使用 `OkxAdaptiveMhfStrategy`、`fixed_size=0.01`、`risk_per_trade=0.0005`，并包含保守的 `max_leverage`、`max_notional_ratio`、`max_trades_per_day`。这个目标用于确认链路和报告是否正常，不用于证明策略已经可上线。

### `make analyze-alpha`

分析回测报告：

```bash
make analyze-alpha REPORT_DIR=reports/backtest/manual_cost
make analyze-alpha REPORT_DIR=reports/backtest/manual_cost COMPARE_REPORT_DIR=reports/backtest/manual_no_cost
```

`REPORT_DIR` 必填。`COMPARE_REPORT_DIR` 为空时只分析单个报告；非空时传入 `--compare-report-dir`，通常用于成本版 vs 无成本版。默认输出到 `REPORT_DIR/alpha_diagnostics/`，也可以用 `OUTPUT_DIR` 覆盖。

### `make alpha-sweep`

运行保守参数 shortlist sweep：

```bash
make alpha-sweep
make alpha-sweep OUTPUT_DIR=reports/alpha_sweep/manual_001
```

默认以 `config/strategy_sanity_min_size.json` 为 base config，固定 `--max-runs 100 --data-check-strict`。脚本内部会强制保守风控上限，例如 `fixed_size=0.01`、`risk_per_trade<=0.0005`、`max_leverage<=0.5`、`max_notional_ratio<=0.5`、`max_trades_per_day<=10`。

### `make test`

运行全部单元测试：

```bash
make test
```

实际调用 `python -m unittest discover -s tests -p "test_*.py"`。

### `make test-one`

运行单个测试文件：

```bash
make test-one TEST=tests/test_history_time_utils.py
```

`TEST` 为空会报错。Makefile 会把 `tests/test_history_time_utils.py` 转为 `tests.test_history_time_utils` 后交给 `unittest`。

### `make compile`

编译检查：

```bash
make compile
```

实际调用 `python -m compileall scripts strategies tests`。该命令会产生 `__pycache__/`，可用 `make clean-cache` 删除。

### `make clean-cache`

删除 `__pycache__`、`.pytest_cache`、`.mypy_cache`、`.ruff_cache`。

### `make clean-logs`

删除 `logs/` 下的 `.log` 文件，保留 `logs/.gitkeep`。

### `make clean-reports`

默认不会删除：

```bash
make clean-reports
```

必须显式确认：

```bash
make clean-reports CONFIRM=1
```

确认后删除 `reports/` 下生成内容，保留 `reports/.gitkeep`。

### `make tail-log`

查看日志尾部：

```bash
make tail-log
make tail-log LOG_FILE=logs/download_okx_history.log
```

默认查看 `logs/backtest_okx_mhf.log`。可用 `TAIL_LINES=200` 覆盖初始显示行数。

## 参数覆盖示例

```bash
make download-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
make verify-history START=2025-01-01 END=2026-03-31
make backtest START=2025-01-01 END=2026-03-31 OUTPUT_DIR=reports/backtest/manual_cost
make backtest-no-cost OUTPUT_DIR=reports/backtest/manual_no_cost
make analyze-alpha REPORT_DIR=reports/backtest/manual_cost COMPARE_REPORT_DIR=reports/backtest/manual_no_cost
make alpha-sweep OUTPUT_DIR=reports/alpha_sweep/manual_001
```

覆盖方式统一是 `make 目标 变量=值`。例如要切换 REAL 连接检查：

```bash
make check-okx SERVER=REAL
```

## 回测报告说明

`make backtest`、`make backtest-no-cost`、`make backtest-sanity` 会生成同一套报告文件：

- `warning.txt`：说明 vn.py 标准回测未自动计入 OKX 永续 funding fee。
- `run_config.json`：本次运行参数、策略配置、合约元数据、输出文件路径、数据检查摘要。
- `stats.json`：vn.py 统计结果和脚本补充的交易次数、胜率、破产标记等。
- `diagnostics.json`：资金曲线、破产检测、日级汇总等诊断信息。
- `daily_pnl.csv`：每日盈亏、余额、手续费、滑点、成交数等。
- `trades.csv`：成交明细。
- `orders.csv`：委托明细。
- `chart.html`：回测图表 HTML。headless 环境不会自动打开浏览器，可把文件路径交给已有浏览器打开。

## Alpha 诊断说明

`make analyze-alpha` 的核心用途是把成本版和无成本版拆开看：

- 成本版：包含手续费和滑点，更接近常规回测成本。
- 无成本版：`rate=0` 且 `slippage=0`，用于判断是否存在毛 alpha。
- 毛 alpha：无成本版仍无法盈利时，不能把亏损归因于成本。
- 成本拖累：无成本版和成本版的差额，用来衡量手续费、滑点对策略的压制。
- 交易频率：成交密度过高时，成本会吞掉弱 alpha。
- 最差小时/最差日期：用于定位时段过滤或风险控制的优先级。
- 不能只看收益曲线：收益曲线可能掩盖交易频率、成本拖累、局部大亏、样本偶然性和破产风险。

诊断输出默认在 `REPORT_DIR/alpha_diagnostics/`，包括 `alpha_summary.json`、`alpha_diagnostics.md`、月/周/日/时段/方向/持仓时长等 CSV。

## 防止过拟合的工作流

- 不要直接大规模参数搜索。
- 先固定 `2025-01-01` 到 `2026-03-31` 为主样本。
- 做成本版和无成本版对照，先确认是否有毛 alpha。
- 优先减少交易频率，不要扩大仓位掩盖问题。
- `alpha-sweep` 只做保守 shortlist，不把 sweep 结果直接当最终参数。
- 后续正式优化应再拆 train、validation、out-of-sample。
- 模拟盘前仍要保持最小仓位、保守风控和可回滚流程。

## OKX 模拟盘前置条件

进入 OKX DEMO 模拟盘前，至少满足：

- `make doctor` 通过。
- `make check-okx SERVER=DEMO` 通过。
- 目标区间历史数据完整，`make verify-history` 通过。
- 成本版和无成本版报告都完成。
- Alpha 诊断证明至少存在毛 alpha，并且成本拖累可解释。
- 策略没有用扩大仓位掩盖亏损。

当前如果没有 `scripts/run_cta.py`，不要声称已经支持模拟盘自动运行。下一步需要补 demo runner，再加入 systemd、日志轮转、异常退出重启、只读配置检查和最小下单保护。

## 常见问题

### `.env` 不存在

运行：

```bash
make env
```

然后编辑 `.env`。不要把 `.env` 提交到 Git。

### OKX DEMO/REAL 不匹配

`OKX_SERVER=DEMO` 要配 DEMO API Key；`OKX_SERVER=REAL` 要配 REAL API Key。也可以临时覆盖：

```bash
make check-okx SERVER=DEMO
```

### 代理错误

如果设置了 `OKX_PROXY_HOST`，必须设置正整数 `OKX_PROXY_PORT`。如果不用代理，保持：

```dotenv
OKX_PROXY_HOST=
OKX_PROXY_PORT=0
```

### 历史数据缺口

先验证：

```bash
make verify-history START=2025-01-01 END=2026-03-31
```

再按提示修复，或直接运行：

```bash
make repair-history START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3
```

### `vnpy_sqlite` 未安装

运行：

```bash
make install
make doctor
```

如果仍失败，检查当前 shell 是否已激活 `.venv`，或确认 `PYTHON=.venv/bin/python` 指向正确环境。

### `BacktestingEngine` 没有 funding fee

这是当前回测口径限制。`warning.txt` 和 `run_config.json` 会记录该提醒。正式评估 OKX 永续策略前，需要后续补资金费率数据和资金费成本模型。

### `chart.html` 如何打开

Makefile 不启动 GUI。可以在有浏览器的环境中打开生成的 `chart.html` 文件；在 VPS/headless 上可下载该 HTML 后本地打开。

### `reports/` 和 `logs/` 如何清理

```bash
make clean-logs
make clean-reports CONFIRM=1
make clean-cache
```

`clean-reports` 必须带 `CONFIRM=1`，避免误删回测结果。
