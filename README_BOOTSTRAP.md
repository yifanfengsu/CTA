# OKX + vn.py Headless CTA Bootstrap

这个仓库只面向命令行和 Linux/headless 运行，不包含任何 GUI/Qt 操作步骤。

## 原则

- 先 `DEMO`，后 `REAL`。
- API 密钥只放在 `.env`，不要硬编码到脚本或配置文件。
- 首个交易标的固定为 `BTCUSDT_SWAP_OKX.GLOBAL`。
- 当前骨架阶段只完成环境检查、目录约定和运行时配置，不包含策略交易逻辑。

## 目录说明

- `strategies/`: CTA 策略类目录，后续策略文件统一放这里。
- `scripts/`: 命令行脚本目录，当前已提供 `doctor.py`。
- `config/`: 运行期配置目录，`runtime.json` 放全局运行参数。
- `config/instruments/`: 单标的或多标的补充配置目录。
- `data/raw/`: 原始行情、原始接口回包、本地缓存数据目录。
- `reports/`: 回测结果、诊断报告、导出报表目录。
- `logs/`: 项目脚本日志目录。
- `deploy/systemd/`: Linux VPS 的 `systemd` 部署文件目录。

## 命令行流程

1. 创建并激活虚拟环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install vnpy vnpy_ctastrategy vnpy_okx vnpy_sqlite python-dotenv
```

2. 复制环境变量模板，并先填 DEMO 账号。

```bash
cp .env.example .env
```

3. 运行环境自检。

```bash
python scripts/doctor.py
```

4. 自检通过后，再进入后续脚本开发和联调。

## 脚本执行顺序

当前阶段仓库只提供 `scripts/doctor.py`，因此实际执行顺序只有一条：

1. `python scripts/doctor.py`

下一阶段建议保持下面的脚本顺序扩展：

1. `doctor.py`: 检查 Python、操作系统、依赖、OKX 网关、数据库设置。
2. `download_history.py`: 下载或拉取 OKX 历史行情。
3. `import_history.py`: 将标准化 Bar 数据写入 `vnpy_sqlite`。
4. `run_cta.py`: 启动无 GUI CTA 实盘/模拟盘主流程。
5. `backtest.py`: 离线回测和参数验证。

## DEMO 到 REAL 的切换

- 初始阶段保持 `.env` 中 `OKX_SERVER=DEMO`。
- DEMO 环境确认下单链路、持仓同步、行情接收、日志输出都正常后，再切换到 `REAL`。
- 切到 `REAL` 前，先补充风控、异常告警、重连策略、systemd 守护和回滚预案。

## 备注

- `doctor.py` 会在项目根目录自动准备 `.vntrader/` 运行目录，避免把运行时文件写到用户家目录。
- 后续新增脚本统一使用 `argparse`，保证可以直接从命令行 headless 运行。

## 历史数据下载失败/断点续传/修复缺口

所有历史数据命令默认使用 `--timezone Asia/Shanghai`。

- `--start 2025-01-01` 表示 `2025-01-01 00:00:00 Asia/Shanghai`。
- `--end 2025-01-03` 表示包含 `2025-01-03` 整天，内部会转换成 `end_exclusive=2025-01-04 00:00:00 Asia/Shanghai`。
- 下载、验证、缺口修复和回测 preflight 全部使用同一套半开区间规则：`[start, end_exclusive)`。

推荐先用支持断点续传、逐 chunk 写库、写后校验、REST fallback 和缺口修复的下载命令：

```bash
python scripts/download_okx_history.py \
  --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL \
  --interval 1m \
  --start 2025-01-01 \
  --end 2026-03-31 \
  --chunk-days 3 \
  --server DEMO \
  --timezone Asia/Shanghai \
  --resume \
  --source auto \
  --save-per-chunk \
  --verify-db \
  --repair-missing \
  --max-retries 8 \
  --retry-base-delay 2 \
  --retry-max-delay 120 \
  --throttle-seconds 0.35 \
  --csv-copy
```

推荐先用短区间做边界验证：

```bash
python scripts/download_okx_history.py \
  --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL \
  --interval 1m \
  --start 2025-01-01 \
  --end 2025-01-03 \
  --chunk-days 1 \
  --server DEMO \
  --timezone Asia/Shanghai \
  --resume \
  --source auto \
  --save-per-chunk \
  --verify-db \
  --repair-missing \
  --strict-completeness
```

下载完成后，建议独立验证本地数据库覆盖是否完整：

```bash
python scripts/verify_okx_history.py \
  --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL \
  --interval 1m \
  --start 2025-01-01 \
  --end 2025-01-03 \
  --timezone Asia/Shanghai \
  --strict
```

回测脚本默认会在 `BacktestingEngine.load_data()` 前做数据完整性 preflight。推荐命令：

```bash
python scripts/backtest_okx_mhf.py \
  --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL \
  --start 2025-01-01 \
  --end 2026-03-31 \
  --timezone Asia/Shanghai \
  --capital 5000 \
  --rate 0.0005 \
  --slippage-mode ticks \
  --slippage 2
```
