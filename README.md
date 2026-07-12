# CTA — OKX 永续合约 CTA 策略研究仓库

OKX 永续合约（+ Deribit 期权 / 期现）系统化策略研究，vnpy headless 框架。
~12 条研究线已全部收敛（除 B2_4h 在做零成本前向观察外，其余全部 CLOSED/FAIL/NOT-VIABLE），
当前处于"方法论固化 + 新方向待开题"阶段。

## 结构导航

| 位置 | 职责 |
|------|------|
| `CLAUDE.md` | 项目最高约束：数据环境铁律、工作原则、铁律 A/B/C |
| `docs/` | 活文档：`PROJECT_GUIDE.md`（当前最佳认知）、`METHODOLOGY.md`（方法论总纲）、`STRATEGY_CODEX.md`（策略代号图例）、`AGENTS.md`（域分工）、`BOOTSTRAP.md`（环境自举）、`README_ARCHIVE.md`（旧 README 历史存档） |
| `.claude/skills/` | 研究流水线：`PIPELINE.md`（策略的一生九站装配图）+ 8 个到站必用 skill |
| `core/` | 共享库（数据 IO / DB 工具；前向依赖模块以 re-export 代理指向 `scripts/`） |
| `data_engineering/` | 数据域：下载 / 校验 / 导入脚本（宪法见其 `CLAUDE.md`） |
| `research/` | 研究域：`_closed/<market>/<line>/` 已关闭研究线（脚本+报告），`_synthesis/` 跨线综合文档 |
| `audit/` | 审计域：audit / postmortem / forensics 脚本 |
| `forward/` | ⚠️ B2_4h 前向观察系统（VPS 正在跑，冻结区，勿动） |
| `scripts/` | ⚠️ 前向冻结区：`forward_b2_4h.py` + 其 import 闭包（两引擎 + 基准引擎 + 共享工具），勿动 |
| `reports/` | 研究产物存档（regime 取证链 / 大宗历史产物 / 盘点报告） |
| `_archive/` | 遗留归档：MR-5m runner、demo 时代脚本与报告（历史证据，不删除） |
| `data/` `.vntrader/` | 数据资产（唯一可信回测源 `.vntrader/database_mainnet.db`；污染库严禁使用） |

## 怎么上手

1. 读 `CLAUDE.md`（铁律）→ `docs/PROJECT_GUIDE.md`（认知现状）→ `docs/METHODOLOGY.md`。
2. 环境自举：`docs/BOOTSTRAP.md`；健康检查 `make doctor`，测试 `make test`。
3. 新研究开题：一律走 `.claude/skills/PIPELINE.md` 流水线，先过
   MR5M 复盘（`research/_closed/_synthesis/MR5M_postmortem.md`）第 8 节检查清单。

## 数据环境（最高优先级）

- 唯一可信回测数据源：`.vntrader/database_mainnet.db`（mainnet 重建，Binance 全量交叉验证 PASS）。
- `.vntrader/database_DEMO_CONTAMINATED.db` 为已确认污染的 DEMO 行情，**严禁用于任何回测或研究**。
- 详细铁律见 `CLAUDE.md`「数据环境铁律」。

> 旧 README（1521 行历史命令与研究模块流水账，demo 时代为主）原文存档于
> `docs/README_ARCHIVE.md`，其中结论的证据基础多已失效，仅作历史记录。
