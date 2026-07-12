# AGENTS — 多 Agent 域分工总说明书

> 本仓库按四个域组织 agent 工作：**data_engineering / research / audit / forward**。
> 每域有域宪法（CLAUDE.md）定其铁律；本文档定**分工、路由与跨域协作**。
> 顶层 `CLAUDE.md` 是全域上位法；方法论见 `docs/METHODOLOGY.md`；
> 流水线与 skills 见 `.claude/skills/PIPELINE.md`。

## 1. 四域职责 / 权限 / 边界

| 域 | 宪法 | 职责 | 权限 | 硬边界 |
|---|---|---|---|---|
| **data_engineering** | `data_engineering/CLAUDE.md` | 下载/校验/入库/manifest；数据真实性可证伪核对 | 写 `data/`、写 manifest；只读两个 DB 之外的新库建设 | 污染库绝对不碰；`.env` 数据环境不得隐式继承；大文件不入 git |
| **research** | `research/CLAUDE.md` | 前置研究/回测/策略验证，走 PIPELINE 九站 | 读可信数据源；写 `reports/<line>/` 与研究脚本 | 不改归档实盘脚本与基准引擎；不碰 forward 资产；skill 到站必用 |
| **audit** | `audit/CLAUDE.md` | 核算诚实性/结论复核/数据取证 | 只读一切被审计对象；写审计脚本与审计产物 | 只核对不修改；独立代码路径；不判策略生死 |
| **forward** | 规则见下 §2（宪法待第二批迁移落地为 `forward/CLAUDE.md`） | B2_4h 零成本前向观察（VPS cron 自主运行） | Claude 仅本地开发与文档 | 见 §2，最严 |

## 2. forward 域规则（本步骤不动 `forward/`，规则先立此处）

- **三角色分离铁律**：开发（Claude，本地）/ 部署（**用户手动**，VPS）/
  运行（VPS cron，自主）。**Claude 绝不 SSH、绝不操作 VPS**；若被要求"部署"，
  只更新 `forward/VPS_DEPLOYMENT_MANUAL.md`，由用户执行。
- **config_frozen.json 绝不修改**（SHA256 `001b0d9e…0012e6`；改动 = 新系统、
  gate 作废）。`baseline_distribution.json`、`gates_preregistered.md` 同级冻结。
- **只读监控**：对前向系统的一切分析都是只读（state/ledger 在 VPS 本地，
  gitignored——本地仓库不含运行态，勿臆造）。
- **路径依赖不可破坏**：`scripts/forward_b2_4h.py` + 两个引擎脚本
  （`research_trend_baseline.py` / `research_trend_validation_r2.py`）+ `forward/`
  的相对关系是 VPS 运行依赖；任何重构移动它们须先更新部署手册并由用户重部署。
- gate 判定由预注册文档自动裁决（K1/K2/K3 KILL、U1/U2 UPGRADE），
  不做临场解释。

## 3. 任务路由（什么任务归哪域）

| 任务形态 | 归属 |
|---|---|
| "下载 X 数据 / 建新数据集 / 验证数据真伪" | data_engineering |
| "研究 X 想法 / 回测 / 验证假设 / 判生死" | research（走 PIPELINE） |
| "核对 X 数字是否正确 / 复核 Y 结论 / 审查核算" | audit |
| "前向系统状态 / gate 触发 / 部署变更" | forward（Claude 只动文档与本地代码） |
| 跨域文档（PROJECT_GUIDE/METHODOLOGY/CODEX 维护） | 产出研究的那个域按外科手术规则执行 |

判断优先级：涉及 forward 冻结资产 → 先按 forward 规则（最严）；
涉及数据真实性 → 先 data_engineering 核对再进 research。

## 4. 跨域协作规则

- **research 需要新数据** → 移交 data_engineering：产出 manifest（server 字段
  + sha256）+ 可证伪核对（端点审计 + 独立锚点）后，research 才可消费。
- **research 判活并产出冻结数字** → 移交 audit：独立重推核对
  （`.claude/skills/audit-independent/`）后数字才可长期引用/进 gate。
- **audit 发现数据源问题** → 移交 data_engineering 走可证伪核对流程；
  发现核算错误 → 报告修正值，由 research 域决定对结论的影响（audit 不改）。
- **任何域触及 forward 资产** → 停，按 §2 处理。
- 跨域交接一律有**落盘交接物**（manifest / 审计 ledger / 判决书），不做口头交接。

## 5. 共同纪律（全域适用，源自顶层 CLAUDE.md）

- 数据环境铁律（唯一可信源 / 污染库红线）。
- 铁律 A（gate 事后不改门）/ B（多重检验打折前置）/ C（正偏不用 Sharpe 主判）。
- 保守优于激进；统计显著性优先；判死判活同标准；措辞纪律
  （资源关闭≠证伪、程序污点≠自我修正）。
- 新研究一律走 `.claude/skills/PIPELINE.md`。
