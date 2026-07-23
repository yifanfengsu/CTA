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

## 6. 域间通信协议（数据流 + 交接物清单）

> 本节把 §4 的跨域协作规则升级为显式协议：**每条域间通道规定唯一合法的
> 交接物，不经交接物的跨域调用即违规**（负面清单见 §6.4）。

### 6.1 数据流图

```
data_engineering ──► research ──► forward ──► (audit 只读观察)
                         │
                         └───────────────────► audit（独立重推）

  所有域 ──► core             单向依赖；core 永不反向依赖任何域
  所有域 ──► .claude/skills/  到站调用；skill 代码不得复制改造进域内

  forward ┈┈┈► live           未激活（虚线）；激活条件 = U1 触发 + 用户人工确认（§6.3）
```

### 6.2 交接物清单（协议核心）

| 通道 | 交接物 | 方向与权限 |
|---|---|---|
| data_engineering → research | manifest 登记（`server` 字段）+ sha256 核验过的数据文件 | research 只消费；**不自行下载/修数据**——发现数据问题 → 提数据域工单，不就地修 |
| research → forward | 冻结策略配置（config_frozen）+ 预注册 gate 文档 + 部署手册 | forward 只读挂载；research 不碰运行中的 forward |
| forward → audit | 日报 / 记账文件 / 日志 /（未来）快照 | audit 只读；**绝不调用研究引擎重跑作"核对"**——独立重推是 audit 唯一合法核对方式（b2_4h_pnl_audit 范式） |
| research → audit | 回测结果 / 报告 / 落盘明细（jsonl/json） | audit 只读原件，用独立代码路径重推 |
| 任何域 → core | `import core` 提供的共享函数 | core 的修改须跑全部 tests 且向后兼容（`core/CLAUDE.md` §2；单向依赖与前向 re-export 冻结约束见其 §1） |
| 任何域 → .claude/skills/ | 到 `PIPELINE.md` 对应站点调用 skill | **skill 代码不得被复制粘贴改造进域内**（防分叉——要改进 skill 就改 skill 本身并过冒烟，另行 commit） |
| forward → live | （未激活，预留）见 §6.3 | 激活前本表不含 live 通道的任何实义条目 |

### 6.3 live 域（未激活——预留节）

- **激活条件**：U1 触发（B2_4h 前向 UPGRADE gate，见 `forward/` 预注册文档）
  **+ 用户人工确认**，二者缺一不可。
- **届时宪法要点**（激活时落地为 live 域宪法，此处仅预留）：资金上限；
  紧急停止机制；风控日志独立于 live 本体存储；与 forward 物理隔离。
- **届时生命周期管理要点**（激活时落地，此处仅预留设计意图，零代码）：
  - **edge decay 监控**：rolling Sharpe 跟踪 edge 是否衰减；
  - **机制指标监控**（按活策略机制分别定义，监控机制本体而非仅业绩）：趋势查 momentum
    persistence（延续是否还在）、协整查 β 稳定性、VRP 查 premium 存在性；
  - **风险状态触发**：回撤超历史分位 → 触发**人工复查**（非自动加/减仓）。
  - **注**：前向阶段的等价需求**已由预注册 gate（K1/K2 KILL、U1/U2 UPGRADE）覆盖且
    更硬**——预注册阈值优于事后人工判断（事后判断易受确认偏误，见铁律 A）。故本节要点
    是 live 域**独有**的补充（前向不需要），**随 live 域一同激活**。
- **元规则**：激活前 `live/` **不得存在任何可执行代码**（含上述生命周期监控）——
  本节文字是 live 域唯一合法的存在形式。

### 6.4 违规负面清单（让边界可判定）

- **审计时 import 研究引擎复跑当核对** —— 循环论证：复跑只能复现引擎自身
  可能存在的同一个 bug。
- **研究脚本直接读交易所 API 补数据** —— 绕过数据域 manifest；数据环境事故
  的同款根因（隐式来源）。
- **把 skill 的 .py 复制进研究目录改两行用** —— 分叉后失效模式脱管，冒烟
  自检形同虚设。
- **任何域代码 import 另一个域的内部模块** —— 跨域必经 core 或本节交接物，
  无第三条路。
