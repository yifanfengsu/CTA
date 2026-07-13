# GIT_WORKFLOW — 本仓库版本控制约定

> 本文档是**本仓库的具体约定**，不是 git 教程。机器强制部分见 `.githooks/pre-commit`
> （新 clone 后执行一次 `make git-hooks` 启用，见 §5）。

## 1. 提交规范

**格式**：`<域>: <动作> <对象>`

域枚举（与目录域对应）：

| 域 | 覆盖范围 |
|---|---|
| `research` | 研究脚本/回测/判定（`research/`、研究线产物） |
| `data` | 数据工程（`data_engineering/`、manifest、下载/校验） |
| `audit` | 审计/取证/复核（`audit/`） |
| `skills` | `.claude/skills/` 流水线与技能 |
| `docs` | `docs/` 活文档、README、CLAUDE.md |
| `core` | `core/` 共享库 |
| `forward` | 前向系统相关（最严，见 §2 分支模型） |
| `archive` | `_archive/` 归档动作 |
| `repo` | 仓库级（重构/hooks/CI 等，如本模块） |

**研究线的标准 commit 序列**（与 `.claude/skills/PIPELINE.md` 九站对应）：

1. `research(<线名>): prereg — <gate 摘要>` —— 预注册 commit，**先于任何结果产出**，
   同时打 `prereg/<研究名>` tag（§3）：tag 时间戳是"gate 先于结果写死"的 git 层公证
   （铁律 A 的机器面）。
2. `research(<线名>): result — <判定> <关键数字>` —— 结果 commit。
3. `research(<线名>)/docs: synthesis — <归宿>` —— 收尾/存档/文档外科手术。

**粒度**：一个逻辑单元一个 commit；禁止"WIP 大杂烩"。分批任务每批一个 commit，
任何一批可单独 revert（重构步骤 2 的 9-commit 链是范例）。

**message 正文**：涉及判定的 commit **必须含判定一行**（过/不过 + 关键数字），
使 `git log` 可直接当研究日志读。范例（既有历史）：
`research(flow-vs-price): FAIL — flow signal (cause) no edge over price (B2_4h), use price`。

## 2. 分支模型（双轨制）

| 轨道 | 适用 | 约定 |
|---|---|---|
| **直进 main** | 研究/文档/审计/skill 增补 | 单人研究仓库，低摩擦优先；commit 规范即门槛 |
| **必须开分支** | 结构性变更：重构/迁移/引擎改动/批量重命名 | 分支名 `restructure/<主题>`；本地验证全过才合入：①tests 红集与基线逐项一致 ②`make -n` 全 target 可解析 ③前向 `config_frozen.json` sha256 核对；合入用 `--no-ff` 保留分支边界 |
| **分支 + 人工 review** | 前向相关任何改动（如未来第二批迁移） | 分支名 `forward/<主题>`；合入前**用户亲自确认**（三角色分离铁律：Claude 不得自行合入前向变更） |

**禁止**：
- force push 到 origin/main（历史不可改写——研究日志的可信性依赖它）；
- rebase 已 push 的 commit。
- （未 push 的本地 commit 允许 amend/rebase——320MB 误提交正是靠未 push 前 amend 救回。）

## 3. tag 约定

| 前缀 | 含义 | 示例 |
|---|---|---|
| `freeze/<名>` | 冻结点（配置/引擎版本定格） | `freeze/b2-4h-forward` |
| `prereg/<研究名>` | 预注册公证（每个 prereg commit 必打） | `prereg/vrp-stageB` |
| `milestone/<名>` | 仓库级里程碑 | `milestone/restructure-v1` |

- tag **一律 annotated**（`git tag -a`），message 写明冻结/预注册内容摘要。
- push 带 tag：`git push origin main --tags`。

既有里程碑：`milestone/skills-v1`（2b141ca，skill 库 v1）、
`milestone/restructure-v1`（1ddd13b，重构步骤 2 完成）、
`freeze/b2-4h-forward`（b2db5ce，config_frozen 定格 = sha256 `001b0d9e…0012e6`）。

## 4. 大文件与数据策略

- git 只入：manifest / 报告 / 代码 / 小型结果 csv。
- 原始数据与 **>10MB** 产物一律 gitignore + manifest 登记（sha256 + 来源 + 路径）。
- ⚠️ **`.gitignore` 的路径在 `git mv` 后不会自动跟随**——迁移类分支合入前必须
  核对 ignore 路径。教训：2026-07-12 重构批次6 迁移 `reports/research/` 时，
  320MB `exit_variant_trades.csv` 因 ignore 规则仍指旧路径而被误提交，靠 push 前
  终检抓回（`rm --cached` + amend）。hooks 的大文件拦截（§5 检查1）是独立于
  ignore 的第二道防线，正为此事故而设。

## 5. hooks 启用（新 clone 后必做一次）

```bash
make git-hooks    # = git config core.hooksPath .githooks
```

`.githooks/pre-commit` 四道检查（详见脚本内注释）：

1. **大文件拦截**（硬）：staged 单文件 >10MB 拒绝；override：`GIT_ALLOW_LARGE=1`。
2. **敏感文件拦截**（硬）：`.env`/`*.pem`/`*.key`/`id_rsa*` 拒绝；staged 内容命中
   secret 形态（保守正则）警告；override：`GIT_ALLOW_SECRET=1`。
3. **前向冻结区守卫**（硬，本仓库特有）：`scripts/` 11 个冻结 `.py`、`forward/**`、
   `data/funding/okx/**` 命中即拒绝；override：`GIT_ALLOW_FROZEN=1`，且须走
   `forward/<主题>` 分支 + 人工 review。这条把"前向零触碰"从纪律变成机器强制。
4. **分支保护提示**（软）：main 上 staged >30 文件或含重命名 → 警告应走
   `restructure/` 分支，不拦截。

**护栏不是牢笼**：每道硬检查都有显式 override 环境变量；override 的使用本身
应写进 commit message（为什么越过防线）。
