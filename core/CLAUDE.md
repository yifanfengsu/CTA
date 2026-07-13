# core/ — 共享库宪法

> core **不是域**，是四域共同依赖的市场无关共享库。顶层 `CLAUDE.md` 为上位法；
> 域间通道规则见 `docs/AGENTS.md` §6.2（"任何域 → core"通道的引用落点即本文件）。

## 1. 单向依赖铁律

- 方向唯一：**所有域 → core；core 永不 import 任何域**
  （`research/`、`data_engineering/`、`audit/`、`forward/`）的模块。
- 唯一例外（方向已冻结）：`core/data_io/` 的两个**薄 re-export 代理**
  （`common_runtime`、`history_time_utils`）指向 `scripts/` 前向冻结区真身——
  那是前向 import 闭包资产，不是域代码；此例外不得再增。

## 2. 修改纪律

- core 的任何修改须：① 跑**全部 tests** 且全过（`make test` =
  `unittest discover -s tests -t .`）；② **向后兼容**——不缩小既有公开
  import 面（模块路径/函数签名），废弃走"新增替代 + 过渡期"，不直接删改。
- 代理模块**零语义修改**：它们只做 re-export；要改真身 = 触碰前向冻结区，
  按 `docs/AGENTS.md` §2 forward 规则（分支 + 用户人工 review）。
- 新函数入 core 的门槛：**≥2 个域需要**；单域专用逻辑留在域内，
  不为"看起来通用"提前抽象。

## 3. 内容边界

- 只放市场无关工具（数据 IO / 时间工具 / DB 初始化与体检）。
- 不放：策略逻辑、gate 判定、研究结论；金额计算规范不在 core 重复实现
  （走 `.claude/skills/financial-precision/`，防分叉）。
- 大文件/数据不入 core（`docs/GIT_WORKFLOW.md` §4）。

## 4. 现状清单（2026-07-13）

| 模块 | 性质 |
|---|---|
| `core/data_io/common_runtime.py` | 代理 → `scripts/common_runtime.py`（冻结真身） |
| `core/data_io/history_time_utils.py` | 代理 → `scripts/history_time_utils.py`（冻结真身） |
| `core/data_io/history_utils.py` | 真身（sqlite 历史覆盖/修补规划） |
| `core/db/init_db.py` | 真身（库初始化） |
| `core/db/doctor.py` | 真身（环境体检，`make doctor`） |
