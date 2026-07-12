# CLAUDE.md — OKX CTA 策略研究

## 项目
OKX 永续合约 CTA 策略研究，vnpy 框架。
**MR-5m 项目已关闭**（2026-06-11，mainnet 重验确认真实数据上无 edge，
test PF 0.826 / 毛利≈0，详见 `research/_closed/_synthesis/MR5M_postmortem.md`）。
**当前阶段：基于 `database_mainnet.db` 干净数据的新策略研究。**
- `scripts/run_mr_5m_direct.py`：MR-5m 实盘脚本，**已关停归档**。保留为工程参考
  （IOC 下单 / force_close / bills 对账等执行栈经实战验证），仍不随意修改。
- 历史标的与参数（BTC/ETH/SOL/LINK/DOGE 5 币、LB=24/ATR=14 Wilder/止损 1.0×ATR/
  midline 止盈）作为归档记录见 `docs/PROJECT_GUIDE.md`，对新研究无约束力。

## 数据环境铁律（最高优先级，违反即停）
- **唯一可信回测数据源：`.vntrader/database_mainnet.db`**（OKX mainnet 公开 REST 重建，
  3.4 年 1m 零缺口，含 `download_meta` 来源元数据）。
- 旧污染库已改名为 `.vntrader/database_DEMO_CONTAMINATED.db`（原 `database.db`，
  2026-06-11 修补改名以防呆；取证：`reports/regime/data_contamination_forensics_20260610/`），
  为**已确认污染的 OKX DEMO 行情**，仅作取证/对比基准，**严禁用于任何回测或研究**。
- mainnet 库已于 2026-06-11 通过 **Binance 全量交叉验证**（205 个月度文件、5 币
  1,790,880 根/币 1:1 对齐，PASS：合成形态 11 起全部为 Binance 同向证实的真实事件、
  无月度脱锚、median 偏差 0.03%，详见 `reports/regime/data_trust_closure_20260611/`）；
  funding 数据同日通过正式核查并补齐至 2026-06-11。
- mainnet 库备份：`~/backups/database_mainnet_20260611.db.gz`（gzip ~273MB），
  原库 SHA256 `a6d6928dbdec108f54ebc413ec84344d3e9cde5f4f54dd07b5adec36f573495d`；
  还原校验：`gunzip -c <备份> | sha256sum` 须等于该值。
- 任何写库脚本必须：启动时向 stdout 打印数据环境（MAINNET/DEMO）、manifest 写
  `server` 字段、新数据与外部独立源抽样交叉验证（≥3 随机日逐 bar）后才可用于研究。
- `.env` 的 `OKX_SERVER` **不得被任何数据下载脚本隐式继承**；数据环境必须显式
  命令行指定（本次事故根因即隐式继承，见复盘第 3 节）。
- 新研究启动门槛：`research/_closed/_synthesis/MR5M_postmortem.md` 第 8 节检查清单**逐项过一遍**。

## 回测引擎（保真复刻实盘）
- `scripts/backtest_mr_5m_compare.py` — 多币种/多方案对比引擎，费用/成交/指标口径与实盘 1:1：
  Wilder ATR、当根浮动止损、close±1tick taker 出场、入场 close 限价(maker返佣)、
  整数张数(ctVal)、maker −0.002% / taker 0.05%、不含 funding。引擎地位不变。
- 复用其数据加载与指标，勿重复造轮子；**数据源一律指向 `database_mainnet.db`**
  （外层注入，参考 `scripts/research_mr5m_mainnet_baseline.py` 的做法）。报告产物在 `reports/`。

## 历史研究发现（已解除约束力）
> ⚠️ 以下结论（含全部"死胡同"清单：exit 优化全败 / 熔断器 74 配置全败 /
> 趋势跟踪 10+ 族全败 / 过滤器 36 组合全败）**基于已确认污染的 DEMO 数据得出，
> 在 mainnet 干净数据上的有效性未知，不再作为新研究的硬边界**。保留原文仅作
> 历史记录与"待重验假设"的索引。详见 `research/_closed/_synthesis/MR5M_postmortem.md` 第 7 节。
1. **保留全部 5 币种（方案A），DOGE 不剔除。** 移除 DOGE 抹掉 ~$78k 历史利润而 PF 几乎不变
   (1.68→1.69)。DOGE 全周期 PF 1.67（第二强），近年走强。实盘单周亏损是噪声，不可据此决策。
2. **2024 是全市场 MR 逆风年**（所有币种该年 PF 最弱），非个别币种问题。
   <!-- 2026-06-11：已被 mainnet 重验证伪（真实数据各年均匀亏损，2023 反而最高） -->
3. **whipsaw（震荡扫损）才是 MR 的敌人，不是 trend。** BTC 涨幅最大的月反而最赚；
   失效月是低效率/低波动/高 stop% 的震荡月。"趋势强就停做"的过滤思路是错的。
4. **反应式状态熔断器无效**（连续止损 / 滚动PF，74 个配置全部负收益）。根因：失效月总亏损
   仅 −$10,690（占总盈利 ~3%），失效与健康月共用信号，事后暂停误伤的健康利润 > 省下的亏损。
   "靠少交易让指标变好"(looking better by trading less) 是陷阱。
   <!-- 注：此条的"少交易≠变好"教训在 mainnet 基线中再次得到独立印证（ATR filter 只能少亏） -->

## 当前阶段
MR-5m 已关闭。当前任务：基于 `database_mainnet.db` 的新策略研究。
新方向的选择是独立决策（不在复盘文档里预设）；任何新想法先过
复盘第 8 节检查清单，先跑裸基线（无过滤、固定仓位），毛利≈0 即放弃。

## 工作原则（用户强调，务必遵守）
- 不基于直觉/短期实盘样本拍板；统计显著性优先。
- **数据不支持就诚实说不支持**，不强行给"积极"结论。否定性结论与肯定性结论
  同等对待、同等固化（同等证据标准，进 PROJECT_GUIDE"已验证的核心事实"）。
- 不做参数过拟合；所有方案用同一段完整历史。gate 阈值在跑之前写死，
  **不可在结果出来后修改**（1.5× DD gate 差 0.023 不放宽 → 整条数据事故发现链的起点）。
- 保守优于激进。
- 回测假设（费率/滑点/成交）显式写进每份报告头部。
- 不修改归档实盘脚本与基准引擎；研究产出独立脚本 + markdown 报告 + 中间 jsonl 数据。
- 任何新策略线开题时必须估算验证周期（基于预期信号频率与 edge 强度估算统计判别
  所需样本长度），并确认其与可接受的资源配置匹配后才立项。验证周期超出可接受
  范围的方向，无论形态多有吸引力，不立项。（来源：趋势线关闭教训，
  `research/_closed/_synthesis/trend_line_closure_20260612.md`）
- **（铁律A）gate 事后改门根治**：涉及剔除/筛选/过滤的 gate（如集中度检验），其
  数值阈值 + "尾部交易须跨 ≥K 币、双样本均复现"这类结构要求，必须看结果前写死。
  预注册 gate 判死后，**不得通过重定义它复活死者**；要复活只能靠另一个事先写好的、
  不同的 gate。（根治 trend_validation_r2 的 V1→V1′ 事后移动球门：V1′ 系预注册 V1
  判死 15/15 后改门复活 5 幸存者，是程序污点而非"自我修正"，幸存证据等级因此降低。
  来源：趋势研究方法论评价 2026-06-22，`research/_closed/_synthesis/trend_methodology_hardening_20260622/`）
- **（铁律B）多重检验打折，立项算术前置**："从 N 配置选最优"的研究，报告必须含：
  搜索配置数 N（名义 N + 因试验相关折算的有效 N）+ deflated Sharpe。验证周期用
  打折后 Sharpe 且从对自相关诚实的 bootstrap SE 反推，不用幸存者 Sharpe、不用 iid
  闭式。双样本不能替代打折（双样本答"另一样本是否成立"，打折答"选择偏差抬高了多少"）。
  （来源同上：趋势线 15 配置选 B2_4h，原"15 年"系未打折 + iid 的乐观估计。）
- **（铁律C）正偏策略不用 Sharpe 做主要生死判据**：对结构性正偏（尾部收割、低胜率、
  长期 flat）策略，评价/关闭看整个 P&L 分布、尾部捕获效率、下行调整指标，Sharpe 仅
  参考之一。Sharpe 惩罚上行波动而上行波动是趋势本体，主导会系统性低估正偏策略、把
  优化推向均值回归味配置。（来源同上。）
- 记忆见 `~/.claude/.../memory/`（mr5m-* 系列，含完整事故链记录）。

## 研究产出后维护 PROJECT_GUIDE.md
`docs/PROJECT_GUIDE.md` 是项目的**活文档**，反映**当前最佳认知**，不是流水账。
每次研究产出后，按下面流程判断并执行。

**何时更新 PROJECT_GUIDE：**
- 研究结论翻转了某条已有叙述（例："ATR filter 是风控" → "不是风控"）。
- 研究新增了一条策略层认知（例："whipsaw 才是 MR 的敌人"）。
- 验证完了一条之前"已知未验证"的假设。
- 一句话标准：**一年后重读项目说明，没有这条会误导我或新来的 Claude → 进 PROJECT_GUIDE。**

**何时不更新 PROJECT_GUIDE：**
- 参数微调、单一币种细节实验、A/B 对比的中间结果。
- 这些只进 `reports/` 和必要的记忆文件。

**更新方式（外科手术式，不是追加）：**
1. 找到当前叙述被推翻或不完整的**具体行**。
2. **修订正文**为新的最佳认知（不是在末尾追加"现在认为……"）。
3. 在被修订处加 HTML 注释，**仅用于被推翻的叙述**（新增内容不需要 historical 注释）：
   `<!-- YYYY-MM-DD 更新：原 X 被 OOS 验证推翻，详见 reports/.../ -->`
4. 在对应 OOS 报告的 README 末尾追加"由本次验证产生的文档更新"一节，列出改动清单。
5. 若某文档考虑过但决定不动，在 OOS 报告里**显式列出"未改动文档及原因"**——避免"考虑过但不告诉你"的隐性决策。

**PROJECT_GUIDE 必须维护的两个章节：**
- **"已验证的核心事实"**：被回测/OOS 充分支持的策略层结论。
- **"已知未验证的假设"**：还没回测验证的关键假设，明确标注"未验证"。
  新研究做完后，从"未验证"区移到"已验证"区（如通过）或撤回（如推翻）。
- 现状：两节**已建立**（2026-06 数据事故系列研究期间），正常按上述流程维护即可。
  注意：其中基于 demo 数据"已验证"的条目，其证据基础已失效，移动/撤回时按
  外科手术规则处理（修订 + 注释，不悄悄删除）。

**改动颗粒度要求：**
- 外科手术式——原句修订 + 必要注释。
- 不新增长篇内容（避免文档臃肿）；不"顺手"修无关错误。
- 单次研究典型改动 **< 30 行**，不应大幅改写文档。

## 结构导航（2026-07 重构）

- **新研究一律走流水线**：`.claude/skills/PIPELINE.md`（策略的一生九站装配图，
  含死因库与闭环）。到站必用对应 skill（`.claude/skills/` 下 8 个：
  preregistration / noise-calibration / multiple-testing / cointegration /
  bootstrap-inference / ic-analysis / honest-verdict / audit-independent，
  各含四段式 SKILL.md + 经冒烟自检的配套代码/模板）。
- **四域宪法**：`data_engineering/CLAUDE.md`（数据域）、`research/CLAUDE.md`
  （研究域）、`audit/CLAUDE.md`（审计域）；forward 域规则暂立于
  `docs/AGENTS.md` §2（待第二批迁移落地为 forward/CLAUDE.md）。
- **docs/ 活文档**：`docs/METHODOLOGY.md`（方法论总纲——skills 的"为什么"层）、
  `docs/STRATEGY_CODEX.md`（策略代号图例）、`docs/AGENTS.md`（域分工与任务路由）。
