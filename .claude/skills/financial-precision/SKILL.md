---
name: financial-precision
description: 财务精度分界纪律——研究/回测 float64 可用（误差量级经一次性诊断确认远小于判定阈值）、实盘/对账/与交易所交互强制 Decimal；含 Decimal 安全构造、盈亏/手续费/保证金/funding 结算工具与 float↔Decimal 边界转换规范。任何涉及金额计算的代码（研究记账、审计对账、未来 live 域激活时）必须使用。
---

# financial-precision — 财务精度分界 + Decimal 工具

## 0. 流水线位置

**横切技能，非单站**（见 `.claude/skills/PIPELINE.md` §2 挂载表）：
- **研究记账（第 3–5 站）**：拟合/单次检验产出金额数字处——float64 可用性
  的依据与边界在此声明（见第 1 段分界）。
- **审计重推（audit 域）**：对账类核对的核对侧走 Decimal。
- **未来 live 域**：激活时强制（与交易所交互的一切金额计算）。

上游输入：任何产生金额的计算。下游产出：判定与对账可依赖的金额数字。

## 1. 何时用（核心是"分界"，不是"一律 Decimal"）

- **研究/回测阶段：float64 可用**。前提：误差量级已确认远小于判定阈值——
  本项目已显式确认（本目录 `DIAGNOSTIC_20260713.md`：纯 float64 累加误差
  ~$1.4×10⁻¹⁰，比 $0.01 小 8 个数量级、比最小金额 gate（forward K1 $1,296）小
  ~13 个数量级；此前 `b2_4h_pnl_audit` 独立重推逐分一致已是间接证明）。
  **强制全链 Decimal 会使数值密集回测慢约一个数量级**（诊断实测纯算术回路
  7×，整链更高），对研究是纯摩擦、无判定收益。
- **实盘/对账/与交易所交互：强制 Decimal**。交易所按 Decimal 结算，float
  对账**必然**出现分差（手续费、保证金、成交金额、funding 结算逐笔累积）；
  vnpy 内部多用 Decimal，口径必须一致。
- **边界场景**：研究结果要与交易所流水逐笔核对时（审计域的对账类任务），
  **核对侧用 Decimal**（研究侧产物可以是 float，进入核对即过
  `to_decimal(x, quantize=...)` 边界）。
- **重跑诊断的条件**：标的量级/笔数比诊断基准（470 笔/~$68k）增长 100 倍
  以上时，重跑 `diagnostic_float_error.py` 再声明 float 可用。

## 2. 怎么用

配套 `precision_toolkit.py`（零第三方依赖，可独立 import；
自检 `python precision_toolkit.py` 冒烟 15/15 PASS 才可用）：

| 函数 | 用途 |
|---|---|
| `safe_decimal(x)` | Decimal 安全构造：只收 str/int/Decimal，**float 直构一律 TypeError**（`Decimal(0.1) ≠ Decimal('0.1')` 陷阱） |
| `to_decimal(x, quantize, rounding=…)` | **float 进入 Decimal 域的唯一合法入口**，强制显式 quantize（误差上界=声明末位的一半，且是声明过的） |
| `pnl(entry, exit, qty, side)` | 毛盈亏，方向符号规范 side=`'long'`/+1、`'short'`/−1（与冻结引擎同式） |
| `fee(notional, rate, exp, rounding)` | 单腿手续费；舍入规则**参数化**（默认 ROUND_HALF_UP@8 位并注释为什么；交易所写明截断则传 ROUND_DOWN） |
| `margin(notional, leverage)` | 保证金；除法非终止 → 强制 quantize |
| `funding_settle(notional, rate, side)` | 单次 funding 结算现金流；口径 = `b2_4h_pnl_audit` 已验证口径（**rate>0 多头付/空头收**，rate<0 反之） |
| `dsum(items)` | Decimal 求和（float 混入即抛错） |
| `amounts_equal(a, b, exp)` | 金额比较：两侧 quantize 后比，**不用裸 ==** |

使用规范（违反即第 3 段的失效模式）：
1. Decimal 域内不得混入 float 运算；float 只能经 `to_decimal` 入域。
2. 金额比较用 quantize 后比较（`amounts_equal`），不用 `==` 裸比。
3. 对账时舍入规则按交易所文档显式传参，不靠默认值蒙对。
4. 口径不自创：pnl/fee/funding 公式与冻结引擎 `build_trades` 及其审计
   （`b2_4h_pnl_audit_20260628`）1:1，工具坏了修工具（过冒烟另行 commit），
   不复制改造。

## 3. 怎么失效

- **float 直构 Decimal**：为什么——`Decimal(0.1)` 精确继承二进制表示误差
  （0.1000000000000000055511…），字符串构造才是十进制 0.1；错误在构造瞬间
  固化，下游全链带毒且不再可检测。
- **Decimal 与 float 混算静默降级**：为什么——Python 中 `Decimal * float` 抛
  TypeError（保护有效），但经 `float()` 包裹或 numpy 广播（object dtype 之外）
  会绕过保护静默丢精度——最危险的不是报错的混算，是不报错的那种。
- **舍入规则不匹配交易所**：为什么——Python 默认 ROUND_HALF_EVEN（银行家
  舍入）统计无偏，但交易所常用四舍五入/截断；规则不匹配则**逐笔分差系统性
  同向累积**成对账缺口，且每笔都"看起来只差一分"。
- **quantize 缺失**：为什么——链路含除法/长乘链即产生非终止或超长小数，
  不同精度的 Decimal 比较/相加产生虚假不等（`1/3×3 ≠ 1`），金额判等随机翻车。
- **性能悬崖**：为什么——全链 Decimal 使数值密集回测慢约一个数量级（诊断
  实测纯算术 7×，数组运算被迫逐元素化后更高）——**这正是研究阶段不强制
  Decimal 的原因**，分界的依据而非缺陷。
- **研究与实盘口径不一致而不自知**：为什么——回测 float、实盘 Decimal，若无
  本 skill 的边界规范（`to_decimal` 唯一入口 + 舍入显式），上线后绩效与回测
  系统性偏离，且偏离会被误诊为"策略失效"而非核算口径差。

## 素材出处

- `research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_pnl_audit_20260628/`
  （fee/funding 口径的独立重推核准；"逐分一致"= float 可用的间接证明）
- 本目录 `DIAGNOSTIC_20260713.md`（浮点误差量级显式确认：路径 A 差额 $0.0000，
  float64 累加误差 ~1.4×10⁻¹⁰；重跑条件写死）
- `scripts/research_trend_baseline.py::build_trades`（冻结引擎，金额公式的
  唯一权威口径；本 skill 不修改它，只对齐它）
- MR-5m 实盘工程参考：`scripts/run_mr_5m_direct.py`（IOC 下单/bills 对账栈，
  实盘侧"交易所按 Decimal 结算"的实战出处）
