#!/usr/bin/env python3
"""financial-precision skill — Decimal 财务计算工具（可独立 import，零第三方依赖）。

金额口径出处（与本项目已验证口径 1:1，不自创）：
  scripts/research_trend_baseline.py::build_trades（冻结引擎，零修改）经
  research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_pnl_audit_20260628/
  独立重推审计核准（独立路径 vs 引擎最大单笔差 $5.0e-5 = 4 位小数存储舍入地板）：
    gross        = (exit − entry) × qty × side     side = +1 多 / −1 空
    fee          = notional × rate                  （现金流为负；进出两腿都扣）
    funding 现金 = −rate × settle_px × qty × side   rate>0 ⇒ 多头付、空头收
    net          = gross − fees + funding_cash

核心规则（SKILL.md 第 2 段的代码面）：
  1. Decimal 只从 str/int 构造（safe_decimal）；float 唯一合法入口 =
     to_decimal(x, quantize=...)，强制显式声明精度。
  2. Decimal 域内不得混入 float 运算（Python 原生抛 TypeError——这是保护不是缺陷；
     危险的是 float() 包裹或 numpy 广播绕过它静默降级）。
  3. 金额比较一律 quantize 后比较（amounts_equal），不裸 ==（链路含除法即产生
     非终止小数，裸 == 虚假不等）。

自检: python precision_toolkit.py   （冒烟测试全 PASS 才可交付/使用）
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP

__all__ = [
    "safe_decimal", "to_decimal", "dsum", "pnl", "fee", "margin",
    "funding_settle", "amounts_equal",
    "ROUND_HALF_UP", "ROUND_DOWN", "ROUND_HALF_EVEN",
]

# 交易所落账常用精度：OKX 资金账单按结算币种 8 位小数落账（USDT 本位）。
# 对账时以交易所流水实际精度为准，显式覆盖。
EXCHANGE_EXP = "0.00000001"


def safe_decimal(x) -> Decimal:
    """str/int/Decimal → Decimal；float（含 numpy 标量）与 bool 一律拒绝。

    为什么拒绝 float：Decimal(0.1) 精确继承二进制表示误差
    （= 0.1000000000000000055511…），Decimal('0.1') 才是十进制 0.1。
    float 必须走 to_decimal(x, quantize=...) 显式声明精度后进入 Decimal 域。
    """
    if isinstance(x, bool):
        raise TypeError("bool 不是金额，禁止进入 Decimal 域")
    if isinstance(x, Decimal):
        return x
    if isinstance(x, int):
        return Decimal(x)
    if isinstance(x, str):
        return Decimal(x)
    raise TypeError(
        f"禁止从 {type(x).__name__} 直构 Decimal；"
        "float 请走 to_decimal(x, quantize=...)（唯一合法入口）"
    )


def to_decimal(x, quantize: str, rounding=ROUND_HALF_EVEN) -> Decimal:
    """float↔Decimal 边界的唯一合法入口：强制显式 quantize。

    float 经 repr()（最短往返表示 = 打印值而非内存二进制值）进入，再 quantize
    到声明精度——误差上界 = 声明末位的一半，且是**声明过的**，不再是静默的。
    str/int/Decimal 也接受（统一入口方便逐列转换），同样 quantize。
    """
    if isinstance(x, float):
        d = Decimal(repr(x))
    else:
        d = safe_decimal(x)
    return d.quantize(Decimal(quantize), rounding=rounding)


def dsum(items) -> Decimal:
    """Decimal 求和（起始值 Decimal(0)；逐项过 safe_decimal，float 混入即抛错）。"""
    total = Decimal(0)
    for it in items:
        total += safe_decimal(it)
    return total


def _side_sign(side) -> Decimal:
    """方向符号规范：'long'/+1 → +1，'short'/−1 → −1（与冻结引擎 side 语义一致）。"""
    if isinstance(side, str):
        s = side.strip().lower()
        if s == "long":
            return Decimal(1)
        if s == "short":
            return Decimal(-1)
        raise ValueError(f"side 字符串仅接受 'long'/'short'，得到 {side!r}")
    if isinstance(side, bool):
        raise TypeError("bool 不是方向")
    if isinstance(side, (int, Decimal)) and side in (1, -1):
        return Decimal(int(side))
    raise ValueError(f"side 仅接受 'long'/'short'/+1/−1，得到 {side!r}")


def pnl(entry, exit_, qty, side, exp: str | None = None,
        rounding=ROUND_HALF_EVEN) -> Decimal:
    """毛盈亏 = (exit − entry) × qty × side（引擎 build_trades 同式）。

    qty = 币量（张数 × ctVal）。exp=None 返回精确积（研究聚合用）；
    对账时传交易所落账精度。
    """
    v = (safe_decimal(exit_) - safe_decimal(entry)) * safe_decimal(qty) * _side_sign(side)
    return v if exp is None else v.quantize(Decimal(exp), rounding=rounding)


def fee(notional, rate, exp: str = EXCHANGE_EXP, rounding=ROUND_HALF_UP) -> Decimal:
    """单腿手续费金额 = |notional| × rate（非负；现金流方向由调用方记账，
    引擎口径为进出两腿各计一次、记负现金流）。

    默认 ROUND_HALF_UP 而非 Python 默认 ROUND_HALF_EVEN：为什么——HALF_EVEN
    （银行家舍入）统计无偏但与交易所流水普遍对不平；四舍五入（HALF_UP）是逐笔
    对账实践中最常对得平的规则。若交易所文档写明截断，显式传 ROUND_DOWN。
    参数存在的原因 = 各所规则不同且必须显式，不允许隐式默认蒙对。
    """
    v = abs(safe_decimal(notional)) * safe_decimal(rate)
    return v.quantize(Decimal(exp), rounding=rounding)


def margin(notional, leverage, exp: str = EXCHANGE_EXP,
           rounding=ROUND_HALF_UP) -> Decimal:
    """保证金 = |notional| / leverage。除法可产生非终止小数，故强制 quantize
    （失效模式：quantize 缺失 → 下游裸 == 虚假不等）。"""
    v = abs(safe_decimal(notional)) / safe_decimal(leverage)
    return v.quantize(Decimal(exp), rounding=rounding)


def funding_settle(position_notional, funding_rate, side, exp: str | None = None,
                   rounding=ROUND_HALF_EVEN) -> Decimal:
    """单次 funding 结算现金流（正 = 收入，负 = 支出）。

    口径 = b2_4h_pnl_audit 已验证口径（冻结引擎 funding_cost 的单次结算项）：
        cash = −rate × |notional| × side，side = +1 多 / −1 空
    ⇒ rate>0：多头付（负现金流）、空头收；rate<0 反之。
    position_notional 应为结算时刻名义价值（引擎口径 settle_px = 结算前一分钟
    1m 收盘 × 币量）。exp=None 返回精确积；对账时传交易所落账精度。
    """
    v = -(safe_decimal(funding_rate) * abs(safe_decimal(position_notional))
          * _side_sign(side))
    return v if exp is None else v.quantize(Decimal(exp), rounding=rounding)


def amounts_equal(a, b, exp: str = "0.01", rounding=ROUND_HALF_EVEN) -> bool:
    """金额相等判定：两侧 quantize 到同一精度后比较。裸 == 只在两侧都是精确
    十进制时可靠；链路含除法/长乘链后必须走本函数。"""
    q = Decimal(exp)
    return (safe_decimal(a).quantize(q, rounding=rounding)
            == safe_decimal(b).quantize(q, rounding=rounding))


# ── 冒烟自检（已知值 + 舍入边界 + 混算拦截；全 PASS 才可交付） ────────────────
def _smoke() -> int:
    results: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        results.append((name, bool(cond)))

    def raises(fn, exc) -> bool:
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    # 1–2 str 构造精确性 vs float 直构陷阱
    check("T01 str构造精确: Decimal('0.1')*3 == Decimal('0.3')",
          safe_decimal("0.1") * 3 == Decimal("0.3"))
    check("T02 float直构陷阱确认: Decimal(0.1) != Decimal('0.1')",
          Decimal(0.1) != Decimal("0.1"))
    # 3–4 safe_decimal 拒绝 float/bool
    check("T03 safe_decimal(0.1) 抛 TypeError",
          raises(lambda: safe_decimal(0.1), TypeError))
    check("T04 safe_decimal(True) 抛 TypeError",
          raises(lambda: safe_decimal(True), TypeError))
    # 5–6 to_decimal 边界入口
    check("T05 to_decimal(0.1,'0.0001') == Decimal('0.1000')",
          to_decimal(0.1, "0.0001") == Decimal("0.1000"))
    check("T06 to_decimal 缺 quantize 抛 TypeError",
          raises(lambda: to_decimal(0.1), TypeError))  # type: ignore[call-arg]
    # 7–8 pnl 已知值（B2_4h 真实成交，B2_4h_trades.jsonl 第 3 行 / 末 2 行）
    check("T07 pnl 多头 BTC 实例 = 281.953",
          pnl("21220.5", "21820.4", "0.47", "long") == Decimal("281.953"))
    check("T08 pnl 空头 LINK 实例 = 840.480",
          pnl("9.71", "8.894", "1030", "short") == Decimal("840.480"))
    # 9 fee 已知值（同一 BTC 成交的双腿，引擎存档 fee_usd=10.1146）
    fee_both = fee("9973.635", "0.0005") + fee("10255.588", "0.0005")
    check("T09 fee 双腿 BTC 实例 quantize4 = 10.1146",
          amounts_equal(fee_both, Decimal("10.1146"), "0.0001"))
    # 10 fee 舍入边界（0.05005 恰为 4 位处半值）
    check("T10 fee 舍入边界: HALF_UP→0.0501 / DOWN→0.0500",
          fee("100.10", "0.0005", exp="0.0001", rounding=ROUND_HALF_UP)
          == Decimal("0.0501")
          and fee("100.10", "0.0005", exp="0.0001", rounding=ROUND_DOWN)
          == Decimal("0.0500"))
    # 11 margin 已知值 + 非终止除法
    check("T11 margin: 10000/10=1000.00000000; 100/3@0.01=33.33",
          margin("10000", "10") == Decimal("1000.00000000")
          and margin("100", "3", exp="0.01") == Decimal("33.33"))
    # 12 funding 符号四象限（审计口径: rate>0 多付空收）
    check("T12 funding 符号: 多+付/空+收/多−收/空−付",
          funding_settle("10000", "0.0001", "long") == Decimal("-1")
          and funding_settle("10000", "0.0001", "short") == Decimal("1")
          and funding_settle("10000", "-0.0001", "long") == Decimal("1")
          and funding_settle("10000", "-0.0001", "short") == Decimal("-1"))
    # 13 混算静默降级拦截
    check("T13 Decimal*float 抛 TypeError",
          raises(lambda: Decimal("1") * 0.5, TypeError))
    # 14 quantize 后比较 vs 裸 ==
    third = Decimal(1) / Decimal(3) * 3
    check("T14 除法链: 裸==不等而 amounts_equal 相等",
          third != Decimal(1) and amounts_equal(third, Decimal(1), "0.00000001"))
    # 15 dsum 守卫
    check("T15 dsum(['0.1','0.2'])==0.3 且 dsum([0.1]) 抛 TypeError",
          dsum(["0.1", "0.2"]) == Decimal("0.3")
          and raises(lambda: dsum([0.1]), TypeError))

    n_pass = sum(ok for _, ok in results)
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nsmoke: {n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
