#!/usr/bin/env python3
"""MR-5m dynamic position sizing — Task C, stages C2 + C3.

C2: three-tier sizing on atr_ratio rolling percentile (N=200, the most robust
window from C1-A). Unlike C1-A (down-only), C2 sizes UP on top signals so total
notional stays ~constant (redistribution, not leverage). Constraint: total
notional within +/-10% of the $500-flat baseline ($24.99M).

  C2-1 conservative : pct<50 ->$250, 50-80 ->$500, >80 ->$750
  C2-2 aggressive   : pct<50 ->$200, 50-80 ->$500, >80 ->$1000
  C2-3 mild         : pct<50 ->$300, 50-80 ->$500, >80 ->$700

Then pick the best C2 scheme (within the notional constraint, by net + Sharpe)
and run C3 robustness on it ONLY:

  C3-1 walk-forward : learn atr_ratio cutpoints from IS (2023-2024), apply to
                      OOS (2025-01..2026-05); compare OOS PF vs IS PF.
  C3-2 param robust : neighbouring percentile pairs around the chosen split.
  C3-3 per-symbol   : each of 5 symbols, dynamic vs flat.

NOT path-dependent: sizing never changes which trades occur, so we rescale the
exact baseline trade sequence. Reuses the validated engine. Live script untouched.

Usage: python scripts/backtest_mr_5m_dynsize_c2c3.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_mr_5m_compare import SYMBOLS, load_1m, r5, parse_history_range
from backtest_mr_5m_dynsize import (
    collect_trades, evaluate, BASE_NOTIONAL, SYM_NAMES, DB, START, END,
    TIMEZONE, OUT_DIR,
)

# ---- C2 schemes: (label, low_pct, high_pct, $low, $mid, $high) ----
C2_SCHEMES = [
    ("C2-1 保守", 0.50, 0.80, 250.0, 500.0, 750.0),
    ("C2-2 激进", 0.50, 0.80, 200.0, 500.0, 1000.0),
    ("C2-3 温和", 0.50, 0.80, 300.0, 500.0, 700.0),
]
IS_END = np.datetime64("2024-12-31")   # IS: ..2024-12-31 ; OOS: 2025-01-01..
NOTIONAL_TOL = 0.10                      # +/-10% of baseline


def pct_sizer(lo, hi, d_low, d_mid, d_high, key="pr200"):
    """Tier by rolling percentile rank (online/causal). nan pct -> mid (neutral)."""
    def f(t):
        p = t[key]
        if p != p:
            return d_mid
        if p < lo:
            return d_low
        if p < hi:
            return d_mid
        return d_high
    return f


def cut_sizer(c_lo, c_hi, d_low, d_mid, d_high):
    """Tier by FIXED atr_ratio cutpoints (used for walk-forward: cutpoints learned IS)."""
    def f(t):
        a = t["atr_ratio"]
        if a != a:
            return d_mid
        if a < c_lo:
            return d_low
        if a < c_hi:
            return d_mid
        return d_high
    return f


def fmt_row(label, m, base):
    return (f"| {label} | {m['trades']:,} | ${m['notional']:,.0f} | "
            f"{m['notional']/base['notional']*100-100:+.0f}% | ${m['net']:,.0f} | "
            f"{m['net']/base['net']*100-100:+.0f}% | {m['pf']:.2f} | ${m['dd']:,.0f} | "
            f"{m['sharpe']:.2f} | {m['eff_bps']:.1f} | "
            f"{m['eff_bps']/base['eff_bps']*100-100:+.0f}% |")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hr = parse_history_range(START, END, timedelta(minutes=1), TIMEZONE)
    print("Collecting baseline trades + atr_ratio percentiles...", flush=True)
    by_sym = {}
    trades = []
    for nm in SYM_NAMES:
        b5 = r5(load_1m(SYMBOLS[nm][0], hr, DB), 5, hr)
        t = collect_trades(nm, b5)
        by_sym[nm] = t
        trades += t
        print(f"  [{nm}] {len(t):,} trades", flush=True)

    base = evaluate(trades, lambda t: BASE_NOTIONAL)
    print(f"\nBASELINE ($500 flat): trades={base['trades']:,} notional=${base['notional']:,.0f} "
          f"net=${base['net']:,.0f} PF={base['pf']:.2f} DD=${base['dd']:,.0f} "
          f"Sharpe={base['sharpe']:.2f} eff={base['eff_bps']:.1f}bps")

    # ---------- C2 ----------
    print("\n== C2: three-tier (atr_ratio pr200) ==")
    c2 = []
    for label, lo, hi, dl, dm, dh in C2_SCHEMES:
        m = evaluate(trades, pct_sizer(lo, hi, dl, dm, dh))
        dev = m["notional"] / base["notional"] - 1.0
        m.update(label=label, lo=lo, hi=hi, dl=dl, dm=dm, dh=dh, dev=dev,
                 within=abs(dev) <= NOTIONAL_TOL)
        c2.append(m)
        flag = "OK" if m["within"] else "OUT-OF-BAND"
        print(f"  {label}: notional={dev*100:+.1f}% [{flag}] net=${m['net']:,.0f} "
              f"({m['net']/base['net']*100-100:+.1f}%) PF={m['pf']:.2f} "
              f"Sharpe={m['sharpe']:.2f} eff={m['eff_bps']:.1f}bps DD=${m['dd']:,.0f}", flush=True)

    # pick best WITHIN notional band, by net then sharpe
    eligible = [m for m in c2 if m["within"]] or c2
    best = max(eligible, key=lambda m: (m["net"], m["sharpe"]))
    print(f"\n-> chosen C2 scheme: {best['label']} (net=${best['net']:,.0f}, "
          f"Sharpe={best['sharpe']:.2f}, notional {best['dev']*100:+.1f}%)")

    lo, hi = best["lo"], best["hi"]
    dl, dm, dh = best["dl"], best["dm"], best["dh"]

    # ---------- C3-1 walk-forward ----------
    print("\n== C3-1: walk-forward (IS 2023-2024 / OOS 2025-2026) ==")
    is_tr = [t for t in trades if t["entry_date"] <= IS_END]
    oos_tr = [t for t in trades if t["entry_date"] > IS_END]
    # learn atr_ratio cutpoints from IS trade distribution at the chosen percentiles
    is_ar = np.array([t["atr_ratio"] for t in is_tr if t["atr_ratio"] == t["atr_ratio"]])
    c_lo = float(np.quantile(is_ar, lo))
    c_hi = float(np.quantile(is_ar, hi))
    print(f"  learned IS cutpoints: atr_ratio {lo:.0%}={c_lo:.3f}, {hi:.0%}={c_hi:.3f}")
    sizer = cut_sizer(c_lo, c_hi, dl, dm, dh)
    wf = {
        "is_dyn": evaluate(is_tr, sizer), "is_base": evaluate(is_tr, lambda t: BASE_NOTIONAL),
        "oos_dyn": evaluate(oos_tr, sizer), "oos_base": evaluate(oos_tr, lambda t: BASE_NOTIONAL),
    }
    pf_ratio = wf["oos_dyn"]["pf"] / wf["is_dyn"]["pf"] if wf["is_dyn"]["pf"] else 0.0
    for k in ("is", "oos"):
        d, b = wf[f"{k}_dyn"], wf[f"{k}_base"]
        print(f"  {k.upper():3} dyn: net=${d['net']:,.0f} PF={d['pf']:.2f} eff={d['eff_bps']:.1f}bps "
              f"| base net=${b['net']:,.0f} PF={b['pf']:.2f} eff={b['eff_bps']:.1f}bps "
              f"| Δeff={d['eff_bps']/b['eff_bps']*100-100:+.1f}%", flush=True)
    print(f"  OOS_PF / IS_PF = {pf_ratio:.2f}  (>0.85 = trustworthy)")

    # ---------- C3-2 param robustness ----------
    print("\n== C3-2: parameter robustness (neighbouring splits, pr200) ==")
    pairs = [(lo, hi), (lo-0.05, hi-0.05), (lo+0.05, hi+0.05),
             (lo-0.10, hi-0.10), (lo+0.10, hi+0.10)]
    c32 = []
    for plo, phi in pairs:
        m = evaluate(trades, pct_sizer(plo, phi, dl, dm, dh))
        beat = m["net"] > base["net"]
        m.update(lo=plo, hi=phi, beat=beat)
        c32.append(m)
        print(f"  ({plo:.0%},{phi:.0%}): net=${m['net']:,.0f} "
              f"({m['net']/base['net']*100-100:+.1f}%) notional={m['notional']/base['notional']*100-100:+.1f}% "
              f"PF={m['pf']:.2f} eff={m['eff_bps']:.1f}bps {'BEAT' if beat else 'miss'}", flush=True)
    n_beat = sum(m["beat"] for m in c32)
    print(f"  {n_beat}/{len(c32)} neighbours beat baseline net ({n_beat/len(c32)*100:.0f}%, need >=60%)")

    # ---------- C3-3 per-symbol ----------
    print("\n== C3-3: per-symbol (dynamic vs flat) ==")
    c33 = []
    sizer_ps = pct_sizer(lo, hi, dl, dm, dh)
    for nm in SYM_NAMES:
        st = by_sym[nm]
        d = evaluate(st, sizer_ps)
        b = evaluate(st, lambda t: BASE_NOTIONAL)
        beat = d["net"] > b["net"]
        c33.append({"sym": nm, "dyn": d, "base": b, "beat": beat})
        print(f"  {nm:5}: dyn net=${d['net']:,.0f} PF={d['pf']:.2f} | base net=${b['net']:,.0f} "
              f"PF={b['pf']:.2f} | Δnet={d['net']/b['net']*100-100:+.1f}% {'BEAT' if beat else 'miss'}",
              flush=True)
    n_sym = sum(x["beat"] for x in c33)
    print(f"  {n_sym}/5 symbols: dynamic beats flat (need >=4)")

    # ---------- decision ----------
    crit = {
        "C2 net > base +5%": best["net"] > base["net"] * 1.05,
        "C3-1 OOS/IS PF > 0.85": pf_ratio > 0.85,
        "C3-2 >=60% neighbours beat": n_beat / len(c32) >= 0.60,
        "C3-3 >=4/5 symbols beat": n_sym >= 4,
        "DD <= 1.2x base": best["dd"] <= base["dd"] * 1.2,
    }
    print("\n== DECISION CRITERIA ==")
    for k, v in crit.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    npass = sum(crit.values())
    verdict = ("RECOMMEND" if npass == len(crit)
               else "CONSERVATIVE" if npass >= 3 else "REJECT")
    print(f"  -> {npass}/{len(crit)} passed => {verdict}")

    write_report(base, c2, best, wf, pf_ratio, c_lo, c_hi, c32, n_beat,
                 c33, n_sym, crit, npass, verdict)
    print(f"\n-> wrote {OUT_DIR/'dynamic_sizing_report.md'}")


def write_report(base, c2, best, wf, pf_ratio, c_lo, c_hi, c32, n_beat,
                 c33, n_sym, crit, npass, verdict):
    L = []; A = L.append
    A("# MR-5m 动态仓位回测 — 任务C（C1-A → C2 → C3）\n")
    A(f"- 区间: {START} → {END}（{TIMEZONE}），全部 5 币种")
    A("- 特征: **atr_ratio**（当根 Wilder ATR / 过去24根 ATR 均值），"
      "取其在过去 **N=200** 根中的滚动分位（在线/因果，无未来信息）")
    A("- **无路径依赖**：仓位不改变开/平仓决策，交易序列与基准完全相同，逐笔重算 size/PnL/费用")
    A("- 效率 eff = 净PnL / 总名义（bps）；Sharpe 为日 PnL 年化(√365)，仅作相对比较\n")
    A(f"**基准（$500 统一仓位）**：{base['trades']:,} 笔，总名义 ${base['notional']:,.0f}，"
      f"净 ${base['net']:,.0f}，PF {base['pf']:.2f}，回撤 ${base['dd']:,.0f}，"
      f"Sharpe {base['sharpe']:.2f}，效率 {base['eff_bps']:.1f}bps。\n")

    A("## C2：三档仓位（在 C1-A 减法上加入 up-sizing）\n")
    A("目标是**重新分配**而非加杠杆——总名义须落在基准 ±10% 内。三档按 atr_ratio 的 pr200 分位划分"
      "（<50% / 50–80% / >80%）。\n")
    A("| 方案 | 档位($) | 交易数 | 总名义 | Δ名义 | 净PnL | Δ净 | PF | 回撤$ | Sharpe | 效率bps | Δ效率 |")
    A("|------|--------|------:|-------:|-----:|------:|----:|---:|------:|-------:|-------:|------:|")
    A(f"| 基准 | 500/500/500 | {base['trades']:,} | ${base['notional']:,.0f} | — | "
      f"${base['net']:,.0f} | — | {base['pf']:.2f} | ${base['dd']:,.0f} | {base['sharpe']:.2f} | "
      f"{base['eff_bps']:.1f} | — |")
    for m in c2:
        tiers = f"{int(m['dl'])}/{int(m['dm'])}/{int(m['dh'])}"
        band = "" if m["within"] else " ⚠️超带"
        A(f"| {m['label']}{band} | {tiers} | {m['trades']:,} | ${m['notional']:,.0f} | "
          f"{m['dev']*100:+.0f}% | ${m['net']:,.0f} | {m['net']/base['net']*100-100:+.0f}% | "
          f"{m['pf']:.2f} | ${m['dd']:,.0f} | {m['sharpe']:.2f} | {m['eff_bps']:.1f} | "
          f"{m['eff_bps']/base['eff_bps']*100-100:+.0f}% |")
    A("")
    all_beat = all(m["net"] > base["net"] for m in c2)
    A(f"- 是否所有方案净 PnL 都超基准：**{'是' if all_beat else '否'}**。")
    bpf = max(c2, key=lambda m: m["pf"]); bsh = max(c2, key=lambda m: m["sharpe"])
    bdd = min(c2, key=lambda m: m["dd"])
    A(f"- PF 提升最大：{bpf['label']}（{bpf['pf']:.2f}）；Sharpe 提升最大：{bsh['label']}"
      f"（{bsh['sharpe']:.2f}）；回撤最小：{bdd['label']}（${bdd['dd']:,.0f}）。")
    A(f"- **选定方案：{best['label']}** —— 在 ±10% 名义约束内按「净 PnL + Sharpe」综合最优"
      f"（净 ${best['net']:,.0f}，Sharpe {best['sharpe']:.2f}，名义 {best['dev']*100:+.0f}%）。"
      "下面的 C3 仅对该方案做验证。\n")

    A("## C3-1：时间序列 walk-forward\n")
    A(f"- IS = 2023-01-01 → 2024-12-31；OOS = 2025-01-01 → {END}")
    A(f"- 用 **IS 段** atr_ratio 分布学习固定切点：{best['lo']:.0%}→{c_lo:.3f}，"
      f"{best['hi']:.0%}→{c_hi:.3f}；再原样套用到 OOS（检验切点是否随波动 regime 漂移而失效）\n")
    A("| 段 | 方案 | 交易数 | 总名义 | 净PnL | PF | 效率bps | vs基准Δ效率 |")
    A("|----|------|------:|-------:|------:|---:|-------:|----------:|")
    for k, tag in (("is", "IS 2023-24"), ("oos", "OOS 2025-26")):
        d, b = wf[f"{k}_dyn"], wf[f"{k}_base"]
        A(f"| {tag} | 动态 | {d['trades']:,} | ${d['notional']:,.0f} | ${d['net']:,.0f} | "
          f"{d['pf']:.2f} | {d['eff_bps']:.1f} | {d['eff_bps']/b['eff_bps']*100-100:+.0f}% |")
        A(f"| {tag} | 基准 | {b['trades']:,} | ${b['notional']:,.0f} | ${b['net']:,.0f} | "
          f"{b['pf']:.2f} | {b['eff_bps']:.1f} | — |")
    A("")
    A(f"- **OOS_PF / IS_PF = {pf_ratio:.2f}**（>0.85 视为可信，<0.85 过拟合警告）。")
    A(f"- IS 动态 vs 基准效率：{wf['is_dyn']['eff_bps']/wf['is_base']['eff_bps']*100-100:+.0f}%；"
      f"OOS 动态 vs 基准效率：{wf['oos_dyn']['eff_bps']/wf['oos_base']['eff_bps']*100-100:+.0f}%。"
      "（更稳健的过拟合判据：动态相对基准的**优势**是否在 OOS 仍然存在，而非绝对 PF——"
      "绝对 PF 受市场 regime 影响，2024 逆风年在 IS 内会压低 IS PF。）\n")

    A("## C3-2：参数稳健性（pr200，邻近分位对）\n")
    A("固定档位金额不变，仅平移分位切点。若只有正中心好、邻近全差 → 过拟合；一片区域都好 → 真规律。\n")
    A("| 分位对 (低,高) | 总名义 | 净PnL | Δ净 | PF | 效率bps | 优于基准 |")
    A("|--------------|-------:|------:|----:|---:|-------:|:------:|")
    for m in c32:
        A(f"| ({m['lo']:.0%}, {m['hi']:.0%}) | ${m['notional']:,.0f} | ${m['net']:,.0f} | "
          f"{m['net']/base['net']*100-100:+.0f}% | {m['pf']:.2f} | {m['eff_bps']:.1f} | "
          f"{'✅' if m['beat'] else '❌'} |")
    A(f"\n- {n_beat}/{len(c32)} 个邻近参数净 PnL 优于基准（{n_beat/len(c32)*100:.0f}%，标准 ≥60%）。\n")

    A("## C3-3：分币种验证（动态 vs 平仓位）\n")
    A("| 币种 | 动态净$ | 动态PF | 基准净$ | 基准PF | Δ净 | 动态更优 |")
    A("|------|-------:|------:|-------:|------:|----:|:------:|")
    for x in c33:
        d, b = x["dyn"], x["base"]
        A(f"| {x['sym']} | ${d['net']:,.0f} | {d['pf']:.2f} | ${b['net']:,.0f} | {b['pf']:.2f} | "
          f"{d['net']/b['net']*100-100:+.0f}% | {'✅' if x['beat'] else '❌'} |")
    A(f"\n- {n_sym}/5 币种动态优于平仓位（标准 ≥4/5）。\n")

    A("## 最终决策\n")
    A("| 判据 | 结果 |")
    A("|------|:----:|")
    for k, v in crit.items():
        A(f"| {k} | {'✅ 通过' if v else '❌ 未过'} |")
    A(f"\n**{npass}/{len(crit)} 项通过 → 结论：{verdict}**\n")
    if verdict == "RECOMMEND":
        A(f"- 全部判据通过：**推荐**采用 {best['label']}（atr_ratio pr200，"
          f"<50%→${int(best['dl'])} / 50–80%→${int(best['dm'])} / >80%→${int(best['dh'])}）。"
          "总名义与基准持平，净 PnL 与效率提升在 OOS、邻近参数、多币种上均成立——是规律而非过拟合。")
    elif verdict == "CONSERVATIVE":
        A(f"- 部分判据通过：**保守推荐**。{best['label']} 在样本内改善明确，但下列判据未过——")
        for k, v in crit.items():
            if not v:
                A(f"  - ❌ {k}")
        A("- 保守做法：要么沿用基准平仓位，要么仅采用**最温和**的一档（如 C2-3）以降低对分位估计误差的敏感性，"
          "并在小资金上先行观察 OOS 表现，确认后再放大。")
    else:
        A("- **不推荐**：样本内看起来更好，但 walk-forward / 参数 / 分币种验证未能支撑。"
          "按既定原则，诚实标注为**不可信**——动态仓位在此数据上未显示可靠、可泛化的改进。")
    A("\n> 原则备注：本轮**未为通过验证而调参**；档位金额与分位切点先验设定，C3 仅检验、不回头优化。")
    (OUT_DIR / "dynamic_sizing_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
