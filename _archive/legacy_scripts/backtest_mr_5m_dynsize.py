#!/usr/bin/env python3
"""MR-5m dynamic position sizing — Task C, stage C1-A.

Premise (from Task B): the edge concentrates in extreme signals; the top
atr_ratio decile (vol expansion) produced ~64% of net. Test whether sizing UP
on high-quality signals and DOWN on ordinary ones improves *capital efficiency*
(PnL per unit of notional deployed) without relying on trading less.

C1-A: single feature (atr_ratio percentile), binary size.
  - feature: rolling percentile of current atr_ratio within trailing N bars
    (atr_ratio = current Wilder ATR / trailing-24-bar mean ATR)
  - N in {50, 100, 200}; threshold in {70%, 80%, 90%}
  - notional = $500 if percentile >= threshold else $250

KEY PROPERTY: sizing does NOT change which trades occur (entry/exit are size-
independent), so this is NOT path-dependent. We take the exact baseline trade
sequence and rescale each trade's size/PnL/fees by its configured notional.

Metrics vs $500-flat baseline: trades (identical), total notional deployed,
net PnL, PF, max DD, Sharpe (daily PnL), and EFFICIENCY = net / total notional.

Reuses the validated engine (Wilder ATR / floating stop / close±tick taker /
maker-rebate+taker fees / int sizing). Live script untouched.

Usage: python scripts/backtest_mr_5m_dynsize.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 2026-07 重构批次6：脚本迁入 _archive/legacy_scripts/；共享依赖真身在
# scripts/（前向冻结区）与 core/data_io/，此处按新深度注入 sys.path。
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (
    str(_REPO_ROOT / "core" / "data_io"),
    str(_REPO_ROOT / "scripts"),
    str(_REPO_ROOT / "data_engineering" / "scripts"),
    *sorted(str(_q) for _q in (_REPO_ROOT / "research" / "_closed").glob("*/*/scripts")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from backtest_mr_5m_compare import (
    SYMBOLS, CONTRACT_SPECS, ATR_THRESHOLDS, LOOKBACK, ATR_STOP, MAX_HOLD,
    FEE_MAKER, FEE_TAKER, LEVERAGE, wilder_atr, load_1m, r5, parse_history_range,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2026-07 重构批次6：迁入 _archive/legacy_scripts/，深度 1→2
DB = PROJECT_ROOT / ".vntrader" / "database.db"
START, END = "2023-01-01", "2026-05-28"
TIMEZONE = "UTC"
OUT_DIR = PROJECT_ROOT / "reports" / "regime"
SYM_NAMES = ["BTC", "ETH", "SOL", "LINK", "DOGE"]
BASE_NOTIONAL = 500.0
N_GRID = [50, 100, 200]
THR_GRID = [0.70, 0.80, 0.90]


def size_for(inst, price, notional):
    cv = price * CONTRACT_SPECS[inst]["ctVal"]
    if cv <= 0:
        return 1
    return max(1, min(round(notional * LEVERAGE / cv), 1000))


def collect_trades(name, bars):
    """Baseline MR loop; record per-trade context + atr_ratio rolling percentiles."""
    inst = SYMBOLS[name][1]
    ct = CONTRACT_SPECS[inst]["ctVal"]
    tick = CONTRACT_SPECS[inst]["tickSz"]
    thr = ATR_THRESHOLDS[inst]

    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    dt = bars["datetime"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    n = len(c)
    atr = wilder_atr(h, l, c)
    dh = bars["high"].rolling(LOOKBACK).max().shift(1).to_numpy()
    dl = bars["low"].rolling(LOOKBACK).min().shift(1).to_numpy()

    atr_ratio = pd.Series(atr) / pd.Series(atr).rolling(24).mean().shift(1)
    pr = {N: atr_ratio.rolling(N).rank(pct=True).to_numpy() for N in N_GRID}
    ar = atr_ratio.to_numpy()

    trades = []
    pos = 0; eb = -1; ep = 0.0
    for i in range(LOOKBACK + 1, n):
        a = atr[i]
        if a != a or a <= 0:
            continue
        if pos != 0:
            reason = ""
            dhi = dh[i]; dli = dl[i]
            if dhi > 0 and dli > 0:
                mid = (dhi + dli) / 2.0
                if (pos == 1 and c[i] >= mid) or (pos == -1 and c[i] <= mid):
                    reason = "midline"
            if not reason:
                sd = ATR_STOP * a
                if pos == 1 and l[i] <= ep - sd:
                    reason = "stop"
                elif pos == -1 and h[i] >= ep + sd:
                    reason = "stop"
            if not reason and (i - eb) >= MAX_HOLD:
                reason = "max_hold"
            if reason:
                exit_px = c[i] - tick if pos == 1 else c[i] + tick
                trades.append({
                    "inst": inst, "symbol": name, "ct": ct,
                    "side": pos, "entry": ep, "exit": exit_px, "reason": reason,
                    "exit_date": dt[i].astype("datetime64[D]"),
                    "entry_date": dt[eb].astype("datetime64[D]"),
                    "atr_ratio": ar[eb],
                    "pr50": pr[50][eb], "pr100": pr[100][eb], "pr200": pr[200][eb],
                })
                pos = 0
                continue
        if pos == 0:
            if thr > 0 and a < thr:
                continue
            dhi = dh[i]; dli = dl[i]
            if dhi != dhi or dli != dli or dhi <= 0 or dli <= 0:
                continue
            cc = c[i]
            if cc > dhi:
                pos = -1
            elif cc < dli:
                pos = 1
            else:
                continue
            ep = cc; eb = i
    return trades


def trade_pnl(t, notional):
    """Net PnL + actual notional for one trade at a given configured notional."""
    sz = size_for(t["inst"], t["entry"], notional)
    ct = t["ct"]
    if t["side"] == 1:
        gross = (t["exit"] - t["entry"]) * sz * ct
    else:
        gross = (t["entry"] - t["exit"]) * sz * ct
    fee = (-FEE_MAKER * t["entry"] * sz * ct) - (FEE_TAKER * t["exit"] * sz * ct)
    return gross + fee


def evaluate(trades, notional_of):
    """notional_of: callable(trade)->notional. Returns metrics dict."""
    nets = np.empty(len(trades))
    notional_sum = 0.0
    daily = {}
    for k, t in enumerate(trades):
        nv = notional_of(t)
        net = trade_pnl(t, nv)
        nets[k] = net
        notional_sum += nv
        d = t["exit_date"]
        daily[d] = daily.get(d, 0.0) + net
    nets = pd.Series(nets)
    wins = nets[nets > 0].sum(); losses = -nets[nets < 0].sum()
    pf = (wins / losses) if losses > 0 else float("inf")
    # equity / DD by exit date order
    ds = pd.Series(daily).sort_index()
    eq = ds.cumsum()
    dd = float((eq - eq.cummax()).min())
    # Sharpe of daily PnL (absolute $), annualised sqrt(365); relative measure
    sharpe = float(np.sqrt(365) * ds.mean() / ds.std()) if ds.std() > 0 else 0.0
    net = float(nets.sum())
    return {
        "trades": len(trades),
        "notional": notional_sum,
        "net": net,
        "pf": float(pf),
        "dd": -dd,
        "sharpe": sharpe,
        "eff_bps": net / notional_sum * 1e4 if notional_sum else 0.0,  # PnL per $ notional, in bps
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hr = parse_history_range(START, END, timedelta(minutes=1), TIMEZONE)
    print("Collecting baseline trades + atr_ratio percentiles...", flush=True)
    trades = []
    for nm in SYM_NAMES:
        b5 = r5(load_1m(SYMBOLS[nm][0], hr, DB), 5, hr)
        t = collect_trades(nm, b5)
        trades += t
        print(f"  [{nm}] {len(t):,} trades", flush=True)

    base = evaluate(trades, lambda t: BASE_NOTIONAL)
    print(f"\nBASELINE ($500 flat): trades={base['trades']:,} notional=${base['notional']:,.0f} "
          f"net=${base['net']:,.0f} PF={base['pf']:.2f} DD=${base['dd']:,.0f} "
          f"Sharpe={base['sharpe']:.2f} eff={base['eff_bps']:.1f}bps")

    results = []
    for N in N_GRID:
        key = f"pr{N}"
        for th in THR_GRID:
            r = evaluate(trades, lambda t, key=key, th=th:
                         BASE_NOTIONAL if (t[key] >= th and t[key] == t[key]) else 250.0)
            r["label"] = f"C1-A N={N} thr={int(th*100)}%"
            r["N"] = N; r["thr"] = th
            results.append(r)
            print(f"  {r['label']:22} net=${r['net']:,.0f} ({r['net']/base['net']*100-100:+.1f}%) "
                  f"notional=${r['notional']:,.0f} ({r['notional']/base['notional']*100-100:+.1f}%) "
                  f"PF={r['pf']:.2f} eff={r['eff_bps']:.1f}bps ({r['eff_bps']/base['eff_bps']*100-100:+.1f}%) "
                  f"Sharpe={r['sharpe']:.2f}", flush=True)

    write_report(base, results)
    print(f"\n-> wrote {OUT_DIR/'dynamic_sizing_report.md'}")


def write_report(base, results):
    L = []; A = L.append
    A("# MR-5m 动态仓位回测 — 任务C / 阶段C1-A\n")
    A(f"- 区间: {START} → {END}（{TIMEZONE}），全部 5 币种")
    A("- 方案: 单特征 atr_ratio + 二元仓位。atr_ratio = 当根 Wilder ATR / 过去24根 ATR 均值；"
      "取其在过去 N 根中的滚动分位，分位 ≥ 阈值则 **$500**，否则 **$250**")
    A("- **关键性质**：仓位不改变开/平仓决策（交易序列与基准完全相同），"
      "故无路径依赖——直接对基准交易逐笔重算 size/PnL/费用")
    A("- 效率 eff = 净PnL / 总名义（bps）= 每投入 $1 名义赚多少；这是资金效率的核心指标")
    A("- Sharpe 为日 PnL（绝对$）年化(√365)，仅作配置间相对比较\n")
    A(f"**基准（$500 统一仓位）**：{base['trades']:,} 笔，总名义 ${base['notional']:,.0f}，"
      f"净 ${base['net']:,.0f}，PF {base['pf']:.2f}，回撤 ${base['dd']:,.0f}，"
      f"Sharpe {base['sharpe']:.2f}，效率 {base['eff_bps']:.1f}bps。\n")

    A("## 对比表（vs 基准）\n")
    A("| 方案 | 交易数 | 总名义 | Δ名义 | 净PnL | Δ净 | PF | 回撤$ | Sharpe | 效率bps | Δ效率 |")
    A("|------|------:|-------:|-----:|------:|----:|---:|------:|-------:|-------:|------:|")
    A(f"| 基准 $500 | {base['trades']:,} | ${base['notional']:,.0f} | — | "
      f"${base['net']:,.0f} | — | {base['pf']:.2f} | ${base['dd']:,.0f} | {base['sharpe']:.2f} | "
      f"{base['eff_bps']:.1f} | — |")
    for r in results:
        A(f"| {r['label']} | {r['trades']:,} | ${r['notional']:,.0f} | "
          f"{r['notional']/base['notional']*100-100:+.0f}% | ${r['net']:,.0f} | "
          f"{r['net']/base['net']*100-100:+.0f}% | {r['pf']:.2f} | ${r['dd']:,.0f} | "
          f"{r['sharpe']:.2f} | {r['eff_bps']:.1f} | "
          f"{r['eff_bps']/base['eff_bps']*100-100:+.0f}% |")

    A("\n## 初步解读\n")
    best = max(results, key=lambda r: r["eff_bps"])
    A(f"- 效率最高：**{best['label']}** — 效率 {best['eff_bps']:.1f}bps "
      f"(基准 {base['eff_bps']:.1f}bps, {best['eff_bps']/base['eff_bps']*100-100:+.0f}%)，"
      f"净 ${best['net']:,.0f} ({best['net']/base['net']*100-100:+.0f}%)，"
      f"总名义 {best['notional']/base['notional']*100-100:+.0f}%。")
    eff_up = [r for r in results if r["eff_bps"] > base["eff_bps"]]
    A(f"- {len(eff_up)}/{len(results)} 个配置效率高于基准。")
    A("- **读法**：减半「普通信号」会同时减少其盈亏，故总净额必然下降；"
      "真正要看的是 **效率(eff) 是否提升**——若提升，说明高分位信号的单位名义回报确实更高，"
      "「按信号强度配资」的逻辑成立。但 C1-A 只下调、不上调，总名义必然下降；"
      "是否值得，要看效率提升幅度，并在 C2（总名义持平）和 C3（walk-forward）中验证。")
    A("\n> 注意：本阶段**尚未做 walk-forward / 参数稳健性 / 分币种验证**（C3）。"
      "在通过 C3 之前，任何「看起来更好」的配置都**不能采信**。")
    (OUT_DIR / "dynamic_sizing_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
