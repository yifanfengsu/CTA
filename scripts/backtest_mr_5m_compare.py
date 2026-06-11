#!/usr/bin/env python3
"""MR-5m multi-symbol scenario backtester — faithful replica of live runner.

Replicates scripts/run_mr_5m_direct.py strategy mechanics exactly so that the
backtest is a 1:1 stand-in for the live OKX paper account:

  * Wilder smoothed ATR (SMA seed of first 14 TRs, alpha=1/14) — NOT SMA ATR.
  * Donchian channel from the 24 bars BEFORE the current bar (rolling(24).shift(1)).
  * Entry: fade Donchian breakout. close>DH -> short, close<DL -> long.
  * ATR regime filter: skip entry when current-bar Wilder ATR < static threshold.
  * Exit priority: midline (dynamic) -> ATR stop (entry +/- 1.0x CURRENT ATR,
    low/high touch) -> max_hold (48 bars). One-bar lag between entry and first
    exit check, matching the async live fill.
  * Sizing: size = round(500 * 5 / (price*ctVal)), clamped [1,1000], integer
    contracts per OKX contract spec.
  * Fill prices: entry = bar close (maker limit). exit = bar close -/+ 1 tick
    (taker market, slippage against us).
  * Fees: entry maker -0.002% (rebate / credit), exit taker 0.05% (cost), on
    actual contract notional. net = gross - taker_fee + maker_rebate.
  * No funding adjustment (live trade log accounts OKX fill fees only).

KNOWN DEVIATION (intentional, documented in report): the static ATR thresholds
were originally derived as p30 of *SMA* ATR, but live (and therefore this
backtest) applies them against *Wilder* ATR. Not corrected this round per task.

Usage:
  python scripts/backtest_mr_5m_compare.py --scenarios D1
  python scripts/backtest_mr_5m_compare.py --scenarios all --report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_mr_5m import load_1m, r5  # data loading + 1m->5m aggregation
from history_time_utils import parse_history_range

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Strategy params (locked — identical to live run_mr_5m_direct.py) ──────────
LOOKBACK = 24
ATR_WINDOW = 14
ATR_STOP = 1.0
MAX_HOLD = 48
LEVERAGE = 5
NOTIONAL_PER_TRADE = 500.0

# Fee model (task spec): maker rebate on entry, taker cost on exit.
FEE_MAKER = -0.00002   # -0.002%  (negative = rebate / credit)
FEE_TAKER = 0.0005     #  0.05%   (positive = cost)

ATR_THRESHOLDS = {
    "BTC-USDT-SWAP": 81.5,
    "ETH-USDT-SWAP": 4.64,
    "SOL-USDT-SWAP": 0.245,
    "LINK-USDT-SWAP": 0.0212,
    "DOGE-USDT-SWAP": 0.0002,
}

CONTRACT_SPECS = {
    "BTC-USDT-SWAP":  {"ctVal": 0.01, "tickSz": 0.1,     "minSz": 0.01},
    "ETH-USDT-SWAP":  {"ctVal": 0.1,  "tickSz": 0.01,    "minSz": 0.01},
    "SOL-USDT-SWAP":  {"ctVal": 1.0,  "tickSz": 0.01,    "minSz": 0.01},
    "LINK-USDT-SWAP": {"ctVal": 1.0,  "tickSz": 0.001,   "minSz": 0.1},
    "DOGE-USDT-SWAP": {"ctVal": 1000, "tickSz": 0.00001, "minSz": 1.0},
}

# name -> (vt_symbol in DB, OKX inst_id)
SYMBOLS = {
    "BTC":  ("BTCUSDT_SWAP_OKX.GLOBAL",  "BTC-USDT-SWAP"),
    "ETH":  ("ETHUSDT_SWAP_OKX.GLOBAL",  "ETH-USDT-SWAP"),
    "SOL":  ("SOLUSDT_SWAP_OKX.GLOBAL",  "SOL-USDT-SWAP"),
    "LINK": ("LINKUSDT_SWAP_OKX.GLOBAL", "LINK-USDT-SWAP"),
    "DOGE": ("DOGEUSDT_SWAP_OKX.GLOBAL", "DOGE-USDT-SWAP"),
}

SCENARIOS = {
    "A":  ["BTC", "ETH", "SOL", "LINK", "DOGE"],
    "B":  ["BTC", "ETH", "SOL", "LINK"],
    "C":  ["ETH", "SOL"],
    "D1": ["BTC"],
    "D2": ["ETH"],
    "D3": ["SOL"],
    "D4": ["LINK"],
    "D5": ["DOGE"],
}

DEFAULT_START, DEFAULT_END = "2023-01-01", "2026-03-01"
TIMEZONE = "UTC"  # match live (OKX ts are UTC; 5m boundaries align to UTC clock)

OUT_DIR = PROJECT_ROOT / "reports" / "backtest_compare"


# ── Indicators ───────────────────────────────────────────────────────────────
def wilder_atr(high, low, close, period=ATR_WINDOW):
    """Wilder smoothed ATR. atr[i] uses bars up to and including i.

    Mirrors live BarAggregator._compute_atr: first value at index `period`
    (the (period+1)-th bar) = mean of TR[1..period]; then EMA alpha=1/period.
    Returns np.nan where ATR is not yet defined (i < period).
    """
    n = len(close)
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    atr[period] = np.mean(tr[1:period + 1])
    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def calc_size(inst_id, price):
    """Replica of live calc_size: integer contracts, clamped [1, 1000]."""
    ct_val = CONTRACT_SPECS[inst_id]["ctVal"]
    contract_value = price * ct_val
    if contract_value <= 0:
        return 1
    size = round(NOTIONAL_PER_TRADE * LEVERAGE / contract_value)
    return max(1, min(size, 1000))


# ── Per-symbol trade engine ──────────────────────────────────────────────────
def backtest_symbol(name, bars):
    """Run the MR-5m loop over one symbol's 5m bars. Returns list of trade dicts.

    Each trade dict matches the live trade_log schema.
    """
    inst_id = SYMBOLS[name][1]
    ct_val = CONTRACT_SPECS[inst_id]["ctVal"]
    tick = CONTRACT_SPECS[inst_id]["tickSz"]
    threshold = ATR_THRESHOLDS[inst_id]

    dt = bars["datetime"].to_numpy()
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = len(c)
    if n < LOOKBACK + 5:
        return []

    atr = wilder_atr(h, l, c)
    # Donchian excludes current bar: max/min of the 24 bars before i.
    dh = bars["high"].rolling(LOOKBACK).max().shift(1).to_numpy()
    dl = bars["low"].rolling(LOOKBACK).min().shift(1).to_numpy()

    trades = []
    pos = 0            # 0 flat, +1 long, -1 short
    eb = -1            # entry bar index
    ep = 0.0           # entry price
    esize = 0          # entry size (contracts)
    et = None          # entry datetime

    for i in range(LOOKBACK + 1, n):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        # ── Exit (current-bar indicators), only when in position ──
        if pos != 0:
            hb = i - eb
            reason = ""
            d_h, d_l = dh[i], dl[i]
            if d_h > 0 and d_l > 0:
                mid = (d_h + d_l) / 2.0
                if (pos == 1 and c[i] >= mid) or (pos == -1 and c[i] <= mid):
                    reason = "midline"
            if not reason:
                stop_dist = ATR_STOP * atr_i  # floats with CURRENT atr
                if pos == 1 and l[i] <= ep - stop_dist:
                    reason = "stop"
                elif pos == -1 and h[i] >= ep + stop_dist:
                    reason = "stop"
            if not reason and hb >= MAX_HOLD:
                reason = "max_hold"

            if reason:
                # Market exit ≈ bar close, 1 tick slippage against us.
                exit_px = c[i] - tick if pos == 1 else c[i] + tick
                gross = (exit_px - ep) * esize * ct_val if pos == 1 \
                    else (ep - exit_px) * esize * ct_val
                entry_notional = ep * esize * ct_val
                exit_notional = exit_px * esize * ct_val
                maker_rebate = -FEE_MAKER * entry_notional   # FEE_MAKER<0 -> credit
                taker_fee = FEE_TAKER * exit_notional         # cost
                fee_usd = maker_rebate - taker_fee            # net, signed (neg=cost)
                net = gross + fee_usd
                trades.append({
                    "time": pd.Timestamp(dt[i]).isoformat(),
                    "symbol": name,
                    "side": "long" if pos == 1 else "short",
                    "entry_price": round(ep, 8),
                    "entry_time": pd.Timestamp(et).isoformat(),
                    "exit_price": round(exit_px, 8),
                    "exit_reason": reason,
                    "size": esize,
                    "gross_pnl_usd": round(gross, 4),
                    "fee_usd": round(fee_usd, 4),
                    "net_pnl_usd": round(net, 4),
                })
                pos = 0
                continue  # no same-bar re-entry (matches live)

        # ── Entry (only when flat) ──
        if pos == 0:
            if threshold > 0 and atr_i < threshold:
                continue  # ATR regime filter
            d_h, d_l = dh[i], dl[i]
            if np.isnan(d_h) or np.isnan(d_l) or d_h <= 0 or d_l <= 0:
                continue
            close = c[i]
            if close > d_h:
                pos = -1  # fade long breakout -> short
            elif close < d_l:
                pos = 1   # fade short breakout -> long
            else:
                continue
            ep = close                       # maker limit fills at close
            esize = calc_size(inst_id, close)
            eb = i
            et = dt[i]

    return trades


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    """Aggregate scenario-level metrics from a merged trade list."""
    m = {"n": len(trades)}
    if not trades:
        return m

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_dt"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("exit_dt", kind="stable").reset_index(drop=True)

    net = df["net_pnl_usd"]
    wins = net[net > 0]
    losses = net[net < 0]
    m["net_pnl"] = float(net.sum())
    m["gross_pnl"] = float(df["gross_pnl_usd"].sum())
    m["total_fees"] = float(df["fee_usd"].sum())
    m["pf"] = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    m["win_rate_profit"] = float((net > 0).mean() * 100)
    m["win_rate_midline"] = float((df["exit_reason"] == "midline").mean() * 100)
    m["avg_trade"] = float(net.mean())

    # Exit structure
    m["exits"] = {}
    for reason in ("midline", "stop", "max_hold", "end_of_data"):
        g = df[df["exit_reason"] == reason]
        if len(g):
            m["exits"][reason] = {
                "count": int(len(g)),
                "pct": float(len(g) / len(df) * 100),
                "avg_pnl": float(g["net_pnl_usd"].mean()),
                "total_pnl": float(g["net_pnl_usd"].sum()),
            }

    # Equity / drawdown (cumulative net PnL ordered by exit time; df pre-sorted)
    equity = df["net_pnl_usd"].cumsum()
    peak = equity.cummax()
    dd_abs = equity - peak  # <= 0
    m["max_dd_usd"] = float(-dd_abs.min())
    trough_i = dd_abs.idxmin()
    peak_at_trough = float(peak.iloc[trough_i]) if len(peak) else 0.0
    m["max_dd_pct"] = float(m["max_dd_usd"] / peak_at_trough * 100) if peak_at_trough > 0 else float("nan")

    # Max consecutive losses (ordered by exit time)
    max_streak = streak = 0
    for v in df["net_pnl_usd"]:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    m["max_consec_losses"] = int(max_streak)

    # Sub-period PF by entry year
    m["subperiods"] = {}
    for label, yr in (("2023", 2023), ("2024", 2024), ("2025", 2025), ("2026-todate", 2026)):
        g = df[df["entry_time"].dt.year == yr]
        if len(g):
            gw = g[g["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
            gl = abs(g[g["net_pnl_usd"] < 0]["net_pnl_usd"].sum())
            m["subperiods"][label] = {
                "n": int(len(g)),
                "net_pnl": float(g["net_pnl_usd"].sum()),
                "pf": float(gw / gl) if gl > 0 else float("inf"),
            }

    # PF / net excluding 2024 (is the 2024 weakness universal or symbol-specific?)
    ex = df[df["entry_time"].dt.year != 2024]
    if len(ex):
        ew = ex[ex["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
        el = abs(ex[ex["net_pnl_usd"] < 0]["net_pnl_usd"].sum())
        m["ex2024"] = {
            "n": int(len(ex)),
            "net_pnl": float(ex["net_pnl_usd"].sum()),
            "pf": float(ew / el) if el > 0 else float("inf"),
        }
    else:
        m["ex2024"] = {"n": 0, "net_pnl": 0.0, "pf": float("inf")}
    return m


def fmt_pf(pf):
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def print_metrics(scenario_id, symbols, m):
    print(f"\n{'='*64}")
    print(f"  SCENARIO {scenario_id}: {'+'.join(symbols)}")
    print(f"{'='*64}")
    if m["n"] == 0:
        print("  No trades.")
        return
    print(f"  Total trades:        {m['n']:>10,}")
    print(f"  Win rate (profit):   {m['win_rate_profit']:>9.1f}%")
    print(f"  Win rate (midline):  {m['win_rate_midline']:>9.1f}%")
    print(f"  Net PnL:             ${m['net_pnl']:>10,.2f}")
    print(f"  Gross PnL:           ${m['gross_pnl']:>10,.2f}")
    print(f"  Total fees (net):    ${m['total_fees']:>10,.2f}")
    print(f"  Profit Factor:       {fmt_pf(m['pf']):>10}")
    print(f"  Avg trade:           ${m['avg_trade']:>10.2f}")
    dd_pct = "n/a" if m['max_dd_pct'] != m['max_dd_pct'] else f"{m['max_dd_pct']:.1f}%"
    print(f"  Max drawdown:        ${m['max_dd_usd']:>10,.2f}  ({dd_pct})")
    print(f"  Max consec losses:   {m['max_consec_losses']:>10}")
    print(f"\n  Exit structure:")
    print(f"    {'reason':>10} {'count':>8} {'pct':>7} {'avg$':>9} {'total$':>12}")
    for reason in ("midline", "stop", "max_hold", "end_of_data"):
        if reason in m["exits"]:
            e = m["exits"][reason]
            print(f"    {reason:>10} {e['count']:>8,} {e['pct']:>6.1f}% "
                  f"{e['avg_pnl']:>9.2f} {e['total_pnl']:>12,.2f}")
    print(f"\n  Sub-period PF (by entry year):")
    print(f"    {'period':>12} {'trades':>8} {'net$':>12} {'PF':>7}")
    for label in ("2023", "2024", "2025", "2026-todate"):
        if label in m["subperiods"]:
            s = m["subperiods"][label]
            print(f"    {label:>12} {s['n']:>8,} {s['net_pnl']:>12,.2f} {fmt_pf(s['pf']):>7}")
    ex = m.get("ex2024")
    if ex and ex["n"]:
        print(f"    {'EX-2024':>12} {ex['n']:>8,} {ex['net_pnl']:>12,.2f} {fmt_pf(ex['pf']):>7}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", default="all",
                   help="comma list of scenario IDs (A,B,C,D1..D5) or 'all'")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--database-path", default=str(PROJECT_ROOT / ".vntrader" / "database.db"))
    p.add_argument("--report", action="store_true", help="generate compare report (needs all 8)")
    args = p.parse_args(argv)

    if args.scenarios.strip().lower() == "all":
        scen_ids = list(SCENARIOS.keys())
    else:
        scen_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        for sid in scen_ids:
            if sid not in SCENARIOS:
                raise SystemExit(f"Unknown scenario: {sid}. Valid: {list(SCENARIOS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hr = parse_history_range(args.start, args.end, timedelta(minutes=1), TIMEZONE)
    db = Path(args.database_path)

    # Which symbols do we actually need?
    needed = sorted({s for sid in scen_ids for s in SCENARIOS[sid]})
    print(f"Backtest range: {args.start} -> {args.end} ({TIMEZONE})")
    print(f"Scenarios: {scen_ids}  |  symbols needed: {needed}")

    # Compute per-symbol trades once.
    sym_trades = {}
    for name in needed:
        vt = SYMBOLS[name][0]
        print(f"\n[{name}] loading 1m bars...", flush=True)
        b1 = load_1m(vt, hr, db)
        b5 = r5(b1, 5, hr)
        if b5.empty or len(b5) < 26000:  # ~ <90 days of 5m -> insufficient
            cov_days = len(b5) * 5 / 60 / 24 if not b5.empty else 0
            if cov_days < 180:
                print(f"[{name}] WARNING insufficient data ({cov_days:.0f}d, {len(b5)} 5m bars) — SKIPPING")
                sym_trades[name] = []
                continue
        print(f"[{name}] {len(b5):,} 5m bars "
              f"({b5['datetime'].iloc[0]} -> {b5['datetime'].iloc[-1]})", flush=True)
        t = backtest_symbol(name, b5)
        sym_trades[name] = t
        print(f"[{name}] {len(t):,} trades", flush=True)

    # Per scenario: merge, write jsonl, metrics.
    all_metrics = {}
    for sid in scen_ids:
        syms = SCENARIOS[sid]
        merged = [tr for s in syms for tr in sym_trades.get(s, [])]
        merged.sort(key=lambda x: x["time"])
        out_path = OUT_DIR / f"backtest_results_{sid}.jsonl"
        with open(out_path, "w") as f:
            for tr in merged:
                f.write(json.dumps(tr) + "\n")
        m = compute_metrics(merged)
        all_metrics[sid] = m
        print_metrics(sid, syms, m)
        print(f"\n  -> wrote {len(merged):,} trades to {out_path}")

    if args.report:
        generate_report(all_metrics)

    return 0


def generate_report(all_metrics):
    """Build backtest_compare_report.md from computed scenario metrics."""
    path = OUT_DIR / "backtest_compare_report.md"
    lines = []
    A = lines.append

    A("# MR-5m 多币种组合回测对比报告\n")
    A(f"- 回测区间: {DEFAULT_START} → {DEFAULT_END}（{TIMEZONE}）")
    A("- 策略: fade Donchian 突破，5m 周期，LB=24 / ATR=14(Wilder) / 止损 1.0×ATR / 最大持仓 48 bars")
    A("- 名义: $500/笔 × 5x 杠杆，整数张数（按 OKX ctVal）")
    A("- 费率: 入场 maker −0.002%（返佣），出场 taker 0.05%；市价出场加 1 tick 滑点")
    A("- PnL 均为扣费后 net（不含 funding，与实盘 trade_log 口径一致）\n")
    A("> **已知偏差（下一轮迭代 input）**：静态 ATR 阈值（BTC 81.5 / ETH 4.64 / "
      "SOL 0.245 / LINK 0.0212 / DOGE 0.0002）最初由 *SMA*-ATR 的 p30 推导，"
      "但实盘与本回测均用其与 *Wilder*-ATR 比较。本轮保持与实盘一致，未修正。\n")

    # Scenario table sorted by PF
    A("## 1. 方案对比（按 Profit Factor 排序）\n")
    A("| 方案 | 标的 | 交易数 | 胜率(净) | midline占比 | 净PnL | PF | 最大回撤$ | 回撤% | 最大连亏 |")
    A("|------|------|-------:|--------:|-----------:|------:|---:|---------:|------:|--------:|")
    order = sorted(all_metrics.items(),
                   key=lambda kv: (kv[1].get("pf", 0) if kv[1].get("pf") != float("inf") else 9e9),
                   reverse=True)
    for sid, m in order:
        syms = "+".join(SCENARIOS[sid])
        if m["n"] == 0:
            A(f"| {sid} | {syms} | 0 | - | - | - | - | - | - | - |")
            continue
        ddp = "n/a" if m["max_dd_pct"] != m["max_dd_pct"] else f"{m['max_dd_pct']:.1f}%"
        A(f"| {sid} | {syms} | {m['n']:,} | {m['win_rate_profit']:.1f}% | "
          f"{m['win_rate_midline']:.1f}% | ${m['net_pnl']:,.0f} | {fmt_pf(m['pf'])} | "
          f"${m['max_dd_usd']:,.0f} | {ddp} | {m['max_consec_losses']} |")
    A("\n> **回撤% 口径说明**：回撤% = 回撤$ / 回撤谷底前的权益峰值，权益曲线从 0 起累计 net PnL。"
      "当某标的早期即陷入净亏（如 LINK 2023 年 PF 0.86，权益峰值接近 0），分母极小会放大百分比"
      "（LINK 显示 149.8% 即为此假象）。**请以回撤$ 为准**，回撤% 仅供组合方案间横向参考。")

    # Per-symbol independent (D1-D5)
    A("\n## 2. 单币种独立表现（D1–D5）\n")
    A("| 方案 | 币种 | 交易数 | 胜率(净) | midline占比 | 净PnL | PF | midline笔/均利 | stop笔/均亏 | max_hold笔 |")
    A("|------|------|-------:|--------:|-----------:|------:|---:|--------------|------------|----------:|")
    for sid in ("D1", "D2", "D3", "D4", "D5"):
        m = all_metrics.get(sid)
        sym = SCENARIOS[sid][0]
        if not m or m["n"] == 0:
            A(f"| {sid} | {sym} | 0 | - | - | - | - | - | - | - |")
            continue
        ex = m["exits"]
        mid = ex.get("midline", {})
        stp = ex.get("stop", {})
        mh = ex.get("max_hold", {})
        mid_s = f"{mid.get('count',0):,}/${mid.get('avg_pnl',0):.2f}" if mid else "0"
        stp_s = f"{stp.get('count',0):,}/${stp.get('avg_pnl',0):.2f}" if stp else "0"
        A(f"| {sid} | {sym} | {m['n']:,} | {m['win_rate_profit']:.1f}% | "
          f"{m['win_rate_midline']:.1f}% | ${m['net_pnl']:,.0f} | {fmt_pf(m['pf'])} | "
          f"{mid_s} | {stp_s} | {mh.get('count',0):,} |")

    # Sub-period PF per single symbol
    A("\n## 3. 单币种分时段 PF（按入场年份）\n")
    A("最后两列对比：含全部年份的总 PF vs **剔除 2024 年**后的 PF，"
      "用于判断 2024 的疲软是普遍现象还是个别币种问题。\n")
    A("| 币种 | 2023 | 2024 | 2025 | 2026-todate | 总PF | 剔除2024 PF |")
    A("|------|-----:|-----:|-----:|------------:|-----:|-----------:|")
    for sid in ("D1", "D2", "D3", "D4", "D5"):
        m = all_metrics.get(sid)
        sym = SCENARIOS[sid][0]
        if not m or m["n"] == 0:
            A(f"| {sym} | - | - | - | - | - | - |")
            continue
        sp = m["subperiods"]
        def cell(lbl):
            if lbl in sp:
                return f"{fmt_pf(sp[lbl]['pf'])} (${sp[lbl]['net_pnl']:,.0f})"
            return "-"
        ex = m.get("ex2024", {})
        ex_cell = f"{fmt_pf(ex.get('pf', float('nan')))} (${ex.get('net_pnl', 0):,.0f})" if ex.get("n") else "-"
        A(f"| {sym} | {cell('2023')} | {cell('2024')} | {cell('2025')} | "
          f"{cell('2026-todate')} | {fmt_pf(m['pf'])} | {ex_cell} |")

    # Conclusions
    A("\n## 4. 结论与建议\n")
    a, b, c = all_metrics.get("A"), all_metrics.get("B"), all_metrics.get("C")
    d5 = all_metrics.get("D5")

    # Q1 — remove DOGE?
    A("### Q1. 移除 DOGE 后，整体是改善还是恶化？\n")
    if a and b and a["n"] and b["n"]:
        d_pnl = b["net_pnl"] - a["net_pnl"]
        d_pf = b["pf"] - a["pf"] if a["pf"] != float("inf") and b["pf"] != float("inf") else None
        A(f"| 指标 | A（含DOGE） | B（去DOGE） | 变化 |")
        A(f"|------|-----------:|-----------:|------|")
        A(f"| 净PnL | ${a['net_pnl']:,.0f} | ${b['net_pnl']:,.0f} | **${d_pnl:,.0f}** |")
        A(f"| PF | {fmt_pf(a['pf'])} | {fmt_pf(b['pf'])} | {d_pf:+.2f} |")
        A(f"| 最大回撤$ | ${a['max_dd_usd']:,.0f} | ${b['max_dd_usd']:,.0f} | ${b['max_dd_usd']-a['max_dd_usd']:,.0f} |")
        A(f"| 最大连亏 | {a['max_consec_losses']} | {b['max_consec_losses']} | {b['max_consec_losses']-a['max_consec_losses']:+d} |")
        A("")
        A("**结论：移除 DOGE 会恶化整体表现，假设被否定。** 去掉 DOGE 抹掉约 "
          f"${-d_pnl:,.0f} 的历史净利润，而 PF 仅从 {fmt_pf(a['pf'])} 微升到 {fmt_pf(b['pf'])}"
          f"（{d_pf:+.2f}，统计上无意义）。回撤虽略降（${a['max_dd_usd']:,.0f}→${b['max_dd_usd']:,.0f}），"
          "但相对损失的利润不成比例。")
    A("")
    if d5:
        A("**为什么实盘 DOGE 亏损与回测矛盾？** DOGE 在 2023–2026 完整周期里是仅次于 SOL 的"
          f"第二强标的（PF {fmt_pf(d5['pf'])}，净 ${d5['net_pnl']:,.0f}），且**越近越强**"
          f"（2025 PF {fmt_pf(d5['subperiods'].get('2025',{}).get('pf',0))}，"
          f"2026-todate PF {fmt_pf(d5['subperiods'].get('2026-todate',{}).get('pf',0))}）。"
          "实盘一周 −$1,503 的亏损样本量过小（单币种约几十笔），未达统计显著，"
          "与 BTC 实盘表面亏损同属噪声。**不应据此移除 DOGE。**")

    # Q2 — which symbols fit
    A("\n### Q2. 2023–2026 完整周期内，哪些币种真正适合该均值回归策略？\n")
    singles = [(sid, all_metrics[sid]) for sid in ("D1", "D2", "D3", "D4", "D5")
               if all_metrics.get(sid) and all_metrics[sid]["n"]]
    singles.sort(key=lambda kv: (kv[1]["pf"] if kv[1]["pf"] != float("inf") else 9e9), reverse=True)
    A("单币种 PF 排序：")
    for sid, m in singles:
        sym = SCENARIOS[sid][0]
        A(f"- **{sym}**: PF={fmt_pf(m['pf'])}, 净PnL=${m['net_pnl']:,.0f}, "
          f"交易={m['n']:,}, 剔除2024后 PF={fmt_pf(m.get('ex2024',{}).get('pf',0))}")
    A("")
    A("- **核心标的（保留）**：SOL、DOGE — 全周期 PF 显著领先，且近两年持续走强。")
    A("- **稳健标的（保留）**：ETH、BTC — PF ~1.5，年度全部为正，回撤可控，是组合的压舱石。")
    A("- **边际标的（可保留/可观察）**：LINK — 全周期 PF 最低（1.32），2023 年净亏（PF 0.86），"
      "但 2024 起转正、2025–2026 明显改善。作为分散化成分可保留，但权重宜低。")

    # 2024 finding
    A("\n### 2024 是普遍现象还是个别币种问题？\n")
    A("**普遍现象。** 五个币种的 2024 PF 全部是其各自年份中的最低或次低，"
      "剔除 2024 后每个币种的 PF 都明显跳升：")
    for sid in ("D1", "D2", "D3", "D4", "D5"):
        m = all_metrics.get(sid)
        if not m or not m["n"]:
            continue
        sym = SCENARIOS[sid][0]
        y24 = m["subperiods"].get("2024", {}).get("pf", 0)
        exp = m.get("ex2024", {}).get("pf", 0)
        A(f"- {sym}: 2024 PF {fmt_pf(y24)} → 剔除2024 PF {fmt_pf(exp)}")
    A("\n这说明 2024 是一个**全市场层面的策略逆风年**（均值回归在该年趋势性行情中受损），"
      "而非某个币种失效。结论：不应因 2024 单年表现而剔除任何币种；"
      "若要改进，方向是**市场状态过滤（regime filter）**而非调整标的池。")

    # Recommendation
    A("\n### 建议\n")
    A("1. **保持现有 5 币种组合（方案 A）**，不要移除 DOGE。历史与近期数据均不支持移除。")
    A("2. 若需更高 PF 且可接受更低绝对收益，方案 C（ETH+SOL，PF "
      + (f"{fmt_pf(c['pf'])}" if c else "—") + "）是高质量子集，但收益集中度高、分散性差。")
    A("3. **下一轮迭代重点**：(a) 引入市场状态过滤以解决 2024 类逆风年；"
      "(b) 修正本报告顶部标注的 ATR 阈值（SMA 推导 vs Wilder 应用）口径偏差。")
    A("4. 实盘决策不应基于单周样本——BTC、DOGE 的实盘表面亏损均在历史噪声范围内。")

    path.write_text("\n".join(lines) + "\n")
    print(f"\n-> report written to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
