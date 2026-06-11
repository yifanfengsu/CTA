#!/usr/bin/env python3
"""MR-5m State Circuit Breaker — prototype backtest (Task A).

Tests whether a simple *reactive* circuit breaker (NOT a predictor) reduces
whipsaw-period losses without damaging healthy months. Breakers pause NEW
entries; existing positions exit normally; trading resumes after the pause.

Because a breaker is path-dependent (skipping an entry frees the strategy to
take a later one, and breaker state must build from actually-taken trades), the
breaker is integrated into the bar-level engine and the whole thing is
re-simulated — it is NOT a post-filter of the baseline trade list.

Breakers:
  A1  consecutive-stop:  pause after N consecutive stop exits
  A2  rolling-window PF: pause when PF over last W taken trades < threshold
  A3  combo:             pause if EITHER A1 or A2 fires (best params of each)

Scopes:
  per_symbol  each symbol has its own breaker (its own danger pattern)
  portfolio   one shared breaker watches all trades; a trip pauses ALL symbols

Reuses the validated engine from backtest_mr_5m_compare (Wilder ATR, current-bar
floating stop, close±1tick taker exits, maker-rebate/taker fees, int sizing).

Usage: python scripts/backtest_mr_5m_breaker.py
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_mr_5m_compare import (
    SYMBOLS, CONTRACT_SPECS, ATR_THRESHOLDS, LOOKBACK, ATR_STOP, MAX_HOLD,
    FEE_MAKER, FEE_TAKER, wilder_atr, calc_size, load_1m, r5,
    parse_history_range, compute_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = PROJECT_ROOT / ".vntrader" / "database.db"
START, END = "2023-01-01", "2026-05-28"
TIMEZONE = "UTC"
OUT_DIR = PROJECT_ROOT / "reports" / "regime"
SYM_NAMES = ["BTC", "ETH", "SOL", "LINK", "DOGE"]
PAUSE_HOURS = [4, 8, 24]
PAST = np.datetime64("1970-01-01")


# ── data prep (indicators computed once per symbol) ──────────────────────────
def prepare_symbol(name, bars):
    inst = SYMBOLS[name][1]
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    # naive UTC datetime64 → clean np.timedelta64 arithmetic + comparisons
    dt = bars["datetime"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    return {
        "name": name, "inst": inst,
        "dt": dt,                                 # datetime64[ns], naive UTC
        "close": c, "high": h, "low": l,
        "atr": wilder_atr(h, l, c),
        "dh": bars["high"].rolling(LOOKBACK).max().shift(1).to_numpy(),
        "dl": bars["low"].rolling(LOOKBACK).min().shift(1).to_numpy(),
        "n": len(c),
        "ct": CONTRACT_SPECS[inst]["ctVal"],
        "tick": CONTRACT_SPECS[inst]["tickSz"],
        "thr": ATR_THRESHOLDS[inst],
    }


# ── breakers ─────────────────────────────────────────────────────────────────
class NoBreaker:
    label = "baseline"
    def allow_entry(self, ts): return True
    def on_close(self, reason, net, exit_ts): pass


class ConsecStop:
    def __init__(self, n, pause_h):
        self.n = n
        self.pause = np.timedelta64(pause_h, "h")
        self.consec = 0
        self.pause_until = PAST
    def allow_entry(self, ts):
        return ts >= self.pause_until
    def on_close(self, reason, net, exit_ts):
        if reason == "stop":
            self.consec += 1
        else:
            self.consec = 0
        if self.consec >= self.n:
            self.pause_until = exit_ts + self.pause
            self.consec = 0


class RollingPF:
    def __init__(self, w, thr, pause_h):
        self.w = w
        self.thr = thr
        self.pause = np.timedelta64(pause_h, "h")
        self.buf = deque(maxlen=w)
        self.pause_until = PAST
    def allow_entry(self, ts):
        return ts >= self.pause_until
    def on_close(self, reason, net, exit_ts):
        self.buf.append(net)
        if len(self.buf) == self.w:
            wins = sum(x for x in self.buf if x > 0)
            losses = -sum(x for x in self.buf if x < 0)
            pf = (wins / losses) if losses > 0 else float("inf")
            if pf < self.thr:
                self.pause_until = exit_ts + self.pause
                self.buf.clear()  # require W fresh trades before re-arming


class Combo:
    def __init__(self, *subs):
        self.subs = subs
    def allow_entry(self, ts):
        return all(s.allow_entry(ts) for s in self.subs)
    def on_close(self, reason, net, exit_ts):
        for s in self.subs:
            s.on_close(reason, net, exit_ts)


# ── one-bar step (shared by single-symbol and portfolio loops) ───────────────
def _step(d, i, st, breaker, trades):
    """Process symbol d at bar index i with mutable state st. Returns 1 if a
    valid entry signal was blocked by the breaker, else 0."""
    a = d["atr"][i]
    if a != a or a <= 0:
        return 0
    c = d["close"]; h = d["high"]; l = d["low"]
    ts = d["dt"][i]

    if st["pos"] != 0:
        pos = st["pos"]; ep = st["ep"]; eb = st["eb"]
        reason = ""
        dhi = d["dh"][i]; dli = d["dl"][i]
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
            ct = d["ct"]; tick = d["tick"]; esize = st["sz"]
            exit_px = c[i] - tick if pos == 1 else c[i] + tick
            gross = (exit_px - ep) * esize * ct if pos == 1 else (ep - exit_px) * esize * ct
            en = ep * esize * ct; xn = exit_px * esize * ct
            fee = (-FEE_MAKER * en) - (FEE_TAKER * xn)
            net = gross + fee
            trades.append({
                "symbol": d["name"],
                "side": "long" if pos == 1 else "short",
                "exit_reason": reason,
                "entry_price": round(ep, 8), "exit_price": round(exit_px, 8),
                "size": esize,
                "gross_pnl_usd": round(gross, 4), "fee_usd": round(fee, 4),
                "net_pnl_usd": round(net, 4),
                "entry_time": pd.Timestamp(st["et"]).tz_localize("UTC").isoformat(),
                "time": pd.Timestamp(ts).tz_localize("UTC").isoformat(),
            })
            breaker.on_close(reason, net, ts)
            st["pos"] = 0
            return 0

    if st["pos"] == 0:
        if d["thr"] > 0 and a < d["thr"]:
            return 0
        dhi = d["dh"][i]; dli = d["dl"][i]
        if dhi != dhi or dli != dli or dhi <= 0 or dli <= 0:
            return 0
        cc = c[i]
        if cc > dhi:
            side = -1
        elif cc < dli:
            side = 1
        else:
            return 0
        if not breaker.allow_entry(ts):
            return 1  # blocked
        st["pos"] = side; st["ep"] = cc; st["sz"] = calc_size(d["inst"], cc)
        st["eb"] = i; st["et"] = ts
    return 0


def _new_state():
    return {"pos": 0, "eb": -1, "ep": 0.0, "sz": 0, "et": None}


def run_per_symbol(datas, make_breaker):
    trades = []; blocked = 0
    for nm in SYM_NAMES:
        d = datas[nm]; br = make_breaker(); st = _new_state()
        for i in range(LOOKBACK + 1, d["n"]):
            blocked += _step(d, i, st, br, trades)
    return trades, blocked


def run_portfolio(datas, events, make_breaker):
    br = make_breaker()
    st = {nm: _new_state() for nm in SYM_NAMES}
    trades = []; blocked = 0
    ts_arr, sym_arr, i_arr = events
    for k in range(len(i_arr)):
        nm = SYM_NAMES[sym_arr[k]]
        blocked += _step(datas[nm], i_arr[k], st[nm], br, trades)
    return trades, blocked


def build_events(datas):
    all_ts, all_sym, all_i = [], [], []
    for k, nm in enumerate(SYM_NAMES):
        d = datas[nm]
        rng = np.arange(LOOKBACK + 1, d["n"])
        all_ts.append(d["dt"][rng])
        all_sym.append(np.full(len(rng), k, dtype=np.int8))
        all_i.append(rng.astype(np.int32))
    ts_arr = np.concatenate(all_ts)
    sym_arr = np.concatenate(all_sym)
    i_arr = np.concatenate(all_i)
    order = np.argsort(ts_arr, kind="stable")
    return ts_arr[order], sym_arr[order], i_arr[order]


# ── monthly helpers ──────────────────────────────────────────────────────────
def monthly_net(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df["m"] = pd.to_datetime(df["entry_time"], utc=True).dt.to_period("M").astype(str)
    return df.groupby("m")["net_pnl_usd"].sum().to_dict()


def monthly_pf(trades):
    df = pd.DataFrame(trades)
    df["m"] = pd.to_datetime(df["entry_time"], utc=True).dt.to_period("M").astype(str)
    out = {}
    for m, g in df.groupby("m"):
        w = g[g["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
        l = abs(g[g["net_pnl_usd"] < 0]["net_pnl_usd"].sum())
        out[m] = (w / l) if l > 0 else float("inf")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hr = parse_history_range(START, END, timedelta(minutes=1), TIMEZONE)
    print(f"Range {START}..{END}; loading + preparing indicators...", flush=True)
    datas = {}
    for nm in SYM_NAMES:
        b5 = r5(load_1m(SYMBOLS[nm][0], hr, DB), 5, hr)
        datas[nm] = prepare_symbol(nm, b5)
        print(f"  [{nm}] {datas[nm]['n']:,} 5m bars", flush=True)
    events = build_events(datas)

    # ── baseline ──
    base_trades, _ = run_per_symbol(datas, NoBreaker)
    base_m = compute_metrics(base_trades)
    base_mnet = monthly_net(base_trades)
    base_mpf = monthly_pf(base_trades)
    fail_months = sorted([m for m, pf in base_mpf.items() if pf < 1.0])
    base_fail = sum(base_mnet[m] for m in fail_months)
    base_heal = sum(v for m, v in base_mnet.items() if m not in fail_months)
    print(f"\nBASELINE: trades={base_m['n']:,} net=${base_m['net_pnl']:,.0f} "
          f"PF={base_m['pf']:.2f} DD=${base_m['max_dd_usd']:,.0f}")
    print(f"  fail months ({len(fail_months)}): {fail_months}")
    print(f"  fail net=${base_fail:,.0f}  healthy net=${base_heal:,.0f}")

    def evaluate(trades, blocked, scope, label):
        m = compute_metrics(trades)
        mnet = monthly_net(trades)
        cfg_fail = sum(mnet.get(mo, 0.0) for mo in fail_months)
        cfg_heal = sum(v for mo, v in mnet.items() if mo not in fail_months)
        return {
            "scope": scope, "label": label,
            "trades": m["n"], "blocked": blocked,
            "dtrades_pct": (m["n"] / base_m["n"] - 1) * 100,
            "net": m["net_pnl"], "pf": m["pf"], "dd": m["max_dd_usd"],
            "fail_net": cfg_fail, "heal_net": cfg_heal,
            "d_fail": cfg_fail - base_fail,
            "d_heal": cfg_heal - base_heal,
            "d_total": m["net_pnl"] - base_m["net_pnl"],
        }

    results = []

    def run_cfg(scope, label, make_breaker):
        if scope == "per_symbol":
            tr, bl = run_per_symbol(datas, make_breaker)
        else:
            tr, bl = run_portfolio(datas, events, make_breaker)
        r = evaluate(tr, bl, scope, label)
        results.append(r)
        print(f"  [{scope:10}] {label:28} trades={r['trades']:,} "
              f"Δfail=${r['d_fail']:+,.0f} Δheal=${r['d_heal']:+,.0f} "
              f"Δtot=${r['d_total']:+,.0f} PF={r['pf']:.2f}", flush=True)
        return r

    # ── A1 grid ──
    print("\n=== A1 consecutive-stop ===")
    for scope in ("per_symbol", "portfolio"):
        for n in (3, 5, 7):
            for ph in PAUSE_HOURS:
                run_cfg(scope, f"A1 N={n} pause={ph}h",
                        (lambda n=n, ph=ph: ConsecStop(n, ph)))

    # ── A2 grid ──
    print("\n=== A2 rolling-PF ===")
    for scope in ("per_symbol", "portfolio"):
        for w in (20, 50, 100):
            for thr in (0.7, 0.8, 0.9):
                for ph in PAUSE_HOURS:
                    run_cfg(scope, f"A2 W={w} thr={thr} pause={ph}h",
                            (lambda w=w, thr=thr, ph=ph: RollingPF(w, thr, ph)))

    # ── A3 combo: best A1 + best A2 per scope ──
    print("\n=== A3 combo (best A1 OR best A2) ===")
    def parse_a1(lbl):
        p = dict(kv.split("=") for kv in lbl.split()[1:])
        return int(p["N"]), int(p["pause"].rstrip("h"))
    def parse_a2(lbl):
        p = dict(kv.split("=") for kv in lbl.split()[1:])
        return int(p["W"]), float(p["thr"]), int(p["pause"].rstrip("h"))

    a3_choices = {}
    for scope in ("per_symbol", "portfolio"):
        best_a1 = pick_best([r for r in results if r["scope"] == scope and r["label"].startswith("A1")], base_heal)
        best_a2 = pick_best([r for r in results if r["scope"] == scope and r["label"].startswith("A2")], base_heal)
        n, ph1 = parse_a1(best_a1["label"])
        w, thr, ph2 = parse_a2(best_a2["label"])
        a3_choices[scope] = (best_a1["label"], best_a2["label"])
        run_cfg(scope, f"A3 [{best_a1['label']}] OR [{best_a2['label']}]",
                (lambda n=n, ph1=ph1, w=w, thr=thr, ph2=ph2:
                 Combo(ConsecStop(n, ph1), RollingPF(w, thr, ph2))))

    write_report(base_m, base_fail, base_heal, fail_months, results, a3_choices)
    print(f"\n-> wrote {OUT_DIR/'regime_filter_report.md'}")


def pick_best(rows, base_heal):
    """Best = max Δfail (loss reduction) among configs that damage healthy net by
    <=1% of baseline healthy; fallback to max Δtotal."""
    cap = 0.01 * abs(base_heal)
    safe = [r for r in rows if r["d_heal"] >= -cap]
    pool = safe if safe else rows
    return max(pool, key=lambda r: (r["d_fail"], r["d_total"]))


def write_report(base_m, base_fail, base_heal, fail_months, results, a3_choices):
    L = []; A = L.append
    A("# MR-5m 状态熔断器（State Circuit Breaker）原型回测 — 任务A\n")
    A(f"- 区间: {START} → {END}（{TIMEZONE}），全部 5 币种")
    A("- 机制: 熔断仅暂停**新开仓**，已有持仓正常出场；暂停结束后从下一个信号恢复")
    A("- 熔断是**事后反应**（识别已进入危险区），非预测；基于**实际成交**的交易序列重算")
    A("- 引擎与基准回测一致（Wilder ATR / 当根浮动止损 / close±1tick taker 出场 / maker返佣+taker费）\n")
    A(f"**基准（方案A，无熔断）**：交易 {base_m['n']:,} 笔，净 ${base_m['net_pnl']:,.0f}，"
      f"PF {base_m['pf']:.2f}，最大回撤 ${base_m['max_dd_usd']:,.0f}。\n")
    A(f"失效月（基准组合月 PF<1，共 {len(fail_months)} 个）：{', '.join(fail_months)}")
    A(f"- 失效月合计净 PnL：${base_fail:,.0f}")
    A(f"- 健康月合计净 PnL：${base_heal:,.0f}\n")
    A("> 判据：在**几乎不误伤健康月**（健康月净损失 ≤ 基准健康月 1%）的前提下，"
      "最大化**减少失效月亏损**（Δfail 越正越好）。Δtot = 相对基准的总净变化。\n")

    def table(rows, title):
        A(f"\n### {title}\n")
        A("| 方案 | 作用域 | 交易数 | Δ交易% | 净PnL | Δtot | PF | 回撤$ | Δfail | Δheal |")
        A("|------|--------|------:|------:|------:|-----:|---:|------:|------:|------:|")
        for r in rows:
            A(f"| {r['label']} | {r['scope']} | {r['trades']:,} | {r['dtrades_pct']:+.1f}% | "
              f"${r['net']:,.0f} | ${r['d_total']:+,.0f} | {r['pf']:.2f} | ${r['dd']:,.0f} | "
              f"${r['d_fail']:+,.0f} | ${r['d_heal']:+,.0f} |")

    A("## 1. A1 连续止损熔断\n")
    for scope in ("per_symbol", "portfolio"):
        table([r for r in results if r["scope"] == scope and r["label"].startswith("A1")],
              f"A1 — {scope}")
    A("\n## 2. A2 滚动短窗 PF 熔断\n")
    for scope in ("per_symbol", "portfolio"):
        table([r for r in results if r["scope"] == scope and r["label"].startswith("A2")],
              f"A2 — {scope}")
    A("\n## 3. A3 组合熔断（A1 OR A2）\n")
    for scope in ("per_symbol", "portfolio"):
        A(f"- {scope}: 选用 [{a3_choices[scope][0]}] OR [{a3_choices[scope][1]}]")
    table([r for r in results if r["label"].startswith("A3")], "A3 — combo")

    # ── recommendation ──
    A("\n## 4. 结论与推荐\n")
    best = pick_best(results, base_heal)
    pos = [r for r in results if r["d_total"] > 0]
    A(f"- 全部 {len(results)} 个配置中，相对基准**总净 PnL 改善（Δtot>0）的有 {len(pos)} 个**。")
    A(f"- 综合最优（先保健康月、再压失效月亏损）：**{best['label']} / {best['scope']}**")
    A(f"  - Δfail=${best['d_fail']:+,.0f}，Δheal=${best['d_heal']:+,.0f}，"
      f"Δtot=${best['d_total']:+,.0f}，交易 {best['dtrades_pct']:+.1f}%，PF {best['pf']:.2f}。")
    _verdict(A, base_m, base_fail, best, pos, results)

    (OUT_DIR / "regime_filter_report.md").write_text("\n".join(L) + "\n")


def _verdict(A, base_m, base_fail, best, pos, results):
    A("\n**判断（数据驱动，不拍脑袋）：**\n")
    if best["d_total"] <= 0 or best["d_fail"] <= 0:
        A("1. 没有任何配置能在不误伤健康月的前提下实质改善整体表现："
          f"综合最优配置 Δtot=${best['d_total']:+,.0f}。熔断器在该策略上**得不偿失**。")
        A("2. 根因（呼应上轮发现）：失效月的亏损本身**幅度小、分布散**"
          f"（基准失效月合计仅 ${base_fail:,.0f}），且与健康月共用同一套信号；"
          "反应式熔断在止损成串后才暂停，往往刚好错过暂停窗口后的反弹/盈利信号，"
          "削掉的健康月利润 > 省下的失效月亏损。")
        A("3. **暂不上熔断器。** 维持方案A原样。")
        A("\n**「不需要熔断」的条件说明：** 当 (a) 失效月亏损占总盈利比例很低、"
          "(b) 失效与健康共用信号源、(c) 暂停窗口后常出现反弹信号 —— 三者同时成立时，"
          "事后反应式熔断的期望收益为负。本策略三条全中，故不需要。")
        A("4. 若仍要追求降回撤，应转向**任务B（前瞻指标）**：在开仓前识别 whipsaw，"
          "而非事后暂停——这正是任务B要验证的方向。")
    else:
        A(f"1. 存在正收益配置（{len(pos)} 个）。综合最优 **{best['label']} / {best['scope']}**："
          f"在健康月几乎无损（Δheal=${best['d_heal']:+,.0f}）下，失效月减亏 ${best['d_fail']:+,.0f}，"
          f"总净 +${best['d_total']:,.0f}。")
        A("2. 推荐作为**保守加装**：参数偏紧（少触发），优先保护健康月。")
        A("3. 即便如此，增益有限，仍建议把主要精力放在任务B的前瞻指标上。")


if __name__ == "__main__":
    raise SystemExit(main())
