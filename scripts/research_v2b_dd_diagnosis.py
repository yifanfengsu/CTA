#!/usr/bin/env python3
"""v2B max-DD culprit diagnosis — the 2025-05-29 cluster-loss event.

PURE DIAGNOSIS. No fix, no new risk rule, no parameter change, no engine edit.
Reuses research_dyn_v2 (which reuses backtest_mr_5m_compare). Touches nothing live.

Steps
 0  reproduce v2B-no-cap maxDD window; must match size-cap-V1 report
    (2025-05-29 09:49 -> 16:14 UTC, 16 trades, DD $3,425.57); dump full detail
 1  basic feature stats of the 16 trades
 2  market level: event-day OHLC plot, 5m-return correlation vs prior 30 days,
    ATR-quantile structure of the day vs all days
 3  strategy behaviour: entry trap pattern, MR reversion lag, repeat entries,
    v2B tier amplification vs FLAT
 4  replication scan. Steps 0-3 revealed the event is a DATA ANOMALY (linear
    +-~10/min staircase ramps in SOL/DOGE with instant full reversion, no
    confirmation in BTC/ETH/LINK, no real-market counterpart). So the step-4
    "pattern" is quantified as a synthetic-ramp detector on 1m closes:
      A) staircase run: >=4 consecutive same-sign 1m returns each |r|>=1.5%,
         cumulative |move|>=10%
      B) single-bar jump: |1m return|>=15%
    runs/jumps within 60min merge into one event; an event is "reverting" if
    price returns to within 2% of the pre-event level within 4h of event end.
    Every event is listed with the strategy PnL (FLAT, and v2B where in test)
    of the event-symbol trades overlapping the event window.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from research_dyn_v2 import (  # noqa: E402
    build_trades, trade_net, make_rules, learn_thresholds,
    SYMBOLS, SMALL, MED, LARGE, FLAT_NOTIONAL, START, END, TZ,
)
from backtest_mr_5m_compare import wilder_atr, LOOKBACK  # noqa: E402
from research_mr_5m import load_1m, r5  # noqa: E402
from history_time_utils import parse_history_range  # noqa: E402

OUT_DIR = PROJECT_ROOT / "reports" / "regime" / "v2b_dd_diagnosis_20260610"
CACHE = OUT_DIR / "_b5_cache.pkl"

EVENT_DAY = pd.Timestamp("2025-05-29", tz="UTC")
# expected window from reports/regime/dyn_v2b_size_cap_v1_20260609/dd_culprit_analysis.json
EXPECT = {"max_dd": 3425.57, "n": 16,
          "peak_dt": "2025-05-29 09:49:00+00:00",
          "trough_dt": "2025-05-29 16:14:00+00:00"}

REVERT_HORIZON = 288          # bars (24h) to look for post-entry midline touch
TIER_NAME = {SMALL: "small", MED: "medium", LARGE: "large"}

# step-4 synthetic-ramp detector params (derived from the 2025-05-29 anomaly:
# SOL/DOGE moved in uniform ~6-9%/min staircases, 13 consecutive bars, then
# jumped back to the pre-event price in a single bar)
RAMP_STEP = 0.015             # per-1m-bar |return| that counts as a ramp step
RAMP_MIN_LEN = 4              # consecutive same-sign steps
RAMP_MIN_MOVE = 0.10          # cumulative |move| of the run
JUMP_1BAR = 0.15              # single-bar anomaly threshold
MERGE_GAP_MIN = 60            # merge runs/jumps within this many minutes
REVERT_TOL = 0.02             # back within 2% of pre-event price ...
REVERT_WIN_MIN = 240          # ... within 4h after event end
STAIR_CV = 0.30               # step-uniformity flag: cv(price steps) <= this


# ── context arrays per symbol (same formulas as build_trades; read-only) ─────
def build_ctx(b5):
    dt = b5["datetime"].to_numpy()
    h = b5["high"].to_numpy(dtype=float)
    l = b5["low"].to_numpy(dtype=float)
    c = b5["close"].to_numpy(dtype=float)
    o = b5["open"].to_numpy(dtype=float)
    v = b5["volume"].to_numpy(dtype=float)
    atr = wilder_atr(h, l, c)
    dh = b5["high"].rolling(LOOKBACK).max().shift(1).to_numpy()
    dl = b5["low"].rolling(LOOKBACK).min().shift(1).to_numpy()
    atr_ma = pd.Series(atr).rolling(24).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        atr_ratio = atr / atr_ma
    slot = b5["open_time"].dt.floor("5min").to_numpy()
    return {"dt": dt, "o": o, "h": h, "l": l, "c": c, "v": v, "atr": atr,
            "dh": dh, "dl": dl, "mid": (dh + dl) / 2.0, "atr_ratio": atr_ratio,
            "slot": slot, "idx": {ts: i for i, ts in enumerate(dt)}}


def q_of(sorted_arr, x):
    """quantile (0..1) of x in a sorted train distribution"""
    if not np.isfinite(x) or len(sorted_arr) == 0:
        return None
    return float(np.searchsorted(sorted_arr, x) / len(sorted_arr))


# ── v2B-no-cap DD window on the test set ──────────────────────────────────────
def v2b_dd_window(test, th_by_sym):
    rows = []
    for tr in test:
        notion = make_rules(th_by_sym[tr["symbol"]])["v2B"](tr)
        net, deployed, size = trade_net(tr, notion)
        rows.append((tr["exit_dt"], net, notion, deployed, size, tr))
    rows.sort(key=lambda x: x[0])
    cum = peak = 0.0
    peak = -1e18
    peak_idx = trough_idx = peak_at_trough = -1
    best_dd = cum = 0.0
    for i, r in enumerate(rows):
        cum += r[1]
        if cum > peak:
            peak, peak_idx = cum, i
        if peak - cum > best_dd:
            best_dd, trough_idx, peak_at_trough = peak - cum, i, peak_idx
    window = rows[peak_at_trough + 1: trough_idx + 1]
    meta = {"max_dd": round(best_dd, 2), "peak_idx": peak_at_trough,
            "trough_idx": trough_idx, "n_window_trades": len(window),
            "window_peak_dt": str(rows[peak_at_trough][0]),
            "window_trough_dt": str(rows[trough_idx][0])}
    return meta, window, rows


# ── enrich one window trade with full diagnostic detail ───────────────────────
def enrich(row, ctx_by_sym, th_by_sym, trainq):
    exit_dt, net, notion, deployed, size, tr = row
    sym = tr["symbol"]
    ctx = ctx_by_sym[sym]
    e = tr["entry_idx"]
    x = ctx["idx"][tr["exit_dt"]]
    atr_e, dh_e, dl_e, mid_e = ctx["atr"][e], ctx["dh"][e], ctx["dl"][e], ctx["mid"][e]
    side = "long" if tr["side"] == 1 else "short"
    # breakout excess beyond the broken band, in ATR units
    if tr["side"] == -1:
        excess_atr = (tr["entry_px"] - dh_e) / atr_e
    else:
        excess_atr = (dl_e - tr["entry_px"]) / atr_e
    # post-entry reversion: first bar i>e where strategy midline condition holds
    revert_off = None
    hi = min(e + REVERT_HORIZON, len(ctx["c"]) - 1)
    for i in range(e + 1, hi + 1):
        m = ctx["mid"][i]
        if not np.isfinite(m):
            continue
        if (tr["side"] == -1 and ctx["c"][i] <= m) or (tr["side"] == 1 and ctx["c"][i] >= m):
            revert_off = i - e
            break
    return {
        "symbol": sym, "side": side,
        "entry_dt": str(tr["entry_dt"]), "exit_dt": str(tr["exit_dt"]),
        "entry_px": tr["entry_px"], "exit_px": tr["exit_px"],
        "exit_reason": tr["exit_reason"],
        "hold_bars": int(x - e),
        "atr_abs": tr["atr_abs"], "atr_q_train": round(trainq[sym]["atr"](tr["atr_abs"]), 4),
        "atr_ratio": (None if not np.isfinite(tr["atr_ratio"]) else round(tr["atr_ratio"], 4)),
        "atr_ratio_q_train": (None if not np.isfinite(tr["atr_ratio"]) else
                              round(trainq[sym]["ratio"](tr["atr_ratio"]), 4)),
        "donchian_up": round(float(dh_e), 6), "donchian_dn": round(float(dl_e), 6),
        "donchian_mid": round(float(mid_e), 6),
        "channel_width_atr": round(float((dh_e - dl_e) / atr_e), 3),
        "breakout_excess_atr": round(float(excess_atr), 4),
        "v2b_notional": notion, "v2b_tier": TIER_NAME[notion],
        "deployed_notional": round(deployed, 2), "size": size,
        "net_pnl": round(net, 2),
        "flat_net_pnl": round(trade_net(tr, FLAT_NOTIONAL)[0], 2),
        "bars_to_midline_revert": revert_off,   # None = never within 24h
        "reverted_after_exit": (revert_off is not None and revert_off > (x - e)),
    }


# ── step-4: synthetic-ramp anomaly detector on 1m closes ─────────────────────
def detect_ramp_events(dt_ns, c):
    """dt_ns = int64 epoch-ns array, c = 1m closes. Returns merged events."""
    gap_min = np.diff(dt_ns) / 60_000_000_000
    r = np.diff(c) / c[:-1]
    valid = gap_min <= 2.0                      # don't treat data gaps as moves
    cands = []                                  # (i_start, i_end) bar idx into c
    # A) staircase runs
    i = 0
    n = len(r)
    while i < n:
        if not (valid[i] and abs(r[i]) >= RAMP_STEP):
            i += 1
            continue
        s = np.sign(r[i])
        j = i
        while j + 1 < n and valid[j + 1] and abs(r[j + 1]) >= RAMP_STEP and np.sign(r[j + 1]) == s:
            j += 1
        if j - i + 1 >= RAMP_MIN_LEN and abs(c[j + 1] / c[i] - 1) >= RAMP_MIN_MOVE:
            cands.append((i, j + 1))
        i = j + 1
    # B) single-bar jumps
    for k in np.where(valid & (np.abs(r) >= JUMP_1BAR))[0]:
        cands.append((int(k), int(k) + 1))
    if not cands:
        return []
    cands.sort()
    merged = [list(cands[0])]
    for a, b in cands[1:]:
        if (dt_ns[a] - dt_ns[merged[-1][1]]) / 60_000_000_000 <= MERGE_GAP_MIN:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    events = []
    for a, b in merged:
        ref = c[max(0, a - 1)]
        seg = c[a:b + 1]
        exc = max(abs(seg.max() / ref - 1), abs(seg.min() / ref - 1))
        # revert check
        hi = dt_ns[b] + REVERT_WIN_MIN * 60_000_000_000
        k = b + 1
        reverted = False
        while k < len(c) and dt_ns[k] <= hi:
            if abs(c[k] / ref - 1) <= REVERT_TOL:
                reverted = True
                break
            k += 1
        # staircase-uniformity flag on the longest run inside the event
        steps = np.diff(c[a:b + 1])
        stair = False
        if len(steps) >= RAMP_MIN_LEN:
            sgn = np.sign(steps)
            i0 = 0
            for i1 in range(1, len(steps) + 1):
                if i1 == len(steps) or sgn[i1] != sgn[i0]:
                    run = steps[i0:i1]
                    if len(run) >= 5 and abs(run.mean()) > 0:
                        if run.std() / abs(run.mean()) <= STAIR_CV:
                            stair = True
                    i0 = i1
        events.append({"i0": int(a), "i1": int(b), "ref_px": float(ref),
                       "max_excursion_pct": round(float(exc) * 100, 2),
                       "reverted": reverted, "staircase": stair})
    return events


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []

    def L(m=""):
        print(m, flush=True)
        log.append(m)

    t0 = datetime.now(timezone.utc)
    L(f"# v2B DD diagnosis (2025-05-29) — start {t0.isoformat()}")
    L(f"Range {START}->{END} {TZ}; engines reused UNMODIFIED; diagnosis only")
    L("")

    # ── load bars (cached) ────────────────────────────────────────────────────
    if CACHE.exists():
        b5_by_sym = pickle.loads(CACHE.read_bytes())
        L("loaded b5 from cache")
    else:
        hr = parse_history_range(START, END, timedelta(minutes=1), TZ)
        db = PROJECT_ROOT / ".vntrader" / "database.db"
        b5_by_sym = {}
        for name in SYMBOLS:
            b1 = load_1m(SYMBOLS[name][0], hr, db)
            b5_by_sym[name] = r5(b1, 5, hr)
            L(f"[{name}] {len(b5_by_sym[name]):,} 5m bars")
        CACHE.write_bytes(pickle.dumps(b5_by_sym))
    L("")

    # ── trades + thresholds + ctx ─────────────────────────────────────────────
    all_trades, ctx_by_sym, th_by_sym, split_by_sym, trainq = [], {}, {}, {}, {}
    for name in SYMBOLS:
        b5 = b5_by_sym[name]
        trades, split_idx, n, atr, dt = build_trades(name, b5)
        ctx = build_ctx(b5)
        mask = np.arange(n) < split_idx
        th_by_sym[name] = learn_thresholds(atr, mask)
        split_by_sym[name] = split_idx
        a = atr[mask]; a = np.sort(a[np.isfinite(a) & (a > 0)])
        r = ctx["atr_ratio"][mask]; r = np.sort(r[np.isfinite(r)])
        trainq[name] = {"atr": (lambda x, _a=a: q_of(_a, x)),
                        "ratio": (lambda x, _r=r: q_of(_r, x)),
                        "atr_sorted": a}
        ctx_by_sym[name] = ctx
        all_trades.extend(trades)
        L(f"[{name}] {n:,} bars split@{split_idx:,} | {len(trades):,} trades")
    test = [t for t in all_trades if t["entry_idx"] >= split_by_sym[t["symbol"]]]
    L(f"TEST trades: {len(test):,}")
    L("")

    # ══ STEP 0: reproduce DD window ═══════════════════════════════════════════
    meta, window, rows_all = v2b_dd_window(test, th_by_sym)
    L("## STEP 0 — v2B-no-cap maxDD window")
    L(json.dumps(meta))
    ok = (abs(meta["max_dd"] - EXPECT["max_dd"]) < 1.0
          and meta["n_window_trades"] == EXPECT["n"]
          and meta["window_peak_dt"] == EXPECT["peak_dt"]
          and meta["window_trough_dt"] == EXPECT["trough_dt"])
    L(f"matches size-cap-V1 report: {ok}")
    if not ok:
        L("!! window mismatch vs prior report — stopping for inspection")
        (OUT_DIR / "run_log.txt").write_text("\n".join(log))
        return

    detail = [enrich(r, ctx_by_sym, th_by_sym, trainq) for r in window]
    detail.sort(key=lambda d: d["entry_dt"])
    (OUT_DIR / "trades_detail.json").write_text(json.dumps(
        {"window_meta": meta, "trades": detail}, indent=2))
    L("\n16-trade detail (by entry):")
    hdr = (f"{'sym':5} {'side':5} {'entry_dt':16} {'exit_dt':16} {'reason':8} "
           f"{'hold':>4} {'atrQ':>5} {'ratQ':>5} {'tier':6} {'net$':>8} {'flat$':>7} "
           f"{'exc.ATR':>7} {'rev@':>5}")
    L(hdr)
    for d in detail:
        L(f"{d['symbol']:5} {d['side']:5} {d['entry_dt'][5:16]:16} {d['exit_dt'][5:16]:16} "
          f"{d['exit_reason']:8} {d['hold_bars']:>4} {d['atr_q_train']:>5.2f} "
          f"{(d['atr_ratio_q_train'] or 0):>5.2f} {d['v2b_tier']:6} {d['net_pnl']:>8.2f} "
          f"{d['flat_net_pnl']:>7.2f} {d['breakout_excess_atr']:>7.3f} "
          f"{str(d['bars_to_midline_revert']):>5}")

    # ══ STEP 1: feature stats ═════════════════════════════════════════════════
    L("\n## STEP 1 — feature stats over the 16 trades")
    df = pd.DataFrame(detail)
    stats = {
        "by_symbol": df.groupby("symbol")["net_pnl"].agg(["count", "sum"]).round(2).to_dict("index"),
        "by_side": df.groupby("side")["net_pnl"].agg(["count", "sum"]).round(2).to_dict("index"),
        "by_exit_reason": df.groupby("exit_reason")["net_pnl"].agg(["count", "sum"]).round(2).to_dict("index"),
        "by_tier": df.groupby("v2b_tier")["net_pnl"].agg(["count", "sum"]).round(2).to_dict("index"),
        "atr_q_train": df["atr_q_train"].describe().round(3).to_dict(),
        "atr_ratio": df["atr_ratio"].describe().round(3).to_dict(),
        "atr_ratio_q_train": df["atr_ratio_q_train"].describe().round(3).to_dict(),
        "hold_bars": df["hold_bars"].describe().round(1).to_dict(),
        "entry_hours_utc": sorted(pd.to_datetime(df["entry_dt"]).dt.hour.tolist()),
        "net_total": round(df["net_pnl"].sum(), 2),
        "flat_total": round(df["flat_net_pnl"].sum(), 2),
    }
    L(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "step1_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    # ══ STEP 2: market level ══════════════════════════════════════════════════
    L("\n## STEP 2 — market level (event day)")
    day_lo, day_hi = EVENT_DAY, EVENT_DAY + pd.Timedelta(days=1)

    # (2a) per-symbol day OHLC + plot with entries/exits
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    day_ohlc = {}
    fig, axes = plt.subplots(len(SYMBOLS), 1, figsize=(14, 16), sharex=True)
    for ax, name in zip(axes, SYMBOLS):
        ctx = ctx_by_sym[name]
        m = (ctx["dt"] >= day_lo) & (ctx["dt"] < day_hi)
        idxs = np.where(m)[0]
        ts = [pd.Timestamp(t) for t in ctx["dt"][m]]
        day_ohlc[name] = {
            "t": [str(t) for t in ts],
            "o": ctx["o"][m].tolist(), "h": ctx["h"][m].tolist(),
            "l": ctx["l"][m].tolist(), "c": ctx["c"][m].tolist(),
            "v": ctx["v"][m].tolist(),
            "atr_q": [q_of(trainq[name]["atr_sorted"], x) for x in ctx["atr"][m]],
        }
        ax.fill_between(ts, ctx["l"][m], ctx["h"][m], alpha=0.25, lw=0)
        ax.plot(ts, ctx["c"][m], lw=0.9)
        if np.isfinite(ctx["dh"][idxs]).any():
            ax.plot(ts, ctx["dh"][m], lw=0.6, ls="--")
            ax.plot(ts, ctx["dl"][m], lw=0.6, ls="--")
            ax.plot(ts, ctx["mid"][m], lw=0.6, ls=":")
        for d in detail:
            if d["symbol"] != name:
                continue
            edt, xdt = pd.Timestamp(d["entry_dt"]), pd.Timestamp(d["exit_dt"])
            mk = "v" if d["side"] == "short" else "^"
            ax.plot([edt], [d["entry_px"]], marker=mk, ms=9, color="red", zorder=5)
            ax.plot([xdt], [d["exit_px"]], marker="x", ms=8, color="black", zorder=5)
        ax.set_ylabel(name)
    axes[0].set_title("2025-05-29 UTC — 5m close (band=hi/lo, dashes=Donchian, red=entry, x=exit)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "event_day_chart.png", dpi=110)
    plt.close(fig)
    L("chart -> event_day_chart.png")

    # (2b)+(2c)+step-4 base: daily feature table over the whole period
    closes = {}
    for name in SYMBOLS:
        b5 = b5_by_sym[name]
        s = pd.Series(b5["close"].to_numpy(dtype=float),
                      index=b5["open_time"].dt.floor("5min"))
        closes[name] = s[~s.index.duplicated(keep="last")]
    panel = pd.DataFrame(closes).sort_index()
    rets = np.log(panel).diff()
    rets["date"] = rets.index.date

    def day_corr(g):
        g = g[list(SYMBOLS)].dropna(how="all")
        if len(g) < 100:
            return np.nan
        cm = g.corr().to_numpy()
        iu = np.triu_indices_from(cm, k=1)
        return float(np.nanmean(cm[iu]))

    corr_by_day = rets.groupby("date").apply(day_corr)

    daily_rows = []
    for name in SYMBOLS:
        ctx = ctx_by_sym[name]
        dts = pd.to_datetime(pd.Series(ctx["dt"]))
        dd = pd.DataFrame({
            "date": dts.dt.date, "o": ctx["o"], "h": ctx["h"], "l": ctx["l"],
            "c": ctx["c"],
            "atr_q": np.searchsorted(trainq[name]["atr_sorted"], ctx["atr"]) / max(1, len(trainq[name]["atr_sorted"])),
        })
        g = dd.groupby("date")
        agg = pd.DataFrame({
            f"{name}_atr_q_med": g["atr_q"].median(),
            f"{name}_move_pct": (g["c"].last() - g["o"].first()) / g["o"].first() * 100,
            f"{name}_range_pct": (g["h"].max() - g["l"].min()) / g["o"].first() * 100,
        })
        daily_rows.append(agg)
    daily = pd.concat(daily_rows, axis=1)
    daily["mean_pair_corr"] = corr_by_day
    qcols = [f"{n}_atr_q_med" for n in SYMBOLS]
    daily["min_atr_q"] = daily[qcols].min(axis=1)
    daily["max_atr_q"] = daily[qcols].max(axis=1)
    daily["max_abs_move"] = daily[[f"{n}_move_pct" for n in SYMBOLS]].abs().max(axis=1)

    # strategy day PnL (FLAT full period; v2B test period), stop counts, entries
    flat_day, v2b_day, stop_day, ent_day = {}, {}, {}, {}
    for tr in all_trades:
        d = tr["exit_dt"].date()
        flat_day[d] = flat_day.get(d, 0.0) + trade_net(tr, FLAT_NOTIONAL)[0]
        if tr["exit_reason"] == "stop":
            stop_day[d] = stop_day.get(d, 0) + 1
        ent_day[tr["entry_dt"].date()] = ent_day.get(tr["entry_dt"].date(), 0) + 1
    for _, net, *_r, tr in [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows_all]:
        d = tr["exit_dt"].date()
        v2b_day[d] = v2b_day.get(d, 0.0) + net
    daily["flat_pnl"] = pd.Series(flat_day).reindex(daily.index).fillna(0)
    daily["v2b_pnl"] = pd.Series(v2b_day).reindex(daily.index)   # NaN outside test
    daily["stop_exits"] = pd.Series(stop_day).reindex(daily.index).fillna(0).astype(int)
    daily["entries"] = pd.Series(ent_day).reindex(daily.index).fillna(0).astype(int)
    daily.to_csv(OUT_DIR / "daily_features.csv")

    ev = daily.loc[EVENT_DAY.date()]
    prior30 = daily.loc[[d for d in daily.index
                         if EVENT_DAY.date() - timedelta(days=30) <= d < EVENT_DAY.date()]]
    snap = {
        "event_day": str(EVENT_DAY.date()),
        "event_features": {k: (None if pd.isna(v) else round(float(v), 4))
                           for k, v in ev.items()},
        "corr": {
            "event_day_mean_pair_corr": round(float(ev["mean_pair_corr"]), 4),
            "prior30_mean": round(float(prior30["mean_pair_corr"].mean()), 4),
            "prior30_std": round(float(prior30["mean_pair_corr"].std()), 4),
            "percentile_all_days": round(float((daily["mean_pair_corr"] <
                                                ev["mean_pair_corr"]).mean()), 4),
        },
        "atr_structure": {
            "event_atr_q_by_sym": {n: round(float(ev[f"{n}_atr_q_med"]), 4) for n in SYMBOLS},
            "event_min_atr_q": round(float(ev["min_atr_q"]), 4),
            "pct_days_min_atr_q_geq_event": round(float(
                (daily["min_atr_q"] >= ev["min_atr_q"]).mean()), 4),
        },
        "moves": {n: {"move_pct": round(float(ev[f"{n}_move_pct"]), 3),
                      "range_pct": round(float(ev[f"{n}_range_pct"]), 3)} for n in SYMBOLS},
        "day_ohlc_5m": day_ohlc,
    }
    (OUT_DIR / "market_snapshot_20250529.json").write_text(json.dumps(snap, indent=2))
    L(f"corr: event {snap['corr']['event_day_mean_pair_corr']} vs prior30 "
      f"{snap['corr']['prior30_mean']}±{snap['corr']['prior30_std']} "
      f"(all-days pct {snap['corr']['percentile_all_days']})")
    L(f"atr_q by sym: {snap['atr_structure']['event_atr_q_by_sym']}")
    L(f"moves: { {n: snap['moves'][n]['move_pct'] for n in SYMBOLS} }")

    # ══ STEP 3: strategy behaviour ════════════════════════════════════════════
    L("\n## STEP 3 — strategy behaviour")
    rev = {"before_exit": 0, "after_exit_within_24h": 0, "never_24h": 0}
    for d in detail:
        if d["bars_to_midline_revert"] is None:
            rev["never_24h"] += 1
        elif d["bars_to_midline_revert"] <= d["hold_bars"]:
            rev["before_exit"] += 1
        else:
            rev["after_exit_within_24h"] += 1
    repeats = {}
    for d in detail:
        repeats.setdefault(d["symbol"], []).append(
            (d["entry_dt"], d["side"], d["exit_reason"], d["net_pnl"]))
    step3 = {
        "reversion": rev,
        "repeat_entries": {k: v for k, v in repeats.items() if len(v) > 1},
        "tier_amplification": {
            "v2b_window_net": round(df["net_pnl"].sum(), 2),
            "flat_window_net": round(df["flat_net_pnl"].sum(), 2),
            "ratio": round(df["net_pnl"].sum() / df["flat_net_pnl"].sum(), 3),
            "tier_counts": df["v2b_tier"].value_counts().to_dict(),
        },
        "breakout_excess_atr": df["breakout_excess_atr"].describe().round(3).to_dict(),
        "channel_width_atr": df["channel_width_atr"].describe().round(3).to_dict(),
    }
    L(json.dumps(step3, indent=2, default=str))
    (OUT_DIR / "step3_behaviour.json").write_text(json.dumps(step3, indent=2, default=str))

    # ══ STEP 4: replication scan (synthetic-ramp anomaly detector, 1m) ════════
    L("\n## STEP 4 — synthetic-ramp anomaly scan (1m, full period)")
    cache1m = OUT_DIR / "_c1m_cache.npz"
    if cache1m.exists():
        z = np.load(cache1m, allow_pickle=False)
        c1m = {n: (z[f"{n}_dt"], z[f"{n}_c"]) for n in SYMBOLS}
    else:
        hr = parse_history_range(START, END, timedelta(minutes=1), TZ)
        db = PROJECT_ROOT / ".vntrader" / "database.db"
        c1m, payload = {}, {}
        for n in SYMBOLS:
            b1 = load_1m(SYMBOLS[n][0], hr, db)
            dt_ns = pd.DatetimeIndex(b1["datetime"]).tz_convert("UTC").as_unit("ns").asi8
            cc = b1["close"].to_numpy(dtype=float)
            c1m[n] = (dt_ns, cc)
            payload[f"{n}_dt"] = dt_ns
            payload[f"{n}_c"] = cc
        np.savez_compressed(cache1m, **payload)

    trades_by_sym = {}
    for tr in all_trades:
        trades_by_sym.setdefault(tr["symbol"], []).append(tr)
    split_dt = {n: ctx_by_sym[n]["dt"][split_by_sym[n]] for n in SYMBOLS}

    all_events = []
    for n in SYMBOLS:
        dt_ns, cc = c1m[n]
        for ev in detect_ramp_events(dt_ns, cc):
            t0e = pd.Timestamp(dt_ns[ev["i0"]], unit="ns", tz="UTC")
            t1e = pd.Timestamp(dt_ns[ev["i1"]], unit="ns", tz="UTC")
            lo = t0e - pd.Timedelta(minutes=5)
            hi = t1e + pd.Timedelta(minutes=30)
            flat_pnl = v2b_pnl = 0.0
            ntr = 0
            for tr in trades_by_sym.get(n, []):
                if tr["entry_dt"] <= hi and tr["exit_dt"] >= lo:
                    ntr += 1
                    flat_pnl += trade_net(tr, FLAT_NOTIONAL)[0]
                    if tr["entry_idx"] >= split_by_sym[n]:
                        notion = make_rules(th_by_sym[n])["v2B"](tr)
                        v2b_pnl += trade_net(tr, notion)[0]
            all_events.append({
                "symbol": n, "start": str(t0e), "end": str(t1e),
                "in_test": bool(t0e >= split_dt[n]),
                "max_excursion_pct": ev["max_excursion_pct"],
                "reverted_within_4h": ev["reverted"],
                "staircase_signature": ev["staircase"],
                "n_overlap_trades": ntr,
                "flat_pnl_overlap": round(flat_pnl, 2),
                "v2b_pnl_overlap": round(v2b_pnl, 2),
            })
    all_events.sort(key=lambda e: e["start"])
    n_rev = sum(1 for e in all_events if e["reverted_within_4h"])
    n_stair = sum(1 for e in all_events if e["staircase_signature"])
    summary4 = {
        "detector": {"ramp_step": RAMP_STEP, "ramp_min_len": RAMP_MIN_LEN,
                     "ramp_min_move": RAMP_MIN_MOVE, "jump_1bar": JUMP_1BAR,
                     "merge_gap_min": MERGE_GAP_MIN, "revert_tol": REVERT_TOL,
                     "revert_win_min": REVERT_WIN_MIN, "stair_cv": STAIR_CV},
        "n_events": len(all_events), "n_reverted": n_rev, "n_staircase": n_stair,
        "total_flat_pnl_overlap": round(sum(e["flat_pnl_overlap"] for e in all_events), 2),
        "total_v2b_pnl_overlap_test": round(sum(e["v2b_pnl_overlap"] for e in all_events), 2),
        "events": all_events,
    }
    (OUT_DIR / "pattern_replication_check.json").write_text(
        json.dumps(summary4, indent=2))
    L(f"events: {len(all_events)} total | reverted<=4h: {n_rev} | staircase-uniform: {n_stair}")
    L(f"{'sym':5} {'start':17} {'exc%':>6} {'rev':>4} {'stair':>5} {'test':>4} "
      f"{'nTr':>4} {'flat$':>9} {'v2b$':>9}")
    for e in all_events:
        L(f"{e['symbol']:5} {e['start'][:16]:17} {e['max_excursion_pct']:>6.1f} "
          f"{str(e['reverted_within_4h'])[:1]:>4} {str(e['staircase_signature'])[:1]:>5} "
          f"{str(e['in_test'])[:1]:>4} {e['n_overlap_trades']:>4} "
          f"{e['flat_pnl_overlap']:>9.2f} {e['v2b_pnl_overlap']:>9.2f}")
    (OUT_DIR / "run_log.txt").write_text("\n".join(log))
    t1 = datetime.now(timezone.utc)
    L(f"\n# end {t1.isoformat()} ({(t1 - t0).total_seconds():.1f}s)")
    (OUT_DIR / "run_log.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()
