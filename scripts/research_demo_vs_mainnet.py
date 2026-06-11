#!/usr/bin/env python3
"""DEMO vs mainnet bar-by-bar comparison + credibility map.

Read-only over both DBs (sqlite URI mode=ro — a write attempt would raise).
Reuses the compare engine's conventions verbatim:
  - norm()/r5() from research_mr_5m for normalization and 5m aggregation
  - wilder_atr / Donchian(24).shift(1) / ATR threshold filter / exit rules
    from backtest_mr_5m_compare (imported, not modified)
No full strategy backtest is run: signal sets are raw position-independent
triggers; exit paths are simulated per sampled matched signal only.
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_mr_5m import norm, r5
from history_time_utils import parse_history_range
from backtest_mr_5m_compare import (
    ATR_STOP,
    ATR_THRESHOLDS,
    CONTRACT_SPECS,
    FEE_MAKER,
    FEE_TAKER,
    LOOKBACK,
    MAX_HOLD,
    SYMBOLS,
    calc_size,
    wilder_atr,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DEMO = PROJECT_ROOT / ".vntrader" / "database.db"
DB_MAIN = PROJECT_ROOT / ".vntrader" / "database_mainnet.db"
OUT = PROJECT_ROOT / "reports/regime/demo_vs_mainnet_comparison_20260610"
DIAG_EVENTS = PROJECT_ROOT / "reports/regime/v2b_dd_diagnosis_20260610/pattern_replication_check.json"

START, END = "2023-01-01", "2026-05-28"
TIMEZONE = "UTC"  # engine convention (DB naive timestamps treated uniformly)
TRAIN_END = pd.Timestamp("2025-04-01", tz="UTC")  # train/test split for pctl drift
DEV_THRESHOLDS = [0.0005, 0.001, 0.005, 0.01, 0.05]
EXIT_SAMPLE_N = 500
RNG_SEED = 20260610


def load_1m_ro(vt_symbol: str, hr, db_path: Path) -> pd.DataFrame:
    """Same query/normalization as research_mr_5m.load_1m, but mode=ro."""
    sym, _, exch = str(vt_symbol).partition(".")
    qs = hr.start.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    qe = hr.end_exclusive.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "select datetime,open_price as open,high_price as high,low_price as low,"
            "close_price as close,volume from dbbardata where symbol=? and exchange=? "
            "and interval='1m' and datetime>=? and datetime<? order by datetime",
            conn, params=(sym, exch, qs, qe),
        )
    finally:
        conn.close()
    return norm(df, hr.timezone_name)


def dist_stats(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "p99_9": float(np.percentile(x, 99.9)),
        "max": float(np.max(x)),
    }


def exceedance(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    n = max(len(x), 1)
    return {f">{t*100:g}%": float(np.sum(x > t) / n) for t in DEV_THRESHOLDS}


def indicators_5m(bars: pd.DataFrame):
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    atr = wilder_atr(h, l, c)
    dh = bars["high"].rolling(LOOKBACK).max().shift(1).to_numpy()
    dl = bars["low"].rolling(LOOKBACK).min().shift(1).to_numpy()
    return atr, dh, dl


def raw_triggers(c, atr, dh, dl, threshold):
    """Position-independent entry triggers per engine entry conditions."""
    valid = (~np.isnan(atr)) & (atr > 0) & (atr >= threshold) \
        & (~np.isnan(dh)) & (~np.isnan(dl)) & (dh > 0) & (dl > 0)
    short_trig = valid & (c > dh)   # fade long breakout
    long_trig = valid & (c < dl)    # fade short breakout
    return short_trig, long_trig


def walk_exit(i, direction, c, h, l, atr, dh, dl, tick, inst_id):
    """Replicate engine exit loop for one isolated trade entered at bar i."""
    ep = c[i]
    esize = calc_size(inst_id, ep)
    ct_val = CONTRACT_SPECS[inst_id]["ctVal"]
    n = len(c)
    for j in range(i + 1, n):
        if np.isnan(atr[j]) or atr[j] <= 0:
            continue  # engine skips the whole iteration
        reason = ""
        if dh[j] > 0 and dl[j] > 0:
            mid = (dh[j] + dl[j]) / 2.0
            if (direction == 1 and c[j] >= mid) or (direction == -1 and c[j] <= mid):
                reason = "midline"
        if not reason:
            stop_dist = ATR_STOP * atr[j]
            if direction == 1 and l[j] <= ep - stop_dist:
                reason = "stop"
            elif direction == -1 and h[j] >= ep + stop_dist:
                reason = "stop"
        if not reason and (j - i) >= MAX_HOLD:
            reason = "max_hold"
        if reason:
            exit_px = c[j] - tick if direction == 1 else c[j] + tick
            gross = (exit_px - ep) * esize * ct_val if direction == 1 \
                else (ep - exit_px) * esize * ct_val
            fee = -FEE_MAKER * ep * esize * ct_val - FEE_TAKER * exit_px * esize * ct_val
            return {"reason": reason, "hold_bars": j - i, "net": gross + fee}
    return {"reason": "eod_open", "hold_bars": n - 1 - i, "net": np.nan}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bar_level_stats").mkdir(exist_ok=True)
    (OUT / "signal_overlap").mkdir(exist_ok=True)
    hr = parse_history_range(start_arg=START, end_arg=END,
                             interval_delta=timedelta(minutes=1), timezone_name=TIMEZONE)
    ramp_events = json.load(DIAG_EVENTS.open())["events"]
    ramp_month = {}
    for ev in ramp_events:
        key = (ev["symbol"], str(ev["start"])[:7])
        ramp_month[key] = ramp_month.get(key, 0) + 1

    months = pd.period_range("2023-01", "2026-05", freq="M").strftime("%Y-%m").tolist()
    cred_map = {}
    exit_summary = {}
    rng = random.Random(RNG_SEED)

    for name, (vt_symbol, inst_id) in SYMBOLS.items():
        print(f"=== {name} ===", flush=True)
        d1 = load_1m_ro(vt_symbol, hr, DB_DEMO)
        m1 = load_1m_ro(vt_symbol, hr, DB_MAIN)
        j = d1.merge(m1, on="datetime", suffixes=("_d", "_m"), how="outer", indicator=True)
        both = j[j["_merge"] == "both"].sort_values("datetime").reset_index(drop=True)
        align = {
            "demo_rows": len(d1), "main_rows": len(m1), "matched": len(both),
            "demo_only": int((j["_merge"] == "left_only").sum()),
            "main_only": int((j["_merge"] == "right_only").sum()),
        }

        cm = both["close_m"].to_numpy(dtype=float)
        cd = both["close_d"].to_numpy(dtype=float)
        close_dev = np.abs(cd - cm) / cm
        high_dev = np.abs(both["high_d"].to_numpy(dtype=float) - both["high_m"].to_numpy(dtype=float)) / cm
        low_dev = np.abs(both["low_d"].to_numpy(dtype=float) - both["low_m"].to_numpy(dtype=float)) / cm
        rd = np.diff(cd) / cd[:-1]
        rm = np.diff(cm) / cm[:-1]
        ret_diff = rd - rm
        vol_m = both["volume_m"].to_numpy(dtype=float)
        vol_d = both["volume_d"].to_numpy(dtype=float)
        vol_ratio = np.where(vol_m > 0, vol_d / np.where(vol_m > 0, vol_m, 1.0), np.nan)

        month_key = both["datetime"].dt.strftime("%Y-%m")
        monthly = []
        for mo, grp_idx in pd.Series(range(len(both))).groupby(month_key.values):
            idx = grp_idx.to_numpy()
            mdev = close_dev[idx]
            monthly.append({
                "month": mo, "bars": int(len(idx)),
                "close_dev_mean": float(np.nanmean(mdev)),
                "close_dev_p99": float(np.nanpercentile(mdev, 99)),
                "close_dev_max": float(np.nanmax(mdev)),
                "frac_gt_0.1%": float(np.mean(mdev > 0.001)),
            })

        bar_stats = {
            "symbol": name, "alignment": align,
            "close_dev": {**dist_stats(close_dev), "exceedance": exceedance(close_dev)},
            "high_dev": {**dist_stats(high_dev), "exceedance": exceedance(high_dev)},
            "low_dev": {**dist_stats(low_dev), "exceedance": exceedance(low_dev)},
            "ret_1m_diff_abs": dist_stats(np.abs(ret_diff)),
            "volume_ratio": {
                **dist_stats(vol_ratio),
                "frac_in_[0.5,2]": float(np.nanmean((vol_ratio >= 0.5) & (vol_ratio <= 2))),
            },
            "monthly": monthly,
        }
        (OUT / "bar_level_stats" / f"{name}.json").write_text(
            json.dumps(bar_stats, indent=2, ensure_ascii=False))

        # ── layer 2: 5m indicators & signals ──
        d5 = r5(d1, 5, hr)
        m5 = r5(m1, 5, hr)
        atr_d, dh_d, dl_d = indicators_5m(d5)
        atr_m, dh_m, dl_m = indicators_5m(m5)
        d5k = d5.assign(atr=atr_d, dh=dh_d, dl=dl_d)
        m5k = m5.assign(atr=atr_m, dh=dh_m, dl=dl_m)
        j5 = d5k.merge(m5k, on="datetime", suffixes=("_d", "_m"), how="inner")
        c5m = j5["close_m"].to_numpy(dtype=float)
        ind_dev = {
            "n_5m_matched": len(j5),
            "atr_rel_dev": dist_stats(np.abs(j5["atr_d"].to_numpy() - j5["atr_m"].to_numpy())
                                      / j5["atr_m"].to_numpy()),
            "donchian_high_dev": dist_stats(np.abs(j5["dh_d"].to_numpy() - j5["dh_m"].to_numpy()) / c5m),
            "donchian_low_dev": dist_stats(np.abs(j5["dl_d"].to_numpy() - j5["dl_m"].to_numpy()) / c5m),
            "midline_dev": dist_stats(np.abs((j5["dh_d"] + j5["dl_d"]).to_numpy() / 2
                                             - (j5["dh_m"] + j5["dl_m"]).to_numpy() / 2) / c5m),
        }
        # ATR percentile drift, each normalized by its own train distribution
        tr_mask_d = d5k["datetime"] < TRAIN_END
        tr_mask_m = m5k["datetime"] < TRAIN_END
        ref_d = np.sort(d5k.loc[tr_mask_d, "atr"].dropna().to_numpy())
        ref_m = np.sort(m5k.loc[tr_mask_m, "atr"].dropna().to_numpy())
        pct_d = np.searchsorted(ref_d, j5["atr_d"].to_numpy()) / max(len(ref_d), 1)
        pct_m = np.searchsorted(ref_m, j5["atr_m"].to_numpy()) / max(len(ref_m), 1)
        ind_dev["atr_pctl_drift_abs"] = dist_stats(np.abs(pct_d - pct_m))

        # raw triggers on each dataset independently (full own series)
        thr = ATR_THRESHOLDS[inst_id]
        sd, ld = raw_triggers(d5k["close"].to_numpy(dtype=float), atr_d, dh_d, dl_d, thr)
        sm, lm = raw_triggers(m5k["close"].to_numpy(dtype=float), atr_m, dh_m, dl_m, thr)
        sig_d = {(t, "S") for t in d5k.loc[sd, "datetime"]} | {(t, "L") for t in d5k.loc[ld, "datetime"]}
        sig_m = {(t, "S") for t in m5k.loc[sm, "datetime"]} | {(t, "L") for t in m5k.loc[lm, "datetime"]}
        matched = sig_d & sig_m
        ghosts = sig_d - sig_m       # demo-only
        missed = sig_m - sig_d       # mainnet-only

        def by_month(sigs):
            out = {}
            for t, _dir in sigs:
                k = t.strftime("%Y-%m")
                out[k] = out.get(k, 0) + 1
            return out

        mm, gm, xm = by_month(matched), by_month(ghosts), by_month(missed)
        overlap = {
            "symbol": name,
            "n_demo": len(sig_d), "n_main": len(sig_m), "n_matched": len(matched),
            "recall_of_mainnet": len(matched) / max(len(sig_m), 1),
            "precision_of_demo": len(matched) / max(len(sig_d), 1),
            "jaccard": len(matched) / max(len(sig_d | sig_m), 1),
            "ghost_signals_demo_only": len(ghosts),
            "missed_signals_main_only": len(missed),
            "monthly": [
                {"month": mo, "matched": mm.get(mo, 0), "demo_only": gm.get(mo, 0),
                 "main_only": xm.get(mo, 0)} for mo in months
            ],
        }
        (OUT / "signal_overlap" / f"{name}.json").write_text(
            json.dumps(overlap, indent=2, ensure_ascii=False))

        # ── layer 2c: sampled exit-path comparison on matched signals ──
        idx_d = {t: i for i, t in enumerate(d5k["datetime"])}
        idx_m = {t: i for i, t in enumerate(m5k["datetime"])}
        tick = CONTRACT_SPECS[inst_id]["tickSz"]
        cdn = d5k["close"].to_numpy(dtype=float); hdn = d5k["high"].to_numpy(dtype=float)
        ldn = d5k["low"].to_numpy(dtype=float)
        cmn = m5k["close"].to_numpy(dtype=float); hmn = m5k["high"].to_numpy(dtype=float)
        lmn = m5k["low"].to_numpy(dtype=float)
        sample = rng.sample(sorted(matched, key=str), min(EXIT_SAMPLE_N, len(matched)))
        recs = []
        for t, direc in sample:
            di, mi = idx_d.get(t), idx_m.get(t)
            if di is None or mi is None:
                continue
            dirn = 1 if direc == "L" else -1
            ed = walk_exit(di, dirn, cdn, hdn, ldn, atr_d, dh_d, dl_d, tick, inst_id)
            em = walk_exit(mi, dirn, cmn, hmn, lmn, atr_m, dh_m, dl_m, tick, inst_id)
            recs.append({"time": t.isoformat(), "dir": direc,
                         "demo": ed, "main": em,
                         "reason_match": ed["reason"] == em["reason"],
                         "pnl_diff": (ed["net"] - em["net"])
                         if not (np.isnan(ed["net"]) or np.isnan(em["net"])) else None})
        pnl_diffs = np.array([r["pnl_diff"] for r in recs if r["pnl_diff"] is not None])
        exit_summary[name] = {
            "sampled": len(recs),
            "exit_reason_agreement": float(np.mean([r["reason_match"] for r in recs])) if recs else None,
            "pnl_diff_usd": dist_stats(np.abs(pnl_diffs)),
            "pnl_diff_signed_mean": float(np.mean(pnl_diffs)) if len(pnl_diffs) else None,
            "samples": recs,
        }

        # ── layer 3: credibility cells ──
        p99_by_month = {row["month"]: row["close_dev_p99"] for row in monthly}
        for mo in months:
            mt, go, mn = mm.get(mo, 0), gm.get(mo, 0), xm.get(mo, 0)
            union = mt + go + mn
            ov = mt / union if union else None
            p99 = p99_by_month.get(mo)
            if p99 is None:
                rating = "no_data"
            elif p99 < 0.001 and (ov is None or ov > 0.95):
                rating = "green"
            elif p99 < 0.01 and (ov is None or ov > 0.80):
                rating = "yellow"
            else:
                rating = "red"
            cred_map[f"{name}|{mo}"] = {
                "symbol": name, "month": mo, "close_dev_p99": p99,
                "signal_overlap": ov, "n_signals_union": union,
                "ramp_events": ramp_month.get((name, mo), 0), "rating": rating,
            }
        print(f"{name}: matched_1m={align['matched']} close_p99={bar_stats['close_dev']['p99']:.2e} "
              f"signals d/m/match={len(sig_d)}/{len(sig_m)}/{len(matched)} "
              f"jaccard={overlap['jaccard']:.3f}", flush=True)

        # layer-2 indicator stats live alongside bar stats
        (OUT / "bar_level_stats" / f"{name}_indicators_5m.json").write_text(
            json.dumps(ind_dev, indent=2, ensure_ascii=False))

    (OUT / "exit_path_sample.json").write_text(
        json.dumps(exit_summary, indent=2, ensure_ascii=False))
    (OUT / "credibility_map.json").write_text(
        json.dumps(cred_map, indent=2, ensure_ascii=False))

    # markdown map
    lines = ["# 数据可信度地图（DEMO vs mainnet）", "",
             "G=green Y=yellow R=red ·=no_data；评级=初始建议阈值（待用户确认）", ""]
    header = "| 币种 | " + " | ".join(m[2:] for m in months) + " |"
    lines += [header, "|" + "---|" * (len(months) + 1)]
    letter = {"green": "G", "yellow": "Y", "red": "R", "no_data": "·"}
    for name in SYMBOLS:
        row = [letter[cred_map[f"{name}|{mo}"]["rating"]] for mo in months]
        lines.append("| " + name + " | " + " | ".join(row) + " |")
    lines.append("")
    for name in SYMBOLS:
        cnt = {"green": 0, "yellow": 0, "red": 0, "no_data": 0}
        for mo in months:
            cnt[cred_map[f"{name}|{mo}"]["rating"]] += 1
        lines.append(f"- {name}: green {cnt['green']}/41, yellow {cnt['yellow']}/41, "
                     f"red {cnt['red']}/41" + (f", no_data {cnt['no_data']}" if cnt["no_data"] else ""))
    (OUT / "credibility_map.md").write_text("\n".join(lines))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
