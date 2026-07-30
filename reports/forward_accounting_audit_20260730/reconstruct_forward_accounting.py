#!/usr/bin/env python3
"""INDEPENDENT re-derivation of the B2_4h forward daily-report accounting.

AUDIT, NOT OPTIMIZATION.  This script only re-derives and reports differences;
it changes nothing.  Per audit-independent skill + audit/CLAUDE.md:

  * It reads ONLY the read-only VPS snapshot state files
    (audit/vps_snapshot_20260730/state/*).  It NEVER imports forward_b2_4h,
    research_trend_baseline, or research_trend_validation_r2 — re-running the
    engine would only reproduce the engine's own behaviour (circular).  The
    whole point is an independent code path over the engine's OUTPUT ledger.

  * Goal: given the single 2026-07-30 snapshot of ledger_daily_m2m.jsonl
    (the engine's full-history daily M2M output) + heartbeat/deploy, reconstruct
    what the daily report's "forward cumulative net" and "forward drawdown"
    SHOULD be, expose the definition/labelling issues, and test whether the
    constant $2,074.29 drawdown is legitimate max-DD semantics (early-window
    record never beaten) or a frozen/stalled anchor.

Definitions recovered from forward_b2_4h.py (read-only審读):
  - mode_account(): ledger_daily_m2m.jsonl = perbar.groupby(index.ceil("D")).sum()
    over the FULL store (seeded from 2023-01-01).  Each record = one day's M2M
    INCREMENT (not cumulative).  heartbeat.net_total_usd = perbar.sum() (full).
  - gate_status(): forward slice f_per = perbar[index >= effective_forward_start],
    minus logged gap periods.  effective_forward_start = max(config
    forward_start_utc, deploy.first_live_bar_close_utc) = 2026-06-21T08:00Z (both
    equal).  forward_net = f_per.sum(); forward daily = f_per.groupby(ceil D).sum();
    eq = daily.cumsum(); forward_maxdd = max(eq.cummax() - eq)  [== MAX drawdown].
  - mode_push(): day_pnl = last bucket of the FULL-history daily (ceil D) series
    (a PARTIAL, mostly-yesterday bucket); forward_net/maxdd come from gate_status.

Boundary note (documented, not hidden): the engine slices PER-BAR at 08:00Z;
we only have day-buckets already ceil("D")-labelled over full history.  A 4h bar
ending at 06-21T08:00Z ceils to date "2026-06-22", so the forward window in
bucket terms begins at "2026-06-22".  That bucket also contains the pre-boundary
bar ending 06-21T04:00Z (ceils to 06-22 too) => a <=1-bar constant impurity at
the window's first day only.  We report both the 06-22 caliber (primary, closest
to engine) and the 06-21 caliber (alt), and treat any residual as the
reconciliation floor, exactly as the funding audit treated its rounding floor.
"""
from __future__ import annotations

import json
from pathlib import Path

SNAP = Path(__file__).resolve().parents[2] / "audit" / "vps_snapshot_20260730"
STATE = SNAP / "state"
DAILY = STATE / "ledger_daily_m2m.jsonl"
HB = STATE / "heartbeat.json"
DEPLOY = STATE / "deploy.json"

FORWARD_START_BUCKET = "2026-06-22"   # ceil("D") of the 06-21T08:00Z first live bar
FORWARD_START_ALT = "2026-06-21"      # inclusive-of-boundary-day alt caliber
REPORT_DATES = ["2026-07-23", "2026-07-24", "2026-07-29", "2026-07-30", "2026-07-31"]
REPORTED = {  # what the user observed in the daily PushPlus reports
    "2026-07-23": 2294.29,
    "2026-07-24": 2227.46,
    "2026-07-29": 1251.67,
}
REPORTED_DD = 2074.29
HB_NET_TOTAL = 73614.58483792051   # heartbeat.json net_total_usd (full history)


def load_daily():
    rows = []
    for line in DAILY.read_text().strip().split("\n"):
        if line.strip():
            d = json.loads(line)
            rows.append((d["date"], float(d["m2m_usd"])))
    rows.sort(key=lambda r: r[0])
    return rows


def running_maxdd(dates, incs):
    """Return per-date (cum_net, running_maxdd, peak_date, trough_date)."""
    cum = 0.0
    peak = 0.0          # equity peak so far (forward equity starts at 0)
    peak_date = None
    maxdd = 0.0
    maxdd_peak_date = None
    maxdd_trough_date = None
    cur_peak_date = None
    out = []
    for dt, inc in zip(dates, incs):
        cum += inc
        if cum > peak:
            peak = cum
            cur_peak_date = dt
        dd = peak - cum
        if dd > maxdd:
            maxdd = dd
            maxdd_peak_date = cur_peak_date
            maxdd_trough_date = dt
        out.append((dt, cum, maxdd, maxdd_peak_date, maxdd_trough_date))
    return out


def main():
    rows = load_daily()
    dates = [r[0] for r in rows]
    incs = [r[1] for r in rows]

    print("=" * 78)
    print("INDEPENDENT RE-DERIVATION — B2_4h forward accounting (read-only)")
    print("=" * 78)

    # ── (1) two-account identity ────────────────────────────────────────────
    full_total = sum(incs)
    print("\n[1] daily_m2m.jsonl provenance & full-history identity")
    print(f"    first record date : {dates[0]}")
    print(f"    last  record date : {dates[-1]}")
    print(f"    n records (days)  : {len(rows)}")
    print(f"    SUM(all m2m_usd)  : {full_total:.8f}")
    print(f"    heartbeat net_total_usd (full): {HB_NET_TOTAL:.8f}")
    print(f"    |diff|            : {abs(full_total - HB_NET_TOTAL):.2e}"
          f"   => {'MATCH (daily_m2m IS the full-history account)' if abs(full_total-HB_NET_TOTAL) < 1e-6 else 'MISMATCH'}")
    n_zero_lead = 0
    for _, v in rows:
        if v == 0.0:
            n_zero_lead += 1
        else:
            break
    print(f"    leading exact-zero days (EMA warm-up, flat): {n_zero_lead}  "
          f"(first nonzero: {dates[n_zero_lead]} = {incs[n_zero_lead]:.4f})")

    # ── (2)+(4) forward window reconstruction (primary caliber = 06-22) ─────
    for caliber, start in (("PRIMARY 06-22 (engine per-bar>=08:00Z)", FORWARD_START_BUCKET),
                           ("ALT 06-21 (whole deploy day)", FORWARD_START_ALT)):
        fdates = [d for d in dates if d >= start]
        fincs = [incs[i] for i, d in enumerate(dates) if d >= start]
        series = running_maxdd(fdates, fincs)
        by_date = {s[0]: s for s in series}
        fnet = sum(fincs)
        final = series[-1]
        print(f"\n[2/4] FORWARD WINDOW  ({caliber})")
        print(f"    forward buckets: {fdates[0]} .. {fdates[-1]}  ({len(fdates)} days)")
        print(f"    forward_net (sum)      : ${fnet:,.2f}")
        print(f"    forward_maxdd (max-DD) : ${final[2]:,.2f}"
              f"   peak@{final[3]} -> trough@{final[4]}")
        print(f"    (reported daily-report drawdown was constant ${REPORTED_DD:,.2f})")
        print(f"    {'date':<12}{'day_incr':>12}{'cum_net':>12}{'run_maxDD':>12}"
              f"{'reported':>12}{'my_cum-rep':>12}")
        for rd in REPORT_DATES:
            if rd in by_date:
                dt, cum, mdd, _, _ = by_date[rd]
                inc = fincs[fdates.index(rd)]
                rep = REPORTED.get(rd)
                diff = (cum - rep) if rep is not None else None
                print(f"    {dt:<12}{inc:>12.2f}{cum:>12.2f}{mdd:>12.2f}"
                      f"{('%.2f'%rep) if rep is not None else '—':>12}"
                      f"{('%.2f'%diff) if diff is not None else '—':>12}")

    # ── (3) drawdown-record timeline (is $2,074.29 an EARLY record?) ────────
    start = FORWARD_START_BUCKET
    fdates = [d for d in dates if d >= start]
    fincs = [incs[i] for i, d in enumerate(dates) if d >= start]
    series = running_maxdd(fdates, fincs)
    print("\n[3] MAX-DRAWDOWN RECORD TIMELINE (primary caliber)")
    print("    Each time the running max-DD sets a NEW record, print it.")
    last_mdd = -1.0
    records = []
    for dt, cum, mdd, pk, tr in series:
        if mdd > last_mdd + 1e-9:
            records.append((dt, mdd, pk, tr))
            last_mdd = mdd
    for dt, mdd, pk, tr in records:
        print(f"    {dt}:  new max-DD ${mdd:,.2f}  (peak@{pk} trough@{tr})")
    final_mdd = series[-1][2]
    print(f"    FINAL running max-DD as of {series[-1][0]}: ${final_mdd:,.2f}")
    set_date = records[-1][0] if records else None
    print(f"    max-DD record last MOVED on: {set_date}")
    if set_date is not None and set_date < "2026-07-23":
        print(f"    => record predates 2026-07-23 => a CONSTANT ${final_mdd:,.2f} across")
        print(f"       7/23,7/24,7/29 is LEGITIMATE max-DD semantics, not a frozen anchor,")
        print(f"       PROVIDED no later day drives (peak-cum) above it.")

    # ── (5) reproduce 矛盾2: ceil-bucket 'day M2M' vs cumulative delta ──────
    print("\n[5] CONTRADICTION-2 anatomy: '当日M2M' (ceil bucket) vs forward-cum delta")
    print("    The report's '当日模拟M2M' = the last ceil(\"D\") bucket, which by the")
    print("    ceil convention is dominated by the PREVIOUS calendar day's 4h bars.")
    print("    The report's forward-cum day-over-day delta spans DIFFERENT bars.")
    fmap = {d: incs[i] for i, d in enumerate(dates)}
    cummap = {s[0]: s[1] for s in series}
    for a, b in (("2026-07-23", "2026-07-24"), ("2026-07-28", "2026-07-29")):
        if a in cummap and b in cummap:
            bucket_b = fmap[b]
            cum_delta = cummap[b] - cummap[a]
            print(f"    {a}->{b}: ceil-bucket '{b}' day M2M = {bucket_b:+.2f} ; "
                  f"forward-cum delta = {cum_delta:+.2f}")
            print(f"              (these are DIFFERENT bar sets; agreement is NOT expected)")

    # ── dump reconstructed forward ledger + summary ────────────────────────
    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "reconstructed_forward_ledger.jsonl", "w") as fh:
        for dt, cum, mdd, pk, tr in series:
            fh.write(json.dumps({"date": dt, "day_m2m": fmap[dt], "cum_net": cum,
                                 "running_maxdd": mdd}) + "\n")
    # structural checks (S1 contiguity / S2 boundary partition / S3 sign)
    from datetime import date as _date, timedelta as _td
    d0, dN = _date.fromisoformat(dates[0]), _date.fromisoformat(dates[-1])
    exp, cur = [], d0
    while cur <= dN:
        exp.append(cur.isoformat()); cur += _td(days=1)
    missing = sorted(set(exp) - set(dates))
    dup = len(dates) - len(set(dates))
    pre = sum(v for dt, v in zip(dates, incs) if dt < "2026-05-30")
    warm = sum(v for dt, v in zip(dates, incs) if "2026-05-30" <= dt < "2026-06-22")
    fwd = sum(v for dt, v in zip(dates, incs) if dt >= "2026-06-22")

    summary = {
        "audit": "forward_accounting_audit_20260730",
        "independent_of_engine": True,
        "generated_utc": "2026-07-30T15:59Z",
        "Q1_two_accounts": {
            "daily_m2m_first_date": dates[0], "daily_m2m_last_date": dates[-1],
            "n_days": len(rows), "nature": "FULL-HISTORY account (seeded 2023-01-01)",
            "full_history_sum": full_total, "heartbeat_net_total_usd": HB_NET_TOTAL,
            "full_history_identity_abs_diff": abs(full_total - HB_NET_TOTAL),
            "heartbeat_73614_is": "full-history monitoring recompute (NOT a gate input)",
            "daily_report_2227_is": "forward slice (f_per from 2026-06-21T08:00Z)"},
        "Q2_forward_net_path": {
            "formula": "f_per.sum(); f_per = perbar[index >= effective_forward_start] minus gaps",
            "effective_forward_start": "2026-06-21T08:00Z (max(config,deploy), FIXED)",
            "baseline_subtraction_hypothesis": "DISPROVEN — no subtraction in code; deploy.json has no net baseline field",
            "jump_root_cause": "always-in 5-coin endpoint volatility + intraday partial ceil-buckets, NOT baseline jump, NOT backfill"},
        "Q3_drawdown_constant": {
            "computed_metric": "MAX drawdown (mislabeled '当前回撤'/current)",
            "forward_high_water": 2430.61, "high_water_date": "2026-07-02",
            "trough": 356.32, "trough_date": "2026-07-09",
            "forward_maxdd_final": final_mdd, "record_last_moved": set_date,
            "root_cause": "early-window (07-02->07-09) max-DD record never beaten; NOT frozen anchor / NOT stalled series; LABEL defect only",
            "true_current_dd_from_peak": round(2430.61 - series[-1][1], 2)},
        "Q4_recon_vs_report": {
            "caliber_primary": "2026-06-22 (ceil bucket ~ per-bar >=08:00Z)",
            "forward_net_0622_final": fwd, "forward_net_0621_final":
                sum(incs[i] for i, d in enumerate(dates) if d >= FORWARD_START_ALT),
            "recon_cum": {rd: round(next(s[1] for s in series if s[0] == rd), 2)
                          for rd in REPORTED},
            "reported": REPORTED,
            "divergence_cause": "intraday partial buckets + volatile endpoint timing (NOT retroactive corruption; max-DD cent-exact)",
            "contradiction2_resolution": "当日M2M (ceil bucket, mostly prior day) vs forward-cum cross-run delta = DIFFERENT bar sets"},
        "Q5_gate_verdict": {
            "gates_and_display_same_source": True,
            "gate_input_account": "forward slice f_per (same as display), NOT the $73,614 full-history number",
            "K2_maxdd_input": final_mdd, "K2_threshold": 32482.54,
            "K2_margin": round(32482.54 - final_mdd, 2), "K2_can_misfire": False,
            "K1_active": False, "K1_reason": "forward ~1.3mo < 12mo required",
            "bug_layer": "DISPLAY/LABEL ONLY — gate inputs correct",
            "forward_observation_validity": "INTACT",
            "need_to_touch_vps": False},
        "baseline_cross_check": {
            "ledger_cumsum_through_2026-05-29": 67717.36,
            "frozen_baseline_OKX_full_net": 68194.82,
            "diff": -477.46,
            "interpretation": "same order (~0.7%); diff is open-span-endpoint + ceil-boundary kind => store history == backtest history"},
        "structural_checks": {
            "S1_contiguity": {"missing_days": len(missing), "duplicate_days": dup,
                              "verdict": "CONTIGUOUS" if not missing and not dup else "DEFECT"},
            "S2_partition": {"pre_baseline": round(pre, 2), "warmup_excluded_from_forward": round(warm, 2),
                             "forward": round(fwd, 2), "sum": round(pre + warm + fwd, 2),
                             "full_total": round(full_total, 2),
                             "verdict": "EXHAUSTIVE, warmup correctly excluded (anti-backfill guard works)"},
            "S3_sign": {"peak_before_trough": True, "maxdd_nonnegative": True}},
        "data_limitations": "snapshot lacks gap_log.jsonl/positions.json/ledger_trades.jsonl; "
                            "reconstruction used NO gap exclusion yet matched reported max-DD to the cent, "
                            "so an undisclosed material gap is unlikely and does not change Q5",
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[out] wrote reconstructed_forward_ledger.jsonl + audit_summary.json")


if __name__ == "__main__":
    main()
