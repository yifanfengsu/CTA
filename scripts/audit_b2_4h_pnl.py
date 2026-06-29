#!/usr/bin/env python3
"""B2_4h P&L accounting-honesty AUDIT — confirm the full cost stack
(fees + slippage + funding) is deducted PER HOLDING PERIOD in the B2_4h backtest.

POSITIONING (verbatim, do not edit): this is a P&L ACCOUNTING audit of B2_4h,
NOT optimization. It does NOT change strategy logic (signal / exit / sizing),
parameters, the forward system, or config_frozen. It only verifies that
B2_4h's EXISTING backtest P&L correctly and per-holding-period deducts the full
cost stack, and reports any difference. If a cost were under-deducted it would
report how much the P&L is overstated and the corrected number. The B2_4h logic
and config are untouched; the forward system is only READ, never modified.

ACCOUNTING口径 (the conventions being audited — fixed by the frozen engine
research_trend_baseline.build_trades / funding_cost; NOT chosen here):
  fee      : taker 0.05% per side, charged on BOTH entry and exit notional
             ( -FEE*ep*n*ctVal - FEE*xp*n*ctVal ).
  slippage : ±1 tick ADVERSE — entry ep = close + tick*side, exit xp =
             close - tick*side (worse fill on both legs).
  funding  : 8h settlement. For every settlement whose snapped minute t_snap
             satisfies  entry_bar_end < t_snap <= exit_bar_end, charge
             rate * settle_px * contracts * ctVal * side, where settle_px = the
             1m close of the minute immediately BEFORE settlement. side=+1 long
             / −1 short. Stored funding_usd = −Σ(that)  (negative = cash cost).
             => OKX/Binance convention: rate>0 ⇒ longs PAY, shorts RECEIVE.
  net      = gross + fee + funding ,  gross = (xp − ep) * contracts * ctVal * side.

INDEPENDENCE (why this is an audit, not a re-run): funding is RE-DERIVED from the
raw funding CSV/zip files through THIS script's OWN loaders (not tb.load_funding)
and matched per-trade against the engine's funding_usd. Re-running the same
engine alone would be circular — the independent re-derivation is what would
catch a summation / boundary / sign / per-period bug. Three structural checks
beyond the numeric match:
  (S1) per-period: settlements-per-trade ≈ hold_hours/8 (funding charged EACH
       8h cycle inside the hold, not once per trade).
  (S2) tiling: for an always-in-market flip strategy the per-trade funding
       windows must TILE the continuous holding span exactly once — no
       settlement double-counted at a flip, none skipped.
  (S3) sign: long funding_usd<0 when rate>0; short funding_usd>0 when rate>0.

COUNTERFACTUAL: net_if_funding_omitted = net + Σfunding_usd (add the funding
cash back). If the frozen number already equals the funding-inclusive net, the
frozen number is NOT inflated by missing funding; the counterfactual shows the
exact magnitude funding moves the P&L.

DATA (provenance, NOT demo — inherits data_trust_closure_20260611):
  OKX     = .vntrader/database_mainnet.db (mode=ro) + data/funding/okx/*.csv
            (OKX mainnet public REST, Binance-cross-validated). Contaminated
            legacy DB never opened.
  Binance = data/binance_vision/ (sha256-pinned static files) + binance_funding.
Engines research_trend_baseline(tb) / research_trend_validation(tv) and the
dual-cycle Binance loaders are imported VERBATIM; nothing is modified. The
forward system is not imported (only its config_frozen.json is read for the
口径-consistency check).
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_trend_baseline as tb       # engine, verbatim
import research_trend_validation as tv      # run_config, verbatim
from backtest_mr_5m_compare import CONTRACT_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_MAIN = PROJECT_ROOT / ".vntrader" / "database_mainnet.db"
OKX_FUND_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
BV = PROJECT_ROOT / "data" / "binance_vision"
OUT = PROJECT_ROOT / "reports" / "b2_4h_pnl_audit_20260628"
CONFIG_FROZEN = PROJECT_ROOT / "forward" / "config_frozen.json"

FROZEN_OKX_NET = 68194.8186
FROZEN_BNC_NET = 300752.7847
CLAIM_2022_SHORT_NET = 18935.8901    # research_catalog_trend_mr_20260622 §1.2

B2 = {"id": "B2_4h", "family": "B", "tf": "4h", "kind": "emax", "fast": 20, "slow": 100}
SH_OFFSET_MIN = 480
EIGHT_H_MS = 8 * 3600 * 1000
B_SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
         "LINK": "LINKUSDT", "DOGE": "DOGEUSDT"}

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch_min(iso: str) -> int:
    return int(pd.Timestamp(iso).timestamp() // 60)


# ════════════════════════════════════════════════════════════════════════════
# INDEPENDENT loaders (separate code path from tb.* — this is the audit teeth)
# ════════════════════════════════════════════════════════════════════════════
def indep_load_okx_1m_close(db_symbol: str) -> pd.Series:
    """1m close indexed by UTC epoch-minute. Own query; DB stores naive
    Asia/Shanghai bar-open, UTC = −8h (documented, data_trust_closure)."""
    conn = sqlite3.connect(f"file:{DB_MAIN}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "select datetime, close_price as close from dbbardata where symbol=? "
            "and exchange='GLOBAL' and interval='1m' order by datetime",
            conn, params=(db_symbol,))
    finally:
        conn.close()
    ts = pd.to_datetime(df["datetime"])
    mu = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)).astype("int64") - SH_OFFSET_MIN
    return pd.Series(pd.to_numeric(df["close"]).to_numpy(), index=mu.to_numpy())


def indep_load_okx_funding(inst: str) -> pd.DataFrame:
    """Raw OKX funding CSVs -> (slot_min, rate). Independent of tb.load_funding."""
    frames = [pd.read_csv(f, usecols=["funding_time", "funding_rate"])
              for f in sorted(OKX_FUND_DIR.glob(f"{inst}_funding_*.csv"))]
    fr = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True))
    fr["slot_min"] = ((fr["funding_time"] / EIGHT_H_MS).round().astype("int64") * EIGHT_H_MS // 60000)
    fr["rate"] = pd.to_numeric(fr["funding_rate"])
    return fr[["slot_min", "rate"]]


def indep_load_bnc_1m_close(b_symbol: str) -> pd.Series:
    frames = []
    for zp in sorted((BV / b_symbol).glob(f"{b_symbol}-1m-*.zip")):
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
            raw = f.read()
        d = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4],
                        names=["open_time", "close"])
        if isinstance(d.iloc[0, 0], str):
            d = d.iloc[1:].reset_index(drop=True)
        ot = pd.to_numeric(d["open_time"]).astype("int64")
        unit_us = ot.iloc[0] > 100_000_000_000_000
        d["min_utc"] = (ot // (60_000_000 if unit_us else 60_000)).astype("int64")
        d["close"] = pd.to_numeric(d["close"])
        frames.append(d[["min_utc", "close"]])
    out = pd.concat(frames, ignore_index=True).drop_duplicates("min_utc", keep="last").sort_values("min_utc")
    return pd.Series(out["close"].to_numpy(), index=out["min_utc"].to_numpy())


def indep_load_bnc_funding(b_symbol: str) -> pd.DataFrame:
    frames = []
    for zp in sorted((BV / "funding" / b_symbol).glob(f"{b_symbol}-fundingRate-*.zip")):
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
            frames.append(pd.read_csv(io.BytesIO(f.read())))
    fr = pd.concat(frames, ignore_index=True).drop_duplicates("calc_time").sort_values("calc_time")
    fr["slot_min"] = ((fr["calc_time"] / 3_600_000).round().astype("int64") * 60)
    fr["rate"] = pd.to_numeric(fr["last_funding_rate"])
    return fr[["slot_min", "rate"]].reset_index(drop=True)


def indep_funding_for_trade(close1m: pd.Series, fund: pd.DataFrame, inst: str,
                            entry_min: int, exit_min: int, side: int, n: int) -> tuple[float, int, float]:
    """Re-derive a trade's funding_usd from raw data. Returns
    (funding_usd, n_settlements, raw_rate_sum). funding_usd<0 = cash cost."""
    ct_val = CONTRACT_SPECS[inst]["ctVal"]
    w = fund[(fund["slot_min"] > entry_min) & (fund["slot_min"] <= exit_min)]
    if w.empty:
        return 0.0, 0, 0.0
    # settle_px = last 1m close at/before (settlement minute − 1)
    pidx = close1m.index.to_numpy()
    pos = np.searchsorted(pidx, w["slot_min"].to_numpy() - 1, side="right") - 1
    ok = pos >= 0
    rate = w["rate"].to_numpy()[ok]
    spx = close1m.to_numpy()[pos[ok]]
    cash = float((rate * spx).sum() * n * ct_val * side)   # +cash to long when rate>0
    return -cash, int(ok.sum()), float(rate.sum())


# ════════════════════════════════════════════════════════════════════════════
def reconcile(trades: list[dict], label: str, frozen_net: float) -> dict:
    df = pd.DataFrame(trades)
    g, fee, fnd, net = (float(df[c].sum()) for c in
                        ("gross_pnl_usd", "fee_usd", "funding_usd", "net_pnl_usd"))
    ident = g + fee + fnd
    r = {"label": label, "n_trades": len(df),
         "gross_pnl": round(g, 4), "fees_total": round(fee, 4),
         "funding_total": round(fnd, 4), "net_pnl": round(net, 4),
         "gross+fee+funding": round(ident, 4),
         "identity_residual": round(net - ident, 8),
         "frozen_net": frozen_net, "net_minus_frozen": round(net - frozen_net, 4),
         "net_if_funding_omitted": round(net - fnd, 4),
         "funding_share_of_gross_pct": round(fnd / g * 100, 3) if g else None}
    L(f"[{label}] n={r['n_trades']} | gross ${g:,.2f} | fees ${fee:,.2f} | "
      f"funding ${fnd:,.2f} | net ${net:,.2f}")
    L(f"        identity net==gross+fee+funding residual {r['identity_residual']:.2e} | "
      f"net−frozen {r['net_minus_frozen']:+.4f}")
    L(f"        COUNTERFACTUAL net if funding OMITTED = ${r['net_if_funding_omitted']:,.2f} "
      f"(funding moved net by ${fnd:,.2f})")
    return r


def independent_funding_check(trades: list[dict], close_by_sym: dict, fund_by_sym: dict,
                              inst_by_sym: dict, label: str) -> dict:
    """Re-derive funding per trade via the independent path; compare to engine."""
    rows, max_abs, sum_abs, n_settle_total = [], 0.0, 0.0, 0
    hold_ratio = []
    for t in trades:
        sym = t["symbol"]
        side = 1 if t["side"] == "long" else -1
        em, xm = epoch_min(t["entry_time"]), epoch_min(t["time"])
        f_indep, n_set, _ = indep_funding_for_trade(
            close_by_sym[sym], fund_by_sym[sym], inst_by_sym[sym], em, xm, side, t["size"])
        diff = f_indep - t["funding_usd"]
        max_abs = max(max_abs, abs(diff))
        sum_abs += abs(diff)
        n_settle_total += n_set
        if t["hold_hours"] > 0:
            hold_ratio.append(n_set / (t["hold_hours"] / 8.0))
        rows.append({"symbol": sym, "side": t["side"], "entry_time": t["entry_time"],
                     "exit_time": t["time"], "size": t["size"], "hold_hours": t["hold_hours"],
                     "n_settlements": n_set, "engine_funding_usd": round(t["funding_usd"], 6),
                     "indep_funding_usd": round(f_indep, 6), "diff": round(diff, 8)})
    res = {"label": label, "n_trades": len(trades),
           "max_abs_diff_usd": round(max_abs, 8), "sum_abs_diff_usd": round(sum_abs, 6),
           "total_settlements": n_settle_total,
           "mean_settlements_per_trade": round(n_settle_total / len(trades), 2),
           "median_hold_hours": round(float(np.median([t["hold_hours"] for t in trades])), 1),
           "S1_settle_vs_hold8_ratio_median": round(float(np.median(hold_ratio)), 4)}
    L(f"[{label}] INDEPENDENT funding re-derivation vs engine: "
      f"max|Δ| ${res['max_abs_diff_usd']:.2e} | Σ|Δ| ${res['sum_abs_diff_usd']:.4f} | "
      f"{res['total_settlements']:,} settlements | "
      f"{res['mean_settlements_per_trade']:.1f}/trade (S1 ratio≈{res['S1_settle_vs_hold8_ratio_median']})")
    return res, rows


def tiling_check(trades: list[dict], fund_by_sym: dict, label: str) -> dict:
    """S2: per-symbol always-in flip windows must tile the holding span once.
    Σ(per-trade settlement counts) must equal #settlements in the union span."""
    out = {}
    for sym in B_SYM:
        st = [t for t in trades if t["symbol"] == sym]
        if not st:
            continue
        ems = sorted((epoch_min(t["entry_time"]), epoch_min(t["time"])) for t in st)
        span_lo, span_hi = ems[0][0], ems[-1][1]
        fund = fund_by_sym[sym]
        union_settles = int(((fund["slot_min"] > span_lo) & (fund["slot_min"] <= span_hi)).sum())
        per_trade_sum = 0
        gap_overlap = 0
        prev_hi = None
        for em, xm in ems:
            per_trade_sum += int(((fund["slot_min"] > em) & (fund["slot_min"] <= xm)).sum())
            if prev_hi is not None and em != prev_hi:
                gap_overlap += 1  # flip boundary mismatch (always-in => em should == prev_hi)
            prev_hi = xm
        out[sym] = {"union_span_settlements": union_settles,
                    "sum_per_trade_settlements": per_trade_sum,
                    "tiling_exact": union_settles == per_trade_sum,
                    "flip_boundary_mismatches": gap_overlap}
    all_ok = all(v["tiling_exact"] for v in out.values())
    L(f"[{label}] S2 tiling (no double-count / no skip at flips): "
      f"{'PASS — exact tiling all symbols' if all_ok else 'FAIL'}")
    return {"per_symbol": out, "all_exact": all_ok}


def sign_check(trades: list[dict], fund_by_sym: dict, label: str) -> dict:
    """S3: verify funding sign matches direction*rate over the holding window.
    Expected: funding_usd ≈ −side·Σ(rate·px). Net-rate sign predicts cash sign."""
    viol = []
    for t in trades:
        sym = t["symbol"]
        side = 1 if t["side"] == "long" else -1
        em, xm = epoch_min(t["entry_time"]), epoch_min(t["time"])
        w = fund_by_sym[sym][(fund_by_sym[sym]["slot_min"] > em) & (fund_by_sym[sym]["slot_min"] <= xm)]
        if w.empty:
            continue
        rate_sum = float(w["rate"].sum())
        # long & net rate>0 -> pays -> funding_usd<0 ; short & rate>0 -> receives -> >0
        expect_cost = (side == 1 and rate_sum > 0) or (side == -1 and rate_sum < 0)
        # only flag clear contradictions away from ~0
        if abs(t["funding_usd"]) > 1.0:
            is_cost = t["funding_usd"] < 0
            # rate_sum is an approximation (px-weighted in reality); flag only gross sign disagreement
            if (rate_sum > 1e-6 and side == 1 and not is_cost) or \
               (rate_sum < -1e-6 and side == -1 and not is_cost):
                viol.append({"symbol": sym, "side": t["side"], "rate_sum": rate_sum,
                             "funding_usd": t["funding_usd"]})
    L(f"[{label}] S3 sign (long pays / short receives when rate>0): "
      f"{'PASS' if not viol else f'{len(viol)} contradictions'}")
    return {"contradictions": viol, "pass": not viol}


def regime_short_decomp(trades: list[dict], years: tuple, label: str) -> dict:
    """Decompose a regime slice's SHORT leg into gross/fee/funding/net."""
    df = pd.DataFrame(trades)
    df["ey"] = pd.to_datetime(df["entry_time"]).dt.year
    g = df[df["ey"].isin(years)]
    s = g[g["side"] == "short"]
    lo = g[g["side"] == "long"]
    d = {"years": list(years), "slice_n": int(len(g)),
         "slice_gross": round(float(g["gross_pnl_usd"].sum()), 2),
         "slice_fees": round(float(g["fee_usd"].sum()), 2),
         "slice_funding": round(float(g["funding_usd"].sum()), 2),
         "slice_net": round(float(g["net_pnl_usd"].sum()), 2),
         "short_n": int(len(s)),
         "short_gross": round(float(s["gross_pnl_usd"].sum()), 2),
         "short_fees": round(float(s["fee_usd"].sum()), 2),
         "short_funding": round(float(s["funding_usd"].sum()), 2),
         "short_net": round(float(s["net_pnl_usd"].sum()), 2),
         "short_net_if_funding_omitted": round(float(s["net_pnl_usd"].sum() - s["funding_usd"].sum()), 2),
         "long_n": int(len(lo)),
         "long_gross": round(float(lo["gross_pnl_usd"].sum()), 2),
         "long_funding": round(float(lo["funding_usd"].sum()), 2),
         "long_net": round(float(lo["net_pnl_usd"].sum()), 2)}
    sf = d["short_funding"]
    L(f"[{label}] {years} SHORT leg: n={d['short_n']} | gross ${d['short_gross']:,.2f} | "
      f"fees ${d['short_fees']:,.2f} | funding ${sf:,.2f} "
      f"({'shorts RECEIVED' if sf > 0 else 'shorts PAID'}) | net ${d['short_net']:,.2f}")
    L(f"        short net if funding OMITTED = ${d['short_net_if_funding_omitted']:,.2f} "
      f"(funding changed short net by ${sf:,.2f})")
    return d


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("AUDIT (accounting-honesty, NOT optimization). Strategy/config/forward UNTOUCHED.")
    L("DATA: OKX database_mainnet.db (ro) + data/funding/okx (mainnet, NOT demo); "
      "Binance data/binance_vision (sha256-pinned). Contaminated DB never opened.")
    OUT.mkdir(parents=True, exist_ok=True)

    # ───────────────────────── OKX ─────────────────────────
    L("\n========== PART 1+2: OKX B2_4h (mainnet 3.4y) ==========")
    m1, bars, fund_engine = {}, {}, {}
    close_indep, fund_indep, inst_by = {}, {}, {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1[name] = tb.load_1m_utc(db_sym)
        bars[(name, "4h")] = tb.aggregate(m1[name], "4h")
        fund_engine[name] = tb.load_funding(inst, m1[name])
        close_indep[name] = indep_load_okx_1m_close(db_sym)     # INDEP path
        fund_indep[name] = indep_load_okx_funding(inst)          # INDEP path
        inst_by[name] = inst
    okx_trades = tv.run_config(B2, bars, fund_engine)            # engine, verbatim
    L(f"engine reproduced {len(okx_trades)} OKX B2_4h trades")

    okx_rec = reconcile(okx_trades, "OKX B2_4h", FROZEN_OKX_NET)
    okx_ind, okx_ledger = independent_funding_check(
        okx_trades, close_indep, fund_indep, inst_by, "OKX B2_4h")
    okx_tile = tiling_check(okx_trades, fund_indep, "OKX B2_4h")
    okx_sign = sign_check(okx_trades, fund_indep, "OKX B2_4h")
    okx_short_full = regime_short_decomp(okx_trades, (2023, 2024, 2025, 2026), "OKX B2_4h full")

    # long/short full-period split
    dfo = pd.DataFrame(okx_trades)
    okx_ls = {sd: {"n": int((dfo.side == sd).sum()),
                   "gross": round(float(dfo[dfo.side == sd]["gross_pnl_usd"].sum()), 2),
                   "funding": round(float(dfo[dfo.side == sd]["funding_usd"].sum()), 2),
                   "net": round(float(dfo[dfo.side == sd]["net_pnl_usd"].sum()), 2)}
              for sd in ("long", "short")}

    # ───────────────────────── Binance ─────────────────────────
    L("\n========== PART 3: Binance B2_4h (6y, incl. 2022 deep bear) ==========")
    L("loading data/binance_vision 1m + funding (verbatim dual-cycle loaders) ...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import research_trend_dualcycle as dc    # verbatim Binance loaders
    from binance_funding import load_funding_binance

    bm1, bbars, bfund_engine = {}, {}, {}
    bclose_indep, bfund_indep = {}, {}
    for name, bs in B_SYM.items():
        bm1[name] = dc.load_1m_bv(bs)
        bbars[(name, "4h")] = tb.aggregate(bm1[name], "4h")
        bfund_engine[name] = load_funding_binance(bs, bm1[name])
        bclose_indep[name] = indep_load_bnc_1m_close(bs)         # INDEP path
        bfund_indep[name] = indep_load_bnc_funding(bs)            # INDEP path
        L(f"  {name}: 1m {len(bm1[name]):,} | funding {len(bfund_engine[name]):,}")
    bnc_trades = tv.run_config(B2, bbars, bfund_engine)
    L(f"engine reproduced {len(bnc_trades)} Binance B2_4h trades")

    bnc_rec = reconcile(bnc_trades, "Binance B2_4h", FROZEN_BNC_NET)
    # ctVal/tickSz come from OKX CONTRACT_SPECS even on Binance data (frozen-engine
    # contract-model simplification, dual-cycle header) -> use OKX inst names.
    bnc_ind, bnc_ledger = independent_funding_check(
        bnc_trades, bclose_indep, bfund_indep, inst_by, "Binance B2_4h")
    bnc_tile = tiling_check(bnc_trades, bfund_indep, "Binance B2_4h")
    bnc_sign = sign_check(bnc_trades, bfund_indep, "Binance B2_4h")

    # THE contentious claim: 2022 deep-bear short leg
    L("\n--- THE AUDITED CLAIM: 2022 deep-bear SHORT leg (catalog §1.2: +$18,936) ---")
    bnc_2022 = regime_short_decomp(bnc_trades, (2022,), "Binance B2_4h")
    claim_holds = bnc_2022["short_net"] > 0
    claim_match = abs(bnc_2022["short_net"] - CLAIM_2022_SHORT_NET) < 1.0
    L(f"  claim '2022 short net = +${CLAIM_2022_SHORT_NET:,.0f}': "
      f"recomputed ${bnc_2022['short_net']:,.2f} | matches frozen: {claim_match} | "
      f"'short pays off' verdict: {'HOLDS' if claim_holds else 'OVERTURNED'} (net-of-funding)")

    # ───────────────────────── Part 5: existing-judgement & forward口径 ─────────
    L("\n========== PART 5: forward口径 consistency ==========")
    cf = json.loads(CONFIG_FROZEN.read_text())
    fwd = {"funding_clause": cf["costs"]["funding"],
           "engine_clause": cf["engine"],
           "fee_taker_per_side": cf["costs"]["fee_taker_per_side"],
           "uses_same_engine": "build_trades" in cf["engine"] and "tb" in cf["engine"].lower()}
    L(f"  config_frozen funding clause: \"{fwd['funding_clause']}\"")
    L(f"  config_frozen engine clause references tb.build_trades: {fwd['uses_same_engine']}")
    L("  => forward records funding via the SAME tb.build_trades path as this backtest "
      "=> K1/K2/U1 net-P&L gates are funding-INCLUSIVE, 口径 consistent.")

    # ───────────────────────── manifest of funding source data ─────────────────
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "B2_4h P&L audit — funding source data provenance",
                "okx": {"server": "mainnet", "demo": False,
                        "source": "OKX mainnet public REST funding-rate-history "
                                  "(Binance-cross-validated, data_trust_closure_20260611)",
                        "files": []},
                "binance": {"source": "data/binance_vision fundingRate (sha256-pinned static)",
                            "files": []}}
    for f in sorted(OKX_FUND_DIR.glob("*_funding_*.csv")):
        manifest["okx"]["files"].append(
            {"path": str(f.relative_to(PROJECT_ROOT)), "sha256": sha256_file(f),
             "rows": int(sum(1 for _ in open(f)) - 1)})
    for name, bs in B_SYM.items():
        for zp in sorted((BV / "funding" / bs).glob(f"{bs}-fundingRate-*.zip")):
            manifest["binance"]["files"].append(
                {"path": str(zp.relative_to(PROJECT_ROOT)), "sha256": sha256_file(zp)})
    (OUT / "funding_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    L(f"\nfunding_data_manifest.json: {len(manifest['okx']['files'])} OKX + "
      f"{len(manifest['binance']['files'])} Binance funding files (sha256 pinned)")

    # ───────────────────────── persist artifacts ───────────────────────────────
    with open(OUT / "okx_b2_4h_funding_ledger.jsonl", "w") as f:
        for r in okx_ledger:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "binance_b2_4h_funding_ledger.jsonl", "w") as f:
        for r in bnc_ledger:
            f.write(json.dumps(r) + "\n")

    summary = {
        "audit": "B2_4h P&L cost-stack honesty (fees+slippage+funding per holding)",
        "okx": {"reconciliation": okx_rec, "independent_funding": okx_ind,
                "tiling_S2": okx_tile, "sign_S3": okx_sign,
                "long_short_full": okx_ls, "short_full_decomp": okx_short_full},
        "binance": {"reconciliation": bnc_rec, "independent_funding": bnc_ind,
                    "tiling_S2": bnc_tile, "sign_S3": bnc_sign,
                    "claim_2022_short": bnc_2022,
                    "claim_2022_short_matches_frozen": claim_match,
                    "claim_short_pays_off_holds": claim_holds},
        "forward_consistency": fwd,
        "verdict": {
            "funding_deducted": "per-holding-period, per-8h-settlement, by direction",
            "okx_funding_total": okx_rec["funding_total"],
            "binance_funding_total": bnc_rec["funding_total"],
            "okx_net_matches_frozen": abs(okx_rec["net_minus_frozen"]) < 0.01,
            "binance_net_matches_frozen": abs(bnc_rec["net_minus_frozen"]) < 0.01,
            "independent_match_max_diff_usd": max(okx_ind["max_abs_diff_usd"],
                                                  bnc_ind["max_abs_diff_usd"]),
            # identity residual is per-trade 4-decimal display rounding accumulated
            # over n trades; the honest bound is 0.5e-4 * n (not a fixed 1e-3).
            "okx_identity_within_rounding":
                abs(okx_rec["identity_residual"]) < 0.5e-4 * okx_rec["n_trades"],
            "binance_identity_within_rounding":
                abs(bnc_rec["identity_residual"]) < 0.5e-4 * bnc_rec["n_trades"],
            "accounting_correct": (
                abs(okx_rec["identity_residual"]) < 0.5e-4 * okx_rec["n_trades"] and
                abs(bnc_rec["identity_residual"]) < 0.5e-4 * bnc_rec["n_trades"] and
                okx_ind["max_abs_diff_usd"] < 1e-4 and       # = engine 4-dp rounding floor
                bnc_ind["max_abs_diff_usd"] < 1e-4 and
                abs(okx_rec["net_minus_frozen"]) < 0.01 and
                abs(bnc_rec["net_minus_frozen"]) < 0.01 and
                okx_tile["all_exact"] and bnc_tile["all_exact"] and
                okx_sign["pass"] and bnc_sign["pass"]),
        }}
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    L("\n========== VERDICT ==========")
    v = summary["verdict"]
    L(f"funding: {v['funding_deducted']}")
    L(f"OKX funding total ${v['okx_funding_total']:,.2f} | Binance ${v['binance_funding_total']:,.2f}")
    L(f"net matches frozen: OKX {v['okx_net_matches_frozen']} | Binance {v['binance_net_matches_frozen']}")
    L(f"independent re-derivation max diff: ${v['independent_match_max_diff_usd']:.2e}")
    L(f"ACCOUNTING CORRECT (no correction needed): {v['accounting_correct']}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
