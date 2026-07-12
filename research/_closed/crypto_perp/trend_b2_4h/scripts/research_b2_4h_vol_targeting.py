#!/usr/bin/env python3
"""B2_4h vol-targeting RISK-CONTROL variant (B2_4h-VT) — pre-registered.

POSITIONING (verbatim into report header, do not edit):
  本任务为 B2_4h 设计 vol-targeting 风控变体（B2_4h-VT）。性质=风控（控制风险），
  **不是 alpha 增强**——它不改变 B2_4h 选哪些交易（信号、出场不变），只改变每笔
  交易承担多少风险（仓位∝1/σ）。这与四次已判死的 alpha 增强（funding/ADX/faster/
  V1，都试图改进 edge 本身）层面不同。
  成败唯一关键：lookback 按经济理由预注册定死，绝不对样本内挑最优（挑最优=描线）。
  诚实预期：成功=更稳（回撤降/Calmar 改善），不是更强（总收益提升）；总收益提升
  → 高度警惕描线（接铁律：声称"纯增益"自动触发最高警惕，已被骗 4 次）。
  不污染前向：正在 VPS 跑的 B2_4h 用 config_frozen，本变体独立预注册验证，绝不改
  config_frozen、不碰前向系统、不替换正在跑的 B2_4h。
  判定哲学：四次增强同一防线（双样本+机制+DSR 打折）；不用 Sharpe 单独主判（正偏，
  看整个分布+回撤+稳健性）；预注册先于结果。

──────────────────────────────────────────────────────────────────────────────
PRE-REGISTERED DESIGN (fixed before results — IRON RULE A; nothing below tuned):

  Position rule  : notional_t = NOTIONAL_base × clip( σ_target / σ_t , FLOOR, CAP )
                   contracts  = max(1, round(notional_t / (entry_px × ctVal)))
                   Signal/exit IDENTICAL to B2_4h (EMA20/100 always-in, flip on
                   reversal). ONLY position size changes. Cost stack (±1 tick
                   taker both sides + per-holding 8h funding) reused VERBATIM
                   from the funding-audited engine (reports/b2_4h_pnl_audit_*).

  σ_t (EWMA)     : EWMA of 4h log-return², annualised (×√(365×6)). Recursive
                   (adjust=False), CAUSAL — σ at entry bar uses returns through
                   the entry close only (sized when we enter, no look-ahead).

  EWMA half-life : **48 bars (4h) = 8 days**, derived from B2_4h MEDIAN HOLD
                   196h ≈ 8.17 days = 49 bars. ECONOMIC REASON: the vol scale
                   relevant to sizing a position held ~8 days is the vol over a
                   ~8-day timescale. Fixed before results; the neighbourhood
                   {24, 48, 96} = {4d, 8d, 16d} is for ROBUSTNESS ONLY — never
                   to pick a best lookback.

  σ_target       : **0.65 annualised (65%)**. ECONOMIC REASON: a moderate crypto
                   vol budget per position (majors realise ~0.5–0.9 ann). The
                   VALUE IS IMMATERIAL TO THE VERDICT: Calmar (= return/maxDD) is
                   scale-invariant, and the head-line maxDD comparison is
                   EXPOSURE-MATCHED (VT analytically scaled so its time-integrated
                   $-exposure equals the base book's), so any constant σ_target
                   cancels. Stated so no one suspects 0.65 was fitted.

  FLOOR / CAP    : multiplier clipped to **[0.25, 4.0]**. RISK REASON: never
                   below 0.25× base (keep meaningful exposure when σ huge), never
                   above 4× base (prevent leverage blow-up when σ tiny). Fixed.

  Baseline       : B2_4h fixed $10,000 notional/signal (config_frozen UNTOUCHED).
                   Same symbols / window / dual sample / cost convention.

  EXPOSURE MATCH : the vol-targeting *mechanism* (does redistributing exposure by
                   1/σ reduce risk?) is isolated from any leverage difference by
                   scaling VT to the base book's time-integrated exposure
                   k = Σ_base(notional·hold) / Σ_VT(notional·hold); VT_matched
                   P&L = k × VT P&L (net_i ∝ n_i, so equity scales by k, maxDD
                   scales by k, Calmar invariant). Head-line DD/return use
                   VT_matched; raw VT and k reported for transparency.

──────────────────────────────────────────────────────────────────────────────
PRE-REGISTERED VERDICT (fixed before results):
  ADOPT B2_4h-VT (as a more robust RISK-CONTROL variant) ⟺ ALL of:
    (i)   maxDD_matched(VT) < maxDD(base)        in BOTH samples (OKX & Binance)
    (ii)  Calmar(VT) > Calmar(base)              in BOTH samples
    (iii) (i)&(ii) SIGN-STABLE across half-life {24,48,96} (effect robust, not a
          single-lookback artefact)
    (iv)  total return FLAT or DOWN; if UP, the "gain" must NOT be lookback-
          dependent (else disguised line-drawing → fail)
    (v)   right-tail capture (top-5% base-winner net, matched) ≥ 0.70 in BOTH
          samples (VT did not gut the big winners the right-skew edge lives in)
  DO NOT ADOPT ⟺ ANY of:
    - DD/Calmar effect appears in only one sample or only one half-life (line)
    - return "improves" AND is lookback-dependent (line disguised as gain)
    - right-tail capture < 0.70 either sample (severe削弱 — iron rule C: cutting
      the right tail of a right-skew strategy is the cardinal sin)
  Sharpe / DSR are a CHECK (right-skew → not the main judge); reported, not gated.
  EVEN IF ADOPTED: B2_4h-VT is a NEW variant needing its OWN forward verification
  before real money; the running B2_4h forward is unaffected and NOT replaced.

DATA: OKX database_mainnet.db (ro) + data/funding/okx (mainnet, audited, NOT
demo); Binance data/binance_vision (sha256-pinned). Engines tb/tv + dual-cycle
loaders imported VERBATIM. config_frozen / forward system NOT touched.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2026-07 重构批次5：脚本迁入 research/_closed/crypto_perp/trend_b2_4h/scripts/；共享依赖真身在
# scripts/（前向冻结区）与 core/data_io/，此处按新深度注入 sys.path。
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[5]
for _p in (
    str(_REPO_ROOT / "core" / "data_io"),
    str(_REPO_ROOT / "scripts"),
    *sorted(str(_q) for _q in (_REPO_ROOT / "research" / "_closed").glob("*/*/scripts")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import research_trend_baseline as tb
import research_trend_validation as tv
from backtest_mr_5m_compare import CONTRACT_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/trend_b2_4h/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "b2_4h_vol_targeting_20260628"

# ── PRE-REGISTERED (fixed before results) ────────────────────────────────────
B2 = {"id": "B2_4h", "tf": "4h", "kind": "emax", "fast": 20, "slow": 100}
HALFLIFE_PREREG = 48           # bars (4h) = 8 days = median hold 196h scale
HALFLIFE_NBHD = [24, 48, 96]   # {0.5x, 1x, 2x} = {4d, 8d, 16d}, robustness only
SIGMA_TARGET = 0.65            # annualised; immaterial to verdict (scale-invariant)
MULT_FLOOR, MULT_CAP = 0.25, 4.0
BARS_PER_YEAR_4H = 365 * 6
FROZEN = {"OKX": 68194.8186, "Binance": 300752.7847}
RIGHT_TAIL_MIN_CAPTURE = 0.70  # top-5% matched capture floor (verdict v)

B_SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
         "LINK": "LINKUSDT", "DOGE": "DOGEUSDT"}
LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── EWMA realised vol (causal, annualised) ───────────────────────────────────
def ewma_sigma_ann(bars: pd.DataFrame, halflife_bars: int) -> np.ndarray:
    c = bars["close"].to_numpy()
    r = np.zeros(len(c))
    r[1:] = np.log(c[1:] / c[:-1])
    var = pd.Series(r ** 2).ewm(halflife=halflife_bars, adjust=False).mean().to_numpy()
    return np.sqrt(var) * np.sqrt(BARS_PER_YEAR_4H)


# ── trade builder (general n; base & VT share one code path) ─────────────────
def build_trades_sized(name, inst, bars, fund, spans, n_of_span) -> list[dict]:
    """Verbatim copy of tb.build_trades cost math; only n is injected.
    n_of_span(ei, ep_raw) -> integer contracts. With fixed-notional n this
    reproduces tb.build_trades to the cent (asserted in selfcheck)."""
    tick = CONTRACT_SPECS[inst]["tickSz"]
    ct_val = CONTRACT_SPECS[inst]["ctVal"]
    c = bars["close"].to_numpy()
    end_min = bars["end_min"].to_numpy()
    trades = []
    for ei, xi, side, reason in spans:
        ep_raw, xp_raw = c[ei], c[xi]
        ep = ep_raw + tick * side
        xp = xp_raw - tick * side
        n = n_of_span(ei, ep_raw)
        gross = (xp - ep) * n * ct_val * side
        fee = -(tb.FEE_TAKER * ep * n * ct_val + tb.FEE_TAKER * xp * n * ct_val)
        fnd = -tb.funding_cost(fund, int(end_min[ei]), int(end_min[xi]), side, n, ct_val)
        t_en = pd.Timestamp(int(end_min[ei]) * 60, unit="s", tz="UTC")
        t_ex = pd.Timestamp(int(end_min[xi]) * 60, unit="s", tz="UTC")
        trades.append({
            "time": t_ex.isoformat(), "symbol": name,
            "side": "long" if side == 1 else "short",
            "entry_time": t_en.isoformat(), "entry_price": round(float(ep), 8),
            "exit_price": round(float(xp), 8), "exit_reason": reason,
            "size": int(n), "notional_usd": round(float(n * ep_raw * ct_val), 2),
            "hold_hours": float((int(end_min[xi]) - int(end_min[ei])) / 60),
            "gross_pnl_usd": round(float(gross), 4), "fee_usd": round(float(fee), 4),
            "funding_usd": round(float(fnd), 4),
            "net_pnl_usd": round(float(gross + fee + fnd), 4)})
    return trades


def build_pair(bars_by, fund, spans_by, sigma_by, sigma_target):
    """Return aligned (base_trades, vt_trades) with VT multiplier recorded."""
    base, vt = [], []
    for name, (_, inst) in tb.SYMBOLS.items():
        b = bars_by[(name, "4h")]
        ct_val = CONTRACT_SPECS[inst]["ctVal"]
        spans = spans_by[name]
        sig = sigma_by[name]

        def n_base(ei, ep_raw, inst=inst):
            return tb.calc_contracts(inst, ep_raw)

        def n_vt(ei, ep_raw, ct_val=ct_val, sig=sig):
            mult = float(np.clip(sigma_target / sig[ei], MULT_FLOOR, MULT_CAP))
            return max(1, round(tb.NOTIONAL * mult / (ep_raw * ct_val)))

        bt = build_trades_sized(name, inst, b, fund[name], spans, n_base)
        vtr = build_trades_sized(name, inst, b, fund[name], spans, n_vt)
        for k, (ei, xi, side, _) in enumerate(spans):
            mult = float(np.clip(sigma_target / sig[ei], MULT_FLOOR, MULT_CAP))
            vtr[k]["vt_mult"] = round(mult, 4)
            vtr[k]["sigma_ann_entry"] = round(float(sig[ei]), 4)
            bt[k]["sigma_ann_entry"] = round(float(sig[ei]), 4)
        base.extend(bt)
        vt.extend(vtr)
    return base, vt


# ── metrics ──────────────────────────────────────────────────────────────────
def equity_dd(trades: list[dict], scale: float = 1.0):
    df = pd.DataFrame(trades).sort_values("time", kind="stable")
    net = df["net_pnl_usd"].to_numpy() * scale
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return eq, dd


def drawdown_episodes(trades: list[dict], scale: float = 1.0, top=5):
    """Top drawdown episodes (depth + duration in days) on the trade-exit clock."""
    df = pd.DataFrame(trades).sort_values("time", kind="stable").reset_index(drop=True)
    t = pd.to_datetime(df["time"]).to_numpy()
    eq = np.cumsum(df["net_pnl_usd"].to_numpy() * scale)
    peak = eq[0]
    peak_t = t[0]
    eps, cur = [], None
    for i in range(len(eq)):
        if eq[i] >= peak:
            if cur:
                eps.append(cur)
                cur = None
            peak, peak_t = eq[i], t[i]
        else:
            depth = peak - eq[i]
            if cur is None or depth > cur["depth"]:
                cur = {"depth": float(depth), "start": peak_t, "trough_t": t[i]}
            if cur:
                cur["dur_days"] = float((t[i] - cur["start"]) / np.timedelta64(1, "D"))
    if cur:
        eps.append(cur)
    eps = sorted(eps, key=lambda e: e["depth"], reverse=True)[:top]
    return [{"depth_usd": round(e["depth"], 0), "dur_days": round(e.get("dur_days", 0), 1)}
            for e in eps]


def variant_metrics(trades: list[dict], scale: float = 1.0) -> dict:
    df = pd.DataFrame(trades)
    net = df["net_pnl_usd"].to_numpy() * scale
    gross = df["gross_pnl_usd"].to_numpy() * scale
    _, dd = equity_dd(trades, scale)
    maxdd = float(dd.max())
    total = float(net.sum())
    # per-trade dollar-vol (risk) = notional × σ_ann (constancy of RISK exposure)
    dvol = (df["notional_usd"].to_numpy() * scale) * df["sigma_ann_entry"].to_numpy()
    return {"n": len(df), "net": round(total, 2), "gross": round(float(gross.sum()), 2),
            "funding": round(float(df["funding_usd"].sum() * scale), 2),
            "max_dd_usd": round(maxdd, 2),
            "calmar": round(total / maxdd, 4) if maxdd > 0 else None,
            "win_rate_pct": round(float((net > 0).mean() * 100), 2),
            "dollar_vol_mean": round(float(dvol.mean()), 1),
            "dollar_vol_cv": round(float(dvol.std() / dvol.mean()), 4) if dvol.mean() else None,
            "avg_notional": round(float(df["notional_usd"].mean() * scale), 1)}


def by_year(trades: list[dict], scale: float = 1.0) -> dict:
    df = pd.DataFrame(trades)
    df["ey"] = pd.to_datetime(df["entry_time"]).dt.year
    return {int(y): round(float(g["net_pnl_usd"].sum() * scale), 0)
            for y, g in df.groupby("ey")}


def exposure_match_k(base: list[dict], vt: list[dict]) -> float:
    """k so VT time-integrated $-exposure == base's (isolates timing from leverage)."""
    be = sum(t["notional_usd"] * t["hold_hours"] for t in base)
    ve = sum(t["notional_usd"] * t["hold_hours"] for t in vt)
    return be / ve if ve else 1.0


def right_tail_capture(base: list[dict], vt: list[dict], k: float, pct: float) -> dict:
    """Select top-pct trades by BASE net; measure VT_matched net on the SAME trades."""
    idx = np.argsort([t["net_pnl_usd"] for t in base])[::-1]
    m = max(1, int(len(base) * pct))
    sel = idx[:m]
    base_sum = sum(base[i]["net_pnl_usd"] for i in sel)
    vt_sum = sum(vt[i]["net_pnl_usd"] * k for i in sel)
    mults = [vt[i]["vt_mult"] for i in sel]
    return {"pct": pct, "n": m, "base_net": round(base_sum, 0),
            "vt_matched_net": round(vt_sum, 0),
            "capture_ratio": round(vt_sum / base_sum, 4) if base_sum else None,
            "avg_vt_mult_on_winners": round(float(np.mean(mults)), 4)}


def daily_m2m(bars_by, fund, spans_by, n_by_span_sym) -> pd.Series:
    """Portfolio per-bar mark-to-market (funding to settlement bar), daily-summed.
    n_by_span_sym[name] = list of contracts aligned with spans_by[name]."""
    total = None
    for name, (_, inst) in tb.SYMBOLS.items():
        spec = CONTRACT_SPECS[inst]
        b = bars_by[(name, "4h")]
        c = b["close"].to_numpy()
        endm = b["end_min"].to_numpy()
        pnl = np.zeros(len(b))
        f = fund[name]
        fmins = f["slot_min"].to_numpy()
        fpay = (f["rate"] * f["settle_px"]).to_numpy()
        for (ei, xi, side, _), n in zip(spans_by[name], n_by_span_sym[name]):
            pnl[ei + 1:xi + 1] += np.diff(c[ei:xi + 1]) * n * spec["ctVal"] * side
            pnl[ei] -= spec["tickSz"] * n * spec["ctVal"] + tb.FEE_TAKER * (c[ei] + spec["tickSz"]) * n * spec["ctVal"]
            pnl[xi] -= spec["tickSz"] * n * spec["ctVal"] + tb.FEE_TAKER * (c[xi] - spec["tickSz"]) * n * spec["ctVal"]
            in_span = (fmins > endm[ei]) & (fmins <= endm[xi])
            bi = np.searchsorted(endm, fmins[in_span], side="left")
            for j, p in zip(bi, fpay[in_span] * n * spec["ctVal"] * side):
                if j < len(b):
                    pnl[j] -= p
        s = pd.Series(pnl, index=pd.to_datetime(endm * 60, unit="s", utc=True))
        total = s if total is None else total.add(s, fill_value=0)
    return total.groupby(total.index.ceil("D")).sum()


def daily_sharpe_ann(daily: np.ndarray) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan")


# ── per-sample run ────────────────────────────────────────────────────────────
def run_sample(label, bars_by, fund) -> dict:
    L(f"\n===== {label} =====")
    spans_by = {name: tb.positions_flip(tb.signal_emax(bars_by[(name, '4h')], 20, 100))
                for name in tb.SYMBOLS}
    # base sized fixed-$10k (n_base), but annotated with REAL pre-reg σ so the
    # base dollar-vol (risk) constancy is measured correctly. σ does NOT affect
    # base sizing -> base_net still == frozen (asserted below).
    sigma_pre = {n: ewma_sigma_ann(bars_by[(n, "4h")], HALFLIFE_PREREG) for n in tb.SYMBOLS}
    base, _ = build_pair(bars_by, fund, spans_by, sigma_pre, SIGMA_TARGET)
    base_net = sum(t["net_pnl_usd"] for t in base)
    assert abs(base_net - FROZEN[label]) < 0.01, f"base {base_net} != frozen {FROZEN[label]}"
    L(f"base reproduces frozen net ${base_net:,.2f} (==${FROZEN[label]:,.2f}) ✓")
    bm = variant_metrics(base)

    out = {"label": label, "base": bm, "base_by_year": by_year(base), "vt": {}}
    base_daily = daily_m2m(bars_by, fund, spans_by,
                           {n: [tb.calc_contracts(tb.SYMBOLS[n][1], bars_by[(n, '4h')]["close"].to_numpy()[ei])
                                for ei, *_ in spans_by[n]] for n in tb.SYMBOLS})
    out["base"]["sharpe_ann"] = round(daily_sharpe_ann(base_daily.to_numpy()), 4)

    for hl in HALFLIFE_NBHD:
        sigma_by = {n: ewma_sigma_ann(bars_by[(n, "4h")], hl) for n in tb.SYMBOLS}
        _, vt = build_pair(bars_by, fund, spans_by, sigma_by, SIGMA_TARGET)
        k = exposure_match_k(base, vt)
        vm = variant_metrics(vt, scale=k)
        vm_raw = variant_metrics(vt, scale=1.0)
        rt5 = right_tail_capture(base, vt, k, 0.05)
        rt1 = right_tail_capture(base, vt, k, 0.01)
        clip_lo = float(np.mean([t["vt_mult"] <= MULT_FLOOR + 1e-9 for t in vt]))
        clip_hi = float(np.mean([t["vt_mult"] >= MULT_CAP - 1e-9 for t in vt]))
        entry = {"halflife_bars": hl, "exposure_match_k": round(k, 4),
                 "vt_matched": vm, "vt_raw_net": vm_raw["net"], "vt_raw_maxdd": vm_raw["max_dd_usd"],
                 "by_year_matched": by_year(vt, scale=k),
                 "right_tail_top5": rt5, "right_tail_top1": rt1,
                 "clip_floor_share": round(clip_lo, 4), "clip_cap_share": round(clip_hi, 4),
                 "dd_episodes_base": drawdown_episodes(base),
                 "dd_episodes_vt_matched": drawdown_episodes(vt, scale=k)}
        if hl == HALFLIFE_PREREG:
            vt_daily = daily_m2m(bars_by, fund, spans_by,
                                 {n: [max(1, round(tb.NOTIONAL * float(np.clip(SIGMA_TARGET / sigma_by[n][ei], MULT_FLOOR, MULT_CAP)) /
                                          (bars_by[(n, '4h')]["close"].to_numpy()[ei] * CONTRACT_SPECS[tb.SYMBOLS[n][1]]["ctVal"])))
                                      for ei, *_ in spans_by[n]] for n in tb.SYMBOLS})
            entry["vt_matched"]["sharpe_ann"] = round(daily_sharpe_ann(vt_daily.to_numpy() * k), 4)
        out["vt"][hl] = entry
        L(f"  hl={hl:>3} | k={k:.3f} | base maxDD ${bm['max_dd_usd']:,.0f} -> VT ${vm['max_dd_usd']:,.0f} "
          f"({(vm['max_dd_usd']/bm['max_dd_usd']-1)*100:+.0f}%) | base net ${bm['net']:,.0f} -> VT ${vm['net']:,.0f} "
          f"({(vm['net']/bm['net']-1)*100:+.0f}%) | Calmar {bm['calmar']:.2f}->{vm['calmar']:.2f} "
          f"| RT5 capture {rt5['capture_ratio']:.2f} (mult {rt5['avg_vt_mult_on_winners']:.2f}) "
          f"| $vol CV {bm['dollar_vol_cv']:.2f}->{vm['dollar_vol_cv']:.2f}")
    return out


def verdict(okx: dict, bnc: dict) -> dict:
    hl = HALFLIFE_PREREG
    res = {}
    for s, d in (("OKX", okx), ("Binance", bnc)):
        b, v = d["base"], d["vt"][hl]["vt_matched"]
        res[s] = {"dd_reduced": v["max_dd_usd"] < b["max_dd_usd"],
                  "calmar_improved": (v["calmar"] or -9) > (b["calmar"] or 9),
                  "return_up": v["net"] > b["net"] * 1.001,
                  "rt5_capture": d["vt"][hl]["right_tail_top5"]["capture_ratio"],
                  "rt5_ok": (d["vt"][hl]["right_tail_top5"]["capture_ratio"] or 0) >= RIGHT_TAIL_MIN_CAPTURE,
                  "dd_pct": round((v["max_dd_usd"] / b["max_dd_usd"] - 1) * 100, 1),
                  "net_pct": round((v["net"] / b["net"] - 1) * 100, 1)}
    # neighbourhood sign-stability of DD reduction & Calmar improvement
    def nbhd_stable(d):
        dd = [d["vt"][h]["vt_matched"]["max_dd_usd"] < d["base"]["max_dd_usd"] for h in HALFLIFE_NBHD]
        cal = [(d["vt"][h]["vt_matched"]["calmar"] or -9) > (d["base"]["calmar"] or 9) for h in HALFLIFE_NBHD]
        return all(dd), all(cal)
    okx_dd_stab, okx_cal_stab = nbhd_stable(okx)
    bnc_dd_stab, bnc_cal_stab = nbhd_stable(bnc)
    dual_dd = res["OKX"]["dd_reduced"] and res["Binance"]["dd_reduced"]
    dual_cal = res["OKX"]["calmar_improved"] and res["Binance"]["calmar_improved"]
    nbhd_ok = okx_dd_stab and bnc_dd_stab and okx_cal_stab and bnc_cal_stab
    rt_ok = res["OKX"]["rt5_ok"] and res["Binance"]["rt5_ok"]
    ret_up_both = res["OKX"]["return_up"] and res["Binance"]["return_up"]
    adopt = dual_dd and dual_cal and nbhd_ok and rt_ok and not (ret_up_both and not nbhd_ok)
    return {"per_sample": res,
            "dual_dd_reduced": dual_dd, "dual_calmar_improved": dual_cal,
            "neighbourhood_sign_stable": {"okx_dd": okx_dd_stab, "okx_calmar": okx_cal_stab,
                                          "bnc_dd": bnc_dd_stab, "bnc_calmar": bnc_cal_stab, "all": nbhd_ok},
            "right_tail_ok_both": rt_ok, "return_up_both_samples": ret_up_both,
            "ADOPT": bool(adopt),
            "reason": _verdict_reason(dual_dd, dual_cal, nbhd_ok, rt_ok, res)}


def _verdict_reason(dual_dd, dual_cal, nbhd_ok, rt_ok, res):
    if not rt_ok:
        return ("不采纳 — 右尾被削弱（top5% 捕获 OKX %.2f / Binance %.2f < %.2f），"
                "vol-targeting 削掉趋势赖以盈利的大赢家（铁律C）" %
                (res["OKX"]["rt5_capture"], res["Binance"]["rt5_capture"], RIGHT_TAIL_MIN_CAPTURE))
    if not (dual_dd and dual_cal):
        return "不采纳 — 回撤/Calmar 改善未在双样本一致成立"
    if not nbhd_ok:
        return "不采纳 — 风控效果依赖特定 lookback（邻域内符号不稳）= 描线"
    return "采纳 — 双样本一致降回撤+Calmar改善、邻域稳健、右尾保留；但需独立前向验证才能上实盘"


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("B2_4h-VT vol-targeting RISK-CONTROL variant — pre-registered (half-life 48 bars "
      "= 8d = median hold scale; σ_target 0.65; clip [0.25,4.0]); config_frozen UNTOUCHED")
    OUT.mkdir(parents=True, exist_ok=True)

    # OKX
    okx_bars, okx_fund = {}, {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1 = tb.load_1m_utc(db_sym)
        okx_bars[(name, "4h")] = tb.aggregate(m1, "4h")
        okx_fund[name] = tb.load_funding(inst, m1)
    okx = run_sample("OKX", okx_bars, okx_fund)

    # Binance (verbatim dual-cycle loaders)
    import research_trend_dualcycle as dc
    from binance_funding import load_funding_binance
    bnc_bars, bnc_fund = {}, {}
    for name, bs in B_SYM.items():
        m1 = dc.load_1m_bv(bs)
        bnc_bars[(name, "4h")] = tb.aggregate(m1, "4h")
        bnc_fund[name] = load_funding_binance(bs, m1)
    bnc = run_sample("Binance", bnc_bars, bnc_fund)

    vd = verdict(okx, bnc)
    L("\n===== VERDICT =====")
    L(f"dual DD reduced: {vd['dual_dd_reduced']} | dual Calmar improved: {vd['dual_calmar_improved']} "
      f"| neighbourhood stable: {vd['neighbourhood_sign_stable']['all']} | right-tail ok both: {vd['right_tail_ok_both']} "
      f"| return up both: {vd['return_up_both_samples']}")
    L(f"ADOPT: {vd['ADOPT']} — {vd['reason']}")

    summary = {"prereg": {"halflife_bars": HALFLIFE_PREREG, "halflife_nbhd": HALFLIFE_NBHD,
                          "sigma_target_ann": SIGMA_TARGET, "mult_clip": [MULT_FLOOR, MULT_CAP],
                          "right_tail_min_capture": RIGHT_TAIL_MIN_CAPTURE,
                          "median_hold_hours": 196, "economic_anchor": "half-life ≈ median hold ≈ 8d"},
               "OKX": okx, "Binance": bnc, "verdict": vd}
    (OUT / "vt_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    L(f"\nrun end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
