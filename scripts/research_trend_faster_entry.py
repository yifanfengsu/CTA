#!/usr/bin/env python3
"""Faster-entry single-point test: EMA10/30 vs B2_4h (EMA20/100) (pre-registered, ZERO search).

POSITIONING (verbatim into report header, never deleted/edited):
  ADX 过滤研究意外发现：~70% 的趋势头部利润落在"早期低 ADX 入场"（OKX 70.0%/Binance 70.7%，
  双样本一致），提示趋势利润集中在趋势启动早期。本研究检验由此引出的单一假设：更早入场
  （更快均线对）是否捕获更多趋势利润、扣成本后优于 B2_4h。
  本研究是已知靶子上的参数检验，过拟合风险极高，且比指标过滤更隐蔽——靶子(B2_4h 历史)已知、
  且裸基线阶段已扫过 EMA 参数族，"更早入场"本质是重新挑参数。防线设为最严：① 仅测一个文献
  标准的更快参数对(EMA10/30)，绝对零搜索零变体——不测 5/20、15/50 或任何其他更快对；
  ② OKX 定义、Binance 6 年确认，双样本不一致即判死；③ 更早入场必然增加假信号，必须扣成本后
  净收益优于 B2_4h 才算数；④ 笔数调整后的验证周期记账。
  诚实预期声明：裸基线阶段(reports/trend_baseline_*)的 EMA 参数测试中，慢参数(EMA20/100→B2_4h)
  胜出。本检验很可能确认"更快入场不如 B2_4h"。无论结果，这是一个干净的双样本句号，替代
  "感觉更早应该更好"的悬念。

═══ PART 1 — CONTRAST DEFINITION (pre-registered, ZERO variants) ═══
  唯一对照 — 两个 EMA 参数对，其余完全一致：
    V0     = B2_4h 原始 EMA20/100，4h，always-in-market，金叉多/死叉空，信号反向出场（基准，复现冻结）。
    V_fast = EMA10/30（更快=更早入场），结构与 V0 完全相同（4h、always-in、反向出场、同 5 币、
             同 $10k 名义、同成本口径 taker±1tick 0.05%/边 + 真实 funding）。
  EMA10/30 选择依据：经典动量/趋势跟踪标准快参数对（文献常见），非历史最优挑选。绝对禁止测试
  任何其他更快对（5/20、15/50、12/26…）——多于一个对即参数搜索=描线。EMA10/30 无优势即结论
  "更早入场(此参数)无优势"，不换参数再试。

═══ PART 2 — DUAL-SAMPLE EVALUATION ═══
  OKX (2023-2026 定义) 与 Binance (2020-2026 确认) 各跑 V0/V_fast 同口径。逐版本：净利、净 Sharpe
  （日 M2M 年化）、笔数、净 PF、maxDD、net/maxDD、分年度、bootstrap 95% CI (10,000 seed=20260611)。
  OKX 上 V_fast 改善不足为信（靶子已知 + 参数已扫过）；Binance 6 年同向改善才可信。

═══ PART 3 — MECHANISM (更早入场是真趋势捕获还是撞成本墙) ═══
   (3a) whipsaw：V_fast vs V0 笔数增加多少；更快均线更多交叉——增加的交易里多少是短命 whipsaw
        （持仓 ≤ WHIP_BARS=2 bar=8h 即被反向平掉、未吃趋势），及其净贡献。
   (3b) 大趋势早期捕获：对 V0 top-10% 盈利交易，检查 V_fast 是否在 V0 入场时已持同向仓（更早入场）、
        早入多少小时。双样本。
   (3c) 毛 vs 净：V_fast 毛利(更早多吃的趋势) vs 净利(扣更多假信号成本)。毛升但净被吃平/吃负 =
        撞成本墙（同 5m MR / 突破回调死因），此市场对更高频趋势入场的结构性惩罚。
   (3d) mean/median 分列（纪律）。

═══ PART 4 — TRADE-COUNT / HORIZON LEDGER (加法陷阱) ═══
  V_fast 是加法（更多信号）。笔数 vs V0；PRIMARY 验证周期 = 日 M2M 年化 Sharpe 的 (1.96/SR_ann)²
  （频率/事件密度调整后）。更多笔数若伴 Sharpe 下降，事件密度提升未必转化为更短周期。
  核心量：V_fast 是否扣成本后同时改善 Sharpe 且缩短验证周期。

═══ PART 5 — JUDGMENT (pre-registered, immutable) ═══
  F1 改善真实：V_fast 净 Sharpe(日年化) > V0，双样本一致（OKX ∧ Binance 同向；仅 OKX = 描线判死）。
  F2 成本可活：V_fast 净利 > V0 净利，双样本一致（更早多吃趋势盖过更多假信号成本）。
  F3 机制成立：V_fast 毛利 > V0 毛利（真多吃趋势）且 V0 大赢家中 V_fast 更早入场占比 ≥ 50%（3b），
        双样本一致（若毛利不增=faster 只是更噪，机制premise假）。
  F4 周期不恶化：V_fast PRIMARY 验证周期 ≤ V0，双样本一致（加法陷阱不发生）。
  全过 = 更早入场(EMA10/30)有效，B2_4h 可考虑切换/并入。任一不过 = 无优势，B2_4h 维持 EMA20/100。

ENGINES: research_trend_baseline(tb)/validation(tv)/validation_r2(r2)/dualcycle(dc) +
  binance_funding imported VERBATIM, zero modification. signal_emax(b,fast,slow) 已支持任意 EMA
  参数；V0 复现冻结数字校验引擎一致性。OKX DB read-only；污染库不触碰。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_trend_baseline as tb
import research_trend_validation as tv
import research_trend_validation_r2 as r2
import research_trend_dualcycle as dc
from binance_funding import load_funding_binance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "trend_faster_entry_20260616"
SEED = 20260611
TF = "4h"
PAIRS = {"V0": (20, 100), "V_fast": (10, 30)}   # frozen; ZERO other pairs
WHIP_BARS = 2                                     # hold <= 2 bars (8h) = whipsaw

V0_EXPECT = {"OKX": 68194.8186, "Binance": 300752.7847}   # frozen B2_4h
LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── version spans (outer layer; just the EMA-pair swap) ──────────────────────
def version_spans(version: str, bars: dict) -> dict:
    fast, slow = PAIRS[version]
    return {name: tb.positions_flip(tb.signal_emax(bars[(name, TF)], fast, slow))
            for name in tb.SYMBOLS}


# ── metrics helpers (identical to funding/ADX studies) ───────────────────────
def to_daily(perbar: pd.Series) -> pd.Series:
    return perbar.groupby(perbar.index.ceil("D")).sum()


def year_net(trades: list[dict]) -> dict:
    df = pd.DataFrame(trades)
    ey = pd.to_datetime(df["entry_time"]).dt.year
    out = {}
    for y in sorted(ey.unique()):
        g = df[ey == y]
        w = g[g["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
        lo = abs(g[g["net_pnl_usd"] < 0]["net_pnl_usd"].sum())
        out[str(int(y))] = {"n": int(len(g)), "net": float(g["net_pnl_usd"].sum()),
                            "pf_net": (float(w / lo) if lo > 0 else float("inf"))}
    return out


def sharpe_horizon(perbar: pd.Series, trades: list[dict]) -> dict:
    daily = to_daily(perbar)
    span_days = max((perbar.index[-1] - perbar.index[0]).days, 1)
    years = span_days / 365.25
    dmu, dsd = float(daily.mean()), float(daily.std(ddof=1))
    sr_d = dmu / dsd if dsd > 0 else 0.0
    sr_ann = sr_d * math.sqrt(365.0)
    horizon_primary = (1.96 / sr_ann) ** 2 if sr_ann > 0 else float("inf")
    x = np.array([t["net_pnl_usd"] for t in trades], dtype=float)
    tmu = float(x.mean()); tsd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    n = len(x); tpy = n / years if years > 0 else 0.0
    sr_t_ann = (tmu / tsd) * math.sqrt(tpy) if tsd > 0 and tpy > 0 else 0.0
    horizon_secondary = (1.96 / sr_t_ann) ** 2 if sr_t_ann > 0 else float("inf")
    eq = perbar.cumsum(); maxdd = float((eq.cummax() - eq).max())
    return {"years_span": years, "daily_mean": dmu, "daily_std": dsd,
            "sharpe_daily": sr_d, "sharpe_annual_daily": sr_ann,
            "verif_horizon_years_PRIMARY": horizon_primary,
            "trade_mean": tmu, "trade_std": tsd, "trades_per_year": tpy,
            "sharpe_annual_pertrade": sr_t_ann,
            "verif_horizon_years_secondary": horizon_secondary,
            "m2m_maxdd_usd": maxdd}


def whipsaw_stats(trades: list[dict]) -> dict:
    h = np.array([t["hold_hours"] for t in trades], dtype=float)
    whip = [t for t in trades if t["hold_hours"] <= WHIP_BARS * 4]
    nwh = len(whip)
    return {"median_hold_h": float(np.median(h)), "mean_hold_h": float(h.mean()),
            "n_whip_le_8h": nwh, "pct_whip": round(100.0 * nwh / len(trades), 1),
            "net_whip_usd": round(float(sum(t["net_pnl_usd"] for t in whip)), 1),
            "winrate_whip_pct": round(100.0 * sum(1 for t in whip if t["net_pnl_usd"] > 0) / nwh, 1) if nwh else 0.0}


def evaluate(version: str, bars: dict, fund: dict) -> dict:
    spans_by = version_spans(version, bars)
    trades = []
    for name, (_, inst) in tb.SYMBOLS.items():
        trades.extend(tb.build_trades(name, inst, bars[(name, TF)], fund[name], spans_by[name]))
    m = tb.metrics(trades)
    perbar, _ = r2.m2m_pnl(TF, bars, fund, spans_by)
    sh = sharpe_horizon(perbar, trades)
    boot = tv.diag_d3_bootstrap(trades, np.random.default_rng(SEED))
    ls = {}
    for side in ("long", "short"):
        g = [t for t in trades if t["side"] == side]
        ls[side] = {"n": len(g), "net": float(sum(t["net_pnl_usd"] for t in g))}
    return {"version": version, "ema": list(PAIRS[version]), "n": m["n"],
            "gross_pnl": m["gross_pnl"], "fees_total": m["fees_total"],
            "funding_total": m["funding_total"], "net_pnl": m["net_pnl"], "pf_net": m["pf_net"],
            "win_rate_pct": m["win_rate_pct"], "avg_hold_hours": m["avg_hold_hours"],
            "whipsaw": whipsaw_stats(trades), "long_short": ls, "by_year": year_net(trades),
            "bootstrap": boot, **sh, "m2m_net_check": float(perbar.sum())}


# ── (3b) early-capture of big trends ─────────────────────────────────────────
def vfast_exposure(b: pd.DataFrame) -> np.ndarray:
    raw = tb.signal_emax(b, *PAIRS["V_fast"])
    exp = np.zeros(len(raw)); cur = 0
    for i in range(len(raw)):
        if not np.isnan(raw[i]) and raw[i] != 0:
            cur = int(raw[i])
        exp[i] = cur
    return exp


def early_capture(bars: dict, fund: dict) -> dict:
    """Of V0 top-10% net winners, how often was V_fast already in same-dir position at
    V0's entry bar, and how many hours earlier did V_fast enter?"""
    rows, vexp = [], {}
    for name, (_, inst) in tb.SYMBOLS.items():
        b = bars[(name, TF)]
        sp0 = tb.positions_flip(tb.signal_emax(b, *PAIRS["V0"]))
        tr0 = tb.build_trades(name, inst, b, fund[name], sp0)
        for (ei, xi, side, _), t in zip(sp0, tr0):
            rows.append({"sym": name, "ei": ei, "side": side, "net": t["net_pnl_usd"]})
        vexp[name] = vfast_exposure(b)
    df = pd.DataFrame(rows)
    k = max(1, int(round(0.10 * len(df))))
    top = df.nlargest(k, "net")
    earlier, leads = 0, []
    for _, r in top.iterrows():
        exp = vexp[r["sym"]]; ei0 = int(r["ei"]); side = int(r["side"])
        if exp[ei0] == side:
            j = ei0
            while j > 0 and exp[j - 1] == side:
                j -= 1
            leads.append((ei0 - j) * 4)   # hours (4h bars)
            earlier += 1
    return {"n_bigwinners": k, "n_vfast_earlier_samedir": earlier,
            "pct_vfast_earlier": round(100.0 * earlier / k, 1),
            "median_lead_hours": float(np.median(leads)) if leads else 0.0,
            "mean_lead_hours": round(float(np.mean(leads)), 1) if leads else 0.0,
            "note": "V_fast already holding same direction at V0 big-winner entry => entered earlier"}


# ── data loading (verbatim) ──────────────────────────────────────────────────
def load_okx():
    m1, bars, fund = {}, {}, {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1[name] = tb.load_1m_utc(db_sym)
        bars[(name, TF)] = tb.aggregate(m1[name], TF)
        fund[name] = tb.load_funding(inst, m1[name])
    return bars, fund


def load_binance():
    m1, bars, fund = {}, {}, {}
    for name, bs in dc.B_SYM.items():
        m1[name] = dc.load_1m_bv(bs)
        bars[(name, TF)] = tb.aggregate(m1[name], TF)
        fund[name] = load_funding_binance(bs, m1[name])
    return bars, fund


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("STUDY: faster-entry single-point EMA10/30 vs B2_4h EMA20/100 (known target, ZERO search)")
    L("DATA: OKX database_mainnet.db (mode=ro) + Binance data/binance_vision/")
    L("engines tb/tv/r2/dc + binance_funding imported verbatim; pollution DB not touched")
    OUT.mkdir(parents=True, exist_ok=True)

    samples = {}
    L("\n== loading OKX (2023-2026) ==")
    samples["OKX"] = load_okx()
    L("== loading Binance (2020-2026) ==")
    samples["Binance"] = load_binance()
    for s, (bars, _) in samples.items():
        L(f"  {s}: 4h bars per sym " + ", ".join(f"{n}={len(bars[(n,TF)])}" for n in tb.SYMBOLS))

    report = {"samples": {}, "judgment": {}}
    evals = {}
    for s, (bars, fund) in samples.items():
        L(f"\n========== SAMPLE: {s} ==========")
        v0 = evaluate("V0", bars, fund)
        vf = evaluate("V_fast", bars, fund)
        evals[s] = {"V0": v0, "V_fast": vf}
        exp = V0_EXPECT[s]
        ok = abs(v0["net_pnl"] - exp) < 1.0
        L(f"  V0 reproduction: net ${v0['net_pnl']:,.2f} vs expected ${exp:,.2f} "
          f"-> {'MATCH' if ok else 'MISMATCH!!'}")
        if not ok:
            L("  !! V0 does not reproduce frozen number -> engine wiring error, stopping")
            (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
            return 1
        for v in ("V0", "V_fast"):
            e = evals[s][v]
            L(f"  [{v} EMA{e['ema'][0]}/{e['ema'][1]}] n={e['n']} | net ${e['net_pnl']:,.0f} "
              f"(gross ${e['gross_pnl']:,.0f}) | PFnet {e['pf_net']:.2f} | "
              f"SR_ann(daily) {e['sharpe_annual_daily']:.3f} | horizon {e['verif_horizon_years_PRIMARY']:.1f}y | "
              f"M2M maxDD ${e['m2m_maxdd_usd']:,.0f} | CI [{e['bootstrap']['ci95'][0]:.1f},{e['bootstrap']['ci95'][1]:.1f}] | "
              f"whip {e['whipsaw']['pct_whip']}% net ${e['whipsaw']['net_whip_usd']:,.0f}")
        # ledger
        n_add = vf["n"] - v0["n"]
        ledger = {"v0_n": v0["n"], "vfast_n": vf["n"], "n_added_abs": n_add,
                  "n_added_pct": round(100.0 * n_add / v0["n"], 1),
                  "gross_v0": v0["gross_pnl"], "gross_vfast": vf["gross_pnl"],
                  "gross_vfast_gt_v0": vf["gross_pnl"] > v0["gross_pnl"],
                  "net_v0": v0["net_pnl"], "net_vfast": vf["net_pnl"],
                  "net_vfast_gt_v0": vf["net_pnl"] > v0["net_pnl"],
                  "cost_wall": (vf["gross_pnl"] > v0["gross_pnl"] and vf["net_pnl"] <= v0["net_pnl"]),
                  "horizon_v0_primary": v0["verif_horizon_years_PRIMARY"],
                  "horizon_vfast_primary": vf["verif_horizon_years_PRIMARY"],
                  "horizon_not_worse": vf["verif_horizon_years_PRIMARY"] <= v0["verif_horizon_years_PRIMARY"]}
        ec = early_capture(bars, fund)
        L(f"  ledger: n {v0['n']}->{vf['n']} (+{ledger['n_added_pct']}%) | "
          f"gross ${v0['gross_pnl']:,.0f}->${vf['gross_pnl']:,.0f} ({'UP' if ledger['gross_vfast_gt_v0'] else 'DOWN'}) | "
          f"net ${v0['net_pnl']:,.0f}->${vf['net_pnl']:,.0f} ({'UP' if ledger['net_vfast_gt_v0'] else 'DOWN'}) | "
          f"cost_wall={ledger['cost_wall']}")
        L(f"  (3b) early-capture: V_fast earlier same-dir at {ec['n_vfast_earlier_samedir']}/{ec['n_bigwinners']} "
          f"({ec['pct_vfast_earlier']}%) of V0 big winners | median lead {ec['median_lead_hours']}h")
        report["samples"][s] = {"V0": v0, "V_fast": vf, "ledger": ledger, "early_capture": ec}
        (OUT / f"{s}_V0.json").write_text(json.dumps(v0, indent=2, default=float))
        (OUT / f"{s}_Vfast.json").write_text(json.dumps(vf, indent=2, default=float))
        (OUT / f"{s}_ledger.json").write_text(json.dumps(ledger, indent=2, default=float))
        (OUT / f"{s}_early_capture.json").write_text(json.dumps(ec, indent=2, default=float))

    # ── pre-registered judgment F1-F4 (dual-sample) ──────────────────────────
    def f1_one(s):
        return evals[s]["V_fast"]["sharpe_annual_daily"] > evals[s]["V0"]["sharpe_annual_daily"]
    f1 = f1_one("OKX") and f1_one("Binance")

    def f2_one(s):
        return evals[s]["V_fast"]["net_pnl"] > evals[s]["V0"]["net_pnl"]
    f2 = f2_one("OKX") and f2_one("Binance")

    def f3_one(s):
        led = report["samples"][s]["ledger"]; ec = report["samples"][s]["early_capture"]
        return led["gross_vfast_gt_v0"] and ec["pct_vfast_earlier"] >= 50.0
    f3 = f3_one("OKX") and f3_one("Binance")

    def f4_one(s):
        return report["samples"][s]["ledger"]["horizon_not_worse"]
    f4 = f4_one("OKX") and f4_one("Binance")

    verdict_pass = f1 and f2 and f3 and f4
    judgment = {
        "F1_improvement_real": {"OKX": f1_one("OKX"), "Binance": f1_one("Binance"), "pass": f1},
        "F2_cost_survivable": {"OKX": f2_one("OKX"), "Binance": f2_one("Binance"), "pass": f2},
        "F3_mechanism_holds": {"OKX": f3_one("OKX"), "Binance": f3_one("Binance"), "pass": f3},
        "F4_horizon_not_worse": {"OKX": f4_one("OKX"), "Binance": f4_one("Binance"), "pass": f4},
        "ALL_PASS": verdict_pass,
        "conclusion": ("更早入场(EMA10/30)有效 -> 可考虑切换/并入" if verdict_pass
                       else "更早入场(EMA10/30)无优势 -> B2_4h 维持原状(EMA20/100)"),
    }
    report["judgment"] = judgment
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=float))

    L("\n========== JUDGMENT (pre-registered) ==========")
    for k in ("F1_improvement_real", "F2_cost_survivable", "F3_mechanism_holds", "F4_horizon_not_worse"):
        j = judgment[k]
        L(f"  {k}: OKX={j['OKX']} Binance={j['Binance']} -> {'PASS' if j['pass'] else 'FAIL'}")
    L(f"  ALL_PASS: {verdict_pass} -> {judgment['conclusion']}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
