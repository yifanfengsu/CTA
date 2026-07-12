#!/usr/bin/env python3
"""B2_4h ADX(14)>25 trend-strength entry filter (HIGHEST overfit-defense level).

POSITIONING (verbatim into report header, never deleted/edited):
  本研究不是独立策略前置研究，而是对已知资产 B2_4h（4h EMA20/100 交叉、always-in-market、
  信号反向出场，Sharpe~0.5、验证 18 月+、资源关闭）的增强尝试。靶子的全部历史表现已知——
  这使本研究处于最高过拟合风险下。直接前车之鉴：funding 同向确认
  (reports/trend_funding_confirm_20260613/) 曾在 OKX 上让每个指标变好（Sharpe 0.65→0.99），
  但 Binance 6 年全部翻车（0.94→0.74、砍 1/3 利润、周期反延长）——"OKX 单边改善正是已知靶子
  上的描线产物"。本研究用与 funding 确认完全相同的防线检验 ADX 过滤：① 单一标准指标、文献
  标准阈值、零参数搜索；② OKX 定义、Binance 6 年确认，双样本不一致即判死；③ 机制验证独立
  于总 Sharpe；④ 笔数调整后的验证周期记账（减法陷阱）。

MECHANISM HYPOTHESIS (verbatim into header, NEUTRAL prior):
  EMA 交叉在震荡市发假信号（whipsaw）、在趋势市发真信号。ADX 衡量趋势强度，只在 ADX>25
  （趋势确立）时接受 B2_4h 入场、过滤弱趋势/震荡信号，可能提升 Sharpe。先验：机制合理但
  未经检验，且存在反向风险——大趋势启动初期 ADX 往往仍低，ADX 过滤可能恰好砍掉大趋势的
  早期入场（利润最肥的部分）。

═══ PART 1 — ADX & FILTER RULE (pre-registered, ZERO variants ZERO search) ═══
  ADX 计算：标准 Wilder 定义，周期 = 14（4h bar 上），+DI/−DI/DX/ADX 经典 Wilder 平滑。
    生产用 talib.ADX(high,low,close,14)（canonical Wilder）；自检：与独立手写 Wilder 重实现
    比对（附录），+ 手算公式校验。绝对不测其他周期。
  唯一过滤规则（无可调旋钮）：B2_4h 在每个 EMA 金/死叉处给出入场信号（入场 bar = ei，执行于
    ei 的收盘，与 EMA 信号同一信息集，零 look-ahead）。V+：仅当 ADX(14)[ei] > 25 才入场；
    ADX ≤ 25（或 warm-up NaN）则放弃该信号、保持空仓，直到下一次 EMA 翻转再判。
    出场不变（仍信号反向出场，不加 ADX 出场逻辑——只测入场过滤一个变量）。
    阈值 = 25（教科书：<20 无趋势 / 20-25 弱 / >25 有趋势）。绝对禁止测 20/30 或任何其他
    阈值/周期。ADX>25 无效即结论"ADX 趋势过滤无用"，不换参数再试。
    => V+ 交易集合是 V0 的真子集（ADX 只能否决入场，不新增）。ADX 方向无关（只过滤强度）。

═══ PART 2 — DUAL-SAMPLE EVALUATION (Binance 是真检验) ═══
  OKX (2023-2026 定义样本) 与 Binance (2020-2026 确认样本) 各跑 V0/V+，同口径（taker ±1tick +
  真实 funding，与 B2_4h 原研究一致）。逐版本：净利、净 Sharpe（日 M2M 年化）、笔数、净 PF、
  maxDD(M2M)、net/maxDD、分年度、bootstrap 95% CI (10,000 次 seed=20260611)。OKX 改善不足为信；
  Binance 6 年 V+ 同样改善且方向一致才可信。

═══ PART 3 — MECHANISM VALIDATION (被 ADX 过滤的信号是否真更差) ═══
  双样本各做（trade-level 主检验 + bar-level 辅检验）：
   (3a) 把 B2_4h 原始信号按入场 bar 的 ADX 分两组：强趋势(ADX>25,保留) vs 弱趋势(ADX≤25,过滤)。
   (3b) 机制为真 → 弱趋势组实际收益显著 < 强趋势组。报告两组 mean/median、差值、Welch t、双样本一致。
   (3c) 弱趋势组不更差(甚至更好,大趋势早期低 ADX) → V+ 任何 Sharpe 改善是偶然切割非机制，判死。
   (3d) 专项：被过滤信号里有多少是全期净 PnL top-10% 大赢家？ADX 砍掉大赢家早期入场 = 趋势
        策略加趋势过滤的反讽，量化之（双样本）。

═══ PART 4 — SUBTRACTION LEDGER (防减法陷阱) ═══
  V+ vs V0 笔数变化；PRIMARY 验证周期 = 日 M2M 年化 Sharpe 的 (1.96/SR_ann)²（频率/事件密度
  调整后：过滤若只减事件则日均降、SR_ann 降、周期延长；只有砍的确是坏信号 SR_ann 才升周期才缩）。
  SECONDARY = per-trade 年化 Sharpe 的 (1.96/SR)²（naive，暴露陷阱）。核心量：V+ 是否缩短
  B2_4h 前向验证周期（PRIMARY）。Sharpe 数字本身不是目的。

═══ PART 5 — JUDGMENT (pre-registered, immutable; identical to funding-confirm) ═══
  E1 机制成立：弱趋势组(过滤)收益显著 < 强趋势组(3b, trade-level)，双样本一致。
  E2 改善真实：V+ 净 Sharpe(日年化) > V0，双样本一致（OKX ∧ Binance 同向；仅 OKX = 描线判死）。
  E3 周期缩短：V+ PRIMARY 验证周期 < V0（减法陷阱不发生）。
  E4 无新害：V+ M2M maxDD ≤ 1.10×V0 且 bootstrap CI 下界不更负，双样本。
  全过 = ADX 过滤有效, B2_4h → B2_4h+ADX。任一不过 = 无效，B2_4h 维持原状（资源关闭、18 月不变）。

ENGINES: research_trend_baseline(tb)/validation(tv)/validation_r2(r2)/dualcycle(dc) +
  binance_funding imported VERBATIM, zero modification. V0 = canonical tb spans 复现既有冻结
  数字校验引擎一致性；V+ 仅外层用 ADX 过滤后信号。OKX DB read-only；污染库不触碰。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import talib

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
import research_trend_validation_r2 as r2
import research_trend_dualcycle as dc
from binance_funding import load_funding_binance

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/trend_b2_4h/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "trend_adx_filter_20260616"
SEED = 20260611
FAST, SLOW, TF = 20, 100, "4h"      # B2_4h, frozen
ADX_PERIOD = 14                     # Wilder standard, NOT tunable
ADX_THRESH = 25.0                   # textbook trend threshold, NOT tunable

# expected V0 reproduction targets (frozen prior runs; V0 = identical B2_4h)
V0_EXPECT = {"OKX": 68194.8186, "Binance": 300752.7847}

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── ADX (production = talib canonical Wilder) ────────────────────────────────
def adx_per_bar(b: pd.DataFrame) -> np.ndarray:
    """Wilder ADX(14) aligned to each bar; NaN during warm-up (~first 27 bars)."""
    return talib.ADX(b["high"].to_numpy(dtype=float),
                     b["low"].to_numpy(dtype=float),
                     b["close"].to_numpy(dtype=float),
                     timeperiod=ADX_PERIOD)


def wilder_adx_ref(high, low, close, period=14) -> np.ndarray:
    """Independent hand-rolled Wilder ADX (self-check reference only, NOT production)."""
    n = len(high)
    tr = np.full(n, np.nan); pdm = np.full(n, np.nan); mdm = np.full(n, np.nan)
    for i in range(1, n):
        up = high[i] - high[i - 1]; dn = low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan); sp = np.full(n, np.nan); sm = np.full(n, np.nan)
    atr[period] = np.nansum(tr[1:period + 1])
    sp[period] = np.nansum(pdm[1:period + 1]); sm[period] = np.nansum(mdm[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
        sp[i] = sp[i - 1] - sp[i - 1] / period + pdm[i]
        sm[i] = sm[i - 1] - sm[i - 1] / period + mdm[i]
    pdi = 100.0 * sp / atr; mdi = 100.0 * sm / atr
    dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = np.full(n, np.nan); first = 2 * period - 1
    adx[first] = np.nanmean(dx[period:first + 1])
    for i in range(first + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def adx_confirms(adx_at_entry: float) -> bool:
    """ADX strength confirmation (direction-agnostic). NaN warm-up -> False -> filtered."""
    return bool(adx_at_entry > ADX_THRESH)   # np.nan > x is False


# ── version spans (outer layer only) ─────────────────────────────────────────
def version_spans(version: str, bars: dict, adx_by: dict) -> dict:
    """V0 = canonical B2_4h flip spans. V+ = same spans minus legs whose entry-bar
    ADX<=25 (subset of V0; ADX only vetoes entries)."""
    spans_by = {}
    for name in tb.SYMBOLS:
        b = bars[(name, TF)]
        spans = tb.positions_flip(tb.signal_emax(b, FAST, SLOW))
        if version == "V0":
            spans_by[name] = spans
        else:
            adx = adx_by[name]
            spans_by[name] = [s for s in spans if adx_confirms(adx[s[0]])]
    return spans_by


# ── metrics helpers (identical to funding-confirm) ───────────────────────────
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


def evaluate(version: str, bars: dict, fund: dict, adx_by: dict) -> dict:
    spans_by = version_spans(version, bars, adx_by)
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
    return {"version": version, "n": m["n"], "gross_pnl": m["gross_pnl"],
            "fees_total": m["fees_total"], "funding_total": m["funding_total"],
            "net_pnl": m["net_pnl"], "pf_net": m["pf_net"],
            "win_rate_pct": m["win_rate_pct"], "avg_hold_hours": m["avg_hold_hours"],
            "long_short": ls, "by_year": year_net(trades),
            "bootstrap": boot, **sh, "m2m_net_check": float(perbar.sum())}


# ── mechanism validation ─────────────────────────────────────────────────────
def normal_two_sided_p(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2.0))


def welch(a: list[float], b: list[float]) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"na": na, "nb": nb, "insufficient": True}
    ma, mb = a.mean(), b.mean(); va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else 0.0
    df = ((va / na + vb / nb) ** 2 /
          ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))) if se > 0 else 0.0
    return {"na": na, "nb": nb, "mean_a": float(ma), "mean_b": float(mb),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "diff_a_minus_b": float(ma - mb), "welch_t": float(t),
            "welch_df": float(df), "p_two_sided_normal_approx": float(normal_two_sided_p(t))}


def mechanism(bars: dict, fund: dict, adx_by: dict) -> dict:
    """Classify ORIGINAL B2_4h signals by ADX>25 at entry; strong(kept,a) vs weak(veto,b).
    trade-level (primary) + bar-level (secondary, autocorr caveat)."""
    tr_strong_net, tr_weak_net = [], []
    tr_strong_ret, tr_weak_ret = [], []
    bar_strong, bar_weak = [], []
    for name, (_, inst) in tb.SYMBOLS.items():
        b = bars[(name, TF)]; c = b["close"].to_numpy(); adx = adx_by[name]
        spans = tb.positions_flip(tb.signal_emax(b, FAST, SLOW))
        trades = tb.build_trades(name, inst, b, fund[name], spans)
        for (ei, xi, side, _), t in zip(spans, trades):
            strong = adx_confirms(adx[ei])
            ret = t["gross_pnl_usd"] / tb.NOTIONAL * 100.0
            (tr_strong_net if strong else tr_weak_net).append(t["net_pnl_usd"])
            (tr_strong_ret if strong else tr_weak_ret).append(ret)
        # bar-level: every bar with nonzero original exposure -> next-bar signed return
        raw = tb.signal_emax(b, FAST, SLOW)
        target = np.zeros(len(raw)); cur = 0
        for i in range(len(raw)):
            if not np.isnan(raw[i]) and raw[i] != 0:
                cur = int(raw[i])
            target[i] = cur
        for i in range(len(target) - 1):
            if target[i] == 0:
                continue
            strong = adx_confirms(adx[i])
            rr = float(target[i]) * (c[i + 1] / c[i] - 1.0) * 100.0
            (bar_strong if strong else bar_weak).append(rr)
    return {
        "trade_level_net_usd": {"agree": welch(tr_strong_net, tr_weak_net),
                                "note": "a=strong(ADX>25,kept) b=weak(ADX<=25,filtered); "
                                        "diff>0 & sig => weak signals worse (mechanism holds)"},
        "trade_level_gross_ret_pct": {"agree": welch(tr_strong_ret, tr_weak_ret)},
        "bar_level_fwd_ret_pct": {"agree": welch(bar_strong, bar_weak),
                                  "caveat": "overlapping holds -> autocorrelated -> t optimistic; secondary only"},
    }


def big_winner_analysis(bars: dict, fund: dict, adx_by: dict) -> dict:
    """(3d) Of the full-period top-10% net-PnL trades, how many / how much net are
    FILTERED OUT by ADX<=25 (trend-strategy + trend-filter irony)."""
    rows = []
    for name, (_, inst) in tb.SYMBOLS.items():
        b = bars[(name, TF)]; adx = adx_by[name]
        spans = tb.positions_flip(tb.signal_emax(b, FAST, SLOW))
        trades = tb.build_trades(name, inst, b, fund[name], spans)
        for (ei, xi, side, _), t in zip(spans, trades):
            rows.append({"net": t["net_pnl_usd"], "kept": adx_confirms(adx[ei])})
    df = pd.DataFrame(rows)
    n = len(df); k = max(1, int(round(0.10 * n)))
    top = df.nlargest(k, "net")
    tot = float(top["net"].sum())
    filt = top[~top["kept"]]
    return {"n_trades": n, "top10pct_k": k,
            "n_big_winners_filtered": int(len(filt)),
            "pct_big_winners_filtered": round(100.0 * len(filt) / k, 1),
            "net_in_all_big_winners": round(tot, 1),
            "net_in_filtered_big_winners": round(float(filt["net"].sum()), 1),
            "pct_bigwinner_net_filtered": round(100.0 * filt["net"].sum() / tot, 1) if tot != 0 else 0.0}


def adx_selfcheck(bars: dict) -> dict:
    """Validate production talib.ADX against independent hand-rolled Wilder reimpl,
    on real BTC 4h bars; + manual textbook spot-check on a tiny known series."""
    b = bars[("BTC", TF)]
    h, lo, c = b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float)
    prod = talib.ADX(h, lo, c, ADX_PERIOD)
    ref = wilder_adx_ref(h, lo, c, ADX_PERIOD)
    both = ~np.isnan(prod) & ~np.isnan(ref)
    diff = np.abs(prod[both] - ref[both])
    # convergence: diff in the steady region (>=70 bars past first valid)
    fv = int(np.argmax(both))
    steady = both.copy(); steady[:fv + 70] = False
    sdiff = np.abs(prod[steady] - ref[steady]) if steady.any() else np.array([0.0])
    return {"bars_compared": int(both.sum()),
            "first_valid_idx": fv,
            "max_abs_diff_all": float(diff.max()),
            "mean_abs_diff_all": float(diff.mean()),
            "max_abs_diff_steady_state": float(sdiff.max()),
            "note": "production=talib.ADX (canonical Wilder); ref=independent reimpl; "
                    "residual is seed-convergence only (decays 13/14 per bar), ~0 at all real "
                    "entry bars (>=bar 100 after EMA-slow warm-up)",
            "sample_prod_idx_200_205": [round(float(x), 4) for x in prod[200:206]],
            "sample_ref_idx_200_205": [round(float(x), 4) for x in ref[200:206]]}


# ── data loading (verbatim from funding-confirm) ─────────────────────────────
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
    L("STUDY: B2_4h ADX(14)>25 trend-strength entry filter (highest overfit-defense; known target)")
    L("RULE: single textbook-threshold ADX filter, period=14, thresh=25, ZERO parameter search")
    L("DATA: OKX database_mainnet.db (mode=ro) + Binance data/binance_vision/")
    L("engines tb/tv/r2/dc + binance_funding imported verbatim; pollution DB not touched")
    OUT.mkdir(parents=True, exist_ok=True)

    samples = {}
    L("\n== loading OKX (2023-2026) ==")
    samples["OKX"] = load_okx()
    L("== loading Binance (2020-2026) ==")
    samples["Binance"] = load_binance()
    # precompute ADX per sample/symbol
    adx_by_sample = {}
    for s, (bars, _) in samples.items():
        adx_by_sample[s] = {n: adx_per_bar(bars[(n, TF)]) for n in tb.SYMBOLS}
        L(f"  {s}: 4h bars per sym " + ", ".join(f"{n}={len(bars[(n,TF)])}" for n in tb.SYMBOLS))

    # ADX self-check (OKX BTC bars)
    selfchk = adx_selfcheck(samples["OKX"][0])
    (OUT / "adx_selfcheck.json").write_text(json.dumps(selfchk, indent=2, default=float))
    L(f"  ADX self-check: max|talib-ref| all={selfchk['max_abs_diff_all']:.4f}, "
      f"steady={selfchk['max_abs_diff_steady_state']:.6f} (bars {selfchk['bars_compared']})")

    report = {"adx_selfcheck": selfchk, "samples": {}, "judgment": {}}
    evals = {}
    for s, (bars, fund) in samples.items():
        adx_by = adx_by_sample[s]
        L(f"\n========== SAMPLE: {s} ==========")
        v0 = evaluate("V0", bars, fund, adx_by)
        vp = evaluate("V+", bars, fund, adx_by)
        evals[s] = {"V0": v0, "V+": vp}
        exp = V0_EXPECT[s]
        ok = abs(v0["net_pnl"] - exp) < 1.0
        L(f"  V0 reproduction: net ${v0['net_pnl']:,.2f} vs expected ${exp:,.2f} "
          f"-> {'MATCH' if ok else 'MISMATCH!!'}")
        if not ok:
            L("  !! V0 does not reproduce frozen number -> engine wiring error, stopping")
            (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
            return 1
        for v in ("V0", "V+"):
            e = evals[s][v]
            L(f"  [{v}] n={e['n']} | net ${e['net_pnl']:,.0f} (gross ${e['gross_pnl']:,.0f}) "
              f"| PFnet {e['pf_net']:.2f} | SR_ann(daily) {e['sharpe_annual_daily']:.3f} "
              f"| horizon {e['verif_horizon_years_PRIMARY']:.1f}y | M2M maxDD ${e['m2m_maxdd_usd']:,.0f} "
              f"| CI [{e['bootstrap']['ci95'][0]:.1f},{e['bootstrap']['ci95'][1]:.1f}]")
        n_drop = v0["n"] - vp["n"]
        ledger = {"v0_n": v0["n"], "vp_n": vp["n"],
                  "n_filtered_abs": n_drop, "n_filtered_pct": 100.0 * n_drop / v0["n"],
                  "horizon_v0_primary": v0["verif_horizon_years_PRIMARY"],
                  "horizon_vp_primary": vp["verif_horizon_years_PRIMARY"],
                  "horizon_shorter_PRIMARY": vp["verif_horizon_years_PRIMARY"] < v0["verif_horizon_years_PRIMARY"],
                  "horizon_v0_secondary": v0["verif_horizon_years_secondary"],
                  "horizon_vp_secondary": vp["verif_horizon_years_secondary"],
                  "pertrade_sharpe_v0": v0["sharpe_annual_pertrade"],
                  "pertrade_sharpe_vp": vp["sharpe_annual_pertrade"],
                  "subtraction_trap": (vp["sharpe_annual_pertrade"] > v0["sharpe_annual_pertrade"]
                                       and vp["sharpe_annual_daily"] <= v0["sharpe_annual_daily"])}
        L(f"  ledger: filtered {n_drop}/{v0['n']} ({ledger['n_filtered_pct']:.1f}%) | "
          f"horizon {v0['verif_horizon_years_PRIMARY']:.1f}y -> {vp['verif_horizon_years_PRIMARY']:.1f}y "
          f"({'SHORTER' if ledger['horizon_shorter_PRIMARY'] else 'LONGER/SAME'})")
        mech = mechanism(bars, fund, adx_by)
        bw = big_winner_analysis(bars, fund, adx_by)
        wt = mech["trade_level_net_usd"]["agree"]
        agree_sum = wt["mean_a"] * wt["na"]
        cons_ok = abs(agree_sum - vp["net_pnl"]) < 1.0 and wt["na"] == vp["n"]
        L(f"  consistency V+==strong-group: V+ net ${vp['net_pnl']:,.0f} (n{vp['n']}) "
          f"vs strong ${agree_sum:,.0f} (n{wt['na']}) -> {'OK' if cons_ok else 'MISMATCH!!'}")
        L(f"  mechanism (trade-level net): strong mean ${wt['mean_a']:,.0f} (n{wt['na']}) vs "
          f"weak mean ${wt['mean_b']:,.0f} (n{wt['nb']}) | diff ${wt['diff_a_minus_b']:,.0f} | "
          f"Welch t {wt['welch_t']:.2f} p {wt['p_two_sided_normal_approx']:.3f}")
        L(f"  (3d) big-winners filtered: {bw['n_big_winners_filtered']}/{bw['top10pct_k']} "
          f"({bw['pct_big_winners_filtered']}%) | net cut ${bw['net_in_filtered_big_winners']:,.0f}/"
          f"${bw['net_in_all_big_winners']:,.0f} ({bw['pct_bigwinner_net_filtered']}%)")
        report["samples"][s] = {"V0": v0, "V+": vp, "ledger": ledger,
                                "mechanism": mech, "big_winners": bw}
        (OUT / f"{s}_V0.json").write_text(json.dumps(v0, indent=2, default=float))
        (OUT / f"{s}_Vplus.json").write_text(json.dumps(vp, indent=2, default=float))
        (OUT / f"{s}_mechanism.json").write_text(json.dumps(mech, indent=2, default=float))
        (OUT / f"{s}_ledger.json").write_text(json.dumps(ledger, indent=2, default=float))
        (OUT / f"{s}_big_winners.json").write_text(json.dumps(bw, indent=2, default=float))

    # ── pre-registered judgment E1-E4 (dual-sample consistency required) ──────
    def e1_one(s):
        w = report["samples"][s]["mechanism"]["trade_level_net_usd"]["agree"]
        return w["diff_a_minus_b"] > 0 and w["p_two_sided_normal_approx"] < 0.05
    e1 = e1_one("OKX") and e1_one("Binance")

    def e2_one(s):
        return evals[s]["V+"]["sharpe_annual_daily"] > evals[s]["V0"]["sharpe_annual_daily"]
    e2 = e2_one("OKX") and e2_one("Binance")

    def e3_one(s):
        return report["samples"][s]["ledger"]["horizon_shorter_PRIMARY"]
    e3 = e3_one("OKX") and e3_one("Binance")

    def e4_one(s):
        v0, vp = evals[s]["V0"], evals[s]["V+"]
        dd_ok = vp["m2m_maxdd_usd"] <= 1.10 * v0["m2m_maxdd_usd"]
        ci_ok = vp["bootstrap"]["ci95"][0] >= v0["bootstrap"]["ci95"][0]
        return dd_ok and ci_ok
    e4 = e4_one("OKX") and e4_one("Binance")

    verdict_pass = e1 and e2 and e3 and e4
    judgment = {
        "E1_mechanism_holds": {"OKX": e1_one("OKX"), "Binance": e1_one("Binance"), "pass": e1},
        "E2_improvement_real": {"OKX": e2_one("OKX"), "Binance": e2_one("Binance"), "pass": e2},
        "E3_horizon_shorter": {"OKX": e3_one("OKX"), "Binance": e3_one("Binance"), "pass": e3},
        "E4_no_new_harm": {"OKX": e4_one("OKX"), "Binance": e4_one("Binance"), "pass": e4},
        "ALL_PASS": verdict_pass,
        "conclusion": ("ADX 过滤有效 -> B2_4h+ADX" if verdict_pass
                       else "ADX 过滤无效 -> B2_4h 维持原状（资源关闭、18 月验证不变）"),
    }
    report["judgment"] = judgment
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=float))

    L("\n========== JUDGMENT (pre-registered) ==========")
    for k in ("E1_mechanism_holds", "E2_improvement_real", "E3_horizon_shorter", "E4_no_new_harm"):
        j = judgment[k]
        L(f"  {k}: OKX={j['OKX']} Binance={j['Binance']} -> {'PASS' if j['pass'] else 'FAIL'}")
    L(f"  ALL_PASS: {verdict_pass} -> {judgment['conclusion']}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
