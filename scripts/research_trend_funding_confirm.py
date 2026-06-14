#!/usr/bin/env python3
"""B2_4h funding same-direction confirmation enhancement (HIGHEST overfit-defense level).

POSITIONING (verbatim into report header, never deleted/edited):
  本研究不是独立策略前置研究，而是对已知资产 B2_4h（4h EMA20/100，趋势线唯一
  双周期幸存信号，Sharpe~0.5、验证 18 月+，资源决策关闭）的增强尝试。靶子的全部
  历史表现已知——这使本研究处于整个项目最高的过拟合风险下：任何"用 funding 过滤
  B2_4h"的调整都可能是对着已知答案描线。防线：① 仅测一个零阈值、有机制依据的规则，
  零参数搜索；② OKX 定义、Binance 6 年确认，双样本不一致即判死；③ 验证机制本身
  （被 funding 否决的信号是否实际更差），而非只看总 Sharpe；④ 同时报告笔数与笔数
  调整后的验证周期，Sharpe 升但笔数降导致周期不改善则判负贡献。

MECHANISM HYPOTHESIS (verbatim into header):
  B2_4h 做多（EMA 金叉）时，若 funding 同向为正（多头付费持有 = 资金认同该方向），
  趋势更可能延续；若做多但 funding 为负（市场为做空付费），该多头信号疑为逆势噪声，
  过滤之。funding 作为"市场资金是否认同方向"的确认。先验：机制合理但未经检验，中性。

═══ PART 1 — CONFIRMATION RULE (pre-registered, ZERO variants ZERO thresholds) ═══
  唯一规则 — 符号一致过滤（无可调旋钮）：B2_4h 在每个 EMA 金/死叉处给出一笔方向
    信号（入场 bar）。在该入场 bar 用 funding_recent 确认：
      原始 long  ∧ funding_recent ≥ 0 → 入场 long（保留该 leg）
      原始 long  ∧ funding_recent < 0 → 不入场（该 leg 改为 flat / 过滤）
      原始 short ∧ funding_recent ≤ 0 → 入场 short（保留该 leg）
      原始 short ∧ funding_recent > 0 → 不入场（该 leg 改为 flat / 过滤）
    被过滤的 leg 在其原 [入场,出场] 区间内空仓；出场时点（下一次 EMA 翻转）不变。
    => V+ 的交易集合是 V0 的子集（funding 只能否决入场，不能新增入场点）。
    funding_recent = 入场 bar 之前最近一次已结算 funding rate（单次值，不平均、不加窗）。
      实现：取 slot_min < 入场 bar 的 end_min 的最近一次结算 rate（严格早于决策时刻，零
      look-ahead）。若入场 bar 之前无任何已结算 funding（仅极早期 warm-up），确认失败 → 过滤。
    分界值 = 0（机制自然点：付费方向，非拟合点）。
  绝对禁止：任何非零阈值、任何 funding 平均窗口、任何 funding 强度分档。零阈值规则
    有效与否即结论；无效则结论为"funding 符号确认无用"，不得换参数再试。
  DEFINITION NOTE（披露的口径选择，非多版本搜索）：规则在"入场 bar"评估（entry-time
    veto），与原文"funding_recent = 入场 bar 之前最近一次"措辞及 PART 4 全程"减法/减笔数"
    语义一致。曾考虑"每个 4h bar 连续重新确认"的逐字面读法，但 funding 符号在 8h 尺度上
    高频翻转，会把单个趋势持仓切碎成 ~6× 的微交易（OKX 470→3057 笔、Binance 836→2913 笔），
    与"减法/减笔数"框架直接矛盾，故弃用。仍是同一条零阈值符号规则，无参数搜索。

═══ PART 2 — DUAL-SAMPLE EVALUATION (Binance 是真检验) ═══
  OKX (2023-2026, 定义样本) 与 Binance (2020-2026, 确认样本) 各跑 V0 / V+，同口径
  （taker ±1tick + 真实 funding 成本，与 B2_4h 原研究一致）。逐版本输出：净利、净
  Sharpe（日 M2M 年化）、笔数、净 PF、maxDD（M2M）、net/maxDD、分年度、bootstrap 95% CI
  (10,000 次 seed=20260611, per-trade net)。OKX 改善不足为信；Binance 6 年 V+ 是否同样
  改善且方向一致才是可信证据。

═══ PART 3 — MECHANISM VALIDATION (被否信号是否真的更差) ═══
  双样本各做（trade-level 主检验 + bar-level 辅检验）：
   (3a) 把 B2_4h 原始信号按 funding 符号分两组：同向组(funding 认同) vs 逆向组(被 V+ 过滤)。
   (3b) 若机制为真，逆向组实际收益应显著 < 同向组。报告两组 mean/median、差值、Welch t、
        双样本一致性。bar-level 收益自相关（持仓重叠）→ t 偏乐观，仅作辅证，主检验用 trade-level。
   (3c) 若逆向组收益不更差(甚至更好) → V+ 任何 Sharpe 改善是偶然切割，非机制，即使数字
        好看也判机制不成立。

═══ PART 4 — SUBTRACTION LEDGER (防减法陷阱) ═══
  报告 V+ 相对 V0 的笔数变化（过滤掉百分之几）；验证周期对比 V0 vs V+。
  PRIMARY 验证周期 = 日 M2M 年化 Sharpe 的 (1.96/Sharpe_ann)² 年——此度量本身就是
    笔数/事件密度调整后的：V+ 的 flat 日贡献 0 PnL，过滤若只是减事件则日均下降、Sharpe_ann
    降、周期延长；只有当被砍的确是坏信号时 Sharpe_ann 才升、周期才缩。
  SECONDARY = per-trade 年化 Sharpe 的 (1.96/SR)²——naive，用于显式暴露陷阱：per-trade
    Sharpe 升而频率调整 Sharpe 降即为减法陷阱。
  判定核心量：V+ 是否缩短 B2_4h 的前向验证周期（PRIMARY）。Sharpe 数字本身不是目的。

═══ PART 5 — JUDGMENT (pre-registered, immutable) ═══
  E1 机制成立：逆向组实际收益显著低于同向组(3b, trade-level)，双样本一致。
  E2 改善真实：V+ 净 Sharpe(日年化) > V0，双样本一致（OKX ∧ Binance 同向改善；仅 OKX = 描线判死）。
  E3 周期缩短：V+ PRIMARY 验证周期 < V0（减法陷阱不发生）。
  E4 无新害：V+ M2M maxDD 不显著恶化(≤ 1.10×V0)、bootstrap CI 不比 V0 更差(下界不更负)。
  全过 = funding 确认有效, B2_4h → B2_4h+（验证周期缩短，进入更新后前向观察决策）。
  任一不过 = funding 确认无效，B2_4h 维持原状（资源关闭、18 月验证不变）。

ENGINES: research_trend_baseline(tb)/validation(tv)/validation_r2(r2) + binance_funding +
  research_trend_dualcycle(dc, for load_1m_bv/B_SYM) imported VERBATIM, zero modification.
  V0 走 canonical tb.positions_flip(signal_emax) 复现既有数字；V+ 仅在外层用确认后信号
  + positions_from_target。OKX DB read-only；污染库不触碰。
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
OUT = PROJECT_ROOT / "reports" / "trend_funding_confirm_20260613"
SEED = 20260611
FAST, SLOW, TF = 20, 100, "4h"  # B2_4h, frozen

# expected V0 reproduction targets (from prior frozen runs)
V0_EXPECT = {"OKX": 68194.8186, "Binance": 300752.7847}

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── confirmation machinery (outer layer only) ────────────────────────────────
def target_orig_exposure(b: pd.DataFrame) -> np.ndarray:
    """B2_4h actual held exposure per bar = forward-filled nonzero EMA-cross sign
    (identical to what positions_flip trades; leading warm-up bars = flat)."""
    raw = tb.signal_emax(b, FAST, SLOW)
    out = np.zeros(len(raw))
    cur = 0
    for i in range(len(raw)):
        s = raw[i]
        if not np.isnan(s) and s != 0:
            cur = int(s)
        out[i] = cur
    return out


def funding_recent_per_bar(b: pd.DataFrame, fund_df: pd.DataFrame) -> np.ndarray:
    """Most recent SETTLED funding rate strictly before each bar's close moment.
    NaN if no settlement precedes the bar (very early warm-up only)."""
    endm = b["end_min"].to_numpy()
    fslot = fund_df["slot_min"].to_numpy()           # ascending (loaders sort)
    frate = fund_df["rate"].to_numpy()
    idx = np.searchsorted(fslot, endm, side="left") - 1   # last slot strictly < endm
    fr = np.where(idx >= 0, frate[np.clip(idx, 0, len(frate) - 1)], np.nan)
    return fr


def leg_agrees(side: int, f: float) -> bool:
    """Entry-time funding sign confirmation (NaN -> False -> filtered)."""
    return (side == 1 and f >= 0) or (side == -1 and f <= 0)


def version_spans(version: str, bars: dict, fund: dict) -> dict:
    """V0 = canonical B2_4h flip spans. V+ = same spans minus the legs whose entry
    funding sign disagrees (subset of V0; funding only vetoes entries)."""
    spans_by = {}
    for name in tb.SYMBOLS:
        b = bars[(name, TF)]
        spans = tb.positions_flip(tb.signal_emax(b, FAST, SLOW))
        if version == "V0":
            spans_by[name] = spans
        else:
            fr = funding_recent_per_bar(b, fund[name])
            spans_by[name] = [s for s in spans if leg_agrees(s[2], fr[s[0]])]
    return spans_by


# ── metrics helpers ──────────────────────────────────────────────────────────
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
    """Primary: daily M2M annualized Sharpe (frequency-aware). Secondary: per-trade."""
    daily = to_daily(perbar)
    span_days = max((perbar.index[-1] - perbar.index[0]).days, 1)
    years = span_days / 365.25
    dmu, dsd = float(daily.mean()), float(daily.std(ddof=1))
    sr_d = dmu / dsd if dsd > 0 else 0.0
    sr_ann = sr_d * math.sqrt(365.0)
    horizon_primary = (1.96 / sr_ann) ** 2 if sr_ann > 0 else float("inf")
    # secondary: per-trade
    x = np.array([t["net_pnl_usd"] for t in trades], dtype=float)
    tmu, tsd = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
    n = len(x)
    tpy = n / years if years > 0 else 0.0
    sr_t_ann = (tmu / tsd) * math.sqrt(tpy) if tsd > 0 and tpy > 0 else 0.0
    horizon_secondary = (1.96 / sr_t_ann) ** 2 if sr_t_ann > 0 else float("inf")
    eq = perbar.cumsum()
    maxdd = float((eq.cummax() - eq).max())
    return {"years_span": years, "daily_mean": dmu, "daily_std": dsd,
            "sharpe_daily": sr_d, "sharpe_annual_daily": sr_ann,
            "verif_horizon_years_PRIMARY": horizon_primary,
            "trade_mean": tmu, "trade_std": tsd, "trades_per_year": tpy,
            "sharpe_annual_pertrade": sr_t_ann,
            "verif_horizon_years_secondary": horizon_secondary,
            "m2m_maxdd_usd": maxdd}


def evaluate(version: str, bars: dict, fund: dict) -> dict:
    spans_by = version_spans(version, bars, fund)
    trades = []
    for name, (_, inst) in tb.SYMBOLS.items():
        trades.extend(tb.build_trades(name, inst, bars[(name, TF)], fund[name],
                                      spans_by[name]))
    m = tb.metrics(trades)
    perbar, _ = r2.m2m_pnl(TF, bars, fund, spans_by)
    sh = sharpe_horizon(perbar, trades)
    boot = tv.diag_d3_bootstrap(trades, np.random.default_rng(SEED))
    # long/short split
    ls = {}
    for side in ("long", "short"):
        g = [t for t in trades if t["side"] == side]
        ls[side] = {"n": len(g), "net": float(sum(t["net_pnl_usd"] for t in g))}
    return {"version": version, "n": m["n"], "gross_pnl": m["gross_pnl"],
            "fees_total": m["fees_total"], "funding_total": m["funding_total"],
            "net_pnl": m["net_pnl"], "pf_net": m["pf_net"],
            "win_rate_pct": m["win_rate_pct"], "avg_hold_hours": m["avg_hold_hours"],
            "long_short": ls, "by_year": year_net(trades),
            "bootstrap": boot, **sh,
            "m2m_net_check": float(perbar.sum())}


# ── mechanism validation ─────────────────────────────────────────────────────
def normal_two_sided_p(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2.0))


def welch(a: list[float], b: list[float]) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"na": na, "nb": nb, "insufficient": True}
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else 0.0
    df = ((va / na + vb / nb) ** 2 /
          ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))) if se > 0 else 0.0
    return {"na": na, "nb": nb, "mean_a": float(ma), "mean_b": float(mb),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "diff_a_minus_b": float(ma - mb), "welch_t": float(t),
            "welch_df": float(df), "p_two_sided_normal_approx": float(normal_two_sided_p(t))}


def mechanism(bars: dict, fund: dict) -> dict:
    """Classify ORIGINAL B2_4h signals by funding sign at entry; agree vs veto.
    trade-level (primary, independent samples) + bar-level (secondary, autocorr caveat)."""
    tr_agree_net, tr_veto_net = [], []     # trade net PnL ($)
    tr_agree_ret, tr_veto_ret = [], []     # trade gross return % of notional
    bar_agree, bar_veto = [], []           # per-bar fwd return * side
    for name, (_, inst) in tb.SYMBOLS.items():
        b = bars[(name, TF)]
        c = b["close"].to_numpy()
        fr = funding_recent_per_bar(b, fund[name])
        # trade-level: canonical B2_4h spans, classify by funding at entry bar
        spans = tb.positions_flip(tb.signal_emax(b, FAST, SLOW))
        trades = tb.build_trades(name, inst, b, fund[name], spans)
        for (ei, xi, side, _), t in zip(spans, trades):
            agree = leg_agrees(side, fr[ei])  # NaN->veto
            ret = t["gross_pnl_usd"] / tb.NOTIONAL * 100.0
            (tr_agree_net if agree else tr_veto_net).append(t["net_pnl_usd"])
            (tr_agree_ret if agree else tr_veto_ret).append(ret)
        # bar-level: every bar with nonzero original exposure -> next-bar signed return
        target = target_orig_exposure(b)
        for i in range(len(target) - 1):
            if target[i] == 0:
                continue
            agree = leg_agrees(int(target[i]), fr[i])
            r = float(target[i]) * (c[i + 1] / c[i] - 1.0) * 100.0
            (bar_agree if agree else bar_veto).append(r)
    return {
        "trade_level_net_usd": {"agree": welch(tr_agree_net, tr_veto_net),
                                "note": "agree-veto>0 & sig => veto signals worse (mechanism holds)"},
        "trade_level_gross_ret_pct": {"agree": welch(tr_agree_ret, tr_veto_ret)},
        "bar_level_fwd_ret_pct": {"agree": welch(bar_agree, bar_veto),
                                  "caveat": "overlapping holds -> autocorrelated -> t optimistic; secondary only"},
    }


# ── data loading ─────────────────────────────────────────────────────────────
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
    L("STUDY: B2_4h funding same-direction confirmation (highest overfit-defense; known target)")
    L("RULE: single zero-threshold sign-agreement filter, ZERO parameter search")
    L("DATA: OKX database_mainnet.db (mode=ro) + Binance data/binance_vision/ (sha256-verified)")
    L("engines tb/tv/r2/dc + binance_funding imported verbatim; pollution DB not touched")
    OUT.mkdir(parents=True, exist_ok=True)

    samples = {}
    L("\n== loading OKX (2023-2026) ==")
    samples["OKX"] = load_okx()
    L("== loading Binance (2020-2026) ==")
    samples["Binance"] = load_binance()
    for s, (bars, _) in samples.items():
        sp = {n: (pd.Timestamp(int(bars[(n, TF)]['end_min'].iloc[0]) * 60, unit='s', tz='UTC').date(),
                  pd.Timestamp(int(bars[(n, TF)]['end_min'].iloc[-1]) * 60, unit='s', tz='UTC').date())
              for n in tb.SYMBOLS}
        L(f"  {s}: 4h bars per sym " + ", ".join(f"{n}={len(bars[(n,TF)])}" for n in tb.SYMBOLS))
        L(f"       span BTC {sp['BTC']}")

    report = {"samples": {}, "judgment": {}}
    evals = {}
    for s, (bars, fund) in samples.items():
        L(f"\n========== SAMPLE: {s} ==========")
        v0 = evaluate("V0", bars, fund)
        vp = evaluate("V+", bars, fund)
        evals[s] = {"V0": v0, "V+": vp}
        # V0 reproduction check
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
        # subtraction ledger
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
        # mechanism
        mech = mechanism(bars, fund)
        wt = mech["trade_level_net_usd"]["agree"]
        # consistency: V+ net must equal the agree-group net sum (V+ = kept legs)
        agree_sum = wt["mean_a"] * wt["na"]
        cons_ok = abs(agree_sum - vp["net_pnl"]) < 1.0 and wt["na"] == vp["n"]
        L(f"  consistency V+==agree-group: V+ net ${vp['net_pnl']:,.0f} (n{vp['n']}) "
          f"vs agree ${agree_sum:,.0f} (n{wt['na']}) -> {'OK' if cons_ok else 'MISMATCH!!'}")
        L(f"  mechanism (trade-level net): agree mean ${wt['mean_a']:,.0f} (n{wt['na']}) vs "
          f"veto mean ${wt['mean_b']:,.0f} (n{wt['nb']}) | diff ${wt['diff_a_minus_b']:,.0f} | "
          f"Welch t {wt['welch_t']:.2f} p {wt['p_two_sided_normal_approx']:.3f}")
        report["samples"][s] = {"V0": v0, "V+": vp, "ledger": ledger, "mechanism": mech}
        (OUT / f"{s}_V0.json").write_text(json.dumps(v0, indent=2, default=float))
        (OUT / f"{s}_Vplus.json").write_text(json.dumps(vp, indent=2, default=float))
        (OUT / f"{s}_mechanism.json").write_text(json.dumps(mech, indent=2, default=float))
        (OUT / f"{s}_ledger.json").write_text(json.dumps(ledger, indent=2, default=float))

    # ── pre-registered judgment E1-E4 (dual-sample consistency required) ──────
    def both(key_fn):
        return key_fn("OKX"), key_fn("Binance")

    # E1 mechanism: veto significantly worse (agree-veto > 0 AND p<0.05), both samples
    def e1_one(s):
        w = report["samples"][s]["mechanism"]["trade_level_net_usd"]["agree"]
        return w["diff_a_minus_b"] > 0 and w["p_two_sided_normal_approx"] < 0.05
    e1 = e1_one("OKX") and e1_one("Binance")

    # E2 improvement: V+ daily-annualized Sharpe > V0, both samples
    def e2_one(s):
        return evals[s]["V+"]["sharpe_annual_daily"] > evals[s]["V0"]["sharpe_annual_daily"]
    e2 = e2_one("OKX") and e2_one("Binance")

    # E3 horizon shorter (PRIMARY), both samples
    def e3_one(s):
        return report["samples"][s]["ledger"]["horizon_shorter_PRIMARY"]
    e3 = e3_one("OKX") and e3_one("Binance")

    # E4 no new harm: M2M maxDD <= 1.10x V0 AND bootstrap CI lower bound not more negative, both
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
        "conclusion": ("funding 确认有效 -> B2_4h+" if verdict_pass
                       else "funding 确认无效 -> B2_4h 维持原状（资源关闭、18 月验证不变）"),
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
