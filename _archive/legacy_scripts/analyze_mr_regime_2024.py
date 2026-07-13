#!/usr/bin/env python3
"""MR-5m regime exploration: when does mean-reversion stop working?

Data-exploration only (NO production filter). Goal: characterise the months in
which the MR-5m strategy fails (PF<1) and surface candidate market-state
indicators that separate "chop" (MR-friendly) from "trend" (MR-hostile).

Steps:
  1. Monthly breakdown 2023-01 .. 2026-05 for each symbol: PF, stop%.
  2. BTC monthly market features (daily-bar based):
       - return%, realized vol (daily-return std)
       - Kaufman Efficiency Ratio (trend vs chop; 0=chop, 1=pure trend)
       - mean deviation from 50d MA
       - trend persistence: max consecutive same-direction days,
         and share of days printing 20d new highs/lows
       - lag-1 autocorrelation of daily returns
  3. Flag PF<1 months (portfolio + per symbol), compare feature means
     fail vs healthy, correlations, and threshold buckets.
  4. Write reports/regime/mr_regime_analysis.md + monthly CSV.

ADX intentionally excluded (prior research: poor).

Usage: python scripts/analyze_mr_regime_2024.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 2026-07 重构批次6：脚本迁入 _archive/legacy_scripts/；共享依赖真身在
# scripts/（前向冻结区）与 core/data_io/，此处按新深度注入 sys.path。
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (
    str(_REPO_ROOT / "core" / "data_io"),
    str(_REPO_ROOT / "scripts"),
    str(_REPO_ROOT / "data_engineering" / "scripts"),
    *sorted(str(_q) for _q in (_REPO_ROOT / "research" / "_closed").glob("*/*/scripts")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from backtest_mr_5m_compare import (
    SYMBOLS, backtest_symbol, load_1m, r5, parse_history_range,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2026-07 重构批次6：迁入 _archive/legacy_scripts/，深度 1→2
DB = PROJECT_ROOT / ".vntrader" / "database.db"
START, END = "2023-01-01", "2026-05-28"
TIMEZONE = "UTC"
OUT_DIR = PROJECT_ROOT / "reports" / "regime"
SYM_NAMES = ["BTC", "ETH", "SOL", "LINK", "DOGE"]


def pf_of(net: pd.Series) -> float:
    w = net[net > 0].sum()
    l = abs(net[net < 0].sum())
    return float(w / l) if l > 0 else float("inf")


def max_consec_same_sign(signs: np.ndarray) -> int:
    best = cur = 0
    prev = 0
    for s in signs:
        if s != 0 and s == prev:
            cur += 1
        elif s != 0:
            cur = 1
        else:
            cur = 0
        prev = s
        best = max(best, cur)
    return best


def build_daily(df1m: pd.DataFrame) -> pd.DataFrame:
    d = df1m.set_index("datetime").resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).dropna()
    d["ret"] = d["close"].pct_change()
    d["ma50"] = d["close"].rolling(50).mean()
    d["dev_ma50"] = d["close"] / d["ma50"] - 1.0
    d["hh20"] = d["high"].rolling(20).max()
    d["ll20"] = d["low"].rolling(20).min()
    d["new_high20"] = d["close"] >= d["hh20"].shift(1)
    d["new_low20"] = d["close"] <= d["ll20"].shift(1)
    return d


def monthly_btc_features(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, g in daily.groupby(daily.index.to_period("M")):
        g = g.dropna(subset=["close"])
        if len(g) < 5:
            continue
        c = g["close"].to_numpy()
        ret_m = c[-1] / c[0] - 1.0
        dret = g["ret"].dropna()
        vol = float(dret.std())
        abs_changes = np.abs(np.diff(c)).sum()
        er = float(abs(c[-1] - c[0]) / abs_changes) if abs_changes > 0 else 0.0
        ac1 = float(dret.autocorr(lag=1)) if len(dret) > 3 else np.nan
        dev = float(g["dev_ma50"].mean())
        signs = np.sign(g["ret"].fillna(0).to_numpy())
        consec = max_consec_same_sign(signs)
        nh = int(g["new_high20"].sum())
        nl = int(g["new_low20"].sum())
        ndays = len(g)
        # directional drive: |monthly return| normalised by month volatility
        drive = float(abs(ret_m) / (vol * np.sqrt(ndays))) if vol > 0 else np.nan
        rows.append({
            "month": str(period),
            "btc_ret_pct": ret_m * 100,
            "btc_vol_d": vol * 100,
            "btc_er": er,
            "btc_dev_ma50_pct": dev * 100,
            "btc_max_consec_days": consec,
            "btc_newext_share": max(nh, nl) / ndays,
            "btc_autocorr1": ac1,
            "btc_drive": drive,
        })
    return pd.DataFrame(rows).set_index("month")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hr = parse_history_range(START, END, timedelta(minutes=1), TIMEZONE)

    # ── per-symbol trades + BTC daily features ──
    sym_trades = {}
    btc_daily = None
    for name in SYM_NAMES:
        vt = SYMBOLS[name][0]
        print(f"[{name}] loading...", flush=True)
        b1 = load_1m(vt, hr, DB)
        if name == "BTC":
            btc_daily = build_daily(b1)
        b5 = r5(b1, 5, hr)
        t = backtest_symbol(name, b5)
        df = pd.DataFrame(t)
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        df["month"] = df["entry_time"].dt.to_period("M").astype(str)
        sym_trades[name] = df
        print(f"[{name}] {len(df):,} trades", flush=True)

    feat = monthly_btc_features(btc_daily)

    # ── monthly per-symbol PF & stop% ──
    all_months = sorted(set(feat.index) | {m for df in sym_trades.values()
                                           for m in df["month"].unique()})
    pf_tbl, stop_tbl, n_tbl = {}, {}, {}
    for name in SYM_NAMES:
        df = sym_trades[name]
        pf_tbl[name], stop_tbl[name], n_tbl[name] = {}, {}, {}
        for m, g in df.groupby("month"):
            pf_tbl[name][m] = pf_of(g["net_pnl_usd"])
            stop_tbl[name][m] = (g["exit_reason"] == "stop").mean() * 100
            n_tbl[name][m] = len(g)

    # portfolio monthly PF (all symbols pooled)
    pool = pd.concat([sym_trades[n][["month", "net_pnl_usd", "exit_reason"]]
                      for n in SYM_NAMES])
    port_pf, port_stop, port_net = {}, {}, {}
    for m, g in pool.groupby("month"):
        port_pf[m] = pf_of(g["net_pnl_usd"])
        port_stop[m] = (g["exit_reason"] == "stop").mean() * 100
        port_net[m] = g["net_pnl_usd"].sum()

    # ── master monthly frame ──
    master = feat.copy()
    master["port_pf"] = pd.Series(port_pf)
    master["port_net"] = pd.Series(port_net)
    master["port_stop_pct"] = pd.Series(port_stop)
    for name in SYM_NAMES:
        master[f"pf_{name}"] = pd.Series(pf_tbl[name])
        master[f"stop_{name}"] = pd.Series(stop_tbl[name])
    master = master.loc[[m for m in all_months if m in master.index]]
    master.to_csv(OUT_DIR / "mr_regime_monthly.csv")

    # ── failure analysis ──
    valid = master.dropna(subset=["port_pf"]).copy()
    valid["fail"] = valid["port_pf"] < 1.0
    valid["btc_abs_ret"] = valid["btc_ret_pct"].abs()
    valid["pf_cap"] = valid["port_pf"].replace([np.inf, -np.inf], np.nan).clip(upper=5.0)
    fail = valid[valid["fail"]]
    ok = valid[~valid["fail"]]

    feat_cols = ["btc_ret_pct", "btc_abs_ret", "btc_vol_d", "btc_er", "btc_dev_ma50_pct",
                 "btc_max_consec_days", "btc_newext_share", "btc_autocorr1", "btc_drive"]

    # correlation of features vs portfolio PF (capped) and vs net$
    corr = {}
    for col in feat_cols:
        pear = float(valid[col].corr(valid["pf_cap"]))
        spear = float(valid[col].rank().corr(valid["pf_cap"].rank()))
        pear_net = float(valid[col].corr(valid["port_net"]))
        corr[col] = {"pf_pearson": pear, "pf_spearman": spear, "net_pearson": pear_net}

    write_report(master, valid, fail, ok, feat_cols, corr)
    print(f"\n-> wrote {OUT_DIR/'mr_regime_analysis.md'}")
    print(f"-> wrote {OUT_DIR/'mr_regime_monthly.csv'}")

    # quick console digest
    print(f"\nMonths analysed: {len(valid)}  |  PF<1 (fail): {len(fail)}")
    print("Feature means (fail vs ok):")
    for col in feat_cols:
        print(f"  {col:>22}: fail={fail[col].mean():8.3f}  ok={ok[col].mean():8.3f}")


def fmt_pf(x):
    if pd.isna(x):
        return "-"
    return "inf" if np.isinf(x) else f"{x:.2f}"


def write_report(master, valid, fail, ok, feat_cols, corr):
    L = []
    A = L.append
    A("# MR-5m 市场状态探索：什么时候停止做均值回归？\n")
    A(f"- 区间: {START} → {END}（{TIMEZONE}），按入场月份归集")
    A("- 失效定义: 当月**组合 PF < 1**（5 币种合并）")
    A("- 市场特征基于 **BTC 日线**（作为整体市场代理）")
    A("- 候选指标（**不含 ADX**）: 效率比 ER、相对50日均线偏离、趋势持续性、收益自相关\n")
    A("> 本报告仅做数据观察，不实现过滤器。目标是先理解市场何时不适合 MR。\n")

    # ── Section 1: monthly master table ──
    A("## 1. 每月数据总表\n")
    A("ER = Kaufman 效率比（0=纯震荡, 1=纯趋势）；drive = |月涨跌|/(日波动×√天数)，方向驱动强度；"
      "newext% = 当月录得 20 日新高/新低的天数占比；AC1 = 日收益 lag-1 自相关。\n")
    A("| 月份 | BTC涨跌% | BTC日波动% | ER | 偏离MA50% | 连续同向天 | newext% | AC1 | drive | 组合PF | 组合net | 组合stop% |")
    A("|------|--------:|----------:|----:|---------:|----------:|--------:|-----:|------:|------:|--------:|---------:|")
    for m, r in master.iterrows():
        flag = " ⚠️" if (pd.notna(r["port_pf"]) and r["port_pf"] < 1) else ""
        dev = "-" if pd.isna(r["btc_dev_ma50_pct"]) else f"{r['btc_dev_ma50_pct']:+.1f}"
        A(f"| {m}{flag} | {r['btc_ret_pct']:+.1f} | {r['btc_vol_d']:.2f} | {r['btc_er']:.2f} | "
          f"{dev} | {r['btc_max_consec_days']:.0f} | "
          f"{r['btc_newext_share']*100:.0f}% | "
          f"{r['btc_autocorr1']:+.2f} | {r['btc_drive']:.2f} | "
          f"{fmt_pf(r['port_pf'])} | {r['port_net']:+,.0f} | {r['port_stop_pct']:.0f}% |")

    # ── Section 1b: per-symbol monthly PF ──
    A("\n## 1b. 各币种每月 PF\n")
    A("| 月份 | " + " | ".join(f"PF_{n}" for n in SYM_NAMES) + " | "
      + " | ".join(f"stop_{n}%" for n in SYM_NAMES) + " |")
    A("|------|" + "----:|" * (2 * len(SYM_NAMES)))
    for m, r in master.iterrows():
        pf_cells = " | ".join(fmt_pf(r.get(f"pf_{n}")) for n in SYM_NAMES)
        st_cells = " | ".join(
            ("-" if pd.isna(r.get(f"stop_{n}")) else f"{r.get(f'stop_{n}'):.0f}")
            for n in SYM_NAMES)
        A(f"| {m} | {pf_cells} | {st_cells} |")

    interp = {
        "btc_ret_pct": "月涨跌%（带方向）",
        "btc_abs_ret": "月涨跌绝对值%（趋势幅度）",
        "btc_vol_d": "日波动率%",
        "btc_er": "效率比 ER（越高越趋势）",
        "btc_dev_ma50_pct": "相对MA50偏离%（带方向）",
        "btc_max_consec_days": "最长连续同向天数",
        "btc_newext_share": "20日新高/新低日占比",
        "btc_autocorr1": "日收益 lag-1 自相关（正=动量/趋势）",
        "btc_drive": "方向驱动 drive（强方向幅度）",
    }

    # ── Section 2: failure characterisation ──
    A("\n## 2. 失效月份（组合 PF<1）的市场特征\n")
    fail_months = list(fail.index)
    A(f"共 **{len(fail)}** 个失效月 / {len(valid)} 个有效月：{', '.join(fail_months)}\n")
    A("> **反直觉的核心发现**：BTC 涨幅最大、趋势最强的几个月（2024-02 +47.7%、"
      "2024-11 +39.6%、2023-10 +26.3%）反而都是**盈利月**；最差的月（2024-07，PF 0.79）"
      "BTC 仅 +5%、效率比仅 0.08，是**震荡市**。失效不是被「强趋势」打败，而是被「来回打脸的"
      "震荡（whipsaw）」打败。\n")
    A("**失效月 vs 健康月 — 特征均值对比：**\n")
    A("| 特征 | 失效月均值 | 健康月均值 | 差异 |")
    A("|------|----------:|----------:|-----:|")
    cols2 = ["btc_abs_ret", "btc_er", "btc_drive", "btc_max_consec_days",
             "btc_newext_share", "btc_autocorr1", "btc_vol_d", "btc_ret_pct"]
    for col in cols2:
        fm, om = fail[col].mean(), ok[col].mean()
        A(f"| {interp.get(col, col)} | {fm:.3f} | {om:.3f} | {fm-om:+.3f} |")
    fstop, ostop = fail["port_stop_pct"].mean(), ok["port_stop_pct"].mean()
    A(f"| 组合 stop 占比% | {fstop:.1f} | {ostop:.1f} | {fstop-ostop:+.1f} |")
    A("\n失效月相对健康月：效率比/方向驱动/连续同向天数**更低**（更震荡），波动率**更低**，"
      "而 stop 占比**更高**。即「低效率、低波动、却频繁触发止损」——典型的假突破来回扫损。")

    # ── Section 3: candidate filter analysis ──
    A("\n## 3. 候选过滤指标初步分析\n")
    A("**3.1 各特征与组合表现的相关性**（PF 截顶 5.0；月度样本 n="
      f"{len(valid)}，相关性整体偏弱，仅作方向参考）：\n")
    A("| 特征 | vs PF (Pearson) | vs PF (Spearman) | vs net$ (Pearson) |")
    A("|------|---------------:|----------------:|-----------------:|")
    for col in sorted(feat_cols, key=lambda c: abs(corr[c]["net_pearson"]), reverse=True):
        cc = corr[col]
        A(f"| {interp.get(col, col)} | {cc['pf_pearson']:+.2f} | "
          f"{cc['pf_spearman']:+.2f} | {cc['net_pearson']:+.2f} |")
    A("\n注意符号：ER、|涨跌|、drive、连续同向天数与 net$ 均为**正相关**——趋势/方向越强，"
      "策略反而越赚；波动率、MA50偏离为负相关。**这与「强趋势杀死 MR」的直觉相反。**")

    # threshold buckets (terciles)
    A("\n**3.2 三分位分桶**：看 MR 表现如何随各指标变化。\n")
    for col, label in (("btc_abs_ret", "|BTC月涨跌|%"), ("btc_er", "效率比 ER"),
                       ("btc_vol_d", "日波动率%"), ("btc_drive", "方向驱动 drive")):
        v = valid.copy()
        try:
            v["bucket"] = pd.qcut(v[col], 3, labels=["低", "中", "高"])
        except Exception:
            continue
        A(f"\n*按 {label} 三分位：*\n")
        A("| 档位 | 月数 | 失效率 | 平均组合PF | 平均组合net$ |")
        A("|------|----:|------:|----------:|-----------:|")
        for b, g in v.groupby("bucket", observed=True):
            A(f"| {b} | {len(g)} | {g['fail'].mean()*100:.0f}% | "
              f"{g['pf_cap'].mean():.2f} | {g['port_net'].mean():+,.0f} |")

    # ── Section 4: observations ──
    A("\n## 4. 初步观察与下一步方向\n")
    A("（数据驱动的初步判断，待下一轮量化验证。）\n")
    _observations(A, fail, ok, corr, valid)

    (OUT_DIR / "mr_regime_analysis.md").write_text("\n".join(L) + "\n")


def _observations(A, fail, ok, corr, valid):
    A("1. **「强趋势 → 停做 MR」的假设被月度数据否定。** BTC 月度趋势/方向指标（ER、|涨跌|、"
      "drive）与策略 net$ 全部呈**正相关**；|涨跌|最高的一档失效率最低（约 7%）、平均 net 最高。"
      "若按「BTC 趋势强就停做」来过滤，会**砍掉最赚钱的月份**——这正是上一轮我提的直觉需要被纠正之处。")
    A("2. **真正的 MR 杀手是「震荡扫损（whipsaw）」而非趋势。** 失效月的共性是：效率比低、"
      f"波动率低、但 stop 占比更高（{fail['port_stop_pct'].mean():.0f}% vs {ok['port_stop_pct'].mean():.0f}%）。"
      "即价格在窄幅内反复假突破，既不回归中线、也不形成干净趋势，把仓位来回扫掉。")
    A("3. **月度 + BTC 代理的区分力本身有限**（相关系数 |r|≤0.34，Spearman 更弱，失效仅 7/41 月）。"
      "这说明：(a) 失效更可能由**月内/日内**的微观结构（盘中反转、事件冲击，如 2024-07 的抛压）驱动，"
      "月度聚合把它平滑掉了；(b) BTC 未必是各币种的好代理，**按币种各自的状态**可能更准。")
    A("4. **stop 占比是目前最直接的同期判别量**，但它几乎是 PF 的镜像（近乎同义），"
      "不能作为**前瞻**闸门——需要找在开仓**之前**就可观测、且能预示 whipsaw 的指标。")
    A("\n**下一轮建议方向（先探索、暂不实现过滤器）：**")
    A("- 把分析下沉到**滚动窗口（如 24/48 根 5m bar 或日内）**与**逐币种**，"
      "而非月度+BTC，验证 whipsaw 是否在更细粒度上可识别。")
    A("- 候选前瞻指标改为刻画 whipsaw 的量：**近期突破后的「跟随度」/「中线命中率」**、"
      "**实现波动与 ATR 阈值的相对位置**、**短窗效率比的快速下降**；避开 ADX（已验证差）。")
    A("- 用「连续 N 笔止损」或「滚动短窗 PF」作为**状态熔断**的对照基准，"
      "评估事后熔断 vs 事前过滤哪种对全周期 PF/回撤更有效。")


if __name__ == "__main__":
    raise SystemExit(main())
