#!/usr/bin/env python3
"""Classic trend-following naked-baseline screen on OKX mainnet data (4h/1d).

First study of the new research phase. Postmortem §8 checklist applies:
data source = .vntrader/database_mainnet.db ONLY (sqlite mode=ro; passed full
Binance cross-validation, reports/regime/data_trust_closure_20260611/).
The contaminated legacy DB is never touched.

METHODOLOGY POSITION: unoptimized full-sample SCREEN. All 15 configs are
literature-classic prototypes, pre-registered below, zero scanning / zero
tuning — no fitting, hence no overfitting; full-sample use is therefore
methodologically sound. Conclusions are "screen pass/fail" (ticket to the
next stage: OOS / neighborhood / robustness), NOT "edge verified".

COST CONVENTION (verbatim into report header, conservative for trend):
  入场与出场均按信号 bar 收盘价 ±1 tick 以 taker 成交（费率 0.05%/边）；
  不假设任何 maker 成交；计入真实 OKX funding（8h 结算）；
  无滑点压力测试（留下一阶段）；这是对趋势策略偏保守的成本口径。
  注：毛利口径 = 成交价（含 ±1 tick）PnL，费前、funding 前。

PRE-REGISTERED GATES (final values, may not be modified after results):
  GATE-1: portfolio gross (pre-fee pre-funding) <= 0      -> FAIL
  GATE-2: gross > 0 but avg round-trip gross < 0.15% of notional -> MARGINAL
  PASS  : gross > 0 and avg round-trip gross >= 0.15%
  Family verdict: all configs FAIL -> family = dead-end candidate;
  any PASS -> family earns next-stage ticket. n<30 -> [样本不足] tag.

FUNDING CONVENTION (OKX): positive rate -> longs PAY shorts. Charge for each
8h settlement (UTC 00/08/16) strictly inside the holding interval:
entry_moment < t_settle <= exit_moment; amount = rate * contracts * ctVal *
settle_price (1m close of the minute immediately before settlement);
sign = +pay for long when rate>0, short receives (mirror).

AGGREGATION: 1m (DB naive Asia/Shanghai bar-open -> UTC = -8h) bucketed on
UTC boundaries: 4h buckets at 00/04/.../20 UTC, 1d buckets at 00 UTC.
First 1d bucket (2022-12-31, 8h of data) and last bucket are partial — kept,
negligible for indicator warm-up, noted in report. Fills happen at bar CLOSE
moment = bucket end.

SIZING: fixed $10,000 notional per signal (pure notional accounting, no
leverage semantics), integer contracts via ctVal, single-position model
(flat -> long/short). Donchian: no same-bar re-entry after exit (house
convention). EMA-cross / TSMOM: always-in-market, flip on signal change
(flip = close + open at the same bar close, both taker fills).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_mr_5m_compare import CONTRACT_SPECS  # specs reused; engine not modified

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_MAIN = PROJECT_ROOT / ".vntrader" / "database_mainnet.db"
FUND_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
OUT = PROJECT_ROOT / "reports" / "trend_baseline_20260611"

SYMBOLS = {
    "BTC": ("BTCUSDT_SWAP_OKX", "BTC-USDT-SWAP"),
    "ETH": ("ETHUSDT_SWAP_OKX", "ETH-USDT-SWAP"),
    "SOL": ("SOLUSDT_SWAP_OKX", "SOL-USDT-SWAP"),
    "LINK": ("LINKUSDT_SWAP_OKX", "LINK-USDT-SWAP"),
    "DOGE": ("DOGEUSDT_SWAP_OKX", "DOGE-USDT-SWAP"),
}
NOTIONAL = 10_000.0
FEE_TAKER = 0.0005
EIGHT_H_MS = 8 * 3600 * 1000
SH_OFFSET_MIN = 480

GATE2_THICKNESS = 0.0015  # 0.15% of notional per round trip
MIN_SAMPLE = 30

# ── 15 pre-registered configs (locked; nothing added, nothing tuned) ─────────
CONFIGS = [
    # family A — Donchian channel breakout (turtle, symmetric exit)
    {"id": "A1_4h", "family": "A", "tf": "4h", "kind": "donchian", "entry_n": 20, "exit_n": 10},
    {"id": "A1_1d", "family": "A", "tf": "1d", "kind": "donchian", "entry_n": 20, "exit_n": 10},
    {"id": "A2_4h", "family": "A", "tf": "4h", "kind": "donchian", "entry_n": 55, "exit_n": 20},
    {"id": "A2_1d", "family": "A", "tf": "1d", "kind": "donchian", "entry_n": 55, "exit_n": 20},
    {"id": "A3_1d", "family": "A", "tf": "1d", "kind": "donchian", "entry_n": 100, "exit_n": 50},
    # family B — EMA cross, always in market
    {"id": "B1_4h", "family": "B", "tf": "4h", "kind": "emax", "fast": 50, "slow": 200},
    {"id": "B1_1d", "family": "B", "tf": "1d", "kind": "emax", "fast": 50, "slow": 200},
    {"id": "B2_4h", "family": "B", "tf": "4h", "kind": "emax", "fast": 20, "slow": 100},
    {"id": "B2_1d", "family": "B", "tf": "1d", "kind": "emax", "fast": 20, "slow": 100},
    # family C — TSMOM (lookback in DAYS), always in market
    {"id": "C1_4h", "family": "C", "tf": "4h", "kind": "tsmom", "days": 30},
    {"id": "C1_1d", "family": "C", "tf": "1d", "kind": "tsmom", "days": 30},
    {"id": "C2_4h", "family": "C", "tf": "4h", "kind": "tsmom", "days": 90},
    {"id": "C2_1d", "family": "C", "tf": "1d", "kind": "tsmom", "days": 90},
    {"id": "C3_4h", "family": "C", "tf": "4h", "kind": "tsmom", "days": 180},
    {"id": "C3_1d", "family": "C", "tf": "1d", "kind": "tsmom", "days": 180},
]
TF_MIN = {"4h": 240, "1d": 1440}
BARS_PER_DAY = {"4h": 6, "1d": 1}

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── data loading ──────────────────────────────────────────────────────────────
def load_1m_utc(db_symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{DB_MAIN}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "select datetime, open_price as open, high_price as high, "
            "low_price as low, close_price as close from dbbardata "
            "where symbol=? and exchange='GLOBAL' and interval='1m' order by datetime",
            conn, params=(db_symbol,),
        )
    finally:
        conn.close()
    ts = pd.to_datetime(df["datetime"])
    df["min_utc"] = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)
                     ).astype("int64") - SH_OFFSET_MIN
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c])
    return df[["min_utc", "open", "high", "low", "close"]]


def aggregate(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """UTC-boundary buckets; bar datetime = bucket START; fill moment = bucket end."""
    step = TF_MIN[tf]
    g = df1m.groupby(df1m["min_utc"] // step)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "n1m": g["close"].size(),
    })
    out["start_min"] = out.index.astype("int64") * step
    out["end_min"] = out["start_min"] + step
    return out.reset_index(drop=True)


def load_funding(inst: str, m1: pd.DataFrame) -> pd.DataFrame:
    """Funding schedule with settle price = 1m close of minute before settlement."""
    frames = []
    for f in sorted(FUND_DIR.glob(f"{inst}_funding_*.csv")):
        frames.append(pd.read_csv(f, usecols=["funding_time", "funding_rate"]))
    fr = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("funding_time").sort_values("funding_time"))
    # snap jittered settlement stamps (<=12s, verified in data_trust_closure) to 8h grid
    fr["slot_min"] = ((fr["funding_time"] / EIGHT_H_MS).round().astype("int64")
                      * EIGHT_H_MS // 60000)
    px = m1.set_index("min_utc")["close"]
    # settle price: last 1m close at/before slot-1 minute
    idx = px.index.searchsorted(fr["slot_min"].to_numpy() - 1, side="right") - 1
    ok = idx >= 0
    fr = fr.loc[ok].copy()
    fr["settle_px"] = px.to_numpy()[idx[ok]]
    fr["rate"] = pd.to_numeric(fr["funding_rate"])
    return fr[["slot_min", "rate", "settle_px"]].reset_index(drop=True)


def funding_cost(fund: pd.DataFrame, entry_min: int, exit_min: int,
                 side: int, contracts: int, ct_val: float) -> float:
    """Sum of funding payments for holding (entry_min, exit_min]. Positive = cost.
    OKX: positive rate -> long pays. side: +1 long / -1 short."""
    w = fund[(fund["slot_min"] > entry_min) & (fund["slot_min"] <= exit_min)]
    if w.empty:
        return 0.0
    return float((w["rate"] * w["settle_px"]).sum() * contracts * ct_val * side)


def calc_contracts(inst: str, price: float) -> int:
    ct_val = CONTRACT_SPECS[inst]["ctVal"]
    return max(1, round(NOTIONAL / (price * ct_val)))


# ── signal engines ────────────────────────────────────────────────────────────
def positions_donchian(bars: pd.DataFrame, entry_n: int, exit_n: int) -> list[tuple]:
    """Returns list of (entry_i, exit_i, side, reason). Channels exclude current bar."""
    h, l, c = (bars[k].to_numpy() for k in ("high", "low", "close"))
    eh = bars["high"].rolling(entry_n).max().shift(1).to_numpy()
    el = bars["low"].rolling(entry_n).min().shift(1).to_numpy()
    xh = bars["high"].rolling(exit_n).max().shift(1).to_numpy()
    xl = bars["low"].rolling(exit_n).min().shift(1).to_numpy()
    res, pos, ei = [], 0, -1
    for i in range(entry_n + 1, len(c)):
        if pos != 0:
            if pos == 1 and c[i] < xl[i]:
                res.append((ei, i, 1, "channel_exit")); pos = 0
                continue  # no same-bar re-entry
            if pos == -1 and c[i] > xh[i]:
                res.append((ei, i, -1, "channel_exit")); pos = 0
                continue
        if pos == 0 and not np.isnan(eh[i]):
            if c[i] > eh[i]:
                pos, ei = 1, i
            elif c[i] < el[i]:
                pos, ei = -1, i
    if pos != 0:
        res.append((ei, len(c) - 1, pos, "end_of_data"))
    return res


def positions_flip(sig: np.ndarray) -> list[tuple]:
    """Always-in-market from first nonzero signal; flip on sign change."""
    res, pos, ei = [], 0, -1
    for i in range(len(sig)):
        s = sig[i]
        if s == 0 or np.isnan(s):
            continue
        if pos == 0:
            pos, ei = int(s), i
        elif int(s) != pos:
            res.append((ei, i, pos, "flip"))
            pos, ei = int(s), i
    if pos != 0:
        res.append((ei, len(sig) - 1, pos, "end_of_data"))
    return res


def signal_emax(bars: pd.DataFrame, fast: int, slow: int) -> np.ndarray:
    c = bars["close"]
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    sig = np.sign(ef - es).to_numpy().copy()
    sig[:slow] = 0  # warm-up: no position before slow EMA has seen `slow` bars
    return sig


def signal_tsmom(bars: pd.DataFrame, days: int, tf: str) -> np.ndarray:
    lb = days * BARS_PER_DAY[tf]
    c = bars["close"].to_numpy()
    sig = np.zeros(len(c))
    r = np.full(len(c), np.nan)
    r[lb:] = c[lb:] / c[:-lb] - 1
    sig[lb:] = np.sign(r[lb:])
    # sign==0 (exact tie): hold previous -> forward-fill nonzero
    for i in range(1, len(sig)):
        if sig[i] == 0 and not np.isnan(r[i]) and i > lb:
            sig[i] = sig[i - 1]
    return sig


# ── trade building ────────────────────────────────────────────────────────────
def build_trades(name: str, inst: str, bars: pd.DataFrame, fund: pd.DataFrame,
                 spans: list[tuple]) -> list[dict]:
    tick = CONTRACT_SPECS[inst]["tickSz"]
    ct_val = CONTRACT_SPECS[inst]["ctVal"]
    c = bars["close"].to_numpy()
    end_min = bars["end_min"].to_numpy()
    trades = []
    for ei, xi, side, reason in spans:
        ep_raw, xp_raw = c[ei], c[xi]
        ep = ep_raw + tick * side          # adverse tick on entry
        xp = xp_raw - tick * side          # adverse tick on exit
        n = calc_contracts(inst, ep_raw)
        gross = (xp - ep) * n * ct_val * side
        fee = -(FEE_TAKER * ep * n * ct_val + FEE_TAKER * xp * n * ct_val)
        fnd = -funding_cost(fund, int(end_min[ei]), int(end_min[xi]), side, n, ct_val)
        t_en = pd.Timestamp(int(end_min[ei]) * 60, unit="s", tz="UTC")
        t_ex = pd.Timestamp(int(end_min[xi]) * 60, unit="s", tz="UTC")
        trades.append({
            "time": t_ex.isoformat(), "symbol": name,
            "side": "long" if side == 1 else "short",
            "entry_time": t_en.isoformat(), "entry_price": round(float(ep), 8),
            "exit_price": round(float(xp), 8), "exit_reason": reason,
            "size": int(n), "hold_hours": float((int(end_min[xi]) - int(end_min[ei])) / 60),
            "gross_pnl_usd": round(float(gross), 4),
            "fee_usd": round(float(fee), 4),
            "funding_usd": round(float(fnd), 4),
            "net_pnl_usd": round(float(gross + fee + fnd), 4),
        })
    return trades


# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(trades: list[dict]) -> dict:
    m = {"n": len(trades)}
    if not trades:
        return m
    df = pd.DataFrame(trades)
    df["exit_dt"] = pd.to_datetime(df["time"])
    df = df.sort_values("exit_dt", kind="stable").reset_index(drop=True)
    net, gross = df["net_pnl_usd"], df["gross_pnl_usd"]
    wins, losses = net[net > 0], net[net < 0]
    gw, gl = gross[gross > 0], gross[gross < 0]
    m.update({
        "gross_pnl": float(gross.sum()),
        "funding_total": float(df["funding_usd"].sum()),
        "fees_total": float(df["fee_usd"].sum()),
        "net_pnl": float(net.sum()),
        "avg_hold_hours": float(df["hold_hours"].mean()),
        "win_rate_pct": float((net > 0).mean() * 100),
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None,
        "pf_gross": float(gw.sum() / abs(gl.sum())) if gl.sum() != 0 else float("inf"),
        "pf_net": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "avg_roundtrip_gross_pct_of_notional": float(gross.sum() / (len(df) * NOTIONAL) * 100),
    })
    eq = net.cumsum()
    m["max_dd_usd"] = float(-(eq - eq.cummax()).min())
    yr = {}
    ey = pd.to_datetime(df["entry_time"]).dt.year
    for y in (2023, 2024, 2025, 2026):
        g = df[ey == y]
        if len(g):
            w = g[g["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
            lo = abs(g[g["net_pnl_usd"] < 0]["net_pnl_usd"].sum())
            yr[str(y)] = {"n": int(len(g)),
                          "gross": float(g["gross_pnl_usd"].sum()),
                          "funding": float(g["funding_usd"].sum()),
                          "fees": float(g["fee_usd"].sum()),
                          "net": float(g["net_pnl_usd"].sum()),
                          "pf_net": float(w / lo) if lo > 0 else float("inf")}
    m["by_year"] = yr
    return m


def gate(m: dict) -> str:
    if m["n"] == 0 or m["gross_pnl"] <= 0:
        return "FAIL"
    if m["avg_roundtrip_gross_pct_of_notional"] < GATE2_THICKNESS * 100:
        return "MARGINAL"
    return "PASS"


# ── funding module self-check (hand-computed comparison, into appendix) ──────
def funding_selfcheck(fund_by_sym: dict, m1_by_sym: dict) -> str:
    lines = ["# funding 模块手算自检", "",
             "构造持仓：BTC long 1 张（ctVal 0.01），entry 2024-03-10 04:00 UTC，"
             "exit 2024-03-11 04:00 UTC。",
             "应计结算时刻（entry < t ≤ exit）：03-10 08:00、03-10 16:00、03-11 00:00 UTC。", ""]
    fund = fund_by_sym["BTC"]
    px = m1_by_sym["BTC"].set_index("min_utc")["close"]
    t0 = int(pd.Timestamp("2024-03-10 04:00", tz="UTC").timestamp() // 60)
    t1 = int(pd.Timestamp("2024-03-11 04:00", tz="UTC").timestamp() // 60)
    w = fund[(fund["slot_min"] > t0) & (fund["slot_min"] <= t1)]
    manual = 0.0
    lines.append("| 结算时刻(UTC) | rate | settle_px(1m close 前一分钟) | 手算费用 = rate×px×1×0.01 |")
    lines.append("|---|---|---|---|")
    for _, r in w.iterrows():
        cost = r["rate"] * r["settle_px"] * 1 * 0.01
        manual += cost
        ts = pd.Timestamp(int(r["slot_min"]) * 60, unit="s", tz="UTC")
        raw_px = px.loc[:int(r["slot_min"]) - 1].iloc[-1]
        assert raw_px == r["settle_px"]
        lines.append(f"| {ts} | {r['rate']:.10f} | {r['settle_px']:.1f} | ${cost:.6f} |")
    module = funding_cost(fund, t0, t1, 1, 1, 0.01)
    lines += ["", f"- 手算合计（long 应付）：**${manual:.6f}**",
              f"- 模块输出 funding_cost()：**${module:.6f}**",
              f"- 差值：{abs(manual - module):.2e} → {'**一致，自检通过**' if abs(manual - module) < 1e-9 else '**不一致，自检失败**'}",
              "", "约定核实：OKX 正费率 = 多头付空头；trade 记录中 funding_usd 取负号表成本"
              "（long 在正费率期间 funding_usd < 0）。结算数 3 个 = 手算预期（entry 时刻本身不计费）。"]
    ok = abs(manual - module) < 1e-9 and len(w) == 3
    if not ok:
        raise SystemExit("FUNDING SELF-CHECK FAILED")
    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: .vntrader/database_mainnet.db (mode=ro, Binance cross-validated PASS)")
    L("contaminated legacy DB: not touched")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    (OUT / "by_year").mkdir(exist_ok=True)

    L("\nloading 1m + funding, aggregating 4h/1d ...")
    m1, bars, fund = {}, {}, {}
    for name, (db_sym, inst) in SYMBOLS.items():
        m1[name] = load_1m_utc(db_sym)
        bars[(name, "4h")] = aggregate(m1[name], "4h")
        bars[(name, "1d")] = aggregate(m1[name], "1d")
        fund[name] = load_funding(inst, m1[name])
        L(f"  {name}: 1m {len(m1[name]):,} | 4h {len(bars[(name,'4h')]):,} | "
          f"1d {len(bars[(name,'1d')]):,} | funding {len(fund[name]):,}")

    sc = funding_selfcheck(fund, m1)
    (OUT / "funding_module_selfcheck.md").write_text(sc)
    L("funding module self-check: PASS (see funding_module_selfcheck.md)")

    summary, yearly_all = [], {}
    for cfg in CONFIGS:
        cid, tf = cfg["id"], cfg["tf"]
        all_trades, per_sym = [], {}
        for name, (_, inst) in SYMBOLS.items():
            b = bars[(name, tf)]
            if cfg["kind"] == "donchian":
                spans = positions_donchian(b, cfg["entry_n"], cfg["exit_n"])
            elif cfg["kind"] == "emax":
                spans = positions_flip(signal_emax(b, cfg["fast"], cfg["slow"]))
            else:
                spans = positions_flip(signal_tsmom(b, cfg["days"], tf))
            tr = build_trades(name, inst, b, fund[name], spans)
            per_sym[name] = metrics(tr)
            all_trades.extend(tr)
        m = metrics(all_trades)
        verdict = gate(m)
        small = m["n"] < MIN_SAMPLE
        out = {"config": cfg, "verdict": verdict, "small_sample": small,
               "portfolio": m, "per_symbol": per_sym}
        (OUT / "configs" / f"{cid}.json").write_text(json.dumps(out, indent=2))
        with open(OUT / "configs" / f"{cid}_trades.jsonl", "w") as f:
            for t in sorted(all_trades, key=lambda x: x["time"]):
                f.write(json.dumps(t) + "\n")
        yearly_all[cid] = m.get("by_year", {})
        summary.append({"id": cid, "family": cfg["family"], "tf": tf,
                        "verdict": verdict, "small_sample": small, **{
                            k: m.get(k) for k in (
                                "n", "gross_pnl", "funding_total", "fees_total", "net_pnl",
                                "avg_roundtrip_gross_pct_of_notional", "pf_gross", "pf_net",
                                "win_rate_pct", "avg_hold_hours", "max_dd_usd")}})
        L(f"[{cid}] {verdict}{' [样本不足]' if small else ''} | n={m['n']} | "
          f"gross ${m['gross_pnl']:,.0f} | funding ${m['funding_total']:,.0f} | "
          f"fees ${m['fees_total']:,.0f} | net ${m['net_pnl']:,.0f} | "
          f"rt_gross {m['avg_roundtrip_gross_pct_of_notional']:.3f}% | PFnet "
          f"{m['pf_net']:.2f}")

    (OUT / "by_year" / "yearly.json").write_text(json.dumps(yearly_all, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    counts = {v: sum(1 for s in summary if s["verdict"] == v)
              for v in ("PASS", "MARGINAL", "FAIL")}
    fam = {}
    for f_ in ("A", "B", "C"):
        vs = [s["verdict"] for s in summary if s["family"] == f_]
        fam[f_] = ("dead-end candidate" if all(v == "FAIL" for v in vs)
                   else ("has PASS -> next-stage ticket" if "PASS" in vs else "marginal only"))
    L(f"\nverdict counts: {counts}")
    L(f"family verdicts: {fam}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
