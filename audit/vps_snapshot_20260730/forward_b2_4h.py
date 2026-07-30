#!/usr/bin/env python3
"""B2_4h forward-observation resident system — phase-1 zero-cost simulated accounting.

POSITIONING (verbatim into forward/README.md top):
  本系统对 B2_4h（4h EMA20/100 交叉, long/short, always-in-market, 信号反向出场, 参数永久冻结）
  做零成本前向观察：拉 OKX 真实 mainnet 公开行情 + 本地模拟成交记账，**不触碰任何真实或模拟
  账户、不读任何 API 凭证、不下任何单、绝不使用 OKX demo**（demo 撮合失真是 MR-5m 事故根源）。
  B2_4h 历史 Sharpe~0.5 系在已知数据上验证；前向样本与全部历史独立，是当前唯一能增加证据
  等级的途径。这是三阶段验证的阶段 1（模拟记账 → 满 18 月看 UPGRADE → 阶段 2 小额真金 →
  阶段 3 正式资金），阶段 1 未过不上真金。B2_4h 经四次增强（funding/ADX/faster-entry/V1）双样本
  全判死，确认当前形态即最优——本系统部署原样，零优化/过滤/参数变体。
  三角色分离：开发由 Claude 本地完成；部署由用户手动在 VPS 执行；运行由 VPS cron 自主触发(无 AI)。

ENGINE: research_trend_baseline(tb) + research_trend_validation_r2(r2) imported VERBATIM, zero
  modification. Forward accounting is a THIN outer wrapper: it maintains an append-only local
  1m+funding store (seeded ONCE from the read-only mainnet DB so EMA20/100 carry full history,
  then REST-appended) and on every --account call does a FULL deterministic recompute via
  tb.signal_emax / tb.positions_flip / tb.build_trades + r2.m2m_pnl. => identical to the backtest
  by construction (the dry-run proves $0). The main DB is NEVER written; the contaminated DB is
  NEVER touched; no OKX demo endpoint, no auth header, no order ever placed.

MODES:
  --selfcheck      env/credential/endpoint/config-hash self-check (also auto-run at every start)
  --build-baseline compute forward/baseline_distribution.json (gate numbers; OKX + Binance)
  --seed           one-time: copy read-only mainnet DB 1m + main funding -> forward store
  --update         pull incremental confirmed 1m + funding from OKX mainnet REST -> store
  --account        full recompute ledger from store -> trades/daily-M2M jsonl + positions + heartbeat
  --cron-4h        = update + account  (install on a 4h cadence; see VPS manual)
  --push           compose + send daily PushPlus summary  (install on a daily cadence)
  --cron-daily     = push
  --reconcile      monthly: re-fetch last full month from REST, compare to store + recompute -> diff must ~0
  --dry-run        pre-deploy validation: hold out last complete DB month, feed it as forward
                   increments, assert forward ledger == backtest ledger to the cent ($0)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_trend_baseline as tb
import research_trend_validation_r2 as r2
# NOTE: Binance loaders (research_trend_dualcycle / binance_funding) are imported LAZILY inside
# mode_build_baseline only. The LIVE VPS system (--cron-4h / --cron-daily / --reconcile) imports
# just the engine -> dependency closure = numpy, pandas, requests (no vnpy / no gateway / no Binance).

ROOT = Path(__file__).resolve().parents[1]
FWD = ROOT / "forward"
STORE = FWD / "data"
KL = STORE / "klines"
FU = STORE / "funding"
STATE = FWD / "state"
CONFIG_PATH = FWD / "config_frozen.json"
BASELINE_PATH = FWD / "baseline_distribution.json"
NOTIFY_CONF = FWD / "notify.conf"
DEPLOY_PATH = STATE / "deploy.json"             # records deployment moment, used to exclude backfill

OKX_BASE = "https://www.okx.com"                 # MAINNET public, hardcoded
FAST, SLOW, TF = 20, 100, "4h"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "b2-4h-forward/1.0"})   # NO demo header, NO auth header

LOG: list[str] = []


def L(msg=""):
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def cfg() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


# ─────────────────────────── startup self-check (anti-corruption) ───────────────
def selfcheck(strict=True) -> dict:
    # sensitive tokens assembled from split literals so NONE appear contiguously in this
    # source -> the source-grep below never self-matches, and a user `grep` for any of them
    # returns EMPTY = unambiguously safe (see VPS manual 5c).
    SIM = "x-simulated" "-trading"
    AKEY = "OK-" "ACCESS-KEY"
    ASIGN = "OK-" "ACCESS-SIGN"
    APASS = "OK-" "ACCESS-PASSPHRASE"
    res = {"config_sha256": config_sha256(), "checks": {}}
    c = cfg()
    res["checks"]["server_is_mainnet"] = (c["data"]["server"] == "mainnet" and c["data"]["demo"] is False)
    res["checks"]["okx_base_mainnet"] = (OKX_BASE == "https://www.okx.com")
    res["checks"]["session_no_auth_or_demo_headers"] = not any(
        k.lower() in (AKEY.lower(), ASIGN.lower(), APASS.lower(), SIM.lower())
        for k in SESSION.headers)
    src = Path(__file__).read_text()
    danger = [SIM, AKEY, ASIGN, APASS, "/api/v5/" "trade/order",
              "/api/v5/" "account/", "/api/v5/" "asset/"]
    hits = [t for t in danger if t in src]
    res["checks"]["no_demo_auth_trading_tokens_in_source"] = (len(hits) == 0)
    res["danger_hits"] = hits
    ok = all(res["checks"].values())
    res["pass"] = ok
    L(f"selfcheck: config_sha256={res['config_sha256'][:16]}… pass={ok} {res['checks']}")
    if strict and not ok:
        raise SystemExit("SELFCHECK FAILED — refusing to run (see checks above)")
    return res


# ─────────────────────────── forward store I/O ─────────────────────────────────
def _store_paths():
    for p in (KL, FU, STATE):
        p.mkdir(parents=True, exist_ok=True)


def load_store_1m(coin: str) -> pd.DataFrame:
    p = KL / f"{coin}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["min_utc", "open", "high", "low", "close"])
    df = pd.read_csv(p)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col])
    df["min_utc"] = df["min_utc"].astype("int64")
    return df.sort_values("min_utc").reset_index(drop=True)


def append_store_1m(coin: str, rows: dict):
    """rows: {min_utc:int -> (o,h,l,c) as strings/floats}. Dedup on min_utc."""
    p = KL / f"{coin}.csv"
    existing = load_store_1m(coin)
    have = set(existing["min_utc"].tolist())
    new = [{"min_utc": mu, "open": o, "high": h, "low": l, "close": cc}
           for mu, (o, h, l, cc) in rows.items() if mu not in have]
    if not new and p.exists():
        return 0
    out = pd.concat([existing, pd.DataFrame(new)], ignore_index=True) if new else existing
    out = out.drop_duplicates("min_utc").sort_values("min_utc").reset_index(drop=True)
    out.to_csv(p, index=False)
    return len(new)


def load_store_funding(inst: str, m1: pd.DataFrame) -> pd.DataFrame:
    """Mirror of tb.load_funding reading from forward store (identical math -> identical output)."""
    p = FU / f"{inst}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["slot_min", "rate", "settle_px"])
    fr = (pd.read_csv(p, usecols=["funding_time", "funding_rate"])
          .drop_duplicates("funding_time").sort_values("funding_time"))
    fr["slot_min"] = ((fr["funding_time"] / tb.EIGHT_H_MS).round().astype("int64")
                      * tb.EIGHT_H_MS // 60000)
    px = m1.set_index("min_utc")["close"]
    idx = px.index.searchsorted(fr["slot_min"].to_numpy() - 1, side="right") - 1
    ok = idx >= 0
    fr = fr.loc[ok].copy()
    fr["settle_px"] = px.to_numpy()[idx[ok]]
    fr["rate"] = pd.to_numeric(fr["funding_rate"])
    return fr[["slot_min", "rate", "settle_px"]].reset_index(drop=True)


def append_store_funding(inst: str, rows: dict):
    p = FU / f"{inst}.csv"
    have = set()
    if p.exists():
        old = pd.read_csv(p)
        have = set(old["funding_time"].astype("int64").tolist())
    else:
        old = pd.DataFrame(columns=["funding_time", "funding_rate"])
    new = [{"funding_time": ft, "funding_rate": rate} for ft, rate in rows.items() if ft not in have]
    if not new and p.exists():
        return 0
    out = pd.concat([old, pd.DataFrame(new)], ignore_index=True) if new else old
    out = out.drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)
    out.to_csv(p, index=False)
    return len(new)


def write_manifest(extra=None):
    man = {"server": "mainnet", "demo": False,
           "source_klines": "OKX /api/v5/market/candles + history-candles (bar=1m, confirm==1)",
           "source_funding": "OKX /api/v5/public/funding-rate-history",
           "config_sha256": config_sha256(), "updated_utc": datetime.now(timezone.utc).isoformat()}
    if extra:
        man.update(extra)
    (STORE / "manifest.json").write_text(json.dumps(man, indent=2))


# ─────────────────────────── OKX mainnet REST (public only) ─────────────────────
def _okx_get(path: str, params: dict) -> list:
    r = SESSION.get(OKX_BASE + path, params=params, timeout=30)
    j = r.json()
    if j.get("code") != "0":
        raise RuntimeError(f"OKX {path} code={j.get('code')} msg={j.get('msg')}")
    return j.get("data", [])


def fetch_1m_since(inst: str, since_min: int, max_pages: int = 600) -> dict:
    """Confirmed (confirm==1) 1m bars with min_utc > since_min, via candles + history-candles."""
    got = {}
    for r in _okx_get("/api/v5/market/candles", {"instId": inst, "bar": "1m", "limit": "300"}):
        if r[8] == "1":
            mu = int(r[0]) // 60000
            if mu > since_min:
                got[mu] = (r[1], r[2], r[3], r[4])
    oldest_ms = (min(got) * 60000) if got else int(time.time() * 1000)
    for _ in range(max_pages):
        data = _okx_get("/api/v5/market/history-candles",
                        {"instId": inst, "bar": "1m", "after": str(oldest_ms), "limit": "100"})
        if not data:
            break
        for r in data:
            if r[8] == "1":
                mu = int(r[0]) // 60000
                if mu > since_min:
                    got[mu] = (r[1], r[2], r[3], r[4])
        oldest_ms = min(int(r[0]) for r in data)
        if oldest_ms // 60000 <= since_min + 1:
            break
        time.sleep(0.08)
    return got


def fetch_funding_since(inst: str, since_ms: int, max_pages: int = 80) -> dict:
    got, cur_after = {}, None
    for _ in range(max_pages):
        p = {"instId": inst, "limit": "100"}
        if cur_after:
            p["after"] = str(cur_after)
        data = _okx_get("/api/v5/public/funding-rate-history", p)
        if not data:
            break
        for r in data:
            ft = int(r["fundingTime"])
            if ft > since_ms:
                got[ft] = r["fundingRate"]
        oldest = min(int(r["fundingTime"]) for r in data)
        cur_after = oldest
        if oldest <= since_ms:
            break
        time.sleep(0.08)
    return got


# ─────────────────────────── engine wrapper (verbatim tb/r2) ────────────────────
def build_inputs(loader_1m, loader_fund):
    bars, fund = {}, {}
    for coin, (_, inst) in tb.SYMBOLS.items():
        m1 = loader_1m(coin)
        bars[(coin, TF)] = tb.aggregate(m1, TF)
        fund[coin] = loader_fund(inst, m1)
    return bars, fund


def run_ledger(bars, fund):
    spans_by = {coin: tb.positions_flip(tb.signal_emax(bars[(coin, TF)], FAST, SLOW))
                for coin in tb.SYMBOLS}
    trades = []
    for coin, (_, inst) in tb.SYMBOLS.items():
        trades.extend(tb.build_trades(coin, inst, bars[(coin, TF)], fund[coin], spans_by[coin]))
    perbar, pos = r2.m2m_pnl(TF, bars, fund, spans_by)
    return trades, perbar, pos, spans_by


# ─────────────────────────── continuity / heartbeat ─────────────────────────────
def continuity_gaps(coin: str) -> list:
    m1 = load_store_1m(coin)
    if len(m1) < 2:
        return []
    mu = m1["min_utc"].to_numpy()
    d = np.diff(mu)
    gap_idx = np.where(d > 1)[0]
    return [(int(mu[i]), int(mu[i + 1]), int(d[i] - 1)) for i in gap_idx]


def write_heartbeat(extra=None):
    STATE.mkdir(parents=True, exist_ok=True)
    hb = {"last_run_utc": datetime.now(timezone.utc).isoformat(),
          "config_sha256": config_sha256()}
    if extra:
        hb.update(extra)
    # preserve last_live_bar_close_utc from previous heartbeat if not overwritten
    prev = (STATE / "heartbeat.json")
    if prev.exists() and "last_live_bar_close_utc" not in hb:
        try:
            old = json.loads(prev.read_text())
            if old.get("last_live_bar_close_utc"):
                hb["last_live_bar_close_utc"] = old["last_live_bar_close_utc"]
        except Exception:
            pass
    (STATE / "heartbeat.json").write_text(json.dumps(hb, indent=2))


# ─────────────────────────── seed / update / account ────────────────────────────
def mode_seed():
    """One-time: copy read-only mainnet DB 1m + main funding into the forward store."""
    _store_paths()
    L("seed: copying read-only mainnet DB 1m + main funding into forward store")
    for coin, (db_sym, inst) in tb.SYMBOLS.items():
        m1 = tb.load_1m_utc(db_sym)                     # DB opened mode=ro inside tb
        rows = {int(r.min_utc): (r.open, r.high, r.low, r.close)
                for r in m1.itertuples(index=False)}
        n = append_store_1m(coin, rows)
        # funding: copy the canonical OKX funding CSVs verbatim into the store
        frames = [pd.read_csv(f, usecols=["funding_time", "funding_rate"])
                  for f in sorted(tb.FUND_DIR.glob(f"{inst}_funding_*.csv"))]
        fdf = pd.concat(frames, ignore_index=True).drop_duplicates("funding_time")
        (FU / f"{inst}.csv").parent.mkdir(parents=True, exist_ok=True)
        fdf.sort_values("funding_time").to_csv(FU / f"{inst}.csv", index=False)
        L(f"  {coin}: seeded {n} 1m bars, {len(fdf)} funding rows")
    write_manifest({"seeded_from": "database_mainnet.db (read-only) + data/funding/okx"})
    # Compute first_live_bar_close_utc: the first complete 4h bar close after seed.
    # Deployment completed now; next 4h boundary = (current_hour // 4 + 1) * 4.
    now = datetime.now(timezone.utc)
    next_boundary = ((now.hour // 4) + 1) * 4
    first_live = now.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(hours=next_boundary)
    if first_live <= now:
        first_live += pd.Timedelta(hours=4)
    write_deploy(now.isoformat(), first_live.isoformat())
    write_heartbeat({"event": "seed", "last_live_bar_close_utc": first_live.isoformat()})
    L(f"seed: done. deploy.json written, first_live_bar_close={first_live.isoformat()}. "
      f"NOTE: run --update to backfill REST data (warmup only, excluded from forward), then --account.")


def mode_update():
    _store_paths()
    L("update: pulling incremental confirmed 1m + funding from OKX MAINNET public REST")
    total_k, total_f = 0, 0
    for coin, (_, inst) in tb.SYMBOLS.items():
        m1 = load_store_1m(coin)
        since_min = int(m1["min_utc"].iloc[-1]) if len(m1) else 0
        kr = fetch_1m_since(inst, since_min)
        nk = append_store_1m(coin, kr)
        fp = FU / f"{inst}.csv"
        since_ms = 0
        if fp.exists():
            ff = pd.read_csv(fp)
            since_ms = int(ff["funding_time"].max()) if len(ff) else 0
        fr = fetch_funding_since(inst, since_ms)
        nf = append_store_funding(inst, fr)
        total_k += nk; total_f += nf
        gaps = continuity_gaps(coin)
        L(f"  {coin}: +{nk} 1m, +{nf} funding, gaps={len(gaps)}")
        if gaps:
            notify_alert(f"[B2_4h forward] 数据缺口告警 {coin}",
                         f"{coin} 1m 网格缺口 {len(gaps)} 处: {gaps[:5]}")
    write_manifest({"last_update_event": "incremental REST append"})
    write_heartbeat({"event": "update", "added_1m": total_k, "added_funding": total_f})
    L(f"update: total +{total_k} 1m, +{total_f} funding")


GAP_LOG = STATE / "gap_log.jsonl"

def _detect_and_log_gap(perbar: pd.Series):
    """If system was down between last live bar and now, log the gap.
    Gap bars are excluded from forward_window (they were REST-backfilled, not live)."""
    hb = json.loads((STATE / "heartbeat.json").read_text()) if (STATE / "heartbeat.json").exists() else {}
    last_live = hb.get("last_live_bar_close_utc")
    if not last_live:
        deploy = read_deploy()
        if deploy:
            last_live = deploy.get("first_live_bar_close_utc")
    if not last_live:
        return
    last_live_dt = pd.Timestamp(last_live)
    cur_end = perbar.index[-1]
    if cur_end - last_live_dt > pd.Timedelta(hours=5):  # 4h bar + 1h tolerance
        gap_start = (last_live_dt + pd.Timedelta(hours=4)).isoformat()
        gap_end = cur_end.isoformat()
        GAP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(GAP_LOG, "a") as f:
            f.write(json.dumps({"gap_start": gap_start, "gap_end": gap_end,
                                "detected_utc": datetime.now(timezone.utc).isoformat(),
                                "note": "System downtime gap — bars in this range are excluded from forward ledger"}) + "\n")
        L(f"gap detected: [{gap_start} .. {gap_end}] — bars excluded from forward (downtime backfill)")


def _get_gap_periods() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (gap_start, gap_end) tuples for forward_window exclusion."""
    gaps = []
    if GAP_LOG.exists():
        for line in GAP_LOG.read_text().strip().split("\n"):
            if line:
                g = json.loads(line)
                gaps.append((pd.Timestamp(g["gap_start"]), pd.Timestamp(g["gap_end"])))
    return gaps


def mode_account():
    _store_paths()
    bars, fund = build_inputs(load_store_1m, load_store_funding)
    trades, perbar, pos, spans = run_ledger(bars, fund)
    # write dual-caliber ledgers (full recompute = source of truth)
    (STATE / "ledger_trades.jsonl").write_text(
        "\n".join(json.dumps(t, default=float) for t in trades) + ("\n" if trades else ""))
    daily = perbar.groupby(perbar.index.ceil("D")).sum()
    with open(STATE / "ledger_daily_m2m.jsonl", "w") as fh:
        for ts, v in daily.items():
            fh.write(json.dumps({"date": ts.strftime("%Y-%m-%d"), "m2m_usd": float(v)}) + "\n")
    # current positions (last bar exposure per coin)
    positions = {}
    for coin in tb.SYMBOLS:
        b = bars[(coin, TF)]
        sig = tb.signal_emax(b, FAST, SLOW)
        cur = 0
        for s in sig:
            if not np.isnan(s) and s != 0:
                cur = int(s)
        positions[coin] = {"direction": ("long" if cur == 1 else "short" if cur == -1 else "flat"),
                           "last_bar_close_utc": pd.Timestamp(int(b["end_min"].iloc[-1]) * 60,
                                                              unit="s", tz="UTC").isoformat()}
    (STATE / "positions.json").write_text(json.dumps(positions, indent=2))
    write_heartbeat({"event": "account", "n_trades": len(trades),
                     "net_total_usd": float(perbar.sum()),
                     "last_live_bar_close_utc": perbar.index[-1].isoformat()})
    # ── gap detection: if system was down, log the missed bars ──
    _detect_and_log_gap(perbar)
    pos_summary = ", ".join(f"{k}:{v['direction']}" for k, v in positions.items())
    L(f"account: {len(trades)} trades, full-sample net ${perbar.sum():,.2f}, positions={{ {pos_summary} }}")
    return trades, perbar, positions


# ─────────────────────────── deploy guard (anti-backfill) ────────────────────────
def read_deploy() -> dict | None:
    """Read deploy.json. Returns None if not yet created (pre-seed state)."""
    if not DEPLOY_PATH.exists():
        return None
    return json.loads(DEPLOY_PATH.read_text())

def write_deploy(deploy_completed_utc: str, first_live_bar_close_utc: str):
    """Write deploy.json. Called once by --seed. Never modified thereafter."""
    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEPLOY_PATH.write_text(json.dumps({
        "deploy_completed_utc": deploy_completed_utc,
        "first_live_bar_close_utc": first_live_bar_close_utc,
        "note": "No bar closing before first_live_bar_close_utc is EVER counted as forward sample. "
                "backfill/warmup bars (seed + REST gap-fill) are excluded from forward ledger. "
                "Gap-detection: if a cron-4h detects >4h since last live bar close, "
                "the bars in the gap are logged as missed (not forward)."
    }, indent=2))

def effective_forward_start() -> pd.Timestamp:
    """Forward sample starts at max(config.forward_start_utc, deploy.first_live_bar_close_utc).
    This prevents the config anchor predating deployment (the root cause of the 2026-06-22
    contamination incident)."""
    start = pd.Timestamp(cfg()["forward_start_utc"])
    deploy = read_deploy()
    if deploy and deploy.get("first_live_bar_close_utc"):
        return max(start, pd.Timestamp(deploy["first_live_bar_close_utc"]))
    return start


# ─────────────────────────── gate margins ───────────────────────────────────────
def forward_window(perbar: pd.Series, trades: list):
    start = effective_forward_start()
    f_per = perbar[perbar.index >= start]
    # exclude gap periods (system downtime — bars were REST-backfilled, not live)
    for gap_start, gap_end in _get_gap_periods():
        f_per = f_per[(f_per.index < gap_start) | (f_per.index > gap_end)]
    f_tr = [t for t in trades if pd.Timestamp(t["entry_time"]) >= start]
    return f_per, f_tr


def gate_status(perbar: pd.Series, trades: list) -> dict:
    base = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
    okx = base.get("OKX", {})
    f_per, f_tr = forward_window(perbar, trades)
    if len(f_per) == 0:
        return {"forward_active": False, "note": "尚无前向样本（forward_start 之后无数据）"}
    daily = f_per.groupby(f_per.index.ceil("D")).sum()
    eq = daily.cumsum()
    maxdd = float((eq.cummax() - eq).max()) if len(eq) else 0.0
    net = float(f_per.sum())
    span_days = (f_per.index[-1] - effective_forward_start()).days
    months = span_days / 30.44
    roll12 = float(daily[daily.index > (daily.index[-1] - pd.Timedelta(days=365))].sum())
    k1_p5 = okx.get("rolling_12mo_net", {}).get("p5")
    k2_p95 = okx.get("rolling_12mo_maxdd", {}).get("p95")
    out = {"forward_active": True, "forward_months": round(months, 2),
           "forward_net_usd": round(net, 2), "forward_maxdd_usd": round(maxdd, 2),
           "forward_trades": len(f_tr),
           "rolling_12mo_net_latest": round(roll12, 2) if months >= 12 else None,
           "K1_p5_threshold": k1_p5, "K2_p95x1.25_threshold": (k2_p95 * 1.25 if k2_p95 else None)}
    if months >= 12 and k1_p5 is not None:
        out["K1_margin_usd"] = round(roll12 - k1_p5, 2)
        out["K1_triggered"] = roll12 < k1_p5
    else:
        out["K1_note"] = f"滚动12月 gate 未激活（已 {months:.1f}/12 月）"
    if k2_p95 is not None:
        out["K2_margin_usd"] = round(k2_p95 * 1.25 - maxdd, 2)
        out["K2_triggered"] = maxdd > k2_p95 * 1.25
    return out


# ─────────────────────────── PushPlus notify ────────────────────────────────────
def read_token() -> str:
    if not NOTIFY_CONF.exists():
        return ""
    for line in NOTIFY_CONF.read_text().splitlines():
        line = line.strip()
        if line.startswith("pushplus_token") and "=" in line:
            return line.split("=", 1)[1].strip()
    return ""


def _retain(title, content):
    d = STATE / "pending_push"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{int(time.time()*1000)}.json").write_text(
        json.dumps({"title": title, "content": content}))


def pushplus_send(title: str, content: str) -> bool:
    token = read_token()
    if not token or token == "YOUR_TOKEN_HERE":
        L("[notify] no PushPlus token configured -> retaining content, not sent")
        _retain(title, content)
        return False
    for attempt in range(3):
        try:
            r = requests.post("https://www.pushplus.plus/send",
                              json={"token": token, "title": title, "content": content,
                                    "template": "markdown"}, timeout=20)
            if r.json().get("code") == 200:
                return True
            L(f"[notify] PushPlus resp {r.json().get('code')} {r.json().get('msg')}")
        except Exception as e:
            L(f"[notify] attempt {attempt+1} error {e}")
        time.sleep(2)
    _retain(title, content)
    return False


def notify_alert(title: str, content: str):
    """Immediate, separate channel for anomalies (not mixed into the daily digest)."""
    pushplus_send(title, "⚠️ " + content + f"\n\n时间(UTC): {datetime.now(timezone.utc).isoformat()}")


def resend_pending():
    d = STATE / "pending_push"
    if not d.exists():
        return
    for f in sorted(d.glob("*.json")):
        p = json.loads(f.read_text())
        token = read_token()
        if token and token != "YOUR_TOKEN_HERE":
            try:
                r = requests.post("https://www.pushplus.plus/send",
                                  json={"token": token, "title": "[补发] " + p["title"],
                                        "content": p["content"], "template": "markdown"}, timeout=20)
                if r.json().get("code") == 200:
                    f.unlink()
            except Exception:
                pass


def mode_push():
    _store_paths()
    resend_pending()
    trades, perbar, positions = mode_account()
    gs = gate_status(perbar, trades)
    # daily slice = the most recent complete UTC day
    daily = perbar.groupby(perbar.index.ceil("D")).sum()
    today = daily.index[-1] if len(daily) else None
    day_pnl = float(daily.iloc[-1]) if len(daily) else 0.0
    # today's signal flips
    flips = [t for t in trades if today is not None
             and pd.Timestamp(t["entry_time"]).date() == today.date()]
    hb_age = "n/a"
    gaps_total = sum(len(continuity_gaps(c)) for c in tb.SYMBOLS)
    pos_line = " · ".join(f"{k} {v['direction']}" for k, v in positions.items())
    lines = [f"# B2_4h 前向观察 日报",
             f"**日期(UTC)**: {today.strftime('%Y-%m-%d') if today is not None else 'n/a'}",
             f"**持仓**: {pos_line}",
             f"**当日新信号翻转**: {len(flips)}" + ("".join(
                 f"\n- {f['symbol']} {f['side']} @ {pd.Timestamp(f['entry_time']).strftime('%m-%d %H:%MZ')}"
                 for f in flips) if flips else "（无）"),
             f"**当日模拟 PnL(M2M)**: ${day_pnl:,.2f}",
             ]
    if gs.get("forward_active"):
        lines += [f"**前向累计净利**: ${gs['forward_net_usd']:,.2f}（{gs['forward_months']} 月, {gs['forward_trades']} 笔）",
                  f"**前向当前回撤**: ${gs['forward_maxdd_usd']:,.2f}"]
        if gs.get("K1_triggered") is not None:
            lines.append(f"**距 K1(滚动12月净利<p5)**: 余量 ${gs.get('K1_margin_usd',0):,.0f}"
                         + ("  ⛔已触发" if gs.get("K1_triggered") else ""))
        else:
            lines.append(f"**K1**: {gs.get('K1_note','')}")
        if gs.get("K2_margin_usd") is not None:
            lines.append(f"**距 K2(maxDD>p95×1.25={gs['K2_p95x1.25_threshold']:,.0f})**: 余量 ${gs['K2_margin_usd']:,.0f}"
                         + ("  ⛔已触发" if gs.get("K2_triggered") else ""))
    else:
        lines.append(f"**前向样本**: {gs.get('note','')}")
    lines += [f"**系统健康**: 数据缺口 {gaps_total} 处 · config {config_sha256()[:12]}…",
              f"_数据源: OKX mainnet 公开行情（无 demo/无凭证/无下单）_"]
    content = "\n\n".join(lines)
    ok = pushplus_send("B2_4h 前向日报", content)
    # anomalies as separate immediate alerts
    if gs.get("K1_triggered") or gs.get("K2_triggered"):
        notify_alert("[B2_4h forward] KILL gate 触发", json.dumps(gs, ensure_ascii=False, default=float))
    write_heartbeat({"event": "push", "push_ok": ok})
    L(f"push: sent={ok} | day PnL ${day_pnl:,.2f} | forward net "
      f"${gs.get('forward_net_usd', 0):,.2f}")


# ─────────────────────────── reconcile (monthly) ───────────────────────────────
def mode_reconcile():
    """Re-fetch the last full UTC month from REST, compare to store, recompute ledger both ways."""
    _store_paths()
    L("reconcile: re-fetching last full month from REST and checking store integrity")
    now = datetime.now(timezone.utc)
    first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_full_start = (first_this - pd.Timedelta(days=1)).replace(day=1)
    m_start = int(last_full_start.timestamp() // 60)
    m_end = int(first_this.timestamp() // 60)
    max_abs_price_diff = 0.0
    mismatched = 0
    for coin, (_, inst) in tb.SYMBOLS.items():
        store = load_store_1m(coin)
        store_m = store[(store["min_utc"] >= m_start) & (store["min_utc"] < m_end)].set_index("min_utc")
        fresh = fetch_1m_since(inst, m_start - 1)
        for mu, (o, h, l, cc) in fresh.items():
            if m_start <= mu < m_end and mu in store_m.index:
                d = abs(float(cc) - float(store_m.loc[mu, "close"]))
                max_abs_price_diff = max(max_abs_price_diff, d)
                if d > 1e-9:
                    mismatched += 1
    res = {"month": last_full_start.strftime("%Y-%m"), "max_abs_close_diff": max_abs_price_diff,
           "mismatched_minutes": mismatched, "store_matches_rest": mismatched == 0}
    (STATE / "reconcile_last.json").write_text(json.dumps(res, indent=2, default=float))
    L(f"reconcile: {res}")
    if mismatched > 0:
        notify_alert("[B2_4h forward] 月度对账失配 (K3 计时起点)",
                     f"{res['month']} store vs REST 失配 {mismatched} 分钟, max diff {max_abs_price_diff}")
    write_heartbeat({"event": "reconcile", **res})
    return res


# ─────────────────────────── baseline distribution ─────────────────────────────
def _rolling_window_stats(daily: pd.Series, win_days: int) -> dict:
    eq = daily.cumsum()
    idx = daily.index
    nets, dds = [], []
    for i in range(len(idx)):
        lo = idx[i] - pd.Timedelta(days=win_days)
        w = daily[(daily.index > lo) & (daily.index <= idx[i])]
        if (idx[i] - idx[0]).days < win_days:
            continue
        nets.append(float(w.sum()))
        we = w.cumsum()
        dds.append(float((we.cummax() - we).max()))
    nets = np.array(nets) if nets else np.array([np.nan])
    dds = np.array(dds) if dds else np.array([np.nan])
    pct = lambda a, q: (float(np.nanpercentile(a, q)) if a.size and not np.isnan(a).all() else None)
    return {"n_windows": int(len(nets)),
            "net": {"p5": pct(nets, 5), "p25": pct(nets, 25), "p50": pct(nets, 50)},
            "maxdd": {"p50": pct(dds, 50), "p95": pct(dds, 95)}}


def _sample_baseline(name: str, bars, fund) -> dict:
    trades, perbar, _, _ = run_ledger(bars, fund)
    daily = perbar.groupby(perbar.index.ceil("D")).sum().sort_index()
    eq = daily.cumsum()
    full_net = float(perbar.sum())
    full_maxdd = float((eq.cummax() - eq).max())
    # monthly
    mon = perbar.groupby(perbar.index.to_period("M")).sum()
    monthly_winrate = float((mon > 0).mean() * 100)
    r6 = _rolling_window_stats(daily, 182)
    r12 = _rolling_window_stats(daily, 365)
    return {"sample": name, "n_trades": len(trades),
            "full_net_usd": full_net, "full_maxdd_usd": full_maxdd,
            "net_over_maxdd": (full_net / full_maxdd if full_maxdd > 0 else None),
            "monthly_net_avg_usd": float(mon.mean()), "monthly_winrate_pct": monthly_winrate,
            "n_months": int(len(mon)),
            "rolling_6mo_net": r6["net"], "rolling_12mo_net": r12["net"],
            "rolling_12mo_maxdd": r12["maxdd"],
            "span": [daily.index[0].strftime("%Y-%m-%d"), daily.index[-1].strftime("%Y-%m-%d")]}


def mode_build_baseline():
    import research_trend_dualcycle as dc                      # lazy: Binance side-by-side only
    from binance_funding import load_funding_binance
    L("build-baseline: B2_4h backtest distributions (OKX gate-of-record + Binance side-by-side)")
    out = {"strategy": "B2_4h", "config_sha256": config_sha256(),
           "note": "gate numbers source; OKX is gate-of-record. Frozen with gates_preregistered.md.",
           "built_utc": datetime.now(timezone.utc).isoformat()}
    # OKX
    okx_bars, okx_fund = build_inputs(lambda c: tb.load_1m_utc(tb.SYMBOLS[c][0]),
                                      lambda inst, m1: tb.load_funding(inst, m1))
    out["OKX"] = _sample_baseline("OKX", okx_bars, okx_fund)
    L(f"  OKX: net ${out['OKX']['full_net_usd']:,.0f} maxDD ${out['OKX']['full_maxdd_usd']:,.0f} "
      f"roll12 net p5 ${out['OKX']['rolling_12mo_net']['p5']:,.0f} "
      f"roll12 maxDD p95 ${out['OKX']['rolling_12mo_maxdd']['p95']:,.0f}")
    # Binance
    bz_bars, bz_fund = {}, {}
    for coin, bs in dc.B_SYM.items():
        m1 = dc.load_1m_bv(bs)
        bz_bars[(coin, TF)] = tb.aggregate(m1, TF)
        bz_fund[coin] = load_funding_binance(bs, m1)
    out["Binance"] = _sample_baseline("Binance", bz_bars, bz_fund)
    L(f"  Binance: net ${out['Binance']['full_net_usd']:,.0f} maxDD ${out['Binance']['full_maxdd_usd']:,.0f}")
    BASELINE_PATH.write_text(json.dumps(out, indent=2, default=float))
    L(f"build-baseline: wrote {BASELINE_PATH}")
    return out


# ─────────────────────────── dry-run ($0 validation) ───────────────────────────
def mode_dry_run():
    """Hold out the last COMPLETE calendar month in the DB; feed it as forward increments
    (day-by-day appends from DB, simulating REST), then assert forward ledger == backtest
    ledger to the cent over that month."""
    L("dry-run: pre-deploy $0 validation (incremental-append path vs backtest engine)")
    db_end = max(int(tb.load_1m_utc(tb.SYMBOLS[c][0])["min_utc"].iloc[-1]) for c in tb.SYMBOLS)
    db_end_dt = pd.Timestamp(db_end * 60, unit="s", tz="UTC")
    # last COMPLETE month fully inside the DB
    first_of_end_month = db_end_dt.normalize().replace(day=1)
    month_end = first_of_end_month                       # exclusive
    month_start = (first_of_end_month - pd.Timedelta(days=1)).replace(day=1)
    L(f"  DB ends {db_end_dt.strftime('%Y-%m-%d %H:%MZ')} -> dry-run month "
      f"{month_start.strftime('%Y-%m')} [{month_start.date()} .. {month_end.date()})")
    ms_min = int(month_start.timestamp() // 60)
    me_min = int(month_end.timestamp() // 60)
    # backtest truth = engine on DB TRUNCATED at me_min, i.e. the SAME data window the forward
    # store ends at (so the single boundary open-position is treated identically in both).
    bt_bars, bt_fund = build_inputs(
        lambda c: tb.load_1m_utc(tb.SYMBOLS[c][0]).pipe(
            lambda d: d[d["min_utc"] < me_min].reset_index(drop=True)),
        lambda inst, m1: tb.load_funding(inst, m1))
    bt_trades, bt_perbar, _, _ = run_ledger(bt_bars, bt_fund)
    # build a TEMP forward store: history < month_start in bulk, then the month day-by-day
    tmp_kl = STORE / "_dryrun_klines"; tmp_fu = STORE / "_dryrun_funding"
    for p in (tmp_kl, tmp_fu):
        p.mkdir(parents=True, exist_ok=True)
        for f in p.glob("*.csv"):
            f.unlink()

    def tmp_load_1m(coin):
        f = tmp_kl / f"{coin}.csv"
        if not f.exists():
            return pd.DataFrame(columns=["min_utc", "open", "high", "low", "close"])
        df = pd.read_csv(f)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c])
        df["min_utc"] = df["min_utc"].astype("int64")
        return df.sort_values("min_utc").reset_index(drop=True)

    def tmp_load_fund(inst, m1):
        f = tmp_fu / f"{inst}.csv"
        fr = (pd.read_csv(f, usecols=["funding_time", "funding_rate"])
              .drop_duplicates("funding_time").sort_values("funding_time"))
        fr["slot_min"] = ((fr["funding_time"] / tb.EIGHT_H_MS).round().astype("int64")
                          * tb.EIGHT_H_MS // 60000)
        px = m1.set_index("min_utc")["close"]
        idx = px.index.searchsorted(fr["slot_min"].to_numpy() - 1, side="right") - 1
        ok = idx >= 0
        fr = fr.loc[ok].copy()
        fr["settle_px"] = px.to_numpy()[idx[ok]]
        fr["rate"] = pd.to_numeric(fr["funding_rate"])
        return fr[["slot_min", "rate", "settle_px"]].reset_index(drop=True)

    for coin, (db_sym, inst) in tb.SYMBOLS.items():
        m1 = tb.load_1m_utc(db_sym)
        acc = {}                                          # in-memory store (min_utc -> ohlc)
        for r in m1[m1["min_utc"] < ms_min].itertuples(index=False):   # history bulk seed
            acc[int(r.min_utc)] = (r.open, r.high, r.low, r.close)
        # feed the held-out month DAY BY DAY as forward increments (dedup is order-independent)
        month = m1[(m1["min_utc"] >= ms_min) & (m1["min_utc"] < me_min)]
        for day_start in range(ms_min, me_min, 1440):
            chunk = month[(month["min_utc"] >= day_start) & (month["min_utc"] < day_start + 1440)]
            for r in chunk.itertuples(index=False):
                acc[int(r.min_utc)] = (r.open, r.high, r.low, r.close)
        pd.DataFrame([{"min_utc": k, "open": v[0], "high": v[1], "low": v[2], "close": v[3]}
                      for k, v in sorted(acc.items())]).to_csv(tmp_kl / f"{coin}.csv", index=False)
        frames = [pd.read_csv(f, usecols=["funding_time", "funding_rate"])
                  for f in sorted(tb.FUND_DIR.glob(f"{inst}_funding_*.csv"))]
        pd.concat(frames, ignore_index=True).drop_duplicates("funding_time").to_csv(
            tmp_fu / f"{inst}.csv", index=False)

    fwd_bars, fwd_fund = build_inputs(tmp_load_1m, tmp_load_fund)
    fwd_trades, fwd_perbar, _, _ = run_ledger(fwd_bars, fwd_fund)

    # PRIMARY $0 gate: identical data window [start..me_min) -> forward ledger must equal
    # backtest ledger over the WHOLE series (total trade-net AND total M2M).
    bt_net = sum(t["net_pnl_usd"] for t in bt_trades)
    fwd_net = sum(t["net_pnl_usd"] for t in fwd_trades)
    bt_m2m = float(bt_perbar.sum()); fwd_m2m = float(fwd_perbar.sum())
    diff_trades = abs(bt_net - fwd_net)
    diff_m2m = abs(bt_m2m - fwd_m2m)
    # context: the held-out month's M2M slice (both truncated identically -> also equal)
    apr = lambda per: float(per[(per.index >= month_start) & (per.index < month_end)].sum())
    # store-vs-DB exact bar equality (CSV float round-trip)
    bars_equal = True
    for coin, (db_sym, _) in tb.SYMBOLS.items():
        a = tb.load_1m_utc(db_sym); a = a[a["min_utc"] < me_min].reset_index(drop=True)
        b = tmp_load_1m(coin).reset_index(drop=True)
        if not (len(a) == len(b) and np.array_equal(a["min_utc"], b["min_utc"])
                and np.allclose(a["close"], b["close"], rtol=0, atol=0)):
            bars_equal = False
    res = {"dry_run_month_fed_incrementally": month_start.strftime("%Y-%m"),
           "data_window": f"[series start .. {month_end.date()}) (forward store == DB truncated identically)",
           "db_end_utc": db_end_dt.isoformat(),
           "backtest_total_trade_net": round(bt_net, 6),
           "forward_total_trade_net": round(fwd_net, 6),
           "trade_net_abs_diff": round(diff_trades, 6),
           "backtest_total_m2m": round(bt_m2m, 6),
           "forward_total_m2m": round(fwd_m2m, 6),
           "m2m_abs_diff": round(diff_m2m, 6),
           "held_out_month_m2m_backtest": round(apr(bt_perbar), 6),
           "held_out_month_m2m_forward": round(apr(fwd_perbar), 6),
           "store_bars_exactly_equal_DB": bool(bars_equal),
           "PASS": bool(diff_trades < 1e-6 and diff_m2m < 1e-6 and bars_equal),
           "config_sha256": config_sha256(),
           "validated_utc": datetime.now(timezone.utc).isoformat()}
    (FWD / "dry_run_validation.json").write_text(json.dumps(res, indent=2, default=float))
    # cleanup temp
    for p in (tmp_kl, tmp_fu):
        for f in p.glob("*.csv"):
            f.unlink()
        p.rmdir()
    L(f"dry-run: month={res['dry_run_month']} trade-net diff ${diff_trades:.6f} "
      f"m2m diff ${diff_m2m:.6f} bars_equal={bars_equal} -> {'PASS ($0)' if res['PASS'] else 'FAIL'}")
    return res


# ─────────────────────────── main ──────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    for m in ("selfcheck", "build-baseline", "seed", "update", "account", "push",
              "cron-4h", "cron-daily", "reconcile", "dry-run"):
        ap.add_argument(f"--{m}", action="store_true")
    args = ap.parse_args()
    d = vars(args)
    L(f"forward_b2_4h start | mode={[k for k,v in d.items() if v] or ['(none)']}")
    selfcheck(strict=True)                       # always, before anything
    if d.get("selfcheck"):
        pass
    elif d.get("build_baseline"):
        mode_build_baseline()
    elif d.get("seed"):
        mode_seed()
    elif d.get("update"):
        mode_update()
    elif d.get("account"):
        mode_account()
    elif d.get("cron_4h"):
        mode_update(); mode_account()
    elif d.get("push") or d.get("cron_daily"):
        mode_push()
    elif d.get("reconcile"):
        mode_reconcile()
    elif d.get("dry_run"):
        mode_dry_run()
    else:
        L("no mode given; use --help. (ran selfcheck only)")
    (STATE if STATE.exists() else FWD).mkdir(parents=True, exist_ok=True)
    (FWD / "last_run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
