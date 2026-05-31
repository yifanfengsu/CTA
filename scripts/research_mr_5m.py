#!/usr/bin/env python3
"""MR-5m: Quick parameter sweep on BTC only, then full-5 validation on top params."""

from __future__ import annotations

import argparse, json, logging, re, sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging, to_jsonable
from history_time_utils import HistoryRange, parse_history_range

# Simplified for speed
DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL", "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL", "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_START, DEFAULT_END = "2023-01-01", "2026-03-31"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_5m"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

FIXED_NOTIONAL, ATR_PERIOD = 1000.0, 14
FEE_TAKER, FEE_MAKER, SLIPPAGE = 5.0, 2.0, 2.0

GRID_LOOKBACK = [12, 24]
GRID_ATR_STOP = [0.5, 1.0]
GRID_MAX_HOLD = [24, 36, 48]

# ------ data ------
def split_vt_symbol(vt): 
    s,_,e=str(vt).partition("."); return s,e
def symbol_to_inst_id(vt):
    s,_=split_vt_symbol(vt); r=s.removesuffix("_OKX"); p=r[:-len("_SWAP")] if r.endswith("_SWAP") else r
    return f"{p[:-4]}-USDT-SWAP" if p.endswith("USDT") else r.replace("_","-")
def norm(df,tz):
    c=["datetime","open","high","low","close","volume"]
    if df.empty: return pd.DataFrame(columns=c)
    o=df.loc[:,c].copy()
    ts=pd.to_datetime(o["datetime"],errors="coerce")
    o["datetime"]=ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    for cc in ["open","high","low","close","volume"]:
        o[cc]=pd.to_numeric(o[cc],errors="coerce")
    return o.dropna(subset=c).sort_values("datetime",kind="stable").drop_duplicates("datetime",keep="last").reset_index(drop=True)

def load_1m(sym,hr,db):
    s,e=split_vt_symbol(sym); qs=hr.start.replace(tzinfo=None).isoformat(sep=" ",timespec="seconds")
    qe=hr.end_exclusive.replace(tzinfo=None).isoformat(sep=" ",timespec="seconds")
    with sqlite3.connect(db) as conn:
        df=pd.read_sql_query("select datetime,open_price as open,high_price as high,low_price as low,close_price as close,volume from dbbardata where symbol=? and exchange=? and interval='1m' and datetime>=? and datetime<? order by datetime",conn,params=(s,e,qs,qe))
    return norm(df,hr.timezone_name)

def r5(df, minutes, hr):
    c=["open_time","datetime","open","high","low","close","volume"]
    if df.empty: return pd.DataFrame(columns=c)
    w=df.sort_values("datetime",kind="stable").drop_duplicates("datetime",keep="last").copy()
    anchor=pd.Timestamp(hr.start)
    if anchor.tzinfo is None: anchor=anchor.tz_localize(w["datetime"].iloc[0].tz)
    deltas=(w["datetime"]-anchor)/pd.Timedelta(minutes=1)
    w=w.loc[deltas>=0].copy()
    w["_s"]=np.floor(deltas.loc[w.index].to_numpy(dtype=float)/minutes).astype(np.int64)
    g=w.groupby("_s",sort=True,dropna=False)
    r=g.agg(open_time=("datetime","min"),datetime=("datetime","max"),open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum"),mc=("datetime","size"))
    return r[r["mc"]>=max(1,int(minutes*0.75))].drop(columns=["mc"]).dropna(subset=["open","high","low","close"]).reset_index(drop=True).loc[:,c]

def load_funding(path):
    df=pd.read_csv(path)
    if df.empty: return pd.DataFrame(columns=["funding_time_utc","funding_rate"])
    k="funding_time_utc" if "funding_time_utc" in df.columns else "funding_time"
    ts=pd.to_datetime(df[k],utc=True,errors="coerce") if k=="funding_time_utc" else pd.to_datetime(pd.to_numeric(df[k],errors="coerce"),unit="ms",utc=True,errors="coerce")
    r=pd.DataFrame({"funding_time_utc":ts,"funding_rate":pd.to_numeric(df.get("funding_rate"),errors="coerce")})
    return r.dropna().sort_values("funding_time_utc",kind="stable").drop_duplicates("funding_time_utc",keep="last").reset_index(drop=True)

# ------ backtest ------
@dataclass
class Trade:
    entry_time: pd.Timestamp; exit_time: pd.Timestamp; direction: int
    entry_price: float; exit_price: float; exit_reason: str; symbol: str

def atr_series(bars, period):
    h,l,c=bars["high"],bars["low"],bars["close"]
    tr=pd.concat([(h-l).abs(),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bt(bars, sym, lb, atr_s, mh):
    n=len(bars)
    if n<lb+5: return []
    atr=atr_series(bars,ATR_PERIOD)
    dh=bars["high"].rolling(lb).max().shift(1); dl=bars["low"].rolling(lb).min().shift(1)
    cl=bars["close"]
    trades=[]; pos=0; eb=-1; ep=0.0; ea=0.0
    for i in range(lb+1,n):
        bar=bars.iloc[i]
        if pos!=0:
            hb=i-eb; ew=False; r=""
            d_h=dh.iloc[i] if i<len(dh) and not pd.isna(dh.iloc[i]) else 0
            d_l=dl.iloc[i] if i<len(dl) and not pd.isna(dl.iloc[i]) else 0
            if d_h>0 and d_l>0:
                mid=(d_h+d_l)/2
                if (pos==1 and bar["close"]>=mid) or (pos==-1 and bar["close"]<=mid): ew=True; r="midline"
            if not ew:
                sd=atr_s*ea
                if pos==1:
                    if bar["low"]<=ep-sd: ew=True; r="stop"
                else:
                    if bar["high"]>=ep+sd: ew=True; r="stop"
            if hb>=mh and not ew: ew=True; r="max_hold"
            if ew:
                if r=="stop":
                    xp=min(bar["open"],ep-sd) if pos==1 else max(bar["open"],ep+sd)
                else: xp=bar["close"]
                trades.append(Trade(bars["datetime"].iloc[eb],bar["datetime"],pos,ep,xp,r,sym))
                pos=0; continue
        if pos==0:
            h=dh.iloc[i]; l=dl.iloc[i]; c=cl.iloc[i]
            if pd.isna(h) or pd.isna(l): continue
            if c>h: pos=-1; eb=i; ep=c; ea=atr.iloc[i] if i<len(atr) and not pd.isna(atr.iloc[i]) and atr.iloc[i]>0 else c*0.005
            elif c<l: pos=1; eb=i; ep=c; ea=atr.iloc[i] if i<len(atr) and not pd.isna(atr.iloc[i]) and atr.iloc[i]>0 else c*0.005
    if pos!=0:
        last=bars.iloc[-1]; trades.append(Trade(bars["datetime"].iloc[eb],last["datetime"],pos,ep,last["close"],"end_of_data",sym))
    return trades

def metrics(trades, fmap, tz, fee_bps, slp_bps):
    if not trades: return {"total_trades":0,"pnl":0,"win_rate":0}
    cpt=FIXED_NOTIONAL*(2*fee_bps+2*slp_bps)/10000.0
    import zoneinfo; tz_obj=zoneinfo.ZoneInfo(tz)
    rec=[]
    for t in trades:
        ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
        nc=ret*FIXED_NOTIONAL; ca=nc-cpt
        et=t.entry_time.tz_convert("UTC") if t.entry_time.tzinfo else t.entry_time.tz_localize(tz_obj).tz_convert("UTC")
        xt=t.exit_time.tz_convert("UTC") if t.exit_time.tzinfo else t.exit_time.tz_localize(tz_obj).tz_convert("UTC")
        inst=symbol_to_inst_id(t.symbol); f=fmap.get(inst); fp=0.0
        if f is not None and not f.empty:
            m=(f["funding_time_utc"]>=et)&(f["funding_time_utc"]<xt)
            if m.any(): s=f.loc[m,"funding_rate"].sum()*FIXED_NOTIONAL; fp=s if t.direction==1 else -s
        rec.append({"reason":t.exit_reason,"pnl":ca-fp})
    df=pd.DataFrame(rec)
    wins=df[df["pnl"]>0]["pnl"].sum(); losses=abs(df[df["pnl"]<0]["pnl"].sum())
    pf=float(wins/losses) if losses>0 else float("inf")
    wr=float((df["pnl"]>0).mean())*100 if len(df)>0 else 0
    eb={}
    for rsn,grp in df.groupby("reason"): eb[rsn]={"count":len(grp),"pnl":float(grp["pnl"].sum())}
    return {"total_trades":len(df),"pnl":float(df["pnl"].sum()),"pf":round(pf,2),"win_rate":round(wr,1),"by_exit":eb}

def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--start",default=DEFAULT_START); p.add_argument("--end",default=DEFAULT_END)
    p.add_argument("--timezone",default=DEFAULT_TIMEZONE)
    p.add_argument("--funding-dir",default=str(DEFAULT_FUNDING_DIR))
    p.add_argument("--output-dir",default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--database-path",default=str(DEFAULT_DATABASE_PATH))
    p.add_argument("--full",action="store_true",help="Run full 5-symbol validation on top params")
    return p.parse_args(argv)

def main(argv=None):
    ensure_headless_runtime()
    args=parse_args(argv)
    logger=setup_logging("mr5m",verbose=True)
    syms=[s.strip() for s in re.split(r"[\s,]+",args.symbols) if s.strip()]
    out=Path(args.output_dir); db=Path(args.database_path); fdir=Path(args.funding_dir)
    hr=parse_history_range(args.start,args.end,timedelta(minutes=1),args.timezone)

    fmap={}
    for sym in syms:
        inst=symbol_to_inst_id(sym); cf=fdir/f"{inst}_funding_{DEFAULT_START}_{DEFAULT_END}.csv"
        if cf.exists(): fmap[inst]=load_funding(cf)
        else:
            ms=sorted(fdir.glob(f"{inst}_funding_*.csv"))
            if ms: fmap[inst]=load_funding(ms[-1])

    # Phase 1: BTC only sweep
    btc_sym = syms[0]
    print(f"Phase 1: BTC sweep ({btc_sym})")
    bars_1m=load_1m(btc_sym,hr,db); bars5=r5(bars_1m,5,hr)
    print(f"  {len(bars5):,} bars loaded")

    grid=[(lb,atr_s,mh) for lb in GRID_LOOKBACK for atr_s in GRID_ATR_STOP for mh in GRID_MAX_HOLD]
    results=[]
    for lb,atr_s,mh in grid:
        trades=bt(bars5,btc_sym,lb,atr_s,mh)
        m_maker=metrics(trades,fmap,args.timezone,FEE_MAKER,SLIPPAGE)
        m_taker=metrics(trades,fmap,args.timezone,FEE_TAKER,SLIPPAGE)
        results.append({"lookback":lb,"atr_stop":atr_s,"max_hold":mh,
            "trades":m_maker["total_trades"],
            "pnl_maker":m_maker["pnl"],"pnl_taker":m_taker["pnl"],
            "wr_maker":m_maker["win_rate"],"wr_taker":m_taker["win_rate"],
            "stop_t":m_maker.get("by_exit",{}).get("stop",{}).get("count",0),
            "stop_p_m":m_maker.get("by_exit",{}).get("stop",{}).get("pnl",0),
            "mid_t":m_maker.get("by_exit",{}).get("midline",{}).get("count",0),
            "mid_p_m":m_maker.get("by_exit",{}).get("midline",{}).get("pnl",0),
        })

    results.sort(key=lambda x: x["pnl_maker"],reverse=True)
    print(f"\n{'='*110}")
    print(f"BTC sweep: {len(grid)} combos")
    print(f"{'LB':>3} {'ATR':>4} {'MH':>3} {'Trades':>7} {'PnL_M':>10} {'PnL_T':>10} {'WR_M':>6} {'WR_T':>6} {'Stop_T':>7} {'Stop$':>10} {'Mid_T':>7} {'Mid$':>10}")
    print("-"*110)
    for r in results:
        print(f"{r['lookback']:>3} {r['atr_stop']:>4.1f} {r['max_hold']:>3} {r['trades']:>7,} ${r['pnl_maker']:>9,.0f} ${r['pnl_taker']:>9,.0f} {r['wr_maker']:>5.1f}% {r['wr_taker']:>5.1f}% {r['stop_t']:>7} ${r['stop_p_m']:>10,.0f} {r['mid_t']:>7} ${r['mid_p_m']:>10,.0f}")

    # Top 3 for full validation
    top3=results[:3]
    print(f"\nPhase 2: Full 5-symbol validation on top 3 params")

    # Load all 5 symbols
    all5={}
    for sym in syms[1:]:
        b1=load_1m(sym,hr,db); b5=r5(b1,5,hr)
        if not b5.empty: all5[sym]=b5; print(f"  {sym.split('_')[0]}: {len(b5):,} bars")
    all5[syms[0]]=bars5

    for i,r in enumerate(top3):
        all_t=[]
        for sym,bars in all5.items():
            all_t.extend(bt(bars,sym,r["lookback"],r["atr_stop"],r["max_hold"]))
        mm=metrics(all_t,fmap,args.timezone,FEE_MAKER,SLIPPAGE)
        mt=metrics(all_t,fmap,args.timezone,FEE_TAKER,SLIPPAGE)
        print(f"\n  #{i+1}: LB={r['lookback']}, ATR={r['atr_stop']}, MH={r['max_hold']}")
        print(f"    MAKER: PnL=${mm['pnl']:,.0f}, Trades={mm['total_trades']:,}, WR={mm['win_rate']:.1f}%, PF={mm['pf']}")
        print(f"    TAKER: PnL=${mt['pnl']:,.0f}, Trades={mt['total_trades']:,}, WR={mt['win_rate']:.1f}%, PF={mt['pf']}")
        for rsn,info in mm.get("by_exit",{}).items():
            print(f"    {rsn:>10}: {info['count']:>6,} trades, ${info['pnl']:>10,.0f}")

    print(f"\n--- vs 4h Baseline ---")
    print(f"  4h MR-v1.2: PnL=+$4,048, Trades=2,216, WR=43.3%, Maker OK")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
