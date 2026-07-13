#!/usr/bin/env python3
"""MR-5m Fix Script: weight balancing + ATR regime filter + re-test.

Fixes:
  1. SOL weight cap: scale PnL per trade based on symbol avg
  2. ATR regime filter: skip trades when market ATR < threshold
  3. Combined: both applied
"""

import sys; sys.path.insert(0,'scripts')
# 2026-07 重构批次6：脚本迁入 _archive/mr5m_runner/；共享依赖真身在
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

from research_mr_5m import *
from datetime import timedelta
import numpy as np

args=parse_args()
hr=parse_history_range(args.start,args.end,timedelta(minutes=1),args.timezone)
db=Path(args.database_path); fdir=Path(args.funding_dir)
import zoneinfo; tz=zoneinfo.ZoneInfo(args.timezone)

fmap={}
for sym in DEFAULT_SYMBOLS:
    inst=symbol_to_inst_id(sym); cf=fdir/f'{inst}_funding_{DEFAULT_START}_{DEFAULT_END}.csv'
    if cf.exists(): fmap[inst]=load_funding(cf)
    else:
        ms=sorted(fdir.glob(f'{inst}_funding_*.csv'))
        if ms: fmap[inst]=load_funding(ms[-1])

bars_map={}
for sym in DEFAULT_SYMBOLS:
    b1=load_1m(sym,hr,db); b5=r5(b1,5,hr)
    if not b5.empty: bars_map[sym]=b5

LB,ATR_V,MH=24,1.0,48

# ============================================================
# Fix 1: Per-symbol weight balancing
# ============================================================
def compute_weight_scales(sym_trades, weight_mult=1.5):
    """Compute per-symbol PnL scaling factors."""
    sym_avg={}
    for sym in sym_trades:
        pnls=[]
        for t in sym_trades[sym]:
            ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
            pnls.append(ret*FIXED_NOTIONAL)
        sym_avg[sym]=np.mean(pnls) if pnls else 0

    scales={}
    for sym in DEFAULT_SYMBOLS:
        others=[v for s,v in sym_avg.items() if s!=sym]
        if not others: scales[sym]=1.0; continue
        avg_o=np.mean(others)
        if avg_o>0 and sym_avg[sym]>weight_mult*avg_o:
            scales[sym]=weight_mult*avg_o/max(sym_avg[sym],0.001)
        else: scales[sym]=1.0
    return scales

# ============================================================
# Fix 2: ATR regime filter
# ============================================================
def compute_atr_thresholds(bars_map, percentile=30):
    """Compute per-symbol ATR percentile thresholds."""
    thresholds={}
    for sym,bars in bars_map.items():
        h,l,c=bars["high"],bars["low"],bars["close"]
        tr=pd.concat([(h-l).abs(),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr=tr.rolling(ATR_PERIOD).mean().dropna()
        thresholds[sym]=np.percentile(atr,percentile)
    return thresholds

def filter_trades_by_atr(sym_trades, bars_map, thresholds):
    """Return set of trade indices filtered by ATR threshold."""
    filtered=set()
    for sym in DEFAULT_SYMBOLS:
        h,l,c=bars_map[sym]["high"],bars_map[sym]["low"],bars_map[sym]["close"]
        tr=pd.concat([(h-l).abs(),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr_series=tr.rolling(ATR_PERIOD).mean()
        thresh=thresholds[sym]
        for idx,t in enumerate(sym_trades[sym]):
            # Find entry bar by matching time
            entry_dt=t.entry_time
            matching=bars_map[sym][bars_map[sym]["datetime"]==entry_dt]
            if len(matching)==0: continue
            i=matching.index[0]
            if i<len(atr_series) and pd.notna(atr_series.iloc[i]):
                if atr_series.iloc[i]<thresh:
                    filtered.add((sym,idx))
    return filtered

def compute_filtered_metrics(sym_trades, funding_map, tz_name, fee_bps, slp_bps,
                             weight_scales=None, filtered_trades=None):
    """Compute metrics with optional fixes."""
    if filtered_trades is None: filtered_trades=set()
    cpt=FIXED_NOTIONAL*(2*fee_bps+2*slp_bps)/10000.0
    records=[]
    filtered_count=0
    for sym in DEFAULT_SYMBOLS:
        scale=weight_scales.get(sym,1.0) if weight_scales else 1.0
        for idx,t in enumerate(sym_trades[sym]):
            if (sym,idx) in filtered_trades: filtered_count+=1; continue
            ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
            nc=ret*FIXED_NOTIONAL*scale
            ca=nc-cpt*scale
            et=t.entry_time.tz_convert("UTC") if t.entry_time.tzinfo else t.entry_time.tz_localize(tz).tz_convert("UTC")
            xt=t.exit_time.tz_convert("UTC") if t.exit_time.tzinfo else t.exit_time.tz_localize(tz).tz_convert("UTC")
            inst=symbol_to_inst_id(sym)
            f=funding_map.get(inst); fp=0.0
            if f is not None and not f.empty:
                m=(f["funding_time_utc"]>=et)&(f["funding_time_utc"]<xt)
                if m.any(): s=f.loc[m,"funding_rate"].sum()*FIXED_NOTIONAL*scale; fp=s if t.direction==1 else -s
            records.append({"symbol":sym,"pnl":ca-fp,"date":t.exit_time.date()})
    df=pd.DataFrame(records)
    if df.empty: return {"total_trades":0,"total_pnl":0,"win_rate":0,"sharpe":0,"uw_pct":0,"max_dd":0,"filtered":0}

    daily_pnl={}
    for _,r in df.iterrows(): daily_pnl[r["date"]]=daily_pnl.get(r["date"],0)+r["pnl"]
    dates=sorted(daily_pnl.keys())
    all_dates=pd.date_range(start=dates[0],end=dates[-1],freq="D").date
    daily=pd.Series({d:daily_pnl.get(d,0) for d in all_dates})
    equity=daily.cumsum()
    rets=daily/FIXED_NOTIONAL
    sharpe=float(np.sqrt(252)*rets.mean()/rets.std()) if rets.std()>0 else 0
    peak=equity.cummax(); dd=(equity-peak)/peak.where(peak!=0,other=np.nan)*100; dd=dd.fillna(0)
    max_dd=float(dd.min()); uw_d=int((dd<0).sum()); uw_p=uw_d/max(len(all_dates),1)*100
    wins=df[df["pnl"]>0]["pnl"].sum(); losses=abs(df[df["pnl"]<0]["pnl"].sum())
    pf=wins/losses if losses>0 else float("inf"); wr=(df["pnl"]>0).mean()*100 if len(df)>0 else 0
    per_sym={}
    for sym in DEFAULT_SYMBOLS: g=df[df["symbol"]==sym]; per_sym[sym]=g["pnl"].sum() if len(g)>0 else 0
    return {"total_trades":len(df),"total_pnl":float(df["pnl"].sum()),"win_rate":wr,"pf":pf,
            "sharpe":sharpe,"max_dd":max_dd,"uw_days":uw_d,"uw_pct":uw_p,
            "per_symbol":per_sym,"filtered":filtered_count}

# ============================================================
# Run all scenarios
# ============================================================
print("Loading data and running baseline backtests...")
all_sym={}
for sym in DEFAULT_SYMBOLS:
    all_sym[sym]=bt(bars_map[sym],sym,LB,ATR_V,MH)
    print(f"  {sym.split('_')[0]:>5}: {len(all_sym[sym]):,} trades")

# Baseline
m_base=compute_filtered_metrics(all_sym,fmap,args.timezone,FEE_MAKER,SLIPPAGE)

# Fix 1: Weight cap
scales=compute_weight_scales(all_sym,weight_mult=1.5)
m_w=compute_filtered_metrics(all_sym,fmap,args.timezone,FEE_MAKER,SLIPPAGE,weight_scales=scales)

# Fix 2: ATR filter
atr_thresh=compute_atr_thresholds(bars_map,percentile=30)
filt=filter_trades_by_atr(all_sym,bars_map,atr_thresh)
m_a=compute_filtered_metrics(all_sym,fmap,args.timezone,FEE_MAKER,SLIPPAGE,filtered_trades=filt)

# Fix 3: Combined
m_c=compute_filtered_metrics(all_sym,fmap,args.timezone,FEE_MAKER,SLIPPAGE,
                              weight_scales=scales,filtered_trades=filt)

# ============================================================
# Report
# ============================================================
print(f"\nWeight scales:")
for sym,s in sorted(scales.items()):
    print(f"  {sym.split('_')[0]:>5}: {s:.2f}x")

print(f"\nATR thresholds (p30):")
for sym,t in sorted(atr_thresh.items()):
    print(f"  {sym.split('_')[0]:>5}: {t:.4f}")

print(f"\n{'='*115}")
print("BEFORE/AFTER COMPARISON")
print(f"{'='*115}")
print(f"{'Variant':<30} {'Trades':>7} {'PnL':>10} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'UW%':>6} {'SOL%':>6}")
print("-"*115)

for name,m in [("Baseline",m_base),("Fix1: Weight cap 1.5x",m_w),
               ("Fix2: ATR filter p30",m_a),("Fix3: Combined",m_c)]:
    sol_p=m["per_symbol"].get("SOLUSDT_SWAP_OKX.GLOBAL",0)/max(m["total_pnl"],1)*100
    filt_str=f" (-{m.get('filtered',0):,})" if m.get('filtered',0)>0 else ""
    print(f"{name:<30} {m['total_trades']:>7,}{filt_str} ${m['total_pnl']:>9,.0f} {m['win_rate']:>5.1f}% {m['pf']:>5.2f} {m['sharpe']:>6.2f} {m['max_dd']:>6.1f}% {m['uw_pct']:>5.1f}% {sol_p:>5.1f}%")

print(f"\n--- Per-symbol PnL shift ---")
print(f"{'Sym':>5} {'Baseline':>10} {'+Weight':>10} {'+ATR':>10} {'+Both':>10}")
for sym in DEFAULT_SYMBOLS:
    b=m_base["per_symbol"].get(sym,0)
    w=m_w["per_symbol"].get(sym,0)
    a=m_a["per_symbol"].get(sym,0)
    c=m_c["per_symbol"].get(sym,0)
    print(f"{sym.split('_')[0]:>5} ${b:>9,.0f} ${w:>9,.0f} ${a:>9,.0f} ${c:>9,.0f}")

# ATR sensitivity sweep
print(f"\n--- ATR Percentile Sensitivity ---")
print(f"{'pct':>4} {'PnL':>10} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'DD':>7} {'UW%':>6} {'Filtered%':>9}")
for pct in [10,20,30,40,50,60]:
    th=compute_atr_thresholds(bars_map,percentile=pct)
    fl=filter_trades_by_atr(all_sym,bars_map,th)
    mm=compute_filtered_metrics(all_sym,fmap,args.timezone,FEE_MAKER,SLIPPAGE,filtered_trades=fl)
    f_pct=mm.get("filtered",0)/max(m_base["total_trades"],1)*100
    print(f"{pct:>4} ${mm['total_pnl']:>9,.0f} {mm['win_rate']:>5.1f}% {mm['pf']:>5.2f} {mm['sharpe']:>6.2f} {mm['max_dd']:>6.1f}% {mm['uw_pct']:>5.1f}% {f_pct:>8.1f}%")
