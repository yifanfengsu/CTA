#!/usr/bin/env python3
"""MR-5m Deep Dive v3: Taker analysis, slippage sensitivity, per-symbol OOS, trade stats."""

import sys; sys.path.insert(0,'scripts')
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

from research_mr_5m import *
from collections import Counter, defaultdict
from datetime import timedelta
import json, numpy as np

args=parse_args()
hr=parse_history_range(args.start,args.end,timedelta(minutes=1),args.timezone)
db=Path(args.database_path); fdir=Path(args.funding_dir)

SPLITS={"train":"2023-01-01_2024-07-01","val":"2024-07-01_2025-07-01","oos":"2025-07-01_2026-04-01"}

fmap={}
for sym in DEFAULT_SYMBOLS:
    inst=symbol_to_inst_id(sym); cf=fdir/f'{inst}_funding_{DEFAULT_START}_{DEFAULT_END}.csv'
    if cf.exists(): fmap[inst]=load_funding(cf)
    else:
        ms=sorted(fdir.glob(f'{inst}_funding_*.csv'))
        if ms: fmap[inst]=load_funding(ms[-1])

bars_map={}
print("Loading 5min data...")
for sym in DEFAULT_SYMBOLS:
    b1=load_1m(sym,hr,db); b5=r5(b1,5,hr)
    if not b5.empty: bars_map[sym]=b5

LB,ATR,MH=24,1.0,48
all_trades={}
print("Running backtests...")
for sym in bars_map:
    all_trades[sym]=bt(bars_map[sym],sym,LB,ATR,MH)
    print(f"  {sym.split('_')[0]}: {len(all_trades[sym]):,} trades")

all_t=[t for ts in all_trades.values() for t in ts]

# ============================================================
# 1. TAKER per-symbol analysis
# ============================================================
print(f"\n{'='*80}")
print("1. TAKER per-symbol (LB=24, ATR=1.0, MH=48)")
print(f"{'Sym':>5} {'Trades':>7} {'PnL':>10} {'WR':>6} {'PF':>6} {'Avg$/T':>7}")
print("-"*55)
total_t=0; total_p=0
for sym in DEFAULT_SYMBOLS:
    mt=metrics(all_trades[sym],fmap,args.timezone,FEE_TAKER,SLIPPAGE)
    avg_t=mt["pnl"]/max(mt["total_trades"],1)
    print(f"{sym.split('_')[0]:>5} {mt['total_trades']:>7,} ${mt['pnl']:>9,.0f} {mt['win_rate']:>5.1f}% {mt['pf']:>5.2f} ${avg_t:>6.2f}")
    total_t+=mt['total_trades']; total_p+=mt['pnl']
print(f"{'TOTAL':>5} {total_t:>7,} ${total_p:>9,.0f}")

# ============================================================
# 2. Slippage sensitivity
# ============================================================
print(f"\n{'='*80}")
print("2. Slippage sensitivity (LB=24, ATR=1.0, MH=48, MAKER base)")
for slp in [0,1,2,3,5,10]:
    mm=metrics(all_t,fmap,args.timezone,FEE_MAKER,slp)
    delta=mm["pnl"]-142320  # baseline
    print(f"  Slippage={slp}bps/side: PnL=${mm['pnl']:>10,.0f} ({delta:+,.0f})  WR={mm['win_rate']:.1f}%")

# ============================================================
# 3. Fee breakeven (combined maker+taker)
# ============================================================
print(f"\n{'='*80}")
print("3. Fee sensitivity: at what cost does strategy break even?")
for fee in [2,3,4,5,6,7,8,10]:
    mm=metrics(all_t,fmap,args.timezone,fee,SLIPPAGE)
    status="✓ PROFITABLE" if mm["pnl"]>0 else "✗ LOSS"
    print(f"  Fee={fee}bps/side: PnL=${mm['pnl']:>10,.0f} {status}")

# ============================================================
# 4. Per-symbol OOS
# ============================================================
print(f"\n{'='*80}")
print("4. Per-symbol OOS (MAKER)")
import zoneinfo; tz=zoneinfo.ZoneInfo(args.timezone)
s_d,e_d="2025-07-01","2026-04-01"
s_ts=pd.Timestamp(s_d).tz_localize(tz); e_ts=pd.Timestamp(e_d).tz_localize(tz)
print(f"{'Sym':>5} {'Trades':>7} {'PnL':>10} {'WR':>6} {'PF':>6}")
print("-"*45)
for sym in DEFAULT_SYMBOLS:
    oos_t=[t for t in all_trades[sym] if s_ts<=t.entry_time<e_ts]
    mm=metrics(oos_t,fmap,args.timezone,FEE_MAKER,SLIPPAGE)
    print(f"{sym.split('_')[0]:>5} {mm['total_trades']:>7,} ${mm['pnl']:>9,.0f} {mm['win_rate']:>5.1f}% {mm['pf']:>5.2f}")

# ============================================================
# 5. Trade duration distribution  
# ============================================================
print(f"\n{'='*80}")
print("5. Trade duration (bars) by exit reason")
for reason in ["midline","stop","max_hold"]:
    durations=[]
    for t in all_t:
        if t.exit_reason==reason:
            dur=(t.exit_time-t.entry_time).total_seconds()/300  # 5min bars
            durations.append(dur)
    if durations:
        arr=np.array(durations)
        print(f"  {reason:>10}: count={len(arr):>6,}  mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
              f"min={arr.min():.0f}  max={arr.max():.0f}  p90={np.percentile(arr,90):.0f}")

# ============================================================
# 6. Concurrent positions
# ============================================================
print(f"\n{'='*80}")
print("6. Concurrent position analysis")
# Build timeline of active positions
import zoneinfo; tz=zoneinfo.ZoneInfo(args.timezone)
events=[]  # (timestamp, delta)
for t in all_t:
    et=t.entry_time; xt=t.exit_time
    events.append((et,1)); events.append((xt,-1))
events.sort(key=lambda x:x[0])

active=0; active_samples=[]; prev_t=None
for ts,delta in events:
    if prev_t is not None and active>0:
        active_samples.append((active,ts-prev_t))
    active+=delta; prev_t=ts

# Weighted average
total_active_time=sum(d.total_seconds() for _,d in active_samples)
weighted_avg=sum(c*d.total_seconds() for c,d in active_samples)/max(total_active_time,1)

print(f"  Max concurrent: {max(c for c,_ in active_samples)}")
print(f"  Weighted avg:   {weighted_avg:.1f}")
print(f"  Distribution:")
dist=Counter()
for c,_ in active_samples: dist[c]+=1
for c in sorted(dist)[:10]:
    print(f"    {c:>2} positions: {dist[c]:>8,} samples")

# ============================================================
# 7. Hour-of-day profitability
# ============================================================
print(f"\n{'='*80}")
print("7. Profit by hour-of-day (UTC) of EXIT")
hour_pnl=defaultdict(float); hour_count=defaultdict(int)
for t in all_t:
    h=t.exit_time.hour
    ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
    nc=ret*FIXED_NOTIONAL; cp=nc-FIXED_NOTIONAL*(2*FEE_MAKER+2*SLIPPAGE)/10000.0
    hour_pnl[h]+=cp; hour_count[h]+=1

print(f"  {'Hour':>5} {'PnL':>10} {'Count':>7} {'Avg$/T':>8}")
for h in sorted(hour_pnl):
    avg=hour_pnl[h]/max(hour_count[h],1)
    print(f"  {h:02d}:00  ${hour_pnl[h]:>9,.0f} {hour_count[h]:>7,} ${avg:>7.2f}")

# ============================================================
# 8. Win/Loss streak
# ============================================================
print(f"\n{'='*80}")
print("8. Win/Loss streaks (MAKER, chronological)")
ordered=sorted(all_t,key=lambda t:t.exit_time)
wins=[]; current=0; streak_type=None
for t in ordered:
    ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
    pnl=ret*FIXED_NOTIONAL-FIXED_NOTIONAL*(2*FEE_MAKER+2*SLIPPAGE)/10000.0
    w=pnl>0
    if streak_type is None or w==streak_type: current+=1
    else: wins.append(current); current=1
    streak_type=w
if current>0: wins.append(current)

win_streaks=[w for i,w in enumerate(wins) if i%2==0]  # even indices = win streaks
loss_streaks=[w for i,w in enumerate(wins) if i%2==1]

if win_streaks:
    wa=np.array(win_streaks)
    print(f"  Win streaks:  mean={wa.mean():.1f}  median={np.median(wa):.0f}  max={wa.max()}  p99={np.percentile(wa,99):.0f}")
if loss_streaks:
    la=np.array(loss_streaks)
    print(f"  Loss streaks: mean={la.mean():.1f}  median={np.median(la):.0f}  max={la.max()}  p99={np.percentile(la,99):.0f}")

print(f"\n  Total streaks: {len(wins)}")
