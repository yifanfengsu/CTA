#!/usr/bin/env python3
"""Streak analysis for MR-5m."""
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
import numpy as np
from collections import defaultdict

args=parse_args()
hr=parse_history_range(args.start,args.end,timedelta(minutes=1),args.timezone)
db=Path(args.database_path); fdir=Path(args.funding_dir)

fmap={}
for sym in DEFAULT_SYMBOLS:
    inst=symbol_to_inst_id(sym); cf=fdir/f'{inst}_funding_{DEFAULT_START}_{DEFAULT_END}.csv'
    if cf.exists(): fmap[inst]=load_funding(cf)
    else:
        ms=sorted(fdir.glob(f'{inst}_funding_*.csv'))
        if ms: fmap[inst]=load_funding(ms[-1])

bars_map={}
print("Loading...")
for sym in DEFAULT_SYMBOLS:
    b1=load_1m(sym,hr,db); b5=r5(b1,5,hr)
    if not b5.empty: bars_map[sym]=b5

LB,ATR,MH=24,1.0,48
all_t=[]
for sym in bars_map:
    all_t.extend(bt(bars_map[sym],sym,LB,ATR,MH))

cost=FIXED_NOTIONAL*(2*FEE_MAKER+2*SLIPPAGE)/10000.0
ordered=sorted(all_t,key=lambda t:t.exit_time)

streaks=[]
current_pnls=[]; current_win=None
for t in ordered:
    ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
    pnl=ret*FIXED_NOTIONAL-cost
    is_win=pnl>0
    if current_win is None or is_win==current_win:
        current_pnls.append(pnl)
    else:
        streaks.append((current_win,current_pnls))
        current_pnls=[pnl]
    current_win=is_win
if current_pnls: streaks.append((current_win,current_pnls))

by_len_win=defaultdict(lambda: {'count':0,'total':0,'worst':1e9,'best':-1e9})
by_len_loss=defaultdict(lambda: {'count':0,'total':0,'worst':1e9,'best':-1e9})

for is_win,pnls in streaks:
    l=len(pnls); tot=sum(pnls); mn=min(pnls); mx=max(pnls)
    target=by_len_win if is_win else by_len_loss
    target[l]['count']+=1; target[l]['total']+=tot
    target[l]['worst']=min(target[l]['worst'],mn)
    target[l]['best']=max(target[l]['best'],mx)

print('=== WIN STREAKS ===')
print(f'  {"Len":>4}  {"Count":>5}  {"Avg Total":>10}  {"Avg/Trade":>10}  {"Worst Trade":>12}  {"Best Trade":>12}')
for l,d in sorted(by_len_win.items()):
    if l<=5 or l%5==0 or l>=15:
        avg_tot=d['total']/d['count']; avg_tr=avg_tot/l
        print(f'  {l:>4}  {d["count"]:>5}  ${avg_tot:>9,.0f}  ${avg_tr:>9,.2f}  ${d["worst"]:>11,.0f}  ${d["best"]:>11,.0f}')

print(f'\n=== LOSS STREAKS ===')
print(f'  {"Len":>4}  {"Count":>5}  {"Avg Total":>10}  {"Avg/Trade":>10}  {"Worst Trade":>12}  {"Best Trade":>12}')
for l,d in sorted(by_len_loss.items()):
    if l<=5 or l%5==0 or l>=10:
        avg_tot=d['total']/d['count']; avg_tr=avg_tot/l
        print(f'  {l:>4}  {d["count"]:>5}  ${avg_tot:>9,.0f}  ${avg_tr:>9,.2f}  ${d["worst"]:>11,.0f}  ${d["best"]:>11,.0f}')

print(f'\n=== WORST LOSS STREAKS (len>=10) ===')
for is_win,pnls in streaks:
    if not is_win and len(pnls)>=10:
        print(f'  len={len(pnls):>2}, total=${sum(pnls):>8,.0f}, avg=${sum(pnls)/len(pnls):>7,.2f}, worst=${min(pnls):>7,.0f}')

print(f'\n=== LONGEST WIN STREAKS (top 5) ===')
ws=[(len(pnls),sum(pnls),pnls) for is_win,pnls in streaks if is_win]
ws.sort(reverse=True)
for l,tot,pnls in ws[:5]:
    print(f'  len={l:>2}, total=${tot:>10,.0f}, avg=${tot/l:>7,.2f}, best=${max(pnls):>7,.0f}')

# Summary
all_wins=[s for s in streaks if s[0]]
all_losses=[s for s in streaks if not s[0]]
print(f'\n=== SUMMARY ===')
print(f'  Win streaks:  {len(all_wins):,} total, avg len={np.mean([len(s[1]) for s in all_wins]):.1f}, avg total=${np.mean([sum(s[1]) for s in all_wins]):,.0f}')
print(f'  Loss streaks: {len(all_losses):,} total, avg len={np.mean([len(s[1]) for s in all_losses]):.1f}, avg total=${np.mean([sum(s[1]) for s in all_losses]):,.0f}')
