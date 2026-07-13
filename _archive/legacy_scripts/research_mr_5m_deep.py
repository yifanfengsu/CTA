#!/usr/bin/env python3
"""MR-5m Deep Dive v2: per-symbol, OOS splits, risk metrics. Cached backtests."""

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
from datetime import timedelta
import numpy as np

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

# Load data
bars_map={}
print("Loading 5min data...")
for sym in DEFAULT_SYMBOLS:
    b1=load_1m(sym,hr,db); b5=r5(b1,5,hr)
    if not b5.empty: bars_map[sym]=b5; print(f"  {sym.split('_')[0]:>5}: {len(b5):,} bars")

LB,ATR,MH=24,1.0,48

# Run backtests ONCE and store
print("\nRunning backtests (LB=24, ATR=1.0, MH=48)...")
all_trades={}
for sym in bars_map:
    all_trades[sym]=bt(bars_map[sym],sym,LB,ATR,MH)
    print(f"  {sym.split('_')[0]:>5}: {len(all_trades[sym]):,} trades")

all_t=[t for ts in all_trades.values() for t in ts]

# ============================================================
# 1. Per-symbol
# ============================================================
print(f"\n{'='*80}")
print("1. Per-symbol (MAKER)")
print(f"{'Sym':>5} {'Trades':>7} {'PnL':>10} {'WR':>6} {'PF':>6} {'Stop':>7} {'Stop$':>10} {'Mid':>7} {'Mid$':>10}")
print("-"*80)
for sym in DEFAULT_SYMBOLS:
    mm=metrics(all_trades[sym],fmap,args.timezone,FEE_MAKER,SLIPPAGE)
    s=mm.get("by_exit",{}).get("stop",{}); mid=mm.get("by_exit",{}).get("midline",{})
    print(f"{sym.split('_')[0]:>5} {mm['total_trades']:>7,} ${mm['pnl']:>9,.0f} {mm['win_rate']:>5.1f}% {mm['pf']:>5.2f} {s.get('count',0):>7,} ${s.get('pnl',0):>10,.0f} {mid.get('count',0):>7,} ${mid.get('pnl',0):>10,.0f}")

# ============================================================
# 2. OOS splits
# ============================================================
print(f"\n{'='*80}")
print("2. OOS splits (MAKER)")
import zoneinfo; tz=zoneinfo.ZoneInfo(args.timezone)
for split_name,range_str in SPLITS.items():
    s_d,e_d=range_str.split("_")
    s_ts=pd.Timestamp(s_d).tz_localize(tz); e_ts=pd.Timestamp(e_d).tz_localize(tz)
    split_t=[t for t in all_t if s_ts<=t.entry_time<e_ts]
    mm=metrics(split_t,fmap,args.timezone,FEE_MAKER,SLIPPAGE)
    s=mm.get("by_exit",{}).get("stop",{}); mid=mm.get("by_exit",{}).get("midline",{})
    print(f"  {split_name:>5}: Trades={mm['total_trades']:>6,}  PnL=${mm['pnl']:>10,.0f}  WR={mm['win_rate']:>5.1f}%  PF={mm['pf']:>5.2f}  Stop: {s.get('count',0):>5,}t ${s.get('pnl',0):>,.0f}  Mid: {mid.get('count',0):>5,}t ${mid.get('pnl',0):>,.0f}")

# ============================================================
# 3. Risk metrics (Sharpe, max DD, monthly)
# ============================================================
print(f"\n{'='*80}")
print("3. Risk metrics (MAKER)")
cost_per_trade=FIXED_NOTIONAL*(2*FEE_MAKER+2*SLIPPAGE)/10000.0
daily_pnl={}
for t in all_t:
    dt=t.exit_time.date()
    ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
    nc=ret*FIXED_NOTIONAL; ca=nc-cost_per_trade
    et=t.entry_time.tz_convert("UTC") if t.entry_time.tzinfo else t.entry_time.tz_localize(tz).tz_convert("UTC")
    xt=t.exit_time.tz_convert("UTC") if t.exit_time.tzinfo else t.exit_time.tz_localize(tz).tz_convert("UTC")
    inst=symbol_to_inst_id(t.symbol); f=fmap.get(inst); fp=0.0
    if f is not None and not f.empty:
        m=(f["funding_time_utc"]>=et)&(f["funding_time_utc"]<xt)
        if m.any(): s=f.loc[m,"funding_rate"].sum()*FIXED_NOTIONAL; fp=s if t.direction==1 else -s
    daily_pnl[dt]=daily_pnl.get(dt,0)+ca-fp

dates=sorted(daily_pnl.keys())
all_dates=pd.date_range(start=dates[0],end=dates[-1],freq="D").date
daily=pd.Series({d:daily_pnl.get(d,0) for d in all_dates})
equity=daily.cumsum()

total_pnl=float(equity.iloc[-1])
daily_ret=daily/FIXED_NOTIONAL
sharpe=float(np.sqrt(252)*daily_ret.mean()/daily_ret.std()) if daily_ret.std()>0 else 0
peak=equity.cummax(); dd=(equity-peak)/FIXED_NOTIONAL; max_dd=float(dd.min())*100

mdf=pd.DataFrame({"date":pd.to_datetime(dates),"pnl":list(daily)}).set_index("date")
mo=mdf.resample("ME")["pnl"].sum()
profitable=sum(1 for v in mo if v>0)

print(f"  Daily PnL: ${total_pnl:,.0f}")
print(f"  Sharpe:    {sharpe:.2f}")
print(f"  Max DD:    {max_dd:.1f}%")
print(f"  Months:    {profitable}/{len(mo)} profitable ({profitable/len(mo)*100:.0f}%)")
print(f"  Avg month: ${mo.mean():,.0f}")
print(f"  Best month: ${mo.max():,.0f} ({mo.idxmax().strftime('%Y-%m')})")
print(f"  Worst month: ${mo.min():,.0f} ({mo.idxmin().strftime('%Y-%m')})")

print(f"\n--- vs 4h ---")
print(f"  4h: PnL=+$4,048, Sharpe=0.51, DD=-408%")
print(f"  5m: PnL=+${total_pnl:,.0f}, Sharpe={sharpe:.2f}, DD={max_dd:.1f}%")
