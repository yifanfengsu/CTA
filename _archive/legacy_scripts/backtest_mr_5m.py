#!/usr/bin/env python3
"""MR-5m Formal Backtest: Phase 3-style comprehensive report."""

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

SPLITS={"Train":"2023-01-01_2024-07-01","Val":"2024-07-01_2025-07-01","OOS":"2025-07-01_2026-04-01"}

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

all_trades={}
total_t=0
for sym in bars_map:
    all_trades[sym]=bt(bars_map[sym],sym,LB,ATR,MH)
    total_t+=len(all_trades[sym])

all_t=[t for ts in all_trades.values() for t in ts]

# ============================================================
# Report
# ============================================================
print(f"\n{'='*70}")
print(f"  MR-5m Phase 3: Formal 5-min Backtest")
print(f"  Parameters: Lookback=24 (2h), ATR Stop=1.0x, Max Hold=48 (4h)")
print(f"  Notional: ${FIXED_NOTIONAL:,.0f}/trade, MAKER Fee: {FEE_MAKER}bps/side, Slippage: {SLIPPAGE}bps/side")
print(f"  Data: {DEFAULT_START} to {DEFAULT_END}, 5 symbols")
print(f"{'='*70}")

# --- 1. Overall Metrics ---
cost=FIXED_NOTIONAL*(2*FEE_MAKER+2*SLIPPAGE)/10000.0
import zoneinfo; tz=zoneinfo.ZoneInfo(args.timezone)

daily_pnl={}
records=[]
for t in all_t:
    ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
    nc=ret*FIXED_NOTIONAL; ca=nc-cost
    et=t.entry_time.tz_convert("UTC") if t.entry_time.tzinfo else t.entry_time.tz_localize(tz).tz_convert("UTC")
    xt=t.exit_time.tz_convert("UTC") if t.exit_time.tzinfo else t.exit_time.tz_localize(tz).tz_convert("UTC")
    inst=symbol_to_inst_id(t.symbol); f=fmap.get(inst); fp=0.0
    if f is not None and not f.empty:
        m=(f["funding_time_utc"]>=et)&(f["funding_time_utc"]<xt)
        if m.any(): s=f.loc[m,"funding_rate"].sum()*FIXED_NOTIONAL; fp=s if t.direction==1 else -s
    pnl=ca-fp
    daily_pnl[t.exit_time.date()]=daily_pnl.get(t.exit_time.date(),0)+pnl
    records.append({"reason":t.exit_reason,"symbol":t.symbol,"pnl":pnl})

df=pd.DataFrame(records)
dates=sorted(daily_pnl.keys())
all_dates=pd.date_range(start=dates[0],end=dates[-1],freq="D").date
daily=pd.Series({d:daily_pnl.get(d,0) for d in all_dates})
equity=daily.cumsum()

total_pnl=float(equity.iloc[-1])
daily_ret=daily/FIXED_NOTIONAL
sharpe=float(np.sqrt(252)*daily_ret.mean()/daily_ret.std()) if daily_ret.std()>0 else 0
peak=equity.cummax(); dd=(equity-peak)/peak*100; max_dd=float(dd.min()) if dd.min()<0 else 0
wr=float((df["pnl"]>0).mean())*100

wins=df[df["pnl"]>0]["pnl"].sum(); losses=abs(df[df["pnl"]<0]["pnl"].sum())
pf=wins/losses if losses>0 else float("inf")

print(f"\n  {'='*50}")
print(f"  OVERALL METRICS (MAKER, funding-adjusted)")
print(f"  {'='*50}")
print(f"  Total Trades:      {len(df):>10,}")
print(f"  Total PnL:         ${total_pnl:>10,.0f}")
print(f"  Win Rate:          {wr:>10.1f}%")
print(f"  Profit Factor:     {pf:>10.2f}")
print(f"  Avg Trade PnL:     ${total_pnl/len(df):>10.2f}")
print(f"  Sharpe (annual):   {sharpe:>10.2f}")
print(f"  Max Drawdown:      {max_dd:>10.1f}%")

# --- 2. Exit Breakdown ---
print(f"\n  {'='*50}")
print(f"  EXIT REASON BREAKDOWN")
print(f"  {'='*50}")
print(f"  {'Reason':>10}  {'Count':>8}  {'PnL':>12}  {'Avg/Trade':>10}")
for reason in ["midline","stop","max_hold","end_of_data"]:
    g=df[df["reason"]==reason]
    if len(g)>0:
        print(f"  {reason:>10}  {len(g):>8,}  ${g['pnl'].sum():>11,.0f}  ${g['pnl'].mean():>9.2f}")

# --- 3. Per-Symbol ---
print(f"\n  {'='*50}")
print(f"  PER-SYMBOL")
print(f"  {'='*50}")
print(f"  {'Sym':>5}  {'Trades':>7}  {'PnL':>10}  {'WR':>6}  {'PF':>6}  {'Stop%':>6}")
for sym in DEFAULT_SYMBOLS:
    g=df[df["symbol"]==sym]
    w=g[g["pnl"]>0]["pnl"].sum(); l=abs(g[g["pnl"]<0]["pnl"].sum());
    pf_s=w/l if l>0 else float("inf")
    wr_s=(g["pnl"]>0).mean()*100
    stop_s=(g["reason"]=="stop").mean()*100
    print(f"  {sym.split('_')[0]:>5}  {len(g):>7,}  ${g['pnl'].sum():>9,.0f}  {wr_s:>5.1f}%  {pf_s:>5.2f}  {stop_s:>5.1f}%")

# --- 4. OOS Splits ---
print(f"\n  {'='*50}")
print(f"  SPLIT ANALYSIS")
print(f"  {'='*50}")
print(f"  {'Split':>6}  {'Trades':>7}  {'PnL':>10}  {'WR':>6}  {'PF':>6}")
for name,rs in SPLITS.items():
    sd,ed=rs.split("_")
    st=pd.Timestamp(sd).tz_localize(tz); et=pd.Timestamp(ed).tz_localize(tz)
    g=df[(pd.to_datetime([t.entry_time for t in all_t])>=st) & (pd.to_datetime([t.entry_time for t in all_t])<et)]
    w=g[g["pnl"]>0]["pnl"].sum(); l=abs(g[g["pnl"]<0]["pnl"].sum())
    pf_s=w/l if l>0 else float("inf"); wr_s=(g["pnl"]>0).mean()*100
    print(f"  {name:>6}  {len(g):>7,}  ${g['pnl'].sum():>9,.0f}  {wr_s:>5.1f}%  {pf_s:>5.2f}")

# --- 5. Monthly PnL ---
mdf=pd.DataFrame({"date":pd.to_datetime(dates),"pnl":list(daily)}).set_index("date")
mo=mdf.resample("ME")["pnl"].sum()
profitable_months=sum(1 for v in mo if v>0)
print(f"\n  {'='*50}")
print(f"  MONTHLY PnL ({profitable_months}/{len(mo)} profitable)")
print(f"  {'='*50}")
print(f"  {'Month':>8}  {'PnL':>10}")
for dt,val in mo.items():
    print(f"  {dt.strftime('%Y-%m'):>8}  ${val:>9,.0f}")

print(f"\n  Avg Monthly: ${mo.mean():>,.0f}  |  Best: ${mo.max():>,.0f} ({mo.idxmax().strftime('%Y-%m')})  |  Worst: ${mo.min():>,.0f} ({mo.idxmin().strftime('%Y-%m')})")

# --- 6. Per-symbol OOS ---
print(f"\n  {'='*50}")
print(f"  PER-SYMBOL OOS ({'2025-07-01'} to {'2026-04-01'})")
print(f"  {'='*50}")
sd,ed="2025-07-01","2026-04-01"
st=pd.Timestamp(sd).tz_localize(tz); et=pd.Timestamp(ed).tz_localize(tz)
for sym in DEFAULT_SYMBOLS:
    g=df[(df["symbol"]==sym) & (pd.to_datetime([t.entry_time for t in all_t])>=st) & (pd.to_datetime([t.entry_time for t in all_t])<et)]
    w=g[g["pnl"]>0]["pnl"].sum(); l=abs(g[g["pnl"]<0]["pnl"].sum())
    pf_s=w/l if l>0 else float("inf"); wr_s=(g["pnl"]>0).mean()*100
    print(f"  {sym.split('_')[0]:>5}: {len(g):>5,} trades  PnL=${g['pnl'].sum():>9,.0f}  WR={wr_s:>5.1f}%  PF={pf_s:>5.2f}")

# --- 7. TAKER scenario ---
cost_t=FIXED_NOTIONAL*(2*FEE_TAKER+2*SLIPPAGE)/10000.0
taker_pnl=0
for t in all_t:
    ret=(t.exit_price-t.entry_price)/t.entry_price if t.direction==1 else (t.entry_price-t.exit_price)/t.entry_price
    nc=ret*FIXED_NOTIONAL; ca=nc-cost_t
    et=t.entry_time.tz_convert("UTC") if t.entry_time.tzinfo else t.entry_time.tz_localize(tz).tz_convert("UTC")
    xt=t.exit_time.tz_convert("UTC") if t.exit_time.tzinfo else t.exit_time.tz_localize(tz).tz_convert("UTC")
    inst=symbol_to_inst_id(t.symbol); f=fmap.get(inst); fp=0.0
    if f is not None and not f.empty:
        m=(f["funding_time_utc"]>=et)&(f["funding_time_utc"]<xt)
        if m.any(): s=f.loc[m,"funding_rate"].sum()*FIXED_NOTIONAL; fp=s if t.direction==1 else -s
    taker_pnl+=ca-fp

print(f"\n  {'='*50}")
print(f"  TAKER SCENARIO ({FEE_TAKER}bps/side)")
print(f"  Total Taker PnL: ${taker_pnl:>10,.0f}")
print(f"  Maker-Taker gap: ${total_pnl-taker_pnl:>,.0f}")

print(f"\n  {'='*50}")
print(f"  vs 4h MR-v1.2")
print(f"  4h: PnL=+$4,048, Trades=2,216, Sharpe=0.51")
print(f"  5m: PnL=+${total_pnl:,.0f}, Trades={len(df):,}, Sharpe={sharpe:.2f}")
print(f"  5m/Taker: PnL=+${taker_pnl:,.0f} (still {'Profitable' if taker_pnl>0 else 'LOSS'})")
