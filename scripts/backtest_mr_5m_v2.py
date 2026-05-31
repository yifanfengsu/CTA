#!/usr/bin/env python3
"""MR-5m Formal Backtest v2: with ATR regime filter (p30).

Filters out trades when 5-min ATR is below the 30th percentile,
avoiding low-volatility chop that causes extended drawdowns.

Benchmarks (vs baseline):
  PnL:    $142k → $152k (+7%)
  Max DD: -36% → -10% (-72%)
  UW:     61% → 53%
"""

import sys; sys.path.insert(0,'scripts')
from research_mr_5m import *
from datetime import timedelta
import numpy as np

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
print("Loading data...")
for sym in DEFAULT_SYMBOLS:
    b1=load_1m(sym,hr,db); b5=r5(b1,5,hr)
    if not b5.empty: bars_map[sym]=b5

LB,ATR_V,MH=24,1.0,48

# Compute ATR thresholds (p30) for all symbols
print("Computing ATR thresholds...")
ATR_PCT=30
atr_thresholds={}
for sym,bars in bars_map.items():
    h,l,c=bars["high"],bars["low"],bars["close"]
    tr=pd.concat([(h-l).abs(),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr_full=tr.rolling(ATR_PERIOD).mean().dropna()
    atr_thresholds[sym]=np.percentile(atr_full,ATR_PCT)
    print(f"  {sym.split('_')[0]:>5}: p{ATR_PCT}={atr_thresholds[sym]:.4f}")

# Run filtered backtest
print("Running filtered backtest...")
all_trades={}; total_t=0; total_filtered=0
for sym in bars_map.keys():
    bars=bars_map[sym]
    threshold=atr_thresholds[sym]
    # Pre-compute ATR
    h,l,c=bars["high"],bars["low"],bars["close"]
    tr=pd.concat([(h-l).abs(),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr_series=tr.rolling(ATR_PERIOD).mean()

    trades=bt(bars,sym,LB,ATR_V,MH)
    filtered_trades=[]
    filtered_count=0
    for t in trades:
        # Find entry bar index
        entry_dt=t.entry_time
        match=bars[bars["datetime"]==entry_dt]
        if len(match)>0:
            i=match.index[0]
            if i<len(atr_series) and pd.notna(atr_series.iloc[i]):
                if atr_series.iloc[i]<threshold:
                    filtered_count+=1
                    continue
        filtered_trades.append(t)
    all_trades[sym]=filtered_trades
    total_t+=len(filtered_trades)
    total_filtered+=filtered_count
    print(f"  {sym.split('_')[0]:>5}: {len(filtered_trades):,} trades (filtered {filtered_count:,} / {filtered_count/max(len(trades),1)*100:.0f}%)")

all_t=[t for ts in all_trades.values() for t in ts]

# Metrics
import zoneinfo; tz=zoneinfo.ZoneInfo(args.timezone)
cost=FIXED_NOTIONAL*(2*FEE_MAKER+2*SLIPPAGE)/10000.0
daily_pnl={}; records=[]
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
TOTAL_PNL=float(equity.iloc[-1]); N_DAYS=len(all_dates)

peak=equity.cummax(); dd_data=(equity-peak)/peak.where(peak!=0,other=np.nan)*100; dd_data=dd_data.fillna(0)
MAX_DD=float(dd_data.min())
UW_DAYS=int((dd_data<0).sum())

daily_ret=daily/FIXED_NOTIONAL
SHARPE=float(np.sqrt(252)*daily_ret.mean()/daily_ret.std()) if daily_ret.std()>0 else 0
WR=float((df["pnl"]>0).mean())*100
wins=df[df["pnl"]>0]["pnl"].sum(); losses=abs(df[df["pnl"]<0]["pnl"].sum())
PF=wins/losses if losses>0 else float("inf")

# Monthly
mdf=pd.DataFrame({"date":pd.to_datetime(all_dates),"pnl":list(daily)}).set_index("date")
mo=mdf.resample("ME")["pnl"].sum()
profitable_months=sum(1 for v in mo if v>0)

# ============================================================
# REPORT
# ============================================================
print(f"\n{'='*70}")
print(f"  MR-5m v2: ATR Filter p{ATR_PCT}")
print(f"  Lookback=24, ATR Stop=1.0x, Max Hold=48, ATR Filter p{ATR_PCT}")
print(f"  MAKER 2bps/side, Slippage 2bps/side")
print(f"  Data: {DEFAULT_START} → {DEFAULT_END}")
print(f"{'='*70}")

print(f"\n  OVERALL:")
print(f"  Total Trades:      {len(df):>10,} (filtered {total_filtered:,})")
print(f"  Total PnL:         ${TOTAL_PNL:>10,.0f}")
print(f"  Win Rate:          {WR:>10.1f}%")
print(f"  Profit Factor:     {PF:>10.2f}")
print(f"  Sharpe:            {SHARPE:>10.2f}")
print(f"  Max DD:            {MAX_DD:>10.1f}%")
print(f"  UW Days:           {UW_DAYS:>10,} / {N_DAYS:,} ({UW_DAYS/N_DAYS*100:.0f}%)")

print(f"\n  EXIT BREAKDOWN:")
for reason in ["midline","stop","max_hold"]:
    g=df[df["reason"]==reason]
    if len(g)>0: print(f"  {reason:>10}: {len(g):>6,}  ${g['pnl'].sum():>12,.0f}  avg ${g['pnl'].mean():>7.2f}")

print(f"\n  PER-SYMBOL:")
for sym in DEFAULT_SYMBOLS:
    g=df[df["symbol"]==sym]; w=g[g["pnl"]>0]["pnl"].sum(); l=abs(g[g["pnl"]<0]["pnl"].sum())
    pf_s=w/l if l>0 else float("inf"); wr_s=(g["pnl"]>0).mean()*100
    sol_pct=g["pnl"].sum()/TOTAL_PNL*100
    print(f"  {sym.split('_')[0]:>5}: {len(g):>6,}  ${g['pnl'].sum():>10,.0f}  WR={wr_s:>5.1f}%  PF={pf_s:>5.2f}  {sol_pct:>5.1f}%")

print(f"\n  MONTHLY ({profitable_months}/{len(mo)} profitable):")
print(f"  Avg: ${mo.mean():>,.0f}  Best: ${mo.max():>,.0f} ({mo.idxmax().strftime('%Y-%m')})  Worst: ${mo.min():>,.0f} ({mo.idxmin().strftime('%Y-%m')})")

# Per-symbol ATR thresholds
print(f"\n  ATR THRESHOLDS (p{ATR_PCT}):")
for sym,t in sorted(atr_thresholds.items()):
    print(f"  {sym.split('_')[0]:>5}: {t:.4f}")

# vs Baseline comparison
print(f"\n{'='*70}")
print(f"  vs BASELINE (no filter)")
print(f"  PnL:    $142,320 → ${TOTAL_PNL:,.0f} ({TOTAL_PNL/142320*100-100:+.0f}%)")
print(f"  Max DD: -36.3% → {MAX_DD:+.1f}%")
print(f"  UW:     60.8% → {UW_DAYS/N_DAYS*100:.1f}%")
print(f"  Sharpe: 5.83 → {SHARPE:.2f}")
print(f"  Trades: 68,149 → {len(df):,} ({len(df)/68149*100-100:+.0f}%)")
