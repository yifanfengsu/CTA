#!/usr/bin/env python3
"""MR-5m Full Diagnostic: SOL concentration, underwater time, rolling windows.

Phase 1: diagnostic analysis → print conclusions
Phase 2: four standalone code modules with visualizations
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
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

# ============================================================
# Shared data loading (same as audit script)
# ============================================================
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
for sym in DEFAULT_SYMBOLS:
    b1=load_1m(sym,hr,db); b5=r5(b1,5,hr)
    if not b5.empty: bars_map[sym]=b5

LB,ATR,MH=24,1.0,48
all_t=[]
for sym in bars_map:
    all_t.extend(bt(bars_map[sym],sym,LB,ATR,MH))

import zoneinfo; tz=zoneinfo.ZoneInfo(args.timezone)
cost=FIXED_NOTIONAL*(2*FEE_MAKER+2*SLIPPAGE)/10000.0

daily_pnl={}
records=[]
symbol_pnl=defaultdict(lambda: defaultdict(float))  # sym → date → pnl
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
    symbol_pnl[t.symbol][t.exit_time.date()]+=pnl
    records.append({"symbol":t.symbol,"pnl":pnl,"date":t.exit_time.date()})

df=pd.DataFrame(records)
dates=sorted(daily_pnl.keys())
all_dates=pd.date_range(start=dates[0],end=dates[-1],freq="D").date
daily=pd.Series({d:daily_pnl.get(d,0) for d in all_dates})
equity=daily.cumsum()
TOTAL_PNL=float(equity.iloc[-1]); N_DAYS=len(all_dates)

peak=equity.cummax()
dd_series=(equity-peak)/peak.where(peak!=0,other=np.nan)*100
dd_series=dd_series.fillna(0)
MAX_DD=float(dd_series.min())

mdf=pd.DataFrame({"date":pd.to_datetime(dates),"pnl":list(daily)}).set_index("date")
mo=mdf.resample("ME")["pnl"].sum()

# Per-symbol monthly
sol_daily=pd.Series({d:symbol_pnl["SOLUSDT_SWAP_OKX.GLOBAL"].get(d,0) for d in all_dates})
sol_daily.index=pd.to_datetime(sol_daily.index)
sol_equity=sol_daily.cumsum()
sol_mo=sol_daily.resample("ME").sum()

# Other symbols combined (ex-SOL)
other_daily=pd.Series({d:sum(symbol_pnl[s].get(d,0) for s in DEFAULT_SYMBOLS if s!="SOLUSDT_SWAP_OKX.GLOBAL") for d in all_dates})
other_daily.index=pd.to_datetime(other_daily.index)
other_equity=other_daily.cumsum()
other_mo=other_daily.resample("ME").sum()

out_dir=PROJECT_ROOT/"reports"/"research"/"mr_5m_diagnostics"
out_dir.mkdir(parents=True,exist_ok=True)

# ============================================================
# Phase 1: Diagnostic Analysis
# ============================================================
print("="*70)
print("PHASE 1: DIAGNOSTIC ANALYSIS")
print("="*70)

# --- Problem 1: SOL Concentration ---
print("\n--- PROBLEM 1: SOL CONCENTRATION ---")
sol_top3=sol_mo.sort_values(ascending=False).head(3)
print(f"  SOL total PnL: ${sol_mo.sum():,.0f}")
sol_labels=[f"{d.strftime('%Y-%m')}=${v:,.0f}" for d,v in sol_top3.items()]
print(f"  SOL Top3 months: {', '.join(sol_labels)}")
print(f"  SOL Top3 / Total SOL: {sol_top3.sum()/sol_mo.sum()*100:.1f}%")

# Market regime label for top months
for dt,val in sol_top3.items():
    y=dt.year; m=dt.month
    if y==2023 and m<=6: regime="recovery"
    elif y==2023: regime="sideways"
    elif y==2024 and m<=6: regime="BULL"
    elif y==2024: regime="correction"
    else: regime="mixed/range"
    print(f"    {dt.strftime('%Y-%m')}: {regime}")

# Check if SOL advantage is consistent or episodic
sol_share_by_half={
    "2023-H1":sol_mo["2023-01":"2023-06"].sum()/max(mo["2023-01":"2023-06"].sum(),1)*100,
    "2023-H2":sol_mo["2023-07":"2023-12"].sum()/max(mo["2023-07":"2023-12"].sum(),1)*100,
    "2024-H1":sol_mo["2024-01":"2024-06"].sum()/max(mo["2024-01":"2024-06"].sum(),1)*100,
    "2024-H2":sol_mo["2024-07":"2024-12"].sum()/max(mo["2024-07":"2024-12"].sum(),1)*100,
    "2025-H1":sol_mo["2025-01":"2025-06"].sum()/max(mo["2025-01":"2025-06"].sum(),1)*100,
    "2025-H2+":sol_mo["2025-07":].sum()/max(mo["2025-07":].sum(),1)*100,
}
print(f"  SOL share by half-year:")
for period,share in sol_share_by_half.items():
    flag="⚠️ >80%" if share>80 else "✓"
    print(f"    {period}: {share:.0f}% {flag}")

# Gini coefficient for SOL monthly contribution
sorted_vals=np.sort(np.abs(sol_mo.values))
n=len(sorted_vals)
gini=2*np.sum(np.arange(1,n+1)*sorted_vals)/(n*np.sum(sorted_vals))-(n+1)/n if np.sum(sorted_vals)>0 else 0
print(f"  SOL monthly Gini: {gini:.3f} (0=uniform, 1=max concentration)")

# Diagnosis
if sol_mo.max()/sol_mo.sum()>0.4:
    sol_cause="B (episodic: single month dominates)"
    sol_risk="HIGH"
elif gini>0.5:
    sol_cause="B (episodic: profit concentrated in few months)"
    sol_risk="HIGH"
elif all(s>40 for s in sol_share_by_half.values()):
    sol_cause="A (structural: SOL consistently dominates across all periods)"
    sol_risk="LOW"
else:
    sol_cause="B (mixed: some periods dominate, others don't)"
    sol_risk="MEDIUM"
print(f"  SOL diagnosis: cause={sol_cause}, risk={sol_risk}")

# --- Problem 2: Underwater Time ---
print("\n--- PROBLEM 2: UNDERWATER TIME ---")
shallow=((dd_series<0)&(dd_series>=-5)).sum()
medium=((dd_series<-5)&(dd_series>=-15)).sum()
deep=(dd_series<-15).sum()
total_uw=int((dd_series<0).sum())
print(f"  Shallow (0 to -5%):   {shallow:>4} days ({shallow/max(total_uw,1)*100:.0f}%)")
print(f"  Medium (-5 to -15%):  {medium:>4} days ({medium/max(total_uw,1)*100:.0f}%)")
print(f"  Deep (< -15%):        {deep:>4} days ({deep/max(total_uw,1)*100:.0f}%)")

# Longest continuous underwater
current_start=None; current_len=0; longest_start=None; longest_len=0; longest_depth=0
for i,d in enumerate(dd_series):
    if d<0:
        if current_start is None: current_start=all_dates[i]
        current_len+=1
        if current_len>longest_len:
            longest_len=current_len; longest_start=current_start
            longest_depth=float(dd_series.iloc[i-longest_len+1:i+1].min())
    else:
        current_start=None; current_len=0
print(f"  Longest continuous underwater: {longest_start} to {all_dates[list(dd_series.index).index(longest_start)+longest_len-1] if longest_start else 'N/A'}")
print(f"    Duration: {longest_len} days, deepest: {longest_depth:.1f}%")
if longest_len>60 and longest_depth<-15:
    print(f"    ⚠️ LONG & DEEP — needs regime filter")
    uw_rating="需过滤 (存在长期深水下)"
else:
    uw_rating="可接受"
print(f"  UW rating: {uw_rating}")

# --- Problem 3: Rolling Windows ---
print("\n--- PROBLEM 3: ROLLING WINDOWS ---")
window_days=270
rolling_pnls=[]; rolling_sharpes=[]; rolling_start_dates=[]
for start_d in range(0,len(daily)-window_days,30):
    window=daily.iloc[start_d:start_d+window_days]
    pnl=window.sum()
    rets=window/FIXED_NOTIONAL
    s=(np.sqrt(252)*rets.mean()/rets.std()) if rets.std()>0 else 0
    rolling_pnls.append(pnl); rolling_sharpes.append(s)
    rolling_start_dates.append(all_dates[start_d])

neg_count=sum(1 for p in rolling_pnls if p<0)
neg_pct=neg_count/len(rolling_pnls)*100
print(f"  Total windows: {len(rolling_pnls)}, negative: {neg_count} ({neg_pct:.0f}%)")
if neg_count>0:
    neg_pnls=[p for p in rolling_pnls if p<0]
    neg_sharpes=[s for p,s in zip(rolling_pnls,rolling_sharpes) if p<0]
    pos_sharpes=[s for p,s in zip(rolling_pnls,rolling_sharpes) if p>0]
    print(f"  Neg PnL range: ${min(neg_pnls):,.0f} to ${max(neg_pnls):,.0f}")
    print(f"  Neg Sharpe mean: {np.mean(neg_sharpes):.2f}, Pos Sharpe mean: {np.mean(pos_sharpes):.2f}")
    # Check if negative windows are clustered
    neg_dates=[d for p,d in zip(rolling_pnls,rolling_start_dates) if p<0]
    if len(neg_dates)>1:
        gaps=[(neg_dates[i+1]-neg_dates[i]).days for i in range(len(neg_dates)-1)]
        avg_gap=np.mean(gaps) if gaps else 0
        print(f"  Neg window gaps: avg {avg_gap:.0f} days ({'CLUSTERED' if avg_gap<60 else 'SCATTERED'})")
        # Overlap with longest underwater
        if longest_start:
            uw_start=longest_start; uw_end=all_dates[list(dd_series.index).index(longest_start)+longest_len-1] if longest_start else None
            overlap=sum(1 for d in neg_dates if uw_start<=d<=uw_end)
            print(f"  Overlap with longest UW: {overlap}/{len(neg_dates)} windows ({overlap/max(len(neg_dates),1)*100:.0f}%)")

rw_cause="局部失效" if neg_pct<20 else "系统性"
rw_risk="MEDIUM" if neg_pct<20 else "HIGH"
print(f"  RW diagnosis: {rw_cause}, risk={rw_risk}")

# --- Phase 1: Summary ---
print("\n"+"="*70)
print("PHASE 1 SUMMARY")
print("="*70)
print(f"  SOL:  cause={sol_cause}, risk={sol_risk}, action={'limit weight' if sol_risk!='LOW' else 'accept'} ")
print(f"  UW:   {uw_rating}, longest={longest_len}d, action={'add regime filter' if longest_len>60 else 'monitor'}")
print(f"  RW:   {rw_cause}, {neg_pct:.0f}% negative, risk={rw_risk}")
readiness="条件就绪" if sol_risk!="CRITICAL" and neg_pct<30 else "未就绪"
print(f"  Readiness: {readiness}")
print(f"  Prerequisites: SOL weight cap, extend OOS to 12mo, regime filter if long UW found")

# ============================================================
# Phase 2: Code Modules
# ============================================================
print("\n"+"="*70)
print("PHASE 2: GENERATING CODE MODULES & CHARTS")
print("="*70)

# ----- Module 1: SOL Concentration -----
def analyze_sol_concentration(df, mo, TOTAL_PNL):
    """Module 1: SOL concentration deep dive."""
    sol_mo_local=sol_mo.copy()

    # Chart 1: SOL monthly PnL vs others
    fig,ax=plt.subplots(figsize=(14,6))
    ax.bar(sol_mo_local.index,sol_mo_local.values,color='#FF6B35',label='SOL',width=20)
    ax.bar(other_mo.index,other_mo.values,color='#4ECDC4',label='Other 4 symbols',width=20,alpha=0.7,align='edge')
    # Market regime bands
    for sd,ed,label,c in [("2023-01-01","2023-07-01","recovery","#e8f5e9"),("2023-07-01","2024-01-01","sideways","#fff3e0"),
                            ("2024-01-01","2024-07-01","BULL","#ffebee"),("2024-07-01","2025-01-01","corr","#e3f2fd"),
                            ("2025-01-01","2025-07-01","range","#f3e5f5"),("2025-07-01","2026-04-01","mixed","#fce4ec")]:
        ax.axvspan(pd.Timestamp(sd),pd.Timestamp(ed),alpha=0.15,color=c,zorder=0)
        ax.text(pd.Timestamp(sd)+timedelta(days=30),ax.get_ylim()[1]*0.95,label,fontsize=8,alpha=0.6)
    ax.legend(); ax.set_title("MR-5m Monthly PnL: SOL vs Others"); ax.set_ylabel("PnL ($)")
    fmt=mticker.FuncFormatter(lambda x,_:f"${x/1000:.0f}k"); ax.yaxis.set_major_formatter(fmt)
    fig.tight_layout(); fig.savefig(out_dir/"sol_monthly_pnl.png",dpi=100); plt.close(fig)
    print("  Saved: sol_monthly_pnl.png")

    # Numbers
    sol_top3=sol_mo_local.sort_values(ascending=False).head(3)
    sol_labels2=[f"{d.strftime('%Y-%m')}:${v:,.0f}" for d,v in sol_top3.items()]
    print(f"  SOL Top3: {', '.join(sol_labels2)}")
    # Gini
    sv=np.abs(sol_mo_local.values); n=len(sv)
    gini=2*np.sum(np.arange(1,n+1)*np.sort(sv))/(n*np.sum(sv))-(n+1)/n
    print(f"  SOL Gini: {gini:.3f}")
    # Without top3
    total_without_top3=TOTAL_PNL-sol_top3.sum()
    print(f"  Total PnL without SOL top3: ${total_without_top3:,.0f} ({total_without_top3/TOTAL_PNL*100:.0f}% of original)")
    # Weight recommendation
    if gini>0.5:
        avg_other_pnl=other_mo.sum()/4  # 4 other symbols
        recommended_weight=avg_other_pnl/(sol_mo_local.sum()/13)  # avg monthly
        recommended_weight=min(recommended_weight,1.5)  # cap at 1.5x
        print(f"  Recommended SOL weight cap: {recommended_weight:.1f}x of other avg")
    assert sol_top3 is not None, "analyze_sol_concentration: 返回值不应为空"

# ----- Module 2: Underwater Time -----
def analyze_drawdown_depth(equity, dd_series, N_DAYS):
    """Module 2: Drawdown depth analysis."""
    # Chart: equity with DD bands
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,8),sharex=True,gridspec_kw={'height_ratios':[3,1]})

    ax1.plot(equity.index,equity.values,color='#2E86AB',linewidth=0.8,label='Equity')
    ax1.fill_between(equity.index,0,equity.values,alpha=0.1,color='#2E86AB')
    # Colored DD bands on ax2
    dd_arr=dd_series.values
    colors=np.where(dd_arr>=0,'#2ecc71',np.where(dd_arr>=-5,'#f1c40f',np.where(dd_arr>=-15,'#e67e22','#e74c3c')))
    ax2.bar(dd_series.index,dd_arr,color=colors,width=1)
    ax2.axhline(y=0,color='black',linewidth=0.5)
    ax2.axhline(y=-5,color='#f1c40f',linewidth=0.5,linestyle='--')
    ax2.axhline(y=-15,color='#e74c3c',linewidth=0.5,linestyle='--')
    ax1.set_title("MR-5m Equity Curve"); ax1.set_ylabel("Equity ($)")
    ax2.set_ylabel("Drawdown %"); ax2.set_xlabel("Date")
    fmt=mticker.FuncFormatter(lambda x,_:f"${x/1000:.0f}k"); ax1.yaxis.set_major_formatter(fmt)
    fig.tight_layout(); fig.savefig(out_dir/"drawdown_analysis.png",dpi=100); plt.close(fig)
    print("  Saved: drawdown_analysis.png")

    # Depth breakdown
    names=["Profit","Shallow (0-5%)","Medium (5-15%)","Deep (>15%)"]
    counts=[int((dd_arr>=0).sum()),int(((dd_arr<0)&(dd_arr>=-5)).sum()),
            int(((dd_arr<-5)&(dd_arr>=-15)).sum()),int((dd_arr<-15).sum())]
    for name,count in zip(names,counts):
        print(f"  {name:>20}: {count:>4} days ({count/N_DAYS*100:.1f}%)")

    # Longest continuous
    current_start=None; cl=0; ls=None; ll=0; ld=0
    for i,d in enumerate(dd_arr):
        if d<0:
            if current_start is None: current_start=all_dates[i]
            cl+=1
            if cl>ll: ll=cl; ls=current_start; ld=float(min(dd_arr[i-cl+1:i+1]))
        else: current_start=None; cl=0
    if ls:
        le=all_dates[list(dd_series.index).index(ls)+ll-1]
        print(f"  Longest UW: {ls} to {le}, {ll} days, deepest {ld:.1f}%")
    # Median recovery
    recovery_days=[]
    in_dd=False; trough_idx=0; trough_val=0
    for i,d in enumerate(dd_arr):
        if d<0 and not in_dd: in_dd=True; trough_idx=i; trough_val=d
        elif d<0 and in_dd:
            if d<trough_val: trough_idx=i; trough_val=d
        elif d>=0 and in_dd:
            recovery_days.append(i-trough_idx); in_dd=False
    if recovery_days:
        print(f"  Median recovery: {np.median(recovery_days):.0f} days")
    # Regime filter suggestion
    if ll>60 and ld<-15:
        print(f"  ⚠️ LONG DEEP UW: suggest ATR-threshold regime filter")
    assert ld<0, "analyze_drawdown_depth: deepest DD 应为负值"

# ----- Module 3: Rolling Windows -----
def analyze_rolling_windows(daily, FIXED_NOTIONAL, window_days=270):
    """Module 3: Rolling window analysis."""
    rp=[]; rs=[]; rd=[]
    for start_d in range(0,len(daily)-window_days,30):
        w=daily.iloc[start_d:start_d+window_days]; rp.append(w.sum())
        rets=w/FIXED_NOTIONAL
        rs.append((np.sqrt(252)*rets.mean()/rets.std()) if rets.std()>0 else 0)
        rd.append(all_dates[start_d])

    # Chart 1: Rolling PnL
    fig,ax=plt.subplots(figsize=(14,5))
    colors=['#e74c3c' if p<0 else '#2ecc71' for p in rp]
    ax.bar(range(len(rp)),rp,color=colors,width=0.8)
    ax.axhline(y=0,color='black',linewidth=0.5)
    ax.set_title(f"MR-5m Rolling {window_days}-Day Window PnL (red=negative)"); ax.set_ylabel("PnL ($)")
    tick_positions=range(0,len(rp),max(1,len(rp)//8))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([rd[i].strftime('%Y-%m') for i in tick_positions],rotation=45,ha='right')
    fmt=mticker.FuncFormatter(lambda x,_:f"${x/1000:.0f}k"); ax.yaxis.set_major_formatter(fmt)
    fig.tight_layout(); fig.savefig(out_dir/"rolling_windows.png",dpi=100); plt.close(fig)
    print("  Saved: rolling_windows.png")

    # Chart 2: Sharpe histogram
    fig,ax=plt.subplots(figsize=(10,4))
    ax.hist(rs,bins=20,color='#3498db',edgecolor='white',alpha=0.8)
    ax.axvline(x=0,color='red',linestyle='--'); ax.axvline(x=np.median(rs),color='green',linestyle='-',label=f'Median={np.median(rs):.2f}')
    ax.set_title("Rolling Window Sharpe Distribution"); ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Count")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir/"rolling_sharpe_hist.png",dpi=100); plt.close(fig)
    print("  Saved: rolling_sharpe_hist.png")

    neg_pnls=[p for p in rp if p<0]; neg_s=[s for p,s in zip(rp,rs) if p<0]
    pos_s=[s for p,s in zip(rp,rs) if p>0]
    print(f"  Negative windows: {len(neg_pnls)}/{len(rp)} ({len(neg_pnls)/len(rp)*100:.0f}%)")
    if neg_pnls:
        print(f"  Neg PnL: ${min(neg_pnls):,.0f} to ${max(neg_pnls):,.0f}")
        print(f"  Neg Sharpe: {np.mean(neg_s):.2f} (pos: {np.mean(pos_s):.2f})")
        neg_dates=[d for p,d in zip(rp,rd) if p<0]
        print(f"  Neg window start dates: {', '.join(d.strftime('%Y-%m') for d in neg_dates[:5])}{'...' if len(neg_dates)>5 else ''}")
    assert len(rp)>0, "analyze_rolling_windows: 窗口数不应为0"

# ----- Module 4: Recommendations -----
def generate_fix_recommendations():
    """Module 4: synthesised fix recommendations."""
    sol_pct=sol_mo.sum()/TOTAL_PNL*100
    uw_pct=float((dd_series<0).sum())/N_DAYS*100
    neg_pct2=len([p for p in rolling_pnls if p<0])/max(len(rolling_pnls),1)*100
    recommendations=[]
    if sol_pct>50:
        avg_other=other_mo.sum()/4
        recommendations.append(f"SOL weight cap: {avg_other/max(sol_mo.mean(),1)*100:.0f}% of current")
    else:
        recommendations.append("SOL acceptable — maintain weight")
    if uw_pct>50:
        recommendations.append("Add ATR-based regime filter to reduce UW time")
    else:
        recommendations.append("UW acceptable — monitor")
    if neg_pct2>20:
        recommendations.append("Extend OOS validation to 12+ months before live")
    else:
        recommendations.append("Rolling windows acceptable — proceed")
    print("Fix Recommendations:")
    for i,r in enumerate(recommendations): print(f"  {i+1}. {r}")
    assert len(recommendations)>0, "generate_fix_recommendations: 无建议输出"

# ============================================================
# Execute all modules
# ============================================================
analyze_sol_concentration(df, mo, TOTAL_PNL)
analyze_drawdown_depth(equity, dd_series, N_DAYS)
analyze_rolling_windows(daily, FIXED_NOTIONAL)
generate_fix_recommendations()

print(f"\n✅ All modules complete. Charts saved to {out_dir}/")
print(f"   sol_monthly_pnl.png, drawdown_analysis.png, rolling_windows.png, rolling_sharpe_hist.png")
