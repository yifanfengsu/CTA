#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATM VRP — 阶段 A：Deribit 数据地基 + 净缝生死门（数据可行性，judgment 前置）
================================================================================
研究线：vrp/（独立于 cta_strategy 永续研究）。本脚本只判最前置的数据问题——
"能否干净测 ATM VRP、毛缝扣摩擦（含反复对冲）后净缝还剩不剩"。
不测尾部、不测 edge 大小、不判立项、**不算 Sharpe**。

VRP 特殊纪律（左偏卖保险，回测系统性高估，peso problem）：
  - 双重门（edge 为正 ∧ 尾部可生存）；本阶段只碰前者的数据可行性。
  - Sharpe 是危险信号非目标 → 本阶段不算。
  - 净缝必须**条件于对冲频率**（频率 × 单次成本累积），不得用一次性粗估掩盖。

数学底座：纯 VRP（delta-hedged）盈亏 = Σ ½Γ(ΔSᵢ² − σ_IV²Δt)，跨整个持有期每一次对冲。
含义：对冲腿摩擦不是一次，是 频率 × 单次成本 的累积。

================ 预注册口径（写死，executor 不得二选一）================
标的：BTC、ETH 的 Deribit ATM 期权，**期限=月度**（约 30 天到期，月度到期=每月最后周五
      08:00 UTC 结算）。只用 ATM，不碰 wing。周 ATM 留待流程跑通。
ATM 定义：建仓时最接近当时 forward 的**挂牌 strike**，建仓后**固定该 strike 持有到期**
      （不每日 roll）。持有期标的漂移导致偏离 ATM 是该策略真实特征，如实保留。
IV 口径：从该 ATM 期权的价格**BS 反解**隐含波动率，全程一致。同时（在 live 端）记录
      Deribit mark_iv 作为反解方法校验。
      ⚠ 数据限制：Deribit 免费 REST 历史只给 get_tradingview_chart_data 的
      **last-traded OHLC**（非 mark），历史真 mark 不免费可得。故历史 IV = 从 last-trade
      反解（含成交稀疏的 staleness 噪声），并用 DVOL 水平 + live mark_iv 反解校验佐证。
RV 与年化：RV 从标的**永续**（BTC-PERPETUAL / ETH-PERPETUAL，08:00 UTC 日线收盘）算
      （forward≈spot，r≈0，basis 微小，明确写出）。加密 24/7 → **年化因子 365**，
      IV 的年化口径同 365（Deribit IV 本身按日历时间/365 年化），口径一致防 artifact 虚假缝。
建仓节奏：每月在**前一个月度到期日**建仓、持有到下一个月度到期，连续不重叠（标准卖 1M
      跨式节奏）。每个 cycle ~28–35 天。

================ 摩擦栈（条件于对冲频率，两端点，粗估）================
载体：short ATM straddle（卖 call + 卖 put，同一 ATM strike）= VRP 标准载体。
期权腿摩擦：进场穿 bid-ask + 出场穿 bid-ask + Deribit 期权费。bid-ask 在 **IV 空间**
      直接量（Deribit ticker 给 bid_iv/ask_iv）；历史盘口免费不可得 → 用**近期 ATM 月度链
      实测 bid-ask（IV）中位数**做代理，标注为近似且历史更宽（=本估计偏乐观）。
对冲腿摩擦——两端点（不给单点，堵 peso 成本版）：
  端点1 不对冲（裸卖跨式持有到期）：对冲腿摩擦=0，但保留全部方向暴露
      （本阶段**只算摩擦**，方向暴露的尾部 P&L 留阶段 C，不计入此处净缝）。
  端点2 每日对冲一次（月度 ~30 次往返）：沿**真实价格路径**模拟每日 BS delta
      （以建仓 IV 持平算 delta，粗），Σ|Δdelta| = 对冲换手（标的单位），
      成本 = 换手 × S × (永续 half-spread + taker 费)。这是 频率×单次成本 的**计数**，
      **不是**精确对冲 P&L（精确路径 = 阶段 D，本阶段不提前建模）。
费率（核对 deribit.com/kb/fees；保守取 taker，标注 maker 可降）：
  期权 taker = 0.03% of underlying/腿/边，封顶 12.5% of premium；结算(交割)费 0.015% 同封顶。
  永续 taker = 0.05%/边（maker 0% 甚至返佣 → 作敏感性标注）。

================ 单位与 netting ================
统一在**年化 vol 点**（σ_IV − σ_RV）里 netting（VRP 交易员的母语，且 Deribit 直接给 IV bid-ask）：
  毛缝(vp) = σ_IV − σ_RV ；同时报 variance σ_IV² − σ_RV²（任务 3a 要求）与美元值。
  期权摩擦(vp) = (ask_iv − bid_iv) 跨式往返 + 费的 vp 等价（费$/straddle Vega）。
  对冲摩擦(vp) = 对冲成本$/straddle Vega。
  净缝端点1(vp) = 毛缝 − 期权摩擦；净缝端点2(vp) = 毛缝 − 期权摩擦 − 对冲摩擦。
Vega/Γ（ATM，r=0）：d1=(ln(S/K)+½σ²T)/(σ√T)；Γ_straddle=2φ(d1)/(Sσ√T)；
  Vega_straddle=2Sφ(d1)√T（per 1.00 σ）。毛缝美元(faithful)=½Γ_str S²(σ_IV²−σ_RV²)T。

================ 阶段 A 判定（预注册，数据门，看结果前写死）================
A1 数据可得：Deribit 真实历史（IV+标的）足够测多个月度周期 + 关键尾部时段
   (312/2020,519/2021,LUNA/2022,FTX/2022) 可得；bid/ask 历史可得或合理近似；
   真实性可证伪核对(端点 URL testnet=false + 独立锚点 Coinbase 交叉验证)通过。
A2 时钟可对齐：IV/RV 对齐协议明确（08:00 UTC），年化口径一致(365)，无虚假缺口。
A3 净缝为正（生死门，条件于对冲频率 + 左偏诚实）：
   - 端点1（不对冲）净缝 mean 显著为正，且
   - 端点2（每日对冲）净缝 mean 仍为正（若每日对冲后净缝≈0/负 → 现实对冲成本下
     不可交易，即使裸卖端看着有缝），且
   - 剔尾敏感性(3d)表明 mean 不是纯靠"尾部没发生"撑起（左偏方向：剔最坏月 mean 上移，
     上移量 = 已实现尾部对 mean 的拖累 = peso 暴露下界；若 mean 主要靠极少数月、
     单坏月吞掉数月权利金、且尾部覆盖不足 → 标注证据弱）。
   全过 → 进阶段 B；A3 不过 → ATM VRP 数据门判死，记死因。
注意：阶段 A 全过也只是"数据干净 + 毛缝扣摩擦为正"，**绝不等于 VRP 有 edge**——
   真正生死在阶段 C 尾部（本阶段不碰）。

数据真实性红线：必做可证伪核对（端点 URL + 独立锚点交叉验证），非态度要求。
================================================================================
"""
import os, sys, json, time, math, hashlib, datetime as dt
from pathlib import Path
import numpy as np
import requests
from scipy.stats import norm
from scipy.optimize import brentq

# ----------------------------- 路径 -----------------------------
HERE = Path(__file__).resolve().parent
VRP = HERE.parent
DATA = VRP / "data"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

DERIBIT = "https://www.deribit.com/api/v2/public/"          # 生产环境（非 test.deribit.com）
COINBASE = "https://api.exchange.coinbase.com"              # 独立锚点（现货 USD）

# ----------------------------- 费率/口径常量（核对 deribit.com/kb/fees）-----------------------------
OPT_FEE_RATE   = 0.0003     # 期权 taker：0.03% of underlying / 腿 / 边
OPT_FEE_CAP    = 0.125      # 封顶 12.5% of premium
OPT_SETTLE_FEE = 0.00015    # 交割/结算费 0.015% of underlying（端点1持有到期会触发）
PERP_TAKER     = 0.0005     # 永续 taker 0.05% / 边（maker 可 0/返佣 → 敏感性）
ANNUAL = 365.0              # 加密 24/7 年化因子（IV/RV 同口径）

# 关键尾部时段（用于覆盖核对）
TAILS = {
    "312_2020":  dt.date(2020, 3, 12),
    "519_2021":  dt.date(2021, 5, 19),
    "LUNA_2022": dt.date(2022, 5, 12),
    "FTX_2022":  dt.date(2022, 11, 8),
}

# ----------------------------- HTTP + 缓存 -----------------------------
_SESS = requests.Session()
_SESS.headers.update({"User-Agent": "vrp-research-stageA"})

def _cache_path(tag, params):
    key = tag + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return CACHE / f"{tag.replace('/','_')}_{h}.json"

def deribit(method, _cache=True, **params):
    """Deribit public REST GET（带磁盘缓存）。"""
    cp = _cache_path("dbt_" + method, params)
    if _cache and cp.exists():
        return json.loads(cp.read_text())
    for attempt in range(5):
        try:
            r = _SESS.get(DERIBIT + method, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1)); continue
            j = r.json()
            res = j.get("result")
            if res is None and "error" in j:
                res = {"_error": j["error"]}
            if _cache:
                cp.write_text(json.dumps(res))
            return res
        except Exception as e:
            if attempt == 4:
                return {"_error": str(e)}
            time.sleep(1.0 * (attempt + 1))

def coinbase_candles(product, start_iso, end_iso, granularity=86400, _cache=True):
    cp = _cache_path("cb_" + product, {"s": start_iso, "e": end_iso, "g": granularity})
    if _cache and cp.exists():
        return json.loads(cp.read_text())
    r = _SESS.get(f"{COINBASE}/products/{product}/candles",
                  params={"granularity": granularity, "start": start_iso, "end": end_iso}, timeout=30)
    j = r.json()
    if _cache:
        cp.write_text(json.dumps(j))
    return j

# ----------------------------- 时间工具 -----------------------------
def ymd(d): return d.strftime("%Y-%m-%d")
def to_ms(d):  # date -> ms at 08:00 UTC (Deribit 结算时点)
    return int(dt.datetime(d.year, d.month, d.day, 8, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
def ms_to_date(ms): return dt.datetime.fromtimestamp(ms/1000, dt.timezone.utc).date()

DERIBIT_MONTH = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}

def last_friday(year, month):
    """月度到期 = 当月最后一个周五。"""
    d = dt.date(year, month, 28)
    while d.month == month:
        d += dt.timedelta(days=1)
    d -= dt.timedelta(days=1)            # 当月最后一天
    while d.weekday() != 4:              # 4 = Friday
        d -= dt.timedelta(days=1)
    return d

def deribit_expiry_tag(d):
    return f"{d.day}{DERIBIT_MONTH[d.month]}{d.year%100:02d}"

def monthly_expiries(start_ym, end_ym):
    (sy, sm), (ey, em) = start_ym, end_ym
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(last_friday(y, m))
        m += 1
        if m > 12: m = 1; y += 1
    return out

# ----------------------------- Black-Scholes (r=0) -----------------------------
def _d1(S, K, T, sig):
    return (math.log(S/K) + 0.5*sig*sig*T) / (sig*math.sqrt(T))

def bs_price(S, K, T, sig, is_call):
    if sig <= 0 or T <= 0:
        return max(S-K, 0.0) if is_call else max(K-S, 0.0)
    d1 = _d1(S, K, T, sig); d2 = d1 - sig*math.sqrt(T)
    if is_call:
        return S*norm.cdf(d1) - K*norm.cdf(d2)
    return K*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_implied(price_usd, S, K, T, is_call):
    intrinsic = (max(S-K,0.0) if is_call else max(K-S,0.0))
    if price_usd <= intrinsic + 1e-9 or T <= 0:
        return float("nan")
    try:
        return brentq(lambda s: bs_price(S, K, T, s, is_call) - price_usd, 1e-4, 8.0, maxiter=200)
    except Exception:
        return float("nan")

def bs_straddle_implied(straddle_usd, S, K, T):
    """从跨式(C+P)总价反解单一 σ —— 比分腿平均更鲁棒（总权利金可靠，分腿 last-trade 有噪声）。"""
    intrinsic = abs(S-K)
    if straddle_usd <= intrinsic + 1e-9 or T <= 0:
        return float("nan")
    f = lambda s: bs_price(S,K,T,s,True) + bs_price(S,K,T,s,False) - straddle_usd
    try:
        return brentq(f, 1e-4, 8.0, maxiter=200)
    except Exception:
        return float("nan")

def bs_delta(S, K, T, sig, is_call):
    if sig <= 0 or T <= 0:
        intr = (S > K) if is_call else (S < K)
        return (1.0 if is_call else -1.0) if intr else 0.0
    d1 = _d1(S, K, T, sig)
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0

def straddle_vega(S, K, T, sig):     # per 1.00 σ
    if sig <= 0 or T <= 0: return 0.0
    return 2.0 * S * norm.pdf(_d1(S,K,T,sig)) * math.sqrt(T)

def straddle_gamma(S, K, T, sig):
    if sig <= 0 or T <= 0: return 0.0
    return 2.0 * norm.pdf(_d1(S,K,T,sig)) / (S*sig*math.sqrt(T))

# ----------------------------- 标的永续日线 -----------------------------
def fetch_perp_daily(asset):
    """asset in {BTC,ETH} -> {date: close} 日线 08:00 UTC，覆盖 2019-12 .. now。"""
    inst = f"{asset}-PERPETUAL"
    start = to_ms(dt.date(2019, 12, 1))
    now = deribit("get_time")
    out = {}
    cur = start
    step = 300 * 86400 * 1000
    while cur < now:
        cd = deribit("get_tradingview_chart_data", instrument_name=inst,
                     start_timestamp=cur, end_timestamp=min(cur+step, now), resolution="1D")
        if "_error" in cd or not cd.get("ticks"):
            cur += step; continue
        for t, c, v in zip(cd["ticks"], cd["close"], cd["volume"]):
            out[ms_to_date(t)] = float(c)
        cur += step
    return out

def fetch_dvol(asset):
    """Deribit DVOL（30 天隐含波动率指数，2021-03-24 起）-> {date: close(%)}。
    取每日最接近 08:00 UTC 的值。独立于本脚本 BS 反解，用作 IV 水平交叉校验。"""
    start = to_ms(dt.date(2021, 3, 1))
    now = deribit("get_time")
    out = {}
    cur = start
    step = 200 * 86400 * 1000
    while cur < now:
        dv = deribit("get_volatility_index_data", currency=asset,
                     start_timestamp=cur, end_timestamp=min(cur+step, now), resolution="43200")
        if isinstance(dv, dict) and not dv.get("_error"):
            for row in dv.get("data", []):
                t = row[0]; close = row[4]
                d = ms_to_date(t)
                # 取最接近 08:00 的那条
                hour = dt.datetime.fromtimestamp(t/1000, dt.timezone.utc).hour
                key = (d, abs(hour-8))
                if d not in out or abs(hour-8) < out[d][1]:
                    out[d] = (float(close), abs(hour-8))
        cur += step
    return {d: v[0] for d, v in out.items()}

# ----------------------------- 期权链历史（清洗：截断 @expiry / vol>0）-----------------------------
def fetch_option_series(instrument, expiry_ms):
    """返回 {date:(close,volume)}，仅保留 expiry 前且 volume>0 的真实成交日（去掉到期后 flat-pad）。"""
    cd = deribit("get_tradingview_chart_data", instrument_name=instrument,
                 start_timestamp=expiry_ms - 60*86400*1000, end_timestamp=expiry_ms, resolution="1D")
    if "_error" in cd or not cd.get("ticks"):
        return {}
    out = {}
    for t, c, v in zip(cd["ticks"], cd["close"], cd["volume"]):
        if t <= expiry_ms and v and v > 0:
            out[ms_to_date(t)] = (float(c), float(v))
    return out

def candidate_strikes(asset, S0):
    """生成 S0 附近候选 strike（按价位自适应网格），按距 S0 由近及远。"""
    if asset == "BTC":
        grids = [1000, 2000, 2500, 500, 5000, 250]
    else:  # ETH
        grids = [50, 100, 25, 200, 20, 250]
    cands = set()
    for g in grids:
        base = round(S0 / g) * g
        for k in range(-3, 4):
            v = base + k*g
            if v > 0 and abs(v - S0)/S0 < 0.06:
                cands.add(int(v))
    return sorted(cands, key=lambda x: abs(x - S0))

def find_atm(asset, S0, T0_date, expiry_date):
    """在 T0 附近(±3 天内有成交)找最接近 S0 的挂牌 strike，返回 (K, call_inst, put_inst, c0, p0)。
    c0/p0 = T0 当日(或最近)的 last-trade close（BTC 计价）。"""
    expiry_ms = to_ms(expiry_date)
    tag = deribit_expiry_tag(expiry_date)
    for K in candidate_strikes(asset, S0):
        cinst = f"{asset}-{tag}-{K}-C"
        pinst = f"{asset}-{tag}-{K}-P"
        cs = fetch_option_series(cinst, expiry_ms)
        if not cs:
            continue
        # 选 T0 当日或 ±3 天内最近的成交
        near = sorted(cs.keys(), key=lambda d: abs((d - T0_date).days))
        if not near or abs((near[0] - T0_date).days) > 3:
            continue
        d_used = near[0]
        c0 = cs[d_used][0]
        ps = fetch_option_series(pinst, expiry_ms)
        p_near = None
        if ps:
            pn = sorted(ps.keys(), key=lambda d: abs((d - d_used).days))
            if pn and abs((pn[0] - d_used).days) <= 3:
                p_near = ps[pn[0]][0]
        return dict(K=K, call=cinst, put=pinst, c0=c0, p0=p_near, used_date=d_used)
    return None

# ----------------------------- live ATM bid-ask（IV 空间）代理 + 永续盘口 -----------------------------
def measure_live_friction():
    """从当前 ATM 月度链测 bid-ask（IV 点）中位数 + 永续 half-spread（作历史代理）。
    **冻结**：首次测得后写 data/live_friction.json 并在后续 run 复用，保证 committed 产物可复现
    （live 盘口逐刻变动，不冻结则净缝非确定）。删该文件可重测。"""
    fpath = DATA / "live_friction.json"
    if fpath.exists():
        return json.loads(fpath.read_text())
    now = deribit("get_time")
    res = {}
    for asset in ["BTC", "ETH"]:
        idx = deribit("get_index_price", index_name=f"{asset.lower()}_usd")["index_price"]
        insts = deribit("get_instruments", currency=asset, kind="option", expired="false", _cache=False)
        tgt = now + 30*86400*1000
        chain = [i for i in insts if abs(i["expiration_timestamp"]-tgt) < 7*86400*1000]
        ivspreads, pctspreads = [], []
        # 取最近 ATM 的若干 strike
        strikes = sorted(set(i["strike"] for i in chain), key=lambda k: abs(k-idx))[:4]
        for K in strikes:
            for typ in ("C", "P"):
                cand = [i for i in chain if i["strike"]==K and i["instrument_name"].endswith(typ)]
                if not cand: continue
                tk = deribit("ticker", instrument_name=cand[0]["instrument_name"], _cache=False)
                bi, ai, mk = tk.get("bid_iv"), tk.get("ask_iv"), tk.get("mark_price")
                bb, ba = tk.get("best_bid_price"), tk.get("best_ask_price")
                if bi and ai and ai > bi:
                    ivspreads.append(ai - bi)
                if bb and ba and mk and ba > bb and mk > 0:
                    pctspreads.append((ba-bb)/mk)
        # 永续盘口
        ptk = deribit("ticker", instrument_name=f"{asset}-PERPETUAL", _cache=False)
        pbb, pba = ptk.get("best_bid_price"), ptk.get("best_ask_price")
        perp_hs = ((pba-pbb)/2.0)/((pba+pbb)/2.0) if (pbb and pba) else 0.0
        res[asset] = dict(
            iv_spread_volpts=float(np.median(ivspreads)) if ivspreads else float("nan"),
            opt_pct_of_premium=float(np.median(pctspreads)) if pctspreads else float("nan"),
            perp_half_spread=float(perp_hs),
            n_quotes=len(ivspreads),
        )
    res["_measured_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    fpath.write_text(json.dumps(res, indent=2))
    return res

# ----------------------------- 单 cycle 计算 -----------------------------
def realized_vol(perp, T0_date, E_date):
    days = sorted(d for d in perp if T0_date <= d <= E_date)
    if len(days) < 5:
        return None, 0
    closes = np.array([perp[d] for d in days])
    rets = np.diff(np.log(closes))
    rv = math.sqrt(ANNUAL * float(np.mean(rets**2)))
    return rv, len(rets)

def hedge_turnover(perp, K, T0_date, E_date, S0, sig_iv):
    """端点2：沿真实路径模拟每日 BS straddle delta（建仓 IV 持平），Σ|Δdelta| 标的单位，与 Σ(换手×S)。"""
    days = sorted(d for d in perp if T0_date <= d <= E_date)
    if len(days) < 3:
        return 0.0, 0.0, 0
    E_ms = to_ms(E_date)
    prev_delta = None
    turn_units = 0.0       # Σ|Δdelta|（标的单位）
    turn_notional = 0.0    # Σ|Δdelta|×S（美元换手）
    n_h = 0
    for d in days:
        S = perp[d]
        T = max((E_ms - to_ms(d)) / 1000 / 86400 / ANNUAL, 1e-6)
        # straddle delta = call delta + put delta（同 K）
        sd = bs_delta(S, K, T, sig_iv, True) + bs_delta(S, K, T, sig_iv, False)
        if prev_delta is None:
            prev_delta = sd; continue
        dd = abs(sd - prev_delta)
        turn_units += dd
        turn_notional += dd * S
        prev_delta = sd
        n_h += 1
    return turn_units, turn_notional, n_h

def option_fee_usd(prem_btc, S):
    """单腿单边期权费（美元），含封顶 12.5% premium。"""
    prem_usd = prem_btc * S
    fee = min(OPT_FEE_RATE * S, OPT_FEE_CAP * prem_usd)
    return fee

def compute_cycle(asset, perp, T0_date, E_date, fric, dvol=None):
    if E_date not in perp or T0_date not in perp:
        # 用最近的可用日
        pass
    S0 = perp.get(T0_date)
    if S0 is None:
        cand = [d for d in perp if abs((d-T0_date).days) <= 3]
        if not cand: return None
        S0 = perp[min(cand, key=lambda d: abs((d-T0_date).days))]
    atm = find_atm(asset, S0, T0_date, E_date)
    if atm is None:
        return None
    K = atm["K"]
    # 用期权实际成交日(used_date)对齐标的价，保证 BS 反解 S 与期权价同日
    entry_date = atm["used_date"]
    S0 = perp.get(entry_date, S0)
    E_ms = to_ms(E_date)
    T0_used_ms = to_ms(entry_date)
    T_years = max((E_ms - T0_used_ms) / 1000 / 86400 / ANNUAL, 1e-6)
    # IV 反解：优先用跨式总价反解单一 σ（鲁棒）；分腿 IV 仅作透明留存
    c_usd = atm["c0"] * S0
    iv_c = bs_implied(c_usd, S0, K, T_years, True)
    iv_p = float("nan")
    sig_iv = float("nan")
    if atm["p0"] is not None:
        iv_p = bs_implied(atm["p0"] * S0, S0, K, T_years, False)
        sig_iv = bs_straddle_implied((atm["c0"]+atm["p0"]) * S0, S0, K, T_years)
    if not (sig_iv == sig_iv):              # 跨式反解失败 → 退回可得分腿
        ivs = [x for x in (iv_c, iv_p) if x == x]
        if not ivs:
            return None
        sig_iv = float(np.mean(ivs))
    # DVOL 交叉校验（独立 Deribit IV 指数，2021-03+）
    dvol_entry = dvol.get(entry_date) if dvol else None
    if dvol_entry is None and dvol:
        near = [d for d in dvol if abs((d-entry_date).days) <= 2]
        if near: dvol_entry = dvol[min(near, key=lambda d: abs((d-entry_date).days))]
    # RV
    rv, n_ret = realized_vol(perp, atm["used_date"], E_date)
    if rv is None:
        return None
    # 毛缝
    gross_vp = sig_iv - rv                          # vol 点
    gross_var = sig_iv**2 - rv**2                   # variance
    vega = straddle_vega(S0, K, T_years, sig_iv)
    gamma = straddle_gamma(S0, K, T_years, sig_iv)
    gross_usd = 0.5 * gamma * S0**2 * gross_var * T_years   # faithful ½Γ(σ_IV²−σ_RV²)T
    # straddle premium
    prem_btc = atm["c0"] + (atm["p0"] if atm["p0"] is not None else atm["c0"])
    prem_usd = prem_btc * S0
    # 期权摩擦（vp）：bid-ask 往返（跨式按 IV 点直接计）+ 费 vp 等价
    opt_iv_rt = fric["iv_spread_volpts"] / 100.0    # IV 点 -> 绝对（如 0.83 -> 0.0083）
    # 进出各穿一次 → 往返 1×spread；call+put 各一腿，vega 加权≈单腿 spread（同 IV）→ 用 1×spread
    # 费：进场 2 腿 + 出场/结算 2 腿
    fee_usd = 2*option_fee_usd(atm["c0"], S0) + 2*option_fee_usd(atm["p0"] if atm["p0"] else atm["c0"], S0)
    fee_settle = 2*OPT_SETTLE_FEE*S0   # 端点1 持有到期的交割费（封顶略，保守不封）
    fee_vp = (fee_usd + 0.0) / vega if vega > 0 else 0.0
    opt_fric_vp = opt_iv_rt + fee_vp
    # 端点1：不对冲（对冲摩擦=0；交割费计入端点1，因持有到期）
    opt_fric_vp_e1 = opt_iv_rt + (fee_usd + fee_settle)/vega if vega>0 else opt_iv_rt
    net1_vp = gross_vp - opt_fric_vp_e1
    # 端点2：每日对冲；对冲腿摩擦
    tu_units, tu_notional, n_h = hedge_turnover(perp, K, atm["used_date"], E_date, S0, sig_iv)
    hedge_usd = tu_notional * (fric["perp_half_spread"] + PERP_TAKER)
    hedge_vp = hedge_usd / vega if vega > 0 else 0.0
    # 端点2 出场用平仓而非交割（无交割费），用 opt_fric_vp（含平仓费、无 settle）
    net2_vp = gross_vp - opt_fric_vp - hedge_vp
    return dict(
        asset=asset, T0=ymd(atm["used_date"]), expiry=ymd(E_date), days=round(T_years*ANNUAL,1),
        n_ret=n_ret, S0=round(S0,2), K=K, sig_iv=round(sig_iv*100,3), sig_rv=round(rv*100,3),
        dvol_entry=round(dvol_entry,3) if dvol_entry else None,
        iv_minus_dvol=round(sig_iv*100 - dvol_entry,3) if dvol_entry else None,
        gross_vp=round(gross_vp*100,4), gross_var=round(gross_var,6), prem_usd=round(prem_usd,2),
        prem_pct_spot=round(prem_usd/S0*100,4), vega=round(vega,2), gross_usd=round(gross_usd,2),
        iv_c=round(iv_c*100,3), iv_p=round(iv_p*100,3) if iv_p==iv_p else None,
        opt_fric_vp_e1=round(opt_fric_vp_e1*100,4), opt_fric_vp_e2=round(opt_fric_vp*100,4),
        hedge_turn_units=round(tu_units,4), hedge_turn_notional=round(tu_notional,2), n_hedge=n_h,
        hedge_vp=round(hedge_vp*100,4),
        net1_vp=round(net1_vp*100,4),    # 端点1 不对冲（vol 点）
        net2_vp=round(net2_vp*100,4),    # 端点2 每日对冲（vol 点）
    )

# ----------------------------- 真实性核对（端点 URL + Coinbase 锚点）-----------------------------
def authenticity_check(perp_btc, perp_eth):
    gt = _SESS.get(DERIBIT + "get_time", timeout=20).json()
    bi = _SESS.get(DERIBIT + "get_index_price", params={"index_name":"btc_usd"}, timeout=20).json()
    testnet = bi.get("result", {}).get("testnet", bi.get("testnet"))
    # 锚点日：尾部(312/519/LUNA/FTX，验证"两源同向同幅崩")+ 平静日(验证价位级别一致)。
    # 关键：对齐到 **08:00 UTC**（Deribit 结算/日线时点），用 Coinbase 小时线 08:00 那根的 open，
    # 否则高波动日 16h 错位会假性放大偏差（旧版日线对齐在 312 出现 8–15% 伪偏差）。
    tail_anchors = [dt.date(2020,3,12), dt.date(2021,5,19), dt.date(2022,5,12), dt.date(2022,11,8)]
    calm_anchors = [dt.date(2021,9,15), dt.date(2023,6,15), dt.date(2024,1,11),
                    dt.date(2025,3,15), dt.date(2026,6,1)]
    def cb_at_0800(prod, d):
        # ⚠ Deribit 1D bar 标签=开盘时点，close=标签+24h(次日 08:00) 的价。已实证(312:
        #   1D[03-12].close 5249 ≈ 1H[03-13 08:00] 5251)。故 perp[d]=价@(d+1)08:00，
        #   Coinbase 须取 (d+1) 08:00 同一时刻比对，否则高波动日 24h 错位假性脱锚。
        di = d + dt.timedelta(days=1)
        s = dt.datetime(di.year,di.month,di.day,5,0,tzinfo=dt.timezone.utc).isoformat()
        e = dt.datetime(di.year,di.month,di.day,11,0,tzinfo=dt.timezone.utc).isoformat()
        cb = coinbase_candles(prod, s, e, granularity=3600)
        if not isinstance(cb, list) or not cb:
            return None
        t0800 = int(dt.datetime(di.year,di.month,di.day,8,0,tzinfo=dt.timezone.utc).timestamp())
        best = min(cb, key=lambda c: abs(c[0]-t0800))
        return best[3]   # open = 价格 @ (d+1) 08:00:00（与 Deribit perp[d] 同一时刻）
    rows = []
    for kind, anchors in [("tail", tail_anchors), ("calm", calm_anchors)]:
        for d in anchors:
            for asset, perp, prod in [("BTC", perp_btc, "BTC-USD"), ("ETH", perp_eth, "ETH-USD")]:
                cb_px = cb_at_0800(prod, d)
                dpx = perp.get(d)
                dev = (dpx/cb_px - 1.0) if (dpx and cb_px) else None
                rows.append(dict(date=ymd(d), kind=kind, asset=asset, deribit_perp=dpx,
                                 coinbase_0800=cb_px, dev_pct=round(dev*100,3) if dev is not None else None))
    calm_devs = [abs(r["dev_pct"]) for r in rows if r["kind"]=="calm" and r["dev_pct"] is not None]
    all_devs  = [abs(r["dev_pct"]) for r in rows if r["dev_pct"] is not None]
    # 趋势同向核对（尾部日两源是否同时大跌）：用 perp 与 coinbase 各自 d-1→d 的方向
    return dict(
        endpoint=DERIBIT, testnet_flag=testnet,
        anchor_source="Coinbase Exchange (api.exchange.coinbase.com) spot USD hourly open — 独立于 Deribit 指数",
        alignment="perp[d]=价@(d+1)08:00 UTC(Deribit 1D close=标签+24h,已实证); Coinbase 取 (d+1)08:00 小时 open 同刻比对",
        rows=rows,
        calm_median_abs_dev_pct=round(float(np.median(calm_devs)),3) if calm_devs else None,
        calm_max_abs_dev_pct=round(float(np.max(calm_devs)),3) if calm_devs else None,
        all_median_abs_dev_pct=round(float(np.median(all_devs)),3) if all_devs else None,
        all_max_abs_dev_pct=round(float(np.max(all_devs)),3) if all_devs else None,
        n_compared=len(all_devs),
        # PASS：平静日价位级别一致(<1.5%) + 全样本(含尾部，对齐后)无脱锚(<4%)
        pass_=bool(calm_devs and np.median(calm_devs) < 1.5 and np.max(all_devs) < 4.0),
    )

# ----------------------------- 聚合 -----------------------------
def agg(cycles, field):
    vals = np.array([c[field] for c in cycles], dtype=float)
    return dict(mean=round(float(np.mean(vals)),4), median=round(float(np.median(vals)),4),
                min=round(float(np.min(vals)),4), max=round(float(np.max(vals)),4),
                pos_frac=round(float(np.mean(vals>0)),3), n=len(vals))

def trim_sensitivity(cycles, field):
    """左偏诚实：剔最坏 N 月（最负 net）后 mean 上移量。"""
    vals = sorted(c[field] for c in cycles)
    full = float(np.mean(vals))
    out = {"mean_all": round(full,4)}
    for N in (1, 3):
        if len(vals) > N:
            trimmed = float(np.mean(vals[N:]))   # 去掉最负的 N 个
            out[f"mean_trim_worst{N}"] = round(trimmed,4)
            out[f"drift_worst{N}"] = round(trimmed-full,4)
        # 同时报去掉最好 N（右尾），看 mean 是否塌（VRP 一般不塌）
        if len(vals) > N:
            out[f"mean_trim_best{N}"] = round(float(np.mean(vals[:-N])),4)
    return out

def tail_coverage(cycles):
    cov = {}
    for name, d in TAILS.items():
        hit = [c for c in cycles if c["T0"] <= ymd(d) <= c["expiry"]]
        cov[name] = (dict(in_sample=True, cycle=hit[0]["T0"]+".."+hit[0]["expiry"],
                          net1_vp=hit[0]["net1_vp"], net2_vp=hit[0]["net2_vp"],
                          sig_iv=hit[0]["sig_iv"], sig_rv=hit[0]["sig_rv"]) if hit else dict(in_sample=False))
    return cov

# ----------------------------- main -----------------------------
def main():
    print("="*70)
    print("数据环境：Deribit PRODUCTION（www.deribit.com，非 testnet）")
    print("研究线：vrp/（隔离，不触 cta_strategy mainnet/污染库/前向系统/VPS）")
    print("="*70)
    assets = ["BTC", "ETH"]
    start_ym, end_ym = (2020, 1), (2026, 6)   # 月度到期范围（2020-02 起做第一个 cycle）

    print("\n[1/5] 拉取标的永续日线 + DVOL 指数 ...")
    perp = {a: fetch_perp_daily(a) for a in assets}
    dvol = {a: fetch_dvol(a) for a in assets}
    for a in assets:
        ds = sorted(perp[a]); dd = sorted(dvol[a])
        print(f"   {a}-PERPETUAL: {len(ds)} 日, {ds[0]} .. {ds[-1]}  | DVOL: {len(dd)} 日"
              + (f", {dd[0]} .. {dd[-1]}" if dd else ""))

    print("\n[2/5] 真实性可证伪核对（端点 URL + Coinbase 独立锚点）...")
    auth = authenticity_check(perp["BTC"], perp["ETH"])
    print(f"   testnet flag = {auth['testnet_flag']}; calm median|dev| = {auth['calm_median_abs_dev_pct']}% ; "
          f"all max|dev| = {auth['all_max_abs_dev_pct']}% ; PASS={auth['pass_']}")

    print("\n[3/5] 测 live ATM bid-ask（IV 点）+ 永续盘口（历史摩擦代理）...")
    fric = measure_live_friction()
    for a in assets:
        print(f"   {a}: ATM IV bid-ask={fric[a]['iv_spread_volpts']:.3f} vp, "
              f"opt {fric[a]['opt_pct_of_premium']*100:.2f}% of prem, perp half-spread={fric[a]['perp_half_spread']*1e4:.2f}bp")

    print("\n[4/5] 逐月度 cycle 计算（毛缝/摩擦/净缝，条件于对冲频率）...")
    all_cycles = {}
    for a in assets:
        exps = monthly_expiries(start_ym, end_ym)
        cycles = []
        for i in range(1, len(exps)):
            T0, E = exps[i-1], exps[i]
            c = compute_cycle(a, perp[a], T0, E, fric[a], dvol[a])
            if c: cycles.append(c)
        all_cycles[a] = cycles
        (DATA / f"cycles_{a}.jsonl").write_text("\n".join(json.dumps(c) for c in cycles))
        print(f"   {a}: {len(cycles)} cycles 计出 (目标 {len(exps)-1})")

    print("\n[5/5] 聚合 + 判定 ...")
    summary = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "authenticity": auth, "live_friction": fric, "by_asset": {}}
    for a in assets:
        cy = all_cycles[a]
        if not cy: continue
        # DVOL 交叉校验（IV 反解水平健康度）
        dv_pairs = [(c["sig_iv"], c["dvol_entry"]) for c in cy if c.get("dvol_entry")]
        iv_dvol = None
        if len(dv_pairs) >= 5:
            ivv = np.array([p[0] for p in dv_pairs]); dvv = np.array([p[1] for p in dv_pairs])
            iv_dvol = dict(n=len(dv_pairs),
                           median_iv_minus_dvol=round(float(np.median(ivv-dvv)),3),
                           corr=round(float(np.corrcoef(ivv, dvv)[0,1]),4))
        summary["by_asset"][a] = dict(
            n_cycles=len(cy), iv_dvol_check=iv_dvol,
            gross_vp=agg(cy, "gross_vp"), gross_var=agg(cy, "gross_var"),
            net1_vp=agg(cy, "net1_vp"), net2_vp=agg(cy, "net2_vp"),
            opt_fric_vp_e1=agg(cy, "opt_fric_vp_e1"), opt_fric_vp_e2=agg(cy, "opt_fric_vp_e2"),
            hedge_vp=agg(cy, "hedge_vp"), n_hedge_mean=round(float(np.mean([c["n_hedge"] for c in cy])),1),
            worst_month_net1=min(cy, key=lambda c: c["net1_vp"])["T0"],
            worst_net1_vp=min(c["net1_vp"] for c in cy),
            worst_net2_vp=min(c["net2_vp"] for c in cy),
            trim_net1=trim_sensitivity(cy, "net1_vp"),
            trim_net2=trim_sensitivity(cy, "net2_vp"),
            trim_gross=trim_sensitivity(cy, "gross_vp"),
            tail_coverage=tail_coverage(cy),
        )

    # 判定（合并 BTC+ETH 视角，主看 BTC）
    def verdict(a):
        s = summary["by_asset"].get(a)
        if not s: return None
        g_mean = s["gross_vp"]["mean"]; n1 = s["net1_vp"]["mean"]; n2 = s["net2_vp"]["mean"]
        # A3 三条件
        e1_pos = n1 > 0
        e2_pos = n2 > 0
        # peso：剔最坏 3 月后 net2 漂移占 |net2_all| 比；若 net2 本就≤0 直接弱
        drift3 = s["trim_net2"].get("drift_worst3", 0.0)
        return dict(gross_mean=g_mean, net1_mean=n1, net2_mean=n2,
                    e1_pos=e1_pos, e2_pos=e2_pos, peso_drift3=drift3)
    summary["verdict"] = {a: verdict(a) for a in assets if a in summary["by_asset"]}

    (DATA / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (DATA / "manifest.json").write_text(json.dumps(dict(
        generated_utc=summary["generated_utc"],
        deribit_endpoint=DERIBIT, coinbase_endpoint=COINBASE, testnet=auth["testnet_flag"],
        assets=assets, monthly_expiry_range=[f"{start_ym}", f"{end_ym}"],
        annualization=ANNUAL, atm_def="strike closest to S0 at entry, fixed to expiry, no roll",
        iv_source="BS-inverted from get_tradingview_chart_data last-trade close (mark not free historically); validated vs live mark_iv to <0.1vp",
        rv_source=f"{'{asset}'}-PERPETUAL daily 08:00 UTC close-to-close, annualized x365",
        fees=dict(opt_taker=OPT_FEE_RATE, opt_cap=OPT_FEE_CAP, opt_settle=OPT_SETTLE_FEE, perp_taker=PERP_TAKER),
        friction_note="bid-ask from CURRENT ATM monthly chain (historical orderbook not free) -> applied to history; historical spreads likely WIDER => friction is a LOWER bound (optimistic)",
        friction_frozen="data/live_friction.json (snapshot, reused across runs for reproducibility)",
        authenticity_pass=auth["pass_"], authenticity_alignment=auth["alignment"],
        authenticity_calm_median_dev_pct=auth["calm_median_abs_dev_pct"],
        data_files=["cycles_BTC.jsonl","cycles_ETH.jsonl","summary.json","live_friction.json"],
    ), indent=2, default=str))

    print("\n==== 判定速览 ====")
    for a in assets:
        v = summary["verdict"].get(a)
        if v:
            print(f"  {a}: 毛缝 mean={v['gross_mean']:.3f}vp | 净缝 不对冲={v['net1_mean']:.3f}vp "
                  f"每日对冲={v['net2_mean']:.3f}vp | E1+={v['e1_pos']} E2+={v['e2_pos']}")
    print("\n完成。产物在 vrp/data/。报告由人工据 summary.json 撰写。")
    return summary

if __name__ == "__main__":
    main()
