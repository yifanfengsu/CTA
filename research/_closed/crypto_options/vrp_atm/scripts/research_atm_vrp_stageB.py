#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATM VRP — 阶段 B：BTC VRP 是否只是尾部补偿（前瞻情景）
================================================================================
预注册见 vrp/reports/atm_vrp_stageB_premium_truth_20260628/PREREGISTRATION.md
（LOCKED 2026-06-28 commit d4597b2，先于本结果）。标的 BTC（ETH 已 Stage A 判死）。

主口径 = **端点2 delta-hedged（剥离方向）**，沿真实日路径逐日对冲：
   Π = Σ_i ½ Γ_i S_i² (σ_IV²·Δt − r_i²) − 摩擦（期权往返+费 + 每日 ~30 次对冲）
端点1（不对冲，含方向尾部）= 并行"可交易性"口径，归 Stage C，仅并列报。

gate 0 前置：样本方向中性核对（净 Σlog、上/下月、drawup/down），结论 conditional。
B1：端点2 P&L 全分布。
B2：block bootstrap{1,3,6 月}5% 下界（异号判不稳）+ 三情景（历史最坏×2 / 上行 squeeze /
    双向抽插）+ break-even 尾部强度。
B-gate：均值>0 ∧ 5%下界≥0(三档同号) ∧ break-even>历史。不算 Sharpe。
口径不可在看结果后改（铁律 A）。
================================================================================
"""
import sys, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import research_atm_vrp_stageA as A   # 复用 deribit/perp/BS/口径

DATA = A.DATA
RNG = np.random.default_rng(20260628)   # 固定种子，复现

# ----------------------------- 载入 cycles + perp -----------------------------
def load_cycles(asset):
    return [json.loads(l) for l in (DATA / f"cycles_{asset}.jsonl").read_text().splitlines()]

import datetime as dt
def dparse(s): return dt.date.fromisoformat(s)

# ----------------------------- 端点2：路径 delta-hedged P&L -----------------------------
def endpoint2_path_pnl(cyc, perp):
    """沿真实日路径逐日对冲的 short straddle gamma P&L（美元，每 1 单位 notional）。
    Π_gross = Σ ½ Γ_i S_i² (σ²Δt − r_i²)。返回 (Π_gross_usd, n_steps, rv_realized)。"""
    T0, E = dparse(cyc["T0"]), dparse(cyc["expiry"])
    K = cyc["K"]; sig = cyc["sig_iv"]/100.0
    E_ms = A.to_ms(E)
    dates = sorted(d for d in perp if T0 <= d <= E)
    if len(dates) < 3:
        return None
    S = [perp[d] for d in dates]
    pnl = 0.0; rsq = 0.0
    for i in range(len(S)-1):
        Si = S[i]; r = (S[i+1]-Si)/Si
        Ti = max((E_ms - A.to_ms(dates[i]))/1000/86400/A.ANNUAL, 1e-6)
        gam = A.straddle_gamma(Si, K, Ti, sig)
        pnl += 0.5 * gam * Si*Si * (sig*sig*(1.0/A.ANNUAL) - r*r)
        rsq += r*r
    rv = math.sqrt(A.ANNUAL * rsq/(len(S)-1))
    return pnl, len(S)-1, rv

def cycle_pnls(cycles, perp):
    """每 cycle 端点2(路径,净) 与 端点1(不对冲,净)，归一化 %spot 与 vol 点。"""
    out = []
    for c in cycles:
        r = endpoint2_path_pnl(c, perp)
        if r is None: continue
        pnl2_gross, nstep, rv_path = r
        vega = c["vega"]; S0 = c["S0"]; K = c["K"]
        fric2_usd = (c["opt_fric_vp_e2"] + c["hedge_vp"])/100.0 * vega   # 端点2 摩擦$
        pnl2_net = pnl2_gross - fric2_usd
        pnl2_gross_vp = pnl2_gross/vega*100
        # 端点1 不对冲实际 P&L = 权利金 − |S_exp−K| − 期权摩擦(端点1)
        E = dparse(c["expiry"]); S_exp = perp.get(E)
        fric1_usd = c["opt_fric_vp_e1"]/100.0 * vega
        pnl1_net = (c["prem_usd"] - abs(S_exp - K) - fric1_usd) if S_exp else None
        out.append(dict(
            T0=c["T0"], expiry=c["expiry"], S0=S0, K=K, sig_iv=c["sig_iv"], rv_path=round(rv_path*100,2),
            vega=vega, nstep=nstep,
            pnl2_usd=round(pnl2_net,2), pnl2_pct=round(pnl2_net/S0*100,4), pnl2_vp=round(pnl2_net/vega*100,4),
            pnl2_gross_vp=round(pnl2_gross_vp,4), pnl2_gross_pct=round(pnl2_gross/S0*100,4),
            pnl1_usd=round(pnl1_net,2) if pnl1_net is not None else None,
            pnl1_pct=round(pnl1_net/S0*100,4) if pnl1_net is not None else None,
            S_exp=round(S_exp,2) if S_exp else None,
            logret=round(math.log(S_exp/S0),5) if S_exp else None,
        ))
    return out

# ----------------------------- gate 0：样本方向中性 -----------------------------
def gate0_directional(rows):
    logrets = [r["logret"] for r in rows if r["logret"] is not None]
    net = float(np.sum(logrets))
    up = sum(1 for x in logrets if x > 0); down = sum(1 for x in logrets if x < 0)
    # 最大累计 drawup/drawdown of cumulative logret path
    cum = np.cumsum(logrets); run_max = np.maximum.accumulate(cum); run_min = np.minimum.accumulate(cum)
    max_dd = float(np.min(cum - run_max))   # 最深回撤
    max_du = float(np.max(cum - run_min))   # 最大涨幅段
    n = len(logrets)
    extreme = (max(up, down)/n > 0.70) or (abs(net) > 1.0)
    return dict(n=n, net_logret=round(net,4), total_x=round(math.exp(net),2),
                up_months=up, down_months=down, up_frac=round(up/n,3),
                max_drawup=round(max_du,4), max_drawdown=round(max_dd,4),
                extreme=bool(extreme),
                criterion="单向月占比>70% 或 |净Σlog|>1.0")

# ----------------------------- block bootstrap 5% 下界 -----------------------------
def block_bootstrap_mean(vals, block, n_boot=20000, q=5):
    vals = np.asarray(vals); n = len(vals)
    nblk = int(math.ceil(n/block)); means = np.empty(n_boot)
    hi = max(1, n-block+1)
    for b in range(n_boot):
        idx = []
        for _ in range(nblk):
            s = int(RNG.integers(0, hi)); idx.extend(range(s, min(s+block, n)))
        means[b] = vals[idx[:n]].mean()
    return dict(block=block, lb5=round(float(np.percentile(means, q)),4),
                boot_mean=round(float(means.mean()),4),
                lb1=round(float(np.percentile(means,1)),4))

# ----------------------------- 情景（端点2 标准月，方向已剥离）-----------------------------
def scenario_pnl(path_returns, S0=60000.0, sig_iv=0.50):
    """给定日收益序列，算 ATM straddle 端点2 路径 gamma P&L（%spot 与 vol 点）。"""
    K = S0; n = len(path_returns); E_days = n
    S = [S0];
    for r in path_returns: S.append(S[-1]*(1+r))
    pnl = 0.0; rsq=0.0
    for i in range(n):
        Si = S[i]; r = path_returns[i]
        Ti = max((E_days - i)/A.ANNUAL, 1e-6)
        gam = A.straddle_gamma(Si, K, Ti, sig_iv)
        pnl += 0.5*gam*Si*Si*(sig_iv*sig_iv*(1.0/A.ANNUAL) - r*r); rsq+=r*r
    vega = A.straddle_vega(S0, K, E_days/A.ANNUAL, sig_iv)
    rv = math.sqrt(A.ANNUAL*rsq/n)
    # 摩擦：端点2 ~2vp 量级（Stage A），粗加 2.0vp
    fric = 0.02*vega
    return dict(pnl_pct=round((pnl-fric)/S0*100,3), pnl_vp=round((pnl-fric)/vega*100,3),
                rv=round(rv*100,1), end_move=round((S[-1]/S0-1)*100,1))

def build_scenarios(worst_pnl_pct):
    sc = {}
    # ① 历史最坏 × 2（直接 P&L 翻倍）
    sc["hist_worst_x2"] = dict(pnl_pct=round(worst_pnl_pct*2,3), note="实测最坏端点2月 P&L×2")
    # ② 上行 squeeze +60% + IV 暴涨：均值漂移 + 高 realized
    n=30; drift=math.log(1.60)/n
    rets=[drift + RNG.normal(0,0.06) for _ in range(n)]   # ~115% realized
    sc["upside_squeeze_+60%"] = {**scenario_pnl(rets, sig_iv=0.50), "note":"+60%月+高realized(IV暴涨同时被卖在高位)"}
    # ③ 双向抽插 whipsaw：±8%/日交替，净~0，realized 极高（端点2 最坏）
    rets=[(0.08 if i%2==0 else -0.08) for i in range(30)]
    sc["whipsaw_pm8"] = {**scenario_pnl(rets, sig_iv=0.50), "note":"±8%/日抽插,净≈0,realized极高,压对冲路径"}
    return sc

# ----------------------------- break-even 尾部强度 -----------------------------
def breakeven_tail(pnls_pct, tail_pct=10.0):
    """尾部=最差 tail_pct% 分位（数据驱动，端点2 分布被对冲压窄，固定 −10% 阈值会无尾）。"""
    arr = np.asarray(pnls_pct)
    thr = float(np.percentile(arr, tail_pct))
    tail = arr[arr <= thr]; normal = arr[arr > thr]
    if len(tail)==0 or len(normal)==0:
        return dict(note="无尾/全尾,不适用", tail_pct=tail_pct)
    mu_n = float(normal.mean()); L = float(-tail.mean())   # 平均尾损(正数,若尾仍>0则为负)
    p_hist = len(tail)/len(arr)
    p_star = mu_n/(mu_n + L) if (mu_n+L)>0 else float("nan")   # E=0 的尾频
    return dict(tail_def=f"bottom {tail_pct}% (≤{round(thr,3)}%spot)", n_tail=len(tail), p_hist=round(p_hist,4),
                mu_normal=round(mu_n,3), mean_tail_loss=round(L,3),
                p_breakeven=round(p_star,4) if p_star==p_star else None,
                survives=bool(p_star==p_star and p_star > p_hist),
                note="p*>p_hist=溢价扛过历史尾部; p*<=p_hist=peso/只是尾部补偿; mean≈0 则已在break-even")

# ----------------------------- main -----------------------------
def main():
    print("="*70); print("ATM VRP 阶段 B — BTC VRP 真伪（端点2 剥离方向）· 预注册 d4597b2"); print("="*70)
    perp = A.fetch_perp_daily("BTC")
    cycles = load_cycles("BTC")
    rows = cycle_pnls(cycles, perp)
    print(f"\nBTC cycles with endpoint-2 path P&L: {len(rows)}")

    g0 = gate0_directional(rows)
    print(f"\n[gate0 方向中性] net Σlog={g0['net_logret']} (×{g0['total_x']}); up/down={g0['up_months']}/{g0['down_months']} (up {g0['up_frac']*100:.0f}%); "
          f"maxDU={g0['max_drawup']} maxDD={g0['max_drawdown']} → 样本方向极端? {g0['extreme']} [{g0['criterion']}]")

    p2 = [r["pnl2_pct"] for r in rows]
    p2vp = [r["pnl2_vp"] for r in rows]
    p1 = [r["pnl1_pct"] for r in rows if r["pnl1_pct"] is not None]
    def stats(v):
        v=np.asarray(v); from scipy.stats import skew
        return dict(mean=round(float(v.mean()),4), median=round(float(np.median(v)),4),
                    min=round(float(v.min()),3), max=round(float(v.max()),3),
                    skew=round(float(skew(v)),3), pos=round(float((v>0).mean()),3), n=len(v))
    p2g = [r["pnl2_gross_vp"] for r in rows]
    print(f"\n[B1 端点2 P&L %spot] {stats(p2)}")
    print(f"[B1 端点2 净 vol点] mean={np.mean(p2vp):.3f} median={np.median(p2vp):.3f} | 毛(扣摩擦前) mean={np.mean(p2g):.3f} median={np.median(p2g):.3f}")
    print(f"[并列 端点1 不对冲 %spot · 归Stage C] {stats(p1)}")

    boots = {b: block_bootstrap_mean(p2, b) for b in (1,3,6)}
    print(f"\n[B2 block bootstrap mean 5% 下界 (%spot)]")
    signs=set()
    for b,bb in boots.items():
        print(f"   block={b}月: 5%下界={bb['lb5']}  boot_mean={bb['boot_mean']}  1%下界={bb['lb1']}")
        signs.add(np.sign(bb['lb5']))
    gate_stable = len(signs)==1
    print(f"   三档下界同号(gate稳)? {gate_stable}")

    worst = min(rows, key=lambda r: r["pnl2_pct"])
    scen = build_scenarios(worst["pnl2_pct"])
    print(f"\n[B2 情景] 实测最坏端点2月 = {worst['T0']} {worst['pnl2_pct']}%spot")
    for k,v in scen.items(): print(f"   {k}: {v}")
    # 情景对样本 mean / 下界 影响（各加一个该情景月）
    print("   情景注入后样本 mean / block=3 下界:")
    for k,v in scen.items():
        aug = p2 + [v["pnl_pct"]]
        bb = block_bootstrap_mean(aug, 3, n_boot=8000)
        print(f"     +{k}: mean={np.mean(aug):.3f} (Δ{np.mean(aug)-np.mean(p2):+.3f}) lb5(block3)={bb['lb5']}")

    be = breakeven_tail(p2)
    print(f"\n[B2 break-even 尾部强度] {be}")

    # ---- B-gate 判定 ----
    mean_pos = np.mean(p2) > 0
    lb_nonneg = all(boots[b]['lb5'] >= 0 for b in boots)
    be_survives = be.get("survives", False)
    verdict_pass = bool(mean_pos and lb_nonneg and gate_stable and be_survives)
    print("\n"+"="*70)
    print(f"[B-gate] 均值>0:{mean_pos} | 5%下界≥0(三档):{lb_nonneg} | 三档同号(稳):{gate_stable} | break-even>历史:{be_survives}")
    print(f"[B 判定] {'过(VRP真,进Stage C)' if verdict_pass else '不过(VRP只是尾部补偿/peso)'} ; 结论 conditional 于 gate0(方向极端={g0['extreme']})")
    print("="*70)

    out = dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               prereg_commit="d4597b2", asset="BTC", n_cycles=len(rows),
               gate0=g0, b1_endpoint2_pct=stats(p2), b1_endpoint2_volpts=dict(mean=round(float(np.mean(p2vp)),4),median=round(float(np.median(p2vp)),4)),
               b1_endpoint1_pct=stats(p1), bootstrap=boots, gate_stable=gate_stable,
               worst_month=dict(T0=worst["T0"], pnl2_pct=worst["pnl2_pct"]),
               scenarios=scen, breakeven=be,
               verdict=dict(mean_pos=mean_pos, lb_nonneg=lb_nonneg, gate_stable=gate_stable,
                            be_survives=be_survives, PASS=verdict_pass,
                            conditional_on_gate0_not_extreme=not g0["extreme"]))
    (DATA / "stageB_summary.json").write_text(json.dumps(out, indent=2, default=str))
    (DATA / "stageB_cycles.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("\n产物: vrp/data/stageB_summary.json, stageB_cycles.jsonl")
    return out

if __name__ == "__main__":
    main()
