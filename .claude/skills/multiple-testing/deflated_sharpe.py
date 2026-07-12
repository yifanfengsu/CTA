#!/usr/bin/env python3
"""multiple-testing skill — Deflated Sharpe / N 三口径 / 验证周期 / FDR（可独立 import）。

提炼自 scripts/research_deflated_sharpe.py（reports/trend_methodology_hardening_20260622/，
原文件未修改）。DSR 按 Bailey & López de Prado (2014) 原文实现，无自创变体。

全部 Sharpe 计算在 DAILY 单位；年化 = 日 Sharpe × √periods_per_year（加密 365）。
偏度/峰度用实际分布（kurt 为非超额口径，正态=3），绝不假设正态。

数学恒等式（唯一"反向即错"的 sanity check）：
    N > 1  ⇒  SR*₀ ≥ 0  ⇒  deflated Sharpe ≤ 观测 Sharpe。

自检：python deflated_sharpe.py
  复现 reports/trend_methodology_hardening_20260622/README.md 的 B2_4h 手验明细：
  Var(SR_d)=1.313e-4, N_eff=2.354 → SR*₀_ann≈0.145, deflated_ann≈0.510, DSR≈0.826。
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = [
    "daily_sharpe", "sortino_annual", "expected_max_sharpe", "psr",
    "effective_n_enb", "dsr_report", "stationary_boot_sharpe_se",
    "years_to_sig", "fdr_bh",
]

EULER_GAMMA = 0.5772156649015329
Z95 = 1.959963984540054                     # norm.ppf(0.975)


def daily_sharpe(daily: np.ndarray) -> float:
    """日 Sharpe = mean/std(ddof=1)。固定名义账本下与美元/收益率口径同值。"""
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd) if sd > 0 else float("nan")


def sortino_annual(daily: np.ndarray, target: float = 0.0,
                   periods_per_year: float = 365.0) -> float:
    """年化 Sortino（下行偏差，target=0）。右偏画像的并列尺子（铁律 C）。"""
    downside = np.minimum(daily - target, 0.0)
    dd = math.sqrt(np.mean(downside ** 2))
    return float((daily.mean() - target) / dd * math.sqrt(periods_per_year)) \
        if dd > 0 else float("inf")


# ── DSR 核心（Bailey & LdP 2014，DAILY 单位） ───────────────────────────────
def expected_max_sharpe(var_sr_daily: float, n_trials: float) -> float:
    """SR*₀ = √V·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]（日单位）。

    V = 全部被测配置日 Sharpe 的样本方差（ddof=1）——这就是为什么 trial-ledger
    必须落盘【全部】配置的 Sharpe，只存赢家算不出 V。
    """
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr_daily) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def psr(sr_daily: float, sr_star_daily: float, n_obs: int,
        skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio（kurt 非超额，正态=3；用实际矩，不假设正态）。"""
    denom = math.sqrt(1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * sr_daily ** 2)
    return float(stats.norm.cdf((sr_daily - sr_star_daily) * math.sqrt(n_obs - 1) / denom))


def effective_n_enb(daily_returns_matrix: np.ndarray) -> dict:
    """有效 N = ENB = (Σλ)²/Σλ²，λ 为 N 配置日收益 Pearson 相关矩阵特征值。

    输入 [T x N]（同一 common-live 窗口，缺日填 0）。高相关配置族 ENB << 名义 N
    （15 个趋势配置 ENB=2.35，PC1 独占 61% 方差）——主算用有效 N 防过度惩罚。
    """
    corr = np.corrcoef(daily_returns_matrix, rowvar=False)
    ev = np.linalg.eigvalsh(corr)[::-1]
    ev = np.clip(ev, 0, None)
    enb = float(ev.sum() ** 2 / (ev ** 2).sum())
    return {"enb": enb, "eigenvalues": [float(x) for x in ev]}


def dsr_report(sr_daily_target: float, skew: float, kurt_nonexcess: float,
               n_obs_days: int, var_sr_daily: float,
               n_calibres: dict[str, float],
               periods_per_year: float = 365.0) -> dict:
    """三 N 口径 DSR 全报（名义/有效/扩展下界——三者都报，主算 = 有效 N）。

    n_calibres 例：{"effective": 2.354, "nominal": 15, "extended_LB": 47}。
    返回每口径 {N, SR*₀(日/年), deflated(日/年), DSR 概率} + PSR(0) 参照。
    """
    ann = math.sqrt(periods_per_year)
    rows = {}
    for label, N in n_calibres.items():
        assert N > 1.0, f"SANITY: N must be > 1 (got {N} for {label})"
        sr0 = expected_max_sharpe(var_sr_daily, N)
        assert sr0 >= 0, f"SANITY FAIL: SR*0 < 0 for N={N}"
        defl = sr_daily_target - sr0
        assert defl <= sr_daily_target + 1e-12, "SANITY FAIL: deflated > observed"
        rows[label] = {"N": float(N),
                       "SR_star0_daily": sr0, "SR_star0_ann": sr0 * ann,
                       "deflated_daily": defl, "deflated_ann": defl * ann,
                       "DSR_prob": psr(sr_daily_target, sr0, n_obs_days,
                                       skew, kurt_nonexcess)}
    return {"rows": rows,
            "PSR_vs_zero": psr(sr_daily_target, 0.0, n_obs_days, skew, kurt_nonexcess),
            "observed_sr_ann": sr_daily_target * ann}


# ── 诚实 SE 与验证周期 ──────────────────────────────────────────────────────
def stationary_boot_sharpe_se(daily: np.ndarray, L: int, B: int = 10_000,
                              seed: int = 0, periods_per_year: float = 365.0) -> dict:
    """stationary bootstrap（Politis & Romano 1994）的年化 Sharpe SE；L=1 即 iid。

    L 取经济尺度（如 round(中位持仓小时/24)），先用 bootstrap-inference skill
    测 ACF 决定 block 是否必要（该测不该假设）。
    """
    rng = np.random.default_rng(seed)
    n = len(daily)
    p = 1.0 / L
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    new_block = rng.random((B, n)) < p
    new_starts = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        cont = ~new_block[:, t]
        idx[:, t] = np.where(cont, (idx[:, t - 1] + 1) % n, new_starts[:, t])
    samp = daily[idx]
    mu = samp.mean(axis=1)
    sd = samp.std(axis=1, ddof=1)
    ann = math.sqrt(periods_per_year)
    sr_ann = np.where(sd > 0, mu / sd * ann, np.nan)
    sr_ann = sr_ann[np.isfinite(sr_ann)]
    return {"L_days": L, "B": B, "se_sharpe_ann": float(sr_ann.std(ddof=1)),
            "ci95_sharpe_ann": [float(np.percentile(sr_ann, 2.5)),
                                float(np.percentile(sr_ann, 97.5))]}


def years_to_sig(sr_annual: float, se_annual: float, t0_years: float) -> float:
    """验证周期：t-stat∝√T ⇒ T* = T₀·(1.96/t₀)²，t₀=SR_ann/SE_ann。

    SR 用【打折后】的（铁律 B），SE 用对自相关诚实的 bootstrap SE——
    不用 iid 闭式（与 bootstrap 自相矛盾）、不用幸存者 Sharpe。effect≤0 → inf。
    """
    if sr_annual <= 0 or se_annual <= 0:
        return float("inf")
    return float(t0_years * (Z95 / (sr_annual / se_annual)) ** 2)


# ── FDR（Benjamini-Hochberg，多假设场景） ───────────────────────────────────
def fdr_bh(pvals: list[float], q: float = 0.05) -> dict:
    """BH 阶梯：p_(i) ≤ i/m·q 的最大 i 之前全过。返回逐假设判定 + 调整 p。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    passed_ranked = ranked <= thresh
    k = int(np.max(np.nonzero(passed_ranked)[0]) + 1) if passed_ranked.any() else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True
    p_adj_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    p_adj = np.empty(m)
    p_adj[order] = np.clip(p_adj_ranked, 0, 1)
    return {"reject": reject.tolist(), "p_adjusted": p_adj.tolist(), "n_rejected": k}


# ── 自检：复现 B2_4h 手验明细（README 计算明细段） ──────────────────────────
def _selftest() -> None:
    # 输入 = trend_methodology_hardening_20260622 冻结口径
    sr_d, skew, kurt, n = 0.03428, 0.191, 7.50, 1228
    var_sr_d = 1.313e-4
    rep = dsr_report(sr_d, skew, kurt, n,
                     var_sr_d, {"effective": 2.354, "nominal": 15.0, "extended_LB": 47.0})
    eff = rep["rows"]["effective"]
    assert abs(eff["SR_star0_ann"] - 0.1452) < 2e-3, eff
    assert abs(eff["deflated_ann"] - 0.510) < 5e-3, eff
    assert abs(eff["DSR_prob"] - 0.826) < 5e-3, eff
    nom = rep["rows"]["nominal"]
    assert abs(nom["deflated_ann"] - 0.267) < 5e-3, nom
    # 验证周期锚点：portfolio 0.5 + iid → 15.4y
    assert abs((Z95 / 0.5) ** 2 - 15.37) < 0.05
    # 恒等式：deflated ≤ observed（三口径）
    for r in rep["rows"].values():
        assert r["deflated_daily"] <= sr_d
    # ENB：两条完全相关 + 一条独立 → ENB ≈ 2 内外
    rng = np.random.default_rng(3)
    a = rng.standard_normal(600)
    m = np.column_stack([a, a * 1.0001 + rng.standard_normal(600) * 1e-6,
                         rng.standard_normal(600)])
    enb = effective_n_enb(m)["enb"]
    assert 1.5 < enb < 2.5, enb
    # FDR 基本行为
    f = fdr_bh([0.001, 0.011, 0.2, 0.9], q=0.05)
    assert f["reject"][0] and not f["reject"][3]
    print(f"deflated_sharpe selftest PASS  (SR*0_ann={eff['SR_star0_ann']:.4f}, "
          f"deflated_ann={eff['deflated_ann']:.4f}, DSR={eff['DSR_prob']:.3f}, ENB={enb:.2f})")


if __name__ == "__main__":
    _selftest()
