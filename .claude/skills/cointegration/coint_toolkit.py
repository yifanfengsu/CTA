#!/usr/bin/env python3
"""cointegration skill — 配对/价差协整工具（可独立 import）。

提炼自 research/_closed/crypto_perp/pairs_cointegration/scripts/research_pairs_cointegration.py（research/_closed/crypto_perp/pairs_cointegration/reports/pairs_cointegration_20260613/，
原文件未修改）。EG 检验经 statsmodels 惰性 import（statsmodels 0.14.6 已在 .venv），
其余纯 numpy。

关键口径（预注册范式）：
  - 价差 = log(A) − β·log(B) − α，β/α 由 formation 窗 OLS 估（非价格比值）。
  - EG 有方向性：coint(A,B) ≠ coint(B,A)，方向必须显式传入并在预注册写死。
  - 滚动 formation/trading 分离：trading 窗只用 formation 参数，不重估（防前视）。
  - 半衰期（AR(1)）必报；协整计数必过噪声底（noise-calibration skill）。

自检：python coint_toolkit.py
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ols_spread", "eg_pvalue", "half_life", "rolling_windows",
    "zscore_first_event", "baseline_reach_rate", "persistence_stats",
]


def ols_spread(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """formation 窗 OLS：log(A)=α+β·log(B)。返回 (α, β, 残差价差)。

    方向约定：A 是被解释腿——(A,B) 与 (B,A) 给出不同 β 与不同残差，
    必须与 eg_pvalue 的方向一致并在预注册写死。
    """
    X = np.column_stack([np.ones(len(log_b)), log_b])
    coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    return alpha, beta, log_a - beta * log_b - alpha


def eg_pvalue(log_a: np.ndarray, log_b: np.ndarray, trend: str = "c",
              maxlag: int = 1) -> float:
    """Engle-Granger p 值（MacKinnon），方向 = log_a 对 log_b 回归（显式，非对称）。

    冻结口径（pairs 研究）：trend='c', maxlag=1, autolag=None（确定性/速度）。
    statsmodels 惰性 import——不装 statsmodels 时其余函数仍可用。
    """
    from statsmodels.tsa.stattools import coint  # lazy
    return float(coint(log_a, log_b, trend=trend, maxlag=maxlag, autolag=None)[1])


def half_life(spread: np.ndarray) -> float:
    """AR(1) 半衰期（天/bar，与输入频率同单位）：Δs_t = a + b·s_{t−1}，
    HL = −ln2/ln(1+b)。b≥0（无回归）→ inf。

    用途：半衰期 > trading 窗长 ⇒ 窗内等不到回归，厚度统计无意义（必报）。
    """
    s = np.asarray(spread, dtype=float)
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([np.ones(len(lag)), lag])
    coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
    b = float(coef[1])
    if b >= 0 or (1.0 + b) <= 0:
        return float("inf")
    return float(-np.log(2.0) / np.log(1.0 + b))


def rolling_windows(n_obs: int, form: int = 90, trade: int = 30,
                    roll: int = 30) -> list[tuple[int, int, int, int]]:
    """滚动窗索引：[(f0,f1,t0,t1)]，formation=[f0,f1) / trading=[t0,t1)，前移 roll。

    防前视结构：β/μ/σ 与 EG 只在 formation 估；trading 只消费，不重估。
    """
    out = []
    for fe in range(form, n_obs - trade + 1, roll):
        out.append((fe - form, fe, fe, min(fe + trade, n_obs)))
    return out


def zscore_first_event(z: np.ndarray, spread: np.ndarray, z_entry: float = 2.0,
                       z_exit: float = 0.5, z_blow: float = 3.0) -> dict | None:
    """trading 窗内首次 |z|≥z_entry 事件，解析至回归/发散/未决（冻结口径）。

    outcome_gross = sign(z_trig)·(spread_trig − spread_exit)·100（%，正=收敛获利）。
    判定必须用【全事件加权】期望（C5），不只回归事件（C4 条件厚度陷阱）。
    """
    n = len(z)
    trig = next((j for j in range(n) if abs(z[j]) >= z_entry), None)
    if trig is None:
        return None
    sgn = float(np.sign(z[trig]))
    sp0 = float(spread[trig])
    for j in range(trig + 1, n):
        if abs(z[j]) < z_exit:
            return {"class": "revert", "outcome_gross_pct": sgn * (sp0 - float(spread[j])) * 100}
        if abs(z[j]) > z_blow:
            return {"class": "blowout", "outcome_gross_pct": sgn * (sp0 - float(spread[j])) * 100}
    return {"class": "unresolved", "outcome_gross_pct": sgn * (sp0 - float(spread[-1])) * 100}


def baseline_reach_rate(z: np.ndarray, z_exit: float = 0.5) -> tuple[int, int]:
    """无条件基线：窗内任一时点，其后剩余时间是否到达 |z|<z_exit。返回 (命中, 总数)。

    与事件回归率同窗同口径对照（distance-matched 的保守替身）——回归率必须
    显著高于该基线才算"偏离后更易回归"（pairs C3：17.9% vs 42.4%，反向）。
    """
    n = len(z)
    below = np.abs(z) < z_exit
    future = np.zeros(n, bool)
    seen = False
    for j in range(n - 1, -1, -1):
        future[j] = seen
        if below[j]:
            seen = True
    return int(future[:-1].sum()), n - 1


def persistence_stats(sig_sets: list[set]) -> dict:
    """逐窗可交易对集合 → 持续窗数分布 + churn 率（相邻窗 1−Jaccard）。

    加密教训：中位持续 1 窗 / churn 0.91 = 协整瞬时，不可前向交易（C2 生死门）。
    """
    runs: dict = {}
    completed = []
    prev: set = set()
    churn = []
    for cur in sig_sets:
        for p in list(runs):
            if p not in cur:
                completed.append(runs.pop(p))
        for p in cur:
            runs[p] = runs.get(p, 0) + 1
        if prev or cur:
            union = len(prev | cur)
            churn.append(1 - len(prev & cur) / union if union else 0.0)
        prev = cur
    completed.extend(runs.values())
    return {"median_persistence": float(np.median(completed)) if completed else 0.0,
            "mean_persistence": float(np.mean(completed)) if completed else 0.0,
            "n_runs": len(completed),
            "mean_churn": float(np.mean(churn)) if churn else float("nan")}


# ── 自检 ────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    rng = np.random.default_rng(7)
    n = 800
    # 合成协整对：x 随机游走，y = 0.7 + 1.5x + AR(1)(phi=0.9) 平稳噪声
    x = np.cumsum(rng.standard_normal(n) * 0.01) + 5.0
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + rng.standard_normal() * 0.01
    y = 0.7 + 1.5 * x + noise
    alpha, beta, sp = ols_spread(y, x)
    assert abs(beta - 1.5) < 0.05 and abs(alpha - 0.7) < 0.3, (alpha, beta)
    hl = half_life(sp)
    assert 3.0 < hl < 15.0, hl                     # 理论 ≈ −ln2/ln(0.9) = 6.6
    # 独立随机游走 → 无回归 → 半衰期大/inf 方向
    rw = np.cumsum(rng.standard_normal(n) * 0.01)
    assert half_life(rw) > hl
    # EG（statsmodels 可用时）：协整对 p 小、独立游走 p 大
    try:
        p_co = eg_pvalue(y, x)
        p_rw = eg_pvalue(rw + 5.0, x)
        assert p_co < 0.05 < p_rw, (p_co, p_rw)
        eg_msg = f"EG p(coint)={p_co:.4f} p(indep)={p_rw:.3f}"
    except ImportError:
        eg_msg = "EG skipped (statsmodels not installed)"
    # 滚动窗结构
    wins = rolling_windows(400, 90, 30, 30)
    assert wins[0] == (0, 90, 90, 120) and all(f1 == t0 for _, f1, t0, _ in wins)
    # 事件解析 + 基线
    z = np.array([0.1, 2.5, 2.8, 0.3])
    ev = zscore_first_event(z, z.copy())
    assert ev is not None and ev["class"] == "revert" and ev["outcome_gross_pct"] > 0
    hit, tot = baseline_reach_rate(z)
    assert tot == 3 and hit == 3
    # 持续性/churn
    ps = persistence_stats([{("a", "b")}, {("a", "b")}, {("c", "d")}])
    assert ps["median_persistence"] >= 1 and 0 <= ps["mean_churn"] <= 1
    print(f"coint_toolkit selftest PASS  (beta={beta:.3f}, HL={hl:.1f}, {eg_msg})")


if __name__ == "__main__":
    _selftest()
