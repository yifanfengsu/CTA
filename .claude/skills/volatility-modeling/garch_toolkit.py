#!/usr/bin/env python3
"""garch_toolkit — GARCH 族波动率建模工具（volatility-modeling skill 配套代码）。

设计约定（违反即 SKILL.md 第 3 段失效模式）：
- 输入一律为**小数标度对数收益**（0.01 = 1%），内部 ×100 喂 arch 优化器后把
  ω 除以 100² 还原——调用方永远只见小数标度，杜绝 percent/decimal 混用。
- 一切波动率输出显式区分 variance / vol、per-period / annualized，
  年化因子必须显式传入（加密日频 = 365，不设默认值以外的魔法）。
- persistence = α+β 报点估计 + delta-method 标准误；半衰期 = ln(0.5)/ln(α+β)。
- 依赖：numpy / pandas / statsmodels（检验）+ arch（拟合，惰性 import；
  安装 `.venv/bin/pip install arch`，冒烟基准 arch 8.0.0）。

自检：`python garch_toolkit.py` 全部 PASS 才可用（模拟已知参数→回收，
数字自证，不引用任何外部示例数值）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# arch 喂给优化器的内部标度（percent）。对外接口永远是小数标度。
_ARCH_SCALE = 100.0


# ──────────────────────────────────────────────────────────────────────
# 模拟（自检与功效分析用）
# ──────────────────────────────────────────────────────────────────────

def simulate_garch(omega: float, alpha: float, beta: float, n: int,
                   dist: str = "normal", nu: float = 5.0,
                   seed: int | None = None, burn: int = 1000) -> np.ndarray:
    """按已知 (ω, α, β) 模拟 GARCH(1,1) 收益序列（小数标度，零均值）。

    dist='normal' 或 't'（Student-t，标准化到单位方差后代入）。
    burn 段丢弃以消除初值影响；初值 = 无条件方差 ω/(1−α−β)。
    """
    if alpha + beta >= 1.0:
        raise ValueError("alpha+beta >= 1：无平稳无条件方差，模拟器不支持 IGARCH")
    rng = np.random.default_rng(seed)
    total = n + burn
    if dist == "normal":
        z = rng.standard_normal(total)
    elif dist == "t":
        if nu <= 2:
            raise ValueError("Student-t 需 nu > 2 才有有限方差")
        z = rng.standard_t(nu, total) / np.sqrt(nu / (nu - 2.0))
    else:
        raise ValueError(f"未知 dist: {dist}")
    uncond = omega / (1.0 - alpha - beta)
    var = np.empty(total)
    r = np.empty(total)
    var[0] = uncond
    r[0] = np.sqrt(var[0]) * z[0]
    for t in range(1, total):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = np.sqrt(var[t]) * z[t]
    return r[burn:]


# ──────────────────────────────────────────────────────────────────────
# 前置检验：ARCH 效应（无 ARCH 效应则 GARCH 无意义）
# ──────────────────────────────────────────────────────────────────────

def arch_effect_test(returns: np.ndarray, lags: int = 10,
                     alpha_level: float = 0.05) -> dict:
    """Engle ARCH-LM + Ljung-Box(平方收益) 双检验。

    返回 dict：两检验的统计量与 p 值 + has_arch_effect（双检验都过
    alpha_level 才判有——保守合取，宁可漏报不虚报）。
    """
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    r = r - r.mean()  # 检验条件异方差，先去均值
    lm_stat, lm_p, _f_stat, _f_p = het_arch(r, nlags=lags)
    lb = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_p = float(lb["lb_pvalue"].iloc[0])
    return {
        "n": int(r.size),
        "lags": lags,
        "arch_lm_stat": float(lm_stat),
        "arch_lm_p": float(lm_p),
        "ljungbox_sq_stat": lb_stat,
        "ljungbox_sq_p": lb_p,
        "has_arch_effect": bool(lm_p < alpha_level and lb_p < alpha_level),
    }


# ──────────────────────────────────────────────────────────────────────
# 拟合
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GarchFit:
    """拟合结果（参数已还原到小数标度；arch 原始结果保留在 raw）。"""
    model: str
    dist: str
    mean: str
    n: int
    params: dict = field(default_factory=dict)      # 小数标度：omega 已 /100²
    std_err: dict = field(default_factory=dict)     # 同标度
    loglik: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    persistence_cov: float = float("nan")           # cov(α̂, β̂)，delta-method 用
    raw: object = None                              # arch 的 ARCHModelResult


def fit_garch(returns: np.ndarray, dist: str = "t", model: str = "GARCH",
              p: int = 1, q: int = 1, mean: str = "Constant",
              o: int = 0) -> GarchFit:
    """封装 arch 包拟合 GARCH 族模型；输入输出均为小数标度收益。

    dist：'t'（默认，加密肥尾）/'normal'/'skewt'。
    model：'GARCH'/'EGARCH'；GJR 用 model='GARCH', o=1。
    mean：'Constant'/'Zero'（μ≈0 是结论不是预设——用 Constant 拟合让数据自证）。
    注意：仅 GARCH(o=0) 的 ω/(1−α−β) 无条件方差换算在 persistence_report
    中实现；EGARCH/GJR 的 persistence 口径不同，报告函数会拒绝。
    """
    from arch import arch_model  # 惰性 import：仅拟合路径需要 arch

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    am = arch_model(r * _ARCH_SCALE, mean=mean, vol=model, p=p, o=o, q=q,
                    dist=dist, rescale=False)
    res = am.fit(disp="off", show_warning=False)

    scale2 = _ARCH_SCALE ** 2
    params, std_err = {}, {}
    for name, value in res.params.items():
        se = float(res.std_err[name])
        if name == "omega":
            params[name], std_err[name] = float(value) / scale2, se / scale2
        elif name == "mu":
            params[name] = float(value) / _ARCH_SCALE
            std_err[name] = se / _ARCH_SCALE
        else:  # alpha/beta/gamma/nu/lambda 等无量纲参数不受标度影响
            params[name], std_err[name] = float(value), se

    cov_ab = float("nan")
    try:
        pc = res.param_cov
        if "alpha[1]" in pc.index and "beta[1]" in pc.index:
            cov_ab = float(pc.loc["alpha[1]", "beta[1]"])
    except Exception:
        pass

    return GarchFit(
        model=model if o == 0 else f"GJR-{model}",
        dist=dist, mean=mean, n=int(r.size),
        params=params, std_err=std_err,
        loglik=float(res.loglikelihood),
        aic=float(res.aic), bic=float(res.bic),
        persistence_cov=cov_ab, raw=res,
    )


# ──────────────────────────────────────────────────────────────────────
# persistence / 半衰期 / 无条件方差（量纲显式）
# ──────────────────────────────────────────────────────────────────────

def half_life(persistence: float) -> float:
    """半衰期（单位 = 数据频率的 bar 数）：ln(0.5)/ln(α+β)。α+β≥1 时 inf。"""
    if not (0.0 < persistence < 1.0):
        return float("inf")
    return float(np.log(0.5) / np.log(persistence))


def persistence_report(fit: GarchFit, periods_per_year: float = 365.0) -> dict:
    """GARCH(1,1) 的 persistence 全套报告，三个量纲全部显式标注。

    - persistence = α+β，SE 由 delta method：√(var_α + var_β + 2cov_αβ)。
    - uncond_var_per_period：**方差**（小数²，per-period）= ω/(1−α−β)。
    - uncond_vol_per_period：**波动率**（小数，per-period）= √方差。
    - uncond_vol_annual：**年化波动率** = per-period vol × √periods_per_year
      （加密日频传 365；量纲混淆是本技能第一大错误源，故全部三层同时报）。
    """
    if not fit.model.startswith("GARCH") or "alpha[1]" not in fit.params:
        raise ValueError("persistence_report 只支持 GARCH(1,1)（o=0）口径")
    omega = fit.params["omega"]
    alpha = fit.params["alpha[1]"]
    beta = fit.params["beta[1]"]
    pers = alpha + beta
    var_a = fit.std_err["alpha[1]"] ** 2
    var_b = fit.std_err["beta[1]"] ** 2
    cov = fit.persistence_cov if np.isfinite(fit.persistence_cov) else 0.0
    pers_se = float(np.sqrt(max(var_a + var_b + 2 * cov, 0.0)))

    if pers < 1.0:
        uncond_var = omega / (1.0 - pers)
        uncond_vol = float(np.sqrt(uncond_var))
        ann_vol = uncond_vol * float(np.sqrt(periods_per_year))
    else:
        uncond_var = uncond_vol = ann_vol = float("inf")

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": float(pers),
        "persistence_se": pers_se,
        "half_life_bars": half_life(pers),
        "uncond_var_per_period": float(uncond_var),   # 方差，per-period
        "uncond_vol_per_period": float(uncond_vol),   # 波动率，per-period（小数）
        "uncond_vol_annual": float(ann_vol),          # 年化波动率（小数）
        "periods_per_year": float(periods_per_year),
    }


# ──────────────────────────────────────────────────────────────────────
# 预测
# ──────────────────────────────────────────────────────────────────────

def forecast_vol(fit: GarchFit, horizon: int,
                 periods_per_year: float = 365.0) -> pd.DataFrame:
    """多步条件波动率预测路径（arch 解析预测，参数还原到小数标度）。

    GARCH(1,1) 的路径按 σ²_{t+h} = σ²_∞ + (α+β)^{h−1}(σ²_{t+1} − σ²_∞)
    向无条件方差均值回归——这正是 EWMA/IGARCH（无回归项，路径走平）缺失的结构。
    返回 DataFrame：h / cond_var（小数²）/ cond_vol（小数）/ ann_vol（年化小数）。
    """
    fc = fit.raw.forecast(horizon=horizon, reindex=False)
    var_path = np.asarray(fc.variance.values[-1], dtype=float) / _ARCH_SCALE ** 2
    vol_path = np.sqrt(var_path)
    return pd.DataFrame({
        "h": np.arange(1, horizon + 1),
        "cond_var": var_path,
        "cond_vol": vol_path,
        "ann_vol": vol_path * np.sqrt(periods_per_year),
    })


# ──────────────────────────────────────────────────────────────────────
# 诊断
# ──────────────────────────────────────────────────────────────────────

def residual_diagnostics(fit: GarchFit, lags: int = 10) -> dict:
    """标准化残差检验套件：模型是否吃干净了条件异方差。

    合格标准（写进报告，不隐式）：std resid 的 Ljung-Box（水平）与
    Ljung-Box（平方）+ ARCH-LM 的 p 均 > 0.05 = 无剩余自相关/剩余 ARCH 效应。
    偏度/峰度用于核对 innovation 分布假设（t 分布拟合后 std resid
    峰度仍可 >3——那是 t 的本意，不是失败）。
    """
    from scipy import stats as sps
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

    res = fit.raw
    std_resid = np.asarray(res.resid / res.conditional_volatility, dtype=float)
    std_resid = std_resid[np.isfinite(std_resid)]
    lb_level = acorr_ljungbox(std_resid, lags=[lags], return_df=True)
    lb_sq = acorr_ljungbox(std_resid ** 2, lags=[lags], return_df=True)
    _lm_stat, lm_p, _f, _fp = het_arch(std_resid, nlags=lags)
    return {
        "n": int(std_resid.size),
        "lags": lags,
        "ljungbox_level_p": float(lb_level["lb_pvalue"].iloc[0]),
        "ljungbox_sq_p": float(lb_sq["lb_pvalue"].iloc[0]),
        "arch_lm_p": float(lm_p),
        "skew": float(sps.skew(std_resid)),
        "kurtosis": float(sps.kurtosis(std_resid, fisher=False)),  # 正态=3
        "clean": bool(float(lb_sq["lb_pvalue"].iloc[0]) > 0.05
                      and float(lm_p) > 0.05),
    }


# ──────────────────────────────────────────────────────────────────────
# EWMA 对照（RiskMetrics = IGARCH 特例）
# ──────────────────────────────────────────────────────────────────────

def ewma_sigma(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """EWMA 条件波动率（RiskMetrics）：σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1}。

    与 GARCH 的关系：这是 IGARCH(1,1) 的特例（ω=0, α=1−λ, β=λ, α+β≡1）
    ——没有 ω/(1−α−β) 无条件方差，多步预测**不**向长期水平回归（路径走平）。
    B2_4h-VT 用的正是这个族（hl=48bars ⇔ λ=0.5^{1/48}≈0.9857），其死因是
    削右尾而非 σ 估计（见 SKILL.md 锚点）。返回与输入对齐的 σ_t 序列
    （σ_t 只用 t−1 及以前的数据，因果）；初值 = 前 20 个 r² 的均值。
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("lam 须在 (0,1)")
    r = np.asarray(returns, dtype=float)
    n = r.size
    var = np.empty(n)
    var[0] = np.mean(r[: min(20, n)] ** 2)
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return np.sqrt(var)


# ──────────────────────────────────────────────────────────────────────
# 自检（python garch_toolkit.py）——模拟已知参数，数字自证
# ──────────────────────────────────────────────────────────────────────

def _selftest() -> int:
    results = []

    def check(name: str, ok: bool, detail: str = ""):
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    print("── garch_toolkit 冒烟自检（模拟已知参数 → 回收）──")

    # 真实参数（自检基准；一切期望值由公式现算，不引用外部示例数）
    OMEGA, ALPHA, BETA = 1e-5, 0.10, 0.85
    PERS = ALPHA + BETA                          # 0.95
    UV = OMEGA / (1.0 - PERS)                    # 无条件【方差】per-period
    UVOL = np.sqrt(UV)                           # 无条件【波动率】per-period
    ANNV = UVOL * np.sqrt(365.0)                 # 年化波动率（加密 365）
    HL = np.log(0.5) / np.log(PERS)              # 半衰期（bar）

    # 1. 量纲换算：方差 → 日 vol → 年化 vol 三层一致
    check("dim-1 无条件方差 = ω/(1−α−β)", abs(UV - 2.0e-4) < 1e-12,
          f"UV={UV:.6e}（方差，非波动率）")
    check("dim-2 日 vol = √UV", abs(UVOL - np.sqrt(2.0e-4)) < 1e-12,
          f"日vol={UVOL:.4%}")
    check("dim-3 年化 = 日 vol×√365", abs(ANNV - UVOL * 365 ** 0.5) < 1e-12,
          f"年化vol={ANNV:.2%}（√365 口径）")

    # 2. 模拟序列样本方差 ≈ 理论无条件方差
    r = simulate_garch(OMEGA, ALPHA, BETA, n=20000, seed=42)
    sv = float(np.var(r))
    check("sim-1 样本方差≈理论无条件方差(±10%)", abs(sv / UV - 1) < 0.10,
          f"sample={sv:.3e} vs theory={UV:.3e}")

    # 3. 双向报警：模拟 GARCH 报警、纯白噪声不报警
    t_garch = arch_effect_test(r)
    check("alarm-1 模拟GARCH → ARCH效应报警", t_garch["has_arch_effect"],
          f"LM p={t_garch['arch_lm_p']:.2e}, LB p={t_garch['ljungbox_sq_p']:.2e}")
    wn = np.random.default_rng(7).standard_normal(5000) * 0.01
    t_wn = arch_effect_test(wn)
    check("alarm-2 纯白噪声 → 不报警", not t_wn["has_arch_effect"],
          f"LM p={t_wn['arch_lm_p']:.3f}, LB p={t_wn['ljungbox_sq_p']:.3f}")

    # 4. 参数回收（normal innovation，同分布拟合）
    fit = fit_garch(r, dist="normal", mean="Zero")
    rep = persistence_report(fit, periods_per_year=365)
    check("recover-1 α 回收(±0.03)", abs(rep["alpha"] - ALPHA) < 0.03,
          f"α̂={rep['alpha']:.4f} (真值 {ALPHA})")
    check("recover-2 β 回收(±0.05)", abs(rep["beta"] - BETA) < 0.05,
          f"β̂={rep['beta']:.4f} (真值 {BETA})")
    check("recover-3 persistence 回收(±0.02)", abs(rep["persistence"] - PERS) < 0.02,
          f"α̂+β̂={rep['persistence']:.4f}±{rep['persistence_se']:.4f} (真值 {PERS})")
    check("recover-4 年化无条件vol 回收(±10%)",
          abs(rep["uncond_vol_annual"] / ANNV - 1) < 0.10,
          f"est={rep['uncond_vol_annual']:.2%} vs theory={ANNV:.2%}")

    # 5. 半衰期：理论式 + 高/低 persistence 两档回收
    check("hl-1 半衰期理论式", abs(half_life(PERS) - HL) < 1e-12,
          f"hl({PERS})={HL:.2f} bars")
    hl_est = rep["half_life_bars"]
    hl_lo = np.log(0.5) / np.log(PERS - 0.02)
    hl_hi = np.log(0.5) / np.log(PERS + 0.02)
    check("hl-2 高persistence 回收落带内", hl_lo <= hl_est <= hl_hi,
          f"ĥl={hl_est:.1f} ∈ [{hl_lo:.1f},{hl_hi:.1f}] (理论 {HL:.1f})")
    r_low = simulate_garch(1e-4, 0.25, 0.35, n=20000, seed=11)   # persistence 0.6
    rep_low = persistence_report(fit_garch(r_low, dist="normal", mean="Zero"))
    hl_low_theory = np.log(0.5) / np.log(0.60)
    check("hl-3 低persistence 回收(hl<4)", rep_low["half_life_bars"] < 4.0,
          f"ĥl={rep_low['half_life_bars']:.2f} (理论 {hl_low_theory:.2f})")

    # 6. 多步预测向无条件方差均值回归 + 解析递推一致
    fc = forecast_vol(fit, horizon=500)
    conv = float(fc["cond_var"].iloc[-1])
    check("fc-1 h=500 收敛到无条件方差(±3%)",
          abs(conv / rep["uncond_var_per_period"] - 1) < 0.03,
          f"var(h=500)={conv:.3e} vs UV̂={rep['uncond_var_per_period']:.3e}")
    o_, a_, b_ = rep["omega"], rep["alpha"], rep["beta"]
    v1 = float(fc["cond_var"].iloc[0])
    manual = [v1]
    for _ in range(9):
        manual.append(o_ + (a_ + b_) * manual[-1])
    max_diff = float(np.max(np.abs(np.array(manual) - fc["cond_var"].values[:10])))
    check("fc-2 arch预测 = 手推递推(前10步)", max_diff < 1e-12,
          f"max|diff|={max_diff:.1e}")

    # 7. t-innovation：t 分布拟合回收自由度
    r_t = simulate_garch(OMEGA, ALPHA, BETA, n=20000, dist="t", nu=5, seed=99)
    fit_t = fit_garch(r_t, dist="t", mean="Zero")
    nu_hat = fit_t.params.get("nu", float("nan"))
    check("t-1 Student-t 自由度回收(3.5<ν̂<7)", 3.5 < nu_hat < 7.0,
          f"ν̂={nu_hat:.2f} (真值 5)")
    diag = residual_diagnostics(fit_t)
    check("t-2 正确设定下 std resid 干净", diag["clean"],
          f"LB² p={diag['ljungbox_sq_p']:.3f}, ARCH-LM p={diag['arch_lm_p']:.3f}")

    # 8. EWMA 对照：多步预测走平（无均值回归）vs GARCH 回归
    lam = 0.94
    sig = ewma_sigma(r, lam=lam)
    check("ewma-1 σ 序列有限且尺度合理", np.all(np.isfinite(sig))
          and 0.3 < float(np.median(sig)) / UVOL < 3.0,
          f"median σ={float(np.median(sig)):.4%} vs 无条件 {UVOL:.4%}")
    # IGARCH 特例：ω=0, α=1−λ, β=λ ⇒ E[σ²_{t+h}] ≡ σ²_{t+1}，路径走平
    v_now = sig[-1] ** 2
    ewma_path = [v_now]
    for _ in range(9):
        ewma_path.append(0.0 + (1 - lam) * ewma_path[-1] + lam * ewma_path[-1])
    flat = float(np.ptp(ewma_path))
    garch_move = abs(float(fc["cond_var"].iloc[9]) - v1)
    check("ewma-2 EWMA 预测路径走平, GARCH 路径回归", flat < 1e-18 and garch_move > 0,
          f"EWMA ptp={flat:.1e} vs GARCH |Δvar(10步)|={garch_move:.2e}")

    n_pass = sum(ok for _, ok, _ in results)
    print(f"── 冒烟结果：{n_pass}/{len(results)} PASS ──")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(_selftest())
