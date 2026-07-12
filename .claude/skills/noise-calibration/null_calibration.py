#!/usr/bin/env python3
"""noise-calibration skill — 假阳性噪声基线工具（可独立 import）。

提炼自 scripts/research_factor_scale.py 的 NULL-A/NULL-B 噪声标定实现
（research/_closed/crypto_tick/factor_scale/reports/factor_scale_feasibility_20260628/，原文件未修改）。
教训出处：pairs_cointegration 漏装噪声标定（231 对 × p<0.05 = 11.55 对/窗
期望假阳性，"12.48 对 PASS"仅高出噪声底 8%）。

核心原则：real 指标必须超【同一管线喂噪声】的基线（p95，单侧 p<0.05），
超 0 不算数。null 用 block bootstrap 保留边际分布 + 短程串行结构
（朴素逐点 shuffle 会把基线做窄——见 SKILL.md 失效模式）。

用法：
    from null_calibration import null_ic_panel, placebo_expected_count
    res = null_ic_panel(score, ret, n_shuffle=200, block=5, seed=42)
    # res["real"]["mean"], res["nullA"]["p95"], res["nullA"]["p_one_sided"], ...

自检：python null_calibration.py  （合成数据双向验证：纯噪声不报警、植入信号报警）
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rank_rows", "rowwise_corr", "block_idx_1d", "block_idx_2d",
    "ic_stats", "null_ic_panel", "null_scalar_stat", "placebo_expected_count",
]


# ── 基础件（提炼自 research_factor_scale.py，逐字口径） ─────────────────────
def rank_rows(X: np.ndarray) -> np.ndarray:
    """按行（axis=1）序数秩（连续数据 ties 可忽略）。"""
    order = X.argsort(axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    T, K = X.shape
    ranks[np.arange(T)[:, None], order] = np.arange(K)
    return ranks.astype(float)


def rowwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """逐行 Pearson 相关（对秩矩阵用 = 逐日 Spearman IC）。"""
    Am = A - A.mean(1, keepdims=True)
    Bm = B - B.mean(1, keepdims=True)
    num = (Am * Bm).sum(1)
    den = np.sqrt((Am * Am).sum(1) * (Bm * Bm).sum(1))
    out = np.full(A.shape[0], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def block_idx_1d(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """moving-block bootstrap 索引（长度 n，共享一条重排）。"""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    offs = np.arange(block)
    return ((starts[:, None] + offs[None, :]) % n).ravel()[:n]


def block_idx_2d(n: int, k: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """k 条相互独立的 moving-block bootstrap 索引列，shape (n, k)。"""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(k, nb))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]) % n
    return idx.reshape(k, nb * block)[:, :n].T


def ic_stats(ic_series: np.ndarray, periods_per_year: float = 365.0) -> dict:
    ic = ic_series[np.isfinite(ic_series)]
    n = len(ic)
    m = float(ic.mean())
    sd = float(ic.std(ddof=1)) if n > 1 else float("nan")
    ir = m / sd if sd and sd > 0 else float("nan")
    return {"mean": m, "std": sd, "ir": ir,
            "ir_ann": ir * np.sqrt(periods_per_year) if np.isfinite(ir) else float("nan"),
            "t": (m / sd * np.sqrt(n)) if sd and sd > 0 else float("nan"),
            "pos_share": float((ic > 0).mean()) if n else float("nan"), "n_days": n}


def _pack(null: np.ndarray, real_mean: float) -> dict:
    return {"null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)),
            "p95": float(np.percentile(null, 95)), "p99": float(np.percentile(null, 99)),
            "p_one_sided": float((null >= real_mean).mean()),
            "real_percentile": float((null < real_mean).mean() * 100.0),
            "exceeds_p95": bool(real_mean > np.percentile(null, 95))}


# ── 主入口 1：面板 IC 的双 null 标定（NULL-A 主 / NULL-B 副） ───────────────
def null_ic_panel(score: np.ndarray, ret: np.ndarray, n_shuffle: int = 200,
                  block: int = 5, seed: int = 42) -> dict:
    """score, ret: complete-case [T x K] 面板。返回 real IC 统计 + 双 null 分布。

    NULL-A（主）：逐列独立 block bootstrap 打乱收益 → 破坏 score→ret 对齐，
      保留每列边际 + 短程串行结构。
    NULL-B（副）：整行共享 block bootstrap → 额外保留横截面同期相关。
    判定读法：real mean-IC > nullA.p95 且 p_one_sided < 0.05 才算超噪声。
    """
    rng = np.random.default_rng(seed)
    S_rank = rank_rows(score)
    real_ic = rowwise_corr(S_rank, rank_rows(ret))
    real = ic_stats(real_ic)
    T, K = ret.shape
    nullA, nullB = np.empty(n_shuffle), np.empty(n_shuffle)
    for b in range(n_shuffle):
        idxA = block_idx_2d(T, K, block, rng)
        Rb = np.take_along_axis(ret, idxA, axis=0)
        nullA[b] = np.nanmean(rowwise_corr(S_rank, rank_rows(Rb)))
        nullB[b] = np.nanmean(rowwise_corr(S_rank, rank_rows(ret[block_idx_1d(T, block, rng)])))
    return {"real": real, "real_ic_series": real_ic,
            "nullA": _pack(nullA, real["mean"]), "nullB": _pack(nullB, real["mean"]),
            "nullA_dist": nullA, "nullB_dist": nullB}


# ── 主入口 2：任意标量统计量的 block-shuffle null ───────────────────────────
def null_scalar_stat(series: np.ndarray, stat_fn, n_shuffle: int = 200,
                     block: int = 5, seed: int = 42) -> dict:
    """对单条序列的标量统计量建 null：block bootstrap 重排后重算 stat_fn。

    stat_fn: callable(np.ndarray) -> float，与 real 用【完全同一】函数
    （管线一致性是标定有效的前提——见 SKILL.md 失效模式）。
    事件研究的安慰剂请优先用"随机时点跑同一测量管线"（order_flow 范式），
    本函数适用于全局统计量（如 ACF、事件计数率）。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(series, dtype=float)
    real = float(stat_fn(x))
    null = np.empty(n_shuffle)
    for b in range(n_shuffle):
        null[b] = float(stat_fn(x[block_idx_1d(len(x), block, rng)]))
    return {"real": real, **_pack(null, real), "null_dist": null}


# ── 主入口 3：多重筛查的期望假阳性底（pairs 教训的算术） ────────────────────
def placebo_expected_count(n_tests: int, p_threshold: float = 0.05) -> float:
    """N 次独立检验在阈值 p 下的期望假阳性数 = N*p。

    报告"检测到 X 个"之前先算这个底：X 必须显著超过它才有意义。
    实例：231 对 × 0.05 = 11.55 → pairs 的 12.48 对/窗仅高 8%（≈噪声）。
    """
    return float(n_tests) * float(p_threshold)


# ── 自检：合成数据双向验证（factor_scale 的 synthetic validation 范式） ─────
def _selftest() -> None:
    rng = np.random.default_rng(0)
    T, K = 800, 30
    ret = rng.standard_normal((T, K)) * 0.02
    # ① 纯噪声 score → 不得报警
    score_noise = rng.standard_normal((T, K))
    r0 = null_ic_panel(score_noise, ret, n_shuffle=200, block=5, seed=1)
    assert not r0["nullA"]["exceeds_p95"] or r0["nullA"]["p_one_sided"] > 0.01, \
        f"pure noise falsely flagged: {r0['nullA']}"
    # ② 植入信号 score = 0.3*未来收益 + 噪声 → 必须报警
    score_sig = 0.3 * ret + rng.standard_normal((T, K)) * 0.02
    r1 = null_ic_panel(score_sig, ret, n_shuffle=200, block=5, seed=1)
    assert r1["nullA"]["exceeds_p95"] and r1["nullA"]["p_one_sided"] < 0.05, \
        f"planted signal missed: {r1['nullA']}"
    # ③ 标量 null + 期望假阳性底
    s = rng.standard_normal(500)
    r2 = null_scalar_stat(s, lambda x: float(np.mean(x)), n_shuffle=200, seed=2)
    assert abs(r2["real"]) < 3 * (r2["null_std"] + 1e-12) + 1e-6
    assert abs(placebo_expected_count(231, 0.05) - 11.55) < 1e-9
    print("null_calibration selftest PASS  "
          f"(noise p={r0['nullA']['p_one_sided']:.3f}, signal p={r1['nullA']['p_one_sided']:.3f}, "
          f"pairs floor={placebo_expected_count(231):.2f})")


if __name__ == "__main__":
    _selftest()
