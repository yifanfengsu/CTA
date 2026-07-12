#!/usr/bin/env python3
"""bootstrap-inference skill — block/stationary bootstrap 误差棒工具（可独立 import）。

提炼自 scripts/research_deflated_sharpe.py（stationary bootstrap, Politis & Romano
1994）与 scripts/research_factor_scale.py（moving-block 索引），多档下界披露范式
提炼自 research/_closed/crypto_options/vrp_atm/reports/atm_vrp_stageB_premium_truth_20260628/（原文件未修改）。

纪律：
  ① 先 acf_check 再选 iid/block——该测不该假设（B2_4h 实测 ACF≈0 的反预设发现）。
  ② 块长按经济尺度预注册（主档），{短/主/长} 多档只作稳定性披露，不挑最优。
  ③ 下界类 gate：多档同号才算稳；bootstrap 抽不出比样本更坏的尾部（peso），
     须配前瞻情景注入（honest-verdict skill），二者互补非替代。

自检：python block_bootstrap.py
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "acf_check", "stationary_boot_indices", "stationary_bootstrap_stat",
    "multi_block_lower_bound",
]


def acf_check(series: np.ndarray, nlags: int = 5) -> dict:
    """样本自相关 lag 1..nlags + 简易读数。写进报告作"测了"的证据。

    读数口径（trend_methodology_hardening Q4）：全部 |ACF| 近 0（阈值默认
    2/√n 的白噪声带）⇒ iid 够用、block 修正 immaterial；否则必须 block。
    """
    x = np.asarray(series, dtype=float)
    x0 = x - x.mean()
    denom = float((x0 * x0).sum())
    acf = [float((x0[:-k] * x0[k:]).sum() / denom) if denom > 0 else 0.0
           for k in range(1, nlags + 1)]
    band = 2.0 / np.sqrt(len(x))
    return {"acf": acf, "white_noise_band": float(band),
            "iid_ok": bool(all(abs(a) <= band for a in acf))}


def stationary_boot_indices(n: int, L: int, B: int,
                            rng: np.random.Generator) -> np.ndarray:
    """stationary bootstrap 索引矩阵 (B, n)：几何分布块长（均值 L），环绕拼接。
    L=1 退化为 iid 重采样（参照档）。"""
    p = 1.0 / L
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    new_block = rng.random((B, n)) < p
    new_starts = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        cont = ~new_block[:, t]
        idx[:, t] = np.where(cont, (idx[:, t - 1] + 1) % n, new_starts[:, t])
    return idx


def stationary_bootstrap_stat(series: np.ndarray, stat_fn, L: int,
                              B: int = 10_000, seed: int = 0) -> dict:
    """任意统计量的 stationary bootstrap 分布：SE / CI95 / 分位。

    stat_fn: callable(np.ndarray 1D 重排样本) -> float（矢量化则传 axis 版更快，
    此处按事件规模保持通用）。L 按经济尺度传入；与 L=1 的 SE 比值 = inflation。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(series, dtype=float)
    idx = stationary_boot_indices(len(x), L, B, rng)
    vals = np.array([stat_fn(x[row]) for row in idx], dtype=float)
    vals = vals[np.isfinite(vals)]
    return {"L": L, "B": B, "mean": float(vals.mean()),
            "se": float(vals.std(ddof=1)),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "q05": float(np.percentile(vals, 5)), "q01": float(np.percentile(vals, 1)),
            "dist": vals}


def multi_block_lower_bound(series: np.ndarray, stat_fn, Ls: list[int],
                            q: float = 0.05, B: int = 10_000, seed: int = 0) -> dict:
    """多档块长的 q 分位下界披露 + 同号稳定性（vrp StageB 范式）。

    gate 用下界时：各档下界须同号才算"稳"；异号 = gate 不稳（预注册裁定）。
    注意"稳定"的答案也可能是稳定地区分不出零——稳 ≠ 过。
    """
    rows = {}
    for i, L in enumerate(Ls):
        r = stationary_bootstrap_stat(series, stat_fn, L, B=B, seed=seed + i)
        rows[L] = {"lower_q": float(np.percentile(r["dist"], q * 100)),
                   "mean": r["mean"], "se": r["se"]}
    signs = {np.sign(v["lower_q"]) for v in rows.values()}
    return {"q": q, "by_L": rows, "same_sign": len(signs) == 1,
            "all_lower_bounds": [v["lower_q"] for v in rows.values()]}


# ── 自检 ────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    rng = np.random.default_rng(11)
    # ① iid 序列：ACF 在白噪声带内；block/iid SE 比 ≈ 1
    iid = rng.standard_normal(1200)
    a = acf_check(iid)
    assert a["iid_ok"], a
    se_iid = stationary_bootstrap_stat(iid, np.mean, L=1, B=800, seed=1)["se"]
    se_blk = stationary_bootstrap_stat(iid, np.mean, L=8, B=800, seed=1)["se"]
    assert 0.8 < se_blk / se_iid < 1.25, (se_iid, se_blk)
    # ② AR(1) phi=0.6：ACF 报警；block SE 显著大于 iid SE（iid 假窄的经典失效）
    ar = np.zeros(1200)
    for t in range(1, 1200):
        ar[t] = 0.6 * ar[t - 1] + rng.standard_normal()
    a2 = acf_check(ar)
    assert not a2["iid_ok"] and a2["acf"][0] > 0.5, a2
    se_iid2 = stationary_bootstrap_stat(ar, np.mean, L=1, B=800, seed=2)["se"]
    se_blk2 = stationary_bootstrap_stat(ar, np.mean, L=10, B=800, seed=2)["se"]
    assert se_blk2 > 1.3 * se_iid2, (se_iid2, se_blk2)
    # ③ 多档下界：正均值序列的 5% 下界多档同号（此例应稳定为正）
    pos = rng.standard_normal(800) * 0.5 + 1.0
    mb = multi_block_lower_bound(pos, np.mean, Ls=[1, 4, 8], q=0.05, B=600, seed=3)
    assert mb["same_sign"] and all(lb > 0 for lb in mb["all_lower_bounds"]), mb
    print(f"block_bootstrap selftest PASS  (iid inflation={se_blk/se_iid:.2f}, "
          f"AR1 inflation={se_blk2/se_iid2:.2f}, multiL same_sign={mb['same_sign']})")


if __name__ == "__main__":
    _selftest()
