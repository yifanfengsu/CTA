#!/usr/bin/env python3
"""ic-analysis skill — 横截面 IC / 分层 / 可交易价差工具（可独立 import）。

提炼自 scripts/research_factor_scale.py（reports/factor_scale_feasibility_20260628/，
原文件未修改）。

核心纪律（SKILL.md 详述）：
  IC ≠ 可交易 alpha —— 三道墙：① 分位单调性；② 成本后价差（换手×费率）；
  ③ 流动性分层（净正只在不可交易层 = 流动性伪装，预注册 FAIL 条款）。
  规模/强度看 IC 点估计（t 随 N 机械上升）；噪声标定用 noise-calibration skill。

自检：python ic_toolkit.py
"""
from __future__ import annotations

import numpy as np

__all__ = ["daily_ic", "quintile_monotonicity", "ls_spread", "tier_split"]


def _rank_rows(X: np.ndarray) -> np.ndarray:
    order = X.argsort(axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    T, K = X.shape
    ranks[np.arange(T)[:, None], order] = np.arange(K)
    return ranks.astype(float)


def _rowwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Am = A - A.mean(1, keepdims=True)
    Bm = B - B.mean(1, keepdims=True)
    num = (Am * Bm).sum(1)
    den = np.sqrt((Am * Am).sum(1) * (Bm * Bm).sum(1))
    out = np.full(A.shape[0], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def daily_ic(score: np.ndarray, fwd_ret: np.ndarray) -> dict:
    """逐日 Spearman IC（score、fwd_ret 均 [T x K] complete-case 面板）。

    信息纪律由调用方保证：score 只用前一收盘前信息（shift(1)，无前视）。
    返回 mean（点估计——判强度/规模用它，不用 t）/ std / t / 序列。
    """
    ic = _rowwise_corr(_rank_rows(score), _rank_rows(fwd_ret))
    ok = ic[np.isfinite(ic)]
    m = float(ok.mean())
    sd = float(ok.std(ddof=1)) if len(ok) > 1 else float("nan")
    return {"mean": m, "std": sd,
            "t": (m / sd * np.sqrt(len(ok))) if sd and sd > 0 else float("nan"),
            "n_days": int(len(ok)), "series": ic}


def quintile_monotonicity(score: np.ndarray, fwd_ret: np.ndarray,
                          n_q: int = 5) -> dict:
    """分位组平均收益 + 单调性（墙①）。非单调 ⇒ 排序信息不存在，
    long-short 价差只是两个尾组的偶然差。"""
    T, K = score.shape
    qret = np.zeros((T, n_q))
    for t in range(T):
        order = np.argsort(score[t], kind="stable")
        splits = np.array_split(order, n_q)
        for qi, grp in enumerate(splits):
            qret[t, qi] = fwd_ret[t, grp].mean()
    means = qret.mean(axis=0)
    diffs = np.diff(means)
    return {"quintile_means": means.tolist(),
            "monotone_up": bool((diffs >= 0).all()),
            "monotone_down": bool((diffs <= 0).all()),
            "top_minus_bottom": float(means[-1] - means[0])}


def ls_spread(score: np.ndarray, ret: np.ndarray, fee_per_side: float,
              periods_per_year: float = 365.0) -> dict:
    """quintile long-short 日调仓：毛/成本/净年化 + 日换手（墙②，逐字口径）。

    成本 = 日组合换手 × 单边费率。反转类信号换手 ~2×/日 → 年化成本几十个点，
    是结构墙不是细节（REV pool100：毛 +16%/yr、成本 40%/yr、净 −23%/yr）。
    """
    T, K = score.shape
    nq = max(1, K // 5)
    gross = np.empty(T)
    turn = np.empty(T)
    prevw = np.zeros(K)
    for t in range(T):
        order = np.argsort(score[t], kind="stable")
        short, long = order[:nq], order[-nq:]
        w = np.zeros(K)
        w[long] = 1.0 / nq
        w[short] = -1.0 / nq
        gross[t] = ret[t, long].mean() - ret[t, short].mean()
        turn[t] = np.abs(w - prevw).sum()
        prevw = w
    cost = turn * fee_per_side
    net = gross - cost
    return {"gross_ann_pct": float(gross.mean() * periods_per_year * 100),
            "cost_ann_pct": float(cost.mean() * periods_per_year * 100),
            "net_ann_pct": float(net.mean() * periods_per_year * 100),
            "avg_daily_turnover": float(turn.mean()),
            "net_positive": bool(net.mean() > 0)}


def tier_split(score: np.ndarray, ret: np.ndarray, liquidity_rank: np.ndarray,
               fees_by_tier: tuple[float, float, float] = (5e-4, 6e-4, 8e-4)) -> dict:
    """按流动性把宇宙切 top/mid/bottom 三层，各层报 IC + 净价差（墙③）。

    liquidity_rank: 长度 K，越小越流动（如按成交额排名）。
    判定读法：净正 alpha 只出现在 bottom（不可交易层）= 流动性伪装
    ——预注册 FAIL 条款（factor_scale 的判死主件）。
    fees_by_tier 对小币"故意乐观"（真实冲击成本更高），判死才稳。
    """
    K = score.shape[1]
    order = np.argsort(liquidity_rank)
    cuts = [K // 3, 2 * K // 3]
    tiers = {"top": order[:cuts[0]], "mid": order[cuts[0]:cuts[1]],
             "bottom": order[cuts[1]:]}
    out = {}
    for (name, idx), fee in zip(tiers.items(), fees_by_tier):
        s, r = score[:, idx], ret[:, idx]
        out[name] = {"ic_mean": daily_ic(s, r)["mean"],
                     "spread": ls_spread(s, r, fee)}
    top_net = out["top"]["spread"]["net_ann_pct"]
    bot_net = out["bottom"]["spread"]["net_ann_pct"]
    out["liquidity_disguise"] = bool(bot_net > 0 and top_net <= 0)
    return out


# ── 自检 ────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    rng = np.random.default_rng(21)
    T, K = 600, 30
    ret = rng.standard_normal((T, K)) * 0.02
    # 植入排序信号：score 与未来收益正相关
    score = 0.5 * ret + rng.standard_normal((T, K)) * 0.02
    ic = daily_ic(score, ret)
    assert ic["mean"] > 0.2, ic["mean"]
    mono = quintile_monotonicity(score, ret)
    assert mono["monotone_up"] and mono["top_minus_bottom"] > 0, mono
    # 零费率下净≈毛为正；高费率吃穿（成本墙行为）
    sp0 = ls_spread(score, ret, fee_per_side=0.0)
    spF = ls_spread(score, ret, fee_per_side=0.005)
    assert sp0["net_positive"] and spF["cost_ann_pct"] > 100, (sp0, spF)
    # 流动性伪装构造：信号只在 bottom 层有效
    liq = np.arange(K, dtype=float)               # 0 最流动
    score2 = rng.standard_normal((T, K)) * 0.02
    score2[:, K * 2 // 3:] = 0.8 * ret[:, K * 2 // 3:] + rng.standard_normal((T, K - K * 2 // 3)) * 0.005
    ts = tier_split(score2, ret, liq, fees_by_tier=(5e-4, 5e-4, 5e-4))
    assert ts["bottom"]["ic_mean"] > ts["top"]["ic_mean"], ts
    assert ts["liquidity_disguise"], {k: v for k, v in ts.items() if k != "liquidity_disguise"}
    print(f"ic_toolkit selftest PASS  (IC={ic['mean']:.3f}, mono_up={mono['monotone_up']}, "
          f"disguise_detected={ts['liquidity_disguise']})")


if __name__ == "__main__":
    _selftest()
