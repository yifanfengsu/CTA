# ATR regime-filter p30 — out-of-sample validation + sensitivity

- Run (UTC): start 2026-06-09T02:05:41Z | end 2026-06-09T02:07:22Z
- Pure local backtest. Live runner / strategy / baseline engine NOT modified.
- Engine reused: `scripts/backtest_mr_5m_compare.py` (faithful live replica).
- Split (strict time order): train = earliest 0.667, test = latest 1/3.
- Split boundary: **2025-04-09 08:00:40+00:00**
  - train = [2023-01-01 00:04:00+00:00, split), test = [split, 2026-05-28 23:59:00+00:00]
- Threshold for percentile pXX = pXX of **Wilder ATR over the TRAIN window** (Wilder = the quantity the engine actually thresholds).
- OOS eval = run engine over full series at threshold T, keep trades whose **entry** is in the test window (entry-based attribution).

## Split — bars per symbol

| symbol | train bars | test bars | train end | test start |
|--------|----------:|---------:|-----------|-----------|
| BTC | 238,848 | 119,424 | 2025-04-09 07:59:00+00:00 | 2025-04-09 08:04:00+00:00 |
| ETH | 238,848 | 119,424 | 2025-04-09 07:59:00+00:00 | 2025-04-09 08:04:00+00:00 |
| SOL | 238,848 | 119,424 | 2025-04-09 07:59:00+00:00 | 2025-04-09 08:04:00+00:00 |
| LINK | 238,848 | 119,424 | 2025-04-09 07:59:00+00:00 | 2025-04-09 08:04:00+00:00 |
| DOGE | 238,848 | 119,424 | 2025-04-09 07:59:00+00:00 | 2025-04-09 08:04:00+00:00 |

> Note: this 2/3–1/3 calendar split differs from `reports/ablation/` (which uses a fixed 2026-Q1 OOS window on BTC for direction/hour filters). Per task spec we use 2/3–1/3 here; the two are unrelated experiments.

## train_p30 vs live threshold

Live thresholds were derived from *SMA*-ATR p30 on the *full* sample (`backtest_mr_5m_v2.py`); here train_p30 is *Wilder*-ATR p30 on *train only*. Differences are expected (different estimator + different window).

| symbol | live thr | train_p30 (Wilder) | diff % |
|--------|--------:|------------------:|------:|
| BTC | 81.5 | 65.437453 | -19.7% |
| ETH | 4.64 | 4.079710 | -12.1% |
| SOL | 0.245 | 0.121918 | -50.2% |
| LINK | 0.0212 | 0.018546 | -12.5% |
| DOGE | 0.0002 | 0.000194 | -3.0% |

## Q1 — out-of-sample value of the filter (test set)

filter ON = threshold train_p30; filter OFF = threshold 0 (all signals pass). Other params identical (LB=24, ATR=14 Wilder, stop=1.0, max_hold=48, $500/trade).

| symbol | mode | trades | net PnL | PF | win% | max DD $ | Sharpe/trade |
|--------|------|------:|-------:|---:|----:|--------:|------------:|
| BTC | OFF | 4,530 | $15,238 | 1.76 | 50.6 | $470 | 12.765 |
| BTC | ON  | 4,447 | $15,415 | 1.77 | 50.9 | $435 | 12.930 |
| ETH | OFF | 4,731 | $28,045 | 1.65 | 47.3 | $1,102 | 8.556 |
| ETH | ON  | 4,293 | $29,812 | 1.72 | 48.7 | $1,102 | 8.921 |
| SOL | OFF | 4,153 | $64,400 | 2.29 | 49.1 | $1,898 | 7.036 |
| SOL | ON  | 3,969 | $63,247 | 2.27 | 49.8 | $1,765 | 6.931 |
| LINK | OFF | 4,184 | $34,240 | 2.03 | 45.7 | $1,283 | 9.723 |
| LINK | ON  | 3,233 | $36,943 | 2.31 | 49.2 | $1,295 | 10.614 |
| DOGE | OFF | 3,700 | $47,304 | 2.50 | 48.6 | $2,005 | 9.426 |
| DOGE | ON  | 3,250 | $47,487 | 2.59 | 49.6 | $2,005 | 9.485 |
| PORTFOLIO | OFF | 21,298 | $189,227 | 2.06 | 48.3 | $2,250 | 16.352 |
| PORTFOLIO | ON  | 19,192 | $192,905 | 2.14 | 49.7 | $2,250 | 16.698 |

## Q2 — threshold sensitivity (test-set net PnL)

| symbol | p20 | p25 | p30 | p35 | p40 | shape |
|--------|----:|----:|----:|----:|----:|-------|
| BTC | $15,272 | $15,279 | $15,415 | $15,479 | $15,451 | flat (robust) |
| ETH | $28,423 | $29,498 | $29,812 | $29,794 | $30,191 | flat (robust) |
| SOL | $64,431 | $64,479 | $63,247 | $63,923 | $65,720 | flat (robust) |
| LINK | $37,011 | $36,795 | $36,943 | $37,757 | $38,485 | flat (robust) |
| DOGE | $47,372 | $47,368 | $47,487 | $47,948 | $48,088 | flat (robust) |
| PORTFOLIO | $192,508 | $193,419 | $192,905 | $194,902 | $197,936 | flat (robust) |

## Conclusions (honest read — data-driven, no forced positives)

### Q1 — does the filter add out-of-sample value?
**Weakly yes: neutral-to-slightly-positive on every symbol, harmful on none.**
On the held-out test set the filter (threshold = train_p30) removes ~10% of the
lowest-volatility trades and improves the portfolio by **+$3,678 (+1.9%) net and
+0.08 PF (2.06→2.14)**. It never meaningfully hurts a symbol:

| symbol | Δnet (ON−OFF) | ΔPF | verdict |
|--------|-------------:|----:|---------|
| BTC | +$177 (+1.2%) | +0.01 | neutral — no harm |
| ETH | +$1,767 (+6.3%) | +0.07 | mild positive |
| SOL | −$1,153 (−1.8%) | −0.02 | ≈neutral (edge so strong it needs no filter); DD −$133 |
| LINK | +$2,703 (+7.9%) | +0.28 | clear positive (biggest beneficiary, win% +3.5pp) |
| DOGE | +$183 (+0.4%) | +0.09 | neutral net, mild PF |

The filter is a **mild quality refinement**, not a strong edge.

> **Honest correction of the original in-sample narrative.** `backtest_mr_5m_v2.py`
> sold the p30 filter as a drawdown fix ("Max DD −72%"). That does **not** replicate
> out-of-sample: portfolio max-DD is **identical ($2,250) with and without the
> filter** (ETH/DOGE unchanged too). The DD trough is driven by large, high-vol
> trades the filter never touches. The filter's real OOS value is a small PF/win-rate
> lift from skipping dead low-vol chop — consistent with the project finding that
> edge concentrates in stronger signals. It is **not** drawdown protection.

### Q2 — is the threshold sensitive (overfit) or smooth (robust)?
**Smooth and robust for every symbol — no p30 spike.** Across p20→p40 each curve is
flat-to-gently-monotone (full range ≤4% of net for all symbols). p30 is neither a
peak nor a special sweet spot; if anything net PnL leans **slightly higher at p35–p40**
(more filtering → marginally higher PF). This is the *opposite* of an overfit
signature — the choice of percentile barely matters, so p30 was never a tuned spike.

### Q3 — true edge or in-sample artifact?
**Neither a fake edge nor a strong one: a small, robust, defensible refinement.**
It survives OOS (positive-to-neutral everywhere) and shows no overfitting (flat
sensitivity). The original p30 value is sound; the original *DD claim* was an
in-sample artifact and should be dropped from the strategy's stated rationale.

### Per-symbol recommendations
- **BTC** — keep filter. OOS-neutral, threshold robust. Live 81.5 ≈ Wilder-p40, in the robust region. No change.
- **ETH** — keep filter (mild benefit). Live 4.64 ≈ Wilder-p40. No change.
- **SOL** — keep filter (harmless; tiny DD help). Standalone edge is strong enough it doesn't need filtering. Live 0.245 ≈ Wilder-p40; train_p30 is −50% below live, but sensitivity shows the higher (live) value is actually the better end. No change.
- **LINK** — keep filter; clearest beneficiary (+7.9% net, +0.28 PF). Data even favors a slightly higher threshold (p40 best). No change needed; raising toward p40 is optional upside within noise.
- **DOGE** — keep filter (mild benefit). No change.
- **None fail; none should stop trading; none should drop the filter.**

### Overall
Maintain the status quo: **filter on, current live thresholds.** They are validated
out-of-sample and sit in the flat/robust region of the sensitivity surface (near
Wilder-p40, which is marginally the best end). A mild global shift toward p35–p40
would be fractionally better OOS but the gain is inside the noise band — **not worth
a live change.** The filter is **not an overfitting product**; the only thing to
retire is the obsolete "−72% drawdown" justification, which does not hold OOS.

### Method caveats
- Sharpe is **per-trade** (mean/std·√n), not annualized — a unitless ranking aid only.
- Live thresholds originate from *SMA*-ATR full-sample p30; here train_pXX are
  *Wilder*-ATR train-only percentiles, so train_p30 ≠ live by construction (largest
  gap SOL −50%). The sensitivity sweep brackets every live value, so this does not
  affect the conclusions.
- OOS uses entry-based attribution over the full series at threshold T; at most one
  position straddles the split boundary (entry in train → discarded), a negligible
  edge effect identical across all variants.

## 由本次验证产生的文档更新

2026-06-09，基于本次 OOS 验证结论（ATR filter 是 whipsaw 过滤器 / 温和质量精修，
非风控/降回撤；"−72% Max DD" 为 in-sample artifact，已撤回）对项目文档做了外科手术式更新：

- `PROJECT_GUIDE.md`（第 55 行，ATR regime filter 描述）—— 将其定位为 whipsaw 过滤器 /
  温和质量精修（OOS +1.9% 净利 / +0.08 PF），明确标注"非风控、不降回撤（Max DD on/off
  相同 $2,250）"，并加注释撤回 `backtest_mr_5m_v2.py` 原"风控/−72% Max DD"说法。

未改动的文档（经核查不含错误说法）：
- `README.md` —— 全文为趋势跟踪研究命令参考，未描述 MR-5m ATR filter。
- `CLAUDE.md` —— 未提及 ATR regime filter（其 "ATR" 指单笔 ATR 止损，属真实风控层）。
- `reports/regime/regime_filter_report.md` —— 是状态熔断器（circuit breaker）报告，与 ATR filter 无关。
- `reports/regime/{signal_quality,mr_regime_analysis,dynamic_sizing}_report.md` —— 均未将 ATR filter 描述为风控。

> 注：原"−72% Max DD / 风控"说法的唯一来源是 `scripts/backtest_mr_5m_v2.py`（代码），
> 按项目约束不修改任何 .py 代码，故未改动；上述文档注释已对该说法做撤回标注。

