# Large-Scale Factor Pre-Gate — Stage A: *does cross-sectional alpha improve with universe scale?*

**Run:** 2026-06-28 · `scripts/research_factor_scale.py` · data `data/binance_vision/` (1d klines + funding, READ-ONLY)
**Status: 不过 / FAIL — the large-scale factor route is NOT worth the big-engineering + institutional-data spend on free ~100-coin data.** (The narrow pre-registered IC-scale gate *technically* PASSED one factor — short-term reversal — but the *required* tradeability + survivor-bias evidence shows that pass is a textbook **IC ≠ tradeable-alpha / liquidity-disguise** artifact. Full transparency below; nothing hidden, no gate redefined.)

---

## POSITIONING & DISCIPLINE (frozen — do not edit)

> This task is the **zero-cost front gate** of the "large-scale factor" big-engineering route. The core question is **NOT** "can we find a factor with significant IC" — with hundreds of coins × many factors under multiple testing you are *guaranteed* to find "significant" factors, and that is exactly the trap (cf. MR cointegration: 231 "cointegrated" pairs ≈ pure noise, 11.55 false-positive pairs/window). The core question is: **taking the already-tested factors from 22 coins to ~100 coins, does alpha get stronger, stay the same, or get weaker?** Stronger = scale itself brings edge → worth the big spend; same = scale contributes nothing (same weak alpha, smaller SE); weaker / indistinguishable-from-noise = scale only adds more false positives.
>
> Two lessons wired in: ① cross-sectional study (22-coin MOM/CAR/REV all FAILED: alpha too weak + decayed + eaten by cost; IC ≠ tradeable alpha); ② MR cointegration (MISSED noise calibration → a possible multiple-testing false positive read as a mechanism — **this task installs noise calibration at step one**).
>
> Judgement philosophy: **noise calibration before IC interpretation** (a real IC must beat the shuffled-returns false-positive baseline, not beat 0); **factor set pre-registered & fixed** (no p-hacking in factor space); **no Sharpe as primary verdict** (look at IC / tradeable spread / noise baseline / tier source).

**Backtest / data assumptions (header per project rule):** free public **Binance Vision UM-perp 1d klines + fundingRate** (production CDN, no demo variant; sha256-verified, 9400/9400 ok, 15 not-available-404). Cross-validated vs OKX `database_mainnet.db` (read-only): **median |close dev| 0.030%, gate <0.5% PASS** (OKX naive-Shanghai → UTC −8h confirmed; reproduces the 0.03% from `data_trust_closure_20260611`). Tradeable cost = daily-turnover × per-side taker fee (6 bps pool / 5 bps top / 8 bps bottom — **deliberately optimistic for small-caps**; rough Part-4c proxy, precise cost later). Contaminated DB / vrp line / forward system / VPS **never touched**.

### Pre-registered universe (frozen before results)
Free daily klines, top by 2026-05 USDT quote volume, **nested pools 22 ⊂ 50 ⊂ 100 by volume rank**, **balanced panel** = top-K *among coins with full history over the window*. Window chosen deterministically = **longest window where ≥100 full-history coins exist = 2022-07 → 2026-05 (~3.9 yr, 1431 days)** (availability-driven, not result-driven). **SURVIVORSHIP (frozen):** current-listed only = survivor bias, alpha **systematically optimistic**; dead coins (LUNA/FTT/…) absent. Direction known: optimistic-alpha judged **DEAD → conclusion more robust** (truth worse); judged **ALIVE → flag "needs delisted-inclusive institutional data; optimistic upper bound."** Point-in-time infeasible on free data; stated, not hidden.

### Pre-registered factors (frozen; zero variants — the core anti-p-hack defence)
Only the three already-tested cross-sectional factors, caliber copied verbatim from `scripts/research_cross_sectional_ic.py` (params NOT re-searched, none added): **F-MOM** = +ret30, **F-CAR** = −funding7 (most-negative funding = long), **F-REV** = −ret3. IC = Spearman(long-pref score, fwd-1d return); daily rebalance UTC 00:00, info through prior close.

### Pre-registered gates (frozen; iron-rule A)
- **G1 beats noise:** real mean-IC > pool-100 NULL-A 95th pct **and** one-sided perm p<0.05 (beats the shuffle baseline, not 0).
- **G2 not bottom-disguise:** top-or-mid tier IC ≥ 0.5× bottom-tier IC **and** that top/mid IC beats its tier null p95.
- **G3 scale improves:** IC(100) − IC(22) > one null SE **and** IC(50) not below IC(22) by > one null SE (upward, not U-down).
- **PASS = all three for ≥1 factor.** **FAIL = any of:** IC indistinguishable from noise / **alpha mainly from the untradeable small-cap tier (liquidity disguise)** / pool-expansion doesn't improve IC.

---

## TL;DR

| | Result |
|---|---|
| **Core question (does alpha improve with scale?)** | **Only short-term reversal's *raw IC* improves with scale** (REV +0.030→+0.043→+0.056, 22→50→100). Momentum is reversal-signed and gets *more* negative (−0.022→−0.049). Carry **washes out** with breadth (+0.019→−0.005). |
| **Noise calibration** | REV beats the block-shuffle null at every K (real IC at 100th pctile, p=0.000, both nulls). MOM is significantly *negative* (reversal). CAR beats noise only at K=22, indistinguishable from noise by K=50/100. |
| **Where does REV's "scale edge" live?** | In the **least-liquid coins** (tier IC top +0.038 < mid +0.055 < bottom +0.073) — and crucially the only **positive *net* spread** is the illiquid bottom tier; the liquid top tier is **gross-negative** (−62%/yr). |
| **Tradeability (IC ≠ alpha)** | REV pool-100 **net −23%/yr** at a conservative 6 bps (turnover 1.8×/day → 40%/yr cost wall). Net positive **only** in bottom tier (+22%/yr at optimistic 8 bps that real small-cap impact would erase). |
| **Pre-registered coded gate** | **REV PASS** (G1∧G2∧G3, IC-based), MOM/CAR FAIL → `results.json.final = "PASS"`. **Reported unaltered.** |
| **Integrated 判生死 (Part 5, weighing 可交易价差 + 分层来源 + 幸存者)** | **不过 / FAIL.** REV triggers the pre-registered FAIL condition *"alpha mainly from the untradeable small-cap tier (liquidity disguise)"* on a **tradeable** basis (net-positive only in illiquid bottom; gross-negative in liquid top), is the **most survivor-bias-flattered** factor, and its scale gain is in **raw IC only, not tradeable alpha**. |
| **Spend saved** | Big-engineering route **not green-lit**; the only conceivable spend (≈$1–6k Tardis survivor-free + realistic-cost recheck of reversal) is predicted by this same run to confirm death → low priority. |

---

## Q1 — Free data availability + institutional data cost

**Free data (obtained):** Binance Vision lists **729 ASCII USDT UM-perps** (3 CJK joke tickers dropped); **588 have a 2026-05 1d file**. Downloaded **1d klines + fundingRate, 100 coins × 47 months (2022-07→2026-05), 9400/9400 verified, funding 100/100**. Coverage ≥2-3 yr satisfied (3.9 yr). **Survivor-biased** (current-listed only). Manifest: `data/binance_vision/factor_scale_manifest.json` (tracked); zips gitignored.

**Full-history frontier (why the design is forced):** the *current* top-100-by-volume coins are mostly **young listings** — even at a 2025-01 start only ~55/100 have full history. #full-history coins by window start: 2022-01→97, **2022-07→105**, 2023-01→116, 2024-01→191. To reach 100 full-history coins the window must start ≤2022-07; reaching 100 dips to volume-rank ~551 (so the bottom tier is genuinely illiquid — making the disguise test meaningful). **This "the liquid universe is young" fact is itself a finding** about free-data large-scale factor work.

**Institutional data cost (reconnaissance only, nothing bought — full table in `institutional_data_cost_survey.md`):**
- **Lower bound (raw survivor-free, build PIT yourself): ~$1,000–6,000 one-time** via **Tardis.dev** (explicitly survivorship-bias-free; min order $300; invoicing >$6,000).
- **Institutional (curated + point-in-time index constituents + reference rates): ~$10,000–55,000/yr** via **Kaiko** (Vendr avg ~$28.5k/yr, range $9.5k–55k) or **Coin Metrics** (quote-only). CoinAPI flat-files usage-based (~$170–465/mo small scope); Amberdata self-serve + enterprise.
- **The data threshold to remove survivor bias is therefore ~$1–6k to start, ~$10–55k/yr to do institutionally.** Justified only if the free gate first shows scale-improving *tradeable* alpha. It does not (below) → spend saved.

## Q2 — Noise calibration: real IC vs false-positive baseline

Primary null **NULL-A** = per-coin moving-block bootstrap (block 5d) of the forward-return series (200×), real scores fixed → destroys score→return cross-sectional alignment, preserves each coin's marginal + serial structure. Secondary **NULL-B** = shared-row block bootstrap (preserves cross-coin correlation). Validated on synthetic data: pure noise → real IC stays inside the null (p≈0.08); planted signal → p=0.000. *(Fig `noise_calibration_pool100.png`.)*

| factor (pool-100) | real mean-IC | NULL-A p95 | NULL-A p (1-sided) | NULL-B p | verdict vs noise |
|---|---|---|---|---|---|
| F-MOM | **−0.0489** | +0.0073 | 1.000 | 1.000 | significantly **negative** (reversal); not positive-signal |
| F-REV | **+0.0556** | +0.0020 | **0.000** | **0.000** | **beats noise** (100th pctile, both nulls) |
| F-CAR | −0.0050 | +0.0038 | 0.880 | 1.000 | **indistinguishable from noise** |

Beating 0 is not enough and not the test: only **REV** beats the shuffle baseline (strongly). CAR beats it at K=22 only.

## Q3 — Scale gradient (the core question): IC **point estimate** trend (not t — t rises mechanically with N)

| factor | IC @ K=22 | IC @ K=50 | IC @ K=100 | null SE | trend |
|---|---|---|---|---|---|
| F-MOM | −0.0223 | −0.0283 | −0.0489 | 0.0027 | more negative (reversal strengthens with scale) |
| **F-REV** | **+0.0305** | **+0.0425** | **+0.0556** | 0.0024 | **rises with scale** (Δ = +0.025 = ~10 null-SE) |
| F-CAR | +0.0185 | −0.0001 | −0.0050 | 0.0032 | **washes out** (carry is a top-coin effect) |

*(Fig `scale_gradient_ic.png`.)* So scale genuinely **adds raw cross-sectional information for the reversal/overreaction signal** (REV up, MOM more-negative = same direction), and **subtracts** it for carry. The point estimate (not just t) moves — this is real breadth, not just smaller SE. **But Q4 shows the rise is concentrated in illiquid coins and is not tradeable.**

## Q4 — Liquidity stratification (anti-disguise) + tradeability

Pool-100 split by 2026-05 volume: top/mid/bottom (33/33/34). IC by tier:

| factor | top IC (liquid) | mid IC | bottom IC (illiquid) | reading |
|---|---|---|---|---|
| F-MOM | −0.0277 | −0.0497 | −0.0762 | reversal strongest in small-caps |
| **F-REV** | **+0.0375** | +0.0546 | **+0.0729** | monotone ↑ as liquidity ↓ — alpha **concentrates in least-tradeable coins** |
| F-CAR | +0.0124 | −0.0261 | −0.0152 | only top tier positive; not coherent |

**Tradeability (rough Part-4c, daily quintile long-short, optimistic small-cap fees):**

| factor / scope | gross %/yr | turnover/d | cost %/yr | **net %/yr** |
|---|---|---|---|---|
| F-REV **pool-100** | +16.3 | 1.81 | 39.6 | **−23.3** |
| F-REV **top (liquid)** | **−62.1** | 1.87 | 34.1 | **−96.2** |
| F-REV mid | +23.4 | 1.88 | 41.2 | −17.8 |
| F-REV **bottom (illiquid)** | +78.3 | 1.93 | 56.2 | **+22.1** (only positive cell) |
| F-CAR pool-100 | +16.2 | 0.51 | 11.2 | +5.1 (but IC ≈ noise at scale → not a real signal) |
| F-MOM pool-100 | −7.6 | 0.62 | 13.6 | −21.2 |

**The decisive picture:** REV has a *positive broad-rank IC* but the **tradeable extremes are gross-negative in the liquid tier** (−62%/yr: the largest recent losers *keep* losing relative to winners — continuation, consistent with the project capstone's "right-skew continuation"), and the only **positive net** is the **illiquid bottom tier at an optimistic 8 bps** that genuine small-cap impact (often 30–100+ bps) would erase. This is **IC ≠ tradeable alpha** and **liquidity disguise** in one — the *monetizable* alpha lives entirely where you cannot trade it.

## Q5 — 判生死 (verdict)

**Literal pre-registered coded gate (unaltered, `results.json`):** F-REV **PASS** (G1 beats noise ✓, G2 IC-disguise-proxy ✓ [top IC 0.0375 ≥ 0.5×0.0729, barely], G3 scale-improves ✓); F-MOM FAIL; F-CAR FAIL.

**Integrated 判生死 = 不过 / FAIL.** Per Part 5 (which weighs *IC / noise baseline / 可交易价差 / 分层来源*, and lists *"alpha mainly from the untradeable small-cap tier (liquidity disguise)"* as a pre-registered FAIL condition), the one gate-passing factor fails the route on the dimension that matters:
1. **Liquidity disguise (pre-registered FAIL):** REV's *tradeable* alpha is net-positive **only** in the illiquid bottom tier; the liquid top tier is **gross-negative**. The coded G2 used an *IC proxy* for disguise and was too weak to catch it; the **required** tradeability evidence catches it.
2. **Not tradeable at scale:** pool-100 net **−23%/yr** at conservative cost; the "scale improvement" is in **raw IC, not tradeable alpha**.
3. **Survivor bias bites hardest here:** REV = buy recent losers; dead coins are losers that never bounced → including them only lowers REV's alpha. The "alive" reading is an **optimistic upper bound**, strongest exactly in the small/old bottom tier.

> **Discipline note (iron-rule A):** I did **not** redefine the coded gate to kill a passing factor (that is the forbidden direction). The coded PASS is reported as-is. The route verdict applies a **pre-registered FAIL condition** (liquidity disguise) using **pre-registered required evidence** (Part-4c tradeability), with the divergence shown in full. Direction is toward conservatism, transparent, not a resurrection.

## Q6 — Next step

**Do NOT fund the large-scale-factor big engineering.** The free gate's only scale-improving, noise-beating factor is an untradeable, survivor-flattered, high-turnover reversal. The only conceivable spend is the **~$1–6k Tardis survivor-free + realistic-small-cap-cost recheck of reversal specifically** — and *this same run already predicts that recheck fails* (dead coins lower REV alpha; realistic small-cap cost erases the only positive cell). So that spend is **low priority / optional**, and the **~$10–55k/yr institutional route is not justified.** **Estimated saving: the institutional data subscription + the multi-month large-scale-factor engineering build.** If reversal is ever revisited it must be as an *execution/microstructure* problem (cost & impact on small-caps), not a breadth problem.

## Q7 — Observations

- **The scalable signal is overreaction-reversal, and it is the least tradeable one** (turnover 1.8×/day; cost wall ~40–56%/yr). Scale "helps" precisely because small illiquid coins overreact more — the edge and the impossibility of harvesting it are the same fact.
- **Liquid large-caps show extreme-tail *continuation*** (REV top-tier gross −62%/yr), echoing the project capstone `PROJECT_FINAL_SUMMARY_20260614` "right-skew continuation". Cross-sectional reversal does not escape it; it just relocates to coins you can't trade.
- **Carry does not benefit from scale** — it is a top-coin funding effect that *washes out* as breadth grows (+0.019→−0.005). Direct negative answer to "does scale help carry?": no.
- **The free liquid universe is young**: reaching 100 full-history coins forces volume-rank ~551 — a structural limit of free-data large-scale work, independent of any factor.
- **Methodology lesson (reusable):** an **IC-based** liquidity-disguise gate (G2) is too weak; the **tradeable-spread-based** disguise check is what bites. Pair every "IC beats noise / IC improves with scale" claim with a tradeable-spread-by-tier check — IC ≠ tradeable alpha, again (third instance after F-MOM and carry-holding in the cross-sectional study).
- **Noise calibration worked as designed** and is now a reusable asset: it correctly classified MOM (negative), CAR-at-scale (in-noise), and REV (beats noise), and prevented the "positive IC = signal" error.

---

## Methods / reproduction
```
.venv/bin/python scripts/research_factor_scale.py --mode survey                       # universe + frontier
.venv/bin/python scripts/research_factor_scale.py --mode download --window-start 2022-07 --n-pool 100
.venv/bin/python scripts/research_factor_scale.py --mode analyze  --window-start 2022-07
```
Frozen definitions, noise method, and decision line live in the script docstring. pool_22 (top-22 by volume, full-history) overlaps the original 22-coin study **14/22**; the 8 non-overlaps (SUI/TON/PEPE/WLD/ONDO/TAO/ENA/INJ) are young listings correctly excluded by the full-history-since-2022-07 rule.

## Files
- `results.json` — scale gradient, noise calibration, strata, tradeability, coded verdict.
- `noise_distributions.json` + `noise_calibration_pool100.png` — null distributions vs real IC.
- `scale_gradient_ic.png` — IC point estimate vs K.
- `survey_candidates.json` — 588 ranked coins + earliest-month frontier.
- `institutional_data_cost_survey.md` — vendor cost reconnaissance.
- `data/binance_vision/factor_scale_manifest.json` — data manifest (server/window/pool/sha256).

## Documentation updates produced by this study
- **`PROJECT_GUIDE.md` → 已验证的核心事实**: added one bullet (`大规模因子前置门——规模不带来可交易 edge，不立项`) after the basis-arbitrage entry. New strategy-layer conclusion (a route judged), surgical (<25 lines), no existing narrative overwritten — it *extends* the cross-sectional entry (closed at 22 coins) to the scale question (closed at 100). No historical comment needed (additive).
- **Memory**: `MEMORY.md` index + `factor_scale_stageA.md` added.
- **Not changed (considered, deliberately left):** the cross-sectional-IC entry (lines 362-379) — its 22-coin conclusion stands and is *confirmed* by this study, not overturned, so no revision. `已知未验证的假设` — no assumption moved (this gate did not validate a prior open hypothesis; it judged a new route). CLAUDE.md iron rules — none changed (no new methodology rule; the IC≠tradeable-alpha lesson is recorded in the report + memory, consistent with existing iron-rule C spirit).

---
*大规模因子前置阶段A完成于2026-06-28T11:58Z/免费数据:100币2022-07→2026-05(3.9yr,1431d)/机构数据成本:Tardis survivor-free~$1–6k一次性,Kaiko/CoinMetrics~$10–55k/yr/噪声标定:三因子real IC vs 噪声基线 REV beats(p0.000)、MOM显著负(反转)、CAR仅K22 beats其余in-noise/规模梯度(IC点估计22→50→100):REV上升(+0.030→+0.056)但仅毛IC、carry下降(washout)/流动性伪装:是(REV净利仅来自不可交易底层,液性顶层毛负)/判定:不过-唯一规模改善因子REV为高换手反转,可交易层净负(pool100 −23%/yr),液性伪装+幸存者偏乐观,规模只改善毛IC不改善可交易alpha(IC-coded门技术性PASS但综合判生死FAIL,全程透明未改门)/已push:[待确认]*
