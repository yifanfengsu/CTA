# Research Decision Dossier

## 1. Executive Summary
- Actual OKX funding data is now complete for BTC/ETH/SOL/LINK/DOGE over 2023-2026.
- Funding-aware analysis completed.
- Funding-aware gates remain closed.
- Strategy development remains blocked.
- Demo/live remains blocked.
- 当前没有任何策略可进入 demo/live。
- 当前没有任何 policy 可进入 Strategy V3 原型开发。
- 当前趋势跟踪 V3.0 family 已失败。
- External Regime Classifier Gate Audit 已完成，未能救回当前 V3 family。
- Derivatives Data Readiness Audit 已完成，但关键历史衍生品特征覆盖未通过。
- 当前五品种趋势跟踪 family 已最终封档。
- 继续趋势跟踪需要新的研究前提，而不是继续调当前参数。
- 当前推荐暂停策略开发，只维护数据与研究工具。
- no policy can be traded from the current research package.

## 2. Data Status
- 2023-2026 五品种数据完整。
- 当前 symbols: BTC / ETH / SOL / LINK / DOGE。
- market_data_complete=true
- data_ready=true
- funding_data_complete=true
- funding_source=OKX Historical Market Data
- rest_funding_endpoint_partial_only=true
- historical_funding_auto_download_succeeded=true
- coverage_window=2023-01-01 to 2026-03-31
- interval=1m
- missing_count=0
- gap_count=0
- 数据不是当前失败原因。

## 3. Funding-aware Final Gate
- funding_data_complete=true
- actual_funding_source=OKX Historical Market Data
- rest_funding_endpoint_partial_only=true
- historical_funding_auto_download_succeeded=true
- funding_adjusted_stable_candidate_exists=false
- can_enter_funding_aware_v3_1_research=false
- strategy_development_allowed=false
- demo_live_allowed=false
- current_v3_family_failed_after_actual_funding=true
- target_policy=v3_1d_ema_50_200_atr5
- why_gates_stay_closed=actual funding completion removes the data blocker, but it does not remove Extended V3 stable_candidate=false, top-trade concentration, or regime diagnostics rejection.

| inst_id | row_count | first_time | last_time | complete |
| --- | ---: | --- | --- | --- |
| BTC-USDT-SWAP | 3558 | 2022-12-31T16:00:00+00:00 | 2026-03-31T08:00:00+00:00 | true |
| ETH-USDT-SWAP | 3558 | 2022-12-31T16:00:01+00:00 | 2026-03-31T08:00:00+00:00 | true |
| SOL-USDT-SWAP | 3558 | 2022-12-31T16:00:12+00:00 | 2026-03-31T08:00:00+00:00 | true |
| LINK-USDT-SWAP | 3558 | 2022-12-31T16:00:05+00:00 | 2026-03-31T08:00:00+00:00 | true |
| DOGE-USDT-SWAP | 3558 | 2022-12-31T16:00:11+00:00 | 2026-03-31T08:00:00+00:00 | true |

## 4. External Regime Classifier Gate Audit Final Result
- 之前 classifier 的 stable_candidate_like 口径存在问题。
- 旧口径使用正收益合计作 top trade concentration 分母，低估 top trade concentration。
- 修正后 v3_1d_ema_50_200_atr5 的 OOS top 5% contribution=1.9818，超过 0.8。
- original_all 不通过 strict gate。
- exclude_hostile_chop_overheated 和 exclude_funding_overheated 没有改变 v3_1d_ema_50_200_atr5 的 OOS trade set。
- trend_friendly_only 删除全部 OOS trades。
- external_regime_classifier_gate_audit_complete=true
- classifier_old_gate_inconsistent=true
- classifier_strict_stable_candidate_exists=false
- can_enter_research_only_v3_1_classifier_experiment=false
- external_classifier_rescued_v3_family=false
- strategy_development_allowed=false
- demo_live_allowed=false
- gate_audit_reason=No non-original classifier filter passed strict gates after Dossier-consistent concentration checks.

## Derivatives Data Readiness Final Gate
- Derivatives-confirmed trend hypothesis 是新的研究假设，不是当前 V3 family 的延续。
- 本次 audit 没有证明有足够历史衍生品数据支持研究。
- Funding 数据完整，但 funding alone 不足以构成 derivatives confirmation。
- Mark/index candle 可用，但只能构造 basis proxy，不足以替代 OI/taker/long-short。
- Open interest 当前只支持 current snapshot，不能用于 2023-2026 historical research。
- Taker buy/sell volume、long/short ratio、contracts OI/volume、OI history、premium history 均未证明能覆盖 2023-2026。
- derivatives_data_readiness_audit_complete=true
- can_enter_derivatives_confirmed_trend_research=false
- derivatives_data_blocker=true
- derivatives_missing_historical_features=open_interest_history, taker_buy_sell_volume, long_short_account_ratio, contracts_open_interest_volume, premium_history
- derivatives_available_features=actual_funding_rate, funding_dispersion, funding_sign_breadth, funding_trend, mark_index_price_candles
- derivatives_research_recommended_next_step=pause_research
- funding_complete_but_not_sufficient=true
- mark_index_available_but_not_sufficient=true
- current_open_interest_snapshot_only=true
- key_historical_features_coverage_not_proven=true
- strategy_development_allowed=false
- demo_live_allowed=false
- recommended_next_step=pause strategy development and maintain research/data tooling.

## 5. Research Timeline
| stage | goal | result | pass/fail | key finding | decision |
| --- | --- | --- | --- | --- | --- |
| 1m breakout | Test raw short-horizon Donchian breakout as trend-following entry. | Failed. | fail | Short-term breakout did not produce stable cost-aware edge. | Stop optimizing 1m breakout. |
| Signal Lab | Identify features explaining signal outcomes. | Stable negative risk features found. | diagnostic_pass_strategy_fail | High volatility, ATR, breakout distance, recent return, volume spike, and large body ratio are negative. | Treat short-term breakout as overheat/exhaustion risk, not a strategy candidate. |
| HTF Signal Research | Use 1h regime plus 15m structure and 5m pullback/reclaim. | Failed. | fail | No stable HTF policy across train/validation/oos. | Do not enter Strategy V2 from HTF policies. |
| Trend V2 | Test single-symbol 1h/4h trend-following families. | Failed. | fail | No stable candidate. | Stop single-symbol V2 direction. |
| Trend V3 | Test multi-symbol portfolio-level 4h/1d Donchian/EMA/ensemble policies. | Failed. | fail | No stable candidate; concentration and OOS fragility remain. | Do not build Strategy V3. |
| Extended V3 | Retest the same V3.0 policy set on 2023-2026 data. | Failed. | fail | stable_candidate_exists=false; only 1d EMA had all no-cost splits positive but failed concentration/funding gates. | Do not enter V3.1 from extended V3 alone. |
| Regime Diagnostics | Check whether trend regimes exist and whether V3.0 aligned with them. | Rejected V3.1. | fail | Strong trend share is 4.79%; V3 profits were not mainly from strong trend; 1d EMA strong-regime PnL was negative. | proceed_to_v3_1_research=false. |
| Funding-aware Final Gate | Apply complete actual OKX funding to Trend V3 extended trades. | Completed; gates still closed. | data_pass_strategy_fail | funding_data_complete=true, but funding_adjusted_stable_candidate_exists=false. | current_v3_family_failed_after_actual_funding=true. |
| External Regime Classifier Gate Audit | Check whether classifier-filtered V3.1 rescue is consistent with Dossier and Extended V3 strict gates. | Completed; no strict stable candidate. | fail | Old classifier gate underestimated top-trade concentration; v3_1d_ema_50_200_atr5 OOS top 5% contribution=1.9818. | can_enter_research_only_v3_1_classifier_experiment=false. |
| Derivatives Data Readiness Audit | Check whether public/no-key OKX derivatives data can support a new derivatives-confirmed trend hypothesis. | Completed; data readiness gate blocked. | data_fail_research_blocked | Funding and mark/index candles are available, but historical OI/taker/long-short coverage for 2023-2026 was not proven. | can_enter_derivatives_confirmed_trend_research=false. |

## 6. Failed Policy Families
| policy family | status | failure reason | evidence | tradable |
| --- | --- | --- | --- | --- |
| 1m Donchian breakout | failed | No stable positive edge; Signal Lab shows short-term breakout behaves like overheat/exhaustion risk. | Original 1m Donchian breakout and alpha diagnostics failed. | false |
| 1h/15m/5m HTF pullback | failed | No stable HTF policy; pullback/reclaim variants were negative across train/validation/oos. | HTF Signal Research compare rejected Strategy V2 entry. | false |
| 4h Donchian | failed | No stable candidate; weak validation/OOS behavior and choppy/high_vol_choppy loss exposure. | Trend V3 and regime attribution. | false |
| 1d Donchian | failed | No stable candidate; OOS fragility and Donchian losses concentrated in choppy/high_vol_choppy. | Extended Trend V3 and Trend Regime Diagnostics. | false |
| 4h EMA | failed | No stable candidate; OOS no-cost/cost-aware failure. | Trend V3 compare and Extended Trend V3 compare. | false |
| 1d EMA as currently defined | failed | Best weak lead, but rejected by top trade concentration, funding stress, and regime diagnostics. | v3_1d_ema_50_200_atr5 was not stable and strong-trend no-cost PnL was negative. | false |
| vol compression Donchian breakout | failed | No stable candidate; negative across validation/OOS after costs. | Trend V3 compare and Extended Trend V3 compare. | false |
| V3 ensemble_core | failed | No stable ensemble edge; inherits failed 4h/1d Donchian and 4h EMA components. | Trend V3 compare and Extended Trend V3 compare. | false |
| funding-adjusted V3 family | failed | Actual OKX funding data is complete, but funding-adjusted stable_candidate_exists remains false. | Funding-aware Trend Research actual funding report and Research Decision Dossier final gate. | false |
| VSVCB-v1 positive breakout hypothesis | failed | Breakout + squeeze + volume confirmation failed train/validation/OOS and reverse test was stronger than the positive breakout hypothesis. | VSVCB-v1 Phase 1 and postmortem. | false |

## 7. Signal Lab Findings
- short-term breakout 更像 overheat/exhaustion risk。
- high volatility / high ATR / large breakout / high recent return / volume spike / large body ratio 都是负向风险特征。
- 稳定负向特征：recent_volatility_30m, atr_pct, breakout_distance_atr, recent_return_30m, recent_return_15m, recent_return_5m, volume_zscore_30m, body_ratio

## 8. Trend Regime Findings
- strong trend 占比 4.79%。
- choppy/high_vol_choppy 占比 38.70%。
- strongest symbol SOLUSDT_SWAP_OKX.GLOBAL。
- weakest symbol BTCUSDT_SWAP_OKX.GLOBAL。
- 1d EMA 不只在 strong trend 有效；strong no-cost PnL=-0.43818000000000007。
- Donchian 亏在 choppy/high_vol_choppy：true。

## 9. Why Strategy Development Is Blocked
Strategy development is blocked because:
- no stable candidate
- regime diagnostics rejects V3.1
- demo/live disallowed
- top trade concentration unresolved
- symbol concentration unresolved
- actual OKX funding complete, but funding-adjusted stable_candidate_exists=false
- Extended Trend V3 stable_candidate_exists=false
- Trend Regime Diagnostics proceed_to_v3_1_research=false
- Trend V3 Postmortem proceed_to_v3_1=false
- v3_1d_ema_50_200_atr5 remains a weak lead only, rejected by concentration and regime gates
- External classifier gate audit complete; no strict stable candidate
- Derivatives data readiness blocks derivatives-confirmed trend research
- current five-symbol trend-following family is fully archived

## 10. Retained Research Hypotheses
| hypothesis | status | reason | not allowed as |
| --- | --- | --- | --- |
| broader universe trend following | optional_research_only | Five-symbol universe may be too narrow for sparse crypto trend regimes, but this is not the recommended main path while the current user scope avoids universe expansion. | Strategy V3 or demo/live without new evidence. |
| true funding-aware trend following | completed_no_gate_opened | Actual OKX funding is now complete for the current universe; funding-aware analysis did not create a stable candidate. | Further funding-only work on the current V3 family. |
| stronger macro/trend regime classifier | closed_for_current_v3_family | External classifier gate audit found no strict stable candidate; OOS trend_friendly was too sparse and exclude filters did not rescue V3. | V3.1 rescue or direct trade filter on current V3 family. |
| longer history beyond 2021 if listing metadata supports it | conditional_research_only | More cycles may be needed, but only if listing metadata and sqlite coverage can be verified. | Backfill assumption without metadata verification. |
| 1d EMA only as weak research lead | weak_lead_research_only | It is still the only all no-cost positive policy, but actual funding completion did not fix concentration and regime failures. | Tradable policy or Strategy V3 prototype. |
| derivatives-confirmed trend following | data_blocked_research_hypothesis | Historical OI/taker/long-short coverage not proven for 2023-2026; funding plus mark/index alone is not enough derivatives confirmation. | Strategy V3, V3.1, demo/live, or a tradable policy from readiness audit output. |

## 11. Do Not Continue List
| item | reason |
| --- | --- |
| do not expand current Donchian grid | Current Donchian families failed stability and regime attribution. |
| do not trade v3_1d_ema_50_200_atr5 | Rejected by top trade concentration, funding fragility, and regime diagnostics. |
| do not develop v3_1d_ema_50_200_atr5 | Actual funding analysis completed, but it is still not a stable strategy candidate. |
| do not continue current V3 family after actual funding completion | Funding-aware final gate did not open V3.1 or Strategy V3. |
| do not enter funding-aware V3.1 without a new hypothesis | Current V3 family failed after actual funding and regime diagnostics remain blocking. |
| do not enter demo/live | No stable candidate and demo_live_allowed=false. |
| do not build Strategy V3 from current results | strategy_development_allowed=false. |
| do not continue V3.0 ensemble_core | No stable candidate and component families failed. |
| do not optimize 1m breakout | Short-term breakout is aligned with overheat/exhaustion risk, not durable trend following. |
| do not continue external regime classifier as V3.1 rescue | Gate consistency audit found no strict stable classifier-filtered candidate. |
| do not treat original_all v3_1d_ema_50_200_atr5 as stable | OOS top 5% trade contribution is 1.9818, above the 0.8 gate. |
| do not ignore top trade concentration | Top-trade concentration is the decisive Extended V3 and classifier gate blocker. |
| do not treat no-cost positive as tradable | No-cost positives still fail concentration/funding/regime gates and are not strategy candidates. |
| do not use classifier filters that do not change OOS trade set | exclude_hostile_chop_overheated and exclude_funding_overheated did not affect v3_1d_ema_50_200_atr5 OOS trades. |
| do not start derivatives-confirmed trend research without historical OI/taker/long-short coverage | Derivatives readiness audit did not prove the required 2023-2026 historical derivatives feature coverage. |
| do not use current open interest snapshot as historical feature | The public open-interest endpoint probe is current snapshot only and cannot stand in for 2023-2026 history. |
| do not treat funding alone as derivatives confirmation | Funding data is complete, but funding alone does not satisfy the required OI/taker/long-short confirmation mix. |
| do not develop Strategy V3 from derivatives readiness audit | Endpoint availability audit is not a strategy result and can_enter_derivatives_confirmed_trend_research=false. |

## 12. Next Research Options
| option | name | prerequisites | acceptance criteria | allowed now |
| --- | --- | --- | --- | --- |
| Option A | Broader universe trend following readiness (optional, not main path) | Verified metadata, listing dates, contract specs, liquidity, and 1m sqlite coverage for a materially broader symbol set. | Multi-symbol universe passes coverage checks; research design pre-registers stable-candidate criteria before any policy comparison. | optional_not_recommended |
| Option B | Funding-aware research complete for current universe | A new non-V3 policy family or new hypothesis before any further funding-only work. | No additional funding-only research unless a new candidate policy family emerges. | no |
| Option C | classifier-filtered V3.1 continuation | Closed for current V3 family. | Not applicable; strict gate audit found no stable classifier-filtered candidate. | no |
| Option D | Pause strategy development and maintain tooling | None beyond maintaining data integrity and reproducible reports. | No new strategy work starts until a new research premise is documented and approved. | yes |
| Option E | Derivatives-confirmed trend research | historical derivatives metrics coverage proven | Historical OI or contracts OI/volume plus taker volume or long/short ratio must cover 2023-2026 without private API keys; funding alone is insufficient. | no |
| Option F | Third-party/external derivatives data audit | user explicitly agrees to external data source or paid/free vendor evaluation | External source must provide reproducible 2023-2026 coverage, licensing clarity, and no strategy conclusion before data audit passes. | optional |

## 13. Final Decision
- strategy_development_allowed=false
- demo_live_allowed=false
- proceed_to_v3_1_research=false
- can_enter_research_only_v3_1_classifier_experiment=false
- can_enter_derivatives_confirmed_trend_research=false
- derivatives_data_blocker=true
- current_v3_family_failed=true
- current_v3_family_failed_after_actual_funding=true
- final_current_trend_family_archived=true
- final_current_research_archived=true
- final_strategy_development_allowed=false
- final_demo_live_allowed=false
- proceed_to_broader_universe_research=optional
- proceed_to_funding_research=complete_for_current_universe_no_gate_opened
- derivatives_research_recommended_next_step=pause_research
- next_default=Pause strategy development and maintain research/data tooling.

## Source Report Warnings
- none
