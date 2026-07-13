# ETC-v1 Inverse: Early Trend Classifier Feature Discovery (Sign-Flipped)

## 1. ETC-v1 Inverse 是什么研究？
ETC-v1 Inverse 是 ETC-v1 的符号翻转版本。原版 ETC-v1 发现 composite score 呈反向：high-score 样本的 early_trend_rate 低于 low-score 样本，且 top10/bottom30 的 PnL 排序在 OOS 中与预期相反。因此本版本将所有 6 个 COMPOSITE_TERMS 符号翻转，原版 bottom30 变为新版 top30。本研究是 research-only，不修改策略、不进入 demo/live。

## 2. 趋势早期标签如何定义？
趋势段前 25% 标记为 early_uptrend 或 early_downtrend，25%-75% 标记为 middle_trend，75% 之后标记为 late_trend。不在趋势段且未来窗口内没有有效 2ATR/3ATR 机会的样本标记为 nontrend；重叠、方向冲突、边界、数据不足和未来窗口不足标记为 excluded_ambiguous。

## 3. 哪些特征完全使用入场前信息？
所有 feature columns 均来自当前 timestamp 已完成 closed bar 及其历史 rolling 窗口，包括效率、广度、相对强弱、波动、funding、回撤位置和成交结构。未来收益、future MFE/MAE、trend_segment_end 和趋势标签没有进入特征计算。

## 4. 哪些特征最能区分 early trend 和 nontrend？
| feature | feature_predictiveness_score | monotonicity_score | train_validation_consistency | oos_consistency |
|---|---|---|---|---|
| market_return_breadth_55 | 0.165382 | 0.800671 | False | False |
| distance_from_60d_high | 0.131905 | 0.608185 | True | False |
| drawdown_from_60d_high | 0.131905 | 0.608185 | True | False |
| symbols_above_rolling_midline_55 | 0.128716 | 0.860293 | False | False |
| trend_efficiency_change_20 | 0.122170 | 0.938197 | True | True |
| distance_from_20d_high | 0.113584 | 0.779942 | True | True |
| efficiency_to_volatility_ratio_20 | 0.107875 | 0.944096 | True | True |
| rebound_from_recent_low_60 | 0.104366 | 0.086689 | False | False |
| atr_percentile_200 | 0.104254 | 0.953739 | True | True |
| directional_efficiency_ratio_55 | 0.103760 | 0.371732 | True | True |
| price_position_in_60d_range | 0.099048 | 0.927087 | True | False |
| trend_efficiency_20 | 0.098036 | 0.909955 | True | True |

## 5. 是否存在 train / validation / oos 一致的单特征？
- consistent_feature_count=6

## 6. composite score 是否优于 random control？
| split | score_bucket | sample_count | early_trend_rate | future_3atr_hit_rate | random_control_comparison |
|---|---|---|---|---|---|
| train_ext | random_control | 2171 | 0.222478 | 0.300278 | 0.000000 |
| train_ext | top10 | 1086 | 0.253223 | 0.196133 | 0.030745 |
| train_ext | top20 | 2171 | 0.255182 | 0.215108 | 0.032704 |
| train_ext | top30 | 3256 | 0.255221 | 0.222666 | 0.032743 |
| train_ext | bottom30 | 3256 | 0.175369 | 0.406020 | -0.047110 |
| validation_ext | random_control | 1587 | 0.236295 | 0.328922 | 0.000000 |
| validation_ext | top10 | 799 | 0.386733 | 0.183980 | 0.150439 |
| validation_ext | top20 | 1587 | 0.356018 | 0.207940 | 0.119723 |
| validation_ext | top30 | 2388 | 0.337102 | 0.224456 | 0.100807 |
| validation_ext | bottom30 | 2112 | 0.174242 | 0.424716 | -0.062052 |
| oos_ext | random_control | 1467 | 0.201091 | 0.241990 | 0.000000 |
| oos_ext | top10 | 763 | 0.284404 | 0.182176 | 0.083313 |
| oos_ext | top20 | 1467 | 0.257669 | 0.194956 | 0.056578 |
| oos_ext | top30 | 2170 | 0.252535 | 0.195853 | 0.051444 |
| oos_ext | bottom30 | 1221 | 0.135954 | 0.335790 | -0.065137 |

## 7. top-score events 是否能在 train / validation / oos 中产生正收益？
| group | event_group | hold | split | event_count | early_trend_rate | no_cost_pnl | cost_aware_pnl | funding_adjusted_pnl |
|---|---|---|---|---|---|---|---|---|
| A | top10_score_events | hold_4h | oos_ext | 760 | 0.285526 | -285.515192 | -1805.515192 | -1793.045047 |
| A | top10_score_events | hold_8h | oos_ext | 760 | 0.285526 | 293.637819 | -1226.362181 | -1211.817844 |
| A | top10_score_events | hold_1d | oos_ext | 760 | 0.285526 | 871.142867 | -648.857133 | -631.565417 |
| A | top10_score_events | hold_3d | oos_ext | 760 | 0.285526 | 2839.646315 | 1319.646315 | 1363.453738 |
| A | top10_score_events | hold_4h | train_ext | 1086 | 0.253223 | 288.058990 | -1883.941010 | -1854.573404 |
| A | top10_score_events | hold_8h | train_ext | 1086 | 0.253223 | 762.540426 | -1409.459574 | -1367.576641 |
| A | top10_score_events | hold_1d | train_ext | 1086 | 0.253223 | -1585.776702 | -3757.776702 | -3669.542944 |
| A | top10_score_events | hold_3d | train_ext | 1086 | 0.253223 | -5535.960279 | -7707.960279 | -7453.709688 |
| A | top10_score_events | hold_4h | validation_ext | 799 | 0.386733 | -1086.729123 | -2684.729123 | -2693.420406 |
| A | top10_score_events | hold_8h | validation_ext | 799 | 0.386733 | -1750.018382 | -3348.018382 | -3356.843107 |
| A | top10_score_events | hold_1d | validation_ext | 799 | 0.386733 | -2674.819097 | -4272.819097 | -4276.591611 |
| A | top10_score_events | hold_3d | validation_ext | 799 | 0.386733 | -4510.145648 | -6108.145648 | -6102.974840 |
| B | top20_score_events | hold_4h | oos_ext | 1461 | 0.258727 | 630.876953 | -2291.123047 | -2299.228135 |
| B | top20_score_events | hold_8h | oos_ext | 1461 | 0.258727 | 1192.847812 | -1729.152188 | -1736.256720 |
| B | top20_score_events | hold_1d | oos_ext | 1461 | 0.258727 | 3982.436113 | 1060.436113 | 1041.823670 |
| B | top20_score_events | hold_3d | oos_ext | 1461 | 0.258727 | 9125.574080 | 6203.574080 | 6160.571081 |
| B | top20_score_events | hold_4h | train_ext | 2171 | 0.255182 | -1385.067760 | -5727.067760 | -5715.630026 |
| B | top20_score_events | hold_8h | train_ext | 2171 | 0.255182 | -789.884244 | -5131.884244 | -5111.931401 |
| B | top20_score_events | hold_1d | train_ext | 2171 | 0.255182 | -1488.853233 | -5830.853233 | -5780.557331 |
| B | top20_score_events | hold_3d | train_ext | 2171 | 0.255182 | -13640.314117 | -17982.314117 | -17859.205991 |
| B | top20_score_events | hold_4h | validation_ext | 1587 | 0.356018 | -1090.541279 | -4264.541279 | -4281.330337 |
| B | top20_score_events | hold_8h | validation_ext | 1587 | 0.356018 | -2231.079438 | -5405.079438 | -5421.991671 |
| B | top20_score_events | hold_1d | validation_ext | 1587 | 0.356018 | -4017.413249 | -7191.413249 | -7195.277121 |
| B | top20_score_events | hold_3d | validation_ext | 1587 | 0.356018 | -4588.015825 | -7762.015825 | -7746.263744 |
| C | top30_score_events | hold_4h | oos_ext | 2159 | 0.253821 | 88.536973 | -4229.463027 | -4254.348272 |
| C | top30_score_events | hold_8h | oos_ext | 2159 | 0.253821 | 578.903294 | -3739.096706 | -3772.626540 |
| C | top30_score_events | hold_1d | oos_ext | 2159 | 0.253821 | 5785.000019 | 1467.000019 | 1411.358996 |
| C | top30_score_events | hold_3d | oos_ext | 2159 | 0.253821 | 15660.418117 | 11342.418117 | 11195.883828 |
| C | top30_score_events | hold_4h | train_ext | 3256 | 0.255221 | -1328.146768 | -7840.146768 | -7861.718992 |
| C | top30_score_events | hold_8h | train_ext | 3256 | 0.255221 | 2018.271142 | -4493.728858 | -4524.343942 |

## 8. cost-aware 和 funding-adjusted 是否通过？
- cost_aware_pass=true
- funding_adjusted_pass=true

## 9. reverse test 是否弱于正向？
| hold | split | forward_no_cost_pnl | reverse_no_cost_pnl | reverse_weaker |
|---|---|---|---|---|
| hold_4h | train_ext | -1385.067760 | 1385.067760 | False |
| hold_4h | validation_ext | -1090.541279 | 1090.541279 | False |
| hold_4h | oos_ext | 630.876953 | -630.876953 | True |
| hold_8h | train_ext | -789.884244 | 789.884244 | False |
| hold_8h | validation_ext | -2231.079438 | 2231.079438 | False |
| hold_8h | oos_ext | 1192.847812 | -1192.847812 | True |
| hold_1d | train_ext | -1488.853233 | 1488.853233 | False |
| hold_1d | validation_ext | -4017.413249 | 4017.413249 | False |
| hold_1d | oos_ext | 3982.436113 | -3982.436113 | True |
| hold_3d | train_ext | -13640.314117 | 13640.314117 | False |
| hold_3d | validation_ext | -4588.015825 | 4588.015825 | False |
| hold_3d | oos_ext | 9125.574080 | -9125.574080 | True |

## 10. 收益是否集中在单一 symbol 或 top trades？
| group | hold | split | trade_count | largest_symbol_pnl_share | largest_symbol_event_share | top_5pct_trade_pnl_contribution | concentration_pass |
|---|---|---|---|---|---|---|---|
| A | hold_1d | oos_ext | 760 | 0.654835 | 0.284211 | None | True |
| A | hold_1d | train_ext | 1086 | 0.612856 | 0.243094 | None | True |
| A | hold_1d | validation_ext | 799 | 0.311400 | 0.222778 | None | True |
| A | hold_3d | oos_ext | 760 | 0.598964 | 0.284211 | 4.798882 | False |
| A | hold_3d | train_ext | 1086 | 0.609521 | 0.243094 | None | True |
| A | hold_3d | validation_ext | 799 | 1.000000 | 0.222778 | None | False |
| A | hold_4h | oos_ext | 760 | 0.324261 | 0.284211 | None | True |
| A | hold_4h | train_ext | 1086 | 0.556816 | 0.243094 | None | True |
| A | hold_4h | validation_ext | 799 | 1.000000 | 0.222778 | None | False |
| A | hold_8h | oos_ext | 760 | 0.957689 | 0.284211 | None | False |
| A | hold_8h | train_ext | 1086 | 0.537362 | 0.243094 | None | True |
| A | hold_8h | validation_ext | 799 | 1.000000 | 0.222778 | None | False |
| B | hold_1d | oos_ext | 1461 | 0.433393 | 0.265572 | 7.928899 | False |
| B | hold_1d | train_ext | 2171 | 0.554395 | 0.241363 | None | True |
| B | hold_1d | validation_ext | 1587 | 0.293879 | 0.226843 | None | True |
| B | hold_3d | oos_ext | 1461 | 0.592721 | 0.265572 | 2.119920 | False |
| B | hold_3d | train_ext | 2171 | 0.775760 | 0.241363 | None | False |
| B | hold_3d | validation_ext | 1587 | 0.393394 | 0.226843 | None | True |
| B | hold_4h | oos_ext | 1461 | 0.329416 | 0.265572 | None | True |
| B | hold_4h | train_ext | 2171 | 0.636996 | 0.241363 | None | True |

## 11. 是否允许进入 Phase 2？
- can_enter_phase2=false
- final_decision=postmortem_or_pause
- recommended_next_step=postmortem_or_pause

## 12. 是否允许修改正式策略？
- strategy_development_allowed=false

## 13. 是否允许 demo/live？
- demo_live_allowed=false

## Controls
| group | hold | split | positive_early_trend_rate | random_early_trend_rate | positive_no_cost_pnl | random_no_cost_pnl | random_weaker |
|---|---|---|---|---|---|---|---|
| A | hold_4h | oos_ext | 0.285526 | 0.188227 | -285.515192 | 1518.011750 | False |
| A | hold_8h | oos_ext | 0.285526 | 0.188227 | 293.637819 | 1172.686665 | False |
| A | hold_1d | oos_ext | 0.285526 | 0.188227 | 871.142867 | 4455.441832 | False |
| A | hold_3d | oos_ext | 0.285526 | 0.188227 | 2839.646315 | 12080.867274 | False |
| A | hold_4h | train_ext | 0.253223 | 0.242745 | 288.058990 | 2518.985665 | False |
| A | hold_8h | train_ext | 0.253223 | 0.242745 | 762.540426 | 4157.365608 | False |
| A | hold_1d | train_ext | 0.253223 | 0.242745 | -1585.776702 | 9657.348131 | False |
| A | hold_3d | train_ext | 0.253223 | 0.242745 | -5535.960279 | 19044.444306 | False |
| A | hold_4h | validation_ext | 0.386733 | 0.262130 | -1086.729123 | 79.868118 | False |
| A | hold_8h | validation_ext | 0.386733 | 0.262130 | -1750.018382 | 1567.564756 | False |
| A | hold_1d | validation_ext | 0.386733 | 0.262130 | -2674.819097 | 4871.671907 | False |
| A | hold_3d | validation_ext | 0.386733 | 0.262130 | -4510.145648 | 4680.385282 | False |
| B | hold_4h | oos_ext | 0.258727 | 0.188227 | 630.876953 | 1518.011750 | False |
| B | hold_8h | oos_ext | 0.258727 | 0.188227 | 1192.847812 | 1172.686665 | True |
| B | hold_1d | oos_ext | 0.258727 | 0.188227 | 3982.436113 | 4455.441832 | False |
| B | hold_3d | oos_ext | 0.258727 | 0.188227 | 9125.574080 | 12080.867274 | False |
| B | hold_4h | train_ext | 0.255182 | 0.242745 | -1385.067760 | 2518.985665 | False |
| B | hold_8h | train_ext | 0.255182 | 0.242745 | -789.884244 | 4157.365608 | False |
| B | hold_1d | train_ext | 0.255182 | 0.242745 | -1488.853233 | 9657.348131 | False |
| B | hold_3d | train_ext | 0.255182 | 0.242745 | -13640.314117 | 19044.444306 | False |