# ETC-v1 Early Trend Classifier Feature Discovery

## 1. ETC-v1 是什么研究？
ETC-v1 是 research-only 的趋势早期识别特征发现研究。它使用 Trend Opportunity Map 的趋势段生成 ex-post label，再检验入场前可见 closed-bar 特征是否能提高未来进入 early trend 的概率。

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
| train_ext | top10 | 1086 | 0.171271 | 0.476059 | -0.051207 |
| train_ext | top20 | 2171 | 0.163980 | 0.434362 | -0.058498 |
| train_ext | top30 | 3256 | 0.175369 | 0.406020 | -0.047110 |
| train_ext | bottom30 | 3256 | 0.255221 | 0.222666 | 0.032743 |
| validation_ext | random_control | 1397 | 0.242663 | 0.323550 | 0.000000 |
| validation_ext | top10 | 679 | 0.135493 | 0.484536 | -0.107169 |
| validation_ext | top20 | 1397 | 0.166786 | 0.446671 | -0.075877 |
| validation_ext | top30 | 2112 | 0.174242 | 0.424716 | -0.068420 |
| validation_ext | bottom30 | 2388 | 0.337102 | 0.224456 | 0.094439 |
| oos_ext | random_control | 769 | 0.198960 | 0.245774 | 0.000000 |
| oos_ext | top10 | 340 | 0.126471 | 0.405882 | -0.072489 |
| oos_ext | top20 | 769 | 0.130039 | 0.362809 | -0.068921 |
| oos_ext | top30 | 1221 | 0.135954 | 0.335790 | -0.063006 |
| oos_ext | bottom30 | 2170 | 0.252535 | 0.195853 | 0.053575 |

## 7. top-score events 是否能在 train / validation / oos 中产生正收益？
| group | event_group | hold | split | event_count | early_trend_rate | no_cost_pnl | cost_aware_pnl | funding_adjusted_pnl |
|---|---|---|---|---|---|---|---|---|
| A | top10_score_events | hold_4h | oos_ext | 339 | 0.126844 | 2138.933650 | 1460.933650 | 1441.874784 |
| A | top10_score_events | hold_8h | oos_ext | 339 | 0.126844 | 3081.551680 | 2403.551680 | 2377.870426 |
| A | top10_score_events | hold_1d | oos_ext | 339 | 0.126844 | 3438.055530 | 2760.055530 | 2713.552099 |
| A | top10_score_events | hold_3d | oos_ext | 339 | 0.126844 | 2146.075971 | 1468.075971 | 1375.553952 |
| A | top10_score_events | hold_4h | train_ext | 1086 | 0.171271 | 4536.062608 | 2364.062608 | 2308.298492 |
| A | top10_score_events | hold_8h | train_ext | 1086 | 0.171271 | 7384.371396 | 5212.371396 | 5133.152730 |
| A | top10_score_events | hold_1d | train_ext | 1086 | 0.171271 | 14060.980994 | 11888.980994 | 11691.299633 |
| A | top10_score_events | hold_3d | train_ext | 1086 | 0.171271 | 31497.277120 | 29325.277120 | 28740.021254 |
| A | top10_score_events | hold_4h | validation_ext | 679 | 0.135493 | 3561.709081 | 2203.709081 | 2173.453589 |
| A | top10_score_events | hold_8h | validation_ext | 679 | 0.135493 | 5403.186116 | 4045.186116 | 4000.002226 |
| A | top10_score_events | hold_1d | validation_ext | 679 | 0.135493 | 8171.396323 | 6813.396323 | 6704.657630 |
| A | top10_score_events | hold_3d | validation_ext | 679 | 0.135493 | 8431.942226 | 7073.942226 | 6802.546463 |
| B | top20_score_events | hold_4h | oos_ext | 760 | 0.131579 | 3049.634739 | 1529.634739 | 1481.669926 |
| B | top20_score_events | hold_8h | oos_ext | 760 | 0.131579 | 4377.851647 | 2857.851647 | 2796.586943 |
| B | top20_score_events | hold_1d | oos_ext | 760 | 0.131579 | 7265.627644 | 5745.627644 | 5638.910366 |
| B | top20_score_events | hold_3d | oos_ext | 760 | 0.131579 | 8332.600890 | 6812.600890 | 6596.943254 |
| B | top20_score_events | hold_4h | train_ext | 2171 | 0.163980 | 8779.936454 | 4437.936454 | 4315.744968 |
| B | top20_score_events | hold_8h | train_ext | 2171 | 0.163980 | 13464.816787 | 9122.816787 | 8952.417899 |
| B | top20_score_events | hold_1d | train_ext | 2171 | 0.163980 | 24022.675772 | 19680.675772 | 19294.179893 |
| B | top20_score_events | hold_3d | train_ext | 2171 | 0.163980 | 53831.442960 | 49489.442960 | 48344.372161 |
| B | top20_score_events | hold_4h | validation_ext | 1397 | 0.166786 | 6657.903359 | 3863.903359 | 3794.343511 |
| B | top20_score_events | hold_8h | validation_ext | 1397 | 0.166786 | 10781.338391 | 7987.338391 | 7882.980130 |
| B | top20_score_events | hold_1d | validation_ext | 1397 | 0.166786 | 15736.928233 | 12942.928233 | 12707.556510 |
| B | top20_score_events | hold_3d | validation_ext | 1397 | 0.166786 | 17982.362683 | 15188.362683 | 14608.270441 |
| C | top30_score_events | hold_4h | oos_ext | 1206 | 0.137645 | 3941.696492 | 1529.696492 | 1461.458120 |
| C | top30_score_events | hold_8h | oos_ext | 1206 | 0.137645 | 6103.003578 | 3691.003578 | 3601.373723 |
| C | top30_score_events | hold_1d | oos_ext | 1206 | 0.137645 | 9986.333649 | 7574.333649 | 7407.598318 |
| C | top30_score_events | hold_3d | oos_ext | 1206 | 0.137645 | 13625.198868 | 11213.198868 | 10885.347742 |
| C | top30_score_events | hold_4h | train_ext | 3256 | 0.175369 | 11943.909233 | 5431.909233 | 5255.604839 |
| C | top30_score_events | hold_8h | train_ext | 3256 | 0.175369 | 18045.113589 | 11533.113589 | 11289.319167 |

## 8. cost-aware 和 funding-adjusted 是否通过？
- cost_aware_pass=true
- funding_adjusted_pass=true

## 9. reverse test 是否弱于正向？
| hold | split | forward_no_cost_pnl | reverse_no_cost_pnl | reverse_weaker |
|---|---|---|---|---|
| hold_4h | train_ext | 8779.936454 | -8779.936454 | True |
| hold_4h | validation_ext | 6657.903359 | -6657.903359 | True |
| hold_4h | oos_ext | 3049.634739 | -3049.634739 | True |
| hold_8h | train_ext | 13464.816787 | -13464.816787 | True |
| hold_8h | validation_ext | 10781.338391 | -10781.338391 | True |
| hold_8h | oos_ext | 4377.851647 | -4377.851647 | True |
| hold_1d | train_ext | 24022.675772 | -24022.675772 | True |
| hold_1d | validation_ext | 15736.928233 | -15736.928233 | True |
| hold_1d | oos_ext | 7265.627644 | -7265.627644 | True |
| hold_3d | train_ext | 53831.442960 | -53831.442960 | True |
| hold_3d | validation_ext | 17982.362683 | -17982.362683 | True |
| hold_3d | oos_ext | 8332.600890 | -8332.600890 | True |

## 10. 收益是否集中在单一 symbol 或 top trades？
| group | hold | split | trade_count | largest_symbol_pnl_share | largest_symbol_event_share | top_5pct_trade_pnl_contribution | concentration_pass |
|---|---|---|---|---|---|---|---|
| A | hold_1d | oos_ext | 339 | 0.285813 | 0.274336 | 0.652831 | True |
| A | hold_1d | train_ext | 1086 | 0.443371 | 0.290976 | 0.625545 | True |
| A | hold_1d | validation_ext | 679 | 0.350238 | 0.294551 | 0.757905 | True |
| A | hold_3d | oos_ext | 339 | 0.337972 | 0.274336 | 1.834291 | False |
| A | hold_3d | train_ext | 1086 | 0.326829 | 0.290976 | 0.429636 | True |
| A | hold_3d | validation_ext | 679 | 0.475477 | 0.294551 | 1.032666 | False |
| A | hold_4h | oos_ext | 339 | 0.458573 | 0.274336 | 0.848435 | False |
| A | hold_4h | train_ext | 1086 | 0.567818 | 0.290976 | 1.393397 | False |
| A | hold_4h | validation_ext | 679 | 0.422988 | 0.294551 | 1.346161 | False |
| A | hold_8h | oos_ext | 339 | 0.352987 | 0.274336 | 0.577639 | True |
| A | hold_8h | train_ext | 1086 | 0.460095 | 0.290976 | 0.838918 | False |
| A | hold_8h | validation_ext | 679 | 0.482306 | 0.294551 | 0.910746 | False |
| B | hold_1d | oos_ext | 760 | 0.507690 | 0.240789 | 0.697247 | True |
| B | hold_1d | train_ext | 2171 | 0.414977 | 0.255643 | 0.701541 | True |
| B | hold_1d | validation_ext | 1397 | 0.366077 | 0.256979 | 0.835804 | False |
| B | hold_3d | oos_ext | 760 | 0.440631 | 0.240789 | 0.931482 | False |
| B | hold_3d | train_ext | 2171 | 0.336520 | 0.255643 | 0.498257 | True |
| B | hold_3d | validation_ext | 1397 | 0.511113 | 0.256979 | 1.148982 | False |
| B | hold_4h | oos_ext | 760 | 0.427088 | 0.240789 | 1.623922 | False |
| B | hold_4h | train_ext | 2171 | 0.646627 | 0.255643 | 1.631491 | False |

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
| A | hold_4h | oos_ext | 0.126844 | 0.172368 | 2138.933650 | 523.018010 | False |
| A | hold_8h | oos_ext | 0.126844 | 0.172368 | 3081.551680 | 981.584468 | False |
| A | hold_1d | oos_ext | 0.126844 | 0.172368 | 3438.055530 | 2575.279930 | False |
| A | hold_3d | oos_ext | 0.126844 | 0.172368 | 2146.075971 | 3495.951525 | False |
| A | hold_4h | train_ext | 0.171271 | 0.242745 | 4536.062608 | 2518.985665 | False |
| A | hold_8h | train_ext | 0.171271 | 0.242745 | 7384.371396 | 4157.365608 | False |
| A | hold_1d | train_ext | 0.171271 | 0.242745 | 14060.980994 | 9657.348131 | False |
| A | hold_3d | train_ext | 0.171271 | 0.242745 | 31497.277120 | 19044.444306 | False |
| A | hold_4h | validation_ext | 0.135493 | 0.239084 | 3561.709081 | 3247.272704 | False |
| A | hold_8h | validation_ext | 0.135493 | 0.239084 | 5403.186116 | 5257.877975 | False |
| A | hold_1d | validation_ext | 0.135493 | 0.239084 | 8171.396323 | 7370.651166 | False |
| A | hold_3d | validation_ext | 0.135493 | 0.239084 | 8431.942226 | 7149.764571 | False |
| B | hold_4h | oos_ext | 0.131579 | 0.172368 | 3049.634739 | 523.018010 | False |
| B | hold_8h | oos_ext | 0.131579 | 0.172368 | 4377.851647 | 981.584468 | False |
| B | hold_1d | oos_ext | 0.131579 | 0.172368 | 7265.627644 | 2575.279930 | False |
| B | hold_3d | oos_ext | 0.131579 | 0.172368 | 8332.600890 | 3495.951525 | False |
| B | hold_4h | train_ext | 0.163980 | 0.242745 | 8779.936454 | 2518.985665 | False |
| B | hold_8h | train_ext | 0.163980 | 0.242745 | 13464.816787 | 4157.365608 | False |
| B | hold_1d | train_ext | 0.163980 | 0.242745 | 24022.675772 | 9657.348131 | False |
| B | hold_3d | train_ext | 0.163980 | 0.242745 | 53831.442960 | 19044.444306 | False |