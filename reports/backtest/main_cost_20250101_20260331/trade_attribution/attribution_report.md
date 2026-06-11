# 交易归因诊断报告

## 结论
- 主要判断：毛 alpha 问题：无成本版本 total_net_pnl 仍为负，不能把亏损主要归因于手续费或滑点。
- gross_alpha_negative：True
- cost_drag_dominant：False
- candidate_viable_for_oos：False
- primary_issue：gross_alpha_negative

## 核心指标
- 总交易次数：17636
- 往返次数：8818（trades 推断 8818）
- 总净利润：-29254.857349
- 平均单笔收益：-3.317630（basis=round_trip）
- 中位数单笔收益：-5.323383
- 胜率：24.495350%
- 平均盈利：8.949693
- 平均亏损：-7.297416
- 盈亏比：1.226419
- 期望值 expectancy：-3.317630
- 最大单笔盈利：92.665511
- 最大单笔亏损：-69.749608
- 连续亏损次数：32
- 每日平均交易次数：38.760440
- 每日平均 PnL：-64.296390

## 最差 10 天
- 1. 2026-01-18: net_pnl=-178.315071, trades=40.00
- 2. 2026-02-16: net_pnl=-174.267529, trades=34.00
- 3. 2026-03-11: net_pnl=-171.359261, trades=40.00
- 4. 2026-02-28: net_pnl=-169.154399, trades=38.00
- 5. 2026-01-25: net_pnl=-165.201858, trades=40.00
- 6. 2025-05-23: net_pnl=-158.140883, trades=40.00
- 7. 2025-05-12: net_pnl=-154.619521, trades=20.00
- 8. 2025-11-26: net_pnl=-152.640695, trades=40.00
- 9. 2025-05-24: net_pnl=-152.290208, trades=36.00
- 10. 2025-07-29: net_pnl=-151.594337, trades=40.00

## 最好 10 天
- 1. 2025-10-11: net_pnl=175.887190, trades=40.00
- 2. 2025-04-23: net_pnl=125.580904, trades=40.00
- 3. 2025-12-11: net_pnl=87.904778, trades=40.00
- 4. 2025-01-13: net_pnl=76.606535, trades=40.00
- 5. 2025-05-14: net_pnl=61.704776, trades=40.00
- 6. 2026-02-23: net_pnl=52.754325, trades=40.00
- 7. 2026-02-24: net_pnl=51.931578, trades=40.00
- 8. 2025-01-14: net_pnl=47.935469, trades=40.00
- 9. 2025-01-17: net_pnl=46.820447, trades=40.00
- 10. 2026-02-05: net_pnl=39.258180, trades=40.00

## 交易频率分布
- 11-20: days=6, net_pnl=-383.522003, avg_daily_pnl=-63.920334
- 21-40: days=440, net_pnl=-28375.181770, avg_daily_pnl=-64.489049
- 41-80: days=9, net_pnl=-496.153576, avg_daily_pnl=-55.128175

## 输出文件
- attribution_summary.json
- attribution_by_side.csv
- attribution_by_hour.csv
- attribution_by_weekday.csv
- attribution_by_month.csv
- attribution_daily_worst.csv
- attribution_frequency_distribution.csv

## Warning
- trades/orders 中没有可用 exit_reason 或 strategy metadata，跳过 exit_reason 分组
