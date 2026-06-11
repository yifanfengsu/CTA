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
- 总净利润：-4402.943180
- 平均单笔收益：-0.499313（basis=round_trip）
- 中位数单笔收益：-2.362000
- 胜率：34.769789%
- 平均盈利：8.725364
- 平均亏损：-5.422961
- 盈亏比：1.608967
- 期望值 expectancy：-0.499313
- 最大单笔盈利：95.226750
- 最大单笔亏损：-66.377520
- 连续亏损次数：25
- 每日平均交易次数：38.760440
- 每日平均 PnL：-9.676798

## 最差 10 天
- 1. 2026-02-16: net_pnl=-126.307020, trades=34.00
- 2. 2025-05-12: net_pnl=-124.558740, trades=20.00
- 3. 2026-01-18: net_pnl=-119.392160, trades=40.00
- 4. 2026-02-11: net_pnl=-117.015420, trades=34.00
- 5. 2026-03-11: net_pnl=-115.778680, trades=40.00
- 6. 2025-06-08: net_pnl=-108.127210, trades=26.00
- 7. 2026-02-28: net_pnl=-106.613100, trades=38.00
- 8. 2025-03-12: net_pnl=-106.597720, trades=36.00
- 9. 2025-07-12: net_pnl=-105.990070, trades=36.00
- 10. 2025-09-28: net_pnl=-103.604960, trades=24.00

## 最好 10 天
- 1. 2025-10-11: net_pnl=224.945450, trades=40.00
- 2. 2025-04-23: net_pnl=180.978200, trades=40.00
- 3. 2025-12-11: net_pnl=153.394790, trades=40.00
- 4. 2025-01-13: net_pnl=126.658500, trades=40.00
- 5. 2025-05-14: net_pnl=125.979740, trades=40.00
- 6. 2026-02-23: net_pnl=119.738240, trades=40.00
- 7. 2026-02-24: net_pnl=117.535540, trades=40.00
- 8. 2025-10-09: net_pnl=104.902260, trades=40.00
- 9. 2025-01-17: net_pnl=103.366650, trades=40.00
- 10. 2025-01-14: net_pnl=96.691950, trades=40.00

## 交易频率分布
- 11-20: days=6, net_pnl=-241.658690, avg_daily_pnl=-40.276448
- 21-40: days=440, net_pnl=-4185.366640, avg_daily_pnl=-9.512197
- 41-80: days=9, net_pnl=24.082150, avg_daily_pnl=2.675794

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
