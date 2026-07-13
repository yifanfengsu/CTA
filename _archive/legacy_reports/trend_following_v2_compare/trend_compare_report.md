# Trend Following V2 跨样本比较

## 核心结论
- stable_candidate_exists=false
- trend_following_v2_failed=false
- overfit_risk=false
- stable_candidate 需要 train/validation/oos no-cost 均为正、OOS 成本后不亏或低频小额成本拖累例外、回撤受控、每段 trade_count>=10。
- 低频成本例外阈值：trade_count<=20 且 abs(oos_net_pnl)<=0.5*oos_no_cost_net_pnl。

## 稳定候选
- 无

## Train 正但 Validation/OOS 不稳定
- 无

## 输出文件
- trend_compare_summary.json
- trend_compare_leaderboard.csv
- trend_compare_report.md
