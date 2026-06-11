# VPS 实盘关停确认清单（需在 VPS 上人工执行）

> 背景：CLAUDE.md 记载 MR-5m 实盘脚本"已关停归档"，但这只是文档声明——
> 本地仓库无法验证 VPS 进程状态（盘点发现：`deploy/systemd/` 为空目录，
> 部署配置在 VPS 本机；`restart_mr5m.sh` 指向 `/run-project/vnpy_strategy_test/CTA`）。
> 本清单产出于漏洞修补任务（`reports/regime/vulnerability_patch_20260611/`），
> 修补任务本身不连 VPS。请在 VPS 上照单执行，并把结果回填到本文档第 5 节。

## 1. 确认进程状态

```bash
ps -ef | grep -v grep | grep run_mr_5m
```

- 无输出 → 进程已停，跳到第 3 步。
- 有输出 → 记录 PID 与启动时间，执行第 2 步。

## 2. 若仍在跑：按既有关停流程

按项目既有的关停提示词流程执行（顺序不可调换）：

1. **最后对账**：以 OKX bills 为准对账当日成交与手续费；
2. **平残仓**：确认两个账号（DYN/FLAT）无持仓、无挂单
   （可用 `scripts/close_positions.py` 紧急平仓；逐币种核对 positions 接口为空）；
3. **停进程**：优雅停止（脚本已有 shutdown 处理），确认进程退出；
4. **归档日志**：将 `trade_log_5m_*.jsonl`、运行日志打包归档
   （建议同步一份回本仓库 `reports/archive/vps_logs_<日期>/`）。

## 3. 确认无自启残留

```bash
crontab -l                                  # 无 run_mr_5m / restart_mr5m 条目
ls /etc/systemd/system/ | grep -i mr        # 无相关 unit
systemctl list-units --all | grep -i mr5m   # 无 active/failed unit
screen -ls; tmux ls 2>/dev/null             # 无残留会话
ls /run-project/vnpy_strategy_test/CTA/restart_mr5m.sh  # 确认该脚本不再被任何机制调用
```

任一处有残留 → 移除后重查。

## 4. 记录最后一条 trade_log 时间戳

```bash
tail -1 /run-project/vnpy_strategy_test/CTA/trade_log_5m_*.jsonl | tail -1
```

把最后一条记录的 `time` 字段抄到下面第 5 节。

## 5. 确认结果回填（执行人填写）

| 项 | 结果 |
|---|---|
| 执行日期（UTC） | ______ |
| 进程状态（第 1 步） | ______ |
| 残仓/挂单清零（第 2 步，如适用） | ______ |
| 自启残留检查（第 3 步） | ______ |
| 最后一条 trade_log 时间戳（第 4 步） | ______ |
| 日志归档位置 | ______ |

## 6. 完成后的文档闭环

在 `CLAUDE.md` 项目节，将：

> `scripts/run_mr_5m_direct.py`：MR-5m 实盘脚本，**已关停归档**。

升级为：

> `scripts/run_mr_5m_direct.py`：MR-5m 实盘脚本，**已于 [日期] 在 VPS 确认关停归档**。

并将本文档第 5 节填写完整后一并提交。
