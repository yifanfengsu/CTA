#!/usr/bin/env python3
"""financial-precision skill — 一次性浮点误差量级诊断（2026-07-13）。

目的：用 precision_toolkit 的 Decimal 路径重算 B2_4h（OKX）全期净 P&L，
与冻结数字 $68,194.8186 对比，把"研究阶段 float64 可用"的依据从间接证明
（b2_4h_pnl_audit 独立重推逐分一致）升级为显式量级确认。
结果写入同目录 DIAGNOSTIC_20260713.md。

只读输入（不写任何研究产出、不修改冻结数字）：
  research/_closed/crypto_perp/trend_b2_4h/reports/trend_baseline_20260611/
      configs/B2_4h_trades.jsonl                       （470 笔成交明细）
  research/_closed/crypto_perp/trend_b2_4h/reports/b2_4h_pnl_audit_20260628/
      audit_summary.json                                （冻结数字与审计基准）
      okx_b2_4h_funding_ledger.jsonl                    （470 笔 / 18,410 次结算）

复现: .venv/bin/python .claude/skills/financial-precision/diagnostic_float_error.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precision_toolkit import amounts_equal, dsum, fee, pnl  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LINE = ROOT / "research/_closed/crypto_perp/trend_b2_4h/reports"
TRADES = LINE / "trend_baseline_20260611/configs/B2_4h_trades.jsonl"
AUDIT = LINE / "b2_4h_pnl_audit_20260628/audit_summary.json"
FUND_LEDGER = LINE / "b2_4h_pnl_audit_20260628/okx_b2_4h_funding_ledger.jsonl"

FROZEN_NET = Decimal("68194.8186")          # audit_summary okx.reconciliation.frozen_net
FEE_TAKER = Decimal("0.0005")               # 引擎 FEE_TAKER（taker 0.05%/边）
CT_VAL = {"BTC": Decimal("0.01"), "ETH": Decimal("0.1"), "SOL": Decimal("1"),
          "LINK": Decimal("1"), "DOGE": Decimal("1000")}   # backtest_mr_5m_compare.CONTRACT_SPECS


def load_jsonl_decimal(path: Path) -> list[dict]:
    """parse_float=Decimal：数字字段按文件中的十进制字面量精确进入 Decimal 域。"""
    with open(path) as f:
        return [json.loads(line, parse_float=Decimal) for line in f if line.strip()]


def main() -> int:
    trades = load_jsonl_decimal(TRADES)
    ledger = load_jsonl_decimal(FUND_LEDGER)
    audit = json.loads(AUDIT.read_text())
    assert len(trades) == 470 and len(ledger) == 470
    n_settle = sum(int(r["n_settlements"]) for r in ledger)

    # ── 路径 A：Decimal 求和引擎逐笔 net（冻结数字的同一构成，聚合层换 Decimal）──
    net_A = dsum(t["net_pnl_usd"] for t in trades)
    diff_A = net_A - FROZEN_NET

    # ── 路径 B：三分量恒等式（gross + fee + funding，各列 Decimal 求和）──────────
    g_B = dsum(t["gross_pnl_usd"] for t in trades)
    f_B = dsum(t["fee_usd"] for t in trades)
    fd_B = dsum(t["funding_usd"] for t in trades)
    net_B = g_B + f_B + fd_B
    diff_B = net_B - FROZEN_NET

    # ── 路径 C：toolkit 全重算（gross/fee 从 entry/exit/size/ctVal 重推）─────────
    g_C = Decimal(0)
    fee_C = Decimal(0)
    max_dg = Decimal(0)
    max_df = Decimal(0)
    for t in trades:
        qty = Decimal(t["size"]) * CT_VAL[t["symbol"]]
        g = pnl(t["entry_price"], t["exit_price"], qty, t["side"])
        fe = (fee(t["entry_price"] * qty, FEE_TAKER)
              + fee(t["exit_price"] * qty, FEE_TAKER))
        g_C += g
        fee_C += fe
        max_dg = max(max_dg, abs(g - t["gross_pnl_usd"]))
        max_df = max(max_df, abs(-fe - t["fee_usd"]))
    net_C = g_C - fee_C + fd_B
    diff_C = net_C - FROZEN_NET

    # ── 路径 D：float64 累加误差隔离（同一 1,410 个分量，float 裸加 vs Decimal）──
    comps = ([t["gross_pnl_usd"] for t in trades]
             + [t["fee_usd"] for t in trades]
             + [t["funding_usd"] for t in trades])
    dec_sum = dsum(comps)
    float_naive = 0.0
    for c in comps:
        float_naive += float(c)
    float_fsum = math.fsum(float(c) for c in comps)
    err_naive = Decimal(repr(float_naive)) - dec_sum
    err_fsum = Decimal(repr(float_fsum)) - dec_sum

    # ── funding 独立路径交叉（审计 ledger：engine vs independent，Decimal 求和）──
    fd_engine = dsum(r["engine_funding_usd"] for r in ledger)
    fd_indep = dsum(r["indep_funding_usd"] for r in ledger)

    # ── 性能：同一 470 笔 gross+fee 重算，float vs Decimal ───────────────────────
    ft = [(float(t["entry_price"]), float(t["exit_price"]),
           float(Decimal(t["size"]) * CT_VAL[t["symbol"]]),
           1.0 if t["side"] == "long" else -1.0) for t in trades]
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        s = 0.0
        for e, x, q, sd in ft:
            s += (x - e) * q * sd - (e * q + x * q) * 0.0005
    t_float = (time.perf_counter() - t0) / reps
    dt = [(t["entry_price"], t["exit_price"],
           Decimal(t["size"]) * CT_VAL[t["symbol"]],
           Decimal(1) if t["side"] == "long" else Decimal(-1)) for t in trades]
    t0 = time.perf_counter()
    for _ in range(reps):
        sD = Decimal(0)
        for e, x, q, sd in dt:
            sD += (x - e) * q * sd - (e * q + x * q) * FEE_TAKER
    t_dec = (time.perf_counter() - t0) / reps

    # ── 判定与输出 ────────────────────────────────────────────────────────────────
    audit_frozen = Decimal(str(audit["okx"]["reconciliation"]["frozen_net"]))
    print(f"trades={len(trades)}  settlements={n_settle}  frozen_net={FROZEN_NET}")
    print(f"audit_summary frozen_net 一致: {audit_frozen == FROZEN_NET}")
    print(f"[A] Decimal求和逐笔net      = {net_A}   diff={diff_A}")
    print(f"[B] 三分量 g+f+fd           = {net_B}   diff={diff_B}"
          f"   (g={g_B} f={f_B} fd={fd_B})")
    print(f"[C] toolkit全重算           = {net_C}   diff={diff_C}")
    print(f"    per-trade max|Δgross|={max_dg}  max|Δfee|={max_df}")
    print(f"[D] float64累加误差: naive={err_naive}  fsum={err_fsum}")
    print(f"[F] funding engine={fd_engine}  indep={fd_indep}  Δ={fd_indep - fd_engine}")
    print(f"[P] per-pass: float={t_float*1e3:.3f}ms  Decimal={t_dec*1e3:.3f}ms"
          f"  ratio={t_dec / t_float:.1f}x")

    stop = abs(diff_A) > Decimal("0.01")
    print(f"\n|diff_A| <= $0.01: {not stop}"
          f"   (amounts_equal@0.01: {amounts_equal(net_A, FROZEN_NET, '0.01')})")
    if stop:
        print("⚠️ Decimal 重算与冻结数字差额 > $0.01 —— 按任务纪律停下，"
              "不修改冻结数字，等待单独调查。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
