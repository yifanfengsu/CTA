#!/usr/bin/env python3
"""BTC 日收益 GARCH 测绘诊断（volatility-modeling skill 配套，一次性）。

定位：第 0 站"出生地检查"波动区的参考件——本市场波动结构的 GARCH 测绘。
**不是研究**：不产生任何交易/立项结论；任何策略研究须走 PIPELINE 九站。

数据：`data/binance_vision/BTCUSDT/BTCUSDT-1m-*.zip`（Binance Vision UM 永续
1m 月度文件，本仓库既有资产，**只读**）→ UTC 日界重采样取每日最后一根 close
→ 对数日收益。年化因子 365（加密 24/7，与 vrp_atm RV 口径一致）。

用法：.venv/bin/python .claude/skills/volatility-modeling/diagnostic_btc_garch.py
输出：本目录 DIAGNOSTIC_BTC_<YYYYMMDD>.md（唯一写入物；不触任何数据库）。
"""

from __future__ import annotations

import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SKILL_DIR = Path(__file__).resolve().parent
REPO = SKILL_DIR.parents[2]
DATA_DIR = REPO / "data" / "binance_vision" / "BTCUSDT"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "count", "taker_buy_volume",
              "taker_buy_quote_volume", "ignore"]

sys.path.insert(0, str(SKILL_DIR))
from garch_toolkit import (arch_effect_test, fit_garch, persistence_report,  # noqa: E402
                           forecast_vol, residual_diagnostics, ewma_sigma)


def load_daily_close() -> pd.Series:
    """1m 月度 zip（只读）→ UTC 日界日线 close。头行/时间戳单位逐文件自适应。"""
    zips = sorted(DATA_DIR.glob("BTCUSDT-1m-*.zip"))
    if not zips:
        raise FileNotFoundError(f"未找到 1m 月度文件：{DATA_DIR}")
    frames = []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            name = z.namelist()[0]
            with z.open(name) as fh:
                first = fh.readline().decode()
            header = 0 if first.startswith("open_time") else None
            with z.open(name) as fh:
                df = pd.read_csv(fh, header=header, names=KLINE_COLS,
                                 usecols=["open_time", "close"])
        ot = pd.to_numeric(df["open_time"], errors="coerce")
        unit = "us" if float(ot.iloc[0]) > 1e14 else "ms"  # Binance 2025+ 部分数据集转 µs
        idx = pd.to_datetime(ot.to_numpy(), unit=unit, utc=True)
        frames.append(pd.Series(pd.to_numeric(df["close"]).to_numpy(),
                                index=idx, name="close"))
    s = pd.concat(frames).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    daily = s.resample("1D").last().dropna()
    return daily


def main() -> int:
    print("=" * 70)
    print("数据环境：Binance Vision UM 永续 1m 月度文件（本仓库 data/ 既有资产）")
    print(f"路径：{DATA_DIR}（只读；本脚本唯一写入物 = 本目录诊断 md）")
    print("=" * 70)

    daily = load_daily_close()
    logret = np.log(daily / daily.shift(1)).dropna()
    r = logret.to_numpy()
    t0, t1 = daily.index[0].date(), daily.index[-1].date()
    print(f"日线 {len(daily)} 根，日收益 {len(r)} 个，区间 {t0} → {t1}（UTC 日界）")

    # ① 平稳性（ADF on returns）
    from statsmodels.tsa.stattools import adfuller
    adf_stat, adf_p, *_ = adfuller(r, autolag="AIC")

    # ② ARCH 效应
    arch_t = arch_effect_test(r, lags=10)

    # ③ 分布形态
    from scipy import stats as sps
    skew, kurt = float(sps.skew(r)), float(sps.kurtosis(r, fisher=False))

    # ④ t 分布 GARCH(1,1) 拟合（Constant mean：μ≈0 让数据自证而非预设）
    fit = fit_garch(r, dist="t", model="GARCH", mean="Constant")
    rep = persistence_report(fit, periods_per_year=365)
    mu, mu_se = fit.params.get("mu", float("nan")), fit.std_err.get("mu", float("nan"))
    nu = fit.params.get("nu", float("nan"))

    # ⑤ 残差诊断
    diag = residual_diagnostics(fit, lags=10)

    # ⑥ 预测路径（向无条件方差回归的形状展示）
    fc = forecast_vol(fit, horizon=30, periods_per_year=365)

    # ⑦ EWMA 对照（RiskMetrics λ=0.94）
    sig_ewma = ewma_sigma(r, lam=0.94)

    # ⑧ 断点敏感性提示：前后半样本分别拟合（只报数字，不下结论）
    half = len(r) // 2
    rep_a = persistence_report(fit_garch(r[:half], dist="t", mean="Constant"), 365)
    rep_b = persistence_report(fit_garch(r[half:], dist="t", mean="Constant"), 365)
    mid_date = logret.index[half].date()

    at_boundary = not np.isfinite(rep["uncond_var_per_period"])

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = SKILL_DIR / f"DIAGNOSTIC_BTC_{today}.md"

    cond_vol_now = float(fit.raw.conditional_volatility[-1]) / 100.0  # 还原小数标度
    lines = [
        f"# DIAGNOSTIC — BTC 日收益 GARCH 测绘（{datetime.now(timezone.utc).date()}）",
        "",
        "> **定位**：第 0 站'出生地检查'波动区参考件（本市场波动结构测绘）。",
        "> **不是研究产物，不含任何交易/立项结论**；策略研究须走 PIPELINE 九站。",
        "> 复现：`.venv/bin/python .claude/skills/volatility-modeling/diagnostic_btc_garch.py`",
        "",
        "## 数据与口径（显式声明）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 数据源 | `data/binance_vision/BTCUSDT/BTCUSDT-1m-*.zip`（Binance Vision UM 永续，只读） |",
        f"| 样本区间 | **{t0} → {t1}**（UTC 日界，{len(daily)} 日线 / {len(r)} 日收益） |",
        "| 收益口径 | UTC 日界 close-to-close 对数收益 |",
        "| 年化因子 | **√365**（加密 24/7；与 vrp_atm RV 口径一致） |",
        "| 量纲约定 | 下文一切数字显式标注 方差 vs vol、日 vs 年化 |",
        "",
        "**样本内已知结构断点提示**（分段数字见末节；本诊断不做断点正式检验）：",
        "COVID '312'（2020-03）、LUNA 崩盘（2022-05）、FTX 崩盘（2022-11）均在样本内，",
        "此类断点会推高全样本 persistence 估计（见 SKILL.md 失效模式③）。",
        "",
        "## ① 前置检验",
        "",
        "| 检验 | 统计量 | p | 读法 |",
        "|---|---|---|---|",
        f"| ADF（收益平稳性） | {adf_stat:.2f} | {adf_p:.2e} | p≪0.05 → 拒绝单位根，收益平稳 |",
        f"| ARCH-LM（lag 10） | {arch_t['arch_lm_stat']:.1f} | {arch_t['arch_lm_p']:.2e} | p≪0.05 → 条件异方差显著 |",
        f"| Ljung-Box(r², lag 10) | {arch_t['ljungbox_sq_stat']:.1f} | {arch_t['ljungbox_sq_p']:.2e} | 同上，双检验合取 |",
        f"| 偏度 / 峰度 | {skew:.2f} / {kurt:.1f} | — | 峰度≫3 → 肥尾，禁用正态 innovation |",
        "",
        f"**ARCH 效应判定：{'显著' if arch_t['has_arch_effect'] else '不显著'}**——"
        "波动聚集是本市场已确认结构（volatility_event 线 5/5×2 样本），此处为 GARCH 口径的再确认。",
        "",
        "## ② GARCH(1,1)-t 拟合（Constant mean）",
        "",
        "| 参数 | 估计 | SE | 说明 |",
        "|---|---|---|---|",
        f"| μ（日均值） | {mu:.2e} | {mu_se:.2e} | \\|μ/SE\\|={abs(mu / mu_se):.2f} → 均值方程无方向信息（与方向区结论一致） |",
        f"| ω | {rep['omega']:.3e} | {fit.std_err['omega']:.1e} | 方差截距（日方差量纲，小数²） |",
        f"| α | {rep['alpha']:.4f} | {fit.std_err['alpha[1]']:.4f} | 冲击反应 |",
        f"| β | {rep['beta']:.4f} | {fit.std_err['beta[1]']:.4f} | 记忆 |",
        f"| ν（t 自由度） | {nu:.2f} | {fit.std_err.get('nu', float('nan')):.2f} | ν 小 → 肥尾确认，正态设定会低估尾部 |",
        "",
        "| 派生量 | 值 | 量纲 |",
        "|---|---|---|",
        f"| **persistence α+β** | **{rep['persistence']:.4f} ± {rep['persistence_se']:.4f}** | 无量纲（±为 delta-method SE） |",
        f"| **半衰期** | **{rep['half_life_bars']:.1f} 天** | ln(0.5)/ln(α+β)，日 bar |",
        f"| 无条件方差 | {rep['uncond_var_per_period']:.3e} | **日方差**（小数²） |",
        f"| 无条件日 vol | {rep['uncond_vol_per_period']:.4%} | 日波动率 |",
        f"| **无条件年化 vol** | **{rep['uncond_vol_annual']:.1%}** | 年化（×√365） |",
        f"| 拟合末日条件 vol | {cond_vol_now:.4%}（日） / {cond_vol_now * 365 ** 0.5:.1%}（年化） | 时变——GARCH 的输出本体 |",
        "",
    ]
    if at_boundary:
        lines += [
            "**边界读法（重要）**：全样本 α+β 估计撞到 **1 边界**（IGARCH 边界）——",
            "无条件方差与半衰期在此**不可识别**（表中 inf 即此义），边界处的",
            "delta-method SE 也不可按常规解读。这不是'BTC 波动率非平稳'的结论，",
            "而是 SKILL.md 失效模式③的现场演示：样本含 312/LUNA/FTX 结构断点，",
            "**断点伪装成极高 persistence**。处置动作 = 分段拟合（见⑤）：",
            f"后半段（{mid_date} → {t1}，无同量级断点）α+β={rep_b['persistence']:.4f}、",
            f"半衰期 {rep_b['half_life_bars']:.1f} 天、年化无条件 vol {rep_b['uncond_vol_annual']:.1%}",
            "——作为近段波动结构的测绘参考值。",
            "",
        ]
    lines += [
        "## ③ 残差诊断（模型是否吃干净条件异方差）",
        "",
        "| 检验（标准化残差） | p / 值 | 合格线 | 结果 |",
        "|---|---|---|---|",
        f"| Ljung-Box 水平 | {diag['ljungbox_level_p']:.3f} | >0.05 | {'过' if diag['ljungbox_level_p'] > 0.05 else '不过（残差仍有线性自相关）'} |",
        f"| Ljung-Box 平方 | {diag['ljungbox_sq_p']:.3f} | >0.05 | {'过' if diag['ljungbox_sq_p'] > 0.05 else '不过'} |",
        f"| 剩余 ARCH-LM | {diag['arch_lm_p']:.3f} | >0.05 | {'过' if diag['arch_lm_p'] > 0.05 else '不过'} |",
        f"| 偏度 / 峰度 | {diag['skew']:.2f} / {diag['kurtosis']:.1f} | — | t 设定下峰度>3 属预期（t 的本意） |",
        "",
        f"**剩余 ARCH 效应：{'无（模型设定充分）' if diag['clean'] else '有——GARCH(1,1) 未吃净，解读时注意'}**",
        "",
        "## ④ 30 日条件 vol 预测路径（均值回归形状）",
        "",
        "| h（日） | 条件日 vol | 年化 |",
        "|---|---|---|",
    ]
    for h in (1, 5, 10, 20, 30):
        row = fc[fc["h"] == h].iloc[0]
        lines.append(f"| {h} | {row['cond_vol']:.4%} | {row['ann_vol']:.1%} |")
    if at_boundary:
        path_note = ("α+β=1（边界）时预测路径**不**均值回归——按 σ²_{t+h}=σ²_{t+1}+(h−1)ω "
                     "线性攀升，上表即此形态（边界行为；有限 persistence 下才是向无条件方差"
                     "按 (α+β)^h 回归的正常形态）；")
    else:
        path_note = (f"路径自当前条件方差向无条件方差（日 {rep['uncond_var_per_period']:.2e}）"
                     "按 (α+β)^h 回归；")
    lines += [
        "",
        path_note,
        f"EWMA(λ=0.94) 对照末日 σ={float(sig_ewma[-1]):.4%}（日）——EWMA 无回归项，",
        "多步预测走平，长 horizon 与 GARCH 系统性分叉（SKILL.md 失效模式⑦）。",
        "",
        "## ⑤ 分段敏感性（断点提示，非正式检验）",
        "",
        "| 段 | 区间 | α+β | 半衰期（天） | 年化无条件 vol |",
        "|---|---|---|---|---|",
        f"| 全样本 | {t0} → {t1} | {rep['persistence']:.4f} | {rep['half_life_bars']:.1f} | {rep['uncond_vol_annual']:.1%} |",
        f"| 前半 | {t0} → {mid_date} | {rep_a['persistence']:.4f} | {rep_a['half_life_bars']:.1f} | {rep_a['uncond_vol_annual']:.1%} |",
        f"| 后半 | {mid_date} → {t1} | {rep_b['persistence']:.4f} | {rep_b['half_life_bars']:.1f} | {rep_b['uncond_vol_annual']:.1%} |",
        "",
        "前半含 312/LUNA/FTX 三断点。分段数字仅提示参数的时变性——任何跨段使用",
        "全样本参数的研究，须自行做断点处理（分段拟合/断点检验）。",
        "",
        "---",
        "*工具：`garch_toolkit.py`（冒烟 19/19 PASS，参数回收+双向报警+量纲换算已自证）；*",
        f"*arch 8.0.0 / statsmodels 0.14.6；生成于 {datetime.now(timezone.utc).isoformat(timespec='seconds')}。*",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n诊断写入：{out}")
    print(f"关键数字：全样本 α+β={rep['persistence']:.4f}"
          f"{'（撞 1 边界，无条件量不可识别）' if at_boundary else f'±{rep['persistence_se']:.4f}'}，"
          f"ARCH-LM p={arch_t['arch_lm_p']:.2e}，ν̂={nu:.2f}，残差干净={diag['clean']}")
    print(f"近段参考（{mid_date}→{t1}）：α+β={rep_b['persistence']:.4f}，"
          f"半衰期 {rep_b['half_life_bars']:.1f} 天，"
          f"年化无条件 vol {rep_b['uncond_vol_annual']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
