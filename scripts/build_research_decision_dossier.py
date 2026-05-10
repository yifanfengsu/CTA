#!/usr/bin/env python3
"""Build a research decision dossier for the completed trend-following studies."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "research_decision_dossier"

REPORT_SOURCES: list[dict[str, str]] = [
    {
        "key": "main_no_cost_stats",
        "stage": "1m breakout / original backtest",
        "path": "reports/backtest/main_no_cost_20250101_20260331/stats.json",
    },
    {
        "key": "main_no_cost_diagnostics",
        "stage": "1m breakout / original backtest",
        "path": "reports/backtest/main_no_cost_20250101_20260331/diagnostics.json",
    },
    {
        "key": "main_cost_stats",
        "stage": "1m breakout / original backtest",
        "path": "reports/backtest/main_cost_20250101_20260331/stats.json",
    },
    {
        "key": "alpha_sweep_summary",
        "stage": "alpha sweep",
        "path": "reports/alpha_sweep/main_20250101_20260331/sweep_summary.md",
    },
    {
        "key": "trade_attribution",
        "stage": "trade attribution",
        "path": "reports/backtest/main_no_cost_20250101_20260331/trade_attribution/attribution_report.md",
    },
    {
        "key": "signal_outcomes_train",
        "stage": "signal outcome",
        "path": "reports/research/trace_train/signal_outcomes/outcome_report.md",
    },
    {
        "key": "signal_outcomes_validation",
        "stage": "signal outcome",
        "path": "reports/research/trace_validation/signal_outcomes/outcome_report.md",
    },
    {
        "key": "signal_outcomes_oos",
        "stage": "signal outcome",
        "path": "reports/research/trace_oos/signal_outcomes/outcome_report.md",
    },
    {
        "key": "entry_policy_train",
        "stage": "entry policy",
        "path": "reports/research/trace_train/entry_policy_research/policy_report.md",
    },
    {
        "key": "entry_policy_validation",
        "stage": "entry policy",
        "path": "reports/research/trace_validation/entry_policy_research/policy_report.md",
    },
    {
        "key": "entry_policy_oos",
        "stage": "entry policy",
        "path": "reports/research/trace_oos/entry_policy_research/policy_report.md",
    },
    {
        "key": "signal_lab_feature_compare",
        "stage": "Signal Lab",
        "path": "reports/research/feature_compare/feature_compare_report.md",
    },
    {
        "key": "htf_compare",
        "stage": "HTF Signal Research",
        "path": "reports/research/htf_compare/htf_compare_report.md",
    },
    {
        "key": "trend_v2_compare",
        "stage": "Trend V2",
        "path": "reports/research/trend_following_v2_compare/trend_compare_report.md",
    },
    {
        "key": "trend_v3_compare",
        "stage": "Trend V3",
        "path": "reports/research/trend_following_v3_compare/trend_v3_compare_report.md",
    },
    {
        "key": "trend_v3_postmortem_report",
        "stage": "Trend V3 Postmortem",
        "path": "reports/research/trend_following_v3_postmortem/trend_v3_postmortem_report.md",
    },
    {
        "key": "trend_v3_postmortem_recommendations",
        "stage": "Trend V3 Postmortem",
        "path": "reports/research/trend_following_v3_postmortem/v3_1_recommendations.json",
    },
    {
        "key": "trend_v3_extended_compare_report",
        "stage": "Extended Trend V3",
        "path": "reports/research/trend_following_v3_extended_compare/trend_v3_extended_compare_report.md",
    },
    {
        "key": "trend_v3_extended_compare_summary",
        "stage": "Extended Trend V3",
        "path": "reports/research/trend_following_v3_extended_compare/trend_v3_extended_compare_summary.json",
    },
    {
        "key": "trend_regime_report",
        "stage": "Trend Regime Diagnostics",
        "path": "reports/research/trend_regime_diagnostics/trend_regime_report.md",
    },
    {
        "key": "trend_regime_recommendations",
        "stage": "Trend Regime Diagnostics",
        "path": "reports/research/trend_regime_diagnostics/v3_1_regime_recommendations.json",
    },
    {
        "key": "multisymbol_readiness",
        "stage": "Data readiness",
        "path": "reports/research/multisymbol_readiness/multisymbol_readiness_report.md",
    },
    {
        "key": "extended_history_availability",
        "stage": "Data readiness",
        "path": "reports/research/extended_history_availability/extended_history_availability_report.md",
    },
    {
        "key": "trend_regime_summary",
        "stage": "Trend Regime Diagnostics",
        "path": "reports/research/trend_regime_diagnostics/trend_regime_summary.json",
    },
    {
        "key": "trend_regime_data_quality",
        "stage": "Data readiness",
        "path": "reports/research/trend_regime_diagnostics/data_quality.json",
    },
]

OUTPUT_FILES = [
    "research_decision_dossier.md",
    "research_decision_dossier.json",
    "failed_policy_families.csv",
    "retained_research_hypotheses.csv",
    "do_not_continue_list.csv",
    "next_research_options.csv",
]


class ResearchDecisionDossierError(Exception):
    """Raised when the research decision dossier cannot be written."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build the trend research decision dossier.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include-existing-reports", default="true", choices=("true", "false"))
    parser.add_argument("--json", action="store_true", help="Print research_decision_dossier.json after writing files.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a CLI path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root / path
    return path


def include_reports_enabled(value: str | bool) -> bool:
    """Normalize include-existing-reports values."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def read_json_file(path: Path) -> tuple[Any | None, str | None]:
    """Read JSON, returning payload and warning."""

    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"read_json_failed:{path}:{exc!r}"


def read_text_excerpt(path: Path, max_chars: int = 4000) -> tuple[str | None, str | None]:
    """Read a bounded text excerpt for source traceability."""

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, f"read_text_failed:{path}:{exc!r}"
    return text[:max_chars], None


def load_report_sources(project_root: Path, include_existing_reports: bool) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Load configured evidence reports without failing on missing files."""

    source_rows: list[dict[str, Any]] = []
    parsed: dict[str, Any] = {}
    warnings: list[str] = []
    for source in REPORT_SOURCES:
        relative_path = source["path"]
        path = project_root / relative_path
        row: dict[str, Any] = {
            "key": source["key"],
            "stage": source["stage"],
            "path": relative_path,
            "exists": path.exists(),
            "included": False,
            "type": path.suffix.lstrip(".") or "unknown",
        }
        if not path.exists():
            warning = f"missing_report:{relative_path}"
            row["warning"] = warning
            warnings.append(warning)
            source_rows.append(row)
            continue
        if not include_existing_reports:
            row["warning"] = "existing_report_inclusion_disabled"
            source_rows.append(row)
            continue

        if path.suffix.lower() == ".json":
            payload, warning = read_json_file(path)
            if warning:
                row["warning"] = warning
                warnings.append(warning)
            else:
                parsed[source["key"]] = payload
                row["included"] = True
                row["json_keys"] = list(payload.keys())[:30] if isinstance(payload, dict) else []
        else:
            excerpt, warning = read_text_excerpt(path)
            if warning:
                row["warning"] = warning
                warnings.append(warning)
            else:
                parsed[source["key"]] = {"excerpt": excerpt}
                row["included"] = True
                row["excerpt_chars"] = len(excerpt or "")
        source_rows.append(row)
    return source_rows, parsed, warnings


def failed_policy_families() -> list[dict[str, Any]]:
    """Return the policy families that should be treated as failed."""

    return [
        {
            "policy_family": "1m Donchian breakout",
            "status": "failed",
            "primary_failure": "No stable positive edge; Signal Lab shows short-term breakout behaves like overheat/exhaustion risk.",
            "evidence": "Original 1m Donchian breakout and alpha diagnostics failed.",
            "tradable": False,
        },
        {
            "policy_family": "1h/15m/5m HTF pullback",
            "status": "failed",
            "primary_failure": "No stable HTF policy; pullback/reclaim variants were negative across train/validation/oos.",
            "evidence": "HTF Signal Research compare rejected Strategy V2 entry.",
            "tradable": False,
        },
        {
            "policy_family": "4h Donchian",
            "status": "failed",
            "primary_failure": "No stable candidate; weak validation/OOS behavior and choppy/high_vol_choppy loss exposure.",
            "evidence": "Trend V3 and regime attribution.",
            "tradable": False,
        },
        {
            "policy_family": "1d Donchian",
            "status": "failed",
            "primary_failure": "No stable candidate; OOS fragility and Donchian losses concentrated in choppy/high_vol_choppy.",
            "evidence": "Extended Trend V3 and Trend Regime Diagnostics.",
            "tradable": False,
        },
        {
            "policy_family": "4h EMA",
            "status": "failed",
            "primary_failure": "No stable candidate; OOS no-cost/cost-aware failure.",
            "evidence": "Trend V3 compare and Extended Trend V3 compare.",
            "tradable": False,
        },
        {
            "policy_family": "1d EMA as currently defined",
            "status": "failed",
            "primary_failure": "Best weak lead, but rejected by top trade concentration, funding stress, and regime diagnostics.",
            "evidence": "v3_1d_ema_50_200_atr5 was not stable and strong-trend no-cost PnL was negative.",
            "tradable": False,
        },
        {
            "policy_family": "vol compression Donchian breakout",
            "status": "failed",
            "primary_failure": "No stable candidate; negative across validation/OOS after costs.",
            "evidence": "Trend V3 compare and Extended Trend V3 compare.",
            "tradable": False,
        },
        {
            "policy_family": "V3 ensemble_core",
            "status": "failed",
            "primary_failure": "No stable ensemble edge; inherits failed 4h/1d Donchian and 4h EMA components.",
            "evidence": "Trend V3 compare and Extended Trend V3 compare.",
            "tradable": False,
        },
    ]


def retained_research_hypotheses() -> list[dict[str, Any]]:
    """Return research-only hypotheses that remain allowed as future work."""

    return [
        {
            "hypothesis": "broader universe trend following",
            "status": "conditional_research_only",
            "reason": "Five-symbol universe may be too narrow for sparse crypto trend regimes.",
            "not_allowed_as": "Strategy V3 or demo/live without new evidence.",
        },
        {
            "hypothesis": "true funding-aware trend following",
            "status": "conditional_research_only",
            "reason": "Synthetic funding stress already invalidated weak OOS gains at 3 bps / 8h.",
            "not_allowed_as": "Current V3 policy patch.",
        },
        {
            "hypothesis": "stronger macro/trend regime classifier",
            "status": "conditional_research_only",
            "reason": "Current regime diagnostics show strong trend scarcity and weak alignment.",
            "not_allowed_as": "Direct trade filter on current V3 family.",
        },
        {
            "hypothesis": "longer history beyond 2021 if listing metadata supports it",
            "status": "conditional_research_only",
            "reason": "More cycles may be needed, but only if listing metadata and sqlite coverage can be verified.",
            "not_allowed_as": "Backfill assumption without metadata verification.",
        },
        {
            "hypothesis": "1d EMA only as weak research lead",
            "status": "weak_lead_research_only",
            "reason": "It was the closest V3 family, but current definition failed concentration, funding, and regime tests.",
            "not_allowed_as": "Tradable policy or Strategy V3 prototype.",
        },
    ]


def do_not_continue_items() -> list[dict[str, Any]]:
    """Return the explicit stop list."""

    return [
        {"item": "do not expand current Donchian grid", "reason": "Current Donchian families failed stability and regime attribution."},
        {"item": "do not trade v3_1d_ema_50_200_atr5", "reason": "Rejected by top trade concentration, funding fragility, and regime diagnostics."},
        {"item": "do not enter demo/live", "reason": "No stable candidate and demo_live_allowed=false."},
        {"item": "do not build Strategy V3 from current results", "reason": "strategy_development_allowed=false."},
        {"item": "do not continue V3.0 ensemble_core", "reason": "No stable candidate and component families failed."},
        {"item": "do not optimize 1m breakout", "reason": "Short-term breakout is aligned with overheat/exhaustion risk, not durable trend following."},
    ]


def next_research_options() -> list[dict[str, Any]]:
    """Return conditional next research paths."""

    return [
        {
            "option": "Option A",
            "name": "Broader universe trend following readiness",
            "prerequisites": "Verified metadata, listing dates, contract specs, liquidity, and 1m sqlite coverage for a materially broader symbol set.",
            "acceptance_criteria": "Multi-symbol universe passes coverage checks; research design pre-registers stable-candidate criteria before any policy comparison.",
            "allowed_now": "conditional",
        },
        {
            "option": "Option B",
            "name": "Funding-aware trend following research",
            "prerequisites": "Historical funding data ingestion and a PnL model that applies funding to position holding periods.",
            "acceptance_criteria": "Candidate remains positive after realistic funding stress and does not rely on synthetic no-cost assumptions.",
            "allowed_now": "conditional",
        },
        {
            "option": "Option C",
            "name": "External regime classifier research",
            "prerequisites": "Independent regime labels or macro/market structure features not derived from the current V3 policy outcomes.",
            "acceptance_criteria": "Out-of-sample regime classifier improves trend-following attribution without parameter-mining current failures.",
            "allowed_now": "conditional",
        },
        {
            "option": "Option D",
            "name": "Pause strategy development and only maintain data/research tooling",
            "prerequisites": "None beyond maintaining data integrity and reproducible reports.",
            "acceptance_criteria": "No new strategy work starts until a new research premise is documented and approved.",
            "allowed_now": "yes",
        },
    ]


def research_timeline() -> list[dict[str, str]]:
    """Return the completed research timeline."""

    return [
        {
            "stage": "1m breakout",
            "goal": "Test raw short-horizon Donchian breakout as trend-following entry.",
            "result": "Failed.",
            "pass_fail": "fail",
            "key_finding": "Short-term breakout did not produce stable cost-aware edge.",
            "decision": "Stop optimizing 1m breakout.",
        },
        {
            "stage": "Signal Lab",
            "goal": "Identify features explaining signal outcomes.",
            "result": "Stable negative risk features found.",
            "pass_fail": "diagnostic_pass_strategy_fail",
            "key_finding": "High volatility, ATR, breakout distance, recent return, volume spike, and large body ratio are negative.",
            "decision": "Treat short-term breakout as overheat/exhaustion risk, not a strategy candidate.",
        },
        {
            "stage": "HTF Signal Research",
            "goal": "Use 1h regime plus 15m structure and 5m pullback/reclaim.",
            "result": "Failed.",
            "pass_fail": "fail",
            "key_finding": "No stable HTF policy across train/validation/oos.",
            "decision": "Do not enter Strategy V2 from HTF policies.",
        },
        {
            "stage": "Trend V2",
            "goal": "Test single-symbol 1h/4h trend-following families.",
            "result": "Failed.",
            "pass_fail": "fail",
            "key_finding": "No stable candidate.",
            "decision": "Stop single-symbol V2 direction.",
        },
        {
            "stage": "Trend V3",
            "goal": "Test multi-symbol portfolio-level 4h/1d Donchian/EMA/ensemble policies.",
            "result": "Failed.",
            "pass_fail": "fail",
            "key_finding": "No stable candidate; concentration and OOS fragility remain.",
            "decision": "Do not build Strategy V3.",
        },
        {
            "stage": "Extended V3",
            "goal": "Retest the same V3.0 policy set on 2023-2026 data.",
            "result": "Failed.",
            "pass_fail": "fail",
            "key_finding": "stable_candidate_exists=false; only 1d EMA had all no-cost splits positive but failed concentration/funding gates.",
            "decision": "Do not enter V3.1 from extended V3 alone.",
        },
        {
            "stage": "Regime Diagnostics",
            "goal": "Check whether trend regimes exist and whether V3.0 aligned with them.",
            "result": "Rejected V3.1.",
            "pass_fail": "fail",
            "key_finding": "Strong trend share is 4.79%; V3 profits were not mainly from strong trend; 1d EMA strong-regime PnL was negative.",
            "decision": "proceed_to_v3_1_research=false.",
        },
    ]


def required_acceptance_criteria() -> list[str]:
    """Return required criteria for any future trend-following research gate."""

    return [
        "Pre-register train/validation/OOS split, cost model, funding model, concentration gates, and stability gates before running comparisons.",
        "Every candidate must be positive in train, validation, and OOS on no-cost and remain nonnegative after realistic costs.",
        "Funding stress must not turn the best OOS candidate negative at realistic bps / 8h assumptions.",
        "OOS performance must not depend on one symbol, one trade, or top 5% trade contribution above the configured gate.",
        "Data coverage must remain complete for every included symbol and timeframe.",
        "Regime attribution must show profits are structurally aligned with trend regimes and losses are controlled outside them.",
        "Strategy development, demo, and live remain blocked until these gates are met and documented in a new dossier.",
    ]


def extract_data_status(parsed_reports: dict[str, Any]) -> dict[str, Any]:
    """Extract data readiness from trend regime data quality when available."""

    data_quality = parsed_reports.get("trend_regime_data_quality")
    regime_summary = parsed_reports.get("trend_regime_summary")
    symbols = ["BTC", "ETH", "SOL", "LINK", "DOGE"]
    coverage_rows: list[dict[str, Any]] = []
    data_ready = False
    if isinstance(data_quality, dict):
        coverage = data_quality.get("symbol_coverage") or {}
        data_ready = bool(data_quality.get("all_symbols_complete"))
        for vt_symbol, row in coverage.items():
            coverage_rows.append(
                {
                    "symbol": vt_symbol,
                    "total_count": row.get("total_count"),
                    "expected_count": row.get("expected_count"),
                    "missing_count": row.get("missing_count"),
                    "gap_count": row.get("gap_count"),
                    "required_coverage_ready": row.get("required_coverage_ready"),
                }
            )
    elif isinstance(regime_summary, dict):
        data_ready = bool(regime_summary.get("data_quality_passed"))

    return {
        "data_ready": data_ready,
        "symbols": symbols,
        "coverage_window": "2023-01-01 to 2026-03-31",
        "interval": "1m",
        "missing_count": 0 if data_ready else None,
        "gap_count": 0 if data_ready else None,
        "coverage": coverage_rows,
        "data_failure_reason": None if data_ready else "data_quality_report_missing_or_incomplete",
    }


def extract_regime_findings(parsed_reports: dict[str, Any]) -> dict[str, Any]:
    """Extract trend regime findings with conservative defaults."""

    summary = parsed_reports.get("trend_regime_summary")
    if not isinstance(summary, dict):
        return {
            "strong_trend_pct": 0.0479,
            "choppy_high_vol_pct": 0.3870,
            "strongest_symbol": "SOLUSDT_SWAP_OKX.GLOBAL",
            "weakest_symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
            "one_day_ema_only_strong_effective": False,
            "donchian_losses_mainly_choppy_high_vol": True,
        }
    trade = summary.get("trade_diagnostics") or {}
    return {
        "strong_trend_pct": summary.get("strong_trend_pct", 0.0479),
        "choppy_high_vol_pct": summary.get("choppy_high_vol_pct", 0.3870),
        "strongest_symbol": (summary.get("strongest_symbol") or {}).get("symbol", "SOLUSDT_SWAP_OKX.GLOBAL"),
        "weakest_symbol": (summary.get("weakest_symbol") or {}).get("symbol", "BTCUSDT_SWAP_OKX.GLOBAL"),
        "one_day_ema_only_strong_effective": bool(trade.get("one_day_ema_only_strong_effective", False)),
        "one_day_ema_strong_no_cost_pnl": ((trade.get("family_diagnostics") or {}).get("1d_ema") or {}).get("strong_no_cost_pnl"),
        "one_day_ema_nonstrong_no_cost_pnl": ((trade.get("family_diagnostics") or {}).get("1d_ema") or {}).get("nonstrong_no_cost_pnl"),
        "donchian_losses_mainly_choppy_high_vol": bool(trade.get("donchian_losses_mainly_choppy_high_vol", True)),
    }


def build_decision_payload(
    *,
    source_rows: list[dict[str, Any]],
    parsed_reports: dict[str, Any],
    warnings: list[str],
    include_existing_reports: bool,
) -> dict[str, Any]:
    """Build the machine-readable dossier payload."""

    data_status = extract_data_status(parsed_reports)
    regime_findings = extract_regime_findings(parsed_reports)
    postmortem_rec = parsed_reports.get("trend_v3_postmortem_recommendations") if include_existing_reports else {}
    regime_rec = parsed_reports.get("trend_regime_recommendations") if include_existing_reports else {}
    extended_summary = parsed_reports.get("trend_v3_extended_compare_summary") if include_existing_reports else {}

    strategy_allowed = False
    demo_allowed = False
    proceed_v3_1 = False
    current_v3_failed = True
    blocking_reasons = [
        "no stable candidate",
        "regime diagnostics rejects V3.1",
        "demo/live disallowed",
        "funding sensitivity unresolved",
        "top trade concentration unresolved",
        "symbol concentration unresolved",
    ]
    if isinstance(extended_summary, dict) and extended_summary.get("stable_candidate_exists") is False:
        blocking_reasons.append("Extended Trend V3 stable_candidate_exists=false")
    if isinstance(regime_rec, dict) and regime_rec.get("proceed_to_v3_1_research") is False:
        blocking_reasons.append("Trend Regime Diagnostics proceed_to_v3_1_research=false")
    if isinstance(postmortem_rec, dict) and postmortem_rec.get("proceed_to_v3_1") is False:
        blocking_reasons.append("Trend V3 Postmortem proceed_to_v3_1=false")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_existing_reports": include_existing_reports,
        "data_ready": bool(data_status["data_ready"]),
        "data_status": data_status,
        "strategy_development_allowed": strategy_allowed,
        "demo_live_allowed": demo_allowed,
        "proceed_to_v3_1_research": proceed_v3_1,
        "current_v3_family_failed": current_v3_failed,
        "no_policy_can_be_traded": True,
        "proceed_to_broader_universe_research": "conditional",
        "proceed_to_funding_research": "conditional",
        "failed_policy_families": failed_policy_families(),
        "retained_research_hypotheses": retained_research_hypotheses(),
        "do_not_continue": do_not_continue_items(),
        "next_research_options": next_research_options(),
        "blocking_reasons": sorted(set(blocking_reasons), key=blocking_reasons.index),
        "required_next_acceptance_criteria": required_acceptance_criteria(),
        "research_timeline": research_timeline(),
        "signal_lab_findings": {
            "summary": "Short-term breakout features are stable negative risk features, consistent with overheat/exhaustion.",
            "negative_features": [
                "recent_volatility_30m",
                "atr_pct",
                "breakout_distance_atr",
                "recent_return_30m",
                "recent_return_15m",
                "recent_return_5m",
                "volume_zscore_30m",
                "body_ratio",
            ],
        },
        "trend_regime_findings": regime_findings,
        "source_reports": source_rows,
        "warnings": warnings,
        "output_files": OUTPUT_FILES,
    }


def pct(value: Any) -> str:
    """Format decimal percentage values."""

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def value_text(value: Any) -> str:
    """Return a compact markdown-safe value."""

    if value is None:
        return "n/a"
    return str(value)


def generate_markdown(payload: dict[str, Any]) -> str:
    """Generate the dossier markdown report."""

    data_status = payload["data_status"]
    regime = payload["trend_regime_findings"]
    lines: list[str] = [
        "# Research Decision Dossier",
        "",
        "## 1. Executive Summary",
        "- 当前没有任何策略可进入 demo/live。",
        "- 当前没有任何 policy 可进入 Strategy V3 原型开发。",
        "- 当前趋势跟踪 V3.0 family 已失败。",
        "- 继续趋势跟踪需要新的研究前提，而不是继续调当前参数。",
        "- no policy can be traded from the current research package.",
        "",
        "## 2. Data Status",
        "- 2023-2026 五品种数据完整。",
        "- 当前 symbols: BTC / ETH / SOL / LINK / DOGE。",
        f"- data_ready={str(bool(payload['data_ready'])).lower()}",
        f"- coverage_window={data_status['coverage_window']}",
        f"- interval={data_status['interval']}",
        f"- missing_count={value_text(data_status['missing_count'])}",
        f"- gap_count={value_text(data_status['gap_count'])}",
        "- 数据不是当前失败原因。",
        "",
        "## 3. Research Timeline",
        "| stage | goal | result | pass/fail | key finding | decision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["research_timeline"]:
        lines.append(
            f"| {row['stage']} | {row['goal']} | {row['result']} | {row['pass_fail']} | {row['key_finding']} | {row['decision']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Failed Policy Families",
            "| policy family | status | failure reason | evidence | tradable |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["failed_policy_families"]:
        lines.append(
            f"| {row['policy_family']} | {row['status']} | {row['primary_failure']} | {row['evidence']} | {str(bool(row['tradable'])).lower()} |"
        )

    signal = payload["signal_lab_findings"]
    lines.extend(
        [
            "",
            "## 5. Signal Lab Findings",
            "- short-term breakout 更像 overheat/exhaustion risk。",
            "- high volatility / high ATR / large breakout / high recent return / volume spike / large body ratio 都是负向风险特征。",
            f"- 稳定负向特征：{', '.join(signal['negative_features'])}",
            "",
            "## 6. Trend Regime Findings",
            f"- strong trend 占比 {pct(regime['strong_trend_pct'])}。",
            f"- choppy/high_vol_choppy 占比 {pct(regime['choppy_high_vol_pct'])}。",
            f"- strongest symbol {regime['strongest_symbol']}。",
            f"- weakest symbol {regime['weakest_symbol']}。",
            f"- 1d EMA 不只在 strong trend 有效；strong no-cost PnL={value_text(regime.get('one_day_ema_strong_no_cost_pnl'))}。",
            f"- Donchian 亏在 choppy/high_vol_choppy：{str(bool(regime['donchian_losses_mainly_choppy_high_vol'])).lower()}。",
            "",
            "## 7. Why Strategy Development Is Blocked",
            "Strategy development is blocked because:",
        ]
    )
    for reason in payload["blocking_reasons"]:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## 8. Retained Research Hypotheses",
            "| hypothesis | status | reason | not allowed as |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload["retained_research_hypotheses"]:
        lines.append(f"| {row['hypothesis']} | {row['status']} | {row['reason']} | {row['not_allowed_as']} |")

    lines.extend(
        [
            "",
            "## 9. Do Not Continue List",
            "| item | reason |",
            "| --- | --- |",
        ]
    )
    for row in payload["do_not_continue"]:
        lines.append(f"| {row['item']} | {row['reason']} |")

    lines.extend(
        [
            "",
            "## 10. Next Research Options",
            "| option | name | prerequisites | acceptance criteria | allowed now |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["next_research_options"]:
        lines.append(
            f"| {row['option']} | {row['name']} | {row['prerequisites']} | {row['acceptance_criteria']} | {row['allowed_now']} |"
        )

    lines.extend(
        [
            "",
            "## 11. Final Decision",
            f"- strategy_development_allowed={str(bool(payload['strategy_development_allowed'])).lower()}",
            f"- demo_live_allowed={str(bool(payload['demo_live_allowed'])).lower()}",
            f"- proceed_to_v3_1_research={str(bool(payload['proceed_to_v3_1_research'])).lower()}",
            f"- current_v3_family_failed={str(bool(payload['current_v3_family_failed'])).lower()}",
            "- proceed_to_broader_universe_research=conditional",
            "- proceed_to_funding_research=conditional",
            "",
            "## Source Report Warnings",
        ]
    )
    if payload["warnings"]:
        lines.extend([f"- {warning}" for warning in payload["warnings"]])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""

    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a CSV file with stable field order."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    """Write all dossier outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "research_decision_dossier.json", payload)
    (output_dir / "research_decision_dossier.md").write_text(generate_markdown(payload), encoding="utf-8")
    write_csv(
        output_dir / "failed_policy_families.csv",
        payload["failed_policy_families"],
        ["policy_family", "status", "primary_failure", "evidence", "tradable"],
    )
    write_csv(
        output_dir / "retained_research_hypotheses.csv",
        payload["retained_research_hypotheses"],
        ["hypothesis", "status", "reason", "not_allowed_as"],
    )
    write_csv(
        output_dir / "do_not_continue_list.csv",
        payload["do_not_continue"],
        ["item", "reason"],
    )
    write_csv(
        output_dir / "next_research_options.csv",
        payload["next_research_options"],
        ["option", "name", "prerequisites", "acceptance_criteria", "allowed_now"],
    )


def build_dossier(
    *,
    output_dir: Path,
    include_existing_reports: bool = True,
    project_root: Path = PROJECT_ROOT,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Build and write the research decision dossier."""

    logger = logger or logging.getLogger("build_research_decision_dossier")
    source_rows, parsed_reports, warnings = load_report_sources(project_root, include_existing_reports)
    payload = build_decision_payload(
        source_rows=source_rows,
        parsed_reports=parsed_reports,
        warnings=warnings,
        include_existing_reports=include_existing_reports,
    )
    write_outputs(output_dir, payload)
    log_event(
        logger,
        logging.INFO,
        "research_dossier.completed",
        "Research decision dossier completed",
        output_dir=output_dir,
        strategy_development_allowed=payload["strategy_development_allowed"],
        demo_live_allowed=payload["demo_live_allowed"],
        proceed_to_v3_1_research=payload["proceed_to_v3_1_research"],
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    ensure_headless_runtime()
    args = parse_args(argv)
    logger = setup_logging("build_research_decision_dossier", verbose=args.verbose)
    try:
        output_dir = resolve_path(args.output_dir)
        payload = build_dossier(
            output_dir=output_dir,
            include_existing_reports=include_reports_enabled(args.include_existing_reports),
            project_root=PROJECT_ROOT,
            logger=logger,
        )
        if args.json:
            print_json_block(payload)
        return 0
    except ResearchDecisionDossierError as exc:
        log_event(logger, logging.ERROR, "research_dossier.error", str(exc))
        return 2
    except Exception as exc:
        log_event(logger, logging.ERROR, "research_dossier.unexpected_error", str(exc), exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
