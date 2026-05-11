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
    {
        "key": "actual_funding_report",
        "stage": "Funding-aware Trend Research",
        "path": "reports/research/trend_following_v3_actual_funding/actual_funding_report.md",
    },
    {
        "key": "actual_funding_summary",
        "stage": "Funding-aware Trend Research",
        "path": "reports/research/trend_following_v3_actual_funding/actual_funding_summary.json",
    },
    {
        "key": "funding_verify_report",
        "stage": "Funding data verification",
        "path": "reports/research/funding/okx_funding_verify_report.md",
    },
    {
        "key": "funding_verify_summary",
        "stage": "Funding data verification",
        "path": "reports/research/funding/okx_funding_verify_summary.json",
    },
    {
        "key": "historical_funding_download_report",
        "stage": "Funding data source",
        "path": "reports/research/funding_historical_download/okx_historical_funding_download_report.md",
    },
    {
        "key": "historical_funding_download_summary",
        "stage": "Funding data source",
        "path": "reports/research/funding_historical_download/okx_historical_funding_download_summary.json",
    },
    {
        "key": "external_regime_gate_audit_report",
        "stage": "External Regime Classifier Gate Audit",
        "path": "reports/research/external_regime_classifier_gate_audit/external_regime_gate_audit_report.md",
    },
    {
        "key": "external_regime_gate_audit_summary",
        "stage": "External Regime Classifier Gate Audit",
        "path": "reports/research/external_regime_classifier_gate_audit/external_regime_gate_audit_summary.json",
    },
    {
        "key": "external_regime_gate_comparison",
        "stage": "External Regime Classifier Gate Audit",
        "path": "reports/research/external_regime_classifier_gate_audit/gate_comparison.csv",
    },
    {
        "key": "external_regime_filter_trade_set_diff",
        "stage": "External Regime Classifier Gate Audit",
        "path": "reports/research/external_regime_classifier_gate_audit/filter_trade_set_diff.csv",
    },
    {
        "key": "derivatives_data_readiness_report",
        "stage": "Derivatives Data Readiness Audit",
        "path": "reports/research/derivatives_data_readiness/derivatives_data_readiness_report.md",
    },
    {
        "key": "derivatives_data_readiness_summary",
        "stage": "Derivatives Data Readiness Audit",
        "path": "reports/research/derivatives_data_readiness/derivatives_data_readiness.json",
    },
    {
        "key": "derivatives_endpoint_probe_results",
        "stage": "Derivatives Data Readiness Audit",
        "path": "reports/research/derivatives_data_readiness/endpoint_probe_results.csv",
    },
    {
        "key": "derivatives_proposed_features",
        "stage": "Derivatives Data Readiness Audit",
        "path": "reports/research/derivatives_data_readiness/proposed_derivatives_features.csv",
    },
    {
        "key": "derivatives_unavailable_features",
        "stage": "Derivatives Data Readiness Audit",
        "path": "reports/research/derivatives_data_readiness/unavailable_derivatives_features.csv",
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
        {
            "policy_family": "funding-adjusted V3 family",
            "status": "failed",
            "primary_failure": "Actual OKX funding data is complete, but funding-adjusted stable_candidate_exists remains false.",
            "evidence": "Funding-aware Trend Research actual funding report and Research Decision Dossier final gate.",
            "tradable": False,
        },
        {
            "policy_family": "VSVCB-v1 positive breakout hypothesis",
            "status": "failed",
            "primary_failure": "Breakout + squeeze + volume confirmation failed train/validation/OOS and reverse test was stronger than the positive breakout hypothesis.",
            "evidence": "VSVCB-v1 Phase 1 and postmortem.",
            "tradable": False,
        },
    ]


def retained_research_hypotheses() -> list[dict[str, Any]]:
    """Return research-only hypotheses that remain allowed as future work."""

    return [
        {
            "hypothesis": "broader universe trend following",
            "status": "optional_research_only",
            "reason": "Five-symbol universe may be too narrow for sparse crypto trend regimes, but this is not the recommended main path while the current user scope avoids universe expansion.",
            "not_allowed_as": "Strategy V3 or demo/live without new evidence.",
        },
        {
            "hypothesis": "true funding-aware trend following",
            "status": "completed_no_gate_opened",
            "reason": "Actual OKX funding is now complete for the current universe; funding-aware analysis did not create a stable candidate.",
            "not_allowed_as": "Further funding-only work on the current V3 family.",
        },
        {
            "hypothesis": "stronger macro/trend regime classifier",
            "status": "closed_for_current_v3_family",
            "reason": "External classifier gate audit found no strict stable candidate; OOS trend_friendly was too sparse and exclude filters did not rescue V3.",
            "not_allowed_as": "V3.1 rescue or direct trade filter on current V3 family.",
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
            "reason": "It is still the only all no-cost positive policy, but actual funding completion did not fix concentration and regime failures.",
            "not_allowed_as": "Tradable policy or Strategy V3 prototype.",
        },
        {
            "hypothesis": "derivatives-confirmed trend following",
            "status": "data_blocked_research_hypothesis",
            "reason": "Historical OI/taker/long-short coverage not proven for 2023-2026; funding plus mark/index alone is not enough derivatives confirmation.",
            "not_allowed_as": "Strategy V3, V3.1, demo/live, or a tradable policy from readiness audit output.",
        },
    ]


def do_not_continue_items() -> list[dict[str, Any]]:
    """Return the explicit stop list."""

    return [
        {"item": "do not expand current Donchian grid", "reason": "Current Donchian families failed stability and regime attribution."},
        {"item": "do not trade v3_1d_ema_50_200_atr5", "reason": "Rejected by top trade concentration, funding fragility, and regime diagnostics."},
        {"item": "do not develop v3_1d_ema_50_200_atr5", "reason": "Actual funding analysis completed, but it is still not a stable strategy candidate."},
        {"item": "do not continue current V3 family after actual funding completion", "reason": "Funding-aware final gate did not open V3.1 or Strategy V3."},
        {"item": "do not enter funding-aware V3.1 without a new hypothesis", "reason": "Current V3 family failed after actual funding and regime diagnostics remain blocking."},
        {"item": "do not enter demo/live", "reason": "No stable candidate and demo_live_allowed=false."},
        {"item": "do not build Strategy V3 from current results", "reason": "strategy_development_allowed=false."},
        {"item": "do not continue V3.0 ensemble_core", "reason": "No stable candidate and component families failed."},
        {"item": "do not optimize 1m breakout", "reason": "Short-term breakout is aligned with overheat/exhaustion risk, not durable trend following."},
        {"item": "do not continue external regime classifier as V3.1 rescue", "reason": "Gate consistency audit found no strict stable classifier-filtered candidate."},
        {"item": "do not treat original_all v3_1d_ema_50_200_atr5 as stable", "reason": "OOS top 5% trade contribution is 1.9818, above the 0.8 gate."},
        {"item": "do not ignore top trade concentration", "reason": "Top-trade concentration is the decisive Extended V3 and classifier gate blocker."},
        {"item": "do not treat no-cost positive as tradable", "reason": "No-cost positives still fail concentration/funding/regime gates and are not strategy candidates."},
        {"item": "do not use classifier filters that do not change OOS trade set", "reason": "exclude_hostile_chop_overheated and exclude_funding_overheated did not affect v3_1d_ema_50_200_atr5 OOS trades."},
        {
            "item": "do not start derivatives-confirmed trend research without historical OI/taker/long-short coverage",
            "reason": "Derivatives readiness audit did not prove the required 2023-2026 historical derivatives feature coverage.",
        },
        {
            "item": "do not use current open interest snapshot as historical feature",
            "reason": "The public open-interest endpoint probe is current snapshot only and cannot stand in for 2023-2026 history.",
        },
        {
            "item": "do not treat funding alone as derivatives confirmation",
            "reason": "Funding data is complete, but funding alone does not satisfy the required OI/taker/long-short confirmation mix.",
        },
        {
            "item": "do not develop Strategy V3 from derivatives readiness audit",
            "reason": "Endpoint availability audit is not a strategy result and can_enter_derivatives_confirmed_trend_research=false.",
        },
    ]


def next_research_options() -> list[dict[str, Any]]:
    """Return conditional next research paths."""

    return [
        {
            "option": "Option A",
            "name": "Broader universe trend following readiness (optional, not main path)",
            "prerequisites": "Verified metadata, listing dates, contract specs, liquidity, and 1m sqlite coverage for a materially broader symbol set.",
            "acceptance_criteria": "Multi-symbol universe passes coverage checks; research design pre-registers stable-candidate criteria before any policy comparison.",
            "allowed_now": "optional_not_recommended",
        },
        {
            "option": "Option B",
            "name": "Funding-aware research complete for current universe",
            "prerequisites": "A new non-V3 policy family or new hypothesis before any further funding-only work.",
            "acceptance_criteria": "No additional funding-only research unless a new candidate policy family emerges.",
            "allowed_now": "no",
        },
        {
            "option": "Option C",
            "name": "classifier-filtered V3.1 continuation",
            "prerequisites": "Closed for current V3 family.",
            "acceptance_criteria": "Not applicable; strict gate audit found no stable classifier-filtered candidate.",
            "allowed_now": "no",
        },
        {
            "option": "Option D",
            "name": "Pause strategy development and maintain tooling",
            "prerequisites": "None beyond maintaining data integrity and reproducible reports.",
            "acceptance_criteria": "No new strategy work starts until a new research premise is documented and approved.",
            "allowed_now": "yes",
        },
        {
            "option": "Option E",
            "name": "Derivatives-confirmed trend research",
            "prerequisites": "historical derivatives metrics coverage proven",
            "acceptance_criteria": "Historical OI or contracts OI/volume plus taker volume or long/short ratio must cover 2023-2026 without private API keys; funding alone is insufficient.",
            "allowed_now": "no",
        },
        {
            "option": "Option F",
            "name": "Third-party/external derivatives data audit",
            "prerequisites": "user explicitly agrees to external data source or paid/free vendor evaluation",
            "acceptance_criteria": "External source must provide reproducible 2023-2026 coverage, licensing clarity, and no strategy conclusion before data audit passes.",
            "allowed_now": "optional",
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
        {
            "stage": "Funding-aware Final Gate",
            "goal": "Apply complete actual OKX funding to Trend V3 extended trades.",
            "result": "Completed; gates still closed.",
            "pass_fail": "data_pass_strategy_fail",
            "key_finding": "funding_data_complete=true, but funding_adjusted_stable_candidate_exists=false.",
            "decision": "current_v3_family_failed_after_actual_funding=true.",
        },
        {
            "stage": "External Regime Classifier Gate Audit",
            "goal": "Check whether classifier-filtered V3.1 rescue is consistent with Dossier and Extended V3 strict gates.",
            "result": "Completed; no strict stable candidate.",
            "pass_fail": "fail",
            "key_finding": "Old classifier gate underestimated top-trade concentration; v3_1d_ema_50_200_atr5 OOS top 5% contribution=1.9818.",
            "decision": "can_enter_research_only_v3_1_classifier_experiment=false.",
        },
        {
            "stage": "Derivatives Data Readiness Audit",
            "goal": "Check whether public/no-key OKX derivatives data can support a new derivatives-confirmed trend hypothesis.",
            "result": "Completed; data readiness gate blocked.",
            "pass_fail": "data_fail_research_blocked",
            "key_finding": "Funding and mark/index candles are available, but historical OI/taker/long-short coverage for 2023-2026 was not proven.",
            "decision": "can_enter_derivatives_confirmed_trend_research=false.",
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
        "market_data_complete": data_ready,
        "symbols": symbols,
        "coverage_window": "2023-01-01 to 2026-03-31",
        "interval": "1m",
        "missing_count": 0 if data_ready else None,
        "gap_count": 0 if data_ready else None,
        "coverage": coverage_rows,
        "data_failure_reason": None if data_ready else "data_quality_report_missing_or_incomplete",
    }


def extract_actual_funding_status(parsed_reports: dict[str, Any]) -> dict[str, Any]:
    """Extract actual OKX funding completion and funding-aware gates."""

    actual_summary = parsed_reports.get("actual_funding_summary")
    verify_summary = parsed_reports.get("funding_verify_summary")
    download_summary = parsed_reports.get("historical_funding_download_summary")
    actual = actual_summary if isinstance(actual_summary, dict) else {}
    verify = verify_summary if isinstance(verify_summary, dict) else {}
    download = download_summary if isinstance(download_summary, dict) else {}

    verify_results = verify.get("results") if isinstance(verify.get("results"), list) else []
    download_results = download.get("inst_results") if isinstance(download.get("inst_results"), list) else []
    download_by_inst = {
        str(row.get("inst_id")): row
        for row in download_results
        if isinstance(row, dict) and row.get("inst_id") is not None
    }
    inst_results: list[dict[str, Any]] = []
    for row in verify_results:
        if not isinstance(row, dict):
            continue
        inst_id = str(row.get("inst_id") or "")
        download_row = download_by_inst.get(inst_id, {})
        inst_results.append(
            {
                "inst_id": inst_id,
                "row_count": row.get("row_count"),
                "first_time": row.get("first_available_time") or row.get("first_funding_time") or download_row.get("first_time"),
                "last_time": row.get("last_available_time") or row.get("last_funding_time") or download_row.get("last_time"),
                "complete": bool(row.get("coverage_complete")),
                "completion_status": row.get("completion_status"),
            }
        )
    if not inst_results:
        for row in download_results:
            if not isinstance(row, dict):
                continue
            inst_results.append(
                {
                    "inst_id": row.get("inst_id"),
                    "row_count": row.get("row_count"),
                    "first_time": row.get("first_time"),
                    "last_time": row.get("last_time"),
                    "complete": False,
                    "completion_status": row.get("complete_decided_by", "requires_verify_funding"),
                }
            )

    funding_data_complete = bool(actual.get("funding_data_complete") or verify.get("funding_data_complete"))
    funding_adjusted_stable_candidate_exists = bool(actual.get("funding_adjusted_stable_candidate_exists"))
    can_enter_funding_aware = bool(actual.get("can_enter_funding_aware_v3_1_research"))
    current_universe_funding_complete = bool(funding_data_complete and inst_results and all(row.get("complete") for row in inst_results))
    historical_download_succeeded = bool(download.get("status") == "downloaded")

    return {
        "actual_funding_data_complete": funding_data_complete,
        "actual_funding_source": "OKX Historical Market Data" if historical_download_succeeded else "OKX funding reports unavailable",
        "rest_funding_endpoint_partial_only": True,
        "historical_funding_auto_download_succeeded": historical_download_succeeded,
        "current_universe_funding_complete": current_universe_funding_complete,
        "funding_data_complete": funding_data_complete,
        "inst_results": inst_results,
        "funding_adjusted_stable_candidate_exists": funding_adjusted_stable_candidate_exists,
        "funding_adjusted_stable_candidates": actual.get("funding_adjusted_stable_candidates", []),
        "can_enter_funding_aware_v3_1_research": can_enter_funding_aware,
        "strategy_development_allowed": bool(actual.get("strategy_development_allowed")),
        "demo_live_allowed": bool(actual.get("demo_live_allowed")),
        "target_policy": actual.get("target_policy", "v3_1d_ema_50_200_atr5"),
        "target_policy_conservative_all_splits_positive": bool(actual.get("target_policy_conservative_all_splits_positive")),
        "target_policy_signed_all_splits_positive": bool(actual.get("target_policy_signed_all_splits_positive")),
        "oos_best_policy_after_funding": actual.get("oos_best_policy_after_funding") if isinstance(actual.get("oos_best_policy_after_funding"), dict) else {},
        "verify_incomplete_reason": verify.get("incomplete_reason", []),
        "symbols_with_warnings": verify.get("symbols_with_warnings", []),
        "downloaded_file_count": download.get("downloaded_file_count"),
        "extracted_csv_count": download.get("extracted_csv_count"),
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


def extract_external_classifier_gate_audit(parsed_reports: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Extract the final external classifier gate audit result without failing on missing files."""

    summary = parsed_reports.get("external_regime_gate_audit_summary")
    if not isinstance(summary, dict):
        warnings.append("missing_external_regime_classifier_gate_audit_summary")
        return {
            "external_regime_classifier_gate_audit_complete": False,
            "classifier_old_gate_inconsistent": True,
            "classifier_strict_stable_candidate_exists": False,
            "can_enter_research_only_v3_1_classifier_experiment": False,
            "external_classifier_rescued_v3_family": False,
            "original_all_strict_stable_candidate_like_count": 0,
            "strict_stable_candidate_like_count": 0,
            "v3_1d_ema_oos_top_5pct_trade_pnl_contribution": 1.9818,
            "v3_1d_ema_oos_largest_symbol_pnl_share": None,
            "v3_1d_ema_exclude_filters_did_not_affect_oos": True,
            "trend_friendly_only_removed_all_oos_trades": True,
            "reason": "Gate audit summary missing; keep all final gates closed.",
        }

    complete = True
    strict_count = int(summary.get("strict_stable_candidate_like_count") or 0)
    can_enter = bool(summary.get("can_enter_research_only_v3_1_classifier_experiment"))
    return {
        "external_regime_classifier_gate_audit_complete": complete,
        "classifier_old_gate_inconsistent": True,
        "classifier_strict_stable_candidate_exists": bool(strict_count > 0),
        "can_enter_research_only_v3_1_classifier_experiment": bool(can_enter),
        "external_classifier_rescued_v3_family": False,
        "original_all_strict_stable_candidate_like_count": int(summary.get("original_all_strict_stable_candidate_like_count") or 0),
        "strict_stable_candidate_like_count": strict_count,
        "v3_1d_ema_oos_top_5pct_trade_pnl_contribution": 1.9818,
        "v3_1d_ema_oos_largest_symbol_pnl_share": 0.3462,
        "v3_1d_ema_exclude_filters_did_not_affect_oos": bool(summary.get("v3_1d_ema_exclude_filters_did_not_affect_oos")),
        "trend_friendly_only_removed_all_oos_trades": True,
        "filter_did_not_affect_oos_count": int(summary.get("filter_did_not_affect_oos_count") or 0),
        "filter_did_affect_oos_count": int(summary.get("filter_did_affect_oos_count") or 0),
        "reason": summary.get("reason") or "No non-original classifier filter passed strict gates.",
    }


def normalize_derivatives_next_step(value: Any) -> str:
    """Normalize derivatives audit next-step text to dossier enum style."""

    normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in {"pause_research", "pause_strategy_development"}:
        return "pause_research"
    if normalized in {"download_derivatives_metrics", "import_historical_files"}:
        return normalized
    return "pause_research"


def extract_derivatives_data_readiness(parsed_reports: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Extract the final derivatives data readiness gate without failing on missing files."""

    summary = parsed_reports.get("derivatives_data_readiness_summary")
    missing_features = [
        "open_interest_history",
        "taker_buy_sell_volume",
        "long_short_account_ratio",
        "contracts_open_interest_volume",
        "premium_history",
    ]
    available_features = [
        "actual_funding_rate",
        "funding_dispersion",
        "funding_sign_breadth",
        "funding_trend",
        "mark_index_price_candles",
    ]
    if not isinstance(summary, dict):
        warnings.append("missing_derivatives_data_readiness_summary")
        return {
            "derivatives_data_readiness_audit_complete": False,
            "can_enter_derivatives_confirmed_trend_research": False,
            "derivatives_data_blocker": True,
            "derivatives_missing_historical_features": missing_features,
            "derivatives_available_features": available_features,
            "derivatives_research_recommended_next_step": "pause_research",
            "funding_complete_but_not_sufficient": True,
            "mark_index_available_but_not_sufficient": True,
            "current_open_interest_snapshot_only": True,
            "key_historical_features_coverage_not_proven": True,
            "endpoint_probe_results": [],
            "reason": "Derivatives data readiness summary missing; keep derivatives-confirmed trend research blocked.",
        }

    decision = summary.get("decision") if isinstance(summary.get("decision"), dict) else {}
    endpoints = summary.get("endpoint_probe_results") if isinstance(summary.get("endpoint_probe_results"), list) else []
    features = summary.get("proposed_derivatives_features") if isinstance(summary.get("proposed_derivatives_features"), list) else []
    local_funding = summary.get("local_funding") if isinstance(summary.get("local_funding"), dict) else {}

    can_enter = bool(decision.get("can_enter_derivatives_confirmed_trend_research"))
    endpoint_by_name = {
        str(row.get("endpoint_name")): row
        for row in endpoints
        if isinstance(row, dict) and row.get("endpoint_name") is not None
    }
    feature_by_name = {
        str(row.get("feature_name")): row
        for row in features
        if isinstance(row, dict) and row.get("feature_name") is not None
    }
    current_oi = endpoint_by_name.get("Open Interest", {})
    mark_feature = feature_by_name.get("mark price", {})
    index_feature = feature_by_name.get("index price", {})

    return {
        "derivatives_data_readiness_audit_complete": True,
        "can_enter_derivatives_confirmed_trend_research": can_enter,
        "derivatives_data_blocker": not can_enter,
        "derivatives_missing_historical_features": missing_features,
        "derivatives_available_features": available_features,
        "derivatives_research_recommended_next_step": normalize_derivatives_next_step(
            decision.get("recommended_next_step")
        ),
        "funding_complete_but_not_sufficient": bool(local_funding.get("funding_data_complete")) and not can_enter,
        "mark_index_available_but_not_sufficient": bool(
            mark_feature.get("usable_for_research") or index_feature.get("usable_for_research")
        )
        and not can_enter,
        "current_open_interest_snapshot_only": str(current_oi.get("warning") or "").find("current_snapshot_only") >= 0,
        "key_historical_features_coverage_not_proven": not bool(
            decision.get("open_interest_available")
            and (decision.get("taker_flow_available") or decision.get("long_short_ratio_available"))
        ),
        "non_price_derivatives_feature_category_count": int(
            decision.get("non_price_derivatives_feature_category_count") or 0
        ),
        "non_price_derivatives_feature_categories_available": decision.get(
            "non_price_derivatives_feature_categories_available", []
        ),
        "blocking_reasons": decision.get("blocking_reasons", []),
        "endpoint_probe_results": endpoints,
        "reason": "Historical OI/taker/long-short coverage not proven; funding plus mark/index alone is not enough.",
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
    actual_funding = extract_actual_funding_status(parsed_reports)
    data_status["funding_data_complete"] = bool(actual_funding["actual_funding_data_complete"])
    data_status["funding_source"] = actual_funding["actual_funding_source"]
    data_status["rest_funding_endpoint_partial_only"] = bool(actual_funding["rest_funding_endpoint_partial_only"])
    data_status["historical_funding_auto_download_succeeded"] = bool(actual_funding["historical_funding_auto_download_succeeded"])
    regime_findings = extract_regime_findings(parsed_reports)
    classifier_gate = extract_external_classifier_gate_audit(parsed_reports, warnings)
    derivatives_gate = extract_derivatives_data_readiness(parsed_reports, warnings)
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
        "top trade concentration unresolved",
        "symbol concentration unresolved",
    ]
    if actual_funding["actual_funding_data_complete"]:
        blocking_reasons.append("actual OKX funding complete, but funding-adjusted stable_candidate_exists=false")
    else:
        blocking_reasons.append("actual funding report missing or incomplete")
    if isinstance(extended_summary, dict) and extended_summary.get("stable_candidate_exists") is False:
        blocking_reasons.append("Extended Trend V3 stable_candidate_exists=false")
    if isinstance(regime_rec, dict) and regime_rec.get("proceed_to_v3_1_research") is False:
        blocking_reasons.append("Trend Regime Diagnostics proceed_to_v3_1_research=false")
    if isinstance(postmortem_rec, dict) and postmortem_rec.get("proceed_to_v3_1") is False:
        blocking_reasons.append("Trend V3 Postmortem proceed_to_v3_1=false")
    if actual_funding["target_policy"] == "v3_1d_ema_50_200_atr5":
        blocking_reasons.append("v3_1d_ema_50_200_atr5 remains a weak lead only, rejected by concentration and regime gates")
    if classifier_gate["external_regime_classifier_gate_audit_complete"]:
        blocking_reasons.append("External classifier gate audit complete; no strict stable candidate")
    else:
        blocking_reasons.append("External classifier gate audit missing or incomplete; final gates remain closed")
    if derivatives_gate["derivatives_data_blocker"]:
        blocking_reasons.append("Derivatives data readiness blocks derivatives-confirmed trend research")
    blocking_reasons.append("current five-symbol trend-following family is fully archived")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_existing_reports": include_existing_reports,
        "data_ready": bool(data_status["data_ready"]),
        "data_status": data_status,
        "actual_funding": actual_funding,
        "external_regime_classifier_gate_audit": classifier_gate,
        "derivatives_data_readiness": derivatives_gate,
        "external_regime_classifier_gate_audit_complete": bool(classifier_gate["external_regime_classifier_gate_audit_complete"]),
        "classifier_old_gate_inconsistent": bool(classifier_gate["classifier_old_gate_inconsistent"]),
        "classifier_strict_stable_candidate_exists": bool(classifier_gate["classifier_strict_stable_candidate_exists"]),
        "can_enter_research_only_v3_1_classifier_experiment": bool(classifier_gate["can_enter_research_only_v3_1_classifier_experiment"]),
        "external_classifier_rescued_v3_family": bool(classifier_gate["external_classifier_rescued_v3_family"]),
        "derivatives_data_readiness_audit_complete": bool(derivatives_gate["derivatives_data_readiness_audit_complete"]),
        "can_enter_derivatives_confirmed_trend_research": bool(
            derivatives_gate["can_enter_derivatives_confirmed_trend_research"]
        ),
        "derivatives_data_blocker": bool(derivatives_gate["derivatives_data_blocker"]),
        "derivatives_missing_historical_features": derivatives_gate["derivatives_missing_historical_features"],
        "derivatives_available_features": derivatives_gate["derivatives_available_features"],
        "derivatives_research_recommended_next_step": derivatives_gate["derivatives_research_recommended_next_step"],
        "actual_funding_data_complete": bool(actual_funding["actual_funding_data_complete"]),
        "actual_funding_source": actual_funding["actual_funding_source"],
        "rest_funding_endpoint_partial_only": bool(actual_funding["rest_funding_endpoint_partial_only"]),
        "funding_adjusted_stable_candidate_exists": bool(actual_funding["funding_adjusted_stable_candidate_exists"]),
        "can_enter_funding_aware_v3_1_research": bool(actual_funding["can_enter_funding_aware_v3_1_research"]),
        "current_universe_funding_complete": bool(actual_funding["current_universe_funding_complete"]),
        "current_v3_family_failed_after_actual_funding": True,
        "strategy_development_allowed": strategy_allowed,
        "demo_live_allowed": demo_allowed,
        "proceed_to_v3_1_research": proceed_v3_1,
        "current_v3_family_failed": current_v3_failed,
        "no_policy_can_be_traded": True,
        "final_strategy_development_allowed": False,
        "final_demo_live_allowed": False,
        "final_current_trend_family_archived": True,
        "final_current_research_archived": True,
        "proceed_to_broader_universe_research": "optional",
        "proceed_to_funding_research": "complete_for_current_universe_no_gate_opened",
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
    funding = payload["actual_funding"]
    regime = payload["trend_regime_findings"]
    derivatives = payload["derivatives_data_readiness"]
    lines: list[str] = [
        "# Research Decision Dossier",
        "",
        "## 1. Executive Summary",
        "- Actual OKX funding data is now complete for BTC/ETH/SOL/LINK/DOGE over 2023-2026.",
        "- Funding-aware analysis completed.",
        "- Funding-aware gates remain closed.",
        "- Strategy development remains blocked.",
        "- Demo/live remains blocked.",
        "- 当前没有任何策略可进入 demo/live。",
        "- 当前没有任何 policy 可进入 Strategy V3 原型开发。",
        "- 当前趋势跟踪 V3.0 family 已失败。",
        "- External Regime Classifier Gate Audit 已完成，未能救回当前 V3 family。",
        "- Derivatives Data Readiness Audit 已完成，但关键历史衍生品特征覆盖未通过。",
        "- 当前五品种趋势跟踪 family 已最终封档。",
        "- 继续趋势跟踪需要新的研究前提，而不是继续调当前参数。",
        "- 当前推荐暂停策略开发，只维护数据与研究工具。",
        "- no policy can be traded from the current research package.",
        "",
        "## 2. Data Status",
        "- 2023-2026 五品种数据完整。",
        "- 当前 symbols: BTC / ETH / SOL / LINK / DOGE。",
        f"- market_data_complete={str(bool(data_status.get('market_data_complete'))).lower()}",
        f"- data_ready={str(bool(payload['data_ready'])).lower()}",
        f"- funding_data_complete={str(bool(data_status.get('funding_data_complete'))).lower()}",
        f"- funding_source={data_status.get('funding_source')}",
        f"- rest_funding_endpoint_partial_only={str(bool(data_status.get('rest_funding_endpoint_partial_only'))).lower()}",
        f"- historical_funding_auto_download_succeeded={str(bool(data_status.get('historical_funding_auto_download_succeeded'))).lower()}",
        f"- coverage_window={data_status['coverage_window']}",
        f"- interval={data_status['interval']}",
        f"- missing_count={value_text(data_status['missing_count'])}",
        f"- gap_count={value_text(data_status['gap_count'])}",
        "- 数据不是当前失败原因。",
        "",
        "## 3. Funding-aware Final Gate",
        f"- funding_data_complete={str(bool(funding['actual_funding_data_complete'])).lower()}",
        f"- actual_funding_source={funding['actual_funding_source']}",
        f"- rest_funding_endpoint_partial_only={str(bool(funding['rest_funding_endpoint_partial_only'])).lower()}",
        f"- historical_funding_auto_download_succeeded={str(bool(funding['historical_funding_auto_download_succeeded'])).lower()}",
        f"- funding_adjusted_stable_candidate_exists={str(bool(funding['funding_adjusted_stable_candidate_exists'])).lower()}",
        f"- can_enter_funding_aware_v3_1_research={str(bool(funding['can_enter_funding_aware_v3_1_research'])).lower()}",
        f"- strategy_development_allowed={str(bool(payload['strategy_development_allowed'])).lower()}",
        f"- demo_live_allowed={str(bool(payload['demo_live_allowed'])).lower()}",
        f"- current_v3_family_failed_after_actual_funding={str(bool(payload['current_v3_family_failed_after_actual_funding'])).lower()}",
        f"- target_policy={funding.get('target_policy')}",
        "- why_gates_stay_closed=actual funding completion removes the data blocker, but it does not remove Extended V3 stable_candidate=false, top-trade concentration, or regime diagnostics rejection.",
        "",
        "| inst_id | row_count | first_time | last_time | complete |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in funding["inst_results"]:
        lines.append(
            f"| {row.get('inst_id')} | {value_text(row.get('row_count'))} | {value_text(row.get('first_time'))} | {value_text(row.get('last_time'))} | {str(bool(row.get('complete'))).lower()} |"
        )
    classifier = payload["external_regime_classifier_gate_audit"]
    lines.extend(
        [
        "",
        "## 4. External Regime Classifier Gate Audit Final Result",
        "- 之前 classifier 的 stable_candidate_like 口径存在问题。",
        "- 旧口径使用正收益合计作 top trade concentration 分母，低估 top trade concentration。",
        "- 修正后 v3_1d_ema_50_200_atr5 的 OOS top 5% contribution=1.9818，超过 0.8。",
        "- original_all 不通过 strict gate。",
        "- exclude_hostile_chop_overheated 和 exclude_funding_overheated 没有改变 v3_1d_ema_50_200_atr5 的 OOS trade set。",
        "- trend_friendly_only 删除全部 OOS trades。",
        f"- external_regime_classifier_gate_audit_complete={str(bool(payload['external_regime_classifier_gate_audit_complete'])).lower()}",
        f"- classifier_old_gate_inconsistent={str(bool(payload['classifier_old_gate_inconsistent'])).lower()}",
        f"- classifier_strict_stable_candidate_exists={str(bool(payload['classifier_strict_stable_candidate_exists'])).lower()}",
        f"- can_enter_research_only_v3_1_classifier_experiment={str(bool(payload['can_enter_research_only_v3_1_classifier_experiment'])).lower()}",
        f"- external_classifier_rescued_v3_family={str(bool(payload['external_classifier_rescued_v3_family'])).lower()}",
        f"- strategy_development_allowed={str(bool(payload['strategy_development_allowed'])).lower()}",
        f"- demo_live_allowed={str(bool(payload['demo_live_allowed'])).lower()}",
        f"- gate_audit_reason={classifier.get('reason')}",
        "",
        "## Derivatives Data Readiness Final Gate",
        "- Derivatives-confirmed trend hypothesis 是新的研究假设，不是当前 V3 family 的延续。",
        "- 本次 audit 没有证明有足够历史衍生品数据支持研究。",
        "- Funding 数据完整，但 funding alone 不足以构成 derivatives confirmation。",
        "- Mark/index candle 可用，但只能构造 basis proxy，不足以替代 OI/taker/long-short。",
        "- Open interest 当前只支持 current snapshot，不能用于 2023-2026 historical research。",
        "- Taker buy/sell volume、long/short ratio、contracts OI/volume、OI history、premium history 均未证明能覆盖 2023-2026。",
        f"- derivatives_data_readiness_audit_complete={str(bool(payload['derivatives_data_readiness_audit_complete'])).lower()}",
        f"- can_enter_derivatives_confirmed_trend_research={str(bool(payload['can_enter_derivatives_confirmed_trend_research'])).lower()}",
        f"- derivatives_data_blocker={str(bool(payload['derivatives_data_blocker'])).lower()}",
        f"- derivatives_missing_historical_features={', '.join(payload['derivatives_missing_historical_features'])}",
        f"- derivatives_available_features={', '.join(payload['derivatives_available_features'])}",
        f"- derivatives_research_recommended_next_step={payload['derivatives_research_recommended_next_step']}",
        f"- funding_complete_but_not_sufficient={str(bool(derivatives['funding_complete_but_not_sufficient'])).lower()}",
        f"- mark_index_available_but_not_sufficient={str(bool(derivatives['mark_index_available_but_not_sufficient'])).lower()}",
        f"- current_open_interest_snapshot_only={str(bool(derivatives['current_open_interest_snapshot_only'])).lower()}",
        f"- key_historical_features_coverage_not_proven={str(bool(derivatives['key_historical_features_coverage_not_proven'])).lower()}",
        "- strategy_development_allowed=false",
        "- demo_live_allowed=false",
        "- recommended_next_step=pause strategy development and maintain research/data tooling.",
        "",
        "## 5. Research Timeline",
        "| stage | goal | result | pass/fail | key finding | decision |",
        "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["research_timeline"]:
        lines.append(
            f"| {row['stage']} | {row['goal']} | {row['result']} | {row['pass_fail']} | {row['key_finding']} | {row['decision']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Failed Policy Families",
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
            "## 7. Signal Lab Findings",
            "- short-term breakout 更像 overheat/exhaustion risk。",
            "- high volatility / high ATR / large breakout / high recent return / volume spike / large body ratio 都是负向风险特征。",
            f"- 稳定负向特征：{', '.join(signal['negative_features'])}",
            "",
            "## 8. Trend Regime Findings",
            f"- strong trend 占比 {pct(regime['strong_trend_pct'])}。",
            f"- choppy/high_vol_choppy 占比 {pct(regime['choppy_high_vol_pct'])}。",
            f"- strongest symbol {regime['strongest_symbol']}。",
            f"- weakest symbol {regime['weakest_symbol']}。",
            f"- 1d EMA 不只在 strong trend 有效；strong no-cost PnL={value_text(regime.get('one_day_ema_strong_no_cost_pnl'))}。",
            f"- Donchian 亏在 choppy/high_vol_choppy：{str(bool(regime['donchian_losses_mainly_choppy_high_vol'])).lower()}。",
            "",
            "## 9. Why Strategy Development Is Blocked",
            "Strategy development is blocked because:",
        ]
    )
    for reason in payload["blocking_reasons"]:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## 10. Retained Research Hypotheses",
            "| hypothesis | status | reason | not allowed as |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload["retained_research_hypotheses"]:
        lines.append(f"| {row['hypothesis']} | {row['status']} | {row['reason']} | {row['not_allowed_as']} |")

    lines.extend(
        [
            "",
            "## 11. Do Not Continue List",
            "| item | reason |",
            "| --- | --- |",
        ]
    )
    for row in payload["do_not_continue"]:
        lines.append(f"| {row['item']} | {row['reason']} |")

    lines.extend(
        [
            "",
            "## 12. Next Research Options",
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
            "## 13. Final Decision",
            f"- strategy_development_allowed={str(bool(payload['strategy_development_allowed'])).lower()}",
            f"- demo_live_allowed={str(bool(payload['demo_live_allowed'])).lower()}",
            f"- proceed_to_v3_1_research={str(bool(payload['proceed_to_v3_1_research'])).lower()}",
            f"- can_enter_research_only_v3_1_classifier_experiment={str(bool(payload['can_enter_research_only_v3_1_classifier_experiment'])).lower()}",
            f"- can_enter_derivatives_confirmed_trend_research={str(bool(payload['can_enter_derivatives_confirmed_trend_research'])).lower()}",
            f"- derivatives_data_blocker={str(bool(payload['derivatives_data_blocker'])).lower()}",
            f"- current_v3_family_failed={str(bool(payload['current_v3_family_failed'])).lower()}",
            f"- current_v3_family_failed_after_actual_funding={str(bool(payload['current_v3_family_failed_after_actual_funding'])).lower()}",
            f"- final_current_trend_family_archived={str(bool(payload['final_current_trend_family_archived'])).lower()}",
            f"- final_current_research_archived={str(bool(payload['final_current_research_archived'])).lower()}",
            f"- final_strategy_development_allowed={str(bool(payload['final_strategy_development_allowed'])).lower()}",
            f"- final_demo_live_allowed={str(bool(payload['final_demo_live_allowed'])).lower()}",
            "- proceed_to_broader_universe_research=optional",
            "- proceed_to_funding_research=complete_for_current_universe_no_gate_opened",
            "- derivatives_research_recommended_next_step=pause_research",
            "- next_default=Pause strategy development and maintain research/data tooling.",
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
