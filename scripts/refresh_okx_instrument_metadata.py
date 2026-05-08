#!/usr/bin/env python3
"""Refresh local OKX SWAP instrument metadata from the public instruments API."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config" / "instruments"
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "multisymbol_readiness"
    / "okx_metadata_refresh.json"
)
DEFAULT_INST_IDS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "LINK-USDT-SWAP",
    "DOGE-USDT-SWAP",
]
OKX_PUBLIC_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"
CANONICAL_TEXT_FIELDS = ("okx_inst_id", "product")
METADATA_FIELDS = ("size", "pricetick", "min_volume")
PLACEHOLDER_NOTE = (
    "Placeholder only. Refresh size/pricetick/min_volume from OKX contract metadata "
    "before formal backtests."
)


class MetadataRefreshError(Exception):
    """Raised when one instrument cannot be refreshed from OKX."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Refresh config/instruments metadata from OKX public SWAP instruments."
    )
    parser.add_argument(
        "--inst-ids",
        default=",".join(DEFAULT_INST_IDS),
        help="Comma-separated OKX instrument ids, e.g. BTC-USDT-SWAP,ETH-USDT-SWAP.",
    )
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument(
        "--server",
        choices=("PUBLIC", "REAL", "DEMO"),
        default="PUBLIC",
        help="Metadata source label only. This script uses public REST and never places orders.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Refresh JSON output path. Relative paths resolve from project root.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Fetch and report without writing files.")
    mode.add_argument("--write", action="store_true", help="Write refreshed metadata into config files.")
    return parser.parse_args(argv)


def utc_now_iso() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def resolve_path(path_arg: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a possibly relative path from the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root / path
    return path


def parse_inst_ids(value: str) -> list[str]:
    """Parse a comma/space separated inst_id string while preserving order."""

    seen: set[str] = set()
    inst_ids: list[str] = []
    for token in re.split(r"[\s,]+", value):
        inst_id = token.strip().upper()
        if not inst_id or inst_id in seen:
            continue
        seen.add(inst_id)
        inst_ids.append(inst_id)
    if not inst_ids:
        raise ValueError("--inst-ids must contain at least one OKX instrument id")
    return inst_ids


def okx_inst_id_to_symbol_parts(inst_id: str) -> dict[str, str]:
    """Derive local vn.py symbol fields from an OKX SWAP inst_id."""

    parts = inst_id.upper().split("-")
    if len(parts) < 3 or parts[-1] != "SWAP":
        raise ValueError(f"Only OKX SWAP inst_id values are supported: {inst_id}")
    base = "".join(parts[:-2])
    quote = parts[-2]
    product = parts[-1]
    symbol = f"{base}{quote}_{product}_OKX"
    return {
        "okx_inst_id": inst_id.upper(),
        "vt_symbol": f"{symbol}.GLOBAL",
        "symbol": symbol,
        "exchange": "GLOBAL",
        "product": product,
        "name": inst_id.upper(),
    }


def config_path_for_vt_symbol(config_dir: Path, vt_symbol: str) -> Path:
    """Return the instrument config filename for a vt_symbol."""

    symbol = vt_symbol.split(".", maxsplit=1)[0].lower()
    return config_dir / f"{symbol}.json"


def read_existing_config(path: Path) -> dict[str, Any]:
    """Read an existing config object, returning an empty object on missing files."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def positive_float(value: Any) -> float | None:
    """Convert OKX numeric text to a positive finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def metadata_complete(payload: dict[str, Any]) -> bool:
    """Return whether local metadata is complete enough for formal backtests."""

    if payload.get("needs_okx_contract_metadata_refresh") is True:
        return False
    if any(not str(payload.get(field) or "").strip() for field in CANONICAL_TEXT_FIELDS):
        return False
    return all(positive_float(payload.get(field)) is not None for field in METADATA_FIELDS)


def build_okx_url(inst_id: str) -> str:
    """Build the public OKX instruments endpoint URL for one SWAP inst_id."""

    query = urlencode({"instType": "SWAP", "instId": inst_id})
    return f"{OKX_PUBLIC_INSTRUMENTS_URL}?{query}"


def fetch_okx_instrument(inst_id: str, timeout: float = 20.0) -> dict[str, Any]:
    """Fetch one instrument from OKX public REST."""

    url = build_okx_url(inst_id)
    request = Request(url, headers={"User-Agent": "cta-strategy-metadata-refresh/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise MetadataRefreshError(f"okx_http_error status={exc.code} body={body[:300]}") from exc
    except URLError as exc:
        raise MetadataRefreshError(f"okx_url_error reason={exc.reason!r}") from exc
    except TimeoutError as exc:
        raise MetadataRefreshError("okx_request_timeout") from exc

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise MetadataRefreshError(f"okx_invalid_json: {exc!r}") from exc

    if not isinstance(payload, dict):
        raise MetadataRefreshError("okx_response_root_not_object")
    if str(payload.get("code")) != "0":
        raise MetadataRefreshError(f"okx_error code={payload.get('code')} msg={payload.get('msg')}")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise MetadataRefreshError("okx_response_data_empty")
    first = data[0]
    if not isinstance(first, dict):
        raise MetadataRefreshError("okx_response_data_item_not_object")
    if str(first.get("instId", "")).upper() != inst_id.upper():
        raise MetadataRefreshError(f"okx_response_inst_id_mismatch: {first.get('instId')}")
    return first


def map_okx_metadata(inst_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Map OKX raw fields to local instrument metadata."""

    raw_inst_id = str(raw.get("instId") or inst_id).upper()
    if raw_inst_id != inst_id.upper():
        raise MetadataRefreshError(f"okx_response_inst_id_mismatch: {raw.get('instId')}")
    product = str(raw.get("instType") or "").upper()
    if not product:
        raise MetadataRefreshError("okx_missing_or_invalid_fields: instType")

    size = positive_float(raw.get("ctVal"))
    pricetick = positive_float(raw.get("tickSz"))
    min_volume = positive_float(raw.get("minSz"))
    if min_volume is None:
        min_volume = positive_float(raw.get("lotSz"))

    missing = [
        name
        for name, value in (
            ("ctVal", size),
            ("tickSz", pricetick),
            ("minSz_or_lotSz", min_volume),
        )
        if value is None
    ]
    if missing:
        raise MetadataRefreshError(f"okx_missing_or_invalid_fields: {', '.join(missing)}")

    symbol_parts = okx_inst_id_to_symbol_parts(raw_inst_id)
    return {
        **symbol_parts,
        "size": size,
        "pricetick": pricetick,
        "min_volume": min_volume,
        "product": product,
        "exchange": "GLOBAL",
    }


def build_placeholder_config(inst_id: str) -> dict[str, Any]:
    """Build a placeholder config for an instrument whose metadata is not available."""

    symbol_parts = okx_inst_id_to_symbol_parts(inst_id)
    return {
        "vt_symbol": symbol_parts["vt_symbol"],
        "symbol": symbol_parts["symbol"],
        "exchange": symbol_parts["exchange"],
        "okx_inst_id": symbol_parts["okx_inst_id"],
        "name": symbol_parts["name"],
        "product": symbol_parts["product"],
        "size": None,
        "pricetick": None,
        "min_volume": None,
        "gateway_name": "OKX",
        "history_data": True,
        "needs_okx_contract_metadata_refresh": True,
        "metadata_note": PLACEHOLDER_NOTE,
    }


def merge_successful_metadata(existing: dict[str, Any], mapped: dict[str, Any]) -> dict[str, Any]:
    """Merge refreshed metadata into an existing config object."""

    merged = dict(existing)
    merged.update(
        {
            "vt_symbol": mapped["vt_symbol"],
            "symbol": mapped["symbol"],
            "exchange": mapped["exchange"],
            "okx_inst_id": mapped["okx_inst_id"],
            "name": mapped["name"],
            "product": mapped["product"],
            "size": mapped["size"],
            "pricetick": mapped["pricetick"],
            "min_volume": mapped["min_volume"],
            "gateway_name": merged.get("gateway_name") or "OKX",
            "history_data": merged.get("history_data", True),
            "needs_okx_contract_metadata_refresh": False,
            "metadata_note": "Refreshed from OKX public instruments endpoint.",
            "okx_metadata_source": OKX_PUBLIC_INSTRUMENTS_URL,
            "okx_metadata_refreshed_at": utc_now_iso(),
        }
    )
    return merged


def merge_failed_metadata(existing: dict[str, Any], inst_id: str) -> dict[str, Any]:
    """Preserve existing values on failure, creating placeholders only when needed."""

    if existing:
        merged = dict(existing)
        if not metadata_complete(merged):
            merged.update(
                {
                    **build_placeholder_config(inst_id),
                    **{
                        key: value
                        for key, value in merged.items()
                        if key not in {"size", "pricetick", "min_volume", "needs_okx_contract_metadata_refresh"}
                    },
                    "size": merged.get("size"),
                    "pricetick": merged.get("pricetick"),
                    "min_volume": merged.get("min_volume"),
                    "needs_okx_contract_metadata_refresh": True,
                }
            )
        return merged
    return build_placeholder_config(inst_id)


def write_config(path: Path, payload: dict[str, Any]) -> None:
    """Write one instrument JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_one_inst_id(
    inst_id: str,
    config_dir: Path,
    dry_run: bool,
    fetcher: Callable[[str, float], dict[str, Any]] = fetch_okx_instrument,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Refresh one inst_id and optionally write its config file."""

    symbol_parts = okx_inst_id_to_symbol_parts(inst_id)
    path = config_path_for_vt_symbol(config_dir, symbol_parts["vt_symbol"])
    existing = read_existing_config(path)
    warnings: list[str] = []
    raw_okx_fields: dict[str, Any] = {}
    mapped: dict[str, Any] | None = None
    status = "failed"

    try:
        raw_okx_fields = fetcher(inst_id, timeout)
        mapped = map_okx_metadata(inst_id, raw_okx_fields)
        status = "refreshed"
    except Exception as exc:
        warnings.append(str(exc))

    if mapped is not None:
        candidate = merge_successful_metadata(existing, mapped)
    else:
        candidate = merge_failed_metadata(existing, inst_id)

    wrote_file = False
    if not dry_run:
        write_config(path, candidate)
        wrote_file = True

    complete = metadata_complete(candidate)
    needs_refresh = bool(candidate.get("needs_okx_contract_metadata_refresh"))
    if not complete and "needs_okx_contract_metadata_refresh" not in warnings:
        warnings.append("needs_okx_contract_metadata_refresh")

    return {
        "okx_inst_id": symbol_parts["okx_inst_id"],
        "vt_symbol": symbol_parts["vt_symbol"],
        "symbol": symbol_parts["symbol"],
        "product": candidate.get("product") or symbol_parts["product"],
        "exchange": candidate.get("exchange") or symbol_parts["exchange"],
        "size": candidate.get("size"),
        "pricetick": candidate.get("pricetick"),
        "min_volume": candidate.get("min_volume"),
        "status": status,
        "dry_run": dry_run,
        "wrote_file": wrote_file,
        "config_path": str(path),
        "metadata_complete": complete,
        "needs_okx_contract_metadata_refresh": needs_refresh,
        "warning": "; ".join(warnings) if warnings else "",
        "warnings": warnings,
        "raw_okx_fields": raw_okx_fields,
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate refresh counts."""

    refreshed = [item for item in results if item["status"] == "refreshed"]
    complete = [item for item in results if item["metadata_complete"]]
    needs_refresh = [item for item in results if item["needs_okx_contract_metadata_refresh"]]
    return {
        "requested": len(results),
        "refreshed": len(refreshed),
        "metadata_complete": len(complete),
        "needs_refresh": len(needs_refresh),
        "failed": len(results) - len(refreshed),
    }


def render_report(payload: dict[str, Any]) -> str:
    """Render a Markdown refresh report."""

    summary = payload["summary"]
    lines = [
        "# OKX Instrument Metadata Refresh",
        "",
        "## Summary",
        f"- dry_run={str(bool(payload['dry_run'])).lower()}",
        f"- write={str(bool(payload['write'])).lower()}",
        f"- server={payload['server']}",
        f"- requested={summary['requested']}",
        f"- refreshed={summary['refreshed']}",
        f"- metadata_complete={summary['metadata_complete']}",
        f"- needs_refresh={summary['needs_refresh']}",
        "",
        "## Instruments",
        "| inst_id | vt_symbol | product | size | pricetick | min_volume | dry_run/write | canonical_schema_complete | needs_refresh | warning |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    mode = "dry_run" if payload["dry_run"] else "write"
    for item in payload["instruments"]:
        warning = item.get("warning") or "-"
        lines.append(
            "| {inst_id} | {vt_symbol} | {product} | {size} | {pricetick} | {min_volume} | {mode} | {complete} | {needs} | {warning} |".format(
                inst_id=item["okx_inst_id"],
                vt_symbol=item["vt_symbol"],
                product=item.get("product") or "",
                size=item.get("size"),
                pricetick=item.get("pricetick"),
                min_volume=item.get("min_volume"),
                mode=mode,
                complete=str(bool(item["metadata_complete"])).lower(),
                needs=str(bool(item["needs_okx_contract_metadata_refresh"])).lower(),
                warning=warning.replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "- This script only uses OKX public contract metadata.",
            "- It does not place orders, connect private trading, or write API keys.",
            "- Failed refreshes keep placeholder metadata marked as needing refresh.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_metadata_for_inst_ids(
    inst_ids: list[str],
    config_dir: Path,
    output_json: Path,
    dry_run: bool,
    server: str = "PUBLIC",
    timeout: float = 20.0,
    fetcher: Callable[[str, float], dict[str, Any]] = fetch_okx_instrument,
) -> dict[str, Any]:
    """Refresh all requested instruments and write JSON/Markdown reports."""

    config_dir = resolve_path(config_dir)
    output_json = resolve_path(output_json)
    results = [
        refresh_one_inst_id(
            inst_id=inst_id,
            config_dir=config_dir,
            dry_run=dry_run,
            fetcher=fetcher,
            timeout=timeout,
        )
        for inst_id in inst_ids
    ]
    payload = {
        "generated_at": utc_now_iso(),
        "dry_run": dry_run,
        "write": not dry_run,
        "server": server,
        "endpoint": OKX_PUBLIC_INSTRUMENTS_URL,
        "config_dir": str(config_dir),
        "instruments": results,
        "summary": build_summary(results),
    }

    report_path = output_json.with_name("okx_metadata_refresh_report.md")
    payload["outputs"] = {"json": str(output_json), "report": str(report_path)}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    dry_run = not args.write
    inst_ids = parse_inst_ids(args.inst_ids)
    payload = refresh_metadata_for_inst_ids(
        inst_ids=inst_ids,
        config_dir=resolve_path(args.config_dir),
        output_json=resolve_path(args.output_json),
        dry_run=dry_run,
        server=args.server,
        timeout=args.timeout,
    )

    print("OKX metadata refresh:")
    print(f"- dry_run={str(bool(payload['dry_run'])).lower()}")
    print(f"- output_json={payload['outputs']['json']}")
    print(f"- output_report={payload['outputs']['report']}")
    print(f"- refreshed={payload['summary']['refreshed']}/{payload['summary']['requested']}")
    if payload["summary"]["failed"]:
        print(f"- warnings={payload['summary']['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
