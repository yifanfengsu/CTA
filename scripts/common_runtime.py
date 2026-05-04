"""Shared runtime helpers for headless command-line scripts."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
LOG_DIR: Path = PROJECT_ROOT / "logs"
VNTRADER_DIR: Path = PROJECT_ROOT / ".vntrader"
VNTRADER_LOG_DIR: Path = VNTRADER_DIR / "log"

_RESERVED_LOG_RECORD_FIELDS: set[str] = {
    "args",
    "asctime",
    "created",
    "event",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def ensure_headless_runtime() -> None:
    """Prepare project-local runtime folders before importing vn.py modules."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VNTRADER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)


def to_jsonable(value: Any) -> Any:
    """Convert Python objects into JSON-safe values for logs and reports."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


class JsonFormatter(logging.Formatter):
    """Emit structured JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record as JSON."""

        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        event_name: str | None = getattr(record, "event", None)
        if event_name:
            payload["event"] = event_name

        fields: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            fields[key] = to_jsonable(value)

        if fields:
            payload["fields"] = fields

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(script_name: str, verbose: bool = False) -> logging.Logger:
    """Configure a JSON logger for console and file output."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file: Path = LOG_DIR / f"{script_name}.log"

    logger: logging.Logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonFormatter()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """Write one structured log event."""

    safe_fields: dict[str, Any] = {}
    for key, value in fields.items():
        normalized_key = f"field_{key}" if key in _RESERVED_LOG_RECORD_FIELDS else key
        safe_fields[normalized_key] = value

    logger.log(level, message, extra={"event": event, **safe_fields})


def print_json_block(title: str, payload: Any) -> None:
    """Pretty-print JSON output for human inspection."""

    print(title)
    print(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))


def normalize_text(text: str) -> str:
    """Normalize a label for fuzzy field matching."""

    return "".join(character.lower() for character in text if character.isalnum())


def mask_secret(value: str, visible_suffix: int = 4) -> str:
    """Mask a secret while leaving a short suffix for confirmation."""

    if not value:
        return ""

    if len(value) <= visible_suffix:
        return "*" * len(value)

    masked_length: int = len(value) - visible_suffix
    return f"{'*' * masked_length}{value[-visible_suffix:]}"
