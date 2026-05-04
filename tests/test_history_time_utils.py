from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from history_time_utils import build_half_open_chunks, expected_bar_count, parse_history_range


class HistoryTimeUtilsTest(unittest.TestCase):
    def test_parse_date_only_as_db_timezone(self) -> None:
        history_range = parse_history_range(
            start_arg="2025-01-01",
            end_arg="2025-01-03",
            interval_delta=timedelta(minutes=1),
            timezone_name="Asia/Shanghai",
        )

        self.assertEqual(history_range.start, datetime(2025, 1, 1, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")))
        self.assertEqual(history_range.end_exclusive, datetime(2025, 1, 4, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")))
        self.assertEqual(history_range.end_display, datetime(2025, 1, 3, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai")))
        self.assertEqual(history_range.start_utc, datetime(2024, 12, 31, 16, 0, tzinfo=timezone.utc))
        self.assertEqual(history_range.end_exclusive_utc, datetime(2025, 1, 3, 16, 0, tzinfo=timezone.utc))
        self.assertEqual(expected_bar_count(history_range), 4320)

    def test_half_open_chunks_no_2359_gap(self) -> None:
        history_range = parse_history_range(
            start_arg="2025-01-01",
            end_arg="2025-01-03",
            interval_delta=timedelta(minutes=1),
            timezone_name="Asia/Shanghai",
        )

        chunks = build_half_open_chunks(history_range, chunk_days=1)

        self.assertEqual(
            [(chunk.start, chunk.end_exclusive) for chunk in chunks],
            [
                (
                    datetime(2025, 1, 1, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                    datetime(2025, 1, 2, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                ),
                (
                    datetime(2025, 1, 2, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                    datetime(2025, 1, 3, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                ),
                (
                    datetime(2025, 1, 3, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                    datetime(2025, 1, 4, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                ),
            ],
        )
        self.assertEqual(chunks[0].end_display, datetime(2025, 1, 1, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai")))
        self.assertEqual(chunks[1].end_display, datetime(2025, 1, 2, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai")))
        self.assertEqual(chunks[2].end_display, datetime(2025, 1, 3, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai")))


if __name__ == "__main__":
    unittest.main()
