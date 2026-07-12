"""薄 re-export 代理 → scripts/history_time_utils.py（前向冻结区真身）。

真身被前向系统 import 闭包引用（backtest_mr_5m_compare / research_mr_5m），
按重构铁律不得移动。新代码请 import 本模块（core.data_io.history_time_utils）。
"""

import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import history_time_utils as _real

sys.modules[__name__] = _real
