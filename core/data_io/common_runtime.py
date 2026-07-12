"""薄 re-export 代理 → scripts/common_runtime.py（前向冻结区真身）。

真身被前向系统 import 闭包引用（research_mr_5m → backtest_mr_5m_compare →
research_trend_baseline → forward_b2_4h），按重构铁律不得移动。
新代码请 import 本模块（core.data_io.common_runtime）。
"""

import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import common_runtime as _real

sys.modules[__name__] = _real
