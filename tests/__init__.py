"""测试套件集中 sys.path 注入（2026-07 重构批次2 起）。

历史上全部脚本平铺在 scripts/，测试以 flat import（import xxx）引用。
重构后脚本分域存放但保持 flat import 不变；此处把所有脚本目录一次性
注入 sys.path（unittest discovery 导入 tests 包时最先执行）。

插入顺序 = 列表逆序生效：scripts/（前向冻结区）最后插入、优先级最高，
同名碰撞时冻结区真身胜出。不存在的目录自动跳过（分批迁移期兼容）。
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_SCRIPT_DIRS = [
    _ROOT / "_archive" / "legacy_scripts",
    _ROOT / "_archive" / "mr5m_runner",
    *sorted((_ROOT / "research" / "_closed").glob("*/*/scripts")),
    _ROOT / "audit" / "scripts",
    _ROOT / "data_engineering" / "scripts",
    _ROOT / "core" / "db",
    _ROOT / "core" / "data_io",
    _ROOT / "scripts",
]

for _d in _SCRIPT_DIRS:
    if _d.is_dir():
        _p = str(_d)
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)

# repo 根也入 path，供 `from core.data_io import ...` 包式引用
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
