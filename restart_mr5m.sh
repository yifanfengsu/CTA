#!/bin/bash
cd /run-project/vnpy_strategy_test/CTA
PYTHONUNBUFFERED=1 exec .venv/bin/python -u scripts/run_mr_5m_direct.py
