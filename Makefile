SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
PIP ?= $(PYTHON) -m pip

VT_SYMBOL ?= BTCUSDT_SWAP_OKX.GLOBAL
INTERVAL ?= 1m
START ?= 2025-01-01
END ?= 2026-03-31
TIMEZONE ?= Asia/Shanghai
CHUNK_DAYS ?= 5
SERVER ?= DEMO
CAPITAL ?= 5000
RATE ?= 0.0005
SLIPPAGE_MODE ?= ticks
SLIPPAGE ?= 2
REPORT_DIR ?=
COMPARE_REPORT_DIR ?=
OUTPUT_DIR ?=
STRATEGY_CONFIG ?= config/strategy_default.json
SANITY_CONFIG ?= config/strategy_sanity_min_size.json
MAX_RETRIES ?= 8
THROTTLE_SECONDS ?= 0.35

TEST ?=
CONFIRM ?=
LOG_FILE ?= logs/backtest_okx_mhf.log
TAIL_LINES ?= 80

.PHONY: help
.PHONY: venv install env
.PHONY: doctor inspect-okx check-okx
.PHONY: download-history-dry-run download-history repair-history verify-history
.PHONY: backtest backtest-no-cost backtest-sanity analyze-alpha alpha-sweep
.PHONY: test test-one compile
.PHONY: clean-cache clean-logs clean-reports tail-log

help:
	@printf '%s\n' \
		"OKX vn.py headless CTA commands" \
		"" \
		"Setup:" \
		"  make venv                 Create .venv with python3 -m venv .venv" \
		"  make install              Install vn.py, OKX gateway, sqlite backend, dotenv, pandas, numpy" \
		"  make env                  Create .env from .env.example when .env is missing" \
		"  make doctor               Run local environment checks" \
		"" \
		"OKX checks:" \
		"  make inspect-okx          Inspect local OKX gateway fields without connecting" \
		"  make check-okx SERVER=DEMO  Connect to OKX without placing orders" \
		"" \
		"History:" \
		"  make download-history-dry-run START=2025-01-01 END=2025-01-02 CHUNK_DAYS=1" \
		"  make download-history     Download and verify sqlite history chunks" \
		"  make repair-history       Repair missing sqlite history ranges" \
		"  make verify-history       Verify sqlite coverage and write reports/history_verify_latest.json" \
		"" \
		"Backtest and diagnostics:" \
		"  make backtest             Cost-aware backtest using STRATEGY_CONFIG" \
		"  make backtest-no-cost     No-cost backtest for gross-alpha comparison" \
		"  make backtest-sanity      Conservative min-size sanity backtest using SANITY_CONFIG" \
		"  make analyze-alpha REPORT_DIR=reports/backtest/cost COMPARE_REPORT_DIR=reports/backtest/no_cost" \
		"  make alpha-sweep          Guarded conservative shortlist sweep" \
		"" \
		"Quality and cleanup:" \
		"  make test                 Run unittest discovery" \
		"  make test-one TEST=tests/test_history_time_utils.py" \
		"  make compile              Compile scripts, strategies, tests" \
		"  make clean-cache          Remove Python/tool caches" \
		"  make clean-logs           Remove logs/*.log, keep logs/.gitkeep" \
		"  make clean-reports CONFIRM=1  Remove generated reports, keep reports/.gitkeep" \
		"  make tail-log LOG_FILE=logs/download_okx_history.log" \
		"" \
		"Common overrides:" \
		"  VT_SYMBOL=$(VT_SYMBOL) INTERVAL=$(INTERVAL) START=$(START) END=$(END)" \
		"  TIMEZONE=$(TIMEZONE) CHUNK_DAYS=$(CHUNK_DAYS) SERVER=$(SERVER)" \
		"  CAPITAL=$(CAPITAL) RATE=$(RATE) SLIPPAGE_MODE=$(SLIPPAGE_MODE) SLIPPAGE=$(SLIPPAGE)" \
		"  OUTPUT_DIR=reports/backtest/manual_cost REPORT_DIR=reports/backtest/manual_cost"

venv:
	@if [[ -d .venv ]]; then \
		echo ".venv already exists"; \
	else \
		echo "Creating .venv"; \
		python3 -m venv .venv; \
	fi

install: venv
	@echo "Upgrading pip with $(PIP)"
	$(PIP) install -U pip
	@echo "Installing runtime dependencies"
	$(PIP) install vnpy vnpy_ctastrategy vnpy_okx vnpy_sqlite python-dotenv pandas numpy

env:
	@if [[ -f .env ]]; then \
		echo ".env already exists; not overwriting"; \
	else \
		echo "Creating .env from .env.example"; \
		cp .env.example .env; \
	fi

doctor:
	@echo "Running local environment doctor"
	$(PYTHON) scripts/doctor.py

inspect-okx:
	@echo "Inspecting local OKX gateway fields without connecting"
	$(PYTHON) scripts/inspect_okx_gateway.py

check-okx:
	@echo "Checking OKX connectivity for $(VT_SYMBOL) on $(SERVER)"
	$(PYTHON) scripts/check_okx_connection.py --vt-symbol "$(VT_SYMBOL)" --server "$(SERVER)" --timeout 30

download-history-dry-run:
	@echo "Planning history download without contacting OKX or writing bars"
	$(PYTHON) scripts/download_okx_history.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--interval "$(INTERVAL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--chunk-days "$(CHUNK_DAYS)" \
		--server "$(SERVER)" \
		--source auto \
		--resume \
		--dry-run

download-history:
	@echo "Downloading OKX history into local vn.py sqlite database"
	$(PYTHON) scripts/download_okx_history.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--interval "$(INTERVAL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--chunk-days "$(CHUNK_DAYS)" \
		--server "$(SERVER)" \
		--source auto \
		--resume \
		--save-per-chunk \
		--verify-db \
		--strict-completeness \
		--max-retries "$(MAX_RETRIES)" \
		--throttle-seconds "$(THROTTLE_SECONDS)"

repair-history:
	@echo "Repairing missing OKX history ranges in local sqlite database"
	$(PYTHON) scripts/download_okx_history.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--interval "$(INTERVAL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--chunk-days "$(CHUNK_DAYS)" \
		--server "$(SERVER)" \
		--repair-missing \
		--source auto \
		--resume \
		--save-per-chunk \
		--verify-db \
		--strict-completeness \
		--max-retries "$(MAX_RETRIES)" \
		--throttle-seconds "$(THROTTLE_SECONDS)"

verify-history:
	@echo "Verifying local sqlite history coverage"
	@mkdir -p reports
	$(PYTHON) scripts/verify_okx_history.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--interval "$(INTERVAL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--strict \
		--output-json reports/history_verify_latest.json

backtest:
	@echo "Running cost-aware backtest"
	@args=( \
		scripts/backtest_okx_mhf.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--capital "$(CAPITAL)" \
		--rate "$(RATE)" \
		--slippage-mode "$(SLIPPAGE_MODE)" \
		--slippage "$(SLIPPAGE)" \
		--strategy-config "$(STRATEGY_CONFIG)" \
		--data-check-strict \
	); \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

backtest-no-cost:
	@echo "Running no-cost backtest for gross-alpha comparison"
	@args=( \
		scripts/backtest_okx_mhf.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--capital "$(CAPITAL)" \
		--rate 0 \
		--slippage-mode absolute \
		--slippage 0 \
		--strategy-config "$(STRATEGY_CONFIG)" \
		--data-check-strict \
	); \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

backtest-sanity:
	@if [[ ! -f "$(SANITY_CONFIG)" ]]; then \
		echo "ERROR: sanity config not found: $(SANITY_CONFIG)"; \
		exit 2; \
	fi
	@echo "Running conservative min-size sanity backtest with $(SANITY_CONFIG)"
	@args=( \
		scripts/backtest_okx_mhf.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--capital "$(CAPITAL)" \
		--rate "$(RATE)" \
		--slippage-mode "$(SLIPPAGE_MODE)" \
		--slippage "$(SLIPPAGE)" \
		--strategy-config "$(SANITY_CONFIG)" \
		--data-check-strict \
	); \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

analyze-alpha:
	@if [[ -z "$(strip $(REPORT_DIR))" ]]; then \
		echo "ERROR: REPORT_DIR is required. Example: make analyze-alpha REPORT_DIR=reports/backtest/manual_cost COMPARE_REPORT_DIR=reports/backtest/manual_no_cost"; \
		exit 2; \
	fi
	@echo "Analyzing alpha diagnostics for $(REPORT_DIR)"
	@args=( \
		scripts/analyze_alpha_diagnostics.py \
		--report-dir "$(REPORT_DIR)" \
	); \
	if [[ -n "$(strip $(COMPARE_REPORT_DIR))" ]]; then args+=(--compare-report-dir "$(COMPARE_REPORT_DIR)"); fi; \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

alpha-sweep:
	@if [[ ! -f "$(SANITY_CONFIG)" ]]; then \
		echo "ERROR: sanity config not found: $(SANITY_CONFIG)"; \
		exit 2; \
	fi
	@echo "Running guarded alpha sweep with $(SANITY_CONFIG)"
	@args=( \
		scripts/run_alpha_sweep.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--base-config "$(SANITY_CONFIG)" \
		--capital "$(CAPITAL)" \
		--rate "$(RATE)" \
		--slippage-mode "$(SLIPPAGE_MODE)" \
		--slippage "$(SLIPPAGE)" \
		--max-runs 100 \
		--data-check-strict \
	); \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

test:
	@echo "Running unittest discovery"
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

test-one:
	@if [[ -z "$(strip $(TEST))" ]]; then \
		echo "ERROR: TEST is required. Example: make test-one TEST=tests/test_history_time_utils.py"; \
		exit 2; \
	fi
	@module="$(TEST)"; \
	module="$${module%.py}"; \
	module="$${module//\//.}"; \
	echo "Running unittest $$module"; \
	$(PYTHON) -m unittest "$$module"

compile:
	@echo "Compiling scripts, strategies, and tests"
	$(PYTHON) -m compileall scripts strategies tests

clean-cache:
	@echo "Removing Python and tool cache directories"
	@find scripts strategies tests -type d -name '__pycache__' -prune -exec rm -rf {} +
	@rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__

clean-logs:
	@echo "Removing logs/*.log and preserving logs/.gitkeep"
	@mkdir -p logs
	@find logs -type f -name '*.log' -delete

clean-reports:
	@if [[ "$(CONFIRM)" != "1" ]]; then \
		echo "Refusing to remove reports without CONFIRM=1"; \
		echo "Run: make clean-reports CONFIRM=1"; \
		exit 0; \
	fi
	@echo "Removing generated reports and preserving reports/.gitkeep"
	@mkdir -p reports
	@find reports -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

tail-log:
	@if [[ ! -f "$(LOG_FILE)" ]]; then \
		echo "ERROR: log file not found: $(LOG_FILE)"; \
		exit 2; \
	fi
	@echo "Tailing $(LOG_FILE)"
	tail -n "$(TAIL_LINES)" -f "$(LOG_FILE)"
