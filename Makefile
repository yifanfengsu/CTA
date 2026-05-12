SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
PIP ?= $(PYTHON) -m pip

VT_SYMBOL ?= BTCUSDT_SWAP_OKX.GLOBAL
SYMBOLS ?= BTCUSDT_SWAP_OKX.GLOBAL ETHUSDT_SWAP_OKX.GLOBAL SOLUSDT_SWAP_OKX.GLOBAL LINKUSDT_SWAP_OKX.GLOBAL DOGEUSDT_SWAP_OKX.GLOBAL
INST_IDS ?= BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP,LINK-USDT-SWAP,DOGE-USDT-SWAP
INTERVAL ?= 1m
START ?= 2023-01-01
END ?= 2026-03-31
TIMEZONE ?= Asia/Shanghai
CHUNK_DAYS ?= 3
SERVER ?= DEMO
CAPITAL ?= 5000
RATE ?= 0.0005
SLIPPAGE_MODE ?= ticks
SLIPPAGE ?= 2
REPORT_DIR ?=
COMPARE_REPORT_DIR ?=
OUTPUT_DIR ?=
SIGNAL_TRACE_PATH ?=
FORMAT ?=
SPLIT ?= train
HORIZONS ?= 5,15,30,60,120
ENTRY_HORIZONS ?= 15,30,60,120
FEATURE_HORIZONS ?= 15,30,60,120
HTF_HORIZONS ?= 60,120,240,480
FEATURE_BINS ?= 5
FEATURE_MIN_COUNT ?= 50
FEATURE_LIST ?=
ENTRY_MAX_WAIT_BARS ?= 10
STOP_ATR_GRID ?= 1.0,1.5,2.0,2.5,3.0,4.0
TP_ATR_GRID ?= 1.5,2.0,2.5,3.0,4.0,5.0
HTF_STOP_ATR_GRID ?= 1.5,2.0,2.5,3.0,4.0
HTF_TP_ATR_GRID ?= 2.0,3.0,4.0,5.0,6.0
HTF_COOLDOWN_BARS_5M ?= 6
HTF_MAX_SIGNALS ?=
HTF_OUTPUT_DIR ?= reports/research/htf_signals/$(SPLIT)
HTF_TRAIN_DIR ?= reports/research/htf_signals/train
HTF_VALIDATION_DIR ?= reports/research/htf_signals/validation
HTF_OOS_DIR ?= reports/research/htf_signals/oos
HTF_COMPARE_OUTPUT_DIR ?= reports/research/htf_compare
TREND_OUTPUT_DIR ?= reports/research/trend_following_v2/$(SPLIT)
TREND_TRAIN_DIR ?= reports/research/trend_following_v2/train
TREND_VALIDATION_DIR ?= reports/research/trend_following_v2/validation
TREND_OOS_DIR ?= reports/research/trend_following_v2/oos
TREND_COMPARE_OUTPUT_DIR ?= reports/research/trend_following_v2_compare
TREND_MAX_RUNS ?=
TREND_V3_OUTPUT_DIR ?= reports/research/trend_following_v3/$(SPLIT)
TREND_V3_TRAIN_DIR ?= reports/research/trend_following_v3/train
TREND_V3_VALIDATION_DIR ?= reports/research/trend_following_v3/validation
TREND_V3_OOS_DIR ?= reports/research/trend_following_v3/oos
TREND_V3_COMPARE_OUTPUT_DIR ?= reports/research/trend_following_v3_compare
TREND_V3_POSTMORTEM_OUTPUT_DIR ?= reports/research/trend_following_v3_postmortem
TREND_V3_MAX_RUNS ?=
TREND_V3_EXT_SPLIT ?= train_ext
EXT_SPLIT = $(if $(filter train,$(SPLIT)),$(TREND_V3_EXT_SPLIT),$(SPLIT))
TREND_V3_EXT_OUTPUT_DIR ?= reports/research/trend_following_v3_extended/$(EXT_SPLIT)
TREND_V3_EXT_TRAIN_DIR ?= reports/research/trend_following_v3_extended/train_ext
TREND_V3_EXT_VALIDATION_DIR ?= reports/research/trend_following_v3_extended/validation_ext
TREND_V3_EXT_OOS_DIR ?= reports/research/trend_following_v3_extended/oos_ext
TREND_V3_EXT_COMPARE_OUTPUT_DIR ?= reports/research/trend_following_v3_extended_compare
FUNDING_OUTPUT_DIR ?= data/funding/okx
FUNDING_REPORTS_DIR ?= reports/research/funding
TREND_V3_ACTUAL_FUNDING_OUTPUT_DIR ?= reports/research/trend_following_v3_actual_funding
EXTENDED_HISTORY_OUTPUT_DIR ?= reports/research/extended_history_availability
TREND_REGIME_OUTPUT_DIR ?= reports/research/trend_regime_diagnostics
TREND_OPPORTUNITY_OUTPUT_DIR ?= reports/research/trend_opportunity_map
TREND_CAPTURE_EXIT_OUTPUT_DIR ?= reports/research/trend_capture_exit_convexity
TREND_HEALTH_EXIT_OUTPUT_DIR ?= reports/research/trend_health_state_exit
TREND_ENTRY_TIMING_OUTPUT_DIR ?= reports/research/trend_entry_timing
TREND_ENTRY_TIMING_POSTMORTEM_OUTPUT_DIR ?= reports/research/trend_entry_timing_postmortem
BREADTH_PHASE15_OUTPUT_DIR ?= reports/research/cross_symbol_breadth_phase15
RESEARCH_DOSSIER_OUTPUT_DIR ?= reports/research/research_decision_dossier
EXTERNAL_REGIME_OUTPUT_DIR ?= reports/research/external_regime_feasibility
EXTERNAL_REGIME_CLASSIFIER_OUTPUT_DIR ?= reports/research/external_regime_classifier
EXTERNAL_REGIME_GATE_AUDIT_OUTPUT_DIR ?= reports/research/external_regime_classifier_gate_audit
VSVCB_OUTPUT_DIR ?= reports/research/vsvcb_v1
VSVCB_POSTMORTEM_OUTPUT_DIR ?= reports/research/vsvcb_v1_postmortem
DERIVATIVES_DATA_READINESS_OUTPUT_DIR ?= reports/research/derivatives_data_readiness
CSRB_OUTPUT_DIR ?= reports/research/csrb_v1
CSRB_POSTMORTEM_OUTPUT_DIR ?= reports/research/csrb_v1_postmortem
TRAIN_DIR ?=
VALIDATION_DIR ?=
OOS_DIR ?=
STRATEGY_CONFIG ?= config/strategy_default.json
SANITY_CONFIG ?= config/strategy_sanity_min_size.json
MAX_RUNS ?= 100
MAX_RETRIES ?= 8
THROTTLE_SECONDS ?= 0.35

TEST ?=
CONFIRM ?=
LOG_FILE ?= logs/backtest_okx_mhf.log
TAIL_LINES ?= 80

.PHONY: help
.PHONY: venv install env
.PHONY: doctor inspect-okx check-okx
.PHONY: download-history-dry-run download-history repair-history verify-history refresh-okx-metadata-dry-run refresh-okx-metadata download-history-batch-dry-run download-history-batch verify-history-batch download-funding-dry-run download-funding verify-funding verify-funding-allow-partial import-funding-csv probe-funding-source download-funding-historical-dry-run download-funding-historical analyze-trend-v3-funding
.PHONY: backtest backtest-no-cost backtest-trace backtest-sanity analyze-alpha analyze-trades analyze-signals research-entry research-features compare-features research-htf compare-htf research-trend-v2 compare-trend-v2 research-trend-v3 compare-trend-v3 research-trend-v3-extended compare-trend-v3-extended postmortem-trend-v3 diagnose-trend-regimes research-trend-opportunity-map research-trend-exit-convexity research-trend-health-exit research-trend-entry-timing postmortem-trend-entry-timing research-breadth-phase15 research-dossier audit-multisymbol audit-extended-history audit-external-regime research-external-regime-classifier audit-external-regime-gates research-vsvcb-v1 postmortem-vsvcb-v1 audit-derivatives-data research-csrb-v1 postmortem-csrb-v1 alpha-sweep ablation
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
		"  make refresh-okx-metadata-dry-run  Fetch OKX metadata report without writing instruments" \
		"  make refresh-okx-metadata          Fetch OKX metadata and update instrument files" \
		"  make download-history-dry-run START=2025-01-01 END=2025-01-02 CHUNK_DAYS=1" \
		"  make download-history     Download and verify sqlite history chunks" \
		"  make download-history-batch-dry-run START=2025-01-01 END=2025-01-07" \
		"  make download-history-batch START=2025-01-01 END=2026-03-31 CHUNK_DAYS=3" \
		"  make repair-history       Repair missing sqlite history ranges" \
		"  make verify-history       Verify sqlite coverage and write reports/history_verify_latest.json" \
		"  make verify-history-batch START=2025-01-01 END=2026-03-31" \
		"  make download-funding-dry-run  Plan OKX public funding history download" \
		"  make download-funding          Download OKX public funding history CSVs" \
		"  make verify-funding            Verify funding CSV coverage and gaps" \
		"  make verify-funding-allow-partial  Verify funding CSVs but return 0 for audited partial data" \
		"  make import-funding-csv INPUT=... INST_ID=...  Import manually downloaded OKX funding CSV" \
		"  make probe-funding-source      Probe OKX historical market data funding query endpoint" \
		"  make download-funding-historical-dry-run  Plan OKX historical funding file download" \
		"  make download-funding-historical          Download OKX historical funding files when probe confirms endpoint" \
		"  make analyze-trend-v3-funding  Analyze Trend V3 extended trades with actual funding" \
		"" \
		"Backtest and diagnostics:" \
		"  make backtest             Cost-aware backtest using STRATEGY_CONFIG" \
		"  make backtest-no-cost     No-cost backtest for gross-alpha comparison" \
		"  make backtest-trace       No-cost backtest plus signal_trace.csv export" \
		"  make backtest-sanity      Conservative min-size sanity backtest using SANITY_CONFIG" \
		"  make analyze-alpha REPORT_DIR=reports/backtest/cost COMPARE_REPORT_DIR=reports/backtest/no_cost" \
		"  make analyze-trades REPORT_DIR=reports/backtest/main_no_cost_20250101_20260331" \
		"  make analyze-signals REPORT_DIR=reports/research/trace_2025q1" \
		"  make research-entry REPORT_DIR=reports/research/trace_train" \
		"  make research-features REPORT_DIR=reports/research/trace_train" \
		"  make compare-features TRAIN_DIR=reports/research/trace_train/signal_feature_research VALIDATION_DIR=reports/research/trace_validation/signal_feature_research OOS_DIR=reports/research/trace_oos/signal_feature_research" \
		"  make research-htf SPLIT=train" \
		"  make compare-htf" \
		"  make research-trend-v2 SPLIT=train" \
		"  make compare-trend-v2" \
		"  make research-trend-v3 SPLIT=train" \
		"  make compare-trend-v3" \
		"  make research-trend-v3-extended SPLIT=train_ext" \
		"  make compare-trend-v3-extended" \
		"  make postmortem-trend-v3" \
		"  make diagnose-trend-regimes  Diagnose 2023-2026 trend regimes and V3 trade attribution" \
		"  make research-trend-opportunity-map  Build ex-post trend opportunity diagnostics" \
		"  make research-trend-exit-convexity  Research legacy trend capture and counterfactual exits" \
		"  make research-trend-health-exit  Research trend health-state exits offline" \
		"  make research-trend-entry-timing  Research early-entry timing candidates offline" \
		"  make postmortem-trend-entry-timing  Postmortem and audit Trend Entry Timing candidates" \
		"  make research-breadth-phase15  Diagnose cross-symbol breadth near-miss lead" \
		"  make research-dossier  Archive final research decision and blocked directions" \
		"  make audit-external-regime  Audit research-only external regime classifier feasibility" \
		"  make research-external-regime-classifier  Research external regime classifier filters offline" \
		"  make audit-external-regime-gates  Audit external regime classifier gate consistency" \
		"  make research-vsvcb-v1  Run VSVCB-v1 Phase 1 event study and fixed-hold benchmark" \
		"  make postmortem-vsvcb-v1  Run VSVCB-v1 Phase 1 failure postmortem" \
		"  make audit-derivatives-data  Audit OKX public derivatives data readiness" \
		"  make research-csrb-v1  Run CSRB-v1 Phase 1 session breakout event study" \
		"  make postmortem-csrb-v1  Run CSRB-v1 Phase 1 postmortem and control audit" \
		"  make audit-multisymbol  Audit multi-symbol metadata and sqlite readiness" \
		"  make audit-extended-history  Audit long-history availability and download plan" \
		"  make alpha-sweep          Guarded conservative shortlist sweep" \
		"  make ablation            Entry-filter ablation diagnostics" \
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
		"  VT_SYMBOL=$(VT_SYMBOL) SYMBOLS=\"$(SYMBOLS)\" INST_IDS=$(INST_IDS)" \
		"  INTERVAL=$(INTERVAL) START=$(START) END=$(END)" \
		"  TIMEZONE=$(TIMEZONE) CHUNK_DAYS=$(CHUNK_DAYS) SERVER=$(SERVER)" \
		"  CAPITAL=$(CAPITAL) RATE=$(RATE) SLIPPAGE_MODE=$(SLIPPAGE_MODE) SLIPPAGE=$(SLIPPAGE)" \
		"  OUTPUT_DIR=reports/backtest/manual_cost REPORT_DIR=reports/backtest/manual_cost" \
		"  SIGNAL_TRACE_PATH=$(SIGNAL_TRACE_PATH) HORIZONS=$(HORIZONS)" \
		"  ENTRY_HORIZONS=$(ENTRY_HORIZONS) ENTRY_MAX_WAIT_BARS=$(ENTRY_MAX_WAIT_BARS)" \
		"  STOP_ATR_GRID=$(STOP_ATR_GRID) TP_ATR_GRID=$(TP_ATR_GRID)" \
		"  FEATURE_HORIZONS=$(FEATURE_HORIZONS) FEATURE_BINS=$(FEATURE_BINS) FEATURE_MIN_COUNT=$(FEATURE_MIN_COUNT)" \
		"  FEATURE_LIST=$(FEATURE_LIST) TRAIN_DIR=$(TRAIN_DIR) VALIDATION_DIR=$(VALIDATION_DIR) OOS_DIR=$(OOS_DIR)" \
		"  HTF_HORIZONS=$(HTF_HORIZONS) HTF_OUTPUT_DIR=$(HTF_OUTPUT_DIR) HTF_COOLDOWN_BARS_5M=$(HTF_COOLDOWN_BARS_5M)" \
		"  TREND_OUTPUT_DIR=$(TREND_OUTPUT_DIR) TREND_MAX_RUNS=$(TREND_MAX_RUNS)" \
		"  TREND_V3_OUTPUT_DIR=$(TREND_V3_OUTPUT_DIR) TREND_V3_MAX_RUNS=$(TREND_V3_MAX_RUNS)" \
		"  TREND_V3_EXT_OUTPUT_DIR=$(TREND_V3_EXT_OUTPUT_DIR) TREND_V3_EXT_COMPARE_OUTPUT_DIR=$(TREND_V3_EXT_COMPARE_OUTPUT_DIR)" \
		"  FUNDING_OUTPUT_DIR=$(FUNDING_OUTPUT_DIR) FUNDING_REPORTS_DIR=$(FUNDING_REPORTS_DIR)" \
		"  TREND_V3_POSTMORTEM_OUTPUT_DIR=$(TREND_V3_POSTMORTEM_OUTPUT_DIR)" \
		"  EXTENDED_HISTORY_OUTPUT_DIR=$(EXTENDED_HISTORY_OUTPUT_DIR) TREND_REGIME_OUTPUT_DIR=$(TREND_REGIME_OUTPUT_DIR) TREND_OPPORTUNITY_OUTPUT_DIR=$(TREND_OPPORTUNITY_OUTPUT_DIR) TREND_CAPTURE_EXIT_OUTPUT_DIR=$(TREND_CAPTURE_EXIT_OUTPUT_DIR) TREND_HEALTH_EXIT_OUTPUT_DIR=$(TREND_HEALTH_EXIT_OUTPUT_DIR) TREND_ENTRY_TIMING_OUTPUT_DIR=$(TREND_ENTRY_TIMING_OUTPUT_DIR) TREND_ENTRY_TIMING_POSTMORTEM_OUTPUT_DIR=$(TREND_ENTRY_TIMING_POSTMORTEM_OUTPUT_DIR) BREADTH_PHASE15_OUTPUT_DIR=$(BREADTH_PHASE15_OUTPUT_DIR)" \
		"  RESEARCH_DOSSIER_OUTPUT_DIR=$(RESEARCH_DOSSIER_OUTPUT_DIR) EXTERNAL_REGIME_OUTPUT_DIR=$(EXTERNAL_REGIME_OUTPUT_DIR)" \
		"  EXTERNAL_REGIME_CLASSIFIER_OUTPUT_DIR=$(EXTERNAL_REGIME_CLASSIFIER_OUTPUT_DIR) EXTERNAL_REGIME_GATE_AUDIT_OUTPUT_DIR=$(EXTERNAL_REGIME_GATE_AUDIT_OUTPUT_DIR)" \
		"  VSVCB_OUTPUT_DIR=$(VSVCB_OUTPUT_DIR) DERIVATIVES_DATA_READINESS_OUTPUT_DIR=$(DERIVATIVES_DATA_READINESS_OUTPUT_DIR) CSRB_OUTPUT_DIR=$(CSRB_OUTPUT_DIR) CSRB_POSTMORTEM_OUTPUT_DIR=$(CSRB_POSTMORTEM_OUTPUT_DIR)" \
		"  SPLIT=$(SPLIT) MAX_RUNS=$(MAX_RUNS)"

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

refresh-okx-metadata-dry-run:
	@echo "Refreshing OKX instrument metadata report without writing config files"
	$(PYTHON) scripts/refresh_okx_instrument_metadata.py \
		--inst-ids "$(INST_IDS)" \
		--server "$(SERVER)" \
		--dry-run

refresh-okx-metadata:
	@echo "Refreshing OKX instrument metadata into config/instruments"
	$(PYTHON) scripts/refresh_okx_instrument_metadata.py \
		--inst-ids "$(INST_IDS)" \
		--server "$(SERVER)" \
		--write

download-history-batch-dry-run:
	@for symbol in $(SYMBOLS); do \
		echo "Planning history download for $$symbol"; \
		$(PYTHON) scripts/download_okx_history.py \
			--vt-symbol "$$symbol" \
			--interval "$(INTERVAL)" \
			--start "$(START)" \
			--end "$(END)" \
			--timezone "$(TIMEZONE)" \
			--chunk-days "$(CHUNK_DAYS)" \
			--server "$(SERVER)" \
			--source auto \
			--resume \
			--dry-run; \
	done

download-history-batch:
	@for symbol in $(SYMBOLS); do \
		echo "Downloading history for $$symbol"; \
		$(PYTHON) scripts/download_okx_history.py \
			--vt-symbol "$$symbol" \
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
			--strict-completeness; \
	done

verify-history-batch:
	@mkdir -p reports/history_verify
	@for symbol in $(SYMBOLS); do \
		safe_symbol="$${symbol//./_}"; \
		safe_symbol="$${safe_symbol//\//_}"; \
		echo "Verifying history for $$symbol"; \
		$(PYTHON) scripts/verify_okx_history.py \
			--vt-symbol "$$symbol" \
			--interval "$(INTERVAL)" \
			--start "$(START)" \
			--end "$(END)" \
			--timezone "$(TIMEZONE)" \
			--strict \
			--output-json "reports/history_verify/$${safe_symbol}_$(START)_$(END).json"; \
	done

download-funding-dry-run:
	@echo "Planning OKX public funding history download without writing funding CSVs"
	$(PYTHON) scripts/download_okx_funding_history.py \
		--inst-ids "$(INST_IDS)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--output-dir "$(FUNDING_OUTPUT_DIR)" \
		--reports-dir "$(FUNDING_REPORTS_DIR)" \
		--dry-run

download-funding:
	@echo "Downloading OKX public funding history"
	$(PYTHON) scripts/download_okx_funding_history.py \
		--inst-ids "$(INST_IDS)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--output-dir "$(FUNDING_OUTPUT_DIR)" \
		--reports-dir "$(FUNDING_REPORTS_DIR)" \
		--throttle-seconds "$(THROTTLE_SECONDS)" \
		--max-retries "$(MAX_RETRIES)"

verify-funding:
	@echo "Verifying OKX funding history coverage"
	$(PYTHON) scripts/verify_okx_funding_history.py \
		--funding-dir "$(FUNDING_OUTPUT_DIR)" \
		--inst-ids "$(INST_IDS)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--output-dir "$(FUNDING_REPORTS_DIR)"

verify-funding-allow-partial:
	@echo "Verifying OKX funding history coverage; partial data is allowed only for audit/reporting"
	$(PYTHON) scripts/verify_okx_funding_history.py \
		--funding-dir "$(FUNDING_OUTPUT_DIR)" \
		--inst-ids "$(INST_IDS)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)" \
		--output-dir "$(FUNDING_REPORTS_DIR)" \
		--allow-partial

import-funding-csv:
	@if [[ -z "$${INPUT:-}" && -z "$${INPUTS:-}" && -z "$${INPUT_DIR:-}" ]]; then \
		echo "INPUT, INPUTS, or INPUT_DIR is required, e.g. make import-funding-csv INPUT=/path/funding.csv INST_ID=BTC-USDT-SWAP" >&2; \
		exit 2; \
	fi
	@if [[ -z "$${INST_ID:-}" ]]; then \
		echo "INST_ID is required, e.g. make import-funding-csv INPUT=/path/funding.csv INST_ID=BTC-USDT-SWAP" >&2; \
		exit 2; \
	fi
	@echo "Importing manually downloaded OKX funding CSV"
	@args=(scripts/import_okx_funding_csv.py \
		--inst-id "$${INST_ID}" \
		--output-dir "$(FUNDING_OUTPUT_DIR)" \
		--reports-dir "$(FUNDING_REPORTS_DIR)" \
		--start "$(START)" \
		--end "$(END)" \
		--timezone "$(TIMEZONE)"); \
	if [[ -n "$${INPUT:-}" ]]; then args+=(--input "$${INPUT}"); fi; \
	if [[ -n "$${INPUTS:-}" ]]; then args+=(--inputs "$${INPUTS}"); fi; \
	if [[ -n "$${INPUT_DIR:-}" ]]; then args+=(--input-dir "$${INPUT_DIR}"); fi; \
	if [[ "$${APPEND:-0}" == "1" || "$${APPEND:-false}" == "true" ]]; then args+=(--append); else args+=(--overwrite); fi; \
	$(PYTHON) "$${args[@]}"

probe-funding-source:
	@echo "Probing OKX historical market data funding source"
	$(PYTHON) scripts/probe_okx_historical_market_data.py \
		--inst-ids "$(INST_IDS)" \
		--start "$(START)" \
		--end "$(END)" \
		--dry-run

download-funding-historical-dry-run:
	@echo "Planning OKX historical funding file download"
	$(PYTHON) scripts/download_okx_historical_funding_files.py \
		--inst-ids "$(INST_IDS)" \
		--start "$(START)" \
		--end "$(END)" \
		--dry-run

download-funding-historical:
	@echo "Downloading OKX historical funding files"
	$(PYTHON) scripts/download_okx_historical_funding_files.py \
		--inst-ids "$(INST_IDS)" \
		--start "$(START)" \
		--end "$(END)"

analyze-trend-v3-funding:
	@echo "Analyzing Trend V3 extended trades with actual OKX funding"
	$(PYTHON) scripts/analyze_trend_v3_actual_funding.py \
		--funding-dir "$(FUNDING_OUTPUT_DIR)" \
		--trend-v3-extended-dir "reports/research/trend_following_v3_extended" \
		--compare-dir "$(TREND_V3_EXT_COMPARE_OUTPUT_DIR)" \
		--output-dir "$(TREND_V3_ACTUAL_FUNDING_OUTPUT_DIR)" \
		--timezone "$(TIMEZONE)"

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

backtest-trace:
	@echo "Running no-cost backtest with signal trace export"
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
		--export-signal-trace \
	); \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	if [[ -n "$(strip $(SIGNAL_TRACE_PATH))" ]]; then args+=(--signal-trace-path "$(SIGNAL_TRACE_PATH)"); fi; \
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

analyze-trades:
	@if [[ -z "$(strip $(REPORT_DIR))" ]]; then \
		echo "ERROR: REPORT_DIR is required. Example: make analyze-trades REPORT_DIR=reports/backtest/main_no_cost_20250101_20260331"; \
		exit 2; \
	fi
	@echo "Analyzing trade attribution for $(REPORT_DIR)"
	@args=( \
		scripts/analyze_trade_attribution.py \
		--report-dir "$(REPORT_DIR)" \
		--timezone "$(TIMEZONE)" \
	); \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	if [[ -n "$(strip $(FORMAT))" ]]; then args+=(--format "$(FORMAT)"); fi; \
	$(PYTHON) "$${args[@]}"

analyze-signals:
	@if [[ -z "$(strip $(REPORT_DIR))" ]]; then \
		echo "ERROR: REPORT_DIR is required. Example: make analyze-signals REPORT_DIR=reports/research/trace_2025q1"; \
		exit 2; \
	fi
	@echo "Analyzing signal outcomes for $(REPORT_DIR)"
	@args=( \
		scripts/analyze_signal_outcomes.py \
		--report-dir "$(REPORT_DIR)" \
		--timezone "$(TIMEZONE)" \
		--horizons "$(HORIZONS)" \
	); \
	if [[ -n "$(strip $(SIGNAL_TRACE_PATH))" ]]; then args+=(--signal-trace "$(SIGNAL_TRACE_PATH)"); fi; \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

research-entry:
	@if [[ -z "$(strip $(REPORT_DIR))" ]]; then \
		echo "ERROR: REPORT_DIR is required. Example: make research-entry REPORT_DIR=reports/research/trace_train"; \
		exit 2; \
	fi
	@echo "Researching offline entry policies for $(REPORT_DIR)"
	@args=( \
		scripts/research_entry_policies.py \
		--report-dir "$(REPORT_DIR)" \
		--timezone "$(TIMEZONE)" \
		--horizons "$(ENTRY_HORIZONS)" \
		--max-wait-bars "$(ENTRY_MAX_WAIT_BARS)" \
		--stop-atr-grid "$(STOP_ATR_GRID)" \
		--tp-atr-grid "$(TP_ATR_GRID)" \
	); \
	if [[ -n "$(strip $(SIGNAL_TRACE_PATH))" ]]; then args+=(--signal-trace "$(SIGNAL_TRACE_PATH)"); fi; \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

research-features:
	@if [[ -z "$(strip $(REPORT_DIR))" ]]; then \
		echo "ERROR: REPORT_DIR is required. Example: make research-features REPORT_DIR=reports/research/trace_train"; \
		exit 2; \
	fi
	@echo "Researching signal features for $(REPORT_DIR)"
	@args=( \
		scripts/research_signal_features.py \
		--report-dir "$(REPORT_DIR)" \
		--timezone "$(TIMEZONE)" \
		--horizons "$(FEATURE_HORIZONS)" \
		--bins "$(FEATURE_BINS)" \
		--min-count "$(FEATURE_MIN_COUNT)" \
	); \
	if [[ -n "$(strip $(SIGNAL_TRACE_PATH))" ]]; then args+=(--signal-trace "$(SIGNAL_TRACE_PATH)"); fi; \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	if [[ -n "$(strip $(FEATURE_LIST))" ]]; then args+=(--feature-list "$(FEATURE_LIST)"); fi; \
	$(PYTHON) "$${args[@]}"

compare-features:
	@if [[ -z "$(strip $(TRAIN_DIR))" || -z "$(strip $(VALIDATION_DIR))" || -z "$(strip $(OOS_DIR))" ]]; then \
		echo "ERROR: TRAIN_DIR, VALIDATION_DIR, and OOS_DIR are required."; \
		echo "Example: make compare-features TRAIN_DIR=reports/research/trace_train/signal_feature_research VALIDATION_DIR=reports/research/trace_validation/signal_feature_research OOS_DIR=reports/research/trace_oos/signal_feature_research"; \
		exit 2; \
	fi
	@echo "Comparing signal feature research across train/validation/oos"
	@args=( \
		scripts/compare_signal_feature_research.py \
		--train-dir "$(TRAIN_DIR)" \
		--validation-dir "$(VALIDATION_DIR)" \
		--oos-dir "$(OOS_DIR)" \
	); \
	if [[ -n "$(strip $(OUTPUT_DIR))" ]]; then args+=(--output-dir "$(OUTPUT_DIR)"); fi; \
	$(PYTHON) "$${args[@]}"

research-htf:
	@echo "Researching HTF signal candidates split=$(SPLIT) output=$(HTF_OUTPUT_DIR)"
	@args=( \
		scripts/research_htf_signals.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--split "$(SPLIT)" \
		--timezone "$(TIMEZONE)" \
		--output-dir "$(HTF_OUTPUT_DIR)" \
		--horizons "$(HTF_HORIZONS)" \
		--stop-atr-grid "$(HTF_STOP_ATR_GRID)" \
		--tp-atr-grid "$(HTF_TP_ATR_GRID)" \
		--cooldown-bars-5m "$(HTF_COOLDOWN_BARS_5M)" \
		--data-check-strict \
	); \
	if [[ "$(origin START)" == "command line" ]]; then args+=(--start "$(START)"); fi; \
	if [[ "$(origin END)" == "command line" ]]; then args+=(--end "$(END)"); fi; \
	if [[ -n "$(strip $(HTF_MAX_SIGNALS))" ]]; then args+=(--max-signals "$(HTF_MAX_SIGNALS)"); fi; \
	$(PYTHON) "$${args[@]}"

compare-htf:
	@echo "Comparing HTF signal research across train/validation/oos"
	$(PYTHON) scripts/compare_htf_signal_research.py \
		--train-dir "$(HTF_TRAIN_DIR)" \
		--validation-dir "$(HTF_VALIDATION_DIR)" \
		--oos-dir "$(HTF_OOS_DIR)" \
		--output-dir "$(HTF_COMPARE_OUTPUT_DIR)"

research-trend-v2:
	@echo "Researching Trend Following V2 split=$(SPLIT) output=$(TREND_OUTPUT_DIR)"
	@args=( \
		scripts/research_trend_following_v2.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--split "$(SPLIT)" \
		--timezone "$(TIMEZONE)" \
		--output-dir "$(TREND_OUTPUT_DIR)" \
		--capital "$(CAPITAL)" \
		--rate "$(RATE)" \
		--slippage-mode "$(SLIPPAGE_MODE)" \
		--slippage "$(SLIPPAGE)" \
		--data-check-strict \
	); \
	if [[ "$(origin START)" == "command line" ]]; then args+=(--start "$(START)"); fi; \
	if [[ "$(origin END)" == "command line" ]]; then args+=(--end "$(END)"); fi; \
	if [[ -n "$(strip $(TREND_MAX_RUNS))" ]]; then args+=(--max-runs "$(TREND_MAX_RUNS)"); fi; \
	$(PYTHON) "$${args[@]}"

compare-trend-v2:
	@echo "Comparing Trend Following V2 research across train/validation/oos"
	$(PYTHON) scripts/compare_trend_following_v2.py \
		--train-dir "$(TREND_TRAIN_DIR)" \
		--validation-dir "$(TREND_VALIDATION_DIR)" \
		--oos-dir "$(TREND_OOS_DIR)" \
		--output-dir "$(TREND_COMPARE_OUTPUT_DIR)"

research-trend-v3:
	@echo "Researching Trend Following V3 split=$(SPLIT) output=$(TREND_V3_OUTPUT_DIR)"
	@args=( \
		scripts/research_trend_following_v3.py \
		--symbols "$(SYMBOLS)" \
		--split "$(SPLIT)" \
		--timezone "$(TIMEZONE)" \
		--interval "$(INTERVAL)" \
		--output-dir "$(TREND_V3_OUTPUT_DIR)" \
		--capital "$(CAPITAL)" \
		--capital-mode portfolio_fixed \
		--position-sizing fixed_contract \
		--fixed-size "0.01" \
		--rate "$(RATE)" \
		--slippage-mode "$(SLIPPAGE_MODE)" \
		--slippage "$(SLIPPAGE)" \
		--max-symbol-weight "0.35" \
		--max-portfolio-positions "3" \
		--data-check-strict \
	); \
	if [[ "$(origin START)" == "command line" ]]; then args+=(--start "$(START)"); fi; \
	if [[ "$(origin END)" == "command line" ]]; then args+=(--end "$(END)"); fi; \
	if [[ -n "$(strip $(TREND_V3_MAX_RUNS))" ]]; then args+=(--max-runs "$(TREND_V3_MAX_RUNS)"); fi; \
	$(PYTHON) "$${args[@]}"

compare-trend-v3:
	@echo "Comparing Trend Following V3 research across train/validation/oos"
	$(PYTHON) scripts/compare_trend_following_v3.py \
		--train-dir "$(TREND_V3_TRAIN_DIR)" \
		--validation-dir "$(TREND_V3_VALIDATION_DIR)" \
		--oos-dir "$(TREND_V3_OOS_DIR)" \
		--output-dir "$(TREND_V3_COMPARE_OUTPUT_DIR)"

research-trend-v3-extended:
	@echo "Researching Trend Following V3 extended split=$(EXT_SPLIT) output=$(TREND_V3_EXT_OUTPUT_DIR)"
	@args=( \
		scripts/research_trend_following_v3.py \
		--symbols "$(SYMBOLS)" \
		--split-scheme extended \
		--split "$(EXT_SPLIT)" \
		--timezone "$(TIMEZONE)" \
		--interval "$(INTERVAL)" \
		--output-dir "$(TREND_V3_EXT_OUTPUT_DIR)" \
		--capital "$(CAPITAL)" \
		--capital-mode portfolio_fixed \
		--position-sizing fixed_contract \
		--fixed-size "0.01" \
		--rate "$(RATE)" \
		--slippage-mode "$(SLIPPAGE_MODE)" \
		--slippage "$(SLIPPAGE)" \
		--max-symbol-weight "0.35" \
		--max-portfolio-positions "3" \
		--data-check-strict \
	); \
	if [[ "$(origin START)" == "command line" ]]; then args+=(--start "$(START)"); fi; \
	if [[ "$(origin END)" == "command line" ]]; then args+=(--end "$(END)"); fi; \
	if [[ -n "$(strip $(TREND_V3_MAX_RUNS))" ]]; then args+=(--max-runs "$(TREND_V3_MAX_RUNS)"); fi; \
	$(PYTHON) "$${args[@]}"

compare-trend-v3-extended:
	@echo "Comparing Trend Following V3 extended research across train_ext/validation_ext/oos_ext"
	$(PYTHON) scripts/compare_trend_following_v3.py \
		--split-scheme extended \
		--train-dir "$(TREND_V3_EXT_TRAIN_DIR)" \
		--validation-dir "$(TREND_V3_EXT_VALIDATION_DIR)" \
		--oos-dir "$(TREND_V3_EXT_OOS_DIR)" \
		--output-dir "$(TREND_V3_EXT_COMPARE_OUTPUT_DIR)"

postmortem-trend-v3:
	@echo "Running Trend Following V3 postmortem diagnostics"
	$(PYTHON) scripts/postmortem_trend_following_v3.py \
		--train-dir "$(TREND_V3_TRAIN_DIR)" \
		--validation-dir "$(TREND_V3_VALIDATION_DIR)" \
		--oos-dir "$(TREND_V3_OOS_DIR)" \
		--compare-dir "$(TREND_V3_COMPARE_OUTPUT_DIR)" \
		--output-dir "$(TREND_V3_POSTMORTEM_OUTPUT_DIR)"

diagnose-trend-regimes:
	@echo "Diagnosing multi-symbol trend regimes"
	$(PYTHON) scripts/diagnose_trend_regimes.py \
		--symbols "$(SYMBOLS)" \
		--start 2023-01-01 \
		--end 2026-03-31 \
		--timezone Asia/Shanghai \
		--output-dir "$(TREND_REGIME_OUTPUT_DIR)" \
		--data-check-strict

research-trend-opportunity-map:
	@echo "Building research-only Trend Opportunity Map"
	$(PYTHON) scripts/research_trend_opportunity_map.py \
		--symbols "$(SYMBOLS)" \
		--start 2023-01-01 \
		--end 2026-03-31 \
		--timezone Asia/Shanghai \
		--timeframes "4h,1d" \
		--output-dir "$(TREND_OPPORTUNITY_OUTPUT_DIR)" \
		--data-check-strict

research-trend-exit-convexity:
	@echo "Researching trend capture and exit convexity without strategy changes"
	$(PYTHON) scripts/research_trend_capture_exit_convexity.py \
		--trend-map-dir reports/research/trend_opportunity_map \
		--trend-v3-dir reports/research/trend_following_v3_extended \
		--vsvcb-dir reports/research/vsvcb_v1 \
		--csrb-dir reports/research/csrb_v1 \
		--funding-dir data/funding/okx \
		--output-dir reports/research/trend_capture_exit_convexity \
		--timezone Asia/Shanghai \
		--data-check-strict

research-trend-health-exit:
	@echo "Researching trend health-state exits without strategy changes"
	$(PYTHON) scripts/research_trend_health_state_exit.py \
		--trend-map-dir reports/research/trend_opportunity_map \
		--trend-v3-dir reports/research/trend_following_v3_extended \
		--funding-dir data/funding/okx \
		--output-dir reports/research/trend_health_state_exit \
		--timezone Asia/Shanghai \
		--data-check-strict

research-trend-entry-timing:
	@echo "Researching trend entry timing without strategy changes"
	$(PYTHON) scripts/research_trend_entry_timing.py \
		--trend-map-dir reports/research/trend_opportunity_map \
		--trend-v3-dir reports/research/trend_following_v3_extended \
		--funding-dir data/funding/okx \
		--output-dir reports/research/trend_entry_timing \
		--timezone Asia/Shanghai \
		--data-check-strict

postmortem-trend-entry-timing:
	@echo "Running Trend Entry Timing postmortem and breadth candidate audit"
	$(PYTHON) scripts/postmortem_trend_entry_timing.py \
		--research-dir reports/research/trend_entry_timing \
		--trend-map-dir reports/research/trend_opportunity_map \
		--funding-dir data/funding/okx \
		--output-dir reports/research/trend_entry_timing_postmortem \
		--focus-family cross_symbol_breadth_acceleration \
		--timezone Asia/Shanghai

research-breadth-phase15:
	@echo "Running cross-symbol breadth acceleration Phase 1.5 diagnostics"
	$(PYTHON) scripts/research_cross_symbol_breadth_phase15.py \
		--entry-timing-dir reports/research/trend_entry_timing \
		--postmortem-dir reports/research/trend_entry_timing_postmortem \
		--trend-map-dir reports/research/trend_opportunity_map \
		--funding-dir data/funding/okx \
		--output-dir reports/research/cross_symbol_breadth_phase15 \
		--focus-family cross_symbol_breadth_acceleration \
		--timezone Asia/Shanghai

research-dossier:
	@echo "Building research decision dossier"
	$(PYTHON) scripts/build_research_decision_dossier.py \
		--output-dir "$(RESEARCH_DOSSIER_OUTPUT_DIR)"

audit-multisymbol:
	@echo "Auditing multi-symbol data readiness"
	$(PYTHON) scripts/audit_multisymbol_readiness.py \
		--start "$(START)" \
		--end "$(END)" \
		--interval "$(INTERVAL)" \
		--timezone "$(TIMEZONE)"

audit-extended-history:
	@echo "Auditing extended local history availability without downloading data"
	$(PYTHON) scripts/audit_extended_history_availability.py \
		--symbols "$(SYMBOLS)" \
		--interval "$(INTERVAL)" \
		--timezone "$(TIMEZONE)" \
		--output-dir "$(EXTENDED_HISTORY_OUTPUT_DIR)"

audit-external-regime:
	@echo "Auditing external regime classifier feasibility without strategy development"
	$(PYTHON) scripts/audit_external_regime_classifier_feasibility.py \
		--symbols "$(SYMBOLS)" \
		--start 2023-01-01 \
		--end 2026-03-31 \
		--timezone Asia/Shanghai \
		--output-dir "$(EXTERNAL_REGIME_OUTPUT_DIR)"

research-external-regime-classifier:
	@echo "Researching external regime classifier filters offline"
	$(PYTHON) scripts/research_external_regime_classifier.py \
		--symbols "$(SYMBOLS)" \
		--start 2023-01-01 \
		--end 2026-03-31 \
		--timezone Asia/Shanghai \
		--output-dir "$(EXTERNAL_REGIME_CLASSIFIER_OUTPUT_DIR)" \
		--data-check-strict

audit-external-regime-gates:
	@echo "Auditing external regime classifier gate consistency"
	$(PYTHON) scripts/audit_external_regime_classifier_gates.py \
		--classifier-dir "$(EXTERNAL_REGIME_CLASSIFIER_OUTPUT_DIR)" \
		--gate-audit-dir "$(EXTERNAL_REGIME_GATE_AUDIT_OUTPUT_DIR)"

research-vsvcb-v1:
	@echo "Researching VSVCB-v1 Phase 1 event study and fixed-hold benchmark"
	$(PYTHON) scripts/research_vsvcb_v1.py \
		--symbols "BTCUSDT_SWAP_OKX.GLOBAL,ETHUSDT_SWAP_OKX.GLOBAL,SOLUSDT_SWAP_OKX.GLOBAL" \
		--timeframes "15m,30m,1h" \
		--start 2023-01-01 \
		--end 2026-03-31 \
		--timezone Asia/Shanghai \
		--output-dir "$(VSVCB_OUTPUT_DIR)" \
		--data-check-strict

postmortem-vsvcb-v1:
	@echo "Running VSVCB-v1 Phase 1 postmortem diagnostics"
	$(PYTHON) scripts/postmortem_vsvcb_v1.py \
		--research-dir reports/research/vsvcb_v1 \
		--output-dir reports/research/vsvcb_v1_postmortem

audit-derivatives-data:
	@echo "Auditing OKX public derivatives data readiness without strategy development"
	$(PYTHON) scripts/audit_okx_derivatives_data_readiness.py \
		--inst-ids "$(INST_IDS)" \
		--ccys "BTC,ETH,SOL,LINK,DOGE" \
		--start 2023-01-01 \
		--end 2026-03-31 \
		--timezone Asia/Shanghai \
		--output-dir "$(DERIVATIVES_DATA_READINESS_OUTPUT_DIR)"

research-csrb-v1:
	@echo "Researching CSRB-v1 Phase 1 session breakout event study"
	$(PYTHON) scripts/research_csrb_v1.py \
		--symbols "BTCUSDT_SWAP_OKX.GLOBAL,ETHUSDT_SWAP_OKX.GLOBAL,SOLUSDT_SWAP_OKX.GLOBAL" \
		--timeframes "15m,30m,1h" \
		--start 2023-01-01 \
		--end 2026-03-31 \
		--session-timezone UTC \
		--report-timezone Asia/Shanghai \
		--output-dir reports/research/csrb_v1 \
		--data-check-strict

postmortem-csrb-v1:
	@echo "Running CSRB-v1 Phase 1 postmortem and control audit"
	$(PYTHON) scripts/postmortem_csrb_v1.py \
		--research-dir reports/research/csrb_v1 \
		--output-dir reports/research/csrb_v1_postmortem

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

ablation:
	@base_config="$(STRATEGY_CONFIG)"; \
	if [[ "$(origin STRATEGY_CONFIG)" == "file" ]]; then base_config="$(SANITY_CONFIG)"; fi; \
	if [[ ! -f "$$base_config" ]]; then \
		echo "ERROR: strategy config not found: $$base_config"; \
		exit 2; \
	fi; \
	output_dir="$(OUTPUT_DIR)"; \
	if [[ -z "$$output_dir" ]]; then output_dir="reports/ablation/main_20250101_20260331"; fi; \
	echo "Running ablation experiments with $$base_config split=$(SPLIT) output=$$output_dir"; \
	args=( \
		scripts/run_ablation_experiments.py \
		--vt-symbol "$(VT_SYMBOL)" \
		--timezone "$(TIMEZONE)" \
		--base-config "$$base_config" \
		--capital "$(CAPITAL)" \
		--rate "$(RATE)" \
		--slippage-mode "$(SLIPPAGE_MODE)" \
		--slippage "$(SLIPPAGE)" \
		--output-dir "$$output_dir" \
		--split "$(SPLIT)" \
		--max-runs "$(MAX_RUNS)" \
		--data-check-strict \
	); \
	if [[ "$(SPLIT)" == "full" ]]; then \
		args+=(--start "$(START)" --end "$(END)"); \
	else \
		if [[ "$(origin START)" != "file" ]]; then args+=(--start "$(START)"); fi; \
		if [[ "$(origin END)" != "file" ]]; then args+=(--end "$(END)"); fi; \
	fi; \
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
