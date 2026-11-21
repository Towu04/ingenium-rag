.DEFAULT_GOAL := help

# =============================================================================
# Toolchain & Environment
# =============================================================================
VENV_BIN := $(if $(wildcard .venv/bin/python),.venv/bin,)
PYTHON   := $(if $(VENV_BIN),$(VENV_BIN)/python,python3)
PIP      := $(PYTHON) -m pip
RUFF     := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)

# Default to online so fresh machines download HF models on first run;
# override with `HF_HUB_OFFLINE=1` for strict offline execution.
HF_HUB_OFFLINE      ?= 0
HF_HUB_ETAG_TIMEOUT ?= 2

PY_ENV := PYTHONPATH=src:. HF_HUB_OFFLINE=$(HF_HUB_OFFLINE) HF_HUB_ETAG_TIMEOUT=$(HF_HUB_ETAG_TIMEOUT)
PYTEST := PYTHONPATH=src:. HF_HUB_OFFLINE=1 $(PYTHON) -m pytest

# =============================================================================
# Runtime Overrides (ingest / query / eval)
# =============================================================================
MODE           ?= etl
RAW_DIR        ?= data/raw_sharepoint
ETL_DIR        ?= data/processed
QDRANT_PATH    ?= data/qdrant_poc
PREFIX         ?=
SITE           ?=
SITES_FILE     ?=
WORKERS        ?=
LIMIT          ?=
FORCE          ?=
Q              ?=
PROJECT        ?=
BUILDING       ?=
PHASE          ?=
RETRIEVAL_ONLY ?=
NO_RERANK      ?=
DEBUG          ?=
JSON           ?=
VERBOSE        ?=
QUIET          ?=
MIN_SCORE      ?=
DOC_ID         ?=
OUT            ?= data/exports
QUESTIONS      ?=
DATASET        ?=

CLI_FLAGS := $(if $(VERBOSE),--verbose,) $(if $(QUIET),--quiet,) $(if $(JSON),--json,)

.PHONY: help install download-models test test-cov lint format check \
        ingest ingest-dry-run report query inspect export-chunks batch-query \
        eval api clean reset-db

# =============================================================================
# Help & Setup
# =============================================================================
help: ## Show available make targets and usage examples
	@echo "Ingenium Parent-Child Hybrid RAG — Developer Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install && make download-models                       # first-time setup"
	@echo "  make check                                                 # lint + offline tests"
	@echo "  make ingest                                                # index data/processed -> data/qdrant_poc"
	@echo "  make ingest MODE=sharepoint PREFIX='General/05_Uitvoering' LIMIT=10"
	@echo "  make ingest-dry-run MODE=sharepoint PREFIX='General/04_Prijsvraag'"
	@echo "  make query Q='Wat zijn de eisen voor brandwerende deuren?' BUILDING='Blok A'"
	@echo "  make query Q='brandwerende deuren' RETRIEVAL_ONLY=1 NO_RERANK=1"
	@echo "  make eval DATASET=data/eval_questions.json"

install: ## Install package in editable mode with all extras (etl, rag, api, dev)
	$(PIP) install -e ".[all]"

download-models: ## Pre-download BGE-M3 embedder and reranker into local HF cache
	$(PYTHON) -c "from huggingface_hub import snapshot_download; \
		snapshot_download('BAAI/bge-m3'); \
		snapshot_download('BAAI/bge-reranker-v2-m3')"

# =============================================================================
# Quality & Testing
# =============================================================================
test: ## Run full offline pytest suite (including AST stage-boundary checks)
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run pytest with terminal coverage report
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

lint: ## Run ruff linter across src/, scripts/, and tests/
	$(RUFF) check src/ scripts/ tests/

format: ## Auto-format and fix imports with ruff
	$(RUFF) format src/ scripts/ tests/
	$(RUFF) check --fix src/ scripts/ tests/

check: lint test ## Run linter and unit/integration tests (pre-PR gate)

# =============================================================================
# Pipeline Execution
# =============================================================================
ingest: ## Run ingestion pipeline (MODE=etl|raw|sharepoint, SITE=..., SITES_FILE=..., WORKERS=4, PREFIX=..., LIMIT=N, FORCE=1)
	@set -- --processed-dir "$(ETL_DIR)" --qdrant-path "$(QDRANT_PATH)"; \
	case "$(MODE)" in \
		sharepoint) \
			set -- "$$@" --from-sharepoint; \
			if [ -n "$(SITE)" ]; then set -- "$$@" --site "$(SITE)"; fi; \
			if [ -n "$(SITES_FILE)" ]; then set -- "$$@" --sites-file "$(SITES_FILE)"; fi; \
			if [ -n "$(PREFIX)" ]; then set -- "$$@" --folder-prefix "$(PREFIX)"; fi; \
			if [ -n "$(LIMIT)" ]; then set -- "$$@" --max-files "$(LIMIT)"; fi ;; \
		raw) \
			set -- "$$@" --raw-dir "$(RAW_DIR)" ;; \
		etl) \
			set -- "$$@" --etl-dir "$(ETL_DIR)" ;; \
		*) \
			echo "Error: Invalid MODE='$(MODE)'. Expected: etl, raw, or sharepoint."; exit 1 ;; \
	esac; \
	if [ -n "$(WORKERS)" ]; then set -- "$$@" --workers "$(WORKERS)"; fi; \
	if [ -n "$(FORCE)" ]; then set -- "$$@" --force; fi; \
	$(PY_ENV) $(PYTHON) scripts/ingest.py "$$@" $(CLI_FLAGS)

ingest-dry-run: ## Preview file inventory and metadata without extracting (MODE=raw|sharepoint)
	@set -- --dry-run; \
	case "$(MODE)" in \
		sharepoint) \
			set -- "$$@" --from-sharepoint; \
			if [ -n "$(SITE)" ]; then set -- "$$@" --site "$(SITE)"; fi; \
			if [ -n "$(SITES_FILE)" ]; then set -- "$$@" --sites-file "$(SITES_FILE)"; fi; \
			if [ -n "$(PREFIX)" ]; then set -- "$$@" --folder-prefix "$(PREFIX)"; fi ;; \
		raw) \
			set -- "$$@" --raw-dir "$(RAW_DIR)" ;; \
		*) \
			echo "Error: Invalid MODE='$(MODE)'. Expected: raw or sharepoint."; exit 1 ;; \
	esac; \
	$(PY_ENV) $(PYTHON) scripts/ingest.py "$$@" $(CLI_FLAGS)

report: ## Print processed documents summary and scanned-PDF/limitations audit report
	$(PY_ENV) $(PYTHON) scripts/ingest.py --report --processed-dir "$(ETL_DIR)" $(CLI_FLAGS)

query: ## Run hybrid search + rerank (Q="...", optional PROJECT/BUILDING/PHASE/DEBUG/JSON/VERBOSE/QUIET/MIN_SCORE/RETRIEVAL_ONLY/NO_RERANK)
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q=\"Your question here\" [PROJECT=...] [BUILDING=...] [PHASE=...] [DEBUG=1] [JSON=1] [VERBOSE=1] [MIN_SCORE=0.05] [RETRIEVAL_ONLY=1] [NO_RERANK=1]"; \
		exit 1; \
	fi
	@set -- --qdrant-path "$(QDRANT_PATH)"; \
	if [ -n "$(PROJECT)" ]; then set -- "$$@" --project "$(PROJECT)"; fi; \
	if [ -n "$(BUILDING)" ]; then set -- "$$@" --building "$(BUILDING)"; fi; \
	if [ -n "$(PHASE)" ]; then set -- "$$@" --phase "$(PHASE)"; fi; \
	if [ -n "$(MIN_SCORE)" ]; then set -- "$$@" --min-score "$(MIN_SCORE)"; fi; \
	if [ -n "$(RETRIEVAL_ONLY)" ]; then set -- "$$@" --retrieval-only; fi; \
	if [ -n "$(NO_RERANK)" ]; then set -- "$$@" --no-rerank; fi; \
	if [ -n "$(DEBUG)" ]; then set -- "$$@" --debug; fi; \
	$(PY_ENV) $(PYTHON) scripts/query.py "$$@" $(CLI_FLAGS) "$(Q)"

inspect: ## Print Qdrant DB stats & metadata distributions (optional DOC_ID=<id>, JSON=1)
	@set -- --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS) stats; \
	if [ -n "$(DOC_ID)" ]; then set -- "$$@" --doc-id "$(DOC_ID)"; fi; \
	$(PY_ENV) $(PYTHON) scripts/inspect_rag.py "$$@"

export-chunks: ## Export parent & child chunks to JSONL for RAGAS (OUT=data/exports, optional DOC_ID=...)
	@set -- --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS) export --out "$(OUT)"; \
	if [ -n "$(DOC_ID)" ]; then set -- "$$@" --doc-id "$(DOC_ID)"; fi; \
	$(PY_ENV) $(PYTHON) scripts/inspect_rag.py "$$@"

batch-query: ## Run batch questions (.txt/.json) and save a retrieval report (QUESTIONS=..., OUT=...)
	@if [ -z "$(QUESTIONS)" ]; then \
		echo "Usage: make batch-query QUESTIONS=path/to/questions.txt [OUT=data/exports/batch_report.md]"; \
		exit 1; \
	fi
	@REPORT_OUT="$(if $(filter data/exports,$(OUT)),data/exports/batch_query_report.md,$(OUT))"; \
	$(PY_ENV) $(PYTHON) scripts/inspect_rag.py --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS) batch-query \
		--questions "$(QUESTIONS)" --out "$$REPORT_OUT"

eval: ## Run evaluation dataset (requires DATASET=path/to/eval.json|csv)
	@if [ -z "$(DATASET)" ]; then \
		echo "Usage: make eval DATASET=path/to/eval.json"; \
		exit 1; \
	fi
	$(PY_ENV) $(PYTHON) scripts/eval.py --dataset "$(DATASET)" --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS)

api: ## Start FastAPI development server on :8000
	$(PY_ENV) $(PYTHON) -m uvicorn api.app:app --reload --port 8000

# =============================================================================
# Maintenance
# =============================================================================
clean: ## Clear Python/pytest/ruff caches (preserves data/ and Qdrant index)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -maxdepth 4 -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -maxdepth 3 -type d -name "*.egg-info" -not -path "./.venv/*" -exec rm -rf {} +

reset-db: ## Wipe local Qdrant storage (./data/qdrant_poc) after confirmation
	@printf "Delete local Qdrant directory '%s'? [y/N] " "$(QDRANT_PATH)"; \
	read ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -rf "$(QDRANT_PATH)"; \
		echo "Removed $(QDRANT_PATH)."; \
	else \
		echo "Aborted."; \
	fi