.DEFAULT_GOAL := help

# Auto-detect project virtualenv if present; otherwise fall back to python3
VENV_BIN := $(if $(wildcard .venv/bin/python),.venv/bin,)
PYTHON   := $(if $(VENV_BIN),$(VENV_BIN)/python,python3)
PIP      := $(PYTHON) -m pip
PYTEST   := PYTHONPATH=src:. HF_HUB_OFFLINE=1 $(PYTHON) -m pytest
RUFF     := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)

# Optional overrides for ingest / query / eval targets
MODE           ?= etl
RAW_DIR        ?= data/raw_sharepoint
ETL_DIR        ?= data/processed
QDRANT_PATH    ?= data/qdrant_poc
PREFIX         ?=
LIMIT          ?=
FORCE          ?=
Q              ?=
PROJECT        ?=
BUILDING       ?=
PHASE          ?=
RETRIEVAL_ONLY ?=
DATASET        ?=

.PHONY: help install test test-cov lint format check ingest ingest-dry-run report query eval api clean reset-db

help: ## Show available make targets and usage examples
	@echo "Ingenium Parent-Child Hybrid RAG — Developer Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make ingest                                                # index data/processed -> data/qdrant_poc"
	@echo "  make ingest MODE=sharepoint PREFIX='General/05_Uitvoering' LIMIT=10"
	@echo "  make ingest-dry-run MODE=sharepoint PREFIX='General/04_Prijsvraag'"
	@echo "  make query Q='Wat zijn de eisen voor brandwerende deuren?' BUILDING='Blok L'"
	@echo "  make query Q='brandwerende deuren' RETRIEVAL_ONLY=1"
	@echo "  make eval DATASET=data/eval_questions.json"

install: ## Install package in editable mode with all extras (etl, rag, api, dev)
	$(PIP) install -e ".[all]"

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

ingest: ## Run ingestion pipeline (MODE=etl|raw|sharepoint, PREFIX=..., LIMIT=N, FORCE=1)
	@args=""; \
	if [ "$(MODE)" = "sharepoint" ]; then \
		args="--from-sharepoint"; \
		if [ -n "$(PREFIX)" ]; then args="$$args --folder-prefix '$(PREFIX)'"; fi; \
		if [ -n "$(LIMIT)" ]; then args="$$args --max-files $(LIMIT)"; fi; \
	elif [ "$(MODE)" = "raw" ]; then \
		args="--raw-dir '$(RAW_DIR)'"; \
	else \
		args="--etl-dir '$(ETL_DIR)'"; \
	fi; \
	if [ -n "$(FORCE)" ]; then args="$$args --force"; fi; \
	eval "PYTHONPATH=src:. $(PYTHON) scripts/ingest.py $$args --processed-dir '$(ETL_DIR)' --qdrant-path '$(QDRANT_PATH)'"

ingest-dry-run: ## Preview file inventory and resolved metadata without extracting (MODE=raw|sharepoint)
	@args="--dry-run"; \
	if [ "$(MODE)" = "sharepoint" ]; then \
		args="$$args --from-sharepoint"; \
		if [ -n "$(PREFIX)" ]; then args="$$args --folder-prefix '$(PREFIX)'"; fi; \
	else \
		args="$$args --raw-dir '$(RAW_DIR)'"; \
	fi; \
	eval "PYTHONPATH=src:. $(PYTHON) scripts/ingest.py $$args"

report: ## Print processed documents summary and scanned-PDF/limitations audit report
	PYTHONPATH=src:. $(PYTHON) scripts/ingest.py --report --processed-dir "$(ETL_DIR)"

query: ## Run hybrid search + rerank query (requires Q="...", optional PROJECT/BUILDING/PHASE/RETRIEVAL_ONLY=1)
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q=\"Your question here\" [PROJECT=...] [BUILDING=...] [PHASE=...] [RETRIEVAL_ONLY=1]"; \
		exit 1; \
	fi
	@args="--qdrant-path '$(QDRANT_PATH)'"; \
	if [ -n "$(PROJECT)" ]; then args="$$args --project '$(PROJECT)'"; fi; \
	if [ -n "$(BUILDING)" ]; then args="$$args --building '$(BUILDING)'"; fi; \
	if [ -n "$(PHASE)" ]; then args="$$args --phase '$(PHASE)'"; fi; \
	if [ -n "$(RETRIEVAL_ONLY)" ]; then args="$$args --retrieval-only"; fi; \
	if [ -n "$(NO_RERANK)" ]; then args="$$args --no-rerank"; fi; \
	eval "HF_HUB_OFFLINE=1 PYTHONPATH=src:. $(PYTHON) scripts/query.py $$args \"$(Q)\""

eval: ## Run evaluation dataset (requires DATASET=path/to/eval.json|csv)
	@if [ -z "$(DATASET)" ]; then \
		echo "Usage: make eval DATASET=path/to/eval.json"; \
		exit 1; \
	fi
	HF_HUB_OFFLINE=1 PYTHONPATH=src:. $(PYTHON) scripts/eval.py --dataset "$(DATASET)" --qdrant-path "$(QDRANT_PATH)"

api: ## Start FastAPI development server on :8000
	PYTHONPATH=src:. $(PYTHON) -m uvicorn api.app:app --reload --port 8000

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
