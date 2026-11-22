.DEFAULT_GOAL := help

# =============================================================================
# Toolchain & Environment
# =============================================================================
VENV_BIN := $(if $(wildcard .venv/bin/python),.venv/bin,$(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts,))
PYTHON   := $(if $(VENV_BIN),$(VENV_BIN)/python,python3)
PIP      := $(PYTHON) -m pip
RUFF     := $(if $(wildcard $(VENV_BIN)/ruff*),$(VENV_BIN)/ruff,ruff)

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
RAW_DIR        ?= data/raw
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
SCENARIOS      ?=
AUTO           ?=
LIVE           ?=
LOW_MEMORY     ?=

CLI_FLAGS := $(if $(VERBOSE),--verbose,) $(if $(QUIET),--quiet,) $(if $(JSON),--json,)

.PHONY: help install download-models test test-cov lint format check \
        ingest ingest-dry-run report query inspect export-chunks batch-query \
        eval demo api clean reset-db

# =============================================================================
# Help & Setup
# =============================================================================
help: ##@Setup Show available make targets and usage examples
	@printf "\033[1mIngenium Parent-Child Hybrid RAG — Developer Commands\033[0m\n"
	@awk 'BEGIN {FS = ":.*?##@"; section = ""} \
		/^[a-zA-Z_-]+:.*?##@/ { \
			split($$2, parts, " "); \
			grp = parts[1]; \
			desc = substr($$2, length(grp) + 2); \
			if (grp != section) { \
				section = grp; \
				gsub(/_/, " ", grp); \
				printf "\n\033[1;33m%s\033[0m\n", grp; \
			} \
			printf "  \033[36m%-18s\033[0m %s\n", $$1, desc; \
		}' $(MAKEFILE_LIST)
	@printf "\n\033[1;33mExamples\033[0m\n"
	@echo "  make install && make download-models                       # first-time setup"
	@echo "  make check                                                 # lint + offline tests"
	@echo "  make ingest MODE=sharepoint SITES_FILE=data/sharepoint_sites.tsv"
	@echo "  make report                                                # compact ETL & audit summary"
	@echo "  make inspect                                               # Qdrant index stats"
	@echo "  make query Q='Wat zijn de eisen voor brandwerende deuren?' BUILDING='Blok A' DEBUG=1"
	@echo "  make eval DATASET=data/eval_questions.json"

install: ##@Setup Install editable package with all extras (etl, rag, api, dev)
	$(PIP) install -e ".[all]"

download-models: ##@Setup Pre-download BGE-M3 embedder and reranker into HF cache
	$(PYTHON) -c "from huggingface_hub import snapshot_download; \
		snapshot_download('BAAI/bge-m3'); \
		snapshot_download('BAAI/bge-reranker-v2-m3')"

# =============================================================================
# Ingestion & Diagnostics
# =============================================================================
ingest: ##@Ingestion_&_Diagnostics Run ETL + chunking + BGE-M3 indexing (MODE=etl|raw|sharepoint)
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

ingest-dry-run: ##@Ingestion_&_Diagnostics Preview file inventory & metadata without downloading (MODE=raw|sharepoint)
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
	if [ -n "$(LIMIT)" ]; then set -- "$$@" --limit "$(LIMIT)"; fi; \
	$(PY_ENV) $(PYTHON) scripts/ingest.py "$$@" $(CLI_FLAGS)

report: ##@Ingestion_&_Diagnostics Print ETL KPI summary & limitations audit (LIMIT=20, 0=all)
	@set -- --report --processed-dir "$(ETL_DIR)"; \
	if [ -n "$(LIMIT)" ]; then set -- "$$@" --limit "$(LIMIT)"; fi; \
	$(PY_ENV) $(PYTHON) scripts/ingest.py "$$@" $(CLI_FLAGS)

# =============================================================================
# Query, Inspection & Evaluation
# =============================================================================
query: ##@Query,_Inspection_&_Evaluation Run hybrid search + rerank (Q="...", DEBUG=1, JSON=1)
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

inspect: ##@Query,_Inspection_&_Evaluation Print Qdrant stats & metadata breakdown (optional DOC_ID=...)
	@set -- --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS) stats; \
	if [ -n "$(DOC_ID)" ]; then set -- "$$@" --doc-id "$(DOC_ID)"; fi; \
	$(PY_ENV) $(PYTHON) scripts/inspect_rag.py "$$@"

export-chunks: ##@Query,_Inspection_&_Evaluation Export parent & child chunks to JSONL (OUT=data/exports)
	@set -- --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS) export --out "$(OUT)"; \
	if [ -n "$(DOC_ID)" ]; then set -- "$$@" --doc-id "$(DOC_ID)"; fi; \
	$(PY_ENV) $(PYTHON) scripts/inspect_rag.py "$$@"

batch-query: ##@Query,_Inspection_&_Evaluation Run batch questions (.txt/.json) -> Markdown/JSON report
	@if [ -z "$(QUESTIONS)" ]; then \
		echo "Usage: make batch-query QUESTIONS=path/to/questions.txt [OUT=data/exports/batch_report.md]"; \
		exit 1; \
	fi
	@REPORT_OUT="$(if $(filter data/exports,$(OUT)),data/exports/batch_query_report.md,$(OUT))"; \
	$(PY_ENV) $(PYTHON) scripts/inspect_rag.py --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS) batch-query \
		--questions "$(QUESTIONS)" --out "$$REPORT_OUT"

eval: ##@Query,_Inspection_&_Evaluation Run evaluation dataset (DATASET=path/to/eval.json|csv)
	@if [ -z "$(DATASET)" ]; then \
		echo "Usage: make eval DATASET=path/to/eval.json"; \
		exit 1; \
	fi
	$(PY_ENV) $(PYTHON) scripts/eval.py --dataset "$(DATASET)" --qdrant-path "$(QDRANT_PATH)" $(CLI_FLAGS)

demo: ##@Query,_Inspection_&_Evaluation Run stakeholder live demo (SCENARIOS=..., AUTO=1, LIVE=1)
	@set -- --qdrant-path "$(QDRANT_PATH)"; \
	if [ -n "$(SCENARIOS)" ]; then set -- "$$@" --scenarios "$(SCENARIOS)"; fi; \
	if [ -n "$(MIN_SCORE)" ]; then set -- "$$@" --min-score "$(MIN_SCORE)"; fi; \
	if [ -n "$(AUTO)" ]; then set -- "$$@" --auto; fi; \
	if [ -n "$(LIVE)" ]; then set -- "$$@" --live; fi; \
	if [ -n "$(LOW_MEMORY)" ]; then set -- "$$@" --low-memory; fi; \
	if [ -n "$(VERBOSE)" ]; then set -- "$$@" --verbose; fi; \
	$(PY_ENV) $(PYTHON) scripts/demo.py "$$@"

api: ##@Query,_Inspection_&_Evaluation Start FastAPI server on http://127.0.0.1:8000 (/docs)
	$(PY_ENV) $(PYTHON) -m uvicorn api.app:app --reload --port 8000

# =============================================================================
# Quality & Maintenance
# =============================================================================
check: lint test ##@Quality_&_Maintenance Run ruff linter and offline pytest suite (pre-PR gate)

test: ##@Quality_&_Maintenance Run offline pytest suite (incl. AST architecture checks)
	$(PYTEST) tests/ -v --tb=short

test-cov: ##@Quality_&_Maintenance Run pytest with terminal coverage report
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

lint: ##@Quality_&_Maintenance Run ruff linter across src/, scripts/, and tests/
	$(RUFF) check src/ scripts/ tests/

format: ##@Quality_&_Maintenance Auto-format and fix imports with ruff
	$(RUFF) format src/ scripts/ tests/
	$(RUFF) check --fix src/ scripts/ tests/

clean: ##@Quality_&_Maintenance Clear Python/pytest/ruff caches (preserves data/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -maxdepth 4 -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -maxdepth 3 -type d -name "*.egg-info" -not -path "./.venv/*" -exec rm -rf {} +

reset-db: ##@Quality_&_Maintenance Wipe local Qdrant storage (./data/qdrant_poc) after confirm
	@printf "Delete local Qdrant directory '%s'? [y/N] " "$(QDRANT_PATH)"; \
	read ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -rf "$(QDRANT_PATH)"; \
		echo "Removed $(QDRANT_PATH)."; \
	else \
		echo "Aborted."; \
	fi