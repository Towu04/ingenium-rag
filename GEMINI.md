# GEMINI.md — Ingenium RAG Architecture & Developer Guide

## Overview
Parent-Child Hybrid RAG PoC for Ingenium engineering/construction documents with metadata filtering on `project`, `building` (multi-value array), and `phase`.

## Architecture: Pipes-and-Filters
Every pipeline stage is strictly isolated under `src/` and communicates **only** through canonical contracts in `src/domain.py`. Stages never import each other; only `src/pipeline.py` wires stages together. This invariant is enforced automatically by `tests/test_architecture.py`.

```text
SharePoint / Local Files
   │
   ▼  Stage 1: src/etl/           (PDF/DOCX/XLSX extractors + 3-layer metadata + OCR/noise audit)
ParsedDoc (domain.py)
   │
   ▼  Stage 2: src/chunking/      (Table-aware ParentChildChunker, BGE-M3 tokenizer)
ParentChunk[], ChildChunk[]
   │
   ▼  Stage 3: src/embedding/     (BGEM3Embedder: 1024-dim dense + sparse lexical weights)
EmbeddedChildBatch
   │
   ▼  Stage 4: src/storage/       (LocalHybridRepository: disk-persistent Qdrant + RRF)
SearchResult[]
   │
   ▼  Stage 5: src/retrieval/     (query_filter + Searcher protocols + BGEM3Reranker)
SearchResult[] (reranked top-k)
   │
   ▼  Stage 6: src/generation/    (prompts + citations + LLMClient protocol + GroundedGenerator)
Answer (text + Citation[] + retrieved_results)
```

## Directory Layout
- `src/domain.py` — Zero-dependency canonical Pydantic contracts (`DocumentMetadata`, `ParsedSection`, `ParsedDoc`, `ParentChunk`, `ChildChunk`, `SearchResult`, `Citation`, `Answer`), `EmbeddedChildBatch`, `EncodedQuery`, and pure metadata normalization/taxonomy helpers.
- `src/config.py` — Single centralized `Settings` (`pydantic-settings` + `.env`), paths, thresholds, model defaults, and `detect_device()` (CUDA > MPS > CPU).
- `src/pipeline.py` — Thin top-level orchestrator (`IndexingPipeline`, `RAGPipeline`, `build_rag_pipeline`, `build_llm_client`).
- `src/etl/` — `extractors/{pdf,docx,xlsx,common}.py`, `catalog.py`, `metadata.py`, `quality.py`, `sharepoint.py`, `runner.py`. Emits and persists `domain.ParsedDoc` JSON directly.
- `src/chunking/` — `parent_child.py` (`ParentChildChunker`: repeats Markdown table headers across split chunks).
- `src/embedding/` — `bge_m3.py` (`BGEM3Embedder`).
- `src/storage/` — `qdrant.py` (`LocalHybridRepository` over `./data/qdrant_poc`, collections: `documents`, `parent_chunks`, `child_chunks`).
- `src/retrieval/` — `query_filter.py`, `searcher.py` (`QueryEncoder` / `ParentStore` protocols), `reranker.py` (`BGEM3Reranker`).
- `src/generation/` — `prompts.py`, `citations.py`, `generator.py` (`LLMClient` protocol, `ExtractiveStubLLM` default, `OpenAICompatibleLLM` for local Ollama/vLLM).
- `src/evaluation/` — `metrics.py` (`citation_precision`, `source_recall`), `runner.py`.
- `src/api/` — FastAPI HTTP layer (`GET /health`, `POST /v1/query`).
- `scripts/` — Thin CLI entry points + shared Rich CLI helper (`_cli.py`): `ingest.py` (`ingenium-ingest`), `query.py` (`ingenium-query`), `inspect_rag.py` (`ingenium-inspect`), `eval.py` (`ingenium-eval`).
- `tests/` — Stage-mirrored test suite + `test_architecture.py` (enforces zero cross-stage imports and zero `print()` inside `src/`).

## Key Invariants & Conventions
1. **Stage isolation**: Do not import sibling stages from within `etl`, `chunking`, `embedding`, `storage`, `retrieval`, `generation`, or `evaluation`. Pass data through `domain.py` types or lightweight `Protocol`s and wire in `src/pipeline.py`.
2. **Logging over `print()`**: All modules under `src/` must use `logging.getLogger(__name__)`. Only CLI scripts under `scripts/` print to stdout (all logs/progress/errors go to `stderr`).
3. **Metadata normalization**: Unresolved `project`, `building`, or `phase` values are always `None` (never `"UNKNOWN_*"`). Multi-building documents store `buildings: list[str]` in Qdrant so `MatchValue(value="Blok A")` works out of the box.
4. **Local LLM**: Defaults to `ExtractiveStubLLM` when `LLM_BASE_URL` / `LLM_MODEL` are unset in `.env` (see `.env.example`). Plugging in a local Ollama/vLLM server requires only setting those env vars.
5. **Qdrant local lock**: Disk-persistent local Qdrant (`./data/qdrant_poc`) allows one process at a time; always use `with LocalHybridRepository(...) as repo:`.

## Common `make` Commands
```bash
make help                                                                # list all targets
make check                                                               # ruff lint + pytest
make test                                                                # run offline pytest suite
make ingest                                                              # index data/processed -> data/qdrant_poc
make ingest MODE=raw RAW_DIR=data/raw                                    # extract local raw files & index
make ingest MODE=sharepoint SITE=/sites/12345001-INT PREFIX="General/05_Uitvoering" LIMIT=10
make ingest MODE=sharepoint SITES_FILE=data/sharepoint_sites.tsv WORKERS=4
make ingest-dry-run MODE=sharepoint SITES_FILE=data/sharepoint_sites.tsv # inventory preview (no downloads)
make report                                                              # manifest, scanned-PDF & unauthorized report
make query Q="Wat zijn de eisen voor brandwerende deuren?" BUILDING="Blok A" DEBUG=1
make query Q="fire-resistant doors" JSON=1 | jq '.trace.results[0]'
make inspect                                                             # Qdrant stats & metadata distributions
make export-chunks OUT=data/exports                                      # export JSONL chunks for RAGAS
make batch-query QUESTIONS=questions.txt OUT=data/exports/report.md      # batch retrieval inspection report
make eval DATASET=data/eval.json                                         # run evaluation dataset
make api                                                                 # start FastAPI on :8000
make clean                                                               # clear caches (preserves data/)
make reset-db                                                            # wipe data/qdrant_poc (with confirmation)
```
