# Ingenium RAG

SharePoint document extraction, metadata-aware Parent-Child Hybrid RAG, and citation-grounded answering for Ingenium engineering projects.

## Architecture

The codebase follows a strict **Pipes-and-Filters** design where every stage lives under `src/`, communicates exclusively through canonical Pydantic models in [src/domain.py](src/domain.py), and is wired in [src/pipeline.py](src/pipeline.py):

| Stage | Module | Input → Output |
| :--- | :--- | :--- |
| 1. Extraction & Metadata | `src/etl/` | SharePoint / local PDF, DOCX, XLSX → `ParsedDoc` |
| 2. Table-Aware Chunking | `src/chunking/` | `ParsedDoc` → `ParentChunk[]`, `ChildChunk[]` |
| 3. Hybrid Embedding | `src/embedding/` | `ChildChunk[]` / query → `EmbeddedChildBatch` / `EncodedQuery` (`BAAI/bge-m3`) |
| 4. Hybrid Vector Storage | `src/storage/` | Local disk-persistent Qdrant (`./data/qdrant_poc`) with native RRF + multi-building filters |
| 5. Retrieval & Reranking | `src/retrieval/` | Query + filters → `SearchResult[]` (`BAAI/bge-reranker-v2-m3`) |
| 6. Grounded Generation | `src/generation/` | `SearchResult[]` → `Answer` + `Citation[]` (`ExtractiveStubLLM` or local OpenAI-compatible LLM) |
| 7. Evaluation | `src/evaluation/` | Q&A dataset → `citation_precision` & `source_recall` |

See [GEMINI.md](GEMINI.md) for the full directory map and architectural invariants.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make check
```

Configure SharePoint credentials (and optional local LLM endpoint) in a git-ignored `.env` at the repo root:

```dotenv
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
SHAREPOINT_HOSTNAME=ingeniumnv.sharepoint.com
SHAREPOINT_SITE_PATH=/sites/06038004-INT
SHAREPOINT_DRIVE_NAME=Documents

# Optional: local OpenAI-compatible LLM (Ollama / vLLM); omit to use ExtractiveStubLLM
# LLM_BASE_URL=http://localhost:11434
# LLM_MODEL=qwen2.5:14b
```

## Common Commands (`Makefile`)

Run `make help` to see all available targets:

```bash
make test                                                                # run pytest (incl. architecture boundary checks)
make lint                                                                # run ruff linter
make format                                                              # auto-format with ruff
make ingest                                                              # index data/processed -> data/qdrant_poc
make ingest MODE=sharepoint PREFIX="General/05_Uitvoering" LIMIT=10      # stream from SharePoint & index
make ingest-dry-run MODE=sharepoint PREFIX="General/04_Prijsvraag"       # preview inventory without downloading
make report                                                              # show extracted docs & scanned-PDF audit
make query Q="Wat zijn de eisen voor brandwerende deuren?" RETRIEVAL_ONLY=1
make query Q="medische gassen" PROJECT="06038.004" BUILDING="Blok L"
make eval DATASET=data/eval.json                                         # evaluate a JSON/CSV question set
make api                                                                 # start FastAPI (GET /health, POST /v1/query)
make clean                                                               # clear caches (never touches data/)
make reset-db                                                            # wipe ./data/qdrant_poc (with confirmation)
```
