# Ingenium RAG

SharePoint document ingestion, metadata-aware retrieval, and citation-grounded
answering for Ingenium projects. The repository is structured as an installable
Python package so each pipeline stage can be developed and deployed independently.

## Quick start

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
uvicorn ingenium_rag.api.app:app --reload
```

The API exposes `GET /health` and `POST /v1/query`. The initial implementation
uses an in-memory vector store and dependency-free text parsing as development
adapters. Production deployments should provide the Microsoft Graph, Docling/OCR,
embedding, vector database, and LLM adapters at the boundaries in `src/ingenium_rag`.

## Repository layout

- `configs/` contains external YAML settings and metadata extraction rules.
- `src/ingenium_rag/ingestion/` tracks SharePoint files and incremental hashes.
- `src/ingenium_rag/etl/` parses documents and extracts project metadata.
- `src/ingenium_rag/indexing/` creates parent-child chunks and persists vectors.
- `src/ingenium_rag/retrieval/` filters, reranks, and generates cited answers.
- `data/` is local-only cache and evaluation data and is intentionally ignored by Git.
- `tests/` contains unit tests for the core contracts and pipeline behavior.

## Local services

`docker compose up -d` starts Qdrant, pgvector/Postgres, and the API container.
Never commit `.env` or downloaded SharePoint content; use `.env.example` as the
configuration template.