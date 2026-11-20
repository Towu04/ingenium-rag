"""CLI entry point for querying the local Parent-Child Hybrid RAG index (`ingenium-query`)."""

import argparse
import logging
from pathlib import Path

from config import get_settings
from embedding import BGEM3Embedder
from generation.citations import format_citation
from generation.generator import GroundedGenerator
from pipeline import RAGPipeline, build_llm_client, build_rag_pipeline
from retrieval import Searcher
from storage import LocalHybridRepository


class _PassThroughReranker:
    def rerank(self, query, candidates, top_k: int = 5):
        return list(candidates)[:top_k]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Query the Ingenium local Qdrant hybrid RAG index."
    )
    parser.add_argument(
        "question", nargs="?", default=None, help="Question to ask (prompts if omitted)."
    )
    parser.add_argument("--qdrant-path", type=Path, default=settings.qdrant_path)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--building", type=str, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument(
        "--auto-filter", action="store_true", help="Infer metadata filters from query text."
    )
    parser.add_argument("--top-k", type=int, default=settings.rerank_k)
    parser.add_argument("--retrieve-k", type=int, default=settings.retrieve_k)
    parser.add_argument("--device", type=str, default=settings.device)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print retrieved & reranked parent chunks without invoking an LLM.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking and return hybrid RRF results directly.",
    )
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    if not question:
        return

    if args.no_rerank:
        repo = LocalHybridRepository(path=str(args.qdrant_path))
        embedder = BGEM3Embedder(model_name=settings.embedding_model, device=args.device)
        pipeline = RAGPipeline(
            searcher=Searcher(embedder=embedder, repository=repo),
            reranker=_PassThroughReranker(),
            generator=GroundedGenerator(llm=build_llm_client(settings)),
            retrieve_k=args.retrieve_k,
            rerank_k=args.top_k,
        )
    else:
        pipeline, repo = build_rag_pipeline(qdrant_path=args.qdrant_path, device=args.device)
    with repo:
        if args.retrieval_only:
            results = pipeline.retrieve(
                question,
                top_k=args.top_k,
                retrieve_k=args.retrieve_k,
                project=args.project,
                building=args.building,
                phase=args.phase,
                auto_filter=args.auto_filter,
            )
            for idx, res in enumerate(results, start=1):
                p = res.parent_chunk
                print(
                    f"\n--- [{idx}] score={res.score:.4f} | {p.metadata.source_path} "
                    f"| pages={p.page_numbers} | section={p.section_header} "
                    f"| phase={p.metadata.phase} | buildings={p.metadata.buildings} ---"
                )
                print(p.text[:600])
            return

        answer = pipeline.run(
            question,
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
            project=args.project,
            building=args.building,
            phase=args.phase,
            auto_filter=args.auto_filter,
        )

    print(answer.text)
    if answer.citations:
        print("\nCitations:")
        for idx, cit in enumerate(answer.citations, start=1):
            print(f"  [{idx}] {format_citation(cit)}")


if __name__ == "__main__":
    main()
