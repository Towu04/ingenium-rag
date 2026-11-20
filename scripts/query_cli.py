"""CLI entry point for querying the local Parent-Child Hybrid RAG index."""

import argparse
import logging
from pathlib import Path

from config import get_settings
from generation.citations import format_citation
from generation.generator import ExtractiveStubLLM, GroundedGenerator, LLMClient, OpenAICompatibleLLM
from pipelines.rag_pipeline import RAGPipeline
from retrieval.reranker import BGEM3Reranker
from retrieval.searcher import Searcher
from storage.embedder import BGEM3Embedder
from storage.repository import LocalHybridRepository


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Query the Ingenium local Qdrant hybrid RAG index."
    )
    parser.add_argument("question", nargs="?", default=None, help="Question to ask (prompts interactively if omitted).")
    parser.add_argument("--qdrant-path", type=Path, default=settings.qdrant_path)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--building", type=str, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--auto-filter", action="store_true", help="Infer metadata filters from query text.")
    parser.add_argument("--top-k", type=int, default=settings.rerank_k)
    parser.add_argument("--retrieve-k", type=int, default=settings.retrieve_k)
    parser.add_argument("--device", type=str, default=settings.device)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print retrieved & reranked parent chunks without invoking an LLM.",
    )
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    if not question:
        return

    repo = LocalHybridRepository(path=str(args.qdrant_path))
    embedder = BGEM3Embedder(model_name=settings.embedding_model, device=args.device)
    reranker = BGEM3Reranker(model_name=settings.reranker_model, device=args.device)
    searcher = Searcher(embedder=embedder, repository=repo)

    if args.retrieval_only:
        candidates = searcher.search(
            question,
            top_k_parents=max(args.retrieve_k, args.top_k),
            project=args.project,
            building=args.building,
            phase=args.phase,
        )
        reranked = reranker.rerank(question, candidates, top_k=args.top_k)
        for idx, res in enumerate(reranked, start=1):
            p = res.parent_chunk
            print(
                f"\n--- [{idx}] score={res.score:.4f} | {p.metadata.source_path} "
                f"| pages={p.page_numbers} | section={p.section_header} ---"
            )
            print(p.text[:600])
        return

    llm: LLMClient
    if settings.llm_base_url and settings.llm_model:
        llm = OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
        )
    else:
        llm = ExtractiveStubLLM()

    pipeline = RAGPipeline(
        searcher=searcher,
        reranker=reranker,
        generator=GroundedGenerator(llm=llm),
        retrieve_k=args.retrieve_k,
        rerank_k=args.top_k,
    )
    answer = pipeline.run(
        question,
        top_k=args.top_k,
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
