from domain import SearchResult
import numpy as np
from typing import Sequence
import torch
from FlagEmbedding import FlagReranker


def _detect_best_device() -> str:
    """Selects CUDA, Apple Silicon MPS, or CPU automatically."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class BGEM3Reranker:
    """Cross-Encoder re-ranker (BAAI/bge-reranker-v2-m3) for post-retrieval precision filtering."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 2048,
    ) -> None:
        self.device = device or _detect_best_device()
        self.batch_size = batch_size
        self.max_length = max_length
        use_fp16 = self.device == "cuda"

        self._reranker = FlagReranker(
            model_name,
            use_fp16=use_fp16,
            devices=self.device,
        )

    @staticmethod
    def _format_passage(item: SearchResult) -> str:
        """Prefix parent text with compact provenance tags so the cross-encoder sees document context."""
        parent = item.parent_chunk
        meta = parent.metadata
        tags = [
            f"Project: {meta.project}" if meta.project else None,
            f"Building: {meta.building}" if meta.building else None,
            f"Phase: {meta.phase}" if meta.phase else None,
            f"Section: {parent.section_header}" if parent.section_header else None,
        ]
        active = [t for t in tags if t]
        return f"[{' | '.join(active)}]\n{parent.text}" if active else parent.text

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Scores [query, parent_chunk.text] pairs with cross-attention and returns the top_k."""
        if not candidates:
            return []

        pairs = [[query, self._format_passage(item)] for item in candidates]
        scores = self._reranker.compute_score(
            pairs,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=True,
        )

        # FlagReranker returns a single float if len(pairs) == 1
        if isinstance(scores, (float, int, np.floating)):
            scores = [float(scores)]

        reranked = [
            SearchResult(
                parent_chunk=item.parent_chunk,
                matched_child_id=item.matched_child_id,
                score=float(rerank_score),
            )
            for item, rerank_score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
