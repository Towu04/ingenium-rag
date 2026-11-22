"""Cross-Encoder reranker (`BAAI/bge-reranker-v2-m3`) for post-retrieval precision filtering."""

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from FlagEmbedding import FlagReranker

from config import detect_device
from domain import SearchResult

logger = logging.getLogger(__name__)


class BGEM3Reranker:
    """Scores `[query, parent_chunk]` pairs with cross-attention and returns the top-k."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 1024,
        min_score: float = 0.0,
    ) -> None:
        self.device = detect_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_score = min_score
        self._reranker = FlagReranker(
            model_name,
            use_fp16=(self.device in {"cuda", "mps"}),
            devices=self.device,
        )

    @staticmethod
    def _format_passage(item: SearchResult) -> str:
        """Prefix parent text with compact provenance tags so the cross-encoder sees document context."""
        parent = item.parent_chunk
        meta = parent.metadata
        file_stem = Path(meta.source_path).stem if meta.source_path else None
        include_building = bool(meta.building) and len(meta.buildings) <= 3
        tags = [
            f"File: {file_stem}" if file_stem else None,
            f"Project: {meta.project}" if meta.project else None,
            f"Building: {meta.building}" if include_building else None,
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
        min_score: float | None = None,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        threshold = self.min_score if min_score is None else min_score
        pairs = [[query, self._format_passage(item)] for item in candidates]
        scores = self._reranker.compute_score(
            pairs,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=True,
        )
        if isinstance(scores, (float, int, np.floating)):
            scores = [float(scores)]

        score_floats = [float(s) for s in scores]
        finite_scores = [s for s in score_floats if math.isfinite(s)]
        non_finite = len(score_floats) - len(finite_scores)
        if non_finite:
            logger.warning(
                "BGEM3Reranker produced %d non-finite score(s) on device=%s; dropping them.",
                non_finite,
                self.device,
            )

        reranked = [
            SearchResult(
                parent_chunk=item.parent_chunk,
                matched_child_id=item.matched_child_id,
                matched_child_text=item.matched_child_text,
                retrieval_score=(
                    item.retrieval_score if item.retrieval_score is not None else item.score
                ),
                score=score,
            )
            for item, score in zip(candidates, score_floats)
            if math.isfinite(score) and score >= threshold
        ]
        if not reranked:
            max_s = max(finite_scores) if finite_scores else float("nan")
            logger.warning(
                "BGEM3Reranker dropped all %d candidate(s) below min_score=%.4f (max_score=%.4f).",
                len(candidates),
                threshold,
                max_s,
            )
            return []

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
