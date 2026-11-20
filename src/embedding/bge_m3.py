"""Hybrid dense (1024-dim float32) + sparse lexical embedder backed by BAAI/bge-m3."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from config import detect_device
from domain import ChildChunk, EmbeddedChildBatch, EncodedQuery


def _to_sparse_dict(raw_weights: Any) -> dict[int, float]:
    """Convert FlagEmbedding `{token_id: np.float32}` lexical weights into `{int: float}`."""
    return {int(k): float(v) for k, v in raw_weights.items() if float(v) > 0}


class BGEM3Embedder:
    """Generates dense and sparse representations in a single BGE-M3 forward pass."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 1024,
    ) -> None:
        self.device = detect_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = BGEM3FlagModel(
            model_name,
            use_fp16=(self.device == "cuda"),
            devices=self.device,
        )

    @property
    def tokenizer(self) -> Any:
        return self._model.tokenizer

    def embed_children(self, children: Sequence[ChildChunk]) -> EmbeddedChildBatch:
        if not children:
            return EmbeddedChildBatch(
                dense_vecs=np.empty((0, 1024), dtype=np.float32),
                sparse_weights=[],
            )
        texts = [c.embedding_text or c.text for c in children]
        output: dict[str, Any] = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
        )
        return EmbeddedChildBatch(
            dense_vecs=np.asarray(output["dense_vecs"], dtype=np.float32),
            sparse_weights=[_to_sparse_dict(w) for w in output["lexical_weights"]],
        )

    def embed_query(self, query: str) -> EncodedQuery:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        output: dict[str, Any] = self._model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
        )
        return EncodedQuery(
            dense_vec=np.asarray(output["dense_vecs"][0], dtype=np.float32),
            sparse_weights=_to_sparse_dict(output["lexical_weights"][0]),
        )
