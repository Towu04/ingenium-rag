from collections.abc import Sequence


class Embedder:
    """Embedding interface. Configure a BGE-M3 implementation at deployment time."""

    model_name = "BAAI/bge-m3"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError("Configure a dense/sparse embedding provider before indexing")
