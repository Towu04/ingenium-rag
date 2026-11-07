from collections.abc import Sequence


class BGEEmbedder:
    model_name = "BAAI/bge-m3"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError("Configure a BGE-M3 embedding provider before indexing")
