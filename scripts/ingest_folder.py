from pathlib import Path

from ingenium_rag.connectors.sharepoint import SharePointConnector
from ingenium_rag.pipelines.indexing_pipeline import IndexingPipeline
from ingenium_rag.storage.repository import InMemoryRepository


def ingest(path: str) -> int:
    repository = InMemoryRepository()
    files = SharePointConnector().fetch_files(Path(path))
    return IndexingPipeline(repository).run(files)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    print(f"Prepared {ingest(args.path)} child chunks")
