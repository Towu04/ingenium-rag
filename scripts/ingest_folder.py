from pathlib import Path

<<<<<<< HEAD
from ingenium_rag.connectors.sharepoint import SharePointConnector
from ingenium_rag.pipelines.indexing_pipeline import IndexingPipeline
from ingenium_rag.storage.repository import InMemoryRepository


def ingest(path: str) -> int:
    repository = InMemoryRepository()
    files = SharePointConnector().fetch_files(Path(path))
    return IndexingPipeline(repository).run(files)
=======
from ingenium_rag.etl.parser import DocumentParser
from ingenium_rag.indexing.chunker import ParentChildChunker


def ingest(path: str) -> int:
    parser = DocumentParser()
    chunker = ParentChildChunker()
    count = 0
    for file in Path(path).rglob("*.md"):
        _, children = chunker.chunk(parser.parse(file))
        count += len(children)
    return count
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    print(f"Prepared {ingest(args.path)} child chunks")
