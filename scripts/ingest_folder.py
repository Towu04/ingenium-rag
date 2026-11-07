from pathlib import Path

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    print(f"Prepared {ingest(args.path)} child chunks")
