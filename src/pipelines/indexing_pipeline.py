from collections.abc import Iterable

from connectors.state_tracker import StateTracker
from domain import RawFile
from processing.chunker import ParentChildChunker
from processing.parser import DocumentParser
from storage.repository import InMemoryRepository


class IndexingPipeline:
    def __init__(
        self,
        repository: InMemoryRepository,
        parser: DocumentParser | None = None,
        chunker: ParentChildChunker | None = None,
        state_tracker: StateTracker | None = None,
    ) -> None:
        self.repository = repository
        self.parser = parser or DocumentParser()
        self.chunker = chunker or ParentChildChunker()
        self.state_tracker = state_tracker

    def run(self, files: Iterable[RawFile]) -> int:
        indexed = 0
        for raw_file in files:
            if self.state_tracker and not self.state_tracker.has_changed(
                raw_file.document_id, raw_file.content_hash
            ):
                continue
            document = self.parser.parse(raw_file)
            parents, children = self.chunker.chunk(document)
            self.repository.delete_by_document(document.document_id)
            self.repository.upsert(parents, children)
            if self.state_tracker:
                self.state_tracker.record(raw_file.document_id, raw_file.content_hash)
            indexed += len(children)
        return indexed
