from domain import ChildChunk, ParentChunk, ParsedDoc


class ParentChildChunker:
    def __init__(self, parent_size: int = 1000, child_size: int = 250) -> None:
        if parent_size <= 0 or child_size <= 0 or child_size > parent_size:
            raise ValueError("chunk sizes must be positive and child_size <= parent_size")
        self.parent_size = parent_size
        self.child_size = child_size

    def chunk(self, document: ParsedDoc) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for parent_index, start in enumerate(range(0, len(document.markdown), self.parent_size)):
            parent_text = document.markdown[start : start + self.parent_size]
            parent_id = f"{document.document_id}:p{parent_index}"
            parent = ParentChunk(
                id=parent_id,
                document_id=document.document_id,
                parent_index=parent_index,
                text=parent_text,
                metadata=document.metadata,
            )
            parents.append(parent)
            for child_index, child_start in enumerate(range(0, len(parent_text), self.child_size)):
                children.append(ChildChunk(
                    id=f"{parent_id}:c{child_index}",
                    parent_id=parent_id,
                    document_id=document.document_id,
                    child_index=child_index,
                    text=parent_text[child_start : child_start + self.child_size],
                    metadata=document.metadata,
                ))
            parent.child_ids = [child.id for child in children if child.parent_id == parent_id]
        return parents, children
