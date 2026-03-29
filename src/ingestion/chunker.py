from dataclasses import dataclass, field
import re
import hashlib


@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """
    Splits documents recursively on paragraph -> sentence -> word boundaries.
    chunk_size: ~600 tokens (~2400 chars)
    chunk_overlap: ~100 tokens (~400 chars)
    """

    def __init__(self, chunk_size: int = 2400, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def _split(self, text: str, separators: list) -> list:
        if not separators:
            return [text]

        sep = separators[0]
        parts = text.split(sep)
        chunks, current = [], ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, separators[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def chunk(self, content: str, doc_id: str, metadata: dict = None) -> list:
        raw_chunks = self._split(content, self.separators)
        chunks = []

        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            chunk_id = hashlib.md5(f"{doc_id}_{i}".encode()).hexdigest()
            chunks.append(Chunk(
                text=text,
                chunk_id=chunk_id,
                doc_id=doc_id,
                metadata={**(metadata or {}), "chunk_index": i}
            ))

        return chunks
