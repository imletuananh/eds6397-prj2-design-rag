from __future__ import annotations

import re
from dataclasses import dataclass

from officeqa_rag.data import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    year: int
    month: int
    chunk_index: int

    @property
    def metadata(self) -> dict[str, int | str]:
        return {
            "source_file": self.source_file,
            "year": self.year,
            "month": self.month,
            "chunk_index": self.chunk_index,
        }


def _window_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        window = normalized[start:end]
        if end < len(normalized):
            boundary = max(window.rfind("\n\n"), window.rfind(". "))
            if boundary > chunk_size * 0.55:
                end = start + boundary + 1
                window = normalized[start:end]
        chunks.append(window.strip())
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def baseline_chunks(documents: list[Document]) -> list[Chunk]:
    """Simple fixed-size chunks used as the baseline system."""
    return make_chunks(documents, chunk_size=1_500, overlap=100)


def engineered_chunks(documents: list[Document]) -> list[Chunk]:
    """Overlapping chunks sized for table context and practical local embedding."""
    return make_chunks(documents, chunk_size=2_000, overlap=300)


def make_chunks(documents: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(_window_text(document.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_file}:{index}",
                    text=text,
                    source_file=document.source_file,
                    year=document.year,
                    month=document.month,
                    chunk_index=index,
                )
            )
    return chunks

