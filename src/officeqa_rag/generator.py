from __future__ import annotations

import re

from officeqa_rag.chunking import Chunk


NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|percent|%))?", re.I)


def answer_from_context(question: str, chunks: list[Chunk]) -> str:
    """Generate a grounded extractive answer from retrieved chunks.

    This keeps the assignment runnable without a paid LLM API. It returns a short quote-like
    answer from the retrieved evidence, which makes groundedness measurable and auditable.
    """
    if not chunks:
        return "Unable to determine from the retrieved Treasury Bulletin text."

    question_terms = {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", question.lower())
        if token not in {"what", "which", "when", "where", "does", "from", "with", "under"}
    }
    candidates: list[tuple[float, str]] = []
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            lowered = sentence.lower()
            term_hits = sum(1 for term in question_terms if term in lowered)
            number_bonus = 2 if NUMBER_RE.search(sentence) else 0
            candidates.append((term_hits + number_bonus, sentence))

    if not candidates:
        return chunks[0].text[:240].strip()
    best = max(candidates, key=lambda item: (item[0], len(item[1]) < 260))
    return best[1][:350].strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if len(part.strip()) > 30]

