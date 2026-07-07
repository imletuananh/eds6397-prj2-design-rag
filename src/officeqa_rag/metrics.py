from __future__ import annotations

import re
from dataclasses import dataclass

from officeqa_rag.chunking import Chunk


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class QueryMetrics:
    uid: str
    hit: float
    reciprocal_rank: float
    recall: float
    groundedness: float
    factual_accuracy: float
    hallucination_rate: float
    prediction: str


def retrieval_metrics(retrieved: list[Chunk], relevant_files: set[str], all_chunks: list[Chunk]) -> tuple[float, float, float]:
    retrieved_relevant = [chunk for chunk in retrieved if chunk.source_file in relevant_files]
    hit = 1.0 if retrieved_relevant else 0.0
    reciprocal_rank = 0.0
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.source_file in relevant_files:
            reciprocal_rank = 1.0 / rank
            break
    total_relevant = sum(1 for chunk in all_chunks if chunk.source_file in relevant_files)
    recall = len({chunk.chunk_id for chunk in retrieved_relevant}) / total_relevant if total_relevant else 0.0
    return hit, reciprocal_rank, recall


def score_answer(ground_truth: str, prediction: str, tolerance: float = 0.01) -> float:
    """Approximate OfficeQA scoring: numeric answers within tolerance, otherwise text containment."""
    gt_numbers = _numbers(ground_truth)
    pred_numbers = _numbers(prediction)
    if gt_numbers:
        for gt in gt_numbers:
            for pred in pred_numbers:
                if gt == 0 and pred == 0:
                    return 1.0
                if gt != 0 and abs(gt - pred) / abs(gt) <= tolerance:
                    return 1.0
        return 0.0

    gt_text = _normalize_text(ground_truth)
    pred_text = _normalize_text(prediction)
    return 1.0 if gt_text and gt_text in pred_text else 0.0


def answer_support_metrics(prediction: str, retrieved: list[Chunk]) -> tuple[float, float]:
    claims = [claim for claim in re.split(r"(?<=[.!?])\s+", prediction.strip()) if claim.strip()]
    if not claims:
        return 0.0, 1.0
    evidence = _normalize_text(" ".join(chunk.text for chunk in retrieved))
    supported = sum(1 for claim in claims if _claim_supported(claim, evidence))
    groundedness = supported / len(claims)
    return groundedness, 1.0 - groundedness


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUM_RE.findall(str(text)):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _claim_supported(claim: str, evidence: str) -> bool:
    normalized_claim = _normalize_text(claim)
    if normalized_claim in evidence:
        return True
    tokens = {token for token in re.findall(r"[a-z0-9]{4,}", normalized_claim)}
    if not tokens:
        return False
    overlap = sum(1 for token in tokens if token in evidence) / len(tokens)
    return overlap >= 0.65

