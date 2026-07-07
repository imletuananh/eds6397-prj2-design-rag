from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from officeqa_rag.chunking import baseline_chunks, engineered_chunks
from officeqa_rag.data import download_officeqa, load_documents, load_questions
from officeqa_rag.filters import infer_time_filters
from officeqa_rag.generator import answer_from_context
from officeqa_rag.metrics import QueryMetrics, answer_support_metrics, retrieval_metrics, score_answer
from officeqa_rag.retrievers import ChromaSentenceRetriever, ChromaTfidfSvdRetriever, TfidfRetriever


def run_system(
    name: str,
    data_dir: Path,
    csv_path: Path,
    years: set[int],
    output_dir: Path,
    source_year_policy: str = "all",
    engineered_embedding: str = "auto",
    sentence_model: Path | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    documents = load_documents(data_dir, years)
    chunks = baseline_chunks(documents) if name == "baseline" else engineered_chunks(documents)
    questions = load_questions(csv_path, years, source_year_policy=source_year_policy)
    if limit:
        questions = questions.head(limit)

    if name == "baseline":
        retriever = TfidfRetriever(chunks)
    elif name == "engineered":
        use_sentence_model = engineered_embedding == "sentence" or (
            engineered_embedding == "auto" and sentence_model is not None and sentence_model.exists()
        )
        if use_sentence_model:
            if sentence_model is None or not sentence_model.exists():
                raise FileNotFoundError(
                    "Sentence-transformer model path does not exist. Pass --sentence-model or "
                    "use --engineered-embedding offline."
                )
            retriever = ChromaSentenceRetriever(
                chunks,
                output_dir / "vector_store",
                model_name=str(sentence_model),
            )
        else:
            retriever = ChromaTfidfSvdRetriever(chunks, output_dir / "vector_store")
    else:
        raise ValueError("System name must be 'baseline' or 'engineered'.")

    rows: list[QueryMetrics] = []
    for row in tqdm(questions.itertuples(index=False), total=len(questions), desc=name):
        metadata_filters = None
        if name == "engineered":
            metadata_filters = infer_time_filters(row.question, years)
        retrieved = retriever.search(row.question, k=5, metadata_filters=metadata_filters)
        relevant_files = set(row.source_file_list)
        hit, reciprocal_rank, recall = retrieval_metrics(retrieved, relevant_files, chunks)
        prediction = answer_from_context(row.question, retrieved)
        groundedness, hallucination = answer_support_metrics(prediction, retrieved)
        factual = score_answer(str(row.answer), prediction, tolerance=0.01)
        rows.append(
            QueryMetrics(
                uid=row.uid,
                hit=hit,
                reciprocal_rank=reciprocal_rank,
                recall=recall,
                groundedness=groundedness,
                factual_accuracy=factual,
                hallucination_rate=hallucination,
                prediction=prediction,
            )
        )

    detail = pd.DataFrame([metric.__dict__ for metric in rows])
    summary = {
        "system": name,
        "years": ",".join(str(year) for year in sorted(years)),
        "embedding": (
            "tfidf"
            if name == "baseline"
            else ("sentence-transformer" if use_sentence_model else "tfidf-svd")
        ),
        "queries": float(len(detail)),
        "hit_rate_at_5": float(detail["hit"].mean()) if not detail.empty else 0.0,
        "mrr": float(detail["reciprocal_rank"].mean()) if not detail.empty else 0.0,
        "recall": float(detail["recall"].mean()) if not detail.empty else 0.0,
        "groundedness": float(detail["groundedness"].mean()) if not detail.empty else 0.0,
        "factual_accuracy": float(detail["factual_accuracy"].mean()) if not detail.empty else 0.0,
        "hallucination_rate": float(detail["hallucination_rate"].mean()) if not detail.empty else 0.0,
    }
    return detail, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OfficeQA baseline and engineered RAG systems.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/officeqa"))
    parser.add_argument("--csv", type=Path, default=None, help="Path to officeqa_full.csv.")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2006, 2026)))
    parser.add_argument("--system", choices=["baseline", "engineered", "both"], default="both")
    parser.add_argument(
        "--engineered-embedding",
        choices=["auto", "sentence", "offline"],
        default="auto",
        help=(
            "Embedding backend for the engineered system. 'auto' uses the local sentence model "
            "when available, otherwise falls back to offline TF-IDF/SVD."
        ),
    )
    parser.add_argument(
        "--sentence-model",
        type=Path,
        default=Path("../models/all-MiniLM-L6-v2"),
        help="Local sentence-transformers model path for the engineered retriever.",
    )
    parser.add_argument(
        "--source-year-policy",
        choices=["all", "any"],
        default="all",
        help=(
            "'all' keeps only questions whose source files all fall inside --years; "
            "'any' keeps questions with at least one source file inside --years."
        ),
    )
    parser.add_argument("--download", action="store_true", help="Download OfficeQA from Hugging Face.")
    parser.add_argument("--limit", type=int, default=None, help="Optional query limit for smoke tests.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        download_officeqa(args.data_dir, token=os.getenv("HF_TOKEN"))

    csv_path = args.csv or args.data_dir / "officeqa_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download with --download after setting HF_TOKEN, "
            "or pass --csv path/to/officeqa_full.csv."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    systems = ["baseline", "engineered"] if args.system == "both" else [args.system]
    summaries = []
    for system_name in systems:
        detail, summary = run_system(
            system_name,
            args.data_dir,
            csv_path,
            set(args.years),
            args.output_dir,
            source_year_policy=args.source_year_policy,
            engineered_embedding=args.engineered_embedding,
            sentence_model=args.sentence_model,
            limit=args.limit,
        )
        detail.to_csv(args.output_dir / f"{system_name}_details.csv", index=False)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "scorecard.csv", index=False)
    (args.output_dir / "scorecard.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

