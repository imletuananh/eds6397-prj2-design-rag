from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(r"treasury_bulletin_(?P<year>\d{4})_(?P<month>\d{2})\.txt")


@dataclass(frozen=True)
class Document:
    source_file: str
    year: int
    month: int
    text: str


def download_officeqa(data_dir: Path, token: str | None = None) -> Path:
    """Download the gated OfficeQA CSV and transformed text corpus from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub before downloading OfficeQA.") from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id="databricks/officeqa",
            repo_type="dataset",
            local_dir=data_dir,
            token=token,
            allow_patterns=[
                "officeqa_full.csv",
                "treasury_bulletins_parsed/transformed/*.txt",
            ],
        )
    )


def parse_source_files(value: object) -> list[str]:
    """Normalize OfficeQA source_files values into a list of filenames."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            return [Path(str(item)).name for item in parsed]
        except (SyntaxError, ValueError):
            pass
    parts = re.split(r"[;,]\s*|\s+\|\s+", text)
    return [Path(part.strip()).name for part in parts if part.strip()]


def file_year_month(source_file: str) -> tuple[int, int] | None:
    match = FILENAME_RE.search(Path(source_file).name)
    if not match:
        return None
    return int(match.group("year")), int(match.group("month"))


def load_questions(csv_path: Path, years: set[int], source_year_policy: str = "all") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"uid", "question", "answer", "source_files"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Question CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["source_file_list"] = df["source_files"].apply(parse_source_files)
    df["source_years"] = df["source_file_list"].apply(
        lambda files: sorted({ym[0] for file in files if (ym := file_year_month(file))})
    )
    if source_year_policy == "all":
        mask = df["source_years"].apply(lambda ys: bool(ys) and set(ys).issubset(years))
    elif source_year_policy == "any":
        mask = df["source_years"].apply(lambda ys: bool(set(ys).intersection(years)))
    else:
        raise ValueError("source_year_policy must be 'all' or 'any'.")
    return df[mask].reset_index(drop=True)


def transformed_text_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir / "treasury_bulletins_parsed" / "transformed",
        data_dir / "transformed",
        data_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("treasury_bulletin_*.txt")):
            return candidate
    raise FileNotFoundError(
        "Could not find transformed Treasury Bulletin .txt files under the data directory."
    )


def load_documents(data_dir: Path, years: set[int]) -> list[Document]:
    text_dir = transformed_text_dir(data_dir)
    documents: list[Document] = []
    for path in sorted(text_dir.glob("treasury_bulletin_*.txt")):
        parsed = file_year_month(path.name)
        if parsed is None:
            continue
        year, month = parsed
        if year not in years:
            continue
        documents.append(
            Document(
                source_file=path.name,
                year=year,
                month=month,
                text=path.read_text(encoding="utf-8", errors="ignore"),
            )
        )
    if not documents:
        raise FileNotFoundError(f"No Treasury Bulletin text files found for years {sorted(years)}.")
    return documents

