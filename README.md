# EDS6397 Project 2: OfficeQA Financial RAG

This repository contains a reproducible baseline-vs-engineered RAG experiment for the
Databricks OfficeQA U.S. Treasury Bulletin assignment.

The project uses 20 recent years by default: **2006 through 2025**. This is well above the
assignment's minimum of four recent years. `officeqa_full.csv` only contains 246 questions spread
across the full 1939-2025 archive, so a strict 4-6 year window (e.g. 2022-2025 or 2020-2025) only
matches 3-10 questions -- too few for a reliable Hit Rate/MRR comparison. Widening to 2006-2025
raises the strict-filtered evaluation set to 43 questions while still describable as "recent years."

## What Is Included

- A baseline TF-IDF retriever with simple fixed-size chunks.
- An engineered ChromaDB retriever with local sentence-transformer embeddings.
- Required `year` and `month` metadata on every chunk.
- K=5 retrieval metrics: Hit Rate, MRR, and Recall.
- Answer metrics: Groundedness, Factual Accuracy, and Hallucination Rate.
- Discussion-board draft in `reports/discussion_submission.md`.
- Architecture notes in `reports/architecture.md`.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Data

This repo can use the local `officeqa_full.csv` file if it is already present in the project root.
The large Treasury Bulletin transformed text corpus still needs to be downloaded or copied into
`data/officeqa/treasury_bulletins_parsed/transformed/`.

OfficeQA benchmark CSVs and corpus files are hosted on Hugging Face. Request access to:

https://huggingface.co/datasets/databricks/officeqa

Then set your token:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Download the transformed Treasury Bulletin text files:

```powershell
python -m officeqa_rag.experiment --download --csv officeqa_full.csv --limit 1
```

The first run can take time because it downloads the transformed `.txt` corpus and installs the
sentence-transformer model. Remove `--limit 1` for the full experiment.

## Run The Assignment Experiment

```powershell
python -m officeqa_rag.experiment `
  --data-dir "D:\Lectures\3. Summer 2026\EDS6397\officeqa" `
  --csv officeqa_full.csv `
  --years 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 `
  --system both `
  --engineered-embedding sentence `
  --sentence-model "D:\Lectures\3. Summer 2026\EDS6397\models\all-MiniLM-L6-v2"
```

To force exactly the assignment's four-year minimum (note: this only matches ~3 questions in
`officeqa_full.csv`, too few for a reliable scorecard):

```powershell
python -m officeqa_rag.experiment `
  --data-dir "D:\Lectures\3. Summer 2026\EDS6397\officeqa" `
  --csv officeqa_full.csv `
  --years 2022 2023 2024 2025 `
  --system both `
  --engineered-embedding sentence `
  --sentence-model "D:\Lectures\3. Summer 2026\EDS6397\models\all-MiniLM-L6-v2"
```

Outputs are written to:

- `outputs/scorecard.csv`
- `outputs/scorecard.json`
- `outputs/baseline_details.csv`
- `outputs/engineered_details.csv`

Copy the scorecard values into `reports/discussion_submission.md`.

## Architecture Summary

Baseline:

- TF-IDF sparse retrieval.
- 1,500-character chunks with 100-character overlap.
- Metadata is stored but not used for filtering.

Engineered:

- ChromaDB vector store.
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- 2,000-character chunks with 300-character overlap.
- Query-time `year` and `month` metadata filters when the question mentions dates.

Both systems use the same extractive generator so the comparison focuses on retrieval, metadata,
and chunking changes.

If the local sentence-transformer model is not available, use `--engineered-embedding offline` to
fall back to ChromaDB with dense TF-IDF/SVD embeddings.

## Question Filtering

By default, `--source-year-policy all` keeps only questions whose complete `source_files` list is
inside the selected years. This avoids scoring questions that require documents outside the local
corpus. Use `--source-year-policy any` only when you intentionally want questions that touch at
least one selected year.
