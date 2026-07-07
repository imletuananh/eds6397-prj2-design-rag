from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from officeqa_rag.chunking import Chunk


class BaseRetriever(ABC):
    @abstractmethod
    def search(
        self,
        question: str,
        k: int = 5,
        metadata_filters: dict[str, list[int]] | None = None,
    ) -> list[Chunk]:
        """Return the top-k chunks for a question."""


def _matches_filters(chunk: Chunk, metadata_filters: dict[str, list[int]] | None) -> bool:
    if not metadata_filters:
        return True
    for key, allowed_values in metadata_filters.items():
        if getattr(chunk, key) not in allowed_values:
            return False
    return True


class TfidfRetriever(BaseRetriever):
    """A simple lexical baseline retriever."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=80_000)
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(
        self,
        question: str,
        k: int = 5,
        metadata_filters: dict[str, list[int]] | None = None,
    ) -> list[Chunk]:
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1]

        results: list[Chunk] = []
        for index in ranked_indices:
            chunk = self.chunks[int(index)]
            if _matches_filters(chunk, metadata_filters):
                results.append(chunk)
            if len(results) == k:
                break
        return results


class ChromaSentenceRetriever(BaseRetriever):
    """Engineered semantic retriever using ChromaDB plus sentence-transformer embeddings."""

    def __init__(
        self,
        chunks: list[Chunk],
        persist_dir: Path,
        collection_name: str = "officeqa_engineered",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install chromadb and sentence-transformers to use the engineered retriever."
            ) from exc

        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.model = SentenceTransformer(model_name)
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = client.create_collection(collection_name)

        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.model.encode(
                [chunk.text for chunk in batch],
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                embeddings=embeddings.tolist(),
            )

    def search(
        self,
        question: str,
        k: int = 5,
        metadata_filters: dict[str, list[int]] | None = None,
    ) -> list[Chunk]:
        where = _chroma_where(metadata_filters)
        query_embedding = self.model.encode([question], normalize_embeddings=True).tolist()[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["metadatas"],
        )
        ids = result.get("ids", [[]])[0]
        return [self.chunks_by_id[chunk_id] for chunk_id in ids if chunk_id in self.chunks_by_id]


def _chroma_where(metadata_filters: dict[str, list[int]] | None) -> dict | None:
    if not metadata_filters:
        return None
    clauses = []
    for key, values in metadata_filters.items():
        if len(values) == 1:
            clauses.append({key: values[0]})
        else:
            clauses.append({key: {"$in": values}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaTfidfSvdRetriever(BaseRetriever):
    """Offline engineered retriever using ChromaDB with dense TF-IDF/SVD embeddings."""

    def __init__(
        self,
        chunks: list[Chunk],
        persist_dir: Path,
        collection_name: str = "officeqa_engineered_offline",
        n_components: int = 256,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb to use the engineered retriever.") from exc

        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=80_000, ngram_range=(1, 2))
        tfidf = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

        max_components = max(1, min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd: TruncatedSVD | None = None
        if max_components >= 2:
            self.svd = TruncatedSVD(n_components=max_components, random_state=42)
            dense = self.svd.fit_transform(tfidf)
        else:
            dense = tfidf.toarray()
        embeddings = normalize(dense).astype(float)

        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = client.create_collection(collection_name)

        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + len(batch)]
            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                embeddings=batch_embeddings.tolist(),
            )

    def search(
        self,
        question: str,
        k: int = 5,
        metadata_filters: dict[str, list[int]] | None = None,
    ) -> list[Chunk]:
        tfidf = self.vectorizer.transform([question])
        if self.svd is not None:
            query_embedding = self.svd.transform(tfidf)
        else:
            query_embedding = tfidf.toarray()
        query_embedding = normalize(query_embedding).astype(float).tolist()[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=_chroma_where(metadata_filters),
            include=["metadatas"],
        )
        ids = result.get("ids", [[]])[0]
        return [self.chunks_by_id[chunk_id] for chunk_id in ids if chunk_id in self.chunks_by_id]

