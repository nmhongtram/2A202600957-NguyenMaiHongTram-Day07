from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            if persist_directory:
                client = chromadb.PersistentClient(path=persist_directory)
                self._collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                client = chromadb.EphemeralClient()
                # ChromaDB 1.x EphemeralClient is process-global (shared singleton),
                # so delete any existing collection to guarantee a clean start.
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass
                self._collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            self._client = client  # keep client alive for ephemeral stores
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    @staticmethod
    def _to_chroma_where(metadata_filter: dict) -> dict:
        """Convert equality dict to ChromaDB where clause with explicit $eq operators."""
        clauses = [{k: {"$eq": v}} for k, v in metadata_filter.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": {**doc.metadata, "doc_id": doc.id},
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)
        scored = [
            {**r, "score": _dot(query_emb, r["embedding"])}
            for r in records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids, documents, embeddings, metadatas = [], [], [], []
            for doc in docs:
                record = self._make_record(doc)
                uid = f"{doc.id}__{self._next_index}"
                self._next_index += 1
                ids.append(uid)
                documents.append(doc.content)
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            count = self._collection.count()
            if count == 0:
                return []
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k, count),
            )
            output = []
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                })
            return output
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        For ChromaDB: uses where clause for server-side filtering.
        For in-memory: filters stored chunks by metadata_filter, then runs similarity search.
        """
        if self._use_chroma and self._collection is not None:
            if metadata_filter is None:
                return self.search(query, top_k=top_k)
            count = self._collection.count()
            if count == 0:
                return []
            query_emb = self._embedding_fn(query)
            try:
                results = self._collection.query(
                    query_embeddings=[query_emb],
                    n_results=min(top_k, count),
                    where=self._to_chroma_where(metadata_filter),
                )
            except Exception:
                return []
            output = []
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                })
            return output

        if metadata_filter is None:
            return self._search_records(query, self._store, top_k)
        filtered = [
            r for r in self._store
            if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            before = self._collection.count()
            try:
                self._collection.delete(where={"doc_id": {"$eq": doc_id}})
            except Exception:
                return False
            return self._collection.count() < before

        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
