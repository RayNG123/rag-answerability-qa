"""
Retriever backends (currently Chroma) extracted from the original
data_loader so they can be reused independently.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional

import chromadb


class ChromaRAGIndex:
    """
    Wraps a Chroma collection for RAG:
      - builds an index over your corpus docs
      - lets you query top-k docs for a question
    """

    def __init__(
        self,
        collection_name: str,
        embedder,
        persist_path: Optional[str] = None,
    ):
        """
        Args:
            collection_name: name of the Chroma collection
            embedder: MultiEmbedder instance (SBERT or OpenAI)
            persist_path: if provided, use PersistentClient at this path;
                          otherwise use in-memory client.
        """
        self.collection_name = collection_name
        self.embedder = embedder

        if persist_path is not None:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder,
        )

    def _batch_add(self, docs: List[dict]):
        ids = [d["doc_id"] for d in docs]
        documents = [d["text"] for d in docs]
        metadatas = [
            {k: v for k, v in d.items() if k not in ("doc_id", "text")}
            for d in docs
        ]
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def build_index(
        self,
        docs: Iterable[dict],
        batch_size: int = 256,
    ):
        """
        Add all docs to the Chroma collection.
        """
        self.build_index_with_callback(docs, None, batch_size=batch_size)

    def build_index_with_callback(
        self,
        docs: Iterable[dict],
        on_batch: Optional[Callable[[int], None]],
        batch_size: int = 256,
    ):
        docs = list(docs)
        if not docs:
            return

        batch_size = max(1, batch_size)
        total_batches = math.ceil(len(docs) / batch_size)

        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            self._batch_add(batch)
            if on_batch is not None:
                on_batch(len(batch))

    def query(self, question: str, k: int = 5) -> List[dict]:
        """
        Query top-k documents for a question.
        """
        res = self.collection.query(
            query_texts=[question],
            n_results=k,
        )

        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]])[0]

        results = []
        for i, doc_text, md, dist in zip(ids, docs, metas, dists):
            entry = {
                "doc_id": i,
                "text": doc_text,
                "score": float(dist) if dist is not None else None,
            }
            if md is not None:
                entry.update(md)
            results.append(entry)
        return results

