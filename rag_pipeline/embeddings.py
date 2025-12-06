"""
Embedding backends for RAG. The MultiEmbedder exposes a simple callable
interface compatible with Chroma (List[str] -> List[List[float]]) and now
lives in its own module so different scripts can reuse it without dragging
in dataset helpers.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class MultiEmbedder:
    """
    Generic embedding wrapper that supports:
      - "sbert" / sentence-transformers
      - "openai" embedding models
    """

    def __init__(
        self,
        backend: str,
        model_name: str,
        openai_api_key_env: str = "OPENAI_API_KEY",
        show_progress_bar: bool = False,
    ):
        self.backend = backend
        self.model_name = model_name
        self.show_progress_bar = show_progress_bar

        if backend == "sbert":
            self._model = SentenceTransformer(model_name)
            self._openai_client = None
        elif backend == "openai":
            self._model = None
            self._openai_client = OpenAI()
        else:
            raise ValueError("Unsupported backend: %s" % backend)

    def _normalize(self, item):
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            return " ".join(self._normalize(sub) for sub in item)
        return str(item)

    def _embed(self, texts):
        texts = [self._normalize(t) for t in texts]
        """
        Chroma expects an embedding_function with signature
        """
        if len(texts) == 0:
            return []

        if self.backend == "sbert":
            emb = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=self.show_progress_bar,
            )
            emb = np.array(emb, dtype="float32")
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
            return emb.tolist()

        elif self.backend == "openai":
            resp = self._openai_client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [d.embedding for d in resp.data]

        else:
            raise RuntimeError("Unknown backend: %s" % self.backend)

    def __call__(self, input):
        return self._embed(input)

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, input):
        return self._embed([input])

    def name(self) -> str:
        return f"multiembedder:{self.backend}:{self.model_name}"

