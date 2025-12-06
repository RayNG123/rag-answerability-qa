"""
Prompting, LLM client, and high-level RAG orchestration utilities.
Supports:
  - single pass with citations (JSON answer + citations)
  - single pass without citations (JSON answer only)
  - two pass: first pass citation-only JSON, second pass answer-only.
"""

from __future__ import annotations

from typing import List

from openai import OpenAI

class LLMClient:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model_name = model_name

    def _extract_text(self, resp) -> str:
        """Robustly extract text from OpenAI response objects."""
        if resp is None:
            return ""
        try:
            output_text = getattr(resp, "output_text", None)
            if isinstance(output_text, str) and output_text:
                return output_text
        except Exception:
            pass
        try:
            output = getattr(resp, "output", None)
            if output:
                out = output[0]
                if hasattr(out, "content") and out.content:
                    return out.content[0].text
        except Exception:
            pass
        try:
            choices = getattr(resp, "choices", None)
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg and isinstance(msg, dict):
                    return msg.get("content", "") or ""
                if msg and hasattr(msg, "content"):
                    return msg.content or ""
        except Exception:
            pass
        try:
            if hasattr(resp, "text"):
                return resp.text
        except Exception:
            pass
        return ""

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        kwargs = {
            "model": self.model_name,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        resp = self.client.responses.create(**kwargs)
        return resp.output_text


class RAGQAModel:
    """
    High-level RAG question answering model.
    """

    def __init__(
        self,
        retriever,
        llm_client: LLMClient,
        max_docs_chars: int = 8000,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.max_docs_chars = max_docs_chars

    def _truncate_docs(self, docs: List[str]) -> List[str]:
        return [str(doc)[:self.max_docs_chars] for doc in docs]

    def retrieve(self, question: str, k: int = 5):
        return self.retriever.query(question, k=k)
       
    def answer(
        self,
        prompt,
        max_tokens: int = 256,
    ) -> str:
        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)
