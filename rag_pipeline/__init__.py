"""
Convenience exports for the rag_pipeline package so that higher-level
scripts can import the most common helpers from a single place.
"""

from .data import build_corpus, build_train_and_test_sets, load_squad_v2
from .embeddings import MultiEmbedder
from .retrieval import ChromaRAGIndex
from .rag import LLMClient, RAGQAModel
from .prompt import build_rag_prompt
from .evaluation import (
    map_citation_indices_to_doc_ids,
    extract_answer,
    extract_citations,
    extract_answerable,
    parse_gold_answers,
    build_judge_prompt,
    judge_answer,
)

__all__ = [
    "build_corpus",
    "build_train_and_test_sets",
    "load_squad_v2",
    "MultiEmbedder",
    "ChromaRAGIndex",
    "LLMClient",
    "RAGQAModel",
    "build_models",
    "build_rag_prompt",
    "map_citation_indices_to_doc_ids",
    "extract_answer",
    "extract_citations",
    "extract_answerable",
    "parse_gold_answers"
    "build_judge_prompt",
    "judge_answer", 
]
