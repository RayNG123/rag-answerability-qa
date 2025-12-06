"""
Run preprocessing (and optionally vectorization) for the RAG pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

from tqdm.auto import tqdm

from rag_pipeline import (
    ChromaRAGIndex,
    MultiEmbedder,
    build_corpus,
    build_train_and_test_sets,
    load_squad_v2,
)

ARTIFACT_DIR = Path("artifacts")
DOCS_FILE = ARTIFACT_DIR / "docs.json"
TRAIN_FILE = ARTIFACT_DIR / "train_df.csv"
TEST_FILE = ARTIFACT_DIR / "test_df.csv"
VECTOR_STORE_DIR = ARTIFACT_DIR / "chroma_index"


@dataclass
class DataSplitConfig:
    n_train_answerable: int = 200
    n_train_unanswerable: int = 200
    n_test_answerable: int = 100
    n_test_unanswerable: int = 100
    seed: int = 42


@dataclass
class RetrieverConfig:
    backend: str = "openai"
    model_name: str = "text-embedding-3-large"
    collection_name: str = "squad_openai"
    persist_path: str = str(VECTOR_STORE_DIR)
    show_progress_bar: bool = True


def prepare_dataset(cfg: DataSplitConfig):
    df = load_squad_v2()
    docs, df_with_doc_id = build_corpus(df)
    train_df, test_df = build_train_and_test_sets(
        df_with_doc_id=df_with_doc_id,
        n_train_answerable=cfg.n_train_answerable,
        n_train_unanswerable=cfg.n_train_unanswerable,
        n_test_answerable=cfg.n_test_answerable,
        n_test_unanswerable=cfg.n_test_unanswerable,
        seed=cfg.seed,
    )
    return docs, train_df, test_df


def save_artifacts(
    docs: List[dict],
    train_df: Any,
    test_df: Any,
    docs_path: Path = DOCS_FILE,
    train_path: Path = TRAIN_FILE,
    test_path: Path = TEST_FILE,
) -> Tuple[Path, Path, Path]:
    for path in (docs_path, train_path, test_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with docs_path.open("w") as f:
        json.dump(docs, f)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return docs_path, train_path, test_path


def load_docs(docs_path: Path = DOCS_FILE):
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Missing docs artifact at {docs_path}. Run the preprocess step first."
        )
    with docs_path.open() as f:
        return json.load(f)


def build_index(docs: List[dict], cfg: RetrieverConfig):
    persist_path = Path(cfg.persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    progress = (
        tqdm(total=len(docs), desc="Vectorizing corpus", unit="doc")
        if cfg.show_progress_bar
        else None
    )

    embedder = MultiEmbedder(
        backend=cfg.backend,
        model_name=cfg.model_name,
        show_progress_bar=False,
    )
    index = ChromaRAGIndex(
        collection_name=cfg.collection_name,
        embedder=embedder,
        persist_path=str(persist_path),
    )
    try:
        callback = (lambda batch_size: progress.update(batch_size)) if progress else None
        index.build_index_with_callback(docs, on_batch=callback)
    finally:
        if progress is not None:
            progress.close()
    print("[prep] vectorize finished.")


def run_prep():
    docs, train_df, test_df = prepare_dataset(DataSplitConfig())
    save_artifacts(docs, train_df, test_df)
    print("[prep] preprocess finished.")

    build_index(docs, RetrieverConfig())


def main():
    run_prep()


if __name__ == "__main__":
    main()
