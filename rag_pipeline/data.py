"""
Dataset helpers for loading SQuAD v2 and building the corpus/train-test splits.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from datasets import load_dataset


def load_squad_v2() -> pd.DataFrame:
    """
    Load SQuAD v2 with HuggingFace datasets and return a combined dataframe
    with a `split` column indicating train/dev membership.
    """
    squad = load_dataset("squad_v2")
    train_df = pd.DataFrame(squad["train"])
    dev_df = pd.DataFrame(squad["validation"])
    train_df["split"] = "train"
    dev_df["split"] = "dev"
    return pd.concat([train_df, dev_df], ignore_index=True)


def build_corpus(df: pd.DataFrame) -> Tuple[List[dict], pd.DataFrame]:
    """
    Deduplicate contexts by (title, context) pair and assign doc_ids.
    Returns (docs_list, df_with_doc_ids).
    """
    docs: List[dict] = []
    seen = {}

    for _, row in df.iterrows():
        title = row.get("title", "") or ""
        context = row["context"]
        key = (title, context)
        if key in seen:
            continue
        doc_id = f"doc_{len(docs)}"
        seen[key] = doc_id
        docs.append(
            {
                "doc_id": doc_id,
                "title": title,
                "split": row["split"],
                "text": context,
            }
        )

    df_with_ids = df.copy()
    df_with_ids["doc_id"] = df_with_ids.apply(
        lambda row: seen[(row.get("title", "") or "", row["context"])], axis=1
    )
    return docs, df_with_ids


def build_train_and_test_sets(
    df_with_doc_id: pd.DataFrame,
    n_train_answerable: int,
    n_train_unanswerable: int,
    n_test_answerable: int,
    n_test_unanswerable: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample balanced train/dev subsets with answerable/unanswerable splits.
    """

    def is_answerable(answers):
        texts = answers.get("text", [])
        if len(texts) == 0:
            return False
        return any(t.strip() for t in texts)

    df = df_with_doc_id.copy()
    df["answerable"] = df["answers"].apply(is_answerable)

    train_df = pd.concat(
        [
            df[(df["split"] == "train") & (df["answerable"])]
            .sample(n=n_train_answerable, random_state=seed),
            df[(df["split"] == "train") & (~df["answerable"])]
            .sample(n=n_train_unanswerable, random_state=seed),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed, ignore_index=True)

    dev_df = pd.concat(
        [
            df[(df["split"] == "dev") & (df["answerable"])]
            .sample(n=n_test_answerable, random_state=seed),
            df[(df["split"] == "dev") & (~df["answerable"])]
            .sample(n=n_test_unanswerable, random_state=seed),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed, ignore_index=True)

    return train_df, dev_df

