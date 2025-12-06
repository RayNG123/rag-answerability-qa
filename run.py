"""
Run RAG over train (and optionally test) splits using settings from config.yaml.
Writes augmented predictions and reports metrics (F1/EM/retrieval/citation hits, optional judge accuracy).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor

from rag_pipeline import (
    ChromaRAGIndex,
    LLMClient,
    MultiEmbedder,
    RAGQAModel,
    build_rag_prompt,
    extract_answer,
    extract_citations,
    extract_answerable,
    judge_answer,
    parse_gold_answers,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file at {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def build_configs(cfg: Dict):
    def require(section: Dict, key: str):
        if key not in section:
            raise KeyError(f"Missing '{key}' in config.yaml")
        return section[key]

    artifact_dir = Path(require(cfg, "artifact_dir"))
    runs_dir = Path(cfg.get("runs_dir", artifact_dir / "runs"))

    data_cfg = require(cfg, "data")
    retr_cfg = require(cfg, "retriever")
    qa_cfg = require(cfg, "qa")
    prompt_cfg = cfg.get("prompt", {})

    paths = {
        "runs_dir": runs_dir,
        "train_file": Path(require(data_cfg, "train_file")),
        "test_file": Path(require(data_cfg, "test_file")),
    }
    require_citations = prompt_cfg.get("require_citations", True)
    return retr_cfg, qa_cfg, paths, require_citations


def make_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg: Dict, run_dir: Path, source_path: Path):
    snapshot_path = run_dir / "config_used.yaml"
    with snapshot_path.open("w") as f:
        yaml.safe_dump(cfg, f)


def load_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {csv_path}. Run the preprocess step first."
        )
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")
    return rows


def build_retriever(cfg: Dict) -> ChromaRAGIndex:
    persist_path = Path(cfg["persist_path"]) if cfg.get("persist_path") else None
    if persist_path is not None and not persist_path.exists():
        raise FileNotFoundError(
            f"No vector store found at {persist_path}. Run the vectorize step first."
        )
    embedder = MultiEmbedder(
        backend=cfg["backend"],
        model_name=cfg["model_name"],
        show_progress_bar=cfg.get("show_progress_bar", False),
    )
    return ChromaRAGIndex(
        collection_name=cfg["collection_name"],
        embedder=embedder,
        persist_path=str(persist_path) if persist_path else None,
    )


def evaluate_dataset(
    qa_model: "RAGQAModel",
    rows: List[dict],
    qa_cfg: Dict,
    require_citations: bool,
    limit: Optional[int] = None,
    show_progress: bool = True,
    save_path: Optional[Path] = None,
    progress_desc: str = "Processing",
    judge_client: Optional["LLMClient"] = None,
    num_workers: int = 1,
    strategy: str = "single",  # "single" or "two_pass"
):
    """
    Single pass over a dataset:
      - run RAG
      - compute basic retrieval/answer metrics (EM, F1)
      - compute citation / retrieval hit metrics
      - optionally judge correctness via a cheap model
      - optionally write predictions to CSV
    """

    # ----------------------------------------------------------------------
    # Setup iteration
    # ----------------------------------------------------------------------
    if not rows:
        raise ValueError("No rows to process.")

    rows = rows if limit is None else rows[:limit]

    def process_row_single(row: dict):
        question = row["question"]
        gold_answer = parse_gold_answers(row.get("answers", "{}"))
        gold_answerable = True if str(row.get("answerable")) == "True" else False

        docs = qa_model.retrieve(question, k=qa_cfg["k"])
        retrieved_doc_ids = {str(doc.get("doc_id")) for doc in docs}
        target_doc_id = str(row.get("doc_id"))
        retrieval_hit = int(target_doc_id in retrieved_doc_ids)
        if retrieval_hit == 0:
            retrieval_answerable = False
        else:
            retrieval_answerable = True

        real_answerable = gold_answerable and retrieval_answerable

        prompt = build_rag_prompt(
            question,
            docs,
            mode="with_citations" if require_citations else "no_citations",
        )
        
        raw_answer = qa_model.answer(
            prompt,
            max_tokens=qa_cfg["max_tokens"],
        )

        answer_text, answer_parsed_ok = extract_answer(
            raw_answer,
        )
        if require_citations:
            pred_doc_ids, citation_parsed_ok = extract_citations(
                raw_answer,
                docs,
            )
            citation_doc_ids = {str(doc_id) for doc_id in pred_doc_ids}
            pred_answerable = len(pred_doc_ids) > 0
            parsed_ok = answer_parsed_ok and citation_parsed_ok
        else:
            citation_doc_ids, citation_parsed_ok = [], True  # skip citation extraction if not required
            pred_answerable, answerable_parsed_ok = extract_answerable(raw_answer)
            parsed_ok = answer_parsed_ok and citation_parsed_ok and answerable_parsed_ok


        if not parsed_ok:
            citation_hit = 0
            judge_correct = 0
            answerable_correct = 0
        else:
            citation_hit = int(target_doc_id in citation_doc_ids)
            if real_answerable != pred_answerable:
                judge_correct = 0
                answerable_correct = 0
            else:
                if real_answerable is False:
                    judge_correct = 1
                else:
                    judge_correct = int(
                        judge_answer(
                            judge_client,
                            question=question,
                            model_answer=answer_text,
                            gold_answer=gold_answer,
                        )
                    ) if judge_client is not None else 0
                answerable_correct = 1

        augmented_row = dict(row)
        augmented_row["real_answerable"] = real_answerable
        augmented_row["prompt"] = prompt
        augmented_row["raw_answer"] = raw_answer
        augmented_row["predicted_answer"] = answer_text
        augmented_row["predicted_answerable"] = pred_answerable
        augmented_row["predicted_citations"] = (
            str(list(citation_doc_ids)) if require_citations else "Not Applied"
        )
        augmented_row["retrieved_doc_ids"] = str(list(retrieved_doc_ids))
        augmented_row["parsed_ok"] = int(parsed_ok)
        augmented_row["retrieval_hit"] = retrieval_hit
        augmented_row["citation_hit"] = citation_hit if require_citations else "Not Applied"
        augmented_row["answerable_correct"] = answerable_correct
        augmented_row["judge_correct"] = judge_correct
        return (
            retrieval_hit,
            citation_hit,
            judge_correct,
            answerable_correct,
            int(parsed_ok),
            int(pred_answerable and not real_answerable),  # FP
            int((not pred_answerable) and real_answerable),  # FN
            augmented_row,
        )

    def process_row_two_pass(row: dict):
        question = row["question"]
        gold_answer = parse_gold_answers(row.get("answers", "{}"))
        gold_answerable = True if str(row.get("answerable")) == "True" else False

        docs = qa_model.retrieve(question, k=qa_cfg["k"])
        retrieved_doc_ids = {str(doc.get("doc_id")) for doc in docs}
        target_doc_id = str(row.get("doc_id"))
        retrieval_hit = int(target_doc_id in retrieved_doc_ids)
        if retrieval_hit == 0:
            retrieval_answerable = False
        else:
            retrieval_answerable = True
        real_answerable = gold_answerable and retrieval_answerable
        
        if require_citations:
            first_pass_prompt = build_rag_prompt(
                question,
                docs,
                mode="citation_only",
            )
            first_pass_answer = qa_model.answer(
                first_pass_prompt,
                max_tokens=qa_cfg["max_tokens"],
            )
            pred_doc_ids, first_pass_parsed_ok = extract_citations(
                first_pass_answer,
                docs,
            )
            citation_doc_ids = {str(doc_id) for doc_id in pred_doc_ids}
            citation_docs = [d for d in docs if str(d.get("doc_id")) in citation_doc_ids]
            pred_answerable = len(citation_docs) > 0
        else:
            first_pass_prompt = build_rag_prompt(
                question,
                docs,
                mode="answerable_only",
            )
            first_pass_answer = qa_model.answer(
                first_pass_prompt,
                max_tokens=qa_cfg["max_tokens"],
            )
            pred_answerable, first_pass_parsed_ok = extract_answerable(first_pass_answer)
            citation_doc_ids = {}
            citation_docs = []


        second_pass_prompt = ""
        second_pass_answer = ""
        if pred_answerable:
            if first_pass_parsed_ok:
                second_pass_prompt = build_rag_prompt(
                    question,
                    citation_docs if require_citations else docs,
                    mode="answer_only",
                )
                second_pass_answer = qa_model.answer(
                    second_pass_prompt,
                    max_tokens=qa_cfg["max_tokens"],
                )
                answer_text, second_pass_parsed_ok = extract_answer(
                second_pass_answer,
                )
            else:
                answer_text, second_pass_parsed_ok = "", False
        else:
            answer_text, second_pass_parsed_ok = "", True
        
        parsed_ok = first_pass_parsed_ok and second_pass_parsed_ok

        if not parsed_ok:
            citation_hit = 0
            judge_correct = 0
            answerable_correct = 0
        else:
            citation_hit = int(target_doc_id in citation_doc_ids)
            if real_answerable != pred_answerable:
                judge_correct = 0
                answerable_correct = 0
            else:
                if real_answerable  is False:
                    judge_correct = 1
                else:
                    judge_correct = int(
                        judge_answer(
                            judge_client,
                            question=question,
                            model_answer=answer_text,
                            gold_answer=gold_answer,
                        )
                    ) if judge_client is not None else 0
                answerable_correct = 1

        augmented_row = dict(row)
        augmented_row["real_answerable"] = real_answerable
        augmented_row["first_pass_prompt"] = first_pass_prompt
        augmented_row["second_pass_prompt"] = second_pass_prompt
        augmented_row["first_pass_answer"] = first_pass_answer
        augmented_row["second_pass_answer"] = second_pass_answer
        augmented_row["predicted_answer"] = answer_text
        augmented_row["predicted_answerable"] = pred_answerable
        augmented_row["predicted_citations"] = (
            str(list(citation_doc_ids))
        ) if require_citations else "Not Applied"
        augmented_row["retrieved_doc_ids"] = str(list(retrieved_doc_ids))
        augmented_row["parsed_ok"] = int(parsed_ok)
        augmented_row["retrieval_hit"] = retrieval_hit
        augmented_row["citation_hit"] = citation_hit if require_citations else "Not Applied"
        augmented_row["answerable_correct"] = answerable_correct
        augmented_row["judge_correct"] = judge_correct
        return (
            retrieval_hit,
            citation_hit,
            judge_correct,
            answerable_correct,
            int(parsed_ok),
            int(pred_answerable and not real_answerable),  # FP
            int((not pred_answerable) and real_answerable),  # FN
            augmented_row,
        )

    augmented: List[dict] = []
    retrieval_hits = 0
    citation_hits = 0
    judge_corrects = 0
    parse_successes = 0
    answerable_successes = 0
    answerable_fp_total = 0
    answerable_fn_total = 0

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        worker = process_row_two_pass if strategy == "two_pass" else process_row_single
        futures = [executor.submit(worker, row) for row in rows]
        fut_iter = tqdm(as_completed(futures), total=len(futures), desc=progress_desc, unit="ex") if show_progress else as_completed(futures)
        for f in fut_iter:
            try:
                retrieval_hit, citation_hit, judge_correct, answerable_correct, parsed_ok, ans_fp, ans_fn, aug = f.result()
            except Exception as e:
                print(f"[rag_demo] worker failed on a row: {e}")
                traceback.print_exc()
                continue
            retrieval_hits += retrieval_hit
            citation_hits += citation_hit if require_citations else 0
            judge_corrects += judge_correct
            parse_successes += parsed_ok
            answerable_successes += answerable_correct
            answerable_fp_total += ans_fp
            answerable_fn_total += ans_fn
            augmented.append(aug)
        if show_progress and hasattr(fut_iter, "close"):
            fut_iter.close()

    total = len(augmented)
    if save_path is not None and augmented:
        pd.DataFrame(augmented).to_csv(save_path, index=False)

    return {
        "num_examples": total,
        "parse_success_rate": parse_successes / total if total else 0.0,
        "retrieval_hit_rate": retrieval_hits / total if total else 0.0,
        "citation_hit_rate": (citation_hits / total if total else 0.0) if require_citations else None,
        "answerable_success_rate": answerable_successes / total if total else 0.0,
        "answerable_fp_rate": answerable_fp_total / total if total else 0.0,
        "answerable_fn_rate": answerable_fn_total / total if total else 0.0,
        "judge_accuracy": judge_corrects / total if total else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the QA demo.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also evaluate on the test split after augmenting train.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (defaults to ./config.yaml).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    retr_cfg, qa_cfg, paths, require_citations = build_configs(cfg)
    eval_cfg = cfg.get("eval", {})
    judge_model = eval_cfg.get("judge_model", cfg.get("judge_model", "gpt-4.1-nano"))
    num_workers = eval_cfg.get("num_workers", cfg.get("num_workers", 1))
    strategy = eval_cfg.get("strategy", cfg.get("strategy", "single"))
    allowed_strategies = {"single", "two_pass"}
    if strategy not in allowed_strategies:
        print(f"[rag] unknown strategy '{strategy}', defaulting to 'single'")
        strategy = "single"

    api_key = cfg.get("openai_api_key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)

    limit = cfg.get("limit")
    client = LLMClient(model_name=qa_cfg["model_name"])

    qa_model = RAGQAModel(
        retriever=build_retriever(retr_cfg),
        llm_client=client,
    )

    run_dir = make_run_dir(paths["runs_dir"])
    save_config_snapshot(cfg, run_dir, args.config)
    train_out = run_dir / "train_with_preds.csv"
    test_out = run_dir / "test_with_preds.csv"
    summary_out = run_dir / "metrics_summary.json"

    train_rows = load_rows(paths["train_file"])
    train_metrics = evaluate_dataset(
        qa_model,
        train_rows,
        qa_cfg,
        require_citations=require_citations,
        limit=limit,
        show_progress=True,
        save_path=train_out,
        progress_desc="Train",
        judge_client=client,
        num_workers=num_workers,
        strategy=strategy,
    )
    print(f"[rag] train predictions -> {train_out} ({train_metrics['num_examples']} rows)")
    cite_train_val = train_metrics.get("citation_hit_rate")
    cite_train = f"{cite_train_val:.4f}" if isinstance(cite_train_val, (int, float)) else "n/a"
    print(
        f"[rag] train metrics: "
        f"Retrieval hit@k {train_metrics['retrieval_hit_rate']:.4f} | "
        f"Citation hit {cite_train} | "
        f"Answerable acc {train_metrics['answerable_success_rate']:.4f} | "
        f"Answerable FP rate {train_metrics['answerable_fp_rate']:.4f} | "
        f"Answerable FN rate {train_metrics['answerable_fn_rate']:.4f} | "
        f"Parse ok {train_metrics['parse_success_rate']:.4f} | "
        f"Judge acc {train_metrics['judge_accuracy']:.4f}"
    )

    summary_payload = {
        "train": train_metrics,
    }
    with summary_out.open("w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[rag] wrote metrics summary to {summary_out}")


if __name__ == "__main__":
    main()
