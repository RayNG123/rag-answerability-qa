"""
Batch experiment runner: test multiple configurations and save results to a single CSV.
Usage:
    python experiment_runner.py
"""

from __future__ import annotations

import copy
import csv
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from run import (
    build_retriever,
    evaluate_dataset,
    load_rows,
    make_run_dir,
    save_config_snapshot,
)
from rag_pipeline import LLMClient, RAGQAModel


# =============================================================================
# Define your experiment configurations here
# Each experiment is a dict with:
#   - "name": a short identifier for the experiment
#   - "overrides": nested dict of config values to override from BASE_CONFIG
# =============================================================================

BASE_CONFIG_PATH = Path("config.yaml")

EXPERIMENTS: List[Dict[str, Any]] = [
    # Example experiments - modify these to your needs
    {
        "name": "gpt3.5_single_citation",
        "overrides": {
            "qa": {"model_name": "gpt-3.5-turbo"},
            "strategy": "single",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt3.5_single_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-3.5-turbo"},
            "strategy": "single",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt3.5_two_pass_citation",
        "overrides": {
            "qa": {"model_name": "gpt-3.5-turbo"},
            "strategy": "two_pass",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt3.5_two_pass_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-3.5-turbo"},
            "strategy": "two_pass",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4o_mini_single_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o-mini"},
            "strategy": "single",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4o_mini_single_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o-mini"},
            "strategy": "single",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4o_mini_two_pass_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o-mini"},
            "strategy": "two_pass",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4o_mini_two_pass_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o-mini"},
            "strategy": "two_pass",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4o_single_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o"},
            "strategy": "single",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4o_single_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o"},
            "strategy": "single",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4o_two_pass_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o"},
            "strategy": "two_pass",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4o_two_pass_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4o"},
            "strategy": "two_pass",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4.1_nano_single_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-nano"},
            "strategy": "single",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4.1_nano_single_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-nano"},
            "strategy": "single",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4.1_nano_two_pass_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-nano"},
            "strategy": "two_pass",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4.1_nano_two_pass_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-nano"},
            "strategy": "two_pass",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4.1_mini_single_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-mini"},
            "strategy": "single",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4.1_mini_single_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-mini"},
            "strategy": "single",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4.1_mini_two_pass_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-mini"},
            "strategy": "two_pass",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4.1_mini_two_pass_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1-mini"},
            "strategy": "two_pass",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4.1_single_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1"},
            "strategy": "single",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4.1_single_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1"},
            "strategy": "single",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt4.1_two_pass_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1"},
            "strategy": "two_pass",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt4.1_two_pass_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-4.1"},
            "strategy": "two_pass",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt5_single_citation",
        "overrides": {
            "qa": {"model_name": "gpt-5-chat-latest"},
            "strategy": "single",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt5_single_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-5-chat-latest"},
            "strategy": "single",
            "prompt": {"require_citations": False},
        },
    },
    {
        "name": "gpt5_two_pass_citation",
        "overrides": {
            "qa": {"model_name": "gpt-5-chat-latest"},
            "strategy": "two_pass",
            "prompt": {"require_citations": True},
        },
    },
    {
        "name": "gpt5_two_pass_no_citation",
        "overrides": {
            "qa": {"model_name": "gpt-5-chat-latest"},
            "strategy": "two_pass",
            "prompt": {"require_citations": False},
        },
    },
]


def load_base_config(path: Path = BASE_CONFIG_PATH) -> Dict:
    """Load the base configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Base config not found at {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge overrides into base config."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_experiment_config(base: Dict, overrides: Dict) -> Dict:
    """Create a full config by merging overrides into base."""
    return deep_merge(base, overrides)


def extract_config_params(cfg: Dict) -> Dict[str, Any]:
    """Extract key config parameters for the results table."""
    return {
        "retriever_backend": cfg.get("retriever", {}).get("backend"),
        "retriever_model": cfg.get("retriever", {}).get("model_name"),
        "qa_model": cfg.get("qa", {}).get("model_name"),
        "k": cfg.get("qa", {}).get("k"),
        "max_tokens": cfg.get("qa", {}).get("max_tokens"),
        "strategy": cfg.get("strategy"),
        "require_citations": cfg.get("prompt", {}).get("require_citations"),
        "limit": cfg.get("limit"),
    }


def run_single_experiment(
    experiment_name: str,
    cfg: Dict,
    base_runs_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single experiment and return metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}")

    # Setup paths
    retr_cfg = cfg["retriever"]
    qa_cfg = cfg["qa"]
    prompt_cfg = cfg.get("prompt", {})
    eval_cfg = cfg.get("eval", {})

    require_citations = prompt_cfg.get("require_citations", True)
    judge_model = eval_cfg.get("judge_model", cfg.get("judge_model", "gpt-4.1-nano"))
    num_workers = eval_cfg.get("num_workers", cfg.get("num_workers", 1))
    strategy = eval_cfg.get("strategy", cfg.get("strategy", "single"))
    limit = cfg.get("limit")

    # Set API key if provided
    api_key = cfg.get("openai_api_key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)

    # Build model
    client = LLMClient(model_name=qa_cfg["model_name"])
    qa_model = RAGQAModel(
        retriever=build_retriever(retr_cfg),
        llm_client=client,
    )

    # Create run directory
    run_dir = make_run_dir(base_runs_dir)
    run_dir = run_dir.parent / f"{run_dir.name}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, run_dir, BASE_CONFIG_PATH)

    # Load data
    train_file = Path(cfg["data"]["train_file"])
    train_rows = load_rows(train_file)

    # Run evaluation
    train_out = run_dir / "train_with_preds.csv"
    train_metrics = evaluate_dataset(
        qa_model,
        train_rows,
        qa_cfg,
        require_citations=require_citations,
        limit=limit,
        show_progress=True,
        save_path=train_out,
        progress_desc=f"[{experiment_name}]",
        judge_client=client,
        num_workers=num_workers,
        strategy=strategy,
    )

    # Build result row
    result = {
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        **extract_config_params(cfg),
        **{f"train_{k}": v for k, v in train_metrics.items()},
    }

    # Save individual metrics
    metrics_out = run_dir / "metrics_summary.json"
    with metrics_out.open("w") as f:
        json.dump({"train": train_metrics, "config": extract_config_params(cfg)}, f, indent=2)

    print(f"[{experiment_name}] Done! Metrics saved to {metrics_out}")
    return result


def main():
    """Run all experiments and save results to CSV."""
    print("=" * 60)
    print("BATCH EXPERIMENT RUNNER")
    print("=" * 60)

    # Load base config
    base_cfg = load_base_config()
    base_runs_dir = Path(base_cfg.get("runs_dir", "artifacts/runs"))

    # Results collection
    all_results: List[Dict[str, Any]] = []
    failed_experiments: List[str] = []

    # Run each experiment
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        overrides = exp.get("overrides", {})

        try:
            cfg = build_experiment_config(base_cfg, overrides)
            result = run_single_experiment(exp_name, cfg, base_runs_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Experiment '{exp_name}' failed: {e}")
            traceback.print_exc()
            failed_experiments.append(exp_name)
            continue

    # Save all results to CSV
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_csv = base_runs_dir / f"experiment_results_{timestamp}.csv"
        df = pd.DataFrame(all_results)

        # Reorder columns for readability
        priority_cols = [
            "experiment_name",
            "qa_model",
            "strategy",
            "require_citations",
            "train_num_examples",
            "train_retrieval_hit_rate",
            "train_citation_hit_rate",
            "train_answerable_success_rate",
            "train_judge_accuracy",
            "train_parse_success_rate",
        ]
        other_cols = [c for c in df.columns if c not in priority_cols]
        ordered_cols = [c for c in priority_cols if c in df.columns] + other_cols
        df = df[ordered_cols]

        df.to_csv(results_csv, index=False)
        print(f"\n{'='*60}")
        print(f"ALL RESULTS SAVED TO: {results_csv}")
        print(f"{'='*60}")

        # Print summary table
        print("\nSummary:")
        print(df[["experiment_name", "train_retrieval_hit_rate", "train_citation_hit_rate", 
                  "train_answerable_success_rate", "train_judge_accuracy"]].to_string(index=False))

    if failed_experiments:
        print(f"\n[WARNING] Failed experiments: {failed_experiments}")

    return all_results


if __name__ == "__main__":
    main()

