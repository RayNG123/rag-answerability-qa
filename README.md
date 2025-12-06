# RAG Answerability QA

A Retrieval-Augmented Generation (RAG) pipeline for evaluating LLM answerability detection on the SQuAD v2 dataset. This project tests whether language models can correctly identify when a question is **answerable** or **unanswerable** given retrieved context.

## Overview

This pipeline:
- Retrieves relevant documents using dense embeddings (OpenAI `text-embedding-3-large`)
- Prompts LLMs to answer questions with optional citation requirements
- Evaluates answerability detection, citation accuracy, and answer correctness
- Supports multiple prompting strategies (single-pass vs two-pass)
- Runs batch experiments across different model configurations

## Features

- **Answerability Detection**: Measures how well models identify unanswerable questions
- **Citation Tracking**: Evaluates if models cite the correct source documents
- **LLM-as-Judge**: Uses a judge model to verify answer correctness
- **Batch Experiments**: Run multiple configurations and compare results
- **Two Strategies**:
  - `single`: Answer + citations in one prompt
  - `two_pass`: First identify citations, then generate answer

## Installation

```bash
# Clone the repository
git clone https://github.com/RayNG123/rag-answerability-qa.git
cd rag-answerability-qa

# Install dependencies
pip install pandas numpy tqdm pyyaml openai chromadb datasets
```

## Setup

1. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. **Configure settings** in `config.yaml`:
   ```yaml
   artifact_dir: artifacts
   runs_dir: artifacts/runs
   
   data:
     train_file: artifacts/train_df.csv
     test_file: artifacts/test_df.csv
   
   retriever:
     backend: openai
     model_name: text-embedding-3-large
     collection_name: squad_openai
     persist_path: artifacts/chroma_index
   
   qa:
     model_name: gpt-4.1-mini
     k: 5                    # Number of documents to retrieve
     temperature: 0.0
     max_tokens: 256
   
   prompt:
     require_citations: true
   
   limit: 400               # Number of examples to evaluate
   judge_model: gpt-4.1-nano
   num_workers: 50
   strategy: single         # "single" or "two_pass"
   ```

## Usage

### 1. Preprocess Data & Build Vector Index

```bash
python preprocess.py
```

This will:
- Load SQuAD v2 dataset
- Create train/test splits with answerable and unanswerable questions
- Build a ChromaDB vector index with OpenAI embeddings

### 2. Run Evaluation

```bash
python run.py
```

Outputs:
- `artifacts/runs/run_<timestamp>/train_with_preds.csv` — predictions with metrics
- `artifacts/runs/run_<timestamp>/metrics_summary.json` — aggregated metrics

### 3. Run Batch Experiments

```bash
python experiment_runner.py
```

Runs multiple configurations (different models, strategies, citation modes) and saves results to a single CSV.

## Metrics

| Metric | Description |
|--------|-------------|
| `retrieval_hit_rate` | % of questions where the gold document was retrieved |
| `citation_hit_rate` | % of questions where the model cited the gold document |
| `answerable_success_rate` | % of correct answerability predictions |
| `answerable_fp_rate` | False positive rate (predicted answerable when not) |
| `answerable_fn_rate` | False negative rate (predicted unanswerable when answerable) |
| `judge_accuracy` | % of answers marked correct by the judge model |
| `parse_success_rate` | % of model outputs successfully parsed as JSON |

## Project Structure

```
rag-answerability-qa/
├── config.yaml              # Main configuration
├── preprocess.py            # Data preprocessing & vectorization
├── run.py                   # Single experiment runner
├── experiment_runner.py     # Batch experiment runner
├── rag_pipeline/
│   ├── __init__.py
│   ├── data.py              # SQuAD loading & preprocessing
│   ├── embeddings.py        # Embedding backends
│   ├── retrieval.py         # ChromaDB index
│   ├── rag.py               # LLM client & RAG model
│   ├── prompt.py            # Prompt templates
│   └── evaluation.py        # Parsing & metrics
├── artifacts/
│   ├── train_df.csv         # Training data
│   ├── test_df.csv          # Test data
│   ├── docs.json            # Document corpus
│   ├── chroma_index/        # Vector store
│   └── runs/                # Experiment outputs
└── out/                     # Analysis outputs
```

## Experiment Configurations

The `experiment_runner.py` includes pre-defined experiments for:

- **Models**: GPT-3.5-turbo, GPT-4o-mini, GPT-4o, GPT-4.1 family
- **Strategies**: `single` (one-shot) vs `two_pass` (citation → answer)
- **Citation modes**: With citations vs without citations

## License

MIT

