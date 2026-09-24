# Enterprise RAG System

A document retrieval and grounded question-answering pipeline.

## Project layout

```text
Config/config.yaml                 Runtime configuration
Data/                              Source PDF documents
Evaluation/Dataset/                Evaluation datasets
src/enterprise_rag/
  config.py                        Typed YAML configuration
  ingestion/                       PDF extraction and chunking
  retrieval/                       BM25, vector, and hybrid retrieval
  generation/                      Prompting and answer generation
  evaluation/                      Dataset loading and retrieval metrics
  rag_system.py                    Application service
  cli.py                           Installed command entry point
src/run_evaluation.py              Local module entry point
src/main.py                        Local query entry point
utils/, src/*.py                   Legacy compatibility modules
```

## Setup

Use the project's Python environment and install the package in editable mode:

```powershell
python -m pip install -e .
```

The Gemini path requires the provider's credentials in the environment. BM25 evaluation does not require Gemini, FAISS, or a sentence-transformer model.

## Run BM25 evaluation

From the repository root:

```powershell
python -m src.run_evaluation
```

After editable installation, this command is also available:

```powershell
evaluate-bm25
```

## Configuration

Edit `Config/config.yaml` to change document paths, evaluation dataset, chunking, retrieval, embedding, and generation settings. Set `ENTERPRISE_RAG_CONFIG` to use another YAML file:

```powershell
$env:ENTERPRISE_RAG_CONFIG = "C:\path\to\config.yaml"
```

## Debugging

Use the **Debug BM25 evaluation** launch configuration in VS Code. Breakpoints in `src/run_evaluation.py`, `src/enterprise_rag/rag_system.py`, and `src/enterprise_rag/evaluation/runner.py` show the complete flow without relying on global state.
