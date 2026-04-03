# CRDC-DH RAG Evaluation Pipeline

End-to-end evaluation of the CRDC Data Hub chatbot's retrieval-augmented generation (RAG) pipeline using [RAGAS](https://docs.ragas.io/) and AWS Bedrock.

## Overview

The pipeline runs in three phases:

| Phase | Script | What it does |
|-------|--------|-------------|
| 1 | `generate_testset.py` | Generates Q&A pairs from datasource documents using RAGAS synthesizers |
| 2a | `retrieve.py` | Queries the Bedrock Knowledge Base per question; scores context precision, recall, and MRR |
| 2b | `generate_and_eval.py` | Generates answers with 4 models; scores faithfulness and answer relevancy via RAGAS |
| 3 | `upload_to_phoenix.py` | Uploads all results to Arize Phoenix for visualisation |

Additional scripts:

- `build_docx_testset.py` — builds a testset from hand-curated questions (Word doc), using Claude Opus on Bedrock to generate ground-truth answers
- `build_negative_testset.py` — builds the negative testset (off-topic and mixed-relevance questions); ground truths are static refusals or Claude Opus answers
- `export_testset_to_docx.py` — exports a testset JSONL to a readable Word document for review
- `recompute_mrr.py` — recomputes MRR at a different similarity threshold without re-querying the KB
- `deepeval_eval.py` — runs DeepEval metrics (faithfulness, answer relevancy, contextual relevancy, G-Eval) against the live chatbot API using Claude Sonnet as judge
- `deepteam_eval.py` — red-team evaluation via DeepTeam; tests for adversarial vulnerabilities (prompt injection, misinformation, excessive agency, robustness); intended for nightly/pre-release runs
- `bedrock_judge.py` — DeepEval-compatible judge wrapper around AWS Bedrock (Claude Sonnet 4.6); imported by `deepeval_eval.py` and `deepteam_eval.py`
- `test_api.py` — thin utility to hit the live chatbot API (`/api/chat/question`) and collect streamed responses; used for smoke tests

## Project Structure

```ini
ragas-test/
├── *.py              # pipeline scripts (see table above)
├── pyproject.toml    # dependencies (managed with uv)
├── uv.lock
├── .env.example      # environment variable template
├── testsets/         # input Q&A testsets (SharePoint)
├── results/          # eval outputs — retrieval_results, generation_results, summaries (SharePoint)
├── docs/             # SOPs, presentations, observations (SharePoint)
└── datasource/       # KB source documents — PDFs, YMLs, CSVs, data models (SharePoint)
```

> **Note:** `testsets/`, `results/`, `docs/`, and `datasource/` are stored in SharePoint and excluded from git.

## Setup

```bash
# 1. Copy and fill in environment variables
cp .env.example .env
# Edit .env: set KB_ID, AWS_REGION, and optionally AWS_PROFILE or explicit credentials

# 2. Install dependencies
uv sync
```

## Usage

```bash
# Phase 1 — generate testset from datasource
uv run python generate_testset.py

# Phase 2a — retrieval evaluation (default input: testsets/testset.jsonl)
uv run python retrieve.py
uv run python retrieve.py --input testsets/docx_testset_good.jsonl   # custom testset

# Phase 2b — generation evaluation (default input: results/retrieval_results.jsonl)
uv run python generate_and_eval.py
uv run python generate_and_eval.py --input results/docx_testset_good_retrieval_results.jsonl

# Phase 3 — upload to Arize Phoenix
uv run python upload_to_phoenix.py

# Testset utilities
uv run python build_docx_testset.py          # hand-curated testset from Word doc
uv run python build_negative_testset.py      # negative (out-of-scope) testset
uv run python export_testset_to_docx.py      # export JSONL testset to Word for review
uv run python recompute_mrr.py               # recompute MRR at different threshold

# DeepEval / red-team (requires live chatbot API access)
uv run python deepeval_eval.py               # all questions
uv run python deepeval_eval.py --limit 3     # quick smoke test
uv run python deepteam_eval.py               # adversarial red-team evaluation
```

## Testsets

### JSONL Schema

Each record in a testset file has this structure:

```json
{
  "question":     "How do I submit a Submission Request?",
  "ground_truth": "To submit a Submission Request...",
  "contexts":     ["chunk text 1", "chunk text 2", "..."],
  "metadata": {
    "source":        "docx",
    "file_type":     "pdf",
    "question_type": "procedural"
  }
}
```

| Field | Description |
|-------|-------------|
| `question` | The evaluation question |
| `ground_truth` | Reference answer used by RAGAS metrics |
| `contexts` | KB chunks retrieved at testset build time (used as ground-truth context for retrieval scoring) |
| `metadata.source` | Where the question came from: `ragas` (auto-generated) or `docx` (hand-curated) |
| `metadata.file_type` | Source document type: `pdf`, `yml`, or `md` |
| `metadata.question_type` | Question category: `factual`, `procedural`, `conceptual`, `troubleshooting` (positive testsets); `off_topic`, `mixed_relevance` (negative testset) |

### Available Testsets

| File | Questions | Source | Notes |
|------|-----------|--------|-------|
| `testset.jsonl` | 53 | Auto-generated by RAGAS from datasource (21 YML, 21 PDF, 11 MD) | Phase 1 output |
| `docx_testset.jsonl` | 16 | Hand-curated from `CRDCDH Chatbot Questions.docx`; ground truths generated by Claude Opus 4.6 via Bedrock | All question types |
| `docx_testset_good.jsonl` | 12 | Filtered subset of `docx_testset.jsonl` — 4 questions removed where the KB lacked sufficient context to produce a reliable ground truth | Recommended for eval |
| `negative_testset.jsonl` | 23 | Hand-curated; 15 `off_topic` (fully unrelated questions) + 8 `mixed_relevance` (uses CRDC vocabulary but unanswerable) | Tests refusal behavior — the chatbot should decline all questions in this set |

## Models

| Role | Model |
|------|-------|
| Embeddings | Amazon Titan Embed Text v2 |
| RAGAS judge | Amazon Nova Pro |
| Answer generation | Nova Pro, Llama 3.3-70B, Mistral Large 3, Claude Opus 4.6 |
| Ground-truth generation (docx testset) | Claude Opus 4.6 (via Bedrock) |
| DeepEval / DeepTeam judge | Claude Sonnet 4.6 (via Bedrock) |

## Requirements

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials with access to Bedrock and the CRDC-DH Knowledge Base
- Arize Phoenix API key
