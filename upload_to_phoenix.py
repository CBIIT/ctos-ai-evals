"""
Phase 3: Upload eval results to Arize Phoenix Cloud.
Uploads:
  - crdc-testset        (dataset anchor from testset.jsonl)
  - retrieval-eval      (experiment from retrieval_results.jsonl)
  - nova-pro-generation
  - llama33-70b-generation
  - mistral-large3-generation

No models are re-run. All scores come from existing JSONL files.
Prints a local summary table before uploading.

Usage:
  uv run python upload_to_phoenix.py
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import phoenix.client as pc
from phoenix.client.resources.experiments.types import ExperimentEvaluation

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]
PHOENIX_COLLECTOR_ENDPOINT = os.environ["PHOENIX_COLLECTOR_ENDPOINT"]
PHOENIX_PROJECT = os.getenv("PHOENIX_PROJECT", "crdc-dh-eval")

# Phoenix uses PHOENIX_PROJECT_NAME env var to route experiments to a named project
os.environ["PHOENIX_PROJECT_NAME"] = PHOENIX_PROJECT

TESTSET_FILE = Path(__file__).parent / "testsets" / "testset.jsonl"
RETRIEVAL_FILE = Path(__file__).parent / "results" / "retrieval_results.jsonl"
GENERATION_FILE = Path(__file__).parent / "results" / "generation_results.jsonl"

MODEL_TO_EXPERIMENT = {
    "us.amazon.nova-pro-v1:0": "nova-pro-generation",
    "us.meta.llama3-3-70b-instruct-v1:0": "llama33-70b-generation",
    "mistral.mistral-large-3-675b-instruct": "mistral-large3-generation",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_local_summary(retrieval_records: list[dict], gen_records: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("LOCAL SUMMARY (pre-upload)")
    print("=" * 80)

    # Retrieval
    cp = _mean([r["context_precision"] for r in retrieval_records])
    cr = _mean([r["context_recall"] for r in retrieval_records])
    mrr = _mean([r["reciprocal_rank"] for r in retrieval_records])
    print(f"\nRetrieval (n={len(retrieval_records)})")
    print(f"  context_precision : {cp:.4f}")
    print(f"  context_recall    : {cr:.4f}")
    print(f"  mrr               : {mrr:.4f}")

    # Generation by model
    models = sorted({r["model"] for r in gen_records})
    print(
        f"\n{'Model':<45} {'faithfulness':>13} {'answer_rel':>11} {'declined':>9} {'n':>5}"
    )
    print(f"{'─'*45} {'─'*13} {'─'*11} {'─'*9} {'─'*5}")
    for m in models:
        recs = [r for r in gen_records if r["model"] == m]
        faith = _mean([r["faithfulness"] for r in recs])
        ar = _mean([r["answer_relevancy"] for r in recs])
        declined = sum(1 for r in recs if r.get("declined_to_answer"))
        short = m.split(".")[-1][:44]
        print(f"{short:<45} {faith:>13.4f} {ar:>11.4f} {declined:>9} {len(recs):>5}")

    # By model × file_type
    print(f"\n{'─'*80}")
    print("By model × file_type  (faithfulness / answer_relevancy):")
    file_types = sorted({r["metadata"]["file_type"] for r in gen_records})
    header = f"  {'Model':<35}" + "".join(f"  {ft:^22}" for ft in file_types)
    print(header)
    for m in models:
        short = m.split(".")[-1][:34]
        row = f"  {short:<35}"
        for ft in file_types:
            recs = [
                r
                for r in gen_records
                if r["model"] == m and r["metadata"]["file_type"] == ft
            ]
            if recs:
                row += f"  {_mean([r['faithfulness'] for r in recs]):.3f} / {_mean([r['answer_relevancy'] for r in recs]):.3f}   "
            else:
                row += f"  {'—':^22}"
        print(row)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Step 1: Upload dataset
# ---------------------------------------------------------------------------


def upload_testset(
    client: pc.Client,
    path: Path = TESTSET_FILE,
    name: str = "crdc-testset-ragas",
    description: str = "CRDC RAGAS evaluation testset",
) -> pc.resources.datasets.Dataset:
    records = load_jsonl(path)
    print(f"\nUploading dataset '{name}' ({len(records)} rows)...")

    df = pd.DataFrame(
        [
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "file_type": r["metadata"]["file_type"],
                "question_type": r["metadata"]["question_type"],
                "source": r["metadata"].get("source", ""),
            }
            for r in records
        ]
    )

    dataset = client.datasets.create_dataset(
        name=name,
        dataframe=df,
        input_keys=["question"],
        output_keys=["ground_truth"],
        metadata_keys=["file_type", "question_type", "source"],
        dataset_description=description,
    )
    url = client.experiments.get_dataset_experiments_url(dataset_id=dataset.id)
    print(f"  OK — dataset id: {dataset.id}  URL: {url}  Rows: {len(records)}")
    return dataset


# ---------------------------------------------------------------------------
# Step 2: Retrieval experiment
# ---------------------------------------------------------------------------


def upload_retrieval_experiment(
    client: pc.Client,
    dataset,
    path: Path = RETRIEVAL_FILE,
    experiment_name: str = "retrieval-eval",
) -> None:
    records = load_jsonl(path)
    by_question = {r["question"]: r for r in records}
    print(f"\nUploading retrieval experiment '{experiment_name}' ({len(records)} rows)...")

    def task(example) -> dict:
        q = example.input["question"]
        r = by_question.get(q, {})
        contexts = r.get("retrieved_contexts", [])
        return {
            "retrieved_contexts": "\n".join(
                f"{i}. {c}" for i, c in enumerate(contexts, 1)
            ),
        }

    def eval_context_precision(output, example) -> ExperimentEvaluation:
        r = by_question.get(example.input["question"], {})
        return ExperimentEvaluation(
            name="context_precision", score=r.get("context_precision", 0.0)
        )

    def eval_context_recall(output, example) -> ExperimentEvaluation:
        r = by_question.get(example.input["question"], {})
        return ExperimentEvaluation(
            name="context_recall", score=r.get("context_recall", 0.0)
        )

    def eval_mrr(output, example) -> ExperimentEvaluation:
        r = by_question.get(example.input["question"], {})
        return ExperimentEvaluation(name="mrr", score=r.get("reciprocal_rank", 0.0))

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[eval_context_precision, eval_context_recall, eval_mrr],
        experiment_name=experiment_name,
        experiment_description="Bedrock KB retrieval: context_precision, context_recall, MRR (threshold=0.50)",
        print_summary=True,
    )
    exp_id = experiment["experiment_id"]
    url = client.experiments.get_experiment_url(
        dataset_id=dataset.id, experiment_id=exp_id
    )
    print(f"  OK — retrieval-eval  URL: {url}")


# ---------------------------------------------------------------------------
# Step 3: Generation experiments (one per model)
# ---------------------------------------------------------------------------


def upload_generation_experiment(
    client: pc.Client,
    dataset,
    model_id: str,
    experiment_name: str,
    records: list[dict],
) -> None:
    by_question = {r["question"]: r for r in records}
    print(
        f"\nUploading generation experiment '{experiment_name}' ({len(records)} rows)..."
    )

    def task(example) -> dict:
        q = example.input["question"]
        r = by_question.get(q, {})
        meta = r.get("metadata", {})
        return {
            "answer": r.get("answer", ""),
            "declined_to_answer": r.get("declined_to_answer", False),
            "file_type": meta.get("file_type", ""),
            "question_type": meta.get("question_type", ""),
            "reciprocal_rank": meta.get("reciprocal_rank", 0.0),
        }

    def eval_faithfulness(output, example) -> ExperimentEvaluation:
        r = by_question.get(example.input["question"], {})
        return ExperimentEvaluation(
            name="faithfulness", score=r.get("faithfulness", 0.0)
        )

    def eval_answer_relevancy(output, example) -> ExperimentEvaluation:
        r = by_question.get(example.input["question"], {})
        return ExperimentEvaluation(
            name="answer_relevancy", score=r.get("answer_relevancy", 0.0)
        )

    def eval_declined(output, example) -> ExperimentEvaluation:
        r = by_question.get(example.input["question"], {})
        declined = r.get("declined_to_answer", False)
        return ExperimentEvaluation(
            name="declined_to_answer",
            score=1.0 if declined else 0.0,
            label="declined" if declined else "answered",
        )

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[eval_faithfulness, eval_answer_relevancy, eval_declined],
        experiment_name=experiment_name,
        experiment_description=f"Generation eval for {model_id} (judge: Nova Pro)",
        print_summary=True,
    )
    exp_id = experiment["experiment_id"]
    url = client.experiments.get_experiment_url(
        dataset_id=dataset.id, experiment_id=exp_id
    )
    print(f"  OK — {experiment_name}  URL: {url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Upload eval results to Arize Phoenix Cloud"
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Upload only the dataset; skip retrieval and generation experiments",
    )
    parser.add_argument(
        "--testset-file",
        default=None,
        help=f"Path to testset JSONL (default: {TESTSET_FILE})",
    )
    parser.add_argument(
        "--dataset-name",
        default="crdc-testset-ragas",
        help='Name for the Phoenix dataset (default: "crdc-testset-ragas")',
    )
    parser.add_argument(
        "--dataset-description",
        default="CRDC RAGAS evaluation testset",
        help='Description for the Phoenix dataset (default: "CRDC RAGAS evaluation testset")',
    )
    parser.add_argument(
        "--retrieval-file",
        default=None,
        help=f"Path to retrieval results JSONL (default: {RETRIEVAL_FILE})",
    )
    parser.add_argument(
        "--retrieval-experiment",
        default="retrieval-eval",
        help='Name for the retrieval experiment in Phoenix (default: "retrieval-eval")',
    )
    parser.add_argument(
        "--generation-file",
        default=None,
        help=f"Path to generation results JSONL (default: {GENERATION_FILE})",
    )
    args = parser.parse_args()

    testset_path = Path(args.testset_file) if args.testset_file else TESTSET_FILE
    retrieval_path = Path(args.retrieval_file) if args.retrieval_file else RETRIEVAL_FILE
    generation_path = Path(args.generation_file) if args.generation_file else GENERATION_FILE

    print(f"Phoenix client version: {pc.__version__}")
    print(f"Endpoint: {PHOENIX_COLLECTOR_ENDPOINT}")
    print(f"Project:  {PHOENIX_PROJECT}")

    client = pc.Client(
        base_url=PHOENIX_COLLECTOR_ENDPOINT,
        api_key=PHOENIX_API_KEY,
    )

    if not args.dataset_only:
        # Print local summary before uploading
        retrieval_records = load_jsonl(retrieval_path)
        gen_records = load_jsonl(generation_path)
        print_local_summary(retrieval_records, gen_records)

    # Step 1: Dataset (anchor for all experiments)
    dataset = upload_testset(client, testset_path, args.dataset_name, args.dataset_description)

    if args.dataset_only:
        print("\n" + "=" * 60)
        print("Dataset upload complete (--dataset-only).")
        print(f"View at: {PHOENIX_COLLECTOR_ENDPOINT}")
        return

    # Step 2: Retrieval experiment
    upload_retrieval_experiment(client, dataset, retrieval_path, args.retrieval_experiment)

    # Step 3: Generation experiments
    print(f"\nLoaded {len(gen_records)} generation records total.")

    for model_id, experiment_name in MODEL_TO_EXPERIMENT.items():
        model_records = [r for r in gen_records if r["model"] == model_id]
        if not model_records:
            print(f"\n[SKIP] No records found for model={model_id}")
            continue
        upload_generation_experiment(
            client, dataset, model_id, experiment_name, model_records
        )

    print("\n" + "=" * 60)
    print("All uploads complete.")
    print(f"View at: {PHOENIX_COLLECTOR_ENDPOINT}")


if __name__ == "__main__":
    main()
