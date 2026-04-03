"""
Phase 2b: Generation Evaluation (3 models)
For each model, generates answers from retrieved_contexts, then evaluates
faithfulness + answer_relevancy via RAGAS (Nova Pro as judge).
Writes generation_results.jsonl and a timestamped generation_summary_<ts>.json.

Usage:  uv run python generate_and_eval.py
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import boto3
import instructor
import litellm
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRIEVAL_RESULTS_FILE = Path(__file__).parent / "results" / "retrieval_results.jsonl"
OUTPUT_FILE            = Path(__file__).parent / "results" / "generation_results.jsonl"

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")

NOVA_PRO_MODEL_ID    = "us.amazon.nova-pro-v1:0"
LLAMA_MODEL_ID       = "us.meta.llama3-3-70b-instruct-v1:0"
MISTRAL_MODEL_ID     = "mistral.mistral-large-3-675b-instruct"  # verify before run; may need us. prefix
CLAUDE_OPUS_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

MODELS = [NOVA_PRO_MODEL_ID, LLAMA_MODEL_ID, MISTRAL_MODEL_ID, CLAUDE_OPUS_MODEL_ID]

SYSTEM_PROMPT = (
    "You are a helpful assistant for the NCI Cancer Research Data Commons (CRDC). "
    "Answer the question using only the provided context. "
    "If the context does not contain enough information to answer, "
    "say 'I don't have enough information to answer this question.' "
    "Be concise and accurate."
)

DECLINE_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "cannot answer",
    "can't answer",
    "no information",
    "not enough information",
    "insufficient information",
    "context does not contain",
    "context does not provide",
    "not mentioned in the context",
    "not provided in the context",
]


# ---------------------------------------------------------------------------
# AWS Setup  (same pattern as retrieve.py / generate_testset.py)
# ---------------------------------------------------------------------------

def build_aws_clients():
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        session = boto3.Session(region_name=AWS_REGION)
    else:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    creds = session.get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"]     = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    os.environ["AWS_REGION_NAME"]       = AWS_REGION
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)

    bedrock_rt = session.client("bedrock-runtime", region_name=AWS_REGION)
    return session, bedrock_rt


def build_ragas_llm_and_embeddings(session):
    # Nova Pro judges all 3 models including itself —
    # self-serving bias is a known caveat, acceptable given approved model constraints
    litellm.drop_params = True
    litellm_client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
    litellm_model  = f"bedrock/converse/{NOVA_PRO_MODEL_ID}"

    ragas_llm = llm_factory(
        model=litellm_model,
        provider="bedrock",
        client=litellm_client,
        adapter="litellm",
    )

    bedrock_embed_client = session.client("bedrock-runtime")
    lc_embeddings = BedrockEmbeddings(
        model_id=TITAN_EMBED_MODEL_ID,
        region_name=AWS_REGION,
        client=bedrock_embed_client,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
    return ragas_llm, ragas_embeddings


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_user_message(retrieved_contexts: list[str], question: str) -> str:
    context_block = "\n".join(
        f"{i}. {chunk}" for i, chunk in enumerate(retrieved_contexts, 1)
    )
    return f"Context:\n{context_block}\n\nQuestion: {question}"


def generate_answer(
    model_id: str,
    question: str,
    retrieved_contexts: list[str],
    bedrock_rt,
) -> str | None:
    user_msg = _build_user_message(retrieved_contexts, question)
    try:
        resp = bedrock_rt.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            inferenceConfig={"temperature": 0.0},
        )
        return resp["output"]["message"]["content"][0]["text"]
    except Exception as exc:
        print(f"  [WARN] Generation failed for model={model_id}: {exc}")
        return None


def _is_declined(answer: str | None) -> bool:
    if not answer:
        return False
    lower = answer.lower()
    return any(phrase in lower for phrase in DECLINE_PHRASES)


# ---------------------------------------------------------------------------
# RAGAS Generation Metrics
# ---------------------------------------------------------------------------

def compute_ragas_generation(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    ragas_llm,
    ragas_embeddings,
    idx: int,
    total: int,
    model_id: str,
) -> dict:
    t0 = time.time()
    print(f"  [{idx:2}/{total}] RAGAS eval ({model_id[-20:]}): {question[:60]}")
    if not answer or not retrieved_contexts:
        print(f"         → skipped (no answer or no contexts) — faith=0.0000, ar=0.0000")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    try:
        hf_dataset = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [retrieved_contexts],
        })
        result = evaluate(
            dataset=hf_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        df  = result.to_pandas()
        row = df.iloc[0]
        faith = float(row.get("faithfulness",     0.0) or 0.0)
        ar    = float(row.get("answer_relevancy", 0.0) or 0.0)
        elapsed = time.time() - t0
        print(f"         → faith={faith:.4f}, ar={ar:.4f}  ({elapsed:.1f}s)")
        return {"faithfulness": faith, "answer_relevancy": ar}
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"         → [WARN] RAGAS eval failed ({elapsed:.1f}s): {exc}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _group_stats(records: list[dict], key: str) -> dict:
    buckets: dict = defaultdict(lambda: {"faith": [], "ar": []})
    for r in records:
        k = r["metadata"][key]
        buckets[k]["faith"].append(r["faithfulness"])
        buckets[k]["ar"].append(r["answer_relevancy"])
    return {
        k: {
            "faithfulness":     _mean(v["faith"]),
            "answer_relevancy": _mean(v["ar"]),
            "n": len(v["faith"]),
        }
        for k, v in sorted(buckets.items())
    }


def print_summary(all_records: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)

    # By model
    by_model: dict = defaultdict(lambda: {"faith": [], "ar": [], "declined": 0})
    for r in all_records:
        m = r["model"]
        by_model[m]["faith"].append(r["faithfulness"])
        by_model[m]["ar"].append(r["answer_relevancy"])
        if r.get("declined_to_answer"):
            by_model[m]["declined"] += 1

    print(f"\n{'Model':<45} {'faithfulness':>13} {'answer_rel':>11} {'declined':>9} {'n':>5}")
    print(f"{'─'*45} {'─'*13} {'─'*11} {'─'*9} {'─'*5}")
    for model, vals in sorted(by_model.items()):
        n = len(vals["faith"])
        print(
            f"{model:<45} {_mean(vals['faith']):>13.4f}"
            f" {_mean(vals['ar']):>11.4f}"
            f" {vals['declined']:>9}"
            f" {n:>5}"
        )

    # By model + file_type
    print(f"\n{'─'*80}")
    print("By model × file_type:")
    cross: dict = defaultdict(lambda: {"faith": [], "ar": []})
    for r in all_records:
        key = (r["model"], r["metadata"]["file_type"])
        cross[key]["faith"].append(r["faithfulness"])
        cross[key]["ar"].append(r["answer_relevancy"])
    print(f"  {'Model':<45} {'file_type':<8} {'faithfulness':>13} {'answer_rel':>11} {'n':>5}")
    print(f"  {'─'*45} {'─'*8} {'─'*13} {'─'*11} {'─'*5}")
    for (model, ft), vals in sorted(cross.items()):
        n = len(vals["faith"])
        print(
            f"  {model:<45} {ft:<8}"
            f" {_mean(vals['faith']):>13.4f}"
            f" {_mean(vals['ar']):>11.4f}"
            f" {n:>5}"
        )

    # Zero-MRR questions only (hardest questions — tests hallucination resistance)
    zero_mrr = [r for r in all_records if r["metadata"].get("reciprocal_rank", 1.0) == 0.0]
    if zero_mrr:
        print(f"\n{'─'*80}")
        print("Zero-MRR questions only (hardest — faithfulness = hallucination resistance):")
        zm_by_model: dict = defaultdict(lambda: {"faith": [], "ar": []})
        for r in zero_mrr:
            zm_by_model[r["model"]]["faith"].append(r["faithfulness"])
            zm_by_model[r["model"]]["ar"].append(r["answer_relevancy"])
        print(f"  {'Model':<45} {'faithfulness':>13} {'answer_rel':>11} {'n':>5}")
        print(f"  {'─'*45} {'─'*13} {'─'*11} {'─'*5}")
        for model, vals in sorted(zm_by_model.items()):
            n = len(vals["faith"])
            print(
                f"  {model:<45}"
                f" {_mean(vals['faith']):>13.4f}"
                f" {_mean(vals['ar']):>11.4f}"
                f" {n:>5}"
            )

    # Expected-decline metrics (off_topic questions)
    off_topic_recs = [r for r in all_records if r.get("expected_decline")]
    if off_topic_recs:
        print(f"\n{'─'*80}")
        print("Expected-decline metrics (off_topic questions — correct behavior = decline):")
        ed_by_model: dict = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in off_topic_recs:
            m = r["model"]
            ed_by_model[m]["n"] += 1
            if r.get("declined_to_answer"):
                ed_by_model[m]["correct"] += 1
        print(f"  {'Model':<45} {'expected_n':>10} {'correct_declines':>16} {'tp_rate':>8}")
        print(f"  {'─'*45} {'─'*10} {'─'*16} {'─'*8}")
        for model, vals in sorted(ed_by_model.items()):
            tp = vals["correct"] / vals["n"] if vals["n"] else 0.0
            print(f"  {model:<45} {vals['n']:>10} {vals['correct']:>16} {tp:>8.2%}")


def write_summary_json(all_records: list[dict]) -> Path:
    by_model: dict = defaultdict(lambda: {"faith": [], "ar": [], "declined": 0, "n": 0})
    for r in all_records:
        m = r["model"]
        by_model[m]["faith"].append(r["faithfulness"])
        by_model[m]["ar"].append(r["answer_relevancy"])
        by_model[m]["n"] += 1
        if r.get("declined_to_answer"):
            by_model[m]["declined"] += 1

    zero_mrr_records = [r for r in all_records if r["metadata"].get("reciprocal_rank", 1.0) == 0.0]
    zm_by_model: dict = defaultdict(lambda: {"faith": [], "ar": []})
    for r in zero_mrr_records:
        zm_by_model[r["model"]]["faith"].append(r["faithfulness"])
        zm_by_model[r["model"]]["ar"].append(r["answer_relevancy"])

    summary = {
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "total_records": len(all_records),
        "models": MODELS,
        "judge_model": NOVA_PRO_MODEL_ID,
        "by_model": {
            m: {
                "faithfulness":     _mean(v["faith"]),
                "answer_relevancy": _mean(v["ar"]),
                "declined_to_answer": v["declined"],
                "n": v["n"],
            }
            for m, v in sorted(by_model.items())
        },
        "by_model_and_file_type": {},
        "zero_mrr_questions": {
            "count": len(zero_mrr_records) // max(len(MODELS), 1),
            "by_model": {
                m: {
                    "faithfulness":     _mean(v["faith"]),
                    "answer_relevancy": _mean(v["ar"]),
                    "n": len(v["faith"]),
                }
                for m, v in sorted(zm_by_model.items())
            },
        },
    }

    # by_model_and_file_type
    cross: dict = defaultdict(lambda: {"faith": [], "ar": []})
    for r in all_records:
        key = (r["model"], r["metadata"]["file_type"])
        cross[key]["faith"].append(r["faithfulness"])
        cross[key]["ar"].append(r["answer_relevancy"])
    for (model, ft), vals in cross.items():
        if model not in summary["by_model_and_file_type"]:
            summary["by_model_and_file_type"][model] = {}
        summary["by_model_and_file_type"][model][ft] = {
            "faithfulness":     _mean(vals["faith"]),
            "answer_relevancy": _mean(vals["ar"]),
            "n": len(vals["faith"]),
        }

    # expected_decline_metrics
    off_topic = [r for r in all_records if r.get("expected_decline")]
    if off_topic:
        ed_by_model: dict = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in off_topic:
            ed_by_model[r["model"]]["n"] += 1
            if r.get("declined_to_answer"):
                ed_by_model[r["model"]]["correct"] += 1
        summary["expected_decline_metrics"] = {
            "by_model": {
                m: {
                    "expected_decline_n": v["n"],
                    "correctly_declined": v["correct"],
                    "failed_to_decline":  v["n"] - v["correct"],
                    "true_positive_rate": v["correct"] / v["n"] if v["n"] else 0.0,
                }
                for m, v in sorted(ed_by_model.items())
            }
        }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(__file__).parent / f"generation_summary_{ts}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_file}")
    return summary_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generation + RAGAS evaluation")
    parser.add_argument("--input",  default=str(RETRIEVAL_RESULTS_FILE),
                        help="Input retrieval results JSONL (default: retrieval_results.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: <stem minus _retrieval_results>_generation_results.jsonl)")
    args = parser.parse_args()

    input_file  = Path(args.input)
    base_stem   = re.sub(r"_retrieval_results$", "", input_file.stem)
    output_file = Path(args.output) if args.output else \
                  input_file.parent / f"{base_stem}_generation_results.jsonl"

    print("Loading retrieval results...")
    with open(input_file, encoding="utf-8") as f:
        retrieval_records = [json.loads(line) for line in f if line.strip()]
    n = len(retrieval_records)
    print(f"  {n} records loaded.")

    print("\nInitialising AWS clients...")
    session, bedrock_rt = build_aws_clients()

    print("Initialising RAGAS LLM and embeddings...")
    ragas_llm, ragas_embeddings = build_ragas_llm_and_embeddings(session)

    # Load existing output to support resume on crash
    completed: set[tuple[str, str]] = set()  # (model_id, question)
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                completed.add((r["model"], r["question"]))
        if completed:
            print(f"\nResuming: {len(completed)} records already written.")

    all_records: list[dict] = []
    # Re-load existing records for summary at the end
    if output_file.exists() and completed:
        with open(output_file, encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

    with open(output_file, "a", encoding="utf-8") as out_f:
        for model_id in MODELS:
            print(f"\n{'='*80}")
            print(f"Model: {model_id}")
            print(f"{'='*80}")

            # Step 1: Generate answers
            print(f"\nStep 1/2 — Generating answers ({n} questions)...")
            model_answers: list[str | None] = []
            for i, rec in enumerate(retrieval_records, 1):
                q = rec["question"]
                if (model_id, q) in completed:
                    print(f"  [{i:2}/{n}] (cached) {q[:70]}")
                    model_answers.append(None)  # placeholder; won't be re-evaluated
                    continue
                print(f"  [{i:2}/{n}] {q[:70]}")
                answer = generate_answer(model_id, q, rec["retrieved_contexts"], bedrock_rt)
                model_answers.append(answer)

            # Step 2: RAGAS eval — only for newly generated answers
            print(f"\nStep 2/2 — RAGAS faithfulness + answer_relevancy ({n} questions)...")
            eval_idx = 0
            new_records: list[dict] = []
            for i, (rec, answer) in enumerate(zip(retrieval_records, model_answers), 1):
                q = rec["question"]
                if (model_id, q) in completed:
                    continue  # already in output file

                eval_idx += 1
                ragas_scores = compute_ragas_generation(
                    q, answer or "", rec["retrieved_contexts"],
                    ragas_llm, ragas_embeddings,
                    eval_idx,
                    sum(1 for a in model_answers if a is not None),
                    model_id,
                )

                out_rec = {
                    "model":              model_id,
                    "question":           q,
                    "answer":             answer,
                    "retrieved_contexts": rec["retrieved_contexts"],
                    "faithfulness":       ragas_scores["faithfulness"],
                    "answer_relevancy":   ragas_scores["answer_relevancy"],
                    "declined_to_answer": _is_declined(answer),
                    "expected_decline":   rec["metadata"].get("question_type") == "off_topic",
                    "metadata": {
                        "file_type":        rec["metadata"]["file_type"],
                        "question_type":    rec["metadata"]["question_type"],
                        "source":           rec["metadata"].get("source", ""),
                        "reciprocal_rank":  rec["reciprocal_rank"],
                        "context_precision": rec["context_precision"],
                        "context_recall":   rec["context_recall"],
                    },
                }
                new_records.append(out_rec)
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                out_f.flush()

            all_records.extend(new_records)
            print(f"\n  {len(new_records)} new records written for {model_id}")

    print(f"\nWrote {len(all_records)} total records to {output_file}")
    print_summary(all_records)
    write_summary_json(all_records)


if __name__ == "__main__":
    main()
