"""
Phase 2a: Retrieval Evaluation (model-agnostic)
Queries Bedrock KB for each question in testset.jsonl, computes
context_precision, context_recall (via RAGAS) and MRR (cosine similarity).
Writes retrieval_results.jsonl.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import boto3
import instructor
import litellm
import numpy as np
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import context_precision, context_recall
from ragas import evaluate
from datasets import Dataset

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TESTSET_FILE      = Path(__file__).parent / "testsets" / "testset.jsonl"
OUTPUT_FILE       = Path(__file__).parent / "results" / "retrieval_results.jsonl"
CHECKPOINT_FILE   = Path(__file__).parent / "results" / "retrieval_checkpoint.jsonl"

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
KB_ID       = os.getenv("KB_ID", "")

NOVA_PRO_MODEL_ID   = "us.amazon.nova-pro-v1:0"
TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

NUMBER_OF_RESULTS   = 8
MRR_SIM_THRESHOLD   = 0.65


# ---------------------------------------------------------------------------
# AWS Setup
# ---------------------------------------------------------------------------

def build_aws_clients():
    # If shell already has credentials in env vars, use them directly (no profile needed).
    # Otherwise fall back to the named profile from .env.
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

    bedrock_agent_rt = session.client("bedrock-agent-runtime", region_name=AWS_REGION)
    bedrock_rt       = session.client("bedrock-runtime",       region_name=AWS_REGION)
    return session, bedrock_agent_rt, bedrock_rt


def build_ragas_llm_and_embeddings(session):
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
    return ragas_llm, ragas_embeddings, lc_embeddings


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(question: str, agent_rt_client, kb_id: str) -> list[str]:
    """Query Bedrock KB and return ordered list of text chunks (rank 1 first)."""
    try:
        resp = agent_rt_client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": NUMBER_OF_RESULTS}
            },
        )
        chunks = [
            r["content"]["text"]
            for r in resp.get("retrievalResults", [])
        ]
        return chunks
    except Exception as exc:
        print(f"  [WARN] Retrieval failed for question '{question[:60]}...': {exc}")
        return []


# ---------------------------------------------------------------------------
# MRR via cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def compute_mrr(ground_truth: str, retrieved_chunks: list[str], embeddings_client) -> float:
    """Embed ground_truth and each chunk, return reciprocal rank of first relevant chunk."""
    if not retrieved_chunks:
        return 0.0
    texts = [ground_truth] + retrieved_chunks
    try:
        vectors = embeddings_client.embed_documents(texts)
    except Exception as exc:
        print(f"  [WARN] Embedding failed for MRR: {exc}")
        return 0.0

    gt_vec     = vectors[0]
    chunk_vecs = vectors[1:]

    for rank, chunk_vec in enumerate(chunk_vecs, start=1):
        sim = _cosine(gt_vec, chunk_vec)
        if sim >= MRR_SIM_THRESHOLD:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# RAGAS context metrics — one question at a time for progress visibility
# ---------------------------------------------------------------------------

def compute_ragas_one(
    question: str,
    ground_truth: str,
    retrieved_chunks: list[str],
    ragas_llm,
    ragas_embeddings,
    idx: int,
    total: int,
) -> dict:
    """Run context_precision + context_recall for a single question. Returns zeros on error."""
    t0 = time.time()
    print(f"  [{idx:2}/{total}] RAGAS eval: {question[:70]}")
    if not retrieved_chunks:
        print(f"         → skipped (no retrieved chunks) — cp=0.0000, cr=0.0000")
        return {"context_precision": 0.0, "context_recall": 0.0}
    try:
        hf_dataset = Dataset.from_dict({
            "question":     [question],
            "ground_truth": [ground_truth],
            "contexts":     [retrieved_chunks],
        })
        result = evaluate(
            dataset=hf_dataset,
            metrics=[context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        df = result.to_pandas()
        row = df.iloc[0]
        cp = float(row.get("context_precision", 0.0) or 0.0)
        cr = float(row.get("context_recall",    0.0) or 0.0)
        elapsed = time.time() - t0
        print(f"         → cp={cp:.4f}, cr={cr:.4f}  ({elapsed:.1f}s)")
        return {"context_precision": cp, "context_recall": cr}
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"         → [WARN] RAGAS eval failed ({elapsed:.1f}s): {exc}")
        return {"context_precision": 0.0, "context_recall": 0.0}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for group_key in ("file_type", "question_type"):
        buckets: dict[str, list] = defaultdict(lambda: {"cp": [], "cr": [], "mrr": []})
        for r in records:
            key = r["metadata"][group_key]
            buckets[key]["cp"].append(r["context_precision"])
            buckets[key]["cr"].append(r["context_recall"])
            buckets[key]["mrr"].append(r["reciprocal_rank"])

        label = "File Type" if group_key == "file_type" else "Question Type"
        print(f"\n{'─'*70}")
        print(f"By {label}:")
        print(f"{'Group':<22} {'ctx_precision':>14} {'ctx_recall':>11} {'MRR':>8} {'n':>5}")
        print(f"{'─'*22} {'─'*14} {'─'*11} {'─'*8} {'─'*5}")
        for grp, vals in sorted(buckets.items()):
            n = len(vals["cp"])
            print(
                f"{grp:<22} {_mean(vals['cp']):>14.4f}"
                f" {_mean(vals['cr']):>11.4f}"
                f" {_mean(vals['mrr']):>8.4f}"
                f" {n:>5}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation against Bedrock KB")
    parser.add_argument("--input",  default=str(TESTSET_FILE),
                        help="Input JSONL testset (default: testset.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: <stem>_retrieval_results.jsonl)")
    args = parser.parse_args()

    input_file      = Path(args.input)
    output_file     = Path(args.output) if args.output else \
                      input_file.parent / f"{input_file.stem}_retrieval_results.jsonl"
    checkpoint_file = input_file.parent / f"{input_file.stem}_retrieval_checkpoint.jsonl"

    if not KB_ID:
        raise ValueError("KB_ID not set in environment / .env file")

    print("Loading testset...")
    with open(input_file, encoding="utf-8") as f:
        testset = [json.loads(line) for line in f if line.strip()]
    n = len(testset)
    print(f"  {n} questions loaded.")

    print("\nInitialising AWS clients...")
    session, agent_rt, _ = build_aws_clients()

    print("Initialising RAGAS LLM and embeddings...")
    ragas_llm, ragas_embeddings, lc_embeddings = build_ragas_llm_and_embeddings(session)

    # -----------------------------------------------------------------------
    # Step 1: Retrieve — with checkpoint so crashes don't lose progress
    # -----------------------------------------------------------------------
    checkpoint: dict[str, list[str]] = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                checkpoint[entry["question"]] = entry["chunks"]
        print(f"\nCheckpoint found: {len(checkpoint)} questions already retrieved.")

    retrieved_all: list[list[str]] = []
    print(f"\nStep 1/3 — Retrieving from KB '{KB_ID}' ({NUMBER_OF_RESULTS} chunks/question)...")
    with open(checkpoint_file, "a", encoding="utf-8") as ckpt_f:
        for i, record in enumerate(testset, 1):
            q = record["question"]
            if q in checkpoint:
                chunks = checkpoint[q]
                print(f"  [{i:2}/{n}] (cached) {q[:70]}  → {len(chunks)} chunks")
            else:
                print(f"  [{i:2}/{n}] {q[:70]}")
                chunks = retrieve_chunks(q, agent_rt, KB_ID)
                print(f"         → {len(chunks)} chunks retrieved")
                ckpt_f.write(json.dumps({"question": q, "chunks": chunks}, ensure_ascii=False) + "\n")
                ckpt_f.flush()
            retrieved_all.append(chunks)

    # -----------------------------------------------------------------------
    # Step 2: RAGAS context metrics — one question at a time
    # -----------------------------------------------------------------------
    print(f"\nStep 2/3 — RAGAS context_precision + context_recall ({n} questions, ~{n*6//60}-{n*10//60} min)...")
    ragas_metrics: list[dict] = []
    for i, (record, chunks) in enumerate(zip(testset, retrieved_all), 1):
        rm = compute_ragas_one(
            record["question"], record["ground_truth"], chunks,
            ragas_llm, ragas_embeddings, i, n,
        )
        ragas_metrics.append(rm)

    # -----------------------------------------------------------------------
    # Step 3: MRR per question
    # -----------------------------------------------------------------------
    print(f"\nStep 3/3 — MRR via cosine similarity (threshold={MRR_SIM_THRESHOLD})...")
    mrr_scores: list[float] = []
    for i, (record, chunks) in enumerate(zip(testset, retrieved_all), 1):
        mrr = compute_mrr(record["ground_truth"], chunks, lc_embeddings)
        mrr_scores.append(mrr)
        print(f"  [{i:2}/{n}] MRR={mrr:.4f}  ({len(chunks)} chunks)")

    # -----------------------------------------------------------------------
    # Assemble and write output
    # -----------------------------------------------------------------------
    output_records: list[dict] = []
    for record, chunks, rm, mrr in zip(testset, retrieved_all, ragas_metrics, mrr_scores):
        out = {
            "question":           record["question"],
            "ground_truth":       record["ground_truth"],
            "contexts":           record.get("contexts", []),
            "retrieved_contexts": chunks,
            "context_precision":  rm["context_precision"],
            "context_recall":     rm["context_recall"],
            "reciprocal_rank":    mrr,
            "metadata":           record["metadata"],
        }
        output_records.append(out)

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(output_records)} records to {output_file}")

    # Clean up checkpoint now that we have final output
    checkpoint_file.unlink(missing_ok=True)

    print_summary(output_records)


if __name__ == "__main__":
    main()
