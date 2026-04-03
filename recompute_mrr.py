"""
Recompute MRR from existing retrieval_results.jsonl at a given threshold.
No KB calls needed — only embeds ground_truth + retrieved_contexts.
Updates retrieval_results.jsonl and writes a new retrieval_summary_<ts>.json.

Usage:  uv run python recompute_mrr.py [--threshold 0.65]
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings

load_dotenv()

RESULTS_FILE         = Path(__file__).parent / "results" / "retrieval_results.jsonl"
AWS_REGION           = os.getenv("AWS_REGION", "us-east-1")
TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"


def _cosine(a, b):
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    threshold = args.threshold

    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"]     = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    os.environ["AWS_REGION_NAME"]       = AWS_REGION
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)

    embed_client = session.client("bedrock-runtime")
    lc_embeddings = BedrockEmbeddings(
        model_id=TITAN_EMBED_MODEL_ID, region_name=AWS_REGION, client=embed_client
    )

    records = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Recomputing MRR@{threshold} for {len(records)} records...")

    updated = []
    for i, r in enumerate(records, 1):
        gt = r["ground_truth"]
        chunks = r["retrieved_contexts"]
        if not chunks:
            mrr = 0.0
        else:
            texts = [gt] + chunks
            vecs = lc_embeddings.embed_documents(texts)
            gt_vec = vecs[0]
            mrr = 0.0
            for rank, cv in enumerate(vecs[1:], 1):
                if _cosine(gt_vec, cv) >= threshold:
                    mrr = 1.0 / rank
                    break
        r2 = dict(r)
        r2["reciprocal_rank"] = mrr
        updated.append(r2)
        ft = r["metadata"]["file_type"]
        print(f"  [{i:2}/{len(records)}] mrr={mrr:.4f}  ({ft})")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for rec in updated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nUpdated {RESULTS_FILE}")

    # Summary
    def group_stats(records, key):
        buckets = defaultdict(lambda: {"cp": [], "cr": [], "mrr": []})
        for r in records:
            k = r["metadata"][key]
            buckets[k]["cp"].append(r["context_precision"])
            buckets[k]["cr"].append(r["context_recall"])
            buckets[k]["mrr"].append(r["reciprocal_rank"])
        return {
            k: {
                "context_precision": _mean(v["cp"]),
                "context_recall":    _mean(v["cr"]),
                "mrr":               _mean(v["mrr"]),
                "n":                 len(v["cp"]),
            }
            for k, v in sorted(buckets.items())
        }

    all_cp  = [r["context_precision"] for r in updated]
    all_cr  = [r["context_recall"]    for r in updated]
    all_mrr = [r["reciprocal_rank"]   for r in updated]
    zeros   = sum(1 for v in all_mrr if v == 0.0)

    summary = {
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "total_records": len(updated),
        "overall": {
            "context_precision": _mean(all_cp),
            "context_recall":    _mean(all_cr),
            "mrr":               _mean(all_mrr),
        },
        "by_file_type":     group_stats(updated, "file_type"),
        "by_question_type": group_stats(updated, "question_type"),
        "score_distributions": {
            "context_precision": {"zeros": sum(1 for v in all_cp  if v == 0.0), "ones": sum(1 for v in all_cp  if v == 1.0)},
            "context_recall":    {"zeros": sum(1 for v in all_cr  if v == 0.0), "ones": sum(1 for v in all_cr  if v == 1.0)},
            "mrr":               {"zeros": zeros, "ones": sum(1 for v in all_mrr if v == 1.0)},
        },
        "config": {"mrr_sim_threshold": threshold, "number_of_results": 8},
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(__file__).parent / "results" / f"retrieval_summary_{ts}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_file}")

    print(f"\nMRR@{threshold} summary:")
    print(f"  Overall: cp={_mean(all_cp):.4f}  cr={_mean(all_cr):.4f}  mrr={_mean(all_mrr):.4f}")
    print(f"  MRR zeros: {zeros}/{len(updated)}")
    print(f"\n  By file type:")
    for k, v in group_stats(updated, "file_type").items():
        print(f"    {k:<6} mrr={v['mrr']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
