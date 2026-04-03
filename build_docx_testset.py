"""
Build a testset.jsonl-compatible file from the 16 questions in
'CRDCDH Chatbot Questions.docx'.

For each question:
  1. Retrieve top-8 chunks from the Bedrock KB (same as retrieve.py)
  2. Ask Claude Opus (via Bedrock) to generate a ground-truth answer from those chunks
  3. Write { question, ground_truth, contexts, metadata } to docx_testset.jsonl

Usage:  uv run python build_docx_testset.py
Output: docx_testset.jsonl
"""

import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_FILE = Path(__file__).parent / "testsets" / "docx_testset.jsonl"

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
KB_ID       = os.getenv("KB_ID", "")

CLAUDE_MODEL_ID   = "us.anthropic.claude-opus-4-6-v1"
NUMBER_OF_RESULTS = 8

SYSTEM_PROMPT = (
    "You are a helpful assistant for the NCI Cancer Research Data Commons (CRDC). "
    "Using ONLY the provided context chunks, write a comprehensive and accurate answer "
    "to the question. Your answer will serve as the ground-truth reference for an "
    "evaluation dataset, so be thorough and precise. "
    "If the context does not contain enough information to fully answer the question, "
    "answer as completely as possible from what is available and note any gaps."
)

# The 16 questions from CRDCDH Chatbot Questions.docx
DOCX_QUESTIONS = [
    {
        "question": (
            "I am a first-time data submitter. Please provide detailed, step-by-step "
            "instructions for the full Data Submission workflow, including all prerequisites."
        ),
        "question_type": "procedural",
    },
    {
        "question": (
            "How do I submit a Submission Request for a new study and obtain CRDC approval? "
            "Please provide step-by-step instructions."
        ),
        "question_type": "procedural",
    },
    {
        "question": (
            "My Submission Request for a study was marked as \"Conditionally Approved.\" "
            "What does this mean? Can I begin data submission?"
        ),
        "question_type": "conceptual",
    },
    {
        "question": (
            "My Submission Request was conditionally approved due to a data model change. "
            "What does this mean, and can I proceed with data submission?"
        ),
        "question_type": "conceptual",
    },
    {
        "question": (
            "As a data submitter, can I transfer raw data files from my own S3 bucket "
            "directly into the CRDC Data Hub bucket?"
        ),
        "question_type": "factual",
    },
    {
        "question": (
            "Are file IDs in the metadata Submission Templates case-sensitive during validation?"
        ),
        "question_type": "factual",
    },
    {
        "question": "How can I link a record to multiple parent records of the same node type?",
        "question_type": "procedural",
    },
    {
        "question": (
            "Which data model version is automatically used for a new data submission?"
        ),
        "question_type": "factual",
    },
    {
        "question": (
            "My new GC data submission was automatically assigned the latest data model version. "
            "How can I use a specific model version instead?"
        ),
        "question_type": "procedural",
    },
    {
        "question": (
            "In the GC data model, what are the permissible values for the \"sex\" property?"
        ),
        "question_type": "factual",
    },
    {
        "question": (
            "In the GC data model, what relationships do the Sample nodes have with other nodes?"
        ),
        "question_type": "factual",
    },
    {
        "question": (
            "I received a \"Value not permitted\" error during validation. "
            "What does this mean, and how do I fix it?"
        ),
        "question_type": "troubleshooting",
    },
    {
        "question": (
            "I see multiple \"Updating existing data\" warnings in the Validation Results. "
            "What does this indicate, and what actions should I take?"
        ),
        "question_type": "troubleshooting",
    },
    {
        "question": (
            "I received a \"Data file not found\" error in the Validation Results. "
            "How can I resolve this issue?"
        ),
        "question_type": "troubleshooting",
    },
    {
        "question": (
            "I received a \"Data file size mismatch\" error during validation. "
            "What causes this, and how do I fix it?"
        ),
        "question_type": "troubleshooting",
    },
    {
        "question": (
            "How can I perform a data submission using the CRDC Submission Portal APIs? "
            "Please provide high-level steps and required endpoints."
        ),
        "question_type": "procedural",
    },
]


# ---------------------------------------------------------------------------
# AWS Setup
# ---------------------------------------------------------------------------

def build_aws_clients():
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        session = boto3.Session(region_name=AWS_REGION)
    else:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)

    bedrock_agent_rt = session.client("bedrock-agent-runtime", region_name=AWS_REGION)
    bedrock_rt       = session.client("bedrock-runtime",       region_name=AWS_REGION)
    return bedrock_agent_rt, bedrock_rt


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(question: str, agent_rt_client) -> list[str]:
    try:
        resp = agent_rt_client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": NUMBER_OF_RESULTS}
            },
        )
        return [r["content"]["text"] for r in resp.get("retrievalResults", [])]
    except Exception as exc:
        print(f"  [WARN] Retrieval failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Ground-truth generation
# ---------------------------------------------------------------------------

def generate_ground_truth(question: str, chunks: list[str], bedrock_rt) -> str:
    if not chunks:
        return "Insufficient context retrieved from the knowledge base to answer this question."

    context_block = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks, 1))
    user_msg = f"Context:\n{context_block}\n\nQuestion: {question}"

    try:
        resp = bedrock_rt.converse(
            modelId=CLAUDE_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 4096},
        )
        return resp["output"]["message"]["content"][0]["text"]
    except Exception as exc:
        print(f"  [WARN] Generation failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not KB_ID:
        raise ValueError("KB_ID not set in environment / .env file")

    print(f"Building docx testset — {len(DOCX_QUESTIONS)} questions")
    print(f"KB: {KB_ID}  |  Region: {AWS_REGION}  |  Model: {CLAUDE_MODEL_ID}\n")

    agent_rt, bedrock_rt = build_aws_clients()

    # Load existing output to support resume on crash
    completed: set[str] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                completed.add(r["question"])
        if completed:
            print(f"Resuming: {len(completed)} questions already done.\n")

    n = len(DOCX_QUESTIONS)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(DOCX_QUESTIONS, 1):
            q = item["question"]
            if q in completed:
                print(f"[{i:2}/{n}] (cached) {q[:70]}")
                continue

            print(f"[{i:2}/{n}] {q[:80]}")

            # Step 1: retrieve
            chunks = retrieve_chunks(q, agent_rt)
            print(f"       → {len(chunks)} chunks retrieved")

            # Step 2: generate ground truth
            ground_truth = generate_ground_truth(q, chunks, bedrock_rt)
            print(f"       → ground truth: {ground_truth[:80]}...")

            record = {
                "question":    q,
                "ground_truth": ground_truth,
                "contexts":    chunks,
                "metadata": {
                    "source":        "docx",
                    "file_type":     "pdf",
                    "question_type": item["question_type"],
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone. Wrote {n} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
