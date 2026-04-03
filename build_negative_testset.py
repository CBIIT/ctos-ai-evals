"""
Build a testset.jsonl-compatible file with negative (out-of-scope) questions.

Two categories:
  1. Off-topic  — completely unrelated to CRDC; no KB call, no LLM call.
                  Ground truth = static refusal string.
  2. Mixed      — valid CRDC topic embedded with nonsense.
                  Retrieves chunks from Bedrock KB, then asks Claude Opus to
                  answer the valid portion and flag the nonsensical portion.

Usage:  uv run python build_negative_testset.py
Output: testsets/negative_testset.jsonl
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

OUTPUT_FILE = Path(__file__).parent / "testsets" / "negative_testset.jsonl"

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
KB_ID       = os.getenv("KB_ID", "")

CLAUDE_MODEL_ID   = "us.anthropic.claude-opus-4-6-v1"
NUMBER_OF_RESULTS = 8

OFF_TOPIC_GROUND_TRUTH = (
    "This question is outside the scope of the CRDC Data Hub. "
    "Please contact the CRDC helpdesk for assistance."
)

MIXED_SYSTEM_PROMPT = (
    "You are a helpful assistant for the NCI Cancer Research Data Commons (CRDC). "
    "The question below contains two parts: one relevant to CRDC data submission, "
    "and one that is nonsensical, impossible, or completely unrelated to CRDC. "
    "Your response will serve as the ground-truth reference for an evaluation dataset.\n\n"
    "Instructions:\n"
    "1. Using ONLY the provided context chunks, answer the valid CRDC-related portion "
    "of the question as completely and accurately as possible.\n"
    "2. For the nonsensical or irrelevant portion, explicitly state that this part of "
    "the question is outside the scope of the CRDC Data Hub and cannot be answered.\n"
    "3. Keep the two portions clearly separated in your response.\n"
    "4. Do not attempt to answer the irrelevant portion — acknowledge it and redirect "
    "the user to the CRDC helpdesk for anything outside CRDC data submission scope."
)

# ---------------------------------------------------------------------------
# Question lists
# ---------------------------------------------------------------------------

# Category 1: completely off-topic — no KB retrieval, no LLM generation
OFF_TOPIC_QUESTIONS = [
    "What is your opinion on the theory of evolution?",
    "What is the meaning of life?",
    "Which is the best form of government?",
    "Should I vote Democrat or Republican in the next election?",
    "How do I bake a sourdough bread from scratch?",
    "What is a good recipe for chocolate chip cookies?",
    "I am feeling sad and lonely. What can I do to feel happier?",
    "Someone at school is bullying me. How do I stop them?",
    "Who is buried in Grant's Tomb?",
    "Who won the most recent FIFA World Cup?",
    "What is the best training regimen for running a marathon?",
    "What is the recommended daily dose of ibuprofen for back pain?",
    "I have a headache every day. Do I have a brain tumor?",
    "How do I factory-reset an iPhone?",
    "What is the capital of Madagascar?",
]

# Category 2: valid CRDC topic mixed with nonsense — retrieve + LLM ground truth
MIXED_QUESTIONS = [
    "How do I submit a can of peas to the TCGA project?",
    "What are the permissible values for the 'flavor' property in the GC data model?",
    "I received a 'Data file not found' error. Also, how do I submit data on behalf of my pet cat?",
    "How do I get my Submission Request approved, and what is Taylor Swift's phone number?",
    "Can I transfer data files from my S3 bucket by mailing a USB drive to the CRDC office?",
    "How do I link a sample record to multiple parent nodes, and can I also link it to my LinkedIn profile?",
    "Which version of the CRDC data model is compatible with my toaster oven?",
    "How do I use the CRDC Submission Portal API to also order pizza for my team?",
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
# Ground-truth generation (mixed questions only)
# ---------------------------------------------------------------------------

def generate_ground_truth_mixed(question: str, chunks: list[str], bedrock_rt) -> str:
    if not chunks:
        return (
            "The CRDC-related portion of this question could not be answered due to "
            "insufficient context. The other portion is outside the scope of the CRDC "
            "Data Hub. Please contact the CRDC helpdesk for assistance."
        )

    context_block = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks, 1))
    user_msg = f"Context:\n{context_block}\n\nQuestion: {question}"

    try:
        resp = bedrock_rt.converse(
            modelId=CLAUDE_MODEL_ID,
            system=[{"text": MIXED_SYSTEM_PROMPT}],
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

    n_off  = len(OFF_TOPIC_QUESTIONS)
    n_mix  = len(MIXED_QUESTIONS)
    n_total = n_off + n_mix
    print(f"Building negative testset — {n_off} off-topic + {n_mix} mixed = {n_total} total")
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

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:

        # --- Category 1: off-topic (no AWS calls) ---
        print(f"--- Category 1: off-topic ({n_off} questions) ---")
        for i, q in enumerate(OFF_TOPIC_QUESTIONS, 1):
            if q in completed:
                print(f"[{i:2}/{n_off}] (cached) {q[:70]}")
                continue

            print(f"[{i:2}/{n_off}] {q[:80]}")
            record = {
                "question":     q,
                "ground_truth": OFF_TOPIC_GROUND_TRUTH,
                "contexts":     [],
                "metadata": {
                    "source":        "negative",
                    "file_type":     "none",
                    "question_type": "off_topic",
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

        # --- Category 2: mixed (retrieve + LLM) ---
        print(f"\n--- Category 2: mixed ({n_mix} questions) ---")
        for i, q in enumerate(MIXED_QUESTIONS, 1):
            if q in completed:
                print(f"[{i:2}/{n_mix}] (cached) {q[:70]}")
                continue

            print(f"[{i:2}/{n_mix}] {q[:80]}")

            chunks = retrieve_chunks(q, agent_rt)
            print(f"       → {len(chunks)} chunks retrieved")

            ground_truth = generate_ground_truth_mixed(q, chunks, bedrock_rt)
            print(f"       → ground truth: {ground_truth[:80]}...")

            record = {
                "question":     q,
                "ground_truth": ground_truth,
                "contexts":     chunks,
                "metadata": {
                    "source":        "negative",
                    "file_type":     "mixed",
                    "question_type": "mixed_relevance",
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone. Wrote to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
