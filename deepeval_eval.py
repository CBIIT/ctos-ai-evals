"""
DeepEval evaluation against the live CRDC chatbot API.

Hits /api/chat/question for each testset question to get the real chatbot answer,
retrieves contexts from Bedrock KB (with reranking, matching chatbot exactly),
then runs DeepEval metrics with Claude Sonnet 4.6 as judge.

Usage:
    uv run python deepeval_eval.py              # all 53 questions
    uv run python deepeval_eval.py --limit 3    # quick smoke test
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import boto3
import httpx
from dotenv import load_dotenv
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


from bedrock_judge import BedrockJudge

load_dotenv()

TESTSET_FILE = Path(__file__).parent / "testsets" / "testset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
CHATBOT_API_URL = os.getenv(
    "CHATBOT_API_URL", "https://hub-dev2.datacommons.cancer.gov/api/chat/question"
)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
KB_ID = os.getenv("KB_ID", "")
RERANK_MODEL_ARN = os.getenv(
    "RERANK_MODEL_ARN",
    "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
)

GUARDRAIL_PHRASES = ["I couldn't find an answer. Please contact support"]


def build_bedrock_client():
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        session = boto3.Session(region_name=AWS_REGION)
    else:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("bedrock-agent-runtime", region_name=AWS_REGION)


def retrieve_contexts(question: str, agent_rt) -> list[str]:
    """
    Fetch 20 chunks then rerank to 8 via Cohere — exactly matches question.ts:69-90.
    Using fewer chunks or no reranking would give different contexts than the chatbot used.
    """
    try:
        resp = agent_rt.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 20,
                    "rerankingConfiguration": {
                        "type": "BEDROCK_RERANKING_MODEL",
                        "bedrockRerankingConfiguration": {
                            "modelConfiguration": {"modelArn": RERANK_MODEL_ARN},
                            "numberOfRerankedResults": 8,
                        },
                    },
                }
            },
        )
        return [r["content"]["text"] for r in resp.get("retrievalResults", [])]
    except Exception as exc:
        print(f"  [WARN] Retrieval failed: {exc}")
        return []


def ask_chatbot(question: str) -> str:
    """POST to live chatbot API and collect streamed answer."""
    payload = {
        "question": question,
        "sessionId": None,
        "conversationHistory": [],
    }
    answer_parts = []
    try:
        with httpx.stream("POST", CHATBOT_API_URL, json=payload, timeout=60.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response":
                    answer_parts.append(event.get("output", ""))
    except Exception as exc:
        print(f"  [WARN] Chatbot call failed: {exc}")
    return "".join(answer_parts)


def is_guardrail_blocked(answer: str) -> bool:
    """Detect chatbot fallback responses triggered by Bedrock Guardrails."""
    return any(phrase in answer for phrase in GUARDRAIL_PHRASES)


def build_metrics(judge: BedrockJudge) -> list:
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
    )
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
    )
    procedural_completeness = GEval(
        name="ProceduralCompleteness",
        criteria=(
            "Determine whether the actual output completely answers the question. "
            "For procedural questions (how do I submit X, how do I get approved), "
            "check that all required steps are present. "
            "Penalize answers that omit steps — a partial procedure is worse than no answer "
            "because the user may think they are done when they are not."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=0.7,
    )
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
    )
    return [
        faithfulness,
        answer_relevancy,
        procedural_completeness,
        contextual_relevancy,
    ]


def main():
    parser = argparse.ArgumentParser(description="DeepEval against live chatbot API")
    parser.add_argument("--input", default=str(TESTSET_FILE))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to evaluate (default: all)",
    )
    args = parser.parse_args()

    if not KB_ID:
        raise ValueError("KB_ID not set in .env")

    with open(args.input, encoding="utf-8") as f:
        testset = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        testset = testset[: args.limit]
    print(f"Loaded {len(testset)} questions.")

    print("Initialising AWS clients...")
    agent_rt = build_bedrock_client()

    print(f"Initialising DeepEval judge...")
    judge = BedrockJudge()
    metrics = build_metrics(judge)

    records = []
    for i, rec in enumerate(testset, 1):
        q = rec["question"]
        print(f"\n[{i}/{len(testset)}] {q[:80]}")

        print("  → retrieving contexts (20→rerank→8)...")
        contexts = retrieve_contexts(q, agent_rt)
        print(f"     {len(contexts)} chunks")

        print("  → asking live chatbot...")
        answer = ask_chatbot(q)
        blocked = is_guardrail_blocked(answer)
        print(f"     guardrail_blocked={blocked}")
        # print(f"     answer: {answer[:100]}...")

        records.append(
            {
                "question": q,
                "answer": answer,
                "ground_truth": rec["ground_truth"],
                "contexts": contexts,
                "guardrail_blocked": blocked,
                "metadata": rec["metadata"],
            }
        )

    # Run DeepEval — skip guardrail-blocked answers (no meaningful answer to score)
    scoreable = [r for r in records if not r["guardrail_blocked"]]
    print(
        f"\n{len(scoreable)}/{len(records)} questions scoreable (rest blocked by guardrail)."
    )

    test_cases = [
        LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            expected_output=(
                None if r["metadata"].get("review_needed") else r["ground_truth"]
            ),
            retrieval_context=r["contexts"],
        )
        for r in scoreable
    ]

    print(f"Running DeepEval metrics...")
    for i, (tc, record) in enumerate(zip(test_cases, scoreable), 1):
        print(f"\n  [{i}/{len(test_cases)}] {tc.input[:60]}")
        record["scores"] = {}
        for metric in metrics:
            metric.measure(tc)
            name = metric.__class__.__name__
            print(f"     {name}: {metric.score:.4f} — {metric.reason}")
            record["scores"][name] = {
                "score": metric.score,
                "reason": metric.reason,
            }

    # Write results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"deepeval_results_{ts}.json"
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(records),
        "guardrail_blocked": sum(1 for r in records if r["guardrail_blocked"]),
        "scored": len(scoreable),
        "records": records,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
