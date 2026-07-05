"""
AWS Bedrock Knowledge Base Evaluation Job — BYOIR (Bring Your Own Inference Results)

Steps:
  1. convert  — convert retrieval JSONL + testset JSONL → AWS eval JSONL and upload to S3
  2. submit   — create a Bedrock evaluation job against the uploaded dataset
  3. status   — poll the job status
  4. results  — download and print the results from S3

Usage:
  uv run python aws_bedrock_eval.py convert  --retrieval-file results/retrieval_xsyf_prose_k8_v3.jsonl
  uv run python aws_bedrock_eval.py submit   --s3-input s3://bucket/prefix/eval_input.jsonl --job-name my-eval-job
  uv run python aws_bedrock_eval.py status   --job-arn arn:aws:bedrock:...
  uv run python aws_bedrock_eval.py results  --job-arn arn:aws:bedrock:...

Prerequisites:
  - AWS_PROFILE / AWS_REGION set in .env or environment
  - S3_BUCKET set in .env
  - BEDROCK_EVAL_ROLE_ARN set in .env  (IAM role Bedrock can assume)
  - Role needs: s3:GetObject on input bucket, s3:PutObject on output bucket,
                bedrock:InvokeModel on the evaluator model
"""

import argparse
import json
import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ["S3_BUCKET"]
BEDROCK_EVAL_ROLE_ARN = os.environ["BEDROCK_EVAL_ROLE_ARN"]

# S3 prefix where the converted input file is uploaded
S3_INPUT_PREFIX = os.getenv("BEDROCK_EVAL_S3_INPUT_PREFIX", "bedrock-eval/input")
# S3 prefix where Bedrock writes results
S3_OUTPUT_PREFIX = os.getenv("BEDROCK_EVAL_S3_OUTPUT_PREFIX", "bedrock-eval/output")

TESTSET_FILE = Path(__file__).parent / "testsets" / "crdc_chatbot_testset_v2.jsonl"

# Evaluator model — Nova Pro used as judge (same as rest of the pipeline)
EVALUATOR_MODEL_ID = os.getenv(
    "BEDROCK_EVAL_MODEL_ID", "us.amazon.nova-pro-v1:0"
)

# RAG source identifier embedded in each BYOIR record (free-form label)
RAG_SOURCE_IDENTIFIER = "crdc-chatbot-kb"


def get_session() -> boto3.Session:
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")

    if aws_access_key and aws_secret_key:
        return boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=AWS_REGION,
        )
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Step 1: Convert
# ---------------------------------------------------------------------------

def convert(retrieval_file: Path, output_file: Path) -> Path:
    """
    Convert retrieval JSONL → AWS Bedrock BYOIR retrieve-only eval JSONL.

    AWS record format per line:
    {
      "conversationTurns": [{
        "prompt": { "content": [{ "text": "<question>" }] },
        "referenceResponses": [{ "content": [{ "text": "<ground_truth>" }] }],
        "output": {
          "knowledgeBaseIdentifier": "<RAG_SOURCE_IDENTIFIER>",
          "retrievedResults": {
            "retrievalResults": [
              { "content": { "text": "<context chunk>" } },
              ...
            ]
          }
        }
      }]
    }
    """
    testset = {r["question"]: r for r in load_jsonl(TESTSET_FILE)}
    retrieval_records = load_jsonl(retrieval_file)

    converted = []
    skipped = 0
    for r in retrieval_records:
        question = r["question"]
        ground_truth = r.get("ground_truth") or testset.get(question, {}).get("ground_truth", "")
        contexts = r.get("retrieved_contexts", [])

        if not contexts:
            skipped += 1
            continue

        record = {
            "conversationTurns": [
                {
                    "prompt": {
                        "content": [{"text": question}]
                    },
                    "referenceResponses": [
                        {
                            "content": [{"text": ground_truth}]
                        }
                    ],
                    "output": {
                        "knowledgeBaseIdentifier": RAG_SOURCE_IDENTIFIER,
                        "retrievedResults": {
                            "retrievalResults": [
                                {"content": {"text": ctx}} for ctx in contexts
                            ]
                        }
                    }
                }
            ]
        }
        converted.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec) + "\n")

    print(f"Converted {len(converted)} records  ({skipped} skipped — no retrieved contexts)")
    print(f"Saved → {output_file}")
    return output_file


def upload_to_s3(local_file: Path, s3_key: str) -> str:
    session = get_session()
    s3 = session.client("s3")
    s3.upload_file(str(local_file), S3_BUCKET, s3_key)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"Uploaded → {s3_uri}")
    return s3_uri


# ---------------------------------------------------------------------------
# Step 2: Submit
# ---------------------------------------------------------------------------

def submit(s3_input_uri: str, job_name: str, job_description: str) -> str:
    """
    Create a Bedrock BYOIR retrieve-only evaluation job.
    Returns the job ARN.
    """
    session = get_session()
    bedrock = session.client("bedrock")

    s3_output_uri = f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/"

    request = {
        "jobName": job_name,
        "jobDescription": job_description,
        "roleArn": BEDROCK_EVAL_ROLE_ARN,
        "applicationType": "RagEvaluation",
        "evaluationConfig": {
            "automated": {
                "datasetMetricConfigs": [
                    {
                        "taskType": "QuestionAndAnswer",
                        "dataset": {
                            "name": job_name,
                            "datasetLocation": {
                                "s3Uri": s3_input_uri
                            }
                        },
                        "metricNames": [
                            "Builtin.ContextRelevance",
                            "Builtin.ContextCoverage",
                        ]
                    }
                ],
                "evaluatorModelConfig": {
                    "bedrockEvaluatorModels": [
                        {"modelIdentifier": EVALUATOR_MODEL_ID}
                    ]
                }
            }
        },
        "inferenceConfig": {
            "ragConfigs": [
                {
                    "precomputedRagSourceConfig": {
                        "retrieveSourceConfig": {
                            "ragSourceIdentifier": RAG_SOURCE_IDENTIFIER
                        }
                    }
                }
            ]
        },
        "outputDataConfig": {
            "s3Uri": s3_output_uri
        }
    }

    response = bedrock.create_evaluation_job(**request)
    job_arn = response["jobArn"]
    print(f"Submitted evaluation job")
    print(f"  Job ARN: {job_arn}")
    print(f"  Results will appear at: {s3_output_uri}")
    return job_arn


# ---------------------------------------------------------------------------
# Step 3: Status
# ---------------------------------------------------------------------------

def status(job_arn: str, poll: bool = False) -> str:
    session = get_session()
    bedrock = session.client("bedrock")

    while True:
        response = bedrock.get_evaluation_job(jobIdentifier=job_arn)
        job_status = response["status"]
        print(f"Status: {job_status}")

        if not poll or job_status in ("Completed", "Failed", "Stopped"):
            return job_status

        print("  Waiting 60s...")
        time.sleep(60)


# ---------------------------------------------------------------------------
# Step 4: Results
# ---------------------------------------------------------------------------

def results(job_arn: str) -> None:
    session = get_session()
    bedrock = session.client("bedrock")
    s3 = session.client("s3")

    response = bedrock.get_evaluation_job(jobIdentifier=job_arn)
    job_status = response["status"]

    if job_status != "Completed":
        print(f"Job is not complete yet (status: {job_status}). Run 'status' first.")
        return

    output_uri = response.get("outputDataConfig", {}).get("s3Uri", "")
    if not output_uri.startswith("s3://"):
        print(f"Cannot determine output location from job response.")
        return

    # Strip s3://bucket/ prefix to get the key prefix
    without_scheme = output_uri[len("s3://"):]
    bucket, prefix = without_scheme.split("/", 1)
    prefix = prefix.rstrip("/")

    paginator = s3.get_paginator("list_objects_v2")
    result_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") or key.endswith(".jsonl"):
                result_files.append(key)

    if not result_files:
        print(f"No result files found under {output_uri}")
        return

    print(f"\nFound {len(result_files)} result file(s):\n")
    for key in result_files:
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        print(f"=== {key} ===")
        # Pretty-print if JSON, otherwise raw
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            for line in content.splitlines():
                try:
                    print(json.dumps(json.loads(line), indent=2))
                except Exception:
                    print(line)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AWS Bedrock KB BYOIR evaluation job helper"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # convert
    p_convert = sub.add_parser("convert", help="Convert retrieval JSONL to AWS eval format and upload to S3")
    p_convert.add_argument(
        "--retrieval-file", required=True,
        help="Path to retrieval results JSONL (e.g. results/retrieval_xsyf_prose_k8_v3.jsonl)"
    )
    p_convert.add_argument(
        "--output-file", default=None,
        help="Where to save the converted JSONL (default: testsets/<stem>_aws_eval.jsonl)"
    )
    p_convert.add_argument(
        "--no-upload", action="store_true",
        help="Convert only; skip S3 upload"
    )

    # submit
    p_submit = sub.add_parser("submit", help="Submit evaluation job to Bedrock")
    p_submit.add_argument("--s3-input", required=True, help="S3 URI of the converted eval JSONL")
    p_submit.add_argument("--job-name", required=True, help="Job name (lowercase, max 63 chars)")
    p_submit.add_argument("--job-description", default="CRDC chatbot KB retrieval evaluation")

    # status
    p_status = sub.add_parser("status", help="Check evaluation job status")
    p_status.add_argument("--job-arn", required=True)
    p_status.add_argument("--poll", action="store_true", help="Poll every 60s until complete")

    # results
    p_results = sub.add_parser("results", help="Download and print evaluation results")
    p_results.add_argument("--job-arn", required=True)

    args = parser.parse_args()

    if args.command == "convert":
        retrieval_path = Path(args.retrieval_file)
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            output_path = Path("testsets") / f"{retrieval_path.stem}_aws_eval.jsonl"

        converted_file = convert(retrieval_path, output_path)

        if not args.no_upload:
            s3_key = f"{S3_INPUT_PREFIX}/{converted_file.name}"
            s3_uri = upload_to_s3(converted_file, s3_key)
            print(f"\nNext step — submit the job:\n")
            print(f"  uv run python aws_bedrock_eval.py submit \\")
            print(f"    --s3-input {s3_uri} \\")
            print(f"    --job-name crdc-kb-eval-$(date +%Y%m%d)")

    elif args.command == "submit":
        job_arn = submit(args.s3_input, args.job_name, args.job_description)
        print(f"\nNext step — check status:\n")
        print(f"  uv run python aws_bedrock_eval.py status --job-arn {job_arn} --poll")

    elif args.command == "status":
        status(args.job_arn, poll=args.poll)

    elif args.command == "results":
        results(args.job_arn)


if __name__ == "__main__":
    main()
