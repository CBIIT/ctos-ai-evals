"""
create_baseline_kb.py
Creates a Phase 1 baseline Bedrock KB with default chunking (fixed-size, Bedrock defaults).
Single data source pointing at s3://<S3_BUCKET>/<s3-prefix>/.

Usage:
  python create_baseline_kb.py
  python create_baseline_kb.py --kb-name my-baseline --s3-prefix chunking-exp/baseline
"""

import argparse
import json
import os
import time

import boto3
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "")
REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "test-dsrc-04022026")

ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/power-user-bedrock-knowledgebase-execution-role"
PERMISSIONS_BOUNDARY_ARN = os.getenv("AWS_PERMISSIONS_BOUNDARY_ARN", "")
VECTOR_BUCKET = os.getenv("KB_VECTOR_BUCKET", "chunking-exp-vectors")
VECTOR_INDEX = "chunking-exp-baseline-index"

EMBED_MODEL_ARN = (
    f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed-text-v2:0"
)
BUCKET_ARN = f"arn:aws:s3:::{S3_BUCKET}"
VECTOR_INDEX_ARN = (
    f"arn:aws:s3vectors:{REGION}:{ACCOUNT_ID}:bucket/{VECTOR_BUCKET}/index/{VECTOR_INDEX}"
)


def build_clients():
    session = boto3.Session(region_name=REGION)
    return (
        session.client("iam"),
        session.client("bedrock-agent"),
        session.client("s3vectors"),
    )


def create_vector_index(s3v):
    print(f"Creating S3 Vectors bucket: {VECTOR_BUCKET}")
    try:
        s3v.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        print("  Bucket created.")
    except s3v.exceptions.ConflictException:
        print("  Bucket already exists, skipping.")

    print(f"Creating vector index: {VECTOR_INDEX}")
    try:
        s3v.create_index(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
        )
        print("  Index created.")
    except s3v.exceptions.ConflictException:
        print("  Index already exists, skipping.")


def create_knowledge_base(bedrock_agent, kb_name: str) -> str:
    print(f"Creating Knowledge Base: {kb_name}")
    resp = bedrock_agent.create_knowledge_base(
        name=kb_name,
        roleArn=ROLE_ARN,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": EMBED_MODEL_ARN,
                "embeddingModelConfiguration": {
                    "bedrockEmbeddingModelConfiguration": {
                        "dimensions": 1024,
                        "embeddingDataType": "FLOAT32",
                    }
                },
            },
        },
        storageConfiguration={
            "type": "S3_VECTORS",
            "s3VectorsConfiguration": {
                "vectorBucketArn": f"arn:aws:s3vectors:{REGION}:{ACCOUNT_ID}:bucket/{VECTOR_BUCKET}",
                "indexName": VECTOR_INDEX,
            },
        },
    )
    kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
    print(f"  KB created: {kb_id}")
    return kb_id


def create_data_source(bedrock_agent, kb_id: str, s3_prefix: str) -> str:
    """Single data source with Bedrock default chunking (no chunkingConfiguration)."""
    print(f"Creating data source: baseline (default chunking, prefix={s3_prefix})")
    resp = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="baseline-default-chunking",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": BUCKET_ARN,
                "inclusionPrefixes": [f"{s3_prefix}/"],
            },
        },
        # No vectorIngestionConfiguration → Bedrock uses default fixed-size chunking
    )
    ds_id = resp["dataSource"]["dataSourceId"]
    print(f"  Data source created: {ds_id}")
    return ds_id


def ingest_and_wait(bedrock_agent, kb_id: str, ds_id: str, ds_name: str):
    print(f"Starting ingestion for: {ds_name}")
    resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    print(f"  Job ID: {job_id}")

    while True:
        status_resp = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )
        job = status_resp["ingestionJob"]
        status = job["status"]
        stats = job.get("statistics", {})
        scanned = stats.get("numberOfDocumentsScanned", 0)
        indexed = stats.get("numberOfNewDocumentsIndexed", 0)
        failed = stats.get("numberOfDocumentsFailed", 0)
        print(f"  Status: {status} | scanned={scanned} indexed={indexed} failed={failed}")

        if status == "COMPLETE":
            print("  Done.")
            break
        elif status == "FAILED":
            raise RuntimeError(f"Ingestion failed for {ds_name}")

        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(
        description="Create Phase 1 baseline KB with Bedrock default chunking"
    )
    parser.add_argument("--kb-name", default="chunking-exp-baseline",
                        help='KB name (default: "chunking-exp-baseline")')
    parser.add_argument("--s3-prefix", default="chunking-exp/baseline",
                        help="S3 prefix for source documents (default: chunking-exp/baseline)")
    args = parser.parse_args()

    if not ACCOUNT_ID:
        raise ValueError("AWS_ACCOUNT_ID not set in environment / .env file")
    if not PERMISSIONS_BOUNDARY_ARN:
        raise ValueError("AWS_PERMISSIONS_BOUNDARY_ARN not set in environment / .env file")

    print(f"Config: KB={args.kb_name}  s3://{S3_BUCKET}/{args.s3_prefix}/")

    iam, bedrock_agent, s3v = build_clients()

    create_vector_index(s3v)
    kb_id = create_knowledge_base(bedrock_agent, args.kb_name)
    ds_id = create_data_source(bedrock_agent, kb_id, args.s3_prefix)
    ingest_and_wait(bedrock_agent, kb_id, ds_id, "baseline-default-chunking")

    print(f"\n=== Done ===")
    print(f"KB_ID={kb_id}")
    print(f"Add to your .env: KB_ID={kb_id}")


if __name__ == "__main__":
    main()
