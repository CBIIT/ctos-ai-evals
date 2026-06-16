"""
create_experiment_kb.py
Creates a new Bedrock KB with 2 data sources:
  - structured/   : pre-split YAML files, fixed-size passthrough (8192 tokens)
  - unstructured/ : PDFs + markdown, semantic chunking
"""

import argparse
import os
import json
import time
import boto3
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "")
REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "test-dsrc-04022026")

# S3 prefixes where chunk_and_upload.py put the files
PREFIX_STRUCTURED   = "chunking-exp/strategy-b/structured"
PREFIX_PDFS         = "chunking-exp/strategy-b/pdfs"
PREFIX_MARKDOWN     = "chunking-exp/strategy-b/markdown"

# Names for new resources
KB_NAME = "chunking-exp-strategy-b"
ROLE_NAME = "power-user-bedrock-chunking-exp"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/power-user-bedrock-knowledgebase-execution-role"
PERMISSIONS_BOUNDARY_ARN = os.getenv("AWS_PERMISSIONS_BOUNDARY_ARN", "")
VECTOR_BUCKET = os.getenv("KB_VECTOR_BUCKET", "chunking-exp-vectors")
VECTOR_INDEX = "chunking-exp-index"

TITAN_EMBED_V2    = "amazon.titan-embed-text-v2:0"
COHERE_EMBED_V3   = "cohere.embed-english-v3"
DEFAULT_EMBED_MODEL = TITAN_EMBED_V2

BUCKET_ARN = f"arn:aws:s3:::{S3_BUCKET}"
VECTOR_INDEX_ARN = f"arn:aws:s3vectors:{REGION}:{ACCOUNT_ID}:bucket/{VECTOR_BUCKET}/index/{VECTOR_INDEX}"


def build_clients():
    session = boto3.Session(region_name=REGION)
    return (
        session.client("iam"),
        session.client("bedrock-agent"),
        session.client("s3vectors"),
    )


def create_vector_index(s3v):
    """
    S3 Vectors is where Bedrock stores the actual vector embeddings.
    Think of it as the database that holds the 1024-dimensional float vectors
    for every chunk. dimension=1024 matches Titan Embed v2's output size.
    """
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


def create_iam_role(iam, embed_model_arn: str):
    """
    Bedrock needs permission to:
    1. Read your S3 data bucket (to ingest documents)
    2. Call the embedding model (to create vectors)
    3. Read/write the S3 Vectors index (to store/query vectors)
    """
    print(f"Creating IAM role: {ROLE_NAME}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:knowledge-base/*"
                    },
                },
            }
        ],
    }

    try:
        resp = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="IAM role for chunking experiment Bedrock KB",
            PermissionsBoundary=PERMISSIONS_BOUNDARY_ARN,  # ← add this
        )
        role_arn = resp["Role"]["Arn"]
        print(f"  Role created: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"
        print(f"  Role already exists: {role_arn}")

    # Policy 1: Read from the S3 data bucket
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="S3DataAccess",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "s3:ListBucket",
                        "Resource": BUCKET_ARN,
                        "Condition": {
                            "StringEquals": {"aws:ResourceAccount": ACCOUNT_ID}
                        },
                    },
                    {
                        "Effect": "Allow",
                        "Action": "s3:GetObject",
                        "Resource": f"{BUCKET_ARN}/*",
                        "Condition": {
                            "StringEquals": {"aws:ResourceAccount": ACCOUNT_ID}
                        },
                    },
                ],
            }
        ),
    )

    # Policy 2: Invoke the embedding model to create vectors
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="BedrockEmbedding",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "bedrock:InvokeModel",
                        "Resource": embed_model_arn,
                    }
                ],
            }
        ),
    )

    # Policy 3: Read/write the S3 Vectors index
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="S3VectorsAccess",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3vectors:GetIndex",
                            "s3vectors:QueryVectors",
                            "s3vectors:PutVectors",
                            "s3vectors:GetVectors",
                            "s3vectors:DeleteVectors",
                        ],
                        "Resource": VECTOR_INDEX_ARN,
                        "Condition": {
                            "StringEquals": {"aws:ResourceAccount": ACCOUNT_ID}
                        },
                    }
                ],
            }
        ),
    )

    print("  Policies attached.")

    # CRITICAL: IAM changes are eventually consistent.
    # If we call create_knowledge_base immediately, Bedrock will get
    # AccessDenied because the role isn't fully propagated yet.
    print("  Waiting 15s for IAM propagation...")
    time.sleep(15)

    return role_arn


def create_knowledge_base(bedrock_agent, role_arn, embed_model_arn: str, embed_dimension: int):
    print(f"Creating Knowledge Base: {KB_NAME}  (embed={embed_model_arn.split('/')[-1]}  dim={embed_dimension})")
    resp = bedrock_agent.create_knowledge_base(
        name=KB_NAME,
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embed_model_arn,
                "embeddingModelConfiguration": {
                    "bedrockEmbeddingModelConfiguration": {
                        "dimensions": embed_dimension,
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


def create_data_sources(bedrock_agent, kb_id, max_tokens: int = 1024, percentile: int = 95):
    # Data source 1: structured YAML — already exists, reuse it
    ds1_id = "XFQIKU0SD6"
    print(f"  Data source 1 (structured-yaml): reusing {ds1_id}")

    # Data source 2: PDFs — semantic chunking
    print(f"Creating data source: pdfs (semantic chunking, maxTokens={max_tokens}, percentile={percentile})")
    ds2 = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="unstructured-pdfs",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": BUCKET_ARN,
                "inclusionPrefixes": [f"{PREFIX_PDFS}/"],
            },
        },
        vectorIngestionConfiguration={
            "parsingConfiguration": {
                "parsingStrategy": "BEDROCK_DATA_AUTOMATION",
            },
            "chunkingConfiguration": {
                "chunkingStrategy": "SEMANTIC",
                "semanticChunkingConfiguration": {
                    "maxTokens": max_tokens,
                    "bufferSize": 1,
                    "breakpointPercentileThreshold": percentile,
                },
            },
        },
    )
    ds2_id = ds2["dataSource"]["dataSourceId"]
    print(f"  Data source 2 created: {ds2_id}")

    # Data source 3: Markdown — no chunking (files are small, keep them whole)
    print("Creating data source: markdown (no chunking)")
    ds3 = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="unstructured-markdown",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": BUCKET_ARN,
                "inclusionPrefixes": [f"{PREFIX_MARKDOWN}/"],
            },
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "NONE",
            },
        },
    )
    ds3_id = ds3["dataSource"]["dataSourceId"]
    print(f"  Data source 3 created: {ds3_id}")

    return ds1_id, ds2_id, ds3_id


def ingest_and_wait(bedrock_agent, kb_id, ds_id, ds_name):
    """
    Ingestion = Bedrock reads the S3 files, chunks them (if needed),
    calls Titan Embed on each chunk, and writes vectors to S3 Vectors.
    """
    print(f"Starting ingestion for: {ds_name}")
    resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    print(f"  Job ID: {job_id}")

    # Poll until done
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
        print(
            f"  Status: {status} | scanned={scanned} indexed={indexed} failed={failed}"
        )

        if status == "COMPLETE":
            print(f"  Done.")
            break
        elif status == "FAILED":
            raise RuntimeError(f"Ingestion failed for {ds_name}")

        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description="Create experiment KB with custom chunking")
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.getenv("KB_PDF_MAX_TOKENS", 1024)),
                        help="Max tokens per PDF semantic chunk (default: 1024, env: KB_PDF_MAX_TOKENS)")
    parser.add_argument("--percentile", type=int,
                        default=int(os.getenv("KB_PDF_PERCENTILE", 95)),
                        help="Breakpoint percentile threshold for PDF chunking (default: 95, env: KB_PDF_PERCENTILE)")
    parser.add_argument("--embedding-model", default=os.getenv("KB_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL),
                        help=f"Bedrock embedding model ID (default: {DEFAULT_EMBED_MODEL}, env: KB_EMBEDDING_MODEL)")
    parser.add_argument("--embedding-dimension", type=int,
                        default=int(os.getenv("KB_EMBEDDING_DIMENSION", 1024)),
                        help="Embedding output dimension — must match the model (default: 1024, env: KB_EMBEDDING_DIMENSION)")
    args = parser.parse_args()

    if not ACCOUNT_ID:
        raise ValueError("AWS_ACCOUNT_ID not set in environment / .env file")
    if not PERMISSIONS_BOUNDARY_ARN:
        raise ValueError("AWS_PERMISSIONS_BOUNDARY_ARN not set in environment / .env file")

    embed_model_arn = f"arn:aws:bedrock:{REGION}::foundation-model/{args.embedding_model}"
    print(f"Config: max_tokens={args.max_tokens}  percentile={args.percentile}  embed={args.embedding_model}  dim={args.embedding_dimension}")

    iam, bedrock_agent, s3v = build_clients()

    create_vector_index(s3v)
    kb_id = "XSYF4HNBQF"  # already created

    # Create separate PDF and markdown data sources (structured-yaml already indexed)
    _, ds2_id, ds3_id = create_data_sources(bedrock_agent, kb_id, args.max_tokens, args.percentile)

    ingest_and_wait(bedrock_agent, kb_id, ds2_id, "unstructured-pdfs")
    ingest_and_wait(bedrock_agent, kb_id, ds3_id, "unstructured-markdown")

    print(f"\n=== Done ===")
    print(f"KB_ID_B={kb_id}")
    print(f"Add this to your .env: KB_ID_B={kb_id}")


if __name__ == "__main__":
    main()
