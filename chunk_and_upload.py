"""
chunk_and_upload.py
Schema-aware chunking for CRDC data model files.
Parses YAML model files into atomic documents and uploads to S3.
"""

import csv
import os
import sys
import yaml
import boto3
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document

# So we can reuse the existing node/relationship parser without copying it
sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_testset import parse_yml_graph_schema, _find_companion_props_file

load_dotenv()

DATASOURCE_DIR = Path(__file__).parent.parent / "datasource"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
S3_BUCKET = os.getenv("S3_BUCKET", "")


def parse_props_to_documents(
    props_path: Path, model_handle: str, version: str
) -> list[Document]:
    """
    Parse a *-props.yml into one Document per PropDefinition.
    Each document is one atomic semantic unit — never split by Bedrock.
    """
    with open(props_path) as f:
        data = yaml.safe_load(f) or {}

    prop_defs = data.get("PropDefinitions", {}) or {}
    documents = []

    for prop_name, meta in prop_defs.items():
        if not isinstance(meta, dict):
            continue

        desc = meta.get("Desc", "") or ""
        req = meta.get("Req", False)
        typ = meta.get("Type", "") or ""
        enum_vals = meta.get("Enum", []) or []

        # CDE reference if present
        terms = meta.get("Term", []) or []
        cde_code = ""
        if terms and isinstance(terms[0], dict):
            cde_code = str(terms[0].get("Code", ""))

        # Build the human-readable content string
        parts = [
            f"Model: {model_handle} {version}.",
            f"Property: {prop_name}.",
        ]
        if desc:
            parts.append(f"Description: {desc}.")
        if typ:
            parts.append(f"Type: {typ}.")
        if req:
            parts.append("Required: true.")
        if enum_vals:
            preview = [str(v) for v in enum_vals[:20]]
            suffix = f" (and {len(enum_vals) - 20} more)" if len(enum_vals) > 20 else ""
            parts.append(f"Allowed values: {', '.join(preview)}{suffix}.")
        if cde_code:
            parts.append(f"CDE code: {cde_code} (caDSR).")

        content = " ".join(parts)

        # Rich metadata — enables filtered retrieval later
        metadata = {
            "model":     model_handle,
            "version":   version,
            "doc_kind":  "prop",
            "prop_name": prop_name,
            "required":  str(req).lower(),
            "has_enum":  "true" if enum_vals else "false",
            "source":    props_path.name,
            "file_type": "yml",
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def load_csv_documents() -> list[Document]:
    """
    Walk datasource/ for CSV files and convert each data row to one Document.
    Rows are already atomic — Bedrock will store them as-is (passthrough chunking).
    """
    docs = []
    for csv_file in sorted(DATASOURCE_DIR.rglob("*.csv")):
        print(f"  Parsing CSV: {csv_file.name}")
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = "\n".join(
                    f"{k}: {v}" for k, v in row.items() if v and str(v).strip()
                )
                if not text.strip():
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": csv_file.name,
                            "row_index": i,
                            "file_type": "csv",
                        },
                    )
                )
        print(f"    → {sum(1 for d in docs if d.metadata['source'] == csv_file.name)} row documents")
    print(f"\nTotal CSV documents: {len(docs)}")
    return docs


def load_all_structured_documents() -> list[Document]:
    """
    Walk data-models/ and return all Documents:
    - Nodes and relationships from model files (via parse_yml_graph_schema)
    - PropDefinitions from props files (via parse_props_to_documents)
    """
    models_dir = DATASOURCE_DIR / "data-models"
    all_docs = []

    for yml_path in sorted(models_dir.rglob("*.yml")) + sorted(
        models_dir.rglob("*.yaml")
    ):
        stem = yml_path.stem.lower()

        # Skip props files here — we handle them separately below
        if "props" in stem or "properties" in stem:
            continue

        print(f"  Parsing model file: {yml_path.name}")

        # Load model handle + version for props parser
        with open(yml_path) as f:
            model_data = yaml.safe_load(f) or {}
        handle = model_data.get("Handle", yml_path.stem)
        version = model_data.get("Version", "")

        # Nodes + relationships
        node_rel_docs = parse_yml_graph_schema(yml_path)
        for doc in node_rel_docs:
            doc.metadata = {
                "doc_kind": "node_or_rel",
                "source": doc.metadata.get("source", ""),
            }

        all_docs.extend(node_rel_docs)

        # Props
        props_path = _find_companion_props_file(yml_path)
        if props_path.exists():
            print(f"    Props file: {props_path.name}")
            prop_docs = parse_props_to_documents(props_path, handle, version)
            print(f"    → {len(prop_docs)} property documents")
            all_docs.extend(prop_docs)

    print(f"\nTotal structured documents: {len(all_docs)}")
    return all_docs


def write_docs_to_files(docs: list[Document], output_dir: Path) -> list[Path]:
    """Write each Document to a separate .txt file. Returns list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for i, doc in enumerate(docs):
        meta = doc.metadata
        model = meta.get("model", "unknown").upper()
        kind = meta.get("doc_kind", "doc")
        prop = meta.get("prop_name", "")

        # Build a unique, readable filename
        if prop:
            safe_name = prop.replace("/", "_").replace(" ", "_")
            filename = f"{model}__prop__{safe_name}.txt"
        else:
            filename = f"{model}__{kind}__{i:04d}.txt"

        file_path = output_dir / filename
        file_path.write_text(doc.page_content, encoding="utf-8")
        written.append(file_path)

    return written


def upload_to_s3(
    files: list[Path], bucket: str, prefix: str, session: boto3.Session
) -> None:
    """Upload a list of local files to s3://bucket/prefix/."""
    s3 = session.client("s3")
    for file_path in files:
        key = f"{prefix}/{file_path.name}"
        s3.upload_file(str(file_path), bucket, key)
        print(f"  Uploaded: {key}")


def upload_unstructured_with_metadata(
    files: list[Path], bucket: str, prefix: str, session: boto3.Session
) -> None:
    """Upload PDFs and markdown with minimal sidecar .metadata.json files.

    Bedrock reads the sidecar instead of auto-generating metadata from file
    content, keeping the stored metadata well under the 2048-byte S3 Vectors limit.
    """
    import json

    s3 = session.client("s3")
    for file_path in files:
        # Upload the file itself
        key = f"{prefix}/{file_path.name}"
        s3.upload_file(str(file_path), bucket, key)
        print(f"  Uploaded: {key}")

        # Upload minimal sidecar metadata
        ext = file_path.suffix.lstrip(".")
        metadata = {
            "metadataAttributes": {
                "file_type": ext,
                "source": file_path.name,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(metadata, tmp)
            tmp_path = Path(tmp.name)

        meta_key = f"{prefix}/{file_path.name}.metadata.json"
        s3.upload_file(str(tmp_path), bucket, meta_key)
        tmp_path.unlink()
        print(f"  Uploaded: {meta_key}")


def main():
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET not set in .env")

    session = boto3.Session(region_name=AWS_REGION)

    print("=== Loading and parsing structured YAML documents ===")
    docs = load_all_structured_documents()

    print("\n=== Writing to temp files ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        files = write_docs_to_files(docs, tmp_path)
        print(f"  {len(files)} files written")

        print("\n=== Uploading structured files to S3 ===")
        upload_to_s3(files, S3_BUCKET, "chunking-exp/strategy-b/structured", session)

    print("\n=== Uploading PDFs to S3 ===")
    pdfs = list(DATASOURCE_DIR.glob("*.pdf"))
    upload_unstructured_with_metadata(
        pdfs, S3_BUCKET, "chunking-exp/strategy-b/pdfs", session
    )

    print("\n=== Uploading markdown to S3 ===")
    mds = list(DATASOURCE_DIR.rglob("*.md"))
    upload_unstructured_with_metadata(
        mds, S3_BUCKET, "chunking-exp/strategy-b/markdown", session
    )

    print("\n=== Loading and uploading CSV row documents ===")
    csv_docs = load_csv_documents()
    if csv_docs:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_files = write_docs_to_files(csv_docs, tmp_path)
            print(f"  {len(csv_files)} files written")
            upload_to_s3(csv_files, S3_BUCKET, "chunking-exp/strategy-b/csvs", session)
    else:
        print("  No CSV files found in datasource/")

    print("\nDone.")


if __name__ == "__main__":
    main()
