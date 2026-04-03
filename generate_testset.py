"""
Phase 1: RAGAS Testset Generator for Bedrock Knowledge Base
Generates 60 questions (20 per group) from YML, PDF, and Markdown sources.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import boto3
import litellm
import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from ragas.embeddings.base import LangchainEmbeddingsWrapper  # private: avoids deprecation warning
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    CustomNodeFilter,
    EmbeddingExtractor,
    OverlapScoreBuilder,
    Parallel,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

load_dotenv()

# Restrict RAGAS query styles to perfect grammar only.
# RAGAS calls list(QueryStyle) in synthesizer base classes to pick a style.
# We replace the module-level QueryStyle reference with a single-value enum
# so only PERFECT_GRAMMAR is ever selected.
import ragas.testset.synthesizers.single_hop.base as _sh_base
import ragas.testset.synthesizers.multi_hop.base as _mh_base
from ragas.testset.synthesizers.base import QueryStyle as _QueryStyle

class _PerfectGrammarOnly:
    """Iterable that yields only PERFECT_GRAMMAR — used to replace QueryStyle in synthesizers."""
    PERFECT_GRAMMAR = _QueryStyle.PERFECT_GRAMMAR
    def __iter__(self):
        return iter([_QueryStyle.PERFECT_GRAMMAR])

_sh_base.QueryStyle = _PerfectGrammarOnly()  # type: ignore
_mh_base.QueryStyle = _sh_base.QueryStyle    # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASOURCE_DIR = Path(__file__).parent / "datasource"
OUTPUT_FILE    = Path(__file__).parent / "testsets" / "testset.jsonl"

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")

NOVA_PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

QUESTIONS_PER_GROUP = 20
MD_QUESTIONS = 10  # fewer MD source files — cap to avoid over-requesting


# ---------------------------------------------------------------------------
# YML Graph Schema Parser
# ---------------------------------------------------------------------------

def _load_props_file(props_path: Path) -> dict:
    """Load a *-props.yml / *-model-props.yml file into a flat dict of prop -> desc."""
    if not props_path.exists():
        return {}
    with open(props_path) as f:
        data = yaml.safe_load(f) or {}
    prop_defs = data.get("PropDefinitions", {})
    descriptions = {}
    for prop_name, meta in (prop_defs.items() if isinstance(prop_defs, dict) else []):
        desc = ""
        if isinstance(meta, dict):
            desc = meta.get("Desc", "") or ""
        descriptions[prop_name] = desc
    return descriptions


def _find_companion_props_file(model_path: Path) -> Path:
    """Given a model YML path, find its companion *-props.yml in the same directory."""
    directory = model_path.parent
    stem = model_path.stem  # e.g. "cds-model", "ctdc_model_file", "popsci-model"

    # Common naming patterns: replace "-model" with "-model-props", etc.
    candidates = [
        directory / (stem.replace("-model", "-model-props") + model_path.suffix),
        directory / (stem.replace("_model_file", "_model_properties_file") + ".yaml"),
        directory / (stem.replace("_model_file", "_model_properties_file") + ".yml"),
        directory / (stem + "-props" + model_path.suffix),
        directory / (stem + "_props" + model_path.suffix),
    ]
    # Also scan directory for any *props* file
    for candidate in candidates:
        if candidate.exists() and candidate != model_path:
            return candidate

    for sibling in directory.iterdir():
        if "props" in sibling.name.lower() and sibling != model_path:
            return sibling

    return Path("/dev/null")  # sentinel — will not exist


def parse_yml_graph_schema(yml_path: Path) -> list[Document]:
    """
    Parse a graph-schema YML model file into LangChain Documents.

    For each node:
      - One document with node description + property list (with descriptions where available).
    For each relationship:
      - One document per (Src, Dst) pair describing the relationship.

    Tags every Document with metadata: file_type='yml', source=filename.
    """
    with open(yml_path) as f:
        data = yaml.safe_load(f) or {}

    source_name = yml_path.name
    model_handle = data.get("Handle", yml_path.stem)
    version = data.get("Version", "")

    props_file = _find_companion_props_file(yml_path)
    prop_descriptions = _load_props_file(props_file)

    documents: list[Document] = []

    # --- Node documents ---
    nodes = data.get("Nodes", {}) or {}
    for node_name, node_meta in nodes.items():
        if not isinstance(node_meta, dict):
            continue

        node_desc = node_meta.get("Desc", "").strip()
        props_list = node_meta.get("Props", []) or []

        # Build human-readable property sentences
        prop_sentences = []
        for prop in props_list:
            if prop is None:
                continue
            prop_str = str(prop)
            prop_desc = prop_descriptions.get(prop_str, "")
            if prop_desc:
                prop_sentences.append(f"{prop_str} ({prop_desc})")
            else:
                prop_sentences.append(prop_str)

        props_text = ", ".join(prop_sentences) if prop_sentences else "none"

        text_parts = [
            f"Model: {model_handle} {version}.",
            f"Node type: {node_name}.",
        ]
        if node_desc:
            text_parts.append(node_desc)
        text_parts.append(f"A {node_name} has properties: {props_text}.")

        # Category tag if present
        tags = node_meta.get("Tags", {}) or {}
        category = tags.get("Category", "")
        if category:
            text_parts.append(f"Category: {category}.")

        documents.append(Document(
            page_content=" ".join(text_parts),
            metadata={"file_type": "yml", "source": source_name},
        ))

    # --- Relationship documents ---
    relationships = data.get("Relationships", {}) or {}
    for rel_name, rel_meta in relationships.items():
        if not isinstance(rel_meta, dict):
            continue

        ends = rel_meta.get("Ends", []) or []
        multiplicity = rel_meta.get("Mul", "")

        # Flatten nested ends (some models nest Ends inside each other)
        flat_ends = []
        for end in ends:
            if isinstance(end, dict):
                src = end.get("Src") or end.get("src")
                dst = end.get("Dst") or end.get("dst")
                if src and dst:
                    flat_ends.append((str(src), str(dst)))

        if not flat_ends:
            # Relationship has no typed ends listed — still emit a generic doc
            text = (
                f"Model: {model_handle} {version}. "
                f"Relationship '{rel_name}' exists in the {model_handle} graph schema."
            )
            if multiplicity:
                text += f" Multiplicity: {multiplicity}."
            documents.append(Document(
                page_content=text,
                metadata={"file_type": "yml", "source": source_name},
            ))
        else:
            for src, dst in flat_ends:
                text = (
                    f"Model: {model_handle} {version}. "
                    f"A {src} relates to {dst} via {rel_name}."
                )
                if multiplicity:
                    text += f" Multiplicity: {multiplicity}."
                documents.append(Document(
                    page_content=text,
                    metadata={"file_type": "yml", "source": source_name},
                ))

    return documents


# ---------------------------------------------------------------------------
# Document Loaders per Group
# ---------------------------------------------------------------------------

def load_yml_documents() -> list[Document]:
    """Group 1: YML graph schema models from datasource/data-models/."""
    data_models_dir = DATASOURCE_DIR / "data-models"
    docs: list[Document] = []

    yml_extensions = {".yml", ".yaml"}
    # Exclude *-props* files from being parsed as model files
    # (props files are companion files consumed by the parser internally)
    props_patterns = re.compile(r"(props|properties)", re.IGNORECASE)

    for yml_path in sorted(data_models_dir.rglob("*")):
        if yml_path.suffix.lower() not in yml_extensions:
            continue
        if props_patterns.search(yml_path.stem):
            continue  # skip companion props files — parsed internally
        parsed = parse_yml_graph_schema(yml_path)
        if parsed:
            docs.extend(parsed)
            print(f"  [yml] Parsed {yml_path.relative_to(DATASOURCE_DIR)}: {len(parsed)} docs")

    return docs


def load_pdf_documents() -> list[Document]:
    """Group 2: PDFs from datasource/."""
    docs: list[Document] = []
    for pdf_path in sorted(DATASOURCE_DIR.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["file_type"] = "pdf"
            page.metadata["source"] = pdf_path.name
        docs.extend(pages)
        print(f"  [pdf] Loaded {pdf_path.name}: {len(pages)} pages")
    return docs


def load_md_documents() -> list[Document]:
    """Group 3: Markdown files from datasource/ (and subdirectories)."""
    docs: list[Document] = []
    for md_path in sorted(DATASOURCE_DIR.rglob("*.md")):
        loader = TextLoader(str(md_path), encoding="utf-8")
        pages = loader.load()
        for page in pages:
            page.metadata["file_type"] = "md"
            page.metadata["source"] = md_path.name
        docs.extend(pages)
        print(f"  [md] Loaded {md_path.relative_to(DATASOURCE_DIR)}: {len(pages)} docs")
    return docs


# ---------------------------------------------------------------------------
# LLM & Embeddings Setup
# ---------------------------------------------------------------------------

def build_llm_and_embeddings():
    # Resolve credentials from profile and inject into environment so that
    # both LiteLLM (for the LLM) and boto3 (for embeddings) pick them up.
    boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    creds = boto_session.get_credentials().get_frozen_credentials()
    # Overwrite env with whatever boto3 resolved (handles both long-term IAM
    # keys and temporary STS/assumed-role credentials consistently).
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    os.environ["AWS_REGION_NAME"] = AWS_REGION
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)

    # instructor-wrapped LiteLLM client — ragas LiteLLMStructuredLLM calls
    # client.chat.completions.create(..., response_model=<PydanticModel>)
    # drop_params: Bedrock Converse does not support tool_choice; drop silently
    import instructor
    litellm.drop_params = True
    # MD_JSON: Nova Pro doesn't reliably invoke tool calls for structured output,
    # but does return well-formed markdown-fenced JSON that instructor can parse.
    litellm_client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
    litellm_model = f"bedrock/converse/{NOVA_PRO_MODEL_ID}"

    generator_llm = llm_factory(
        model=litellm_model,
        provider="bedrock",
        client=litellm_client,
        adapter="litellm",
    )

    # Embeddings: BedrockEmbeddings via langchain-aws (no native ragas Bedrock provider)
    bedrock_embed_client = boto_session.client("bedrock-runtime")
    bedrock_embeddings = BedrockEmbeddings(
        model_id=TITAN_EMBED_MODEL_ID,
        region_name=AWS_REGION,
        client=bedrock_embed_client,
    )
    generator_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)

    return generator_llm, generator_embeddings


# ---------------------------------------------------------------------------
# Testset Generation per Group
# ---------------------------------------------------------------------------

def build_md_transforms(llm, embeddings):
    """MEDIUM pipeline without HeadlinesExtractor/HeadlineSplitter (safe for all MD files)."""
    return [
        SummaryExtractor(llm=llm),
        CustomNodeFilter(llm=llm),
        Parallel(
            EmbeddingExtractor(embedding_model=embeddings),
            ThemesExtractor(llm=llm),
            NERExtractor(llm=llm),
        ),
        Parallel(
            CosineSimilarityBuilder(),
            OverlapScoreBuilder(),
        ),
    ]


def generate_for_group(
    docs: list[Document],
    generator: TestsetGenerator,
    n: int,
    file_type: str,
    transforms=None,
) -> list[dict]:
    """Generate n questions from docs, return list of output records."""
    if not docs:
        print(f"  [WARN] No documents for group '{file_type}', skipping.")
        return []

    print(f"\n  Generating {n} questions for group '{file_type}' ({len(docs)} source docs)...")
    dataset = generator.generate_with_langchain_docs(docs, testset_size=n, transforms=transforms)
    df = dataset.to_pandas()

    records = []
    for _, row in df.iterrows():
        contexts = row.get("reference_contexts") or row.get("contexts") or []
        if not isinstance(contexts, list):
            contexts = [str(contexts)] if contexts else []

        # Infer question type from ragas column names
        question_type = _infer_question_type(row)

        # Best-effort source from contexts metadata
        source = _extract_source(row, file_type)

        records.append({
            "question": str(row.get("user_input", row.get("question", ""))),
            "ground_truth": str(row.get("reference", row.get("ground_truth", ""))),
            "contexts": contexts,
            "metadata": {
                "source": source,
                "file_type": file_type,
                "question_type": question_type,
            },
        })

    return records


def _infer_question_type(row) -> str:
    """Map ragas synthesizer/evolution type to simple|reasoning|multi_context."""
    # ragas 0.4.x exposes a 'synthesizer_name' or 'evolution_type' column
    for col in ("synthesizer_name", "evolution_type", "question_type"):
        val = str(row.get(col, "")).lower()
        if not val or val == "nan":
            continue
        if "multi" in val or "multi_context" in val:
            return "multi_context"
        if "reason" in val or "abstract" in val or "conditional" in val:
            return "reasoning"
        if "simple" in val or "single" in val or "factoid" in val:
            return "simple"
    return "simple"


def _extract_source(row, file_type: str) -> str:
    """Best-effort extraction of source filename from row metadata."""
    contexts = row.get("contexts", [])
    if isinstance(contexts, list) and contexts:
        # Contexts may be strings; look for a source key in metadata if available
        pass
    # Fall back to file_type label
    return file_type


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], output_path: Path, mode: str = "w") -> None:
    with open(output_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nAppended {len(records)} records to {output_path}")


def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_file_type: dict[str, int] = defaultdict(int)
    by_question_type: dict[str, int] = defaultdict(int)
    cross: dict[tuple, int] = defaultdict(int)

    for r in records:
        ft = r["metadata"]["file_type"]
        qt = r["metadata"]["question_type"]
        by_file_type[ft] += 1
        by_question_type[qt] += 1
        cross[(ft, qt)] += 1

    print(f"\nTotal questions: {len(records)}")

    print("\nBy file_type:")
    for ft in sorted(by_file_type):
        print(f"  {ft:10s}: {by_file_type[ft]}")

    print("\nBy question_type:")
    for qt in sorted(by_question_type):
        print(f"  {qt:15s}: {by_question_type[qt]}")

    print("\nCross-tab (file_type x question_type):")
    header = f"  {'':12s}" + "".join(f"{qt:15s}" for qt in sorted(by_question_type))
    print(header)
    for ft in sorted(by_file_type):
        row_str = f"  {ft:12s}"
        for qt in sorted(by_question_type):
            row_str += f"{cross.get((ft, qt), 0):<15d}"
        print(row_str)

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading documents...")

    print("\nGroup 1 — YML graph schemas:")
    yml_docs = load_yml_documents()

    print("\nGroup 2 — PDFs:")
    pdf_docs = load_pdf_documents()

    print("\nGroup 3 — Markdown:")
    md_docs = load_md_documents()

    print(f"\nDocument counts: yml={len(yml_docs)}, pdf={len(pdf_docs)}, md={len(md_docs)}")

    print("\nInitialising LLM and embeddings (Amazon Bedrock)...")
    generator_llm, generator_embeddings = build_llm_and_embeddings()

    # Pre-define personas to avoid ragas 0.4.3 bug where auto-generated
    # duplicate persona names cause a KeyError during scenario generation.
    personas = [
        Persona(name="Data Scientist", role_description="Analyzes datasets and builds models using data from the knowledge base."),
        Persona(name="Clinical Researcher", role_description="Studies disease patterns and treatment outcomes using clinical data models."),
        Persona(name="Data Submitter", role_description="Uploads and manages data submissions to the CRDC Data Hub."),
        Persona(name="Bioinformatician", role_description="Processes genomic and molecular data using CRDC data models."),
        Persona(name="Software Engineer", role_description="Integrates with CRDC APIs and data models to build applications."),
    ]
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
    )

    # Read any already-completed groups from a previous partial run.
    all_records: list[dict] = []
    completed_types: set[str] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                all_records.append(r)
                completed_types.add(r["metadata"]["file_type"])
        print(f"\nResuming: found {len(all_records)} existing records for groups: {sorted(completed_types)}")
    else:
        OUTPUT_FILE.touch()

    for file_type, docs, n, transforms in [
        ("yml", yml_docs, QUESTIONS_PER_GROUP, None),
        ("pdf", pdf_docs, QUESTIONS_PER_GROUP, None),
        ("md", md_docs, MD_QUESTIONS, build_md_transforms(generator_llm, generator_embeddings)),
    ]:
        if file_type in completed_types:
            print(f"\n  Skipping '{file_type}' — already in output file.")
            continue
        records = generate_for_group(docs, generator, n, file_type, transforms)
        if records:
            write_jsonl(records, OUTPUT_FILE, mode="a")
            all_records.extend(records)

    print_summary(all_records)


if __name__ == "__main__":
    main()
