"""
build_model_docs.py
Converts CRDC data model YML files into rich markdown documents for KB ingestion.

Produces one markdown file per model (CDS.md, ICDC.md, CTDC.md, PSDC.md).
Each file has one H2 section per node containing:
  - Node description and category
  - Required properties: name, type, description, enum values
  - Optional properties: name, type, description, enum values
  - Relationships: grouped by relationship name, listing all endpoints with multiplicity

Output goes to datasource/data-models/markdown/ by default.

Usage:
  uv run python build_model_docs.py
  uv run python build_model_docs.py --output-dir /tmp/model-docs
"""

import argparse
from pathlib import Path

import yaml

DATASOURCE_DIR = Path(__file__).parent / "datasource" / "data-models"

MODELS = [
    {
        "handle": "CDS",
        "model_file": DATASOURCE_DIR / "CDS" / "11.0.3" / "cds-model.yml",
        "props_file": DATASOURCE_DIR / "CDS" / "11.0.3" / "cds-model-props.yml",
    },
    {
        "handle": "ICDC",
        "model_file": DATASOURCE_DIR / "ICDC" / "1.3.0" / "icdc-model.yml",
        "props_file": DATASOURCE_DIR / "ICDC" / "1.3.0" / "icdc-model-props.yml",
    },
    {
        "handle": "CTDC",
        "model_file": DATASOURCE_DIR / "CTDC" / "1.21.0" / "ctdc_model_file.yaml",
        "props_file": DATASOURCE_DIR / "CTDC" / "1.21.0" / "ctdc_model_properties_file.yaml",
    },
    {
        "handle": "PSDC",
        "model_file": DATASOURCE_DIR / "PSDC" / "1.0.0" / "popsci-model.yml",
        "props_file": DATASOURCE_DIR / "PSDC" / "1.0.0" / "popsci-model-props.yml",
    },
]

HANDLE_FULL_NAMES = {
    "CDS": "Cancer Data Service",
    "ICDC": "Integrated Canine Data Commons",
    "CTDC": "Clinical Trial Data Commons",
    "PSDC": "Population Science Data Commons",
}


def load_model(model_file: Path, props_file: Path) -> tuple[dict, dict, dict]:
    with open(model_file) as f:
        model = yaml.safe_load(f) or {}
    with open(props_file) as f:
        props = yaml.safe_load(f) or {}
    return (
        model.get("Nodes", {}),
        model.get("Relationships", {}),
        props.get("PropDefinitions", {}),
    )


def format_type(type_val) -> str:
    if not type_val:
        return ""
    if isinstance(type_val, str):
        return type_val
    if isinstance(type_val, dict):
        vt = type_val.get("value_type", "")
        it = type_val.get("item_type", "")
        if it:
            return f"{vt} of {it}"
        return vt
    return str(type_val)


def build_prop_line(prop_name: str, meta: dict) -> str:
    if not isinstance(meta, dict):
        return f"- **{prop_name}**"

    parts = [f"- **{prop_name}**"]

    typ = format_type(meta.get("Type"))
    if typ:
        parts.append(f"*(type: {typ})*")

    req = meta.get("Req", False)
    if req:
        parts.append("*(required)*")

    desc = (meta.get("Desc") or "").strip()
    if desc:
        parts.append(f"— {desc}")

    enum_vals = meta.get("Enum") or []
    if enum_vals:
        preview = [str(v) for v in enum_vals[:20]]
        suffix = f" … and {len(enum_vals) - 20} more" if len(enum_vals) > 20 else ""
        parts.append(f"\n  Allowed values: {', '.join(preview)}{suffix}.")

    cde_terms = meta.get("Term") or []
    if cde_terms and isinstance(cde_terms[0], dict):
        code = cde_terms[0].get("Code", "")
        if code:
            parts.append(f"(CDE: {code})")

    return " ".join(parts)


def build_node_section(
    node_name: str,
    node_data: dict,
    prop_defs: dict,
    relationships: dict,
) -> str:
    lines = []
    lines.append(f"## Node: {node_name}")
    lines.append("")

    desc = (node_data.get("Desc") or "").strip()
    if desc:
        lines.append(desc)
        lines.append("")

    tags = node_data.get("Tags") or {}
    tag_parts = []
    if tags.get("Category"):
        tag_parts.append(f"Category: {tags['Category']}")
    if tags.get("Assignment"):
        tag_parts.append(f"Assignment: {tags['Assignment']}")
    if tags.get("Class"):
        tag_parts.append(f"Class: {tags['Class']}")
    node_req = tags.get("nodeReq", False)
    if node_req:
        tag_parts.append("Required node: yes")
    if tag_parts:
        lines.append(" | ".join(tag_parts))
        lines.append("")

    prop_names = node_data.get("Props") or []
    required_props = []
    optional_props = []
    for pname in prop_names:
        meta = prop_defs.get(pname, {})
        if isinstance(meta, dict) and meta.get("Req"):
            required_props.append((pname, meta))
        else:
            optional_props.append((pname, meta))

    if required_props:
        lines.append("### Required Properties")
        lines.append("")
        for pname, meta in required_props:
            lines.append(build_prop_line(pname, meta))
        lines.append("")

    if optional_props:
        lines.append("### Optional Properties")
        lines.append("")
        for pname, meta in optional_props:
            lines.append(build_prop_line(pname, meta))
        lines.append("")

    # Relationships: collect all where this node is Src or Dst
    node_rels = []
    for rel_name, rel_data in relationships.items():
        ends = rel_data.get("Ends") or []
        top_mul = rel_data.get("Mul", "")
        matching_ends = [e for e in ends if e.get("Src") == node_name or e.get("Dst") == node_name]
        if matching_ends:
            node_rels.append((rel_name, rel_data, top_mul, ends))

    if node_rels:
        lines.append("### Relationships")
        lines.append("")
        for rel_name, rel_data, top_mul, all_ends in node_rels:
            lines.append(f"**{rel_name}**")
            for end in all_ends:
                src = end.get("Src", "")
                dst = end.get("Dst", "")
                mul = end.get("Mul") or top_mul
                marker = " ← (this node)" if dst == node_name else (" → (this node)" if src == node_name else "")
                lines.append(f"  - {src} → {dst} ({mul}){marker}")
            lines.append("")

    return "\n".join(lines)


def build_properties_index(nodes: dict, prop_defs: dict) -> str:
    """
    Flat index of every property across all nodes: property name, which node(s)
    it belongs to, required status, type, enum values. Allows retrieval by
    property name alone without knowing the parent node.
    """
    # Build map: prop_name -> list of node names that reference it
    prop_to_nodes: dict[str, list[str]] = {}
    for node_name, node_data in nodes.items():
        for pname in node_data.get("Props") or []:
            prop_to_nodes.setdefault(pname, []).append(node_name)

    lines = []
    lines.append("## Properties Index")
    lines.append("")
    lines.append(
        "Quick-reference index of all properties in this model, listing which node(s) "
        "each property belongs to along with its type, required status, allowed values, "
        "and description."
    )
    lines.append("")

    for prop_name in sorted(prop_to_nodes.keys()):
        node_names = prop_to_nodes[prop_name]
        meta = prop_defs.get(prop_name, {})
        if not isinstance(meta, dict):
            meta = {}

        req = meta.get("Req", False)
        typ = format_type(meta.get("Type"))
        desc = (meta.get("Desc") or "").strip()
        enum_vals = meta.get("Enum") or []

        parts = [f"- **{prop_name}**"]
        parts.append(f"(node: {', '.join(node_names)})")
        if req:
            parts.append("*(required)*")
        if typ:
            parts.append(f"*(type: {typ})*")
        if desc:
            parts.append(f"— {desc}")
        if enum_vals:
            preview = [str(v) for v in enum_vals[:20]]
            suffix = f" … and {len(enum_vals) - 20} more" if len(enum_vals) > 20 else ""
            parts.append(f"Allowed values: {', '.join(preview)}{suffix}.")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def build_model_doc(handle: str, model_file: Path, props_file: Path) -> str:
    nodes, relationships, prop_defs = load_model(model_file, props_file)

    with open(model_file) as f:
        raw = yaml.safe_load(f) or {}
    version = raw.get("Version", "")
    full_name = HANDLE_FULL_NAMES.get(handle, handle)

    lines = []
    lines.append(f"# {full_name} ({handle}) Data Model {version}")
    lines.append("")
    lines.append(
        f"This document describes all nodes, properties, and relationships in the {handle} data model. "
        f"Each section covers one node type: its description, required and optional properties "
        f"(with types, allowed values, and CDE codes where applicable), and all relationships "
        f"the node participates in."
    )
    lines.append("")

    for node_name, node_data in nodes.items():
        lines.append(build_node_section(node_name, node_data, prop_defs, relationships))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build markdown model docs from YML data model files")
    parser.add_argument(
        "--output-dir",
        default=str(DATASOURCE_DIR / "markdown"),
        help="Directory to write markdown files (default: datasource/data-models/markdown/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in MODELS:
        handle = model_cfg["handle"]
        print(f"Building {handle}.md ...")
        doc = build_model_doc(handle, model_cfg["model_file"], model_cfg["props_file"])
        out_path = out_dir / f"{handle}.md"
        out_path.write_text(doc, encoding="utf-8")
        size_kb = len(doc.encode()) / 1024
        node_count = doc.count("\n## Node:")
        print(f"  Written: {out_path}  ({size_kb:.1f} KB, {node_count} nodes)")

    print(f"\nDone. Files in: {out_dir}")


if __name__ == "__main__":
    main()
