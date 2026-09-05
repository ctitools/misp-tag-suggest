#!/usr/bin/env python3
"""Build deterministic eligible-tag documents from a local taxonomy snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tagrec.allowlist import decide_tag, parse_tag
from tagrec.families import label_family
from tagrec.io import write_json_secure, write_jsonl_secure
from tagrec.taxonomy import TaxonomyCatalog


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def tag_document(tag: str, metadata: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_tag(tag)
    if parsed is None:
        raise ValueError(f"canonical tag is not parseable: {tag}")
    synonyms: list[str] = []
    external_ids: list[str] = []
    for element in metadata.get("elements", []) or []:
        if not isinstance(element, dict):
            continue
        key = str(element.get("key", "")).lower()
        value = compact(element.get("value"))
        if not value:
            continue
        if key in {"synonym", "synonyms"}:
            synonyms.append(value)
        elif key == "external_id":
            external_ids.append(value)
    value = compact(parsed.value)
    namespace = parsed.namespace.replace("-", " ").replace("_", " ")
    predicate = parsed.predicate.replace("-", " ").replace("_", " ")
    description = next(
        (
            compact(metadata.get(key))
            for key in (
                "description",
                "expanded",
                "galaxy_description",
                "taxonomy_description",
            )
            if compact(metadata.get(key))
        ),
        "",
    )
    parts = [f"{namespace}: {predicate}" + (f" = {value}" if value else "") + "."]
    if external_ids:
        parts.append("External identifier: " + ", ".join(sorted(set(external_ids))) + ".")
    if description:
        parts.append(description)
    if synonyms:
        parts.append("Synonyms: " + ", ".join(sorted(set(synonyms))) + ".")
    if metadata.get("kind") == "custom-tag" and not description:
        parts.append(
            "No catalog description is available; this document uses only the "
            "canonical name and namespace expansion."
        )
    return {
        "tag": tag,
        "family": label_family(tag),
        "namespace": parsed.namespace,
        "predicate": parsed.predicate,
        "value": parsed.value,
        "kind": metadata.get("kind", "unknown"),
        "description_available": bool(description),
        "synonyms": sorted(set(synonyms)),
        "external_ids": sorted(set(external_ids)),
        "document": " ".join(parts),
    }


def build(catalog: TaxonomyCatalog) -> list[dict[str, Any]]:
    rows = []
    for tag in sorted(catalog.valid_tags):
        decision = decide_tag(tag)
        if not decision.eligible:
            continue
        normalized, reason = catalog.normalize(tag, decision)
        if normalized != tag or reason != "eligible-and-valid":
            continue
        rows.append(tag_document(tag, catalog.tag_metadata.get(tag, {})))
    if len({str(row["tag"]) for row in rows}) != len(rows):
        raise ValueError("duplicate tag documents")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/taxonomy/snapshot.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/lfm25/tag-docs.jsonl"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/lfm25/tag-docs-manifest.json")
    )
    args = parser.parse_args()
    catalog = TaxonomyCatalog.load(args.taxonomy)
    rows = build(catalog)
    write_jsonl_secure(args.out, rows)
    families = Counter(str(row["family"]) for row in rows)
    kinds = Counter(str(row["kind"]) for row in rows)
    missing = Counter(str(row["kind"]) for row in rows if not row["description_available"])
    manifest = {
        "schema_version": 1,
        "taxonomy_snapshot_date": catalog.snapshot_date,
        "taxonomy_snapshot_sha256": hashlib.sha256(args.taxonomy.read_bytes()).hexdigest(),
        "tag_documents": len(rows),
        "by_family": dict(sorted(families.items())),
        "by_kind": dict(sorted(kinds.items())),
        "missing_description_by_kind": dict(sorted(missing.items())),
        "custom_tag_limitation": (
            "custom tags have no catalog descriptions; "
            "documents use canonical name and namespace expansion"
        ),
        "test_used": False,
    }
    write_json_secure(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
