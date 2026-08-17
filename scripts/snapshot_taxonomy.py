#!/usr/bin/env python3
"""Snapshot MISP taxonomy entries and galaxy clusters into a compact catalog."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tagrec.allowlist import CUSTOM_TAG_NAMESPACES, parse_tag
from tagrec.io import write_json_secure
from tagrec.misp import create_misp_client


def unwrap(item: dict[str, Any], key: str) -> dict[str, Any]:
    value = item.get(key, item)
    return value if isinstance(value, dict) else {}


def custom_tag_names(tags_index: Any) -> list[str]:
    """Instance tags from local custom vocabularies that never appear in /taxonomies."""
    if isinstance(tags_index, dict):
        tags_index = tags_index.get("Tag", [])
    if not isinstance(tags_index, list):
        raise RuntimeError("Unexpected MISP tags index response")
    names: set[str] = set()
    for item in tags_index:
        if not isinstance(item, dict):
            continue
        tag = unwrap(item, "Tag")
        name = tag.get("name")
        if not name or tag.get("hide_tag"):
            continue
        parsed = parse_tag(str(name))
        if parsed and parsed.namespace.lower() in CUSTOM_TAG_NAMESPACES:
            names.add(str(name))
    return sorted(names)


def compact_taxonomy(detail: dict[str, Any]) -> dict[str, Any]:
    taxonomy = unwrap(detail, "Taxonomy")
    entries = detail.get("entries", []) or []
    return {
        "id": taxonomy.get("id"),
        "namespace": taxonomy.get("namespace"),
        "description": taxonomy.get("description", ""),
        "version": taxonomy.get("version"),
        "enabled": bool(taxonomy.get("enabled")),
        "exclusive": bool(taxonomy.get("exclusive")),
        "required": bool(taxonomy.get("required")),
        "entries": [
            {
                "tag": entry.get("tag"),
                "expanded": entry.get("expanded", ""),
                "description": entry.get("description", ""),
                "exclusive_predicate": bool(entry.get("exclusive_predicate")),
                "existing_tag": bool(entry.get("existing_tag")),
            }
            for entry in entries
            if isinstance(entry, dict) and entry.get("tag")
        ],
    }


def compact_galaxy(detail: dict[str, Any]) -> dict[str, Any]:
    galaxy = unwrap(detail, "Galaxy")
    clusters = detail.get("GalaxyCluster", []) or []
    return {
        "id": galaxy.get("id"),
        "uuid": galaxy.get("uuid"),
        "name": galaxy.get("name"),
        "type": galaxy.get("type"),
        "namespace": galaxy.get("namespace"),
        "description": galaxy.get("description", ""),
        "version": galaxy.get("version"),
        "enabled": bool(galaxy.get("enabled")),
        "clusters": [
            {
                "id": cluster.get("id"),
                "uuid": cluster.get("uuid"),
                "type": cluster.get("type"),
                "value": cluster.get("value"),
                "tag_name": cluster.get("tag_name"),
                "description": cluster.get("description", ""),
                "source": cluster.get("source", ""),
                "authors": cluster.get("authors", []),
                "elements": [
                    {"key": element.get("key"), "value": element.get("value")}
                    for element in (cluster.get("GalaxyElement", []) or [])
                    if isinstance(element, dict) and element.get("key")
                ],
            }
            for cluster in clusters
            if isinstance(cluster, dict) and cluster.get("tag_name")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--out-dir", default="data/taxonomy")
    parser.add_argument("--max-taxonomies", type=int, help="Bounded smoke-test snapshot")
    parser.add_argument("--max-galaxies", type=int, help="Bounded smoke-test snapshot")
    args = parser.parse_args()

    misp = create_misp_client(args.env)
    generated_at = datetime.now(timezone.utc)
    taxonomies_index = misp.taxonomies(pythonify=False)
    galaxies_index = misp.galaxies(withCluster=False, pythonify=False)
    if not isinstance(taxonomies_index, list) or not isinstance(galaxies_index, list):
        raise RuntimeError("Unexpected MISP taxonomy or galaxy index response")

    if args.max_taxonomies is not None:
        taxonomies_index = taxonomies_index[: args.max_taxonomies]
    if args.max_galaxies is not None:
        galaxies_index = galaxies_index[: args.max_galaxies]

    taxonomies: list[dict[str, Any]] = []
    for number, item in enumerate(taxonomies_index, start=1):
        metadata = unwrap(item, "Taxonomy")
        detail = misp.get_taxonomy(str(metadata["id"]), pythonify=False)
        if not isinstance(detail, dict):
            raise RuntimeError(f"Unexpected taxonomy response for {metadata.get('namespace')}")
        taxonomies.append(compact_taxonomy(detail))
        print(f"Taxonomy {number}/{len(taxonomies_index)}: {metadata.get('namespace')}", file=sys.stderr)

    galaxies: list[dict[str, Any]] = []
    for number, item in enumerate(galaxies_index, start=1):
        metadata = unwrap(item, "Galaxy")
        detail = misp.get_galaxy(str(metadata["id"]), withCluster=True, pythonify=False)
        if not isinstance(detail, dict):
            raise RuntimeError(f"Unexpected galaxy response for {metadata.get('type')}")
        galaxies.append(compact_galaxy(detail))
        print(
            f"Galaxy {number}/{len(galaxies_index)}: {metadata.get('type')} "
            f"({len(galaxies[-1]['clusters'])} clusters)",
            file=sys.stderr,
        )

    custom_tags = custom_tag_names(misp.tags(pythonify=False))
    print(f"Custom org-specific tags: {len(custom_tags)}", file=sys.stderr)

    snapshot = {
        "schema_version": 2,
        "snapshot_date": generated_at.isoformat(),
        "misp_root_url": getattr(misp, "root_url", ""),
        "taxonomies": taxonomies,
        "galaxies": galaxies,
        "custom_tags": custom_tags,
        "summary": {
            "taxonomy_count": len(taxonomies),
            "enabled_taxonomy_count": sum(t["enabled"] for t in taxonomies),
            "taxonomy_entry_count": sum(len(t["entries"]) for t in taxonomies),
            "galaxy_count": len(galaxies),
            "enabled_galaxy_count": sum(g["enabled"] for g in galaxies),
            "galaxy_cluster_count": sum(len(g["clusters"]) for g in galaxies),
            "custom_tag_count": len(custom_tags),
            "bounded_smoke_snapshot": args.max_taxonomies is not None or args.max_galaxies is not None,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    destination = out_dir / f"snapshot-{stamp}.json"
    write_json_secure(destination, snapshot)

    latest = out_dir / "snapshot.json"
    temporary_link = out_dir / ".snapshot.json.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(destination.name)
    os.replace(temporary_link, latest)

    print(f"Wrote taxonomy snapshot to {destination}")
    print(f"Summary: {snapshot['summary']}")
    if snapshot["summary"]["enabled_taxonomy_count"] == 0:
        print(
            "WARNING: this MISP instance reports zero enabled standard taxonomies; "
            "dataset policy must explicitly decide whether installed-but-disabled entries are valid targets.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
