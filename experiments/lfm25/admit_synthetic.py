#!/usr/bin/env python3
"""Create a fail-closed admitted synthetic corpus and rejection ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.lfm25.validate_synthetic import validate_bundle, validate_seed_contract
from tagrec.allowlist import parse_tag
from tagrec.io import iter_jsonl, write_json_secure, write_jsonl_secure

GENERIC_VALUES = {"other", "unknown", "none", "unspecified", "miscellaneous"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leakage_errors(bundle: dict[str, Any]) -> list[str]:
    parsed = parse_tag(str(bundle.get("target_tag", "")))
    target = str(parsed.value if parsed else "").strip().casefold()
    if not target or target in GENERIC_VALUES:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(target) + r"(?!\w)")
    errors = []
    for scenario in bundle.get("scenarios", []) or []:
        if scenario.get("variant") not in {"implicit_positive", "partial_positive"}:
            continue
        event = json.dumps(scenario.get("event", {}), ensure_ascii=False).casefold()
        if pattern.search(event):
            errors.append(f"exact target value leaked in {scenario.get('variant')}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument(
        "--schema", type=Path, default=Path(__file__).with_name("synthetic_bundle.schema.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    seeds = {str(row["bundle_id"]): row for row in iter_jsonl(args.seeds)}
    accepted = []
    rejected = []
    families: Counter[str] = Counter()
    for bundle in iter_jsonl(args.bundles):
        bundle_id = str(bundle.get("bundle_id", ""))
        seed = seeds.get(bundle_id)
        errors = []
        if seed is None:
            errors.append("missing frozen seed")
        else:
            errors.extend(
                validate_bundle(bundle, schema=schema, valid_tags=set(seed["candidate_tags"]))
            )
            errors.extend(validate_seed_contract(bundle, seed))
        errors.extend(leakage_errors(bundle))
        if errors:
            rejected.append({"bundle_id": bundle_id, "errors": errors})
        else:
            accepted.append(bundle)
            families[str(seed["target_family"])] += 1
    args.out.mkdir(parents=True, exist_ok=True)
    accepted_path = args.out / "bundles.jsonl"
    rejected_path = args.out / "rejected.jsonl"
    write_jsonl_secure(accepted_path, accepted)
    write_jsonl_secure(rejected_path, rejected)
    manifest = {
        "schema_version": 1,
        "input_bundles": len(accepted) + len(rejected),
        "accepted_bundles": len(accepted),
        "rejected_bundles": len(rejected),
        "accepted_by_family": dict(sorted(families.items())),
        "source_sha256": digest(args.bundles),
        "accepted_sha256": digest(accepted_path),
        "rejected_sha256": digest(rejected_path),
        "synthetic_is_training_only": True,
        "validation_used": False,
        "test_used": False,
    }
    write_json_secure(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
