#!/usr/bin/env python3
"""Build admitted synthetic contrastive pairs without synthetic evaluation rows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.lfm25.audit_synthetic_alignment import generated_canonical
from tagrec.io import iter_jsonl, write_json_secure


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_id(bundle_id: str, variant: str, kind: str) -> str:
    return hashlib.sha256(f"{bundle_id}\x1f{variant}\x1f{kind}".encode()).hexdigest()[:24]


def write_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = {str(row["bundle_id"]): row for row in iter_jsonl(args.seeds)}
    pairs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for bundle in iter_jsonl(args.bundles):
        bundle_id = str(bundle["bundle_id"])
        seed = seeds[bundle_id]
        scenarios = {str(row["variant"]): row for row in bundle["scenarios"]}
        for variant in ("explicit_positive", "implicit_positive", "partial_positive", "same_family_confuser"):
            scenario = scenarios[variant]
            expected = list(scenario["expected_tags"])
            if len(expected) != 1:
                raise ValueError(f"{bundle_id}/{variant}: expected one positive tag")
            tag = str(expected[0])
            document = str(seed["candidate_documents"][tag])
            pairs.append(
                {
                    "pair_id": pair_id(bundle_id, variant, "event-tag"),
                    "query": generated_canonical(scenario["event"]),
                    "positive": document,
                    "pair_type": "synthetic-event-tag",
                    "target_tag": tag,
                    "target_family": seed["target_family"],
                    "bundle_id": bundle_id,
                    "variant": variant,
                    "synthetic": True,
                }
            )
            counts["synthetic-event-tag"] += 1
        explicit_text = generated_canonical(scenarios["explicit_positive"]["event"])
        for variant in ("implicit_positive", "partial_positive"):
            pairs.append(
                {
                    "pair_id": pair_id(bundle_id, variant, "preservation"),
                    "query": generated_canonical(scenarios[variant]["event"]),
                    "positive": explicit_text,
                    "pair_type": "synthetic-partial-full",
                    "target_tag": bundle["target_tag"],
                    "target_family": seed["target_family"],
                    "bundle_id": bundle_id,
                    "variant": variant,
                    "synthetic": True,
                }
            )
            counts["synthetic-partial-full"] += 1
    if len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError("duplicate synthetic pair IDs")
    args.out.mkdir(parents=True, exist_ok=True)
    pair_path = args.out / "train.jsonl.gz"
    write_gzip(pair_path, pairs)
    manifest = {
        "schema_version": 1,
        "rows": len(pairs),
        "unique_pairs": len(pairs),
        "by_type": dict(sorted(counts.items())),
        "bundles_sha256": digest(args.bundles),
        "pairs_sha256": digest(pair_path),
        "synthetic_is_training_only": True,
        "validation_used": False,
        "test_used": False,
    }
    write_json_secure(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
