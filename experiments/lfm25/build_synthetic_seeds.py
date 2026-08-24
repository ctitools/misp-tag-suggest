#!/usr/bin/env python3
"""Build a taxonomy-only, privacy-safe synthetic-generation seed bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from tagrec.allowlist import parse_tag
from tagrec.io import iter_jsonl, write_json_secure, write_jsonl_secure

FAMILIES = ("incident-class", "attack-technique", "malware-family", "context")
LANGUAGES = ("en", "en", "en", "de", "fr", "es")
DIFFICULTIES = ("ordinary", "hard", "boundary")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def sanitize_public_document(value: str) -> str:
    """Remove public reference locators that must not look like event indicators."""
    value = _UUID_RE.sub("[public-reference-removed]", value)
    value = _URL_RE.sub("[public-url-removed]", value)
    return " ".join(value.split())


def frequency_bucket(count: int) -> str:
    if count < 20:
        return "under-20"
    if count < 100:
        return "20-99"
    return "100+"


def rank_key(seed: int, *parts: object) -> str:
    value = "\x1f".join(map(str, (seed, *parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def similarity(target: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float, str]:
    same_predicate = float(
        target.get("namespace") == candidate.get("namespace")
        and target.get("predicate") == candidate.get("predicate")
    )
    lexical = SequenceMatcher(
        None,
        str(target.get("value", "")).casefold(),
        str(candidate.get("value", "")).casefold(),
    ).ratio()
    return same_predicate, lexical, str(candidate["tag"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-docs", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/lfm25/synthetic-seeds"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < len(FAMILIES) or args.candidates < 3:
        parser.error("seed count or candidate count is too small")

    tag_rows = list(iter_jsonl(args.tag_docs))
    by_tag = {str(row["tag"]): row for row in tag_rows}
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    frequencies = {
        str(tag): int(count)
        for tag, count in manifest.get(
            "native_train_label_frequencies", manifest.get("train_label_frequencies", {})
        ).items()
    }
    eligible = [
        row
        for row in tag_rows
        if str(row.get("family")) in FAMILIES
        and frequencies.get(str(row["tag"]), 0) >= 20
    ]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_family[str(row["family"])].append(row)

    quotas = {family: args.count // len(FAMILIES) for family in FAMILIES}
    for family in FAMILIES[: args.count % len(FAMILIES)]:
        quotas[family] += 1
    targets: list[dict[str, Any]] = []
    for family in FAMILIES:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_family[family]:
            buckets[frequency_bucket(frequencies[str(row["tag"])])].append(row)
        for values in buckets.values():
            values.sort(key=lambda row: rank_key(args.seed, family, row["tag"]))
        chosen: list[dict[str, Any]] = []
        bucket_order = ("20-99", "100+")
        position = 0
        while len(chosen) < quotas[family]:
            added = False
            for bucket in bucket_order:
                values = buckets.get(bucket, [])
                if position < len(values):
                    chosen.append(values[position])
                    added = True
                    if len(chosen) == quotas[family]:
                        break
            if not added:
                break
            position += 1
        if len(chosen) < quotas[family]:
            raise RuntimeError(f"not enough eligible targets in family {family}")
        targets.extend(chosen)
    targets.sort(key=lambda row: rank_key(args.seed, "target-order", row["tag"]))

    output_rows = []
    family_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for index, target in enumerate(targets):
        target_tag = str(target["tag"])
        family = str(target["family"])
        candidates = [row for row in by_family[family] if row["tag"] != target_tag]
        candidates.sort(key=lambda row: similarity(target, row), reverse=True)
        selected = candidates[: args.candidates - 1]
        candidate_tags = [target_tag] + [str(row["tag"]) for row in selected]
        confuser = candidate_tags[1]
        bundle_id = "syn-" + hashlib.sha256(
            f"{args.seed}\x1f{target_tag}".encode()
        ).hexdigest()[:16]
        documents = {
            tag: sanitize_public_document(str(by_tag[tag]["document"]))
            for tag in candidate_tags
        }
        output_rows.append(
            {
                "schema_version": 1,
                "bundle_id": bundle_id,
                "language": LANGUAGES[index % len(LANGUAGES)],
                "difficulty": DIFFICULTIES[index % len(DIFFICULTIES)],
                "target_tag": target_tag,
                "target_family": family,
                "target_frequency_bucket": frequency_bucket(frequencies[target_tag]),
                "candidate_tags": candidate_tags,
                "candidate_documents": documents,
                "fixed_label_contract": {
                    "explicit_positive": {"expected_tags": [target_tag], "relevance": 2},
                    "implicit_positive": {"expected_tags": [target_tag], "relevance": 2},
                    "partial_positive": {"expected_tags": [target_tag], "relevance": 1},
                    "same_family_confuser": {"expected_tags": [confuser], "relevance": 0},
                    "unrelated_negative": {"expected_tags": [], "relevance": 0},
                    "insufficient_evidence": {"expected_tags": [], "relevance": 0}
                }
            }
        )
        family_counts[family] += 1
        bucket_counts[frequency_bucket(frequencies[target_tag])] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    output_path = args.out / "pilot-seeds.jsonl"
    write_jsonl_secure(output_path, output_rows)
    valid_tags_path = args.out / "valid-tags.json"
    write_json_secure(valid_tags_path, sorted(by_tag))
    output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    seed_manifest = {
        "schema_version": 1,
        "purpose": "taxonomy-grounded synthetic scenario pilot",
        "rows": len(output_rows),
        "by_family": dict(sorted(family_counts.items())),
        "by_frequency_bucket": dict(sorted(bucket_counts.items())),
        "languages": list(LANGUAGES),
        "candidate_count": args.candidates,
        "seed": args.seed,
        "input_policy": "taxonomy definitions and aggregate frequencies only; no event text",
        "labels_fixed_before_generation": True,
        "pilot_seeds_sha256": output_sha,
        "test_used": False,
    }
    write_json_secure(args.out / "manifest.json", seed_manifest)
    print(json.dumps(seed_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
