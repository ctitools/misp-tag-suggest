#!/usr/bin/env python3
"""Build capped authoritative real-only retrieval pairs from the training split.

The default corpus uses analyst-authored native event→tag relationships plus a small
partial-event→full-event preservation objective. It never reads validation or test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO

from tagrec.families import label_family
from tagrec.io import iter_jsonl, write_json_secure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: object, seed: int) -> str:
    joined = "\x1f".join(map(str, (*parts, seed)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def partial_text(text: str) -> str:
    lines = text.splitlines()
    info = [line for line in lines if line.startswith("[INFO]")]
    evidence = [line for line in lines if line.startswith("[ATTR]") or line.startswith("[OBJECT]")]
    take = max(1, math.ceil(0.25 * len(evidence))) if evidence else 0
    return "\n".join(info + evidence[:take])


def _secure_gzip(path: Path) -> tuple[BinaryIO, gzip.GzipFile]:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    raw = temporary.open("wb")
    os.chmod(temporary, 0o600)
    return raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)


def write_row(compressed: gzip.GzipFile, value: dict[str, Any]) -> None:
    compressed.write(
        (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--tag-docs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/lfm25/real-pairs"))
    parser.add_argument("--per-tag-cap", type=int, default=40)
    parser.add_argument("--partial-pairs", type=int, default=6000)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_tag_cap < 1 or args.partial_pairs < 0 or args.max_chars < 4096:
        parser.error("invalid pair limits")

    tag_rows = list(iter_jsonl(args.tag_docs))
    document_by_tag = {str(row["tag"]): str(row["document"]) for row in tag_rows}
    label_rows: dict[str, list[int]] = defaultdict(list)
    partial_candidates: list[tuple[str, int]] = []
    analyst_rows = native_assignments = 0

    for index, row in enumerate(iter_jsonl(args.train)):
        if row.get("feed_derived"):
            continue
        analyst_rows += 1
        sources = row.get("label_sources", {}) or {}
        for label in row.get("labels", []) or []:
            tag = str(label)
            if sources.get(tag, "native") == "native" and tag in document_by_tag:
                label_rows[tag].append(index)
                native_assignments += 1
        reduced = str(row.get("text", ""))[: args.max_chars]
        partial = partial_text(reduced)
        if partial and partial != reduced:
            partial_candidates.append((stable_rank("partial", index, seed=args.seed), index))

    selected_by_row: dict[int, list[str]] = defaultdict(list)
    selected_per_tag: dict[str, int] = {}
    for tag, indices in sorted(label_rows.items()):
        ordered = sorted(indices, key=lambda index: stable_rank(tag, index, seed=args.seed))
        chosen = ordered[: args.per_tag_cap]
        selected_per_tag[tag] = len(chosen)
        for index in chosen:
            selected_by_row[index].append(tag)
    selected_partial = {index for _rank, index in sorted(partial_candidates)[: args.partial_pairs]}
    needed_rows = set(selected_by_row) | selected_partial

    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    destination = args.out / "train.jsonl.gz"
    raw, compressed = _secure_gzip(destination)
    event_tag_count = partial_count = leakage_count = 0
    family_counts: Counter[str] = Counter()
    try:
        for index, row in enumerate(iter_jsonl(args.train)):
            if index not in needed_rows:
                continue
            text = str(row.get("text", ""))[: args.max_chars]
            for tag in sorted(selected_by_row.get(index, [])):
                if tag.casefold() in text.casefold():
                    leakage_count += 1
                    continue
                pair_id = hashlib.sha256(
                    f"event-tag\x1f{index}\x1f{tag}\x1f{args.seed}".encode()
                ).hexdigest()[:24]
                write_row(
                    compressed,
                    {
                        "pair_id": pair_id,
                        "group_id": f"train-row-{index:06d}",
                        "pair_type": "event_to_tag",
                        "target_tag": tag,
                        "query": text,
                        "positive": document_by_tag[tag],
                    },
                )
                event_tag_count += 1
                family_counts[label_family(tag)] += 1
            if index in selected_partial:
                partial = partial_text(text)
                if partial and partial != text:
                    pair_id = hashlib.sha256(
                        f"partial-full\x1f{index}\x1f{args.seed}".encode()
                    ).hexdigest()[:24]
                    write_row(
                        compressed,
                        {
                            "pair_id": pair_id,
                            "group_id": f"train-row-{index:06d}",
                            "pair_type": "partial_to_full_event",
                            "target_tag": None,
                            "query": partial,
                            "positive": text,
                        },
                    )
                    partial_count += 1
        compressed.close()
        raw.close()
        temporary = Path(raw.name)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        try:
            compressed.close()
        finally:
            raw.close()
            Path(raw.name).unlink(missing_ok=True)
        raise

    if leakage_count:
        raise RuntimeError(f"refused {leakage_count} event/tag pairs with exact target leakage")
    pair_count = event_tag_count + partial_count
    if pair_count == 0:
        raise RuntimeError("no training pairs were produced")
    manifest = {
        "schema_version": 1,
        "purpose": "real-only LFM2.5 retrieval fine-tuning",
        "source_train_sha256": sha256(args.train),
        "source_manifest_sha256": sha256(args.source_manifest),
        "tag_documents_sha256": sha256(args.tag_docs),
        "splits_read": ["train"],
        "validation_used": False,
        "test_used": False,
        "selection": {
            "analyst_only": True,
            "native_assignments_only": True,
            "per_tag_cap": args.per_tag_cap,
            "partial_pair_cap": args.partial_pairs,
            "seed": args.seed,
            "max_characters": args.max_chars,
        },
        "source_counts": {
            "analyst_rows": analyst_rows,
            "native_assignments_with_tag_document": native_assignments,
            "eligible_target_tags": len(label_rows),
        },
        "pair_counts": {
            "total": pair_count,
            "event_to_tag": event_tag_count,
            "partial_to_full_event": partial_count,
            "by_family": dict(sorted(family_counts.items())),
            "target_tags": len(selected_per_tag),
        },
        "output": {
            "path": destination.name,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        },
        "privacy": "pseudonymous groups; no UUID, event ID, or organization field",
        "sensitivity": "private canonical event text; mode 0600; do not publish",
    }
    write_json_secure(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
