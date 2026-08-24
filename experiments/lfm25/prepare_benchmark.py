#!/usr/bin/env python3
"""Prepare a minimized, pseudonymous train/validation corpus for GPU benchmarking.

The output deliberately excludes event IDs, UUIDs, organization names, comments outside
canonical text, and the untouched test split. Canonical text remains sensitive and must
only be transferred to an operator-authorized access-controlled compute host.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

from tagrec.io import iter_jsonl, write_json_secure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _secure_gzip(path: Path) -> tuple[BinaryIO, gzip.GzipFile]:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    raw = path.with_suffix(path.suffix + ".tmp").open("wb")
    os.chmod(raw.name, 0o600)
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, compressed


def prepare_split(source: Path, destination: Path, *, max_chars: int) -> dict[str, int]:
    raw, compressed = _secure_gzip(destination)
    rows = labeled = truncated = canonical_chars = written_chars = 0
    try:
        for index, row in enumerate(iter_jsonl(source)):
            text = str(row.get("text", ""))
            reduced = text[:max_chars]
            native_labels = sorted(
                str(label)
                for label in row.get("labels", []) or []
                if (row.get("label_sources", {}) or {}).get(str(label), "native")
                != "freetext-mapped"
            )
            value: dict[str, Any] = {
                "row_id": f"{source.stem}-{index:06d}",
                "text": reduced,
                "labels": sorted(map(str, row.get("labels", []) or [])),
                "native_labels": native_labels,
                "feed_derived": bool(row.get("feed_derived")),
            }
            encoded = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            compressed.write(encoded)
            rows += 1
            labeled += bool(value["labels"])
            truncated += len(reduced) < len(text)
            canonical_chars += len(text)
            written_chars += len(reduced)
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
    return {
        "rows": rows,
        "labeled_rows": labeled,
        "truncated_rows": truncated,
        "canonical_characters": canonical_chars,
        "written_characters": written_chars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/lfm25/remote-data"))
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12_000,
        help="Prefix cap safely above both encoders' frozen 512-token limit",
    )
    args = parser.parse_args()
    if args.max_chars < 4096:
        parser.error("--max-chars must be at least 4096")

    source_manifest = args.source_data / "manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    counts = {}
    outputs = {}
    for split in ("train", "val"):
        destination = args.out / f"{split}.jsonl.gz"
        counts[split] = prepare_split(
            args.source_data / f"{split}.jsonl", destination, max_chars=args.max_chars
        )
        outputs[split] = {
            "path": destination.name,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }

    prepared_manifest = {
        "schema_version": 1,
        "purpose": "frozen LFM2.5 validation benchmark",
        "source_manifest_sha256": sha256(source_manifest),
        "source_schema_version": manifest.get("schema_version"),
        "taxonomy_snapshot_date": manifest.get("taxonomy_snapshot_date"),
        "splits_included": ["train", "val"],
        "test_split_used": False,
        "fields": ["row_id", "text", "labels", "native_labels", "feed_derived"],
        "excluded_fields": ["event_id", "uuid", "orgc"],
        "max_characters": args.max_chars,
        "encoder_max_tokens": 512,
        "counts": counts,
        "outputs": outputs,
        "sensitivity": "private canonical event text; mode 0600; do not publish",
    }
    write_json_secure(args.out / "manifest.json", prepared_manifest)
    print(json.dumps(prepared_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
