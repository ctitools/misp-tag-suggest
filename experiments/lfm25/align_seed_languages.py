#!/usr/bin/env python3
"""Align synthetic seed language mix to coarse analyst-training observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tagrec.io import iter_jsonl, write_json_secure, write_jsonl_secure


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spanish-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.spanish_fraction <= 0.2:
        parser.error("Spanish fraction must be in 0..0.2")
    rows = list(iter_jsonl(args.input))
    spanish = round(len(rows) * args.spanish_fraction)
    ordered = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(str(rows[index]["bundle_id"]).encode()).digest(),
    )
    spanish_indexes = set(ordered[:spanish])
    for index, row in enumerate(rows):
        row["language"] = "es" if index in spanish_indexes else "en"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_secure(args.out, rows)
    manifest = {
        "schema_version": 1,
        "rows": len(rows),
        "languages": dict(sorted(Counter(str(row["language"]) for row in rows).items())),
        "source_sha256": digest(args.input),
        "output_sha256": digest(args.out),
        "policy": "coarse language alignment; no event text used",
        "validation_used": False,
        "test_used": False,
    }
    write_json_secure(args.out.with_name(args.out.stem + "-manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
