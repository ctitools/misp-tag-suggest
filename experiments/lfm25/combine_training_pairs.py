#!/usr/bin/env python3
"""Combine all real pairs with a declared synthetic training fraction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tagrec.io import iter_jsonl, write_json_secure


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_key(seed: int, row: dict[str, Any], repetition: int) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{row.get('pair_id')}\x1f{repetition}".encode()).digest()


def write_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(
                    (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.synthetic_fraction < 0.5:
        parser.error("synthetic fraction must be between zero and 0.5")
    real = list(iter_jsonl(args.real))
    synthetic_unique = list(iter_jsonl(args.synthetic))
    if not real or not synthetic_unique:
        raise ValueError("real and synthetic pair files must be nonempty")
    target_synthetic = round(len(real) * args.synthetic_fraction / (1 - args.synthetic_fraction))
    synthetic_rows = []
    for index in range(target_synthetic):
        source = synthetic_unique[index % len(synthetic_unique)]
        repetition = index // len(synthetic_unique)
        row = dict(source)
        row["oversample_repetition"] = repetition
        row["sample_id"] = hashlib.sha256(
            f"{source['pair_id']}\x1f{repetition}".encode()
        ).hexdigest()[:24]
        synthetic_rows.append(row)
    combined = [(dict(row), 0) for row in real] + [
        (row, int(row["oversample_repetition"]) + 1) for row in synthetic_rows
    ]
    combined.sort(key=lambda item: order_key(args.seed, item[0], item[1]))
    output_rows = [row for row, _ in combined]
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "train.jsonl.gz"
    write_gzip(output, output_rows)
    actual_fraction = len(synthetic_rows) / len(output_rows)
    manifest = {
        "schema_version": 1,
        "rows": len(output_rows),
        "real_rows": len(real),
        "synthetic_rows": len(synthetic_rows),
        "synthetic_unique_pairs": len(synthetic_unique),
        "synthetic_fraction": actual_fraction,
        "maximum_synthetic_repetitions": max(
            int(row["oversample_repetition"]) for row in synthetic_rows
        )
        + 1,
        "seed": args.seed,
        "real_sha256": digest(args.real),
        "synthetic_sha256": digest(args.synthetic),
        "output_sha256": digest(output),
        "synthetic_is_training_only": True,
        "validation_used": False,
        "test_used": False,
    }
    write_json_secure(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
