#!/usr/bin/env python3
"""Select a balanced canary and gap-targeted pilot from the frozen seed bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tagrec.io import iter_jsonl, write_json_secure, write_jsonl_secure

CANARY_QUOTAS = {
    "incident-class": 5,
    "attack-technique": 5,
    "malware-family": 5,
    "context": 5,
}
PILOT_QUOTAS = {
    # Use every remaining incident seed because M2 had a small incident candidate
    # recall regression. Keep the other families as explicit regression guards.
    "incident-class": 45,
    "attack-technique": 20,
    "malware-family": 20,
    "context": 15,
}


def rank(seed: int, stage: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{stage}\x1f{value}".encode()).hexdigest()


def choose(
    rows: list[dict[str, Any]], quotas: dict[str, int], *, seed: int, stage: str
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["target_family"])].append(row)
    result = []
    for family, quota in quotas.items():
        candidates = sorted(
            by_family[family], key=lambda row: rank(seed, stage, str(row["bundle_id"]))
        )
        if len(candidates) < quota:
            raise ValueError(f"not enough {family} seeds for {stage}")
        result.extend(candidates[:quota])
    return sorted(result, key=lambda row: rank(seed, stage + "-order", str(row["bundle_id"])))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "family": dict(sorted(Counter(str(row["target_family"]) for row in rows).items())),
        "language": dict(sorted(Counter(str(row["language"]) for row in rows).items())),
        "difficulty": dict(sorted(Counter(str(row["difficulty"]) for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=56)
    args = parser.parse_args()
    rows = list(iter_jsonl(args.input))
    canary = choose(rows, CANARY_QUOTAS, seed=args.seed, stage="canary")
    canary_ids = {str(row["bundle_id"]) for row in canary}
    pilot = choose(
        [row for row in rows if str(row["bundle_id"]) not in canary_ids],
        PILOT_QUOTAS,
        seed=args.seed,
        stage="pilot",
    )
    pilot_ids = {str(row["bundle_id"]) for row in pilot}
    if canary_ids & pilot_ids:
        raise AssertionError("canary and pilot overlap")
    args.out.mkdir(parents=True, exist_ok=True)
    canary_path = args.out / "canary-seeds.jsonl"
    pilot_path = args.out / "pilot-seeds.jsonl"
    write_jsonl_secure(canary_path, canary)
    write_jsonl_secure(pilot_path, pilot)
    manifest = {
        "schema_version": 1,
        "selection_seed": args.seed,
        "source_sha256": digest(args.input),
        "canary": {"rows": len(canary), "sha256": digest(canary_path), **summarize(canary)},
        "pilot": {
            "rows": len(pilot),
            "sha256": digest(pilot_path),
            "targeting": "incident candidate-recall gap with all-family regression guards",
            **summarize(pilot),
        },
        "overlap": 0,
        "test_used": False,
    }
    write_json_secure(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
