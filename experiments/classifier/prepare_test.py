#!/usr/bin/env python3
"""Prepare the untouched test split once after all adoption gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from experiments.lfm25.prepare_benchmark import prepare_split
from tagrec.io import write_json_secure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_gate(
    adoption: dict,
    bge_cpu: dict,
    classifier_cpu: dict,
    *,
    model_name: str,
) -> dict:
    comparison_name = f"{model_name}-minus-bge"
    comparison = adoption["paired_comparisons"].get(comparison_name)
    if comparison is None:
        raise RuntimeError(f"missing adoption comparison: {comparison_name}")
    precision = comparison["native-only/analyst/precision@3"]
    if precision["delta"] < 0.03 or precision["ci95_low"] <= 0:
        raise RuntimeError("analyst precision adoption gate did not pass")
    for family in (
        "incident-class",
        "attack-technique",
        "malware-family",
        "context",
    ):
        result = comparison[f"native-only/analyst/{family}-recall@3"]
        if result["ci95_high"] < 0:
            raise RuntimeError(f"significant family regression: {family}")
    models = adoption["models"]
    baseline_candidates = models["bge"]["cuts"]["native-only"]["analyst"][
        "candidate_recall@50"
    ]["macro_recall"]
    challenger_candidates = models[model_name]["cuts"]["native-only"]["analyst"][
        "candidate_recall@50"
    ]["macro_recall"]
    if challenger_candidates < baseline_candidates:
        raise RuntimeError("candidate recall adoption gate did not pass")
    if bge_cpu.get("test_used") is not False or classifier_cpu.get("test_used") is not False:
        raise RuntimeError("CPU gate reports must not use test")
    if classifier_cpu["p95_milliseconds"] > bge_cpu["p95_milliseconds"] * 1.2:
        raise RuntimeError("classifier CPU p95 exceeds the 20% deployment allowance")
    if classifier_cpu["process_max_rss_kib"] > bge_cpu["process_max_rss_kib"] * 1.2:
        raise RuntimeError("classifier CPU RSS exceeds the 20% deployment allowance")
    return {
        "precision_delta": precision["delta"],
        "precision_ci95_low": precision["ci95_low"],
        "candidate_delta": challenger_candidates - baseline_candidates,
        "classifier_to_bge_cpu_p95_ratio": (
            classifier_cpu["p95_milliseconds"] / bge_cpu["p95_milliseconds"]
        ),
        "classifier_to_bge_rss_ratio": (
            classifier_cpu["process_max_rss_kib"]
            / bge_cpu["process_max_rss_kib"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--prepared-train", type=Path, required=True)
    parser.add_argument("--adoption-report", type=Path, required=True)
    parser.add_argument("--bge-cpu-report", type=Path, required=True)
    parser.add_argument("--classifier-cpu-report", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=12_000)
    args = parser.parse_args()

    if args.out.exists():
        raise RuntimeError("test preparation output already exists; refusing a second read")
    source_manifest_path = args.source_data / "manifest.json"
    prepared_manifest_path = args.prepared_train / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if prepared_manifest.get("test_split_used") is not False:
        raise RuntimeError("training preparation unexpectedly used test")
    if prepared_manifest.get("source_manifest_sha256") != sha256(source_manifest_path):
        raise RuntimeError("prepared train does not match the source dataset manifest")
    adoption = json.loads(args.adoption_report.read_text(encoding="utf-8"))
    if adoption.get("test_used") is not False:
        raise RuntimeError("adoption report must precede test access")
    gate = verify_gate(
        adoption,
        json.loads(args.bge_cpu_report.read_text(encoding="utf-8")),
        json.loads(args.classifier_cpu_report.read_text(encoding="utf-8")),
        model_name=args.model_name,
    )

    args.out.mkdir(parents=True, mode=0o700)
    os.chmod(args.out, 0o700)
    train_destination = args.out / "train.jsonl.gz"
    try:
        os.link(args.prepared_train / "train.jsonl.gz", train_destination)
    except OSError:
        shutil.copyfile(args.prepared_train / "train.jsonl.gz", train_destination)
    os.chmod(train_destination, 0o600)
    test_destination = args.out / "test.jsonl.gz"
    counts = prepare_split(
        args.source_data / "test.jsonl",
        test_destination,
        max_chars=args.max_chars,
    )
    manifest = {
        "schema_version": 1,
        "purpose": "one-time classifier adoption test",
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_schema_version": source_manifest.get("schema_version"),
        "splits_included": ["train", "test"],
        "test_split_used": True,
        "model_name": args.model_name,
        "adoption_report_sha256": sha256(args.adoption_report),
        "adoption_gate": gate,
        "fields": ["row_id", "text", "labels", "native_labels", "feed_derived"],
        "excluded_fields": ["event_id", "uuid", "orgc"],
        "max_characters": args.max_chars,
        "encoder_max_tokens": 512,
        "counts": {"train": prepared_manifest["counts"]["train"], "test": counts},
        "outputs": {
            "train": {"path": "train.jsonl.gz", "sha256": sha256(train_destination)},
            "test": {"path": "test.jsonl.gz", "sha256": sha256(test_destination)},
        },
        "sensitivity": "private canonical event text; mode 0600; do not publish",
    }
    write_json_secure(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
