#!/usr/bin/env python3
"""Benchmark private BGE multilabel classifiers against frozen BGE retrieval.

Only the prepared train and chronological validation files are opened. Test remains
untouched. The first model is always the pinned BGE nearest-event baseline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.classifier.model import checkpoint_size, load_checkpoint
from experiments.lfm25.benchmark import (
    BGE_MODEL,
    BGE_REVISION,
    ModelSpec,
    evaluate_model,
    search_topk,
)
from experiments.lfm25.metrics import (
    FAMILIES,
    candidate_recall_metrics,
    paired_bootstrap_delta,
    ranking_metrics,
)
from tagrec.families import label_family
from tagrec.io import iter_jsonl, write_json_secure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_metrics(
    truth: list[set[str]],
    predictions: list[list[str]],
    *,
    selected: list[bool],
    top_k: int = 50,
) -> dict[str, Any]:
    values: list[float] = []
    family_values: dict[str, list[float]] = defaultdict(list)
    for include, row_truth, row_predictions in zip(
        selected, truth, predictions, strict=True
    ):
        if not include or not row_truth:
            continue
        candidates = set(row_predictions[:top_k])
        values.append(len(row_truth & candidates) / len(row_truth))
        for family in FAMILIES:
            family_truth = {tag for tag in row_truth if label_family(tag) == family}
            if family_truth:
                family_values[family].append(
                    len(family_truth & candidates) / len(family_truth)
                )
    return {
        "macro_recall": float(np.mean(values)) if values else 0.0,
        "events_with_truth": len(values),
        "per_family": {
            family: {
                "macro_recall": (
                    float(np.mean(family_values[family]))
                    if family_values[family]
                    else 0.0
                ),
                "events": len(family_values[family]),
            }
            for family in FAMILIES
        },
    }


def evaluate_classifier(
    name: str,
    path: Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model, tokenizer, config = load_checkpoint(path, device=device)
    load_seconds = time.perf_counter() - load_started
    labels = list(map(str, config["labels"]))
    if len(labels) < 50:
        raise RuntimeError("classifier must expose at least 50 labels")

    predictions: list[list[str]] = []
    validation_embedding_parts: list[np.ndarray] = []
    score_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(validation_rows), batch_size):
            inputs = tokenizer(
                [str(row["text"]) for row in validation_rows[start : start + batch_size]],
                padding=True,
                truncation=True,
                max_length=int(config["max_length"]),
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.startswith("cuda")
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with context:
                embeddings = model.embeddings(inputs)
                logits = model.classifier(
                    embeddings * model.logit_scale.exp().clamp(max=100.0)
                ) + model.calibration_bias
            validation_embedding_parts.append(embeddings.float().cpu().numpy())
            indices = torch.topk(logits.float(), k=50, dim=1).indices.cpu().tolist()
            predictions.extend([[labels[index] for index in row] for row in indices])
    score_seconds = time.perf_counter() - score_started
    validation_embeddings = np.concatenate(validation_embedding_parts)

    train_embedding_parts: list[np.ndarray] = []
    train_encode_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(train_rows), batch_size):
            inputs = tokenizer(
                [str(row["text"]) for row in train_rows[start : start + batch_size]],
                padding=True,
                truncation=True,
                max_length=int(config["max_length"]),
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.startswith("cuda")
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with context:
                train_embedding_parts.append(
                    model.embeddings(inputs).float().cpu().numpy()
                )
    train_encode_seconds = time.perf_counter() - train_encode_started
    train_embeddings = np.concatenate(train_embedding_parts)
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    )
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    _, neighbors, search_seconds = search_topk(
        train_embeddings,
        validation_embeddings,
        top_k=50,
        device=device,
    )
    del train_embeddings, validation_embeddings
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    train_vocabulary = {
        str(label) for row in train_rows for label in row.get("labels", [])
    }
    truth = [
        {
            str(label)
            for label in row.get("labels", [])
            if str(label) in train_vocabulary
        }
        for row in validation_rows
    ]
    native_truth = [
        {
            str(label)
            for label in row.get("native_labels", [])
            if str(label) in train_vocabulary
        }
        for row in validation_rows
    ]
    train_labels = [list(map(str, row.get("labels", []))) for row in train_rows]
    slices = {
        "all": [True] * len(validation_rows),
        "analyst": [not bool(row.get("feed_derived")) for row in validation_rows],
        "feed": [bool(row.get("feed_derived")) for row in validation_rows],
    }
    cuts: dict[str, Any] = {}
    row_values: dict[str, np.ndarray] = {}
    for truth_name, row_truth in (("all-labels", truth), ("native-only", native_truth)):
        cuts[truth_name] = {}
        for slice_name, selected in slices.items():
            ranking, rows = ranking_metrics(
                row_truth, predictions, selected=selected, top_k=3
            )
            cuts[truth_name][slice_name] = {
                "ranking": ranking,
                "candidate_recall@50": candidate_recall_metrics(
                    row_truth,
                    neighbors,
                    train_labels,
                    selected=selected,
                    neighbor_count=50,
                ),
                "classifier_label_recall@50": candidate_metrics(
                    row_truth, predictions, selected=selected, top_k=50
                ),
            }
            for metric_name, values in rows.items():
                row_values[f"{truth_name}/{slice_name}/{metric_name}"] = values

    result = {
        "name": name,
        "architecture": config["architecture"],
        "base_model": config["base_model"],
        "base_revision": config["base_revision"],
        "encoder_frozen": config["encoder_frozen"],
        "label_count": len(labels),
        "max_sequence_length": int(config["max_length"]),
        "checkpoint_bytes": checkpoint_size(path),
        "classifier": {
            "ranking": "top sigmoid logits",
            "reported_label_count": 50,
            "abstention_applied": False,
        },
        "candidate_retrieval": {
            "encoder": "same classifier encoder; no second model",
            "index": "exact normalized inner product",
            "neighbor_count": 50,
        },
        "timing": {
            "load_seconds": load_seconds,
            "validation_score_seconds": score_seconds,
            "train_encode_seconds": train_encode_seconds,
            "candidate_search_seconds": search_seconds,
            "validation_events_per_second": len(validation_rows) / score_seconds,
            "train_events_per_second": len(train_rows) / train_encode_seconds,
        },
        "peak_gpu_allocated_bytes": peak_gpu_bytes,
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuts": cuts,
    }
    return result, row_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--classifier",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="private classifier checkpoint; repeat as needed",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_split_used") is not False:
        raise RuntimeError("prepared benchmark manifest must explicitly exclude test")
    train_rows = list(iter_jsonl(args.data / "train.jsonl.gz"))
    validation_rows = list(iter_jsonl(args.data / "val.jsonl.gz"))

    baseline_spec = ModelSpec("bge", BGE_MODEL, BGE_REVISION, False, False)
    print("benchmarking bge retrieval baseline", flush=True)
    baseline_result, baseline_rows = evaluate_model(
        baseline_spec,
        train_rows,
        validation_rows,
        batch_size=args.batch_size,
        device=args.device,
    )
    model_results: dict[str, Any] = {"bge": baseline_result}
    rows_by_model: dict[str, dict[str, np.ndarray]] = {"bge": baseline_rows}

    for value in args.classifier:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("--classifier must use NAME=PATH")
        name = name.strip()
        if name in model_results:
            raise ValueError(f"duplicate model name: {name}")
        print(f"benchmarking {name}: {raw_path}", flush=True)
        result, rows = evaluate_classifier(
            name,
            Path(raw_path),
            train_rows,
            validation_rows,
            batch_size=args.batch_size,
            device=args.device,
        )
        model_results[name] = result
        rows_by_model[name] = rows

    comparisons: dict[str, Any] = {}
    for name in list(model_results)[1:]:
        comparison: dict[str, Any] = {}
        for key in (
            "all-labels/all/precision@3",
            "all-labels/analyst/precision@3",
            "native-only/analyst/precision@3",
            "native-only/analyst/recall@3",
            "native-only/analyst/incident-class-recall@3",
            "native-only/analyst/attack-technique-recall@3",
            "native-only/analyst/malware-family-recall@3",
            "native-only/analyst/context-recall@3",
        ):
            comparison[key] = paired_bootstrap_delta(
                baseline_rows[key], rows_by_model[name][key]
            )
        comparisons[f"{name}-minus-bge"] = comparison

    report = {
        "schema_version": 1,
        "benchmark": "supervised multilabel classifier versus nearest-event retrieval",
        "frozen": True,
        "splits_used": ["train", "val"],
        "test_used": False,
        "prepared_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "counts": {
            "train_events": len(train_rows),
            "validation_events": len(validation_rows),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": args.device,
        },
        "models": model_results,
        "paired_comparisons": comparisons,
    }
    write_json_secure(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
