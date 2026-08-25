#!/usr/bin/env python3
"""Frozen nearest-event benchmark for BGE, LFM2.5, or a local fine-tune.

Only train and validation are accepted. The untouched test split is never opened.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.lfm25.metrics import (
    candidate_recall_metrics,
    paired_bootstrap_delta,
    ranking_metrics,
    weighted_vote_predictions,
)
from tagrec.io import iter_jsonl, write_json_secure

BGE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
LFM_MODEL = "LiquidAI/LFM2.5-Embedding-350M"
LFM_REVISION = "f35ae2c91d687658dbf1f2b449382f0b019b9808"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    revision: str | None
    trust_remote_code: bool
    asymmetric_prompts: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    requested = [value.strip() for value in args.models.split(",") if value.strip()]
    for name in requested:
        if name == "bge":
            specs.append(ModelSpec("bge", BGE_MODEL, BGE_REVISION, False, False))
        elif name == "lfm":
            specs.append(ModelSpec("lfm-frozen", LFM_MODEL, LFM_REVISION, True, True))
        else:
            raise ValueError(f"unknown built-in model: {name}")
    if args.custom_model:
        specs.append(
            ModelSpec(
                args.custom_name,
                str(args.custom_model),
                None,
                True,
                True,
            )
        )
    for value in args.extra_model:
        name, separator, model_path = value.partition("=")
        if not separator or not name.strip() or not model_path.strip():
            raise ValueError("--extra-model must use NAME=PATH")
        specs.append(ModelSpec(name.strip(), model_path.strip(), None, True, True))
    if not specs:
        raise ValueError("at least one model is required")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("model names must be unique")
    return specs


def search_topk(
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    *,
    top_k: int,
    device: str,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    started = time.perf_counter()
    if device.startswith("cuda"):
        corpus = torch.from_numpy(train_embeddings).to(device)
        similarity_parts: list[np.ndarray] = []
        neighbor_parts: list[np.ndarray] = []
        for start in range(0, len(validation_embeddings), chunk_size):
            query = torch.from_numpy(validation_embeddings[start : start + chunk_size]).to(
                device
            )
            similarities, neighbors = torch.topk(query @ corpus.T, k=top_k, dim=1)
            similarity_parts.append(similarities.float().cpu().numpy())
            neighbor_parts.append(neighbors.int().cpu().numpy())
        del corpus
        similarities = np.concatenate(similarity_parts)
        neighbors = np.concatenate(neighbor_parts)
    else:
        import faiss

        index = faiss.IndexFlatIP(train_embeddings.shape[1])
        index.add(train_embeddings)
        similarities, neighbors = index.search(validation_embeddings, top_k)
    return similarities, neighbors, time.perf_counter() - started


def evaluate_model(
    spec: ModelSpec,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    if spec.trust_remote_code and not transformers.__version__.startswith("4.57."):
        raise RuntimeError(
            f"LFM2.5 benchmark requires audited transformers 4.57.*, got {transformers.__version__}"
        )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = SentenceTransformer(
        spec.model,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        device=device,
    )
    model.max_seq_length = 512
    load_seconds = time.perf_counter() - load_started

    options: dict[str, Any] = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    train_options = dict(options)
    validation_options = dict(options)
    if spec.asymmetric_prompts:
        train_options["prompt_name"] = "document"
        validation_options["prompt_name"] = "query"

    train_started = time.perf_counter()
    train_embeddings = np.asarray(
        model.encode([str(row["text"]) for row in train_rows], **train_options),
        dtype=np.float32,
    )
    train_seconds = time.perf_counter() - train_started
    validation_started = time.perf_counter()
    validation_embeddings = np.asarray(
        model.encode(
            [str(row["text"]) for row in validation_rows], **validation_options
        ),
        dtype=np.float32,
    )
    validation_seconds = time.perf_counter() - validation_started
    embedding_dimensions = int(train_embeddings.shape[1])
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    )
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    similarities, neighbors, search_seconds = search_topk(
        train_embeddings,
        validation_embeddings,
        top_k=50,
        device=device,
    )
    del train_embeddings, validation_embeddings
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    train_labels = [list(map(str, row.get("labels", []))) for row in train_rows]
    vocabulary = {label for labels in train_labels for label in labels}
    truth = [
        {str(label) for label in row.get("labels", []) if str(label) in vocabulary}
        for row in validation_rows
    ]
    native_truth = [
        {
            str(label)
            for label in row.get("native_labels", [])
            if str(label) in vocabulary
        }
        for row in validation_rows
    ]
    slices = {
        "all": [True] * len(validation_rows),
        "analyst": [not bool(row.get("feed_derived")) for row in validation_rows],
        "feed": [bool(row.get("feed_derived")) for row in validation_rows],
    }
    predictions = weighted_vote_predictions(
        similarities, neighbors, train_labels, neighbor_count=25, top_k=3
    )
    cuts: dict[str, Any] = {}
    row_values: dict[str, np.ndarray] = {}
    for truth_name, row_truth in (("all-labels", truth), ("native-only", native_truth)):
        cuts[truth_name] = {}
        for slice_name, selected in slices.items():
            ranking, rows = ranking_metrics(
                row_truth, predictions, selected=selected, top_k=3
            )
            candidates = candidate_recall_metrics(
                row_truth,
                neighbors,
                train_labels,
                selected=selected,
                neighbor_count=50,
            )
            cuts[truth_name][slice_name] = {
                "ranking": ranking,
                "candidate_recall@50": candidates,
            }
            for metric_name, values in rows.items():
                row_values[f"{truth_name}/{slice_name}/{metric_name}"] = values

    result = {
        "name": spec.name,
        "model": spec.model,
        "revision": spec.revision,
        "trust_remote_code": spec.trust_remote_code,
        "asymmetric_prompts": spec.asymmetric_prompts,
        "max_sequence_length": 512,
        "embedding_dimensions": embedding_dimensions,
        "retrieval": {
            "index": "exact normalized inner product",
            "ranking_neighbor_count": 25,
            "candidate_neighbor_count": 50,
            "abstention_applied": False,
        },
        "timing": {
            "load_seconds": load_seconds,
            "train_encode_seconds": train_seconds,
            "validation_encode_seconds": validation_seconds,
            "search_seconds": search_seconds,
            "train_events_per_second": len(train_rows) / train_seconds,
            "validation_events_per_second": len(validation_rows) / validation_seconds,
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
    parser.add_argument("--models", default="bge,lfm")
    parser.add_argument("--custom-model", type=Path)
    parser.add_argument("--custom-name", default="lfm-real-only")
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="additional local asymmetric model; repeat as needed",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    specs = resolve_specs(args)
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_split_used") is not False:
        raise RuntimeError("prepared benchmark manifest must explicitly exclude test")
    train_rows = list(iter_jsonl(args.data / "train.jsonl.gz"))
    validation_rows = list(iter_jsonl(args.data / "val.jsonl.gz"))

    model_results: dict[str, Any] = {}
    rows_by_model: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        print(f"benchmarking {spec.name}: {spec.model}", flush=True)
        result, row_values = evaluate_model(
            spec,
            train_rows,
            validation_rows,
            batch_size=args.batch_size,
            device=args.device,
        )
        model_results[spec.name] = result
        rows_by_model[spec.name] = row_values

    comparisons: dict[str, Any] = {}
    baseline_name = specs[0].name
    for challenger in specs[1:]:
        name = f"{challenger.name}-minus-{baseline_name}"
        comparisons[name] = {}
        for key in (
            "all-labels/all/precision@3",
            "all-labels/analyst/precision@3",
            "native-only/analyst/precision@3",
            "native-only/analyst/recall@3",
        ):
            comparisons[name][key] = paired_bootstrap_delta(
                rows_by_model[baseline_name][key],
                rows_by_model[challenger.name][key],
            )

    report = {
        "schema_version": 1,
        "benchmark": "nearest-event tag voting",
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
