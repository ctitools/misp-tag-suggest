#!/usr/bin/env python3
"""Build and tune full-corpus semantic retrieval using train/validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tagrec.families import label_family
from tagrec.io import iter_jsonl, write_json_secure, write_jsonl_secure

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
_LONG_HEX_RE = re.compile(r"[0-9a-fA-F]{8,}")
_DIGITS_RE = re.compile(r"\d+")


def normalized_info(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    return _DIGITS_RE.sub("<N>", _LONG_HEX_RE.sub("<H>", first))


def save_npz_secure(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def vote_predictions(
    similarities: np.ndarray,
    neighbors: np.ndarray,
    train_rows: list[dict[str, Any]],
    k: int,
    top_n: int = 3,
) -> list[list[str]]:
    output: list[list[str]] = []
    for row_similarities, row_neighbors in zip(similarities[:, :k], neighbors[:, :k], strict=True):
        weights = np.maximum(row_similarities, 0.0)
        denominator = float(weights.sum()) or 1.0
        scores: dict[str, float] = defaultdict(float)
        for neighbor, weight in zip(row_neighbors, weights, strict=True):
            if neighbor < 0 or weight <= 0:
                continue
            for label in train_rows[int(neighbor)].get("labels", []):
                scores[str(label)] += float(weight) / denominator
        output.append(sorted(scores, key=lambda label: (-scores[label], label))[:top_n])
    return output


def operational_metrics(
    truth: list[set[str]],
    predictions: list[list[str]],
    max_similarity: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    precisions: list[float] = []
    recalls: list[float] = []
    covered = 0
    for index, row_truth in enumerate(truth):
        selected = predictions[index] if max_similarity[index] >= threshold else []
        covered += bool(selected)
        if row_truth:
            hits = len(row_truth & set(selected))
            precisions.append(hits / 3)
            recalls.append(hits / len(row_truth))
    return {
        "precision@3": float(np.mean(precisions)) if precisions else 0.0,
        "recall@3": float(np.mean(recalls)) if recalls else 0.0,
        "coverage_all_events": covered / len(truth) if truth else 0.0,
        "events_with_truth": len(precisions),
    }


def candidate_recall(
    truth: list[set[str]],
    neighbors: np.ndarray,
    train_rows: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    values: list[float] = []
    family_values: dict[str, list[float]] = defaultdict(list)
    for index, row_truth in enumerate(truth):
        if not row_truth:
            continue
        candidates = {
            str(label)
            for neighbor in neighbors[index, :k]
            if neighbor >= 0
            for label in train_rows[int(neighbor)].get("labels", [])
        }
        values.append(len(row_truth & candidates) / len(row_truth))
        for family in ("incident-class", "attack-technique", "malware-family", "context"):
            family_truth = {label for label in row_truth if label_family(label) == family}
            if family_truth:
                family_values[family].append(len(family_truth & candidates) / len(family_truth))
    return {
        "overall": float(np.mean(values)) if values else 0.0,
        "per_family": {
            family: float(np.mean(items)) if items else 0.0
            for family, items in sorted(family_values.items())
        },
        "events_with_truth": len(values),
    }


def main() -> int:
    import faiss
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--artifacts", default="artifacts/retrieval")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k-values", default="10,25,50")
    parser.add_argument("--target-precision", type=float, default=0.8)
    parser.add_argument("--device")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    os.chmod(artifacts, 0o700)
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    train_rows = list(iter_jsonl(data_dir / "train.jsonl"))
    validation_rows = list(iter_jsonl(data_dir / "val.jsonl"))
    vocabulary = {str(label) for row in train_rows for label in row.get("labels", [])}
    truth = [
        {str(label) for label in row.get("labels", []) if str(label) in vocabulary}
        for row in validation_rows
    ]

    model = SentenceTransformer(
        args.model, revision=args.revision, trust_remote_code=False, device=args.device
    )
    options = {
        "batch_size": args.batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    train_embeddings = np.asarray(
        model.encode([str(row.get("text", "")) for row in train_rows], **options),
        dtype=np.float32,
    )
    validation_embeddings = np.asarray(
        model.encode([str(row.get("text", "")) for row in validation_rows], **options),
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(train_embeddings.shape[1])
    index.add(train_embeddings)
    k_values = sorted({min(int(value), len(train_rows)) for value in args.k_values.split(",")})
    similarities, neighbors = index.search(validation_embeddings, max(k_values))
    max_similarity = similarities[:, 0]
    threshold_values = sorted(
        set(
            [0.0]
            + [float(value) for value in np.quantile(max_similarity, [0.1, 0.25, 0.5, 0.75, 0.9])]
        )
    )
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for k in k_values:
        predictions = vote_predictions(similarities, neighbors, train_rows, k)
        for threshold in threshold_values:
            candidates.append(
                (k, threshold, operational_metrics(truth, predictions, max_similarity, threshold))
            )
    feasible = [value for value in candidates if value[2]["precision@3"] >= args.target_precision]
    selected_k, selected_threshold, selected_metrics = max(
        feasible or candidates,
        key=lambda value: (
            value[2]["recall@3"] if feasible else value[2]["precision@3"],
            value[2]["coverage_all_events"],
            -value[0],
        ),
    )

    save_npz_secure(artifacts / "train_embeddings.npz", train=train_embeddings)
    write_jsonl_secure(
        artifacts / "index_meta.jsonl",
        (
            {
                "event_id": row.get("event_id"),
                "uuid": row.get("uuid"),
                "date": row.get("date"),
                "feed_derived": bool(row.get("feed_derived")),
                "labels": row.get("labels", []),
                "title": str(row.get("text", "")).splitlines()[0].removeprefix("[INFO] "),
                "orgc": str(row.get("orgc", "")),
                "normalized_info": normalized_info(str(row.get("text", ""))),
                "dup_cluster": str(row.get("dup_cluster", "")),
            }
            for row in train_rows
        ),
    )
    validation_candidate_recall = {
        f"candidate_recall@{k}": candidate_recall(truth, neighbors, train_rows, k)
        for k in k_values
    }
    params = {
        "schema_version": 1,
        "frozen": True,
        "model": args.model,
        "revision": args.revision,
        "selected_k": selected_k,
        "similarity_threshold": selected_threshold,
        "thresholds": {"global": 0.0, "per_family": {}, "per_label": {}},
        "normalization": "L2",
        "index": "exact inner product rebuilt from NPZ",
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "taxonomy_snapshot_date": manifest.get("taxonomy_snapshot_date"),
        "serialization": "JSON, JSONL and NPZ; allow_pickle=False",
    }
    write_json_secure(artifacts / "params.json", params)
    write_json_secure(
        artifacts / "training_metrics.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split_used": ["train", "val"],
            "test_set_used": False,
            "selected_validation_operational": selected_metrics,
            "candidate_recall_validation": validation_candidate_recall,
            "counts": {
                "train_events": len(train_rows),
                "validation_events": len(validation_rows),
                "train_labels": len(vocabulary),
                "embedding_dimensions": int(train_embeddings.shape[1]),
            },
        },
    )
    print(
        json.dumps(
            {
                "selected_k": selected_k,
                "similarity_threshold": selected_threshold,
                "validation": selected_metrics,
                "candidate_recall": validation_candidate_recall,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
