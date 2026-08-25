"""Pure evaluation helpers for nearest-event tag retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from tagrec.families import label_family

FAMILIES = ("incident-class", "attack-technique", "malware-family", "context")


def weighted_vote_predictions(
    similarities: np.ndarray,
    neighbors: np.ndarray,
    train_labels: Sequence[Sequence[str]],
    *,
    neighbor_count: int = 25,
    top_k: int = 3,
) -> list[list[str]]:
    """Convert nearest events into deterministic similarity-weighted tag votes."""
    if similarities.shape != neighbors.shape:
        raise ValueError("similarity and neighbor matrices differ")
    predictions: list[list[str]] = []
    for row_similarities, row_neighbors in zip(similarities, neighbors, strict=True):
        scores: dict[str, float] = defaultdict(float)
        weights = np.maximum(row_similarities[:neighbor_count], 0.0)
        denominator = float(weights.sum()) or 1.0
        for weight, neighbor in zip(
            weights, row_neighbors[:neighbor_count], strict=True
        ):
            if weight <= 0 or neighbor < 0:
                continue
            for label in train_labels[int(neighbor)]:
                scores[str(label)] += float(weight) / denominator
        predictions.append(
            sorted(scores, key=lambda label: (-scores[label], label))[:top_k]
        )
    return predictions


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def ranking_metrics(
    truth: Sequence[set[str]],
    predictions: Sequence[Sequence[str]],
    *,
    selected: Sequence[bool] | None = None,
    top_k: int = 3,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Macro ranking metrics and row values suitable for paired bootstrapping."""
    if len(truth) != len(predictions):
        raise ValueError("truth and prediction lengths differ")
    keep = list(selected) if selected is not None else [True] * len(truth)
    if len(keep) != len(truth):
        raise ValueError("slice selector length differs")

    precision_rows: list[float] = []
    recall_rows: list[float] = []
    hit_rows: list[float] = []
    family_recall: dict[str, list[float]] = defaultdict(list)
    truth_assignments = 0
    hits_total = 0

    for include, row_truth, row_predictions in zip(
        keep, truth, predictions, strict=True
    ):
        if not include or not row_truth:
            continue
        chosen = set(row_predictions[:top_k])
        hits = len(row_truth & chosen)
        precision_rows.append(hits / top_k)
        recall_rows.append(hits / len(row_truth))
        hit_rows.append(float(hits > 0))
        truth_assignments += len(row_truth)
        hits_total += hits
        for family in FAMILIES:
            family_truth = {tag for tag in row_truth if label_family(tag) == family}
            if family_truth:
                family_recall[family].append(len(family_truth & chosen) / len(family_truth))

    metrics = {
        "precision@3": _mean(precision_rows),
        "recall@3": _mean(recall_rows),
        "hit_rate@3": _mean(hit_rows),
        "events_with_truth": len(precision_rows),
        "truth_assignments": truth_assignments,
        "hits": hits_total,
        "per_family_recall@3": {
            family: {
                "macro_recall": _mean(family_recall[family]),
                "events": len(family_recall[family]),
            }
            for family in FAMILIES
        },
    }
    rows = {
        "precision@3": np.asarray(precision_rows, dtype=np.float32),
        "recall@3": np.asarray(recall_rows, dtype=np.float32),
        "hit@3": np.asarray(hit_rows, dtype=np.float32),
        **{
            f"{family}-recall@3": np.asarray(family_recall[family], dtype=np.float32)
            for family in FAMILIES
        },
    }
    return metrics, rows


def candidate_recall_metrics(
    truth: Sequence[set[str]],
    neighbors: np.ndarray,
    train_labels: Sequence[Sequence[str]],
    *,
    selected: Sequence[bool] | None = None,
    neighbor_count: int = 50,
) -> dict[str, Any]:
    """Recall of labels attached to the nearest historical events."""
    keep = list(selected) if selected is not None else [True] * len(truth)
    values: list[float] = []
    family_values: dict[str, list[float]] = defaultdict(list)
    for include, row_truth, row_neighbors in zip(keep, truth, neighbors, strict=True):
        if not include or not row_truth:
            continue
        candidates = {
            str(label)
            for neighbor in row_neighbors[:neighbor_count]
            if neighbor >= 0
            for label in train_labels[int(neighbor)]
        }
        values.append(len(row_truth & candidates) / len(row_truth))
        for family in FAMILIES:
            family_truth = {tag for tag in row_truth if label_family(tag) == family}
            if family_truth:
                family_values[family].append(
                    len(family_truth & candidates) / len(family_truth)
                )
    return {
        "macro_recall": _mean(values),
        "events_with_truth": len(values),
        "per_family": {
            family: {
                "macro_recall": _mean(family_values[family]),
                "events": len(family_values[family]),
            }
            for family in FAMILIES
        },
    }


def paired_bootstrap_delta(
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired row-bootstrap interval for challenger minus baseline."""
    if baseline.shape != challenger.shape:
        raise ValueError("paired bootstrap arrays differ")
    if baseline.ndim != 1 or not len(baseline):
        raise ValueError("paired bootstrap requires non-empty vectors")
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(baseline), size=len(baseline))
        deltas[index] = float(np.mean(challenger[draw] - baseline[draw]))
    point = float(np.mean(challenger - baseline))
    return {
        "delta": point,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "rows": len(baseline),
        "bootstrap_samples": samples,
        "seed": seed,
    }
