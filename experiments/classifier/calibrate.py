#!/usr/bin/env python3
"""Fit four family logit offsets on a reserved chronological tail of train."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.classifier.model import load_checkpoint
from experiments.lfm25.metrics import FAMILIES
from tagrec.families import label_family
from tagrec.io import iter_jsonl, write_json_secure


def evaluate_offsets(
    logits: np.ndarray,
    truth_indices: list[set[int]],
    label_families: list[str],
    offsets: dict[str, float],
) -> dict[str, Any]:
    bias = np.asarray([offsets.get(family, 0.0) for family in label_families])
    top = np.argpartition(-(logits + bias), kth=2, axis=1)[:, :3]
    precision_rows: list[float] = []
    recall_rows: list[float] = []
    family_rows: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for row_top, truth in zip(top, truth_indices, strict=True):
        chosen = set(map(int, row_top))
        hits = len(chosen & truth)
        precision_rows.append(hits / 3)
        recall_rows.append(hits / len(truth))
        for family in FAMILIES:
            family_truth = {index for index in truth if label_families[index] == family}
            if family_truth:
                family_rows[family].append(len(chosen & family_truth) / len(family_truth))
    family_recall = {
        family: float(np.mean(family_rows[family])) if family_rows[family] else 0.0
        for family in FAMILIES
    }
    precision = float(np.mean(precision_rows))
    return {
        "precision@3": precision,
        "recall@3": float(np.mean(recall_rows)),
        "events_with_truth": len(truth_indices),
        "per_family_recall@3": family_recall,
        "objective": precision + 0.2 * float(np.mean(list(family_recall.values()))),
    }


def optimize_offsets(
    logits: np.ndarray,
    truth_indices: list[set[int]],
    label_families: list[str],
    *,
    maximum_family_regression: float = 0.01,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Coordinate search with an absolute per-family train-calibration guardrail."""
    offsets = {family: 0.0 for family in FAMILIES}
    history: list[dict[str, Any]] = []
    baseline = evaluate_offsets(logits, truth_indices, label_families, offsets)
    family_floor = {
        family: baseline["per_family_recall@3"][family] - maximum_family_regression
        for family in FAMILIES
    }
    grid = np.arange(-4.0, 4.0001, 0.1)
    for round_index in range(3):
        changed = False
        for family in FAMILIES[1:]:
            best_value = offsets[family]
            best_metrics = evaluate_offsets(logits, truth_indices, label_families, offsets)
            best_key = (
                best_metrics["objective"],
                best_metrics["precision@3"],
                -abs(best_value),
            )
            for candidate in grid:
                trial = dict(offsets)
                trial[family] = float(candidate)
                metrics = evaluate_offsets(logits, truth_indices, label_families, trial)
                if any(
                    metrics["per_family_recall@3"][name] < floor
                    for name, floor in family_floor.items()
                ):
                    continue
                key = (metrics["objective"], metrics["precision@3"], -abs(candidate))
                if key > best_key:
                    best_key = key
                    best_value = float(candidate)
                    best_metrics = metrics
            if best_value != offsets[family]:
                changed = True
                offsets[family] = best_value
            history.append(
                {
                    "round": round_index + 1,
                    "family": family,
                    "selected_offset": best_value,
                    "objective": best_metrics["objective"],
                }
            )
        if not changed:
            break
    return offsets, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--maximum-family-regression",
        type=float,
        default=0.0,
        help="maximum absolute train-calibration recall loss allowed per family",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    if args.maximum_family_regression < 0:
        raise ValueError("maximum family regression cannot be negative")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_split_used") is not False:
        raise RuntimeError("input manifest must explicitly exclude test")
    config_path = args.checkpoint / "classifier_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("family_biases"):
        raise RuntimeError("checkpoint is already calibrated")
    calibration_fraction = float(config.get("calibration_fraction", 0.0))
    calibration_start = int(config.get("calibration_start_index", 0))
    if not 0.0 < calibration_fraction < 0.5 or calibration_start < 1:
        raise RuntimeError("checkpoint did not reserve a calibration partition")

    all_rows = list(iter_jsonl(args.data / "train.jsonl.gz"))
    if calibration_start >= len(all_rows):
        raise RuntimeError("reserved calibration partition is empty")
    labels = list(map(str, config["labels"]))
    label_to_index = {label: index for index, label in enumerate(labels)}
    calibration_rows = []
    truth_indices: list[set[int]] = []
    for row in all_rows[calibration_start:]:
        truth = {
            label_to_index[str(label)]
            for label in row.get("native_labels", [])
            if str(label) in label_to_index
        }
        if truth:
            calibration_rows.append(row)
            truth_indices.append(truth)
    if len(calibration_rows) < 100:
        raise RuntimeError("too few calibration events with in-vocabulary truth")

    model, tokenizer, loaded_config = load_checkpoint(args.checkpoint, device=args.device)
    if loaded_config["labels"] != config["labels"]:
        raise RuntimeError("loaded label order differs")
    logits_parts = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(calibration_rows), args.batch_size):
            inputs = tokenizer(
                [str(row["text"]) for row in calibration_rows[start : start + args.batch_size]],
                padding=True,
                truncation=True,
                max_length=int(config["max_length"]),
                return_tensors="pt",
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_parts.append(model(inputs).float().cpu().numpy())
    logits = np.concatenate(logits_parts)
    if not np.isfinite(logits).all():
        raise RuntimeError("non-finite calibration logits")
    label_families = [label_family(label) for label in labels]
    before = evaluate_offsets(
        logits,
        truth_indices,
        label_families,
        {family: 0.0 for family in FAMILIES},
    )
    offsets, history = optimize_offsets(
        logits,
        truth_indices,
        label_families,
        maximum_family_regression=args.maximum_family_regression,
    )
    after = evaluate_offsets(logits, truth_indices, label_families, offsets)
    if after["objective"] < before["objective"] or not math.isfinite(after["objective"]):
        raise RuntimeError("family calibration did not improve its train-only objective")

    backup = args.checkpoint / "classifier_config.uncalibrated.json"
    shutil.copyfile(config_path, backup)
    os.chmod(backup, 0o600)
    config["family_biases"] = offsets
    config["calibration"] = {
        "partition": "chronological tail of train",
        "rows_total": len(all_rows) - calibration_start,
        "rows_with_in_vocabulary_truth": len(calibration_rows),
        "objective": "precision@3 + 0.2 * mean family macro recall@3",
        "validation_used": False,
        "test_used": False,
    }
    write_json_secure(config_path, config)
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "splits_used": ["train-calibration"],
        "validation_used": False,
        "test_used": False,
        "calibration_fraction": calibration_fraction,
        "calibration_start_index": calibration_start,
        "calibration_rows": len(calibration_rows),
        "family_biases": offsets,
        "maximum_absolute_family_regression": args.maximum_family_regression,
        "before": before,
        "after": after,
        "search_history": history,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_secure(args.checkpoint / "calibration_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
