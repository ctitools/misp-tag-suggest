#!/usr/bin/env python3
"""Audit synthetic bundles against private analyst-training style, locally only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from experiments.lfm25.validate_synthetic import validate_bundle, validate_seed_contract
from tagrec.allowlist import parse_tag
from tagrec.canonicalize import canonical_hash, canonicalize_event
from tagrec.io import iter_jsonl


def generated_canonical(event: dict[str, Any]) -> str:
    raw: dict[str, Any] = {"info": event.get("info", ""), "Attribute": [], "Object": []}
    for attribute in event.get("attributes", []) or []:
        raw["Attribute"].append(dict(attribute))
    for item in event.get("objects", []) or []:
        raw["Object"].append(
            {
                "name": item.get("name", ""),
                "Attribute": [
                    {"category": "Other", **dict(attribute)}
                    for attribute in item.get("attributes", []) or []
                ],
            }
        )
    return canonicalize_event(raw)


def info_text(canonical: str) -> str:
    first = canonical.splitlines()[0] if canonical else ""
    return first.removeprefix("[INFO] ")


def words(value: str) -> int:
    return len(re.findall(r"\w+", value))


def quantiles(values: list[int] | np.ndarray) -> dict[str, float]:
    return {
        str(q): float(np.quantile(np.asarray(values, dtype=np.float64), q))
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def read_real_train(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if not bool(row.get("feed_derived")):
                rows.append(str(row.get("text", "")))
    if len(rows) < 1000:
        raise ValueError("real alignment corpus is unexpectedly small")
    return rows


def deterministic_sample(values: list[str], count: int) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(value.encode()).digest())[:count]


def distinguishability(real_info: list[str], synthetic_info: list[str]) -> dict[str, float]:
    real = deterministic_sample(real_info, len(synthetic_info))
    texts = real + synthetic_info
    labels = np.asarray([0] * len(real) + [1] * len(synthetic_info))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30_000,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(texts)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        features,
        labels,
        cv=folds,
        method="predict_proba",
    )[:, 1]
    return {
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions >= 0.5)),
        "examples_per_class": len(real),
    }


def nearest_lexical(real: list[str], synthetic: list[str]) -> dict[str, float | int]:
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=40_000,
        sublinear_tf=True, strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(real + synthetic)
    real_matrix = matrix[: len(real)]
    synthetic_matrix = matrix[len(real) :]
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbors.fit(real_matrix)
    distances, _ = neighbors.kneighbors(synthetic_matrix)
    similarities = 1.0 - distances[:, 0]
    exact = len(set(map(canonical_hash, real)) & set(map(canonical_hash, synthetic)))
    return {
        "exact_matches": exact,
        "max_cosine": float(similarities.max()),
        "p95_cosine": float(np.quantile(similarities, 0.95)),
        "median_cosine": float(np.median(similarities)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("synthetic_bundle.schema.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    seeds = {str(row["bundle_id"]): row for row in iter_jsonl(args.seeds)}
    bundles = list(iter_jsonl(args.bundles))
    validation_errors: list[str] = []
    synthetic_texts: list[str] = []
    synthetic_info: list[str] = []
    attribute_counts: list[int] = []
    target_leaks = 0
    variants: Counter[str] = Counter()
    for bundle in bundles:
        bundle_id = str(bundle.get("bundle_id", ""))
        seed = seeds.get(bundle_id)
        if seed is None:
            validation_errors.append(f"{bundle_id}: missing frozen seed")
            continue
        for error in validate_bundle(bundle, schema=schema, valid_tags=set(seed["candidate_tags"])):
            validation_errors.append(f"{bundle_id}: {error}")
        for error in validate_seed_contract(bundle, seed):
            validation_errors.append(f"{bundle_id}: {error}")
        parsed = parse_tag(str(bundle.get("target_tag", "")))
        target_value = str(parsed.value if parsed else "").casefold()
        for scenario in bundle.get("scenarios", []) or []:
            event = scenario["event"]
            canonical = generated_canonical(event)
            synthetic_texts.append(canonical)
            synthetic_info.append(info_text(canonical))
            attribute_counts.append(
                len(event.get("attributes", []) or [])
                + sum(len(item.get("attributes", []) or []) for item in event.get("objects", []) or [])
            )
            variant = str(scenario.get("variant", ""))
            variants[variant] += 1
            if (
                variant in {"implicit_positive", "partial_positive"}
                and target_value
                and re.search(r"(?<!\w)" + re.escape(target_value) + r"(?!\w)", canonical.casefold())
            ):
                target_leaks += 1
    real = read_real_train(args.real_train)
    real_info = [info_text(text) for text in real]
    real_info_words = [words(text) for text in real_info]
    synthetic_info_words = [words(text) for text in synthetic_info]
    real_attribute_counts = [sum(line.startswith("[ATTR]") for line in text.splitlines()) for text in real]
    style = {
        "real_info_words": quantiles(real_info_words),
        "synthetic_info_words": quantiles(synthetic_info_words),
        "synthetic_info_within_real_5_95": float(
            np.mean(
                (np.asarray(synthetic_info_words) >= np.quantile(real_info_words, 0.05))
                & (np.asarray(synthetic_info_words) <= np.quantile(real_info_words, 0.95))
            )
        ),
        "real_attribute_count": quantiles(real_attribute_counts),
        "synthetic_attribute_count": quantiles(attribute_counts),
    }
    classifier = distinguishability(real_info, synthetic_info)
    lexical = nearest_lexical(real, synthetic_texts)
    pilot_gates = {
        "all_structurally_valid": not validation_errors,
        "no_exact_implicit_target_leak": target_leaks == 0,
        "info_band_fraction_at_least_0_80": style["synthetic_info_within_real_5_95"] >= 0.80,
        "attribute_median_inside_real_5_75": (
            style["synthetic_attribute_count"]["0.5"] >= style["real_attribute_count"]["0.05"]
            and style["synthetic_attribute_count"]["0.5"] <= style["real_attribute_count"]["0.75"]
        ),
        "no_exact_real_match": lexical["exact_matches"] == 0,
        "no_near_copy_above_0_95": lexical["max_cosine"] < 0.95,
    }
    scale_gates = {
        **pilot_gates,
        # This lexical baseline conflates safe fictionalization with style. It is
        # diagnostic at pilot admission and becomes a hard gate only before scale.
        "synthetic_classifier_auc_at_most_0_90": classifier["roc_auc"] <= 0.90,
    }
    report = {
        "schema_version": 1,
        "bundles": len(bundles),
        "scenarios": len(synthetic_texts),
        "variants": dict(sorted(variants.items())),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors[:50],
        "implicit_or_partial_target_leaks": target_leaks,
        "style": style,
        "synthetic_vs_real_info_classifier": classifier,
        "nearest_real_lexical": lexical,
        "pilot_gates": pilot_gates,
        "scale_gates": scale_gates,
        "pilot_ready": all(pilot_gates.values()),
        "scale_ready": all(scale_gates.values()),
        "passed": all(pilot_gates.values()),
        "real_scope": "analyst-authored training only",
        "validation_used": False,
        "test_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.chmod(0o600)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
