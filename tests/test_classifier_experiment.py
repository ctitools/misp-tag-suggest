from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from experiments.classifier.benchmark import candidate_metrics
from experiments.classifier.calibrate import evaluate_offsets, optimize_offsets
from experiments.classifier.model import (
    asymmetric_loss,
    prevalence_bias,
    select_top_labels,
    supervised_rows,
)


def test_top_labels_are_native_frequency_ranked_with_stable_ties() -> None:
    rows = [
        {"native_labels": ["tag-b", "tag-a"]},
        {"native_labels": ["tag-c", "tag-b"]},
        {"native_labels": ["tag-c", "excluded"]},
    ]
    labels, counts = select_top_labels(
        rows, limit=3, allowed={"tag-a", "tag-b", "tag-c"}
    )
    assert labels == ["tag-b", "tag-c", "tag-a"]
    assert counts == Counter({"tag-b": 2, "tag-c": 2, "tag-a": 1})


def test_supervised_rows_require_an_in_vocabulary_positive() -> None:
    rows = [
        {"row_id": "a", "native_labels": ["head"]},
        {"row_id": "b", "native_labels": ["tail"]},
        {"row_id": "c", "native_labels": []},
    ]
    assert [row["row_id"] for row in supervised_rows(rows, ["head"])] == ["a"]


def test_prevalence_bias_is_finite_and_frequency_ordered() -> None:
    biases = prevalence_bias(["common", "rare"], Counter(common=50, rare=1), 100)
    assert biases[0] > biases[1]
    assert all(np.isfinite(value) for value in biases)


def test_asymmetric_loss_is_finite_and_rejects_shape_mismatch() -> None:
    logits = torch.tensor([[3.0, -3.0], [-2.0, 2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = asymmetric_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    with pytest.raises(ValueError, match="differ"):
        asymmetric_loss(torch.zeros((1, 2)), torch.zeros((1, 3)))


def test_classifier_candidate_metrics_keep_tail_truth_in_denominator() -> None:
    truth = [{"head", "tail"}, {"other"}]
    predictions = [["head", "wrong"], ["other", "wrong"]]
    result = candidate_metrics(truth, predictions, selected=[True, True], top_k=2)
    assert result["macro_recall"] == pytest.approx(0.75)
    assert result["events_with_truth"] == 2


def test_family_calibration_can_promote_relevant_attack_labels() -> None:
    logits = np.asarray(
        [
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    truth = [{3}, {3}]
    families = ["incident-class", "malware-family", "context", "attack-technique"]
    before = evaluate_offsets(logits, truth, families, {})
    offsets, _ = optimize_offsets(logits, truth, families)
    after = evaluate_offsets(logits, truth, families, offsets)
    assert before["precision@3"] == 0.0
    assert offsets["attack-technique"] > 0
    assert after["precision@3"] == pytest.approx(1 / 3)
