"""Pinned BGE multilabel-classifier model and pure training helpers."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

BGE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def select_top_labels(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int,
    allowed: set[str] | None = None,
) -> tuple[list[str], Counter[str]]:
    """Select frequent native labels with deterministic lexical tie-breaking."""
    if limit < 1:
        raise ValueError("label limit must be positive")
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(
            str(label)
            for label in row.get("native_labels", [])
            if allowed is None or str(label) in allowed
        )
    labels = [
        label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
    if not labels:
        raise ValueError("no native training labels are available")
    return labels, counts


def supervised_rows(rows: Iterable[dict[str, Any]], labels: Sequence[str]) -> list[dict[str, Any]]:
    """Keep rows with at least one in-vocabulary native positive."""
    vocabulary = set(labels)
    return [
        row
        for row in rows
        if any(str(label) in vocabulary for label in row.get("native_labels", []))
    ]


def prevalence_bias(labels: Sequence[str], counts: Counter[str], row_count: int) -> list[float]:
    """Return finite prior logits for classifier-head bias initialization."""
    if row_count < 1:
        raise ValueError("row count must be positive")
    result = []
    for label in labels:
        probability = min(max(counts[label] / row_count, 1e-4), 1 - 1e-4)
        result.append(max(-8.0, min(8.0, math.log(probability / (1 - probability)))))
    return result


def asymmetric_loss(
    logits: "Any",
    targets: "Any",
    *,
    gamma_negative: float = 4.0,
    gamma_positive: float = 0.0,
    negative_clip: float = 0.05,
) -> "Any":
    """Asymmetric multilabel loss, normalized per event rather than per class."""
    import torch

    if logits.shape != targets.shape:
        raise ValueError("logit and target tensors differ")
    probabilities = torch.sigmoid(logits)
    positive = probabilities
    negative = 1.0 - probabilities
    if negative_clip > 0:
        negative = (negative + negative_clip).clamp(max=1.0)
    loss = targets * torch.log(positive.clamp(min=1e-8))
    loss += (1.0 - targets) * torch.log(negative.clamp(min=1e-8))
    if gamma_negative > 0 or gamma_positive > 0:
        with torch.no_grad():
            positive_selected = positive * targets
            negative_selected = negative * (1.0 - targets)
            gamma = gamma_positive * targets + gamma_negative * (1.0 - targets)
            weight = torch.pow(1.0 - positive_selected - negative_selected, gamma)
        loss *= weight
    return -loss.sum() / logits.shape[0]


def load_tag_documents(path: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            tag = str(row["tag"])
            document = str(row["document"])
            if tag in documents:
                raise ValueError(f"duplicate tag document: {tag}")
            documents[tag] = document
    return documents


def checkpoint_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_model(
    *,
    label_count: int,
    encoder_source: str | Path = BGE_MODEL,
    revision: str | None = BGE_REVISION,
    device: str | None = None,
) -> "Any":
    """Construct the classifier without importing PyTorch at module import time."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel

    class BgeMultilabelClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                str(encoder_source),
                revision=revision,
                trust_remote_code=False,
            )
            hidden_size = int(self.encoder.config.hidden_size)
            self.classifier = torch.nn.Linear(hidden_size, label_count)
            self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(5.0)))
            self.register_buffer("calibration_bias", torch.zeros(label_count), persistent=False)

        def embeddings(self, inputs: dict[str, Any]) -> Any:
            output = self.encoder(**inputs)
            return functional.normalize(output.last_hidden_state[:, 0], p=2, dim=1)

        def forward(self, inputs: dict[str, Any]) -> Any:
            scale = self.logit_scale.exp().clamp(max=100.0)
            return self.classifier(self.embeddings(inputs) * scale) + self.calibration_bias

    model = BgeMultilabelClassifier()
    if device is not None:
        model.to(device)
    return model


def load_checkpoint(path: Path, *, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load a private classifier checkpoint and its tokenizer."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    config = json.loads((path / "classifier_config.json").read_text(encoding="utf-8"))
    local_encoder = path / "encoder"
    encoder_source: str | Path = local_encoder if local_encoder.exists() else config["base_model"]
    revision = None if local_encoder.exists() else config["base_revision"]
    tokenizer_source: str | Path = (
        path / "tokenizer" if (path / "tokenizer").exists() else encoder_source
    )
    model = build_model(
        label_count=len(config["labels"]),
        encoder_source=encoder_source,
        revision=revision,
        device=device,
    )
    state = load_file(str(path / "classifier_head.safetensors"), device=device)
    model.classifier.load_state_dict(
        {"weight": state["classifier.weight"], "bias": state["classifier.bias"]}
    )
    model.logit_scale.data.copy_(state["logit_scale"].to(device))
    family_biases = config.get("family_biases", {})
    if family_biases:
        from tagrec.families import label_family

        model.calibration_bias.copy_(
            torch.tensor(
                [float(family_biases.get(label_family(label), 0.0)) for label in config["labels"]],
                device=device,
            )
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source), revision=revision, trust_remote_code=False
    )
    if device.startswith("cuda"):
        model.to(dtype=torch.bfloat16)
    return model, tokenizer, config
