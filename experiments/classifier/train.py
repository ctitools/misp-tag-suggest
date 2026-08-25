#!/usr/bin/env python3
"""Train a pinned BGE multilabel head using train data only.

The chronological validation and test files are never opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.classifier.model import (
    BGE_MODEL,
    BGE_REVISION,
    asymmetric_loss,
    build_model,
    load_checkpoint,
    load_tag_documents,
    prevalence_bias,
    select_top_labels,
    supervised_rows,
)
from tagrec.io import iter_jsonl, write_json_secure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_from_documents(
    model: Any,
    tokenizer: Any,
    documents: list[str],
    biases: list[float],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> None:
    import torch

    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            inputs = tokenizer(
                documents[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            embeddings.append(model.embeddings(inputs).float().cpu())
    weights = torch.cat(embeddings)
    if weights.shape != model.classifier.weight.shape:
        raise RuntimeError("tag-document initialization shape differs from classifier head")
    with torch.no_grad():
        model.classifier.weight.copy_(weights.to(device))
        model.classifier.bias.copy_(torch.tensor(biases, device=device))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tag-docs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=int, choices=(500, 1000), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.0,
        help="chronological tail of train to reserve for family-score calibration",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the controlled classifier training run")
    if not 0.0 <= args.calibration_fraction < 0.5:
        raise ValueError("calibration fraction must be in [0, 0.5)")
    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not transformers.__version__.startswith("4.57."):
        raise RuntimeError(
            f"pinned classifier training requires transformers 4.57.*, got {transformers.__version__}"
        )
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_split_used") is not False:
        raise RuntimeError("input manifest must explicitly exclude test")
    if set(manifest.get("splits_included", [])) != {"train", "val"}:
        raise RuntimeError("expected the frozen train/validation preparation manifest")

    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tag_documents = load_tag_documents(args.tag_docs)
    all_train_rows = list(iter_jsonl(args.data / "train.jsonl.gz"))
    calibration_start = round(len(all_train_rows) * (1.0 - args.calibration_fraction))
    fit_rows = all_train_rows[:calibration_start]
    labels, counts = select_top_labels(
        fit_rows, limit=args.labels, allowed=set(tag_documents)
    )
    if len(labels) != args.labels:
        raise RuntimeError(f"requested {args.labels} labels but selected {len(labels)}")
    train_rows = supervised_rows(fit_rows, labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    biases = prevalence_bias(labels, counts, len(train_rows))

    class EventDataset(Dataset):
        def __len__(self) -> int:
            return len(train_rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return train_rows[index]

    tokenizer = AutoTokenizer.from_pretrained(
        BGE_MODEL, revision=BGE_REVISION, trust_remote_code=False
    )

    def collate(batch: list[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
        inputs = tokenizer(
            [str(row["text"]) for row in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        targets = torch.zeros((len(batch), len(labels)), dtype=torch.float32)
        for row_index, row in enumerate(batch):
            for label in row.get("native_labels", []):
                label_index = label_to_index.get(str(label))
                if label_index is not None:
                    targets[row_index, label_index] = 1.0
        return inputs, targets

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        EventDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    model = build_model(label_count=len(labels), device=args.device)
    initialize_from_documents(
        model,
        tokenizer,
        [tag_documents[label] for label in labels],
        biases,
        batch_size=max(args.batch_size, 64),
        max_length=args.max_length,
        device=args.device,
    )
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    parameter_groups = [
        {
            "params": list(model.classifier.parameters()) + [model.logit_scale],
            "lr": args.head_learning_rate,
            "weight_decay": args.weight_decay,
        }
    ]
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": args.encoder_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    started = time.perf_counter()
    epoch_losses: list[float] = []
    global_step = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        running_loss = 0.0
        for inputs, targets in loader:
            inputs = {
                key: value.to(args.device, non_blocking=True)
                for key, value in inputs.items()
            }
            targets = targets.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = asymmetric_loss(logits.float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
            global_step += 1
            if global_step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "steps": total_steps,
                            "loss": running_loss / (global_step - epoch * len(loader)),
                        }
                    ),
                    flush=True,
                )
        epoch_losses.append(running_loss / len(loader))
    runtime_seconds = time.perf_counter() - started

    head_state = {
        "classifier.weight": model.classifier.weight.detach().float().cpu().contiguous(),
        "classifier.bias": model.classifier.bias.detach().float().cpu().contiguous(),
        "logit_scale": model.logit_scale.detach().float().cpu().contiguous(),
    }
    save_file(head_state, str(args.output / "classifier_head.safetensors"))
    tokenizer.save_pretrained(args.output / "tokenizer")
    if not args.freeze_encoder:
        model.encoder.save_pretrained(
            args.output / "encoder", safe_serialization=True, max_shard_size="2GB"
        )

    frequencies = [counts[label] for label in labels]
    config = {
        "schema_version": 1,
        "architecture": "BGE CLS encoder plus multilabel linear head",
        "base_model": BGE_MODEL,
        "base_revision": BGE_REVISION,
        "encoder_frozen": args.freeze_encoder,
        "labels": labels,
        "max_length": args.max_length,
        "loss": {
            "name": "asymmetric",
            "gamma_negative": 4.0,
            "gamma_positive": 0.0,
            "negative_clip": 0.05,
        },
        "calibration_fraction": args.calibration_fraction,
        "calibration_start_index": calibration_start,
        "family_biases": {},
    }
    write_json_secure(args.output / "classifier_config.json", config)
    del model
    torch.cuda.empty_cache()
    probe_model, probe_tokenizer, _ = load_checkpoint(args.output, device=args.device)
    probe_inputs = probe_tokenizer(
        ["Compact fictional security incident"],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    probe_inputs = {key: value.to(args.device) for key, value in probe_inputs.items()}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        probe = probe_model(probe_inputs).float().cpu()

    report = {
        "schema_version": 1,
        "model": "BGE multilabel classifier",
        "base_model": BGE_MODEL,
        "base_revision": BGE_REVISION,
        "input_manifest_sha256": sha256(manifest_path),
        "train_file_sha256": sha256(args.data / "train.jsonl.gz"),
        "tag_documents_sha256": sha256(args.tag_docs),
        "splits_used": ["train"],
        "validation_used": False,
        "test_used": False,
        "training": {
            "all_train_rows": len(all_train_rows),
            "fit_partition_rows": len(fit_rows),
            "reserved_calibration_rows": len(all_train_rows) - len(fit_rows),
            "supervised_train_rows": len(train_rows),
            "label_count": len(labels),
            "minimum_label_frequency": min(frequencies),
            "median_label_frequency": statistics.median(frequencies),
            "maximum_label_frequency": max(frequencies),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "encoder_frozen": args.freeze_encoder,
            "encoder_learning_rate": args.encoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "steps": total_steps,
            "epoch_losses": epoch_losses,
        },
        "runtime_seconds": runtime_seconds,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "reload_probe": {
            "shape": list(probe.shape),
            "finite": bool(torch.isfinite(probe).all()),
        },
    }
    write_json_secure(args.output / "training_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
