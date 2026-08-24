#!/usr/bin/env python3
"""One-epoch LFM2.5 fine-tune on capped authoritative real-only pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LFM_MODEL = "LiquidAI/LFM2.5-Embedding-350M"
LFM_REVISION = "f35ae2c91d687658dbf1f2b449382f0b019b9808"
QUERY_PROMPT = "query: "
DOCUMENT_PROMPT = "document: "
REMOTE_CODE_SHA256 = "c7f98a43764dbb93d4d912cb09dc828672680d762452c59637917522fbaedae0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs != 1.0:
        parser.error("the first real-only experiment is frozen to exactly one epoch")

    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    if not transformers.__version__.startswith("4.57."):
        raise RuntimeError(
            f"audited LFM2.5 training requires transformers 4.57.*, got {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    pair_manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    if pair_manifest.get("validation_used") is not False or pair_manifest.get("test_used") is not False:
        raise RuntimeError("real-only pair manifest must exclude validation and test")

    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_dataset("json", data_files=str(args.pairs), split="train")
    dataset = dataset.select_columns(["query", "positive"]).shuffle(seed=args.seed)
    model = SentenceTransformer(
        LFM_MODEL,
        revision=LFM_REVISION,
        trust_remote_code=True,
        device="cuda",
    )
    model.max_seq_length = 512
    loss = CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=args.mini_batch_size
    )
    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"query": QUERY_PROMPT, "positive": DOCUMENT_PROMPT},
        save_strategy="no",
        eval_strategy="no",
        logging_steps=max(1, min(10, steps_per_epoch // 10)),
        seed=args.seed,
        dataloader_num_workers=4,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )
    started = time.perf_counter()
    train_result = trainer.train()
    runtime_seconds = time.perf_counter() - started
    final_dir = args.output / "final"
    model.save_pretrained(str(final_dir))
    # SentenceTransformers 5.7 does not copy Hub remote code into the checkpoint.
    # Copy only the audited file from the exact pinned revision and verify its hash.
    from huggingface_hub import hf_hub_download

    remote_code = Path(
        hf_hub_download(
            LFM_MODEL,
            filename="modeling_lfm2_bidirectional.py",
            revision=LFM_REVISION,
        )
    )
    if sha256(remote_code) != REMOTE_CODE_SHA256:
        raise RuntimeError("pinned LFM remote-code hash differs from the audited file")
    shutil.copy2(remote_code, final_dir / remote_code.name)
    os.chmod(final_dir / remote_code.name, 0o600)

    # A same-process reload catches incomplete SentenceTransformer serialization. A
    # separate shell invocation performs the clean-process check before retrieval.
    del trainer, model
    torch.cuda.empty_cache()
    reloaded = SentenceTransformer(
        str(final_dir), trust_remote_code=True, device="cuda"
    )
    probe = reloaded.encode(
        ["Fictional defensive incident summary."],
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if probe.shape != (1, 1024) or not np.isfinite(probe).all():
        raise RuntimeError(f"saved model reload probe failed: shape={probe.shape}")

    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    (args.output / "dependency-freeze.txt").write_text(freeze, encoding="utf-8")
    os.chmod(args.output / "dependency-freeze.txt", 0o600)
    report = {
        "schema_version": 1,
        "base_model": LFM_MODEL,
        "base_revision": LFM_REVISION,
        "pair_file_sha256": sha256(args.pairs),
        "pair_manifest_sha256": sha256(args.pair_manifest),
        "validation_used": False,
        "test_used": False,
        "training": {
            "rows": len(dataset),
            "epochs": args.epochs,
            "effective_batch_size": args.batch_size,
            "cached_loss_mini_batch_size": args.mini_batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "seed": args.seed,
            "query_prompt": QUERY_PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
            "batch_sampler": "NO_DUPLICATES",
            "loss": "CachedMultipleNegativesRankingLoss",
        },
        "runtime_seconds": runtime_seconds,
        "trainer_metrics": train_result.metrics,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "reload_probe": {"shape": list(probe.shape), "finite": True},
        "serialization": "safetensors plus audited remote-code module",
    }
    write_json(args.output / "training_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
