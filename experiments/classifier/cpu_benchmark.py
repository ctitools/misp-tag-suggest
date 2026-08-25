#!/usr/bin/env python3
"""Measure deterministic single-event CPU latency for BGE or its classifier."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

from experiments.classifier.model import BGE_MODEL, BGE_REVISION, load_checkpoint
from tagrec.io import iter_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", choices=("bge", "classifier"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "classifier" and args.checkpoint is None:
        parser.error("--checkpoint is required in classifier mode")

    import torch

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 16)))
    rows = list(iter_jsonl(args.data / "val.jsonl.gz"))
    texts = [str(row["text"]) for row in rows[: args.samples]]
    if len(texts) != args.samples:
        raise RuntimeError("insufficient validation rows for latency sample")

    if args.mode == "bge":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            BGE_MODEL,
            revision=BGE_REVISION,
            trust_remote_code=False,
            device="cpu",
        )
        model.max_seq_length = 512

        def invoke(text: str) -> None:
            model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    else:
        model, tokenizer, config = load_checkpoint(args.checkpoint, device="cpu")

        def invoke(text: str) -> None:
            inputs = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=int(config["max_length"]),
                return_tensors="pt",
            )
            with torch.inference_mode():
                model(inputs)

    for text in texts[: args.warmup]:
        invoke(text)
    timings = []
    for text in texts:
        started = time.perf_counter()
        invoke(text)
        timings.append((time.perf_counter() - started) * 1000)
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "samples": len(timings),
        "warmup": args.warmup,
        "single_event": True,
        "device": "cpu",
        "p50_milliseconds": float(np.quantile(timings, 0.5)),
        "p95_milliseconds": float(np.quantile(timings, 0.95)),
        "mean_milliseconds": float(np.mean(timings)),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "test_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.chmod(0o600)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
