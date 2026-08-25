# BGE multilabel-classifier experiment

This track tests whether direct supervised classification of frequent MISP tags is
stronger and cheaper than nearest-event retrieval. It is deliberately separate
from the LFM2.5 retrieval work.

## Design

- Select the top 500 or 1,000 labels by `native_labels` frequency using **train
  only** and deterministic lexical tie-breaking.
- Omit events with no in-vocabulary native positive from training. Unlabeled
  MISP events are not treated as known all-negative examples.
- Encode each event with pinned `BAAI/bge-base-en-v1.5` and apply a multilabel
  sigmoid head. This is not a single-class softmax because events can have
  several tags.
- Initialize head weights from BGE embeddings of deterministic taxonomy tag
  documents and initialize bias from train prevalence.
- Use asymmetric loss to reduce domination by easy absent-label negatives.
- Benchmark chronologically on the unchanged real validation split. Test is
  never opened.

The checkpoint contains private vocabulary information and, when the encoder is
fine-tuned, learned private-data parameters. It belongs under ignored
`artifacts/` and must not be published.

## Training

```bash
uv run --group training python -m experiments.classifier.train \
  --data artifacts/lfm25/remote-data \
  --tag-docs artifacts/lfm25/tag-docs-new.jsonl \
  --output artifacts/classifier/bge-top500-frozen \
  --labels 500 --epochs 4 --batch-size 128 --freeze-encoder

uv run --group training python -m experiments.classifier.train \
  --data artifacts/lfm25/remote-data \
  --tag-docs artifacts/lfm25/tag-docs-new.jsonl \
  --output artifacts/classifier/bge-top1000-tuned \
  --labels 1000 --epochs 2 --batch-size 32
```

Training reads `train.jsonl.gz` only. The data manifest may mention validation,
but the training script does not open it.

## Validation benchmark

```bash
uv run --group training python -m experiments.classifier.benchmark \
  --data artifacts/lfm25/remote-data \
  --out artifacts/classifier/benchmark.json \
  --classifier top500=artifacts/classifier/bge-top500-frozen \
  --classifier top1000=artifacts/classifier/bge-top1000-tuned \
  --batch-size 128 --device cuda
```

The benchmark first recomputes the pinned BGE nearest-event baseline, then
reports paired bootstrap intervals. A frequent-label classifier cannot emit a
long-tail tag, so metrics retain all validation truth available in the training
vocabulary rather than silently restricting truth to the classifier labels.
Candidate recall uses the classifier's top 50 outputs. No abstention or retrieval
fallback is applied in this first controlled comparison.
