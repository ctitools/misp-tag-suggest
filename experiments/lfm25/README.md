# LFM2.5 research workflow

This directory reproduces the validation-only LFM2.5 experiment. It does not change
the BGE model used by the demo API.

All commands below use train and validation only. Do not substitute test until a
final model and decision rule have been frozen.

## 1. Install research dependencies

```bash
uv sync --group training
```

LFM remote code is restricted to the pinned revision and audited hash recorded in
[`../../docs/lfm25-m1-m3-report.md`](../../docs/lfm25-m1-m3-report.md).
Transformers is pinned to `4.57.*`.

## 2. Prepare the minimized benchmark

```bash
uv run python -m experiments.lfm25.prepare_benchmark \
  --source-data data/processed \
  --out artifacts/lfm25/remote-data
```

The prepared files still contain private canonical event text. They are mode `0600`,
ignored by Git, and must not be published or sent to model APIs.

## 3. Benchmark frozen BGE and LFM

On a CUDA host with the prepared directory available:

```bash
uv run --group training python -m experiments.lfm25.benchmark \
  --data artifacts/lfm25/remote-data \
  --out artifacts/lfm25/frozen-benchmark.json \
  --models bge,lfm \
  --device cuda
```

The script uses exact search, LFM's required asymmetric prompts and no abstention.

## 4. Build real-only pairs

Build tag documents from the accepted local snapshot:

```bash
uv run python -m experiments.lfm25.build_tag_docs \
  --taxonomy data/taxonomy/snapshot.json \
  --out artifacts/lfm25/tag-docs.jsonl \
  --manifest artifacts/lfm25/tag-docs-manifest.json
```

Then build the pairs:

```bash
uv run python -m experiments.lfm25.build_real_pairs \
  --train data/processed/train.jsonl \
  --source-manifest data/processed/manifest.json \
  --tag-docs artifacts/lfm25/tag-docs.jsonl \
  --out artifacts/lfm25/real-pairs
```

The builder uses analyst-authored native assignments, caps each target at 40, adds
6,000 partial-to-full preservation pairs, and refuses exact target leakage.

## 5. Train one frozen epoch

```bash
uv run --group training python -m experiments.lfm25.train_real_only \
  --pairs artifacts/lfm25/real-pairs/train.jsonl.gz \
  --pair-manifest artifacts/lfm25/real-pairs/manifest.json \
  --output artifacts/lfm25/real-only
```

A GPU is required. The script copies the audited Liquid modeling file into the saved
checkpoint because SentenceTransformers does not currently include it automatically.
The file hash is checked before reload.

## 6. Compare the fine-tune

```bash
uv run --group training python -m experiments.lfm25.benchmark \
  --data artifacts/lfm25/remote-data \
  --out artifacts/lfm25/matched-three-way.json \
  --models bge,lfm \
  --custom-model artifacts/lfm25/real-only/final \
  --custom-name lfm-real-only \
  --device cuda
```

## 7. Prepare taxonomy-only synthetic seeds

```bash
uv run python -m experiments.lfm25.build_synthetic_seeds \
  --tag-docs artifacts/lfm25/tag-docs.jsonl \
  --dataset-manifest data/processed/manifest.json \
  --out artifacts/lfm25/synthetic-seeds \
  --count 200
```

Read [`SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md) before implementing generation. The
seed output contains taxonomy definitions and aggregate frequency buckets only. No
script in this directory currently makes an OpenAI request.

## 8. Validate generated bundles

After a future generation stage writes one bundle per JSONL line:

```bash
uv run python -m experiments.lfm25.validate_synthetic \
  artifacts/lfm25/generated/bundles.jsonl \
  --valid-tags artifacts/lfm25/synthetic-seeds/valid-tags.json
```

Any schema, label-contract, evidence-path, UUID or reserved-indicator failure returns
a nonzero status.
