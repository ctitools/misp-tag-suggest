# LFM2.5 milestones M1–M3 report

**Status:** complete  
**Date:** 2026-08-24  
**Test split used:** no

## Executive result

The frozen `LiquidAI/LFM2.5-Embedding-350M` did not beat the current BGE
nearest-event retriever. One epoch of narrowly capped, real-only authoritative
fine-tuning reversed that result and produced a statistically positive improvement on
analyst-authored validation events.

The gain is promising but does not yet clear the proposed deployment gate of roughly
`+0.03` analyst P@3. BGE remains the shipped demo retriever while synthetic
augmentation is tested as a controlled ablation.

## Frozen protocol

All three models used the same prepared chronological corpus:

- 58,948 training events;
- 7,368 validation events;
- no test read;
- canonical text capped at 12,000 characters before the encoders' common 512-token
  truncation point;
- exact normalized inner-product search;
- 25-neighbor similarity-weighted votes for P@3 and recall@3;
- 50-neighbor candidate recall;
- no abstention threshold, so the comparison measures ranking rather than calibration.

BGE retained its existing symmetric no-prefix encoding. LFM used its required
`query: ` and `document: ` asymmetric prompts.

## M1 — frozen comparison

### Analyst events, native-label truth

| Model | P@3 | Recall@3 | Candidate recall@50 |
|---|---:|---:|---:|
| BGE base | **0.3632** | **0.3230** | **0.7824** |
| Frozen LFM2.5 | 0.3185 | 0.3031 | 0.7571 |

Paired frozen-LFM minus BGE deltas:

- P@3: `-0.0446`, 95% paired bootstrap interval `[-0.0558, -0.0339]`;
- recall@3: `-0.0200`, interval `[-0.0314, -0.0084]`.

Frozen LFM was therefore a clear no-go as a drop-in replacement. Its only notable
family advantage was context candidate recall (`0.6528` versus BGE `0.6296`).

### Remote-code audit

The model was pinned to revision:

```text
f35ae2c91d687658dbf1f2b449382f0b019b9808
```

The only executable Hub file, `modeling_lfm2_bidirectional.py`, had SHA-256:

```text
c7f98a43764dbb93d4d912cb09dc828672680d762452c59637917522fbaedae0
```

The complete 135-line file was reviewed. It imports only `typing`, PyTorch and
Transformers LFM2 modules. It patches the causal attention mask and short convolution
for bidirectional retrieval. It contains no filesystem access, networking,
subprocesses, dynamic execution, credential access or serialization calls.

Training and inference are pinned to Transformers `4.57.*`. The model remains under
LFM Open License v1.0, including its commercial-use restriction for legal entities
with annual revenue of at least USD 10 million.

## M2 — real-only fine-tuning

### Training corpus

The builder read training only and created 30,922 pairs:

| Pair type | Rows |
|---|---:|
| Analyst event → native recorded tag document | 24,922 |
| Partial event → full event | 6,000 |

Controls:

- feed-derived events excluded;
- native assignments only;
- maximum 40 pairs per target tag;
- 2,846 target tags represented;
- exact target-tag leakage rejected;
- no UUID, event ID or organization field in the pair artifact;
- validation and test excluded.

Family distribution for event-to-tag pairs:

| Family | Pairs |
|---|---:|
| ATT&CK technique | 12,401 |
| Malware family | 4,818 |
| Context | 4,350 |
| Incident class | 3,353 |

### Training configuration

- base: pinned `LFM2.5-Embedding-350M`;
- one epoch, seed 42;
- `CachedMultipleNegativesRankingLoss`;
- effective batch 256, forward mini-batch 32;
- `BatchSamplers.NO_DUPLICATES`;
- learning rate `2e-5`, 10% warmup;
- BF16 on one RTX 5090;
- 121 optimizer updates in 693 seconds;
- final training loss `3.1929`.

SentenceTransformers did not initially copy Liquid's remote-code file into the saved
checkpoint. Reload correctly failed rather than silently randomizing or changing the
architecture. The training script now copies only the audited file from the pinned
revision and verifies its hash before reload. The repaired checkpoint passed a fresh
local CPU process reload with finite 1,024-dimensional output.

### Matched validation result

| Model | Analyst native P@3 | Analyst native R@3 | Analyst native candidate R@50 |
|---|---:|---:|---:|
| BGE base | 0.3632 | 0.3230 | 0.7824 |
| Frozen LFM2.5 | 0.3185 | 0.3031 | 0.7571 |
| **Real-only LFM2.5** | **0.3853** | **0.3553** | **0.8296** |

Real-only LFM minus BGE paired deltas:

- native analyst P@3: `+0.0221`, interval `[+0.0127, +0.0321]`;
- native analyst recall@3: `+0.0323`, interval `[+0.0200, +0.0448]`;
- all-label analyst P@3: `+0.0114`, interval `[+0.0017, +0.0214]`;
- all-event P@3: `+0.0032`, interval `[+0.0002, +0.0061]`.

Analyst native family results:

| Family | BGE R@3 | Real-only LFM R@3 | BGE candidate R@50 | Real-only LFM candidate R@50 |
|---|---:|---:|---:|---:|
| Incident class | 0.3355 | **0.3566** | **0.8293** | 0.8257 |
| ATT&CK technique | 0.2239 | **0.2359** | 0.7336 | **0.8154** |
| Malware family | 0.4142 | **0.4490** | 0.8092 | **0.8543** |
| Context | 0.3504 | **0.4319** | 0.6296 | **0.6987** |

The real-only model beats frozen LFM substantially and beats BGE overall, especially
in candidate recall and the malware/context slices. Incident candidate recall has a
small point regression and remains a declared guardrail.

### Efficiency

On the RTX 5090 benchmark:

| Model | Train-corpus encoding | Validation encoding | Peak allocated VRAM |
|---|---:|---:|---:|
| BGE | 418 events/s | 505 events/s | 2.80 GiB |
| Real-only LFM | 171 events/s | 216 events/s | 7.61 GiB |

On the local two-core Xeon host, warm single-query CPU inference was:

| Model | p50 | p95 | Process RSS |
|---|---:|---:|---:|
| BGE | 440 ms | 456 ms | 0.82 GiB |
| Real-only LFM | 730 ms | 780 ms | 1.66 GiB |

The quality gain therefore costs approximately 1.7× single-query latency and 2× model
RSS on this old CPU.

## M3 — synthetic design and offline preparation

No OpenAI generation request was made. The direct OpenAI key incurred zero model
calls during M1–M3.

Completed artifacts and controls:

- strict six-scenario JSON Schema;
- fail-closed local validator;
- `.example`, TEST-NET and `2001:db8::/32` indicator enforcement;
- fictional SHA-256 placeholder rule;
- UUID rejection;
- candidate-only label and evidence-path validation;
- immutable local label contracts fixed before generation;
- defensive-content and privacy boundary;
- explicit GPT-5.6 prompt-cache design;
- whole-run refusal/content-filter brake specification;
- 200 taxonomy-only pilot seeds.

Pilot seeds are balanced 50 each across incident, ATT&CK, malware and context. They
contain taxonomy definitions and aggregate frequency buckets only—no event text,
indicator, UUID, event ID or organization name. There are 129 targets in the 20–99
frequency bucket and 71 in the 100+ bucket.

## Compute and reproducibility

A dedicated verified Vast.ai RTX 5090 rental was used. Only pseudonymous minimized
train/validation text, training pairs and an explicit source-code allowlist were
transferred. No local credential was copied. Input hashes were verified after
transfer; checkpoint hashes were verified after retrieval. The rental was destroyed
and its absence confirmed.

- elapsed rental time at final check: 3,352 seconds;
- compute plus storage estimate: **USD 0.4423**;
- minor traffic charges excluded from that estimate;
- local checkpoint size: approximately 1.4 GiB, ignored by Git.

## Decision

M1: **frozen LFM rejected**.  
M2: **real-only LFM retained as the challenger**.  
M3: **synthetic pilot inputs ready, but no generation authorized by this report**.

The challenger does not replace BGE yet because analyst P@3 improves by `0.0221`,
short of the approximate `0.03` adoption target, and this is one training seed with
no untouched-test read. The next justified experiment is the small GPT-5.6 Sol Flex
canary and 100–200-bundle pilot, followed by a real-only versus real+synthetic paired
validation comparison.

## Reproduction paths

- `experiments/lfm25/prepare_benchmark.py`
- `experiments/lfm25/benchmark.py`
- `experiments/lfm25/build_tag_docs.py`
- `experiments/lfm25/build_real_pairs.py`
- `experiments/lfm25/train_real_only.py`
- `experiments/lfm25/build_synthetic_seeds.py`
- `experiments/lfm25/synthetic_bundle.schema.json`
- `experiments/lfm25/validate_synthetic.py`
- `experiments/lfm25/SYNTHETIC_DATA.md`

Private reports, prepared corpora and checkpoints remain under ignored
`artifacts/lfm25/`.
