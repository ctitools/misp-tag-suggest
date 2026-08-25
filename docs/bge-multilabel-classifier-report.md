# BGE multilabel classifier report

Date: 2026-08-25

## Decision

**Keep the pinned BGE nearest-event retriever in production.** A fine-tuned
1,000-label BGE classifier cleared the validation adoption gate and retained the
same encoder for candidate retrieval, but the one-time chronological test showed
significant incident-class and attack-technique ranking regressions. The test split
is now consumed and must not be used for further tuning.

The classifier remains a private research checkpoint. It and its vocabulary are
ignored artifacts and must not be published.

## Method

The experiment selected labels by native-label frequency using train only. MISP
Events are multilabel, so the model uses 1,000 sigmoid outputs rather than a
single-class softmax. Head weights were initialized from deterministic taxonomy
tag-document embeddings. Training used asymmetric loss and omitted rows with no
in-vocabulary native positive rather than treating unlabeled events as known
all-negative examples.

The selected configuration was:

- pinned `BAAI/bge-base-en-v1.5` at revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`;
- top 1,000 train-native labels;
- 53,053 fit-partition events and 21,419 events with an in-vocabulary positive;
- 5,895 chronological train events reserved for calibration;
- two epochs, batch size 32, encoder LR `2e-5`, head LR `1e-3`;
- train-only family offsets with an absolute 0.01 calibration-family guardrail;
- 354 seconds of training on an RTX 3090;
- 422 MiB private checkpoint.

Validation and test truth retained every eligible label available in the training
vocabulary. It was not silently restricted to the classifier's 1,000 outputs.
Candidate recall used 50 nearest historical events encoded by the same fine-tuned
encoder, not the classifier's 50 highest-scoring labels.

## Chronological validation

Analyst-authored native-label results:

| Model | P@3 | Recall@3 | Candidate R@50 |
|---|---:|---:|---:|
| BGE retrieval | 0.3632 | 0.3230 | 0.7824 |
| BGE top-1,000 classifier | **0.4540** | **0.3540** | **0.8465** |
| Delta | **+0.0908** | **+0.0309** | **+0.0641** |

Paired validation P@3 CI: `[+0.0781, +0.1032]`.

Validation family recall@3 deltas were:

| Family | Delta | Paired 95% CI |
|---|---:|---:|
| Incident class | -0.0041 | [-0.0231, +0.0152] |
| Attack technique | +0.0341 | [+0.0198, +0.0486] |
| Malware family | +0.0042 | [-0.0244, +0.0338] |
| Context | +0.0475 | [+0.0156, +0.0804] |

There was no statistically significant validation family regression. Candidate
recall also rose in every family. The result exceeded the predeclared `+0.03` P@3
gate, had a positive paired interval, and passed local CPU cost checks. This
permitted the one-time test read.

## CPU deployment cost

On the local CPU host over the same 100 validation events:

| Model | p50 | p95 | Max RSS |
|---|---:|---:|---:|
| BGE retrieval encoder | 172 ms | 637 ms | 0.88 GiB |
| Fine-tuned BGE classifier | 168 ms | 648 ms | 0.87 GiB |

The 1,000-output head therefore adds negligible runtime or memory. Candidate
retrieval can reuse the fine-tuned encoder, so a second encoder is unnecessary.

## One-time chronological test

The final configuration was frozen before test preparation. The gate-enforcing
preparation script verified validation quality, family intervals, candidate
coverage, CPU latency and memory before opening `test.jsonl`. The benchmark was
then run once.

Analyst-authored native-label results:

| Model | P@3 | Recall@3 | Candidate R@50 |
|---|---:|---:|---:|
| BGE retrieval | 0.2903 | 0.2559 | 0.6822 |
| BGE top-1,000 classifier | **0.3530** | 0.2699 | **0.7680** |
| Delta | **+0.0627** | +0.0139 | **+0.0858** |

Paired test intervals:

- P@3: `[+0.0481, +0.0775]` — positive;
- Recall@3: `[-0.0013, +0.0290]` — includes zero.

However, family recall@3 failed the final guardrail:

| Family | BGE | Classifier | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Incident class | 0.3335 | 0.2980 | -0.0355 | [-0.0538, -0.0180] |
| Attack technique | 0.1860 | 0.1481 | -0.0379 | [-0.0564, -0.0188] |
| Malware family | 0.4129 | 0.4539 | +0.0410 | [-0.0060, +0.0876] |
| Context | 0.1559 | 0.2329 | +0.0770 | [+0.0463, +0.1099] |

Candidate recall improved overall and for attack, malware and context, but
incident candidate recall fell from 0.6511 to 0.6360. The stronger overall P@3 is
not sufficient to override two significant ranking-family regressions.

## Interpretation

Direct classification is substantially better at frequent native-label ranking
than nearest-event voting and is operationally cheap. Its weakness is temporal
and family calibration: gains concentrated in malware and context did not
preserve incident and attack performance on the later test period. It also
regressed on the broader all-label metric because it was intentionally trained on
native analyst labels rather than propagated/freetext-mapped supervision.

A future hybrid could route incident and attack ranking through retrieval while
using classifier scores for malware and context. That hypothesis must be designed
using train or new shadow data and evaluated on a **new future chronological
holdout**. The consumed test split cannot be reused for model selection.

## Reproducibility and cost

Code is under `experiments/classifier/`. It includes deterministic label
selection, training, train-only calibration, validation benchmarking, CPU
benchmarking, and fail-closed one-time test preparation. The suite has 49 tests.

Estimated Vast cost was `$0.35`, including `$0.06` discarded before data transfer
when an offer was found to expose only a quarter GPU. The successful full RTX
3090 instance was destroyed after reports and the hash-verified checkpoint were
retrieved. No project training instance remains active.
