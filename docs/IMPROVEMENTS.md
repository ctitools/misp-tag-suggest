# Improvements backlog

Numbered like generic-ai-misp-module `docs/IMPROVEMENTS.md`; resolved items stay with a date.

## 1. Benchmark and model quality (2026-09-05, first nanu deployment)

Context: the module-side benchmark `benchmarks/run_suggest.py` (generic-ai-misp-module)
measured precision 0.27, recall 0.13, hit@1 0.35 on 20 random dev-MISP events, while this
service's own validation (`artifacts/retrieval/training_metrics.json`, 4,583 labelled events)
gives precision@3 0.45, recall@3 0.73 and candidate recall@25 0.88. Analysis of the 20 events:
of 93 gold tags, 22 were `tlp:*`, 4 `PAP:*`, 4 `admiralty-scale`, plus `source:*`, `type:*`,
`stone:source`, `stone:threat-level`, `emotet:epoch`, all excluded by design in
`tagrec/allowlist.py`; 10 of 20 events (feed events: urlhaus, Emotet epochs) had no admissible
gold and correctly got no suggestion; two events carried only `tlp:white`, so every correct
suggestion counted as a false positive. The benchmark measured the policy, not the model.

Items 1 to 4 fix the measurement, 5 to 7 the model. Do them in this order; every model change
is judged with the fixed benchmark only.

1. **`GET /labels` endpoint.** Return the label vocabulary of the built index (the distinct
   `labels` in `index_meta.jsonl`, ~3,300 strings), `dataset_manifest_sha256`, `model_version`
   and the split boundaries from the manifest (`val_min_date`, `test_min_date`). The module
   benchmark intersects its gold set with this vocabulary and reports the excluded tags
   separately. Contract test in `tests/test_api.py`; document in README next to `/suggest`.
2. **Sample events with admissible gold.** In the module's `run_suggest.py`: at least 100
   events, several seeds, keep only events with ≥1 admissible gold tag; report the
   "no suggestion" rate on unlabelled events as a sanity number, not as recall.
3. **No training leakage.** Benchmark only events dated after `test_min_date` (2026-02-23 on
   the 2026-09-05 index), so the module benchmark is a held-out measurement like the
   service's validation.
4. **Same metrics as `training_metrics.json`.** precision@1/3/5, recall@3/5, hit@1, per family
   (attack-technique, incident-class, malware-family, context, see `tagrec/families.py`) so
   both sides compare directly. Expected after 1 to 4: P@3 near 0.45, the real baseline.
5. **Per-family thresholds and smaller k.** `params.json` has `selected_k: 25` and empty
   per-family thresholds; with 25 neighbours an exact indexed twin of the query contributes
   only 1/25 of the vote. Sweep k in 5..25 and a minimum vote score per family on the
   validation split (`retrieval/build_index.py` selection step); the module passes
   `suggest_min_score` accordingly.
6. **Near-duplicate collapse.** The first index was built with `--skip-near-dedup` (README
   first-run advice). Feed events dominate the index; rebuild with MinHash near-dedup
   (`scripts/build_dataset.py --near-threshold 0.85`) and re-run 1 to 4. Cost: the dataset
   build (25 min without near-dedup) grows; measure.
7. **LLM re-ranking of candidates (in the module).** Candidate recall@25 0.88 vs precision@3
   0.45: the answer is usually in the top 25, not the top 3. This service stays the candidate
   generator (`candidate_tags_many`, or `/suggest` with `limit` raised to 25); the module
   renders the event (its UC2 `render_event`) plus the candidates with their taxonomy
   descriptions and asks its LLM to pick. Needs: `limit` upper bound raised or a
   `/candidates` endpoint returning tag + description + vote score; the module side is a
   `tag_suggestion` variant with its own tests and golden. Largest expected gain.

Not planned again: fine-tuned LFM encoder and the supervised classifier head (both evaluated
and rejected, see `docs/lfm25-m1-m3-report.md`, `docs/bge-multilabel-classifier-report.md`).

## 2. Operations

8. **Service as a systemd unit** on the developer host; today it is started with `nohup`
   (generic-ai-misp-module `docs/DEPLOY_TAG_SUGGEST.md` step 4) and does not survive a reboot.
9. **GPU torch in the lock.** The CUDA build is installed by hand after `uv sync` (README
   step 8). A `cu128` extra with its own index in `pyproject.toml` would make it one command;
   check cp314 wheel availability on the cu128 index first (2.11.0 was the newest on
   2026-09-05).
10. **Index rebuild cadence.** The index is a snapshot of the dev MISP (82,327 events,
    2026-09-05). Document when to re-export (new taxonomies enabled, N new tagged events) and
    re-run the module benchmark after every rebuild; keep the results files as history.
