# misp-tag-suggest

A read-only MISP proof of concept using one method: **BGE semantic nearest-event retrieval**.

Given a MISP Event JSON document, it can:

- suggest taxonomy tags by similarity-weighted voting over previously tagged events;
- return similar historical events after an ACL check and duplicate/template collapse;
- abstain when the nearest historical event is not similar enough.

The repository contains all code needed to export your own MISP data and build the index. It intentionally contains no event data, credentials, or trained artifacts.

## 1. Prerequisites

You need:

- Linux or macOS and Python 3.14 (same floor as generic-ai-misp-module);
- [`uv`](https://docs.astral.sh/uv/);
- a MISP API key with **read-only** access to events, tags, taxonomies, and galaxies;
- enough private disk space for your MISP export and generated embeddings.

The default installation deliberately uses CPU-only PyTorch, avoiding several gigabytes of unexpected CUDA packages. Allow roughly 2 GB for the Python environment and downloaded encoder before accounting for your data. A small smoke export works on an ordinary machine. A large MISP may need 10+ GB of additional free disk and several hours to export and encode. Keep the repository and generated files on an access-controlled host.

## 2. Install

```bash
git clone https://github.com/ctitools/misp-tag-suggest.git
cd misp-tag-suggest
uv sync
cp .env.example .env
chmod 600 .env
```

Edit `.env`:

```dotenv
MISP_BASE_URL=https://misp.example.org
MISP_API_KEY=your-read-only-api-key
MISP_VERIFY_SSL=true
```

The three keys are named exactly as in generic-ai-misp-module, so one `.env` can serve both.
Keep TLS verification enabled. Never commit `.env` or put the API key in a command line.

## 3. Verify MISP export access

Run a ten-event smoke export first:

```bash
uv run python -m scripts.export_events \
  --out data/raw/smoke.jsonl.gz \
  --max-events 10 \
  --workers 1 \
  --min-free-gb 0

gzip -cd data/raw/smoke.jsonl.gz | wc -l
```

The final command should print `10`. Remove the smoke file after checking it:

```bash
rm data/raw/smoke.jsonl.gz
```

This is the first command that contacts MISP. It only reads events and writes a local mode-0600 export.

## 4. Export the dataset

Export all accessible events with one worker, which minimizes memory use:

```bash
uv run python -m scripts.export_events --workers 1
```

The exporter is resumable. Completed page shards are retained if a request fails, so running the same command again continues the export. The final file is:

```text
data/raw/events.jsonl.gz
```

For a time-bounded demo, add dates such as:

```bash
uv run python -m scripts.export_events \
  --workers 1 \
  --since 2025-01-01 \
  --until 2025-12-31
```

## 5. Snapshot the allowed tag vocabulary

The model must only emit tags that exist on the target MISP. Snapshot its taxonomies, galaxies, and supported custom tags:

```bash
uv run python -m scripts.snapshot_taxonomy
```

This creates a timestamped snapshot and the link:

```text
data/taxonomy/snapshot.json
```

## 6. Build and validate the processed dataset

Canonicalization removes existing tags from model input and normalizes high-cardinality IOC values. The build performs exact deduplication and chronological train/validation/test splitting.

```bash
uv run python -m scripts.build_dataset --skip-near-dedup
uv run python -m scripts.validate_dataset
```

Expected files:

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/manifest.json
data/processed/validation_report.md
```

Open `data/processed/validation_report.md` and confirm it says `Status: PASS`. Also check that the manifest reports labeled training events. An index built from mostly untagged events cannot produce useful tag suggestions.

## 7. Build the one retrieval model

The index builder downloads the pinned `BAAI/bge-base-en-v1.5` encoder, embeds train and validation events, selects retrieval depth and abstention threshold on validation, and writes safe JSON/JSONL/NPZ artifacts.

Build the index on CPU:

```bash
uv run python -m retrieval.build_index --device cpu
```

Advanced users can substitute a CUDA-enabled PyTorch installation, but the documented PoC intentionally stays CPU-only and predictable.

Expected artifacts:

```text
artifacts/retrieval/train_embeddings.npz
artifacts/retrieval/index_meta.jsonl
artifacts/retrieval/params.json
artifacts/retrieval/training_metrics.json
```

These artifacts are specific to your MISP and are ignored by Git. `index_meta.jsonl` contains historical event identifiers and tags; protect it like the processed dataset.

## 8. Run the demo API

Start one worker. Multiple workers would load multiple copies of the model and index.

```bash
MODEL_DEVICE=cpu uv run uvicorn app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

Open:

- API documentation: <http://127.0.0.1:8000/docs>
- health check: <http://127.0.0.1:8000/health>

A healthy, indexed service reports `"artifacts_ready": true`. The model loads lazily on the first inference request.

GPU: the lock pins CPU-only torch. After `uv sync`, swap it for a CUDA build and start the
service from `.venv/bin/` (`uv run` would re-sync back to CPU):

```bash
uv pip install --python .venv/bin/python "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
MODEL_DEVICE=cuda .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

The full deployment on the CIRCL developer host, with the numbers of the first index build,
is written up in generic-ai-misp-module `docs/DEPLOY_TAG_SUGGEST.md`.

Set `SUGGEST_API_KEY=<secret>` in the environment to require the header `X-Api-Key: <secret>` on
`/suggest` and `/similar` (unset: no check, loopback demo only).

## 9. Request tag suggestions

```bash
curl -s http://127.0.0.1:8000/suggest \
  -H 'Content-Type: application/json' \
  -d '{
    "limit": 5,
    "event": {
      "Event": {
        "info": "Credential-harvesting email campaign",
        "Attribute": [
          {
            "category": "Network activity",
            "type": "domain",
            "value": "example.invalid"
          }
        ]
      }
    }
  }' | uv run python -m json.tool
```

The response contains ranked tags and retrieval vote scores. If the event is unlike the indexed history, the service returns:

```json
{
  "suggestions": [],
  "abstained": true
}
```

Abstention is expected behavior, not an error. Suggestions are never written back to MISP.

## 10. Request similar events

`/similar` is deny-by-default. For a local demo, create an ACL file mapping an actor to allowed MISP event IDs:

```bash
printf '%s\n' '{"demo-analyst": ["*"]}' > acl.json
chmod 600 acl.json
```

`"*"` grants that actor access to every indexed event and is suitable only for a loopback demo. In a real integration, generate explicit event-ID lists from your authorization system.

Restart the API with the ACL file:

```bash
POC_ACL_FILE=acl.json MODEL_DEVICE=cpu uv run uvicorn app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Then request similar events:

```bash
curl -s http://127.0.0.1:8000/similar \
  -H 'Content-Type: application/json' \
  -H 'X-Actor: demo-analyst' \
  -d '{
    "limit": 5,
    "event": {
      "Event": {
        "info": "Credential-harvesting email campaign"
      }
    }
  }' | uv run python -m json.tool
```

The response includes similarity, event ID, title, date, historical tags, and `group_size`. ACL filtering happens before duplicate/template collapse.

## Troubleshooting

- **`503 Retrieval index not built`** — complete step 7 or set `RETRIEVAL_ARTIFACTS` to the artifact directory.
- **PyTorch or device error** — use the documented CPU commands unless you intentionally installed a compatible CUDA build.
- **No tag suggestions** — the system may be abstaining, or the indexed training events may have too few eligible tags. Inspect `manifest.json` and `training_metrics.json`.
- **No similar events** — check `X-Actor` and `POC_ACL_FILE`; access is denied by default.
- **TLS certificate error** — install the correct CA certificate. Set `MISP_VERIFY_SSL=false` only for an isolated test where the risk is understood.
- **Export interrupted** — rerun the same export command without `--overwrite`; it resumes completed pages.
- **Out of memory** — keep export workers at one, use a smaller date range, and build the embedding index on a larger machine.

## Run the test suite

The tests use synthetic events and a local HTTP MISP double; they do not contact a real MISP or download a model.

```bash
uv run pytest -q
```

They cover export pagination and credential handling, taxonomy snapshotting, dataset construction and validation, index creation, retrieval voting and abstention, API contracts, ACL ordering, canonicalization, and repository hygiene.

Lint must be clean before a commit, same rules and tools as generic-ai-misp-module
(`AGENTS.md`); GitHub Actions (`.github/workflows/checks.yml`) runs all four on every push:

```bash
uv run ruff check . && uv run ruff format --check .
uv run pylint tagrec retrieval scripts app.py tests
uvx semgrep scan --config p/python --config p/security-audit --error tagrec retrieval scripts app.py tests
```

`git config core.hooksPath .githooks` runs ruff on every commit. `experiments/` is research
code: ruff only.

## Used by generic-ai-misp-module

[generic-ai-misp-module](https://github.com/ctitools/generic-ai-misp-module) calls this service
over HTTP for its `tag_suggestion` use-case (`genai/suggest.py` there): it POSTs the MISP Event
to `/suggest`, adds the returned tags to the event and tags the event `ai-computer-assisted`.
The module reads `MISP_TAG_SUGGEST_URL` and `MISP_TAG_SUGGEST_API_KEY` from its `.env`.
It relies on this contract, asserted in `tests/test_api.py`:

```text
GET  /health                  -> {"status", "artifacts_ready", "model_loaded", "read_only"}
POST /suggest {"event": {"Event": {...}}, "limit": 1..10}
                              -> {"suggestions": [{"tag", "score"}], "abstained",
                                  "model_version", "dataset_manifest_sha256"}
```

Why HTTP and not `import`: this service needs torch, faiss and sentence-transformers; the
module allows only `pymisp` as a dependency and runs inside the misp-modules server.

## Experimental research

The shipped demo remains the pinned BGE retriever.

A real-only `LiquidAI/LFM2.5-Embedding-350M` fine-tune improved analyst native-label
P@3 over BGE by `0.0221`, below the predeclared adoption target. A later controlled
25% synthetic mix reached a `+0.0275` validation delta versus BGE but did not
significantly beat real-only LFM and introduced family regressions. LFM is therefore
not enabled in the API. Reproducible LFM and privacy-preserving generation code is
under `experiments/lfm25/`.

A supervised top-1,000 multilabel head on BGE produced a much larger validation P@3
gain with essentially unchanged CPU latency and memory. It retained the fine-tuned
encoder for nearest-event candidate retrieval. The frozen configuration was allowed
one test read, where overall P@3 still improved but incident and attack ranking recall
regressed significantly. It was rejected and the test is now consumed. Code is under
`experiments/classifier/`; see
[`docs/bge-multilabel-classifier-report.md`](docs/bge-multilabel-classifier-report.md).

Training dependencies are installed with:

```bash
uv sync --group training
```

See [`docs/lfm25-m1-m3-report.md`](docs/lfm25-m1-m3-report.md) for LFM retrieval
results and [`docs/sol-flex-canary-report.md`](docs/sol-flex-canary-report.md) for
synthetic-generation controls and diagnostics.

## Security boundaries

- MISP is contacted only by `scripts.export_events` and `scripts.snapshot_taxonomy`.
- The FastAPI application has no MISP client and no write path.
- Existing event tags are excluded from model input.
- `/similar` requires authorization before historical metadata is returned.
- `.env`, `data/`, `artifacts/`, logs, and ACL files are excluded from Git.
- Bind to `127.0.0.1`; `SUGGEST_API_KEY` is a shared secret, not authentication. Add TLS, rate limiting, and audited authorization before any network deployment.
