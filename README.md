# misp-suggest

A small, read-only MISP demo using one method: BGE semantic nearest-event retrieval. It suggests tags by similarity-weighted voting and exposes ACL-filtered, deduplicated similar events.

No dataset or trained artifact is shipped. Each installation builds its index from its own MISP.

## Build the local index

```bash
cp .env.example .env                 # add a read-only MISP key
uv sync

uv run python -m scripts.export_events --workers 1
uv run python -m scripts.snapshot_taxonomy
uv run python -m scripts.build_dataset --skip-near-dedup
uv run python -m scripts.validate_dataset
uv run python -m retrieval.build_index --device cpu
```

Use `--device cuda` for index building when CUDA is available. Exports, processed events, and artifacts are private, mode-0600 where applicable, and ignored by Git.

## Run

```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/docs>. Endpoints are `/suggest`, `/similar`, and `/health`.

`/similar` requires an `X-Actor` header and denies all events by default. Set `POC_ACL_FILE` to a local JSON file such as `{"analyst": ["123", "456"]}`. `POC_ALLOW_ALL_SIMILAR=1` is only for a loopback demo.

The app has no MISP write path. Do not expose this PoC publicly or commit `.env`, `data/`, or `artifacts/`.
