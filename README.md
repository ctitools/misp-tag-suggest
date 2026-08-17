# misp-suggest

Minimal, read-only FastAPI PoC around the frozen MISP Hybrid v2 recommender.

Requires the model checkout and artifacts from `misp-research`:

```bash
export MISP_RESEARCH_ROOT=/path/to/misp-research
uv sync
uv run uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/docs>. The API provides `/suggest`, `/similar`, and `/health`.

`/similar` denies every historical event unless `POC_ACL_FILE` points to an actor-to-event-ID JSON map. For a loopback-only demo, `POC_ALLOW_ALL_SIMILAR=1` explicitly disables that protection. The app has no MISP client or write path; do not expose this PoC publicly.
