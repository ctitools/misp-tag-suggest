# AGENTS.md

This repository follows the same rules as
[generic-ai-misp-module](https://github.com/ctitools/generic-ai-misp-module), whose `AGENTS.md`
is the binding contributor guide. Read it first. What differs here:

## Stack

- Python `>=3.14`, `uv` (`uv sync --group dev`; `.venv` at the repo root, do not recreate).
- Tests: `pytest`, all offline (synthetic events, a local HTTP MISP double, fake faiss and
  encoder in `tests/conftest.py`). No test contacts a MISP or downloads a model.
- Lint: `ruff` (E, F, I, B, line 100), `pylint` 10/10 on `tagrec retrieval scripts app.py tests`,
  `semgrep` (`p/python`, `p/security-audit`). Config in `pyproject.toml`, CI in
  `.github/workflows/checks.yml`, pre-commit hook in `.githooks/`.
- `experiments/` is research code: ruff only, not part of the API.

## Dependencies

The "only pymisp" rule of the module does **not** apply here: this service is the home of the
ML stack (torch CPU, faiss-cpu, sentence-transformers, transformers, numpy) and the API
(fastapi, uvicorn, requests, python-dotenv, ijson, jsonschema, datasketch, tldextract). Keep
them as they are; they are pinned in `uv.lock`. Ask before adding one.

## Boundaries

- Read-only towards MISP: only `scripts.export_events` and `scripts.snapshot_taxonomy` contact
  it, and only through `MISP_BASE_URL` / `MISP_API_KEY` / `MISP_VERIFY_SSL` from `.env`.
- The API never writes to MISP. Writing suggestions back is the job of the caller
  (generic-ai-misp-module), which also applies the `ai-computer-assisted` tags.
- `/suggest` and `/health` are a contract (README "Used by generic-ai-misp-module"); change
  them together with `tests/test_api.py` and the module's `genai/suggest.py`.

## Loop

1. Read the relevant files first.
2. Write the test, then the smallest change that makes it pass. Less code is more.
3. `uv run pytest -q`, then ruff, pylint, semgrep (README "Run the test suite").
4. Document behaviour changes in README and add a line to `CHANGELOG.md` in the same pass.
