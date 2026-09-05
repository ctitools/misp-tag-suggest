@AGENTS.md

# CLAUDE.md

Read `AGENTS.md` first. Commands (the `.venv` exists; use `uv`):

```bash
uv sync --group dev                                            # runtime + pytest/ruff/pylint
uv run pytest -q                                               # whole suite, offline
uv run pytest -q tests/test_api.py -k api_key                  # one file / one test by keyword
uv run ruff check . && uv run ruff format --check .            # lint + formatting
uv run pylint tagrec retrieval scripts app.py tests            # must be 10/10
uvx semgrep scan --config p/python --config p/security-audit --error tagrec retrieval scripts app.py tests
MODEL_DEVICE=cpu uv run uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1   # the API
```

`ruff format .` fixes formatting. Data flow, artifacts and the API contract are in `README.md`.
