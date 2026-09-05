# Changelog

## 2026-09-05 (docs review)

- `app.py` loads `.env` (python-dotenv, shell wins), so `SUGGEST_API_KEY`, `MODEL_DEVICE`, `RETRIEVAL_ARTIFACTS`, `POC_ACL_FILE` can live next to the MISP keys; `.env.example` lists `SUGGEST_API_KEY`. README: which settings `.env` covers, disk/time reference numbers, detached start and stop, "Rebuilding" (restart after a new index), complete abstention example and its second meaning, troubleshooting for 401 and `No route to host`.

## 2026-09-05 (backlog)

- `docs/IMPROVEMENTS.md`: why the first module-side benchmark (P 0.27 / R 0.13) measured the label policy rather than the model, and the ordered plan: `/labels` endpoint, admissible-gold sampling, held-out dates, family metrics, then per-family thresholds and smaller k, near-dedup rebuild, LLM re-ranking of candidates in the module; operations items (systemd, GPU torch extra, rebuild cadence).

## 2026-09-05 (harmonized with generic-ai-misp-module)

- Renamed the project `misp-suggest` → `misp-tag-suggest` (pyproject, API title, README, research code identifiers).
- Python floor raised to 3.14, lock regenerated (`uv lock`); all runtime dependencies unchanged.
- Lint and CI now identical to generic-ai-misp-module: ruff (E/F/I/B, line 100), pylint 10/10 on `tagrec retrieval scripts app.py tests` (complexity metrics disabled with a reason in `pyproject.toml`; `experiments/` ruff only), semgrep `p/python` + `p/security-audit`, `.githooks/pre-commit`. `.github/workflows/checks.yml` replaces `test.yml`. Whole tree reformatted; a handful of long lines, a `zip(strict=)`, an unused variable and a `raise ... from` fixed on the way; no behaviour change.
- `.env` keys renamed to the module's: `MISP_BASE_URL`, `MISP_API_KEY`, `MISP_VERIFY_SSL` (old names are no longer read).
- API: optional shared secret `SUGGEST_API_KEY` checked as header `X-Api-Key` on `/suggest` and `/similar` (unset: no check). `tests/test_api.py` pins the `/suggest` response keys as the contract generic-ai-misp-module relies on.
- README: GPU torch swap after `uv sync`, pointer to the deployment runbook in generic-ai-misp-module.
- Docs: `AGENTS.md`, `CLAUDE.md`, README sections "Run the test suite" (lint commands) and "Used by generic-ai-misp-module" (contract, why HTTP not import).
