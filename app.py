from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path(
    os.getenv("MISP_RESEARCH_ROOT", HERE.parent / "misp-research")
).resolve()
DEVICE = os.getenv("MODEL_DEVICE") or None

if not (MODEL_ROOT / "models/09-hybrid-production/service.py").is_file():
    raise RuntimeError(f"MISP_RESEARCH_ROOT is not a model checkout: {MODEL_ROOT}")

sys.path.insert(0, str(MODEL_ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hybrid_module = load_module(
    "misp_suggest_hybrid", MODEL_ROOT / "models/09-hybrid-production/service.py"
)
similar_module = load_module(
    "misp_suggest_similar", MODEL_ROOT / "models/09-hybrid-production/similar_endpoint.py"
)
HybridV2Service = hybrid_module.HybridV2Service
SimilarEventService = similar_module.SimilarEventService

_LONG_HEX_RE = re.compile(r"[0-9a-fA-F]{8,}")
_DIGITS_RE = re.compile(r"\d+")


def normalized_info(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    return _DIGITS_RE.sub("<N>", _LONG_HEX_RE.sub("<H>", first))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class SuggestRequest(BaseModel):
    event: dict[str, Any]
    mode: Literal["full", "partial-info", "partial-25", "partial-50"] = "full"
    limit: int = Field(default=5, ge=1, le=10)


class SimilarRequest(BaseModel):
    event: dict[str, Any]
    limit: int = Field(default=10, ge=1, le=20)


class LocalSimilarRuntime:
    """Load Track 02 for one request, then release it to bound memory."""

    def __init__(self) -> None:
        self.artifacts = MODEL_ROOT / "models/02-semantic-retrieval/artifacts-full"
        self.display_rows: list[dict[str, Any]] | None = None
        self.allow_all = os.getenv("POC_ALLOW_ALL_SIMILAR") == "1"
        self.acl = self._load_acl(os.getenv("POC_ACL_FILE"))

    @staticmethod
    def _load_acl(path: str | None) -> dict[str, set[str]]:
        if not path:
            return {}
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(actor): set(map(str, ids)) for actor, ids in raw.items()}

    def authorize(self, actor: str, event_id: str) -> bool:
        allowed = self.acl.get(actor, set())
        return self.allow_all or "*" in allowed or event_id in allowed

    def _display(self) -> list[dict[str, Any]]:
        if self.display_rows is not None:
            return self.display_rows

        compact: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(MODEL_ROOT / "data/processed/train.jsonl"):
            text = str(row.get("text", ""))
            first = text.splitlines()[0] if text else ""
            compact[str(row["uuid"])] = {
                "title": first.removeprefix("[INFO] "),
                "orgc": str(row.get("orgc", "")),
                "normalized_info": normalized_info(text),
                "dup_cluster": str(row.get("dup_cluster", "")),
            }

        rows = []
        for meta in iter_jsonl(self.artifacts / "index_meta.jsonl"):
            item = compact.get(str(meta.get("uuid")), {})
            rows.append(
                {
                    "event_id": str(meta.get("event_id", "")),
                    "uuid": str(meta.get("uuid", "")),
                    "date": str(meta.get("date", "")),
                    "tags": list(map(str, meta.get("labels", []))),
                    **item,
                }
            )
        self.display_rows = rows
        return rows

    def search(self, event: dict[str, Any], actor: str, top_k: int) -> dict[str, Any]:
        display = self._display()
        retrieval_module = load_module(
            "misp_suggest_retrieval",
            MODEL_ROOT / "models/02-semantic-retrieval/retrieve.py",
        )
        model = retrieval_module.SemanticRetrievalSuggester(
            self.artifacts, device=DEVICE
        )

        def retrieve(raw_event: dict[str, Any], depth: int):
            text = hybrid_module._canonical_text(raw_event)
            similarities, indices = model._search([{"text": text}], k=depth)
            return [
                display[int(index)] | {"similarity": float(score)}
                for score, index in zip(similarities[0], indices[0], strict=True)
                if index >= 0
            ]

        try:
            service = SimilarEventService(retrieve, self.authorize, search_depth=5000)
            result = service.similar(event, actor, top_k)
            result["suggestions"] = [
                {
                    key: row[key]
                    for key in (
                        "event_id",
                        "uuid",
                        "date",
                        "title",
                        "tags",
                        "similarity",
                        "group_size",
                    )
                    if key in row
                }
                for row in result["suggestions"]
            ]
            return result
        finally:
            del model
            gc.collect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.hybrid = HybridV2Service(device=DEVICE)
    app.state.similar = LocalSimilarRuntime()
    app.state.inference_lock = asyncio.Lock()
    yield


app = FastAPI(title="MISP Suggest PoC", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    version = app.state.hybrid.version
    return {
        "status": "ok",
        "read_only": True,
        "hybrid_version": version["hybrid_version"],
        "parity": version["parity"],
    }


@app.post("/suggest")
async def suggest(request: SuggestRequest):
    async with app.state.inference_lock:
        result = await asyncio.to_thread(
            app.state.hybrid.suggest, request.event, request.mode, 50
        )

    suggestions = [
        {"tag": row["tag"], "score": round(float(row["score"]), 4)}
        for row in result["suggestions"]
        if row["threshold_pass"]
    ][: request.limit]
    version = result["model_version"]
    return {
        "suggestions": suggestions,
        "abstained": not suggestions,
        "mode": request.mode,
        "model_version": {
            "hybrid_version": version["hybrid_version"],
            "params_sha256": version["params_sha256"],
            "parity": version["parity"],
        },
    }


@app.post("/similar")
async def similar(
    request: SimilarRequest,
    x_actor: str | None = Header(default=None),
):
    if not x_actor:
        raise HTTPException(status_code=401, detail="X-Actor is required")
    async with app.state.inference_lock:
        return await asyncio.to_thread(
            app.state.similar.search, request.event, x_actor, request.limit
        )
