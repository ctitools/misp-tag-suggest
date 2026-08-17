from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from retrieval.retrieve import SemanticRetrievalSuggester
from tagrec.canonicalize import canonicalize_event
from tagrec.diversify import collapse_similar_neighbors

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.getenv("RETRIEVAL_ARTIFACTS", ROOT / "artifacts/retrieval"))
DEVICE = os.getenv("MODEL_DEVICE") or None
SEARCH_DEPTH = int(os.getenv("SIMILAR_SEARCH_DEPTH", "5000"))


class EventRequest(BaseModel):
    event: dict[str, Any]
    limit: int = Field(default=5, ge=1, le=10)


class RetrievalRuntime:
    def __init__(self) -> None:
        self._model: SemanticRetrievalSuggester | None = None
        self.allow_all = os.getenv("POC_ALLOW_ALL_SIMILAR") == "1"
        self.acl = self._load_acl(os.getenv("POC_ACL_FILE"))

    @staticmethod
    def _load_acl(path: str | None) -> dict[str, set[str]]:
        if not path:
            return {}
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(actor): set(map(str, ids)) for actor, ids in raw.items()}

    @property
    def ready(self) -> bool:
        return all(
            (ARTIFACTS / name).is_file()
            for name in ("params.json", "train_embeddings.npz", "index_meta.jsonl")
        )

    def model(self) -> SemanticRetrievalSuggester:
        if not self.ready:
            raise FileNotFoundError(
                f"Retrieval artifacts are missing from {ARTIFACTS}; run retrieval.build_index"
            )
        if self._model is None:
            self._model = SemanticRetrievalSuggester(ARTIFACTS, device=DEVICE)
        return self._model

    def authorize(self, actor: str, event_id: str) -> bool:
        allowed = self.acl.get(actor, set())
        return self.allow_all or "*" in allowed or event_id in allowed

    @staticmethod
    def input_row(event: dict[str, Any]) -> dict[str, str]:
        return {"text": canonicalize_event(event)}

    def suggest(self, event: dict[str, Any], limit: int) -> dict[str, Any]:
        model = self.model()
        rows = [self.input_row(event)]
        values = model.suggest_many(rows, top_k=limit)[0]
        accepted = model.event_acceptance_many(rows)[0]
        suggestions = (
            [{"tag": value.tag, "score": round(float(value.score), 4)} for value in values]
            if accepted
            else []
        )
        return {
            "suggestions": suggestions,
            "abstained": not suggestions,
            "model_version": model.model_version,
            "dataset_manifest_sha256": model.params["dataset_manifest_sha256"],
        }

    def similar(
        self, event: dict[str, Any], actor: str, limit: int
    ) -> dict[str, Any]:
        model = self.model()
        rows = [self.input_row(event)]
        similarities, indices = model._search(rows, k=SEARCH_DEPTH)
        visible: list[dict[str, Any]] = []
        denied = 0
        for score, index in zip(similarities[0], indices[0], strict=True):
            if index < 0:
                continue
            metadata = dict(model.metadata[int(index)])
            event_id = str(metadata.get("event_id", ""))
            if not event_id or not self.authorize(actor, event_id):
                denied += 1
                continue
            metadata["event_id"] = event_id
            metadata["similarity"] = float(score)
            visible.append(metadata)

        diversified = collapse_similar_neighbors(visible, top_k=limit)
        suggestions = [
            {
                key: row[key]
                for key in (
                    "event_id",
                    "uuid",
                    "date",
                    "title",
                    "labels",
                    "similarity",
                    "group_size",
                )
                if key in row
            }
            for row in diversified
        ]
        return {
            "suggestions": suggestions,
            "read_only": True,
            "writes_performed": 0,
            "search_depth": SEARCH_DEPTH,
            "acl_summary": {
                "retrieved": len(indices[0]),
                "allowed": len(visible),
                "denied": denied,
            },
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = RetrievalRuntime()
    app.state.inference_lock = asyncio.Lock()
    yield


app = FastAPI(title="MISP Suggest", version="0.2.0", lifespan=lifespan)


def require_ready() -> None:
    if not app.state.runtime.ready:
        raise HTTPException(
            status_code=503,
            detail="Retrieval index not built; run `uv run python -m retrieval.build_index`",
        )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {
        "status": "ok" if app.state.runtime.ready else "setup-required",
        "artifacts_ready": app.state.runtime.ready,
        "model_loaded": app.state.runtime._model is not None,
        "read_only": True,
    }


@app.post("/suggest")
async def suggest(request: EventRequest):
    require_ready()
    async with app.state.inference_lock:
        return await asyncio.to_thread(
            app.state.runtime.suggest, request.event, request.limit
        )


@app.post("/similar")
async def similar(
    request: EventRequest,
    x_actor: str | None = Header(default=None),
):
    require_ready()
    if not x_actor:
        raise HTTPException(status_code=401, detail="X-Actor is required")
    async with app.state.inference_lock:
        return await asyncio.to_thread(
            app.state.runtime.similar, request.event, x_actor, request.limit
        )
