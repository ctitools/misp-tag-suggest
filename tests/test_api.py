# pylint: disable=protected-access,invalid-name  # fakes mirror PyMISP / http.server names
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np

import app as api
from tagrec.interface import Suggestion


class FakeRetrievalModel:
    model_version = "fake@revision"
    params = {"dataset_manifest_sha256": "manifest"}

    def __init__(self):
        self.metadata = [
            {
                "event_id": "denied",
                "uuid": "d",
                "date": "2025-01-01",
                "title": "Repeated",
                "labels": ["tag:a"],
                "orgc": "org",
                "normalized_info": "same",
                "dup_cluster": "c1",
            },
            {
                "event_id": "allowed-1",
                "uuid": "a1",
                "date": "2025-01-02",
                "title": "Allowed repeated",
                "labels": ["tag:a"],
                "orgc": "org",
                "normalized_info": "same",
                "dup_cluster": "c2",
            },
            {
                "event_id": "allowed-2",
                "uuid": "a2",
                "date": "2025-01-03",
                "title": "Distinct",
                "labels": ["tag:b"],
                "orgc": "other",
                "normalized_info": "different",
                "dup_cluster": "c3",
            },
        ]
        self.accept = True

    def suggest_many(self, rows, top_k=10):
        assert "Tag" not in rows[0]["text"]
        return [[Suggestion("tag:a", 0.9, ("denied",), self.model_version)]]

    def event_acceptance_many(self, rows):
        return [self.accept]

    def _search(self, rows, k=50):
        return (
            np.asarray([[0.99, 0.98, 0.8]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )


def make_ready(tmp_path: Path, monkeypatch):
    for name in ("params.json", "train_embeddings.npz", "index_meta.jsonl"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(api, "ARTIFACTS", tmp_path)
    runtime = api.RetrievalRuntime()
    runtime._model = FakeRetrievalModel()
    return runtime


def test_runtime_suggests_and_abstains(tmp_path: Path, monkeypatch):
    runtime = make_ready(tmp_path, monkeypatch)
    event = {"Event": {"info": "Phishing", "Tag": [{"name": "must-not-leak"}]}}
    result = runtime.suggest(event, 5)
    assert result["suggestions"] == [{"tag": "tag:a", "score": 0.9}]
    assert result["abstained"] is False
    runtime._model.accept = False
    result = runtime.suggest(event, 5)
    assert result["suggestions"] == [] and result["abstained"] is True


def test_runtime_filters_acl_before_diversification(tmp_path: Path, monkeypatch):
    runtime = make_ready(tmp_path, monkeypatch)
    runtime.acl = {"alice": {"allowed-1", "allowed-2"}}
    result = runtime.similar({"Event": {"info": "query"}}, "alice", 10)
    assert [row["event_id"] for row in result["suggestions"]] == [
        "allowed-1",
        "allowed-2",
    ]
    assert result["acl_summary"] == {"retrieved": 3, "allowed": 2, "denied": 1}
    assert result["writes_performed"] == 0


def test_acl_file_and_default_deny(tmp_path: Path, monkeypatch):
    acl = tmp_path / "acl.json"
    acl.write_text(json.dumps({"alice": ["7"]}), encoding="utf-8")
    monkeypatch.setenv("POC_ACL_FILE", str(acl))
    monkeypatch.delenv("POC_ALLOW_ALL_SIMILAR", raising=False)
    runtime = api.RetrievalRuntime()
    assert runtime.authorize("alice", "7")
    assert not runtime.authorize("alice", "8")
    assert not runtime.authorize("bob", "7")


def test_http_health_setup_and_route_contracts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(api, "ARTIFACTS", tmp_path)

    async def scenario():
        async with api.lifespan(api.app):
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/health")
                assert health.status_code == 200
                assert health.json()["status"] == "setup-required"
                response = await client.post("/suggest", json={"event": {}})
                assert response.status_code == 503

                runtime = make_ready(tmp_path, monkeypatch)
                runtime.acl = {"alice": {"allowed-1", "allowed-2"}}
                api.app.state.runtime = runtime
                suggested = await client.post("/suggest", json={"event": {"Event": {"info": "x"}}})
                assert suggested.status_code == 200
                # contract relied on by generic-ai-misp-module (genai/suggest.py): keep stable
                assert set(suggested.json()) == {
                    "suggestions",
                    "abstained",
                    "model_version",
                    "dataset_manifest_sha256",
                }
                assert suggested.json()["suggestions"] == [{"tag": "tag:a", "score": 0.9}]
                response = await client.post("/similar", json={"event": {}})
                assert response.status_code == 401
                similar = await client.post(
                    "/similar",
                    json={"event": {}},
                    headers={"X-Actor": "alice"},
                )
                assert similar.status_code == 200
                assert similar.json()["read_only"] is True

    asyncio.run(scenario())


def test_api_key_is_checked_only_when_configured(tmp_path: Path, monkeypatch):
    async def scenario():
        async with api.lifespan(api.app):
            api.app.state.runtime = make_ready(tmp_path, monkeypatch)
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                body = {"event": {"Event": {"info": "x"}}}
                monkeypatch.setattr(api, "API_KEY", "")
                assert (await client.post("/suggest", json=body)).status_code == 200
                monkeypatch.setattr(api, "API_KEY", "s3cret")
                assert (await client.post("/suggest", json=body)).status_code == 401
                wrong = await client.post("/suggest", json=body, headers={"X-Api-Key": "nope"})
                assert wrong.status_code == 401
                right = await client.post("/suggest", json=body, headers={"X-Api-Key": "s3cret"})
                assert right.status_code == 200
                similar = await client.post(
                    "/similar", json=body, headers={"X-Actor": "alice", "X-Api-Key": "nope"}
                )
                assert similar.status_code == 401

    asyncio.run(scenario())
