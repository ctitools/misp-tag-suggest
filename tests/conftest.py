from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


class FakeIndexFlatIP:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.values = np.empty((0, dimensions), dtype=np.float32)

    def add(self, values):
        values = np.asarray(values, dtype=np.float32)
        assert values.ndim == 2 and values.shape[1] == self.dimensions
        self.values = values.copy()

    def search(self, queries, k: int):
        queries = np.asarray(queries, dtype=np.float32)
        scores = queries @ self.values.T
        take = min(k, len(self.values))
        order = np.argsort(-scores, axis=1)[:, :take]
        ranked = np.take_along_axis(scores, order, axis=1).astype(np.float32)
        if take < k:
            order = np.pad(order, ((0, 0), (0, k - take)), constant_values=-1)
            ranked = np.pad(ranked, ((0, 0), (0, k - take)), constant_values=-np.inf)
        return ranked, order.astype(np.int64)


class FakeSentenceTransformer:
    def __init__(self, model, revision=None, trust_remote_code=False, device=None):
        self.model = model
        self.revision = revision
        self.device = device
        assert trust_remote_code is False

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lower = text.lower()
        digest = hashlib.sha256(text.encode()).digest()
        return np.asarray(
            [
                2.0 if "phish" in lower else digest[0] / 255,
                2.0 if "malware" in lower else digest[1] / 255,
                2.0 if "ddos" in lower else digest[2] / 255,
                1.0,
            ],
            dtype=np.float32,
        )

    def encode(self, texts, normalize_embeddings=True, **_kwargs):
        values = np.stack([self._vector(str(text)) for text in texts])
        if normalize_embeddings:
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values.astype(np.float32)


@pytest.fixture
def fake_ml_modules(monkeypatch):
    faiss = types.ModuleType("faiss")
    faiss.IndexFlatIP = FakeIndexFlatIP
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "faiss", faiss)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    return faiss, sentence_transformers


@pytest.fixture
def taxonomy_snapshot() -> dict:
    tag = 'rsit:fraud="phishing"'
    return {
        "schema_version": 2,
        "snapshot_date": "2026-01-01T00:00:00+00:00",
        "taxonomies": [
            {
                "id": "1",
                "namespace": "rsit",
                "description": "Reference Security Incident Taxonomy",
                "enabled": True,
                "entries": [
                    {
                        "tag": tag,
                        "expanded": "Fraud: Phishing",
                        "description": "Phishing",
                        "exclusive_predicate": False,
                        "existing_tag": True,
                    }
                ],
            }
        ],
        "galaxies": [],
        "custom_tags": [],
        "summary": {
            "enabled_taxonomy_count": 1,
            "enabled_galaxy_count": 0,
            "bounded_smoke_snapshot": False,
        },
    }


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
