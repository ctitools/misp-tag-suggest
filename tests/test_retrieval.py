# pylint: disable=protected-access  # exercises the fake encoder directly
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from conftest import write_jsonl

from retrieval import build_index
from retrieval.retrieve import SemanticRetrievalSuggester
from tagrec.io import iter_jsonl


def test_index_builder_creates_complete_portable_artifacts(
    tmp_path: Path, monkeypatch, fake_ml_modules
):
    data = tmp_path / "processed"
    data.mkdir()
    train = [
        {
            "event_id": 1,
            "uuid": "u1",
            "date": "2025-01-01",
            "orgc": "A",
            "dup_cluster": "c1",
            "text": "[INFO] Phishing campaign 100",
            "labels": ['rsit:fraud="phishing"'],
        },
        {
            "event_id": 2,
            "uuid": "u2",
            "date": "2025-01-02",
            "orgc": "A",
            "dup_cluster": "c2",
            "text": "[INFO] Malware campaign abcdef123456",
            "labels": ['rsit:malicious-code="malware-distribution"'],
        },
        {
            "event_id": 3,
            "uuid": "u3",
            "date": "2025-01-03",
            "orgc": "B",
            "dup_cluster": "c3",
            "text": "[INFO] DDoS campaign",
            "labels": ['ddos:type="flooding-attack"'],
        },
    ]
    validation = [
        {"uuid": "v1", "text": "[INFO] New phishing message", "labels": train[0]["labels"]},
        {"uuid": "v2", "text": "[INFO] New malware", "labels": train[1]["labels"]},
    ]
    write_jsonl(data / "train.jsonl", train)
    write_jsonl(data / "val.jsonl", validation)
    (data / "manifest.json").write_text(
        json.dumps({"taxonomy_snapshot_date": "2026-01-01"}), encoding="utf-8"
    )
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_index",
            "--data-dir",
            str(data),
            "--artifacts",
            str(artifacts),
            "--device",
            "cpu",
        ],
    )
    assert build_index.main() == 0
    assert {path.name for path in artifacts.iterdir()} == {
        "train_embeddings.npz",
        "index_meta.jsonl",
        "params.json",
        "training_metrics.json",
    }
    with np.load(artifacts / "train_embeddings.npz", allow_pickle=False) as values:
        assert values["train"].shape == (3, 4)  # pylint: disable=no-member
    metadata = list(iter_jsonl(artifacts / "index_meta.jsonl"))
    assert metadata[0]["title"] == "Phishing campaign 100"
    assert metadata[0]["normalized_info"] == "[INFO] Phishing campaign <N>"
    assert metadata[0]["orgc"] == "A"
    assert metadata[0]["dup_cluster"] == "c1"
    assert (
        json.loads((artifacts / "params.json").read_text())["revision"]
        == build_index.DEFAULT_REVISION
    )


def test_retriever_votes_labels_and_applies_event_abstention(tmp_path: Path, fake_ml_modules):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    vectors = np.stack(
        [
            fake_ml_modules[1].SentenceTransformer._vector("phishing"),
            fake_ml_modules[1].SentenceTransformer._vector("malware"),
        ]
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with (artifacts / "train_embeddings.npz").open("wb") as handle:
        np.savez_compressed(handle, train=vectors.astype(np.float32))
    write_jsonl(
        artifacts / "index_meta.jsonl",
        [
            {"event_id": 1, "labels": ['rsit:fraud="phishing"']},
            {"event_id": 2, "labels": ['rsit:malicious-code="malware-distribution"']},
        ],
    )
    (artifacts / "params.json").write_text(
        json.dumps(
            {
                "model": "fake",
                "revision": "abc1234567890",
                "selected_k": 1,
                "similarity_threshold": 0.8,
            }
        ),
        encoding="utf-8",
    )
    model = SemanticRetrievalSuggester(artifacts, device="cpu")
    rows = [{"text": "obvious phishing email"}]
    output = model.suggest_many(rows, top_k=2)[0]
    assert output[0].tag == 'rsit:fraud="phishing"'
    assert output[0].evidence == ("1",)
    assert model.event_acceptance_many(rows) == [True]
    assert model.candidate_tags_many(rows, neighbor_count=1) == [{'rsit:fraud="phishing"'}]
