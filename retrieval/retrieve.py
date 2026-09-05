#!/usr/bin/env python3
"""Safe NPZ semantic nearest-event retrieval inference."""

# pylint: disable=import-outside-toplevel  # torch/faiss are imported lazily; tests replace them
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tagrec.canonicalize import canonicalize_event
from tagrec.interface import Suggestion
from tagrec.io import iter_jsonl


class SemanticRetrievalSuggester:
    def __init__(self, artifacts_dir: Path, device: str | None = None):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.artifacts_dir = artifacts_dir
        self.params = json.loads((artifacts_dir / "params.json").read_text(encoding="utf-8"))
        with np.load(artifacts_dir / "train_embeddings.npz", allow_pickle=False) as arrays:
            embeddings = np.asarray(arrays["train"], dtype=np.float32)
        self.metadata = list(iter_jsonl(artifacts_dir / "index_meta.jsonl"))
        if len(self.metadata) != len(embeddings):
            raise ValueError("embedding and metadata counts differ")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        # Faiss owns its copy; retaining the source matrix wastes ~170 MB in service RSS.
        del embeddings
        self.model = SentenceTransformer(
            self.params["model"],
            revision=self.params["revision"],
            trust_remote_code=False,
            device=device,
        )
        self.selected_k = int(self.params["selected_k"])
        self.similarity_threshold = float(self.params["similarity_threshold"])
        self.model_version = f"{self.params['model']}@{self.params['revision'][:12]}"
        self._cached_rows_id: int | None = None
        self._cached_similarities: np.ndarray | None = None
        self._cached_neighbors: np.ndarray | None = None

    def _search(self, rows: list[dict[str, Any]], k: int = 50) -> tuple[np.ndarray, np.ndarray]:
        search_k = min(max(k, self.selected_k, 50), len(self.metadata))
        if self._cached_rows_id == id(rows) and self._cached_neighbors is not None:
            if self._cached_neighbors.shape[1] >= search_k:
                return (
                    self._cached_similarities,
                    self._cached_neighbors,
                )  # type: ignore[return-value]
        embeddings = np.asarray(
            self.model.encode(
                [str(row.get("text", "")) for row in rows],
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        similarities, neighbors = self.index.search(embeddings, search_k)
        self._cached_rows_id = id(rows)
        self._cached_similarities = similarities
        self._cached_neighbors = neighbors
        return similarities, neighbors

    def suggest_many(self, rows: list[dict[str, Any]], top_k: int = 10) -> list[list[Suggestion]]:
        if not rows:
            return []
        similarities, neighbors = self._search(rows)
        output: list[list[Suggestion]] = []
        for row_similarities, row_neighbors in zip(similarities, neighbors, strict=True):
            weights = np.maximum(row_similarities[: self.selected_k], 0.0)
            denominator = float(weights.sum()) or 1.0
            scores: dict[str, float] = defaultdict(float)
            evidence: dict[str, list[str]] = defaultdict(list)
            for neighbor, weight in zip(row_neighbors[: self.selected_k], weights, strict=True):
                if neighbor < 0 or weight <= 0:
                    continue
                metadata = self.metadata[int(neighbor)]
                for label in metadata.get("labels", []):
                    tag = str(label)
                    scores[tag] += float(weight) / denominator
                    if len(evidence[tag]) < 5:
                        evidence[tag].append(str(metadata.get("event_id", "")))
            ranked = [tag for tag, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][
                :top_k
            ]
            output.append(
                [
                    Suggestion(
                        tag=tag,
                        score=scores[tag],
                        evidence=tuple(evidence[tag]),
                        model_version=self.model_version,
                    )
                    for tag in ranked
                ]
            )
        return output

    def event_acceptance_many(self, rows: list[dict[str, Any]]) -> list[bool]:
        """Return whole-event abstention decisions without altering ranking output."""
        if not rows:
            return []
        similarities, _neighbors = self._search(rows)
        return [float(row[0]) >= self.similarity_threshold for row in similarities]

    def candidate_tags_many(
        self, rows: list[dict[str, Any]], neighbor_count: int = 50
    ) -> list[set[str]]:
        _similarities, neighbors = self._search(rows, neighbor_count)
        return [
            {
                str(label)
                for neighbor in row_neighbors[:neighbor_count]
                if neighbor >= 0
                for label in self.metadata[int(neighbor)].get("labels", [])
            }
            for row_neighbors in neighbors
        ]

    def suggest(self, event: dict[str, Any], top_k: int = 10) -> list[Suggestion]:
        return self.suggest_many([event], top_k=top_k)[0]


def load_suggester(artifacts_dir: Path) -> SemanticRetrievalSuggester:
    return SemanticRetrievalSuggester(artifacts_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    raw: Any = json.loads(args.event.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Input must be a JSON object")
    row = dict(raw)
    if not row.get("text"):
        row["text"] = canonicalize_event(raw)
    suggester = SemanticRetrievalSuggester(args.artifacts, device=args.device)
    suggestions = suggester.suggest(row, top_k=args.top_k)
    print(json.dumps({"suggestions": [value.to_dict() for value in suggestions]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
