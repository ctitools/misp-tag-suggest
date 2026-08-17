"""Shared model-independent suggestion interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One validated taxonomy-tag recommendation."""

    tag: str
    score: float
    evidence: tuple[str, ...] = field(default_factory=tuple)
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Suggester(Protocol):
    """Interface implemented by every model track."""

    def suggest(self, event: dict[str, Any], top_k: int = 10) -> list[Suggestion]: ...
