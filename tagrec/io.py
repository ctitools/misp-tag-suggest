"""Small secure file helpers for sensitive MISP-derived data."""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"} if source.suffix == ".gz" else {"encoding": "utf-8"}
    with opener(source, **kwargs) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_json_secure(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_jsonl_secure(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count
