#!/usr/bin/env python3
"""Build coarse analyst-training style statistics without retaining event text."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def summary(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in QUANTILES},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    info_words: list[int] = []
    canonical_chars: list[int] = []
    canonical_lines: list[int] = []
    attribute_lines: list[int] = []
    object_lines: list[int] = []
    with args.train.open("rb") as raw:
        source = gzip.open(raw, "rt", encoding="utf-8") if args.train.suffix == ".gz" else raw
        for line in source:
            row = json.loads(line)
            if bool(row.get("feed_derived")):
                continue
            text = str(row.get("text", ""))
            lines = text.splitlines()
            info = lines[0].removeprefix("[INFO] ") if lines else ""
            info_words.append(len(re.findall(r"\w+", info)))
            canonical_chars.append(len(text))
            canonical_lines.append(len(lines))
            attribute_lines.append(sum(value.startswith("[ATTR]") for value in lines))
            object_lines.append(sum(value.startswith("[OBJECT]") for value in lines))
    if len(info_words) < 1000:
        raise ValueError("alignment profile requires at least 1,000 analyst rows")
    report = {
        "schema_version": 1,
        "scope": "analyst-authored training events only",
        "rows": len(info_words),
        "minimum_group_size": 1000,
        "metrics": {
            "info_words": summary(info_words),
            "canonical_chars": summary(canonical_chars),
            "canonical_lines": summary(canonical_lines),
            "attribute_lines": summary(attribute_lines),
            "object_lines": summary(object_lines),
        },
        "source_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "contains_event_text": False,
        "validation_used": False,
        "test_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.chmod(0o600)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
