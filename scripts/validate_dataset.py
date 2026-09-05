#!/usr/bin/env python3
"""Validate the shared processed dataset and write a Markdown report."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from tagrec.allowlist import decide_tag, tag_namespace
from tagrec.io import iter_jsonl
from tagrec.taxonomy import TaxonomyCatalog

SPLITS = ("train", "val", "test")


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return date.min


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--taxonomy", default="data/taxonomy/snapshot.json")
    parser.add_argument("--out", default="data/processed/validation_report.md")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = TaxonomyCatalog.load(
        args.taxonomy,
        require_enabled_taxonomies=bool(manifest.get("require_enabled_taxonomies")),
    )

    failures: list[str] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    namespace_counts: dict[str, Counter[str]] = {}
    label_counts: Counter[str] = Counter()
    train_label_counts: Counter[str] = Counter()
    date_ranges: dict[str, tuple[date, date] | None] = {}
    feed_counts: dict[str, int] = {}

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        rows = list(iter_jsonl(path))
        rows_by_split[split] = rows
        namespace_counts[split] = Counter()
        feed_counts[split] = sum(bool(row.get("feed_derived")) for row in rows)
        dates = [parse_date(str(row.get("date", ""))) for row in rows]
        date_ranges[split] = (min(dates), max(dates)) if dates else None

        for line_number, row in enumerate(rows, start=1):
            if row.get("split") != split:
                failures.append(f"{path}:{line_number}: record split is {row.get('split')!r}")
            cluster_id = str(row.get("dup_cluster", ""))
            if not cluster_id:
                failures.append(f"{path}:{line_number}: missing dup_cluster")
            cluster_splits[cluster_id].add(split)
            text = str(row.get("text", ""))
            for label in row.get("labels", []) or []:
                label = str(label)
                label_counts[label] += 1
                if split == "train":
                    train_label_counts[label] += 1
                namespace_counts[split][tag_namespace(label)] += 1
                if label not in catalog.valid_tags:
                    failures.append(f"{path}:{line_number}: label absent from snapshot: {label}")
                if not decide_tag(label).eligible:
                    failures.append(f"{path}:{line_number}: label violates allowlist: {label}")
                if label.lower() in text.lower():
                    failures.append(f"{path}:{line_number}: exact label leakage: {label}")

    for cluster_id, splits in cluster_splits.items():
        if len(splits) > 1:
            failures.append(f"Duplicate cluster {cluster_id} spans splits: {sorted(splits)}")

    for earlier, later in zip(SPLITS, SPLITS[1:], strict=False):
        earlier_range, later_range = date_ranges[earlier], date_ranges[later]
        if earlier_range and later_range and earlier_range[1] > later_range[0]:
            failures.append(
                f"Non-monotonic dates: {earlier} max {earlier_range[1]} "
                f"> {later} min {later_range[0]}"
            )

    frequencies = Counter(label_counts.values())
    buckets = {
        "1": sum(count for frequency, count in frequencies.items() if frequency == 1),
        "2-4": sum(count for frequency, count in frequencies.items() if 2 <= frequency <= 4),
        "5-19": sum(count for frequency, count in frequencies.items() if 5 <= frequency <= 19),
        "20-99": sum(count for frequency, count in frequencies.items() if 20 <= frequency <= 99),
        "100+": sum(count for frequency, count in frequencies.items() if frequency >= 100),
    }

    lines = [
        "# Dataset Validation Report",
        "",
        f"Status: **{'PASS' if not failures else 'FAIL'}**",
        "",
        "## Hard checks",
        "",
    ]
    if failures:
        lines.extend(f"- ❌ {failure}" for failure in failures[:200])
        if len(failures) > 200:
            lines.append(f"- … {len(failures) - 200} additional failures omitted")
    else:
        lines.extend(
            [
                "- ✅ No exact tag leakage detected.",
                "- ✅ No duplicate cluster spans multiple splits.",
                "- ✅ Every label exists in the taxonomy snapshot.",
                "- ✅ Split date ranges are monotonic.",
            ]
        )

    lines.extend(
        [
            "",
            "## Split summary",
            "",
            "| Split | Events | Labeled | Feed-derived | Date range |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for split in SPLITS:
        rows = rows_by_split[split]
        labeled = sum(bool(row.get("labels")) for row in rows)
        feed = feed_counts[split]
        date_range = date_ranges[split]
        range_text = f"{date_range[0]} to {date_range[1]}" if date_range else "n/a"
        feed_text = f"{feed} ({feed / len(rows):.1%})" if rows else "0"
        lines.append(f"| {split} | {len(rows)} | {labeled} | {feed_text} | {range_text} |")

    raw_count = int(manifest.get("counts", {}).get("raw_events", 0))
    dedup_count = int(manifest.get("counts", {}).get("deduplicated_representatives", 0))
    shrinkage = 1 - dedup_count / raw_count if raw_count else 0
    lines.extend(
        [
            "",
            "## Deduplication",
            "",
            f"- Raw events: {raw_count}",
            f"- Representatives: {dedup_count}",
            f"- Shrinkage: {shrinkage:.1%}",
            f"- Exact unions: {manifest.get('counts', {}).get('exact_unions', 0)}",
            f"- Near-duplicate unions: {manifest.get('counts', {}).get('near_unions', 0)}",
            "",
            "## Label frequency buckets",
            "",
            "| Examples per label | Labels |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {bucket} | {count} |" for bucket, count in buckets.items())
    trainable = sum(1 for count in train_label_counts.values() if count >= 20)
    lines.extend(
        [
            "",
            f"Labels with ≥ 20 train examples (acceptance criterion): **{trainable}**",
        ]
    )

    lines.extend(["", "## Labels by namespace and split", ""])
    all_namespaces = sorted(set().union(*(counts.keys() for counts in namespace_counts.values())))
    lines.extend(["| Namespace | Train | Validation | Test |", "|---|---:|---:|---:|"])
    for namespace in all_namespaces:
        lines.append(
            f"| `{namespace}` | {namespace_counts['train'][namespace]} | "
            f"{namespace_counts['val'][namespace]} | {namespace_counts['test'][namespace]} |"
        )

    if manifest.get("warning"):
        lines.extend(["", "## Manifest warning", "", f"> {manifest['warning']}"])

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)

    print(f"Validation {'passed' if not failures else 'failed'}: {len(failures)} hard failures")
    print(f"Report: {destination}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
