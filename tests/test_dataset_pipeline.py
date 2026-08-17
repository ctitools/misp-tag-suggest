from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from scripts import build_dataset, validate_dataset
from tagrec.io import iter_jsonl


def test_dataset_build_and_validation_end_to_end(
    tmp_path: Path, monkeypatch, taxonomy_snapshot
):
    taxonomy = tmp_path / "data/taxonomy/snapshot.json"
    taxonomy.parent.mkdir(parents=True)
    taxonomy.write_text(json.dumps(taxonomy_snapshot), encoding="utf-8")

    tag = 'rsit:fraud="phishing"'
    events = []
    for number in range(12):
        events.append(
            {
                "Event": {
                    "id": str(number + 1),
                    "uuid": f"00000000-0000-0000-0000-{number:012d}",
                    "date": f"2025-{number + 1:02d}-01",
                    "timestamp": str(1_700_000_000 + number),
                    "info": f"Phishing investigation {number}",
                    "Orgc": {"name": "Example CSIRT"},
                    "Tag": [{"name": tag}],
                    "Attribute": [
                        {
                            "category": "Network activity",
                            "type": "url",
                            "value": f"hxxp://case{number}.example.org/path",
                        }
                    ],
                }
            }
        )
    duplicate = json.loads(json.dumps(events[0]))
    duplicate["Event"]["id"] = "99"
    duplicate["Event"]["uuid"] = "00000000-0000-0000-0000-999999999999"
    duplicate["Event"]["date"] = "2025-12-31"
    events.append(duplicate)

    raw = tmp_path / "data/raw/events.jsonl.gz"
    raw.parent.mkdir(parents=True)
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    processed = tmp_path / "data/processed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset",
            "--raw",
            str(raw),
            "--taxonomy",
            str(taxonomy),
            "--out-dir",
            str(processed),
            "--skip-near-dedup",
        ],
    )
    assert build_dataset.main() == 0

    report = processed / "validation_report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_dataset",
            "--data-dir",
            str(processed),
            "--taxonomy",
            str(taxonomy),
            "--out",
            str(report),
        ],
    )
    assert validate_dataset.main() == 0
    assert "Status: **PASS**" in report.read_text(encoding="utf-8")

    manifest = json.loads((processed / "manifest.json").read_text())
    assert manifest["counts"]["raw_events"] == 13
    assert manifest["counts"]["deduplicated_representatives"] == 12
    assert manifest["counts"]["exact_unions"] == 1
    assert manifest["counts"]["train"] > 0
    assert manifest["counts"]["val"] > 0
    assert manifest["counts"]["test"] > 0

    rows = [
        row
        for split in ("train", "val", "test")
        for row in iter_jsonl(processed / f"{split}.jsonl")
    ]
    assert len(rows) == 12
    assert all(row["labels"] == [tag] for row in rows)
    assert all(tag not in row["text"] for row in rows)
    assert all("Tag" not in row["text"] for row in rows)


def test_builder_rejects_bounded_taxonomy_without_override(
    tmp_path: Path, monkeypatch, taxonomy_snapshot
):
    taxonomy_snapshot["summary"]["bounded_smoke_snapshot"] = True
    taxonomy = tmp_path / "snapshot.json"
    taxonomy.write_text(json.dumps(taxonomy_snapshot), encoding="utf-8")
    raw = tmp_path / "events.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_dataset", "--raw", str(raw), "--taxonomy", str(taxonomy)],
    )
    try:
        build_dataset.main()
    except SystemExit as error:
        assert "bounded smoke-test" in str(error)
    else:
        raise AssertionError("bounded taxonomy was accepted")
