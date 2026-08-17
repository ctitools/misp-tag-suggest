from __future__ import annotations

import json
from pathlib import Path

from tagrec.taxonomy import TaxonomyCatalog


def test_catalog_validates_and_normalizes_rsit(tmp_path: Path, taxonomy_snapshot):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(taxonomy_snapshot), encoding="utf-8")
    catalog = TaxonomyCatalog.load(path)
    expected = 'rsit:fraud="phishing"'
    assert catalog.normalize(expected) == (expected, "eligible-and-valid")
    assert catalog.normalize('misp-galaxy:rsit="Fraud:Phishing"') == (
        expected,
        "eligible-and-valid",
    )
    assert catalog.normalize('rsit:fraud="not-in-snapshot"') == (
        None,
        "absent-from-taxonomy-snapshot",
    )


def test_bounded_snapshot_flag_is_preserved(tmp_path: Path, taxonomy_snapshot):
    taxonomy_snapshot["summary"]["bounded_smoke_snapshot"] = True
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(taxonomy_snapshot), encoding="utf-8")
    assert json.loads(path.read_text())["summary"]["bounded_smoke_snapshot"] is True
