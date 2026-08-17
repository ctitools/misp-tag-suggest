from __future__ import annotations

import gzip
import json
from pathlib import Path

from tagrec.allowlist import decide_tag, parse_tag, tag_namespace
from tagrec.canonicalize import canonicalize_event, normalize_attribute_value
from tagrec.diversify import collapse_similar_neighbors
from tagrec.families import label_family
from tagrec.io import iter_jsonl


def test_ioc_normalization():
    assert normalize_attribute_value("sha256", "a" * 64) == "<HASH:sha256>"
    assert normalize_attribute_value("ip-src", "192.0.2.1") == "<IP>"
    assert normalize_attribute_value("ip-src|port", "192.0.2.1|443") == "<IP>:443"
    assert normalize_attribute_value("url", "hxxps://evil.example.com/a") == "<URL:example.com>"
    assert normalize_attribute_value("domain", "evil.example.com") == "<DOMAIN:example.com>"
    assert normalize_attribute_value("email-src", "user@evil.example.com") == "<EMAIL:example.com>"
    assert normalize_attribute_value("filename", "invoice.docx") == "<FILE:.docx>"


def test_canonicalization_excludes_tags_and_galaxies():
    event = {
        "Event": {
            "info": "ClickFix malware delivery",
            "Tag": [{"name": 'rsit:malicious-code="malware-distribution"'}],
            "Galaxy": [{"GalaxyCluster": [{"tag_name": 'misp-galaxy:malpedia="Secret"'}]}],
            "EventReport": [{"id": "1", "name": "Analysis", "content": "Uses **PowerShell**."}],
            "Attribute": [{"category": "Network activity", "type": "url", "value": "hxxp://payload.example.org/a"}],
        }
    }
    text = canonicalize_event(event)
    assert "[INFO] ClickFix malware delivery" in text
    assert "[REPORT] Analysis — Uses PowerShell ." in text
    assert "<URL:example.org>" in text
    assert "rsit:" not in text
    assert "Secret" not in text


def test_allowlist_keeps_descriptive_and_rejects_policy_tags():
    assert decide_tag('rsit:fraud="phishing"').eligible
    assert decide_tag('misp-galaxy:mitre-attack-pattern="PowerShell - T1059.001"').eligible
    assert not decide_tag("tlp:white").eligible
    assert not decide_tag('source:feed="example"').eligible
    assert not decide_tag('misp-galaxy:threat-actor="Example"').eligible
    assert not decide_tag('ncsc-nl-ndn:feed="generic"').eligible


def test_tag_parsing_namespace_and_family_classification():
    parsed = parse_tag('rsit:fraud="phishing"')
    assert parsed and parsed.namespace == "rsit" and parsed.value == "phishing"
    assert tag_namespace('misp-galaxy:sector="Energy"') == "misp-galaxy:sector"
    assert label_family('rsit:fraud="phishing"') == "incident-class"
    assert label_family('misp-galaxy:malpedia="Emotet"') == "malware-family"
    assert label_family('misp-galaxy:sector="Finance"') == "context"


def test_diversification_collapses_template_or_duplicate_cluster():
    rows = [
        {"event_id": "1", "orgc": "a", "normalized_info": "x", "dup_cluster": "c1"},
        {"event_id": "2", "orgc": "a", "normalized_info": "x", "dup_cluster": "c2"},
        {"event_id": "3", "orgc": "b", "normalized_info": "y", "dup_cluster": "c1"},
        {"event_id": "4", "orgc": "b", "normalized_info": "z", "dup_cluster": "c4"},
    ]
    output = collapse_similar_neighbors(rows, top_k=10)
    assert [row["event_id"] for row in output] == ["1", "4"]
    assert output[0]["group_size"] == 2


def test_jsonl_reader_supports_plain_and_gzip(tmp_path: Path):
    rows = [{"id": 1}, {"id": 2}]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    plain = tmp_path / "rows.jsonl"
    compressed = tmp_path / "rows.jsonl.gz"
    plain.write_text(content, encoding="utf-8")
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write(content)
    assert list(iter_jsonl(plain)) == rows
    assert list(iter_jsonl(compressed)) == rows
