from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from experiments.lfm25.admit_synthetic import leakage_errors
from experiments.lfm25.audit_synthetic_alignment import generated_canonical
from experiments.lfm25.build_real_pairs import partial_text
from experiments.lfm25.metrics import (
    candidate_recall_metrics,
    paired_bootstrap_delta,
    ranking_metrics,
    weighted_vote_predictions,
)
from experiments.lfm25.validate_synthetic import validate_bundle


def test_weighted_votes_and_metrics_are_deterministic():
    similarities = np.asarray([[0.9, 0.5], [0.8, 0.2]], dtype=np.float32)
    neighbors = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    train_labels = [["rsit:fraud=\"phishing\""], ["misp-galaxy:malpedia=\"Example\""]]
    predictions = weighted_vote_predictions(
        similarities, neighbors, train_labels, neighbor_count=2, top_k=3
    )
    assert predictions[0][0] == "rsit:fraud=\"phishing\""
    truth = [
        {"rsit:fraud=\"phishing\""},
        {"misp-galaxy:malpedia=\"Example\""},
    ]
    ranking, rows = ranking_metrics(truth, predictions)
    assert ranking["precision@3"] == 1 / 3
    assert ranking["recall@3"] == 1.0
    assert rows["precision@3"].shape == (2,)
    candidates = candidate_recall_metrics(truth, neighbors, train_labels)
    assert candidates["macro_recall"] == 1.0


def test_paired_bootstrap_requires_and_preserves_pairing():
    baseline = np.asarray([0.0, 0.25, 0.5], dtype=np.float32)
    challenger = baseline + 0.1
    result = paired_bootstrap_delta(baseline, challenger, samples=100, seed=7)
    assert abs(float(result["delta"]) - 0.1) < 1e-6
    assert float(result["ci95_low"]) > 0


def test_partial_text_preserves_info_and_first_evidence_quarter():
    text = "\n".join(
        ["[INFO] fictional incident"] + [f"[ATTR] evidence-{i}" for i in range(8)]
    )
    partial = partial_text(text)
    assert partial.splitlines() == [
        "[INFO] fictional incident",
        "[ATTR] evidence-0",
        "[ATTR] evidence-1",
    ]


def _valid_bundle() -> dict:
    target = 'rsit:fraud="phishing"'
    confuser = 'rsit:fraud="masquerade"'
    candidates = [target, confuser, 'rsit:fraud="other"']

    def scenario(variant: str, relevance: int, expected: list[str], evidence: list[str]):
        return {
            "scenario_id": f"syn-12345678-{variant.replace('_', '-')}",
            "variant": variant,
            "expected_relevance": relevance,
            "expected_tags": expected,
            "evidence_paths": evidence,
            "event": {
                "info": f"Fictional defensive report for {variant} with sufficient detail.",
                "attributes": [
                    {
                        "category": "Network activity",
                        "type": "domain",
                        "value": f"{variant.replace('_', '-')}.example",
                    }
                ],
                "objects": [],
            },
        }

    evidence = ["event.attributes.0.value"]
    return {
        "schema_version": 1,
        "bundle_id": "syn-12345678",
        "language": "en",
        "difficulty": "ordinary",
        "target_tag": target,
        "candidate_tags": candidates,
        "scenarios": [
            scenario("explicit_positive", 2, [target], evidence),
            scenario("implicit_positive", 2, [target], evidence),
            scenario("partial_positive", 1, [target], evidence),
            scenario("same_family_confuser", 0, [confuser], evidence),
            scenario("unrelated_negative", 0, [], []),
            scenario("insufficient_evidence", 0, [], []),
        ],
    }


def test_synthetic_bundle_validator_accepts_only_reserved_indicators():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "experiments/lfm25/synthetic_bundle.schema.json").read_text()
    )
    bundle = _valid_bundle()
    assert validate_bundle(bundle, schema=schema, valid_tags=set(bundle["candidate_tags"])) == []

    invalid = copy.deepcopy(bundle)
    invalid["scenarios"][0]["event"]["attributes"][0]["value"] = "public.example.com"
    errors = validate_bundle(invalid, schema=schema)
    assert any("not under .example" in error for error in errors)


def test_synthetic_validator_allows_shell_observation_but_rejects_command_syntax():
    observed = _valid_bundle()
    observed["scenarios"][0]["event"]["info"] = (
        "Defensive endpoint telemetry showed cmd.exe launching an unexpected child process."
    )
    assert validate_bundle(observed) == []

    operational = copy.deepcopy(observed)
    operational["scenarios"][0]["event"]["info"] = (
        "The report contained an executable command: cmd.exe /c whoami."
    )
    errors = validate_bundle(operational)
    assert any("command-like content" in error for error in errors)


def test_synthetic_admission_leakage_check_ignores_generic_other_only():
    generic = _valid_bundle()
    generic["target_tag"] = 'cssa:origin="other"'
    generic["scenarios"][1]["event"]["info"] = "Other provenance remains uncategorized."
    assert leakage_errors(generic) == []

    leaked = _valid_bundle()
    leaked["scenarios"][1]["event"]["info"] = "Exact phishing target appears here."
    assert leakage_errors(leaked) == ["exact target value leaked in implicit_positive"]


def test_generated_event_uses_production_canonical_markers():
    event = _valid_bundle()["scenarios"][0]["event"]
    text = generated_canonical(event)
    assert text.startswith("[INFO] ")
    assert "[ATTR] Network activity/domain:" in text


def test_synthetic_bundle_validator_rejects_label_contract_mutation():
    bundle = _valid_bundle()
    bundle["scenarios"][0]["expected_tags"] = [bundle["candidate_tags"][1]]
    errors = validate_bundle(bundle)
    assert any("positive contract" in error for error in errors)
