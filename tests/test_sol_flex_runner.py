from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.lfm25.sol_flex_runner import (
    EmergencyHalt,
    HTTPResult,
    MODEL,
    SERVICE_TIER,
    make_request,
    run,
    strict_response_schema,
)


def seed(number: int = 1) -> dict:
    target = 'rsit:fraud="phishing"'
    confuser = 'rsit:fraud="masquerade"'
    candidates = [target, confuser, 'rsit:fraud="other"']
    return {
        "schema_version": 1,
        "bundle_id": f"syn-testcase{number:02d}",
        "language": "en",
        "difficulty": "ordinary",
        "target_tag": target,
        "target_family": "incident-class",
        "target_frequency_bucket": "20-99",
        "candidate_tags": candidates,
        "candidate_documents": {tag: f"Public definition for candidate {i}." for i, tag in enumerate(candidates)},
        "fixed_label_contract": {
            "explicit_positive": {"expected_tags": [target], "relevance": 2},
            "implicit_positive": {"expected_tags": [target], "relevance": 2},
            "partial_positive": {"expected_tags": [target], "relevance": 1},
            "same_family_confuser": {"expected_tags": [confuser], "relevance": 0},
            "unrelated_negative": {"expected_tags": [], "relevance": 0},
            "insufficient_evidence": {"expected_tags": [], "relevance": 0},
        },
    }


def bundle(value: dict) -> dict:
    scenarios = []
    for variant, contract in value["fixed_label_contract"].items():
        evidence = [] if not contract["expected_tags"] else ["event.attributes.0.value"]
        scenarios.append(
            {
                "scenario_id": value["bundle_id"] + "-" + variant.replace("_", "-"),
                "variant": variant,
                "expected_relevance": contract["relevance"],
                "expected_tags": contract["expected_tags"],
                "evidence_paths": evidence,
                "event": {
                    "info": f"Fictional defensive scenario {variant} number {value['bundle_id']} with enough distinct explanatory detail.",
                    "attributes": [
                        {
                            "category": "Network activity",
                            "type": "domain",
                            "value": f"{variant.replace('_', '-')}.example",
                            "comment": "",
                        }
                    ],
                    "objects": [],
                },
            }
        )
    return {
        "schema_version": 1,
        "bundle_id": value["bundle_id"],
        "language": value["language"],
        "difficulty": value["difficulty"],
        "target_tag": value["target_tag"],
        "candidate_tags": value["candidate_tags"],
        "scenarios": scenarios,
    }


def response_for(value: dict, **overrides) -> HTTPResult:
    body = {
        "id": "resp_synthetic_test",
        "model": MODEL,
        "service_tier": SERVICE_TIER,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": json.dumps(bundle(value))}
                ],
            }
        ],
        "usage": {
            "input_tokens": 3000,
            "input_tokens_details": {"cached_tokens": 2000, "cache_write_tokens": 0},
            "output_tokens": 1000,
        },
    }
    body.update(overrides)
    return HTTPResult(200, json.dumps(body).encode(), {})


def paths(tmp_path: Path) -> dict:
    return {
        "output": tmp_path / "bundles.jsonl",
        "ledger_path": tmp_path / "ledger.jsonl",
        "brake_path": tmp_path / "HALTED.json",
    }


def schema() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "experiments/lfm25/synthetic_bundle.schema.json").read_text())


def test_request_is_exact_sol_flex_explicit_cache_and_strict_schema():
    request, body, prompt_hash, schema_hash = make_request(
        seed(), strict_response_schema(schema()), max_output_tokens=6000
    )
    assert request["model"] == "gpt-5.6-sol"
    assert request["service_tier"] == "flex"
    assert request["store"] is False
    assert request["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    block = request["input"][0]["content"][0]
    assert block["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert len(block["text"].encode()) > 8_000
    assert request["text"]["format"]["strict"] is True
    assert len(body) > len(block["text"])
    assert len(prompt_hash) == len(schema_hash) == 64


def test_refusal_on_n_prevents_n_plus_one(tmp_path: Path):
    values = [seed(1), seed(2), seed(3)]
    calls = []

    def transport(body, headers, timeout):
        calls.append((body, headers, timeout))
        if len(calls) == 1:
            return response_for(values[0])
        refusal = json.loads(response_for(values[1]).body)
        refusal["output"][0]["content"] = [
            {"type": "refusal", "refusal": "I cannot comply with this request."}
        ]
        return HTTPResult(200, json.dumps(refusal).encode(), {})

    with pytest.raises(EmergencyHalt, match="refusal"):
        run(
            values,
            schema=schema(),
            stage="canary",
            budget_usd=2,
            transport=transport,
            sleep=lambda _: None,
            **paths(tmp_path),
        )
    assert len(calls) == 2
    assert (tmp_path / "HALTED.json").exists()
    assert len((tmp_path / "bundles.jsonl").read_text().splitlines()) == 1


def test_content_filter_prevents_later_request(tmp_path: Path):
    values = [seed(1), seed(2)]
    calls = 0

    def transport(body, headers, timeout):
        nonlocal calls
        calls += 1
        filtered = json.loads(response_for(values[0]).body)
        filtered["status"] = "incomplete"
        filtered["incomplete_details"] = {"reason": "content_filter"}
        return HTTPResult(200, json.dumps(filtered).encode(), {})

    with pytest.raises(EmergencyHalt, match="content_filter"):
        run(values, schema=schema(), stage="canary", budget_usd=2, transport=transport, **paths(tmp_path))
    assert calls == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda response: response.update(service_tier="default"), "service tier"),
        (lambda response: response["output"][0]["content"][0].update(text="not json"), "malformed JSON"),
        (lambda response: response.update(model="gpt-5.6-luna"), "response model"),
    ],
)
def test_malformed_or_unexpected_response_halts(tmp_path: Path, mutation, message):
    value = seed()

    def transport(body, headers, timeout):
        response = json.loads(response_for(value).body)
        mutation(response)
        return HTTPResult(200, json.dumps(response).encode(), {})

    with pytest.raises(EmergencyHalt, match=message):
        run([value], schema=schema(), stage="canary", budget_usd=2, transport=transport, **paths(tmp_path))
    assert (tmp_path / "HALTED.json").exists()


def test_privacy_and_budget_fail_before_http(tmp_path: Path):
    calls = 0

    def transport(body, headers, timeout):
        nonlocal calls
        calls += 1
        return response_for(seed())

    private = seed()
    private["event_id"] = 1853
    with pytest.raises(EmergencyHalt, match="privacy gate"):
        run([private], schema=schema(), stage="canary", budget_usd=2, transport=transport, **paths(tmp_path))
    assert calls == 0

    other = tmp_path / "budget"
    with pytest.raises(EmergencyHalt, match="budget reservation"):
        run([seed()], schema=schema(), stage="canary", budget_usd=0.001, transport=transport, **paths(other))
    assert calls == 0


def test_only_documented_flex_errors_retry(tmp_path: Path):
    value = seed()
    results = [
        HTTPResult(429, b'{"error":{"message":"Resource Unavailable"}}', {}),
        response_for(value),
    ]

    def transport(body, headers, timeout):
        return results.pop(0)

    report = run(
        [value], schema=schema(), stage="canary", budget_usd=2, transport=transport,
        sleep=lambda _: None, **paths(tmp_path)
    )
    assert report["http_attempts"] == 2
    assert report["accepted"] == 1


def test_policy_http_error_is_never_retried(tmp_path: Path):
    calls = 0

    def transport(body, headers, timeout):
        nonlocal calls
        calls += 1
        return HTTPResult(400, b'{"error":{"message":"safety policy violation"}}', {})

    with pytest.raises(EmergencyHalt, match="policy-related"):
        run([seed()], schema=schema(), stage="canary", budget_usd=2, transport=transport, **paths(tmp_path))
    assert calls == 1


def test_seed_contract_mutation_halts(tmp_path: Path):
    value = seed()

    def transport(body, headers, timeout):
        invalid = bundle(value)
        invalid["candidate_tags"] = list(reversed(invalid["candidate_tags"]))
        response = json.loads(response_for(value).body)
        response["output"][0]["content"][0]["text"] = json.dumps(invalid)
        return HTTPResult(200, json.dumps(response).encode(), {})

    with pytest.raises(EmergencyHalt, match="seed contract"):
        run([value], schema=schema(), stage="canary", budget_usd=2, transport=transport, **paths(tmp_path))
