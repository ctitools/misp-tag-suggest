#!/usr/bin/env python3
"""Fail-closed structural and fictional-indicator validation for synthetic bundles."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

VARIANTS = {
    "explicit_positive",
    "implicit_positive",
    "partial_positive",
    "same_family_confuser",
    "unrelated_negative",
    "insufficient_evidence",
}
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])")
_HEX64_RE = re.compile(r"\b[0-9a-f]{64}\b", re.IGNORECASE)
_CVE_RE = re.compile(r"\bCVE-[0-9]{4}-[0-9]{4,}\b", re.IGNORECASE)
_PUBLIC_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|ru|cn|io|co|xyz|top|info|biz|de|fr|es|uk)\b",
    re.IGNORECASE,
)
_OPERATIONAL_RE = re.compile(
    r"```|-----BEGIN [A-Z ]+PRIVATE KEY-----|\b(?:password|passwd|api[_ -]?key)\s*[:=]"
    r"|\b(?:powershell|cmd\.exe|/bin/(?:ba)?sh)\s+(?:-[A-Za-z]+\s+)*[-/A-Za-z0-9]"
    r"|\b(?:curl|wget)\s+https?://",
    re.IGNORECASE,
)


def _is_example_domain(value: str) -> bool:
    name = value.rstrip(".").lower()
    return name == "example" or name.endswith(".example")


def _is_documentation_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    networks = (
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
        ipaddress.ip_network("2001:db8::/32"),
    )
    return any(address in network for network in networks)


def _path_exists(scenario: dict[str, Any], path: str) -> bool:
    current: Any = scenario
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False
    return isinstance(current, (str, int, float, bool)) and current not in ("", None)


def _indicator_error(indicator_type: str, value: str) -> str | None:
    if indicator_type in {"domain", "hostname"}:
        return None if _is_example_domain(value) else "domain is not under .example"
    if indicator_type == "url":
        parsed = urlparse(value)
        return (
            None
            if parsed.scheme in {"http", "https"}
            and parsed.hostname
            and _is_example_domain(parsed.hostname)
            else "URL does not use http(s) with a .example host"
        )
    if indicator_type in {"ip-src", "ip-dst"}:
        return None if _is_documentation_ip(value) else "IP is not from a documentation range"
    if indicator_type in {"email-src", "email-dst"}:
        local, separator, domain = value.rpartition("@")
        return (
            None
            if separator and local and _is_example_domain(domain)
            else "email does not use a .example domain"
        )
    if indicator_type == "sha256":
        return (
            None
            if _SHA256_RE.fullmatch(value) and value.lower().startswith("0" * 32)
            else "SHA-256 must be a 64-hex fictional placeholder beginning with 32 zeroes"
        )
    return None


def _event_strings(event: dict[str, Any]) -> list[tuple[str, str]]:
    values = [("info", str(event.get("info", "")))]
    for index, attribute in enumerate(event.get("attributes", []) or []):
        indicator_type = str(attribute.get("type", ""))
        if indicator_type not in {"domain", "hostname", "url", "ip-src", "ip-dst", "email-src", "email-dst", "sha256", "filename"}:
            values.append((f"attributes.{index}.value", str(attribute.get("value", ""))))
        if "comment" in attribute:
            values.append((f"attributes.{index}.comment", str(attribute.get("comment", ""))))
    for object_index, item in enumerate(event.get("objects", []) or []):
        for attr_index, attribute in enumerate(item.get("attributes", []) or []):
            values.append(
                (
                    f"objects.{object_index}.attributes.{attr_index}.value",
                    str(attribute.get("value", "")),
                )
            )
    return values


def _free_text_errors(value: str) -> list[str]:
    errors: list[str] = []
    for match in _URL_RE.findall(value):
        parsed = urlparse(match.rstrip(".,;:!?)"))
        if not parsed.hostname or not _is_example_domain(parsed.hostname):
            errors.append("free text contains a non-.example URL")
    for match in _EMAIL_RE.findall(value):
        if not _is_example_domain(match.rpartition("@")[2]):
            errors.append("free text contains a non-.example email")
    for match in _IPV4_RE.findall(value):
        if not _is_documentation_ip(match):
            errors.append("free text contains a non-documentation IPv4 address")
    for match in _HEX64_RE.findall(value):
        if not match.lower().startswith("0" * 32):
            errors.append("free text contains a non-placeholder SHA-256")
    for match in _PUBLIC_DOMAIN_RE.findall(value):
        if not _is_example_domain(match):
            errors.append("free text contains a public-looking domain")
    if _CVE_RE.search(value):
        errors.append("generated event must not cite a real CVE identifier")
    if _OPERATIONAL_RE.search(value):
        errors.append("generated event contains credential, code, or command-like content")
    return sorted(set(errors))


def validate_seed_contract(bundle: dict[str, Any], seed: dict[str, Any]) -> list[str]:
    """Ensure model output did not mutate any locally frozen label decision."""
    errors: list[str] = []
    for key in ("bundle_id", "language", "difficulty", "target_tag", "candidate_tags"):
        if bundle.get(key) != seed.get(key):
            errors.append(f"seed contract changed: {key}")
    contracts = seed.get("fixed_label_contract", {}) or {}
    by_variant = {
        str(scenario.get("variant", "")): scenario
        for scenario in bundle.get("scenarios", []) or []
    }
    for variant, contract in contracts.items():
        scenario = by_variant.get(str(variant))
        if scenario is None:
            errors.append(f"seed contract missing variant: {variant}")
            continue
        if scenario.get("expected_tags") != contract.get("expected_tags"):
            errors.append(f"seed contract changed expected_tags: {variant}")
        if scenario.get("expected_relevance") != contract.get("relevance"):
            errors.append(f"seed contract changed relevance: {variant}")
    return errors


def validate_bundle(
    bundle: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    valid_tags: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if schema is not None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:  # pragma: no cover - command-line environment guard
            raise RuntimeError("install the training extra to validate JSON Schema") from exc
        validator = Draft202012Validator(schema)
        for error in sorted(validator.iter_errors(bundle), key=lambda item: list(item.path)):
            location = ".".join(map(str, error.path)) or "$"
            errors.append(f"schema:{location}: {error.message}")
        if errors:
            return errors

    target = str(bundle.get("target_tag", ""))
    candidates = list(map(str, bundle.get("candidate_tags", []) or []))
    if target not in candidates:
        errors.append("target_tag must occur in candidate_tags")
    if valid_tags is not None:
        unknown = sorted(set(candidates) - valid_tags)
        if unknown:
            errors.append(f"candidate tags absent from supplied catalog: {unknown[:3]}")

    scenarios = bundle.get("scenarios", []) or []
    observed_variants = [str(row.get("variant", "")) for row in scenarios]
    if set(observed_variants) != VARIANTS or len(observed_variants) != len(VARIANTS):
        errors.append("scenarios must contain each required variant exactly once")
    scenario_ids = [str(row.get("scenario_id", "")) for row in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        errors.append("scenario_id values must be unique")

    normalized_events: set[str] = set()
    for index, scenario in enumerate(scenarios):
        prefix = f"scenarios[{index}]"
        variant = str(scenario.get("variant", ""))
        expected = list(map(str, scenario.get("expected_tags", []) or []))
        relevance = scenario.get("expected_relevance")
        evidence = list(map(str, scenario.get("evidence_paths", []) or []))
        if not set(expected).issubset(candidates):
            errors.append(f"{prefix}: expected_tags must be candidate-only")
        if variant in {"explicit_positive", "implicit_positive"}:
            if expected != [target] or relevance != 2 or not evidence:
                errors.append(f"{prefix}: positive contract must be target/relevance-2 with evidence")
        elif variant == "partial_positive":
            if expected != [target] or relevance != 1 or not evidence:
                errors.append(f"{prefix}: partial contract must be target/relevance-1 with evidence")
        elif variant == "same_family_confuser":
            if target in expected or not expected or relevance != 0 or not evidence:
                errors.append(f"{prefix}: confuser must support a non-target candidate at relevance 0")
        elif variant in {"unrelated_negative", "insufficient_evidence"}:
            if expected or relevance != 0 or evidence:
                errors.append(f"{prefix}: negative/abstention contract must have no tag or evidence")

        for path in evidence:
            if not _path_exists(scenario, path):
                errors.append(f"{prefix}: evidence path does not resolve: {path}")
        event = scenario.get("event", {}) or {}
        normalized = json.dumps(event, sort_keys=True, ensure_ascii=False).casefold()
        event_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if event_hash in normalized_events:
            errors.append(f"{prefix}: duplicate event within bundle")
        normalized_events.add(event_hash)
        if _UUID_RE.search(normalized):
            errors.append(f"{prefix}: UUID-like value is forbidden")
        for location, value in _event_strings(event):
            for problem in _free_text_errors(value):
                errors.append(f"{prefix}.event.{location}: {problem}")
        for attr_index, attribute in enumerate(event.get("attributes", []) or []):
            indicator_type = str(attribute.get("type", ""))
            value = str(attribute.get("value", ""))
            problem = _indicator_error(indicator_type, value)
            if problem:
                errors.append(f"{prefix}.event.attributes[{attr_index}]: {problem}")
        for object_index, item in enumerate(event.get("objects", []) or []):
            for attr_index, attribute in enumerate(item.get("attributes", []) or []):
                indicator_type = str(attribute.get("type", ""))
                value = str(attribute.get("value", ""))
                problem = _indicator_error(indicator_type, value)
                if problem:
                    errors.append(
                        f"{prefix}.event.objects[{object_index}].attributes[{attr_index}]: {problem}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("synthetic_bundle.schema.json"),
    )
    parser.add_argument("--valid-tags", type=Path, help="optional JSON list of candidate tags")
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    valid_tags = (
        set(map(str, json.loads(args.valid_tags.read_text(encoding="utf-8"))))
        if args.valid_tags
        else None
    )
    failures = 0
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            errors = validate_bundle(value, schema=schema, valid_tags=valid_tags)
            for error in errors:
                print(f"{args.input}:{line_number}: {error}")
            failures += bool(errors)
    print(f"validated with {failures} failing bundles")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
