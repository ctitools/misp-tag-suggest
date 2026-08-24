#!/usr/bin/env python3
"""Fail-closed direct OpenAI Flex runner for fictional synthetic bundles.

The key is read only from the fixed mode-0600 file. It is never accepted through an
environment variable or command-line argument. Requests are sequential so a safety
signal on request N is processed before request N+1 can be sent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from experiments.lfm25.validate_synthetic import validate_bundle, validate_seed_contract

API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-5.6-sol"
SERVICE_TIER = "flex"
KEY_PATH = Path.home() / ".config/misp-suggest/openai_api_key"
CACHE_KEY = "misp-suggest-fictional-bundles-v1"
SAFETY_IDENTIFIER = "misp-suggest-research"
TIMEOUT_SECONDS = 900.0
MAX_RETRIES = 2
HARD_BUDGET_USD = 200.0
# Promotional Flex prices per million tokens, documented 2026-08-24.
INPUT_USD_PER_M = 2.0
CACHED_INPUT_USD_PER_M = 0.2
CACHE_WRITE_USD_PER_M = 2.5
OUTPUT_USD_PER_M = 10.0
PRICE_VERSION = "openai-flex-2026-08-24"

_SEED_KEYS = {
    "schema_version",
    "bundle_id",
    "language",
    "difficulty",
    "target_tag",
    "target_family",
    "target_frequency_bucket",
    "candidate_tags",
    "candidate_documents",
    "fixed_label_contract",
}
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_WARNING_RE = re.compile(
    r"\b(?:content[_ -]?filter(?:ed|ing)?|safety policy|usage policy|policy violation|"
    r"moderation|request (?:was )?blocked|I (?:cannot|can't|am unable to)|"
    r"I(?:'m| am) sorry,? but|refus(?:e|al))\b",
    re.IGNORECASE,
)
_POLICY_ERROR_RE = re.compile(
    r"content[_ -]?filter|safety|policy|moderation|blocked|refus|guardrail",
    re.IGNORECASE,
)


class EmergencyHalt(RuntimeError):
    """A global, persistent stop condition; no later request may be sent."""


class OrdinaryFlexRetry(RuntimeError):
    """A retryable 408 or exact Flex resource-unavailable response."""


@dataclass(frozen=True)
class HTTPResult:
    status_code: int
    body: bytes
    headers: dict[str, str]


class Transport(Protocol):
    def __call__(
        self, body: bytes, headers: dict[str, str], timeout: float
    ) -> HTTPResult: ...


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def strict_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt the local schema to OpenAI's all-object-fields-required subset."""
    value = copy.deepcopy(schema)
    value.pop("$schema", None)
    value.pop("$id", None)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            # OpenAI's strict subset does not advertise these local admission
            # constraints. The full local schema is applied after generation.
            for unsupported in ("minLength", "maxLength", "uniqueItems"):
                node.pop(unsupported, None)
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
                node["additionalProperties"] = False
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return value


def stable_prompt(schema: dict[str, Any]) -> str:
    rendered_schema = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return f"""You generate privacy-safe fictional cyber-threat-intelligence training bundles for a defensive tag-retrieval experiment. This is data composition, not operational assistance. Follow every rule below literally.

PRIVACY BOUNDARY
The user message contains only public taxonomy identifiers, sanitized public taxonomy descriptions, aggregate frequency buckets, fictional IDs, and locally frozen candidate/label contracts. Never ask for, infer, mention, or recreate any real event, analyst, organization, UUID, credential, internal host, historical report, or retrieved neighbor. Never claim that a fictional scenario is real. Do not add source citations, links, reference identifiers, timestamps, case numbers, organizations, people, campaigns, victims, or geographic details. Treat taxonomy descriptions as semantic definitions only; do not copy reference URLs or identifier-like strings from them.

DEFENSIVE CONTENT BOUNDARY
Write concise analyst-facing observations only. Do not provide executable code, shell commands, PowerShell, exploit instructions, payloads, persistence procedures, credentials, private keys, evasion steps, weaponization details, or instructions that enable intrusion. High-level observed behavior is allowed only when needed to support a defensive taxonomy label. Do not output warnings, policy commentary, refusals, apologies, or prose outside the requested JSON object.

FICTIONAL INDICATORS
Every event is invented. Typed domains and hostnames must end in .example. Typed URLs must use http or https and a .example host. Typed email addresses must use a .example domain. IPv4 must be in 192.0.2.0/24, 198.51.100.0/24, or 203.0.113.0/24. IPv6 must be in 2001:db8::/32. SHA-256 placeholders must contain exactly 64 hexadecimal characters, begin with 32 zeroes, and be invented. Never emit a UUID, CVE identifier, public-looking domain, public IP, real hash, or real email anywhere, including info text and comments. Filenames must be generic and fictional.

FIXED LABEL CONTRACT
Copy schema_version, bundle_id, language, difficulty, target_tag, and candidate_tags exactly from the user seed. Do not reorder candidate_tags. Return exactly six scenarios and exactly one of each required variant. Copy expected_tags and expected_relevance exactly from fixed_label_contract. The target and candidates were selected locally and are immutable. You may not correct, replace, broaden, or invent a tag.

VARIANTS
1. explicit_positive: clear direct evidence for the target; relevance 2; expected_tags contains only target; cite at least one generated scalar evidence path.
2. implicit_positive: distinct paraphrased evidence for target without simply restating the canonical tag; relevance 2; expected_tags contains only target; cite evidence.
3. partial_positive: incomplete but genuinely useful target evidence; relevance 1; expected_tags contains only target; cite evidence.
4. same_family_confuser: support only the preselected non-target expected tag; target relevance 0; cite evidence for the confuser. Do not support target.
5. unrelated_negative: support no candidate; expected_tags and evidence_paths empty.
6. insufficient_evidence: plausible but too sparse to apply any candidate; expected_tags and evidence_paths empty.

EVENT CONSTRUCTION
All six events must be materially different, not a template with substitutions. Write event.info in the requested language, while MISP-style category/type names may remain English. Keep each info string between roughly 35 and 120 words. Use zero to six compact attributes and zero to two compact objects. Prefer a small number of decisive facts over elaborate narratives. Positive evidence must appear in generated fields. Negatives must not accidentally mention candidate-defining behavior. Explicit positives may use an ordinary public malware-family name when natural, but implicit and partial positives must not spell out the target tag string. Do not place taxonomy tags in event text.

EVIDENCE PATHS
Paths start with event and must resolve to a nonempty scalar in that scenario. Allowed forms are event.info; event.attributes.N.category; event.attributes.N.type; event.attributes.N.value; event.attributes.N.comment; event.objects.N.name; event.objects.N.attributes.M.type; and event.objects.N.attributes.M.value. Use actual zero-based indexes. Negative and insufficient variants have no evidence paths.

QUALITY AND DIVERSITY
Use natural analyst prose rather than repetitive boilerplate. Vary whether evidence appears in the summary, attributes, or a small defensive object. Do not manufacture certainty: partial evidence stays partial and insufficient evidence stays insufficient. Ensure confusers test a genuine semantic boundary instead of merely negating the target. Do not encode the variant, relevance, target, language, or difficulty in indicator values, filenames, comments, sentence openings, or recurring phrases. Avoid stock openings such as "an analyst observed" across multiple scenarios. These rows will be checked for shortcuts and real-versus-synthetic artifacts, so prioritize semantically necessary evidence and omit decorative details.

OUTPUT DISCIPLINE
Return one JSON object and nothing else. Every object has only schema-defined keys. Because strict output requires attribute.comment, use an empty string when no comment is needed. Arrays may be empty where allowed. Scenario IDs must begin with the supplied bundle_id followed by a hyphen and the variant name with underscores changed to hyphens. Do not include markdown.

A minimal formatting illustration follows. It is fictional and is not a semantic example to imitate: an event has an info string, an attributes array such as a domain value demo-node.example, and an objects array. A cited path event.attributes.0.value resolves to that domain. A negative has expected_tags [], evidence_paths [], and expected_relevance 0. Every real request still follows its own fixed seed.

The strict response schema is:
{rendered_schema}

Final preflight: preserve all frozen IDs and labels; exactly six distinct events; reserved indicators only; no real identifiers; no operational instructions; valid evidence paths; JSON only.
"""


def validate_seed_privacy(seed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(seed) != _SEED_KEYS:
        errors.append("seed has unexpected or missing top-level fields")
    if seed.get("schema_version") != 1:
        errors.append("seed schema_version must be 1")
    candidates = seed.get("candidate_tags")
    documents = seed.get("candidate_documents")
    if not isinstance(candidates, list) or not (3 <= len(candidates) <= 12):
        errors.append("seed candidates must be a list of length 3..12")
    if not isinstance(documents, dict) or set(documents) != set(candidates or []):
        errors.append("candidate_documents must map exactly the candidate tags")
    if seed.get("target_tag") not in (candidates or []):
        errors.append("seed target must be a candidate")
    if seed.get("target_family") not in {
        "incident-class",
        "attack-technique",
        "malware-family",
        "context",
    }:
        errors.append("unexpected target family")
    rendered = json.dumps(seed, ensure_ascii=False)
    if _UUID_RE.search(rendered):
        errors.append("seed contains UUID-like material")
    lowered_keys = {str(key).casefold() for key in _all_keys(seed)}
    forbidden_keys = {
        "event_id",
        "event_uuid",
        "uuid",
        "org",
        "orgc",
        "organization",
        "analyst",
        "comment",
        "attributes",
        "objects",
        "real_event",
        "neighbor",
    }
    if lowered_keys & forbidden_keys:
        errors.append("seed contains a forbidden private-data-shaped key")
    return errors


def _all_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def load_key() -> str:
    metadata = KEY_PATH.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise EmergencyHalt("API key path is not a regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise EmergencyHalt("API key file mode must be exactly 0600")
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key or "\n" in key or "\r" in key or len(key) > 512:
        raise EmergencyHalt("API key file is malformed")
    return key


def direct_transport(api_key: str) -> Transport:
    def send(body: bytes, headers: dict[str, str], timeout: float) -> HTTPResult:
        request_headers = dict(headers)
        request_headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            API_URL, data=body, headers=request_headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HTTPResult(
                    status_code=int(response.status),
                    body=response.read(),
                    headers={key.casefold(): value for key, value in response.headers.items()},
                )
        except urllib.error.HTTPError as error:
            return HTTPResult(
                status_code=int(error.code),
                body=error.read(),
                headers={key.casefold(): value for key, value in error.headers.items()},
            )

    return send


class AppendLedger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        if path.exists() and stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise EmergencyHalt("ledger mode must be exactly 0600")

    def append(self, value: dict[str, Any]) -> None:
        payload = canonical_bytes(value) + b"\n"
        descriptor = os.open(
            self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(self.path, 0o600)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise EmergencyHalt(
                        f"malformed append-only ledger at line {line_number}"
                    ) from exc
        return rows

    def budget_committed(self) -> float:
        reservations: dict[str, float] = {}
        spent = 0.0
        for row in self.rows():
            kind = row.get("kind")
            reservation_id = str(row.get("reservation_id", ""))
            if kind == "reservation":
                reservations[reservation_id] = float(row["reservation_usd"])
            elif kind == "settlement" and reservation_id in reservations:
                reservations.pop(reservation_id)
                spent += float(row["actual_cost_usd"])
            elif kind == "uncharged" and reservation_id in reservations:
                reservations.pop(reservation_id)
        return spent + sum(reservations.values())

    def accepted_bundle_ids(self) -> set[str]:
        return {
            str(row["bundle_id"])
            for row in self.rows()
            if row.get("kind") == "accepted"
        }


def output_append(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600
    )
    try:
        os.write(descriptor, canonical_bytes(bundle) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def actual_cost(usage: dict[str, Any]) -> tuple[float, dict[str, int]]:
    input_tokens = int(usage["input_tokens"])
    output_tokens = int(usage["output_tokens"])
    details = usage["input_tokens_details"]
    cached = int(details["cached_tokens"])
    written = int(details["cache_write_tokens"])
    ordinary = input_tokens - cached - written
    if min(input_tokens, output_tokens, cached, written, ordinary) < 0:
        raise ValueError("impossible token accounting")
    cost = (
        ordinary * INPUT_USD_PER_M
        + cached * CACHED_INPUT_USD_PER_M
        + written * CACHE_WRITE_USD_PER_M
        + output_tokens * OUTPUT_USD_PER_M
    ) / 1_000_000
    return cost, {
        "input_tokens": input_tokens,
        "uncached_input_tokens": ordinary,
        "cached_tokens": cached,
        "cache_write_tokens": written,
        "output_tokens": output_tokens,
    }


def pessimistic_reservation(body: bytes, max_output_tokens: int) -> float:
    # UTF-8 bytes are a conservative upper bound on input token count. Treat every
    # input token as a cache write and every allowed output token as generated.
    return (
        len(body) * CACHE_WRITE_USD_PER_M + max_output_tokens * OUTPUT_USD_PER_M
    ) / 1_000_000


def make_request(
    seed: dict[str, Any], response_schema: dict[str, Any], *, max_output_tokens: int
) -> tuple[dict[str, Any], bytes, str, str]:
    prompt = stable_prompt(response_schema)
    variable = (
        "Generate the one bundle specified by this frozen taxonomy-only seed. "
        "Candidate documents are semantic context, not text to copy.\nSEED_JSON:\n"
        + json.dumps(seed, sort_keys=True, ensure_ascii=False)
    )
    request = {
        "model": MODEL,
        "service_tier": SERVICE_TIER,
        "store": False,
        "reasoning": {"effort": "low"},
        "max_output_tokens": max_output_tokens,
        "truncation": "disabled",
        "prompt_cache_key": CACHE_KEY,
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "safety_identifier": SAFETY_IDENTIFIER,
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": variable}],
            },
        ],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "fictional_misp_bundle",
                "strict": True,
                "schema": response_schema,
            },
        },
    }
    body = canonical_bytes(request)
    return request, body, sha256_bytes(prompt.encode()), sha256_bytes(canonical_bytes(response_schema))


def _error_text(result: HTTPResult) -> str:
    return result.body.decode("utf-8", errors="replace")[:20_000]


def safe_http_error_facts(result: HTTPResult) -> dict[str, Any]:
    """Retain diagnostic categories without persisting a possibly echoed prompt."""
    facts: dict[str, Any] = {
        "http_status": result.status_code,
        "body_sha256": sha256_bytes(result.body),
    }
    try:
        error = json.loads(result.body).get("error", {})
    except (json.JSONDecodeError, AttributeError):
        return facts
    if not isinstance(error, dict):
        return facts
    for key in ("type", "code", "param"):
        value = error.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            facts[f"error_{key}"] = value
    message = error.get("message")
    if isinstance(message, str):
        facts["message_sha256"] = sha256_bytes(message.encode("utf-8"))
    return facts


def classify_http_error(result: HTTPResult) -> None:
    text = _error_text(result)
    if _POLICY_ERROR_RE.search(text):
        raise EmergencyHalt(f"policy-related API error (HTTP {result.status_code})")
    if result.status_code == 408:
        raise OrdinaryFlexRetry("Flex HTTP 408 Request Timeout")
    if result.status_code == 429 and "resource unavailable" in text.casefold():
        raise OrdinaryFlexRetry("Flex HTTP 429 Resource Unavailable")
    raise EmergencyHalt(f"unexpected OpenAI HTTP status {result.status_code}")


def extract_bundle(response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], float]:
    if response.get("service_tier") != SERVICE_TIER:
        raise EmergencyHalt("unexpected response service tier")
    if response.get("model") != MODEL:
        raise EmergencyHalt("unexpected response model")
    if response.get("error") is not None:
        message = json.dumps(response["error"], ensure_ascii=False)
        if _POLICY_ERROR_RE.search(message):
            raise EmergencyHalt("policy error in response body")
        raise EmergencyHalt("response contains an error object")
    incomplete = response.get("incomplete_details")
    if incomplete is not None:
        if str(incomplete.get("reason", "")) == "content_filter":
            raise EmergencyHalt("response content_filter signal")
        raise EmergencyHalt("response is incomplete")
    if response.get("status") != "completed":
        raise EmergencyHalt("response status is not completed")
    texts: list[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "refusal" or content.get("refusal"):
                raise EmergencyHalt("model refusal")
            if content.get("type") == "output_text":
                texts.append(str(content.get("text", "")))
    if len(texts) != 1 or not texts[0].strip():
        raise EmergencyHalt("response did not contain exactly one output_text")
    if _WARNING_RE.search(texts[0]):
        raise EmergencyHalt("suspicious warning or filtering language in output")
    try:
        bundle = json.loads(texts[0])
    except json.JSONDecodeError as exc:
        raise EmergencyHalt("malformed JSON output") from exc
    if not isinstance(bundle, dict):
        raise EmergencyHalt("structured output root is not an object")
    try:
        cost, usage = actual_cost(response["usage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EmergencyHalt("missing or malformed usage accounting") from exc
    return bundle, usage, cost


def write_brake(path: Path, *, reason: str, stage: str, bundle_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = canonical_bytes(
        {
            "schema_version": 1,
            "halted": True,
            "reason": reason,
            "stage": stage,
            "bundle_id": bundle_id,
            "unix_time": time.time(),
        }
    ) + b"\n"
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
    except FileExistsError:
        return
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(
    seeds: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    output: Path,
    ledger_path: Path,
    brake_path: Path,
    stage: str,
    budget_usd: float,
    max_output_tokens: int = 6000,
    transport: Transport,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if budget_usd <= 0 or budget_usd > HARD_BUDGET_USD:
        raise EmergencyHalt("budget must be positive and at most USD 200")
    if brake_path.exists():
        raise EmergencyHalt("global emergency brake already exists")
    ledger = AppendLedger(ledger_path)
    accepted_before = ledger.accepted_bundle_ids()
    response_schema = strict_response_schema(schema)
    accepted = 0
    skipped = 0
    sent = 0
    for seed in seeds:
        bundle_id = str(seed.get("bundle_id", "unknown"))
        if bundle_id in accepted_before:
            skipped += 1
            continue
        try:
            privacy_errors = validate_seed_privacy(seed)
            if privacy_errors:
                raise EmergencyHalt("privacy gate failed: " + "; ".join(privacy_errors))
            _, body, prompt_hash, schema_hash = make_request(
                seed, response_schema, max_output_tokens=max_output_tokens
            )
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "misp-suggest-sol-flex/1",
            }
            response: dict[str, Any] | None = None
            reservation_id = ""
            for retry in range(MAX_RETRIES + 1):
                reservation = pessimistic_reservation(body, max_output_tokens)
                committed = ledger.budget_committed()
                if committed + reservation > budget_usd + 1e-12:
                    raise EmergencyHalt("pessimistic budget reservation would breach ceiling")
                reservation_id = sha256_bytes(
                    f"{stage}\x1f{bundle_id}\x1f{retry}\x1f{sha256_bytes(body)}".encode()
                )[:24]
                ledger.append(
                    {
                        "kind": "reservation",
                        "stage": stage,
                        "bundle_id": bundle_id,
                        "retry": retry,
                        "reservation_id": reservation_id,
                        "reservation_usd": reservation,
                        "budget_ceiling_usd": budget_usd,
                        "request_sha256": sha256_bytes(body),
                        "prompt_sha256": prompt_hash,
                        "schema_sha256": schema_hash,
                        "model": MODEL,
                        "requested_service_tier": SERVICE_TIER,
                        "price_version": PRICE_VERSION,
                        "unix_time": time.time(),
                    }
                )
                headers["Idempotency-Key"] = "misp-suggest-" + reservation_id
                result = transport(body, headers, TIMEOUT_SECONDS)
                sent += 1
                if result.status_code != 200:
                    ledger.append(
                        {
                            "kind": "http_error",
                            "stage": stage,
                            "bundle_id": bundle_id,
                            "reservation_id": reservation_id,
                            **safe_http_error_facts(result),
                            "unix_time": time.time(),
                        }
                    )
                    try:
                        classify_http_error(result)
                    except OrdinaryFlexRetry:
                        ledger.append(
                            {
                                "kind": "uncharged" if result.status_code == 429 else "retry_error",
                                "stage": stage,
                                "bundle_id": bundle_id,
                                "reservation_id": reservation_id,
                                "http_status": result.status_code,
                                "documented_uncharged": result.status_code == 429,
                                "unix_time": time.time(),
                            }
                        )
                        if retry == MAX_RETRIES:
                            raise EmergencyHalt("bounded Flex retries exhausted")
                        sleep(float(2**retry))
                        continue
                try:
                    response = json.loads(result.body)
                except json.JSONDecodeError as exc:
                    raise EmergencyHalt("malformed OpenAI response body") from exc
                break
            if response is None:
                raise EmergencyHalt("no response after Flex request")
            bundle, usage, cost = extract_bundle(response)
            ledger.append(
                {
                    "kind": "settlement",
                    "stage": stage,
                    "bundle_id": bundle_id,
                    "reservation_id": reservation_id,
                    "actual_cost_usd": cost,
                    "usage": usage,
                    "response_id": response.get("id"),
                    "model": response.get("model"),
                    "service_tier": response.get("service_tier"),
                    "unix_time": time.time(),
                }
            )
            errors = validate_bundle(bundle, schema=schema, valid_tags=set(seed["candidate_tags"]))
            errors.extend(validate_seed_contract(bundle, seed))
            if errors:
                raise EmergencyHalt("generated bundle validation failed: " + "; ".join(errors[:8]))
            output_append(output, bundle)
            ledger.append(
                {
                    "kind": "accepted",
                    "stage": stage,
                    "bundle_id": bundle_id,
                    "bundle_sha256": sha256_bytes(canonical_bytes(bundle)),
                    "response_id": response.get("id"),
                    "prompt_sha256": prompt_hash,
                    "schema_sha256": schema_hash,
                    "usage": usage,
                    "actual_cost_usd": cost,
                    "validator": "validate_synthetic-v2",
                    "unix_time": time.time(),
                }
            )
            accepted_before.add(bundle_id)
            accepted += 1
        except EmergencyHalt as exc:
            ledger.append(
                {
                    "kind": "halt",
                    "stage": stage,
                    "bundle_id": bundle_id,
                    "reason": str(exc),
                    "unix_time": time.time(),
                }
            )
            write_brake(brake_path, reason=str(exc), stage=stage, bundle_id=bundle_id)
            raise
    return {
        "stage": stage,
        "requested": len(seeds),
        "accepted": accepted,
        "skipped": skipped,
        "http_attempts": sent,
        "budget_committed_usd": ledger.budget_committed(),
        "budget_ceiling_usd": budget_usd,
        "halted": False,
    }


def read_seeds(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EmergencyHalt("seed row is not an object")
            rows.append(value)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--brake", type=Path, required=True)
    parser.add_argument("--stage", choices=("canary", "pilot", "scale"), required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    args = parser.parse_args()
    if args.max_output_tokens < 1000 or args.max_output_tokens > 10_000:
        parser.error("max output tokens must be in 1000..10000")
    schema = json.loads(
        Path(__file__).with_name("synthetic_bundle.schema.json").read_text(encoding="utf-8")
    )
    seeds = read_seeds(args.seeds, args.limit)
    key = load_key()
    try:
        report = run(
            seeds,
            schema=schema,
            output=args.output,
            ledger_path=args.ledger,
            brake_path=args.brake,
            stage=args.stage,
            budget_usd=args.budget_usd,
            max_output_tokens=args.max_output_tokens,
            transport=direct_transport(key),
        )
    except EmergencyHalt as exc:
        print(json.dumps({"halted": True, "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
