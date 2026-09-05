"""Executable taxonomy eligibility policy from taxonomy-allowlist.md."""

from __future__ import annotations

from dataclasses import dataclass

STANDARD_NAMESPACES = {
    "rsit",
    "ecsirt",
    "ddos",
    "kill-chain",
    "ms-caro-malware-full",
    "malware_classification",
    "dhs-ciip-sectors",
    "cssa",
    "enisa",
    "cccs",
    "ccirc",
}

ELIGIBLE_GALAXY_TYPES = {
    "mitre-attack-pattern",
    "mitre-malware",
    "mitre-tool",
    "malpedia",
    "tool",
    "software",
    "ransomware",
    "rat",
    "stealer",
    "banker",
    "botnet",
    "backdoor",
    "android",
    "exploit-kit",
    "sector",
    "country",
    "target-information",
}

LEGACY_NORMALIZABLE_GALAXIES = {
    "rsit",
    "mitre-enterprise-attack-attack-pattern",
    "stix-2.1-attack-pattern",
}

ORG_SPECIFIC_NAMESPACES = {
    "dnsc-ro",
    "cert-xlm",
    "csirt-americas",
    "malware_name",
    "mn-malware",
    "certego",
    "cert-ist",
    "ncsc-nl-ndn",
}

# Local custom-tag vocabularies: not installed taxonomy definitions, so their
# eligible tags are validated against the instance tag list snapshot instead.
CUSTOM_TAG_NAMESPACES = ORG_SPECIFIC_NAMESPACES | {"stone"}

EXCLUDED_NAMESPACES = {
    "tlp",
    "pap",
    "retention",
    "admiralty-scale",
    "source",
    "type",
    "information-origin",
    "feedly",
    "opencti_label",
    "recorded-future",
    "splunk",
}

JUDGMENT_OR_PROVENANCE_PREDICATES = {
    "confidence",
    "confidence-level",
    "certainty",
    "threat-level",
    "automation-level",
    "source",
    "source-type",
    "lifetime",
    "provenance",
    "feed",
    "handling",
    "distribution",
    "state",
    "status",
    "attribution",
}


@dataclass(frozen=True, slots=True)
class ParsedTag:
    raw: str
    namespace: str
    predicate: str
    value: str | None


@dataclass(frozen=True, slots=True)
class TagDecision:
    eligible: bool
    reason: str
    parsed: ParsedTag | None = None
    needs_normalization: bool = False


def parse_tag(tag: str) -> ParsedTag | None:
    """Parse a MISP machine tag without altering its canonical spelling."""
    raw = tag.strip()
    if not raw or ":" not in raw:
        return None
    namespace, remainder = raw.split(":", 1)
    if not namespace or not remainder:
        return None
    if "=" in remainder:
        predicate, value = remainder.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
    else:
        predicate, value = remainder, None
    return ParsedTag(raw, namespace, predicate, value)


def _looks_judgmental(predicate: str) -> bool:
    normalized = predicate.lower().replace("_", "-")
    return any(
        normalized == blocked
        or normalized.startswith(blocked + "-")
        or normalized.endswith("-" + blocked)
        for blocked in JUDGMENT_OR_PROVENANCE_PREDICATES
    )


def decide_tag(tag: str) -> TagDecision:
    """Return whether a tag is an eligible prediction target and why."""
    parsed = parse_tag(tag)
    if parsed is None:
        return TagDecision(False, "free-text-or-malformed")

    namespace = parsed.namespace.lower()
    predicate = parsed.predicate.lower()

    if namespace in EXCLUDED_NAMESPACES:
        return TagDecision(False, f"excluded-namespace:{namespace}", parsed)

    if namespace == "osint":
        return TagDecision(False, "excluded-osint-judgment-or-provenance", parsed)

    if namespace == "misp-galaxy":
        if predicate == "threat-actor":
            return TagDecision(False, "excluded-attribution", parsed)
        if predicate in ELIGIBLE_GALAXY_TYPES:
            return TagDecision(True, "eligible-descriptive-galaxy", parsed)
        if predicate in LEGACY_NORMALIZABLE_GALAXIES:
            return TagDecision(
                True,
                "eligible-after-normalization",
                parsed,
                needs_normalization=True,
            )
        return TagDecision(False, "galaxy-type-not-allowlisted", parsed)

    if namespace == "circl":
        eligible = predicate in {"incident-classification", "topic"}
        return TagDecision(
            eligible, "eligible-circl" if eligible else "circl-predicate-excluded", parsed
        )

    if namespace == "misp":
        eligible = predicate == "event-type"
        return TagDecision(
            eligible, "eligible-misp-event-type" if eligible else "misp-predicate-excluded", parsed
        )

    if namespace == "veris" and predicate == "confidence":
        return TagDecision(False, "excluded-estimative", parsed)

    if namespace == "phishing" and predicate == "state":
        return TagDecision(False, "excluded-time-varying-state", parsed)

    if namespace in STANDARD_NAMESPACES or namespace in {"veris", "phishing"}:
        return TagDecision(True, "eligible-standard-taxonomy", parsed)

    if namespace == "stone":
        eligible = predicate in {"incident-classification", "malware-categorization"}
        return TagDecision(
            eligible,
            "eligible-stone-descriptive" if eligible else "stone-predicate-excluded",
            parsed,
        )

    if namespace in ORG_SPECIFIC_NAMESPACES:
        if _looks_judgmental(predicate):
            return TagDecision(False, "org-specific-non-descriptive-predicate", parsed)
        return TagDecision(True, "eligible-org-specific-descriptive", parsed)

    return TagDecision(False, "namespace-not-allowlisted", parsed)


def tag_namespace(tag: str) -> str:
    parsed = parse_tag(tag)
    if parsed is None:
        return "<free-text>"
    if parsed.namespace.lower() == "misp-galaxy":
        return f"misp-galaxy:{parsed.predicate}"
    return parsed.namespace
