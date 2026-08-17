"""Map eligible MISP tags into the evaluation families from training-plan.md."""

from __future__ import annotations

from .allowlist import parse_tag

INCIDENT_NAMESPACES = {
    "rsit",
    "ecsirt",
    "circl",
    "dnsc-ro",
    "cert-xlm",
    "csirt-americas",
    "enisa",
    "cccs",
    "ccirc",
    "cssa",
    "misp",
    "ddos",
    "kill-chain",
    "veris",
    "phishing",
    "cert-ist",
    "ncsc-nl-ndn",
}
MALWARE_GALAXY_TYPES = {
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
}
MALWARE_NAMESPACES = {
    "malware_name",
    "mn-malware",
    "ms-caro-malware-full",
    "malware_classification",
    "certego",
}
CONTEXT_GALAXY_TYPES = {"sector", "country", "target-information"}


def label_family(tag: str) -> str:
    """Return incident-class, attack-technique, malware-family, context or other."""
    parsed = parse_tag(tag)
    if parsed is None:
        return "other"
    namespace = parsed.namespace.lower()
    predicate = parsed.predicate.lower()
    if namespace == "misp-galaxy":
        if predicate == "mitre-attack-pattern":
            return "attack-technique"
        if predicate in MALWARE_GALAXY_TYPES:
            return "malware-family"
        if predicate in CONTEXT_GALAXY_TYPES:
            return "context"
        return "other"
    if namespace == "dhs-ciip-sectors":
        return "context"
    if namespace == "stone":
        if predicate == "incident-classification":
            return "incident-class"
        if predicate == "malware-categorization":
            return "malware-family"
        return "other"
    if namespace in MALWARE_NAMESPACES:
        return "malware-family"
    if namespace in INCIDENT_NAMESPACES:
        return "incident-class"
    return "other"
