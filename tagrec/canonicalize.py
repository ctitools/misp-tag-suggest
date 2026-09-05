"""Leakage-safe deterministic MISP Event canonicalization."""

from __future__ import annotations

import hashlib
import html
import ipaddress
import re
from pathlib import PurePath
from typing import Any, Iterable
from urllib.parse import urlsplit

import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())
_HASH_TYPES = {
    "md5",
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "ssdeep",
    "tlsh",
    "imphash",
    "authentihash",
}
_IP_TYPES = {"ip-src", "ip-dst", "ip-src|port", "ip-dst|port"}
_DOMAIN_TYPES = {"domain", "hostname", "domain|ip"}
_URL_TYPES = {"url", "uri"}
_EMAIL_TYPES = {"email", "email-src", "email-dst", "email-reply-to", "email-x-mailer"}
_FILENAME_TYPES = {"filename", "attachment", "malware-sample"}
_SPACE_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"!?\[([^]]*)\]\([^)]+\)")
_MD_MARK_RE = re.compile(r"[`*_~>#]+")
_HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{32,128}$")


def unwrap_event(raw: dict[str, Any]) -> dict[str, Any]:
    event = raw.get("Event", raw)
    return event if isinstance(event, dict) else {}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = _HTML_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_MARK_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _registered_domain(value: str) -> str:
    cleaned = value.strip().lower().replace("[.]", ".").replace("(.)", ".")
    try:
        ext = _EXTRACT(cleaned)
    except TypeError, ValueError:
        return ""
    return ext.top_domain_under_public_suffix or cleaned.strip(".")


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip().strip("[]").replace("[.]", "."))
        return True
    except ValueError:
        return False


def _normalize_url(value: str) -> str:
    defanged = (
        value.strip()
        .replace("hxxps://", "https://")
        .replace("hxxp://", "http://")
        .replace("[.]", ".")
    )
    try:
        parsed = urlsplit(defanged if "://" in defanged else "//" + defanged)
        host = parsed.hostname or ""
    except ValueError:
        # Malformed or defanged bracket syntax is common in imported CTI.
        host_match = re.search(r"(?:[a-z][a-z0-9+.-]*://)?([^/\s:]+)", defanged, re.IGNORECASE)
        host = host_match.group(1).strip("[]") if host_match else ""
    domain = _registered_domain(host) if host else ""
    return f"<URL:{domain}>" if domain else "<URL>"


def _normalize_email(value: str) -> str:
    domain = value.rsplit("@", 1)[-1] if "@" in value else ""
    domain = _registered_domain(domain) if domain else ""
    return f"<EMAIL:{domain}>" if domain else "<EMAIL>"


def _normalize_filename(value: str) -> str:
    name = value.split("|", 1)[0].strip()
    suffix = PurePath(name).suffix.lower()
    return f"<FILE:{suffix}>" if suffix else "<FILE>"


def normalize_attribute_value(attribute_type: str, value: Any) -> str:
    """Normalize high-cardinality IOC values while retaining useful type information."""
    text = clean_text(value)
    attr_type = (attribute_type or "").lower()
    if not text:
        return ""

    if attr_type in {"ip-src|port", "ip-dst|port"} and "|" in text:
        _ip, port = text.split("|", 1)
        return f"<IP>:{clean_text(port)}"
    if attr_type == "malware-sample" and "|" in text:
        filename, digest = text.split("|", 1)
        digest_type = "md5" if len(digest) == 32 else "hash"
        return f"{_normalize_filename(filename)}|<HASH:{digest_type}>"

    # Other composite MISP types are normalized component by component.
    if "|" in attr_type and "|" in text:
        subtypes = attr_type.split("|")
        values = text.split("|", len(subtypes) - 1)
        if len(subtypes) == len(values):
            return "|".join(
                normalize_attribute_value(t, v) for t, v in zip(subtypes, values, strict=True)
            )

    if attr_type in _HASH_TYPES or _HEX_HASH_RE.fullmatch(text):
        return f"<HASH:{attr_type}>" if attr_type in _HASH_TYPES else "<HASH>"
    if attr_type == "ip" or attr_type in _IP_TYPES or attr_type.startswith("ip-"):
        if "|" in text:
            _ip, port = text.split("|", 1)
            return f"<IP>:{clean_text(port)}"
        return "<IP>"
    if attr_type in _URL_TYPES:
        return _normalize_url(text)
    if attr_type in _DOMAIN_TYPES:
        domain = text.split("|", 1)[0]
        registered = _registered_domain(domain)
        return f"<DOMAIN:{registered}>" if registered else "<DOMAIN>"
    if attr_type in _EMAIL_TYPES or attr_type.startswith("email-"):
        return _normalize_email(text)
    if attr_type in _FILENAME_TYPES or attr_type.startswith("filename"):
        return _normalize_filename(text)
    if _is_ip(text):
        return "<IP>"
    return text


def _attributes(event: dict[str, Any]) -> Iterable[tuple[dict[str, Any], str | None]]:
    for attribute in event.get("Attribute", []) or []:
        if isinstance(attribute, dict) and not attribute.get("deleted"):
            yield attribute, None
    for obj in event.get("Object", []) or []:
        if not isinstance(obj, dict) or obj.get("deleted"):
            continue
        object_name = clean_text(obj.get("name"))
        meta = clean_text(obj.get("meta-category"))
        object_description = clean_text(obj.get("comment") or obj.get("description"))
        label = object_name
        if meta:
            label += f" ({meta})"
        if object_description:
            label += f" — {object_description}"
        yield {"__object_marker__": label}, None
        for attribute in obj.get("Attribute", []) or []:
            if isinstance(attribute, dict) and not attribute.get("deleted"):
                yield attribute, object_name


def canonicalize_event(raw: dict[str, Any]) -> str:
    """Convert a MISP Event into stable text without reading tag/galaxy structures."""
    event = unwrap_event(raw)
    lines: list[str] = []

    info = clean_text(event.get("info"))
    if info:
        lines.append(f"[INFO] {info}")

    reports = sorted(
        (
            r
            for r in (event.get("EventReport", []) or [])
            if isinstance(r, dict) and not r.get("deleted")
        ),
        key=lambda r: (str(r.get("id", "")), str(r.get("uuid", ""))),
    )
    for report in reports:
        name = clean_text(report.get("name"))
        content = clean_text(report.get("content"))
        report_text = " — ".join(part for part in (name, content) if part)
        if report_text:
            lines.append(f"[REPORT] {report_text}")

    for attribute, object_name in _attributes(event):
        marker = attribute.get("__object_marker__")
        if marker:
            lines.append(f"[OBJECT] {marker}")
            continue
        category = clean_text(attribute.get("category"))
        attr_type = clean_text(attribute.get("type"))
        value = normalize_attribute_value(attr_type, attribute.get("value"))
        comment = clean_text(attribute.get("comment"))
        prefix = f"{category}/{attr_type}" if category else attr_type
        if object_name:
            prefix = f"{object_name}:{prefix}"
        body = f"{prefix}: {value}" if value else prefix
        if comment:
            body += f" — {comment}"
        if body:
            lines.append(f"[ATTR] {body}")

    return "\n".join(lines)


def canonical_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
