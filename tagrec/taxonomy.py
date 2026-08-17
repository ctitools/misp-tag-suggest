"""Taxonomy snapshot loading, validation and legacy tag normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .allowlist import ELIGIBLE_GALAXY_TYPES, TagDecision, decide_tag

_TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
_TECHNIQUE_SUFFIX_RE = re.compile(r"-\s*(T\d{4}(?:\.\d{3})?)\s*\"?$", re.IGNORECASE)
_SOFTWARE_ID_RE = re.compile(r"\bS\d{4}\b", re.IGNORECASE)
_DEPRECATED_SUFFIX_RE = re.compile(r"\s*\(deprecated\)\s*$", re.IGNORECASE)
_GALAXY_FALLBACK_TYPES = {
    "tool": ("software", "malpedia", "stealer"),
    "malpedia": ("software", "tool", "stealer", "ransomware"),
    "ransomware": ("software", "malpedia"),
    "mitre-tool": ("mitre-malware", "software", "tool"),
}


def _expanded_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _technique_ids(cluster: dict[str, Any]) -> set[str]:
    """IDs this cluster *is*, not IDs it mentions — descriptions cite other techniques."""
    ids: set[str] = set()
    for element in cluster.get("elements", []) or []:
        if isinstance(element, dict) and element.get("key") == "external_id":
            match = _TECHNIQUE_RE.fullmatch(str(element.get("value", "")).strip())
            if match:
                ids.add(match.group(0).upper())
    for source in (cluster.get("value"), cluster.get("tag_name")):
        match = _TECHNIQUE_SUFFIX_RE.search(str(source or ""))
        if match:
            ids.add(match.group(1).upper())
    return ids


@dataclass(slots=True)
class TaxonomyCatalog:
    snapshot_path: Path
    snapshot_date: str
    valid_tags: set[str]
    tag_metadata: dict[str, dict[str, Any]]
    rsit_expanded_map: dict[str, str]
    attack_technique_map: dict[str, str]
    attack_software_map: dict[str, str]
    casefold_tag_map: dict[str, str]
    galaxy_alias_map: dict[tuple[str, str], str]
    enabled_taxonomy_count: int
    enabled_galaxy_count: int
    require_enabled_taxonomies: bool

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_enabled_taxonomies: bool = False,
    ) -> "TaxonomyCatalog":
        snapshot_path = Path(path)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        valid_tags: set[str] = set()
        metadata: dict[str, dict[str, Any]] = {}
        rsit_map: dict[str, str] = {}
        attack_candidates: dict[str, list[str]] = {}
        software_candidates: dict[str, list[str]] = {}
        galaxy_alias_candidates: dict[tuple[str, str], set[str]] = {}

        taxonomies = snapshot.get("taxonomies", []) or []
        for taxonomy in taxonomies:
            if not isinstance(taxonomy, dict):
                continue
            if require_enabled_taxonomies and not taxonomy.get("enabled"):
                continue
            for entry in taxonomy.get("entries", []) or []:
                if not isinstance(entry, dict) or not entry.get("tag"):
                    continue
                tag = str(entry["tag"])
                valid_tags.add(tag)
                metadata[tag] = {
                    "kind": "taxonomy",
                    "namespace": taxonomy.get("namespace"),
                    "taxonomy_description": taxonomy.get("description", ""),
                    **entry,
                }
                if str(taxonomy.get("namespace", "")).lower() == "rsit":
                    for source in (entry.get("expanded"), entry.get("tag")):
                        if source:
                            rsit_map[_expanded_key(str(source))] = tag

        galaxies = snapshot.get("galaxies", []) or []
        for galaxy in galaxies:
            if not isinstance(galaxy, dict) or not galaxy.get("enabled"):
                continue
            galaxy_type = str(galaxy.get("type", ""))
            for cluster in galaxy.get("clusters", []) or []:
                if not isinstance(cluster, dict) or not cluster.get("tag_name"):
                    continue
                tag = str(cluster["tag_name"])
                valid_tags.add(tag)
                metadata[tag] = {
                    "kind": "galaxy",
                    "galaxy_type": galaxy_type,
                    "galaxy_description": galaxy.get("description", ""),
                    **cluster,
                }
                if galaxy_type.lower() in ELIGIBLE_GALAXY_TYPES:
                    aliases = [cluster.get("value")]
                    value_without_deprecated = _DEPRECATED_SUFFIX_RE.sub(
                        "", str(cluster.get("value") or "")
                    )
                    if value_without_deprecated != str(cluster.get("value") or ""):
                        aliases.append(value_without_deprecated)
                    for element in cluster.get("elements", []) or []:
                        if isinstance(element, dict) and str(element.get("key", "")).lower() in {
                            "synonym",
                            "synonyms",
                        }:
                            aliases.append(element.get("value"))
                    for alias in aliases:
                        key = _expanded_key(str(alias or ""))
                        if not key:
                            continue
                        galaxy_alias_candidates.setdefault((galaxy_type.lower(), key), set()).add(tag)
                if galaxy_type == "mitre-attack-pattern":
                    for technique_id in _technique_ids(cluster):
                        attack_candidates.setdefault(technique_id, []).append(tag)
                if galaxy_type == "mitre-malware":
                    for element in cluster.get("elements", []) or []:
                        if not isinstance(element, dict) or element.get("key") != "external_id":
                            continue
                        match = _SOFTWARE_ID_RE.fullmatch(str(element.get("value", "")).strip())
                        if match:
                            software_candidates.setdefault(match.group(0).upper(), []).append(tag)

        for name in snapshot.get("custom_tags", []) or []:
            tag = str(name)
            valid_tags.add(tag)
            metadata.setdefault(tag, {"kind": "custom-tag"})

        attack_map = {
            technique_id: sorted(tags, key=lambda tag: (len(tag), tag))[0]
            for technique_id, tags in attack_candidates.items()
            if tags
        }
        software_map = {
            software_id: sorted(tags, key=lambda tag: (len(tag), tag))[0]
            for software_id, tags in software_candidates.items()
            if tags
        }
        casefold_candidates: dict[str, set[str]] = {}
        for tag in valid_tags:
            casefold_candidates.setdefault(tag.casefold(), set()).add(tag)
        casefold_map = {
            key: next(iter(tags)) for key, tags in casefold_candidates.items() if len(tags) == 1
        }
        galaxy_alias_map = {
            key: next(iter(tags))
            for key, tags in galaxy_alias_candidates.items()
            if len(tags) == 1
        }
        summary = snapshot.get("summary", {}) or {}
        return cls(
            snapshot_path=snapshot_path,
            snapshot_date=str(snapshot.get("snapshot_date", "")),
            valid_tags=valid_tags,
            tag_metadata=metadata,
            rsit_expanded_map=rsit_map,
            attack_technique_map=attack_map,
            attack_software_map=software_map,
            casefold_tag_map=casefold_map,
            galaxy_alias_map=galaxy_alias_map,
            enabled_taxonomy_count=int(summary.get("enabled_taxonomy_count", 0)),
            enabled_galaxy_count=int(summary.get("enabled_galaxy_count", 0)),
            require_enabled_taxonomies=require_enabled_taxonomies,
        )

    def normalize_detailed(
        self, tag: str, decision: TagDecision | None = None
    ) -> tuple[str | None, str, str | None]:
        """Normalize a candidate and expose the exact successful remap mechanism.

        The middle value preserves the historical keep/drop reason contract used by
        dataset construction. The final value is ``direct`` for unchanged valid tags,
        ``remap:*`` for successful remaps, and ``None`` for failures.
        """
        decision = decision or decide_tag(tag)
        if not decision.eligible:
            return None, decision.reason, None
        parsed = decision.parsed
        if parsed is None:
            return None, "unparseable", None

        normalized = tag.strip()
        mechanism = "remap:strip" if normalized != tag else "direct"
        if decision.needs_normalization:
            galaxy_type = parsed.predicate.lower()
            if galaxy_type == "rsit" and parsed.value:
                normalized = self.rsit_expanded_map.get(_expanded_key(parsed.value), "")
                if not normalized:
                    return None, "rsit-normalization-unresolved", None
                mechanism = "remap:rsit-expanded"
            elif galaxy_type in {
                "mitre-enterprise-attack-attack-pattern",
                "stix-2.1-attack-pattern",
            }:
                match = _TECHNIQUE_RE.search(parsed.value or tag)
                if not match:
                    return None, "attack-technique-id-missing", None
                normalized = self.attack_technique_map.get(match.group(0).upper(), "")
                if not normalized:
                    return None, "attack-technique-normalization-unresolved", None
                mechanism = "remap:technique-id"

        if normalized in self.valid_tags:
            return normalized, "eligible-and-valid", mechanism

        # Historical event tags frequently differ only by case or missing quote delimiters.
        value = (parsed.value or "").strip().strip('"')
        reconstructed = (
            f'{parsed.namespace}:{parsed.predicate}="{value}"' if parsed.value is not None else normalized
        )
        casefold_match = self.casefold_tag_map.get(reconstructed.casefold())
        if casefold_match:
            return casefold_match, "eligible-and-valid", "remap:casefold"

        if parsed.namespace.lower() == "misp-galaxy":
            galaxy_type = parsed.predicate.lower()
            # Event tags often carry pre-rename ATT&CK technique names; the ID is stable.
            if galaxy_type == "mitre-attack-pattern":
                match = _TECHNIQUE_RE.search(value)
                remapped = self.attack_technique_map.get(match.group(0).upper()) if match else None
                if remapped:
                    return remapped, "eligible-and-valid", "remap:technique-id"
            if galaxy_type == "mitre-tool":
                match = _SOFTWARE_ID_RE.search(value)
                remapped = self.attack_software_map.get(match.group(0).upper()) if match else None
                if remapped:
                    return remapped, "eligible-and-valid", "remap:software-id"
            alias_key = _expanded_key(value)
            alias_keys = [alias_key]
            if alias_key.startswith("the") and len(alias_key) > 3:
                alias_keys.append(alias_key[3:])

            def alias_match(alias_type: str) -> tuple[str | None, str | None]:
                for key in alias_keys:
                    candidate = self.galaxy_alias_map.get((alias_type, key))
                    if candidate:
                        return candidate, key
                return None, None

            remapped, matched_key = alias_match(galaxy_type)
            matched_mechanism = "remap:alias:same-type"
            if remapped is None:
                for fallback_type in _GALAXY_FALLBACK_TYPES.get(galaxy_type, ()):
                    remapped, matched_key = alias_match(fallback_type)
                    if remapped:
                        matched_mechanism = "remap:alias:fallback"
                        break
            if remapped:
                if matched_key != alias_key:
                    matched_mechanism = "remap:the-prefix"
                return remapped, "eligible-and-valid", matched_mechanism

        return None, "absent-from-taxonomy-snapshot", None

    def normalize(self, tag: str, decision: TagDecision | None = None) -> tuple[str | None, str]:
        """Normalize a candidate while preserving the dataset's keep/drop contract."""
        normalized, reason, _mechanism = self.normalize_detailed(tag, decision)
        return normalized, reason
