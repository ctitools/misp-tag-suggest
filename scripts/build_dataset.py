#!/usr/bin/env python3
"""Build the shared leakage-safe, deduplicated chronological dataset.

Two streaming passes keep memory bounded on small hosts: pass 1 keeps only
per-event metadata, content hash and MinHash signature; pass 2 re-reads the
raw export and writes canonical text for the chosen representatives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

from tagrec.allowlist import decide_tag
from tagrec.canonicalize import canonical_hash, canonicalize_event, unwrap_event
from tagrec.io import iter_jsonl, write_json_secure
from tagrec.taxonomy import TaxonomyCatalog

_TOKEN_RE = re.compile(r"<[^>]+>|[\w.-]+", re.UNICODE)
FEED_TAG_PREFIXES = ("source:", "type:osint", "stone:source", "stone:feed")
SPLITS = ("train", "val", "test")
EMPTY_TEXT_HASH = canonical_hash("")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(slots=True)
class EventRecord:
    uuid: str
    event_id: int | str
    date: str
    timestamp: int
    orgc: str
    feed_derived: bool
    text_hash: str
    labels: list[str]
    n_attributes: int
    n_objects: int


def event_tag_names(event: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tag in event.get("Tag", []) or []:
        if isinstance(tag, dict) and tag.get("name"):
            names.append(str(tag["name"]))
    for event_tag in event.get("EventTag", []) or []:
        if not isinstance(event_tag, dict):
            continue
        tag = event_tag.get("Tag", event_tag)
        if isinstance(tag, dict) and tag.get("name"):
            names.append(str(tag["name"]))
    for galaxy in event.get("Galaxy", []) or []:
        if not isinstance(galaxy, dict):
            continue
        for cluster in galaxy.get("GalaxyCluster", []) or []:
            if isinstance(cluster, dict) and cluster.get("tag_name"):
                names.append(str(cluster["tag_name"]))
    return sorted(set(names))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def make_record(
    raw: dict[str, Any],
    catalog: TaxonomyCatalog,
    feed_orgs: set[str],
    dropped: Counter[str],
) -> tuple[EventRecord, str]:
    event = unwrap_event(raw)
    all_tags = event_tag_names(event)
    labels: set[str] = set()
    for tag in all_tags:
        decision = decide_tag(tag)
        if not decision.eligible:
            dropped[decision.reason] += 1
            continue
        normalized, reason = catalog.normalize(tag, decision)
        if normalized:
            labels.add(normalized)
        else:
            dropped[reason] += 1

    orgc_value = event.get("Orgc", {})
    orgc = str(orgc_value.get("name", "")) if isinstance(orgc_value, dict) else ""
    feed_derived = orgc.lower() in feed_orgs or any(
        tag.lower().startswith(FEED_TAG_PREFIXES) for tag in all_tags
    )
    objects = [
        obj
        for obj in (event.get("Object", []) or [])
        if isinstance(obj, dict) and not obj.get("deleted")
    ]
    event_attributes = [
        attr
        for attr in (event.get("Attribute", []) or [])
        if isinstance(attr, dict) and not attr.get("deleted")
    ]
    object_attribute_count = sum(
        1
        for obj in objects
        for attr in (obj.get("Attribute", []) or [])
        if isinstance(attr, dict) and not attr.get("deleted")
    )
    text = canonicalize_event(event)
    event_id_raw = event.get("id", "")
    event_id: int | str = _safe_int(event_id_raw, -1)
    if event_id == -1:
        event_id = str(event_id_raw)
    record = EventRecord(
        uuid=str(event.get("uuid", "")),
        event_id=event_id,
        date=str(event.get("date", "")),
        timestamp=_safe_int(event.get("timestamp")),
        orgc=orgc,
        feed_derived=feed_derived,
        text_hash=canonical_hash(text),
        labels=sorted(labels),
        n_attributes=len(event_attributes) + object_attribute_count,
        n_objects=len(objects),
    )
    return record, text


def shingles(text: str, size: int = 5) -> set[bytes]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    if not tokens:
        return {b"<EMPTY>"}
    if len(tokens) < size:
        return {" ".join(tokens).encode("utf-8")}
    return {
        " ".join(tokens[index : index + size]).encode("utf-8")
        for index in range(len(tokens) - size + 1)
    }


def minhash(text: str, num_perm: int) -> MinHash:
    value = MinHash(num_perm=num_perm)
    for shingle in shingles(text):
        value.update(shingle)
    return value


def deduplicate(
    records: list[EventRecord],
    signatures: list[MinHash | None] | None,
    *,
    near_threshold: float,
    num_perm: int,
) -> tuple[dict[int, list[int]], dict[str, int]]:
    uf = UnionFind(len(records))
    by_uuid: dict[str, int] = {}
    by_text: dict[str, int] = {}
    exact_unions = 0
    for index, record in enumerate(records):
        keys = [(record.uuid, by_uuid)]
        # Content-less events are not copies of each other; only UUID identity applies.
        if record.text_hash != EMPTY_TEXT_HASH:
            keys.append((record.text_hash, by_text))
        for key, mapping in keys:
            if not key:
                continue
            previous = mapping.get(key)
            if previous is None:
                mapping[key] = index
            else:
                if uf.find(previous) != uf.find(index):
                    exact_unions += 1
                uf.union(previous, index)

    near_unions = 0
    if signatures is not None:
        lsh = MinHashLSH(threshold=near_threshold, num_perm=num_perm)
        for index, signature in enumerate(signatures):
            if signature is None:
                continue
            for candidate_key in lsh.query(signature):
                candidate = int(candidate_key)
                candidate_signature = signatures[candidate]
                if candidate_signature is None:
                    continue
                if signature.jaccard(candidate_signature) >= near_threshold:
                    if uf.find(candidate) != uf.find(index):
                        near_unions += 1
                    uf.union(candidate, index)
            lsh.insert(str(index), signature)
            if (index + 1) % 1000 == 0:
                print(
                    f"Near-dedup signatures: {index + 1}/{len(records)}",
                    file=sys.stderr,
                    flush=True,
                )

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        clusters[uf.find(index)].append(index)
    return dict(clusters), {
        "exact_unions": exact_unions,
        "near_unions": near_unions,
        "cluster_count": len(clusters),
    }


def _date_key(record: EventRecord) -> tuple[date, int, str]:
    try:
        parsed = date.fromisoformat(record.date)
    except ValueError:
        parsed = date.min
    return parsed, record.timestamp, str(record.event_id)


def choose_representatives(
    records: list[EventRecord], clusters: dict[int, list[int]]
) -> list[tuple[str, int]]:
    """Return (cluster_id, stream_index) representatives in chronological order."""
    representatives: list[tuple[str, int]] = []
    ordered_clusters = sorted(clusters.values(), key=min)
    for cluster_number, members in enumerate(ordered_clusters, start=1):
        representative_index = min(members, key=lambda index: _date_key(records[index]))
        representatives.append((f"c{cluster_number:06d}", representative_index))
    representatives.sort(key=lambda item: _date_key(records[item[1]]))
    return representatives


def assign_splits(
    records: list[EventRecord],
    representatives: list[tuple[str, int]],
) -> tuple[dict[int, tuple[str, str]], dict[str, str]]:
    """Map representative stream index -> (split, cluster_id) plus boundary dates."""
    total = len(representatives)
    train_end = math.floor(total * 0.8)
    val_end = math.floor(total * 0.9)
    if total >= 3:
        train_end = max(1, min(train_end, total - 2))
        val_end = max(train_end + 1, min(val_end, total - 1))
    ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, total)}
    assignments: dict[int, tuple[str, str]] = {}
    boundaries: dict[str, str] = {}
    for split, (start, end) in ranges.items():
        section = representatives[start:end]
        for cluster_id, index in section:
            assignments[index] = (split, cluster_id)
        if section:
            boundaries[f"{split}_min_date"] = records[section[0][1]].date
            boundaries[f"{split}_max_date"] = records[section[-1][1]].date
    return assignments, boundaries


def write_splits(
    out_dir: Path,
    raw_path: str,
    records: list[EventRecord],
    assignments: dict[int, tuple[str, str]],
    max_events: int | None,
) -> dict[str, int]:
    """Pass 2: re-stream the raw export and write representative rows per split."""
    paths = {split: out_dir / f"{split}.jsonl" for split in SPLITS}
    temporaries = {split: path.with_suffix(path.suffix + ".tmp") for split, path in paths.items()}
    counts = {split: 0 for split in SPLITS}
    written = 0
    try:
        handles = {
            split: temporary.open("w", encoding="utf-8") for split, temporary in temporaries.items()
        }
        try:
            for index, raw in enumerate(iter_jsonl(raw_path)):
                if max_events and index >= max_events:
                    break
                assignment = assignments.get(index)
                if assignment is None:
                    continue
                split, cluster_id = assignment
                record = records[index]
                text = canonicalize_event(raw)
                if canonical_hash(text) != record.text_hash:
                    raise RuntimeError(
                        f"Raw export changed between passes at line {index + 1} ({record.uuid})"
                    )
                row = {
                    "uuid": record.uuid,
                    "event_id": record.event_id,
                    "date": record.date,
                    "orgc": record.orgc,
                    "split": split,
                    "feed_derived": record.feed_derived,
                    "dup_cluster": cluster_id,
                    "text": text,
                    "labels": record.labels,
                    "n_attributes": record.n_attributes,
                    "n_objects": record.n_objects,
                }
                handles[split].write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counts[split] += 1
                written += 1
                if written % 1000 == 0:
                    print(
                        f"Wrote {written}/{len(assignments)} representatives",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            for handle in handles.values():
                handle.close()
        if written != len(assignments):
            raise RuntimeError(f"Expected {len(assignments)} representatives, wrote {written}")
        for split, path in paths.items():
            os.chmod(temporaries[split], 0o600)
            os.replace(temporaries[split], path)
            os.chmod(path, 0o600)
    except BaseException:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        raise
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/events.jsonl.gz")
    parser.add_argument("--taxonomy", default="data/taxonomy/snapshot.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--feed-org", action="append", default=[])
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--skip-near-dedup", action="store_true")
    parser.add_argument("--max-events", type=int, help="Bounded smoke-test build")
    parser.add_argument("--require-enabled-taxonomies", action="store_true")
    parser.add_argument("--allow-bounded-snapshot", action="store_true")
    args = parser.parse_args()
    if not 0 < args.near_threshold <= 1:
        parser.error("--near-threshold must be in (0, 1]")

    snapshot = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    if (
        snapshot.get("summary", {}).get("bounded_smoke_snapshot")
        and not args.allow_bounded_snapshot
    ):
        raise SystemExit("Refusing to build from a bounded smoke-test taxonomy snapshot")
    catalog = TaxonomyCatalog.load(
        args.taxonomy,
        require_enabled_taxonomies=args.require_enabled_taxonomies,
    )
    if args.require_enabled_taxonomies and catalog.enabled_taxonomy_count == 0:
        raise SystemExit("Snapshot contains zero enabled standard taxonomies")

    feed_orgs = {org.lower() for org in args.feed_org}
    dropped: Counter[str] = Counter()
    records: list[EventRecord] = []
    signatures: list[MinHash | None] | None = None if args.skip_near_dedup else []
    empty_text_events = 0
    for raw in iter_jsonl(args.raw):
        record, text = make_record(raw, catalog, feed_orgs, dropped)
        records.append(record)
        if record.text_hash == EMPTY_TEXT_HASH:
            empty_text_events += 1
        if signatures is not None:
            signatures.append(
                None if record.text_hash == EMPTY_TEXT_HASH else minhash(text, args.num_perm)
            )
        if args.max_events and len(records) >= args.max_events:
            break
        if len(records) % 1000 == 0:
            print(f"Canonicalized {len(records)} events", file=sys.stderr, flush=True)
    if not records:
        raise SystemExit("No events found")

    clusters, dedup_stats = deduplicate(
        records,
        signatures,
        near_threshold=args.near_threshold,
        num_perm=args.num_perm,
    )
    del signatures
    representatives = choose_representatives(records, clusters)
    assignments, boundaries = assign_splits(records, representatives)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_counts = write_splits(out_dir, args.raw, records, assignments, args.max_events)

    label_counts: Counter[str] = Counter()
    train_label_counts: Counter[str] = Counter()
    labeled_representatives = 0
    for index, (split, _cluster_id) in assignments.items():
        labels = records[index].labels
        labeled_representatives += bool(labels)
        label_counts.update(labels)
        if split == "train":
            train_label_counts.update(labels)

    manifest = {
        "schema_version": 2,
        "raw_path": str(Path(args.raw)),
        "taxonomy_snapshot": str(Path(args.taxonomy).resolve()),
        "taxonomy_snapshot_date": catalog.snapshot_date,
        "require_enabled_taxonomies": args.require_enabled_taxonomies,
        "warning": (
            "MISP reports zero enabled standard taxonomies; "
            "installed taxonomy definitions were used for validation."
            if catalog.enabled_taxonomy_count == 0 and not args.require_enabled_taxonomies
            else None
        ),
        "config": {
            "near_threshold": args.near_threshold,
            "num_perm": args.num_perm,
            "skip_near_dedup": args.skip_near_dedup,
            "feed_orgs": sorted(feed_orgs),
            "max_events": args.max_events,
        },
        "counts": {
            "raw_events": len(records),
            "empty_text_events": empty_text_events,
            "deduplicated_representatives": len(representatives),
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
            "labeled_representatives": labeled_representatives,
            "unique_labels": len(label_counts),
            "train_labels_with_20_examples": sum(
                1 for count in train_label_counts.values() if count >= 20
            ),
            **dedup_stats,
        },
        "split_boundaries": boundaries,
        "dropped_label_reasons": dict(dropped.most_common()),
        "label_frequencies": dict(label_counts.most_common()),
        "train_label_frequencies": dict(train_label_counts.most_common()),
    }
    write_json_secure(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote processed dataset to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
