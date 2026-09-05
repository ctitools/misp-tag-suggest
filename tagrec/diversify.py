"""Deterministic, display-only similar-event diversification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def collapse_similar_neighbors(
    neighbors: Iterable[dict[str, Any]], top_k: int = 10
) -> list[dict[str, Any]]:
    """Keep the highest-scoring exemplar for each template/duplicate group.

    Input must already be similarity-ranked. A neighbor conflicts with a retained
    exemplar when either its non-empty exact duplicate cluster or its exact
    ``(orgc, normalized_info)`` key matches. The returned copy carries group_size.
    """
    rows = list(neighbors)
    template_counts = Counter(
        (str(x.get("orgc", "")), str(x.get("normalized_info", ""))) for x in rows
    )
    cluster_counts = Counter(str(x.get("dup_cluster", "")) for x in rows if x.get("dup_cluster"))
    seen_templates = set()
    seen_clusters = set()
    out = []
    for row in rows:
        template = (str(row.get("orgc", "")), str(row.get("normalized_info", "")))
        cluster = str(row.get("dup_cluster", ""))
        if template in seen_templates or (cluster and cluster in seen_clusters):
            continue
        item = dict(row)
        item["group_size"] = max(template_counts[template], cluster_counts.get(cluster, 0))
        out.append(item)
        seen_templates.add(template)
        if cluster:
            seen_clusters.add(cluster)
        if len(out) >= top_k:
            break
    return out
