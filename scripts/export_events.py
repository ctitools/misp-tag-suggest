#!/usr/bin/env python3
"""Stream full MISP Events into resumable, secure JSONL gzip shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ijson
import requests

from tagrec.canonicalize import unwrap_event
from tagrec.misp import MISPSettings, load_misp_settings

_THREAD_LOCAL = threading.local()


def thread_session(settings: MISPSettings) -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": settings.key,
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "misp-taxonomy-recommender-export/1",
            }
        )
        _THREAD_LOCAL.session = session
    return session


def part_path(parts_dir: Path, page: int, compressed: bool) -> Path:
    suffix = ".jsonl.gz" if compressed else ".jsonl"
    return parts_dir / f"page-{page:07d}{suffix}"


def fetch_page_to_part(
    settings: MISPSettings,
    page: int,
    page_size: int,
    since: str | None,
    until: str | None,
    retries: int,
    path: Path,
    event_limit: int | None = None,
) -> tuple[int, int, str]:
    """Stream one REST page directly to an atomic shard and return metadata only."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, object] = {
        "returnFormat": "json",
        "limit": page_size,
        "page": page,
        "metadata": False,
        "withAttachments": False,
    }
    if since:
        payload["from"] = since
    if until:
        payload["to"] = until

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        count = 0
        digest = hashlib.sha256()
        response: requests.Response | None = None
        try:
            response = thread_session(settings).post(
                settings.url + "/events/restSearch",
                json=payload,
                stream=True,
                verify=settings.verify_tls,
                timeout=(20, 900),
            )
            response.raise_for_status()
            response.raw.decode_content = True
            if path.suffix == ".gz":
                output = gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3)
            else:
                output = temporary.open("w", encoding="utf-8")
            with output as destination:
                # MISP's direct REST response is {"response": [{"Event": ...}, ...]}.
                for raw in ijson.items(response.raw, "response.item"):
                    if not isinstance(raw, dict):
                        continue
                    event = unwrap_event(raw)
                    if not event:
                        continue
                    destination.write(
                        json.dumps({"Event": event}, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    identity = f"{event.get('id', '')}:{event.get('uuid', '')}\n"
                    digest.update(identity.encode("utf-8"))
                    count += 1
                    if event_limit is not None and count >= event_limit:
                        break
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            return page, count, digest.hexdigest()
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            if response is not None:
                response.close()
    raise RuntimeError(f"Page {page} failed after {retries} attempts") from last_error


def count_part(path: Path) -> int:
    if path.suffix == ".gz":
        source = gzip.open(path, "rt", encoding="utf-8")
    else:
        source = path.open(encoding="utf-8")
    with source as f:
        return sum(1 for line in f if line.strip())


def existing_progress(
    parts_dir: Path,
    page_size: int,
    compressed: bool,
) -> tuple[int, int, int | None]:
    """Return next page, event count and an already-known final page."""
    page = 1
    total = 0
    while True:
        path = part_path(parts_dir, page, compressed)
        if not path.exists():
            return page, total, None
        count = count_part(path)
        total += count
        if count < page_size:
            return page + 1, total, page
        page += 1


def concatenate_parts(destination: Path, parts_dir: Path, final_page: int, compressed: bool) -> None:
    """Concatenate gzip members (or plain shards) without decompression."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as output:
            for page in range(1, final_page + 1):
                path = part_path(parts_dir, page, compressed)
                if not path.exists():
                    raise RuntimeError(f"Missing export shard {path}")
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--out", default="data/raw/events.jsonl.gz")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--since", help="Earliest event date accepted by MISP")
    parser.add_argument("--until", help="Latest event date accepted by MISP")
    parser.add_argument("--max-events", type=int, help="Bounded smoke-test export")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Refuse to start unless this much disk remains free",
    )
    args = parser.parse_args()
    if args.page_size < 1 or args.workers < 1 or args.retries < 1:
        parser.error("page size, workers and retries must be positive")
    if args.max_events is not None and args.max_events < 1:
        parser.error("max events must be positive")
    if args.max_events is not None:
        args.workers = 1

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed = destination.suffix == ".gz"
    parts_dir = destination.parent / f".{destination.name}.parts"

    if destination.exists() and not args.overwrite:
        print(f"Export already exists: {destination}; use --overwrite to replace it")
        return 0
    if args.overwrite:
        destination.unlink(missing_ok=True)
        shutil.rmtree(parts_dir, ignore_errors=True)
    destination.with_suffix(destination.suffix + ".tmp").unlink(missing_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(parts_dir, 0o700)

    stat = os.statvfs(destination.parent)
    free_gb = stat.f_bavail * stat.f_frsize / 1024**3
    if free_gb < args.min_free_gb:
        raise SystemExit(f"Only {free_gb:.2f} GiB free; require {args.min_free_gb:.2f} GiB")

    settings = load_misp_settings(args.env)
    next_page, total, final_page = existing_progress(parts_dir, args.page_size, compressed)
    if total:
        print(f"Resuming after page {next_page - 1} ({total} events already saved)", file=sys.stderr)

    previous_signature: str | None = None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while final_page is None:
            pages = list(range(next_page, next_page + args.workers))
            futures = {}
            for page in pages:
                remaining = args.max_events - total if args.max_events is not None else None
                future = executor.submit(
                    fetch_page_to_part,
                    settings,
                    page,
                    args.page_size,
                    args.since,
                    args.until,
                    args.retries,
                    part_path(parts_dir, page, compressed),
                    remaining,
                )
                futures[future] = page

            results: dict[int, tuple[int, str]] = {}
            for future in as_completed(futures):
                page, count, signature = future.result()
                results[page] = (count, signature)

            for page in pages:
                count, signature = results[page]
                if signature and signature == previous_signature:
                    print(
                        f"MISP repeated page {page - 1} as page {page}; stopping before duplicate data",
                        file=sys.stderr,
                    )
                    # The repeated shard is discarded, so the export ends at its predecessor.
                    final_page = page - 1
                    for extra_page in pages:
                        if extra_page > final_page:
                            part_path(parts_dir, extra_page, compressed).unlink(missing_ok=True)
                    break
                total += count
                print(
                    f"Saved page {page}: {count} events ({total} total)",
                    file=sys.stderr,
                    flush=True,
                )
                if count < args.page_size or (
                    args.max_events is not None and total >= args.max_events
                ):
                    final_page = page
                    # Remove concurrently fetched shards beyond the first final page.
                    for extra_page in pages:
                        if extra_page > final_page:
                            part_path(parts_dir, extra_page, compressed).unlink(missing_ok=True)
                    break
                previous_signature = signature
            next_page += args.workers

    if final_page is None:
        raise RuntimeError("Exporter stopped without finding a final page")
    concatenate_parts(destination, parts_dir, final_page, compressed)
    shutil.rmtree(parts_dir)
    print(f"Exported {total} full events to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
