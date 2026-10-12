"""Pinned public dataset downloads and split isolation for source adapters."""

import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from localdesk.core import validate_text
from localdesk.data import file_hash, group_signature


def read_bounded(url: str, limit: int, *, attempts: int = 3) -> bytes:
    """Read a whole response, retrying bodies cut short before their Content-Length.

    A single read() may legally return part of a body, and a dropped connection can
    end the stream early without an error.
    """
    for _ in range(attempts):
        chunks, size = [], 0
        with urllib.request.urlopen(url, timeout=60) as response:
            expected = response.headers.get("Content-Length")
            while chunk := response.read(1 << 16):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"{url}: unexpected source file size")
                chunks.append(chunk)
        if expected is None or size == int(expected):
            return b"".join(chunks)
    raise OSError(f"{url}: incomplete download after {attempts} attempts")


def fetch_verified(url: str, sha256: str, *, limit: int, cache: Path | None = None) -> bytes:
    """Download a pinned public file (or reuse a cached copy) and verify its SHA-256."""
    if cache is not None and cache.is_file():
        payload = cache.read_bytes()
    else:
        payload = read_bounded(url, limit)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"{url}: source SHA-256 mismatch")
    if cache is not None and not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    return payload


def isolate_splits(records: list[dict], preference=("test", "calibration", "train")):
    """Collapse model-equivalent texts so no group crosses splits; drop conflicting labels.

    Each record needs id, text, label and split. A duplicate group keeps one member,
    preferring the earliest split in ``preference`` (official test examples survive).
    """
    groups = defaultdict(list)
    for record in records:
        groups[group_signature(validate_text(record["text"]))].append(record)
    rank = {split: index for index, split in enumerate(preference)}
    kept, audit = defaultdict(list), {"source_rows": len(records), "conflicting_rows": 0}
    audit["duplicate_rows"] = 0
    for members in groups.values():
        if len({record["label"] for record in members}) > 1:
            audit["conflicting_rows"] += len(members)
            continue
        keep = min(members, key=lambda record: (rank[record["split"]], record["id"]))
        audit["duplicate_rows"] += len(members) - 1
        kept[keep["split"]].append(keep)
    for split in kept:
        kept[split].sort(key=lambda record: record["id"])
    audit["counts"] = {split: len(kept[split]) for split in preference}
    return dict(kept), audit


def write_jsonl_splits(directory: Path, splits: dict, manifest: dict) -> dict:
    """Write id/text/label JSONL per split plus a manifest of file hashes."""
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, examples in splits.items():
        path = directory / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {"id": r["id"], "text": r["text"], "label": r["label"]}, ensure_ascii=False
                )
                + "\n"
                for r in examples
            ),
            encoding="utf-8",
            newline="\n",
        )
        hashes[split] = file_hash(path)
    manifest = dict(manifest, sha256=hashes)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
