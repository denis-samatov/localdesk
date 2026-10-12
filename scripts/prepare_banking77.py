"""Fetch the pinned Banking77 CSVs, verify them, and write isolated JSONL splits.

Calibration is a deterministic, label-stratified 15% of the official training data;
the official test set is kept intact (minus model-equivalent duplicates).
"""

import argparse
import csv
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from localdesk.sources import fetch_verified, isolate_splits, write_jsonl_splits

COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{COMMIT}/banking_data"
SOURCES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
SEED = 3407


def read_rows(payload: bytes, split: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    if not rows or set(rows[0]) != {"text", "category"}:
        raise ValueError(f"{split}: unexpected Banking77 columns")
    return [
        {"id": f"{split}-{index:05d}", "text": row["text"], "label": row["category"]}
        for index, row in enumerate(rows)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--out", type=Path, default=Path("data/banking77"))
    args = parser.parse_args()
    source = {
        name: read_rows(
            fetch_verified(f"{BASE}/{name}", digest, limit=2_000_000, cache=args.cache / name),
            name.removesuffix(".csv"),
        )
        for name, digest in SOURCES.items()
    }
    by_label = defaultdict(list)
    for row in source["train.csv"]:
        by_label[row["label"]].append(row)
    records = [dict(row, split="test") for row in source["test.csv"]]
    rng = random.Random(SEED)
    for label in sorted(by_label):
        rows = sorted(by_label[label], key=lambda row: row["id"])
        rng.shuffle(rows)
        cut = round(len(rows) * 0.15)
        records += [dict(row, split="calibration") for row in rows[:cut]]
        records += [dict(row, split="train") for row in rows[cut:]]
    splits, audit = isolate_splits(records)
    manifest = write_jsonl_splits(
        args.out,
        splits,
        {
            "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
            "commit": COMMIT,
            "source_sha256": SOURCES,
            "license": "CC-BY-4.0",
            "seed": SEED,
            "audit": audit,
        },
    )
    print(json.dumps({"counts": audit["counts"], "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
