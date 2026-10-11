"""Verified Zenodo ingestion, explicit exclusions and leakage-free split construction."""

import csv
import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from localdesk.baseline import features
from localdesk.core import validate_text
from localdesk.schema import CATEGORIES, QUESTIONS, schema_hash

SOURCE = "https://zenodo.org/records/7648117"
SOURCE_HASHES = {
    "X_train.csv": "789712f97fb472544eeb82d8f5a142ed46e0bf254c6c34ac53a9f1575a9d460e",
    "y_train.csv": "cb851ff4d5e792886504b3040bcc85a5dbf94514d31293eddd6d69fcb7278d53",
    "X_test.csv": "4b808bc32c57f063960a3a4e7a4a30769c96be8143649aadc86835f4cb9763e5",
    "y_test.csv": "2b10db5e2f071e649efb3d40dd007ec62bc661f35b4afaf44117a00865760f94",
}
SPLITS = ("train", "calibration", "test", "ood")


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_signature(text: str) -> str:
    # Group the representation the baseline actually sees, including masked-tag
    # removal, punctuation/tokenization and feature hashing equivalence.
    representation = [[index, round(value, 12)] for index, value in features(text).items()]
    return hashlib.sha256(json.dumps(representation, separators=(",", ":")).encode()).hexdigest()


def read_source(directory: Path) -> list[dict]:
    for name, expected in SOURCE_HASHES.items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"{name}: source SHA-256 mismatch")
    records = []
    for split in ("train", "test"):
        with (directory / f"X_{split}.csv").open(encoding="utf-8", newline="") as file:
            texts = list(csv.DictReader(file))
        with (directory / f"y_{split}.csv").open(encoding="utf-8", newline="") as file:
            labels = list(csv.DictReader(file))
        by_id = {row["id"]: row["category_truth"] for row in labels}
        if len(by_id) != len(labels) or len({row["id"] for row in texts}) != len(texts):
            raise ValueError("source IDs must be unique within each split")
        if set(by_id) != {row["id"] for row in texts}:
            raise ValueError("source text and label IDs do not match")
        records.extend(
            {
                "id": row["id"],
                "text": row["text"],
                "label": by_id[row["id"]],
                "source_split": split,
            }
            for row in texts
        )
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("source IDs overlap across official splits")
    return records


def build_splits(records: list[dict], seed: int = 3407) -> tuple[dict, dict]:
    groups = defaultdict(list)
    raw_groups = defaultdict(list)
    for record in records:
        validate_text(record["text"])
        if record["label"] not in {*CATEGORIES, "EOL"}:
            raise ValueError("unexpected source label")
        groups[group_signature(record["text"])].append(record)
        raw_groups[normalized(record["text"])].append(record)
    rows = {split: [] for split in SPLITS}
    audit = {
        "source_rows": len(records),
        "normalized_groups": len(raw_groups),
        "feature_equivalent_groups": len(groups),
        "raw_cross_split_groups": sum(
            len({r["source_split"] for r in members}) > 1 for members in raw_groups.values()
        ),
        "conflicting_groups": 0,
        "cross_split_groups": 0,
        "excluded": [],
    }
    candidates = defaultdict(list)
    for signature, members in sorted(groups.items()):
        if len({r["source_split"] for r in members}) > 1:
            audit["cross_split_groups"] += 1
        if len({r["label"] for r in members}) > 1:
            audit["conflicting_groups"] += 1
            audit["excluded"].extend(
                {"id": r["id"], "reason": "conflicting_labels"} for r in members
            )
            continue
        # Keep one official test representative when a group crosses the source split.
        member = min(members, key=lambda r: (r["source_split"] != "test", r["id"]))
        for record in members:
            if record is not member:
                audit["excluded"].append(
                    {
                        "id": record["id"],
                        "reason": "preprocessing_equivalent_duplicate",
                    }
                )
        label = member["label"]
        row = {
            "id": member["id"],
            "group": signature,
            "state": member["text"],
            "questions": QUESTIONS,
            "gold": {"category": label},
            "source_label": label,
            "source_split": member["source_split"],
        }
        if label == "EOL":
            # Two unique automated templates cannot support three disjoint splits.
            rows["ood"].append(row)
        elif member["source_split"] == "test":
            rows["test"].append(row)
        else:
            candidates[label].append(row)
    rng = random.Random(seed)
    for label in CATEGORIES:
        examples = candidates[label]
        if len(examples) < 2:
            raise ValueError(f"{label}: not enough unique training examples")
        rng.shuffle(examples)
        calibration_count = max(1, int(len(examples) * 0.2))
        rows["calibration"].extend(examples[:calibration_count])
        rows["train"].extend(examples[calibration_count:])
    for split in rows:
        rows[split].sort(key=lambda r: r["id"])
    audit["seed"] = seed
    audit["supported_categories"] = list(CATEGORIES)
    audit["ood_category"] = "EOL"
    return rows, audit


def write_dataset(directory: Path, rows: dict, audit: dict, source_hashes: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, examples in rows.items():
        path = directory / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in examples),
            encoding="utf-8",
        )
        hashes[split] = file_hash(path)
    manifest = {
        "schema_sha256": schema_hash(),
        "sha256": hashes,
        "source_sha256": source_hashes,
        "counts": {split: len(examples) for split, examples in rows.items()},
        "evidence": "public_real_ticket_cleaned_holdout",
        "source": SOURCE,
        "license": "CC-BY-4.0",
        "creator": "Leonardo Santiago Benitez Pereira",
        "audit": audit,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_splits(directory: Path) -> dict[str, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_sha256"] != schema_hash():
        raise ValueError("dataset schema does not match this application")
    result, seen_ids, seen_groups = {}, set(), set()
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if file_hash(path) != manifest["sha256"][split]:
            raise ValueError(f"{split}: dataset hash mismatch")
        examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(examples) != manifest["counts"][split] or (not examples and split != "ood"):
            raise ValueError(f"{split}: invalid row count")
        for row in examples:
            required = {"id", "group", "state", "questions", "gold", "source_label", "source_split"}
            if set(row) != required or row["questions"] != QUESTIONS:
                raise ValueError(f"{split}: invalid dataset schema")
            allowed = {"EOL"} if split == "ood" else set(CATEGORIES)
            if (
                set(row["gold"]) != {"category"}
                or row["gold"]["category"] not in allowed
                or row["gold"]["category"] != row["source_label"]
            ):
                raise ValueError(f"{split}: invalid label")
            text = validate_text(row["state"])
            group = group_signature(text)
            if group != row["group"]:
                raise ValueError("text does not match its group hash")
            if row["id"] in seen_ids or group in seen_groups:
                raise ValueError("duplicate id or normalized text group overlap")
            seen_ids.add(row["id"])
            seen_groups.add(group)
        result[split] = examples
    return result
