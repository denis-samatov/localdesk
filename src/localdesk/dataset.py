"""Generic text intake with transitive groups and strict split isolation."""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from localdesk.config import Experiment
from localdesk.core import validate_text
from localdesk.data import file_hash, group_signature


@dataclass
class Prepared:
    rows: dict[str, list[dict]]
    manifest: dict


def _mixed_allocation(groups, labels, seed, calibration_fraction, test_fraction):
    """Allocate whole mixed conversations, covering every class in every split.

    This is a bounded deterministic heuristic, not a universal graph partition solver.
    Explicit splits remain available when constrained conversations cannot be allocated.
    """
    support = Counter(label for _, members in groups for label in {r["label"] for r in members})
    if any(support[label] < 3 for label in labels):
        raise ValueError("insufficient independent groups per label: need >=3")
    totals = Counter(r["label"] for _, members in groups for r in members)
    proportions = {
        "train": 1 - calibration_fraction - test_fraction,
        "calibration": calibration_fraction,
        "test": test_fraction,
    }
    for attempt in range(32):
        ordered = list(groups)
        random.Random(seed + attempt).shuffle(ordered)
        ordered.sort(key=lambda item: min(support[r["label"]] for r in item[1]))
        remaining = support.copy()
        counts = {split: Counter() for split in proportions}
        rows = {split: [] for split in proportions}
        for group, members in ordered:
            present = {r["label"] for r in members}
            candidates = set(proportions)
            for label in present:
                missing = {split for split in proportions if counts[split][label] == 0}
                if missing and remaining[label] <= len(missing):
                    candidates &= missing
            if not candidates:
                break

            def score(split, counts=counts, present=present):
                coverage = sum(counts[split][label] == 0 for label in present)
                deficit = sum(
                    proportions[split] - counts[split][label] / totals[label] for label in present
                )
                return (coverage, deficit, split)

            chosen = max(sorted(candidates), key=score)
            rows[chosen].extend(dict(record, group=group) for record in members)
            counts[chosen].update(r["label"] for r in members)
            remaining.subtract(present)
        else:
            if all(all(counts[split][label] for label in labels) for split in proportions):
                return rows
    raise ValueError("cannot allocate mixed conversation groups; provide explicit isolated splits")


_BLANK = object()


def _json_line(split: str, number: int, line: str):
    if not line.strip():
        return _BLANK
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{split} line {number}: invalid JSON ({error.msg})") from error


def _field(row: dict, path: str):
    if path in row:
        return row[path]
    value = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing dataset column: {path}")
        value = value[key]
    return value


def prepare(experiment: Experiment) -> Prepared:
    settings = experiment.dataset
    paths = {"all": settings["path"]} if "path" in settings else settings["splits"]
    records, sources, ids = [], {}, set()
    for split, relative in sorted(paths.items()):
        path = (experiment.base / relative).resolve()
        if path.stat().st_size > 100_000_000:
            raise ValueError("dataset file exceeds 100 MB; partition it before intake")
        sources[split] = file_hash(path)
        if "sha256" in settings and settings["sha256"][split] != sources[split]:
            raise ValueError(f"{split}: source checksum mismatch")
        with path.open(encoding="utf-8-sig", newline="") as file:
            raw_rows = (
                csv.DictReader(file)
                if settings["format"] == "csv"
                else (_json_line(split, number, line) for number, line in enumerate(file, 1))
            )
            raw_rows = (raw for raw in raw_rows if raw is not _BLANK)
            for index, raw in enumerate(raw_rows):
                if not isinstance(raw, dict):
                    raise ValueError(f"{split} row {index + 1}: expected object")
                text = validate_text(_field(raw, settings["text_column"]))
                label = _field(raw, settings["label_column"])
                if not isinstance(label, str) or label not in experiment.task.labels:
                    raise ValueError(f"{split} row {index + 1}: unknown label")
                identifier = (
                    _field(raw, settings["id_column"])
                    if "id_column" in settings
                    else f"{split}-{index}"
                )
                if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
                    raise ValueError("record id must be a string or integer")
                identifier = str(identifier)
                if not identifier or identifier in ids:
                    raise ValueError("duplicate or empty record id")
                ids.add(identifier)
                conversation = (
                    _field(raw, settings["group_column"])
                    if "group_column" in settings
                    else identifier
                )
                if (
                    not isinstance(conversation, (str, int))
                    or isinstance(conversation, bool)
                    or not str(conversation)
                ):
                    raise ValueError("group must be a non-empty string or integer")
                records.append(
                    {
                        "id": identifier,
                        "state": text,
                        "label": label,
                        "split": split,
                        "conversation": str(conversation),
                        "signature": group_signature(text),
                    }
                )
    if not records:
        raise ValueError("dataset is empty")
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen, signatures = {}, defaultdict(list)
    for index, record in enumerate(records):
        signatures[record["signature"]].append(index)
        for key in ("conversation:" + record["conversation"], "feature:" + record["signature"]):
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    for members in signatures.values():
        if len({records[i]["label"] for i in members}) != 1:
            raise ValueError("conflicting labels for equivalent model inputs")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    collapsed = []
    for members in groups.values():
        if len({r["split"] for r in members}) > 1:
            raise ValueError("conversation or equivalent-input group crosses explicit splits")
        unique = {r["signature"]: r for r in sorted(members, key=lambda r: r["id"], reverse=True)}
        group = hashlib.sha256("\n".join(sorted(r["id"] for r in members)).encode()).hexdigest()
        collapsed.append((group, list(unique.values())))
    rows = {split: [] for split in ("train", "calibration", "test")}
    if "all" in paths and any(len({r["label"] for r in members}) > 1 for _, members in collapsed):
        rows = _mixed_allocation(
            collapsed,
            experiment.task.labels,
            experiment.seed,
            experiment.calibration_fraction,
            experiment.test_fraction,
        )
    elif "all" in paths:
        # Stratify groups by their label sets; mixed-label conversations stay intact.
        buckets = defaultdict(list)
        for group, members in collapsed:
            buckets[tuple(sorted({r["label"] for r in members}))].append((group, members))
        rng = random.Random(experiment.seed)
        for labels, bucket in sorted(buckets.items()):
            bucket.sort(key=lambda item: item[0])
            rng.shuffle(bucket)
            if len(bucket) < 3:
                raise ValueError(
                    f"insufficient independent groups for label set {labels}: need >=3"
                )
            n_cal = max(1, int(len(bucket) * experiment.calibration_fraction))
            n_test = max(1, int(len(bucket) * experiment.test_fraction))
            for index, (group, members) in enumerate(bucket):
                split = (
                    "calibration"
                    if index < n_cal
                    else "test"
                    if index < n_cal + n_test
                    else "train"
                )
                for record in members:
                    record = dict(record, group=group)
                    rows[split].append(record)
    else:
        for group, members in collapsed:
            rows[members[0]["split"]].extend(dict(record, group=group) for record in members)
    for split, examples in rows.items():
        if {r["label"] for r in examples} != set(experiment.task.labels):
            raise ValueError(f"{split}: each configured label requires support; supply more groups")
        rows[split] = [
            {
                "id": r["id"],
                "state": r["state"],
                "group": r["group"],
                "questions": experiment.task.questions,
                "gold": {"category": r["label"]},
            }
            for r in sorted(examples, key=lambda r: r["id"])
        ]
    manifest = {
        "format": "localdesk-dataset-v1",
        "task_sha256": experiment.task.digest,
        "source_sha256": sources,
        "seed": experiment.seed,
        "raw_count": len(records),
        "duplicate_count": len(records) - sum(len(v) for v in rows.values()),
        "counts": {s: len(v) for s, v in rows.items()},
        "class_support": {
            s: dict(Counter(r["gold"]["category"] for r in v)) for s, v in rows.items()
        },
        "split_sha256": {
            s: hashlib.sha256(
                json.dumps(v, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            for s, v in rows.items()
        },
        "provenance": settings.get("provenance", "user_supplied_unverified"),
    }
    return Prepared(rows, manifest)
