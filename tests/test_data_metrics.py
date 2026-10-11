import math

import pytest
from test_core import FixedBackend

from localdesk.data import build_splits, load_splits, write_dataset
from localdesk.metrics import classification_metrics, evaluate_rows
from localdesk.schema import CATEGORIES, QUESTIONS


def source_records():
    return [
        {
            "id": f"{label}-{i}",
            "text": f"{label} unique symptom number{i}",
            "label": label,
            "source_split": "test" if i == 5 else "train",
        }
        for label in CATEGORIES
        for i in range(6)
    ]


def test_split_is_deterministic_and_no_groups_cross_splits(tmp_path):
    rows, audit = build_splits(source_records())
    assert rows == build_splits(source_records())[0]
    groups = {split: {r["group"] for r in examples} for split, examples in rows.items()}
    assert not (groups["train"] & groups["calibration"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["calibration"] & groups["test"])
    for split in ("train", "calibration", "test"):
        assert {row["gold"]["category"] for row in rows[split]} == set(CATEGORIES)
        assert all(row["questions"] == QUESTIONS for row in rows[split])
    write_dataset(tmp_path, rows, audit, {})
    assert load_splits(tmp_path) == rows


def test_duplicates_prioritize_official_test_and_conflicts_are_removed():
    raw = source_records()
    raw.append(
        {**raw[-1], "id": "leaked", "source_split": "train", "text": raw[-1]["text"].upper()}
    )
    raw.extend(
        [
            {"id": "c1", "text": "conflicting label", "label": "Software", "source_split": "train"},
            {
                "id": "c2",
                "text": "conflicting label",
                "label": "Fileservice",
                "source_split": "test",
            },
        ]
    )
    rows, audit = build_splits(raw)
    assert audit["cross_split_groups"] == 2  # Includes the conflicting train/test group.
    assert audit["conflicting_groups"] == 1
    assert not any(
        row["id"] in {"leaked", "c1", "c2"} for examples in rows.values() for row in examples
    )


def test_preprocessing_equivalent_inputs_never_cross_splits():
    from localdesk.baseline import features

    raw = source_records()
    raw.extend(
        [
            {
                "id": "p1",
                "text": "Cannot open [NAME] shared folder.",
                "label": "Fileservice",
                "source_split": "train",
            },
            {
                "id": "p2",
                "text": "Cannot open [EMAIL] shared folder!",
                "label": "Fileservice",
                "source_split": "test",
            },
        ]
    )
    rows, _ = build_splits(raw)
    signatures = {
        split: {tuple(features(row["state"]).items()) for row in examples}
        for split, examples in rows.items()
    }
    assert not (signatures["train"] & signatures["test"])
    assert not (signatures["calibration"] & signatures["test"])
    assert not (signatures["train"] & signatures["calibration"])


def test_eol_is_not_given_invented_training_examples():
    rows, audit = build_splits(
        source_records()
        + [
            {"id": "eol1", "text": "End Of Life machine", "label": "EOL", "source_split": "train"},
            {"id": "eol2", "text": "End Of Life machine", "label": "EOL", "source_split": "test"},
        ]
    )
    assert len(rows["ood"]) == 1
    assert rows["ood"][0]["source_label"] == "EOL"
    assert all(
        row["source_label"] != "EOL"
        for split in ("train", "calibration", "test")
        for row in rows[split]
    )


def test_loader_rejects_overlap_even_when_hashes_match(tmp_path):
    rows, audit = build_splits(source_records())
    rows["test"].append({**rows["train"][0], "id": "different-id"})
    write_dataset(tmp_path, rows, audit, {})
    with pytest.raises(ValueError, match="overlap|duplicate"):
        load_splits(tmp_path)


def test_loader_detects_tampering(tmp_path):
    rows, audit = build_splits(source_records())
    write_dataset(tmp_path, rows, audit, {})
    with (tmp_path / "test.jsonl").open("a") as file:
        file.write("{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_splits(tmp_path)


def test_loader_validates_labels_after_hash_check(tmp_path):
    rows, audit = build_splits(source_records())
    rows["test"][0]["gold"]["category"] = "invented"
    write_dataset(tmp_path, rows, audit, {})
    with pytest.raises(ValueError, match="label"):
        load_splits(tmp_path)


def test_known_metric_values():
    metrics = classification_metrics(
        ["a", "b"], [{"a": 0.8, "b": 0.2}, {"a": 0.4, "b": 0.6}], ["a", "b"]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] == pytest.approx(0.2)
    assert metrics["nll"] == pytest.approx((-math.log(0.8) - math.log(0.6)) / 2)
    assert metrics["ece"] == pytest.approx(0.3)
    assert metrics["per_class_recall"] == {"a": 1.0, "b": 1.0}


def test_evaluation_counts_acceptance_and_reviews():
    rows, _ = build_splits(source_records())
    result = evaluate_rows(rows["test"], FixedBackend())
    assert result["coverage"] == 1
    assert result["accepted_accuracy"] == pytest.approx(1 / len(CATEGORIES))
    rejected = evaluate_rows(rows["test"], FixedBackend(0.3))
    assert rejected["accepted_accuracy"] is None
    assert rejected["coverage"] == 0


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        classification_metrics([], [], ["a", "b"])
