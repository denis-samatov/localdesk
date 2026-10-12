import json

import pytest

from localdesk.config import Experiment, TaskSpec
from localdesk.dataset import prepare


def spec(labels=("red", "green", "blue")):
    return TaskSpec.from_dict(
        {
            "id": "sample",
            "version": "1",
            "instructions": "Choose intent.",
            "labels": {label: f"{label} requests" for label in labels},
        }
    )


def write_config(tmp_path, labels=("red", "green", "blue"), **changes):
    (tmp_path / "task.json").write_text(json.dumps(spec(labels).to_dict()))
    rows = [
        {"id": f"{label}-{i}", "message": f"{label} request topic{i}", "intent": label}
        for label in labels
        for i in range(12)
    ]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {
        "task": "task.json",
        "dataset": {
            "path": "rows.jsonl",
            "format": "jsonl",
            "text_column": "message",
            "label_column": "intent",
            "id_column": "id",
        },
        "registry": "registry",
        "training": {"backend": "baseline"},
        "gates": {},
    }
    config.update(changes)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    return Experiment.load(path)


def test_arbitrary_label_taxonomy_and_hash():
    task = spec(("sales", "returns"))
    assert task.labels == ("sales", "returns")
    assert task.questions["category"]["criteria"] == {
        "sales": "sales requests",
        "returns": "returns requests",
    }
    assert task.digest != spec(("sales", "billing")).digest


@pytest.mark.parametrize(
    "change",
    [
        {"labels": {"only": "single"}},
        {"extra": True},
        {"labels": {"a": "", "b": "ok"}},
        {"min_confidence": float("nan")},
    ],
)
def test_bad_task_fails_explicitly(change):
    raw = spec().to_dict()
    raw.update(change)
    with pytest.raises(ValueError):
        TaskSpec.from_dict(raw)


def test_unknown_config_fields_are_not_ignored(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        write_config(tmp_path, trainning={})


def test_data_is_deterministic_stratified_and_task_owned(tmp_path):
    experiment = write_config(tmp_path)
    first = prepare(experiment)
    assert first == prepare(experiment)
    for rows in first.rows.values():
        assert {row["gold"]["category"] for row in rows} == set(experiment.task.labels)
        assert all(row["questions"] == experiment.task.questions for row in rows)
    groups = [set(row["group"] for row in rows) for rows in first.rows.values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))


def test_unknown_data_label_fails(tmp_path):
    experiment = write_config(tmp_path)
    with (tmp_path / "rows.jsonl").open("a") as file:
        file.write(
            json.dumps({"id": "bad", "message": "unexpected class", "intent": "wrong"}) + "\n"
        )
    with pytest.raises(ValueError, match="unknown label"):
        prepare(experiment)


def test_conflicting_equivalent_text_labels_fail(tmp_path):
    experiment = write_config(tmp_path)
    with (tmp_path / "rows.jsonl").open("a") as file:
        file.write(
            json.dumps({"id": "bad", "message": "red request topic0!", "intent": "green"}) + "\n"
        )
    with pytest.raises(ValueError, match="conflicting"):
        prepare(experiment)


def test_two_unique_samples_per_class_are_insufficient(tmp_path):
    experiment = write_config(tmp_path)
    rows = [
        {"id": f"{label}-{i}", "message": f"{label} topic{i}", "intent": label}
        for label in experiment.task.labels
        for i in range(2)
    ]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="groups|insufficient"):
        prepare(experiment)


def test_conversation_and_equivalence_merge_transitively(tmp_path):
    write_config(tmp_path)
    raw = json.loads((tmp_path / "experiment.json").read_text())
    raw["dataset"]["group_column"] = "conversation"
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    for row in rows:
        row["conversation"] = row["id"]
    rows[0]["conversation"] = rows[1]["conversation"] = "thread-A"
    rows.append({**rows[0], "id": "duplicate-A", "conversation": "thread-B"})
    rows[2]["conversation"] = "thread-B"
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "experiment.json").write_text(json.dumps(raw))
    result = prepare(Experiment.load(tmp_path / "experiment.json"))
    merged = [
        (split, row["group"])
        for split, examples in result.rows.items()
        for row in examples
        if row["id"] in {rows[0]["id"], rows[1]["id"], rows[2]["id"], "duplicate-A"}
    ]
    assert len(set(merged)) == 1


def test_explicit_equivalent_cross_split_is_rejected(tmp_path):
    experiment = write_config(tmp_path)
    raw = json.loads((tmp_path / "experiment.json").read_text())
    raw["dataset"].pop("path")
    raw["dataset"]["splits"] = {
        split: f"{split}.jsonl" for split in ("train", "calibration", "test")
    }
    for split in ("train", "calibration", "test"):
        examples = [
            {"id": f"{split}-{label}", "message": f"{label} identical request", "intent": label}
            for label in experiment.task.labels
        ]
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in examples)
        )
    (tmp_path / "experiment.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="crosses"):
        prepare(Experiment.load(tmp_path / "experiment.json"))


def test_one_mixed_conversation_does_not_require_three_of_its_label_combination(tmp_path):
    write_config(tmp_path, ("red", "green"))
    raw = json.loads((tmp_path / "experiment.json").read_text())
    raw["dataset"]["group_column"] = "conversation"
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    for row in rows:
        row["conversation"] = row["id"]
    rows[0]["conversation"] = rows[12]["conversation"] = "mixed"
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "experiment.json").write_text(json.dumps(raw))
    result = prepare(Experiment.load(tmp_path / "experiment.json"))
    for examples in result.rows.values():
        assert {r["gold"]["category"] for r in examples} == {"red", "green"}
    locations = [
        split
        for split, examples in result.rows.items()
        if any(r["id"] == rows[0]["id"] for r in examples)
    ]
    assert len(locations) == 1


def test_csv_column_names_with_dots_are_literal(tmp_path):
    import csv

    write_config(tmp_path, ("red", "green"))
    config = json.loads((tmp_path / "experiment.json").read_text())
    config["dataset"] = {
        "path": "rows.csv",
        "format": "csv",
        "text_column": "ticket.text",
        "label_column": "ticket.intent",
    }
    with (tmp_path / "rows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ticket.text", "ticket.intent"])
        writer.writeheader()
        for label in ("red", "green"):
            for i in range(12):
                writer.writerow(
                    {"ticket.text": f"{label} message topic{i}", "ticket.intent": label}
                )
    (tmp_path / "experiment.json").write_text(json.dumps(config))
    prepared = prepare(Experiment.load(tmp_path / "experiment.json"))
    assert all(
        {row["gold"]["category"] for row in examples} == {"red", "green"}
        for examples in prepared.rows.values()
    )


def test_invalid_jsonl_reports_line(tmp_path):
    write_config(tmp_path)  # Provides task.json beside the experiment.
    (tmp_path / "broken.jsonl").write_text(
        '{"message": "a", "intent": "red"}\n\n{broken\n', encoding="utf-8"
    )
    config = tmp_path / "broken.json"
    config.write_text(
        json.dumps(
            {
                "task": "task.json",
                "dataset": {
                    "path": "broken.jsonl",
                    "format": "jsonl",
                    "text_column": "message",
                    "label_column": "intent",
                },
                "registry": "registry",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="all line 3: invalid JSON"):
        prepare(Experiment.load(config))
