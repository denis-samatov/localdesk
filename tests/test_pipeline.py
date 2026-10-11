import gzip
import json

import pytest
from test_configurable import write_config

from localdesk.pipeline import run_experiment
from localdesk.registry import Registry


@pytest.mark.parametrize("labels", [("red", "green"), ("red", "green", "blue")])
def test_real_fit_export_and_registry_support_new_labels(tmp_path, labels):
    experiment = write_config(tmp_path, labels, gates={"min_accuracy": 0.0})
    run = run_experiment(experiment, run_id="first")
    registry = Registry(experiment.registry)
    registry.promote("first")
    backend, manifest = registry.load_active()
    assert manifest["run_id"] == "first"
    assert set(backend.predict("red request topic0")["category"]) == set(labels)
    assert backend.task.digest == experiment.task.digest
    assert run["report"]["classification"]["count"] > 0
    assert "macro_f1" in run["report"]["classification"]
    assert (experiment.registry / "first" / "training-plan.json").is_file()


def test_failed_gate_cannot_replace_active_version(tmp_path):
    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="good")
    registry = Registry(experiment.registry)
    registry.promote("good")
    impossible = write_config(tmp_path, gates={"min_coverage": 1.0})
    raw_task = json.loads((tmp_path / "task.json").read_text())
    raw_task["min_confidence"] = 1.0  # Force every finite softmax below acceptance.
    (tmp_path / "task.json").write_text(json.dumps(raw_task))
    from localdesk.config import Experiment

    impossible = Experiment.load(tmp_path / "experiment.json")
    run_experiment(impossible, run_id="candidate")
    with pytest.raises(ValueError, match="quality"):
        registry.promote("candidate")
    assert registry.load_active()[1]["run_id"] == "good"


def test_artifact_corruption_fails_before_serving(tmp_path):
    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="first")
    model = experiment.registry / "first" / "model.json.gz"
    model.write_bytes(model.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        Registry(experiment.registry).load("first")


def test_model_dataset_mismatch_is_rejected_even_with_a_valid_payload_checksum(tmp_path):
    from localdesk.data import file_hash

    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="first")
    directory = experiment.registry / "first"
    model = directory / "model.json.gz"
    bundle = json.loads(gzip.decompress(model.read_bytes()))
    bundle["dataset_sha256"] = {key: "0" * 64 for key in bundle["dataset_sha256"]}
    model.write_bytes(gzip.compress(json.dumps(bundle).encode()))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["model.json.gz"] = file_hash(model)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dataset"):
        Registry(experiment.registry).load("first")


def test_run_ids_are_immutable_and_safe(tmp_path):
    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="first")
    with pytest.raises(ValueError, match="exists"):
        run_experiment(experiment, run_id="first")
    with pytest.raises(ValueError):
        Registry(experiment.registry).load("../first")


def test_ungated_runs_are_candidates(tmp_path):
    experiment = write_config(tmp_path)
    run_experiment(experiment, run_id="ungated")
    with pytest.raises(ValueError, match="quality"):
        Registry(experiment.registry).promote("ungated")


def test_failed_training_retains_failure_record(tmp_path, monkeypatch):
    import localdesk.pipeline

    experiment = write_config(tmp_path)
    monkeypatch.setattr(
        localdesk.pipeline, "fit_baseline", lambda *_: (_ for _ in ()).throw(RuntimeError("failed"))
    )
    with pytest.raises(RuntimeError):
        run_experiment(experiment, run_id="failed")
    status = json.loads((experiment.registry / ".pending-failed" / "status.json").read_text())
    assert status["state"] == "failed"
    assert not (experiment.registry / "failed").exists()


def test_partial_qwen_checkpoint_cannot_replace_active_model(tmp_path):
    from localdesk.data import file_hash

    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="good")
    registry = Registry(experiment.registry)
    registry.promote("good")
    run_experiment(experiment, run_id="partial")
    directory = experiment.registry / "partial"
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir()
    metadata = {
        "task": experiment.task.to_dict(),
        "task_sha256": experiment.task.digest,
        "base_model": "test/base",
        "dataset_sha256": json.loads((directory / "dataset.json").read_text())["split_sha256"],
        "upstream": {"unsloth": "test"},
        "calibration": {"ece": 0.1},
    }
    (checkpoint / "localdesk_manifest.json").write_text(json.dumps(metadata))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["backend"] = "unsloth"
    for name in ("report.json", "training-plan.json"):
        value = json.loads((directory / name).read_text())
        value["backend"] = "unsloth"
        (directory / name).write_text(json.dumps(value))
        manifest["files_sha256"][name] = file_hash(directory / name)
    manifest["files_sha256"]["checkpoint/localdesk_manifest.json"] = file_hash(
        checkpoint / "localdesk_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint|model files"):
        registry.promote("partial")
    assert registry.load_active()[1]["run_id"] == "good"


def test_smoke_run_never_qualifies_even_if_metrics_pass():
    from localdesk.registry import assess_quality

    report = {
        "classification": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0},
        "accepted_accuracy": 1.0,
        "coverage": 1.0,
        "training": {"smoke_run": True},
    }
    assert not assess_quality(report, {"min_accuracy": 0.9})["qualified"]


def test_hashing_large_checkpoint_does_not_read_whole_payload(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    from localdesk.data import file_hash

    payload = b"model-checkpoint" * 200000
    path = tmp_path / "payload.safetensors"
    path.write_bytes(payload)

    def forbidden_read(self):
        raise AssertionError("whole-file allocation is forbidden for model hashes")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    assert file_hash(path) == hashlib.sha256(payload).hexdigest()


def test_qwen_serving_configuration_is_required_before_activation(tmp_path):
    from localdesk.data import file_hash

    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="partial")
    directory = experiment.registry / "partial"
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir()
    dataset = json.loads((directory / "dataset.json").read_text())
    metadata = {
        "task": experiment.task.to_dict(),
        "task_sha256": experiment.task.digest,
        "base_model": "test/base",
        "upstream": {"unsloth": "test"},
        "dataset_sha256": dataset["split_sha256"],
        "calibration": {"ece": 0.1},
    }
    (checkpoint / "localdesk_manifest.json").write_text(json.dumps(metadata))
    for name in (
        "joint_head.safetensors",
        "joint_head_config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (checkpoint / name).write_bytes(b"fixture-only-not-real-model")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["backend"] = "unsloth"
    for name in ("report.json", "training-plan.json"):
        raw = json.loads((directory / name).read_text())
        raw["backend"] = "unsloth"
        (directory / name).write_text(json.dumps(raw))
        manifest["files_sha256"][name] = file_hash(directory / name)
    manifest["files_sha256"].update(
        {f"checkpoint/{path.name}": file_hash(path) for path in checkpoint.iterdir()}
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint"):
        Registry(experiment.registry).promote("partial")
