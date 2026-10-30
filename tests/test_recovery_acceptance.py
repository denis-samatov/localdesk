"""Synthetic, local registry restore and rollback acceptance; no production claim."""

import shutil

import pytest
from test_configurable import write_config

from localdesk.pipeline import run_experiment
from localdesk.registry import Registry


def test_restored_snapshot_preserves_active_version_and_predictions(tmp_path):
    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="stable")
    registry = Registry(experiment.registry)
    registry.promote("stable")
    expected = registry.load_active()[0].predict("red request topic0")
    backup = tmp_path / "backup"
    shutil.copytree(experiment.registry, backup)
    run_experiment(experiment, run_id="candidate")
    registry.promote("candidate")
    restored = Registry(backup)
    backend, manifest = restored.load_active()
    assert manifest["run_id"] == "stable"
    assert backend.predict("red request topic0") == expected
    registry.promote("stable")
    assert registry.load_active()[1]["run_id"] == "stable"


def test_restored_snapshot_rejects_corrupted_model(tmp_path):
    experiment = write_config(tmp_path, gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="stable")
    Registry(experiment.registry).promote("stable")
    restored = tmp_path / "restore"
    shutil.copytree(experiment.registry, restored)
    (restored / "stable" / "model.json.gz").write_bytes(b"corrupt backup")
    with pytest.raises(ValueError, match="checksum"):
        Registry(restored).load_active()


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -0.1, 1.1, True])
def test_quality_gates_reject_invalid_probability_metrics(value):
    from localdesk.registry import assess_quality

    report = {
        "classification": {"accuracy": value, "macro_f1": 0.9, "ece": 0.1},
        "accepted_accuracy": 0.9,
        "coverage": 0.8,
    }
    assert not assess_quality(report, {"min_accuracy": 0.0})["qualified"]
