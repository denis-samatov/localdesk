"""Harness measurement and isolation checks; not a real-data/service qualification."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "qualify_local_cpu", Path(__file__).parents[1] / "scripts" / "qualify_local_cpu.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_nearest_rank_latency_preserves_count_units_and_tail():
    summary = pilot.latency_summary([0.2, 0.001, 0.003, 0.004])
    assert summary == {
        "count": 4,
        "min_ms": 1,
        "p50_ms": 3,
        "p95_ms": 200,
        "p99_ms": 200,
        "max_ms": 200,
    }
    with pytest.raises(ValueError, match="empty"):
        pilot.latency_summary([])


def test_owned_child_does_not_inherit_provider_credentials_or_import_overrides(monkeypatch):
    monkeypatch.setenv("PROVIDER_API_KEY", "ambient-do-not-forward")
    monkeypatch.setenv("LOCALDESK_API_KEY", "ambient-do-not-reuse")
    monkeypatch.setenv("PYTHONPATH", "untrusted-import-path")
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.invalid")
    env = pilot.child_environment("ephemeral-owned-key")
    assert env["LOCALDESK_API_KEY"] == "ephemeral-owned-key"
    assert not {"PROVIDER_API_KEY", "PYTHONPATH", "HTTPS_PROXY"} & env.keys()
    assert "ambient-do-not-forward" not in env.values()


def test_prediction_identity_drift_is_rejected_even_when_decision_matches():
    decision = {"category": "a", "action": "route"}
    manifest = {"run_id": "original", "task_sha256": "task"}
    pilot.assert_prediction(
        {**decision, "model_version": "original", "task_sha256": "task"}, decision, manifest
    )
    with pytest.raises(AssertionError, match="identity drift"):
        pilot.assert_prediction(
            {**decision, "model_version": "other", "task_sha256": "task"}, decision, manifest
        )


def test_optimized_python_cannot_silently_disable_acceptance_assertions():
    result = subprocess.run(
        [sys.executable, "-O", str(Path(pilot.__file__)), "--help"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "optimization disables acceptance assertions" in result.stderr
