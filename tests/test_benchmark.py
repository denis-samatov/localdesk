import dataclasses

import pytest
from test_configurable import write_config

from localdesk.config import digest
from localdesk.dataset import prepare


def test_calibration_partition_is_group_isolated_and_supported(tmp_path):
    from localdesk.benchmark import partition_calibration

    experiment = write_config(tmp_path)
    prepared = prepare(experiment)
    # This small fixture needs more than one calibration group per label.
    experiment = dataclasses.replace(experiment, calibration_fraction=0.3)
    prepared = prepare(experiment)
    tuning, calibration = partition_calibration(prepared.rows["calibration"], experiment.task, 7)
    assert {r["group"] for r in tuning}.isdisjoint(r["group"] for r in calibration)
    assert {r["id"] for r in tuning + calibration} == {
        r["id"] for r in prepared.rows["calibration"]
    }
    for rows in (tuning, calibration):
        assert {r["gold"]["category"] for r in rows} == set(experiment.task.labels)
    assert (tuning, calibration) == partition_calibration(
        prepared.rows["calibration"], experiment.task, 7
    )


def test_partition_fails_with_insufficient_independent_groups(tmp_path):
    from localdesk.benchmark import partition_calibration

    experiment = write_config(tmp_path)
    prepared = prepare(experiment)
    rows = [
        next(r for r in prepared.rows["calibration"] if r["gold"]["category"] == label)
        for label in experiment.task.labels
    ]
    with pytest.raises(ValueError, match="independent|support"):
        partition_calibration(rows, experiment.task, 7)


def test_tfidf_vocabulary_is_train_only_and_new_binary_labels_work(tmp_path):
    from localdesk.benchmark import fit_tfidf

    experiment = dataclasses.replace(
        write_config(tmp_path, ("sales", "returns")), calibration_fraction=0.3
    )
    prepared = prepare(experiment)
    for row in prepared.rows["calibration"] + prepared.rows["test"]:
        row["state"] += " heldoutuniquetoken"
    backend, info = fit_tfidf(
        experiment, prepared.rows["train"], prepared.rows["calibration"], kind="logistic"
    )
    word = backend.vectorizer.transformer_list[0][1]
    assert "heldoutuniquetoken" not in word.vocabulary_
    values = backend.predict("sales request")["category"]
    assert set(values) == {"sales", "returns"}
    assert sum(values.values()) == pytest.approx(1.0)
    assert info["selection_split"] == "tuning"
    assert len(info["candidates"]) == 4


def test_binary_scores_preserve_probability_mapping():
    import numpy as np

    from localdesk.baseline import softmax
    from localdesk.benchmark import class_logits

    logits = class_logits(np.array([-2.0, 2.0]), 2)
    assert logits.tolist() == [[0.0, -2.0], [0.0, 2.0]]
    assert softmax(logits[0].tolist(), 1)[1] < 0.5
    assert softmax(logits[1].tolist(), 1)[1] > 0.5


def test_risk_policy_never_splits_equal_confidence_and_can_abstain_all():
    from localdesk.benchmark import fit_review_policy, policy_metrics

    predictions = [{"a": 0.9, "b": 0.1}] * 10
    # Wilson's upper bound requires evidence, not just zero empirical errors.
    policy = fit_review_policy(["a"] * 10, predictions, ["a", "b"], 0.05)
    assert policy["threshold"] is None
    assert policy_metrics(["a"] * 10, predictions, ["a", "b"], policy)["coverage"] == 0
    policy = fit_review_policy(["a"] * 100, predictions * 10, ["a", "b"], 0.05)
    assert policy["threshold"] == 0.9
    assert policy["calibration_upper_error_95"] < 0.05
    assert policy_metrics(["a"] * 100, predictions * 10, ["a", "b"], policy)["coverage"] == 1


def test_paired_cluster_comparison_rejects_mismatched_ids_and_labels():
    from localdesk.benchmark import paired_comparison

    left = [{"id": "a", "group": "g", "gold": "x", "prediction": "x"}]
    right = [{**left[0], "id": "different"}]
    with pytest.raises(ValueError, match="aligned"):
        paired_comparison(left, right, ["x", "y"], seed=7, samples=20)
    right = [{**left[0], "gold": "y"}]
    with pytest.raises(ValueError, match="aligned"):
        paired_comparison(left, right, ["x", "y"], seed=7, samples=20)


def test_bootstrap_uses_whole_groups_and_identical_pairing():
    from localdesk.benchmark import paired_comparison

    rows = [
        {
            "id": str(i),
            "group": str(i // 2),
            "gold": "x" if i % 2 else "y",
            "prediction": "x" if i % 2 else "y",
        }
        for i in range(20)
    ]
    result = paired_comparison(rows, rows, ["x", "y"], seed=7, samples=100)
    assert result["test_groups"] == 10
    assert result["accuracy_delta_ci95"] == [0.0, 0.0]
    assert result["macro_f1_delta_ci95"] == [0.0, 0.0]
    assert not result["quality_gain_demonstrated"]


def test_cpu_run_has_explicit_missing_qwen_and_frozen_protocol(tmp_path):
    from localdesk.benchmark import run_benchmark

    experiment = dataclasses.replace(write_config(tmp_path), calibration_fraction=0.3)
    out = tmp_path / "benchmark"
    report = run_benchmark(experiment, out, bootstrap_samples=20)
    assert report["conclusion"]["status"] == "incomplete_qwen_comparison"
    assert report["methods"]["qwen_zero_shot"]["status"] == "not_run"
    assert report["methods"]["qwen_decision"]["status"] == "not_run"
    assert report["protocol_sha256"] == digest(report["protocol"])
    assert set(report["protocol"]["split_sha256"]) == {"train", "calibration", "test"}
    assert set(report["methods"]) == {
        "majority",
        "hashed_logistic",
        "tfidf_logistic",
        "tfidf_svm",
        "qwen_zero_shot",
        "qwen_decision",
    }
    assert all(
        report["methods"][name]["status"] == "complete"
        for name in ("majority", "hashed_logistic", "tfidf_logistic", "tfidf_svm")
    )
    assert (out / "protocol.json").is_file()
    assert (out / "report.json").is_file()
    with pytest.raises(ValueError, match="exists"):
        run_benchmark(experiment, out, bootstrap_samples=20)


def test_correlated_test_rows_cannot_validate_iid_wilson_target():
    from test_configurable import spec

    from localdesk.benchmark import MajorityBackend, score_backend

    task = spec(("a", "b"))
    rows = [
        {
            "id": str(i),
            "group": "one-conversation",
            "state": f"message{i}",
            "gold": {"category": "a"},
        }
        for i in range(100)
    ]
    backend = MajorityBackend(task, rows)
    result, _ = score_backend(backend, rows, rows, task, target_error=0.05)
    assert result["target_policy_test"]["error_upper_95"] < 0.05
    assert result["target_validated_on_test"] is False
    assert result["test_independent_groups"] == 1


def test_gpu_parameters_are_frozen_in_protocol_before_scoring(tmp_path):
    from localdesk.benchmark import run_benchmark

    experiment = dataclasses.replace(write_config(tmp_path), calibration_fraction=0.3)
    changed = dataclasses.replace(experiment, training={"backend": "unsloth", "rank": 32})
    first = run_benchmark(experiment, tmp_path / "one", bootstrap_samples=20)
    second = run_benchmark(changed, tmp_path / "two", bootstrap_samples=20)
    assert first["protocol"]["qwen"]["training"]["rank"] == 16
    assert second["protocol"]["qwen"]["training"]["rank"] == 32
    assert first["protocol_sha256"] != second["protocol_sha256"]
