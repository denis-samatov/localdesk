"""One fit/calibrate/evaluate/publish lifecycle for every configured text classification task."""

import gzip
import importlib.metadata
import json
import math
import time
import uuid
import warnings
from datetime import UTC, datetime

from localdesk.baseline import N_FEATURES, BaselineBackend, features, softmax
from localdesk.config import Experiment
from localdesk.data import file_hash
from localdesk.dataset import Prepared, prepare
from localdesk.metrics import evaluate_rows
from localdesk.registry import Registry, assess_quality, atomic_json


def fit_baseline(experiment: Experiment, prepared: Prepared):
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    def matrix(rows):
        indices, columns, values = [], [], []
        for index, row in enumerate(rows):
            for column, value in features(row["state"]).items():
                indices.append(index)
                columns.append(column)
                values.append(value)
        return csr_matrix((values, (indices, columns)), shape=(len(rows), N_FEATURES))

    classifier = LogisticRegression(
        C=experiment.training.get("C", 4.0),
        class_weight="balanced",
        max_iter=experiment.training.get("max_iter", 1000),
        solver="lbfgs",
        random_state=experiment.seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(
            matrix(prepared.rows["train"]), [r["gold"]["category"] for r in prepared.rows["train"]]
        )
    x_cal = matrix(prepared.rows["calibration"])
    logits = classifier.decision_function(x_cal)
    coef, intercept = classifier.coef_, classifier.intercept_
    if len(classifier.classes_) == 2:
        # sklearn stores one binary logit. [0, z] has exactly the same sigmoid probabilities.
        logits = np.column_stack([np.zeros(len(logits)), logits])
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]])
    labels = classifier.classes_.tolist()
    label_ids = {label: index for index, label in enumerate(labels)}
    gold = [label_ids[r["gold"]["category"]] for r in prepared.rows["calibration"]]

    def loss(temperature):
        return -sum(
            math.log(max(softmax(list(z), temperature)[target], 1e-15))
            for z, target in zip(logits, gold, strict=True)
        ) / len(gold)

    temperature = min([0.5 + i * 0.05 for i in range(71)], key=loss)
    bundle = {
        "format": "localdesk-linear-v2",
        "task": experiment.task.to_dict(),
        "task_sha256": experiment.task.digest,
        "classes": labels,
        "n_features": N_FEATURES,
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
        "temperature": temperature,
        "dataset_sha256": prepared.manifest["split_sha256"],
        "parameters": classifier.get_params(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "scipy")
        },
        "calibration_nll": loss(temperature),
    }
    backend = BaselineBackend(bundle=bundle)
    actual = np.array([backend.logits(r["state"]) for r in prepared.rows["calibration"]])
    if not np.allclose(actual, logits, rtol=1e-10, atol=1e-10):
        raise RuntimeError("portable model export differs from fitted logits")
    uncalibrated = np.array([softmax(list(z), 1.0) for z in actual])
    if not np.allclose(uncalibrated, classifier.predict_proba(x_cal), rtol=1e-10, atol=1e-10):
        raise RuntimeError("class mapping/export probability mismatch")
    return backend, bundle


def run_experiment(experiment: Experiment, *, run_id: str | None = None) -> dict:
    prepared = prepare(experiment)
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    registry = Registry(experiment.registry)
    pending = registry.reserve(run_id)
    started = time.monotonic()
    try:
        kind = experiment.training.get("backend", "baseline")
        if kind == "baseline":
            backend, bundle = fit_baseline(experiment, prepared)
            (pending / "model.json.gz").write_bytes(
                gzip.compress(json.dumps(bundle, sort_keys=True, allow_nan=False).encode(), mtime=0)
            )
            training_info = {
                "temperature": bundle["temperature"],
                "versions": bundle["versions"],
                "calibration_nll": bundle["calibration_nll"],
            }
        else:
            from localdesk.training import fit_qwen

            backend, training_info = fit_qwen(experiment, prepared, pending / "checkpoint")
        report = evaluate_rows(prepared.rows["test"], backend)
        report.update(
            {
                "task_sha256": experiment.task.digest,
                "dataset_split_sha256": prepared.manifest["split_sha256"],
                "evidence": prepared.manifest["provenance"],
                "split_counts": prepared.manifest["counts"],
                "training": training_info,
                "runtime_seconds": time.monotonic() - started,
            }
        )
        quality = assess_quality(report, experiment.gates)
        plan = {
            "backend": kind,
            "seed": experiment.seed,
            "parameters": experiment.training,
            "task_sha256": experiment.task.digest,
            "quality_gates": experiment.gates,
            "stages": [
                "validate groups and labels",
                "fit on train",
                "fit temperature on calibration",
                "evaluate untouched test",
                "verify gates",
                "publish immutable version",
            ],
            "policy": {
                "min_confidence": experiment.task.min_confidence,
                "min_margin": experiment.task.min_margin,
            },
        }
        for name, value in [
            ("task.json", experiment.task.to_dict()),
            ("dataset.json", prepared.manifest),
            ("report.json", report),
            ("training-plan.json", plan),
        ]:
            atomic_json(pending / name, value)
        atomic_json(pending / "status.json", {"run_id": run_id, "state": "complete"})
        files = {
            path.relative_to(pending).as_posix(): file_hash(path)
            for path in pending.rglob("*")
            if path.is_file()
        }
        manifest = {
            "format": "localdesk-run-v1",
            "complete": True,
            "run_id": run_id,
            "task_sha256": experiment.task.digest,
            "backend": kind,
            "quality": quality,
            "files_sha256": files,
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(pending / "manifest.json", manifest)
        pending.rename(registry.root / run_id)
        registry.metadata(run_id)
        return {"run_id": run_id, "manifest": manifest, "report": report}
    except Exception as error:
        if pending.exists():
            # Store the type, not private exception strings or ticket content.
            atomic_json(
                pending / "status.json",
                {"run_id": run_id, "state": "failed", "error_type": type(error).__name__},
            )
        raise
