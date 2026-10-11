"""Offline, group-isolated comparisons. Never promotes or replaces service models."""

import dataclasses
import hashlib
import importlib.metadata
import math
import os
import platform
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from localdesk.baseline import softmax
from localdesk.config import Experiment, TaskSpec, digest, read_json
from localdesk.core import validate_distribution
from localdesk.data import file_hash
from localdesk.dataset import Prepared, prepare
from localdesk.metrics import classification_metrics
from localdesk.model import validate_manifest
from localdesk.pipeline import fit_baseline
from localdesk.registry import atomic_json
from localdesk.training import BASE_MODEL, BASE_REVISION, UPSTREAM, resolve_qwen_training

C_GRID = (0.25, 1.0, 4.0, 16.0)
TEMPERATURE_GRID = tuple(0.5 + i * 0.05 for i in range(71))
CPU_METHODS = ("majority", "hashed_logistic", "tfidf_logistic", "tfidf_svm")
QWEN_METHODS = ("qwen_zero_shot", "qwen_decision")


def partition_calibration(rows: list[dict], task: TaskSpec, seed: int):
    """Whole groups, deterministic two-way split, every class supported on both sides."""
    import random

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    support = Counter(
        label for members in grouped.values() for label in {r["gold"]["category"] for r in members}
    )
    if any(support[label] < 2 for label in task.labels):
        raise ValueError("calibration needs >=2 independent groups per label")
    groups = sorted(grouped)
    for attempt in range(256):
        ordered = groups.copy()
        random.Random(seed + attempt).shuffle(ordered)
        left = set(ordered[: len(ordered) // 2])
        tuning = [r for r in rows if r["group"] in left]
        calibration = [r for r in rows if r["group"] not in left]
        if all(
            {r["gold"]["category"] for r in part} == set(task.labels)
            for part in (tuning, calibration)
        ):
            return tuning, calibration
    raise ValueError("cannot partition calibration with class support; supply more groups")


def class_logits(scores, classes: int):
    import numpy as np

    scores = np.asarray(scores)
    if classes == 2 and scores.ndim == 1:
        return np.column_stack([np.zeros(len(scores)), scores])
    if scores.ndim != 2 or scores.shape[1] != classes or not np.isfinite(scores).all():
        raise ValueError("invalid classifier logit shape or values")
    return scores


class TfidfBackend:
    probability_kind = "softmax_margins_then_temperature_on_calibration"

    def __init__(self, task, vectorizer, classifier, kind):
        self.task, self.vectorizer, self.classifier = task, vectorizer, classifier
        self.labels = classifier.classes_.tolist()
        self.name = "tfidf_" + kind

    def predict(self, text):
        scores = self.classifier.decision_function(self.vectorizer.transform([text]))
        logits = class_logits(scores, len(self.labels))[0]
        return {"category": dict(zip(self.labels, softmax(logits.tolist(), 1.0), strict=True))}


def fit_tfidf(experiment, train, tuning, *, kind):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC

    if kind not in {"logistic", "svm"}:
        raise ValueError("unknown TF-IDF classifier")
    vectorizer = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=40000)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=40000
                ),
            ),
        ]
    )
    x_train = vectorizer.fit_transform([r["state"] for r in train])
    x_tuning = vectorizer.transform([r["state"] for r in tuning])
    gold = [r["gold"]["category"] for r in tuning]
    candidates, fitted = [], []
    for c in C_GRID:
        classifier = (
            LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=experiment.seed,
            )
            if kind == "logistic"
            else LinearSVC(
                C=c,
                class_weight="balanced",
                dual="auto",
                max_iter=5000,
                random_state=experiment.seed,
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(x_train, [r["gold"]["category"] for r in train])
        logits = class_logits(classifier.decision_function(x_tuning), len(classifier.classes_))
        probabilities = [
            dict(zip(classifier.classes_.tolist(), softmax(z.tolist(), 1), strict=True))
            for z in logits
        ]
        metrics = classification_metrics(gold, probabilities, list(experiment.task.labels))
        candidates.append({"C": c, "tuning_macro_f1": metrics["macro_f1"]})
        fitted.append(classifier)
    selected = max(range(len(candidates)), key=lambda i: candidates[i]["tuning_macro_f1"])
    return TfidfBackend(experiment.task, vectorizer, fitted[selected], kind), {
        "selection_split": "tuning",
        "selected_C": candidates[selected]["C"],
        "tuning_macro_f1": candidates[selected]["tuning_macro_f1"],
        "candidates": candidates,
        "features": {
            name: len(transform.vocabulary_) for name, transform in vectorizer.transformer_list
        },
        "raw_score_kind": "logistic_probabilities"
        if kind == "logistic"
        else "softmax_of_svm_margins_not_calibrated_probabilities",
    }


class MajorityBackend:
    name = "majority"
    probability_kind = "training_class_frequencies"

    def __init__(self, task, rows):
        self.task = task
        support = Counter(r["gold"]["category"] for r in rows)
        self.probabilities = {label: support[label] / len(rows) for label in task.labels}

    def predict(self, text):
        return {"category": self.probabilities.copy()}


def rescale(probabilities, labels, temperature):
    # Probability floors preserve finite scoring for underflowed float32 model outputs.
    return [
        dict(
            zip(
                labels,
                softmax([math.log(max(p[label], 1e-15)) for label in labels], temperature),
                strict=True,
            )
        )
        for p in probabilities
    ]


def fit_temperature(gold, probabilities, labels):
    return min(
        TEMPERATURE_GRID,
        key=lambda t: -sum(
            math.log(max(p[target], 1e-15))
            for target, p in zip(gold, rescale(probabilities, labels, t), strict=True)
        ),
    )


def wilson_upper(errors: int, count: int):
    if not count:
        return None
    z = 1.959963984540054
    p, denominator = errors / count, 1 + z * z / count
    return (
        p + z * z / (2 * count) + z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    ) / denominator


def policy_metrics(gold, probabilities, labels, policy):
    probabilities = [validate_distribution(p, labels) for p in probabilities]
    threshold = policy["threshold"]
    accepted = [
        (target, p)
        for target, p in zip(gold, probabilities, strict=True)
        if threshold is not None and max(p.values()) >= threshold
    ]
    errors = sum(max(p, key=p.get) != target for target, p in accepted)
    return {
        "accepted_count": len(accepted),
        "coverage": len(accepted) / len(gold),
        "accepted_accuracy": 1 - errors / len(accepted) if accepted else None,
        "error_rate": errors / len(accepted) if accepted else None,
        "error_upper_95": wilson_upper(errors, len(accepted)),
    }


def fit_review_policy(gold, probabilities, labels, target_error):
    if not 0 < target_error < 1:
        raise ValueError("target error must be between zero and one")
    candidates = sorted({max(p.values()) for p in probabilities})
    selected = None
    for threshold in candidates:
        metrics = policy_metrics(gold, probabilities, labels, {"threshold": threshold})
        if metrics["error_upper_95"] <= target_error:
            selected = threshold
            break
    result = policy_metrics(gold, probabilities, labels, {"threshold": selected})
    return {
        "threshold": selected,
        "target_error": target_error,
        "selection_split": "calibration",
        "calibration_count": len(gold),
        "calibration_accepted": result["accepted_count"],
        "calibration_upper_error_95": result["error_upper_95"],
        "limitation": "selection bound is descriptive; validate target on untouched test",
    }


def _fixed_policy(gold, probabilities, task):
    accepted = []
    for target, p in zip(gold, probabilities, strict=True):
        ranked = sorted(p.values(), reverse=True)
        if ranked[0] >= task.min_confidence and ranked[0] - ranked[1] >= task.min_margin:
            accepted.append(target == max(p, key=p.get))
    return {
        "coverage": len(accepted) / len(gold),
        "accepted_count": len(accepted),
        "accepted_accuracy": sum(accepted) / len(accepted) if accepted else None,
        "min_confidence": task.min_confidence,
        "min_margin": task.min_margin,
    }


def _predict_rows(backend, rows, labels, *, timed=False):
    probabilities, latency = [], []
    synchronize = getattr(backend, "synchronize", lambda: None)
    if timed:
        backend.predict(rows[0]["state"])
        synchronize()
    for row in rows:
        synchronize()
        start = time.perf_counter()
        values = validate_distribution(backend.predict(row["state"])["category"], labels)
        synchronize()
        if timed:
            latency.append((time.perf_counter() - start) * 1000)
        probabilities.append(values)
    return probabilities, latency


def score_backend(backend, calibration, test, task, *, target_error):
    import numpy as np

    labels = list(task.labels)
    cal_gold, gold = ([r["gold"]["category"] for r in rows] for rows in (calibration, test))
    raw_cal, _ = _predict_rows(backend, calibration, labels)
    temperature = fit_temperature(cal_gold, raw_cal, labels)
    calibrated_cal = rescale(raw_cal, labels, temperature)
    policy = fit_review_policy(cal_gold, calibrated_cal, labels, target_error)
    raw_test, latency = _predict_rows(backend, test, labels, timed=True)
    calibrated_test = rescale(raw_test, labels, temperature)
    metrics = policy_metrics(gold, calibrated_test, labels, policy)
    independent_groups = len({row["group"] for row in test})
    records = [
        {
            "id": hashlib.sha256(r["id"].encode()).hexdigest(),
            "group": r["group"],
            "gold": target,
            "prediction": max(p, key=p.get),
            "probabilities": p,
        }
        for r, target, p in zip(test, gold, calibrated_test, strict=True)
    ]
    result = {
        "status": "complete",
        "probability_kind": backend.probability_kind,
        "classification": classification_metrics(gold, calibrated_test, labels),
        "raw_classification": classification_metrics(gold, raw_test, labels),
        "temperature": temperature,
        "target_policy": policy,
        "target_policy_test": metrics,
        "test_independent_groups": independent_groups,
        "uncertainty_assumption": (
            "Wilson is a descriptive IID-row bound; multiple rows per group invalidate this bound"
        ),
        "target_validated_on_test": independent_groups == len(test)
        and metrics["error_upper_95"] is not None
        and metrics["error_upper_95"] <= target_error,
        "fixed_task_policy": _fixed_policy(gold, calibrated_test, task),
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "mean": float(np.mean(latency)),
            "requests": len(latency),
            "scope": "warmed sequential batch=1 local backend+validation; no HTTP",
        },
    }
    return result, records


def paired_comparison(reference, candidate, labels, *, seed=3407, samples=2000):
    import numpy as np

    if (
        not reference
        or len(reference) != len(candidate)
        or any(
            tuple(a[k] for k in ("id", "group", "gold"))
            != tuple(b[k] for k in ("id", "group", "gold"))
            for a, b in zip(reference, candidate, strict=True)
        )
    ):
        raise ValueError("paired predictions must be aligned by id, group and gold")
    if type(samples) is not int or samples < 20:
        raise ValueError("bootstrap requires >=20 repetitions")
    label_id = {label: i for i, label in enumerate(labels)}
    grouped = defaultdict(list)
    for i, r in enumerate(reference):
        grouped[r["group"]].append(i)
    groups = [np.array(indices) for _, indices in sorted(grouped.items())]
    gold = np.array([label_id[r["gold"]] for r in reference])
    predictions = [
        np.array([label_id[r["prediction"]] for r in rows]) for rows in (reference, candidate)
    ]

    def measurements(indices, predicted):
        matrix = np.bincount(
            gold[indices] * len(labels) + predicted[indices], minlength=len(labels) ** 2
        ).reshape(len(labels), len(labels))
        tp = np.diag(matrix)
        denom = matrix.sum(axis=0) + matrix.sum(axis=1)
        f1 = np.divide(2 * tp, denom, out=np.zeros(len(labels)), where=denom != 0)
        return np.array([tp.sum() / len(indices), f1.mean()])

    observed = measurements(np.arange(len(gold)), predictions[1]) - measurements(
        np.arange(len(gold)), predictions[0]
    )
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        deltas.append(measurements(indices, predictions[1]) - measurements(indices, predictions[0]))
    intervals = np.percentile(deltas, [2.5, 97.5], axis=0)
    return {
        "direction": "candidate minus reference",
        "accuracy_delta": float(observed[0]),
        "macro_f1_delta": float(observed[1]),
        "accuracy_delta_ci95": intervals[:, 0].tolist(),
        "macro_f1_delta_ci95": intervals[:, 1].tolist(),
        "test_groups": len(groups),
        "repetitions": samples,
        "seed": seed,
        "resampling": "paired whole test groups; fixed classes; unsupported class F1=0",
        "quality_gain_demonstrated": bool((intervals[0] > 0).all()),
    }


def _source_fingerprint():
    return digest(
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        }
    )


def _peak_rss():
    try:
        import resource

        amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(amount if platform.system() == "Darwin" else amount * 1024)
    except ImportError:
        return None


def checkpoint_provenance(experiment, prepared, checkpoint):
    manifest = read_json(checkpoint / "localdesk_manifest.json")
    validate_manifest(manifest)
    origin = read_json(checkpoint.parent / "protocol.json")
    parameters = resolve_qwen_training(experiment.training)
    epoch = manifest.get("training", {}).get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, (int, float))
        or not math.isfinite(epoch)
        or epoch + 1e-6 < parameters["epochs"]
    ):
        raise ValueError("resume requires evidence of full training epochs")
    if manifest.get("smoke_run") is not False or manifest["calibration"].get("method") != (
        "identity_for_benchmark"
    ):
        raise ValueError(
            "resume requires a full benchmark checkpoint; smoke/calibrated heads refused"
        )
    if type(manifest["calibration"].get("temperature")) not in (int, float) or (
        manifest["calibration"]["temperature"] != 1.0
    ):
        raise ValueError("resume requires identity temperature")
    serving = read_json(checkpoint / "unsloth_decision_config.json")
    temperatures = serving.get("temperature", [1, 1, 1])
    if (
        not isinstance(temperatures, list)
        or len(temperatures) != 3
        or any(type(v) not in (int, float) or v != 1 for v in temperatures)
        or any(
            type(serving.get(k, 1)) not in (int, float) or serving.get(k, 1) != 1
            for k in ("head_temperature", "folded_temperature")
        )
        or serving.get("temperature_by_options")
    ):
        raise ValueError("resume requires identity serving temperatures without folded calibration")
    if (
        manifest.get("task_sha256") != experiment.task.digest
        or manifest.get("dataset_sha256") != prepared.manifest["split_sha256"]
        or manifest.get("base_model") != BASE_MODEL
        or manifest.get("base_revision") != BASE_REVISION
        or manifest.get("upstream") != UPSTREAM
        or origin.get("task_sha256") != experiment.task.digest
        or origin.get("split_sha256") != prepared.manifest["split_sha256"]
        or origin.get("seed") != experiment.seed
        or origin.get("qwen", {}).get("training") != resolve_qwen_training(experiment.training)
    ):
        raise ValueError("checkpoint task/data/training identity mismatch")
    paths = sorted(checkpoint.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("checkpoint symlinks are forbidden")
    return {
        "training_source_sha256": origin["source_code_sha256"],
        "origin_protocol_sha256": digest(origin),
        "files_sha256": {
            p.relative_to(checkpoint).as_posix(): file_hash(p) for p in paths if p.is_file()
        },
    }


def run_benchmark(
    experiment: Experiment,
    output: Path,
    *,
    qwen=False,
    target_error=0.05,
    bootstrap_samples=2000,
    checkpoint: Path | None = None,
):
    if output.exists():
        raise ValueError("benchmark output exists; choose a fresh directory")
    if not 0 < target_error < 1 or type(bootstrap_samples) is not int or bootstrap_samples < 20:
        raise ValueError("invalid error target or bootstrap count")
    prepared = prepare(experiment)
    if checkpoint is not None and not qwen:
        raise ValueError("checkpoint resumption requires --qwen")
    resumed = checkpoint_provenance(experiment, prepared, checkpoint) if checkpoint else None
    tuning, calibration = partition_calibration(
        prepared.rows["calibration"], experiment.task, experiment.seed
    )
    protocol = {
        "format": "localdesk-benchmark-v1",
        "task_sha256": experiment.task.digest,
        "split_sha256": prepared.manifest["split_sha256"],
        "source_sha256": prepared.manifest["source_sha256"],
        "source_code_sha256": _source_fingerprint(),
        "resumed_checkpoint": resumed,
        "qwen": {
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "upstream": UPSTREAM,
            "training": resolve_qwen_training(experiment.training),
            "zero_shot_context": 8192,
            "decision_training_context": 2048,
            "zero_shot_option_order": list(experiment.task.labels),
            "zero_shot_scoring": "conditional symbolic next-token likelihood; no thinking",
        },
        "split_counts": prepared.manifest["counts"],
        "tuning_count": len(tuning),
        "calibration_count": len(calibration),
        "tuning_sha256": digest({"rows": tuning}),
        "probability_calibration_sha256": digest({"rows": calibration}),
        "seed": experiment.seed,
        "C_grid": list(C_GRID),
        "temperature_grid": list(TEMPERATURE_GRID),
        "target_error": target_error,
        "primary_comparison": "qwen_decision minus best tuning-selected TF-IDF method",
        "quality_gain_rule": "both paired accuracy and macro-F1 CI95 lower bounds >0",
        "bootstrap": {
            "seed": experiment.seed,
            "samples": bootstrap_samples,
            "unit": "whole test groups",
            "class_absence": "F1=0",
        },
        "evidence": prepared.manifest["provenance"],
        "test_reuse": "existing published holdout; not a new independent blind holdout",
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    report = {
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        },
        "methods": {
            name: {"status": "not_run", "reason": "CUDA stage not requested"}
            for name in QWEN_METHODS
        },
        "comparisons": {},
        "conclusion": {"status": "incomplete_qwen_comparison"},
    }
    predictions = {}

    def save_method(name, backend, info, started):
        result, records = score_backend(
            backend, calibration, prepared.rows["test"], experiment.task, target_error=target_error
        )
        result["fit"] = info
        result["stage_seconds"] = time.monotonic() - started
        report["methods"][name] = result
        predictions[name] = records
        atomic_json(
            output / (name + "-predictions.json"),
            {"records": records, "protocol_sha256": report["protocol_sha256"]},
        )
        atomic_json(output / "report.json", report)

    fitted_tfidf = {}
    for name, kind in (("tfidf_logistic", "logistic"), ("tfidf_svm", "svm")):
        start = time.monotonic()
        backend, info = fit_tfidf(experiment, prepared.rows["train"], tuning, kind=kind)
        info["fit_seconds"] = time.monotonic() - start
        fitted_tfidf[name] = (backend, info)
    reference = max(fitted_tfidf, key=lambda name: fitted_tfidf[name][1]["tuning_macro_f1"])
    report["reference"] = reference
    # Freeze the actual reference before ANY test predictions are scored.
    atomic_json(output / "reference.json", {"name": reference, "selected_on": "tuning_macro_f1"})
    start = time.monotonic()
    save_method(
        "majority",
        MajorityBackend(experiment.task, prepared.rows["train"]),
        {"fit_split": "train"},
        start,
    )
    start = time.monotonic()
    cpu_experiment = dataclasses.replace(
        experiment, training={"backend": "baseline", "C": 4, "max_iter": 1000}
    )
    # Benchmark fits its own temperature with the same procedure as every other method.
    subset = Prepared({**prepared.rows, "calibration": calibration}, prepared.manifest)
    backend, _ = fit_baseline(cpu_experiment, subset)
    backend.bundle["temperature"] = 1.0
    save_method("hashed_logistic", backend, {"C": 4, "fit_split": "train"}, start)
    for name, (backend, info) in fitted_tfidf.items():
        save_method(name, backend, info, time.monotonic())
    if qwen:
        from localdesk.benchmark_qwen import run_qwen_stages

        run_qwen_stages(
            experiment, prepared, calibration, output, report, save_method, checkpoint=checkpoint
        )
    for name in ("hashed_logistic", "qwen_zero_shot", "qwen_decision"):
        if name in predictions:
            report["comparisons"][name + "_vs_" + reference] = paired_comparison(
                predictions[reference],
                predictions[name],
                list(experiment.task.labels),
                seed=experiment.seed,
                samples=bootstrap_samples,
            )
    if all(report["methods"][name]["status"] == "complete" for name in QWEN_METHODS):
        primary = report["comparisons"]["qwen_decision_vs_" + reference]
        report["comparisons"]["qwen_decision_vs_qwen_zero_shot"] = paired_comparison(
            predictions["qwen_zero_shot"],
            predictions["qwen_decision"],
            list(experiment.task.labels),
            seed=experiment.seed,
            samples=bootstrap_samples,
        )
        report["conclusion"] = {
            "status": "quality_gain_on_this_holdout"
            if primary["quality_gain_demonstrated"]
            else "no_quality_gain_proven",
            "scope": "this reused public holdout only; no field or cost proof",
        }
    atomic_json(output / "report.json", report)
    report["memory"] = {
        "process_peak_rss_bytes": _peak_rss(),
        "scope": "cumulative whole benchmark process; not per-model resident size",
        "gpu_scope": "per-stage peak VRAM includes loading/training/inference",
    }
    atomic_json(output / "report.json", report)
    return report
