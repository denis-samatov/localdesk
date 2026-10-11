"""Fit on real tickets, calibrate separately, export portable weights and evaluate once."""

import argparse
import gzip
import importlib.metadata
import json
import math
import platform
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from localdesk.baseline import N_FEATURES, BaselineBackend, features, softmax
from localdesk.core import decide
from localdesk.data import file_hash, load_splits
from localdesk.metrics import evaluate_rows
from localdesk.schema import schema_hash


def matrix(rows):
    positions, columns, values = [], [], []
    for position, row in enumerate(rows):
        for index, value in features(row["state"]).items():
            positions.append(position)
            columns.append(index)
            values.append(value)
    return csr_matrix((values, (positions, columns)), shape=(len(rows), N_FEATURES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--out", type=Path, default=Path("src/localdesk/assets/baseline.json.gz"))
    parser.add_argument("--report", type=Path, default=Path("reports/baseline.json"))
    args = parser.parse_args()
    rows = load_splits(args.data)
    source_manifest = json.loads((args.data / "manifest.json").read_text())
    classifier = LogisticRegression(
        C=4.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=3407
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(matrix(rows["train"]), [row["gold"]["category"] for row in rows["train"]])
    calibration_logits = classifier.decision_function(matrix(rows["calibration"]))
    indices = {label: index for index, label in enumerate(classifier.classes_)}
    calibration_gold = [indices[row["gold"]["category"]] for row in rows["calibration"]]

    def loss(temperature):
        return -sum(
            math.log(max(softmax(list(logits), temperature)[target], 1e-15))
            for logits, target in zip(calibration_logits, calibration_gold, strict=True)
        ) / len(calibration_gold)

    # Predefined search on calibration only; test labels are never used for fitting or selection.
    temperature = min([0.5 + i * 0.05 for i in range(71)], key=loss)
    bundle = {
        "format": "localdesk-linear-v1",
        "schema_sha256": schema_hash(),
        "classes": classifier.classes_.tolist(),
        "n_features": N_FEATURES,
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "temperature": temperature,
        "dataset_sha256": source_manifest["sha256"],
        "parameters": classifier.get_params(),
        "feature_recipe": "SHA256 word and bigram counts, L2 normalized",
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        },
        "python": platform.python_version(),
        "calibration_nll": loss(temperature),
        "data_source": source_manifest["source"],
        "data_license": source_manifest["license"],
    }
    root = Path(__file__).resolve().parents[1]
    recipe_files = [
        "scripts/train_baseline.py",
        "src/localdesk/baseline.py",
        "src/localdesk/core.py",
        "src/localdesk/data.py",
        "src/localdesk/metrics.py",
        "src/localdesk/schema.py",
    ]
    bundle["recipe_sha256"] = {name: file_hash(root / name) for name in recipe_files}
    backend = BaselineBackend(bundle=bundle)
    # Verify exported stdlib scoring against the actual training runtime.
    portable_logits = np.array([backend.logits(row["state"]) for row in rows["calibration"]])
    if not np.allclose(portable_logits, calibration_logits, atol=1e-10, rtol=1e-10):
        raise RuntimeError("exported scoring does not match scikit-learn")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gzip.compress(json.dumps(bundle, sort_keys=True).encode(), mtime=0))
    report = evaluate_rows(rows["test"], backend)
    report.update(
        {
            "evidence": source_manifest["evidence"],
            "dataset_sha256": source_manifest["sha256"],
            "source": source_manifest["source"],
            "model_sha256": file_hash(args.out),
            "recipe_sha256": bundle["recipe_sha256"],
            "split_counts": source_manifest["counts"],
            "temperature": temperature,
            "majority_class_accuracy": max(report["classification"]["per_class_support"].values())
            / len(rows["test"]),
            "ood": {
                "count": len(rows["ood"]),
                "reviewed": sum(
                    decide(row["state"], backend)["action"] == "review" for row in rows["ood"]
                ),
                "limitation": (
                    "Two EOL templates only; this does not establish general OOD detection."
                ),
            },
            "limitations": [
                "Cleaned public corpus from one organization, not a field trial.",
                "Russian performance is unqualified; source has no per-row language labels.",
                "Fixed review thresholds are illustrative, not validated service requirements.",
                "This is a linear baseline, not Qwen or an Unsloth GPU run.",
            ],
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "test_count": report["count"],
                "accuracy": report["classification"]["accuracy"],
                "coverage": report["coverage"],
                "accepted_accuracy": report["accepted_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
