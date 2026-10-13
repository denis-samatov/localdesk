"""Build the packaged demo model with the standard pipeline and publish its report.

Trains experiments/it-support.json (TF-IDF backend) into a temporary registry, then
copies the verified model file to src/localdesk/assets/demo.json.gz and the complete,
promoted registry version to docker/demo-registry/ (served by the Docker image).
The report binds the model to its data, task and the recipe files that determine
its output. Prepare the data first: python scripts/prepare_data.py
"""

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from localdesk.config import Experiment
from localdesk.data import file_hash
from localdesk.pipeline import run_experiment
from localdesk.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "it-support-demo"
RECIPE = [
    "scripts/train_demo.py",
    "src/localdesk/tfidf.py",
    "src/localdesk/core.py",
    "experiments/it-support.json",
    "tasks/it-support.json",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "src/localdesk/assets/demo.json.gz")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/demo.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "docker/demo-registry")
    args = parser.parse_args()
    experiment = Experiment.load(ROOT / "experiments/it-support.json")
    if experiment.training.get("backend") != "tfidf":
        raise ValueError("the demo experiment must use the tfidf backend")
    with tempfile.TemporaryDirectory() as directory:
        experiment = replace(experiment, registry=Path(directory))
        result = run_experiment(experiment, run_id=RUN_ID)
        # Load through the registry so every checksum, identity and gate check runs.
        registry = Registry(Path(directory))
        run, manifest, _ = registry.verify(RUN_ID)
        registry.promote(RUN_ID)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "model.json.gz", args.out)
        shutil.rmtree(args.registry, ignore_errors=True)
        args.registry.mkdir(parents=True)
        shutil.copytree(run, args.registry / RUN_ID)
        shutil.copyfile(Path(directory) / "active.json", args.registry / "active.json")
    Registry(args.registry).load_active()
    report = dict(
        result["report"],
        model_sha256=file_hash(args.out),
        task_sha256=experiment.task.digest,
        quality=manifest["quality"],
        recipe_sha256={name: file_hash(ROOT / name) for name in RECIPE},
        limitations=[
            "Public corpus from one organization; reused holdout, not a field trial.",
            "Fixed review thresholds are the task's illustrative policy.",
            "Russian and other-language performance is unqualified.",
        ],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    classification = report["classification"]
    print(
        json.dumps(
            {
                "accuracy": classification["accuracy"],
                "macro_f1": classification["macro_f1"],
                "coverage": report["coverage"],
                "model_bytes": args.out.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
