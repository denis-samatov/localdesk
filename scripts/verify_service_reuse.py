"""Exercise real exported versions through the ASGI API; reports contain no input text."""

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from localdesk.api import create_app
from localdesk.config import Experiment, read_json
from localdesk.dataset import prepare
from localdesk.registry import Registry, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/service-reuse.json"))
    args = parser.parse_args()
    if len(args.configs) != len(args.runs):
        parser.error("configs and runs must align")
    store = Registry(args.registry)
    pointer = store.root / "active.json"
    previous = read_json(pointer) if pointer.exists() else None
    results = []
    try:
        for config, run_id in zip(args.configs, args.runs, strict=True):
            experiment = Experiment.load(config)
            prepared = prepare(experiment)
            directory, manifest, task = store.metadata(run_id)
            report = read_json(directory / "report.json")
            if (
                task.digest != experiment.task.digest
                or report["dataset_split_sha256"] != prepared.manifest["split_sha256"]
            ):
                raise ValueError("experiment differs from measured artifact")
            store.promote(run_id)
            with TestClient(
                create_app(registry=store.root, api_key="verification-only-key")
            ) as client:
                client.headers["X-API-Key"] = "verification-only-key"
                response = client.post(
                    "/v1/predictions", json={"text": prepared.rows["test"][0]["state"]}
                )
                assert response.status_code == 200
                assert set(response.json()["probabilities"]["category"]) == set(task.labels)
                assert response.json()["model_version"] == run_id
                assert client.get("/health/ready").status_code == 200
                assert (
                    client.get("/v1/model/report").json()["classification"]
                    == report["classification"]
                )
            results.append(
                {
                    "run_id": run_id,
                    "task_id": task.id,
                    "label_count": len(task.labels),
                    "api_prediction_status": response.status_code,
                    "task_sha256": task.digest,
                    "dataset_split_sha256": prepared.manifest["split_sha256"],
                    "classification": report["classification"],
                    "quality": manifest["quality"],
                    "evidence": "real_exported_model_asgi_requests",
                    "limitation": "One API contract request per domain, not a load or field test.",
                }
            )
    finally:
        if previous is not None:
            atomic_json(pointer, previous)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "run_id": r["run_id"],
                    "labels": r["label_count"],
                    "status": r["api_prediction_status"],
                }
                for r in results
            ]
        )
    )


if __name__ == "__main__":
    main()
