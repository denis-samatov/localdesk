"""Owned loopback CPU drill on prepared public data; never a production SLO claim."""

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from localdesk.config import Experiment
from localdesk.core import decide
from localdesk.data import file_hash
from localdesk.dataset import prepare
from localdesk.metrics import classification_metrics
from localdesk.pipeline import run_experiment
from localdesk.registry import Registry


def latency_summary(seconds):
    values = sorted(seconds)
    if not values:
        raise ValueError("latency measurements must not be empty")
    return {
        "count": len(values),
        "min_ms": 1000 * values[0],
        "p50_ms": 1000 * values[math.ceil(0.50 * len(values)) - 1],
        "p95_ms": 1000 * values[math.ceil(0.95 * len(values)) - 1],
        "p99_ms": 1000 * values[math.ceil(0.99 * len(values)) - 1],
        "max_ms": 1000 * values[-1],
    }


def tree_hashes(root):
    return {
        p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


def child_environment(token):
    # No ambient provider credentials/PYTHONPATH are passed to the owned service.
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL"}
    return {
        **{k: v for k, v in os.environ.items() if k in allowed},
        "PYTHONDONTWRITEBYTECODE": "1",
        "LOCALDESK_API_KEY": token,
    }


class OwnedService:
    def __init__(self, registry, directory, label, lifecycle):
        self.registry = registry
        self.directory = directory
        self.label = label
        self.lifecycle = lifecycle
        self.token = secrets.token_urlsafe(32)
        self.process = None

    def request(self, path, payload=None, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["X-API-Key"] = self.token
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
        )
        start = time.monotonic()
        try:
            response = self.client.open(request, timeout=60)
        except urllib.error.HTTPError as response:
            status = response.code
            body = response.read()
            return status, json.loads(body), time.monotonic() - start
        with response:
            return response.status, json.load(response), time.monotonic() - start

    def __enter__(self):
        started = time.monotonic()
        # A proxy must never receive this ephemeral local credential or public text.
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.log_path = self.directory / f"{self.label}-service.log"
        self.log = self.log_path.open("w")
        argv = [
            sys.executable,
            "-c",
            "from localdesk.cli import main; main()",
            "api",
            "--registry",
            str(self.registry),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        self.record = {
            "phase": self.label,
            "argv": argv,
            "host": "127.0.0.1",
            "port": self.port,
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            self.process = subprocess.Popen(
                argv, env=child_environment(self.token), stdout=self.log, stderr=subprocess.STDOUT
            )
            self.record["pid"] = self.process.pid
            self.lifecycle.append(self.record)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("owned service exited before liveness")
                try:
                    if self.request("/health/live", authenticated=False)[0] == 200:
                        if os.name == "posix":
                            command = subprocess.check_output(
                                ["ps", "-o", "command=", "-p", str(self.process.pid)], text=True
                            )
                            assert self.token not in command, "credential in process argv"
                        self.record["credential_in_argv"] = False
                        self.record["startup_liveness_seconds"] = time.monotonic() - started
                        self.record["rss_kib_after_start"] = self.rss_kib()
                        return self
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.05)
            raise TimeoutError("owned service startup watchdog expired")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def rss_kib(self):
        if os.name != "posix":
            return None
        return int(
            subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(self.process.pid)], text=True
            ).strip()
        )

    def __exit__(self, *_):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.record["forced_kill"] = True
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.record.update(
                exit_code=self.process.returncode, reaped=self.process.poll() is not None
            )
        self.log.close()
        # Check without retaining the token or request bodies in evidence.
        if self.token in self.log_path.read_text() or self.token in repr(self.record["argv"]):
            raise RuntimeError("ephemeral credential leaked into service evidence")
        self.record["credential_in_service_log"] = False


def assert_prediction(response, expected, manifest):
    assert response == {
        **expected,
        "model_version": manifest["run_id"],
        "task_sha256": manifest["task_sha256"],
    }, "prediction/identity drift"


def check_restored(registry, expected, row, directory, label, lifecycle):
    backend, manifest = Registry(registry).load_active()
    assert decide(row["state"], backend) == expected
    with OwnedService(registry, directory, label, lifecycle) as service:
        ready, _, ready_seconds = service.request("/health/ready")
        service.record.update(ready_status=ready, ready_latency_ms=1000 * ready_seconds)
        assert ready == 200
        status, result, seconds = service.request("/v1/predictions", {"text": row["state"]})
        service.record.update(prediction_status=status, prediction_latency_ms=1000 * seconds)
        assert status == 200
        assert_prediction(result, expected, manifest)
    return manifest["run_id"]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "scope": "owned local CPU HTTP/public-corpus regression, not field trial or SLO",
        "machine": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "protocol_sha256": file_hash(args.protocol),
        "lifecycle": [],
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("localdesk", "fastapi", "uvicorn", "numpy", "scipy", "scikit-learn")
        },
    }
    try:
        registry = args.out / "registry"
        source_hashes = tree_hashes(args.registry)
        shutil.copytree(args.registry, registry)
        backend, manifest = Registry(registry).load_active()
        raw = json.loads(args.experiment.read_text())
        raw["task"] = str((args.experiment.parent / raw["task"]).resolve())
        raw["registry"] = str(registry.resolve())
        raw["dataset"]["splits"] = {
            name: str((args.data / f"{name}.jsonl").resolve())
            for name in ("train", "calibration", "test")
        }
        config = args.out / "experiment.json"
        config.write_text(json.dumps(raw, indent=2) + "\n")
        experiment = Experiment.load(config)
        prepared = prepare(experiment)
        # Raw JSONL file hashes and canonical model split hashes are distinct identities.
        assert prepared.manifest["split_sha256"] == backend.bundle["dataset_sha256"]
        assert experiment.task.digest == manifest["task_sha256"]
        rows = prepared.rows["test"]
        source = json.loads((args.data / "manifest.json").read_text())
        assert file_hash(args.data / "test.jsonl") == source["sha256"]["test"]
        expected = [decide(row["state"], backend) for row in rows]
        result.update(
            dataset_manifest_sha256=file_hash(args.data / "manifest.json"),
            source_split_files_sha256=source["sha256"],
            dataset_split_sha256=prepared.manifest["split_sha256"],
            original_registry_sha256=source_hashes,
            original_model=manifest["run_id"],
            task_sha256=manifest["task_sha256"],
        )
        with OwnedService(registry, args.out, "public-http", result["lifecycle"]) as service:
            assert service.request("/health/ready")[0] == 200
            actual, elapsed = [], []
            for row, wanted in zip(rows, expected, strict=True):
                status, value, seconds = service.request("/v1/predictions", {"text": row["state"]})
                assert status == 200
                assert_prediction(value, wanted, manifest)
                actual.append(value)
                elapsed.append(seconds)
            accepted = [
                (row, value)
                for row, value in zip(rows, actual, strict=True)
                if value["action"] == "route"
            ]
            result["public_http"] = {
                "status_counts": {"200": len(actual)},
                "latency": latency_summary(elapsed),
                "classification": classification_metrics(
                    [r["gold"]["category"] for r in rows],
                    [v["probabilities"]["category"] for v in actual],
                    list(backend.task.labels),
                ),
                "accepted": len(accepted),
                "coverage": len(accepted) / len(rows),
                "accepted_accuracy": sum(
                    r["gold"]["category"] == v["category"] for r, v in accepted
                )
                / len(accepted)
                if accepted
                else None,
                "direct_artifact_matches": len(actual),
            }
            result["load"] = []
            for concurrency in (1, 4, 16):

                def predict(index):
                    i = index % len(rows)
                    status, value, seconds = service.request(
                        "/v1/predictions", {"text": rows[i]["state"]}
                    )
                    if status == 200:
                        assert_prediction(value, expected[i], manifest)
                    return status, seconds

                start = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    responses = list(pool.map(predict, range(256)))
                duration = time.monotonic() - start
                counts = dict(Counter(str(status) for status, _ in responses))
                result["load"].append(
                    {
                        "concurrency": concurrency,
                        "requests": len(responses),
                        "status_counts": counts,
                        "elapsed_seconds": duration,
                        "requests_per_second": len(responses) / duration,
                        "latency": latency_summary([s for _, s in responses]),
                        "rss_kib_after_phase": service.rss_kib(),
                    }
                )
                assert counts == {"200": 256}, "normal bounded load had errors"
            result["negative_controls"] = {
                "missing_key": service.request("/v1/model", authenticated=False)[0],
                "empty_text": service.request("/v1/predictions", {"text": ""})[0],
                "oversized_batch": service.request("/v1/predictions/batch", {"texts": ["a"] * 17})[
                    0
                ],
            }
            assert result["negative_controls"] == {
                "missing_key": 401,
                "empty_text": 422,
                "oversized_batch": 422,
            }
        result["restart_version"] = check_restored(
            registry, expected[0], rows[0], args.out, "restart", result["lifecycle"]
        )
        backup, restored = args.out / "backup", args.out / "restored"
        # No training/promotion/service writer is running during this whole-registry copy.
        shutil.copytree(registry, backup)
        Registry(backup).load_active()
        shutil.copytree(backup, restored)
        assert tree_hashes(backup) == tree_hashes(restored) == source_hashes
        result["backup_sha256"] = tree_hashes(backup)
        result["restored_version"] = check_restored(
            restored, expected[0], rows[0], args.out, "restore", result["lifecycle"]
        )
        corrupt = args.out / "corrupt"
        shutil.copytree(backup, corrupt)
        (corrupt / manifest["run_id"] / "model.json.gz").write_bytes(b"disposable corrupt model")
        with OwnedService(corrupt, args.out, "corrupt", result["lifecycle"]) as service:
            result["corruption"] = {
                "ready": service.request("/health/ready")[0],
                "prediction": service.request("/v1/predictions", {"text": "a"})[0],
            }
            assert result["corruption"] == {"ready": 503, "prediction": 503}
        result["training_config_sha256"] = file_hash(config)
        run_experiment(experiment, run_id="cpu-pilot-candidate")
        store = Registry(registry)
        store.promote("cpu-pilot-candidate")
        candidate_backend, _ = store.load_active()
        result["promoted_version"] = check_restored(
            registry,
            decide(rows[0]["state"], candidate_backend),
            rows[0],
            args.out,
            "promote",
            result["lifecycle"],
        )
        store.promote(manifest["run_id"])
        result["rollback_version"] = check_restored(
            registry, expected[0], rows[0], args.out, "rollback", result["lifecycle"]
        )
        assert tree_hashes(args.registry) == source_hashes, "source registry changed"
        result.update(
            status="passed",
            original_registry_unchanged=True,
            all_owned_processes_reaped=all(x["reaped"] for x in result["lifecycle"]),
        )
        assert not any(x.get("forced_kill") for x in result["lifecycle"])
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["finished_at_utc"] = datetime.now(UTC).isoformat()
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "public_rows": len(rows),
                "load_requests": sum(x["requests"] for x in result["load"]),
                "all_owned_processes_reaped": result["all_owned_processes_reaped"],
            }
        )
    )


def main():
    if not __debug__:
        raise RuntimeError("Python optimization disables acceptance assertions; run without -O")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "data", "experiment", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
