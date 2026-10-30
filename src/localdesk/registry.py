"""Immutable completed artifacts; atomic active pointer; hashes and gates checked on every load."""

import gzip
import json
import math
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from localdesk.baseline import BaselineBackend
from localdesk.config import TaskSpec, read_json
from localdesk.data import file_hash


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def assess_quality(report: dict, gates: dict) -> dict:
    names = {
        "min_accuracy": report["classification"]["accuracy"],
        "min_macro_f1": report["classification"]["macro_f1"],
        "max_ece": report["classification"]["ece"],
        "min_accepted_accuracy": report["accepted_accuracy"],
        "min_coverage": report["coverage"],
    }
    outcomes = {}
    for name, threshold in gates.items():
        if name not in names:
            raise ValueError("unknown quality gate")
        value = names[name]
        outcomes[name] = {
            "threshold": threshold,
            "actual": value,
            "passed": isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
            and (value <= threshold if name.startswith("max_") else value >= threshold),
        }
    return {
        "qualified": bool(outcomes)
        and all(g["passed"] for g in outcomes.values())
        and not report.get("training", {}).get("smoke_run", False),
        "gates": outcomes,
        "requirements": gates,
    }


class Registry:
    def __init__(self, root: Path):
        self.root = root.resolve()

    @staticmethod
    def validate_id(run_id: str) -> None:
        if not isinstance(run_id, str) or not re.fullmatch(
            "[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id
        ):
            raise ValueError("invalid run id")

    def reserve(self, run_id: str) -> Path:
        self.validate_id(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / run_id).exists():
            raise ValueError("immutable run already exists")
        pending = self.root / f".pending-{run_id}"
        try:
            pending.mkdir()
        except FileExistsError as error:
            raise ValueError(
                f"run {run_id} is in progress; if no training process is running, "
                f"remove the stale staging directory {pending}"
            ) from error
        atomic_json(pending / "status.json", {"run_id": run_id, "state": "running"})
        return pending

    def retire_failed(self, pending: Path, run_id: str) -> Path:
        """Keep the failure record but release the run id for a retry."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        failed = self.root / f".failed-{run_id}-{stamp}-{uuid.uuid4().hex[:8]}"
        pending.rename(failed)
        return failed

    def metadata(self, run_id: str) -> tuple[Path, dict, TaskSpec]:
        self.validate_id(run_id)
        directory = self.root / run_id
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("completed run not found")
        manifest = read_json(directory / "manifest.json")
        if manifest.get("complete") is not True or manifest.get("run_id") != run_id:
            raise ValueError("incomplete or mismatched artifact")
        for relative, expected in manifest["files_sha256"].items():
            path = directory / relative
            if path.resolve().is_relative_to(directory.resolve()) is False or path.is_symlink():
                raise ValueError("unsafe artifact path")
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError("artifact checksum mismatch")
        mandatory = {"task.json", "report.json", "dataset.json", "training-plan.json"}
        if not mandatory <= set(manifest["files_sha256"]):
            raise ValueError("artifact is missing required provenance files")
        task = TaskSpec.from_dict(read_json(directory / "task.json"))
        if task.digest != manifest["task_sha256"]:
            raise ValueError("artifact task mismatch")
        report = read_json(directory / "report.json")
        dataset = read_json(directory / "dataset.json")
        if report["task_sha256"] != task.digest or dataset["task_sha256"] != task.digest:
            raise ValueError("report/dataset task mismatch")
        if report["dataset_split_sha256"] != dataset["split_sha256"]:
            raise ValueError("report/dataset fingerprint mismatch")
        plan = read_json(directory / "training-plan.json")
        if report["backend"] != manifest["backend"] or plan["backend"] != manifest["backend"]:
            raise ValueError("backend metadata mismatch")
        if plan["task_sha256"] != task.digest or report["split_counts"] != dataset["counts"]:
            raise ValueError("plan/report dataset identity mismatch")
        quality = assess_quality(report, manifest["quality"]["requirements"])
        if quality != manifest["quality"]:
            raise ValueError("quality result does not match report")
        return directory, manifest, task

    def verify(self, run_id: str, *, require_qualified: bool = True):
        directory, manifest, task = self.metadata(run_id)
        if require_qualified and not manifest["quality"]["qualified"]:
            raise ValueError("artifact has not passed declared quality gates")
        if manifest["backend"] in {"baseline", "tfidf"}:
            if "model.json.gz" not in manifest["files_sha256"]:
                raise ValueError("artifact model is missing")
            bundle = json.loads(gzip.decompress((directory / "model.json.gz").read_bytes()))
            if manifest["backend"] == "baseline":
                payload = BaselineBackend(bundle=bundle)
            else:
                from localdesk.tfidf import TfidfBackend

                payload = TfidfBackend(bundle=bundle)
            model_task = payload.task
            training_data = payload.bundle["dataset_sha256"]
        elif manifest["backend"] == "unsloth":
            from localdesk.model import validate_manifest

            if "checkpoint/localdesk_manifest.json" not in manifest["files_sha256"]:
                raise ValueError("checkpoint provenance is missing")
            # Clef LoRA layout inspected at the pinned Unsloth revision. The
            # generic Qwen trainer saves an adapter, not a merged backbone.
            required = {
                "joint_head.safetensors",
                "joint_head_config.json",
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "unsloth_decision_config.json",
            }
            if any(
                "checkpoint/" + name not in manifest["files_sha256"]
                or (directory / "checkpoint" / name).stat().st_size == 0
                for name in required
            ):
                raise ValueError("checkpoint is missing required non-empty model files")
            payload = read_json(directory / "checkpoint/localdesk_manifest.json")
            validate_manifest(payload)
            model_task = TaskSpec.from_dict(payload["task"])
            training_data = payload["dataset_sha256"]
        else:
            raise ValueError("unsupported artifact backend")
        if not model_task or model_task.digest != task.digest:
            raise ValueError("model and artifact task differ")
        dataset = read_json(directory / "dataset.json")
        if training_data != dataset["split_sha256"]:
            raise ValueError("model and qualification dataset fingerprints differ")
        return directory, manifest, payload

    def load(self, run_id: str, *, require_qualified: bool = True):
        directory, manifest, payload = self.verify(run_id, require_qualified=require_qualified)
        if manifest["backend"] in {"baseline", "tfidf"}:
            backend = payload
        else:
            from localdesk.model import ModelBackend

            backend = ModelBackend(directory / "checkpoint")
        return backend, manifest

    def promote(self, run_id: str) -> None:
        directory, manifest, _ = self.verify(run_id)
        atomic_json(
            self.root / "active.json",
            {
                "run_id": run_id,
                "task_sha256": manifest["task_sha256"],
                "manifest_sha256": file_hash(directory / "manifest.json"),
            },
        )

    def load_active(self):
        pointer = read_json(self.root / "active.json")
        backend, manifest = self.load(pointer["run_id"])
        if pointer["task_sha256"] != manifest["task_sha256"] or pointer[
            "manifest_sha256"
        ] != file_hash(self.root / pointer["run_id"] / "manifest.json"):
            raise ValueError("active pointer identity mismatch")
        return backend, manifest

    def list_runs(self) -> list[dict]:
        if not self.root.exists():
            return []
        entries = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                # One damaged or foreign directory must not hide every other version.
                try:
                    _, manifest, task = self.metadata(path.name)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    entries.append({"run_id": path.name, "valid": False, "error": str(error)})
                    continue
                entries.append(
                    {
                        "run_id": path.name,
                        "valid": True,
                        "task_id": task.id,
                        "qualified": manifest["quality"]["qualified"],
                        "backend": manifest["backend"],
                    }
                )
        return entries
