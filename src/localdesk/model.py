"""Explicit local Unsloth checkpoint backend; errors never trigger a baseline fallback."""

import json
import math
import os
from pathlib import Path

from localdesk.config import TaskSpec
from localdesk.core import validate_distribution, validate_text
from localdesk.schema import CATEGORIES, QUESTIONS, schema_hash


def _unsloth_distribution(values, labels):
    """Pinned Decision API rounds each probability to four decimal places."""
    try:
        return validate_distribution(values, labels)
    except ValueError:
        # Normalize only the documented rounding envelope; keep the core contract strict.
        if not isinstance(values, dict) or set(values) != set(labels):
            raise
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            or abs(v - round(v, 4)) > 1e-12
            for v in values.values()
        ):
            raise
        total = sum(values.values())
        if total <= 0 or abs(total - 1) > len(labels) * 0.00005 + 1e-7:
            raise
        return validate_distribution({key: value / total for key, value in values.items()}, labels)


def validate_manifest(manifest: dict) -> None:
    required = {"base_model", "dataset_sha256", "upstream", "calibration"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError("checkpoint requires a complete LocalDesk manifest")
    if "task" in manifest:
        if TaskSpec.from_dict(manifest["task"]).digest != manifest.get("task_sha256"):
            raise ValueError("checkpoint task mismatch")
    elif manifest.get("schema_sha256") != schema_hash():
        raise ValueError("checkpoint schema mismatch")
    if not isinstance(manifest["calibration"], dict) or not manifest["calibration"]:
        raise ValueError("checkpoint must include calibration provenance")


class ModelBackend:
    name = "unsloth"
    probability_kind = "unsloth_softmax_calibrated_on_heldout"

    def __init__(self, directory: Path):
        directory = directory.resolve()
        path = directory / "localdesk_manifest.json"
        if not directory.is_dir() or not path.is_file():
            raise ValueError(
                "an existing local checkpoint with localdesk_manifest.json is required"
            )
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(self.manifest)
        self.task = TaskSpec.from_dict(self.manifest["task"]) if "task" in self.manifest else None
        # Offline inference needs the adapter AND its base model already in the local HF cache.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            from unsloth import FastDecisionModel
        except ImportError as error:
            raise RuntimeError("install the optional GPU stack; see docs/TRAINING.md") from error
        self.api = FastDecisionModel
        self.model, self.tokenizer = self.api.from_pretrained(
            model_name=str(directory),
            max_seq_length=2048,
            load_in_4bit=True,
            local_files_only=True,
            use_exact_model_name=True,
        )
        self.api.for_inference(self.model)

    @classmethod
    def from_loaded(cls, api, model, tokenizer, manifest):
        validate_manifest(manifest)
        instance = cls.__new__(cls)
        instance.api, instance.model, instance.tokenizer, instance.manifest = (
            api,
            model,
            tokenizer,
            manifest,
        )
        instance.task = TaskSpec.from_dict(manifest["task"]) if "task" in manifest else None
        api.for_inference(model)
        return instance

    def predict(self, text: str) -> dict[str, dict[str, float]]:
        text = validate_text(text)
        questions = self.task.questions if self.task else QUESTIONS
        labels = self.task.labels if self.task else tuple(CATEGORIES)
        raw = self.api.predict(self.model, self.tokenizer, text, questions)
        if not isinstance(raw, dict) or set(raw) != set(questions):
            raise ValueError("Unsloth returned an invalid question set")
        if not isinstance(raw["category"], dict) or "probabilities" not in raw["category"]:
            raise ValueError("Unsloth did not return category probabilities")
        return {"category": _unsloth_distribution(raw["category"]["probabilities"], list(labels))}
