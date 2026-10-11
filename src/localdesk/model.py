"""Explicit local Unsloth checkpoint backend; errors never trigger a baseline fallback."""

import json
import os
from pathlib import Path

from localdesk.core import validate_distribution, validate_text
from localdesk.schema import CATEGORIES, QUESTIONS, schema_hash


def validate_manifest(manifest: dict) -> None:
    required = {"schema_sha256", "base_model", "dataset_sha256", "upstream", "calibration"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError("checkpoint requires a complete LocalDesk manifest")
    if manifest["schema_sha256"] != schema_hash():
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
        )
        self.api.for_inference(self.model)

    def predict(self, text: str) -> dict[str, dict[str, float]]:
        text = validate_text(text)
        raw = self.api.predict(self.model, self.tokenizer, text, QUESTIONS)
        if not isinstance(raw, dict) or set(raw) != set(QUESTIONS):
            raise ValueError("Unsloth returned an invalid question set")
        if not isinstance(raw["category"], dict) or "probabilities" not in raw["category"]:
            raise ValueError("Unsloth did not return category probabilities")
        return {
            "category": validate_distribution(raw["category"]["probabilities"], list(CATEGORIES))
        }
