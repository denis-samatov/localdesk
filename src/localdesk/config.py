"""Strict, versioned task and experiment contracts; no dataset-specific labels in the core."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def fields(raw: dict, allowed: set[str], required: set[str] = frozenset()) -> None:
    if not isinstance(raw, dict):
        raise ValueError("expected a configuration object")
    if set(raw) - allowed:
        raise ValueError(f"unknown fields: {sorted(set(raw) - allowed)}")
    if required - set(raw):
        raise ValueError(f"missing fields: {sorted(required - set(raw))}")


def number(value, name, low=0, high=1) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return float(value)


@dataclass(frozen=True)
class TaskSpec:
    id: str
    version: str
    instructions: str
    options: tuple[tuple[str, str], ...]
    min_confidence: float = 0.75
    min_margin: float = 0.2

    @classmethod
    def from_dict(cls, raw: dict):
        fields(
            raw,
            {"id", "version", "instructions", "labels", "min_confidence", "min_margin"},
            {"id", "version", "instructions", "labels"},
        )
        if not isinstance(raw["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", raw["id"]):
            raise ValueError("task id must be a safe lowercase identifier")
        if not isinstance(raw["version"], str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,32}", raw["version"]
        ):
            raise ValueError("task version must be a non-empty identifier")
        if (
            not isinstance(raw["instructions"], str)
            or not 1 <= len(raw["instructions"].strip()) <= 2000
        ):
            raise ValueError("instructions must contain 1..2000 characters")
        labels = raw["labels"]
        if not isinstance(labels, dict) or not 2 <= len(labels) <= 200:
            raise ValueError("task requires 2..200 label descriptions")
        for key, description in labels.items():
            if (
                not isinstance(key, str)
                or not 1 <= len(key) <= 80
                or key != key.strip()
                or any(ord(char) < 32 for char in key)
            ):
                raise ValueError("invalid label key")
            if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1000:
                raise ValueError("label descriptions must contain 1..1000 characters")
        return cls(
            raw["id"],
            raw["version"],
            raw["instructions"],
            tuple(labels.items()),
            number(raw.get("min_confidence", 0.75), "min_confidence"),
            number(raw.get("min_margin", 0.2), "min_margin"),
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.options)

    @property
    def questions(self) -> dict:
        return {
            "category": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": dict(self.options),
            }
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "instructions": self.instructions,
            "labels": dict(self.options),
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
        }

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class Experiment:
    task: TaskSpec
    dataset: dict
    training: dict
    gates: dict
    registry: Path
    base: Path
    seed: int
    calibration_fraction: float
    test_fraction: float

    @classmethod
    def load(cls, path: Path):
        raw = read_json(path)
        fields(
            raw,
            {
                "task",
                "dataset",
                "training",
                "gates",
                "registry",
                "seed",
                "calibration_fraction",
                "test_fraction",
            },
            {"task", "dataset", "registry"},
        )
        base = path.resolve().parent
        if not isinstance(raw["task"], str) or not isinstance(raw["registry"], str):
            raise ValueError("task and registry must be paths")
        task = TaskSpec.from_dict(read_json(base / raw["task"]))
        dataset = raw["dataset"]
        fields(
            dataset,
            {
                "path",
                "splits",
                "format",
                "text_column",
                "label_column",
                "id_column",
                "group_column",
                "sha256",
                "provenance",
            },
            {"format", "text_column", "label_column"},
        )
        if ("path" in dataset) == ("splits" in dataset):
            raise ValueError("dataset requires exactly one path or splits mapping")
        paths = {"all": dataset["path"]} if "path" in dataset else dataset["splits"]
        if not isinstance(paths, dict) or (
            set(paths) != {"all"} and set(paths) != {"train", "calibration", "test"}
        ):
            raise ValueError("splits must define train, calibration and test")
        if any(not isinstance(value, str) or not value for value in paths.values()):
            raise ValueError("data paths must be non-empty strings")
        if dataset["format"] not in {"csv", "jsonl"}:
            raise ValueError("dataset format must be csv or jsonl")
        for name in ("text_column", "label_column", "id_column", "group_column"):
            if name in dataset and (not isinstance(dataset[name], str) or not dataset[name]):
                raise ValueError(f"{name} must be a non-empty field path")
        if "sha256" in dataset:
            if not isinstance(dataset["sha256"], dict) or set(dataset["sha256"]) != set(paths):
                raise ValueError("sha256 must cover every dataset file")
            if any(
                not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
                for v in dataset["sha256"].values()
            ):
                raise ValueError("invalid SHA-256")
        training = raw.get("training", {})
        fields(
            training,
            {
                "backend",
                "C",
                "max_iter",
                "epochs",
                "max_steps",
                "rank",
                "batch_size",
                "gradient_accumulation",
                "learning_rate",
            },
        )
        if training.get("backend", "baseline") not in {"baseline", "unsloth"}:
            raise ValueError("unsupported training backend")
        used = (
            {"backend", "C", "max_iter"}
            if training.get("backend", "baseline") == "baseline"
            else {
                "backend",
                "epochs",
                "max_steps",
                "rank",
                "batch_size",
                "gradient_accumulation",
                "learning_rate",
            }
        )
        if set(training) - used:
            raise ValueError("training fields do not apply to the selected backend")
        for name, low, high in [("C", 0.0001, 1000), ("learning_rate", 1e-8, 1)]:
            if name in training:
                number(training[name], name, low, high)
        for name in (
            "max_iter",
            "epochs",
            "max_steps",
            "rank",
            "batch_size",
            "gradient_accumulation",
        ):
            if name in training and (
                type(training[name]) is not int or not 1 <= training[name] <= 10000
            ):
                raise ValueError(f"{name} must be a positive bounded integer")
        gates = raw.get("gates", {})
        fields(
            gates,
            {"min_accuracy", "min_macro_f1", "max_ece", "min_accepted_accuracy", "min_coverage"},
        )
        for key, value in gates.items():
            number(value, key)
        seed = raw.get("seed", 3407)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a non-negative 32-bit integer")
        cal = number(raw.get("calibration_fraction", 0.15), "calibration_fraction", 0.01, 0.45)
        test = number(raw.get("test_fraction", 0.2), "test_fraction", 0.01, 0.45)
        if cal + test >= 0.8:
            raise ValueError("training fraction must exceed 0.2")
        return cls(
            task,
            dataset,
            training,
            gates,
            (base / raw["registry"]).resolve(),
            base,
            seed,
            cal,
            test,
        )
