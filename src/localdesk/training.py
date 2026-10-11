"""GPU training path matched to inspected upstream APIs; not a claim of an executed GPU run."""

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from localdesk.data import file_hash, load_splits
from localdesk.schema import schema_hash

BASE_MODEL = "unsloth/Qwen3.5-4B"
BASE_REVISION = "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"
UPSTREAM = {
    "unsloth": "62354edbf4fd246cf120ef07272934fb405ec705",
    "unsloth_zoo": "ac1410cc766efedd5e4ced1b1d1338138eb7eaa5",
}
QWEN_DEFAULTS = {
    "epochs": 2,
    "rank": 16,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
}


def resolve_qwen_training(training: dict) -> dict:
    return {
        "backend": "unsloth",
        **QWEN_DEFAULTS,
        **{key: value for key, value in training.items() if key in {*QWEN_DEFAULTS, "max_steps"}},
    }


def fit_qwen(experiment, prepared, output: Path, *, calibrate: bool = True):
    """Same prepared TaskSpec/splits as CPU fitting. Requires actual CUDA execution."""
    from localdesk.model import ModelBackend
    from localdesk.registry import atomic_json

    verify_upstream_revisions()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # isort: off
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported
    import torch
    from transformers import TrainingArguments

    # isort: on
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen training requires an NVIDIA CUDA GPU")
    parameters = resolve_qwen_training(experiment.training)
    bf16 = is_bfloat16_supported()
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=BASE_MODEL,
        revision=BASE_REVISION,
        use_exact_model_name=True,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=parameters.get("rank", 16),
        lora_alpha=parameters.get("rank", 16),
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=experiment.seed,
    )
    items, reports = build_training_items(FastDecisionModel, model, tokenizer, prepared.rows)
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=items["train"],
        args=TrainingArguments(
            per_device_train_batch_size=parameters.get("batch_size", 2),
            gradient_accumulation_steps=parameters.get("gradient_accumulation", 4),
            num_train_epochs=parameters.get("epochs", 2),
            max_steps=parameters.get("max_steps", -1),
            learning_rate=parameters.get("learning_rate", 2e-4),
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            bf16=bf16,
            fp16=False,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=10,
            output_dir=str(output.parent / "trainer-state"),
            report_to="none",
            seed=experiment.seed,
        ),
    )
    stats = trainer.train()
    calibration = (
        FastDecisionModel.calibrate(model, tokenizer, items["calibration"])
        if calibrate
        else {"method": "identity_for_benchmark", "temperature": 1.0}
    )
    model.save_pretrained(str(output), tokenizer)
    manifest = {
        "task": experiment.task.to_dict(),
        "task_sha256": experiment.task.digest,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "dataset_sha256": prepared.manifest["split_sha256"],
        "upstream": UPSTREAM,
        "calibration": calibration,
        "build_reports": reports,
        "training": stats.metrics,
        "smoke_run": "max_steps" in parameters,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "unsloth", "unsloth_zoo")
        },
    }
    atomic_json(output / "localdesk_manifest.json", manifest)
    backend = ModelBackend.from_loaded(FastDecisionModel, model, tokenizer, manifest)
    return backend, manifest


def build_training_items(api, model, tokenizer, splits: dict) -> tuple[dict, dict]:
    items, reports = {}, {}
    for split in ("train", "calibration", "test"):
        items[split], reports[split] = api.build_dataset(splits[split], tokenizer, model)
        if not items[split] or reports[split]["skipped"] or reports[split].get("truncated"):
            raise ValueError(
                f"{split}: upstream skipped/truncated decisions; inspect the build report"
            )
    return items, reports


def verify_upstream_revisions() -> None:
    for name, expected in UPSTREAM.items():
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
        revision = json.loads(raw or "{}").get("vcs_info", {}).get("commit_id")
        if revision != expected:
            raise RuntimeError(f"{name}: install the pinned Git revision from requirements-gpu.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/qwen-localdesk"))
    parser.add_argument(
        "--max-steps", type=int, help="optional smoke run; default is two full epochs"
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.out.exists():
        parser.error("use a new output directory; existing checkpoints will not be overwritten")
    splits = load_splits(args.data)
    verify_upstream_revisions()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    # Import Unsloth first, as required by its patching workflow.
    # isort: off
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported
    import torch
    from transformers import TrainingArguments
    # isort: on

    if not torch.cuda.is_available():
        raise RuntimeError(
            "this training recipe requires an NVIDIA CUDA GPU; use the Colab notebook"
        )
    bf16 = is_bfloat16_supported()
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=BASE_MODEL,
        revision=BASE_REVISION,
        use_exact_model_name=True,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    items, build_reports = build_training_items(FastDecisionModel, model, tokenizer, splits)
    checkpoint_directory = args.out.parent / (args.out.name + "-checkpoints")
    if checkpoint_directory.exists():
        raise RuntimeError("checkpoint directory exists; choose a different --out")
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=items["train"],
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            max_steps=args.max_steps if args.max_steps is not None else -1,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            bf16=bf16,
            fp16=False,
            eval_strategy="no",
            logging_steps=10,
            output_dir=str(checkpoint_directory),
            report_to="none",
            seed=3407,
        ),
    )
    stats = trainer.train()
    calibration = FastDecisionModel.calibrate(model, tokenizer, items["calibration"])
    test_metrics = FastDecisionModel.evaluate(model, tokenizer, items["test"])
    model.save_pretrained(str(args.out), tokenizer)
    dataset_manifest = json.loads((args.data / "manifest.json").read_text())
    manifest = {
        "schema_sha256": schema_hash(),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "upstream": UPSTREAM,
        "dataset_sha256": dataset_manifest["sha256"],
        "source": dataset_manifest["source"],
        "split_counts": dataset_manifest["counts"],
        "calibration": calibration,
        "test_metrics": test_metrics,
        "build_reports": build_reports,
        "training": stats.metrics,
        "smoke_run": args.max_steps is not None,
        "gpu": torch.cuda.get_device_name(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("unsloth", "unsloth_zoo", "torch", "transformers")
        },
        "checkpoint_sha256": {
            path.name: file_hash(path) for path in args.out.iterdir() if path.is_file()
        },
        "policy": {"min_confidence": 0.75, "min_margin": 0.2},
        "limitations": [
            "One organization public holdout, not field validation.",
            "Russian quality and general OOD detection remain unqualified.",
        ],
    }
    (args.out / "localdesk_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    freeze = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
    (args.out / "runtime-freeze.txt").write_text(freeze)
    print(
        json.dumps(
            {"saved_to": str(args.out), "calibration": calibration, "test": test_metrics}, indent=2
        )
    )


if __name__ == "__main__":
    main()
