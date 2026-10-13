"""Repeat the CPU comparison over several seeds and publish mean, spread and paired gains.

The logistic models are deterministic, so a seed changes which examples are used for
fitting and which for calibration: for every seed the non-test rows are re-split into
train and calibration (whole groups, stratified by label, same proportions as the
shipped split). The test set stays fixed so seeds are comparable. Writes
reports/seed-suite.json and reports/seed-suite.md. Prepare the data first.
"""

import argparse
import dataclasses
import json
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from localdesk.analysis import out_of_scope
from localdesk.config import Experiment
from localdesk.core import decide
from localdesk.dataset import prepare
from localdesk.pipeline import run_experiment
from localdesk.registry import Registry

DATASETS = {
    "it-support": "IT Support Tickets",
    "sms-filter": "UCI SMS Spam",
    "banking77": "Banking77",
    "clinc150": "CLINC150",
}
BACKENDS = {
    "baseline": {"backend": "baseline", "C": 4.0, "max_iter": 2000},
    "tfidf": {"backend": "tfidf", "C": 4.0, "max_features": 40000},
}
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("accuracy", "macro_f1", "ece", "coverage", "accepted_accuracy", "coverage_at_95")


def resplit(rows, seed):
    """Re-split pooled train+calibration rows by whole groups, stratified by label."""
    groups = defaultdict(list)
    for row in rows["train"] + rows["calibration"]:
        groups[row["group"]].append(row)
    fraction = len(rows["calibration"]) / (len(rows["train"]) + len(rows["calibration"]))
    by_label = defaultdict(list)
    for group in sorted(groups):
        by_label[groups[group][0]["gold"]["category"]].append(group)
    rng = random.Random(seed)
    calibration = set()
    for label in sorted(by_label):
        members = by_label[label]
        rng.shuffle(members)
        calibration.update(members[: max(1, round(len(members) * fraction))])
    return {
        "train": [r for g in sorted(groups) if g not in calibration for r in groups[g]],
        "calibration": [r for g in sorted(groups) if g in calibration for r in groups[g]],
        "test": rows["test"],
    }


def write_experiment(directory, experiment, splits, seed):
    for split, rows in splits.items():
        (directory / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": r["id"],
                        "text": r["state"],
                        "label": r["gold"]["category"],
                        "group": r["group"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for r in rows
            ),
            encoding="utf-8",
        )
    (directory / "task.json").write_text(json.dumps(experiment.task.to_dict()), encoding="utf-8")
    config = {
        "task": "task.json",
        "dataset": {
            "splits": {split: f"{split}.jsonl" for split in splits},
            "format": "jsonl",
            "text_column": "text",
            "label_column": "label",
            "id_column": "id",
            "group_column": "group",
        },
        "registry": "registry",
        "seed": seed,
        "gates": experiment.gates,
    }
    path = directory / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return Experiment.load(path)


def run(name, seed, workspace):
    shipped = Experiment.load(Path("experiments") / f"{name}.json")
    rows = prepare(shipped).rows
    directory = workspace / f"{name}-{seed}"
    directory.mkdir()
    experiment = write_experiment(directory, shipped, resplit(rows, seed), seed)
    ood = Path("data") / name / "ood.jsonl"
    outside = (
        [json.loads(line)["text"] for line in ood.read_text("utf-8").splitlines()]
        if ood.is_file()
        else None
    )
    entries = {}
    for kind, training in BACKENDS.items():
        variant = dataclasses.replace(experiment, training=training)
        result = run_experiment(variant, run_id=kind)
        report = result["report"]
        entry = {
            "accuracy": report["classification"]["accuracy"],
            "macro_f1": report["classification"]["macro_f1"],
            "ece": report["classification"]["ece"],
            "coverage": report["coverage"],
            "accepted_accuracy": report["accepted_accuracy"],
            "coverage_at_95": report["risk_coverage"]["coverage_at_accuracy"]["0.95"],
            "split_counts": report["split_counts"],
        }
        if outside:
            backend, _ = Registry(variant.registry).load(kind, require_qualified=False)
            in_scope = [
                max(decide(r["state"], backend)["probabilities"]["category"].values())
                for r in rows["test"]
            ]
            entry["out_of_scope_review"] = out_of_scope(outside, in_scope, backend)[
                "sent_to_review"
            ]
        entries[kind] = entry
    return entries


def summary(values):
    values = [v for v in values if v is not None]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def aggregate(runs):
    result = {}
    for name, seeds in runs.items():
        result[name] = {"seeds": list(seeds), "backends": {}, "tfidf_minus_baseline": {}}
        keys = [
            *METRICS,
            *(
                ["out_of_scope_review"]
                if "out_of_scope_review" in next(iter(seeds.values()))["tfidf"]
                else []
            ),
        ]
        for kind in BACKENDS:
            result[name]["backends"][kind] = {
                key: summary([seeds[s][kind][key] for s in seeds]) for key in keys
            }
        for key in ("accuracy", "macro_f1", "coverage", "coverage_at_95"):
            result[name]["tfidf_minus_baseline"][key] = summary(
                [seeds[s]["tfidf"][key] - seeds[s]["baseline"][key] for s in seeds]
            )
    return result


def cell(stats):
    return f"{stats['mean'] * 100:.1f} ± {stats['std'] * 100:.1f}%"


def markdown(result):
    first = next(iter(result.values()))
    lines = [
        "# CPU backends over several seeds",
        "",
        "Generated by `scripts/publish_seed_suite.py`. For each seed "
        f"({', '.join(map(str, first['seeds']))}) the non-test rows are re-split into train",
        "and calibration by whole groups, stratified by label, with the shipped proportions.",
        "The test set is fixed. Values are mean ± sample standard deviation over seeds.",
        "",
        "| Dataset | Backend | Accuracy | Macro F1 | ECE | Auto-routed | Accuracy when routed "
        "| Coverage at 95% accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in result.items():
        for kind, stats in entry["backends"].items():
            lines.append(
                f"| {DATASETS[name]} | {kind} | {cell(stats['accuracy'])} "
                f"| {cell(stats['macro_f1'])} | {cell(stats['ece'])} | {cell(stats['coverage'])} "
                f"| {cell(stats['accepted_accuracy'])} | {cell(stats['coverage_at_95'])} |"
            )
    lines += [
        "",
        "## Paired gain of TF-IDF over the baseline (same seed, same split)",
        "",
        "| Dataset | Accuracy gain: mean (min … max) | Macro-F1 gain: mean (min … max) "
        "| Auto-routed gain: mean (min … max) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, entry in result.items():
        gains = entry["tfidf_minus_baseline"]
        lines.append(
            f"| {DATASETS[name]} | "
            + " | ".join(
                f"{gains[k]['mean'] * 100:+.1f} pp ({gains[k]['min'] * 100:+.1f} … "
                f"{gains[k]['max'] * 100:+.1f})"
                for k in ("accuracy", "macro_f1", "coverage")
            )
            + " |"
        )
    for name, entry in result.items():
        if "out_of_scope_review" in entry["backends"]["tfidf"]:
            lines += [
                "",
                f"## Out-of-scope queries ({DATASETS[name]})",
                "",
                "| Backend | Sent to review |",
                "| --- | ---: |",
            ]
            for kind, stats in entry["backends"].items():
                lines.append(f"| {kind} | {cell(stats['out_of_scope_review'])} |")
    lines += [
        "",
        "## Limits",
        "",
        "Seeds vary the train/calibration partition only; the fixed test sets are public",
        "holdouts reused across methods, so this measures sensitivity to the training and",
        "calibration sample, not to a new test population. Qwen fine-tuning needs a GPU and",
        "is not part of this repeat.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    runs = {}
    with tempfile.TemporaryDirectory() as directory:
        for name in args.datasets:
            runs[name] = {}
            for seed in args.seeds:
                runs[name][seed] = run(name, seed, Path(directory))
                tfidf, baseline = runs[name][seed]["tfidf"], runs[name][seed]["baseline"]
                print(
                    f"{name} seed={seed} baseline={baseline['accuracy']:.4f} "
                    f"tfidf={tfidf['accuracy']:.4f}",
                    flush=True,
                )
    result = aggregate(runs)
    Path("reports/seed-suite.json").write_text(
        json.dumps({"runs": runs, "summary": result}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("reports/seed-suite.md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
