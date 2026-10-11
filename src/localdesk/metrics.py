"""Classification and abstention metrics with explicit support counts."""

import math

from localdesk.core import Backend, decide, validate_distribution
from localdesk.schema import CATEGORIES


def classification_metrics(gold: list[str], probabilities: list[dict], choices: list[str]) -> dict:
    if not gold or len(gold) != len(probabilities) or any(label not in choices for label in gold):
        raise ValueError("metrics require non-empty, aligned predictions and valid gold labels")
    predictions = [validate_distribution(values, choices) for values in probabilities]
    selected = [max(values, key=values.get) for values in predictions]
    correct = [answer == target for answer, target in zip(selected, gold, strict=True)]
    count = len(gold)
    bins = [[] for _ in range(10)]
    for values, hit in zip(predictions, correct, strict=True):
        confidence = max(values.values())
        bins[min(int(confidence * 10), 9)].append((confidence, hit))
    ece = sum(
        len(bucket)
        / count
        * abs(
            sum(conf for conf, _ in bucket) / len(bucket)
            - sum(hit for _, hit in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    support = {label: gold.count(label) for label in choices}
    recalls = {
        label: sum(hit and target == label for hit, target in zip(correct, gold, strict=True))
        / support[label]
        if support[label]
        else None
        for label in choices
    }
    return {
        "count": count,
        "accuracy": sum(correct) / count,
        "ece": ece,
        "nll": -sum(
            math.log(max(values[target], 1e-15))
            for values, target in zip(predictions, gold, strict=True)
        )
        / count,
        "brier": sum(
            sum((values[label] - (label == target)) ** 2 for label in choices)
            for values, target in zip(predictions, gold, strict=True)
        )
        / count,
        "macro_recall": sum(v for v in recalls.values() if v is not None)
        / sum(v is not None for v in recalls.values()),
        "per_class_recall": recalls,
        "per_class_support": support,
    }


def evaluate_rows(rows: list[dict], backend: Backend) -> dict:
    if not rows:
        raise ValueError("evaluation requires rows")
    results = [decide(row["state"], backend) for row in rows]
    metrics = classification_metrics(
        [row["gold"]["category"] for row in rows],
        [result["probabilities"]["category"] for result in results],
        list(CATEGORIES),
    )
    accepted = [
        (row, result)
        for row, result in zip(rows, results, strict=True)
        if result["action"] == "route"
    ]
    return {
        "backend": backend.name,
        "probability_kind": backend.probability_kind,
        "count": len(rows),
        "classification": metrics,
        "accepted_count": len(accepted),
        "coverage": len(accepted) / len(rows),
        "accepted_accuracy": sum(
            result["category"] == row["gold"]["category"] for row, result in accepted
        )
        / len(accepted)
        if accepted
        else None,
        "policy": results[0]["policy"],
    }
