"""Selective-classification analysis: risk-coverage curves and out-of-scope behaviour."""

import math

from localdesk.core import Backend, decide


def risk_coverage(confidences: list[float], correct: list[bool]) -> dict:
    """Accuracy of the most confident share of predictions, at each coverage level.

    Descriptive only: thresholds read off a test set are not a calibrated policy.
    AURC is the mean selective risk over all prefixes (lower is better).
    """
    if not confidences or len(confidences) != len(correct):
        raise ValueError("risk-coverage requires aligned, non-empty predictions")
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    count, hits, risks, prefix = len(order), 0, [], []
    for rank, index in enumerate(order, 1):
        hits += correct[index]
        prefix.append(hits / rank)
        risks.append(1 - hits / rank)

    def point(size):
        return {
            "coverage": size / count,
            "accuracy": prefix[size - 1],
            "min_confidence": confidences[order[size - 1]],
        }

    levels = sorted({max(1, math.ceil(count * step / 20)) for step in range(1, 21)})
    return {
        "ranking": "max_probability",
        "aurc": sum(risks) / count,
        "points": [point(size) for size in levels],
        "coverage_at_accuracy": {
            f"{target:.2f}": max(
                (size / count for size in range(1, count + 1) if prefix[size - 1] >= target),
                default=0.0,
            )
            for target in (0.9, 0.95, 0.99)
        },
    }


def selective_report(rows: list[dict], backend: Backend) -> dict:
    """Risk-coverage curve of a backend on labelled rows."""
    results = [decide(row["state"], backend) for row in rows]
    return risk_coverage(
        [max(result["probabilities"]["category"].values()) for result in results],
        [
            result["category"] == row["gold"]["category"]
            for row, result in zip(rows, results, strict=True)
        ],
    )


def out_of_scope(texts: list[str], in_scope: list[float], backend: Backend) -> dict:
    """How a model treats queries outside every label, versus in-scope test queries.

    ``in_scope`` holds the max probabilities on the in-scope test rows. AUROC measures
    how well max probability separates in-scope from out-of-scope (1.0 is perfect).
    """
    if not texts or not in_scope:
        raise ValueError("out-of-scope analysis requires both query sets")
    results = [decide(text, backend) for text in texts]
    outside = [max(result["probabilities"]["category"].values()) for result in results]
    # Mann-Whitney AUROC with ties counted as one half.
    ranked = sorted([(value, 1) for value in in_scope] + [(value, 0) for value in outside])
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[index][0]:
            end += 1
        average = (index + end) / 2 + 1
        rank_sum += average * sum(flag for _, flag in ranked[index : end + 1])
        index = end + 1
    positives = len(in_scope)
    auroc = (rank_sum - positives * (positives + 1) / 2) / (positives * len(outside))
    reviewed = sum(result["action"] == "review" for result in results)
    return {
        "count": len(texts),
        "sent_to_review": reviewed / len(texts),
        "auto_routed": 1 - reviewed / len(texts),
        "auroc_max_probability": auroc,
        "policy": results[0]["policy"],
    }
