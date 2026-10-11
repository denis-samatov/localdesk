"""Backend-independent validation and abstention; no learned action labels are invented."""

import math
from typing import Protocol

from localdesk.schema import QUESTIONS

MAX_TEXT = 4000


class Backend(Protocol):
    name: str
    probability_kind: str

    def predict(self, text: str) -> dict[str, dict[str, float]]: ...


def validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError(f"text must be a non-empty string of at most {MAX_TEXT} characters")
    return text.strip()


def validate_distribution(values: dict, choices: list[str]) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(choices):
        raise ValueError("backend probabilities must contain exactly the expected option keys")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values.values()
    ):
        raise ValueError("backend probabilities must be finite numbers between zero and one")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-5):
        raise ValueError("backend probabilities must sum to one")
    return {key: float(values[key]) for key in choices}


def decide(
    text: str,
    backend: Backend,
    *,
    min_confidence: float = 0.75,
    min_margin: float = 0.2,
) -> dict:
    text = validate_text(text)
    for threshold in (min_confidence, min_margin):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("thresholds must be finite numbers between zero and one")
    raw = backend.predict(text)
    if not isinstance(raw, dict) or set(raw) != set(QUESTIONS):
        raise ValueError("backend must return exactly the category question")
    probs = validate_distribution(raw["category"], list(QUESTIONS["category"]["criteria"]))
    ranked = sorted(probs, key=probs.get, reverse=True)
    reasons = []
    if probs[ranked[0]] + 1e-12 < min_confidence:
        reasons.append("low_confidence")
    if probs[ranked[0]] - probs[ranked[1]] + 1e-12 < min_margin:
        reasons.append("small_margin")
    return {
        "category": ranked[0],
        "action": "review" if reasons else "route",
        "probabilities": {"category": probs},
        "review_reasons": reasons,
        "backend": backend.name,
        "probability_kind": backend.probability_kind,
        "policy": {"min_confidence": min_confidence, "min_margin": min_margin},
    }
