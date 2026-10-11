"""Portable learned linear baseline. Inference needs only the Python standard library."""

import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from importlib.resources import files

from localdesk.core import validate_text
from localdesk.schema import CATEGORIES, schema_hash

N_FEATURES = 4096


def features(text: str) -> dict[int, float]:
    # Ignore authors' PII placeholders. Do not ship a vocabulary of ticket content.
    text = unicodedata.normalize("NFKC", validate_text(text))
    text = re.sub(r"\[[^\]]*\]", " ", text).casefold()
    words = re.findall(r"\b\w\w+\b", text)
    grams = words + [f"{left} {right}" for left, right in zip(words, words[1:], strict=False)]
    counts = Counter(
        int.from_bytes(hashlib.sha256(gram.encode()).digest()[:8], "big") % N_FEATURES
        for gram in grams
    )
    norm = math.sqrt(sum(value * value for value in counts.values()))
    return {key: value / norm for key, value in sorted(counts.items())} if norm else {}


def softmax(logits: list[float], temperature: float) -> list[float]:
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("softmax requires logits and a positive finite temperature")
    maximum = max(logits)
    exp = [math.exp((value - maximum) / temperature) for value in logits]
    total = sum(exp)
    return [value / total for value in exp]


class BaselineBackend:
    name = "baseline"
    probability_kind = "softmax_temperature_fitted_on_calibration"

    def __init__(self, *, bundle: dict | None = None):
        if bundle is None:
            payload = files("localdesk").joinpath("assets/baseline.json.gz").read_bytes()
            bundle = json.loads(gzip.decompress(payload))
        if bundle.get("schema_sha256") != schema_hash():
            raise ValueError("baseline schema mismatch")
        if set(bundle["classes"]) != set(CATEGORIES) or len(bundle["classes"]) != len(CATEGORIES):
            raise ValueError("baseline class keys mismatch")
        if bundle["n_features"] != N_FEATURES or len(bundle["coef"]) != len(CATEGORIES):
            raise ValueError("baseline dimensions mismatch")
        if any(len(row) != N_FEATURES for row in bundle["coef"]) or len(bundle["intercept"]) != len(
            CATEGORIES
        ):
            raise ValueError("baseline coefficient shape mismatch")
        if not math.isfinite(bundle["temperature"]) or bundle["temperature"] <= 0:
            raise ValueError("invalid calibration temperature")
        self.bundle = bundle

    def logits(self, text: str) -> list[float]:
        values = features(text)
        return [
            intercept + sum(row[index] * value for index, value in values.items())
            for row, intercept in zip(self.bundle["coef"], self.bundle["intercept"], strict=True)
        ]

    def predict(self, text: str) -> dict[str, dict[str, float]]:
        probabilities = softmax(self.logits(text), self.bundle["temperature"])
        return {"category": dict(zip(self.bundle["classes"], probabilities, strict=True))}
