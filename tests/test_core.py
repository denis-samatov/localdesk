import math

import pytest

from localdesk.core import decide, validate_distribution
from localdesk.schema import CATEGORIES, QUESTIONS, schema_hash


class FixedBackend:
    name = "test"
    probability_kind = "test"

    def __init__(self, confidence=0.8):
        self.confidence = confidence

    def predict(self, text):
        choices = list(CATEGORIES)
        return {
            "category": {
                key: self.confidence
                if key == choices[0]
                else (1 - self.confidence) / (len(choices) - 1)
                for key in choices
            }
        }


def test_confident_request_routes():
    result = decide("A routine request", FixedBackend())
    assert result["category"] == "Active Directory"
    assert result["action"] == "route"


def test_low_confidence_requires_review():
    result = decide("A request", FixedBackend(0.3))
    assert result["action"] == "review"
    assert "low_confidence" in result["review_reasons"]


@pytest.mark.parametrize("text", ["", "  ", None, 42, "x" * 4001])
def test_invalid_text_rejected(text):
    with pytest.raises(ValueError):
        decide(text, FixedBackend())


@pytest.mark.parametrize(
    "distribution",
    [
        {"a": 0.5},
        {"a": 0.4, "b": 0.4},
        {"a": -0.1, "b": 1.1},
        {"a": math.nan, "b": 1},
        {"a": math.inf, "b": 0},
        {"a": True, "b": 0},
        {"a": "0.5", "b": 0.5},
        {"a": 0.5, "b": 0.5, "extra": 0},
    ],
)
def test_invalid_distributions_fail_closed(distribution):
    with pytest.raises(ValueError):
        validate_distribution(distribution, ["a", "b"])


def test_confidence_boundary():
    assert decide("A request", FixedBackend(0.75))["action"] == "route"
    assert decide("A request", FixedBackend(0.749))["action"] == "review"


def test_margin_boundary():
    class Tied(FixedBackend):
        def predict(self, text):
            values = dict.fromkeys(CATEGORIES, 0.0)
            values["Active Directory"], values["Fileservice"] = 0.6, 0.4
            return {"category": values}

    assert decide("A request", Tied(), min_confidence=0.5, min_margin=0.2)["action"] == "route"
    assert decide("A request", Tied(), min_confidence=0.5, min_margin=0.201)["action"] == "review"


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), True, "0.5", -1, 2])
def test_bad_policy_rejected(threshold):
    with pytest.raises(ValueError):
        decide("A request", FixedBackend(), min_confidence=threshold)


def test_backend_failure_is_not_hidden():
    class Broken(FixedBackend):
        def predict(self, text):
            raise RuntimeError("GPU unavailable")

    with pytest.raises(RuntimeError, match="GPU unavailable"):
        decide("A request", Broken())


def test_fixed_choice_contract():
    assert set(QUESTIONS) == {"category"}
    assert len(CATEGORIES) == 6
    assert "EOL" not in CATEGORIES
    assert len(schema_hash()) == 64
