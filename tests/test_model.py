import json

import pytest

from localdesk.model import ModelBackend, validate_manifest
from localdesk.schema import schema_hash


def test_model_requires_explicit_existing_local_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        ModelBackend(tmp_path)
    with pytest.raises(ValueError):
        ModelBackend(tmp_path / "missing")


def test_manifest_rejects_wrong_schema_or_missing_calibration():
    with pytest.raises(ValueError):
        validate_manifest({"schema_sha256": "wrong"})
    with pytest.raises(ValueError):
        validate_manifest({"schema_sha256": schema_hash(), "calibration": None})


def test_adapter_uses_explicit_probability_keys_and_never_falls_back(tmp_path, monkeypatch):
    import sys
    import types

    calls = []

    class FakeDecisionModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(kwargs)
            return object(), object()

        @staticmethod
        def for_inference(model):
            pass

        @staticmethod
        def predict(model, tokenizer, state, questions):
            return {
                name: {
                    "probabilities": {
                        key: 1 / len(question["criteria"]) for key in reversed(question["criteria"])
                    }
                }
                for name, question in questions.items()
            }

    manifest = {
        "schema_sha256": schema_hash(),
        "calibration": {"ece": 0.1},
        "base_model": "test/base",
        "dataset_sha256": {"train": "a", "calibration": "b", "test": "c"},
        "upstream": {"unsloth": "test", "unsloth_zoo": "test"},
    }
    (tmp_path / "localdesk_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setitem(
        sys.modules, "unsloth", types.SimpleNamespace(FastDecisionModel=FakeDecisionModel)
    )
    backend = ModelBackend(tmp_path)
    assert calls[0]["model_name"] == str(tmp_path.resolve())
    assert calls[0]["local_files_only"] is True
    assert backend.predict("A request")["category"]["Fileservice"] == 1 / 6
    assert backend.name == "unsloth"


def test_pinned_four_decimal_unsloth_output_is_normalized_at_adapter_boundary():
    from localdesk.core import validate_distribution
    from localdesk.schema import CATEGORIES

    values = dict(zip(CATEGORIES, [0.3, 0.25, 0.2, 0.15, 0.08, 0.0201], strict=True))
    with pytest.raises(ValueError, match="sum to one"):
        validate_distribution(values, list(CATEGORIES))

    class API:
        @staticmethod
        def for_inference(model):
            pass

        @staticmethod
        def predict(*args):
            return {"category": {"probabilities": values}}

    manifest = {
        "schema_sha256": schema_hash(),
        "calibration": {"temperature": 1},
        "base_model": "fixture",
        "dataset_sha256": {},
        "upstream": {},
    }
    backend = ModelBackend.from_loaded(API, object(), object(), manifest)
    actual = backend.predict("request")["category"]
    assert sum(actual.values()) == pytest.approx(1.0, abs=1e-12)
    assert actual == pytest.approx(
        {key: value / sum(values.values()) for key, value in values.items()}
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.3, 0.25, 0.2, 0.05, 0, 0],
        [0.3, 0.25, 0.2, 0.15, 0.08, 0.020123],
        [True, 0, 0, 0, 0, 0],
    ],
)
def test_adapter_does_not_accept_missing_mass_arbitrary_precision_or_booleans(values):
    from localdesk.schema import CATEGORIES

    class API:
        @staticmethod
        def for_inference(model):
            pass

        @staticmethod
        def predict(*args):
            return {"category": {"probabilities": dict(zip(CATEGORIES, values, strict=True))}}

    manifest = {
        "schema_sha256": schema_hash(),
        "calibration": {"temperature": 1},
        "base_model": "fixture",
        "dataset_sha256": {},
        "upstream": {},
    }
    backend = ModelBackend.from_loaded(API, object(), object(), manifest)
    with pytest.raises(ValueError):
        backend.predict("request")
