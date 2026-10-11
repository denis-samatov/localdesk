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
