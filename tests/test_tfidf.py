import gzip
import io
import json
import math
from collections import Counter

import pytest
from test_configurable import write_config

import localdesk.sources
from localdesk.analysis import out_of_scope, risk_coverage
from localdesk.config import Experiment
from localdesk.core import decide
from localdesk.pipeline import run_experiment
from localdesk.registry import Registry
from localdesk.sources import isolate_splits, read_bounded
from localdesk.tfidf import TfidfBackend, pack, term_counts, unpack


@pytest.mark.parametrize("labels", [("red", "green"), ("red", "green", "blue")])
def test_tfidf_fit_export_and_registry_round_trip(tmp_path, labels):
    experiment = write_config(tmp_path, labels, training={"backend": "tfidf", "C": 4.0})
    result = run_experiment(experiment, run_id="tfidf")
    assert result["manifest"]["backend"] == "tfidf"
    assert "risk_coverage" in result["report"]
    backend, manifest = Registry(experiment.registry).load("tfidf", require_qualified=False)
    assert isinstance(backend, TfidfBackend)
    assert manifest["task_sha256"] == experiment.task.digest
    value = decide(f"{labels[0]} request topic3", backend)
    assert set(value["probabilities"]["category"]) == set(labels)
    assert value["category"] == labels[0]


def test_tampered_tfidf_payload_is_rejected(tmp_path):
    experiment = write_config(tmp_path, training={"backend": "tfidf"})
    run_experiment(experiment, run_id="tfidf")
    path = experiment.registry / "tfidf" / "model.json.gz"
    bundle = json.loads(gzip.decompress(path.read_bytes()))
    with pytest.raises(ValueError, match="shape"):
        TfidfBackend(bundle=dict(bundle, coef=pack(unpack(bundle["coef"])[:-1])))
    with pytest.raises(ValueError, match="non-finite"):
        TfidfBackend(bundle=dict(bundle, intercept=[math.inf] * len(bundle["intercept"])))
    path.write_bytes(path.read_bytes()[:-8] + b"\0" * 8)
    with pytest.raises(ValueError, match="checksum"):
        Registry(experiment.registry).load("tfidf", require_qualified=False)


def test_character_grams_match_sklearn_char_wb():
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = "VPN is down at HQ; a ok"
    analyzer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False)
    expected = analyzer.build_analyzer()(text.casefold())
    from localdesk import tfidf

    grams = []
    original = tfidf._bucket
    try:
        tfidf._bucket = lambda prefix, gram: grams.append((prefix, gram)) or 0
        term_counts(text)
    finally:
        tfidf._bucket = original
    assert Counter(gram for prefix, gram in grams if prefix == b"c") == Counter(expected)


@pytest.mark.parametrize(
    "training, message",
    [
        ({"backend": "tfidf", "epochs": 2}, "do not apply"),
        ({"backend": "baseline", "max_features": 1000}, "do not apply"),
        ({"backend": "tfidf", "max_features": 10}, "max_features"),
        ({"backend": "tfidf", "max_features": True}, "max_features"),
    ],
)
def test_tfidf_training_fields_are_validated(tmp_path, training, message):
    with pytest.raises(ValueError, match=message):
        write_config(tmp_path, training=training)


def test_risk_coverage_orders_by_confidence():
    curve = risk_coverage([0.9, 0.8, 0.7, 0.6], [True, True, False, True])
    assert curve["points"][-1] == {"coverage": 1.0, "accuracy": 0.75, "min_confidence": 0.6}
    assert curve["coverage_at_accuracy"]["0.95"] == 0.5
    assert curve["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 1 / 4) / 4)


class FixedBackend:
    name, probability_kind, task = "fixed", "test", None

    def __init__(self, confidence):
        self.confidence = confidence

    def predict(self, text):
        rest = (1 - self.confidence) / 5
        labels = ["Active Directory", "Computer-Services", "Fileservice", "O365", "Software"]
        return {"category": {"Support general": self.confidence, **dict.fromkeys(labels, rest)}}


def test_out_of_scope_reports_review_share_and_auroc():
    result = out_of_scope(["anything"] * 4, [0.9, 0.95, 0.99], FixedBackend(0.3))
    assert result["sent_to_review"] == 1.0
    assert result["auroc_max_probability"] == 1.0
    tied = out_of_scope(["anything"], [0.3], FixedBackend(0.3))
    assert tied["auroc_max_probability"] == 0.5


def test_isolate_splits_keeps_test_copy_and_drops_conflicts():
    records = [
        {"id": "a", "text": "Reset my password", "label": "auth", "split": "train"},
        {"id": "b", "text": "reset my password!", "label": "auth", "split": "test"},
        {"id": "c", "text": "Printer jam", "label": "hardware", "split": "train"},
        {"id": "d", "text": "printer jam", "label": "auth", "split": "calibration"},
        {"id": "e", "text": "New laptop please", "label": "hardware", "split": "train"},
    ]
    splits, audit = isolate_splits(records)
    assert [r["id"] for r in splits["test"]] == ["b"]
    assert [r["id"] for r in splits["train"]] == ["e"]
    assert splits["calibration"] == []
    assert audit["duplicate_rows"] == 1 and audit["conflicting_rows"] == 2


class FakeResponse(io.BytesIO):
    def __init__(self, body, length):
        super().__init__(body)
        self.headers = {"Content-Length": str(length)}

    def __exit__(self, *args):
        self.close()


def test_truncated_downloads_are_retried(monkeypatch):
    bodies = iter([FakeResponse(b"abc", 6), FakeResponse(b"abcdef", 6)])
    monkeypatch.setattr(localdesk.sources.urllib.request, "urlopen", lambda *_, **__: next(bodies))
    assert read_bounded("https://example.invalid/file", 100) == b"abcdef"


def test_downloads_fail_after_repeated_truncation(monkeypatch):
    monkeypatch.setattr(
        localdesk.sources.urllib.request, "urlopen", lambda *_, **__: FakeResponse(b"abc", 6)
    )
    with pytest.raises(OSError, match="incomplete"):
        read_bounded("https://example.invalid/file", 100)
    with pytest.raises(ValueError, match="size"):
        read_bounded("https://example.invalid/file", 2)


def test_shipped_experiments_use_tfidf():
    from pathlib import Path

    for path in sorted((Path(__file__).parents[1] / "experiments").glob("*.json")):
        assert Experiment.load(path).training["backend"] == "tfidf"


def test_tfidf_version_serves_through_api(tmp_path):
    from fastapi.testclient import TestClient

    from localdesk.api import create_app

    experiment = write_config(tmp_path, training={"backend": "tfidf"}, gates={"min_accuracy": 0})
    run_experiment(experiment, run_id="served")
    Registry(experiment.registry).promote("served")
    with TestClient(create_app(registry=experiment.registry, api_key="key")) as client:
        client.headers["X-API-Key"] = "key"
        assert client.get("/health/ready").json()["model_version"] == "served"
        value = client.post("/v1/predictions", json={"text": "green request topic1"}).json()
        assert value["backend"] == "tfidf" and value["category"] == "green"
        assert "risk_coverage" in client.get("/v1/model/report").json()
