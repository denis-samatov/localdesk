import gzip
import json
import math
from pathlib import Path

import pytest

from localdesk.baseline import BaselineBackend, features, softmax
from localdesk.schema import CATEGORIES, schema_hash


def test_features_are_deterministic_and_mask_tags_are_removed():
    assert features("Cannot open [NAME] shared folder") == features("cannot open shared folder")
    values = features("A normal request about software")
    assert sum(value * value for value in values.values()) == pytest.approx(1)
    assert all(0 <= index < 4096 for index in values)


def test_softmax_is_stable_and_temperature_changes_confidence():
    values = softmax([1001, 1000], 1)
    assert sum(values) == pytest.approx(1)
    assert values[0] > softmax([1001, 1000], 2)[0]
    assert all(math.isfinite(v) for v in values)


def test_baseline_respects_exported_class_order():
    labels = list(reversed(CATEGORIES))
    bundle = {
        "schema_sha256": schema_hash(),
        "classes": labels,
        "n_features": 4096,
        "coef": [[0] * 4096 for _ in labels],
        "intercept": [5] + [0] * 5,
        "temperature": 1,
        "dataset_sha256": {},
    }
    backend = BaselineBackend(bundle=bundle)
    assert (
        max(
            backend.predict("Any request")["category"],
            key=backend.predict("Any request")["category"].get,
        )
        == labels[0]
    )


def test_baseline_rejects_wrong_schema():
    with pytest.raises(ValueError):
        BaselineBackend(bundle={"schema_sha256": "wrong"})


def test_published_report_is_bound_to_shipped_weights_and_recipe():
    from localdesk.data import file_hash

    root = Path(__file__).resolve().parents[1]
    artifact = root / "src/localdesk/assets/baseline.json.gz"
    report = json.loads((root / "reports/baseline.json").read_text())
    bundle = json.loads(gzip.decompress(artifact.read_bytes()))
    assert file_hash(artifact) == report["model_sha256"]
    assert bundle["dataset_sha256"] == report["dataset_sha256"]
    assert bundle["schema_sha256"] == schema_hash()
    for name, expected in report["recipe_sha256"].items():
        assert file_hash(root / name) == expected
