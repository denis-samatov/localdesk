import gzip
import json
import sys
from pathlib import Path

import pytest

from localdesk import cli
from localdesk.config import TaskSpec, read_json
from localdesk.core import decide
from localdesk.data import file_hash
from localdesk.tfidf import TfidfBackend, demo_backend

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/localdesk/assets/demo.json.gz"


def test_demo_report_is_bound_to_shipped_model_data_and_recipe():
    report = json.loads((ROOT / "reports/demo.json").read_text(encoding="utf-8"))
    bundle = json.loads(gzip.decompress(ASSET.read_bytes()))
    task = TaskSpec.from_dict(read_json(ROOT / "tasks/it-support.json"))
    assert file_hash(ASSET) == report["model_sha256"]
    assert bundle["task_sha256"] == task.digest == report["task_sha256"]
    assert bundle["dataset_sha256"] == report["dataset_split_sha256"]
    assert report["quality"]["qualified"] is True
    for name, expected in report["recipe_sha256"].items():
        assert file_hash(ROOT / name) == expected, f"{name} changed: run scripts/train_demo.py"


def test_demo_routes_clear_requests_and_reviews_ambiguous_ones():
    backend = demo_backend()
    assert isinstance(backend, TfidfBackend)
    clear = decide("File share access request", backend)
    assert (clear["category"], clear["action"]) == ("Fileservice", "route")
    assert decide("hello", backend)["action"] == "review"


@pytest.mark.parametrize(
    "flags, expected", [([], "tfidf"), (["--backend", "baseline"], "baseline")]
)
def test_cli_predict_uses_tfidf_demo_by_default(monkeypatch, capsys, flags, expected):
    monkeypatch.setattr(sys, "argv", ["localdesk", "predict", *flags, "File share access request"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["backend"] == expected


def test_docker_demo_registry_is_a_verified_copy_of_the_packaged_model():
    from localdesk.registry import Registry

    registry = ROOT / "docker/demo-registry"
    backend, manifest = Registry(registry).load_active()
    assert isinstance(backend, TfidfBackend)
    assert manifest["quality"]["qualified"] is True
    copy = registry / manifest["run_id"] / "model.json.gz"
    assert copy.read_bytes() == ASSET.read_bytes(), "run scripts/train_demo.py"
