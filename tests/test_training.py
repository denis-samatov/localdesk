import ast
import json
from pathlib import Path

import pytest

from localdesk.training import build_training_items


def test_unsloth_import_precedes_torch_and_transformers():
    import inspect

    import localdesk.training

    source = inspect.getsource(localdesk.training.main)
    assert source.index("from unsloth import") < source.index("import torch")
    assert source.index("from unsloth import") < source.index("from transformers import")


def test_build_preserves_three_disjoint_splits_and_excludes_ood():
    calls = []

    class FakeAPI:
        @staticmethod
        def build_dataset(rows, tokenizer, model):
            calls.append(rows)
            return rows, {"skipped": 0, "truncated": 0}

    splits = {key: [{"id": key}] for key in ("train", "calibration", "test", "ood")}
    items, reports = build_training_items(FakeAPI, object(), object(), splits)
    assert set(items) == {"train", "calibration", "test"}
    assert calls == [splits["train"], splits["calibration"], splits["test"]]
    assert splits["ood"] not in calls


@pytest.mark.parametrize("report", [{"skipped": 1}, {"skipped": 0, "truncated": 1}])
def test_upstream_data_loss_stops_training(report):
    class FakeAPI:
        @staticmethod
        def build_dataset(rows, tokenizer, model):
            return rows, report

    with pytest.raises(ValueError, match="skipped/truncated"):
        build_training_items(FakeAPI, None, None, {"train": [{"id": "one"}]})


def test_notebook_cells_are_clean_and_python_cells_parse():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/train_qwen.ipynb"
    document = json.loads(notebook.read_text())
    assert document["nbformat"] == 4
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            source = "".join(cell["source"])
            ast.parse(source)
