import json

import pytest
from test_configurable import spec, write_config


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [100 + int(text)]

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [1, 2, 3]


def test_zero_shot_uses_unique_symbolic_tokens_and_descriptions():
    from localdesk.benchmark_qwen import answer_tokens, prompt_ids

    task = spec(("sales", "refunds"))
    tokenizer = Tokenizer()
    assert answer_tokens(tokenizer, task.labels) == [100, 101]
    assert prompt_ids(tokenizer, task, "customer message", 10) == [1, 2, 3]
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert "customer message" in tokenizer.messages[1]["content"]
    assert "0: sales" in tokenizer.messages[0]["content"]
    assert "refunds requests" in tokenizer.messages[0]["content"]


def test_symbol_aliases_and_multitoken_answers_fail():
    from localdesk.benchmark_qwen import answer_tokens

    class Alias:
        def encode(self, *args, **kwargs):
            return [1]

    class Multiple:
        def encode(self, *args, **kwargs):
            return [1, 2]

    for tokenizer in (Alias(), Multiple()):
        with pytest.raises(ValueError, match="single|unique"):
            answer_tokens(tokenizer, ("a", "b"))


def test_oversize_prompt_fails_instead_of_truncating():
    from localdesk.benchmark_qwen import prompt_ids

    with pytest.raises(ValueError, match="context|truncat"):
        prompt_ids(Tokenizer(), spec(("a", "b")), "message", 2)


def test_gpu_unavailable_is_recorded_without_fake_metrics(tmp_path, monkeypatch):
    import localdesk.benchmark_qwen as module
    from localdesk.dataset import prepare

    experiment = write_config(tmp_path)
    prepared = prepare(experiment)
    report = {
        "methods": {"qwen_zero_shot": {"status": "not_run"}, "qwen_decision": {"status": "not_run"}}
    }

    def unavailable():
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(module, "gpu_stack", unavailable)
    module.run_qwen_stages(
        experiment,
        prepared,
        prepared.rows["calibration"],
        tmp_path,
        report,
        lambda *_: pytest.fail("must not score"),
    )
    assert all(method["status"] == "blocked" for method in report["methods"].values())
    assert all("classification" not in method for method in report["methods"].values())
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_qwen_benchmark_can_save_uncalibrated_head_without_changing_default():
    import inspect

    from localdesk.training import fit_qwen

    assert inspect.signature(fit_qwen).parameters["calibrate"].default is True


def test_zero_shot_loader_requests_logits_and_exact_pinned_weights():
    from localdesk.benchmark_qwen import ZeroShotBackend
    from localdesk.training import BASE_REVISION

    calls = []

    class Model:
        def eval(self):
            pass

    class API:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(kwargs)
            return Model(), Tokenizer()

        @staticmethod
        def for_inference(model):
            pass

    class Torch:
        bfloat16 = "bf16"
        float32 = "fp32"

    ZeroShotBackend(spec(("a", "b")), API, Torch, lambda: True)
    assert calls[0]["return_logits"] is True
    assert calls[0]["use_exact_model_name"] is True
    assert calls[0]["revision"] == BASE_REVISION
