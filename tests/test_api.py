import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from test_configurable import write_config

import localdesk.api
from localdesk.api import create_app
from localdesk.pipeline import run_experiment
from localdesk.registry import Registry


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("api-artifact")
    experiment = write_config(tmp_path, ("red", "green"), gates={"min_accuracy": 0.0})
    run_experiment(experiment, run_id="first")
    Registry(experiment.registry).promote("first")
    return experiment.registry


@pytest.fixture
def client(artifact):
    with TestClient(create_app(registry=artifact, api_key="test-key")) as client:
        client.headers["X-API-Key"] = "test-key"
        yield client


def test_task_owned_labels_through_actual_api(client):
    response = client.post("/v1/predictions", json={"text": "red request topic0"})
    assert response.status_code == 200
    value = response.json()
    assert set(value["probabilities"]["category"]) == {"red", "green"}
    assert value["model_version"] == "first"
    assert len(response.headers["X-Request-ID"]) == 32


def test_batch_and_metadata(client):
    response = client.post("/v1/predictions/batch", json={"texts": ["red topic0", "green topic1"]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert set(client.get("/v1/model").json()["task"]["labels"]) == {"red", "green"}
    assert client.get("/v1/model/report").json()["count"] > 0
    assert client.get("/v1/model/report").json()["model_version"] == "first"
    assert client.get("/v1/model/plan").json()["model_version"] == "first"
    assert client.get("/v1/model/plan").json()["stages"]
    assert "LocalDesk" in client.get("/").text


def test_api_rejects_unknown_fields_and_non_strings(client):
    for payload in (
        {"text": "red", "extra": True},
        {"text": 123},
        {"text": " "},
        {"text": "x" * 4001},
    ):
        response = client.post("/v1/predictions", json=payload)
        assert response.status_code == 422
        assert "error" in response.json()
        assert "text" not in str(response.json())  # Do not reflect input field contents.


def test_auth_is_not_a_cli_only_check(client):
    client.headers.pop("X-API-Key")
    assert client.post("/v1/predictions", json={"text": "red"}).status_code == 401
    assert client.get("/health/live").status_code == 200
    assert client.get("/v1/model").status_code == 401


def test_factory_requires_explicit_auth_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALDESK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        create_app(registry=tmp_path)


def test_transport_body_limit_and_batch_bound(client):
    assert (
        client.post(
            "/v1/predictions", content=b"x" * 131073, headers={"Content-Type": "application/json"}
        ).status_code
        == 413
    )
    assert client.post("/v1/predictions/batch", json={"texts": ["red"] * 17}).status_code == 422


def test_readiness_and_metrics(client):
    assert client.get("/health/ready").status_code == 200
    client.post("/v1/predictions", json={"text": "red request topic0"})
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "localdesk_predictions_total" in metrics.text
    assert "red request" not in metrics.text


def test_http_errors_match_documented_envelope_and_openapi(client):
    for response in (client.get("/not-a-route"), client.get("/v1/predictions")):
        assert response.status_code in {404, 405}
        assert set(response.json()) == {"error"}
    schema = client.get("/openapi.json").json()
    error_schema = schema["paths"]["/v1/predictions"]["post"]["responses"]["422"]["content"][
        "application/json"
    ]["schema"]
    assert error_schema["$ref"].endswith("/ErrorResponse")


def test_dashboard_does_not_depend_on_system_text_encoding(client, monkeypatch):
    from pathlib import Path

    original = Path.read_text

    def windows_read(self, encoding=None, errors=None):
        return original(self, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", windows_read)
    response = client.get("/")
    assert response.status_code == 200
    assert "Данные меняются" in response.text


def test_unavailable_artifact_fails_readiness_and_predictions(tmp_path):
    with TestClient(create_app(registry=tmp_path, api_key="test-key")) as client:
        client.headers["X-API-Key"] = "test-key"
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/predictions", json={"text": "red request"}).status_code == 503


def test_legacy_predict_alias_is_preserved(client):
    assert client.post("/api/decide", json={"text": "red request topic0"}).status_code == 200


def test_anonymous_loopback_rejects_rebinding_host_and_origin(artifact, monkeypatch):
    monkeypatch.delenv("LOCALDESK_API_KEY", raising=False)
    app = create_app(registry=artifact, anonymous_local=True)
    with TestClient(app, client=("127.0.0.1", 12345), base_url="http://attacker.invalid") as client:
        assert client.get("/v1/model").status_code == 403
    with TestClient(app, client=("127.0.0.1", 12345), base_url="http://127.0.0.1:8765") as client:
        assert client.get("/v1/model").status_code == 200
        assert (
            client.get("/v1/model", headers={"Origin": "http://attacker.invalid"}).status_code
            == 403
        )


class BlockingBackend:
    """Wraps a loaded backend so a test can hold one inference in flight."""

    def __init__(self, backend):
        self.backend, self.entered, self.release = backend, threading.Event(), threading.Event()
        self.name, self.probability_kind, self.task = (
            backend.name,
            backend.probability_kind,
            backend.task,
        )

    def predict(self, text):
        self.entered.set()
        assert self.release.wait(10)
        return self.backend.predict(text)


def test_concurrent_requests_queue_instead_of_failing(artifact):
    with TestClient(create_app(registry=artifact, api_key="test-key")) as client:
        client.headers["X-API-Key"] = "test-key"
        blocking = BlockingBackend(client.app.state.backend)
        client.app.state.backend = blocking
        with ThreadPoolExecutor(4) as pool:
            first = pool.submit(client.post, "/v1/predictions", json={"text": "red topic0"})
            assert blocking.entered.wait(10)
            queued = [
                pool.submit(client.post, "/v1/predictions", json={"text": "green topic1"})
                for _ in range(3)
            ]
            time.sleep(0.2)
            blocking.release.set()
            responses = [first.result(10), *(future.result(10) for future in queued)]
        assert [response.status_code for response in responses] == [200] * 4


def test_queue_timeout_reports_busy(artifact, monkeypatch):
    monkeypatch.setattr(localdesk.api, "QUEUE_TIMEOUT", 0.1)
    with TestClient(create_app(registry=artifact, api_key="test-key")) as client:
        client.headers["X-API-Key"] = "test-key"
        blocking = BlockingBackend(client.app.state.backend)
        client.app.state.backend = blocking
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(client.post, "/v1/predictions", json={"text": "red topic0"})
            assert blocking.entered.wait(10)
            waiting = client.post("/v1/predictions", json={"text": "green topic1"})
            blocking.release.set()
            assert first.result(10).status_code == 200
        assert waiting.status_code == 503
        assert waiting.json()["error"]["code"] == "MODEL_BUSY"


def test_metrics_declare_every_counter_type(client):
    lines = client.get("/metrics").text.splitlines()
    samples = [line.split()[0] for line in lines if not line.startswith("#")]
    assert samples and all(f"# TYPE {name} counter" in lines for name in samples)
