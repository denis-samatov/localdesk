import pytest
from fastapi.testclient import TestClient
from test_configurable import write_config

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
