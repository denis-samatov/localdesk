import json
import threading
import urllib.error
import urllib.request

import pytest

from localdesk.baseline import BaselineBackend
from localdesk.server import make_server


@pytest.fixture
def server_url():
    server = make_server(BaselineBackend(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def request(url, body):
    return urllib.request.Request(
        url + "/api/decide", data=body, headers={"Content-Type": "application/json"}
    )


def test_serves_local_ui(server_url):
    with urllib.request.urlopen(server_url) as response:
        assert b"LocalDesk" in response.read()
        assert "default-src" in response.headers["Content-Security-Policy"]


def test_json_prediction(server_url):
    with urllib.request.urlopen(
        request(server_url, json.dumps({"text": "File share access request"}).encode())
    ) as response:
        assert json.load(response)["category"] == "Fileservice"


@pytest.mark.parametrize(
    "body", [b"{", b"[]", b'{"text": ""}', b'{"text": 123}', b'{"text": "help", "extra": true}']
)
def test_bad_requests_are_client_errors(server_url, body):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request(server_url, body))
    assert error.value.code == 400


def test_oversized_request_rejected(server_url):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request(server_url, b"x" * 20001))
    assert error.value.code == 413


def test_browser_cross_origin_post_rejected(server_url):
    req = request(server_url, b'{"text":"WiFi disconnects"}')
    req.add_header("Origin", "https://example.invalid")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req)
    assert error.value.code == 403
