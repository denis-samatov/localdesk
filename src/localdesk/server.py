"""Loopback-only local demo; no ticket storage, external assets or request-body logging."""

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from localdesk.core import Backend, decide

MAX_BODY = 20_000


def make_server(backend: Backend, *, port: int = 8765) -> ThreadingHTTPServer:
    lock = threading.Lock()  # Serialize inference for one model/GPU.

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send(self, status, data, content_type="application/json; charset=utf-8", nonce=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            script = f"'nonce-{nonce}'" if nonce else "'none'"
            self.send_header(
                "Content-Security-Policy",
                f"default-src 'self'; script-src {script}; style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status, value):
            self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

        def _valid_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):
            if not self._valid_host():
                return self._json(403, {"error": "invalid local host"})
            path = urlsplit(self.path).path  # Ignore query strings such as ?lang=ru.
            if path == "/":
                from importlib.resources import files

                nonce = secrets.token_urlsafe(18)
                html = files("localdesk").joinpath("assets/index.html").read_text(encoding="utf-8")
                return self._send(
                    200,
                    html.replace("__NONCE__", nonce).encode(),
                    "text/html; charset=utf-8",
                    nonce,
                )
            if path == "/api/info":
                return self._json(
                    200, {"backend": backend.name, "probability_kind": backend.probability_kind}
                )
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._valid_host():
                return self._json(403, {"error": "invalid local host"})
            origin = self.headers.get("Origin")
            if origin and origin not in {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }:
                return self._json(403, {"error": "cross-origin requests are not allowed"})
            if self.path != "/api/decide":
                return self._json(404, {"error": "not found"})
            if self.headers.get_content_type() != "application/json":
                return self._json(415, {"error": "expected application/json"})
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("chunked requests are not supported")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    return self._json(413, {"error": "request body size is out of bounds"})
                self.connection.settimeout(10)
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("incomplete request body")
                payload = json.loads(body)
                if not isinstance(payload, dict) or set(payload) != {"text"}:
                    raise ValueError("expected an object with exactly one text field")
                from localdesk.core import validate_text

                text = validate_text(payload["text"])
            except (ValueError, UnicodeError, TimeoutError):
                return self._json(400, {"error": "invalid JSON request or text"})
            try:
                with lock:
                    result = decide(text, backend)
            except Exception:
                # A provider/contract error is a server failure, never a made-up decision.
                return self._json(503, {"error": "local model unavailable or invalid output"})
            return self._json(200, result)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
