"""Typed ASGI inference boundary, independent of the CLI and training processes."""

import asyncio
import hmac
import ipaddress
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from localdesk.core import decide
from localdesk.registry import Registry

MAX_BODY = 131072
# Inference is serialized (one model/GPU); excess requests wait in a bounded queue.
MAX_QUEUED = 32
QUEUE_TIMEOUT = 30.0
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class PredictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Text


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    texts: Annotated[list[Text], Field(min_length=1, max_length=16)]


class Prediction(BaseModel):
    category: str
    action: Literal["route", "review"]
    probabilities: dict[str, dict[str, float]]
    review_reasons: list[str]
    backend: str
    probability_kind: str
    policy: dict[str, float]
    model_version: str
    task_sha256: str


class BatchOutput(BaseModel):
    predictions: list[Prediction]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error(code: str, message: str, status: int, request_id: str = ""):
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": request_id}}, status_code=status
    )


class Boundary:
    def __init__(self, app, api_key: str | None, anonymous_local: bool):
        self.app, self.api_key, self.anonymous_local = app, api_key, anonymous_local

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope["state"] = {**scope.get("state", {}), "request_id": request_id}
        headers = {k.lower(): v for k, v in scope["headers"]}

        async def response_send(message):
            if message["type"] == "http.response.start":
                message["headers"].extend(
                    [(b"x-request-id", request_id.encode()), (b"cache-control", b"no-store")]
                )
            await send(message)

        if self.anonymous_local and not self.api_key:
            host = headers.get(b"host", b"").decode("ascii", errors="replace")
            try:
                hostname = urlsplit("http://" + host).hostname
                local_host = (
                    hostname == "localhost" or ipaddress.ip_address(hostname or "").is_loopback
                )
            except ValueError:
                local_host = False
            origin = headers.get(b"origin")
            local_origin = origin is None or origin.decode("ascii", errors="replace") in {
                "http://" + host,
                "https://" + host,
            }
            if not local_host or not local_origin:
                return await error(
                    "INVALID_ORIGIN",
                    "Anonymous access requires a local host and origin",
                    403,
                    request_id,
                )(scope, receive, response_send)

        if scope["path"] not in {"/", "/health/live", "/health/ready"}:
            if self.api_key:
                allowed = hmac.compare_digest(headers.get(b"x-api-key", b""), self.api_key.encode())
            else:
                try:
                    peer = scope.get("client", ("", 0))[0]
                    allowed = self.anonymous_local and ipaddress.ip_address(peer).is_loopback
                except ValueError:
                    allowed = False
            if not allowed:
                return await error("UNAUTHORIZED", "Valid API key required", 401, request_id)(
                    scope, receive, response_send
                )
        # Buffer only up to a fixed transport bound, including chunked requests.
        if scope["method"] in {"POST", "PUT", "PATCH"}:
            content_type = headers.get(b"content-type", b"").split(b";")[0].strip().lower()
            if content_type != b"application/json":
                return await error("CONTENT_TYPE", "Expected application/json", 415, request_id)(
                    scope, receive, response_send
                )
            chunks, size = [], 0
            deadline = time.monotonic() + 10
            while True:
                try:
                    message = await asyncio.wait_for(
                        receive(), timeout=max(0, deadline - time.monotonic())
                    )
                except TimeoutError:
                    return await error("REQUEST_TIMEOUT", "Request body timeout", 408, request_id)(
                        scope, receive, response_send
                    )
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > MAX_BODY:
                    return await error(
                        "BODY_TOO_LARGE", "Request body exceeds limit", 413, request_id
                    )(scope, receive, response_send)
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            delivered = False

            async def buffered_receive():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

            return await self.app(scope, buffered_receive, response_send)
        return await self.app(scope, receive, response_send)


def create_app(*, registry: Path, api_key: str | None = None, anonymous_local: bool = False):
    api_key = api_key or os.environ.get("LOCALDESK_API_KEY")
    if not api_key and not anonymous_local:
        raise ValueError("API key required; anonymous loopback must be explicitly selected")
    store = Registry(registry)
    lock = threading.Lock()
    statistics = {"predictions": 0, "reviewed": 0, "failures": 0, "seconds": 0.0}

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.backend, app.state.manifest = store.load_active()
        except (ValueError, OSError, KeyError, ImportError, RuntimeError):
            app.state.backend = None
            app.state.manifest = None
        app.state.admission = asyncio.Semaphore(1)
        yield
        app.state.backend = None

    app = FastAPI(
        title="LocalDesk Decision Service",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        responses={
            status: {"model": ErrorResponse}
            for status in (401, 403, 404, 405, 408, 413, 415, 422, 500, 503)
        },
    )  # No external Swagger UI CDN in private sessions.
    app.add_middleware(Boundary, api_key=api_key, anonymous_local=anonymous_local)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exception):
        return error(
            "VALIDATION_ERROR",
            "Invalid request structure or field value",
            422,
            request.state.request_id,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, exception):
        return error(
            "HTTP_ERROR",
            "Requested operation is unavailable",
            exception.status_code,
            request.state.request_id,
        )

    @app.exception_handler(Exception)
    async def internal_error(request, exception):
        return error(
            "INTERNAL_ERROR", "Request could not be processed", 500, request.state.request_id
        )

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        from importlib.resources import files

        return HTMLResponse(
            files("localdesk").joinpath("assets/service.html").read_text(encoding="utf-8")
        )

    @app.get("/health/ready")
    def ready(request: Request):
        if app.state.backend is None:
            return error(
                "MODEL_UNAVAILABLE", "No qualified model loaded", 503, request.state.request_id
            )
        return {"status": "ready", "model_version": app.state.manifest["run_id"]}

    queued = 0

    def predict_many(texts):
        # The thread lock still guarantees serialized inference if a cancelled request's
        # worker thread outlives its admission slot.
        with lock:
            return run_inference(texts)

    def run_inference(texts):
        started = time.monotonic()
        try:
            predictions = [
                dict(
                    decide(text, app.state.backend),
                    model_version=app.state.manifest["run_id"],
                    task_sha256=app.state.manifest["task_sha256"],
                )
                for text in texts
            ]
            statistics["predictions"] += len(predictions)
            statistics["reviewed"] += sum(value["action"] == "review" for value in predictions)
            return predictions
        except Exception:
            statistics["failures"] += 1
            raise
        finally:
            statistics["seconds"] += time.monotonic() - started

    def busy(request):
        return error(
            "MODEL_BUSY", "Inference capacity is busy; retry later", 503, request.state.request_id
        )

    async def infer(texts, request):
        nonlocal queued
        if app.state.backend is None:
            return error(
                "MODEL_UNAVAILABLE", "No qualified model loaded", 503, request.state.request_id
            )
        if queued >= MAX_QUEUED:
            return busy(request)
        queued += 1
        try:
            async with asyncio.timeout(QUEUE_TIMEOUT):
                await app.state.admission.acquire()
        except TimeoutError:
            return busy(request)
        finally:
            queued -= 1
        try:
            return await run_in_threadpool(predict_many, texts)
        except Exception:
            return error(
                "MODEL_FAILURE",
                "Local model failed to return valid output",
                503,
                request.state.request_id,
            )
        finally:
            app.state.admission.release()

    @app.post("/v1/predictions", response_model=Prediction)
    @app.post("/api/decide", response_model=Prediction, include_in_schema=False)
    async def prediction(payload: PredictInput, request: Request):
        result = await infer([payload.text], request)
        return result if isinstance(result, JSONResponse) else result[0]

    @app.post("/v1/predictions/batch", response_model=BatchOutput)
    async def batch(payload: BatchInput, request: Request):
        result = await infer(payload.texts, request)
        return result if isinstance(result, JSONResponse) else {"predictions": result}

    @app.get("/v1/model")
    @app.get("/api/info", include_in_schema=False)
    def model_info(request: Request):
        if app.state.backend is None:
            return error(
                "MODEL_UNAVAILABLE", "No qualified model loaded", 503, request.state.request_id
            )
        return {
            "model_version": app.state.manifest["run_id"],
            "task": app.state.backend.task.to_dict(),
            "backend": app.state.backend.name,
            "task_sha256": app.state.manifest["task_sha256"],
            "quality": app.state.manifest["quality"],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return "".join(
            f"# TYPE localdesk_{name}_total counter\nlocaldesk_{name}_total {statistics[key]}\n"
            for name, key in (
                ("predictions", "predictions"),
                ("reviews", "reviewed"),
                ("model_failures", "failures"),
                ("inference_seconds", "seconds"),
            )
        )

    @app.get("/v1/model/report")
    def model_report(request: Request):
        if app.state.backend is None:
            return error(
                "MODEL_UNAVAILABLE", "No qualified model loaded", 503, request.state.request_id
            )
        from localdesk.config import read_json

        directory, _, _ = store.metadata(app.state.manifest["run_id"])
        return dict(
            read_json(directory / "report.json"), model_version=app.state.manifest["run_id"]
        )

    @app.get("/v1/model/plan")
    def model_plan(request: Request):
        if app.state.backend is None:
            return error(
                "MODEL_UNAVAILABLE", "No qualified model loaded", 503, request.state.request_id
            )
        from localdesk.config import read_json

        directory, _, _ = store.metadata(app.state.manifest["run_id"])
        return dict(
            read_json(directory / "training-plan.json"), model_version=app.state.manifest["run_id"]
        )

    return app


def application():
    """Uvicorn --factory entrypoint always requires an environment API key."""
    return create_app(registry=Path(os.environ.get("LOCALDESK_REGISTRY", "artifacts/registry")))
