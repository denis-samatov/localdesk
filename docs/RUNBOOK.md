# Service runbook and API

## Setup and a healthy version

Install `.[baseline,service]`, prepare a licensed source, validate the experiment,
train into a new ID, inspect the report/gates, then promote. No existing completed
ID is overwritten. `localdesk runs` lists candidates and qualified versions;
a damaged or foreign directory is listed with `"valid": false` and its error.
A failed fit keeps its failure record under `.failed-RUN-<timestamp>-<id>/status.json`,
cannot be activated, and releases the run ID so the same ID can be retried.
A process killed mid-training leaves `.pending-RUN`; remove it once no training
process is running. Do not edit completed artifacts.

For local use: `localdesk api --anonymous-local` on 127.0.0.1. Open `/` and click
connect without a key. For authenticated mode export `LOCALDESK_API_KEY` or set
it through your process manager; the dashboard key field holds it only in memory.
The direct `uvicorn localdesk.api:application --factory` entrypoint requires this
key and reads `LOCALDESK_REGISTRY` (default artifacts/registry).

## HTTP contract

OpenAPI JSON: `GET /openapi.json` with X-API-Key. The dashboard uses local assets;
it does not load an external Swagger CDN. All business endpoints require the
key unless explicitly running in guarded anonymous loopback mode.

| Endpoint | Behavior |
| --- | --- |
| GET / | Operator dashboard shell; no private metadata embedded |
| GET /health/live | Process liveness, no key required |
| GET /health/ready | 200 for qualified loaded model; 503 otherwise |
| GET /v1/model | Active task, labels, version, task hash, gates |
| GET /v1/model/report | Verified active model quality report |
| GET /v1/model/plan | Verified active training plan |
| POST /v1/predictions | `{"text":"..."}` -> decision and model/task identity |
| POST /v1/predictions/batch | `{"texts":["...", "..."]}` -> ordered predictions |
| GET /metrics | Prometheus text counters, no request bodies |

Example (API_KEY is a shell variable set by the operator):

```bash
curl http://127.0.0.1:8765/v1/predictions \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"text":"File share access request"}'
```

Input: strict strings, 1–4000 characters after stripping; batches 1–16; JSON
transport body at most 128 KiB; bounded body deadline. Unknown fields are invalid.
Errors share `{"error":{"code":"...","message":"...","request_id":"..."}}`.
401 auth, 403 invalid anonymous origin/host, 413 transport size, 415 content type,
422 input structure, 503 unavailable/busy/invalid model. Inference runs one request
at a time; up to 32 further requests wait up to 30 s, beyond that `MODEL_BUSY` is
returned. Retry busy requests later;
inference has no externally consequential side effects. No source inputs or
stack traces are reflected. Each response has X-Request-ID and Cache-Control:no-store.

## Promotion and rollback

```bash
localdesk promote RUN --registry artifacts/registry
```

This verifies task, data/model/report identity and quality before replacing the
active pointer. It does not hot-swap a running worker. Restart after promoting
or rolling back; check `/health/ready` and model_version. Each registry has one
active model; use separate registries/processes for simultaneously served tasks.
Protect registry writes and mount it read-only in inference containers.

## Docker

The image supports CPU inference. Train outside it and promote first.
Set LOCALDESK_API_KEY in your environment (compose refuses a missing key):

```bash
docker compose up --build -d
docker compose ps
```

Compose binds to 127.0.0.1, runs as an unprivileged user, mounts models read-only,
uses a read-only filesystem/tmpfs and drops Linux capabilities. The image includes
a readiness healthcheck. Remote exposure requires the operator's authenticated
HTTPS proxy and network policy; no remote deployment is performed by this repo.
The optional Qwen backend needs a CUDA image/runtime and cached base weights;
the provided CPU image does not qualify GPU serving.

## Observability and limitations

Watch readiness, model_version, prediction/review/error counters and accumulated
inference seconds. These are not ground-truth accuracy measurements. Collect
independently reviewed outcomes before changing operational policy. Request-body
logging and telemetry are disabled; raw training data remain outside Git.
No distributed queue, web training job system, tenant isolation or automatic
production promotion is included in this single-node service.
