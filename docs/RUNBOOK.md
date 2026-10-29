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

For a consistent backup, stop writers first, copy the whole registry (completed
versions and active pointer together) to a fresh location, then verify it by
loading its active model before restarting inference. Do not copy only the active
pointer or overwrite the live registry while a training/promotion process runs.
Restore to a new directory, compare the version/task identity and representative
predictions, mount read-only and restart. Retain the previous verified registry so
rollback is another explicit pointer/version selection and restart.

`python -m pytest tests/test_recovery_acceptance.py` performs a synthetic local
backup/restore/rollback drill and rejects a corrupted model. This is an offline
artifact-contract check; operators still need a target-machine restore/SLO drill.
Define domain acceptance thresholds before opening a fresh held-out set. Reused
public benchmarks are regression evidence, not an independent field estimate.

```bash
localdesk promote RUN --registry artifacts/registry
```

This verifies task, data/model/report identity and quality before replacing the
active pointer. It does not hot-swap a running worker. Restart after promoting
or rolling back; check `/health/ready` and model_version. Each registry has one
active model; use separate registries/processes for simultaneously served tasks.
Protect registry writes and mount it read-only in inference containers.

## Docker

The image supports CPU inference. A published image for `linux/amd64` and
`linux/arm64` ships the IT-tickets demo as a verified, promoted registry version
in `/models`, so it serves immediately:

```bash
docker run --rm -p 127.0.0.1:8765:8765 -e LOCALDESK_API_KEY=change-me \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true \
  ghcr.io/denis-samatov/localdesk:0.3.1
```

Mount your own registry over `/models` (read-only) to serve your versions instead;
the demo is then not visible. Tags: `<version>`, `<major>.<minor>`, `latest` from
releases, `edge` from manual builds. To build locally with your trained registry,
train outside the container and promote first.
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

## Experimental GPU dependency gate

`requirements-gpu.txt` now excludes the audited torch2.8/transformers5.2
versions: torch2.13/torchvision0.28 follow the upstream pairing, and transformers
requires >=5.10. This remains an experimental input recipe, not a qualified
CUDA environment. The pinned Unsloth sources must still resolve together with
those versions; do not override conflicts, use `--no-deps`, or silently fall back
to the vulnerable pins. A clean Linux/CUDA resolution, installed-package audit,
one short training/inference/checkpoint replay, and recorded GPU/driver
versions are required before using or advertising this backend. CPU tests
and CPU release qualification do not cover it.

Relevant upstream version pairs: <https://pytorch.org/get-started/previous-versions/>.
Audited advisory bounds include GHSA-rrmf-rvhw-rf47 (torch>=2.13) and
GHSA-xrqw-3rrv-vx5w (transformers>=5.10). These bounds do not replace a new
whole-environment advisory scan.
