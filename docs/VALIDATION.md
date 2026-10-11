# Validation evidence — configurable service v0.2

## Real configuration reuse

The same pipeline/CLI executed two configurations with different files and labels:

| Version | Task / labels | Train / cal / test | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| it-v2 | IT support, 6 source labels | 690 / 170 / 524 | 0.694656 | 0.649195 |
| sms-v1 | SMS, ham/spam | 3334 / 768 / 1024 | 0.974609 | 0.938196 |

Both are actually fitted CPU logistic models. Temperature was fitted on calibration
only; class mapping/export logits and uncalibrated probabilities matched sklearn.
The same importer, fitter, evaluator and registry handled both tasks. Configured
gates passed; no task-specific core branches or invented gold labels were added.

`verify_service_reuse.py` then loaded/promoted each real exported artifact and
called the actual ASGI application with a held-out source request, confirming
readiness, returned option keys, version and report agreement. It restored the
previous pointer. Proof/report: `reports/service-reuse.json`. This is a functional
contract check, not load testing or a field accuracy trial.

## Executed software checks

- Full pytest suite on the locked Python 3.11 environment: 95 tests. Tests cover arbitrary
  labels, binary/multiclass export parity, strict config/types, group transitivity,
  mixed conversations, conflicting/equivalent inputs, split leakage, corrupted
  artifact hashes, model/data/report identity, failed/ungated candidates, partial
  Qwen checkpoint rejection and active-pointer preservation, smoke exclusion,
  streamed hashing, API/auth/origin/body/batch/readiness/metrics and legacy paths.
- Ruff lint and format checks passed. Wheel/sdist builds passed.
- Both original public source adapters verify pinned SHA-256 source bytes.
- CPU Docker image built; an unprivileged container with read-only filesystem and
  model mount reached healthy/ready and returned actual predictions on loopback.
  No other user's containers were changed. The operator dashboard displayed
  active labels, measured per-class quality/gates and the concrete training plan.
- Independent review identified and helped resolve mixed-group allocation,
  report/model dataset identity, anonymous host/origin protection, and completeness
  of pinned Qwen head/adapter/tokenizer/serving-config files. The same regressions
  guard against recurrence.

Raw corpora, checkpoints, registry/private paths and secrets are excluded from Git.
The packaged legacy CPU baseline was regenerated after its source changed and its
report remains bound to model/data/recipe hashes.

## Boundaries

GPU training, Qwen checkpoint runtime reload, GPU memory/latency and CUDA dependency
installation remain unexecuted. Fake/structural checkpoint tests do not establish
GPU correctness. The pinned CUDA recipe and notebook remain experimental; notebook
Python syntax is checked, not a completed Colab run. The provided Docker image
qualifies only CPU inference, not Qwen serving.

No claim of Russian accuracy, broad OOD detection, prompt-injection immunity,
current spam coverage, multi-tenant isolation or production field performance.
Exact-feature grouping is not semantic/temporal/organization holdout independence.
Registry hashes are integrity checks, not signatures against a writer controlling
all files. Repeated test-set use for human experiment selection needs a fresh final
holdout before consequential accuracy claims.

Hosted CI is attached to a concrete GitHub commit and is verified separately after
publication; having a workflow file alone is not a successful hosted execution.
