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

- Full pytest suite on the locked Python 3.11 environment: 118 tests. Tests cover arbitrary
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

The comparative CUDA recipe installed and ran on a fresh free Colab Tesla T4.
It trained for two full epochs, saved/reloaded the checkpoint, scored all six
methods on 524 identical test rows and returned exit 0. Complete aggregate report
and protocol/source fingerprints were captured and verified locally:
`reports/comparison-colab-it.json`. The source of the fully retained run is
`4ac77a0309bab8e35cda9cb1abdf0c3bd254f87e`.

The complete aggregate report is durable in this repository. The experimental
checkpoint was saved/reloaded and a 214,143,248-byte archive was built in Colab,
but its browser download failed when the connection disappeared. No durable local
checkpoint or downloadable trained-model release is claimed. Reproduce the pinned
notebook and export its `qwen-checkpoint` directory before the temporary runtime ends.

Real execution found a pinned Unsloth API integration issue: returned probabilities
are rounded to four decimals and can sum to 0.9999. The adapter now normalizes only
the documented per-option rounding envelope. Tests reject invalid mass, arbitrary
precision, booleans and nonidentity/incomplete checkpoint resumption. Core validators
remain strict. The original successful GPU console run is separately retained as
an incomplete observation after the transient runtime lost its unexported files.

A separate full production-calibration run saved/reloaded its head and passed real
Uvicorn HTTP evaluation on all 524 rows, plus batch/auth/input/offline-parity checks.
Its original quality gates passed on actual HTTP outputs; see `docs/GPU_HTTP.md`.
The model export was interrupted by GPU quota, leaving only a complete recovered
LoRA adapter and an incomplete head. The durable Drive notebook is prepared but
its Drive/GPU execution remains unqualified.

These results condition on these recipes and reused holdout. Lasting CUDA API
deployment, other GPUs and seed/language/organization robustness remain unqualified. Fake/structural tests do not establish
GPU correctness by themselves. The provided Docker image
qualifies only CPU inference, not Qwen serving.

No claim of Russian accuracy, broad OOD detection, prompt-injection immunity,
current spam coverage, multi-tenant isolation or production field performance.
Exact-feature grouping is not semantic/temporal/organization holdout independence.
Registry hashes are integrity checks, not signatures against a writer controlling
all files. Repeated test-set use for human experiment selection needs a fresh final
holdout before consequential accuracy claims.

Hosted CI is attached to a concrete GitHub commit and is verified separately after
publication; having a workflow file alone is not a successful hosted execution.
