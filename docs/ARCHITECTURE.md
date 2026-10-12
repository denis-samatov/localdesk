# Architecture and contracts

## Scope

A single-node text decision service: 2–200 named choices, one label per input,
CSV/JSONL data, a shared fit/calibrate/evaluate lifecycle, immutable versions and
an inference API. Training is a separate operator process; serving has no job
execution or upload capabilities. A local filesystem registry is deliberate:
there is no multi-tenant control plane or distributed training queue to justify
a database/broker. Add those only with an actual multi-host/multi-tenant requirement.

## Ownership

| Module | Responsibility |
| --- | --- |
| `config.py` | Strict JSON, TaskSpec/Experiment validation and task digest |
| `dataset.py` | Format/column adaptation, transitive groups, split invariants |
| `data.py` | Original Zenodo adapter and shared feature-group signatures |
| `pipeline.py` | Train/calibrate/test lifecycle and portable CPU export |
| `tfidf.py` | Default CPU backend: hashed word+char TF-IDF, stdlib-only inference |
| `analysis.py` | Risk-coverage curves and out-of-scope behaviour |
| `sources.py` | Pinned public downloads and cross-split duplicate isolation |
| `training.py` | Optional Qwen backend using the same prepared TaskSpec/splits |
| `metrics.py` | Classification, calibration and abstention metrics |
| `registry.py` | Immutable artifacts, integrity, qualification and active pointer |
| `core.py` | Probability contract and configured review policy |
| `api.py` | Typed ASGI boundary, auth, bounds, readiness and observability |
| `cli.py` | Operator workflows, legacy CLI compatibility |

The old IT constants in `schema.py` belong only to the original adapter and
compatibility baseline. New model artifacts carry their TaskSpec, which owns the
labels used by core decisions, evaluation and both inference backends.

## Configuration

Task IDs and versions are bounded safe identifiers; label keys and descriptions
are validated strings. Canonical task hashes include instructions, labels and
review policy. Dataset column mappings are explicit; nested JSON paths are read
as dictionary keys, never evaluated as expressions. Unknown configuration fields
and irrelevant backend hyperparameters are rejected. Source checksums may be
required by the experiment. Missing checksums mean the run records the observed
bytes, not that it verified a publisher's claim.

## Data isolation

Equivalent feature inputs merge with declared conversation IDs transitively.
Mixed labels in a conversation are valid; conflicting labels for equivalent
inputs are rejected. Equivalent copies collapse, preserving original texts and
human labels. The shared signature conservatively covers NFKC, PII-placeholder
removal, tokenization, SHA-256 feature hashing and normalized values rounded to
12 decimals. It does not establish semantic or temporal independence.

Explicit split groups must never intersect. Automatic single-label groups use
deterministic per-class stratification. Mixed conversations use bounded seeded
allocation with mandatory class coverage; impossible/unresolved constraints
fail with a request for explicit isolated splits. Training, calibration and test
must each support every configured class. Rare classes are not fabricated,
silently omitted or renamed.

## Training and qualification

Only train data fits model weights. Only calibration data fits temperature;
review thresholds and gates are declared in configuration, not tuned on test.
The CPU export handles sklearn's binary logit and multiclass matrices separately;
calibration logits/probabilities must match the portable scorer before publication.
The test split is evaluated after fitting/calibration. Repeated experiments on the
same test set can still bias human model selection: reserve an independent final
holdout for consequential deployment claims.

An absent gate set creates an unqualified candidate. Each declared gate records
its threshold, actual value and outcome. Null accepted accuracy fails a gate
requiring it. Artifact loading and promotion recompute gate outcomes from the
checksummed report. A failed candidate never replaces an active version.

## Artifact lifecycle

`reserve(run_id)` uses atomic directory creation. Completed IDs and concurrent
reservations are refused. Files and hashes are finalized in staging, then the
completed directory is renamed into the registry on the same filesystem.
Unfinished staging directories cannot be loaded. A failure retains its error
type/status, excluding private exception strings and texts.

Load verifies completion, safe paths, file hashes, TaskSpec identity, model/data/
report fingerprints and quality results. Hashes protect against accidental
corruption/mixing; they are not digital signatures against an attacker with
write access to all registry files. Operators must protect registry writes.

`active.json` contains the run ID, task hash and manifest hash and is atomically
replaced after verification. Concurrent promotions use last completed write wins;
there is one active model per registry. Workers retain an immutable in-memory
snapshot and must restart to adopt promotion/rollback. Serving never reloads
half-written files during a request.

## API operation

JSON-only requests, strict field types, bounded text/batch/body sizes and a body
deadline. API keys are checked in ASGI middleware, independent of CLI. Anonymous
mode requires explicit selection and checks loopback peer, Host and Origin.
No permissive CORS or external dashboard assets. Inference capacity rejects busy
requests instead of accumulating an unbounded model queue.

Liveness distinguishes process presence from readiness of a qualified loaded
model. Each response carries a generated request ID; predictions also identify
task/model. Metrics count predictions, reviews, errors and inference seconds
without recording bodies. Training data, secrets and original text are excluded
from telemetry. Deployment authentication/TLS and infrastructure belong to the
operator; Docker binds host ports to loopback by default.
