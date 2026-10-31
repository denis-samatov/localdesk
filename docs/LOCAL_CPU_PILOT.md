# Local CPU operational pilot

This drill starts only its own authenticated HTTP process on `127.0.0.1`, uses
public licensed IT tickets, and copies the bundled registry into a **new** output
directory. It does not deploy a service, contact model providers, use CUDA, or
establish customer/production performance. Read the
[frozen protocol](LOCAL_CPU_PILOT_PROTOCOL.md) before running it. Retain that
file's SHA-256 before measurement; do not tune acceptance after seeing results.

## Data and environment

The real public source is Leonardo Santiago Benitez Pereira's
[Classification of IT Support Tickets, DOI 10.5281/zenodo.7648117](https://zenodo.org/records/7648117),
licensed CC BY 4.0. Recheck the source record/license and retain the four CSV
checksums. Use only `X_train.csv`, `y_train.csv`, `X_test.csv`, and `y_test.csv`;
upstream Python, pickle, H5 and model files are unnecessary. See
[datasets](DATASETS.md) for attribution and cleaning boundaries. The author
describes source PII scrubbing; that is not an independent privacy audit.

Install the existing CPU/service/dev lock, then prepare the four source CSVs
downloaded from that record into a local source directory:

```bash
uv sync --frozen --extra baseline --extra service --extra dev
uv run --frozen --extra baseline --extra service --extra dev \
  python scripts/prepare_data.py --source /path/to/four-csvs --out data/zenodo
```

The checked-in recipe expects 690 training, 170 calibration and 524 test rows;
two EOL examples are a separate limited challenge. Raw-file fingerprints and
canonical model split fingerprints are different: the harness validates both
through the existing experiment/data preparation path and verified registry.
Equivalent-feature grouping does not establish semantic, temporal or
organization independence. This benchmark was reused previously, so quality
results are regression evidence, not a fresh field holdout.

## Run and inspect

Use a new output directory; the harness refuses to overwrite one:

```bash
uv run --frozen --extra baseline --extra service --extra dev \
  python scripts/qualify_local_cpu.py \
  --registry docker/demo-registry --data data/zenodo \
  --experiment experiments/it-support.json \
  --protocol docs/LOCAL_CPU_PILOT_PROTOCOL.md \
  --out /path/to/new-owned-pilot
```

No external credential is needed. The harness generates a temporary key, passes
it only in the owned child environment, disables HTTP proxies and service access
logs, and checks that the key is absent from process arguments and service logs.
It does not log ticket text. Ambient provider credentials and import-path
overrides are not forwarded. Optimized Python (`-O`/`PYTHONOPTIMIZE`) is rejected
because it would disable the acceptance assertions.

`result.json` retains machine/interpreter/package versions, protocol/data/model
hashes, model/task identities, exact request/status counts, latency nearest-rank
percentiles, throughput and sampled RSS (KiB on POSIX; unavailable elsewhere).
RSS samples are not peak-memory measurements. POSIX RSS/argument inspection uses
the system `ps` command. Each normal load phase sends 256 requests at concurrency
1, 4 or 16. Startup/request/cleanup timeouts bound
the harness; they are not availability or latency SLOs. Negative authentication
and validation requests are reported separately from normal load errors.
Measurements on a shared machine are observational: unrelated concurrent work,
warm caches, short phases and loopback transport prevent capacity estimates or
causal comparisons between concurrency levels. Do not extrapolate these rates.

All 524 held-out rows pass through actual HTTP and must match direct artifact
decisions and identity. Quality uses their public gold labels; predictions are
compared with the frozen model, not a test-time optimized alternative.

The drill gracefully stops/restarts owned processes, makes and verifies a
whole-registry backup with no writers, restores into a fresh directory, and
rejects an intentionally corrupted disposable copy. It fits a second real CPU
candidate using the same frozen task/data/hyperparameters and declared quality
gates, promotes and restarts it, then explicitly rolls back and restarts the
retained original version. Existing registry/model/data files are never edited.
After argument/output/environment setup, `result.json` is also written when a
drill fails; child processes are stopped in context-manager cleanup. Input or
environment setup errors can stop before that record exists. A new output directory is
required for a retry, preserving the failed evidence.

## Interpretation

Synthetic unit and boundary tests remain separate from this real-data HTTP
drill. A successful run qualifies these operations only on its recorded local
CPU profile. It does not establish an OS sandbox, tenant isolation, malicious
registry-writer resistance, Russian accuracy, broad OOD detection, target-host
recovery, field reliability or sustained availability. A future external pilot
requires its own host, operator, traffic/data permissions, representative fresh
outcomes and acceptance criteria agreed before measurement.

The [2026-10-31 local record](../reports/local-cpu-pilot-2026-10-31.json) includes
the actual machine profile, counts, quality, operational outcomes and measured
latencies. No raw ticket text, client credential or personal filesystem path is
included in that public aggregate.
