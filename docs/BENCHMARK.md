# Comparative benchmark

The experiment tests whether fine-tuned Qwen improves over a strong, cheap text
classifier, and whether training improves over ordinary zero-shot Qwen. It does
not presume that the larger model wins. The same Experiment/TaskSpec supports
new CSV/JSONL datasets and 2–200 labels without changing benchmark code.

## Reproduce

```bash
python -m pip install '.[baseline]'
python scripts/prepare_data.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  localdesk benchmark experiments/it-support.json --out artifacts/benchmarks/it-new
python scripts/prepare_sms.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  localdesk benchmark experiments/sms-filter.json --out artifacts/benchmarks/sms-new
```

On Windows, set the thread environment variables in PowerShell before running
the same CLI. Output directories are immutable: choose a new one for each run.
Benchmarking does not promote models or replace the active service version.

For the full CUDA comparison, install `requirements-gpu.txt` and append `--qwen`.
The clean [Colab notebook](../notebooks/benchmark_qwen.ipynb) runs the same CLI.
Free T4 allocation is subject to Google's quota; no paid resources are requested.
Exit 2 means at least one GPU stage is missing/blocked/failed. CPU-only execution
exits 0 while explicitly retaining an incomplete Qwen conclusion. Setup/CPU data
errors stop execution. GPU diagnostics are local; never commit them unreviewed.

Resume evaluation after a runtime/integration failure using a completed benchmark
checkpoint, without repeating training:

```bash
localdesk benchmark experiments/it-support.json --out artifacts/benchmarks/recovered \
  --qwen --checkpoint artifacts/benchmarks/original/qwen-checkpoint
```

This repeats the CPU and ordinary-LM evaluations and reloads the existing trained
head. Task/data/seed/resolved parameters/pinned base and upstream identities must
match its original protocol. Full epoch completion, non-smoke status and identity
serving temperatures are required. The new protocol records origin and checkpoint
file hashes separately from the evaluation source fingerprint.

## Frozen protocol and data separation

1. Validate the configured data using the service's existing intake. Keep the
   original train/test sets unchanged. Partition original calibration **groups**
   into tuning and probability/policy calibration. Every label needs at least
   two independent calibration groups. Fail when support is insufficient.
2. Write `protocol.json` with task/data/split/code hashes, partition hashes,
   candidate grid, resolved GPU parameters, pinned base/upstream revisions,
   contexts, zero-shot option order, error target and bootstrap recipe.
3. Fit vocabulary/weights on **train** only. Fit word (1–2 gram) and character
   (3–5 gram, char_wb) TF-IDF, up to 40,000 features each. Independently select
   C from 0.25/1/4/16 for balanced logistic regression and linear SVM using
   **tuning macro-F1**. Choose the stronger TF-IDF method on tuning, record it
   in `reference.json`, then score test. Ties prefer the first grid/method.
4. Fit temperature on the separate calibration portion using NLL and a frozen
   grid 0.5–4.0, step 0.05. Underflowed probabilities are floored at 1e-15 before
   log scaling. Store raw and calibrated test metrics to expose calibration costs.
5. Choose the confidence threshold with maximum calibration coverage whose
   descriptive Wilson upper error bound is at most the declared target
   (default 5%). Equal-confidence decisions stay together. If no threshold has
   enough evidence, accept none; accuracy/error/upper bound are **null**.
6. Evaluate the frozen policy on test and report coverage/error/Wilson bound.
   The calibration bound was used in threshold search and is **not** a validated
   error guarantee. The test bound assumes IID rows; multiple rows from a shared
   conversation disable `target_validated_on_test`. Singleton groups do not prove
   that unrelated organization/time correlations are absent. Report the original
   task's fixed confidence+margin policy separately.

The majority reference predicts the most frequent **training** class, with
training class-frequency scores. It does not inspect test support to choose a
label. Hashed logistic uses the service's fixed C=4 recipe with independently
recalibrated benchmark probabilities. The benchmark does not rewrite its shipped
weights. SVM softmax margins are explicitly uncalibrated scores, not probabilities
of correctness. Temperature calibration does not ensure perfect calibration.

## Qwen methods

`qwen_zero_shot` loads the pinned ordinary language model and provides instructions,
label keys/descriptions and input in its chat template. Thinking is disabled. It
normalizes the next-token logits over unique, single-token numeric answer IDs.
One prefill scores all options; it does not generate a rationale. These conditional
answer probabilities are not unconditional confidence. Fixed option ordering and
prompt choice may influence quality; no prompt tuning uses test labels.

The pinned Unsloth Decision API formats probabilities to four decimals. The model
adapter normalizes finite, correctly keyed four-decimal outputs only within the
per-option rounding error envelope (0.00005 per option). Other invalid masses,
booleans, malformed keys and non-finite values remain errors; core/API distribution
validation stays strict. Probability metrics therefore include presentation
quantization as well as model uncertainty.

`qwen_decision` uses the existing Unsloth LoRA/decision-head trainer on train only,
with default two epochs/rank 16/batch 2/accumulation 4/learning rate 2e-4. CPU-only
training fields are excluded. Benchmark parameters are frozen before evaluation.
Smoke `max_steps` is forbidden. The head is saved at temperature 1, then loaded
from its checkpoint before applying the common calibration procedure. Skipped or
truncated training/evaluation examples fail. Plain-LM context is 8192; decision
training context is 2048; different encodings/context limits remain a comparison
limitation, not evidence of equivalence. Exact model names prevent Unsloth mirror
remapping from silently discarding the requested model revision.

## Quality and resource evidence

All completed methods report accuracy, macro/weighted F1, per-class metrics,
confusion, ECE/NLL/Brier, raw-vs-calibrated metrics, fixed-policy coverage and
target-policy coverage/error. Primary comparison is trained Qwen minus the
**tuning-selected** TF-IDF reference. A quality-gain verdict requires positive
lower 95% paired bootstrap bounds for **both accuracy and macro-F1**. Samples
resample entire test groups with identical paired indices, fixed class vocabulary
and F1=0 for absent classes. Default 2000 replicates, fixed seed. Other comparisons
are secondary and exploratory; no familywise multiple-testing claim is made.
Missing Qwen results leave the primary conclusion incomplete.
Intervals condition on the fitted models and one training seed; they do not
estimate variation across training seeds or new organizations. Benchmark Qwen
checkpoints are raw T=1 experimental heads, not qualified registry service
versions; calibrated probability metrics use the temperature stored in the report.

Timing is warmed, sequential batch=1, local backend inference and distribution
validation. CUDA work is synchronized. It includes tokenization/vectorization,
but excludes HTTP, installation, download and initial warmup; it is not a load
or end-to-end service benchmark. Report hardware and thread settings. A CPU Mac
run must not be used as a latency comparator against a separate Colab GPU run;
run all methods together on the target machine for the resource comparison.

Process peak RSS is measured for the whole benchmark, cumulatively across methods
(and is unavailable on Windows). Per-GPU-stage allocated/reserved VRAM peaks
include loading/training/inference. Neither quantity is a model's pure serving
footprint. Model-size, energy and monetary/time-saving advantages are unproven.

Test IDs are hashed in local prediction files; texts/vocabularies are never
included in public reports. Task/data/code hashes bind reports to the precise
inputs. Existing public holdouts were previously evaluated and are reused here;
these are reproducible corpus comparisons, **not a new independent blind holdout**,
Russian qualification, new-organization qualification or a production trial.

## Public summaries

Generate aggregate JSON/Markdown from an executed run:

```bash
python scripts/publish_benchmark_report.py artifacts/benchmarks/it-new reports/comparison-it
```

This copies only report/protocol aggregates. It does not publish per-row predictions,
checkpoint weights, private GPU logs, datasets or learned vocabularies.

Primary API references: [sklearn text features](https://scikit-learn.org/stable/modules/feature_extraction.html),
[probability calibration](https://scikit-learn.org/stable/modules/calibration.html),
[pinned Unsloth decision loader](https://github.com/unslothai/unsloth/blob/62354edbf4fd246cf120ef07272934fb405ec705/unsloth/models/decision.py),
[pinned model loader](https://github.com/unslothai/unsloth/blob/62354edbf4fd246cf120ef07272934fb405ec705/unsloth/models/loader.py).
