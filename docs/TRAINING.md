# Qwen training path

This path uses the real dataset documented in `data/README.md` and the public
Unsloth API inspected at the revisions below. The full comparative IT recipe **was executed on a free Colab Tesla T4**:
two epochs, saved/reloaded checkpoint, shared heldout temperature calibration and
all 524 test rows. See [the complete report](../reports/comparison-colab-it.md).
The original production-calibration/registry-serving path remains experimental;
it was not deployed as a CUDA API. Notebook syntax and fake tests alone cannot
qualify a new GPU configuration, dataset or language.

## Pinned direct sources

| Component | Revision |
| --- | --- |
| Unsloth | 62354edbf4fd246cf120ef07272934fb405ec705 |
| Unsloth Zoo | ac1410cc766efedd5e4ced1b1d1338138eb7eaa5 |
| unsloth/Qwen3.5-4B | 3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636 |

The experimental Linux/CUDA recipe in `requirements-gpu.txt` also specifies
PyTorch 2.8.0, torchvision 0.23.0 and Transformers 5.2.0. Source packages are
installed directly from the pinned upstream Git revisions. The trainer checks
installed `direct_url.json` metadata and refuses mismatched revisions.
This is not a complete transitive dependency lock; each configurable training run records its relevant package versions and pinned
source revisions. The legacy standalone trainer also records `pip freeze`. Do not substitute the
CPU `uv.lock` for a CUDA environment lock.

## Configurable service backend

Use the same Experiment and TaskSpec as the CPU baseline; set `training.backend`
to `unsloth` and declare `epochs`, `rank`, `batch_size`, `gradient_accumulation`
and `learning_rate`. Run `localdesk train experiments/it-qwen.json --run-id qwen-v1`.
The version contains checkpoint/, task/data/plan/report and checksummed manifest.
Changing labels or text files requires configuration only. Smoke max_steps runs
remain unqualified even if their measured gates pass. Promotion validates the
pinned adapter/head/tokenizer layout without loading GPU weights. Serving loads
GPU weights separately and requires the pinned base model cached locally.

The original notebook now creates this experiment and runs the shared pipeline.

## Colab

Open `notebooks/train_qwen.ipynb` using its README Colab link. Choose a fresh GPU
runtime; availability and free quota are controlled by Google. The notebook:

1. Clones the project and installs the optional GPU recipe.
2. Downloads four source CSVs and verifies their SHA-256 checksums.
3. Reproduces the cleaned train/calibration/test split.
4. Fits a decision head with LoRA, two epochs, rank 16, learning rate 2e-4,
   batch size 2, gradient accumulation 4, maximum training length 2,048.
5. Fits probability calibration on the 170 calibration examples only.
6. Evaluates on the 524 test examples and saves adapters, decision head and
   calibration, plus a LocalDesk provenance manifest.
7. Checks saved-checkpoint inference in a separate offline command and offers
   a ZIP download of the adapter.

Change `REPO_REF` from main to a reviewed project commit to freeze your run.
Use a new output directory for every run. `--max-steps` produces a smoke run and
is recorded as such; it is not evidence for a fully trained model.

BF16-capable GPUs use bfloat16. Other CUDA devices (including T4) use float32;
pure float16 is avoided for this Qwen recipe, following the upstream notebook's
reported numerical issue. Batch size, sequence length, precision and kernels
all affect memory use. This project does not promise 8 GB VRAM or a runtime.

## Linux with CUDA

Use a dedicated Python 3.11/3.12 environment following the upstream CUDA install
requirements, then:

```bash
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
python scripts/prepare_data.py
python -m localdesk.training --out artifacts/qwen-localdesk
```

If package installation changes PyTorch or Transformers unexpectedly, check the
resolved environment against these pins before training. The recipe is optional;
default CPU installation does not install Unsloth or PyTorch.

## Local inference

The saved LoRA adapter is not a standalone base model. Cache the pinned base
weights on the target machine before offline use; downloading them is a separate
setup step. The inference backend uses `local_files_only=True` and sets
HF_HUB_OFFLINE=1 / HF_HUB_DISABLE_TELEMETRY=1. Missing cached weights are an error.
Keep the checkpoint manifest with its schema, calibration and dataset provenance.

```bash
localdesk predict --backend unsloth --model artifacts/qwen-localdesk "File share access request"
localdesk evaluate --backend unsloth --model artifacts/qwen-localdesk
localdesk serve --backend unsloth --model artifacts/qwen-localdesk
```

The baseline and Qwen share option keys and review policy. Compare their own
results on the same data hashes. Do not tune either model on test results.
Evaluate unsupported classes on an independently collected OOD set; the two EOL
texts are a pipeline challenge, not a sufficient detector benchmark. Validate
Russian and other target languages separately before using them operationally.

Upstream sources:
- https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
- https://github.com/unslothai/unsloth/blob/62354edbf4fd246cf120ef07272934fb405ec705/unsloth/models/decision.py
- https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb
