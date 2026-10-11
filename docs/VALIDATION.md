# Validation evidence — 2026-10-11

The published result is a **CPU logistic-regression baseline trained on real
public IT tickets**. It is not a Qwen model or a GPU training result.

## Executed locally

- Source download: four CSVs from Zenodo 7648117. Fixed SHA-256 hashes match.
  The record's metadata license is CC BY 4.0.
- `python scripts/prepare_data.py`: 690 train / 170 calibration / 524 test / 2 OOD.
  Source IDs join correctly; there are 33 raw normalized cross-split groups,
  43 cross-split groups under actual baseline preprocessing, and two groups
  with contradictory labels. Prepared groups do not cross any split.
- `python scripts/train_baseline.py`: weights fitted on training only; temperature
  0.60 fitted on calibration only. Exported stdlib logits match scikit-learn
  calibration logits at `atol=1e-10, rtol=1e-10`.
- Cleaned test: 69.4656% category accuracy; 231/524 route proposals (44.0840%
  coverage); 211/231 correct proposals (91.3420%). Per-class support/recall,
  NLL, Brier and ECE are recorded in `reports/baseline.json`.
- `pytest -q`: 57 passed on Python 3.11.15. Tests cover text validation,
  distributions, policy boundaries, duplicate/conflicting-label handling,
  feature equivalence, dataset tampering, model/recipe hashes, saved-model
  adapter errors, loopback HTTP requests and notebook Python syntax.
- `ruff check .`: passed. `ruff format --check .`: passed.
- `python -m build`: wheel and sdist built successfully.
- Browser interaction: local UI produced a Fileservice route for the shared-folder
  example and a review decision for a low-confidence password example. The
  full-page screenshot is `docs/demo.png`.
- Separate review reproduced and helped resolve preprocessing leakage and a GPU
  import-order bug. Reviewer then independently confirmed zero feature-vector
  overlap between all pairs of splits and the final baseline coverage/accuracy.

The local commands ran from the repository checkout with its development
virtual environment. CPU fitting used Python 3.13.5, scikit-learn 1.6.0,
NumPy 2.2.6 and SciPy 1.14.1. Model/data/recipe hashes bind the baseline report
to the shipped artifact and source; raw ticket text is not published in Git.

## Not established

- Qwen GPU training, CUDA dependency installation, GPU memory consumption,
  throughput, convergence and real saved-Qwen checkpoint reloading.
  The optional Linux/CUDA dependency-resolution attempt was interrupted during
  upstream Git fetching; it is not a successful resolver check. Direct revisions
  are inspected and pinned, but the GPU recipe remains experimental.
- Fake `FastDecisionModel` boundary tests establish adapter/orchestration
  contracts only. Notebook syntax checks are not executed notebook cells.
- Russian accuracy, general out-of-domain detection, robustness to arbitrary
  prompt injection, temporal generalization or field performance in an organization.
- The two EOL challenge texts receive review; two templates are insufficient to
  qualify an OOD detector.

Hosted CI status belongs to the exact GitHub commit and must be read from
GitHub Actions. The workflow definition itself is not evidence that hosted jobs ran.
