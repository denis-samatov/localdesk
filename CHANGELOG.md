# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-10-12

First release on PyPI. The distribution is now named `localdesk`
(previously `localdesk-decisions`, never published).

### Added
- `tfidf` CPU backend: hashed word 1–2 grams and character 3–5 grams with
  sublinear TF-IDF and balanced logistic regression. Inference needs only the
  standard library; artifacts store hash buckets, not a training vocabulary.
  Shipped experiments use it.
- Packaged TF-IDF demo model, now the default for `localdesk predict`, `serve`
  and `evaluate` without `--model` (76.5% vs 69.5% accuracy on the IT holdout).
  `--backend baseline` keeps the previous model.
- Risk-coverage curves in every training report (coverage at 90/95/99%
  accuracy, AURC) and out-of-scope analysis (review share, AUROC).
- Banking77 and CLINC150 adapters with pinned sources and checksums, and
  `scripts/publish_cpu_suite.py` for the four-dataset comparison.
- Release workflow using PyPI Trusted Publishing; Python 3.12 in CI.

### Changed
- Concurrent API requests wait in a bounded queue (32 requests, 30 s) instead
  of failing immediately with `MODEL_BUSY`.
- A failed training run moves its record to `.failed-<id>-…` and frees the
  run ID for a retry.
- Optional dependencies use version ranges instead of exact pins; exact
  tested versions stay in `uv.lock`. Tested at both ends of each range.
- README rewritten around a quick start and measured results.

### Fixed
- A version that fails post-publish verification is withdrawn instead of
  remaining under a completed run ID.
- `localdesk runs` lists damaged registry entries as invalid instead of
  aborting.
- Dataset downloads read until EOF and retry truncated bodies.
- `/metrics` declares a type for every counter; JSONL errors name the line.

## [0.2.0] - 2026-10-11

Configurable service: task/experiment JSON, immutable registry with quality
gates, FastAPI inference API, operator dashboard, Qwen/Unsloth recipe and the
comparative benchmark. Not published to PyPI.

[0.3.0]: https://github.com/denis-samatov/localdesk/compare/52a3f7c...v0.3.0
[0.2.0]: https://github.com/denis-samatov/localdesk/commit/52a3f7c
