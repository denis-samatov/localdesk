# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-10-13

### Added
- Ready Docker image on GitHub Container Registry (`ghcr.io/denis-samatov/localdesk`,
  `linux/amd64` + `linux/arm64`) that serves the demo model out of the box; mount
  your own registry over `/models`. Built and smoke-tested by `.github/workflows/docker.yml`.
- `scripts/train_demo.py` also exports the demo as a complete, promoted registry
  version (`docker/demo-registry/`), checked by the test suite.
- Multi-seed CPU report (`scripts/publish_seed_suite.py`, `reports/seed-suite.md`):
  five train/calibration re-splits per dataset with fixed test sets.
- English dashboards with a Russian toggle (button or `?lang=ru`) for both
  `localdesk api` and `localdesk serve`.

### Changed
- Release dry runs publish a unique `<version>.devN` to TestPyPI and install it
  back to verify; the `testpypi` GitHub environment exists.
- README: Docker quick start, seed robustness, out-of-scope review share corrected
  to about 84% (83.6 ± 1.4% over five seeds) from the single-split 85%.

### Fixed
- `localdesk serve` returned 404 for `/` with a query string.

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

[0.3.1]: https://github.com/denis-samatov/localdesk/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/denis-samatov/localdesk/compare/52a3f7c...v0.3.0
[0.2.0]: https://github.com/denis-samatov/localdesk/commit/52a3f7c
