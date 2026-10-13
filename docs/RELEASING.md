# Releasing

Releases are built and uploaded by `.github/workflows/release.yml` with PyPI
Trusted Publishing, so no API token is stored anywhere.

## One-time setup (maintainer)

1. On https://pypi.org/manage/account/publishing/ add a **pending publisher**:
   project `localdesk`, owner `denis-samatov`, repository `localdesk`,
   workflow `release.yml`, environment `pypi`.
2. Do the same on https://test.pypi.org/manage/account/publishing/ (a separate
   TestPyPI account) with environment `testpypi`.
3. The GitHub environments `pypi` and `testpypi` exist. Requiring a reviewer on
   `pypi` (Settings → Environments) adds a manual approval before upload.
4. After the first image push, make the `localdesk` package public once:
   GitHub → Packages → localdesk → Package settings → Change visibility.

## Each release

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/localdesk/__init__.py`; add a `CHANGELOG.md` section. Merge to `main`.
2. Dry run (recommended): Actions → Release → Run workflow on `main`. It builds
   a unique `<version>.dev<run>` (TestPyPI, like PyPI, never accepts a version
   twice), uploads it to TestPyPI, then installs it back from TestPyPI and runs
   the CLI. Repeat as often as needed; real versions are never consumed.
3. Tag and push: `git tag -a vX.Y.Z -m "LocalDesk X.Y.Z" && git push origin vX.Y.Z`.
   The Release workflow refuses a tag that does not match the package version,
   builds, runs `twine check --strict`, smoke-tests the wheel and publishes to
   PyPI. The Docker workflow builds both platforms, smoke-tests the demo image
   and pushes `ghcr.io/denis-samatov/localdesk:<version>`, `:<major>.<minor>`
   and `:latest`.
4. Create the GitHub release from the tag with the changelog section.

PyPI versions are immutable: a published version can be yanked but never
re-uploaded. Fix mistakes with a new patch version.
