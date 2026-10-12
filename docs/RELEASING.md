# Releasing

Releases are built and uploaded by `.github/workflows/release.yml` with PyPI
Trusted Publishing, so no API token is stored anywhere.

## One-time setup (maintainer)

1. On https://pypi.org/manage/account/publishing/ add a **pending publisher**:
   project `localdesk`, owner `denis-samatov`, repository `localdesk`,
   workflow `release.yml`, environment `pypi`.
2. Do the same on https://test.pypi.org/manage/account/publishing/ with
   environment `testpypi`.
3. In GitHub → Settings → Environments, create `pypi` and `testpypi`.
   Requiring a reviewer on `pypi` adds a manual approval before upload.

## Each release

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/localdesk/__init__.py`; add a `CHANGELOG.md` section. Merge to `main`.
2. Optional dry run: Actions → Release → Run workflow. It builds, checks and
   uploads to TestPyPI. Verify with
   `pip install -i https://test.pypi.org/simple/ localdesk`.
3. Tag and push: `git tag v0.3.0 && git push origin v0.3.0`. The workflow
   refuses a tag that does not match the package version, builds, runs
   `twine check --strict`, smoke-tests the wheel and publishes to PyPI.
4. Create the GitHub release from the tag with the changelog section.

PyPI versions are immutable: a published version can be yanked but never
re-uploaded. Fix mistakes with a new patch version.
