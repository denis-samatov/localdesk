"""Release artifacts must be built only after the same revision passes CI."""

from pathlib import Path


def test_release_build_depends_on_reused_quality_workflow():
    root = Path(__file__).parents[1]
    release = (root / ".github/workflows/release.yml").read_text()
    ci = (root / ".github/workflows/ci.yml").read_text()
    assert "workflow_call:" in ci
    assert "uses: ./.github/workflows/ci.yml" in release
    build = release.split("  build:\n", 1)[1].split("    steps:", 1)[0]
    assert "needs: quality" in build
