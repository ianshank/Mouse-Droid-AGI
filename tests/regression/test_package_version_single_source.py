"""Regression: the declared package version has one source of truth.

The version is declared in ``[project] version`` (pyproject.toml) and restated
in five other places: a release heading in ``CHANGELOG.md``, ``CITATION.cff``,
``HARNESS_SPEC.md``, and a ``LABEL version`` in each of the three Dockerfiles.
``CITATION.cff``'s own comment documents that set. Every one of them is pinned
here — an earlier version of this file compared only pyproject against the
changelog, so the other four could drift freely while this test stayed green
(and they had: all three Dockerfiles and CITATION.cff were still on 0.3.0).

``src/mousedroid/__init__.py``
derives ``__version__`` from *installed distribution metadata*, so the
pyproject value is what the runtime — and every telemetry sample, build label,
and PyPI artefact carrying it — actually reports.

Nothing asserted the two agree, and they did not: pyproject said ``0.3.0``
while the newest release in the changelog was ``v0.4.0``, so the whole
post-0.4.0 tree reported itself as 0.3.0.

The cause is mechanical rather than an oversight. ``release.yml``'s ``build``
job *does* compare the pushed tag against ``pyproject.toml`` — but the workflow
triggers only on ``push: tags: v*.*.*`` and the repo has never carried a tag,
so that comparison has never executed. This test is the part of that check that
runs on every PR, with no tag required.

Two parsing decisions are load-bearing:

- **Document order, not semver maximum.** The file is reverse-chronological
  Keep a Changelog, so the newest release is the *first* version heading. Its
  tail still carries ``## [0.12.0] — Previous unreleased work`` from a legacy
  pre-1.0 numbering scheme, and ``0.12.0`` sorts *above* ``0.4.0`` — a
  ``max()`` over parsed versions would therefore pin the wrong heading.
- **The ``v`` prefix is optional.** The changelog is inconsistent about it
  (``## [v0.4.0]`` vs. ``## [0.3.0]``); pyproject never carries it, so the
  prefix is stripped before comparing. Marker headings (``## [Unreleased]``,
  ``## [Released]``) do not match the version pattern and are skipped by
  construction.

Mirrors ``test_ruff_version_single_source.py`` and
``test_coverage_gate_single_source.py``, which guard the ruff pin and the
coverage threshold the same way. Both values are parsed, never hardcoded, so
this keeps holding across the next release.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._pyproject import load_pyproject

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

# A released section heading: `## [v0.4.0] — 2026-05-16 — ...` or `## [0.3.0] — ...`.
# Anything after the closing bracket (date, release title) is free-form and ignored.
_RELEASE_HEADING_RE = re.compile(
    r"^## \[v?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?)\]",
    re.MULTILINE,
)

# Bracketed headings that are section markers, not releases.
_MARKER_HEADINGS = ("## [Unreleased]", "## [Released]")


def _pyproject_version() -> str:
    """Return the version declared in ``[project] version``."""
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    version = project["version"]
    assert isinstance(version, str), f"[project] version must be a string, got {version!r}"
    return version


def _released_changelog_versions() -> list[str]:
    """Return every released version heading in CHANGELOG order (newest first)."""
    text = _CHANGELOG.read_text(encoding="utf-8")
    return _RELEASE_HEADING_RE.findall(text)


def test_pyproject_version_matches_newest_changelog_release() -> None:
    declared = _pyproject_version()
    released = _released_changelog_versions()

    assert released, (
        f"no `## [<version>]` release heading found in {_CHANGELOG.name} — if the "
        f"heading format changed, update _RELEASE_HEADING_RE rather than deleting "
        f"this gate"
    )

    newest = released[0]
    assert declared == newest, (
        f"pyproject.toml [project] version {declared!r} != newest CHANGELOG.md "
        f"release heading {newest!r}; bump both in the same change — the runtime "
        f"__version__ comes from the pyproject value, so a mismatch ships a "
        f"tree that misreports itself"
    )


def test_marker_headings_are_not_parsed_as_releases() -> None:
    text = _CHANGELOG.read_text(encoding="utf-8")

    for marker in _MARKER_HEADINGS:
        assert marker in text, (
            f"{marker} no longer present in {_CHANGELOG.name} — drop it from "
            f"_MARKER_HEADINGS so this test keeps asserting something real"
        )
        assert not _RELEASE_HEADING_RE.match(marker), (
            f"{marker} parses as a release version; the newest-release lookup "
            f"would pin a section marker instead of a release"
        )


# ---------------------------------------------------------------------------
# The other four restatement surfaces
# ---------------------------------------------------------------------------
#: Every file that restates ``[project] version``, and how to find it. Kept as
#: data so adding a surface is one row, not a new test -- and so the failure
#: message can name the file an author has to edit.
_VERSION_RESTATEMENTS: tuple[tuple[str, str], ...] = (
    ("CITATION.cff", r"^version:\s*(?P<version>[0-9][^\s]*)\s*$"),
    ("HARNESS_SPEC.md", r"^\s*version:\s*\"(?P<version>[0-9][^\"]*)\"\s*$"),
    ("Dockerfile.dev", r'^LABEL version="(?P<version>[0-9][^"]*)"'),
    ("Dockerfile.jetson", r'^LABEL version="(?P<version>[0-9][^"]*)"'),
    ("docker/Dockerfile.cloud", r'^LABEL version="(?P<version>[0-9][^"]*)"'),
)


@pytest.mark.parametrize(("relpath", "pattern"), _VERSION_RESTATEMENTS)
def test_restated_version_matches_pyproject(relpath: str, pattern: str) -> None:
    """Each restatement must equal ``[project] version``.

    These are the surfaces a release actually ships: the citation metadata, the
    harness spec, and the image labels. Nothing derives them, so only a test
    keeps them honest -- and the drift this catches is not hypothetical, it is
    what this change had to repair by hand.
    """
    path = _REPO_ROOT / relpath
    assert path.is_file(), f"{relpath} is missing — update _VERSION_RESTATEMENTS"
    found = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert found is not None, (
        f"no version literal matched in {relpath}; the file's format changed, so "
        "this pin is no longer reading what it thinks it is"
    )
    assert found.group("version") == _pyproject_version(), (
        f"{relpath} declares version {found.group('version')!r} but pyproject.toml "
        f"declares {_pyproject_version()!r}. Bump both, or derive one from the other."
    )


def test_every_restatement_surface_is_covered() -> None:
    """Anti-vacuity: the roster must not silently shrink.

    A parametrised test over an empty (or truncated) tuple passes by collecting
    nothing, which is exactly the failure mode this file already suffered once.
    """
    assert len(_VERSION_RESTATEMENTS) == 5
    covered = {relpath for relpath, _ in _VERSION_RESTATEMENTS}
    assert covered == {
        "CITATION.cff",
        "HARNESS_SPEC.md",
        "Dockerfile.dev",
        "Dockerfile.jetson",
        "docker/Dockerfile.cloud",
    }
