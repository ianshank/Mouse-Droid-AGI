# tests/regression/test_portfolio_reframe_aqa.py
"""AQA: locks the "MouseDroid" edge-robotics-portfolio reframe.

Pins the contracts the reframe established so a careless later edit cannot
silently regress them:

* the headline docs no longer overclaim "AGI" (brand token / bare ``AGI`` word);
* the Python package stays ``mousedroid`` (rename is docs-only — no import churn);
* the pillar table stays split on the *runtime-integration* axis and never
  re-labels the working ``curiosity``/``meta``/``growth``/``scaling`` modules a
  "stub";
* the large binary artefacts stay untracked with pointer READMEs in their place;
* the deployment SHA pin stays a full, reachable-format hash (Phase A must not
  touch it — the history purge re-pins it separately);
* the stop-tracking + purge tooling keeps its load-bearing contracts
  (regeneration-first fetch, dry-run-default purge, config-compat re-pin).

Pure-stdlib file/text assertions — no package import — so the contract holds
even where the runtime deps are absent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Headline surfaces a reviewer sees first; these must carry zero "AGI" framing.
_HEADLINE_DOCS = ("README.md", "docs/CHARTER.md", "CLAUDE.md")
_AGI_WORD = re.compile(r"\bAGI\b")

_CAD_DIR = _REPO_ROOT / "docs" / "3D_printing_files"
_JETSON_IMAGE = _REPO_ROOT / "deployments" / "jetson-image.json"
_FETCH = _REPO_ROOT / "scripts" / "fetch_data.sh"
_PURGE = _REPO_ROOT / "scripts" / "purge_history.sh"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT = shutil.which("git")

# Large binaries that must stay OUT of git. The contract is "not *tracked*", not
# "absent from disk" — a contributor who runs scripts/fetch_data.sh (or downloads
# CAD) legitimately materialises these gitignored files in the working tree.
_UNTRACKED_BLOB_PATHSPECS = (
    "training/data/bdi_annotations.npz",
    "docs/3D_printing_files/*.stl",
    "docs/3D_printing_files/*.FCStd",
)

# Modules the original ask mis-labelled "stub"; they are implemented + unit-tested.
_WORKING_MODULES = ("curiosity", "meta", "growth", "scaling")

# Forward-facing surfaces that must not re-assert the removed "cohesive agentic system" framing.
_FORWARD_DOCS = (
    "README.md",
    "docs/CHARTER.md",
    "CLAUDE.md",
    "HARNESS_SPEC.md",
    "docs/architecture/c4-overview.md",
)


def _read(rel: str) -> str:
    """Read a repo-relative text file as UTF-8."""
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def _git_tracked(*pathspecs: str) -> list[str]:
    """Return the git-tracked files matching ``pathspecs`` (empty when none are tracked)."""
    result = subprocess.run(
        ["git", "ls-files", "--", *pathspecs],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_headline_docs_drop_agi_framing() -> None:
    """The headline docs carry no MouseDroidAGI token or bare 'AGI' overclaim."""
    for rel in _HEADLINE_DOCS:
        text = _read(rel)
        assert "MouseDroidAGI" not in text, f"{rel} still carries the MouseDroidAGI brand token"
        assert not _AGI_WORD.search(text), f"{rel} still contains a bare 'AGI' overclaim"


def test_package_name_is_unchanged() -> None:
    """The rename stays docs-only: the ``mousedroid`` package name is unchanged."""
    # The rename is brand/docs only — the import surface must not move.
    pyproject = _read("pyproject.toml")
    assert re.search(
        r'(?m)^name\s*=\s*"mousedroid"', pyproject
    ), "pyproject [project] name drifted off 'mousedroid' — the rename must stay docs-only"


def test_pillar_table_uses_integration_axis_not_stub_labels() -> None:
    """The pillar table splits on runtime-integration and never labels working modules stubs."""
    readme = _read("README.md")
    # The honest tiers must both be present ...
    assert "Wired into the runtime loop" in readme, "README lost the runtime-integrated pillar tier"
    assert "not yet wired into the loop" in readme, "README lost the not-yet-wired pillar tier"
    # ... and the working modules must never be re-branded scaffolding/stubs.
    assert "🔬 Scaffolding" not in readme, "README re-introduced a 'Scaffolding' stub label"
    for module in _WORKING_MODULES:
        assert (
            f"`{module}/` | 🔬" not in readme
        ), f"README labels {module}/ a stub — it is implemented"
    # curiosity is factory-wired; it belongs in the integrated tier, not the roadmap.
    wired_section = readme.split("not yet wired", 1)[0]
    assert "`curiosity/`" in wired_section, "curiosity/ must sit in the runtime-integrated tier"


def test_large_blobs_untracked_with_pointers() -> None:
    """The large blobs are untracked by git while the pointer READMEs stay tracked."""
    if _GIT is None:
        pytest.skip("git not available — cannot verify tracking status")
    # Assert the blobs are not *tracked* (robust to a locally-regenerated .npz or
    # downloaded CAD, which are gitignored but may be present on disk).
    tracked = _git_tracked(*_UNTRACKED_BLOB_PATHSPECS)
    assert not tracked, f"large binaries are tracked in git (should be purged): {tracked}"
    # The pointer READMEs, by contrast, MUST be tracked + present on disk.
    assert (
        _CAD_DIR.is_dir()
    ), "docs/3D_printing_files/ directory vanished (pointer README lives here)"
    assert (_CAD_DIR / "README.md").is_file(), "docs/3D_printing_files/README.md pointer is missing"
    assert (
        _REPO_ROOT / "training" / "data" / "README.md"
    ).is_file(), "training/data/README.md pointer is missing"


def test_gitignore_covers_cad_and_data() -> None:
    """`.gitignore` keeps ignoring the CAD binaries and the generated .npz."""
    gitignore = _read(".gitignore")
    for pattern in ("*.stl", "*.FCStd", "docs/3D_printing_files/*", "training/data/*.npz"):
        assert pattern in gitignore, f".gitignore no longer ignores '{pattern}'"


def test_dockerignore_and_c4_reflect_artifact_handling() -> None:
    """The Docker build context excludes the generated artefacts and the C4 doc exists."""
    dockerignore = _read(".dockerignore")
    for pattern in ("training/data/*.npz", "docs/3D_printing_files/"):
        assert pattern in dockerignore, f".dockerignore no longer excludes '{pattern}'"
    assert (
        _REPO_ROOT / "docs" / "architecture" / "c4-artifact-storage.md"
    ).is_file(), "the artifact-storage C4 doc referenced by CLAUDE.md / CHANGELOG is missing"


def test_jetson_image_sha_stays_a_reachable_hash() -> None:
    """The deploy-record SHA stays a full 40-hex hash (never blank/short/garbage)."""
    # Phase A must not mutate the deploy pin; Phase B re-pins it to a rewritten,
    # still-40-hex, reachable SHA. Either way it is never blank/short/garbage.
    record = json.loads(_JETSON_IMAGE.read_text(encoding="utf-8"))
    sha = record.get("sha", "")
    assert _SHA_RE.match(sha), f"deployments/jetson-image.json sha is not a 40-hex hash: {sha!r}"


def test_fetch_data_is_regeneration_first() -> None:
    """fetch_data.sh regenerates via the pipeline by default, HF as an opt-in fast path."""
    fetch = _read("scripts/fetch_data.sh")
    assert _FETCH.exists()
    # Regeneration is the authoritative path; the HF mirror is an opt-in fast path.
    assert "training.run_pipeline" in fetch, "fetch_data.sh lost its regeneration path"
    assert "--phases 0" in fetch, "fetch_data.sh must call pipeline phase 0 (argparse rejects '0b')"
    assert "--from-hf" in fetch, "fetch_data.sh lost the optional HF fast-path flag"
    assert "HF_DATASET" in fetch, "fetch_data.sh should keep the HF dataset id env-overridable"


def test_purge_script_is_safe_and_repin_aware() -> None:
    """purge_history.sh keeps its opt-in push, commit-map re-pin, and CAD-glob contracts."""
    purge = _read("scripts/purge_history.sh")
    assert _PURGE.exists()
    assert "git filter-repo" in purge, "purge_history.sh lost the filter-repo call"
    # The config-compat gate survives the rewrite only if the deploy SHA is re-pinned.
    assert "commit-map" in purge, "purge_history.sh lost the commit-map re-pin lookup"
    assert "jetson-image.json" in purge, "purge_history.sh lost the deploy-record re-pin"
    assert "check_config_compat.py" in purge, "purge_history.sh lost the post-repin verification"
    # Destructive push must be opt-in, never the default.
    assert "--push" in purge, "purge_history.sh lost its opt-in --push gate"
    # Purge CAD *binaries* by glob, never the whole dir — that would also delete the
    # pointer README the Phase A contract keeps in place.
    assert (
        "docs/3D_printing_files/*.stl" in purge
    ), "purge must target CAD blobs by glob, not the dir"
    assert (
        "docs/3D_printing_files/*.FCStd" in purge
    ), "purge must target CAD blobs by glob, not the dir"


def test_forward_docs_drop_cohesive_agentic_overclaim() -> None:
    """No forward-facing doc re-asserts the removed 'cohesive agentic system' overclaim."""
    # The reframe replaced "cohesive agentic system" with the wired/not-wired split
    # everywhere a reviewer sees it; none of these surfaces may re-assert it.
    for rel in _FORWARD_DOCS:
        assert "cohesive agentic system" not in _read(
            rel
        ), f"{rel} re-introduced the 'cohesive agentic system' overclaim"


def test_new_scripts_are_fail_fast() -> None:
    """Both new shell scripts run under ``set -euo pipefail``."""
    for rel in ("scripts/fetch_data.sh", "scripts/purge_history.sh"):
        assert "set -euo pipefail" in _read(rel), f"{rel} must be fail-fast (set -euo pipefail)"


def test_curiosity_is_factory_wired() -> None:
    """The factory defines and calls build_curiosity_module (backs the 'wired' claim)."""
    # Backs README's claim that curiosity is runtime-integrated (not a stub):
    # the factory both defines and calls the builder.
    factory = _read("src/mousedroid/factory.py")
    assert "def build_curiosity_module" in factory, "curiosity factory builder vanished"
    assert (
        factory.count("build_curiosity_module") >= 2
    ), "build_curiosity_module is defined but never called — curiosity is no longer wired"
