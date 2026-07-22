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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Headline surfaces a reviewer sees first; these must carry zero "AGI" framing.
_HEADLINE_DOCS = ("README.md", "docs/CHARTER.md", "CLAUDE.md")
_AGI_WORD = re.compile(r"\bAGI\b")

_NPZ = _REPO_ROOT / "training" / "data" / "bdi_annotations.npz"
_CAD_DIR = _REPO_ROOT / "docs" / "3D_printing_files"
_JETSON_IMAGE = _REPO_ROOT / "deployments" / "jetson-image.json"
_FETCH = _REPO_ROOT / "scripts" / "fetch_data.sh"
_PURGE = _REPO_ROOT / "scripts" / "purge_history.sh"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Modules the original ask mis-labelled "stub"; they are implemented + unit-tested.
_WORKING_MODULES = ("curiosity", "meta", "growth", "scaling")


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def test_headline_docs_drop_agi_framing() -> None:
    for rel in _HEADLINE_DOCS:
        text = _read(rel)
        assert "MouseDroidAGI" not in text, f"{rel} still carries the MouseDroidAGI brand token"
        assert not _AGI_WORD.search(text), f"{rel} still contains a bare 'AGI' overclaim"


def test_package_name_is_unchanged() -> None:
    # The rename is brand/docs only — the import surface must not move.
    pyproject = _read("pyproject.toml")
    assert re.search(r'(?m)^name\s*=\s*"mousedroid"', pyproject), (
        "pyproject [project] name drifted off 'mousedroid' — the rename must stay docs-only"
    )


def test_pillar_table_uses_integration_axis_not_stub_labels() -> None:
    readme = _read("README.md")
    # The honest tiers must both be present ...
    assert "Wired into the runtime loop" in readme, "README lost the runtime-integrated pillar tier"
    assert "not yet wired into the loop" in readme, "README lost the not-yet-wired pillar tier"
    # ... and the working modules must never be re-branded scaffolding/stubs.
    assert "🔬 Scaffolding" not in readme, "README re-introduced a 'Scaffolding' stub label"
    for module in _WORKING_MODULES:
        assert f"`{module}/` | 🔬" not in readme, f"README labels {module}/ a stub — it is implemented"
    # curiosity is factory-wired; it belongs in the integrated tier, not the roadmap.
    wired_section = readme.split("not yet wired", 1)[0]
    assert "`curiosity/`" in wired_section, "curiosity/ must sit in the runtime-integrated tier"


def test_large_blobs_untracked_with_pointers() -> None:
    assert not _NPZ.exists(), "bdi_annotations.npz is back in the tree — keep it out (regeneratable)"
    stray = sorted(p.name for p in _CAD_DIR.glob("*") if p.suffix in {".stl", ".FCStd"})
    assert not stray, f"CAD binaries back under docs/3D_printing_files/: {stray}"
    assert (_CAD_DIR / "README.md").is_file(), "docs/3D_printing_files/README.md pointer is missing"
    assert (_REPO_ROOT / "training" / "data" / "README.md").is_file(), (
        "training/data/README.md pointer is missing"
    )


def test_gitignore_covers_cad_and_data() -> None:
    gitignore = _read(".gitignore")
    for pattern in ("*.stl", "*.FCStd", "docs/3D_printing_files/*", "training/data/*.npz"):
        assert pattern in gitignore, f".gitignore no longer ignores '{pattern}'"


def test_jetson_image_sha_stays_a_reachable_hash() -> None:
    # Phase A must not mutate the deploy pin; Phase B re-pins it to a rewritten,
    # still-40-hex, reachable SHA. Either way it is never blank/short/garbage.
    record = json.loads(_JETSON_IMAGE.read_text(encoding="utf-8"))
    sha = record.get("sha", "")
    assert _SHA_RE.match(sha), f"deployments/jetson-image.json sha is not a 40-hex hash: {sha!r}"


def test_fetch_data_is_regeneration_first() -> None:
    fetch = _read("scripts/fetch_data.sh")
    assert _FETCH.exists()
    # Regeneration is the authoritative path; the HF mirror is an opt-in fast path.
    assert "training.run_pipeline" in fetch, "fetch_data.sh lost its regeneration path"
    assert "--from-hf" in fetch, "fetch_data.sh lost the optional HF fast-path flag"
    assert "HF_DATASET" in fetch, "fetch_data.sh should keep the HF dataset id env-overridable"


def test_purge_script_is_safe_and_repin_aware() -> None:
    purge = _read("scripts/purge_history.sh")
    assert _PURGE.exists()
    assert "git filter-repo" in purge, "purge_history.sh lost the filter-repo call"
    # The config-compat gate survives the rewrite only if the deploy SHA is re-pinned.
    assert "commit-map" in purge, "purge_history.sh lost the commit-map re-pin lookup"
    assert "jetson-image.json" in purge, "purge_history.sh lost the deploy-record re-pin"
    assert "check_config_compat.py" in purge, "purge_history.sh lost the post-repin verification"
    # Destructive push must be opt-in, never the default.
    assert "--push" in purge, "purge_history.sh lost its opt-in --push gate"
