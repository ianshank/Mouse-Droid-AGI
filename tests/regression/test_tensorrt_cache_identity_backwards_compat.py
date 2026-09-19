"""What must NOT have changed when the engine cache gained an identity.

Peer review D-7/D-25. The fix deliberately invalidates every previously
cached engine -- old fingerprints simply never match again, so a stale engine
is ignored rather than mis-loaded -- and that is the one behaviour change.
CLAUDE.md invariant 6 covers everything else: no new config field, every
shipped YAML loads unchanged, and the mock path is untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import JetsonConfig, Settings
from mousedroid.efficiency.tensorrt import (
    JetsonTensorRTCompiler,
    MockTensorRTCompiler,
    _model_fingerprint,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_SRC = _REPO_ROOT / "src" / "mousedroid"
_NOT_AN_OVERLAY = {"baselines.yaml", "default.yaml"}


# -- invariant 6: no new config surface ------------------------------------


def test_no_new_field_was_added() -> None:
    """The fix is entirely in code; nothing to migrate in anyone's YAML."""
    assert set(JetsonConfig.model_fields) == {
        "tensorrt_enabled",
        "gpu_memory_fraction",
        "power_mode",
        "dla_enabled",
        "thermal_zone_path",
        "gpu_load_path",
        "precision",
        "workspace_gb",
        "tensorrt_cache_dir",
    }


def test_the_cache_dir_default_is_unchanged() -> None:
    assert JetsonConfig().tensorrt_cache_dir == Path("/opt/mousedroid/tensorrt_cache")


def test_tensorrt_enabled_default_is_unchanged() -> None:
    """Deliberately still ``True``: the flag expresses intent for the day the
    seam is wired. Its *description* now says it is inert, which is the honest
    half of the fix."""
    assert JetsonConfig().tensorrt_enabled is True


def test_the_inertness_is_stated_where_an_operator_reads_it() -> None:
    """S-11 class: four shipped configs set ``tensorrt_enabled: true`` while
    nothing constructs a compiler. The description is where that gets said."""
    description = JetsonConfig.model_fields["tensorrt_enabled"].description or ""
    assert "INERT" in description


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in _CONFIG_DIR.glob("*.yaml") if p.name not in _NOT_AN_OVERLAY),
)
def test_every_shipped_overlay_still_loads(overlay: str) -> None:
    raw = yaml.safe_load((_CONFIG_DIR / overlay).read_text(encoding="utf-8")) or {}
    raw["mock_hardware"] = True
    assert isinstance(Settings.model_validate(raw), Settings)


def test_the_stacked_production_load_is_unaffected() -> None:
    cfg = load_settings(_CONFIG_DIR / "jetson_production.yaml", config_dir=_CONFIG_DIR)
    assert cfg.jetson.tensorrt_enabled is True
    assert cfg.jetson.tensorrt_cache_dir == Path("/opt/mousedroid/tensorrt_cache")


# -- the mock path, which every non-hardware test rides --------------------


async def test_the_mock_compiler_is_unchanged() -> None:
    mock = MockTensorRTCompiler()
    model = nn.Linear(4, 2)
    assert await mock.compile_model(model, {"x": [1, 4]}, "fp16") is not None


async def test_disabled_compilation_is_still_a_pass_through(tmp_path: Path) -> None:
    """``tensorrt_enabled: false`` must return the original object untouched --
    no cache directory created, no permissions applied, no fingerprint taken."""
    cache = tmp_path / "trt_cache"
    compiler = JetsonTensorRTCompiler(
        JetsonConfig(tensorrt_enabled=False, tensorrt_cache_dir=cache)
    )
    model = nn.Linear(4, 2)
    assert await compiler.compile_model(model, {"x": [1, 4]}, "fp16") is model
    assert not cache.exists()


# -- the fingerprint's model half is untouched -----------------------------


def test_model_structure_still_moves_the_fingerprint() -> None:
    """Adding runtime identity must not displace what was already hashed."""
    shapes = {"x": [1, 4]}
    assert _model_fingerprint(nn.Linear(4, 2), shapes, "fp16") != _model_fingerprint(
        nn.Linear(4, 8), shapes, "fp16"
    )


def test_precision_still_moves_the_fingerprint() -> None:
    model = nn.Linear(4, 2)
    shapes = {"x": [1, 4]}
    assert _model_fingerprint(model, shapes, "fp16") != _model_fingerprint(model, shapes, "fp32")


def test_the_fingerprint_is_still_a_short_hex_digest() -> None:
    """The cache filename format is unchanged, so an operator's ``ls`` and any
    external cleanup script still recognise it."""
    digest = _model_fingerprint(nn.Linear(4, 2), {"x": [1, 4]}, "fp16")
    assert len(digest) == 16
    assert all(character in "0123456789abcdef" for character in digest)


# -- the latent rating, made revisitable -----------------------------------


def test_nothing_in_src_constructs_the_inference_wrapper() -> None:
    """The pin that makes D-25's *latent* rating honest and revisitable.

    ``OptimizedInference`` is the only caller of ``compile_model``. Nothing
    builds it, so the pickle path this change guards is unreachable from any
    production path today -- which is why D-25 is rated latent rather than a
    live RCE. The day someone wires it into the tick, this test goes red and
    the rating gets revisited deliberately instead of by accident.
    """
    hits = [
        path
        for path in _SRC.rglob("*.py")
        if path.name != "optimized_inference.py"
        and "OptimizedInference(" in path.read_text(encoding="utf-8")
    ]
    assert not hits, (
        f"OptimizedInference is now constructed in {[str(p) for p in hits]} — "
        "the TensorRT pickle path is reachable, so re-rate peer review D-25 "
        "from latent to live and re-check the 30 Hz cold-compile hazard"
    )


def test_nothing_in_src_builds_the_tensorrt_compiler() -> None:
    """Same rating, the other entry point."""
    hits = [
        path
        for path in _SRC.rglob("*.py")
        if path.name not in {"hardware.py", "__init__.py"}
        and "build_tensorrt_compiler(" in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"build_tensorrt_compiler now has a caller: {[str(p) for p in hits]}"
