"""Integration — the boot-path digest gate wired to a real metrics registry.

The unit tiers check each piece against a stub. This tier checks the seam the
rover actually runs: config field -> resolver -> ``verify_sha256`` ->
``MetricsRegistry`` -> Prometheus exposition text, for both download paths
(world-model ``.onnx`` and BDI weights) through one registry.

That composition is where a refusal goes unnoticed: a gate that raises without
a counter is invisible on ``/metrics``, and a counter written into a registry
nobody renders is the same defect one layer down. Stubbing only the Hub keeps
everything else real.

No ``onnxruntime``, no network: runs in the blocking ``test`` job.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mousedroid.config.schema import MetricsConfig, Settings, WorldModelConfig
from mousedroid.factory.cognitive import BDI_WEIGHT_FILENAMES, build_cognitive_core
from mousedroid.factory.world_model import _resolve_world_model_onnx_path
from mousedroid.telemetry.metrics.registry import MetricsRegistry
from mousedroid.utils.artifact_integrity import ArtifactIntegrityError

_MANIFEST = "sha256.txt"
_ARTIFACT = "observe_step.onnx"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _digest(payload)


def _world_model_settings(cache_dir: Path, **overrides: Any) -> Settings:
    return Settings(
        mock_hardware=True,
        world_model=WorldModelConfig(
            engine="onnx_trt",
            onnx_cache_dir=str(cache_dir),
            **overrides,
        ),
    )


def _bdi_settings(weights_dir: Path, **overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "weights_dir": str(weights_dir),
        "enabled": True,
        "auto_download": False,
    }
    payload.update(overrides)
    return Settings(mock_hardware=True, cognitive=payload)


@pytest.fixture
def registry() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig.model_validate({}))


class TestWorldModelRefusalReachesMetrics:
    def test_a_tampered_cached_artifact_is_refused_and_visible_on_metrics(
        self, tmp_path: Path, registry: MetricsRegistry
    ) -> None:
        cache = tmp_path / "weights" / "dual_stream_rssm"
        _write(cache / _ARTIFACT, b"tampered-graph")
        (cache / _MANIFEST).write_text(_digest(b"the-real-graph") + "\n", encoding="utf-8")
        cfg = _world_model_settings(cache)

        with pytest.raises(ArtifactIntegrityError):
            _resolve_world_model_onnx_path(cfg, metrics=registry)

        rendered = registry.render_prometheus()
        assert 'model_artifact_sha256_mismatches_total{artifact="world_model_onnx"} 1' in rendered

    def test_a_good_artifact_leaves_the_family_unrendered(
        self, tmp_path: Path, registry: MetricsRegistry
    ) -> None:
        """A healthy boot must not emit the refusal family at all."""
        cache = tmp_path / "weights" / "dual_stream_rssm"
        digest = _write(cache / _ARTIFACT, b"the-real-graph")
        (cache / _MANIFEST).write_text(digest + "\n", encoding="utf-8")
        cfg = _world_model_settings(cache)

        assert _resolve_world_model_onnx_path(cfg, metrics=registry) == cache / _ARTIFACT
        assert "model_artifact_sha256_mismatches" not in registry.render_prometheus()


class TestBdiRefusalReachesMetrics:
    def test_a_tampered_weight_file_is_refused_and_visible_on_metrics(
        self, tmp_path: Path, registry: MetricsRegistry
    ) -> None:
        weights = tmp_path / "weights" / "bdi"
        lines: list[str] = []
        for name in BDI_WEIGHT_FILENAMES:
            lines.append(_write(weights / name, name.encode()) + "  " + name)
        (weights / _MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
        (weights / "belief.npz").write_bytes(b"tampered")
        cfg = _bdi_settings(weights)

        with (
            patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()),
            pytest.raises(ArtifactIntegrityError),
        ):
            build_cognitive_core(cfg, metrics=registry)

        rendered = registry.render_prometheus()
        assert 'model_artifact_sha256_mismatches_total{artifact="bdi_weights"} 1' in rendered


class TestBothSeamsShareOneFamily:
    def test_the_two_artifact_kinds_are_distinguishable_on_one_registry(
        self, tmp_path: Path, registry: MetricsRegistry
    ) -> None:
        """One family, two label values — an operator can tell which model failed."""
        cache = tmp_path / "weights" / "dual_stream_rssm"
        _write(cache / _ARTIFACT, b"tampered-graph")
        (cache / _MANIFEST).write_text(_digest(b"real") + "\n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError):
            _resolve_world_model_onnx_path(_world_model_settings(cache), metrics=registry)

        weights = tmp_path / "weights" / "bdi"
        lines = [
            _write(weights / name, name.encode()) + "  " + name for name in BDI_WEIGHT_FILENAMES
        ]
        (weights / _MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
        (weights / "affect.npz").write_bytes(b"tampered")
        with (
            patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()),
            pytest.raises(ArtifactIntegrityError),
        ):
            build_cognitive_core(_bdi_settings(weights), metrics=registry)

        rendered = registry.render_prometheus()
        assert 'artifact="world_model_onnx"} 1' in rendered
        assert 'artifact="bdi_weights"} 1' in rendered


class TestStrictPolicyIsReachableFromConfig:
    def test_the_require_switch_travels_from_settings_to_the_gate(self, tmp_path: Path) -> None:
        """The fail-closed shape is operator-reachable, not code-only."""
        cache = tmp_path / "weights" / "dual_stream_rssm"
        _write(cache / _ARTIFACT, b"unverifiable-graph")
        cfg = _world_model_settings(cache, onnx_require_sha256_manifest=True)
        with pytest.raises(ArtifactIntegrityError, match="manifest_missing"):
            _resolve_world_model_onnx_path(cfg)

    def test_the_default_config_still_boots_without_a_manifest(self, tmp_path: Path) -> None:
        cache = tmp_path / "weights" / "dual_stream_rssm"
        _write(cache / _ARTIFACT, b"unverifiable-graph")
        cfg = _world_model_settings(cache)
        assert _resolve_world_model_onnx_path(cfg) == cache / _ARTIFACT
