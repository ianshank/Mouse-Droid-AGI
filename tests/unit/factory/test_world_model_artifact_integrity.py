"""The world-model ``.onnx`` resolver's integrity gate (Phase 7, tasks 7.1/7.1b).

Before this gate the resolver checked only ``model_path.is_file()`` and passed
no ``revision`` — ADR-008's own "Negative" section already named the
consequence, "wrong weights = wrong inference, silently".

Deliberately exercises ``_resolve_world_model_onnx_path`` rather than
``build_world_model``: the resolver is the seam where the download happens, and
it needs neither ``onnxruntime`` nor a real ``.onnx`` graph, so every
assertion here runs in the blocking ``test`` job (which installs no ONNX
extras). Hub access is stubbed by monkeypatching the same
``download_weights_from_huggingface`` the resolver imports, so no test touches
the network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from mousedroid.config.schema import MetricsConfig, Settings, WorldModelConfig
from mousedroid.factory.world_model import _resolve_world_model_onnx_path
from mousedroid.telemetry.metrics.registry import MetricsRegistry
from mousedroid.utils.artifact_integrity import (
    ArtifactContractError,
    ArtifactIntegrityError,
    ArtifactMissingError,
)

_MANIFEST = "sha256.txt"


def _settings(**world_model: Any) -> Settings:
    return Settings(mock_hardware=True, world_model=WorldModelConfig(**world_model))


def _write_artifact(path: Path, payload: bytes = b"onnx-graph-bytes") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class _DownloadSpy:
    """Stand-in for ``download_weights_from_huggingface``.

    Records every call's kwargs so the revision pin can be asserted, and
    materialises the requested files from ``payloads`` so the resolver's
    ``is_file()`` checks see a real download.
    """

    def __init__(self, payloads: dict[str, bytes] | None = None, *, succeed: bool = True) -> None:
        self.payloads = payloads or {}
        self.succeed = succeed
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        if not self.succeed:
            return False
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        wrote_any = False
        for name in kwargs["filenames"]:
            payload = self.payloads.get(name)
            if payload is None:
                continue
            (target / name).write_bytes(payload)
            wrote_any = True
        return wrote_any

    @property
    def revisions(self) -> list[str | None]:
        return [call.get("revision") for call in self.calls]

    @property
    def requested_filenames(self) -> list[str]:
        return [name for call in self.calls for name in call["filenames"]]


@pytest.fixture
def stub_download(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a ``_DownloadSpy`` in place of the real HF fetch."""

    def _install(payloads: dict[str, bytes] | None = None, *, succeed: bool = True) -> _DownloadSpy:
        spy = _DownloadSpy(payloads, succeed=succeed)
        monkeypatch.setattr(
            "mousedroid.utils.weights_manager.download_weights_from_huggingface",
            spy,
        )
        return spy

    return _install


class TestExplicitPathBranch:
    def test_missing_explicit_path_is_returned_unchanged(self, tmp_path: Path) -> None:
        """Pre-existing contract: the clear error comes from ``warmup()``.

        ``tests/unit/factory/test_factory_world_model_engine.py`` pins that
        ``build_world_model`` succeeds for an absent ``onnx_path`` and the
        ``FileNotFoundError`` surfaces at warmup, leaving the composite's
        PyTorch imagine engine usable. The gate verifies bytes it has; it does
        not take over the missing-file report.
        """
        absent = tmp_path / "does_not_exist.onnx"
        cfg = _settings(engine="onnx_trt", onnx_path=str(absent))
        assert _resolve_world_model_onnx_path(cfg) == absent

    def test_explicit_path_with_matching_sibling_manifest_is_accepted(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        digest = _write_artifact(artifact)
        (tmp_path / _MANIFEST).write_text(digest + "\n", encoding="utf-8")
        cfg = _settings(engine="onnx_trt", onnx_path=str(artifact))
        assert _resolve_world_model_onnx_path(cfg) == artifact

    def test_explicit_path_with_mismatched_sibling_manifest_is_refused(
        self, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write_artifact(artifact)
        (tmp_path / _MANIFEST).write_text("00" * 32 + "\n", encoding="utf-8")
        cfg = _settings(engine="onnx_trt", onnx_path=str(artifact))
        with pytest.raises(ArtifactIntegrityError):
            _resolve_world_model_onnx_path(cfg)

    def test_explicit_path_without_a_manifest_is_accepted_by_default(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write_artifact(artifact)
        cfg = _settings(engine="onnx_trt", onnx_path=str(artifact))
        assert _resolve_world_model_onnx_path(cfg) == artifact

    def test_explicit_path_without_a_manifest_is_refused_under_strict_policy(
        self, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write_artifact(artifact)
        cfg = _settings(
            engine="onnx_trt",
            onnx_path=str(artifact),
            onnx_require_sha256_manifest=True,
        )
        with pytest.raises(ArtifactIntegrityError, match="manifest_missing"):
            _resolve_world_model_onnx_path(cfg)

    def test_a_pt_checkpoint_is_never_accepted_as_the_graph(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "final.pt"
        _write_artifact(checkpoint)
        cfg = _settings(engine="onnx_trt", onnx_path=str(checkpoint))
        with pytest.raises(ArtifactContractError, match=r"expected a '\.onnx' file"):
            _resolve_world_model_onnx_path(cfg)


class TestCacheHitBranch:
    def test_cache_hit_is_verified_against_the_local_manifest(self, tmp_path: Path) -> None:
        cache = tmp_path / "weights"
        digest = _write_artifact(cache / "observe_step.onnx")
        (cache / _MANIFEST).write_text(digest + "\n", encoding="utf-8")
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(cache))
        assert _resolve_world_model_onnx_path(cfg) == cache / "observe_step.onnx"

    def test_cache_hit_with_a_stale_digest_is_refused(self, tmp_path: Path) -> None:
        cache = tmp_path / "weights"
        _write_artifact(cache / "observe_step.onnx")
        (cache / _MANIFEST).write_text("00" * 32 + "\n", encoding="utf-8")
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(cache))
        with pytest.raises(ArtifactIntegrityError):
            _resolve_world_model_onnx_path(cfg)

    def test_cache_hit_does_not_reach_the_network(self, tmp_path: Path, stub_download: Any) -> None:
        """A cached artifact must stay bootable with no connectivity.

        The manifest was fetched alongside the artifact on the download that
        populated the cache, so verification reads the local copy. Re-fetching
        it here would add a retry-with-backoff stall to every offline boot.
        """
        cache = tmp_path / "weights"
        digest = _write_artifact(cache / "observe_step.onnx")
        (cache / _MANIFEST).write_text(digest + "\n", encoding="utf-8")
        spy = stub_download()
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(cache))
        _resolve_world_model_onnx_path(cfg)
        assert spy.calls == []

    def test_cache_hit_without_a_manifest_is_refused_under_strict_policy(
        self, tmp_path: Path
    ) -> None:
        cache = tmp_path / "weights"
        _write_artifact(cache / "observe_step.onnx")
        cfg = _settings(
            engine="onnx_trt",
            onnx_cache_dir=str(cache),
            onnx_require_sha256_manifest=True,
        )
        with pytest.raises(ArtifactIntegrityError, match="manifest_missing"):
            _resolve_world_model_onnx_path(cfg)


class TestDownloadBranch:
    def test_every_fetch_pins_the_configured_revision(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        payload = b"onnx-graph-bytes"
        digest = hashlib.sha256(payload).hexdigest()
        spy = stub_download(
            {
                "observe_step.onnx": payload,
                _MANIFEST: (digest + "\n").encode(),
            }
        )
        cfg = _settings(
            engine="onnx_trt",
            onnx_cache_dir=str(tmp_path / "weights"),
            onnx_revision="deadbeef",
        )
        _resolve_world_model_onnx_path(cfg)
        assert spy.revisions == ["deadbeef", "deadbeef"]

    def test_the_manifest_is_fetched_alongside_the_artifact(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        payload = b"onnx-graph-bytes"
        digest = hashlib.sha256(payload).hexdigest()
        spy = stub_download(
            {"observe_step.onnx": payload, _MANIFEST: (digest + "\n").encode()},
        )
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(tmp_path / "weights"))
        _resolve_world_model_onnx_path(cfg)
        assert spy.requested_filenames == ["observe_step.onnx", _MANIFEST]

    def test_a_freshly_downloaded_mismatch_is_refused_and_counted(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        stub_download(
            {
                "observe_step.onnx": b"onnx-graph-bytes",
                _MANIFEST: (("00" * 32) + "\n").encode(),
            }
        )
        registry = MetricsRegistry(MetricsConfig.model_validate({}))
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(tmp_path / "weights"))
        with pytest.raises(ArtifactIntegrityError):
            _resolve_world_model_onnx_path(cfg, metrics=registry)
        rendered = registry.render_prometheus()
        assert 'model_artifact_sha256_mismatches_total{artifact="world_model_onnx"} 1' in rendered

    def test_an_unpublished_artifact_raises_the_named_missing_error(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        """The live case: the repo holds only ``.gitattributes`` and ``README.md``."""
        stub_download(succeed=False)
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(tmp_path / "weights"))
        with pytest.raises(ArtifactMissingError) as excinfo:
            _resolve_world_model_onnx_path(cfg)
        assert "revision" in str(excinfo.value)

    def test_the_missing_error_stays_catchable_as_file_not_found(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        stub_download(succeed=False)
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(tmp_path / "weights"))
        with pytest.raises(FileNotFoundError):
            _resolve_world_model_onnx_path(cfg)

    def test_a_missing_manifest_after_download_is_permitted_by_default(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        """A repo that has not published a manifest must not become unbootable."""
        stub_download({"observe_step.onnx": b"onnx-graph-bytes"})
        cfg = _settings(engine="onnx_trt", onnx_cache_dir=str(tmp_path / "weights"))
        resolved = _resolve_world_model_onnx_path(cfg)
        assert resolved.is_file()

    def test_a_missing_manifest_after_download_fails_closed_under_strict_policy(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        stub_download({"observe_step.onnx": b"onnx-graph-bytes"})
        cfg = _settings(
            engine="onnx_trt",
            onnx_cache_dir=str(tmp_path / "weights"),
            onnx_require_sha256_manifest=True,
        )
        with pytest.raises(ArtifactIntegrityError):
            _resolve_world_model_onnx_path(cfg)
