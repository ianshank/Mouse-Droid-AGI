"""The BDI weight download's integrity gate (Phase 7, task 7.1b).

This is the seam that runs on **every boot** — ``config/jetson_production.yaml``
sets ``cognitive.enabled: true`` and ``auto_download: true`` — and until now it
pinned no revision and verified no digest, while the fail-closed
``verify_sha256`` + ``sha256.txt`` machinery already existed on the OTA path.

``NeuralBDI`` is patched out throughout: these tests are about the gate, not
about ``.npz`` deserialisation, and the four weight files are written as
arbitrary bytes so their digests are controllable. No network, no torch
inference, no ONNX — the whole file runs in the blocking ``test`` job.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.factory.cognitive import (
    BDI_WEIGHT_FILENAMES,
    _resolve_bdi_weights,
    build_cognitive_core,
)
from mousedroid.telemetry.metrics.registry import MetricsRegistry
from mousedroid.utils.artifact_integrity import ArtifactIntegrityError

_MANIFEST = "sha256.txt"
_SUBFOLDER = "bdi"


def _settings(weights_dir: Path, **cognitive: Any) -> Settings:
    payload: dict[str, Any] = {
        "weights_dir": str(weights_dir),
        "enabled": True,
        "auto_download": False,
    }
    payload.update(cognitive)
    return Settings(mock_hardware=True, cognitive=payload)


def _write_weight_set(weights_dir: Path) -> dict[str, str]:
    """Write the four BDI files and return ``{filename: digest}``."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in BDI_WEIGHT_FILENAMES:
        payload = name.encode()
        (weights_dir / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _write_manifest(weights_dir: Path, digests: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in digests.items()]
    (weights_dir / _MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")


class _DownloadSpy:
    """Stand-in for ``download_weights_from_huggingface`` on the BDI path."""

    def __init__(self, payloads: dict[str, bytes], *, succeed: bool = True) -> None:
        self.payloads = payloads
        self.succeed = succeed
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        if not self.succeed:
            return False
        # Mirrors the real helper's layout: <local_dir>/<subfolder>/<filename>.
        target = Path(kwargs["local_dir"]) / kwargs.get("subfolder", "")
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
    def requested_filenames(self) -> list[list[str]]:
        return [list(call["filenames"]) for call in self.calls]


@pytest.fixture
def stub_download(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a ``_DownloadSpy`` in place of the real HF fetch."""

    def _install(payloads: dict[str, bytes], *, succeed: bool = True) -> _DownloadSpy:
        spy = _DownloadSpy(payloads, succeed=succeed)
        monkeypatch.setattr("mousedroid.utils.download_weights_from_huggingface", spy)
        return spy

    return _install


def _hub_payloads(weights_dir: Path) -> dict[str, bytes]:
    """The four weight files plus a matching manifest, as Hub bytes."""
    digests = {name: hashlib.sha256(name.encode()).hexdigest() for name in BDI_WEIGHT_FILENAMES}
    manifest = "\n".join(f"{digest}  {name}" for name, digest in digests.items()) + "\n"
    payloads: dict[str, bytes] = {name: name.encode() for name in BDI_WEIGHT_FILENAMES}
    payloads[_MANIFEST] = manifest.encode()
    return payloads


class TestLocalWeightsBranch:
    def test_matching_manifest_is_accepted(self, tmp_path: Path) -> None:
        weights = tmp_path / "bdi"
        _write_manifest(weights, _write_weight_set(weights))
        cfg = _settings(weights)
        with patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()):
            _, source = _resolve_bdi_weights(cfg)
        assert source == "local"

    def test_a_tampered_weight_file_is_refused(self, tmp_path: Path) -> None:
        weights = tmp_path / "bdi"
        digests = _write_weight_set(weights)
        _write_manifest(weights, digests)
        (weights / "belief.npz").write_bytes(b"tampered")
        cfg = _settings(weights)
        with (
            patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()),
            pytest.raises(ArtifactIntegrityError),
        ):
            _resolve_bdi_weights(cfg)

    def test_a_refusal_increments_the_counter(self, tmp_path: Path) -> None:
        weights = tmp_path / "bdi"
        _write_manifest(weights, _write_weight_set(weights))
        (weights / "affect.npz").write_bytes(b"tampered")
        cfg = _settings(weights)
        registry = MetricsRegistry(MetricsConfig.model_validate({}))
        with (
            patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()),
            pytest.raises(ArtifactIntegrityError),
        ):
            _resolve_bdi_weights(cfg, metrics=registry)
        assert (
            'model_artifact_sha256_mismatches_total{artifact="bdi_weights"} 1'
            in registry.render_prometheus()
        )

    def test_absent_manifest_still_loads_by_default(self, tmp_path: Path) -> None:
        """Today's behaviour is preserved: no manifest published yet."""
        weights = tmp_path / "bdi"
        _write_weight_set(weights)
        cfg = _settings(weights)
        with patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()):
            _, source = _resolve_bdi_weights(cfg)
        assert source == "local"

    def test_absent_manifest_fails_closed_under_a_strict_policy(self, tmp_path: Path) -> None:
        weights = tmp_path / "bdi"
        _write_weight_set(weights)
        cfg = _settings(weights, require_sha256_manifest=True)
        with (
            patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()),
            pytest.raises(ArtifactIntegrityError, match="manifest_missing"),
        ):
            _resolve_bdi_weights(cfg)


class TestDownloadBranch:
    def test_every_fetch_pins_the_configured_revision(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        weights = tmp_path / "weights" / _SUBFOLDER
        spy = stub_download(_hub_payloads(weights))
        cfg = _settings(
            weights,
            auto_download=True,
            huggingface_subfolder=_SUBFOLDER,
            huggingface_revision="cafebabe",
        )
        with patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()):
            _, source = _resolve_bdi_weights(cfg)
        assert source == "huggingface"
        assert spy.revisions == ["cafebabe", "cafebabe"]

    def test_the_manifest_is_fetched_after_the_weight_set(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        weights = tmp_path / "weights" / _SUBFOLDER
        spy = stub_download(_hub_payloads(weights))
        cfg = _settings(weights, auto_download=True, huggingface_subfolder=_SUBFOLDER)
        with patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()):
            _resolve_bdi_weights(cfg)
        assert spy.requested_filenames == [list(BDI_WEIGHT_FILENAMES), [_MANIFEST]]

    def test_a_downloaded_mismatch_is_refused(self, tmp_path: Path, stub_download: Any) -> None:
        weights = tmp_path / "weights" / _SUBFOLDER
        payloads = _hub_payloads(weights)
        payloads["desire.npz"] = b"not-what-the-manifest-says"
        stub_download(payloads)
        cfg = _settings(weights, auto_download=True, huggingface_subfolder=_SUBFOLDER)
        with (
            patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()),
            pytest.raises(ArtifactIntegrityError),
        ):
            _resolve_bdi_weights(cfg)

    def test_a_failed_download_still_falls_back_to_random(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        """Unchanged behaviour: the gate never converts a fetch failure into a raise."""
        weights = tmp_path / "weights" / _SUBFOLDER
        stub_download({}, succeed=False)
        cfg = _settings(weights, auto_download=True, huggingface_subfolder=_SUBFOLDER)
        with patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()):
            _, source = _resolve_bdi_weights(cfg)
        assert source == "random"

    def test_a_failed_download_does_not_attempt_the_manifest(
        self, tmp_path: Path, stub_download: Any
    ) -> None:
        weights = tmp_path / "weights" / _SUBFOLDER
        spy = stub_download({}, succeed=False)
        cfg = _settings(weights, auto_download=True, huggingface_subfolder=_SUBFOLDER)
        with patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()):
            _resolve_bdi_weights(cfg)
        assert spy.requested_filenames == [list(BDI_WEIGHT_FILENAMES)]


class TestBuildCognitiveCoreSeam:
    def test_the_metrics_registry_reaches_the_weight_resolver(self, tmp_path: Path) -> None:
        """``build_cognitive_core`` threads ``metrics=`` through to the gate."""
        weights = tmp_path / "bdi"
        _write_manifest(weights, _write_weight_set(weights))
        (weights / "intention.npz").write_bytes(b"tampered")
        cfg = _settings(weights)
        registry = MetricsRegistry(MetricsConfig.model_validate({}))
        with (
            patch("mousedroid.cognitive.bdi_model.NeuralBDI", return_value=MagicMock()),
            pytest.raises(ArtifactIntegrityError),
        ):
            build_cognitive_core(cfg, metrics=registry)
        assert "bdi_weights" in registry.render_prometheus()

    def test_the_metrics_keyword_is_optional(self, tmp_path: Path) -> None:
        """Every pre-existing ``build_cognitive_core(cfg)`` call site keeps working."""
        cfg = _settings(tmp_path / "absent")
        core = build_cognitive_core(cfg)
        assert core is not None

    def test_the_weight_filename_set_is_the_shared_constant(self) -> None:
        """Three call sites (probe, download, gate) read one list."""
        assert BDI_WEIGHT_FILENAMES == (
            "belief.npz",
            "desire.npz",
            "intention.npz",
            "affect.npz",
        )
