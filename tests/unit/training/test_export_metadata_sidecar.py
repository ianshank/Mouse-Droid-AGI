"""The export script's integrity fence and metadata sidecar (Phase 7, tasks 7.1b/7.2).

Two behaviours are pinned here:

* ``_load_checkpoint`` loads with ``weights_only=True`` unconditionally and
  verifies ``--checkpoint-sha256`` before reading. The removed
  ``except TypeError`` fallback silently retried with the legacy
  arbitrary-pickle loader, and its trigger condition cannot occur at the
  project's ``torch>=2.1`` floor — so it could only ever have fired where
  running unverified pickle was the worst available response.
* ``_write_metadata_sidecar`` writes the record beside the ``.onnx``, with the
  filename taken from ``WorldModelConfig.onnx_metadata_filename`` rather than a
  literal in the script.

No ``onnxruntime`` and no ``onnx``: the sidecar is exercised against a stand-in
artifact file (the writer only hashes bytes), so nothing here needs a real
graph and the whole file runs in the blocking ``test`` job. Real-export
coverage lives in ``test_export_dual_stream_rssm_onnx.py`` behind
``importorskip``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from mousedroid.config.schema import ModelConfig, WorldModelConfig
from tests._script_loader import load_script_module

_OPSET = 17


@pytest.fixture(scope="module")
def export_module() -> Any:
    return load_script_module("export_dual_stream_rssm_onnx")


def _cfg() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "vision_dim": 16,
            "vision_proj_dim": 8,
            "ultrasonic_dim": 1,
            "ultrasonic_proj_dim": 4,
            "motor_state_dim": 4,
            "motor_proj_dim": 4,
            "hidden_dim": 32,
            "latent_dim": 8,
            "action_dim": 2,
            "obs_dim": 16,
            "cfc_hidden_dim": 16,
        }
    )


def _args(**overrides: Any) -> argparse.Namespace:
    payload: dict[str, Any] = {
        "output": Path("observe_step.onnx"),
        "opset": _OPSET,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "config": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


_GIT_FAILURE_RETURNCODE = 128
"""``git rev-parse`` exit code when the directory is not a repository."""


class _StubModel:
    """Captures ``load_state_dict`` without needing a real ``DualStreamRSSM``."""

    def __init__(self) -> None:
        self.loaded: dict[str, Any] | None = None

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.loaded = payload


class TestCheckpointLoaderFence:
    def test_loads_a_bare_state_dict(self, tmp_path: Path, export_module: Any) -> None:
        checkpoint = tmp_path / "final.pt"
        torch.save({"weight": torch.zeros(2)}, checkpoint)
        model = _StubModel()
        export_module._load_checkpoint(model, checkpoint)
        assert model.loaded is not None

    def test_unwraps_a_training_checkpoint(self, tmp_path: Path, export_module: Any) -> None:
        checkpoint = tmp_path / "final.pt"
        torch.save({"model_state_dict": {"weight": torch.zeros(2)}}, checkpoint)
        model = _StubModel()
        export_module._load_checkpoint(model, checkpoint)
        assert model.loaded is not None
        assert "weight" in model.loaded

    def test_a_matching_digest_is_accepted(self, tmp_path: Path, export_module: Any) -> None:
        from mousedroid.common.hashing import digest_file_sha256

        checkpoint = tmp_path / "final.pt"
        torch.save({"weight": torch.zeros(2)}, checkpoint)
        export_module._load_checkpoint(
            _StubModel(),
            checkpoint,
            expected_sha256=digest_file_sha256(checkpoint),
        )

    def test_a_mismatched_digest_aborts_the_export(
        self, tmp_path: Path, export_module: Any
    ) -> None:
        checkpoint = tmp_path / "final.pt"
        torch.save({"weight": torch.zeros(2)}, checkpoint)
        with pytest.raises(ValueError, match="SHA-256 verification failed"):
            export_module._load_checkpoint(_StubModel(), checkpoint, expected_sha256="00" * 32)

    def test_a_missing_checkpoint_still_raises_file_not_found(
        self, tmp_path: Path, export_module: Any
    ) -> None:
        with pytest.raises(FileNotFoundError):
            export_module._load_checkpoint(_StubModel(), tmp_path / "absent.pt")

    def test_the_silent_legacy_pickle_fallback_is_gone(self, export_module: Any) -> None:
        """No ``try``/``except`` retry, and ``weights_only`` is always ``True``.

        Asserted over the parsed function, not its source text, so the
        docstring that explains why the fallback was removed cannot satisfy
        the test by mentioning it.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(export_module._load_checkpoint)))
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        weights_only_values = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "weights_only"
        ]
        assert weights_only_values
        assert all(
            isinstance(value, ast.Constant) and value.value is True for value in weights_only_values
        )

    def test_the_cli_exposes_the_digest_flag(self, export_module: Any) -> None:
        args = export_module._parse_args(["--output", "out.onnx", "--checkpoint-sha256", "ab" * 32])
        assert args.checkpoint_sha256 == "ab" * 32


class TestMetadataSidecar:
    def test_writes_the_sidecar_beside_the_artifact(
        self, tmp_path: Path, export_module: Any
    ) -> None:
        artifact = tmp_path / "observe_step.onnx"
        artifact.write_bytes(b"onnx-graph-bytes")
        sidecar = export_module._write_metadata_sidecar(_args(output=artifact), _cfg())
        assert sidecar.parent == tmp_path
        assert sidecar.name == WorldModelConfig.model_validate({}).onnx_metadata_filename
        assert sidecar.is_file()

    def test_records_the_artifact_digest(self, tmp_path: Path, export_module: Any) -> None:
        from mousedroid.common.hashing import digest_file_sha256

        artifact = tmp_path / "observe_step.onnx"
        artifact.write_bytes(b"onnx-graph-bytes")
        sidecar = export_module._write_metadata_sidecar(_args(output=artifact), _cfg())
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert record["artifact"]["sha256"] == digest_file_sha256(artifact)
        assert record["artifact"]["filename"] == "observe_step.onnx"

    def test_records_the_checkpoint_digest_when_one_was_used(
        self, tmp_path: Path, export_module: Any
    ) -> None:
        from mousedroid.common.hashing import digest_file_sha256

        artifact = tmp_path / "observe_step.onnx"
        artifact.write_bytes(b"onnx-graph-bytes")
        checkpoint = tmp_path / "final.pt"
        torch.save({"weight": torch.zeros(2)}, checkpoint)
        sidecar = export_module._write_metadata_sidecar(
            _args(output=artifact, checkpoint=checkpoint), _cfg()
        )
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert record["checkpoint"]["sha256"] == digest_file_sha256(checkpoint)

    def test_a_weightless_export_is_recorded_as_such(
        self, tmp_path: Path, export_module: Any
    ) -> None:
        artifact = tmp_path / "observe_step.onnx"
        artifact.write_bytes(b"onnx-graph-bytes")
        sidecar = export_module._write_metadata_sidecar(_args(output=artifact), _cfg())
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert record["checkpoint"] == {"path": None, "sha256": None}

    def test_records_the_opset_and_the_io_contract(
        self, tmp_path: Path, export_module: Any
    ) -> None:
        artifact = tmp_path / "observe_step.onnx"
        artifact.write_bytes(b"onnx-graph-bytes")
        sidecar = export_module._write_metadata_sidecar(_args(output=artifact), _cfg())
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert record["opset"] == _OPSET
        assert list(record["outputs"]) == ["new_h", "new_z", "obs_embed", "surprise"]
        assert "lidar" not in record["inputs"]

    def test_records_the_tool_versions(self, tmp_path: Path, export_module: Any) -> None:
        artifact = tmp_path / "observe_step.onnx"
        artifact.write_bytes(b"onnx-graph-bytes")
        sidecar = export_module._write_metadata_sidecar(_args(output=artifact), _cfg())
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert record["tool_versions"]["torch"]

    def test_the_sidecar_filename_comes_from_config_not_a_literal(
        self, tmp_path: Path, export_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator repointing the schema field moves the sidecar."""
        settings_stub = type(
            "_Settings",
            (),
            {"world_model": WorldModelConfig(onnx_metadata_filename="provenance.json")},
        )()
        monkeypatch.setattr(
            "mousedroid.config.loader.load_settings",
            lambda _path: settings_stub,
        )
        artifact = tmp_path / "observe_step.onnx"
        artifact.write_bytes(b"onnx-graph-bytes")
        sidecar = export_module._write_metadata_sidecar(
            _args(output=artifact, config=tmp_path / "overlay.yaml"), _cfg()
        )
        assert sidecar.name == "provenance.json"


class TestGitShaResolution:
    def test_returns_none_when_git_is_absent(
        self, export_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(export_module.shutil, "which", lambda _name: None)
        assert export_module._resolve_git_sha() is None

    def test_returns_none_when_git_exits_nonzero(
        self, export_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        monkeypatch.setattr(export_module.shutil, "which", lambda _name: "/usr/bin/git")
        monkeypatch.setattr(
            export_module.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=_GIT_FAILURE_RETURNCODE, stdout="", stderr=""
            ),
        )
        assert export_module._resolve_git_sha() is None

    def test_returns_none_when_the_process_cannot_start(
        self, export_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: Any, **_k: Any) -> None:
            raise OSError("no exec")

        monkeypatch.setattr(export_module.shutil, "which", lambda _name: "/usr/bin/git")
        monkeypatch.setattr(export_module.subprocess, "run", _boom)
        assert export_module._resolve_git_sha() is None

    def test_returns_the_trimmed_sha(
        self, export_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        monkeypatch.setattr(export_module.shutil, "which", lambda _name: "/usr/bin/git")
        monkeypatch.setattr(
            export_module.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=0, stdout="abc123\n", stderr=""
            ),
        )
        assert export_module._resolve_git_sha() == "abc123"
