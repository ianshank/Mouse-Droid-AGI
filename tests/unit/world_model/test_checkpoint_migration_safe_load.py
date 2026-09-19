"""``load_rssm_with_migration``'s deserialisation fence (Phase 7, task 7.1b).

The pre-gate loader called ``torch.load(..., weights_only=False)`` with no
preceding digest check — an arbitrary-code path on any ``.pt`` that reached the
rover, and the exact site the runtime spec names. These tests pin the fence:

* the safe mode is the default and every project-written checkpoint loads under
  it (``rssm_pretrainer`` saves a bare ``state_dict()``, so there is nothing
  non-tensor to lose);
* a digest, when supplied, is verified *before* deserialisation;
* the unsafe mode cannot be reached by omission — requesting it without a
  digest is refused rather than honoured.

Torch only: no ONNX, so this runs in the blocking ``test`` job.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from mousedroid.common.hashing import digest_file_sha256
from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.checkpoint_migration import (
    CheckpointIntegrityError,
    load_rssm_with_migration,
)
from mousedroid.world_model.rssm import RSSM


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
        }
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A real RSSM state dict, saved the way the pretrainer saves one."""
    cfg = _cfg()
    path = tmp_path / "rssm.pt"
    torch.save(RSSM(cfg).state_dict(), path)
    return path


class TestSafeLoadIsTheDefault:
    def test_a_project_written_checkpoint_loads_unchanged(self, checkpoint: Path) -> None:
        model = load_rssm_with_migration(checkpoint, _cfg())
        assert isinstance(model, RSSM)

    def test_a_training_checkpoint_wrapper_loads_unchanged(self, tmp_path: Path) -> None:
        cfg = _cfg()
        path = tmp_path / "wrapped.pt"
        torch.save({"model_state_dict": RSSM(cfg).state_dict()}, path)
        assert isinstance(load_rssm_with_migration(path, cfg), RSSM)

    def test_the_loader_never_defaults_to_arbitrary_pickle(self) -> None:
        """``allow_unsafe_pickle`` defaults to ``False`` — checked on the signature."""
        signature = inspect.signature(load_rssm_with_migration)
        assert signature.parameters["allow_unsafe_pickle"].default is False
        assert signature.parameters["expected_sha256"].default is None

    def test_no_bare_weights_only_false_literal_remains(self) -> None:
        """The unsafe mode must be derived from the flag, never hardcoded on."""
        from mousedroid.world_model import checkpoint_migration

        source = Path(checkpoint_migration.__file__).read_text(encoding="utf-8")
        constants = [
            keyword.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "weights_only" and isinstance(keyword.value, ast.Constant)
        ]
        assert not [value for value in constants if value.value is False]


class TestDigestVerification:
    def test_a_matching_digest_is_accepted(self, checkpoint: Path) -> None:
        model = load_rssm_with_migration(
            checkpoint, _cfg(), expected_sha256=digest_file_sha256(checkpoint)
        )
        assert isinstance(model, RSSM)

    def test_a_mismatched_digest_refuses_the_load(self, checkpoint: Path) -> None:
        with pytest.raises(CheckpointIntegrityError, match="SHA-256 verification failed"):
            load_rssm_with_migration(checkpoint, _cfg(), expected_sha256="00" * 32)

    def test_a_malformed_digest_refuses_the_load(self, checkpoint: Path) -> None:
        """``verify_sha256`` fails closed on a malformed expectation."""
        with pytest.raises(CheckpointIntegrityError):
            load_rssm_with_migration(checkpoint, _cfg(), expected_sha256="not-a-digest")


class TestUnsafePickleFence:
    def test_unsafe_pickle_without_a_digest_is_refused(self, checkpoint: Path) -> None:
        with pytest.raises(CheckpointIntegrityError, match="requires expected_sha256"):
            load_rssm_with_migration(checkpoint, _cfg(), allow_unsafe_pickle=True)

    def test_unsafe_pickle_with_a_matching_digest_is_permitted(self, checkpoint: Path) -> None:
        model = load_rssm_with_migration(
            checkpoint,
            _cfg(),
            expected_sha256=digest_file_sha256(checkpoint),
            allow_unsafe_pickle=True,
        )
        assert isinstance(model, RSSM)

    def test_unsafe_pickle_with_a_bad_digest_is_refused(self, checkpoint: Path) -> None:
        with pytest.raises(CheckpointIntegrityError):
            load_rssm_with_migration(
                checkpoint,
                _cfg(),
                expected_sha256="11" * 32,
                allow_unsafe_pickle=True,
            )

    def test_the_refusal_is_not_an_assert(self) -> None:
        """``PYTHONOPTIMIZE=1`` in ``Dockerfile.jetson`` would strip an assert."""
        from mousedroid.world_model import checkpoint_migration

        source = Path(checkpoint_migration.__file__).read_text(encoding="utf-8")
        asserts = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assert)]
        assert not asserts


class TestNonDictCheckpointStillRejected:
    def test_a_tensor_only_checkpoint_raises_type_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bare_tensor.pt"
        torch.save(torch.zeros(3), path)
        with pytest.raises(TypeError, match="expected dict"):
            load_rssm_with_migration(path, _cfg())
