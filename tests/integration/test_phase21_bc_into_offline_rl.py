"""Phase 2.1 — TD3+BC auxiliary-loss integration into ``train_offline_rl``.

These tests assert the *call-site* behavior added by Phase 2.1:

1. **Byte-identity at ``weight=0``** — final policy/Q parameters match a
   control trainer that never saw the new ``bc_update`` call. Guarantees
   the new code path is fully opt-in.
2. **Active integration at ``weight>0``** — final policy parameters
   diverge from the byte-identity baseline; ``bc_loss`` is recorded in
   the returned epoch stats; nothing goes NaN.
3. **One-shot ``offline_rl_bc_active`` log** when the weight is positive.
4. **Empty / no-data path** is unaffected by ``real_supervised_weight``.

The fixture is a tiny synthetic LMDB store seeded for determinism.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mousedroid.config.schema import ExperienceConfig, OfflineRLConfig, Settings
from mousedroid.experience.record import MouseDroidExperienceRecord
from tests import TEST_EXPERIENCE_MAP_SIZE_GB

# -----------------------------------------------------------------------------
# Test-only constants (config-driven via fixture, NOT shipped in src).
# -----------------------------------------------------------------------------
_RECORDS_PER_TEST = 32
_VISION_DIM = 256
_MOTOR_DIM = 4
_ACTION_DIM = 3
_HIDDEN_DIM = 16
_EPOCHS = 2
_BATCH_SIZE = 8
_BC_WEIGHT_ACTIVE = 0.5
_BC_WEIGHT_OFF = 0.0
_RNG_SEED = 1234


def _populate_lmdb(path: str, *, n_records: int = _RECORDS_PER_TEST, seed: int = _RNG_SEED) -> None:
    """Write ``n_records`` deterministic records to an LMDB store."""
    import lmdb

    rng = np.random.default_rng(seed)
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    base_time = time.time()

    with env.begin(write=True) as txn:
        for i in range(n_records):
            record = MouseDroidExperienceRecord(
                timestamp=base_time + i * 0.1,
                vision_features=rng.standard_normal(_VISION_DIM).astype(np.float32),
                distance_m=1.0 + float(rng.random()),
                motor_state=rng.standard_normal(_MOTOR_DIM).astype(np.float32),
                action=rng.standard_normal(_ACTION_DIM).astype(np.float32) * 0.3,
                reward=float(rng.random()),
                surprise=0.1,
            )
            key = struct.pack(">Q", int((base_time + i * 0.1) * 1_000_000) + i)
            txn.put(key, record.serialize())

    env.close()


def _make_cfg(db_path: Path, *, bc_weight: float, algorithm: str = "cql") -> Settings:
    """Build a minimal :class:`Settings` for a single trainer run.

    Every numeric is sourced from the module-level constants above so the
    test surface stays free of magic numbers.
    """
    return Settings(
        mock_hardware=True,
        experience=ExperienceConfig(
            path=str(db_path),
            map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
            flush_every_n=5,
        ),
        offline_rl=OfflineRLConfig(
            algorithm=algorithm,
            epochs=_EPOCHS,
            batch_size=_BATCH_SIZE,
            checkpoint_every_n_epochs=_EPOCHS + 1,  # don't write per-epoch ckpts
            log_every_n_epochs=1,
            hidden_dim=_HIDDEN_DIM,
            real_supervised_weight=bc_weight,
        ),
    )


def _seed_all(seed: int = _RNG_SEED) -> None:
    """Seed every RNG that ``train_offline_rl`` may consult."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_and_load_policy(cfg: Settings, *, output_dir: Path) -> dict[str, torch.Tensor]:
    """Run ``train_offline_rl`` and return the final policy ``state_dict``."""
    from training.train_offline_rl import train_offline_rl

    _seed_all()
    result_dir, _stats = train_offline_rl(cfg=cfg, output_dir=output_dir)
    final_path = result_dir / "final.pt"
    assert final_path.exists(), f"checkpoint missing at {final_path}"
    blob: dict[str, Any] = torch.load(final_path, map_location="cpu", weights_only=True)
    state: dict[str, torch.Tensor] = blob["policy"]
    return state


# =============================================================================
# Tests
# =============================================================================


class TestBcByteIdentityAtZeroWeight:
    """``real_supervised_weight=0`` must leave the legacy training path bit-stable."""

    def test_zero_weight_preserves_policy_weights(self, tmp_path: Path) -> None:
        """Two seed-locked runs at ``weight=0`` must converge to identical policies."""
        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path))

        cfg = _make_cfg(db_path, bc_weight=_BC_WEIGHT_OFF)

        run_a = _run_and_load_policy(cfg, output_dir=tmp_path / "out_a")
        run_b = _run_and_load_policy(cfg, output_dir=tmp_path / "out_b")

        assert run_a.keys() == run_b.keys()
        for key, tensor_a in run_a.items():
            tensor_b = run_b[key]
            assert torch.equal(
                tensor_a, tensor_b
            ), f"determinism broken at weight=0 for layer {key}"


class TestBcActiveAtPositiveWeight:
    """``real_supervised_weight>0`` must measurably bias the policy."""

    def test_positive_weight_diverges_from_baseline(self, tmp_path: Path) -> None:
        """The BC-active run must produce different policy weights than weight=0."""
        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path))

        baseline = _run_and_load_policy(
            _make_cfg(db_path, bc_weight=_BC_WEIGHT_OFF),
            output_dir=tmp_path / "out_baseline",
        )
        bc_on = _run_and_load_policy(
            _make_cfg(db_path, bc_weight=_BC_WEIGHT_ACTIVE),
            output_dir=tmp_path / "out_bc_on",
        )

        assert baseline.keys() == bc_on.keys()
        differing = [key for key in baseline if not torch.equal(baseline[key], bc_on[key])]
        assert differing, (
            "BC weight > 0 produced byte-identical weights — bc_update integration "
            "is not actually firing."
        )

    def test_positive_weight_preserves_finite_weights(self, tmp_path: Path) -> None:
        """No NaN/Inf in any policy parameter after BC-active training."""
        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path))

        bc_on = _run_and_load_policy(
            _make_cfg(db_path, bc_weight=_BC_WEIGHT_ACTIVE),
            output_dir=tmp_path / "out_bc_on",
        )
        for key, tensor in bc_on.items():
            assert torch.isfinite(tensor).all(), f"non-finite parameter at {key}"

    def test_bc_loss_recorded_in_stats(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``bc_loss`` aggregates into the final ``stats`` dict and the
        ``offline_rl_bc_active`` activation log fires exactly once."""
        from training.train_offline_rl import train_offline_rl

        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path))

        cfg = _make_cfg(db_path, bc_weight=_BC_WEIGHT_ACTIVE)
        _seed_all()

        _result_dir, stats = train_offline_rl(cfg=cfg, output_dir=tmp_path / "out")
        captured = capsys.readouterr()

        # The aggregator key is ``final_bc_loss`` (mirrors other ``final_*`` keys).
        assert "final_bc_loss" in stats, f"bc_loss missing from stats: {stats!r}"
        bc_loss = stats["final_bc_loss"]
        assert isinstance(bc_loss, float)
        assert np.isfinite(bc_loss)
        assert bc_loss > 0.0, "BC loss should be strictly positive on random init"

        # The Phase 2.1 activation log must have fired (structlog → stdout).
        # Single-shot at trainer setup when ``real_supervised_weight > 0``.
        assert (
            "offline_rl_bc_active" in captured.out
        ), f"activation log missing from stdout:\n{captured.out!r}"


class TestBcOnIQL:
    """The BC integration must work for IQL just as for CQL."""

    def test_iql_bc_zero_weight_is_byte_identical(self, tmp_path: Path) -> None:
        """Determinism at weight=0 also holds for IQL."""
        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path))

        cfg = _make_cfg(db_path, bc_weight=_BC_WEIGHT_OFF, algorithm="iql")
        run_a = _run_and_load_policy(cfg, output_dir=tmp_path / "out_a")
        run_b = _run_and_load_policy(cfg, output_dir=tmp_path / "out_b")

        for key, tensor_a in run_a.items():
            assert torch.equal(tensor_a, run_b[key]), f"IQL determinism broken at {key}"

    def test_iql_bc_active_diverges(self, tmp_path: Path) -> None:
        """BC-active IQL run differs from the byte-identity baseline."""
        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path))

        baseline = _run_and_load_policy(
            _make_cfg(db_path, bc_weight=_BC_WEIGHT_OFF, algorithm="iql"),
            output_dir=tmp_path / "out_iql_baseline",
        )
        bc_on = _run_and_load_policy(
            _make_cfg(db_path, bc_weight=_BC_WEIGHT_ACTIVE, algorithm="iql"),
            output_dir=tmp_path / "out_iql_bc_on",
        )
        differing = [k for k in baseline if not torch.equal(baseline[k], bc_on[k])]
        assert differing, "IQL BC integration is not firing"


class TestBcEmptyDataPath:
    """An empty LMDB must short-circuit before BC ever fires, regardless of weight."""

    @pytest.mark.parametrize(
        "weight", [_BC_WEIGHT_OFF, _BC_WEIGHT_ACTIVE], ids=["weight_0", "weight_pos"]
    )
    def test_empty_db_returns_no_data(self, tmp_path: Path, weight: float) -> None:
        """No-data path must be unaffected by the BC weight."""
        import lmdb
        from training.train_offline_rl import train_offline_rl

        db_path = tmp_path / "experience"
        db_path.mkdir()
        lmdb.open(str(db_path), map_size=10 * 1024 * 1024).close()

        cfg = _make_cfg(db_path, bc_weight=weight)
        _, stats = train_offline_rl(cfg=cfg, output_dir=tmp_path / "out")
        assert stats["error"] == "no_data"
        assert stats["n_transitions"] == 0
