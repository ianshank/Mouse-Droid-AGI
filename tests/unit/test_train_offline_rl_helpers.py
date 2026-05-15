"""Phase 2.1 — unit tests for ``train_offline_rl`` helper functions.

These tests isolate ``_resolve_real_replay_dataset`` so each of its four
branches is exercised independently of the full ``train_offline_rl`` flow.
Integration coverage of the same paths lives in
``tests/integration/test_train_offline_rl_mixer.py``; the unit tests below
guarantee the changed-line coverage gate (``scripts/check_branch_coverage.py``,
min 85%) catches drift even when the integration tests are skipped (e.g.
on hardware that cannot open LMDB stores).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from mousedroid.config.schema import (
    ExperienceConfig,
    GPUConfig,
    OfflineRLConfig,
    Settings,
    TrainingConfig,
    TrainingReplayConfig,
)

if TYPE_CHECKING:
    pass


_SIM_PATH = "experience/sim"
_REAL_PATH = "experience/real"
_DEVICE = "cpu"


def _make_settings(
    *,
    sim_path: str = _SIM_PATH,
    replay_enabled: bool,
    replay_source_path: str | None,
) -> Settings:
    """Build a minimal :class:`Settings` for helper-function tests."""
    return Settings(
        mock_hardware=True,
        experience=ExperienceConfig(path=sim_path),
        offline_rl=OfflineRLConfig(),
        training=TrainingConfig(
            replay=TrainingReplayConfig(
                enabled=replay_enabled,
                source_path=replay_source_path,
            ),
            gpu=GPUConfig(device=_DEVICE, require_cuda=False),
        ),
    )


class TestResolveRealReplayDatasetEarlyReturns:
    """Each branch of ``_resolve_real_replay_dataset`` returns ``None`` distinctly."""

    def test_replay_disabled_returns_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        from training.train_offline_rl import _resolve_real_replay_dataset

        cfg = _make_settings(replay_enabled=False, replay_source_path=_REAL_PATH)
        result = _resolve_real_replay_dataset(cfg, _DEVICE)

        assert result is None
        captured = capsys.readouterr()
        # Debug log is gated by structlog's filtering level; just assert no
        # OfflineRLDataset was opened (i.e., no INFO `offline_dataset_opened`).
        assert "offline_dataset_opened" not in captured.out

    def test_empty_source_path_returns_none(self) -> None:
        from training.train_offline_rl import _resolve_real_replay_dataset

        cfg = _make_settings(replay_enabled=True, replay_source_path=None)
        result = _resolve_real_replay_dataset(cfg, _DEVICE)

        assert result is None

    def test_empty_source_path_when_empty_string_returns_none(self) -> None:
        """Empty string is falsy — must short-circuit same as ``None``."""
        from training.train_offline_rl import _resolve_real_replay_dataset

        cfg = _make_settings(replay_enabled=True, replay_source_path="")
        result = _resolve_real_replay_dataset(cfg, _DEVICE)

        assert result is None

    def test_identical_path_returns_none(self) -> None:
        from training.train_offline_rl import _resolve_real_replay_dataset

        cfg = _make_settings(
            sim_path=_SIM_PATH,
            replay_enabled=True,
            replay_source_path=_SIM_PATH,
        )
        result = _resolve_real_replay_dataset(cfg, _DEVICE)

        assert result is None


class TestResolveRealReplayDatasetOpensDataset:
    """When all preconditions are met, a fresh dataset is opened over the real path."""

    def test_opens_dataset_with_distinct_path(self) -> None:
        """The happy path constructs an ``OfflineRLDataset`` for the real LMDB.

        We patch the dataset class at the module boundary so the test does
        not touch the filesystem — the function under test is the helper, not
        the dataset.
        """
        from training import train_offline_rl as module

        cfg = _make_settings(replay_enabled=True, replay_source_path=_REAL_PATH)

        with patch.object(module, "OfflineRLDataset") as mock_dataset_cls:
            mock_instance = MagicMock()
            mock_dataset_cls.return_value = mock_instance

            result = module._resolve_real_replay_dataset(cfg, _DEVICE)

            assert result is mock_instance
            # Verify the dataset was constructed with the redirected path.
            call_kwargs = mock_dataset_cls.call_args.kwargs
            assert call_kwargs["experience_cfg"].path == _REAL_PATH
            # Other ExperienceConfig fields must be inherited from cfg.experience.
            assert call_kwargs["experience_cfg"].map_size_gb == cfg.experience.map_size_gb
            assert call_kwargs["model_cfg"] is cfg.model
            mock_instance.open.assert_called_once()


class TestBuildMixedBatchIterator:
    """``_build_mixed_batch_iterator`` must request batches from both datasets."""

    def test_requests_batches_from_both_datasets(self) -> None:
        from training.train_offline_rl import _build_mixed_batch_iterator

        from mousedroid.training.replay.mixer import MixerConfig

        sim_dataset = MagicMock()
        real_dataset = MagicMock()
        sim_dataset.iterate_batches.return_value = iter([{"states": "sim_a"}])
        real_dataset.iterate_batches.return_value = iter([{"states": "real_a"}])

        mixer_cfg = MixerConfig(alpha_target=0.0)
        offline_cfg = OfflineRLConfig(batch_size=4, terminal_gap_s=2.0)

        it = _build_mixed_batch_iterator(
            sim_dataset=sim_dataset,
            real_dataset=real_dataset,
            offline_cfg=offline_cfg,
            mixer_cfg=mixer_cfg,
            epoch=7,
        )
        # Consume one item to drive the iterator setup.
        _ = next(it, None)

        # Both datasets must have been asked for an iterator with the same
        # batch_size/terminal_gap_s, and seed wired from the epoch.
        for ds in (sim_dataset, real_dataset):
            ds.iterate_batches.assert_called_once_with(
                batch_size=4,
                seed=7,
                terminal_gap_s=2.0,
            )


class TestResolveDeviceForwardsConfig:
    """Sanity-check that ``_resolve_device`` honors the GPU override."""

    def test_resolve_device_uses_cpu_override(self) -> None:
        from training.train_offline_rl import _resolve_device

        cfg = Settings(
            mock_hardware=True,
            training=TrainingConfig(gpu=GPUConfig(device="cpu", require_cuda=False)),
        )
        device = _resolve_device(cfg)
        assert device == "cpu"
