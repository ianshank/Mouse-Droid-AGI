"""Tests for Phase 2.2 — MCTS policy warm-start."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mousedroid.cognitive.constitutional_rl import PolicyMLP, ValueMLP
from mousedroid.config.schema import Settings
from mousedroid.constants import DEFAULT_ACTION_DIM, DEFAULT_BELIEF_DIM


class TestPolicyMLPSaveLoad:
    """Test save/load for PolicyMLP."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        policy = PolicyMLP(input_dim=64, action_dim=3)
        path = tmp_path / "policy.npz"
        policy.save(path)
        assert path.exists()

    def test_roundtrip_preserves_weights(self, tmp_path: Path) -> None:
        policy = PolicyMLP(input_dim=64, action_dim=3)
        path = tmp_path / "policy.npz"
        policy.save(path)

        policy2 = PolicyMLP(input_dim=64, action_dim=3)
        policy2.load(path)

        np.testing.assert_array_equal(policy._w1, policy2._w1)
        np.testing.assert_array_equal(policy._b1, policy2._b1)
        np.testing.assert_array_equal(policy._w2, policy2._w2)
        np.testing.assert_array_equal(policy._b2, policy2._b2)

    def test_loaded_policy_same_output(self, tmp_path: Path) -> None:
        policy = PolicyMLP(input_dim=64, action_dim=3)
        state = np.random.randn(64).astype(np.float32)
        out1 = policy.forward(state)

        path = tmp_path / "policy.npz"
        policy.save(path)

        policy2 = PolicyMLP(input_dim=64, action_dim=3)
        policy2.load(path)
        out2 = policy2.forward(state)

        np.testing.assert_allclose(out1, out2, atol=1e-6)


class TestValueMLPSaveLoad:
    """Test save/load for ValueMLP."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        value_fn = ValueMLP(input_dim=64)
        path = tmp_path / "value.npz"
        value_fn.save(path)
        assert path.exists()

    def test_roundtrip_preserves_weights(self, tmp_path: Path) -> None:
        value_fn = ValueMLP(input_dim=64)
        path = tmp_path / "value.npz"
        value_fn.save(path)

        value_fn2 = ValueMLP(input_dim=64)
        value_fn2.load(path)

        np.testing.assert_array_equal(value_fn._w1, value_fn2._w1)
        np.testing.assert_array_equal(value_fn._b1, value_fn2._b1)

    def test_loaded_value_same_output(self, tmp_path: Path) -> None:
        value_fn = ValueMLP(input_dim=64)
        state = np.random.randn(64).astype(np.float32)
        out1 = value_fn.forward(state)

        path = tmp_path / "value.npz"
        value_fn.save(path)

        value_fn2 = ValueMLP(input_dim=64)
        value_fn2.load(path)
        out2 = value_fn2.forward(state)

        assert abs(out1 - out2) < 1e-6


class TestWarmstartPolicy:
    """Test policy weight initialization from latent statistics."""

    def test_warmstart_produces_valid_policy(self) -> None:
        from training.warmstart_policy import warmstart_policy

        latent_mean = np.random.randn(64).astype(np.float32)
        latent_std = np.abs(np.random.randn(64).astype(np.float32)) + 0.1

        policy = warmstart_policy(latent_mean, latent_std, input_dim=64, action_dim=3)
        state = np.random.randn(64).astype(np.float32)
        action = policy.forward(state)

        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0)  # tanh output

    def test_warmstart_defaults_match_shared_dimensions(self) -> None:
        from training.warmstart_policy import warmstart_policy

        latent_mean = np.zeros(DEFAULT_BELIEF_DIM, dtype=np.float32)
        latent_std = np.ones(DEFAULT_BELIEF_DIM, dtype=np.float32)

        policy = warmstart_policy(latent_mean, latent_std)

        assert policy._w1.shape[0] == DEFAULT_BELIEF_DIM
        assert policy._w2.shape[1] == DEFAULT_ACTION_DIM

    def test_tune_ucb_uses_config_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from training import warmstart_policy as warmstart_module

        candidates = [0.25, 0.75]
        base_cfg = warmstart_module.MCTSConfig(ucb_candidates=candidates, ucb_target_ms=1.0)

        class _FakePlanner:
            def __init__(self, cfg: warmstart_module.MCTSConfig, _rssm: object) -> None:
                self.ucb_c = cfg.ucb_c

            def plan(self, _h: object, _z: object) -> torch.Tensor:
                return torch.zeros(1, 3)

        class _FakeRssm:
            class _Cfg:
                hidden_dim = 4
                latent_dim = 4

            _cfg = _Cfg()

            def imagine_step(
                self,
                _action: torch.Tensor,
                h: torch.Tensor,
                z: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                reward = torch.tensor([float(h[0, 0].item())])
                return h, z, reward

        import torch

        seen: list[float] = []

        class _RecordingPlanner(_FakePlanner):
            def __init__(self, cfg: warmstart_module.MCTSConfig, rssm: object) -> None:
                super().__init__(cfg, rssm)
                seen.append(cfg.ucb_c)

        monkeypatch.setattr(warmstart_module, "MCTSPlanner", _RecordingPlanner)

        best_ucb, results = warmstart_module.tune_ucb(
            _FakeRssm(),
            base_cfg,
            n_episodes=1,
            n_simulations=1,
            target_ms=1_000.0,
        )

        assert seen == candidates
        assert best_ucb in candidates
        assert "ucb_0.25" in results
        assert "ucb_0.75" in results


class TestComputeLatentStatistics:
    """Test latent statistics computation from RSSM + dataset."""

    def test_returns_correct_shapes(self) -> None:
        from training.warmstart_policy import compute_latent_statistics

        latent_dim = 8

        # Build a minimal fake RSSM
        class _FakeRSSM:
            class _Cfg:
                hidden_dim = 8
                latent_dim = 8

            _cfg = _Cfg()

            def encoder(
                self, v: torch.Tensor, u: torch.Tensor, m: torch.Tensor, mask: torch.Tensor
            ) -> torch.Tensor:
                return torch.randn(v.shape[0], latent_dim)

            gru = MagicMock(side_effect=lambda x, h: torch.randn_like(h))

            def posterior(self, x: torch.Tensor) -> torch.Tensor:
                return torch.randn(x.shape[0], latent_dim * 2)

            @staticmethod
            def _sample_gaussian(
                params: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                half = params.shape[-1] // 2
                mean = params[..., :half]
                logvar = params[..., half:]
                return mean, mean, logvar

        # Build a fake dataset that yields one episode
        class _FakeDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(
                self, idx: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                seq_len = 3
                return (
                    torch.randn(seq_len, 256),  # vision
                    torch.randn(seq_len, 1),  # ultrasonic
                    torch.randn(seq_len, 4),  # motor_state
                    torch.ones(seq_len, 1),  # valid_mask
                    torch.randn(seq_len, 3),  # actions
                )

        rssm = _FakeRSSM()
        dataset = _FakeDataset()
        device = torch.device("cpu")

        mean, std = compute_latent_statistics(rssm, dataset, device, max_episodes=1)

        assert mean.shape == (latent_dim,)
        assert std.shape == (latent_dim,)
        assert np.all(std > 0), "std should be positive (includes epsilon)"


class TestRunWarmstart:
    """Test the full run_warmstart orchestration."""

    @pytest.fixture
    def cfg(self) -> Settings:
        return Settings(
            mock_hardware=True,
            training={
                "epochs": 2,
                "n_episodes": 5,
                "sequence_length": 3,
                "batch_size": 2,
            },
        )

    @patch("training.warmstart_policy.tune_ucb")
    @patch("training.warmstart_policy.compute_latent_statistics")
    @patch("training.warmstart_policy.RSSMSequenceDataset")
    @patch("training.warmstart_policy.torch.load")
    @patch("training.warmstart_policy.RSSM")
    def test_run_warmstart_creates_artifacts(
        self,
        mock_rssm_cls: MagicMock,
        mock_torch_load: MagicMock,
        mock_dataset_cls: MagicMock,
        mock_compute: MagicMock,
        mock_tune: MagicMock,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """run_warmstart should create policy_init.npz and tuned_config.json."""
        from training.warmstart_policy import run_warmstart

        latent_dim = cfg.model.latent_dim

        # Mock RSSM: returns an object with .to().eval()
        mock_rssm = MagicMock()
        mock_rssm.to.return_value = mock_rssm
        mock_rssm_cls.return_value = mock_rssm
        mock_torch_load.return_value = {}

        # Mock latent stats
        mock_compute.return_value = (
            np.zeros(latent_dim, dtype=np.float32),
            np.ones(latent_dim, dtype=np.float32),
        )

        # Mock UCB tuning
        mock_tune.return_value = (1.41, {"best_ucb_c": 1.41})

        # Create fake checkpoint
        ckpt = tmp_path / "rssm" / "final.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"fake")

        data_path = tmp_path / "sequences.pt"
        data_path.write_bytes(b"fake")

        output_dir = tmp_path / "mcts"

        run_warmstart(cfg, ckpt, data_path, output_dir=output_dir)

        assert (output_dir / "policy_init.npz").exists()
        assert (output_dir / "tuned_config.json").exists()
        mock_compute.assert_called_once()
        mock_tune.assert_called_once()

    @patch("training.warmstart_policy.tune_ucb")
    @patch("training.warmstart_policy.compute_latent_statistics")
    @patch("training.warmstart_policy.RSSMSequenceDataset")
    @patch("training.warmstart_policy.torch.load")
    @patch("training.warmstart_policy.RSSM")
    def test_run_warmstart_passes_ucb_target_from_config(
        self,
        mock_rssm_cls: MagicMock,
        mock_torch_load: MagicMock,
        mock_dataset_cls: MagicMock,
        mock_compute: MagicMock,
        mock_tune: MagicMock,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """run_warmstart should pass cfg.mcts.ucb_target_ms to tune_ucb."""
        from training.warmstart_policy import run_warmstart

        mock_rssm = MagicMock()
        mock_rssm.to.return_value = mock_rssm
        mock_rssm_cls.return_value = mock_rssm
        mock_torch_load.return_value = {}
        mock_compute.return_value = (
            np.zeros(cfg.model.latent_dim, dtype=np.float32),
            np.ones(cfg.model.latent_dim, dtype=np.float32),
        )
        mock_tune.return_value = (1.41, {"best_ucb_c": 1.41})

        ckpt = tmp_path / "rssm" / "final.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"fake")
        data_path = tmp_path / "sequences.pt"
        data_path.write_bytes(b"fake")

        cfg.mcts.ucb_target_ms = 99.0
        run_warmstart(cfg, ckpt, data_path, output_dir=tmp_path / "mcts")

        _, kwargs = mock_tune.call_args
        assert kwargs.get("target_ms") == 99.0 or mock_tune.call_args[0][-1] == 99.0
