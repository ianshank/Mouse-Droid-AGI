"""Tests for training.run_pipeline — configuration and orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from training.run_pipeline import _load_training_settings

from mousedroid.config.schema import Settings


def test_load_training_settings_forces_mock_hardware(tmp_path: Path) -> None:
    """Training CLI preserves legacy mock-hardware forcing for synthetic runs."""
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "mock_hardware": False,
                "debug": True,
                "training": {"batch_size": 16},
            }
        )
    )

    settings = _load_training_settings(str(config_path))

    assert settings.mock_hardware is True
    assert settings.debug is True
    assert settings.training.batch_size == 16


def test_load_training_settings_honours_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared loader semantics remain active for top-level env overrides."""
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.dump({"debug": False}))
    monkeypatch.setenv("MOUSEDROID_DEBUG", "true")

    settings = _load_training_settings(str(config_path))

    assert settings.debug is True


class TestRunPipeline:
    """Tests for run_pipeline()."""

    @pytest.fixture
    def cfg(self) -> Settings:
        """Create test settings with mock hardware."""
        return Settings.model_validate(
            {
                "mock_hardware": True,
                "training": {
                    "epochs": 2,
                    "n_episodes": 5,
                    "sequence_length": 10,
                    "batch_size": 4,
                },
            }
        )

    @patch("training.run_pipeline.run_upload")
    @patch("training.run_pipeline.run_phase_4_constitutional_rl")
    @patch("training.run_pipeline.run_phase_3_bdi")
    @patch("training.run_pipeline.run_phase_2_warmstart")
    @patch("training.run_pipeline.run_phase_1_rssm")
    @patch("training.run_pipeline.run_phase_0b_annotations")
    @patch("training.run_pipeline.run_phase_0_data_gen")
    def test_full_pipeline_calls_all_phases(
        self,
        mock_p0: MagicMock,
        mock_p0b: MagicMock,
        mock_p1: MagicMock,
        mock_p2: MagicMock,
        mock_p3: MagicMock,
        mock_p4: MagicMock,
        mock_upload: MagicMock,
        cfg: Settings,
    ) -> None:
        """Full pipeline runs all phases sequentially."""
        from training.run_pipeline import run_pipeline

        mock_p0.return_value = Path("data/")
        mock_p0b.return_value = Path("data/bdi_annotations.npz")
        mock_p1.return_value = Path("weights/rssm/final.pt")
        mock_p3.return_value = Path("weights/bdi/")
        mock_p4.return_value = Path("weights/")

        run_pipeline(cfg)

        mock_p0.assert_called_once()
        mock_p0b.assert_called_once()
        mock_p1.assert_called_once()
        mock_p2.assert_called_once()
        mock_p3.assert_called_once()
        mock_p4.assert_called_once()
        mock_upload.assert_not_called()

    @patch("training.run_pipeline.run_upload")
    @patch("training.run_pipeline.run_phase_4_constitutional_rl")
    @patch("training.run_pipeline.run_phase_3_bdi")
    @patch("training.run_pipeline.run_phase_2_warmstart")
    @patch("training.run_pipeline.run_phase_1_rssm")
    @patch("training.run_pipeline.run_phase_0b_annotations")
    @patch("training.run_pipeline.run_phase_0_data_gen")
    def test_selective_phases(
        self,
        mock_p0: MagicMock,
        mock_p0b: MagicMock,
        mock_p1: MagicMock,
        mock_p2: MagicMock,
        mock_p3: MagicMock,
        mock_p4: MagicMock,
        mock_upload: MagicMock,
        cfg: Settings,
    ) -> None:
        """Only specified phases run when phases parameter is set."""
        from training.run_pipeline import run_pipeline

        mock_p1.return_value = Path("weights/rssm/final.pt")

        run_pipeline(cfg, phases={1})

        mock_p0.assert_not_called()
        mock_p0b.assert_not_called()
        mock_p1.assert_called_once()
        mock_p2.assert_not_called()
        mock_p3.assert_not_called()
        mock_p4.assert_not_called()

    @patch("training.run_pipeline.run_upload")
    @patch("training.run_pipeline.run_phase_4_constitutional_rl")
    @patch("training.run_pipeline.run_phase_3_bdi")
    @patch("training.run_pipeline.run_phase_2_warmstart")
    @patch("training.run_pipeline.run_phase_1_rssm")
    @patch("training.run_pipeline.run_phase_0b_annotations")
    @patch("training.run_pipeline.run_phase_0_data_gen")
    def test_upload_flag(
        self,
        mock_p0: MagicMock,
        mock_p0b: MagicMock,
        mock_p1: MagicMock,
        mock_p2: MagicMock,
        mock_p3: MagicMock,
        mock_p4: MagicMock,
        mock_upload: MagicMock,
        cfg: Settings,
    ) -> None:
        """Upload is called when upload=True."""
        from training.run_pipeline import run_pipeline

        mock_p0.return_value = Path("data/")
        mock_p0b.return_value = Path("data/bdi_annotations.npz")
        mock_p1.return_value = Path("weights/rssm/final.pt")
        mock_p3.return_value = Path("weights/bdi/")
        mock_p4.return_value = Path("weights/")

        run_pipeline(cfg, upload=True)

        mock_upload.assert_called_once()

    def test_phase_2_requires_existing_rssm_checkpoint_when_phase_1_skipped(
        self,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """Selective Phase 2 runs should fail clearly without a prior RSSM artifact."""
        from training.run_pipeline import run_pipeline

        cfg.training.data_dir = str(tmp_path)
        cfg.training.weights_dir = str(tmp_path / "weights")

        with pytest.raises(FileNotFoundError, match="Phase 2 requires RSSM checkpoint"):
            run_pipeline(cfg, phases={2})

    def test_phase_3_requires_existing_annotations_when_phase_0_skipped(
        self,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """Selective Phase 3 runs should fail clearly without collected annotations."""
        from training.run_pipeline import run_pipeline

        cfg.training.data_dir = str(tmp_path)
        cfg.training.weights_dir = str(tmp_path / "weights")

        with pytest.raises(FileNotFoundError, match="Phase 3 requires annotation dataset"):
            run_pipeline(cfg, phases={3})

    @patch("training.run_pipeline.run_phase_1_rssm")
    def test_resume_from_is_forwarded_to_phase_1(
        self,
        mock_p1: MagicMock,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """run_pipeline should forward resume_from to Phase 1 RSSM training."""
        from training.run_pipeline import run_pipeline

        cfg.training.data_dir = str(tmp_path)
        (tmp_path / "sequences.pt").write_bytes(b"fake")
        resume_path = tmp_path / "resume.pt"
        resume_path.write_bytes(b"checkpoint")
        mock_p1.return_value = tmp_path / "weights" / "rssm" / "final.pt"

        run_pipeline(cfg, phases={1}, resume_from=str(resume_path))

        _, kwargs = mock_p1.call_args
        assert kwargs["resume_from"] == resume_path

    @patch("training.run_pipeline.run_phase_1_rssm")
    def test_config_resume_from_is_forwarded_when_cli_resume_missing(
        self,
        mock_p1: MagicMock,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """run_pipeline should honor cfg.training.resume_from when CLI resume is absent."""
        from training.run_pipeline import run_pipeline

        cfg.training.data_dir = str(tmp_path)
        cfg.training.resume_from = str(tmp_path / "config_resume.pt")
        (tmp_path / "sequences.pt").write_bytes(b"fake")
        Path(cfg.training.resume_from).write_bytes(b"checkpoint")
        mock_p1.return_value = tmp_path / "weights" / "rssm" / "final.pt"

        run_pipeline(cfg, phases={1})

        _, kwargs = mock_p1.call_args
        assert kwargs["resume_from"] == Path(cfg.training.resume_from)

    @patch("training.train_bdi.train_bdi")
    def test_phase_3_uses_training_hyperparameters(
        self,
        mock_train_bdi: MagicMock,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """Phase 3 should forward configured BDI training hyperparameters."""
        from training.run_pipeline import run_phase_3_bdi

        cfg.training.weights_dir = str(tmp_path / "weights")
        cfg.training.learning_rate = 1e-3
        cfg.training.epochs = 7
        cfg.training.batch_size = 5
        cfg.training.gradient_scale = 3.5
        annotations_path = tmp_path / "bdi_annotations.npz"
        mock_train_bdi.return_value = tmp_path / "weights" / "bdi"

        result = run_phase_3_bdi(cfg, annotations_path)

        assert result == mock_train_bdi.return_value
        mock_train_bdi.assert_called_once_with(
            annotations_path,
            output_dir=tmp_path / "weights" / "bdi",
            lr=1e-3,
            epochs=7,
            batch_size=5,
            gradient_scale=3.5,
        )

    @patch("training.run_pipeline.run_phase_2_warmstart")
    @patch("training.run_pipeline.run_phase_0b_annotations")
    @patch("training.run_pipeline.run_phase_0_data_gen")
    def test_phase_0_and_2_runs_without_prior_rssm_artifact_if_phase_1_skipped(
        self,
        mock_p0: MagicMock,
        mock_p0b: MagicMock,
        mock_p2: MagicMock,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """Running phases {0,2} should still require an existing RSSM checkpoint."""
        from training.run_pipeline import run_pipeline

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mock_p0.return_value = data_dir
        mock_p0b.return_value = data_dir / "bdi_annotations.npz"
        cfg.training.weights_dir = str(tmp_path / "weights")

        with pytest.raises(FileNotFoundError, match="Phase 2 requires RSSM checkpoint"):
            run_pipeline(cfg, phases={0, 2})

        mock_p2.assert_not_called()

    def test_phases_2_3_4_require_missing_upstream_artifacts(
        self,
        cfg: Settings,
        tmp_path: Path,
    ) -> None:
        """Running phases {2,3,4} should fail on the earliest missing upstream artifact."""
        from training.run_pipeline import run_pipeline

        cfg.training.data_dir = str(tmp_path)
        cfg.training.weights_dir = str(tmp_path / "weights")

        with pytest.raises(FileNotFoundError, match="Phase 2 requires RSSM checkpoint"):
            run_pipeline(cfg, phases={2, 3, 4})
