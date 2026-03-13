"""Tests for training.run_pipeline — pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mousedroid.config.schema import Settings

# ── Phase runners ────────────────────────────────────────────


class TestRunPipeline:
    """Tests for run_pipeline()."""

    @pytest.fixture
    def cfg(self) -> Settings:
        """Create test settings with mock hardware."""
        return Settings(
            mock_hardware=True,
            training={
                "epochs": 2,
                "n_episodes": 5,
                "sequence_length": 10,
                "batch_size": 4,
            },
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
