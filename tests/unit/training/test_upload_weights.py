"""Tests for training.upload_weights — HuggingFace upload + model card."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from training.upload_weights import _create_model_card, upload_weights


class TestUploadWeights:
    """Tests for upload_weights()."""

    @patch("training.upload_weights._HF_AVAILABLE", False)
    def test_no_huggingface_returns_false(self, tmp_path: Path) -> None:
        """Returns False when huggingface_hub is not installed."""
        result = upload_weights(tmp_path)
        assert result is False

    @patch("training.upload_weights._HF_AVAILABLE", True)
    def test_missing_dir_returns_false(self, tmp_path: Path) -> None:
        """Returns False when weights dir doesn't exist."""
        result = upload_weights(tmp_path / "nonexistent")
        assert result is False

    @patch("training.upload_weights._HF_AVAILABLE", True)
    def test_empty_dir_returns_false(self, tmp_path: Path) -> None:
        """Returns False when no weight files found."""
        result = upload_weights(tmp_path)
        assert result is False

    @patch("training.upload_weights.HfApi")
    @patch("training.upload_weights._HF_AVAILABLE", True)
    def test_upload_success(self, mock_api_cls: MagicMock, tmp_path: Path) -> None:
        """Successful upload returns True."""
        # Create some weight files
        (tmp_path / "rssm").mkdir()
        (tmp_path / "rssm" / "final.pt").write_bytes(b"fake")
        (tmp_path / "policy.npz").write_bytes(b"fake")

        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        result = upload_weights(tmp_path)

        assert result is True
        mock_api.upload_folder.assert_called_once()

    @patch("training.upload_weights.HfApi")
    @patch("training.upload_weights._HF_AVAILABLE", True)
    def test_upload_failure_returns_false(self, mock_api_cls: MagicMock, tmp_path: Path) -> None:
        """Upload failure returns False."""
        (tmp_path / "model.pt").write_bytes(b"fake")

        mock_api = MagicMock()
        mock_api.upload_folder.side_effect = RuntimeError("Network error")
        mock_api_cls.return_value = mock_api

        result = upload_weights(tmp_path)

        assert result is False


class TestCreateModelCard:
    """Tests for _create_model_card()."""

    def test_returns_valid_markdown(self, tmp_path: Path) -> None:
        """Model card contains expected sections."""
        card = _create_model_card(tmp_path, "ianshank/mousedroid-weights")
        assert "# ianshank/mousedroid-weights" in card
        assert "RSSM World Model" in card
        assert "BDI Belief" in card
        assert "Constitutional RL" in card

    def test_includes_mcts_tuning_metadata(self, tmp_path: Path) -> None:
        """Model card works with tuning metadata file present."""
        mcts_dir = tmp_path / "mcts"
        mcts_dir.mkdir()
        (mcts_dir / "tuned_config.json").write_text('{"best_ucb_c": 1.41}')

        card = _create_model_card(tmp_path, "test/repo")
        assert "# test/repo" in card
