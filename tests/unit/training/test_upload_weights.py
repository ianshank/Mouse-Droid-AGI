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

    def test_empty_dir_advertises_no_components(self, tmp_path: Path) -> None:
        """An empty weights dir must not claim components it does not contain.

        This previously asserted the opposite: the card listed RSSM, BDI and
        Constitutional-RL rows from a static template, so an empty or partial
        upload published a table of files that were never pushed. The card is
        a public claim about repo contents, and the Jetson-side
        ``HuggingFaceWeightUpdatePoller`` is one of its readers.
        """
        card = _create_model_card(tmp_path, "ianshank/mousedroid-weights")
        assert "# ianshank/mousedroid-weights" in card
        assert "No weight files were present" in card
        for phantom in ("RSSM World Model", "BDI Belief", "Constitutional RL Policy"):
            assert phantom not in card, (
                f"model card advertised {phantom!r} for an empty weights directory"
            )

    def test_lists_only_the_files_present(self, tmp_path: Path) -> None:
        """The table is built from what is on disk, component by component."""
        (tmp_path / "bdi").mkdir()
        (tmp_path / "bdi" / "belief.npz").write_bytes(b"")
        (tmp_path / "bdi" / "affect.npz").write_bytes(b"")

        card = _create_model_card(tmp_path, "test/repo")
        assert "BDI Belief" in card
        assert "`bdi/belief.npz`" in card
        assert "BDI Affect" in card
        # Absent from the directory, so absent from the card.
        assert "RSSM World Model" not in card
        assert "Constitutional RL Policy" not in card

    def test_unrecognised_file_is_still_reported(self, tmp_path: Path) -> None:
        """An unknown weight file is listed, not silently dropped.

        Omitting it would understate the upload just as a static table
        overstates it.
        """
        (tmp_path / "experimental.npz").write_bytes(b"")
        card = _create_model_card(tmp_path, "test/repo")
        assert "`experimental.npz`" in card

    def test_includes_mcts_tuning_metadata(self, tmp_path: Path) -> None:
        """Model card works with tuning metadata file present."""
        mcts_dir = tmp_path / "mcts"
        mcts_dir.mkdir()
        (mcts_dir / "tuned_config.json").write_text('{"best_ucb_c": 1.41}')

        card = _create_model_card(tmp_path, "test/repo")
        assert "# test/repo" in card
