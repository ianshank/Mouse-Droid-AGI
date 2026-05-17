"""C1.1: GCS->HF Hub closed-loop weight publication leg."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from training.upload_weights import sync_gcs_to_hf


def test_sync_gcs_to_hf_downloads_then_uploads(tmp_path: Path) -> None:
    """sync_gcs_to_hf downloads all blobs under prefix then calls upload_weights."""
    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    fake_blob = MagicMock()
    fake_blob.name = "trained/policy.onnx"
    fake_blob.download_to_filename = MagicMock(
        side_effect=lambda dest: Path(dest).write_bytes(b"weights")
    )
    fake_bucket.list_blobs.return_value = [fake_blob]
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True) as upload:
        ok = sync_gcs_to_hf(
            gcs_bucket="mousedroid-weights",
            gcs_prefix="trained/",
            repo_id="ianshank/mousedroid-policy-v2",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
        )

    assert ok is True
    upload.assert_called_once()
    assert (tmp_path / "policy.onnx").read_bytes() == b"weights"


def test_sync_gcs_to_hf_returns_false_when_bucket_empty(tmp_path: Path) -> None:
    """Empty prefix short-circuits — returns False and logs a warning."""
    fake_gcs_client = MagicMock()
    fake_gcs_client.bucket.return_value.list_blobs.return_value = []
    ok = sync_gcs_to_hf(
        gcs_bucket="mousedroid-weights",
        gcs_prefix="trained/",
        repo_id="ianshank/mousedroid-policy-v2",
        local_dir=tmp_path,
        gcs_client=fake_gcs_client,
    )
    assert ok is False
