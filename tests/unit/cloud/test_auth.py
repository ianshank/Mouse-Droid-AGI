"""Unit tests for GCP credential helper."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.cloud.conftest import _make_gcp_cfg


def test_resolve_credentials_raises_when_no_google_auth() -> None:
    """resolve_credentials should raise ImportError when google-auth is missing."""
    from mousedroid.cloud import _auth

    original = _auth._GCP_AUTH_AVAILABLE
    try:
        _auth._GCP_AUTH_AVAILABLE = False
        cfg = _make_gcp_cfg()
        with pytest.raises(ImportError, match="google-auth is not installed"):
            _auth.resolve_credentials(cfg)
    finally:
        _auth._GCP_AUTH_AVAILABLE = original


def test_resolve_credentials_file_not_found() -> None:
    """resolve_credentials should raise FileNotFoundError for missing creds file."""
    from mousedroid.cloud import _auth

    if not _auth._GCP_AUTH_AVAILABLE:
        pytest.skip("google-auth not installed")

    cfg = _make_gcp_cfg(credentials_path="/nonexistent/credentials.json")
    with pytest.raises(FileNotFoundError, match="GCP credentials file not found"):
        _auth.resolve_credentials(cfg)


def test_resolve_credentials_from_file() -> None:
    """resolve_credentials should load from file when credentials_path is set."""
    from mousedroid.cloud import _auth

    if not _auth._GCP_AUTH_AVAILABLE:
        pytest.skip("google-auth not installed")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b'{"type": "service_account"}')
        creds_path = f.name

    try:
        cfg = _make_gcp_cfg(credentials_path=creds_path)
        with patch("google.auth.load_credentials_from_file") as mock_load:
            mock_creds = MagicMock()
            mock_load.return_value = (mock_creds, "file-project")
            creds, project = _auth.resolve_credentials(cfg)
            assert creds is mock_creds
            assert project == "test-project"  # cfg.project_id takes precedence
            mock_load.assert_called_once_with(creds_path)
    finally:
        Path(creds_path).unlink()


def test_resolve_credentials_adc() -> None:
    """resolve_credentials should use ADC when no credentials_path."""
    from mousedroid.cloud import _auth

    if not _auth._GCP_AUTH_AVAILABLE:
        pytest.skip("google-auth not installed")

    cfg = _make_gcp_cfg()
    with patch("google.auth.default") as mock_default:
        mock_creds = MagicMock()
        mock_default.return_value = (mock_creds, "adc-project")
        creds, project = _auth.resolve_credentials(cfg)
        assert creds is mock_creds
        assert project == "test-project"  # cfg.project_id takes precedence
        mock_default.assert_called_once()
