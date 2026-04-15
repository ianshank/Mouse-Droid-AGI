"""Shared GCP credential helper.

Resolves credentials via explicit service-account key path or Application
Default Credentials (ADC).  ADC works automatically on GCE/GKE instances and
locally when ``GOOGLE_APPLICATION_CREDENTIALS`` is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import GCPConfig

_log = get_logger(__name__)

_GCP_AUTH_AVAILABLE = False
try:
    import google.auth

    _GCP_AUTH_AVAILABLE = True
except ImportError:
    pass


def resolve_credentials(
    cfg: GCPConfig,
) -> tuple[Any, str]:
    """Resolve GCP credentials and project ID.

    Args:
        cfg: GCP configuration with optional ``credentials_path``.

    Returns:
        Tuple of ``(credentials, project_id)``.

    Raises:
        ImportError: If ``google-auth`` is not installed.
        FileNotFoundError: If an explicit credentials path does not exist.
    """
    if not _GCP_AUTH_AVAILABLE:
        msg = "google-auth is not installed. Install via: pip install 'mousedroid[gcp]'"
        raise ImportError(msg)

    if cfg.credentials_path is not None:
        creds_path = Path(cfg.credentials_path)
        if not creds_path.exists():
            msg = f"GCP credentials file not found: {creds_path}"
            raise FileNotFoundError(msg)
        creds, project = google.auth.load_credentials_from_file(str(creds_path))
        effective_project = cfg.project_id or project or ""
        _log.info(
            "gcp_credentials_loaded",
            source="file",
            path=str(creds_path),
            project_id=effective_project,
        )
        return creds, effective_project

    # Application Default Credentials (GCE metadata, GOOGLE_APPLICATION_CREDENTIALS, etc.)
    creds, project = google.auth.default()
    effective_project = cfg.project_id or project or ""
    _log.info(
        "gcp_credentials_loaded",
        source="adc",
        project_id=effective_project,
    )
    return creds, effective_project
