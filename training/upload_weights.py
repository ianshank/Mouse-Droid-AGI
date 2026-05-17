r"""Upload trained MouseDroid weights to HuggingFace Hub.

Usage::

    python -m training.upload_weights \
        --weights-dir weights/ --repo ianshank/mousedroid-weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

_HF_AVAILABLE = False
try:
    from huggingface_hub import HfApi

    _HF_AVAILABLE = True
except ImportError:
    HfApi = None


def upload_weights(
    weights_dir: str | Path,
    *,
    repo_id: str = "ianshank/mousedroid-weights",
    commit_message: str = "Update trained weights",
    extensions: set[str] | None = None,
) -> bool:
    """Upload all weight files from a directory to HuggingFace Hub.

    Uploads files with specified extensions (default: ``.pt``, ``.npz``, ``.json``).

    Args:
        weights_dir: Local directory containing weight files.
        repo_id: HuggingFace repository ID.
        commit_message: Commit message for the upload.
        extensions: Set of file extensions to upload. Defaults to standard weight extensions.

    Returns:
        True if upload succeeded, False otherwise.
    """
    if not _HF_AVAILABLE:
        _log.warning(
            "huggingface_hub_not_installed",
            hint="Install via: pip install huggingface-hub",
        )
        return False

    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        _log.error("weights_dir_not_found", path=str(weights_dir))
        return False

    # Collect files to upload
    exts = extensions or {".pt", ".npz", ".json"}
    files_to_upload = [f for f in weights_dir.rglob("*") if f.is_file() and f.suffix in exts]

    if not files_to_upload:
        _log.warning("no_weight_files_found", path=str(weights_dir))
        return False

    _log.info(
        "uploading_weights",
        repo_id=repo_id,
        file_count=len(files_to_upload),
        files=[str(f.relative_to(weights_dir)) for f in files_to_upload],
    )

    try:
        api = HfApi()

        # Create model card
        model_card = _create_model_card(weights_dir, repo_id)
        card_path = weights_dir / "README.md"
        card_path.write_text(model_card, encoding="utf-8")

        # Restrict upload to selected extensions + model card
        allow_patterns = [f"**/*{ext}" for ext in exts]
        allow_patterns.append("README.md")

        api.upload_folder(
            folder_path=str(weights_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            allow_patterns=allow_patterns,
        )

        _log.info("upload_complete", repo_id=repo_id, file_count=len(files_to_upload))
        return True

    except Exception as e:
        _log.error("upload_failed", repo_id=repo_id, error=str(e))
        return False


def _create_model_card(weights_dir: Path, repo_id: str) -> str:
    """Create a HuggingFace model card with training metadata.

    Args:
        weights_dir: Directory containing weight files.
        repo_id: HuggingFace repository ID.

    Returns:
        Model card content as string.
    """
    # Try to load training metadata if available
    metadata: dict[str, Any] = {}
    tuned_config = weights_dir / "mcts" / "tuned_config.json"
    if tuned_config.exists():
        with open(tuned_config) as f:
            metadata["mcts_tuning"] = json.load(f)

    card = f"""---
tags:
  - mousedroid
  - robotics
  - rssm
  - bdi
  - constitutional-rl
library_name: pytorch
---

# {repo_id}

Trained weights for MouseDroid autonomous navigation system.

## Components

| Component | File | Description |
|-----------|------|-------------|
| RSSM World Model | `rssm/final.pt` | Recurrent State-Space Model |
| MCTS Policy Init | `mcts/policy_init.npz` | Warm-started PolicyMLP |
| BDI Belief | `bdi/belief.npz` | Belief encoder weights |
| BDI Desire | `bdi/desire.npz` | Desire encoder weights |
| BDI Intention | `bdi/intention.npz` | Intention predictor weights |
| BDI Affect | `bdi/affect.npz` | Affect estimator weights |
| Constitutional RL | `policy.npz`, `value.npz` | PPO policy + value networks |

## Training

Trained on Jetson Orin Nano (8 GB) using synthetic observation sequences.
"""
    # Append training metadata as JSON, if available
    if metadata:
        metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
        card += "\n\n## Training Metadata\n\n```json\n" + metadata_json + "\n```"

    return card


def sync_gcs_to_hf(
    *,
    gcs_bucket: str,
    gcs_prefix: str,
    repo_id: str,
    local_dir: Path,
    gcs_client: Any | None = None,
    commit_message: str = "Cloud trainer auto-upload",
) -> bool:
    """Download every blob under ``gs://<bucket>/<prefix>/*`` and push to HF Hub.

    The function is the cloud-side leg of the OTA loop closed by PR #94's
    Jetson-side ``HuggingFaceWeightUpdatePoller``. It deliberately accepts
    a pre-built ``gcs_client`` so tests inject a ``MagicMock`` and the
    production path resolves the real ``google.cloud.storage.Client`` lazily.

    Args:
        gcs_bucket: GCS bucket name (operators typically wire from
            ``cfg.gcp.gcs.weights_bucket``).
        gcs_prefix: Object prefix inside the bucket (e.g. ``"trained/"``).
            Trailing slash is preserved verbatim — blobs are listed via
            ``bucket.list_blobs(prefix=gcs_prefix)``.
        repo_id: Destination HF Hub repo id (e.g. ``"ianshank/mousedroid-policy"``).
        local_dir: Temp directory to stage downloads in. Created if missing.
        gcs_client: Pre-built GCS client. Production wires from a factory
            (``build_gcs_client``); tests inject a ``MagicMock``. ``None``
            triggers a lazy ``google.cloud.storage.Client()`` construction.
        commit_message: Forwarded to ``upload_weights``.

    Returns:
        ``True`` iff at least one blob was downloaded **and** the subsequent
        ``upload_weights`` call returned ``True``. ``False`` short-circuits
        on an empty prefix (logged as a warning).
    """
    if gcs_client is None:
        from google.cloud import storage  # local import — optional dep

        gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    if not blobs:
        _log.warning(
            "gcs_sync_empty_prefix",
            bucket=gcs_bucket,
            prefix=gcs_prefix,
        )
        return False
    local_dir.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        dest = local_dir / Path(blob.name).name
        blob.download_to_filename(str(dest))
        _log.info("gcs_blob_downloaded", blob=blob.name, dest=str(dest))
    return upload_weights(
        weights_dir=local_dir,
        repo_id=repo_id,
        commit_message=commit_message,
        # Cloud trainer emits .onnx alongside .pt/.npz/.json — the default
        # extension filter in upload_weights() omits .onnx, which would
        # silently drop the world-model artifact and log
        # "no_weight_files_found". Include the cloud-side extension set
        # explicitly so the round-trip is observable.
        extensions={".onnx", ".pt", ".npz", ".json", ".safetensors"},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for uploading weights to HuggingFace Hub.

    Supports two modes:

    * Default — upload an existing local ``--weights-dir`` to HF Hub.
    * ``--from-gcs`` — download every blob under
      ``gs://<gcs-bucket>/<gcs-prefix>/*`` into ``--weights-dir`` (used as a
      staging area) and then upload to ``--repo``. Defaults for the
      bucket/prefix/repo are resolved lazily from ``Settings`` so the
      operator runbook stays config-driven and the CLI still imports in
      minimal cloud environments that lack pydantic.
    """
    parser = argparse.ArgumentParser(
        description="Upload trained MouseDroid weights to HuggingFace Hub.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help=(
            "Local directory containing weight files to upload. When "
            "``--from-gcs`` is set, the directory is used as a staging area "
            "for downloaded blobs (created if missing)."
        ),
    )
    parser.add_argument(
        "--repo",
        dest="repo_id",
        type=str,
        default=None,
        help=(
            "HuggingFace Hub repository ID. Defaults to "
            "``Settings().cloud.weight_update.policy_repo_id`` when "
            "``--from-gcs`` is set, otherwise ``ianshank/mousedroid-weights``."
        ),
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Update trained weights",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--from-gcs",
        action="store_true",
        help=(
            "Download blobs from ``gs://<gcs-bucket>/<gcs-prefix>/*`` into "
            "``--weights-dir`` before uploading. Closes the cloud-trainer "
            "leg of the OTA loop (Tier C1.1)."
        ),
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help=(
            "GCS bucket holding trained artifacts. Defaults to "
            "``Settings().gcp.training.training_bucket`` when ``--from-gcs`` "
            "is set."
        ),
    )
    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default="trained/",
        help="GCS object prefix to sync (only used with ``--from-gcs``).",
    )
    args = parser.parse_args(argv)

    if args.from_gcs:
        # Resolve config-driven defaults lazily so the CLI still imports in
        # minimal cloud environments that may not have pydantic installed.
        gcs_bucket = args.gcs_bucket
        repo_id = args.repo_id
        if gcs_bucket is None or repo_id is None:
            from mousedroid.config.loader import load_settings  # local import

            settings = load_settings()
            if gcs_bucket is None:
                if settings.gcp is None or settings.gcp.training is None:
                    parser.error(
                        "--gcs-bucket is required when ``cfg.gcp.training`` "
                        "is disabled (no Settings default available).",
                    )
                else:
                    gcs_bucket = settings.gcp.training.training_bucket
            if repo_id is None:
                repo_id = settings.cloud.weight_update.policy_repo_id
        success = sync_gcs_to_hf(
            gcs_bucket=gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            repo_id=repo_id,
            local_dir=args.weights_dir,
            commit_message=args.commit_message,
        )
    else:
        repo_id = args.repo_id or "ianshank/mousedroid-weights"
        success = upload_weights(
            weights_dir=args.weights_dir,
            repo_id=repo_id,
            commit_message=args.commit_message,
        )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
