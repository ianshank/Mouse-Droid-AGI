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

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# Legacy default repo for the non-``--from-gcs`` CLI mode. Centralised here so
# the constant is not duplicated between the function signature, the CLI help
# text, and the CLI fallback branch.
_DEFAULT_LEGACY_REPO_ID = "ianshank/mousedroid-weights"
# Default extension set used by ``upload_weights()`` when no override is
# supplied. ``sync_gcs_to_hf`` overrides this via the schema-driven
# ``cloud.weight_update.upload_extensions`` field so the cloud-trainer leg
# can include ``.onnx`` / ``.safetensors`` without mutating the legacy CLI
# default.
_DEFAULT_UPLOAD_EXTENSIONS: frozenset[str] = frozenset({".pt", ".npz", ".json"})
# Cloud-trainer extension set — mirrors the
# ``WeightUpdatePollConfig.upload_extensions`` schema default. Defined as a
# module-level fallback so ``sync_gcs_to_hf`` still works when called
# programmatically without a ``Settings`` instance (tests, ad-hoc scripts).
# The CLI ``--from-gcs`` mode prefers the schema-driven value resolved from
# ``Settings`` so an operator override flows through.
_CLOUD_TRAINER_UPLOAD_EXTENSIONS: tuple[str, ...] = (
    ".onnx",
    ".pt",
    ".npz",
    ".json",
    ".safetensors",
)

_HF_AVAILABLE = False
try:
    from huggingface_hub import HfApi

    _HF_AVAILABLE = True
except ImportError:
    # Optional dependency — module still imports without huggingface_hub so
    # the CLI can fail gracefully via the ``_HF_AVAILABLE`` guard.
    HfApi = None  # type: ignore[assignment,misc]


def upload_weights(
    weights_dir: str | Path,
    *,
    repo_id: str = _DEFAULT_LEGACY_REPO_ID,
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
    exts = extensions if extensions is not None else set(_DEFAULT_UPLOAD_EXTENSIONS)
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
    upload_extensions: tuple[str, ...] | set[str] | None = None,
) -> bool:
    """Download every blob under ``gs://<bucket>/<prefix>/*`` and push to HF Hub.

    The function is the cloud-side leg of the OTA loop closed by PR #94's
    Jetson-side ``HuggingFaceWeightUpdatePoller``. It deliberately accepts
    a pre-built ``gcs_client`` so tests inject a ``MagicMock`` and the
    production path resolves the real ``google.cloud.storage.Client`` lazily.

    Args:
        gcs_bucket: GCS bucket name (operators typically wire from
            ``cfg.gcp.training.training_bucket``).
        gcs_prefix: Object prefix inside the bucket (e.g. ``"trained/"``).
            Trailing slash is preserved verbatim — blobs are listed via
            ``bucket.list_blobs(prefix=gcs_prefix)``.
        repo_id: Destination HF Hub repo id (e.g. ``"ianshank/mousedroid-policy"``).
        local_dir: Temp directory to stage downloads in. Created if missing.
        gcs_client: Pre-built GCS client. Production wires from a factory
            (``build_gcs_client``); tests inject a ``MagicMock``. ``None``
            triggers a lazy ``google.cloud.storage.Client()`` construction.
        commit_message: Forwarded to ``upload_weights``.
        upload_extensions: Extension filter forwarded to ``upload_weights``.
            ``None`` (default) uses the cloud-trainer extension set —
            ``.onnx``/``.pt``/``.npz``/``.json``/``.safetensors`` — so the
            world-model export and HF-native weight formats round-trip.
            Operators override per-call (or via
            ``cfg.cloud.weight_update.upload_extensions``) to extend the
            filter without mutating the legacy CLI default.

    Returns:
        ``True`` iff at least one blob was downloaded **and** the subsequent
        ``upload_weights`` call returned ``True``. ``False`` short-circuits
        on an empty prefix (logged as a warning).
    """
    if gcs_client is None:
        # Local import — optional dep. The project-wide mypy override
        # ``ignore_missing_imports = true`` handles the missing typeshed
        # stubs without needing a per-call ``# type: ignore``.
        from google.cloud import storage

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
        # Skip prefix-itself blobs (e.g. a zero-byte object named "trained/")
        # which list_blobs may surface alongside real files. ``pathlib`` strips
        # trailing slashes from ``.name`` on every platform, so guard on the
        # raw blob name's trailing separator before computing the filename —
        # otherwise the prefix-itself blob would resolve to a sibling file
        # named ``trained`` inside ``local_dir`` (or, when the prefix had no
        # leading path components, to ``local_dir`` itself, raising
        # ``IsADirectoryError`` on Linux).
        if blob.name.endswith("/"):
            continue
        filename = Path(blob.name).name
        if not filename:
            continue
        dest = local_dir / filename
        blob.download_to_filename(str(dest))
        _log.info("gcs_blob_downloaded", blob=blob.name, dest=str(dest))
    # Cloud trainer emits .onnx alongside .pt/.npz/.json — the default
    # extension filter in upload_weights() omits .onnx, which would silently
    # drop the world-model artifact and log "no_weight_files_found". Resolve
    # the schema-driven default (cloud.weight_update.upload_extensions) when
    # the caller has not supplied an override so the round-trip is observable.
    if upload_extensions is None:
        upload_extensions = _CLOUD_TRAINER_UPLOAD_EXTENSIONS
    return upload_weights(
        weights_dir=local_dir,
        repo_id=repo_id,
        commit_message=commit_message,
        extensions=set(upload_extensions),
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
        default=None,
        help=(
            "GCS object prefix to sync (only used with ``--from-gcs``). "
            "Defaults to ``Settings().cloud.weight_update.gcs_artifact_prefix``."
        ),
    )
    args = parser.parse_args(argv)

    if args.from_gcs:
        # Resolve config-driven defaults lazily so the CLI still imports in
        # minimal cloud environments that may not have pydantic installed.
        gcs_bucket: str | None = args.gcs_bucket
        repo_id: str | None = args.repo_id
        gcs_prefix: str | None = args.gcs_prefix
        upload_extensions: tuple[str, ...] | None = None
        if gcs_bucket is None or repo_id is None or gcs_prefix is None:
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
            if gcs_prefix is None:
                gcs_prefix = settings.cloud.weight_update.gcs_artifact_prefix
            # Forward the schema-driven extension filter so an operator
            # override (e.g. adding ``.bin`` for the HF native format)
            # propagates without code changes.
            upload_extensions = settings.cloud.weight_update.upload_extensions
        # Defensive explicit guards instead of ``assert`` — Python's ``-O``
        # flag strips ``assert`` and would let ``None`` slip into
        # ``sync_gcs_to_hf`` and raise a confusing downstream
        # ``AttributeError`` deep inside the GCS client.
        if gcs_bucket is None:
            parser.error("--gcs-bucket could not be resolved from CLI or Settings.")
        if repo_id is None:
            parser.error("--repo could not be resolved from CLI or Settings.")
        if gcs_prefix is None:
            parser.error("--gcs-prefix could not be resolved from CLI or Settings.")
        success = sync_gcs_to_hf(
            gcs_bucket=gcs_bucket,
            gcs_prefix=gcs_prefix,
            repo_id=repo_id,
            local_dir=args.weights_dir,
            commit_message=args.commit_message,
            upload_extensions=upload_extensions,
        )
    else:
        legacy_repo_id: str = args.repo_id or _DEFAULT_LEGACY_REPO_ID
        success = upload_weights(
            weights_dir=args.weights_dir,
            repo_id=legacy_repo_id,
            commit_message=args.commit_message,
        )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
