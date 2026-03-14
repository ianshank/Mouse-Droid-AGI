"""Upload trained MouseDroid weights to HuggingFace Hub.

Usage:
    python -m training.upload_weights --weights-dir weights/ --repo ianshank/mousedroid-weights
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

_HF_AVAILABLE = False
try:
    from huggingface_hub import HfApi

    _HF_AVAILABLE = True
except ImportError:
    pass


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

        api.upload_folder(
            folder_path=str(weights_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
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
    metadata: dict[str, object] = {}
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
    return card
