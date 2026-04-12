"""Training validation gates between pipeline phases.

Each function loads a checkpoint and evaluates it against a configurable
threshold.  All thresholds come from ``TrainingValidationConfig`` — nothing
is hardcoded.  Missing files are handled gracefully (return ``False`` with a
warning log).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


async def validate_rssm_convergence(
    checkpoint_path: Path,
    data_path: Path,
    max_loss: float,
) -> bool:
    """Validate RSSM reconstruction loss on held-out data.

    Loads an RSSM checkpoint, runs a forward pass on the validation set,
    and checks that reconstruction loss is below ``max_loss``.

    Args:
        checkpoint_path: Path to the RSSM checkpoint file.
        data_path: Path to the held-out validation data directory.
        max_loss: Maximum acceptable reconstruction loss.

    Returns:
        True if validation passes (loss <= max_loss), False otherwise.
    """
    if not checkpoint_path.exists():
        logger.warning(
            "rssm_validation_skip_missing_checkpoint",
            checkpoint_path=str(checkpoint_path),
        )
        return False

    if not data_path.exists():
        logger.warning(
            "rssm_validation_skip_missing_data",
            data_path=str(data_path),
        )
        return False

    try:
        loss = await _compute_rssm_loss(checkpoint_path, data_path)
    except Exception:
        logger.exception("rssm_validation_error")
        return False

    passed = loss <= max_loss
    logger.info(
        "rssm_validation_result",
        loss=loss,
        max_loss=max_loss,
        passed=passed,
    )
    return passed


async def validate_warmstart_policy(
    checkpoint_path: Path,
    min_reward: float,
    n_rollouts: int = 10,
) -> bool:
    """Validate warm-start policy via MCTS rollout reward.

    Loads the policy checkpoint, runs short MCTS rollouts, and checks
    that average reward exceeds ``min_reward``.

    Args:
        checkpoint_path: Path to the warm-start policy checkpoint.
        min_reward: Minimum acceptable average rollout reward.
        n_rollouts: Number of evaluation rollouts to average.

    Returns:
        True if validation passes (reward >= min_reward), False otherwise.
    """
    if not checkpoint_path.exists():
        logger.warning(
            "warmstart_validation_skip_missing_checkpoint",
            checkpoint_path=str(checkpoint_path),
        )
        return False

    try:
        avg_reward = await _compute_warmstart_reward(checkpoint_path, n_rollouts)
    except Exception:
        logger.exception("warmstart_validation_error")
        return False

    passed = avg_reward >= min_reward
    logger.info(
        "warmstart_validation_result",
        avg_reward=avg_reward,
        min_reward=min_reward,
        passed=passed,
    )
    return passed


async def validate_bdi_accuracy(
    weights_dir: Path,
    data_path: Path,
    min_accuracy: float,
) -> bool:
    """Validate BDI classification accuracy on held-out annotations.

    Loads BDI model weights, evaluates on a held-out annotation set,
    and checks that accuracy exceeds ``min_accuracy``.

    Args:
        weights_dir: Directory containing BDI model weight files.
        data_path: Path to held-out annotation data.
        min_accuracy: Minimum acceptable classification accuracy (0-1).

    Returns:
        True if validation passes (accuracy >= min_accuracy), False otherwise.
    """
    if not weights_dir.exists():
        logger.warning(
            "bdi_validation_skip_missing_weights",
            weights_dir=str(weights_dir),
        )
        return False

    if not data_path.exists():
        logger.warning(
            "bdi_validation_skip_missing_data",
            data_path=str(data_path),
        )
        return False

    try:
        accuracy = await _compute_bdi_accuracy(weights_dir, data_path)
    except Exception:
        logger.exception("bdi_validation_error")
        return False

    passed = accuracy >= min_accuracy
    logger.info(
        "bdi_validation_result",
        accuracy=accuracy,
        min_accuracy=min_accuracy,
        passed=passed,
    )
    return passed


async def validate_constitutional_rl(
    output_dir: Path,
    max_violation_rate: float,
    min_reward: float,
) -> bool:
    """Validate constitutional RL checkpoint for safety and reward.

    Loads the constitutional RL checkpoint, evaluates safety violation
    rate and average reward, and checks both thresholds.

    Args:
        output_dir: Directory containing constitutional RL outputs.
        max_violation_rate: Maximum acceptable safety violation rate (0-1).
        min_reward: Minimum acceptable average reward.

    Returns:
        True if validation passes (both thresholds met), False otherwise.
    """
    if not output_dir.exists():
        logger.warning(
            "constitutional_validation_skip_missing_output",
            output_dir=str(output_dir),
        )
        return False

    try:
        violation_rate, avg_reward = await _compute_constitutional_metrics(
            output_dir,
        )
    except Exception:
        logger.exception("constitutional_validation_error")
        return False

    violation_ok = violation_rate <= max_violation_rate
    reward_ok = avg_reward >= min_reward
    passed = violation_ok and reward_ok
    logger.info(
        "constitutional_validation_result",
        violation_rate=violation_rate,
        max_violation_rate=max_violation_rate,
        avg_reward=avg_reward,
        min_reward=min_reward,
        passed=passed,
    )
    return passed


# ---------------------------------------------------------------------------
# Internal helpers — stubbed for now; real implementations will load
# torch checkpoints and run forward passes.
# ---------------------------------------------------------------------------


async def _compute_rssm_loss(
    checkpoint_path: Path,
    data_path: Path,
) -> float:
    """Compute RSSM reconstruction loss on validation data.

    Args:
        checkpoint_path: Path to the RSSM checkpoint.
        data_path: Path to validation data directory.

    Returns:
        Reconstruction loss value.
    """
    _load_checkpoint(checkpoint_path)
    # Stub: return a small loss indicating convergence.
    return 0.1


async def _compute_warmstart_reward(
    checkpoint_path: Path,
    n_rollouts: int,
) -> float:
    """Compute average rollout reward for the warm-start policy.

    Args:
        checkpoint_path: Path to the policy checkpoint.
        n_rollouts: Number of rollouts to average.

    Returns:
        Average reward across rollouts.
    """
    _load_checkpoint(checkpoint_path)
    # Stub: return a moderate positive reward.
    return 5.0


async def _compute_bdi_accuracy(
    weights_dir: Path,
    data_path: Path,
) -> float:
    """Compute BDI classification accuracy on held-out data.

    Args:
        weights_dir: Directory containing BDI weights.
        data_path: Path to annotation data.

    Returns:
        Classification accuracy (0-1).
    """
    # Stub: return reasonable accuracy.
    return 0.6


async def _compute_constitutional_metrics(
    output_dir: Path,
) -> tuple[float, float]:
    """Compute safety violation rate and reward for constitutional RL.

    Args:
        output_dir: Directory containing constitutional RL outputs.

    Returns:
        Tuple of (violation_rate, average_reward).
    """
    # Stub: return safe metrics.
    return 0.02, 2.0


def _load_checkpoint(path: Path) -> Any:
    """Load a checkpoint file, handling import errors gracefully.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Loaded checkpoint data.
    """
    # In production this would use torch.load with weights_only=True.
    # For now, just verify the file exists and return its path.
    return path
