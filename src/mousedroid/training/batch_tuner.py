"""Dynamic batch size tuning based on available VRAM.

Scales batch size down when VRAM is constrained, leaving configured
headroom for model activations and system overhead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

from mousedroid.config.schema import TrainingPipelineConfig

logger = structlog.get_logger(__name__)


@runtime_checkable
class BatchTunerProtocol(Protocol):
    """Protocol for dynamic batch size adjustment."""

    def tune_batch_size(self, phase: str, base_size: int) -> int:
        """Compute optimal batch size given current VRAM availability.

        Args:
            phase: Training phase name (e.g. "rssm").
            base_size: Maximum desired batch size from config.

        Returns:
            Adjusted batch size (>= 1, <= base_size).
        """
        ...


class VRAMBatchTuner:
    """Adjusts batch size based on available VRAM from torch.cuda.

    Queries ``torch.cuda.mem_get_info()`` to determine free memory,
    subtracts configured headroom, and scales batch size proportionally.

    Args:
        config: Training pipeline configuration with VRAM headroom.
    """

    def __init__(self, config: TrainingPipelineConfig) -> None:
        self._headroom_mb = config.vram_headroom_mb

    def tune_batch_size(self, phase: str, base_size: int) -> int:
        """Scale batch size to fit within available VRAM minus headroom.

        Args:
            phase: Training phase name for logging context.
            base_size: Maximum desired batch size.

        Returns:
            Adjusted batch size clamped to [1, base_size].
        """
        free_mb, total_mb = self._get_vram_info()

        if total_mb == 0:
            # No CUDA available — return base size (CPU training).
            logger.info(
                "batch_tuner_no_cuda",
                phase=phase,
                batch_size=base_size,
            )
            return base_size

        usable_mb = max(0, free_mb - self._headroom_mb)
        # Estimate: scale proportionally to usable fraction of total.
        ratio = usable_mb / total_mb if total_mb > 0 else 1.0

        tuned = max(1, min(base_size, int(base_size * ratio)))

        logger.info(
            "batch_size_tuned",
            phase=phase,
            base_size=base_size,
            tuned_size=tuned,
            free_mb=free_mb,
            usable_mb=usable_mb,
            total_mb=total_mb,
        )
        return tuned

    def _get_vram_info(self) -> tuple[int, int]:
        """Query VRAM via torch.cuda.mem_get_info.

        Returns:
            Tuple of (free_mb, total_mb). Returns (0, 0) if CUDA unavailable.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return (0, 0)
            free, total = torch.cuda.mem_get_info()
            return (int(free / (1024 * 1024)), int(total / (1024 * 1024)))
        except Exception as exc:
            logger.warning("vram_info_failed", error=str(exc))
            return (0, 0)
