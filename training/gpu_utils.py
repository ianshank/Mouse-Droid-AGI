"""GPU training utilities — device resolution, memory monitoring, AMP helpers.

Provides shared primitives used across all PyTorch training scripts for
GPU auto-detection, memory budget enforcement, and batch-size scaling.
"""

from __future__ import annotations

import structlog
import torch

_log = structlog.get_logger(__name__)


def resolve_device(
    device: str | torch.device | None = None,
    *,
    require_cuda: bool = False,
) -> torch.device:
    """Resolve the best available torch device.

    Auto-detects CUDA GPU when ``device`` is None.  Falls back to CPU
    when CUDA is not available, including when CUDA is explicitly requested
    in a config overlay on a host without CUDA support, unless strict CUDA
    enforcement is requested.

    Args:
        device: Explicit device string or ``torch.device`` (e.g. ``"cuda:0"``,
            ``"cpu"``). ``None`` triggers auto-detection.
        require_cuda: When ``True``, fail instead of falling back to CPU.

    Returns:
        Resolved ``torch.device``.

    Raises:
        RuntimeError: If ``require_cuda`` is set and CUDA is unavailable, or a
            non-CUDA device is selected.
    """
    if device:
        resolved = device if isinstance(device, torch.device) else torch.device(device)
        if require_cuda and resolved.type != "cuda":
            msg = f"GPU-only execution requires a CUDA device, but got '{resolved}'."
            raise RuntimeError(msg)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            if require_cuda:
                msg = (
                    "GPU-only execution requires CUDA, but the requested "
                    f"device '{resolved}' is unavailable."
                )
                raise RuntimeError(msg)
            _log.warning(
                "cuda_device_unavailable_falling_back_to_cpu",
                requested_device=str(resolved),
            )
            return torch.device("cpu")
        _log.info("device_forced", device=str(resolved), require_cuda=require_cuda)
        return resolved

    if torch.cuda.is_available():
        resolved = torch.device("cuda:0")
        _log.info(
            "cuda_device_detected",
            device=str(resolved),
            name=torch.cuda.get_device_name(0),
            memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        )
        return resolved

    if require_cuda:
        msg = "GPU-only execution requires CUDA, but no CUDA device is available."
        raise RuntimeError(msg)

    _log.info("cuda_not_available_using_cpu")
    return torch.device("cpu")


def log_gpu_info(device: torch.device) -> None:
    """Log detailed GPU information.

    No-op if device is CPU.

    Args:
        device: Torch device to query.
    """
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        _log.warning("cuda_logging_skipped_unavailable", device=str(device))
        return

    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        _log.warning("gpu_info_unavailable", device=str(device), exc_info=True)
        return

    _log.info(
        "gpu_info",
        name=props.name,
        total_memory_gb=round(props.total_memory / 1e9, 2),
        major=props.major,
        minor=props.minor,
        multi_processor_count=props.multi_processor_count,
        cuda_version=torch.version.cuda or "unknown",
    )


def check_memory_budget(limit_gb: float, device: torch.device | None = None) -> None:
    """Log a warning if current GPU memory usage exceeds the budget.

    Args:
        limit_gb: Maximum allowed GPU memory in gigabytes.
        device: CUDA device to check. Defaults to current device.
    """
    if device is not None and device.type != "cuda":
        return
    if not torch.cuda.is_available():
        return

    allocated_gb = torch.cuda.memory_allocated(device) / 1e9
    if allocated_gb > limit_gb:
        _log.warning(
            "gpu_memory_budget_exceeded",
            allocated_gb=round(allocated_gb, 2),
            limit_gb=limit_gb,
        )


def get_optimal_batch_size(
    default: int,
    *,
    model_memory_mb: float = 500.0,
    memory_limit_gb: float = 6.0,
    device: torch.device | None = None,
) -> int:
    """Compute an optimal batch size for the available GPU memory.

    Falls back to ``default`` on CPU or when CUDA memory info is unavailable.

    Args:
        default: Default batch size (used as upper bound).
        model_memory_mb: Estimated model memory footprint in MB.
        memory_limit_gb: Maximum GPU memory budget in GB.
        device: CUDA device to query.

    Returns:
        Batch size clamped to headroom.
    """
    if device is not None and device.type != "cuda":
        return default
    if not torch.cuda.is_available():
        return default

    try:
        total_mem_gb = torch.cuda.get_device_properties(device or 0).total_memory / 1e9
        usable_gb = min(total_mem_gb, memory_limit_gb) - (model_memory_mb / 1000.0)
        if usable_gb <= 0:
            _log.warning("insufficient_gpu_memory_for_training", total_gb=round(total_mem_gb, 2))
            return max(1, default // 4)

        # Heuristic: ~50 MB per batch element for RSSM-class models
        estimated_max = int(usable_gb * 1000 / 50)
        optimal = min(default, max(1, estimated_max))
        _log.info(
            "batch_size_computed",
            default=default,
            optimal=optimal,
            usable_gb=round(usable_gb, 2),
        )
        return optimal
    except Exception:
        _log.debug("batch_size_auto_failed_using_default", default=default, exc_info=True)
        return default
