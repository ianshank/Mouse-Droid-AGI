"""Unit tests for training.gpu_utils — device resolution, memory monitoring, batch sizing.

Tests assert on behavior (returned device type, batch size bounds) rather than
implementation details. Hardware I/O (CUDA queries) is mocked at the torch level.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from training.gpu_utils import (
    check_memory_budget,
    get_optimal_batch_size,
    log_gpu_info,
    resolve_device,
)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    """Tests for resolve_device()."""

    def test_explicit_cpu_string(self) -> None:
        device = resolve_device("cpu")
        assert device.type == "cpu"

    def test_explicit_cuda_string_returned_as_is(self) -> None:
        # Only parse — don't require CUDA hardware.
        device = resolve_device("cuda:0")
        assert str(device) == "cuda:0"

    def test_auto_detects_cpu_when_cuda_unavailable(self) -> None:
        with patch("training.gpu_utils.torch.cuda.is_available", return_value=False):
            device = resolve_device(None)
        assert device.type == "cpu"

    def test_auto_detects_cuda_when_available(self) -> None:
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1_000_000_000
        with (
            patch("training.gpu_utils.torch.cuda.is_available", return_value=True),
            patch("training.gpu_utils.torch.cuda.get_device_name", return_value="FakeGPU"),
            patch("training.gpu_utils.torch.cuda.get_device_properties", return_value=mock_props),
        ):
            device = resolve_device(None)
        assert device.type == "cuda"

    def test_none_device_falls_back_gracefully(self) -> None:
        """resolve_device(None) must always return a valid torch.device."""
        with patch("training.gpu_utils.torch.cuda.is_available", return_value=False):
            device = resolve_device(None)
        assert isinstance(device, torch.device)


# ---------------------------------------------------------------------------
# log_gpu_info
# ---------------------------------------------------------------------------


class TestLogGpuInfo:
    """Tests for log_gpu_info()."""

    def test_noop_on_cpu(self) -> None:
        """Should return without error on CPU device."""
        cpu = torch.device("cpu")
        log_gpu_info(cpu)  # Should not raise

    def test_logs_cuda_properties(self) -> None:
        mock_props = MagicMock()
        mock_props.name = "FakeGPU"
        mock_props.total_memory = 4_000_000_000
        mock_props.major = 8
        mock_props.minor = 0
        mock_props.multi_processor_count = 40

        cuda_dev = torch.device("cuda:0")
        with (
            patch("training.gpu_utils.torch.cuda.get_device_properties", return_value=mock_props),
            patch.object(torch.version, "cuda", "11.7"),
        ):
            log_gpu_info(cuda_dev)  # Should not raise


# ---------------------------------------------------------------------------
# check_memory_budget
# ---------------------------------------------------------------------------


class TestCheckMemoryBudget:
    """Tests for check_memory_budget()."""

    def test_noop_on_cpu_device(self) -> None:
        check_memory_budget(limit_gb=4.0, device=torch.device("cpu"))

    def test_noop_when_cuda_unavailable(self) -> None:
        with patch("training.gpu_utils.torch.cuda.is_available", return_value=False):
            check_memory_budget(limit_gb=4.0, device=None)  # Should not raise

    def test_no_warning_under_budget(self) -> None:
        with (
            patch("training.gpu_utils.torch.cuda.is_available", return_value=True),
            patch("training.gpu_utils.torch.cuda.memory_allocated", return_value=1_000_000_000),
            patch("training.gpu_utils._log") as mock_log,
        ):
            check_memory_budget(limit_gb=4.0)
        mock_log.warning.assert_not_called()

    def test_warning_over_budget(self) -> None:
        with (
            patch("training.gpu_utils.torch.cuda.is_available", return_value=True),
            patch("training.gpu_utils.torch.cuda.memory_allocated", return_value=5_000_000_000),
            patch("training.gpu_utils._log") as mock_log,
        ):
            check_memory_budget(limit_gb=4.0)
        mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# get_optimal_batch_size
# ---------------------------------------------------------------------------


class TestGetOptimalBatchSize:
    """Tests for get_optimal_batch_size()."""

    def test_returns_default_on_cpu(self) -> None:
        result = get_optimal_batch_size(32, device=torch.device("cpu"))
        assert result == 32

    def test_returns_default_when_cuda_unavailable(self) -> None:
        with patch("training.gpu_utils.torch.cuda.is_available", return_value=False):
            result = get_optimal_batch_size(16)
        assert result == 16

    def test_clamps_to_memory_headroom(self) -> None:
        mock_props = MagicMock()
        mock_props.total_memory = 2_000_000_000  # 2 GB → very limited

        with (
            patch("training.gpu_utils.torch.cuda.is_available", return_value=True),
            patch("training.gpu_utils.torch.cuda.get_device_properties", return_value=mock_props),
        ):
            result = get_optimal_batch_size(128, model_memory_mb=500, memory_limit_gb=2.0)

        assert isinstance(result, int)
        assert result >= 1

    def test_insufficient_memory_returns_reduced_batch(self) -> None:
        mock_props = MagicMock()
        mock_props.total_memory = 400_000_000  # 0.4 GB — too small

        with (
            patch("training.gpu_utils.torch.cuda.is_available", return_value=True),
            patch("training.gpu_utils.torch.cuda.get_device_properties", return_value=mock_props),
        ):
            result = get_optimal_batch_size(32, model_memory_mb=500, memory_limit_gb=0.4)

        assert result == max(1, 32 // 4)

    def test_exception_in_query_falls_back_to_default(self) -> None:
        with (
            patch("training.gpu_utils.torch.cuda.is_available", return_value=True),
            patch(
                "training.gpu_utils.torch.cuda.get_device_properties",
                side_effect=RuntimeError("device error"),
            ),
        ):
            result = get_optimal_batch_size(64)
        assert result == 64
