"""Tests for training.gpu_utils — device resolution, memory checks, batch sizing."""

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

# ── resolve_device ──────────────────────────────────────────


class TestResolveDevice:
    """Tests for resolve_device()."""

    def test_explicit_cpu(self) -> None:
        """Explicit 'cpu' returns CPU device."""
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=True)
    def test_explicit_cuda_string(self, mock_avail: MagicMock) -> None:
        """Explicit CUDA device is honored when CUDA is available."""
        device = resolve_device("cuda:0")
        assert device == torch.device("cuda:0")

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_falls_back_when_unavailable(self, mock_avail: MagicMock) -> None:
        """Explicit CUDA device falls back to CPU when CUDA is unavailable by default."""
        device = resolve_device("cuda:0")
        assert device == torch.device("cpu")

    def test_explicit_cpu_rejected_when_cuda_required(self) -> None:
        """Strict GPU mode rejects explicit CPU selection."""
        with pytest.raises(RuntimeError, match="requires a CUDA device"):
            resolve_device("cpu", require_cuda=True)

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_raises_when_cuda_required(self, mock_avail: MagicMock) -> None:
        """Strict GPU mode rejects unavailable explicit CUDA devices."""
        with pytest.raises(RuntimeError, match="requested device 'cuda:0' is unavailable"):
            resolve_device("cuda:0", require_cuda=True)

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_auto_detect_no_cuda(self, mock_avail: MagicMock) -> None:
        """Auto-detect falls back to CPU when CUDA unavailable."""
        device = resolve_device(None)
        assert device == torch.device("cpu")

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_auto_detect_no_cuda_raises_when_required(self, mock_avail: MagicMock) -> None:
        """Strict GPU mode rejects auto-detect fallback to CPU."""
        with pytest.raises(RuntimeError, match="no CUDA device is available"):
            resolve_device(None, require_cuda=True)

    @patch("training.gpu_utils.torch.cuda.get_device_properties")
    @patch("training.gpu_utils.torch.cuda.get_device_name", return_value="Test GPU")
    @patch("training.gpu_utils.torch.cuda.is_available", return_value=True)
    def test_auto_detect_with_cuda(
        self,
        mock_avail: MagicMock,
        mock_name: MagicMock,
        mock_props: MagicMock,
    ) -> None:
        """Auto-detect selects cuda:0 when GPU is available."""
        mock_props.return_value = MagicMock(total_memory=8 * 1e9)
        device = resolve_device(None)
        assert device == torch.device("cuda:0")


# ── log_gpu_info ────────────────────────────────────────────


class TestLogGpuInfo:
    """Tests for log_gpu_info()."""

    def test_cpu_is_noop(self) -> None:
        """log_gpu_info is a no-op for CPU device."""
        # Should not raise
        log_gpu_info(torch.device("cpu"))

    @patch("training.gpu_utils.torch.cuda.get_device_properties")
    @patch("training.gpu_utils.torch.cuda.is_available", return_value=True)
    def test_cuda_logs_info(
        self,
        mock_avail: MagicMock,
        mock_props: MagicMock,
    ) -> None:
        """log_gpu_info queries GPU properties for CUDA device."""
        mock_props.return_value = MagicMock(
            name="TestGPU",
            total_memory=8_000_000_000,
            major=8,
            minor=7,
            multi_processor_count=16,
        )
        # Should not raise
        log_gpu_info(torch.device("cuda:0"))
        mock_props.assert_called_once()

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_cuda_is_noop_when_runtime_unavailable(self, mock_avail: MagicMock) -> None:
        """log_gpu_info should not raise when CUDA runtime is unavailable."""
        log_gpu_info(torch.device("cuda:0"))


# ── check_memory_budget ─────────────────────────────────────


class TestCheckMemoryBudget:
    """Tests for check_memory_budget()."""

    def test_cpu_is_noop(self) -> None:
        """No warning for CPU device."""
        check_memory_budget(6.0, torch.device("cpu"))

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_no_cuda_is_noop(self, mock_avail: MagicMock) -> None:
        """No warning when CUDA unavailable."""
        check_memory_budget(6.0)

    @patch("training.gpu_utils.torch.cuda.memory_allocated", return_value=3_000_000_000)
    @patch("training.gpu_utils.torch.cuda.is_available", return_value=True)
    def test_within_budget(
        self,
        mock_avail: MagicMock,
        mock_alloc: MagicMock,
    ) -> None:
        """No warning when memory is within budget."""
        check_memory_budget(6.0)

    @patch("training.gpu_utils.torch.cuda.memory_allocated", return_value=7_000_000_000)
    @patch("training.gpu_utils.torch.cuda.is_available", return_value=True)
    def test_exceeds_budget(
        self,
        mock_avail: MagicMock,
        mock_alloc: MagicMock,
    ) -> None:
        """Warning logged when memory exceeds budget."""
        # Should not raise, just log
        check_memory_budget(6.0)


# ── get_optimal_batch_size ──────────────────────────────────


class TestGetOptimalBatchSize:
    """Tests for get_optimal_batch_size()."""

    def test_cpu_returns_default(self) -> None:
        """CPU device returns the default batch size."""
        result = get_optimal_batch_size(32, device=torch.device("cpu"))
        assert result == 32

    @patch("training.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_no_cuda_returns_default(self, mock_avail: MagicMock) -> None:
        """Returns default when CUDA unavailable."""
        result = get_optimal_batch_size(32)
        assert result == 32

    @patch("training.gpu_utils.torch.cuda.get_device_properties")
    @patch("training.gpu_utils.torch.cuda.is_available", return_value=True)
    def test_scales_down_for_small_gpu(
        self,
        mock_avail: MagicMock,
        mock_props: MagicMock,
    ) -> None:
        """Scales batch size down for small GPU memory."""
        mock_props.return_value = MagicMock(total_memory=2_000_000_000)  # 2 GB
        result = get_optimal_batch_size(64, memory_limit_gb=6.0)
        assert result < 64
        assert result >= 1

    @patch("training.gpu_utils.torch.cuda.get_device_properties")
    @patch("training.gpu_utils.torch.cuda.is_available", return_value=True)
    def test_large_gpu_keeps_default(
        self,
        mock_avail: MagicMock,
        mock_props: MagicMock,
    ) -> None:
        """Large GPU memory returns default batch size."""
        mock_props.return_value = MagicMock(total_memory=16_000_000_000)  # 16 GB
        result = get_optimal_batch_size(32, memory_limit_gb=12.0)
        assert result == 32
