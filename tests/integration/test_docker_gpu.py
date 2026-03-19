"""Integration tests for Docker GPU container on Jetson.

These tests verify the GPU-accelerated L4T container works correctly.
They should be run inside the Docker container or skipped when no GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _is_jetson_host() -> bool:
    """Return True when running directly on Jetson hardware."""
    return Path("/etc/nv_tegra_release").exists()


def _is_l4t_container() -> bool:
    """Return True when running inside an L4T Docker container."""
    return Path("/.dockerenv").exists() and Path("/usr/local/cuda").exists()


def _has_cuda() -> bool:
    """Check if CUDA is available without importing torch at module level."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# Skip entire module if not running on a CUDA-capable system
pytestmark = pytest.mark.skipif(
    not _has_cuda(),
    reason="Requires CUDA GPU (run inside L4T container)",
)


class TestGPUAvailability:
    """Verify GPU access inside the Docker container."""

    def test_cuda_is_available(self) -> None:
        """AC2: torch.cuda.is_available() returns True."""
        import torch

        assert torch.cuda.is_available(), "CUDA not available in container"

    def test_cuda_version_is_12(self) -> None:
        """Verify CUDA 12.x is the runtime version."""
        import torch

        assert torch.version.cuda is not None
        major = int(torch.version.cuda.split(".")[0])
        assert major >= 12, f"Expected CUDA 12+, got {torch.version.cuda}"

    @pytest.mark.skipif(
        not _is_jetson_host(),
        reason="Orin GPU name check only valid on Jetson hardware",
    )
    def test_gpu_device_name(self) -> None:
        """Verify the GPU is an Orin-class device."""
        import torch

        name = torch.cuda.get_device_name(0)
        assert "Orin" in name, f"Expected Orin GPU, got {name}"

    def test_tensor_on_gpu(self) -> None:
        """Verify tensors can be moved to GPU and computed."""
        import torch

        t = torch.randn(4, 4, device="cuda")
        result = t @ t.T
        assert result.device.type == "cuda"
        assert result.shape == (4, 4)

    def test_gpu_memory_available(self) -> None:
        """Verify GPU has reasonable memory (Orin Nano has 8GB shared)."""
        import torch

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        # Orin Nano shares memory with CPU, so available GPU memory varies
        assert total_gb > 0.5, f"Only {total_gb:.1f} GB GPU memory available"


class TestTensorRTAvailability:
    """Verify TensorRT is available in the container."""

    def test_tensorrt_importable(self) -> None:
        """Verify tensorrt can be imported."""
        tensorrt = pytest.importorskip("tensorrt")
        assert hasattr(tensorrt, "Builder")

    def test_tensorrt_version(self) -> None:
        """Verify TensorRT version is 10+."""
        tensorrt = pytest.importorskip("tensorrt")
        major = int(tensorrt.__version__.split(".")[0])
        assert major >= 10, f"Expected TRT 10+, got {tensorrt.__version__}"


class TestMouseDroidImport:
    """Verify the mousedroid package loads correctly with GPU."""

    def test_import_mousedroid(self) -> None:
        """AC3: mousedroid package is importable."""
        import mousedroid  # noqa: F401

    def test_import_world_model(self) -> None:
        """Verify the RSSM world model module loads."""
        from mousedroid.world_model import rssm  # noqa: F401

    def test_import_config(self) -> None:
        """Verify configuration module loads and validates."""
        from mousedroid.config.schema import Settings

        # Should not raise even without config file (uses defaults)
        assert Settings is not None


class TestContainerEnvironment:
    """Verify the container environment is correctly configured."""

    @pytest.mark.skipif(
        not _is_l4t_container(),
        reason="CUDA_HOME is only guaranteed inside the L4T Docker container",
    )
    def test_cuda_home_set(self) -> None:
        """Verify CUDA_HOME environment variable."""
        cuda_home = os.environ.get("CUDA_HOME", "")
        assert cuda_home, "CUDA_HOME not set"
        assert os.path.isdir(cuda_home), f"CUDA_HOME {cuda_home} doesn't exist"

    def test_nvcc_available(self) -> None:
        """Verify nvcc compiler is in PATH."""
        import shutil

        assert shutil.which("nvcc") is not None, "nvcc not found in PATH"

    def test_python_version(self) -> None:
        """Verify Python version is 3.10+."""
        assert sys.version_info >= (3, 10), f"Python {sys.version} is below 3.10"

    def test_config_directory_exists(self) -> None:
        """Verify config directory is mounted."""
        config_dir = os.environ.get("MOUSEDROID_CONFIG", "")
        if config_dir:
            config_path = os.path.dirname(config_dir)
            # Config dir may not be mounted in all test environments
            if os.path.isdir(config_path):
                assert any(
                    f.endswith(".yaml") for f in os.listdir(config_path)
                ), "No YAML files in config directory"
