"""MiDaS monocular depth estimation for Jetson.

Produces dense depth maps from camera frames using MiDaS small model.
The depth map is fused with ultrasonic point readings via Kalman filter.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import FusionConfig

_log = get_logger(__name__)

_torch: Any
try:
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

_MIDAS_TRUSTED_REPO = "intel-isl/MiDaS"
_MIDAS_TRUSTED_REVISION = "master"


class MiDaSDepthEstimator:
    """MiDaS small monocular depth model.

    Produces relative inverse-depth maps from single RGB images.
    The output is normalised to [0, 1] range where higher values
    indicate closer objects.

    ~50 MB memory footprint. GPU-accelerated via PyTorch/CUDA.
    """

    def __init__(self, cfg: FusionConfig) -> None:
        self._cfg = cfg
        self._model: Any = None
        self._transform: Any = None
        self._device: str = "cpu"
        self._last_inference_t: float = 0.0
        self._min_interval: float = 1.0 / cfg.depth_max_hz

    async def start(self) -> None:
        """Load MiDaS model."""
        if _torch is None:
            msg = "torch is not installed — install mousedroid[ai-fusion]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._load_model)
        _log.info(
            "midas_depth_started",
            model=self._cfg.depth_model,
            device=self._device,
        )

    def _load_model(self) -> None:
        """Load MiDaS via torch.hub (blocking)."""
        if _torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        if self._cfg.midas_hub_repo != _MIDAS_TRUSTED_REPO:
            _log.warning(
                "midas_hub_repo_override_ignored",
                configured_repo=self._cfg.midas_hub_repo,
                trusted_repo=_MIDAS_TRUSTED_REPO,
            )

        model_type = self._cfg.depth_model  # "MiDaS_small"
        try:
            self._model = _torch.hub.load(
                _MIDAS_TRUSTED_REPO,
                model_type,
                trust_repo=True,
                revision=_MIDAS_TRUSTED_REVISION,
            )
        except TypeError:
            # Backward compatibility with older torch.hub versions.
            self._model = _torch.hub.load(
                _MIDAS_TRUSTED_REPO,
                model_type,
                trust_repo=True,
            )
        self._model.to(self._device)
        self._model.eval()

        # Load appropriate transform
        try:
            midas_transforms = _torch.hub.load(
                _MIDAS_TRUSTED_REPO,
                "transforms",
                trust_repo=True,
                revision=_MIDAS_TRUSTED_REVISION,
            )
        except TypeError:
            midas_transforms = _torch.hub.load(
                _MIDAS_TRUSTED_REPO,
                "transforms",
                trust_repo=True,
            )
        if "small" in model_type.lower() or "256" in model_type.lower():
            self._transform = midas_transforms.small_transform
        else:
            self._transform = midas_transforms.dpt_transform

    async def stop(self) -> None:
        """Release model resources."""
        self._model = None
        self._transform = None
        _log.info("midas_depth_stopped")

    async def estimate(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Estimate depth from a BGR frame.

        Rate-limited to ``depth_max_hz``.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            Normalised inverse depth map, shape ``(H, W)``,
            values in [0, 1] where higher = closer.
            Returns zeros if called too frequently.
        """
        h, w = frame.shape[:2]
        now = time.monotonic()
        if now - self._last_inference_t < self._min_interval:
            return np.zeros((h, w), dtype=np.float32)
        self._last_inference_t = now

        if self._model is None:
            return np.zeros((h, w), dtype=np.float32)

        return await asyncio.to_thread(self._infer, frame, h, w)

    def _infer(self, frame: NDArray[np.uint8], h: int, w: int) -> NDArray[np.float32]:
        """Run MiDaS inference (blocking)."""
        import cv2  # noqa: PLC0415

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with _torch.no_grad():
            prediction = self._model(input_batch)

            # Interpolate to original resolution
            prediction = _torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map: NDArray[np.float32] = prediction.cpu().numpy().astype(np.float32)

        # Normalise to [0, 1] — MiDaS outputs inverse depth
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map
