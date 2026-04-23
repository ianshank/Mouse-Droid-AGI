"""Mock camera driver for testing and simulation.

Implements ``VisionProtocol`` with random feature vectors.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import CameraConfig

_log = get_logger(__name__)


class MockCamera:
    """Mock camera implementing ``VisionProtocol`` and ``RawFrameSourceProtocol``.

    Two raw-frame modes, selected by ``CameraConfig.mock_source``:

    * ``procedural`` — synthesises an animated colour-bar pattern with a
      sweep line and tick overlay. Self-contained, no OS dependencies.
    * ``screen_capture`` — grabs the host desktop via ``PIL.ImageGrab``.
      Real photographic content; useful when validating the dashboard
      against live input without real camera hardware.

    In ``screen_capture`` mode, :meth:`capture_features` is derived from
    the actual pixel content (mean-pooled colour tiles → L2-normalised
    feature vector) so the feature heatmap reflects the real scene.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        """Initialise mock camera from config.

        Args:
            cfg: Camera configuration.
        """
        self._cfg = cfg
        self._rng = np.random.default_rng()
        self._raw_width = 320
        self._raw_height = 240
        self._frame_counter = 0
        self._latest_rgb: NDArray[np.uint8] | None = None
        self._mode = cfg.mock_source
        _log.info("mock_camera_init", feature_dim=cfg.feature_dim, source=self._mode)

    async def capture_features(self) -> NDArray[np.float32]:
        """Return a feature vector.

        In procedural mode this is random noise. In screen-capture mode
        the vector is derived from the latest raw frame so the
        dashboard heatmap tracks real scene content.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        if self._mode == "screen_capture" and self._latest_rgb is not None:
            return self._features_from_rgb(self._latest_rgb)
        return self._rng.standard_normal(self._cfg.feature_dim).astype(np.float32)

    def _features_from_rgb(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Derive a deterministic feature vector from an RGB frame.

        Splits the frame into a grid of tiles and takes the per-channel
        mean of each tile; flattened + L2 normalised to match the shape
        IMX500/Hailo extractors emit.
        """
        dim = self._cfg.feature_dim
        # Use 3 channels (R,G,B); derive tile grid so tiles*3 == feature_dim
        n_tiles = max(1, dim // 3)
        side = max(1, int(np.sqrt(n_tiles)))
        h, w = rgb.shape[:2]
        th, tw = h // side, w // side
        if th == 0 or tw == 0:
            return self._rng.standard_normal(dim).astype(np.float32)
        trimmed = rgb[: th * side, : tw * side]
        # (side, th, side, tw, 3) → mean over th, tw → (side, side, 3)
        tiles = trimmed.reshape(side, th, side, tw, 3).mean(axis=(1, 3))
        flat = tiles.reshape(-1).astype(np.float32) / 255.0
        # pad/truncate to feature_dim
        flat = np.pad(flat, (0, dim - flat.size)) if flat.size < dim else flat[:dim]
        # centre + L2 normalise
        flat -= flat.mean()
        norm = float(np.linalg.norm(flat))
        if norm > 1e-8:
            flat = flat / norm
        result: NDArray[np.float32] = flat.astype(np.float32, copy=False)
        return result

    async def capture_raw_jpeg(self) -> bytes | None:
        """Capture a JPEG frame from the configured mock source.

        Screen capture and Pillow-based JPEG encoding are synchronous and
        can stall the aiohttp event loop when called from ``/camera/stream``
        or ``/camera/frame.jpg``; offload the blocking work to the default
        thread pool so other telemetry requests stay responsive.
        """
        if self._mode == "screen_capture":
            return await asyncio.to_thread(self._capture_screen_jpeg)
        return await asyncio.to_thread(self._capture_procedural_jpeg)

    def _capture_screen_jpeg(self) -> bytes | None:
        """Grab the host desktop and return a downscaled JPEG.

        Returns ``None`` when Pillow / ImageGrab is unavailable (e.g.
        headless Linux without xdisplay), which allows the server to
        respond with a graceful 503.
        """
        try:
            from PIL import Image, ImageGrab
        except ImportError:
            return None
        try:
            img = ImageGrab.grab()
        except OSError:
            # Headless or screen access denied.
            return None
        img = img.convert("RGB")
        w = self._raw_width * 2
        h = self._raw_height * 2
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        self._latest_rgb = np.asarray(img, dtype=np.uint8)
        out = Image.fromarray(self._latest_rgb, mode="RGB")
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=72)
        self._frame_counter += 1
        return buf.getvalue()

    def _capture_procedural_jpeg(self) -> bytes | None:
        """Synthesise a procedural RGB frame and encode as JPEG.

        Generates a moving colour-bar pattern with a rotating vertical
        sweep line and tick overlay so the dashboard shows obvious
        motion even without real camera hardware.

        Returns:
            JPEG-encoded bytes. ``None`` if Pillow is unavailable.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None

        w, h = self._raw_width, self._raw_height
        t = monotonic()
        self._frame_counter += 1

        # Horizontal gradient bars that drift with time.
        xs = np.arange(w, dtype=np.float32)
        shift = (t * 30.0) % w
        bar = ((xs + shift) % w) / w  # 0..1
        r = (0.5 + 0.5 * np.sin(2 * np.pi * bar * 2.0)) * 255
        g = (0.5 + 0.5 * np.sin(2 * np.pi * bar * 3.0 + 1.0)) * 255
        b = (0.5 + 0.5 * np.sin(2 * np.pi * bar * 5.0 + 2.0)) * 255
        row = np.stack([r, g, b], axis=-1).astype(np.uint8)  # (w, 3)
        img_arr = np.broadcast_to(row[None, :, :], (h, w, 3)).copy()

        # Rotating vertical sweep line to make motion unambiguous.
        sweep_x = int((t * 80.0) % w)
        img_arr[:, max(0, sweep_x - 1) : min(w, sweep_x + 1)] = (255, 255, 255)

        # Centre cross-hair.
        cx, cy = w // 2, h // 2
        img_arr[cy, :] = np.minimum(255, img_arr[cy, :].astype(np.int32) + 80)
        img_arr[:, cx] = np.minimum(255, img_arr[:, cx].astype(np.int32) + 80)

        self._latest_rgb = img_arr

        img = Image.fromarray(img_arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except OSError:
            font = None
        label = f"MOCK #{self._frame_counter:06d}  t={t:8.2f}s"
        draw.rectangle((4, 4, 4 + 8 * len(label), 20), fill=(0, 0, 0))
        draw.text((6, 6), label, fill=(255, 255, 255), font=font)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._cfg.feature_dim

    async def start(self) -> None:
        """Simulate starting the camera pipeline."""
        _log.info("mock_camera_started", source=self._mode)

    async def stop(self) -> None:
        """Simulate stopping the camera pipeline."""
        _log.info("mock_camera_stopped")
