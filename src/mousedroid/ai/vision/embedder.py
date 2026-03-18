"""CLIP semantic embedder for Jetson — TensorRT/ONNX-accelerated.

Implements ``SemanticEmbedderProtocol`` using ``open_clip`` for rich
512-dim visual embeddings that augment the existing IMX500 features.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VisionAIConfig

_log = get_logger(__name__)

_open_clip: Any
_torch: Any
try:
    import open_clip as _open_clip  # type: ignore[import-untyped]
    import torch as _torch
except ImportError:  # pragma: no cover
    _open_clip = None
    _torch = None


class CLIPEmbedder:
    """CLIP ViT-B/32 embedder implementing ``SemanticEmbedderProtocol``.

    Produces 512-dim semantic feature vectors from camera frames.
    All blocking inference is delegated to ``asyncio.to_thread()``.
    """

    def __init__(self, cfg: VisionAIConfig) -> None:
        self._cfg = cfg
        self._model: Any = None
        self._preprocess: Any = None
        self._device: str = "cpu"
        self._last_inference_t: float = 0.0
        self._min_interval: float = 1.0 / cfg.embedder_max_hz
        self._embed_dim = cfg.embedder_dim

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension."""
        return self._embed_dim

    async def start(self) -> None:
        """Load CLIP model."""
        if _open_clip is None:
            msg = "open_clip is not installed — install mousedroid[ai-vision]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._load_model)
        _log.info(
            "clip_embedder_started",
            model=self._cfg.embedder_model,
            pretrained=self._cfg.embedder_pretrained,
            device=self._device,
        )

    def _load_model(self) -> None:
        """Load CLIP model onto available device (blocking)."""
        if _torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        model, _, preprocess = _open_clip.create_model_and_transforms(
            self._cfg.embedder_model,
            pretrained=self._cfg.embedder_pretrained,
            device=self._device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

    async def stop(self) -> None:
        """Release model resources."""
        self._model = None
        self._preprocess = None
        _log.info("clip_embedder_stopped")

    async def embed(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract CLIP embedding from a BGR frame.

        Rate-limited to ``embedder_max_hz``. Returns zero vector if called
        too frequently.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            Embedding vector, shape ``(embed_dim,)``.
        """
        now = time.monotonic()
        if now - self._last_inference_t < self._min_interval:
            return np.zeros(self._embed_dim, dtype=np.float32)
        self._last_inference_t = now

        if self._model is None:
            return np.zeros(self._embed_dim, dtype=np.float32)

        embedding = await asyncio.to_thread(self._infer, frame)
        return embedding

    def _infer(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Run CLIP inference (blocking)."""
        from PIL import Image  # noqa: PLC0415

        # Convert BGR (OpenCV) to RGB PIL Image
        rgb = frame[:, :, ::-1]  # BGR → RGB
        pil_img = Image.fromarray(rgb)
        tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

        with _torch.no_grad(), _torch.amp.autocast(self._device, enabled=self._device == "cuda"):
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        result: NDArray[np.float32] = features.cpu().numpy().flatten().astype(np.float32)
        # Ensure correct output dimension
        if result.shape[0] != self._embed_dim:
            padded = np.zeros(self._embed_dim, dtype=np.float32)
            n = min(result.shape[0], self._embed_dim)
            padded[:n] = result[:n]
            return padded
        return result
