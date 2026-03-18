"""YOLOv8-nano object detector for Jetson — TensorRT-accelerated.

Implements ``ObjectDetectorProtocol`` using the ``ultralytics`` library.
Auto-exports a TensorRT engine on first run and caches it for subsequent boots.
Person detections feed into Three Laws Law 1 safety.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.ai.vision.protocols import Detection
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VisionAIConfig

_log = get_logger(__name__)

_ultralytics: Any
try:
    from ultralytics import YOLO as _YOLO  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _YOLO = None


class JetsonYOLODetector:
    """YOLOv8-nano detector implementing ``ObjectDetectorProtocol``.

    On first ``start()``, exports a TensorRT FP16 engine to the model cache
    directory. Subsequent starts load the cached engine directly.

    All blocking inference is delegated to ``asyncio.to_thread()``.
    """

    def __init__(self, cfg: VisionAIConfig) -> None:
        self._cfg = cfg
        self._model: Any = None
        self._last_inference_t: float = 0.0
        self._min_interval: float = 1.0 / cfg.detector_max_hz

    async def start(self) -> None:
        """Load YOLO model and prepare TensorRT engine."""
        if _YOLO is None:
            msg = "ultralytics is not installed — install mousedroid[ai-vision]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._load_model)
        _log.info(
            "yolo_detector_started",
            model=self._cfg.detector_model,
            confidence=self._cfg.detector_confidence,
        )

    def _load_model(self) -> None:
        """Load or export TensorRT engine (blocking)."""
        cache_dir = Path(self._cfg.model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        engine_path = cache_dir / f"{self._cfg.detector_model}.engine"

        if engine_path.exists():
            _log.info("yolo_loading_cached_engine", path=str(engine_path))
            self._model = _YOLO(str(engine_path), task="detect")
            return

        # Load PyTorch model and attempt TensorRT export
        pt_model = _YOLO(f"{self._cfg.detector_model}.pt")
        if self._cfg.tensorrt_enabled:
            try:
                export_path = pt_model.export(
                    format="engine",
                    half=self._cfg.detector_half_precision,
                    imgsz=self._cfg.detector_imgsz,
                    device=0,
                )
                # Move exported engine to cache
                exported = Path(export_path)
                if exported.exists() and exported != engine_path:
                    exported.rename(engine_path)
                self._model = _YOLO(str(engine_path), task="detect")
                _log.info("yolo_tensorrt_exported", path=str(engine_path))
                return
            except Exception:
                _log.warning("yolo_tensorrt_export_failed_using_pytorch", exc_info=True)

        # Fallback to PyTorch model
        self._model = pt_model
        _log.info("yolo_using_pytorch_fallback")

    async def stop(self) -> None:
        """Release model resources."""
        self._model = None
        _log.info("yolo_detector_stopped")

    async def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run YOLO detection on a frame.

        Rate-limited to ``detector_max_hz``. Returns empty list if called
        too frequently.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            Detections sorted by confidence (descending).
        """
        now = time.monotonic()
        if now - self._last_inference_t < self._min_interval:
            return []
        self._last_inference_t = now

        if self._model is None:
            return []

        results = await asyncio.to_thread(
            self._model.predict,
            frame,
            conf=self._cfg.detector_confidence,
            verbose=False,
        )
        return self._parse_results(results)

    def _parse_results(self, results: Any) -> list[Detection]:
        """Parse ultralytics results into Detection dataclasses."""
        detections: list[Detection] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
