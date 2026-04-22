"""Camera feature extraction backends.

Provides a reusable :class:`FeatureExtractorProtocol` with concrete
implementations for mean-pool fallback and TensorRT/ONNX model inference.
Camera drivers delegate feature extraction here instead of duplicating logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import CameraConfig
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency handles
# ---------------------------------------------------------------------------

_ort: Any
try:
    import onnxruntime as _ort  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    _ort = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Interface for camera feature extraction backends."""

    def extract(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract feature vector from a raw camera frame.

        Args:
            frame: Raw camera frame as uint8 numpy array.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        ...  # pragma: no cover

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Mean-pool fallback extractor
# ---------------------------------------------------------------------------


class MeanPoolExtractor:
    """Feature extractor using mean-pooling of raw pixel values.

    Divides flattened pixel data into ``feature_dim`` equal chunks, takes the
    mean of each chunk, and optionally L2-normalizes the result.

    Args:
        feature_dim: Output feature vector dimension.
        l2_normalize: Whether to L2-normalize the output vector.
    """

    def __init__(self, feature_dim: int, *, l2_normalize: bool = True) -> None:
        self._feature_dim = feature_dim
        self._l2_normalize = l2_normalize
        _log.info(
            "mean_pool_extractor_init",
            feature_dim=feature_dim,
            l2_normalize=l2_normalize,
        )

    def extract(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract features via mean-pooling.

        Args:
            frame: Raw camera frame as uint8 numpy array.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        flat = frame.astype(np.float32).flatten()
        dim = self._feature_dim
        if len(flat) >= dim:
            stride = len(flat) // dim
            features = flat[: stride * dim].reshape(dim, stride).mean(axis=1)
        else:
            features = np.zeros(dim, dtype=np.float32)
            features[: len(flat)] = flat
        if self._l2_normalize:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
        return cast(NDArray[np.float32], features)

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._feature_dim


# ---------------------------------------------------------------------------
# TensorRT / ONNX extractor
# ---------------------------------------------------------------------------


class TensorRTExtractor:
    """Feature extractor using an ONNX model via ONNX Runtime.

    Falls back to :class:`MeanPoolExtractor` when ONNX Runtime is unavailable
    or the model fails to load.

    Args:
        model_path: Path to the ONNX model file.
        feature_dim: Expected output feature vector dimension.
        l2_normalize: Whether to L2-normalize the output vector.
    """

    def __init__(
        self,
        model_path: Path,
        feature_dim: int,
        *,
        l2_normalize: bool = True,
    ) -> None:
        self._feature_dim = feature_dim
        self._l2_normalize = l2_normalize
        self._session: Any = None
        self._fallback = MeanPoolExtractor(feature_dim, l2_normalize=l2_normalize)

        if _ort is None:
            _log.warning(
                "onnxruntime_unavailable_using_mean_pool_fallback",
                model_path=str(model_path),
            )
            return

        try:
            self._session = _ort.InferenceSession(str(model_path))
            _log.info(
                "tensorrt_extractor_init",
                model_path=str(model_path),
                feature_dim=feature_dim,
                l2_normalize=l2_normalize,
            )
        except Exception:
            _log.warning(
                "onnx_model_load_failed_using_mean_pool_fallback",
                model_path=str(model_path),
                exc_info=True,
            )

    def extract(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract features via ONNX model or fallback to mean-pool.

        Args:
            frame: Raw camera frame as uint8 numpy array.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        if self._session is None:
            return self._fallback.extract(frame)

        try:
            input_name = self._session.get_inputs()[0].name
            input_data = frame.astype(np.float32)
            # Assume model expects (1, C, H, W) from (H, W, C)
            if input_data.ndim == 3:
                input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            outputs = self._session.run(None, {input_name: input_data})
            features = outputs[0].flatten().astype(np.float32)
            # Truncate or pad to expected dim
            if len(features) >= self._feature_dim:
                features = features[: self._feature_dim]
            else:
                padded = np.zeros(self._feature_dim, dtype=np.float32)
                padded[: len(features)] = features
                features = padded
            if self._l2_normalize:
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
            return cast(NDArray[np.float32], features)
        except Exception:
            _log.warning("onnx_inference_failed_using_fallback", exc_info=True)
            return self._fallback.extract(frame)

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._feature_dim


# ---------------------------------------------------------------------------
# Hailo-8 accelerator extractor
# ---------------------------------------------------------------------------


class HailoFeatureExtractor:
    """Feature extractor using Hailo-8 neural accelerator.

    Dispatches inference to the shared :class:`HailoRuntime` and falls
    back to :class:`MeanPoolExtractor` when the runtime is unavailable
    or inference fails.

    Args:
        runtime: Hailo-8 runtime instance for inference dispatch.
        feature_dim: Expected output feature vector dimension.
        l2_normalize: Whether to L2-normalize the output vector.
    """

    def __init__(
        self,
        runtime: HailoRuntimeProtocol,
        feature_dim: int,
        *,
        l2_normalize: bool = True,
    ) -> None:
        self._runtime = runtime
        self._feature_dim = feature_dim
        self._l2_normalize = l2_normalize
        self._fallback = MeanPoolExtractor(feature_dim, l2_normalize=l2_normalize)
        _log.info(
            "hailo_feature_extractor_init",
            feature_dim=feature_dim,
            l2_normalize=l2_normalize,
        )

    def extract(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract features via Hailo-8 or fallback to mean-pool.

        Args:
            frame: Raw camera frame as uint8 numpy array.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        if not self._runtime.is_available():
            return self._fallback.extract(frame)

        try:
            # Hailo expects uint8 input; reshape to (H, W, C) if needed
            input_data = frame
            if input_data.ndim == 2:
                input_data = np.expand_dims(input_data, axis=-1)

            # Direct synchronous call — thread-safe via threading.Lock in runtime
            result = self._runtime.infer_sync("feature_extractor", input_data)
            features = result.flatten().astype(np.float32)

            # Truncate or pad to expected dimension
            if len(features) >= self._feature_dim:
                features = features[: self._feature_dim]
            else:
                padded = np.zeros(self._feature_dim, dtype=np.float32)
                padded[: len(features)] = features
                features = padded

            if self._l2_normalize:
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm

            return features
        except Exception:
            _log.warning("hailo_feature_extraction_failed_using_fallback", exc_info=True)
            return self._fallback.extract(frame)

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._feature_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_feature_extractor(
    cfg: CameraConfig,
    hailo_runtime: HailoRuntimeProtocol | None = None,
) -> FeatureExtractorProtocol:
    """Build a feature extractor based on camera configuration.

    Args:
        cfg: Camera configuration with feature_extractor and model_path.
        hailo_runtime: Optional Hailo-8 runtime for accelerated extraction.

    Returns:
        Feature extractor conforming to :class:`FeatureExtractorProtocol`.
    """
    # Hailo-8 accelerator backend
    if cfg.feature_extractor == "hailo" and hailo_runtime is not None:
        extractor: FeatureExtractorProtocol = HailoFeatureExtractor(
            runtime=hailo_runtime,
            feature_dim=cfg.feature_dim,
            l2_normalize=cfg.l2_normalize,
        )
        _log.info("feature_extractor_selected", backend="hailo")
        return extractor

    # Auto: try Hailo first, then TensorRT, then MeanPool
    if (
        cfg.feature_extractor == "auto"
        and hailo_runtime is not None
        and hailo_runtime.is_available()
    ):
        extractor = HailoFeatureExtractor(
            runtime=hailo_runtime,
            feature_dim=cfg.feature_dim,
            l2_normalize=cfg.l2_normalize,
        )
        _log.info("feature_extractor_selected", backend="hailo")
        return extractor

    # TensorRT / ONNX backend
    if cfg.feature_extractor in ("tensorrt", "auto") and cfg.model_path is not None:
        extractor = TensorRTExtractor(
            model_path=cfg.model_path,
            feature_dim=cfg.feature_dim,
            l2_normalize=cfg.l2_normalize,
        )
        _log.info(
            "feature_extractor_selected",
            backend="tensorrt",
            model_path=str(cfg.model_path),
        )
        return extractor

    # Fallback: MeanPool
    extractor = MeanPoolExtractor(
        feature_dim=cfg.feature_dim,
        l2_normalize=cfg.l2_normalize,
    )
    _log.info("feature_extractor_selected", backend="mean_pool")
    return extractor
