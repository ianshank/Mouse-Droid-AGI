"""AI vision protocol interfaces — structural typing for all vision AI components.

All interfaces use ``@runtime_checkable`` structural typing following
the project's protocol-based dependency injection pattern.
"""

from __future__  import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Detection:
    """A single object detection result.

    Attributes:
        bbox: Bounding box ``[x1, y1, x2, y2]`` in pixel coordinates.
        class_id: Integer class identifier.
        class_name: Human-readable class label.
        confidence: Detection confidence in ``[0.0, 1.0]``.
    """

    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class FaceDetection:
    """A single face detection result.

    Attributes:
        bbox: Bounding box ``[x1, y1, x2, y2]`` in pixel coordinates.
        confidence: Detection confidence in ``[0.0, 1.0]``.
    """

    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class Gesture:
    """A recognised hand gesture.

    Attributes:
        label: Gesture label (e.g. ``"stop"``, ``"come"``, ``"go"``, ``"point"``).
        confidence: Recognition confidence in ``[0.0, 1.0]``.
        hand: Which hand — ``"left"`` or ``"right"``.
    """

    label: str
    confidence: float
    hand: str


@dataclass(frozen=True)
class VisionAIResult:
    """Combined output from all vision AI models for one frame.

    Attributes:
        detections: Object detection results (may be empty).
        embedding: Semantic embedding vector.
        faces: Face detection results (may be empty).
        gestures: Gesture recognition results (may be empty).
        frame_shape: Original frame shape ``(H, W, C)``.
        timestamp: Monotonic timestamp.
    """

    detections: list[Detection]
    embedding: NDArray[np.float32]
    faces: list[FaceDetection]
    gestures: list[Gesture]
    frame_shape: tuple[int, ...]
    timestamp: float


@runtime_checkable
class ObjectDetectorProtocol(Protocol):
    """Interface for object detection models (YOLO, SSD, etc)."""

    async def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run object detection on a frame.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            List of detections sorted by confidence (descending).
        """
        ...

    async def start(self) -> None:
        """Load model and prepare for inference."""
        ...

    async def stop(self) -> None:
        """Release model resources."""
        ...


@runtime_checkable
class SemanticEmbedderProtocol(Protocol):
    """Interface for semantic embedding models (CLIP, ViT, etc)."""

    async def embed(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract semantic embedding from a frame.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            Embedding vector, shape ``(embed_dim,)``.
        """
        ...

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension."""
        ...

    async def start(self) -> None:
        """Load model and prepare for inference."""
        ...

    async def stop(self) -> None:
        """Release model resources."""
        ...


@runtime_checkable
class FaceDetectorProtocol(Protocol):
    """Interface for face detection models."""

    async def detect_faces(self, frame: NDArray[np.uint8]) -> list[FaceDetection]:
        """Detect faces in a frame.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            List of face detections.
        """
        ...

    async def start(self) -> None:
        """Load model and prepare for inference."""
        ...

    async def stop(self) -> None:
        """Release model resources."""
        ...


@runtime_checkable
class GestureRecognizerProtocol(Protocol):
    """Interface for hand gesture recognition models."""

    async def recognize(self, frame: NDArray[np.uint8]) -> list[Gesture]:
        """Recognise hand gestures in a frame.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            List of recognised gestures.
        """
        ...

    async def start(self) -> None:
        """Load model and prepare for inference."""
        ...

    async def stop(self) -> None:
        """Release model resources."""
        ...
