"""Shared test fixtures for MouseDroid test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


@pytest.fixture(autouse=True)
def _mock_hardware_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mock hardware is enabled for all tests."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")


@pytest.fixture
def mock_settings() -> Settings:
    """Create a Settings instance with mock hardware enabled."""
    os.environ["MOUSEDROID_MOCK_HARDWARE"] = "true"
    from mousedroid.config.schema import Settings

    return Settings(mock_hardware=True)


# ---------------------------------------------------------------------------
# Mock gpiod fixture
# ---------------------------------------------------------------------------


class MockGPIOLine:
    """Simulated gpiod Line with configurable echo return values."""

    def __init__(self, offset: int) -> None:
        """Initialise mock line.

        Args:
            offset: GPIO line offset number.
        """
        self._offset = offset
        self._value: int = 0
        self._requested: bool = False
        self._direction: str | None = None
        self._echo_sequence: list[int] = []
        self._echo_index: int = 0

    def request(
        self,
        consumer: str = "test",
        type: int = 0,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        """Simulate requesting the line."""
        self._requested = True

    def set_value(self, value: int) -> None:
        """Simulate setting output value."""
        self._value = value

    def get_value(self) -> int:
        """Return next value from echo sequence or current value.

        Returns:
            Simulated GPIO pin value.
        """
        if self._echo_sequence:
            val = self._echo_sequence[self._echo_index % len(self._echo_sequence)]
            self._echo_index += 1
            return val
        return self._value

    def release(self) -> None:
        """Simulate releasing the line."""
        self._requested = False

    @property
    def offset(self) -> int:
        """Return line offset."""
        return self._offset

    @property
    def is_requested(self) -> bool:
        """Return whether line is currently requested."""
        return self._requested

    def set_echo_sequence(self, sequence: list[int]) -> None:
        """Configure return values for get_value calls.

        Args:
            sequence: List of values to return in order (cycles).
        """
        self._echo_sequence = sequence
        self._echo_index = 0


class MockGPIOChip:
    """Simulated gpiod Chip with line management."""

    def __init__(self, name: str = "gpiochip0") -> None:
        """Initialise mock chip.

        Args:
            name: Chip device name.
        """
        self._name = name
        self._lines: dict[int, MockGPIOLine] = {}
        self._closed: bool = False

    def get_line(self, offset: int) -> MockGPIOLine:
        """Get or create a mock line by offset.

        Args:
            offset: GPIO line number.

        Returns:
            MockGPIOLine instance.

        Raises:
            RuntimeError: If chip is closed.
        """
        if self._closed:
            msg = f"Chip {self._name} is closed"
            raise RuntimeError(msg)
        if offset not in self._lines:
            self._lines[offset] = MockGPIOLine(offset)
        return self._lines[offset]

    def close(self) -> None:
        """Close the chip."""
        self._closed = True

    @property
    def name(self) -> str:
        """Return chip name."""
        return self._name

    @property
    def is_closed(self) -> bool:
        """Return whether chip is closed."""
        return self._closed


@pytest.fixture
def mock_gpiod() -> Generator[MockGPIOChip, None, None]:
    """Provide a mock gpiod chip for GPIO integration tests.

    Yields:
        MockGPIOChip instance simulating gpiod.Chip behaviour.
    """
    chip = MockGPIOChip("gpiochip0")
    mock_module = MagicMock()
    mock_module.Chip = MagicMock(return_value=chip)
    mock_module.LINE_REQ_DIR_OUT = 1
    mock_module.LINE_REQ_DIR_IN = 2
    mock_module.LINE_REQ_EV_RISING_EDGE = 3

    with patch.dict(sys.modules, {"gpiod": mock_module}):
        yield chip


# ---------------------------------------------------------------------------
# Mock jetson_utils fixture
# ---------------------------------------------------------------------------


class MockVideoSource:
    """Simulated jetson_utils.videoSource returning fake CUDA frames."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
    ) -> None:
        """Initialise mock video source.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            channels: Number of colour channels.
        """
        import numpy as np

        self._width = width
        self._height = height
        self._channels = channels
        self._rng = np.random.default_rng(42)
        self._frame_count = 0
        self._connected = True

    def Capture(self) -> Any:  # noqa: N802
        """Return a mock CUDA image object.

        Returns:
            Mock object representing a CUDA image.
        """
        if not self._connected:
            msg = "Camera disconnected"
            raise RuntimeError(msg)
        self._frame_count += 1
        return MagicMock(name=f"cuda_frame_{self._frame_count}")

    def get_numpy_frame(self) -> Any:
        """Return the numpy frame that would result from cudaToNumpy.

        Returns:
            Numpy array of configured shape.
        """
        import numpy as np

        return self._rng.integers(
            0, 255, (self._height, self._width, self._channels), dtype=np.uint8
        )

    def disconnect(self) -> None:
        """Simulate camera disconnect."""
        self._connected = False

    def reconnect(self) -> None:
        """Simulate camera reconnect."""
        self._connected = True

    @property
    def frame_count(self) -> int:
        """Return number of frames captured."""
        return self._frame_count


@pytest.fixture
def mock_jetson_utils() -> Generator[tuple[MockVideoSource, MagicMock], None, None]:
    """Provide mock jetson_utils module with videoSource and cudaToNumpy.

    Yields:
        Tuple of (MockVideoSource instance, mock jetson_utils module).
    """
    source = MockVideoSource()
    mock_module = MagicMock()
    mock_module.videoSource = MagicMock(return_value=source)

    def _cuda_to_numpy(cuda_img: Any) -> Any:
        return source.get_numpy_frame()

    mock_module.cudaToNumpy = MagicMock(side_effect=_cuda_to_numpy)

    # Also support configuring resolution
    def _make_source(uri: str, argv: list[str] | None = None) -> MockVideoSource:
        width = 640
        height = 480
        if argv:
            for arg in argv:
                if "--input-width=" in arg:
                    width = int(arg.split("=")[1])
                elif "--input-height=" in arg:
                    height = int(arg.split("=")[1])
        nonlocal source
        source = MockVideoSource(width=width, height=height)
        return source

    mock_module.videoSource = MagicMock(side_effect=_make_source)

    with patch.dict(sys.modules, {"jetson_utils": mock_module}):
        yield source, mock_module
