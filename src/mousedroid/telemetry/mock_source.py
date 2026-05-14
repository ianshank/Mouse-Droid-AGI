"""Mock telemetry data source — synthesises plausible scans + frames.

Useful for dashboard development without a rover attached. The source
runs as an async task that pushes synthetic ``TelemetryFrame`` and
``LidarRawScan`` payloads onto the publisher's queues at the rates
configured by ``TelemetryConfig.publish_hz`` and
``TelemetryConfig.lidar_raw_publish_hz``.

The output is deterministic given a seed so screenshot diffs and
visual-regression tests stay stable across CI runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.protocol import LidarRawScan, TelemetryFrame

if TYPE_CHECKING:
    from mousedroid.config.schema import TelemetryConfig
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol

_log = get_logger(__name__)


class MockTelemetrySource:
    """Synthesises telemetry payloads for the dashboard in mock mode.

    Independent of the orchestrator — runs in its own asyncio task so
    operators can ``MOUSEDROID_MOCK_HARDWARE=true mousedroid`` and see
    moving LiDAR + camera patterns in the browser.

    The source is rate-limited by the publisher's existing logic; this
    task just paces frame emission with its own sleeps. Deterministic
    given ``seed``.
    """

    def __init__(
        self,
        cfg: TelemetryConfig,
        publisher: TelemetryPublisherProtocol,
        *,
        seed: int = 0,
        lidar_points_per_scan: int = 360,
        lidar_max_range_m: float = 8.0,
    ) -> None:
        """Initialise a mock source.

        Args:
            cfg: Telemetry configuration (rates, queue sizes).
            publisher: Real publisher to push synthetic frames into.
            seed: PRNG seed for deterministic output.
            lidar_points_per_scan: Number of synthetic points per scan
                (1° resolution = 360).
            lidar_max_range_m: Synthetic LiDAR max range in metres.
        """
        self._cfg = cfg
        self._publisher = publisher
        self._lidar_points_per_scan = max(8, lidar_points_per_scan)
        self._lidar_max_range_m = max(0.1, lidar_max_range_m)
        # Non-cryptographic — only used for synthetic dashboard noise.
        self._rng = random.Random(seed)  # noqa: S311
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._tick = 0
        self._frame_period = 1.0 / cfg.publish_hz
        self._scan_period = 1.0 / cfg.lidar_raw_publish_hz

    async def start(self) -> None:
        """Spawn the synthesis task.

        Idempotent — calling ``start`` twice keeps the existing task.
        """
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="mock_telemetry_source")
        _log.info(
            "mock_telemetry_source_started",
            publish_hz=self._cfg.publish_hz,
            lidar_raw_hz=self._cfg.lidar_raw_publish_hz,
        )

    async def stop(self) -> None:
        """Cancel the synthesis task and wait for it to drain."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        _log.info("mock_telemetry_source_stopped")

    async def _run(self) -> None:
        """Emit synthetic frames + scans on a fixed interval."""
        # Use the shorter of the two periods as the loop tick so the
        # publisher's rate-limit naturally drops duplicates.
        tick_period = min(self._frame_period, self._scan_period)
        try:
            while self._running:
                await self._emit_frame()
                await self._emit_scan()
                await asyncio.sleep(tick_period)
        except asyncio.CancelledError:
            return

    async def _emit_frame(self) -> None:
        """Build and publish a synthetic ``TelemetryFrame``.

        The values follow a slow sine wave so the dashboard shows
        gentle motion rather than random noise.
        """
        t = self._tick * self._frame_period
        battery_v = 7.4 + 0.05 * math.sin(t * 0.05)
        loop_ms = 18.0 + 4.0 * math.sin(t * 0.2)
        frame = TelemetryFrame(
            timestamp=t,
            distance_m=1.5 + 0.5 * math.sin(t * 0.4),
            motor_state=[0.0, 0.0, 0.0, battery_v],
            vision_norm=0.4 + 0.2 * math.cos(t * 0.3),
            audio_rms=0.05 + 0.02 * abs(math.sin(t * 0.6)),
            valid_mask=[1.0, 1.0, 1.0, 1.0],
            battery_voltage=battery_v,
            safety={
                "is_emergency": False,
                "violations": [],
                "forward_clearance_ok": True,
                "lidar_clearance_ok": True,
            },
            lidar_min_dist_m=0.8 + 0.2 * math.sin(t * 0.7),
            lidar_sectors=self._synthesize_sectors(t),
            lidar_n_points=self._lidar_points_per_scan,
            vision_features=self._synthesize_features(t),
            loop_time_ms=loop_ms,
            tick_count=self._tick,
            sensor_liveness={
                "lidar": {"state": "live", "age_s": 0.05},
                "vision": {"state": "live", "age_s": 0.05},
                "audio": {"state": "live", "age_s": 0.05},
                "motor": {"state": "live", "age_s": 0.05},
            },
        )
        self._tick += 1
        await self._publisher.publish(frame)

    async def _emit_scan(self) -> None:
        """Build and publish a synthetic raw LiDAR scan.

        Two synthetic obstacles rotate slowly so the polar plot
        animates without any sensor attached.
        """
        t = self._tick * self._frame_period
        angles: list[float] = []
        distances: list[float] = []
        n = self._lidar_points_per_scan
        for i in range(n):
            theta = (i / n) * 2.0 * math.pi
            angles.append(theta)
            # Two obstacles at rotating angles to give the dashboard
            # visible motion. Background ring at max range.
            d = self._lidar_max_range_m
            for obs_offset in (0.0, math.pi):
                obs_angle = (t * 0.6 + obs_offset) % (2 * math.pi)
                arc_width = math.pi / 12  # 15° wedge
                diff = abs(((theta - obs_angle + math.pi) % (2 * math.pi)) - math.pi)
                if diff < arc_width:
                    d = min(d, 1.0 + 0.5 * math.sin(t * 0.4 + obs_offset))
            distances.append(max(0.05, d))

        scan = LidarRawScan(
            timestamp=t,
            angles_rad=angles,
            distances_m=distances,
            n_points=n,
            scan_duration_s=self._scan_period,
        )
        await self._publisher.publish_lidar_raw(scan)

    def _synthesize_sectors(self, t: float) -> list[float]:
        """Generate normalised sector distances in ``[0, 1]``."""
        n = 8
        return [
            max(0.05, min(1.0, 0.6 + 0.35 * math.sin(t * 0.3 + i * math.pi / 4))) for i in range(n)
        ]

    def _synthesize_features(self, t: float) -> list[float]:
        """Generate a small vision feature vector with slow drift."""
        n = 256
        base = math.sin(t * 0.1)
        out: list[float] = []
        for i in range(n):
            out.append(0.5 * base + 0.5 * math.sin(i * 0.05 + t * 0.2))
        return out
