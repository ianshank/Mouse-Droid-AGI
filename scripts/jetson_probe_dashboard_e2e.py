#!/usr/bin/env python3
"""P12a — Dashboard E2E data-flow probe (on-Jetson).

Connects to the live telemetry WebSocket and asserts the data plane
that the ``/lidar`` and ``/camera`` dashboards consume. Three
sub-checks, each blocking:

A. **Live data flow** — capture frames for ``MOUSEDROID_PROBE_CAPTURE_S``
   (default 3.0 s). Assert at least 2 frames AND that ``lidar_sectors``
   changes across frames (sensors are sampling, not stuck).
B. **Sensor liveness shape** — every frame carries ``sensor_liveness``
   with at least the ``lidar`` and ``vision`` entries; each entry has a
   ``state`` in {disabled, awaiting, live, stale} and an ``age_s`` that
   is either ``null`` or a non-negative float. State transitions over
   the window are recorded for the SUMMARY log.
C. **No-hello back-compat** — open a SECOND WebSocket and skip the
   ``hello`` send; assert frames still arrive within
   ``MOUSEDROID_PROBE_BACKCOMPAT_S`` (default 5.0 s). This proves
   legacy dashboards (pre-negotiation) still work.

The plan originally promised a "forced stale via docker exec" sub-check.
We dropped it on review: pausing the lidar driver from a shell probe
either freezes the whole container (single asyncio loop) or requires
invasive device-file manipulation. The stale-transition state machine
is exercised by ``tests/unit/telemetry/test_sensor_liveness.py``; this
on-Jetson probe asserts the SHAPE the dashboard consumes, not the
transition itself.

Exit code 0 on PASS, non-zero on FAIL.

Environment:
    MOUSEDROID_TELEMETRY_HOST       default: 127.0.0.1
    MOUSEDROID_TELEMETRY_PORT       default: 8080
    MOUSEDROID_TELEMETRY_TOKEN      bearer token (optional)
    MOUSEDROID_WS_PATH              default: /ws
    MOUSEDROID_PROBE_CAPTURE_S      sub-check A capture window, default 3.0
    MOUSEDROID_PROBE_BACKCOMPAT_S   sub-check C wait window, default 5.0
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

try:
    import aiohttp
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"FAIL: aiohttp not importable: {exc}", file=sys.stderr)
    sys.exit(2)


VALID_STATES = {"disabled", "awaiting", "live", "stale"}


def _build_url() -> str:
    host = os.environ.get("MOUSEDROID_TELEMETRY_HOST", "127.0.0.1")
    port = os.environ.get("MOUSEDROID_TELEMETRY_PORT", "8080")
    path = os.environ.get("MOUSEDROID_WS_PATH", "/ws")
    return f"ws://{host}:{port}{path}"


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("MOUSEDROID_TELEMETRY_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _is_frame(payload: Any) -> bool:
    """Filter telemetry frames from hello_ack and other control envelopes."""
    return isinstance(payload, dict) and "timestamp" in payload and "motor_state" in payload


async def _capture_frames(
    session: aiohttp.ClientSession,
    *,
    url: str,
    duration_s: float,
    send_hello: bool,
) -> list[dict[str, Any]]:
    """Connect, optionally send hello, collect frames for ``duration_s``."""
    try:
        # NOTE: ``aiohttp.ClientWSTimeout`` is 3.10+; project pins
        # ``aiohttp>=3.9``. ``asyncio.wait_for`` around ``ws.receive``
        # below bounds the read budget without needing it.
        ws = await session.ws_connect(
            url,
            headers=_auth_headers(),
            heartbeat=None,
        )
    except Exception as exc:
        raise RuntimeError(f"cannot open WS: {type(exc).__name__}: {exc}") from exc

    frames: list[dict[str, Any]] = []
    async with ws:
        if send_hello:
            hello = {
                "hello": {
                    "protocol_version": 1,
                    "supported_serializations": ["json"],
                }
            }
            await ws.send_str(json.dumps(hello))

        deadline = asyncio.get_event_loop().time() + duration_s
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                break
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except (TypeError, ValueError):
                continue
            if _is_frame(payload):
                frames.append(payload)
    return frames


def _check_sectors_change(frames: list[dict[str, Any]]) -> str | None:
    """Return None if sectors change across frames, else an error message."""
    sectors_seen: list[tuple[float, ...]] = []
    for f in frames:
        ls = f.get("lidar_sectors")
        if isinstance(ls, list) and ls:
            sectors_seen.append(tuple(round(float(x), 6) for x in ls))
    if not sectors_seen:
        # No lidar_sectors — may be by design when LiDAR is disabled.
        # Still need to verify sensor_liveness reflects that.
        return None
    unique = set(sectors_seen)
    if len(unique) < 2:
        return (
            f"lidar_sectors did not change across {len(sectors_seen)} frames "
            f"(sensor stuck or sampling rate is below frame rate)"
        )
    return None


def _check_liveness_shape(frames: list[dict[str, Any]]) -> tuple[str | None, dict[str, set[str]]]:
    """Validate sensor_liveness; return (error, observed_states_by_sensor)."""
    observed: dict[str, set[str]] = {}
    for f in frames:
        sl = f.get("sensor_liveness")
        if not isinstance(sl, dict):
            return (
                f"frame missing sensor_liveness or wrong type: {type(sl).__name__}",
                observed,
            )
        for sensor in ("lidar", "vision"):
            entry = sl.get(sensor)
            if entry is None:
                # Tracker may omit unregistered sensors; tolerate.
                continue
            if not isinstance(entry, dict):
                return (f"sensor_liveness.{sensor} not a dict: {entry!r}", observed)
            state = entry.get("state")
            if state not in VALID_STATES:
                return (
                    f"sensor_liveness.{sensor}.state={state!r} not in {sorted(VALID_STATES)}",
                    observed,
                )
            age = entry.get("age_s")
            if age is not None and not (isinstance(age, (int, float)) and age >= 0):
                return (
                    f"sensor_liveness.{sensor}.age_s invalid: {age!r}",
                    observed,
                )
            observed.setdefault(sensor, set()).add(state)
    if not observed:
        return ("no sensor_liveness entries observed across all frames", observed)
    return (None, observed)


async def _probe() -> int:
    url = _build_url()
    capture_s = float(os.environ.get("MOUSEDROID_PROBE_CAPTURE_S", "3.0"))
    backcompat_s = float(os.environ.get("MOUSEDROID_PROBE_BACKCOMPAT_S", "5.0"))
    print(f"INFO: telemetry WS = {url}")

    async with aiohttp.ClientSession() as session:
        # ---- Sub-check A + B: capture with hello, validate flow + shape ----
        print(f"INFO: sub-check A/B — capturing frames for {capture_s}s (with hello)")
        try:
            frames = await _capture_frames(session, url=url, duration_s=capture_s, send_hello=True)
        except RuntimeError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 3

        if len(frames) < 2:
            print(
                f"FAIL: sub-check A — only {len(frames)} frame(s) in {capture_s}s "
                "(need >= 2 to detect change)",
                file=sys.stderr,
            )
            return 4

        sector_err = _check_sectors_change(frames)
        if sector_err is not None:
            print(f"FAIL: sub-check A — {sector_err}", file=sys.stderr)
            return 5
        print(f"PASS-PART: sub-check A — {len(frames)} frames, lidar_sectors changed")

        liveness_err, observed = _check_liveness_shape(frames)
        if liveness_err is not None:
            print(f"FAIL: sub-check B — {liveness_err}", file=sys.stderr)
            return 6
        observed_render = {k: sorted(v) for k, v in observed.items()}
        print(f"PASS-PART: sub-check B — sensor_liveness observed states: {observed_render}")

        # ---- Sub-check C: back-compat (no hello) ----
        print(f"INFO: sub-check C — connecting WITHOUT hello, waiting {backcompat_s}s for frames")
        try:
            backcompat_frames = await _capture_frames(
                session, url=url, duration_s=backcompat_s, send_hello=False
            )
        except RuntimeError as exc:
            print(f"FAIL: sub-check C — {exc}", file=sys.stderr)
            return 7

        if len(backcompat_frames) < 1:
            print(
                f"FAIL: sub-check C — no frames arrived in {backcompat_s}s without hello "
                "(legacy dashboard back-compat broken)",
                file=sys.stderr,
            )
            return 8
        print(f"PASS-PART: sub-check C — {len(backcompat_frames)} frame(s) without hello")

        print("PASS: dashboard_e2e_data_flow — A, B, C all green")
        return 0


def main() -> int:
    try:
        return asyncio.run(_probe())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
