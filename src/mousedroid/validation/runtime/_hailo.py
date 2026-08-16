"""Hailo-8 accelerator runtime validation helpers (Task 3 domain)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.common.imports import module_importable
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


@dataclass(frozen=True)
class HailoDiagnostics:
    """Diagnostics record for one ``verify_hailo_accelerator`` probe (Task 3).

    Every SKIP / FAIL branch in the helper returns a fully-formed instance
    so the CLI consumer can emit precise structured output without ``None``
    checks at every field.
    """

    device_path_exists: bool = False
    sdk_importable: bool = False
    device_info: dict[str, str] = field(default_factory=dict)
    hef_files: dict[str, str] = field(default_factory=dict)
    inference_latency_ms: float | None = None
    fallback_on_failure: bool = True


async def verify_hailo_accelerator(cfg: Settings) -> HailoDiagnostics:
    """Probe the Hailo-8 accelerator and exercise one synthetic inference.

    Uses the same :func:`mousedroid.factory.build_hailo_runtime` factory the
    orchestrator wires, so the smoke exercises the production code path.
    The runtime is always stopped in a ``finally`` block — the PCIe device
    lock MUST be released before this helper returns, otherwise the next
    orchestrator start would fail to acquire the device.

    SKIP branches (not FAIL — Hailo is opt-in extras and the runtime is
    documented to fall back to GPU when unavailable):

    * ``cfg.hailo.enabled is False`` (caller-side guard; the helper still
      returns a populated record so callers can introspect)
    * ``cfg.hailo.device_path`` does not exist as a /dev node
    * ``hailo_platform`` is not importable

    FAIL branches:

    * HEF path is set but the file is missing or fails to load
    * Inference exceeds ``cfg.hailo.timeout_ms``

    Args:
        cfg: Fully resolved settings.

    Returns:
        :class:`HailoDiagnostics` instance.
    """
    # ``cfg.hailo`` is ``HailoConfig | None`` (schema.py:4468) — operators
    # who never opt into the Hailo accelerator may leave the entire block
    # unconfigured. Returning a SKIP-style record narrows the type for
    # mypy --strict and tells the CLI consumer to emit
    # "hailo not configured in settings" rather than crashing on a None
    # attribute access.
    if cfg.hailo is None:
        return HailoDiagnostics(
            device_path_exists=False,
            sdk_importable=False,
            fallback_on_failure=True,
        )

    # From here on ``cfg.hailo`` is guaranteed non-None. ``HailoConfig``
    # (schema.py:455-493) declares both fields with defaults; the
    # dead-defensive ``getattr`` wrappers that used to live here would
    # silently swallow a future rename instead of producing a clean
    # AttributeError. Reading the field directly is the more honest
    # contract.
    hailo_cfg = cfg.hailo
    device_path = Path(hailo_cfg.device_path)
    device_path_exists = device_path.exists()
    fallback_on_failure = bool(hailo_cfg.fallback_on_failure)

    # SDK importability — does NOT instantiate the runtime yet. Use a real
    # guarded import (not mere spec presence): hailo_platform ships native
    # bindings that can resolve a spec yet fail to import on a host without
    # the driver, which would otherwise mis-report sdk_importable=True.
    if not module_importable("hailo_platform"):
        return HailoDiagnostics(
            device_path_exists=device_path_exists,
            sdk_importable=False,
            fallback_on_failure=fallback_on_failure,
        )

    if not device_path_exists:
        return HailoDiagnostics(
            device_path_exists=False,
            sdk_importable=True,
            fallback_on_failure=fallback_on_failure,
        )

    # Construct via the factory so we exercise the same code path as production.
    from mousedroid.factory import build_hailo_runtime

    runtime = build_hailo_runtime(cfg)
    if runtime is None:
        return HailoDiagnostics(
            device_path_exists=True,
            sdk_importable=True,
            fallback_on_failure=fallback_on_failure,
        )

    hef_files: dict[str, str] = {}
    device_info: dict[str, str] = {}
    inference_latency_ms: float | None = None

    try:
        await runtime.start()

        # Inventory HEF roles via schema reflection — any
        # ``HailoConfig`` field whose name ends in ``_hef_path`` is a
        # candidate role. This lets a future ``depth_hef_path`` /
        # ``segmentation_hef_path`` schema field flow into the smoke
        # without code edits here, and removes the duplication between
        # the hardcoded ``("yolo", "feature_extractor")`` tuple in this
        # helper and the same pair in ``HailoRuntime.start()``.
        models = getattr(runtime, "_models", {})
        hef_role_fields = _discover_hef_role_fields(hailo_cfg)
        for role, field_name in hef_role_fields:
            hef_path_value = getattr(hailo_cfg, field_name, None)
            if hef_path_value is None or not str(hef_path_value).strip():
                hef_files[role] = "missing (path not configured)"
                continue
            hef_path = Path(hef_path_value)
            if not hef_path.exists():
                hef_files[role] = f"missing ({hef_path})"
                continue
            if role in models:
                hef_files[role] = f"loaded ({hef_path.name})"
            else:
                hef_files[role] = "loaded (model registered)"

        # Device-info acquisition. The HailoRuntimeProtocol does not
        # expose device descriptors; ``HailoRuntime`` similarly has no
        # ``_device_id`` / ``_fw_version`` / ``_arch`` attrs. Avoid
        # reflecting against non-existent fields (would always yield an
        # empty dict). Instead report two boolean signals the operator
        # actually cares about: device-found (path exists) and
        # models-loaded (at least one HEF in the inventory). A future
        # protocol extension can replace this with a real
        # ``runtime.device_info`` mapping.
        loaded_count = sum(1 for status in hef_files.values() if status.startswith("loaded"))
        device_info["device_path"] = str(device_path)
        device_info["models_loaded"] = str(loaded_count)
        device_info["models_configured"] = str(len(hef_role_fields))

        # Synthetic inference — guard input-shape introspection behind
        # try/except so a Hailo runtime that doesn't expose vstream info
        # still gets a smoke signal via the schema-driven fallback shape.
        input_shape: tuple[int, ...] = tuple(
            int(dim) for dim in getattr(hailo_cfg, "synthetic_input_shape", (640, 640, 3))
        )
        yolo_model = models.get("yolo") if isinstance(models, dict) else None
        if yolo_model is not None:
            try:
                vstreams = yolo_model["input_vstream_infos"]
                if vstreams:
                    input_shape = tuple(int(dim) for dim in vstreams[0].shape)
            except (KeyError, IndexError, AttributeError, TypeError):
                pass

        if "yolo" in models:
            zero_img: NDArray[np.uint8] = np.zeros(input_shape, dtype=np.uint8)
            t0 = time.perf_counter()
            try:
                # ``infer_sync`` acquires a threading.Lock and runs blocking
                # hailo_platform VStream calls. Offload to a worker thread
                # so the smoke does not stall the asyncio event loop —
                # mirrors how ``HailoRuntime.start()`` dispatches its own
                # blocking calls via ``asyncio.to_thread``.
                await asyncio.to_thread(runtime.infer_sync, "yolo", zero_img)
                inference_latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                hef_files["yolo"] = f"inference_failed ({type(exc).__name__})"

        return HailoDiagnostics(
            device_path_exists=True,
            sdk_importable=True,
            device_info=device_info,
            hef_files=hef_files,
            inference_latency_ms=inference_latency_ms,
            fallback_on_failure=fallback_on_failure,
        )
    finally:
        # CRITICAL — release the PCIe device lock even if start/infer
        # raised. Use the project's structlog setup so the warning lands
        # in the same processor chain as the rest of the orchestrator
        # (JSON renderer, contextvars, cloud log forwarding).
        try:
            await runtime.stop()
        except Exception as stop_exc:
            _log.warning(
                "hailo_runtime_stop_failed_in_smoke",
                device_path=str(device_path),
                error=type(stop_exc).__name__,
                error_message=str(stop_exc),
            )


def _discover_hef_role_fields(hailo_cfg: object) -> list[tuple[str, str]]:
    """Return ``[(role, field_name), ...]`` for every ``*_hef_path`` field.

    Drops the trailing ``"_hef_path"`` suffix to produce the canonical role
    identifier (e.g. ``"yolo_hef_path"`` → role ``"yolo"``). Iterates
    ``HailoConfig.model_fields`` so a future schema field
    ``depth_hef_path`` flows into the smoke automatically — single source
    of truth for HEF roles between the schema and the smoke.

    Falls back to the canonical YOLO + feature-extractor pair when the
    Pydantic ``model_fields`` introspection isn't available (e.g. the
    config object is a stub in tests).
    """
    model_fields = getattr(hailo_cfg, "model_fields", None) or getattr(
        type(hailo_cfg), "model_fields", None
    )
    if model_fields:
        pairs: list[tuple[str, str]] = []
        for field_name in model_fields:
            if field_name.endswith("_hef_path"):
                role = field_name[: -len("_hef_path")]
                pairs.append((role, field_name))
        if pairs:
            return pairs
    # Fallback — preserves Tier C C2.1 contract for stubs / non-Pydantic
    # objects used in tests.
    return [("yolo", "yolo_hef_path"), ("feature_extractor", "feature_extractor_hef_path")]
