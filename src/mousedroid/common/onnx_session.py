"""Neutral ONNX Runtime session-lifecycle helpers.

Both :class:`mousedroid.vla.policy.DistilledVLAOnnx` and
:class:`mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
wrap an ``onnxruntime.InferenceSession`` with the same cohesive
lifecycle: a TensorRT → CUDA → CPU execution-provider fallback, a lazy
``onnxruntime`` import deferred to warmup, a zero-filled warmup pass that
inspects the live graph's input metadata, and structured start/pass/
complete logging.

This module captures that shared logic ONCE so neither wrapper
copy-pastes it. It is deliberately **neutral**: it imports neither the
``vla`` package nor the ``world_model`` package, depending only on
``onnxruntime`` (lazily, inside :func:`warmup_session`), the standard
library, and :mod:`mousedroid.logging`. Routing both wrappers through
here therefore removes the duplication WITHOUT introducing a
``world_model → vla`` (or reverse) import — preserving the original
"keep the runtime independent of the VLA module" constraint that made
the two wrappers carry their own copies.

Per-wrapper differences are parameters, not branches:

* the structlog event-name prefix is the ``log_prefix`` argument
  (``"distilled_vla_onnx"`` vs ``"dual_stream_rssm_onnx"``);
* the ONNX output names requested from ``session.run`` are the
  ``output_names`` argument (``[action_output_name]`` vs the world-model
  ``OBSERVE_STEP_OUTPUT_NAMES`` list).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_log = get_logger(__name__)


def resolve_providers(
    requested: tuple[str, ...],
    available: tuple[str, ...],
) -> tuple[str, ...]:
    """Intersect ``requested`` with ``available`` preserving order.

    Always falls back to ``CPUExecutionProvider`` if it is available and
    the intersection is empty, so warmup never raises on a host that has
    at least the CPU provider.

    Args:
        requested: The desired execution-provider chain, in priority
            order (e.g. TensorRT → CUDA → CPU).
        available: Providers the local ``onnxruntime`` build reports via
            ``onnxruntime.get_available_providers()``.

    Returns:
        The requested providers that are available, in requested order;
        ``("CPUExecutionProvider",)`` when the intersection is empty but
        CPU is available; or an empty tuple in the pathological case
        where no providers are available (the caller lets ORT raise).
    """
    chosen = tuple(p for p in requested if p in available)
    if chosen:
        return chosen
    if "CPUExecutionProvider" in available:
        return ("CPUExecutionProvider",)
    # Pathological case: no providers available — let ORT raise.
    return ()


def run_session_with_zeros(
    session: Any,
    output_names: Sequence[str],
) -> None:
    """Run a single dummy inference using zero-filled inputs.

    Inspects the live session's input metadata so warmup does not require
    knowing latent shapes a priori — those come from the ONNX graph
    itself. Dynamic / non-positive dimensions collapse to ``1``.

    Args:
        session: A live ``onnxruntime.InferenceSession`` (or a stub with
            the same ``get_inputs()`` / ``run()`` surface).
        output_names: ONNX output names to request from ``session.run``.
            Passed through unchanged so each wrapper supplies its own
            output list.
    """
    feeds: dict[str, Any] = {}
    # numpy is a project dependency (torch -> numpy); import locally to
    # keep the module-level import graph minimal.
    import numpy as _np

    # Map each input's declared ONNX element type to a NumPy dtype so the
    # zero-filled warmup feeds match the graph (a model may declare int64 /
    # bool inputs, not only float — feeding float32 to those raises
    # InvalidArgument in ONNX Runtime). Unknown / absent types fall back to
    # float32, preserving the original all-float behaviour for current models.
    dtype_by_onnx_type: dict[str, Any] = {
        "tensor(float)": _np.float32,
        "tensor(double)": _np.float64,
        "tensor(float16)": _np.float16,
        "tensor(int64)": _np.int64,
        "tensor(int32)": _np.int32,
        "tensor(int16)": _np.int16,
        "tensor(int8)": _np.int8,
        "tensor(uint8)": _np.uint8,
        "tensor(bool)": _np.bool_,
    }

    for inp in session.get_inputs():
        shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in (inp.shape or []))
        dtype = dtype_by_onnx_type.get(getattr(inp, "type", ""), _np.float32)
        feeds[inp.name] = _np.zeros(shape, dtype=dtype)
    session.run(list(output_names), feeds)


def warmup_session(
    model_path: Path,
    requested_providers: tuple[str, ...],
    warmup_iterations: int,
    output_names: Sequence[str],
    *,
    log_prefix: str,
) -> tuple[Any, tuple[str, ...]]:
    """Create an ORT session and run zero-filled warmup inferences.

    This is the only place that imports ``onnxruntime``. The caller owns
    the idempotency guard (skipping this call once a session exists) and
    assigns the returned session / providers onto its own attributes, so
    each wrapper keeps its public method surface unchanged.

    Args:
        model_path: Filesystem path to the ``.onnx`` graph. Must exist.
        requested_providers: Desired execution-provider chain, in
            priority order.
        warmup_iterations: Number of zero-filled dummy passes to run
            against the freshly created session.
        output_names: ONNX output names to request on each warmup pass.
        log_prefix: Event-name prefix for the structured ``_warmup_start``
            / ``_warmup_pass`` / ``_warmup_complete`` logs (e.g.
            ``"distilled_vla_onnx"`` or ``"dual_stream_rssm_onnx"``).

    Returns:
        A ``(session, active_providers)`` tuple — the created
        ``onnxruntime.InferenceSession`` and the concrete provider chain
        chosen after intersecting ``requested_providers`` with the
        locally available providers.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ImportError: If ``onnxruntime`` is not installed.
    """
    if not model_path.is_file():
        msg = f"ONNX model not found at {model_path}"
        raise FileNotFoundError(msg)

    # Lazy import keeps importers of this module free of onnxruntime.
    import onnxruntime as ort

    available = tuple(ort.get_available_providers())
    active = resolve_providers(requested_providers, available)
    _log.info(
        f"{log_prefix}_warmup_start",
        requested_providers=list(requested_providers),
        available_providers=list(available),
        active_providers=list(active),
        model_path=str(model_path),
    )

    session = ort.InferenceSession(
        str(model_path),
        providers=list(active),
    )

    for i in range(warmup_iterations):
        run_session_with_zeros(session, output_names)
        _log.debug(f"{log_prefix}_warmup_pass", iteration=i + 1)

    _log.info(
        f"{log_prefix}_warmup_complete",
        active_providers=list(active),
        warmup_iterations=warmup_iterations,
    )
    return session, active
