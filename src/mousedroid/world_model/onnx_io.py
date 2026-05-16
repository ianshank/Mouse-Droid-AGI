"""Shared ONNX I/O contract for ``DualStreamRSSM.observe_step``.

Single source of truth for the tensor names that flow between
:func:`scripts.export_dual_stream_rssm_onnx.run_export` (which declares
them at export time) and
:class:`mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
(which feeds them at inference time). Without this module the two
sites would drift over time — e.g. renaming ``"new_h"`` to
``"hidden"`` in the export would silently break inference because the
runtime would look up a key the ONNX graph no longer publishes.

The names are stable strings (not enums) because :mod:`onnxruntime`
takes ``dict[str, np.ndarray]`` for ``run()`` feeds and
``list[str]`` for output names. ``Final[str]`` annotations lock the
identifiers at import time so a future drive-by rename has to update
this single file.

Modality-conditional inputs (``ultrasonic``, ``audio``, ``lidar``) are
declared via the :class:`ObserveStepInputNames` accessor — the
:meth:`enabled_for_cfg` helper takes a :class:`ModelConfig` and returns
the tuple of input names actually present in the exported graph,
preserving the modality-disabled contract from the packer.
"""

from __future__ import annotations

from typing import Final

from mousedroid.config.schema import ModelConfig

# ---------------------------------------------------------------------------
# Required input names (always present in the exported ONNX graph)
# ---------------------------------------------------------------------------

OBSERVE_STEP_INPUT_VISION: Final[str] = "vision"
OBSERVE_STEP_INPUT_MOTOR: Final[str] = "motor"
OBSERVE_STEP_INPUT_VALID_MASK: Final[str] = "valid_mask"
OBSERVE_STEP_INPUT_PREV_ACTION: Final[str] = "prev_action"
OBSERVE_STEP_INPUT_H: Final[str] = "h"
OBSERVE_STEP_INPUT_Z: Final[str] = "z"

# ---------------------------------------------------------------------------
# Optional input names (present only when the corresponding modality is
# enabled in ``ModelConfig``)
# ---------------------------------------------------------------------------

OBSERVE_STEP_INPUT_ULTRASONIC: Final[str] = "ultrasonic"
OBSERVE_STEP_INPUT_AUDIO: Final[str] = "audio"
OBSERVE_STEP_INPUT_LIDAR: Final[str] = "lidar"

# ---------------------------------------------------------------------------
# Output names — order matches DualStreamRSSM.observe_step_traceable's
# return tuple ``(new_h, new_z, obs_embed, surprise)``.
# ---------------------------------------------------------------------------

OBSERVE_STEP_OUTPUT_NEW_H: Final[str] = "new_h"
OBSERVE_STEP_OUTPUT_NEW_Z: Final[str] = "new_z"
OBSERVE_STEP_OUTPUT_OBS_EMBED: Final[str] = "obs_embed"
OBSERVE_STEP_OUTPUT_SURPRISE: Final[str] = "surprise"

OBSERVE_STEP_OUTPUT_NAMES: Final[tuple[str, str, str, str]] = (
    OBSERVE_STEP_OUTPUT_NEW_H,
    OBSERVE_STEP_OUTPUT_NEW_Z,
    OBSERVE_STEP_OUTPUT_OBS_EMBED,
    OBSERVE_STEP_OUTPUT_SURPRISE,
)
"""Ordered output names — index matches the tuple position from
``observe_step_traceable``. The runtime class indexes this tuple so a
reorder propagates correctly."""


# ---------------------------------------------------------------------------
# Dynamic axis naming — exposed so both the exporter and the runtime
# agree on how dynamic batch is declared.
# ---------------------------------------------------------------------------

OBSERVE_STEP_BATCH_DIM_NAME: Final[str] = "batch"
"""Symbolic name for the dynamic batch axis. Used by
``torch.onnx.export`` ``dynamic_axes={name: {0: BATCH_DIM_NAME}}``."""


def required_input_names() -> tuple[str, str, str, str, str, str]:
    """Return the 6 input names always present in the exported graph."""
    return (
        OBSERVE_STEP_INPUT_VISION,
        OBSERVE_STEP_INPUT_MOTOR,
        OBSERVE_STEP_INPUT_VALID_MASK,
        OBSERVE_STEP_INPUT_PREV_ACTION,
        OBSERVE_STEP_INPUT_H,
        OBSERVE_STEP_INPUT_Z,
    )


def optional_input_names_for_cfg(cfg: ModelConfig) -> tuple[str, ...]:
    """Return the modality-conditional input names enabled by ``cfg``.

    The order matches the order operators see in the export script:
    ``ultrasonic`` -> ``audio`` -> ``lidar``. Disabled modalities
    (``cfg.<modality>_dim == 0``) are omitted, mirroring
    :func:`mousedroid.world_model.observation_packer.pack_observation`.

    Args:
        cfg: The model config the export was produced from.

    Returns:
        Tuple of input names. Empty if all three optional modalities are
        disabled (which is a degenerate config but valid).
    """
    names: list[str] = []
    if cfg.ultrasonic_dim > 0:
        names.append(OBSERVE_STEP_INPUT_ULTRASONIC)
    if cfg.audio_dim > 0:
        names.append(OBSERVE_STEP_INPUT_AUDIO)
    if cfg.lidar_dim > 0:
        names.append(OBSERVE_STEP_INPUT_LIDAR)
    return tuple(names)


def all_input_names_for_cfg(cfg: ModelConfig) -> tuple[str, ...]:
    """Return required + optional input names in the export declaration order."""
    return required_input_names() + optional_input_names_for_cfg(cfg)
