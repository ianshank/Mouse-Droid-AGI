"""Tests for the shared ONNX I/O contract in ``world_model.onnx_io``.

The module is the single source of truth for tensor names that flow
between the export script and the runtime class. These tests lock in:

1. The required-input list (always present in the exported graph)
2. The optional-input filtering driven by ``ModelConfig`` toggles
3. The output names matching the ``observe_step_traceable`` tuple
   position (new_h, new_z, obs_embed, surprise)

If a future PR renames any of these strings, the export script and the
runtime class import the constants — so the rename happens in one
place and these tests catch the side-effect on the public ONNX schema.
"""

from __future__ import annotations

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.onnx_io import (
    OBSERVE_STEP_BATCH_DIM_NAME,
    OBSERVE_STEP_INPUT_AUDIO,
    OBSERVE_STEP_INPUT_H,
    OBSERVE_STEP_INPUT_LIDAR,
    OBSERVE_STEP_INPUT_MOTOR,
    OBSERVE_STEP_INPUT_PREV_ACTION,
    OBSERVE_STEP_INPUT_ULTRASONIC,
    OBSERVE_STEP_INPUT_VALID_MASK,
    OBSERVE_STEP_INPUT_VISION,
    OBSERVE_STEP_INPUT_Z,
    OBSERVE_STEP_OUTPUT_NAMES,
    OBSERVE_STEP_OUTPUT_NEW_H,
    OBSERVE_STEP_OUTPUT_NEW_Z,
    OBSERVE_STEP_OUTPUT_OBS_EMBED,
    OBSERVE_STEP_OUTPUT_SURPRISE,
    all_input_names_for_cfg,
    optional_input_names_for_cfg,
    required_input_names,
)


def _cfg(*, ultrasonic: bool, audio: bool, lidar: bool) -> ModelConfig:
    """Build a ModelConfig with the requested optional modalities enabled."""
    return ModelConfig(
        ultrasonic_dim=1 if ultrasonic else 0,
        ultrasonic_proj_dim=4 if ultrasonic else 0,
        audio_dim=8 if audio else 0,
        audio_proj_dim=4 if audio else 0,
        lidar_dim=12 if lidar else 0,
        lidar_proj_dim=4 if lidar else 0,
        # cfc_hidden_dim > 0 so the DualStreamRSSM path is exercised.
        cfc_hidden_dim=16,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


class TestRequiredInputNames:
    """The 6 required inputs are stable across all configs."""

    def test_returns_6_names_in_canonical_order(self) -> None:
        names = required_input_names()
        assert names == (
            OBSERVE_STEP_INPUT_VISION,
            OBSERVE_STEP_INPUT_MOTOR,
            OBSERVE_STEP_INPUT_VALID_MASK,
            OBSERVE_STEP_INPUT_PREV_ACTION,
            OBSERVE_STEP_INPUT_H,
            OBSERVE_STEP_INPUT_Z,
        )

    def test_all_input_names_unique(self) -> None:
        """No duplicates — ONNX inputs use names as keys, dup would crash."""
        names = required_input_names()
        assert len(names) == len(set(names))


class TestOptionalInputNames:
    """Optional inputs are filtered by ``ModelConfig`` modality toggles."""

    def test_all_disabled_returns_empty(self) -> None:
        # Need at least one distance modality enabled to satisfy the
        # Pydantic validator (`ultrasonic_dim == 0 AND lidar_dim == 0` is
        # rejected); enable ultrasonic, disable the other two.
        cfg = _cfg(ultrasonic=True, audio=False, lidar=False)
        optional = optional_input_names_for_cfg(cfg)
        assert optional == (OBSERVE_STEP_INPUT_ULTRASONIC,)

    def test_all_enabled_returns_three_in_order(self) -> None:
        cfg = _cfg(ultrasonic=True, audio=True, lidar=True)
        optional = optional_input_names_for_cfg(cfg)
        assert optional == (
            OBSERVE_STEP_INPUT_ULTRASONIC,
            OBSERVE_STEP_INPUT_AUDIO,
            OBSERVE_STEP_INPUT_LIDAR,
        )

    def test_lidar_only_skips_ultrasonic_audio(self) -> None:
        cfg = _cfg(ultrasonic=False, audio=False, lidar=True)
        optional = optional_input_names_for_cfg(cfg)
        assert optional == (OBSERVE_STEP_INPUT_LIDAR,)


class TestAllInputNames:
    """``all_input_names_for_cfg`` concatenates required + optional in order."""

    def test_combines_required_and_optional(self) -> None:
        cfg = _cfg(ultrasonic=True, audio=True, lidar=True)
        names = all_input_names_for_cfg(cfg)
        assert names[:6] == required_input_names()
        assert names[6:] == (
            OBSERVE_STEP_INPUT_ULTRASONIC,
            OBSERVE_STEP_INPUT_AUDIO,
            OBSERVE_STEP_INPUT_LIDAR,
        )
        # Full list is unique (no key collision in ONNX feeds).
        assert len(names) == len(set(names))


class TestOutputNames:
    """The output tuple order matches ``observe_step_traceable``'s return."""

    def test_output_names_tuple_has_4_elements(self) -> None:
        assert len(OBSERVE_STEP_OUTPUT_NAMES) == 4

    def test_output_names_in_canonical_order(self) -> None:
        """Index 0=new_h, 1=new_z, 2=obs_embed, 3=surprise.

        Locked in so the runtime class can index the ORT output list by
        position without doing string lookups — and so a future renaming
        propagates through this constant tuple instead of the runtime.
        """
        assert OBSERVE_STEP_OUTPUT_NAMES[0] == OBSERVE_STEP_OUTPUT_NEW_H
        assert OBSERVE_STEP_OUTPUT_NAMES[1] == OBSERVE_STEP_OUTPUT_NEW_Z
        assert OBSERVE_STEP_OUTPUT_NAMES[2] == OBSERVE_STEP_OUTPUT_OBS_EMBED
        assert OBSERVE_STEP_OUTPUT_NAMES[3] == OBSERVE_STEP_OUTPUT_SURPRISE


class TestBatchDimName:
    """The dynamic batch axis name is a stable symbolic string."""

    def test_batch_dim_name_is_non_empty_string(self) -> None:
        assert isinstance(OBSERVE_STEP_BATCH_DIM_NAME, str)
        assert OBSERVE_STEP_BATCH_DIM_NAME
