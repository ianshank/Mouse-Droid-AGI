"""AQA: schema-field hygiene + protocol conformance for the growth pillar.

Pins the non-negotiable contracts so a future refactor cannot silently drift:
the ``growth`` block is Optional/default-None with an operator description, the
metrics flag defaults on with a description, the ``slot_dir`` validator enforces
experience-root containment, and ``KnowledgeDistiller`` still satisfies
``GrowthProtocol``.
"""

from __future__ import annotations

import inspect

import pytest
import torch.nn as nn

from mousedroid.config.schema import GrowthConfig, MetricsConfig, Settings
from mousedroid.growth.distillation import KnowledgeDistiller
from mousedroid.growth.protocol import GrowthProtocol


def test_growth_field_is_optional_default_none() -> None:
    field = Settings.model_fields["growth"]
    assert field.default is None
    assert field.description, "growth must carry an operator description"


def test_metrics_flag_default_on_with_description() -> None:
    field = MetricsConfig.model_fields["track_growth_distillation"]
    assert field.default is True
    assert field.description


def test_enabled_defaults_off() -> None:
    """The master switch defaults off (soak-gated deployment)."""
    assert GrowthConfig().enabled is False


@pytest.mark.parametrize("bad", ["/abs", "..", "../escape", "", "  "])
def test_slot_dir_validator_rejects_escapes(bad: str) -> None:
    with pytest.raises(ValueError, match="slot_dir"):
        GrowthConfig(slot_dir=bad)


def test_slot_dir_accepts_relative_leaf() -> None:
    assert GrowthConfig(slot_dir="growth_slot").slot_dir == "growth_slot"


def test_distiller_satisfies_growth_protocol() -> None:
    d = KnowledgeDistiller(nn.Linear(4, 3), nn.Linear(4, 3), temperature=2.0, alpha=0.5, lr=0.05)
    assert isinstance(d, GrowthProtocol)


def test_distill_step_hard_labels_optional_signature() -> None:
    """The protocol + impl expose ``hard_labels`` as optional (regression seam)."""
    sig = inspect.signature(KnowledgeDistiller.distill_step)
    assert sig.parameters["hard_labels"].default is None
