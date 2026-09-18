"""Automated Quality Assurance (AQA) — schema hygiene for the MCTS candidate strategy.

Checks hygiene on ``model_fields`` (the ``FieldInfo``), never by instantiating:
instantiation only proves the default is *legal*, not that it is *declared*
the way we think, so a refactor replacing ``Field(...)`` with a plain class
attribute must still be caught here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from mousedroid.config.schema import MCTSConfig

_FIELD = "action_candidate_strategy"


def test_action_candidate_strategy_has_description() -> None:
    """The field carries a non-empty, explanatory description."""
    info: FieldInfo = MCTSConfig.model_fields[_FIELD]
    assert info.description
    assert len(info.description) > 20, info.description


def test_action_candidate_strategy_default_is_shared_axis() -> None:
    """Pinned off FieldInfo — see module docstring.

    The default must stay ``shared_axis``: flipping it would silently change
    which actions the planner may propose on every existing deployment.
    """
    info: FieldInfo = MCTSConfig.model_fields[_FIELD]
    assert info.default == "shared_axis"


def test_description_documents_the_rank_deficiency() -> None:
    """The docstring must keep explaining *why* the default is the weaker set.

    Without this, a future reader sees two strategies and no reason to prefer
    either, and the defect the field exists to expose becomes invisible again.
    """
    info: FieldInfo = MCTSConfig.model_fields[_FIELD]
    description = (info.description or "").lower()
    assert "rank 1" in description
    assert "turn in place" in description


def test_unknown_strategy_is_rejected_at_load() -> None:
    """A typo'd strategy fails at YAML-load time, not silently as a no-op."""
    with pytest.raises(ValidationError, match=_FIELD):
        MCTSConfig(action_candidate_strategy="per-axis")  # type: ignore[arg-type]


@pytest.mark.parametrize("strategy", ["shared_axis", "per_axis"])
def test_documented_strategies_load_cleanly(strategy: str) -> None:
    """Both literals are accepted — proves the validator isn't always-raise."""
    cfg = MCTSConfig(action_candidate_strategy=strategy)  # type: ignore[arg-type]
    assert cfg.action_candidate_strategy == strategy
