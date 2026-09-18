"""MCTS candidate-strategy backwards-compatibility regression tests.

Pins CLAUDE.md invariant 9: new config fields MUST have defaults, and existing
YAML files must load unchanged. The stakes are higher than usual here — this
field selects which actions the planner may propose, so a default drift or an
accidental behaviour change alters actuation on every deployed rover without
touching a single YAML file.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from mousedroid.config.schema import MCTSConfig, Settings
from mousedroid.constants import DEFAULT_ACTION_DIM, DEFAULT_ACTION_LIMIT
from mousedroid.world_model.mcts import MCTSPlanner

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class _StubWorldModel:
    def imagine_step(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return h, z, torch.zeros(1)


def test_mcts_block_without_the_new_key_defaults_to_legacy() -> None:
    """A pre-existing ``mcts:`` block has no ``action_candidate_strategy``."""
    cfg = Settings.model_validate(
        {"mock_hardware": True, "mcts": {"n_action_candidates": 9, "ucb_c": 1.41}}
    )
    assert cfg.mcts.action_candidate_strategy == "shared_axis"
    assert cfg.mcts.n_action_candidates == 9


def test_legacy_yaml_without_the_key_loads_unchanged() -> None:
    """A minimal legacy YAML loads cleanly and unrelated fields survive."""
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    mcts:
      n_simulations_base: 50
      rollout_depth: 5
    """
    cfg = Settings.model_validate(yaml.safe_load(legacy_yaml))
    assert cfg.mcts.action_candidate_strategy == "shared_axis"
    assert cfg.mcts.n_simulations_base == 50
    assert cfg.mcts.rollout_depth == 5


def test_committed_config_overlays_still_load_and_stay_on_legacy() -> None:
    """Every shipped overlay keeps the pre-existing planner behaviour.

    No committed YAML may opt into ``per_axis`` without a deliberate, reviewed
    change — this test is what makes that opt-in visible.
    """
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):  # pragma: no cover - defensive
            continue
        mcts_block = data.get("mcts")
        if isinstance(mcts_block, dict):
            assert "action_candidate_strategy" not in mcts_block, (
                f"{path.name} opts into a non-default planner candidate set"
            )


def test_default_candidate_set_is_byte_identical_to_the_pre_change_formula() -> None:
    """The legacy set still equals ``linspace(-1, 1, n)`` broadcast across axes.

    This is the actual backwards-compatibility contract: not merely "the
    default literal is unchanged", but "the tensor the planner builds under
    that default is unchanged".

    The bounds are spelled as literals on purpose. Building ``expected`` from
    ``DEFAULT_ACTION_LIMIT`` — the same symbol the implementation now reads —
    would move both sides together, so re-pointing that constant would silently
    rescale every legacy deployment while this test stayed green.
    """
    n = 9
    planner = MCTSPlanner(
        MCTSConfig(n_action_candidates=n), _StubWorldModel(), action_dim=DEFAULT_ACTION_DIM
    )
    produced = planner._generate_candidate_actions(torch.device("cpu"))
    expected = torch.linspace(-1.0, 1.0, n).unsqueeze(-1).expand(n, 3)
    assert torch.equal(produced, expected)


def test_action_limit_constant_still_matches_the_legacy_bound() -> None:
    """``DEFAULT_ACTION_LIMIT`` is the normalised bound the legacy set used.

    Separate from the tensor pin above so a constant change fails loudly here
    rather than silently rescaling the planner's action space.
    """
    assert DEFAULT_ACTION_LIMIT == 1.0
    assert DEFAULT_ACTION_DIM == 3
