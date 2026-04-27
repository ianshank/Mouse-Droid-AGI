"""Phase 4: tests for VLM-derived dense progress reward.

Covers:
    * :class:`VLMProgressHead` cache hit/miss + LRU eviction.
    * :class:`MockVLMProgress` value validation.
    * Integration with :class:`MultiObjectiveRewardModel`:
        - Byte-identical aggregation when ``vlm_head`` is ``None``.
        - Additive contribution when present and ``weight_vlm_progress > 0``.
        - **Constitutional override**: a contrived high VLM reward that
          coincides with a Law-1 harm signal is still suppressed by the
          multiplicative sigmoid gate (Hypothesis property test).
    * :func:`build_reward_model` wiring (default-off, opt-in).
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import (
    ModelConfig,
    RewardConfig,
    ThreeLawsConfig,
    VLMProgressConfig,
)
from mousedroid.reward.model import MultiObjectiveRewardModel
from mousedroid.reward.vlm_progress import (
    MockVLMProgress,
    VLMProgressBackend,
    VLMProgressHead,
)


def _make_model_cfg() -> ModelConfig:
    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        motor_state_dim=4,
        hidden_dim=32,
        latent_dim=8,
        action_dim=3,
        obs_dim=16,
        vision_proj_dim=8,
        ultrasonic_proj_dim=4,
        motor_proj_dim=4,
    )


class _CountingBackend:
    """Counts every call so cache hits don't reach the backend."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value
        self.calls = 0

    def score(self, prev_obs: torch.Tensor, curr_obs: torch.Tensor, instruction: str) -> float:
        del prev_obs, curr_obs, instruction
        self.calls += 1
        return self.value


class TestMockVLMProgress:
    def test_returns_constant(self) -> None:
        backend = MockVLMProgress(0.42)
        out = backend.score(torch.zeros(1, 4), torch.zeros(1, 4), "go forward")
        assert out == pytest.approx(0.42)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            MockVLMProgress(1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            MockVLMProgress(-0.1)

    def test_satisfies_protocol(self) -> None:
        backend = MockVLMProgress(0.0)
        assert isinstance(backend, VLMProgressBackend)


class TestVLMProgressHead:
    def test_score_shape_and_value(self) -> None:
        cfg = VLMProgressConfig(enabled=True, mock_progress_value=0.3)
        head = VLMProgressHead(cfg)
        prev = torch.zeros(1, 8)
        curr = torch.ones(1, 8)
        out = head.score(prev, curr)
        assert out.shape == (1, 1)
        assert out.item() == pytest.approx(0.3)

    def test_cache_hit_skips_backend(self) -> None:
        cfg = VLMProgressConfig(enabled=True, cache_size=8)
        backend = _CountingBackend(value=0.7)
        head = VLMProgressHead(cfg, backend=backend)
        prev = torch.zeros(1, 4)
        curr = torch.ones(1, 4)
        head.score(prev, curr, instruction="task")
        head.score(prev, curr, instruction="task")
        head.score(prev, curr, instruction="task")
        assert backend.calls == 1
        info = head.cache_info
        assert info["hits"] == 2
        assert info["misses"] == 1

    def test_different_instruction_is_different_key(self) -> None:
        cfg = VLMProgressConfig(enabled=True, cache_size=8)
        backend = _CountingBackend(value=0.4)
        head = VLMProgressHead(cfg, backend=backend)
        prev = torch.zeros(1, 4)
        curr = torch.ones(1, 4)
        head.score(prev, curr, instruction="go left")
        head.score(prev, curr, instruction="go right")
        assert backend.calls == 2

    def test_lru_eviction_respects_maxsize(self) -> None:
        cfg = VLMProgressConfig(enabled=True, cache_size=2)
        backend = _CountingBackend(value=0.5)
        head = VLMProgressHead(cfg, backend=backend)
        # Fill cache with 3 distinct keys → first one evicts.
        for i in range(3):
            head.score(torch.full((1, 4), float(i)), torch.zeros(1, 4), instruction="x")
        # Re-querying the first key must miss again.
        head.score(torch.full((1, 4), 0.0), torch.zeros(1, 4), instruction="x")
        assert backend.calls == 4
        assert head.cache_info["size"] <= 2

    def test_shape_mismatch_raises(self) -> None:
        cfg = VLMProgressConfig(enabled=True)
        head = VLMProgressHead(cfg)
        with pytest.raises(ValueError, match="shape mismatch"):
            head.score(torch.zeros(1, 4), torch.zeros(1, 8))

    def test_backend_out_of_range_raises(self) -> None:
        class BadBackend:
            def score(
                self, prev_obs: torch.Tensor, curr_obs: torch.Tensor, instruction: str
            ) -> float:
                del prev_obs, curr_obs, instruction
                return 2.0

        cfg = VLMProgressConfig(enabled=True)
        head = VLMProgressHead(cfg, backend=cast(VLMProgressBackend, BadBackend()))
        with pytest.raises(ValueError, match="out-of-range"):
            head.score(torch.zeros(1, 4), torch.zeros(1, 4))

    def test_hash_decimals_treats_close_floats_as_equal(self) -> None:
        cfg = VLMProgressConfig(enabled=True, hash_decimals=2, cache_size=8)
        backend = _CountingBackend(value=0.5)
        head = VLMProgressHead(cfg, backend=backend)
        prev = torch.zeros(1, 4)
        # Two values that round to the same 2-decimal grid:
        head.score(prev, torch.full((1, 4), 0.12345), instruction="t")
        head.score(prev, torch.full((1, 4), 0.12399), instruction="t")
        assert backend.calls == 1

    def test_forward_alias_matches_score(self) -> None:
        cfg = VLMProgressConfig(enabled=True, mock_progress_value=0.25)
        head = VLMProgressHead(cfg)
        prev = torch.zeros(1, 4)
        curr = torch.ones(1, 4)
        out_score = head.score(prev, curr, instruction="x").item()
        out_forward = head(prev, curr, instruction="x").item()
        assert out_score == pytest.approx(out_forward)

    def test_instruction_is_keyword_only(self) -> None:
        cfg = VLMProgressConfig(enabled=True)
        head = VLMProgressHead(cfg)
        with pytest.raises(TypeError):
            # Positional instruction must be rejected for API safety.
            head.score(torch.zeros(1, 4), torch.zeros(1, 4), "go")  # type: ignore[misc]


class TestMultiObjectiveBackwardCompat:
    """Without a VLM head the aggregator must be byte-identical to before."""

    def test_aggregate_unchanged_without_vlm_head(self) -> None:
        torch.manual_seed(0)
        model_cfg = _make_model_cfg()
        reward_cfg = RewardConfig(weight_vlm_progress=0.5)  # weight set, but no head
        law_cfg = ThreeLawsConfig()
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg)
        assert model.vlm_head is None
        state = torch.randn(3, 16)
        # Should not raise even though weight is non-zero — head is absent.
        out = model(state)
        assert out.shape == (3, 1)


class TestMultiObjectiveWithVLM:
    def test_vlm_term_contributes_when_enabled(self) -> None:
        torch.manual_seed(0)
        model_cfg = _make_model_cfg()
        reward_cfg = RewardConfig(weight_vlm_progress=0.0)
        vlm_cfg = VLMProgressConfig(enabled=True, mock_progress_value=1.0)
        head = VLMProgressHead(vlm_cfg)
        model_no_weight = MultiObjectiveRewardModel(
            model_cfg, reward_cfg, law_cfg=None, vlm_head=head
        )
        reward_cfg_with = RewardConfig(weight_vlm_progress=0.5)
        head_with = VLMProgressHead(vlm_cfg)
        model_with_weight = MultiObjectiveRewardModel(
            model_cfg, reward_cfg_with, law_cfg=None, vlm_head=head_with
        )
        # Copy weights so the four standard heads agree.
        model_with_weight.heads.load_state_dict(model_no_weight.heads.state_dict())

        state = torch.randn(1, 16)
        prev = torch.zeros(1, 16)
        curr = torch.ones(1, 16)
        r0 = model_no_weight(state, prev_obs=prev, curr_obs=curr).item()
        r1 = model_with_weight(state, prev_obs=prev, curr_obs=curr).item()
        # weight=0.5 * vlm=1.0 added → r1 - r0 ≈ 0.5
        assert math.isclose(r1 - r0, 0.5, abs_tol=1e-5)

    def test_weight_zero_makes_vlm_invisible(self) -> None:
        torch.manual_seed(0)
        model_cfg = _make_model_cfg()
        reward_cfg = RewardConfig(weight_vlm_progress=0.0)
        vlm_cfg = VLMProgressConfig(enabled=True, mock_progress_value=1.0)
        head = VLMProgressHead(vlm_cfg)
        with_head = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=None, vlm_head=head)
        without_head = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=None, vlm_head=None)
        without_head.heads.load_state_dict(with_head.heads.state_dict())

        state = torch.randn(1, 16)
        prev = torch.zeros(1, 16)
        curr = torch.ones(1, 16)
        r_with = with_head(state, prev_obs=prev, curr_obs=curr).item()
        r_without = without_head(state, prev_obs=prev, curr_obs=curr).item()
        assert math.isclose(r_with, r_without, abs_tol=1e-6)

    def test_compute_reward_omits_vlm_when_obs_missing(self) -> None:
        model_cfg = _make_model_cfg()
        reward_cfg = RewardConfig(weight_vlm_progress=0.5)
        vlm_cfg = VLMProgressConfig(enabled=True, mock_progress_value=0.5)
        head = VLMProgressHead(vlm_cfg)
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=None, vlm_head=head)
        state = torch.randn(1, 16)
        scores = model.compute_reward(state)
        assert "vlm_progress" not in scores


class TestConstitutionalOverride:
    """A contrived high VLM reward must not bypass Law-1 harm gating."""

    def test_law1_violation_zeros_vlm_contribution(self) -> None:
        torch.manual_seed(0)
        model_cfg = _make_model_cfg()
        reward_cfg = RewardConfig(weight_vlm_progress=1.0)
        law_cfg = ThreeLawsConfig()
        vlm_cfg = VLMProgressConfig(enabled=True, mock_progress_value=1.0)
        head = VLMProgressHead(vlm_cfg)
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg, vlm_head=head)
        assert model.law_head is not None

        # Force law1_harm to a strongly-negative bias → sigmoid ≈ 0.
        with torch.no_grad():
            model.law_head.heads["law1_harm"].weight.fill_(0.0)
            model.law_head.heads["law1_harm"].bias.fill_(-20.0)
            model.law_head.heads["law2_obedience"].weight.fill_(0.0)
            model.law_head.heads["law2_obedience"].bias.fill_(0.0)
            model.law_head.heads["law3_preservation"].weight.fill_(0.0)
            model.law_head.heads["law3_preservation"].bias.fill_(0.0)
            # Zero the four base heads so we isolate the VLM contribution.
            for name in ("truthfulness", "helpfulness", "safety", "engagement"):
                model.heads[name].weight.fill_(0.0)
                model.heads[name].bias.fill_(0.0)

        state = torch.randn(1, 16)
        prev = torch.zeros(1, 16)
        curr = torch.ones(1, 16)
        out = model(state, prev_obs=prev, curr_obs=curr).item()
        # VLM = 1.0, weight = 1.0, harm gate ≈ 0 → contribution ≈ 0.
        assert abs(out) < 1e-3

    def test_no_harm_lets_vlm_contribute(self) -> None:
        torch.manual_seed(0)
        model_cfg = _make_model_cfg()
        reward_cfg = RewardConfig(weight_vlm_progress=1.0)
        law_cfg = ThreeLawsConfig()
        vlm_cfg = VLMProgressConfig(enabled=True, mock_progress_value=1.0)
        head = VLMProgressHead(vlm_cfg)
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg, vlm_head=head)
        assert model.law_head is not None
        with torch.no_grad():
            # Strongly positive law1 → sigmoid ≈ 1.
            model.law_head.heads["law1_harm"].weight.fill_(0.0)
            model.law_head.heads["law1_harm"].bias.fill_(20.0)
            for name in ("law2_obedience", "law3_preservation"):
                model.law_head.heads[name].weight.fill_(0.0)
                model.law_head.heads[name].bias.fill_(0.0)
            for name in ("truthfulness", "helpfulness", "safety", "engagement"):
                model.heads[name].weight.fill_(0.0)
                model.heads[name].bias.fill_(0.0)

        state = torch.randn(1, 16)
        prev = torch.zeros(1, 16)
        curr = torch.ones(1, 16)
        out = model(state, prev_obs=prev, curr_obs=curr).item()
        # gate ≈ 1, VLM contribution ≈ 1.0 — only.
        assert math.isclose(out, 1.0, abs_tol=1e-3)

    @settings(max_examples=50, deadline=None)
    @given(
        harm_bias=st.floats(min_value=-30.0, max_value=30.0),
        vlm_value=st.floats(min_value=0.0, max_value=1.0),
        weight=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_property_law1_gate_bounds_vlm(
        self, harm_bias: float, vlm_value: float, weight: float
    ) -> None:
        """For any (harm, vlm, weight), VLM contribution ≤ sigmoid(harm)*weight*vlm."""
        torch.manual_seed(0)
        model_cfg = _make_model_cfg()
        reward_cfg = RewardConfig(weight_vlm_progress=weight)
        law_cfg = ThreeLawsConfig()
        vlm_cfg = VLMProgressConfig(enabled=True, mock_progress_value=vlm_value)
        head = VLMProgressHead(vlm_cfg)
        model = MultiObjectiveRewardModel(model_cfg, reward_cfg, law_cfg=law_cfg, vlm_head=head)
        assert model.law_head is not None
        with torch.no_grad():
            model.law_head.heads["law1_harm"].weight.fill_(0.0)
            model.law_head.heads["law1_harm"].bias.fill_(harm_bias)
            for name in ("law2_obedience", "law3_preservation"):
                model.law_head.heads[name].weight.fill_(0.0)
                model.law_head.heads[name].bias.fill_(0.0)
            for name in ("truthfulness", "helpfulness", "safety", "engagement"):
                model.heads[name].weight.fill_(0.0)
                model.heads[name].bias.fill_(0.0)

        state = torch.randn(1, 16)
        prev = torch.zeros(1, 16)
        curr = torch.ones(1, 16)
        out = model(state, prev_obs=prev, curr_obs=curr).item()
        expected = (1.0 / (1.0 + math.exp(-harm_bias))) * weight * vlm_value
        assert math.isclose(out, expected, abs_tol=1e-4)


class TestFactoryWiring:
    def test_build_reward_model_default_no_vlm(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_reward_model

        cfg = Settings()
        model = build_reward_model(cfg)
        assert model.vlm_head is None

    def test_build_reward_model_enabled_attaches_head(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_reward_model

        cfg = Settings()
        cfg.reward.weight_vlm_progress = 0.3
        cfg.reward.vlm_progress.enabled = True
        model = build_reward_model(cfg)
        assert model.vlm_head is not None

    def test_build_reward_model_enabled_but_zero_weight_no_head(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_reward_model

        cfg = Settings()
        cfg.reward.vlm_progress.enabled = True  # but weight stays 0
        model = build_reward_model(cfg)
        assert model.vlm_head is None
