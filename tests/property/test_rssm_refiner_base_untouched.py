"""Property test: the RSSM refiner NEVER mutates the base RSSM in place.

The load-bearing safety invariant for Phase-6 WS-E2 (mirrors
``tests/property/test_on_device_no_inplace_corruption.py`` for the EWC stand-in):
the live RSSM the refiner holds must survive a bounded ``train_sequence``-driven
refinement bitwise-unchanged, so the regression gate (WS-E3) can always fall back
to it. Hypothesis fuzzes the seed, refinement window, batch episodes, learning
rate and step count; for every draw we snapshot every base parameter, run the
refine, and assert the base is still ``torch.equal`` to its snapshot AND the
candidate tensors are SEPARATE objects (distinct ``data_ptr``).
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import ModelConfig, OnDeviceLearningConfig
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.learning.on_device.rssm_refiner import (
    RSSMRefiner,
    build_sequence_batch,
)
from mousedroid.world_model.rssm import RSSM

_DEVICE = torch.device("cpu")


def _make_records(n: int, *, seed: int) -> list[MouseDroidExperienceRecord]:
    rng = np.random.default_rng(seed)
    return [
        MouseDroidExperienceRecord(
            timestamp=float(i),
            vision_features=np.zeros(0, dtype=np.float32),
            distance_m=float(rng.uniform(0.1, 2.0)),
            motor_state=rng.standard_normal(4).astype(np.float32),
            action=rng.standard_normal(3).astype(np.float32),
            reward=float(rng.uniform(-1.0, 1.0)),
        )
        for i in range(n)
    ]


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**16),
    sequence_length=st.integers(min_value=2, max_value=5),
    n_episodes=st.integers(min_value=1, max_value=3),
    steps=st.integers(min_value=1, max_value=4),
    learning_rate=st.floats(min_value=1e-3, max_value=0.5),
)
def test_base_rssm_unchanged_after_refine(
    seed: int,
    sequence_length: int,
    n_episodes: int,
    steps: int,
    learning_rate: float,
) -> None:
    """Base RSSM parameters are bitwise-identical before and after refine."""
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vision_dim=0,
        vision_proj_dim=0,
        ultrasonic_dim=1,
        ultrasonic_proj_dim=4,
        motor_state_dim=4,
        hidden_dim=6,
        latent_dim=4,
        action_dim=3,
        obs_dim=6,
        motor_proj_dim=4,
    )
    base = RSSM(cfg)
    base.eval()

    before = {name: p.detach().clone() for name, p in base.named_parameters()}

    records = _make_records(n_episodes * sequence_length + 5, seed=seed + 1)
    batch = build_sequence_batch(
        records,
        cfg,
        base.encoder,
        sequence_length=sequence_length,
        n_episodes=n_episodes,
        device=_DEVICE,
    )
    ocfg = OnDeviceLearningConfig(
        enabled=True,
        update_steps=steps,
        learning_rate=learning_rate,
        ewc_lambda=0.0,
        refine_sequence_length=sequence_length,
        refine_batch_episodes=n_episodes,
    )

    result = RSSMRefiner(base, ocfg).update(batch)

    # The base must be untouched in place...
    for name, param in base.named_parameters():
        assert torch.equal(param.detach(), before[name]), f"base param {name!r} mutated"

    # ...and the candidate must be a SEPARATE object (not the base's tensors).
    base_params = dict(base.named_parameters())
    for name in before:
        assert result.candidate_state_dict[name].data_ptr() != base_params[name].data_ptr(), (
            f"candidate shares storage with base for {name!r}"
        )
