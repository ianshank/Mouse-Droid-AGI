from __future__ import annotations

import time

import numpy as np
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.world_model.encoder import MultimodalEncoder
from mousedroid.world_model.rssm import RSSM


def _default_cfg() -> ModelConfig:
    return ModelConfig()


def test_encoder_forward_under_50ms() -> None:
    cfg = _default_cfg()
    enc = MultimodalEncoder(cfg)
    enc.eval()
    vision = torch.randn(1, cfg.vision_dim)
    ultrasonic = torch.randn(1, cfg.ultrasonic_dim)
    motor = torch.randn(1, cfg.motor_state_dim)
    mask = torch.ones(1, 3)

    # Warm up
    enc(vision, ultrasonic, motor, mask)

    start = time.perf_counter()
    for _ in range(10):
        enc(vision, ultrasonic, motor, mask)
    elapsed_ms = (time.perf_counter() - start) / 10 * 1000

    assert elapsed_ms < 50, f"Encoder forward took {elapsed_ms:.2f}ms"


def test_rssm_observe_step_under_100ms() -> None:
    cfg = _default_cfg()
    rssm = RSSM(cfg)
    rssm.eval()
    obs = MouseDroidObservationBundle(
        _vision_features=np.zeros(cfg.vision_dim, dtype=np.float32),
        _distance_m=2.0,
        _motor_state=np.zeros(cfg.motor_state_dim, dtype=np.float32),
        _valid_mask=np.ones(3, dtype=np.float32),
    )
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    prev_action = torch.zeros(1, cfg.action_dim)

    # Warm up
    rssm.observe_step(obs, prev_action, h, z)

    start = time.perf_counter()
    for _ in range(10):
        rssm.observe_step(obs, prev_action, h, z)
    elapsed_ms = (time.perf_counter() - start) / 10 * 1000

    assert elapsed_ms < 100, f"RSSM observe_step took {elapsed_ms:.2f}ms"


def test_planning_budget_under_1ms() -> None:
    from mousedroid.agents._planning import compute_mcts_budget

    start = time.perf_counter()
    for _ in range(1000):
        compute_mcts_budget(surprise=1.0, base=50, maximum=200)
    elapsed_ms = (time.perf_counter() - start) / 1000 * 1000

    assert elapsed_ms < 1, f"Budget computation took {elapsed_ms:.4f}ms"


def test_observation_bundle_creation_under_1ms() -> None:
    start = time.perf_counter()
    for _ in range(1000):
        MouseDroidObservationBundle(
            _vision_features=np.zeros(256, dtype=np.float32),
            _distance_m=2.0,
            _motor_state=np.zeros(4, dtype=np.float32),
            _valid_mask=np.ones(3, dtype=np.float32),
        )
    elapsed_ms = (time.perf_counter() - start) / 1000 * 1000

    assert elapsed_ms < 1, f"Bundle creation took {elapsed_ms:.4f}ms"


def test_safety_evaluate_under_5ms() -> None:
    from mousedroid.config.schema import SafetyConfig
    from mousedroid.safety.monitor import MouseDroidSafetyMonitor

    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    obs = MouseDroidObservationBundle(
        _distance_m=2.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _valid_mask=np.ones(3, dtype=np.float32),
    )

    # Warm up
    monitor.evaluate(obs, 10.0)

    start = time.perf_counter()
    for _ in range(100):
        monitor.evaluate(obs, 10.0)
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000

    assert elapsed_ms < 5, f"Safety evaluate took {elapsed_ms:.2f}ms"
