"""Sub-second sanity smoke for the commentary subsystem."""

from __future__ import annotations

import numpy as np

from mousedroid.commentary.facts import extract_commentary_facts
from mousedroid.commentary.protocol import CommentaryFacts
from mousedroid.config.schema import CommentaryConfig, MetricsConfig
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.metrics import MetricsRegistry


def test_imports_and_config_parse() -> None:
    import mousedroid.commentary.composers
    import mousedroid.commentary.engine  # noqa: F401

    cfg = CommentaryConfig()
    assert cfg.enabled is False


def test_extract_facts_on_default_bundle() -> None:
    obs = MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(4, dtype=np.float32),
        _distance_m=1.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(4, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        _lidar_features=None,
    )
    facts = extract_commentary_facts(obs, novelty=None, is_emergency=False)
    assert isinstance(facts, CommentaryFacts)
    assert facts.lidar_valid is False


def test_metrics_render_after_one_write() -> None:
    reg = MetricsRegistry(MetricsConfig())
    reg.inc_commentary_emitted()
    assert "commentary_emitted_total 1" in reg.render_prometheus()
