"""Unit tests for extract_commentary_facts (pure, NaN/empty-safe)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mousedroid.commentary.facts import extract_commentary_facts


class _Obs:
    """Minimal ObservationProtocol stand-in with overridable fields."""

    def __init__(
        self,
        *,
        distance_m: float = 2.0,
        motor_state: np.ndarray | None = None,
        audio_chunk: np.ndarray | None = None,
        lidar_features: np.ndarray | None = None,
        timestamp: float = 1.0,
    ) -> None:
        self.timestamp = timestamp
        self.vision_features = np.zeros(4, dtype=np.float32)
        self.distance_m = distance_m
        self.motor_state = (
            motor_state
            if motor_state is not None
            else np.array([0.0, 0.0, 0.0, 11.5], dtype=np.float32)
        )
        self.audio_chunk = audio_chunk if audio_chunk is not None else np.zeros(8, dtype=np.float32)
        self.lidar_features = lidar_features
        self.valid_mask = np.ones(5, dtype=np.float32)
        self.n_modalities = 5


def test_nominal_floats_not_numpy_scalars() -> None:
    obs = _Obs(
        lidar_features=np.array([1.5, 2.0, 3.0], dtype=np.float32),
        motor_state=np.array([0.3, 0.4, 0.2, 11.0], dtype=np.float32),
    )
    f = extract_commentary_facts(obs, novelty=0.5, is_emergency=False)
    assert f.min_clearance_m == 1.5
    assert f.lidar_valid is True
    assert type(f.min_clearance_m) is float
    assert type(f.speed_mps) is float
    assert f.speed_mps == pytest.approx(0.5, abs=1e-5)
    assert f.battery_v == 11.0
    assert f.novelty == 0.5


def test_lidar_none_falls_back_to_forward_distance() -> None:
    obs = _Obs(distance_m=2.0, lidar_features=None)
    f = extract_commentary_facts(obs, novelty=None, is_emergency=False)
    assert f.lidar_valid is False
    assert f.min_clearance_m == 2.0
    assert f.novelty is None


def test_lidar_empty_array_invalid() -> None:
    obs = _Obs(lidar_features=np.array([], dtype=np.float32))
    f = extract_commentary_facts(obs, novelty=None, is_emergency=False)
    assert f.lidar_valid is False


def test_empty_audio_no_warning_and_invalid() -> None:
    obs = _Obs(audio_chunk=np.array([], dtype=np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning would fail the test
        f = extract_commentary_facts(obs, novelty=None, is_emergency=False)
    assert f.audio_rms == 0.0
    assert f.audio_valid is False


def test_audio_with_nan_inf_is_finite() -> None:
    obs = _Obs(audio_chunk=np.array([np.inf, np.nan, 1.0], dtype=np.float32))
    f = extract_commentary_facts(obs, novelty=None, is_emergency=False)
    assert np.isfinite(f.audio_rms)
    assert f.audio_valid is False


def test_short_motor_state_no_index_error() -> None:
    obs = _Obs(motor_state=np.array([0.1], dtype=np.float32))  # only vx
    f = extract_commentary_facts(obs, novelty=None, is_emergency=False)
    assert f.battery_v == 12.0  # default when index absent
    assert f.turn_rate == 0.0


def test_emergency_and_timestamp_passthrough() -> None:
    obs = _Obs(timestamp=42.0)
    f = extract_commentary_facts(obs, novelty=1.0, is_emergency=True)
    assert f.is_emergency is True
    assert f.timestamp == 42.0


def test_nonfinite_novelty_coerced() -> None:
    obs = _Obs()
    f = extract_commentary_facts(obs, novelty=float("nan"), is_emergency=False)
    assert f.novelty == 0.0  # NaN coerced to safe 0.0 (still not None)


def test_embedding_defaults_none_and_threads_through() -> None:
    obs = _Obs()
    assert extract_commentary_facts(obs, novelty=None, is_emergency=False).embedding is None
    emb = np.arange(4, dtype=np.float32)
    f = extract_commentary_facts(obs, novelty=None, is_emergency=False, embedding=emb)
    assert f.embedding is emb
