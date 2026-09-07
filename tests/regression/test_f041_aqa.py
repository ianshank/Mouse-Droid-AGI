"""AQA pins for F-041 BDI portfolio trainer honesty."""

from __future__ import annotations

from pathlib import Path

from training.collect_annotations import INTENTION_FEATURE_DIM, INTENTION_FEATURE_NAMES
from training.train_bdi import passes_bdi_publish_bars

from mousedroid.config.schema import WeightUpdatePollConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_he_init_is_local_to_training() -> None:
    text = (_REPO_ROOT / "training" / "train_bdi.py").read_text(encoding="utf-8")
    assert "sqrt(2.0 / fan_in)" in text
    assert "WEIGHT_INIT_SCALE" in text  # mentioned as the runtime constant we must not touch
    assert "from mousedroid.constants import (\n    WEIGHT_INIT_SCALE" not in text


def test_belief_trainer_uses_adam() -> None:
    text = (_REPO_ROOT / "training" / "train_bdi.py").read_text(encoding="utf-8")
    assert "AdamOptimizer" in text
    assert "w1 -= lr * d_w1" not in text


def test_train_bdi_does_not_upload() -> None:
    text = (_REPO_ROOT / "training" / "train_bdi.py").read_text(encoding="utf-8")
    assert "upload_weights" not in text


def test_intention_feature_layout_is_pinned() -> None:
    assert INTENTION_FEATURE_NAMES == (
        "vx",
        "vy",
        "omega",
        "speed",
        "abs_omega",
        "distance_m",
        "battery_v",
        "human_detected",
        "human_dist_m",
        "commanded",
    )
    assert INTENTION_FEATURE_DIM == 10


def test_ota_poller_default_is_policy_v2_not_weights_repo() -> None:
    info = WeightUpdatePollConfig.model_fields["policy_repo_id"]
    assert info.default == "ianshank/mousedroid-policy-v2"
    assert "mousedroid-weights" not in str(info.default)


def test_publish_bars_are_strict() -> None:
    assert not passes_bdi_publish_bars(
        belief_mse=1.0,
        pca_mse=0.46,
        intention_acc=0.12,
        majority_acc=0.12,
    )
