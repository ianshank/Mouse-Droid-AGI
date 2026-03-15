"""Post-training weight validation and results reporting.

Validates that all training phases produced correct weight files,
checks convergence metrics, and generates a JSON training report.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

from mousedroid.config.schema import Settings
from mousedroid.world_model.rssm import RSSM

_log = structlog.get_logger(__name__)

# Expected weight artefacts per phase
_EXPECTED_FILES: dict[str, list[str]] = {
    "rssm": ["rssm/final.pt"],
    "mcts": ["mcts/policy_init.npz"],
    "bdi": [
        "bdi/belief.npz",
        "bdi/desire.npz",
        "bdi/intention.npz",
        "bdi/affect.npz",
    ],
    "constitutional_rl": ["constitutional_rl/policy.npz", "constitutional_rl/value.npz"],
}


@dataclass
class PhaseResult:
    """Result for a single training phase."""

    phase: str
    passed: bool
    wall_time_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class TrainingReport:
    """Aggregated training report across all phases."""

    timestamp: str = ""
    device: str = "unknown"
    phases: dict[str, PhaseResult] = field(default_factory=dict)
    total_wall_time_s: float = 0.0
    all_checks_passed: bool = False


def validate_weight_files(weights_dir: Path) -> PhaseResult:
    """Check that all expected weight files exist.

    Args:
        weights_dir: Root weights directory.

    Returns:
        PhaseResult with pass/fail and list of missing files.
    """
    errors: list[str] = []
    metrics: dict[str, Any] = {"files_found": 0, "files_expected": 0}

    for _phase, files in _EXPECTED_FILES.items():
        for f in files:
            metrics["files_expected"] += 1
            full_path = weights_dir / f
            if full_path.exists():
                metrics["files_found"] += 1
            else:
                errors.append(f"Missing: {f}")

    passed = len(errors) == 0
    _log.info(
        "weight_file_check",
        passed=passed,
        found=metrics["files_found"],
        expected=metrics["files_expected"],
    )
    return PhaseResult(
        phase="weight_files",
        passed=passed,
        metrics=metrics,
        errors=errors,
    )


def validate_rssm_shapes(
    weights_dir: Path,
    cfg: Settings,
) -> PhaseResult:
    """Validate RSSM checkpoint shapes match model config.

    Args:
        weights_dir: Root weights directory.
        cfg: Settings with model config.

    Returns:
        PhaseResult with shape validation results.
    """
    errors: list[str] = []
    metrics: dict[str, Any] = {}

    ckpt_path = weights_dir / "rssm" / "final.pt"
    if not ckpt_path.exists():
        return PhaseResult(
            phase="rssm_shapes",
            passed=False,
            errors=["RSSM checkpoint not found"],
        )

    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle both raw state_dict and CheckpointState format
        if isinstance(data, dict) and "model_state_dict" in data:
            state_dict = data["model_state_dict"]
            if "best_loss" in data:
                metrics["best_loss"] = float(data["best_loss"])
        else:
            state_dict = data

        # Try loading into model
        rssm = RSSM(cfg.model)
        rssm.load_state_dict(state_dict)
        metrics["param_count"] = sum(p.numel() for p in rssm.parameters())
        _log.info(
            "rssm_shape_check",
            passed=True,
            param_count=metrics["param_count"],
        )
    except Exception as e:
        errors.append(f"RSSM load failed: {e}")

    return PhaseResult(
        phase="rssm_shapes",
        passed=len(errors) == 0,
        metrics=metrics,
        errors=errors,
    )


def validate_bdi_accuracy(
    weights_dir: Path,
    annotations_path: Path,
    holdout_fraction: float = 0.2,
) -> PhaseResult:
    """Evaluate BDI intention predictor accuracy on held-out data.

    Args:
        weights_dir: Root weights directory.
        annotations_path: Path to bdi_annotations.npz.
        holdout_fraction: Fraction of data to hold out for evaluation.

    Returns:
        PhaseResult with accuracy metrics.
    """
    from mousedroid.utils.numpy_ops import relu

    errors: list[str] = []
    metrics: dict[str, Any] = {}

    intention_path = weights_dir / "bdi" / "intention.npz"
    belief_path = weights_dir / "bdi" / "belief.npz"
    desire_path = weights_dir / "bdi" / "desire.npz"
    norm_stats_path = weights_dir / "bdi" / "belief_norm_stats.npz"

    for p in [intention_path, belief_path, desire_path, annotations_path]:
        if not p.exists():
            return PhaseResult(
                phase="bdi_accuracy",
                passed=False,
                errors=[f"Missing file: {p}"],
            )

    try:
        ann = np.load(annotations_path)
        observations = ann["observations"]
        intentions = ann["intentions"]

        # Split into holdout
        n = len(observations)
        n_holdout = max(1, int(n * holdout_fraction))
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        holdout_idx = indices[:n_holdout]

        obs_holdout = observations[holdout_idx].astype(np.float32)
        int_holdout = intentions[holdout_idx]

        # Apply z-score normalization (must match training)
        if norm_stats_path.exists():
            norm = np.load(norm_stats_path)
            obs_holdout = (obs_holdout - norm["mean"]) / (norm["std"] + 1e-8)
            _log.info("bdi_validation_normalised", norm_stats=str(norm_stats_path))
        else:
            _log.warning("bdi_norm_stats_missing", path=str(norm_stats_path))

        # Load weights and run forward pass
        belief_w = np.load(belief_path)
        desire_w = np.load(desire_path)
        intent_w = np.load(intention_path)

        # Belief: two-layer encoder (must match train_belief_encoder)
        h1 = relu(obs_holdout @ belief_w["w1"] + belief_w["b1"])
        belief = relu(h1 @ belief_w["w2"] + belief_w["b2"])
        # Desire: belief → relu(belief @ w1 + b1)
        desire = relu(belief @ desire_w["w1"] + desire_w["b1"])
        # Intention: desire → logits
        logits = desire @ intent_w["w1"] + intent_w["b1"]
        predictions = logits.argmax(axis=-1)

        accuracy = float(np.mean(predictions == int_holdout))
        metrics["accuracy"] = round(accuracy, 4)
        metrics["n_holdout"] = n_holdout
        metrics["n_classes"] = len(np.unique(intentions))

        passed = accuracy > 0.60
        if not passed:
            errors.append(f"Intention accuracy {accuracy:.2%} below 60% threshold")

        _log.info(
            "bdi_accuracy_check",
            accuracy=round(accuracy, 4),
            n_holdout=n_holdout,
            passed=passed,
        )
    except Exception as e:
        errors.append(f"BDI evaluation failed: {e}")
        passed = False

    return PhaseResult(
        phase="bdi_accuracy",
        passed=passed,
        metrics=metrics,
        errors=errors,
    )


def validate_constitutional_rl(
    weights_dir: Path,
) -> PhaseResult:
    """Check Constitutional RL training results.

    Args:
        weights_dir: Root weights directory.

    Returns:
        PhaseResult with violation rate and reward metrics.
    """
    errors: list[str] = []
    metrics: dict[str, Any] = {}

    policy_path = weights_dir / "constitutional_rl" / "policy.npz"
    value_path = weights_dir / "constitutional_rl" / "value.npz"

    for p in [policy_path, value_path]:
        if not p.exists():
            return PhaseResult(
                phase="constitutional_rl",
                passed=False,
                errors=[f"Missing file: {p}"],
            )

    try:
        # Load and verify shapes
        policy_data = np.load(policy_path)
        value_data = np.load(value_path)
        metrics["policy_keys"] = list(policy_data.files)
        metrics["value_keys"] = list(value_data.files)

        # Check training results if available
        results_path = weights_dir / "constitutional_rl_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            metrics["violation_rate"] = results.get("violation_rate", -1)
            metrics["mean_reward"] = results.get("mean_reward", 0)

            if metrics["violation_rate"] > 0.05:
                errors.append(f"Violation rate {metrics['violation_rate']:.2%} exceeds 5%")
            if metrics["mean_reward"] <= 0:
                errors.append(f"Mean reward {metrics['mean_reward']:.4f} is not positive")

        _log.info("constitutional_rl_check", metrics=metrics)
    except Exception as e:
        errors.append(f"Constitutional RL validation failed: {e}")

    return PhaseResult(
        phase="constitutional_rl",
        passed=len(errors) == 0,
        metrics=metrics,
        errors=errors,
    )


def validate_mcts_latency(
    weights_dir: Path,
    target_p50_ms: float = 50.0,
) -> PhaseResult:
    """Check MCTS tuning results for latency regression.

    Reads ``mcts/tuned_config.json`` and flags if the best UCB candidate's
    p50 search latency exceeds the ``target_p50_ms`` threshold.

    Args:
        weights_dir: Root weights directory.
        target_p50_ms: Acceptable p50 search latency in milliseconds.

    Returns:
        PhaseResult with latency metrics.
    """
    errors: list[str] = []
    metrics: dict[str, Any] = {}

    config_path = weights_dir / "mcts" / "tuned_config.json"
    if not config_path.exists():
        return PhaseResult(
            phase="mcts_latency",
            passed=True,  # Not blocking if tuning hasn't run yet
            metrics={"skipped": True, "reason": "tuned_config.json not found"},
        )

    try:
        with open(config_path) as f:
            data = json.load(f)

        best_ucb = data.get("best_ucb_c")
        if best_ucb is not None:
            key = f"ucb_{best_ucb}"
            if key in data:
                p50 = data[key]["p50_ms"]
                p95 = data[key]["p95_ms"]
                mean_reward = data[key]["mean_reward"]
                metrics["best_ucb_c"] = best_ucb
                metrics["p50_ms"] = p50
                metrics["p95_ms"] = p95
                metrics["mean_reward"] = mean_reward
                metrics["target_p50_ms"] = target_p50_ms

                if p50 > target_p50_ms:
                    errors.append(
                        f"MCTS p50 latency {p50:.1f}ms exceeds target {target_p50_ms:.0f}ms "
                        f"({p50/target_p50_ms:.1f}x over) — consider reducing n_simulations or rollout_depth"
                    )

        _log.info("mcts_latency_check", metrics=metrics, passed=len(errors) == 0)
    except Exception as e:
        errors.append(f"MCTS latency check failed: {e}")

    return PhaseResult(
        phase="mcts_latency",
        passed=len(errors) == 0,
        metrics=metrics,
        errors=errors,
    )

def generate_training_report(
    weights_dir: Path,
    cfg: Settings,
    annotations_path: Path | None = None,
    phase_timings: dict[str, float] | None = None,
    output_path: Path | None = None,
) -> TrainingReport:
    """Run all validations and generate a training report.

    Args:
        weights_dir: Root weights directory.
        cfg: Settings with model config.
        annotations_path: Path to BDI annotations for accuracy check.
        phase_timings: Optional per-phase wall-clock timings.
        output_path: Path to write JSON report. Defaults to
            ``training/results/training_report.json``.

    Returns:
        TrainingReport with aggregated results.
    """
    report = TrainingReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        device=str(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")),
    )

    # 1. Weight file existence
    report.phases["weight_files"] = validate_weight_files(weights_dir)

    # 2. RSSM shapes
    report.phases["rssm_shapes"] = validate_rssm_shapes(weights_dir, cfg)

    # 3. BDI accuracy (if annotations available)
    if annotations_path and annotations_path.exists():
        report.phases["bdi_accuracy"] = validate_bdi_accuracy(weights_dir, annotations_path)
    else:
        report.phases["bdi_accuracy"] = PhaseResult(
            phase="bdi_accuracy",
            passed=True,
            metrics={"skipped": True, "reason": "no annotations file"},
        )

    # 4. Constitutional RL
    report.phases["constitutional_rl"] = validate_constitutional_rl(weights_dir)

    # 5. MCTS latency
    report.phases["mcts_latency"] = validate_mcts_latency(weights_dir)

    # Apply timings
    if phase_timings:
        for phase_name, timing in phase_timings.items():
            if phase_name in report.phases:
                report.phases[phase_name].wall_time_s = timing
        report.total_wall_time_s = sum(phase_timings.values())

    # Overall pass
    report.all_checks_passed = all(r.passed for r in report.phases.values())

    _log.info(
        "training_report",
        all_passed=report.all_checks_passed,
        phases={k: v.passed for k, v in report.phases.items()},
    )

    # Write JSON
    output_path = output_path or Path("training/results/training_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts for JSON
    report_dict: dict[str, Any] = {
        "timestamp": report.timestamp,
        "device": report.device,
        "phases": {k: asdict(v) for k, v in report.phases.items()},
        "total_wall_time_s": round(report.total_wall_time_s, 1),
        "all_checks_passed": report.all_checks_passed,
    }
    output_path.write_text(
        json.dumps(report_dict, indent=2, default=str),
        encoding="utf-8",
    )
    _log.info("report_saved", path=str(output_path))

    return report
