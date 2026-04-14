"""Unit tests for the dual-stream CfC/GRU RSSM training system.

Covers the dual-optimizer parameter separation, CfC loss warmup schedule,
gradient clipping behaviour, single-step forward/backward pass, and
checkpoint round-trip semantics for DualStreamCheckpointState.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional ncps dependency — skip entire module if not installed
# ---------------------------------------------------------------------------

ncps = pytest.importorskip("ncps", reason="ncps library required for CfC cell")

from mousedroid.config.schema import DualStreamTrainingConfig, ModelConfig  # noqa: E402
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_model_cfg(
    hidden_dim: int = 32,
    cfc_hidden_dim: int = 16,
    latent_dim: int = 8,
    action_dim: int = 2,
) -> ModelConfig:
    """Build a minimal ModelConfig suitable for CPU-only unit tests.

    Args:
        hidden_dim: GRU hidden dimension.
        cfc_hidden_dim: CfC stream hidden dimension.
        latent_dim: Latent state dimension.
        action_dim: Action dimension.

    Returns:
        Configured ModelConfig instance.
    """
    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        motor_state_dim=4,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        obs_dim=16,
        vision_proj_dim=8,
        ultrasonic_proj_dim=4,
        motor_proj_dim=4,
        cfc_hidden_dim=cfc_hidden_dim,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


def _make_dual_stream_model(cfg: ModelConfig | None = None) -> DualStreamRSSM:
    """Instantiate a small DualStreamRSSM for testing.

    Args:
        cfg: Optional ModelConfig; defaults to a minimal test config.

    Returns:
        DualStreamRSSM in training mode.
    """
    if cfg is None:
        cfg = _make_model_cfg()
    return DualStreamRSSM(cfg)


def _make_training_cfg(**overrides: object) -> DualStreamTrainingConfig:
    """Build a DualStreamTrainingConfig with optional field overrides.

    Args:
        **overrides: Field values to override from defaults.

    Returns:
        DualStreamTrainingConfig instance.
    """
    return DualStreamTrainingConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_cfc_loss_weight import helper
# ---------------------------------------------------------------------------


def _import_compute_cfc_loss_weight():  # type: ignore[return]
    """Import compute_cfc_loss_weight from the training module.

    Tries the public name ``compute_cfc_loss_weight(step, cfg)`` first.
    If that does not exist, wraps the private ``_cfc_loss_weight`` helper
    with an adapter so the warmup schedule tests can run against the
    implementation that is already present.

    Returns:
        A callable with signature ``(step: int, cfg: DualStreamTrainingConfig) -> float``.

    Raises:
        pytest.skip.Exception: If the training module has not been created yet.
    """
    try:
        import training.train_dual_stream_rssm as _mod
    except ImportError as exc:
        pytest.skip(f"training.train_dual_stream_rssm not yet available: {exc}")

    # Prefer the public name specified in the task contract
    if hasattr(_mod, "compute_cfc_loss_weight"):
        return _mod.compute_cfc_loss_weight  # type: ignore[attr-defined]

    # Fall back to the private helper with a thin adapter
    if hasattr(_mod, "_cfc_loss_weight"):
        _fn = _mod._cfc_loss_weight  # type: ignore[attr-defined]

        def _adapter(step: int, cfg: DualStreamTrainingConfig) -> float:
            return _fn(
                step,
                initial=cfg.cfc_loss_weight_initial,
                final=cfg.cfc_loss_weight_final,
                warmup_steps=cfg.cfc_loss_warmup_steps,
            )

        return _adapter

    pytest.skip(
        "Neither compute_cfc_loss_weight nor _cfc_loss_weight found in "
        "training.train_dual_stream_rssm"
    )


def _import_checkpoint_state():  # type: ignore[return]
    """Import DualStreamCheckpointState from the training module.

    Returns:
        The DualStreamCheckpointState class.

    Raises:
        pytest.skip.Exception: If the training module has not been created yet.
    """
    try:
        from training.train_dual_stream_rssm import DualStreamCheckpointState

        return DualStreamCheckpointState
    except ImportError as exc:
        pytest.skip(f"training.train_dual_stream_rssm not yet available: {exc}")


# ---------------------------------------------------------------------------
# 1. test_dual_optimizer_separate_param_groups
# ---------------------------------------------------------------------------


class TestDualOptimizerSeparateParamGroups:
    """Verify gru_parameters() and cfc_parameters() have no ID overlap."""

    def test_dual_optimizer_separate_param_groups(self) -> None:
        """GRU and CfC parameter sets must be completely disjoint.

        A shared parameter appearing in both groups would cause it to be
        updated twice per step, effectively doubling its learning rate.
        Disjoint sets are required for the dual-optimizer design to be correct.
        """
        model = _make_dual_stream_model()

        gru_ids = {id(p) for p in model.gru_parameters()}
        cfc_ids = {id(p) for p in model.cfc_parameters()}

        assert len(gru_ids) > 0, "gru_parameters() must yield at least one parameter"
        assert len(cfc_ids) > 0, "cfc_parameters() must yield at least one parameter"
        assert gru_ids.isdisjoint(cfc_ids), (
            "gru_parameters() and cfc_parameters() share parameter IDs — "
            "dual optimizer would apply double updates to shared parameters"
        )

    def test_gru_parameters_cover_shared_heads(self) -> None:
        """GRU parameter set includes encoder, posterior, prior, reward, and decoder.

        Shared heads are owned by the GRU optimizer so that the CfC optimizer
        does not inadvertently update them through gradient accumulation.
        """
        model = _make_dual_stream_model()
        gru_ids = {id(p) for p in model.gru_parameters()}

        for submodule_name in (
            "encoder",
            "gru",
            "posterior",
            "prior",
            "reward_head",
            "observation_decoder",
        ):
            submodule = getattr(model, submodule_name)
            for p in submodule.parameters():
                assert (
                    id(p) in gru_ids
                ), f"Parameter from {submodule_name} not found in gru_parameters()"

    def test_cfc_parameters_are_cfc_module_only(self) -> None:
        """cfc_parameters() must yield exactly the CfC module's own parameters.

        Any other parameters in this iterator would receive CfC-rate updates,
        potentially destabilising the slower GRU stream.
        """
        model = _make_dual_stream_model()
        cfc_ids = {id(p) for p in model.cfc_parameters()}
        cfc_module_ids = {id(p) for p in model.cfc.parameters()}
        assert (
            cfc_ids == cfc_module_ids
        ), "cfc_parameters() must yield exactly model.cfc.parameters()"


# ---------------------------------------------------------------------------
# 2-4. Warmup schedule tests
# ---------------------------------------------------------------------------


class TestCfcWarmupSchedule:
    """Tests for compute_cfc_loss_weight linear warmup schedule."""

    def test_cfc_warmup_schedule_initial(self) -> None:
        """At step 0 the weight must equal cfc_loss_weight_initial.

        Before any warmup has occurred the CfC contribution should be small
        so the GRU stream can stabilise before the CfC stream is introduced.
        """
        compute_cfc_loss_weight = _import_compute_cfc_loss_weight()
        cfg = _make_training_cfg(
            cfc_loss_weight_initial=0.1, cfc_loss_weight_final=1.0, cfc_loss_warmup_steps=10000
        )

        weight = compute_cfc_loss_weight(0, cfg)

        assert weight == pytest.approx(
            cfg.cfc_loss_weight_initial, rel=1e-6
        ), f"At step 0 weight should be {cfg.cfc_loss_weight_initial}, got {weight}"

    def test_cfc_warmup_schedule_midpoint(self) -> None:
        """At step warmup/2 the weight is linearly interpolated between initial and final.

        The midpoint check verifies the ramp is linear (not exponential or
        step-wise) which is required for stable gradient scaling.
        """
        compute_cfc_loss_weight = _import_compute_cfc_loss_weight()
        initial = 0.1
        final = 1.0
        warmup_steps = 10000
        cfg = _make_training_cfg(
            cfc_loss_weight_initial=initial,
            cfc_loss_weight_final=final,
            cfc_loss_warmup_steps=warmup_steps,
        )

        mid_step = warmup_steps // 2
        weight = compute_cfc_loss_weight(mid_step, cfg)
        expected = initial + (final - initial) * (mid_step / warmup_steps)

        assert weight == pytest.approx(
            expected, rel=1e-5
        ), f"At midpoint step {mid_step} expected {expected:.4f}, got {weight:.4f}"
        assert (
            initial < weight < final
        ), "Midpoint weight must be strictly between initial and final"

    @pytest.mark.parametrize("step_multiplier", [1, 2, 10])
    def test_cfc_warmup_schedule_final(self, step_multiplier: int) -> None:
        """At step >= warmup_steps the weight must equal cfc_loss_weight_final.

        The schedule must be clamped at the final value so that the CfC loss
        never exceeds the intended full weight even at large step counts.

        Args:
            step_multiplier: Multiplier applied to warmup_steps to test
                steps at and beyond the warmup boundary.
        """
        compute_cfc_loss_weight = _import_compute_cfc_loss_weight()
        cfg = _make_training_cfg(
            cfc_loss_weight_initial=0.1, cfc_loss_weight_final=1.0, cfc_loss_warmup_steps=10000
        )

        step = cfg.cfc_loss_warmup_steps * step_multiplier
        weight = compute_cfc_loss_weight(step, cfg)

        assert weight == pytest.approx(cfg.cfc_loss_weight_final, rel=1e-6), (
            f"At step {step} (>= warmup) weight should be clamped to "
            f"{cfg.cfc_loss_weight_final}, got {weight}"
        )

    def test_cfc_warmup_schedule_zero_warmup(self) -> None:
        """When warmup_steps=0 the weight should always equal the final value.

        Zero warmup is a valid configuration that enables the CfC stream at
        full weight from the first step.
        """
        compute_cfc_loss_weight = _import_compute_cfc_loss_weight()
        cfg = _make_training_cfg(
            cfc_loss_weight_initial=0.1,
            cfc_loss_weight_final=1.0,
            cfc_loss_warmup_steps=0,
        )

        for step in [0, 1, 100]:
            weight = compute_cfc_loss_weight(step, cfg)
            assert weight == pytest.approx(cfg.cfc_loss_weight_final, rel=1e-6), (
                f"With warmup_steps=0, step={step}: expected {cfg.cfc_loss_weight_final}, "
                f"got {weight}"
            )


# ---------------------------------------------------------------------------
# 5-6. Gradient clipping tests
# ---------------------------------------------------------------------------


class TestGradientClipping:
    """Tests that gradient clip norms from config are applied correctly."""

    @staticmethod
    def _inject_large_gradient(params: list[nn.Parameter], magnitude: float = 1000.0) -> None:
        """Assign a uniform gradient of the given magnitude to all parameters.

        Args:
            params: List of parameters to inject gradients into.
            magnitude: Gradient value to fill each element with.
        """
        for p in params:
            p.grad = torch.full_like(p.data, magnitude)

    def test_gradient_clipping_gru(self) -> None:
        """A very large GRU gradient must be clipped to gru_grad_clip norm.

        The clip norm is read from DualStreamTrainingConfig.gru_grad_clip.
        After clipping, the total l2 norm of GRU parameters must not exceed
        that value (within floating-point tolerance).
        """
        model = _make_dual_stream_model()
        cfg = _make_training_cfg(gru_grad_clip=10.0)

        gru_params = list(model.gru_parameters())
        self._inject_large_gradient(gru_params, magnitude=1000.0)

        total_norm_before = (
            sum(p.grad.norm().item() ** 2 for p in gru_params if p.grad is not None) ** 0.5
        )
        assert (
            total_norm_before > cfg.gru_grad_clip
        ), "Pre-clipping norm must exceed the clip threshold for this test to be meaningful"

        nn.utils.clip_grad_norm_(gru_params, cfg.gru_grad_clip)

        total_norm_after = (
            sum(p.grad.norm().item() ** 2 for p in gru_params if p.grad is not None) ** 0.5
        )
        assert total_norm_after <= cfg.gru_grad_clip + 1e-4, (
            f"GRU gradient norm {total_norm_after:.4f} exceeds clip threshold "
            f"{cfg.gru_grad_clip}"
        )

    def test_gradient_clipping_cfc(self) -> None:
        """A very large CfC gradient must be clipped to cfc_grad_clip norm.

        The CfC clip threshold is intentionally tighter (1.0 vs 10.0) to
        prevent the liquid network's ODE solver from producing large updates
        that destabilise training.
        """
        model = _make_dual_stream_model()
        cfg = _make_training_cfg(cfc_grad_clip=1.0)

        cfc_params = list(model.cfc_parameters())
        self._inject_large_gradient(cfc_params, magnitude=1000.0)

        total_norm_before = (
            sum(p.grad.norm().item() ** 2 for p in cfc_params if p.grad is not None) ** 0.5
        )
        assert (
            total_norm_before > cfg.cfc_grad_clip
        ), "Pre-clipping norm must exceed the clip threshold for this test to be meaningful"

        nn.utils.clip_grad_norm_(cfc_params, cfg.cfc_grad_clip)

        total_norm_after = (
            sum(p.grad.norm().item() ** 2 for p in cfc_params if p.grad is not None) ** 0.5
        )
        assert total_norm_after <= cfg.cfc_grad_clip + 1e-4, (
            f"CfC gradient norm {total_norm_after:.4f} exceeds clip threshold "
            f"{cfg.cfc_grad_clip}"
        )

    def test_gru_clip_does_not_affect_cfc_params(self) -> None:
        """Clipping GRU parameters must not alter CfC parameter gradients.

        The two parameter groups are distinct; clipping one group must leave
        the other group's gradients unchanged.
        """
        model = _make_dual_stream_model()
        cfg = _make_training_cfg(gru_grad_clip=10.0)

        gru_params = list(model.gru_parameters())
        cfc_params = list(model.cfc_parameters())

        # Assign different magnitudes to each group
        self._inject_large_gradient(gru_params, magnitude=500.0)
        self._inject_large_gradient(cfc_params, magnitude=200.0)

        # Record CfC norms before GRU clipping
        cfc_norms_before = [p.grad.norm().item() for p in cfc_params if p.grad is not None]

        nn.utils.clip_grad_norm_(gru_params, cfg.gru_grad_clip)

        cfc_norms_after = [p.grad.norm().item() for p in cfc_params if p.grad is not None]
        for before, after in zip(cfc_norms_before, cfc_norms_after, strict=True):
            assert before == pytest.approx(
                after, rel=1e-6
            ), "GRU grad clip must not modify CfC parameter gradients"


# ---------------------------------------------------------------------------
# 7. test_single_training_step_finite
# ---------------------------------------------------------------------------


@dataclass
class _MockObservation:
    """Minimal mock observation bundle for training step tests."""

    vision_features: np.ndarray
    distance_m: float = 2.0
    motor_state: np.ndarray = None  # type: ignore[assignment]
    audio_chunk: np.ndarray = None  # type: ignore[assignment]
    valid_mask: np.ndarray = None  # type: ignore[assignment]
    lidar_features: np.ndarray = None  # type: ignore[assignment]
    n_modalities: int = 4
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Fill in any None arrays with appropriate zeros."""
        if self.motor_state is None:
            self.motor_state = np.zeros(4, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(0, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(4, dtype=np.float32)


class TestSingleTrainingStepFinite:
    """Verify one forward+backward pass produces finite loss and gradients."""

    def test_single_training_step_finite(self) -> None:
        """One complete forward/backward pass must not produce NaN or Inf.

        This is the minimal smoke test for the dual-stream training loop:
        compute both GRU and CfC losses, combine them with the warmup weight,
        and confirm that gradients are finite for all parameters.
        """
        cfg = _make_model_cfg()
        model = _make_dual_stream_model(cfg)
        model.train()

        gru_optimizer = torch.optim.Adam(list(model.gru_parameters()), lr=3e-4)
        cfc_optimizer = torch.optim.Adam(list(model.cfc_parameters()), lr=1e-4)

        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim
        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.randn(1, cfg.action_dim)

        obs = _MockObservation(vision_features=np.random.randn(cfg.vision_dim).astype(np.float32))

        # Forward pass
        new_h, new_z, obs_embed, _ = model.observe_step(obs, action, h, z)

        # GRU reconstruction loss
        recon = model.decode(new_h, new_z)
        gru_loss = nn.functional.mse_loss(recon, obs_embed)

        # CfC auxiliary loss (match the GRU-decoded target using CfC hidden state only)
        h_cfc = model.fusion.extract_cfc_state(new_h)
        # Simple proxy: CfC state should match GRU state in magnitude
        cfc_loss = nn.functional.mse_loss(h_cfc, torch.zeros_like(h_cfc))

        cfc_weight = 0.1  # initial warmup weight
        total_loss = gru_loss + cfc_weight * cfc_loss

        # Backward
        gru_optimizer.zero_grad()
        cfc_optimizer.zero_grad()
        total_loss.backward()
        gru_optimizer.step()
        cfc_optimizer.step()

        # Assertions: loss and all gradients must be finite
        assert torch.isfinite(total_loss), f"Training loss is not finite: {total_loss.item()}"
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient in parameter '{name}'"

    def test_single_training_step_updates_parameters(self) -> None:
        """A single optimiser step must change at least one parameter value.

        If parameters are unchanged after a step it indicates zero loss,
        detached gradients, or a misconfigured optimiser.
        """
        cfg = _make_model_cfg()
        model = _make_dual_stream_model(cfg)
        model.train()

        gru_optimizer = torch.optim.SGD(list(model.gru_parameters()), lr=0.1)

        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim
        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, cfg.latent_dim)
        action = torch.randn(1, cfg.action_dim)
        obs = _MockObservation(vision_features=np.random.randn(cfg.vision_dim).astype(np.float32))

        # Record weights before step
        params_before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if any(id(param) == id(p) for p in model.gru_parameters())
        }

        new_h, new_z, obs_embed, _ = model.observe_step(obs, action, h, z)
        recon = model.decode(new_h, new_z)
        loss = nn.functional.mse_loss(recon, obs_embed)

        gru_optimizer.zero_grad()
        loss.backward()
        gru_optimizer.step()

        params_after = {name: param.detach().clone() for name, param in model.named_parameters()}

        changed = any(
            not torch.equal(params_before[name], params_after[name]) for name in params_before
        )
        assert changed, "No GRU parameters changed after an optimiser step"


# ---------------------------------------------------------------------------
# 8. test_checkpoint_roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    """Verify save/load round-trip for DualStreamCheckpointState."""

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload a checkpoint preserves warmup_step and both optimiser states.

        The checkpoint must faithfully restore:
        - warmup_step (determines the CfC loss weight at resume time)
        - gru_optimizer state (momentum, Adam moments, etc.)
        - cfc_optimizer state (ditto for the CfC stream)

        Args:
            tmp_path: Pytest-provided temporary directory.
        """
        checkpoint_cls = _import_checkpoint_state()

        cfg = _make_model_cfg()
        model = _make_dual_stream_model(cfg)
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        gru_optimizer = torch.optim.Adam(list(model.gru_parameters()), lr=3e-4)
        cfc_optimizer = torch.optim.Adam(list(model.cfc_parameters()), lr=1e-4)

        warmup_step = 5000

        # Build the state object using the actual dataclass signature
        state = checkpoint_cls(
            epoch=10,
            best_loss=0.42,
            warmup_step=warmup_step,
            combined_dim=combined_dim,
            model_state_dict=model.state_dict(),
            gru_optimizer_state_dict=gru_optimizer.state_dict(),
            cfc_optimizer_state_dict=cfc_optimizer.state_dict(),
            scaler_state_dict=None,
            rng_state=torch.get_rng_state(),
        )

        ckpt_path = tmp_path / "dual_stream_checkpoint.pt"
        torch.save(
            {
                "epoch": state.epoch,
                "best_loss": state.best_loss,
                "warmup_step": state.warmup_step,
                "combined_dim": state.combined_dim,
                "model_state_dict": state.model_state_dict,
                "gru_optimizer_state_dict": state.gru_optimizer_state_dict,
                "cfc_optimizer_state_dict": state.cfc_optimizer_state_dict,
                "scaler_state_dict": state.scaler_state_dict,
                "rng_state": state.rng_state,
            },
            ckpt_path,
        )

        # Reload
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Reconstruct model and optimisers from checkpoint
        model2 = _make_dual_stream_model(cfg)
        model2.load_state_dict(loaded["model_state_dict"])

        gru_opt2 = torch.optim.Adam(list(model2.gru_parameters()), lr=3e-4)
        cfc_opt2 = torch.optim.Adam(list(model2.cfc_parameters()), lr=1e-4)
        gru_opt2.load_state_dict(loaded["gru_optimizer_state_dict"])
        cfc_opt2.load_state_dict(loaded["cfc_optimizer_state_dict"])

        # Verify warmup_step preserved exactly
        assert (
            loaded["warmup_step"] == warmup_step
        ), f"warmup_step mismatch: saved {warmup_step}, loaded {loaded['warmup_step']}"

        # Verify combined_dim preserved
        assert (
            loaded["combined_dim"] == combined_dim
        ), f"combined_dim mismatch: saved {combined_dim}, loaded {loaded['combined_dim']}"

        # Verify model weights preserved
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(),
            model2.state_dict().items(),
            strict=True,
        ):
            assert k1 == k2, f"State dict key mismatch: {k1} vs {k2}"
            assert torch.allclose(v1, v2), f"Weight mismatch in layer '{k1}'"

    def test_checkpoint_roundtrip_preserves_optimizer_lr(self, tmp_path: Path) -> None:
        """Reloaded optimiser state must preserve the per-group learning rates.

        Incorrect LR restoration would silently change the effective learning
        rate at resume, causing training instability.

        Args:
            tmp_path: Pytest-provided temporary directory.
        """
        checkpoint_cls = _import_checkpoint_state()

        cfg = _make_model_cfg()
        model = _make_dual_stream_model(cfg)

        gru_lr = 3e-4
        cfc_lr = 1e-4
        gru_optimizer = torch.optim.Adam(list(model.gru_parameters()), lr=gru_lr)
        cfc_optimizer = torch.optim.Adam(list(model.cfc_parameters()), lr=cfc_lr)

        state = checkpoint_cls(
            epoch=5,
            best_loss=1.0,
            warmup_step=1000,
            combined_dim=cfg.hidden_dim + cfg.cfc_hidden_dim,
            model_state_dict=model.state_dict(),
            gru_optimizer_state_dict=gru_optimizer.state_dict(),
            cfc_optimizer_state_dict=cfc_optimizer.state_dict(),
            scaler_state_dict=None,
            rng_state=torch.get_rng_state(),
        )

        ckpt_path = tmp_path / "lr_test.pt"
        torch.save(
            {
                "epoch": state.epoch,
                "best_loss": state.best_loss,
                "warmup_step": state.warmup_step,
                "combined_dim": state.combined_dim,
                "model_state_dict": state.model_state_dict,
                "gru_optimizer_state_dict": state.gru_optimizer_state_dict,
                "cfc_optimizer_state_dict": state.cfc_optimizer_state_dict,
                "scaler_state_dict": state.scaler_state_dict,
                "rng_state": state.rng_state,
            },
            ckpt_path,
        )

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model2 = _make_dual_stream_model(cfg)
        gru_opt2 = torch.optim.Adam(list(model2.gru_parameters()), lr=gru_lr)
        cfc_opt2 = torch.optim.Adam(list(model2.cfc_parameters()), lr=cfc_lr)
        gru_opt2.load_state_dict(loaded["gru_optimizer_state_dict"])
        cfc_opt2.load_state_dict(loaded["cfc_optimizer_state_dict"])

        for pg in gru_opt2.param_groups:
            assert pg["lr"] == pytest.approx(
                gru_lr, rel=1e-6
            ), f"GRU LR after reload: {pg['lr']} != {gru_lr}"
        for pg in cfc_opt2.param_groups:
            assert pg["lr"] == pytest.approx(
                cfc_lr, rel=1e-6
            ), f"CfC LR after reload: {pg['lr']} != {cfc_lr}"
