# Phase 5 — MuJoCo Skid-Steer Rover → RSSM Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the NumPy kinematic rover sim with a MuJoCo skid-steer physics simulator and pretrain the RSSM world-model dynamics core on its episodes, end-to-end through the training pipeline orchestrator — all backwards-compatible and opt-in.

**Architecture:** Three layers, strict build order. **Layer 1 (pure ML, no MuJoCo dep):** make the multimodal encoder's vision branch optional (byte-identical default) and add a gradient-enabled `RSSM.train_sequence` that reconstructs *raw* modalities (not the encoder's own embedding) with free-bits/balanced KL. **Layer 2 (MuJoCo env, independent of L1):** a `RoverMuJoCoEnv` behind the existing `RoverEnvProtocol`, a hand-authored MJCF, the reserved factory branch, and domain-randomization consumption. **Layer 3 (glue):** observation adapter, episode generator, pretrainer, and orchestrator wiring (`asyncio.to_thread`).

**Tech Stack:** Python 3.10+, PyTorch (RSSM + AMP), MuJoCo `>=3.0` (classic C engine, already in the `[arm]` extra), Pydantic v2, structlog, NumPy, pytest + pytest-asyncio + hypothesis, ruff 0.8.0, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-06-07-phase5-mujoco-rssm-pretraining-design.md` (read it first — it carries the 3-agent peer-review rationale behind every non-obvious choice).

---

## Conventions for every task

- Work in the worktree branch `claude/phase5-mujoco-rssm-pretraining`.
- Pytest is always invoked with `--import-mode=importlib` (matches `scripts/ci.sh`).
- After each code change, the per-file gate is:
  `python -m ruff check <files> && python -m ruff format --check <files> && python -m mypy --strict <module>`.
- `from __future__ import annotations` at the top of every new module.
- Google-style docstrings on every public function/class.
- structlog via `from mousedroid.logging.setup import get_logger` — never `print`.
- Commit after each task (frequent commits). Commit trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File map (locked)

| File | Status | Layer | Responsibility |
|---|---|---|---|
| `src/mousedroid/world_model/encoder.py` | MODIFY | 1 | Make vision branch optional (`_vision_enabled`) |
| `src/mousedroid/world_model/latent_utils.py` | MODIFY | 1 | Add `balanced_free_bits_kl` helper |
| `src/mousedroid/world_model/rssm.py` | MODIFY | 1 | Raw-modality decoder heads + `train_sequence` |
| `src/mousedroid/factory.py` | MODIFY | 1,2,3 | `build_rssm_trainable`; `build_rover_env` mujoco branch |
| `src/mousedroid/config/schema.py` | MODIFY | 1,2,3 | `MujocoSimConfig`; pretrain knobs on `TrainingConfig` |
| `assets/rover/mse6_4wd.xml` | CREATE | 2 | Skid-steer MJCF (chassis+4 wheels+walls+sensors) |
| `src/mousedroid/sim/mujoco_rover_env.py` | CREATE | 2 | `RoverMuJoCoEnv` (RoverEnvProtocol) |
| `src/mousedroid/training/rover_obs_adapter.py` | CREATE | 3 | Rover obs dict → RSSM encoder tensors |
| `src/mousedroid/training/sim_episode_generator.py` | CREATE | 3 | `EpisodeBatch` + `SimEpisodeGenerator` |
| `src/mousedroid/training/rssm_pretrainer.py` | CREATE | 3 | Adam loop, AMP, checkpoint |
| `src/mousedroid/training/pipeline_orchestrator.py` | MODIFY | 3 | Wire `_train_rssm` |
| `tests/unit/world_model/...` | CREATE | 1 | encoder/kl/train_sequence units |
| `tests/unit/sim/test_mujoco_rover_env.py` | CREATE | 2 | env units (importorskip mujoco) |
| `tests/unit/training/...` | CREATE | 3 | adapter/generator/pretrainer units |
| `tests/integration/test_phase5_rssm_pretrain.py` | CREATE | 3 | end-to-end round-trip |
| `tests/regression/test_phase5_backwards_compat.py` | CREATE | 1 | byte-identical defaults |
| `tests/regression/test_phase5_rssm_golden.py` | CREATE | 3 | golden loss (non-gating) |
| `tests/smoke/test_phase5_sanity.py` | CREATE | 2 | sub-second import + factory type |

---

# LAYER 1 — Encoder vision-optional + trainable RSSM (pure ML)

### Task 1: Make the encoder's vision branch optional

**Files:**
- Modify: `src/mousedroid/world_model/encoder.py`
- Test: `tests/unit/world_model/test_encoder_vision_optional.py`

- [ ] **Step 1: Write the failing test**

```python
"""Vision branch is optional and gated on vision_dim, mirroring audio/lidar."""
from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.encoder import MultimodalEncoder


def _cfg(**over: object) -> ModelConfig:
    return ModelConfig(**over)  # type: ignore[arg-type]


def test_vision_enabled_by_default() -> None:
    enc = MultimodalEncoder(_cfg())
    assert enc.vision_enabled is True
    assert hasattr(enc, "vision_proj")


def test_vision_disabled_when_vision_dim_zero() -> None:
    enc = MultimodalEncoder(_cfg(vision_dim=0))
    assert enc.vision_enabled is False
    assert not hasattr(enc, "vision_proj")


def test_forward_without_vision_runs() -> None:
    enc = _cfg(vision_dim=0)
    model = MultimodalEncoder(enc)
    motor = torch.zeros(2, enc.motor_state_dim)
    mask = torch.ones(2, 5)
    out = model(None, None, motor, mask)
    assert out.shape == (2, enc.obs_dim)


def test_default_forward_byte_identical_path() -> None:
    """vision_dim=256 default keeps the original 3-modality fused width."""
    cfg = _cfg()
    enc = MultimodalEncoder(cfg)
    vision = torch.zeros(1, cfg.vision_dim)
    motor = torch.zeros(1, cfg.motor_state_dim)
    mask = torch.ones(1, 5)
    out = enc(vision, None, motor, mask)
    assert out.shape == (1, cfg.obs_dim)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/world_model/test_encoder_vision_optional.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: FAIL — `MultimodalEncoder` has no `vision_enabled`; `vision_dim=0` still builds `vision_proj`; `forward(None, ...)` raises on `self.vision_proj(None)`.

- [ ] **Step 3: Implement the optional vision branch**

In `src/mousedroid/world_model/encoder.py` `__init__`, replace the unconditional vision build (currently `self.vision_proj = nn.Linear(cfg.vision_dim, cfg.vision_proj_dim)` and `fused_dim = cfg.vision_proj_dim + cfg.motor_proj_dim`) with the gated pattern mirroring audio/lidar:

```python
        self._vision_enabled = cfg.vision_dim > 0 and cfg.vision_proj_dim > 0
        self.motor_proj = nn.Linear(cfg.motor_state_dim, cfg.motor_proj_dim)

        self._ultrasonic_enabled = cfg.ultrasonic_dim > 0 and cfg.ultrasonic_proj_dim > 0
        self._audio_enabled = cfg.audio_dim > 0 and cfg.audio_proj_dim > 0
        self._lidar_enabled = cfg.lidar_dim > 0 and cfg.lidar_proj_dim > 0

        fused_dim = cfg.motor_proj_dim
        if self._vision_enabled:
            self.vision_proj = nn.Linear(cfg.vision_dim, cfg.vision_proj_dim)
            fused_dim += cfg.vision_proj_dim
        if self._ultrasonic_enabled:
            self.ultrasonic_proj = nn.Linear(cfg.ultrasonic_dim, cfg.ultrasonic_proj_dim)
            fused_dim += cfg.ultrasonic_proj_dim
        # ... (audio, lidar blocks unchanged) ...
```

Add the property near the other `*_enabled` properties:

```python
    @property
    def vision_enabled(self) -> bool:
        """Whether the vision modality is active (vision_dim > 0)."""
        return self._vision_enabled
```

Update `forward` signature `vision: Tensor | None` and gate the vision branch. Replace the head of `forward` body:

```python
        m = self.act(self.motor_proj(motor_state))
        m = self._gate_projection(m, valid_mask, "motor")
        parts: list[Tensor] = []
        if self._vision_enabled:
            if vision is None:
                msg = "vision tensor required when vision modality is enabled"
                raise ValueError(msg)
            v = self.act(self.vision_proj(vision))
            v = self._gate_projection(v, valid_mask, "vision")
            parts.append(v)
        parts.append(m)
```

The remaining `ultrasonic`/`audio`/`lidar` blocks already append to `parts`; change the ultrasonic `parts.insert(1, ...)` to `parts.append(...)` for order-independence (gating is name-keyed via `SENSOR_SLOT_MAP`, so concat order only needs to be internally consistent). Replace any `vision.shape[0]`/`vision.device`/`vision.dtype` references used to size zero-fallback tensors with a `_ref` tensor derived from `motor_state` (always present):

```python
        ref = motor_state
        # in each disabled-data zero-fill: device=ref.device, dtype=ref.dtype, batch=ref.shape[0]
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/unit/world_model/test_encoder_vision_optional.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: 4 passed.

- [ ] **Step 5: Run the existing encoder + RSSM tests to prove no regression**

Run: `python -m pytest tests/unit/world_model/ --import-mode=importlib --no-cov -q 2>&1 | tail -15`
Expected: 0 failures (default `vision_dim=256` path unchanged).

- [ ] **Step 6: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/world_model/encoder.py tests/unit/world_model/test_encoder_vision_optional.py && python -m ruff format --check src/mousedroid/world_model/encoder.py tests/unit/world_model/test_encoder_vision_optional.py && python -m mypy --strict src/mousedroid/world_model/encoder.py`
Expected: clean.

```bash
git add src/mousedroid/world_model/encoder.py tests/unit/world_model/test_encoder_vision_optional.py
git commit -m "feat(world_model): make encoder vision branch optional (vision_dim=0)

Mirrors the existing audio/lidar gating pattern. Default vision_dim=256
keeps the deployed model byte-identical (invariant #9); vision_dim=0
drops the branch so the RSSM can pretrain on dynamics+proprioception
without a camera. Closes peer-review BLOCKER B2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add free-bits / KL-balancing helper

**Files:**
- Modify: `src/mousedroid/world_model/latent_utils.py`
- Test: `tests/unit/world_model/test_balanced_free_bits_kl.py`

- [ ] **Step 1: Write the failing test**

```python
"""Balanced, free-bits, fp32-stable KL for RSSM training."""
from __future__ import annotations

import torch

from mousedroid.world_model.latent_utils import balanced_free_bits_kl


def _g(mean: float, logvar: float, n: int = 4, d: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.full((n, d), mean), torch.full((n, d), logvar)


def test_identical_distributions_clamped_to_free_nats() -> None:
    pm, pl = _g(0.0, 0.0)
    km, kl = _g(0.0, 0.0)
    out = balanced_free_bits_kl(pm, pl, km, kl, alpha=0.8, free_nats=1.0)
    # zero KL is clamped up to free_nats
    assert torch.isfinite(out)
    assert float(out) == 1.0


def test_free_nats_zero_recovers_plain_kl_scale() -> None:
    pm, pl = _g(2.0, 0.0)
    rm, rl = _g(0.0, 0.0)
    out = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.5, free_nats=0.0)
    assert float(out) > 0.0


def test_fp16_inputs_do_not_overflow() -> None:
    pm = torch.full((2, 4), 0.0, dtype=torch.float16)
    pl = torch.full((2, 4), 30.0, dtype=torch.float16)  # exp(30) overflows fp16
    rm = torch.zeros(2, 4, dtype=torch.float16)
    rl = torch.zeros(2, 4, dtype=torch.float16)
    out = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.8, free_nats=1.0)
    assert torch.isfinite(out)


def test_balancing_is_convex_combination() -> None:
    pm, pl = _g(1.0, 0.5)
    rm, rl = _g(0.0, 0.0)
    a0 = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.0, free_nats=0.0)
    a1 = balanced_free_bits_kl(pm, pl, rm, rl, alpha=1.0, free_nats=0.0)
    amid = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.5, free_nats=0.0)
    assert min(float(a0), float(a1)) - 1e-4 <= float(amid) <= max(float(a0), float(a1)) + 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/world_model/test_balanced_free_bits_kl.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: FAIL — `ImportError: cannot import name 'balanced_free_bits_kl'`.

- [ ] **Step 3: Implement the helper**

Append to `src/mousedroid/world_model/latent_utils.py` (it already exports `kl_divergence`):

```python
_LOGVAR_CLAMP = 10.0


def balanced_free_bits_kl(
    post_mean: Tensor,
    post_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
    *,
    alpha: float,
    free_nats: float,
) -> Tensor:
    """KL-balanced, free-bits, fp32-stable KL(posterior || prior).

    Implements Dreamer-v2/v3-style KL balancing — ``alpha`` weights the
    prior-update term (posterior detached) against the posterior-update term
    (prior detached) — followed by a free-bits floor at ``free_nats`` nats.
    Computed in float32 with logvars clamped to ``[-10, 10]`` so an fp16 AMP
    context cannot overflow ``exp(logvar)`` into NaN.

    Args:
        post_mean: Posterior mean, shape ``(batch, latent_dim)``.
        post_logvar: Posterior log-variance, same shape.
        prior_mean: Prior mean, same shape.
        prior_logvar: Prior log-variance, same shape.
        alpha: Balancing weight in ``[0, 1]`` (Dreamer default ~0.8).
        free_nats: Per-batch free-bits floor (nats). ``0`` disables the floor.

    Returns:
        Scalar mean KL (after balancing + free-bits), as a float32 tensor.
    """

    def _kl(pm: Tensor, plv: Tensor, qm: Tensor, qlv: Tensor) -> Tensor:
        pm, plv = pm.float(), plv.float().clamp(-_LOGVAR_CLAMP, _LOGVAR_CLAMP)
        qm, qlv = qm.float(), qlv.float().clamp(-_LOGVAR_CLAMP, _LOGVAR_CLAMP)
        return 0.5 * (qlv - plv + (plv.exp() + (pm - qm) ** 2) / qlv.exp() - 1.0)

    kl_lhs = _kl(post_mean.detach(), post_logvar.detach(), prior_mean, prior_logvar)
    kl_rhs = _kl(post_mean, post_logvar, prior_mean.detach(), prior_logvar.detach())
    kl = alpha * kl_lhs + (1.0 - alpha) * kl_rhs
    kl = kl.sum(dim=-1).mean()  # sum over latent dims, mean over batch
    if free_nats > 0.0:
        kl = torch.clamp(kl, min=free_nats)
    return kl
```

Ensure `Tensor` is imported at the top of the module (`from torch import Tensor`); add if missing.

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/unit/world_model/test_balanced_free_bits_kl.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 4 passed.

- [ ] **Step 5: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/world_model/latent_utils.py tests/unit/world_model/test_balanced_free_bits_kl.py && python -m ruff format --check src/mousedroid/world_model/latent_utils.py tests/unit/world_model/test_balanced_free_bits_kl.py && python -m mypy --strict src/mousedroid/world_model/latent_utils.py`
Expected: clean.

```bash
git add src/mousedroid/world_model/latent_utils.py tests/unit/world_model/test_balanced_free_bits_kl.py
git commit -m "feat(world_model): add balanced free-bits fp32 KL helper

Dreamer-style KL balancing + free-bits floor, computed in float32 with
clamped logvars so an fp16 AMP context cannot overflow exp(logvar).
Closes peer-review MAJOR (posterior collapse + AMP NaN).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Add raw-modality decoders + `RSSM.train_sequence`

**Files:**
- Modify: `src/mousedroid/world_model/rssm.py`
- Modify: `src/mousedroid/config/schema.py` (add pretrain knobs — see Step 3a)
- Test: `tests/unit/world_model/test_rssm_train_sequence.py`

- [ ] **Step 1: Write the failing test (loss decreases, grads flow, no collapse)**

```python
"""RSSM.train_sequence: grad-enabled raw-modality reconstruction + KL."""
from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.rssm import RSSM


def _model() -> RSSM:
    # vision off (pretraining variant); lidar on so we exercise the lidar head.
    cfg = ModelConfig(vision_dim=0, lidar_dim=16, lidar_proj_dim=32)  # type: ignore[arg-type]
    torch.manual_seed(0)
    return RSSM(cfg)


def _batch(model: RSSM, b: int = 4, t: int = 6) -> dict[str, torch.Tensor]:
    cfg = model.cfg
    return {
        "motor": torch.randn(b, t, cfg.motor_state_dim),
        "ultrasonic": torch.rand(b, t, 1),
        "lidar": torch.rand(b, t, cfg.lidar_dim),
        "valid_mask": torch.ones(b, t, 5),
        "action": torch.randn(b, t, cfg.action_dim),
    }


def test_train_sequence_returns_finite_losses() -> None:
    model = _model()
    out = model.train_sequence(_batch(model))
    for key in ("loss", "recon", "kl"):
        assert torch.isfinite(out[key])
    assert out["loss"].requires_grad


def test_train_sequence_backward_populates_grads() -> None:
    model = _model()
    out = model.train_sequence(_batch(model))
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_overfits_single_batch_loss_decreases() -> None:
    model = _model()
    batch = _batch(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(40):
        opt.zero_grad()
        out = model.train_sequence(batch)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = float(out["loss"])
    assert float(out["loss"]) < first  # learns something


def test_no_posterior_collapse_probe() -> None:
    """posterior_std stays above a floor — guards the B1 collapse failure."""
    model = _model()
    out = model.train_sequence(_batch(model))
    assert float(out["posterior_std"]) > 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/world_model/test_rssm_train_sequence.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: FAIL — `RSSM` has no `cfg` property and no `train_sequence`.

- [ ] **Step 3a: Add KL knobs to `ModelConfig` and pretrain knobs to `TrainingConfig`**

**(i) `ModelConfig`** (~line 1567). `train_sequence` reads the KL knobs off
`self._cfg` (a `ModelConfig`), so they live here. Add (additive, Dreamer-default
values; `kl_beta` may already exist on `TrainingConfig` but the *model* needs its
own copy for the training forward):

```python
    kl_beta: float = Field(
        1.0, ge=0.0, description="KL weight in the RSSM training ELBO (recon + kl_beta*KL)."
    )
    kl_balance_alpha: float = Field(
        0.8, ge=0.0, le=1.0, description="Dreamer KL-balancing weight (prior-update term)."
    )
    kl_free_nats: float = Field(
        1.0, ge=0.0, description="Free-bits floor (nats) for the training KL."
    )
```

**(ii) `TrainingConfig`** (~line 3557, already has `kl_beta`, `sequence_length`,
`n_episodes`, `learning_rate`, `epochs`, `batch_size`, `weights_dir`). These are
the operator-facing overrides `build_rssm_trainable` copies onto the model cfg
(Task 4), plus the loop knobs (additive — invariant #9):

```python
    rssm_pretrain_enabled: bool = Field(
        False,
        description="Opt-in: run the MuJoCo->RSSM dynamics pretraining loop in the rssm phase.",
    )
    rssm_free_nats: float = Field(
        1.0, ge=0.0, description="Free-bits floor (nats) for the RSSM training KL."
    )
    rssm_kl_balance_alpha: float = Field(
        0.8, ge=0.0, le=1.0, description="Dreamer KL-balancing weight (prior-update term)."
    )
    rssm_grad_clip: float = Field(
        100.0, gt=0.0, description="Global grad-norm clip for RSSM pretraining."
    )
    rssm_checkpoint_name: str = Field(
        "rssm_pretrained.pt", min_length=1, description="Filename for the RSSM pretrain checkpoint."
    )
```

- [ ] **Step 3b: Add `cfg` property, raw decoders, and `train_sequence` to `RSSM`**

In `src/mousedroid/world_model/rssm.py` `__init__`, after the existing heads, add raw-modality decoder heads (built to match the configured modalities) and keep `self._cfg`:

```python
        # Raw-modality decoders used ONLY by train_sequence (deployment path
        # uses observation_decoder/reward_head, untouched). Reconstructing the
        # RAW sim observations (fixed targets) avoids the obs_embed self-
        # reconstruction collapse (peer-review BLOCKER B1).
        self.decode_motor = nn.Linear(cfg.hidden_dim + cfg.latent_dim, cfg.motor_state_dim)
        self._range_enabled = cfg.ultrasonic_dim > 0
        if self._range_enabled:
            self.decode_range = nn.Linear(cfg.hidden_dim + cfg.latent_dim, 1)
        self._lidar_enabled = cfg.lidar_dim > 0
        if self._lidar_enabled:
            self.decode_lidar = nn.Linear(cfg.hidden_dim + cfg.latent_dim, cfg.lidar_dim)
```

Add a public read-only accessor:

```python
    @property
    def cfg(self) -> ModelConfig:
        """Return the model configuration (read-only)."""
        return self._cfg
```

Add the trainable forward (NO `@torch.no_grad()`):

```python
    def train_sequence(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Gradient-enabled sequence rollout for dynamics pretraining.

        Reconstructs the RAW per-modality sim observations (motor/range/lidar)
        — fixed targets, so the objective cannot collapse the way an
        obs_embed self-reconstruction would. KL uses the balanced free-bits
        helper in float32. Vision is expected OFF (encoder built with
        vision_dim=0); ``batch`` therefore carries no vision tensor.

        Args:
            batch: Dict of ``(B, T, ...)`` tensors with keys ``motor``,
                ``valid_mask``, ``action`` (always) and ``ultrasonic`` /
                ``lidar`` when those modalities are enabled.

        Returns:
            Dict with scalar tensors ``loss``, ``recon``, ``kl`` and a
            detached ``posterior_std`` collapse probe.
        """
        motor = batch["motor"]
        actions = batch["action"]
        mask = batch["valid_mask"]
        b, t, _ = motor.shape
        device = motor.device
        h = torch.zeros(b, self._cfg.hidden_dim, device=device)
        z = torch.zeros(b, self._cfg.latent_dim, device=device)

        recon = torch.zeros((), device=device)
        kl_total = torch.zeros((), device=device)
        post_stds: list[Tensor] = []

        for step in range(t):
            ultra = batch["ultrasonic"][:, step] if self._range_enabled else None
            lidar = batch["lidar"][:, step] if self._lidar_enabled else None
            obs_embed = self.encoder(None, ultra, motor[:, step], mask[:, step], lidar=lidar)

            gru_in = torch.cat([z, actions[:, step]], dim=-1)
            h = self.gru(gru_in, h)

            post_params = self.posterior(torch.cat([h, obs_embed], dim=-1))
            z, post_mean, post_logvar = self._sample_gaussian(post_params)
            prior_params = self.prior(h)
            _, prior_mean, prior_logvar = self._sample_gaussian(prior_params)

            with torch.autocast(device_type=device.type, enabled=False):
                kl_total = kl_total + balanced_free_bits_kl(
                    post_mean,
                    post_logvar,
                    prior_mean,
                    prior_logvar,
                    alpha=self._cfg.kl_balance_alpha,
                    free_nats=self._cfg.kl_free_nats,
                )

            hz = torch.cat([h, z], dim=-1)
            recon = recon + nn.functional.mse_loss(self.decode_motor(hz), motor[:, step])
            if self._range_enabled and ultra is not None:
                recon = recon + nn.functional.mse_loss(self.decode_range(hz), ultra)
            if self._lidar_enabled and lidar is not None:
                recon = recon + nn.functional.mse_loss(self.decode_lidar(hz), lidar)
            post_stds.append((0.5 * post_logvar).exp().mean().detach())

        recon = recon / t
        kl = kl_total / t
        loss = recon + self._cfg.kl_beta * kl
        return {
            "loss": loss,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "posterior_std": torch.stack(post_stds).mean(),
        }
```

Add imports at the top of `rssm.py`: `from mousedroid.world_model.latent_utils import balanced_free_bits_kl` (alongside the existing `kl_divergence, sample_gaussian` import).

> **Config-field ownership (single source of truth):** `kl_beta`,
> `kl_balance_alpha`, `kl_free_nats` are read off `self._cfg` (a `ModelConfig`)
> inside `train_sequence` — they are added to `ModelConfig` in Step 3a(i). The
> operator-facing `TrainingConfig.rssm_free_nats` / `rssm_kl_balance_alpha` are
> copied onto the model cfg by `build_rssm_trainable` (Task 4), so an operator
> tunes them in one place (`training:`) and the model picks them up at build
> time. No call-site threading; no duplicate authority.

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/unit/world_model/test_rssm_train_sequence.py --import-mode=importlib -v 2>&1 | tail -20`
Expected: 4 passed.

- [ ] **Step 5: Prove deployment path untouched**

Run: `python -m pytest tests/unit/world_model/ --import-mode=importlib --no-cov -q 2>&1 | tail -10`
Expected: 0 failures (observe_step/imagine_step unchanged).

- [ ] **Step 6: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/world_model/rssm.py src/mousedroid/config/schema.py tests/unit/world_model/test_rssm_train_sequence.py && python -m ruff format --check src/mousedroid/world_model/rssm.py tests/unit/world_model/test_rssm_train_sequence.py && python -m mypy --strict src/mousedroid/world_model/rssm.py`
Expected: clean.

```bash
git add src/mousedroid/world_model/rssm.py src/mousedroid/config/schema.py tests/unit/world_model/test_rssm_train_sequence.py
git commit -m "feat(world_model): RSSM.train_sequence + raw-modality decoders

Grad-enabled dynamics pretraining forward that reconstructs RAW sim
observations (motor/range/lidar) with balanced free-bits KL in fp32.
Fixed reconstruction targets close the obs_embed self-reconstruction
collapse (BLOCKER B1). Deployment no_grad paths untouched (invariant #7).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `build_rssm_trainable` factory

**Files:**
- Modify: `src/mousedroid/factory.py` (near `build_world_model`, line ~520)
- Test: `tests/unit/test_factory_rssm_trainable.py`

- [ ] **Step 1: Write the failing test**

```python
"""build_rssm_trainable returns a concrete trainable RSSM with vision off."""
from __future__ import annotations

import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import build_rssm_trainable
from mousedroid.world_model.rssm import RSSM


def test_returns_trainable_rssm_vision_off() -> None:
    cfg = Settings(mock_hardware=True)
    model = build_rssm_trainable(cfg)
    assert isinstance(model, RSSM)
    assert model.encoder.vision_enabled is False  # pretraining drops vision
    assert any(p.requires_grad for p in model.parameters())


def test_overrides_pretrain_knobs_from_training_config() -> None:
    cfg = Settings(mock_hardware=True)
    model = build_rssm_trainable(cfg)
    assert model.cfg.kl_free_nats == cfg.training.rssm_free_nats
    assert model.cfg.kl_balance_alpha == cfg.training.rssm_kl_balance_alpha
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_factory_rssm_trainable.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: FAIL — `ImportError: cannot import name 'build_rssm_trainable'`.

- [ ] **Step 3: Implement the factory**

In `src/mousedroid/factory.py`, add near `build_world_model`:

```python
def build_rssm_trainable(cfg: Settings) -> RSSM:
    """Build the concrete trainable RSSM for MuJoCo dynamics pretraining.

    Unlike :func:`build_world_model` (which returns a ``WorldModelProtocol``
    wrapper for deployment), this returns the concrete ``nn.Module`` so the
    pretrainer can run ``train_sequence`` + backprop. Vision is disabled
    (``vision_dim=0``) — the sim has no camera; the dynamics core is what
    gets pretrained. Operator pretrain knobs from ``TrainingConfig`` are
    copied onto the model config.
    """
    from mousedroid.world_model.rssm import RSSM  # concrete import inside factory

    model_cfg = cfg.model.model_copy(
        update={
            "vision_dim": 0,
            "kl_free_nats": cfg.training.rssm_free_nats,
            "kl_balance_alpha": cfg.training.rssm_kl_balance_alpha,
        }
    )
    return RSSM(model_cfg)
```

Add a module-level `from mousedroid.world_model.rssm import RSSM` under `TYPE_CHECKING` for the return annotation, or use a string annotation `-> "RSSM"`. Prefer the `TYPE_CHECKING` import to satisfy mypy without an eager concrete import.

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/unit/test_factory_rssm_trainable.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 2 passed.

- [ ] **Step 5: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/factory.py tests/unit/test_factory_rssm_trainable.py && python -m ruff format --check src/mousedroid/factory.py tests/unit/test_factory_rssm_trainable.py && python -m mypy --strict src/mousedroid/factory.py`
Expected: clean.

```bash
git add src/mousedroid/factory.py tests/unit/test_factory_rssm_trainable.py
git commit -m "feat(factory): build_rssm_trainable (concrete RSSM, vision off)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Layer-1 backwards-compat regression gate

**Files:**
- Test: `tests/regression/test_phase5_backwards_compat.py`

- [ ] **Step 1: Write the regression test**

```python
"""Phase 5 Layer-1 additions are byte-identical by default."""
from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.world_model.encoder import MultimodalEncoder


def test_default_model_has_vision_enabled() -> None:
    assert MultimodalEncoder(ModelConfig()).vision_enabled is True


def test_default_training_config_pretrain_disabled() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.training.rssm_pretrain_enabled is False


def test_default_encoder_output_shape_unchanged() -> None:
    cfg = ModelConfig()
    enc = MultimodalEncoder(cfg)
    out = enc(torch.zeros(1, cfg.vision_dim), None, torch.zeros(1, cfg.motor_state_dim), torch.ones(1, 5))
    assert out.shape == (1, cfg.obs_dim)
```

- [ ] **Step 2: Run + verify pass + commit**

Run: `python -m pytest tests/regression/test_phase5_backwards_compat.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 3 passed.

```bash
git add tests/regression/test_phase5_backwards_compat.py
git commit -m "test(regression): pin Phase 5 Layer-1 byte-identical defaults

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# LAYER 2 — MuJoCo skid-steer env (independent of Layer 1)

> The MJCF parameters below were empirically validated against `mujoco` 3.8.0
> (skid-steer yaw-from-slip, IMU sensors, rangefinder lidar all confirmed
> working). `mujoco>=3.0` is **already** in the `[arm]` optional-dependency
> extra — no new extra is added.

### Task 6: Add `MujocoSimConfig` schema block

**Files:**
- Modify: `src/mousedroid/config/schema.py`
- Test: `tests/regression/test_mujoco_sim_config.py`

- [ ] **Step 1: Write the failing test**

```python
"""MujocoSimConfig is additive and defaults are sane."""
from __future__ import annotations

import yaml

from mousedroid.config.schema import Settings


def test_default_settings_have_no_mujoco_block_required() -> None:
    cfg = Settings(mock_hardware=True)
    # rover.sim.mujoco resolves to defaults even when YAML omits it
    assert cfg.rover.sim.mujoco.mjcf_path.endswith("mse6_4wd.xml")
    assert cfg.rover.sim.mujoco.arena_half_extent_m > 0


def test_opt_in_overrides_parse() -> None:
    raw = yaml.safe_load(
        """
        mock_hardware: true
        rover:
          sim:
            backend: mujoco
            mujoco:
              lidar_num_sectors: 12
              wheel_friction_default: 1.1
        """
    )
    cfg = Settings.model_validate(raw)
    assert cfg.rover.sim.backend == "mujoco"
    assert cfg.rover.sim.mujoco.lidar_num_sectors == 12
    assert cfg.rover.sim.mujoco.wheel_friction_default == 1.1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/regression/test_mujoco_sim_config.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: FAIL — `RoverSimConfig` has no `mujoco` attribute.

- [ ] **Step 3: Add the model + nest it on `RoverSimConfig`**

In `src/mousedroid/config/schema.py`, define before `RoverSimConfig`:

```python
class MujocoSimConfig(BaseModel):
    """MuJoCo backend parameters (consumed only when rover.sim.backend == 'mujoco').

    Every physics knob is config-driven (invariant #3). ``wheel_slip_default``
    is a documented OBSERVATION-NOISE proxy — MuJoCo has no first-class slip
    parameter — applied as multiplicative noise on wheel_vel/pose, NOT a
    contact-solver field.
    """

    mjcf_path: str = Field(
        "assets/rover/mse6_4wd.xml",
        min_length=1,
        description="Repo-relative path to the skid-steer MJCF (resolved against repo root).",
    )
    arena_half_extent_m: float = Field(
        2.0, gt=0.0, description="Half-size of the walled arena (walls give the lidar signal)."
    )
    lidar_num_sectors: int = Field(
        16, gt=0, description="Number of rangefinder sectors fanned around yaw."
    )
    lidar_max_range_m: float = Field(
        4.0, gt=0.0, description="Rangefinder clip; readings normalised to [0,1] by this."
    )
    battery_voltage_const_v: float = Field(
        12.0, gt=0.0, description="Constant battery voltage stamped into motor_state[3]."
    )
    wheel_friction_default: float = Field(
        1.0, gt=0.0, description="Default tangential friction (geom_friction[:,0])."
    )
    wheel_slip_default: float = Field(
        0.0, ge=0.0, description="Observation-noise proxy magnitude (NOT a MuJoCo field)."
    )
    motor_gain_default: float = Field(
        1.0, gt=0.0, description="Default actuator gain (actuator_gainprm[:,0])."
    )
    chassis_mass_default_kg: float = Field(
        2.7, gt=0.0, description="Default chassis mass (body_mass + inertia recompute)."
    )
```

On `class RoverSimConfig(BaseModel)` add the nested field (additive; default factory keeps pre-feature YAML valid — invariant #9):

```python
    mujoco: MujocoSimConfig = Field(
        default_factory=MujocoSimConfig,
        description="MuJoCo backend parameters (used only when backend == 'mujoco').",
    )
```

- [ ] **Step 4: Run + verify pass**

Run: `python -m pytest tests/regression/test_mujoco_sim_config.py tests/regression/ -k "config or overlay or compat" --import-mode=importlib --no-cov -q 2>&1 | tail -10`
Expected: 0 failures (existing config overlays still load).

- [ ] **Step 5: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/config/schema.py tests/regression/test_mujoco_sim_config.py && python -m ruff format --check src/mousedroid/config/schema.py tests/regression/test_mujoco_sim_config.py && python -m mypy --strict src/mousedroid/config/schema.py`
Expected: clean.

```bash
git add src/mousedroid/config/schema.py tests/regression/test_mujoco_sim_config.py
git commit -m "feat(config): add MujocoSimConfig (additive, slip = documented obs-noise proxy)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Author the skid-steer MJCF asset

**Files:**
- Create: `assets/rover/mse6_4wd.xml`
- Test: `tests/unit/sim/test_mjcf_loads.py`

- [ ] **Step 1: Write the failing test (model loads, sensors present, rest-state finite)**

```python
"""The MJCF loads, exposes the expected sensors, and is stable at rest."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MJCF = _REPO_ROOT / "assets" / "rover" / "mse6_4wd.xml"


def test_model_loads() -> None:
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    assert model.nu == 4  # 4 wheel velocity actuators
    assert model.nsensor >= 3  # accel + gyro + rangefinders


def test_rest_state_is_finite() -> None:
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qacc).all()
    # the chassis should not be free-falling through the floor
    assert abs(float(data.qacc[2])) < 50.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/sim/test_mjcf_loads.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: FAIL — file does not exist (or SKIP if mujoco absent — then `pip install "mujoco>=3.0"` first).

- [ ] **Step 3: Write the MJCF**

Create `assets/rover/mse6_4wd.xml` (parameters empirically validated; wheels grounded so the rest-state assertion passes; 4 perimeter walls for lidar signal; FL/FR/RL/RR wheel order matches the mock):

```xml
<mujoco model="mse6_4wd_skidsteer">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.00833" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default>
    <geom condim="3" friction="1 0.005 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <joint damping="0.01"/>
  </default>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.4 0.4 0.4 1"/>
    <!-- Perimeter walls so the rangefinder lidar has geometry to hit. -->
    <geom name="wall_n" type="box" pos="0 2 0.15" size="2 0.05 0.15"/>
    <geom name="wall_s" type="box" pos="0 -2 0.15" size="2 0.05 0.15"/>
    <geom name="wall_e" type="box" pos="2 0 0.15" size="0.05 2 0.15"/>
    <geom name="wall_w" type="box" pos="-2 0 0.15" size="0.05 2 0.15"/>
    <body name="chassis" pos="0 0 0.06">
      <freejoint name="root"/>
      <geom name="chassis_geom" type="box" size="0.11 0.09 0.03" mass="2.7"/>
      <site name="imu_site" pos="0 0 0"/>
      <!-- Lidar fan: sites are added programmatically? No — declare 16 here for parity. -->
      <site name="lidar_0" pos="0.11 0 0.03" zaxis="1 0 0"/>
      <body name="wheel_fl" pos="0.09 0.10 -0.018">
        <joint name="j_fl" type="hinge" axis="0 1 0"/>
        <geom name="g_fl" type="cylinder" size="0.042 0.02" quat="0.7071 0.7071 0 0" mass="0.05"/>
      </body>
      <body name="wheel_fr" pos="0.09 -0.10 -0.018">
        <joint name="j_fr" type="hinge" axis="0 1 0"/>
        <geom name="g_fr" type="cylinder" size="0.042 0.02" quat="0.7071 0.7071 0 0" mass="0.05"/>
      </body>
      <body name="wheel_rl" pos="-0.09 0.10 -0.018">
        <joint name="j_rl" type="hinge" axis="0 1 0"/>
        <geom name="g_rl" type="cylinder" size="0.042 0.02" quat="0.7071 0.7071 0 0" mass="0.05"/>
      </body>
      <body name="wheel_rr" pos="-0.09 -0.10 -0.018">
        <joint name="j_rr" type="hinge" axis="0 1 0"/>
        <geom name="g_rr" type="cylinder" size="0.042 0.02" quat="0.7071 0.7071 0 0" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="a_fl" joint="j_fl" kv="0.05"/>
    <velocity name="a_fr" joint="j_fr" kv="0.05"/>
    <velocity name="a_rl" joint="j_rl" kv="0.05"/>
    <velocity name="a_rr" joint="j_rr" kv="0.05"/>
  </actuator>
  <sensor>
    <accelerometer name="imu_acc" site="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <rangefinder name="lidar_s0" site="lidar_0"/>
  </sensor>
</mujoco>
```

> **Lidar sectors note:** authoring N rotated `<site>` + `<rangefinder>` pairs by
> hand is verbose. The `RoverMuJoCoEnv` (Task 8) instead builds the full
> N-sector fan by editing the parsed MJCF spec (`mujoco.MjSpec`) at construction
> time from `cfg.rover.sim.mujoco.lidar_num_sectors`, so the XML ships ONE
> reference sector and the env adds the rest. This keeps `lidar_num_sectors`
> config-driven (invariant #3). If `MjSpec` site-injection proves fiddly, the
> fallback is to emit the rotated sites into a temp MJCF string before load.

- [ ] **Step 4: Run + verify pass**

Run: `python -m pytest tests/unit/sim/test_mjcf_loads.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 2 passed (or skipped if mujoco unavailable on this host — fine for CI gating, env tests are importorskip-gated).

- [ ] **Step 5: Commit** (no lint on XML; ruff/mypy skip non-Python)

```bash
git add assets/rover/mse6_4wd.xml tests/unit/sim/test_mjcf_loads.py
git commit -m "feat(sim): skid-steer MJCF for the MuJoCo rover backend

Empirically-validated params (implicitfast, friction cone, grounded wheels).
4 perimeter walls give the rangefinder lidar signal; accel+gyro sensors
supply the 6D IMU. Rest-state finite-qacc test guards the silent-NaN footgun.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Implement `RoverMuJoCoEnv`

**Files:**
- Create: `src/mousedroid/sim/mujoco_rover_env.py`
- Test: `tests/unit/sim/test_mujoco_rover_env.py`

- [ ] **Step 1: Write the failing test (protocol + obs parity vs mock + NaN guard)**

```python
"""RoverMuJoCoEnv conforms to the protocol and matches the mock obs contract."""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import Settings
from mousedroid.sim.mock_rover_env import MockRoverEnv
from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv
from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
    RoverEnvProtocol,
)


def _envs() -> tuple[RoverMuJoCoEnv, MockRoverEnv]:
    cfg = Settings(mock_hardware=True)
    mj = RoverMuJoCoEnv(cfg.rover, wheel_radius_m=cfg.robot.wheel_radius_m, track_width_m=cfg.robot.track_width_m)
    mock = MockRoverEnv(cfg.rover, wheel_radius_m=cfg.robot.wheel_radius_m, track_width_m=cfg.robot.track_width_m)
    return mj, mock


def test_satisfies_protocol() -> None:
    mj, _ = _envs()
    assert isinstance(mj, RoverEnvProtocol)


def test_observation_keys_match_mock() -> None:
    mj, mock = _envs()
    assert mj.observation_keys == mock.observation_keys


def test_reset_obs_shapes_match_contract() -> None:
    mj, _ = _envs()
    obs, _info = mj.reset(seed=0)
    if "imu" in obs:
        assert obs["imu"].shape == (ROVER_IMU_DIM,)
    if "chassis_pose" in obs:
        assert obs["chassis_pose"].shape == (ROVER_CHASSIS_POSE_DIM,)
    if "wheel_vel" in obs:
        assert obs["wheel_vel"].shape == (ROVER_NUM_WHEELS,)


def test_step_advances_and_is_finite() -> None:
    mj, _ = _envs()
    mj.reset(seed=0)
    action = np.full((mj.action_dim,), 8.0, dtype=np.float32)
    obs, reward, term, trunc, info = mj.step(action)
    assert np.isfinite(reward)
    for v in obs.values():
        assert np.isfinite(v).all()


def test_spin_in_place_changes_heading() -> None:
    """Skid-steer: opposite wheel commands rotate the chassis (sanity on physics)."""
    mj, _ = _envs()
    mj.reset(seed=0)
    # differential mode: action = [left, right]; spin = [-v, +v]
    for _ in range(60):
        obs, *_ = mj.step(np.asarray([-8.0, 8.0], dtype=np.float32))
    pose = obs["chassis_pose"]
    heading = float(np.arctan2(pose[3], pose[2]))
    assert abs(heading) > 0.1


def test_close_is_idempotent() -> None:
    mj, _ = _envs()
    mj.close()
    mj.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/sim/test_mujoco_rover_env.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `RoverMuJoCoEnv`**

Create `src/mousedroid/sim/mujoco_rover_env.py`. Key design points: lazy `mujoco` import in `__init__`; resolve `mjcf_path` against repo root; build the N-sector lidar fan from config; rest-state assertion; obs dict identical to `MockRoverEnv`; differential `[L,R]` → wheel velocity setpoints; IMU from `sensordata`; DR-param application via a public `apply_domain_params(...)` (Task 9 fills the body).

```python
"""MuJoCo skid-steer rover environment (RoverEnvProtocol backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import RoverConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
)

_log = get_logger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_HEADING_WRAP = 2.0 * np.pi


class RoverMuJoCoEnv:
    """4-wheel skid-steer rover backed by the MuJoCo classic engine.

    Conforms structurally to :class:`RoverEnvProtocol`; produces the SAME
    observation-dict contract as :class:`MockRoverEnv` (keys, shapes, FL/FR/
    RL/RR wheel order) so backends are interchangeable. ``mujoco`` is imported
    lazily so importing this module never requires the engine.
    """

    def __init__(self, cfg: RoverConfig, wheel_radius_m: float, track_width_m: float) -> None:
        import mujoco  # noqa: PLC0415 — lazy; only the mujoco backend pays the cost

        self._mj = mujoco
        self._cfg = cfg
        self._mjcfg = cfg.sim.mujoco
        self._wheel_radius = wheel_radius_m
        self._track_width = track_width_m
        self._control_dt_s = cfg.sim.sim_dt_s * cfg.sim.decimation
        self._decimation = cfg.sim.decimation
        self._max_steps = max(1, int(cfg.sim.episode_length_s / self._control_dt_s))
        self._action_dim = cfg.action.action_dim
        self._obs_keys: tuple[str, ...] = cfg.observation.enabled_keys()
        self._lidar_sectors = self._mjcfg.lidar_num_sectors

        path = (_REPO_ROOT / self._mjcfg.mjcf_path).resolve()
        self._model = self._build_model(path)
        self._data = mujoco.MjData(self._model)
        self._wheel_vel = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        self._step_idx = 0
        self._noise_rng = np.random.default_rng(0)
        self._slip_noise = self._mjcfg.wheel_slip_default

        self._assert_rest_state_stable()
        _log.info(
            "mujoco_rover_env_initialised",
            mjcf=str(path),
            nu=int(self._model.nu),
            lidar_sectors=self._lidar_sectors,
            control_dt_s=self._control_dt_s,
        )

    def _build_model(self, path: Path) -> Any:
        """Load the MJCF and inject the configured N-sector lidar fan."""
        if not path.exists():
            msg = f"MJCF not found at {path}"
            raise FileNotFoundError(msg)
        spec = self._mj.MjSpec.from_file(str(path))
        # Inject (lidar_sectors - 1) extra rangefinder sites/sensors fanned in yaw.
        # The XML ships sector 0; add the rest from config. Implementation detail:
        # locate the chassis body, add sites rotated about z, add rangefinder sensors.
        self._inject_lidar_fan(spec)
        return spec.compile()

    def _inject_lidar_fan(self, spec: Any) -> None:
        """Add (N-1) rotated rangefinder sites+sensors to the chassis body."""
        if self._lidar_sectors <= 1:
            return
        chassis = spec.body("chassis")
        for i in range(1, self._lidar_sectors):
            ang = 2.0 * np.pi * i / self._lidar_sectors
            site = chassis.add_site()
            site.name = f"lidar_{i}"
            site.pos = [0.11 * np.cos(ang), 0.11 * np.sin(ang), 0.03]
            site.zaxis = [np.cos(ang), np.sin(ang), 0.0]
            sensor = spec.add_sensor()
            sensor.name = f"lidar_s{i}"
            sensor.type = self._mj.mjtSensor.mjSENS_RANGEFINDER
            sensor.objtype = self._mj.mjtObj.mjOBJ_SITE
            sensor.objname = f"lidar_{i}"

    def _assert_rest_state_stable(self) -> None:
        """Raise if the model free-falls / interpenetrates at rest (silent-NaN guard)."""
        self._mj.mj_forward(self._model, self._data)
        if not np.isfinite(self._data.qacc).all():
            msg = "MuJoCo rover unstable at rest: non-finite qacc (check wheel grounding)"
            raise RuntimeError(msg)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return self._obs_keys

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        self._mj.mj_resetData(self._model, self._data)
        if seed is not None:
            self._noise_rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._wheel_vel = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        self._mj.mj_forward(self._model, self._data)
        return self._observe(), {"step_idx": self._step_idx}

    def step(self, action: NDArray[np.float32]) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        if action.shape != (self._action_dim,):
            msg = f"action shape must be ({self._action_dim},), got {action.shape}"
            raise ValueError(msg)
        left, right = self._action_to_wheel_setpoints(action)
        # FL, FR, RL, RR (parity with MockRoverEnv wheel order).
        self._data.ctrl[:] = np.asarray([left, right, left, right], dtype=np.float64)
        for _ in range(self._decimation):
            self._mj.mj_step(self._model, self._data)
        self._step_idx += 1

        obs = self._observe()
        goal = np.asarray(self._cfg.task.goal_xy_m, dtype=np.float32)
        pose = self._data.qpos[:2]
        distance = float(np.hypot(goal[0] - pose[0], goal[1] - pose[1]))
        reward = -distance
        truncated = self._step_idx >= self._max_steps
        terminated = distance < self._cfg.task.goal_reach_radius_m
        info = {"step_idx": self._step_idx, "distance_to_goal_m": distance}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Release MuJoCo data (idempotent)."""
        self._data = None  # type: ignore[assignment]

    def _action_to_wheel_setpoints(self, action: NDArray[np.float32]) -> tuple[float, float]:
        cap = self._cfg.action.max_wheel_rad_s
        if self._cfg.action.mode == "differential":
            return float(np.clip(action[0], -cap, cap)), float(np.clip(action[1], -cap, cap))
        # body_velocity: [vx, omega] -> wheel setpoints
        vx, omega = float(action[0]), float(action[1])
        left = (vx - 0.5 * omega * self._track_width) / self._wheel_radius
        right = (vx + 0.5 * omega * self._track_width) / self._wheel_radius
        return float(np.clip(left, -cap, cap)), float(np.clip(right, -cap, cap))

    def _observe(self) -> dict[str, NDArray[np.float32]]:
        obs: dict[str, NDArray[np.float32]] = {}
        oc = self._cfg.observation
        if oc.include_imu:
            obs["imu"] = self._read_imu()
        if oc.include_chassis_pose:
            obs["chassis_pose"] = self._read_pose()
        if oc.include_wheel_encoders:
            obs["wheel_vel"] = self._read_wheel_vel()
        if oc.include_lidar_sectors:
            obs["lidar"] = self._read_lidar(oc.lidar_num_sectors)
        return obs

    def _read_imu(self) -> NDArray[np.float32]:
        acc = self._sensor("imu_acc", 3)
        gyro = self._sensor("imu_gyro", 3)
        out = np.concatenate([acc, gyro]).astype(np.float32)
        assert out.shape == (ROVER_IMU_DIM,)
        return out

    def _read_pose(self) -> NDArray[np.float32]:
        x, y = float(self._data.qpos[0]), float(self._data.qpos[1])
        # heading from the freejoint quaternion (w,x,y,z at qpos[3:7]) about z
        qw, qz = float(self._data.qpos[3]), float(self._data.qpos[6])
        theta = float(np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz))
        pose = np.zeros(ROVER_CHASSIS_POSE_DIM, dtype=np.float32)
        pose[0], pose[1], pose[2], pose[3] = x, y, np.cos(theta), np.sin(theta)
        if self._slip_noise > 0.0:
            pose[:2] += self._noise_rng.normal(0.0, self._slip_noise, size=2).astype(np.float32)
        return pose

    def _read_wheel_vel(self) -> NDArray[np.float32]:
        # 4 hinge joint velocities live at the tail of qvel (after the 6-DoF freejoint).
        wv = np.asarray(self._data.qvel[6:6 + ROVER_NUM_WHEELS], dtype=np.float32)
        if self._slip_noise > 0.0:
            wv = wv * (1.0 + self._noise_rng.normal(0.0, self._slip_noise, size=wv.shape)).astype(np.float32)
        return wv

    def _read_lidar(self, n: int) -> NDArray[np.float32]:
        out = np.zeros(n, dtype=np.float32)
        rng = self._mjcfg.lidar_max_range_m
        for i in range(min(n, self._lidar_sectors)):
            raw = self._sensor(f"lidar_s{i}", 1)[0]
            # -1 sentinel (no hit) -> full range; normalise to [0,1].
            out[i] = 1.0 if raw < 0 else float(np.clip(raw / rng, 0.0, 1.0))
        return out

    def _sensor(self, name: str, dim: int) -> NDArray[np.float32]:
        sid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self._model.sensor_adr[sid])
        return np.asarray(self._data.sensordata[adr:adr + dim], dtype=np.float32)
```

- [ ] **Step 4: Run + verify pass**

Run: `python -m pytest tests/unit/sim/test_mujoco_rover_env.py --import-mode=importlib -v 2>&1 | tail -20`
Expected: all pass (skid-steer spin test confirms physics). If `MjSpec` site-injection API differs on the installed mujoco version, use the temp-MJCF-string fallback noted in Task 7 Step 3.

- [ ] **Step 5: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/sim/mujoco_rover_env.py tests/unit/sim/test_mujoco_rover_env.py && python -m ruff format --check src/mousedroid/sim/mujoco_rover_env.py tests/unit/sim/test_mujoco_rover_env.py && python -m mypy --strict src/mousedroid/sim/mujoco_rover_env.py`
Expected: clean (mujoco is `ignore_missing_imports`).

```bash
git add src/mousedroid/sim/mujoco_rover_env.py tests/unit/sim/test_mujoco_rover_env.py
git commit -m "feat(sim): RoverMuJoCoEnv skid-steer backend (RoverEnvProtocol)

Drop-in obs-contract parity with MockRoverEnv; IMU from accel+gyro sensors;
config-driven N-sector rangefinder lidar; rest-state finite-qacc guard.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Domain-randomization consumption + factory branch

**Files:**
- Modify: `src/mousedroid/sim/mujoco_rover_env.py` (add `apply_domain_params`)
- Modify: `src/mousedroid/factory.py` (`build_rover_env` mujoco branch)
- Test: `tests/unit/sim/test_mujoco_domain_params.py`, `tests/unit/test_factory_rover_env_mujoco.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/sim/test_mujoco_domain_params.py
"""DR params map onto concrete mjModel fields (friction/mass/gain); slip = obs noise."""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import Settings
from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv


def _env() -> RoverMuJoCoEnv:
    cfg = Settings(mock_hardware=True)
    return RoverMuJoCoEnv(cfg.rover, wheel_radius_m=cfg.robot.wheel_radius_m, track_width_m=cfg.robot.track_width_m)


def test_friction_param_writes_geom_friction() -> None:
    env = _env()
    env.apply_domain_params(friction=1.25, slip=0.0, mass_kg=2.7, motor_gain=1.0)
    # at least one wheel geom now carries the requested tangential friction
    frics = np.asarray(env._model.geom_friction[:, 0])  # noqa: SLF001 — white-box check
    assert np.isclose(frics.max(), 1.25, atol=1e-6)


def test_mass_param_writes_body_mass() -> None:
    env = _env()
    env.apply_domain_params(friction=1.0, slip=0.0, mass_kg=3.0, motor_gain=1.0)
    masses = np.asarray(env._model.body_mass)  # noqa: SLF001
    assert np.isclose(masses.max(), 3.0, atol=1e-3)


def test_slip_is_obs_noise_not_physics() -> None:
    env = _env()
    env.apply_domain_params(friction=1.0, slip=0.1, mass_kg=2.7, motor_gain=1.0)
    env.reset(seed=1)
    obs = env.step(np.asarray([5.0, 5.0], dtype=np.float32))[0]
    assert np.isfinite(obs["wheel_vel"]).all()  # noise applied, still finite
```

```python
# tests/unit/test_factory_rover_env_mujoco.py
"""Factory resolves backend='mujoco' to RoverMuJoCoEnv."""
from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import Settings
from mousedroid.factory import build_rover_env
from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv
from mousedroid.sim.protocols import RoverEnvProtocol


def test_mujoco_backend_builds_env() -> None:
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={"rover": cfg.rover.model_copy(update={"sim": cfg.rover.sim.model_copy(update={"backend": "mujoco"})})})
    env = build_rover_env(cfg)
    assert isinstance(env, RoverMuJoCoEnv)
    assert isinstance(env, RoverEnvProtocol)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/sim/test_mujoco_domain_params.py tests/unit/test_factory_rover_env_mujoco.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: FAIL — `apply_domain_params` missing; factory still raises `NotImplementedError`.

- [ ] **Step 3a: Implement `apply_domain_params`**

Add to `RoverMuJoCoEnv`:

```python
    def apply_domain_params(self, *, friction: float, slip: float, mass_kg: float, motor_gain: float) -> None:
        """Apply per-episode domain-randomization params to the live model.

        friction -> geom_friction[:,0] on wheel geoms; mass_kg -> chassis
        body_mass + inertia recompute; motor_gain -> actuator_gainprm[:,0];
        slip -> observation-noise magnitude (documented proxy; MuJoCo has no
        first-class slip). Reload-free: edits mjModel arrays in place.
        """
        for name in ("g_fl", "g_fr", "g_rl", "g_rr"):
            gid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_GEOM, name)
            self._model.geom_friction[gid, 0] = friction
        bid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_BODY, "chassis")
        scale = mass_kg / max(float(self._model.body_mass[bid]), 1e-6)
        self._model.body_mass[bid] = mass_kg
        self._model.body_inertia[bid] *= scale  # keep inertia consistent with new mass
        for name in ("a_fl", "a_fr", "a_rl", "a_rr"):
            aid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_ACTUATOR, name)
            self._model.actuator_gainprm[aid, 0] = motor_gain
        self._slip_noise = max(0.0, slip)
        self._assert_rest_state_stable()
```

- [ ] **Step 3b: Implement the factory branch**

In `src/mousedroid/factory.py` `build_rover_env`, replace the `backend == "mujoco"` `NotImplementedError` branch with:

```python
    if backend == "mujoco":
        from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv  # concrete import inside factory

        return RoverMuJoCoEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
        )
```

- [ ] **Step 4: Run + verify pass**

Run: `python -m pytest tests/unit/sim/test_mujoco_domain_params.py tests/unit/test_factory_rover_env_mujoco.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: all pass.

- [ ] **Step 5: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/sim/mujoco_rover_env.py src/mousedroid/factory.py tests/unit/sim/test_mujoco_domain_params.py tests/unit/test_factory_rover_env_mujoco.py && python -m ruff format --check src/mousedroid/sim/mujoco_rover_env.py src/mousedroid/factory.py && python -m mypy --strict src/mousedroid/sim/mujoco_rover_env.py src/mousedroid/factory.py`
Expected: clean.

```bash
git add src/mousedroid/sim/mujoco_rover_env.py src/mousedroid/factory.py tests/unit/sim/test_mujoco_domain_params.py tests/unit/test_factory_rover_env_mujoco.py
git commit -m "feat(sim,factory): DR-param consumption + build_rover_env mujoco branch

friction/mass/motor_gain -> mjModel fields; slip -> documented obs-noise proxy.
Fills the reserved 'mujoco' factory slot.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Phase 5 smoke sanity

**Files:**
- Test: `tests/smoke/test_phase5_sanity.py`

- [ ] **Step 1: Write the smoke test**

```python
"""Sub-second import + module-presence smoke for Phase 5."""
from __future__ import annotations

import importlib


def test_sim_module_imports_without_mujoco() -> None:
    # importing the module must NOT require the mujoco engine (lazy import).
    mod = importlib.import_module("mousedroid.sim.mujoco_rover_env")
    assert hasattr(mod, "RoverMuJoCoEnv")


def test_factory_exposes_rssm_trainable() -> None:
    from mousedroid import factory

    assert hasattr(factory, "build_rssm_trainable")
```

- [ ] **Step 2: Run + verify pass + commit**

Run: `python -m pytest tests/smoke/test_phase5_sanity.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 2 passed (import works even without mujoco installed — proves lazy import).

```bash
git add tests/smoke/test_phase5_sanity.py
git commit -m "test(smoke): Phase 5 lazy-import + factory-surface sanity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# LAYER 3 — Glue: adapter, generator, pretrainer, orchestrator wiring

### Task 11: `RoverObsAdapter`

**Files:**
- Create: `src/mousedroid/training/rover_obs_adapter.py`
- Test: `tests/unit/training/test_rover_obs_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
"""RoverObsAdapter maps rover obs dict + info -> RSSM encoder tensors."""
from __future__ import annotations

import numpy as np

from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.training.rover_obs_adapter import RoverObsAdapter


def _obs() -> dict[str, np.ndarray]:
    return {
        "imu": np.zeros(6, dtype=np.float32),
        "chassis_pose": np.asarray([0.1, 0.2, 1.0, 0.0], dtype=np.float32),
        "wheel_vel": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "lidar": np.full(16, 0.5, dtype=np.float32),
    }


def test_motor_state_is_vx_vy_omega_battery() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    out = adp.adapt(_obs(), info={"vx_body_mps": 0.3, "omega_rads": 0.05})
    assert out["motor"].shape == (4,)
    assert out["motor"][1] == 0.0  # vy == 0 (skid-steer)
    assert out["motor"][3] == 12.0  # battery const


def test_vision_omitted_mask_has_vision_slot_zero() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    out = adp.adapt(_obs(), info={"vx_body_mps": 0.0, "omega_rads": 0.0})
    mask = out["valid_mask"]
    assert mask[SENSOR_SLOT_MAP["vision"]] == 0.0
    assert mask[SENSOR_SLOT_MAP["motor"]] == 1.0
    assert "vision" not in out


def test_lidar_and_range_passed_through() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    out = adp.adapt(_obs(), info={"vx_body_mps": 0.0, "omega_rads": 0.0})
    assert out["lidar"].shape == (16,)
    assert out["ultrasonic"].shape == (1,)  # min-forward range scalar
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/training/test_rover_obs_adapter.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the adapter**

Create `src/mousedroid/training/rover_obs_adapter.py`:

```python
"""Adapter: rover env obs dict (+ step info) -> RSSM encoder-input tensors."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import SENSOR_SLOT_MAP

_VISION_SLOT = SENSOR_SLOT_MAP["vision"]
_N_SLOTS = 5  # vision, ultrasonic, motor, audio, lidar


class RoverObsAdapter:
    """Map a rover observation dict to the modality vectors the RSSM encoder reads.

    Vision is OMITTED (pretraining RSSM is built with vision_dim=0); the mask's
    vision slot is set to 0. ``motor_state`` is synthesised as
    ``[vx, vy=0, omega, battery_v]`` from the env ``info`` (skid-steer ⇒ vy=0).
    Pure + deterministic — no hidden state.
    """

    def __init__(self, *, battery_v: float) -> None:
        self._battery_v = float(battery_v)

    def adapt(self, obs: dict[str, NDArray[np.float32]], info: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
        vx = float(info.get("vx_body_mps", 0.0))
        omega = float(info.get("omega_rads", 0.0))
        motor = np.asarray([vx, 0.0, omega, self._battery_v], dtype=np.float32)

        lidar = np.asarray(obs.get("lidar", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        # forward range = min normalised lidar (or 1.0 when no lidar).
        forward = float(lidar.min()) if lidar.size else 1.0
        ultrasonic = np.asarray([forward], dtype=np.float32)

        mask = np.ones(_N_SLOTS, dtype=np.float32)
        mask[_VISION_SLOT] = 0.0  # vision omitted

        out: dict[str, NDArray[np.float32]] = {
            "motor": motor,
            "ultrasonic": ultrasonic,
            "valid_mask": mask,
        }
        if lidar.size:
            out["lidar"] = lidar
        return out
```

- [ ] **Step 4: Run + verify pass + lint/type + commit**

Run: `python -m pytest tests/unit/training/test_rover_obs_adapter.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 3 passed.

Run: `python -m ruff check src/mousedroid/training/rover_obs_adapter.py tests/unit/training/test_rover_obs_adapter.py && python -m ruff format --check src/mousedroid/training/rover_obs_adapter.py tests/unit/training/test_rover_obs_adapter.py && python -m mypy --strict src/mousedroid/training/rover_obs_adapter.py`
Expected: clean.

```bash
git add src/mousedroid/training/rover_obs_adapter.py tests/unit/training/test_rover_obs_adapter.py
git commit -m "feat(training): RoverObsAdapter (rover obs -> RSSM encoder inputs, vision off)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: `EpisodeBatch` + `SimEpisodeGenerator`

**Files:**
- Create: `src/mousedroid/training/sim_episode_generator.py`
- Test: `tests/unit/training/test_sim_episode_generator.py`

- [ ] **Step 1: Write the failing test**

```python
"""SimEpisodeGenerator rolls deterministic episodes into batched RSSM tensors."""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import Settings
from mousedroid.factory import build_rover_env
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator


def _gen(n: int, t: int) -> SimEpisodeGenerator:
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={"rover": cfg.rover.model_copy(update={"sim": cfg.rover.sim.model_copy(update={"backend": "mujoco"})})})
    env = build_rover_env(cfg)
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    return SimEpisodeGenerator(env, adapter, n_episodes=n, seq_len=t, seed=0)


def test_batch_tensor_shapes() -> None:
    gen = _gen(n=2, t=5)
    batch = gen.generate()
    assert batch.motor.shape == (2, 5, 4)
    assert batch.action.shape[:2] == (2, 5)
    assert batch.valid_mask.shape == (2, 5, 5)


def test_deterministic_for_fixed_seed() -> None:
    b1 = _gen(2, 5).generate()
    b2 = _gen(2, 5).generate()
    assert np.allclose(b1.action, b2.action)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/training/test_sim_episode_generator.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement generator + `EpisodeBatch`**

Create `src/mousedroid/training/sim_episode_generator.py`:

```python
"""In-process sim episode generation -> batched tensors for RSSM pretraining."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.sim.protocols import RoverEnvProtocol
from mousedroid.training.rover_obs_adapter import RoverObsAdapter

_log = get_logger(__name__)


@dataclass(frozen=True)
class EpisodeBatch:
    """Batched ``(B, T, ...)`` tensors consumed by ``RSSM.train_sequence``.

    A new in-memory container — NOT ``MouseDroidExperienceRecord`` (that schema
    cannot hold the rover modalities). LMDB persistence is a deferred follow-on.
    """

    motor: Tensor
    ultrasonic: Tensor
    lidar: Tensor
    valid_mask: Tensor
    action: Tensor
    reward: Tensor


class SimEpisodeGenerator:
    """Roll N episodes of T steps under a smoothed-random policy; adapt + stack."""

    def __init__(
        self,
        env: RoverEnvProtocol,
        adapter: RoverObsAdapter,
        *,
        n_episodes: int,
        seq_len: int,
        seed: int,
    ) -> None:
        self._env = env
        self._adapter = adapter
        self._n = n_episodes
        self._t = seq_len
        self._rng = np.random.default_rng(seed)
        self._action_dim = env.action_dim

    def _sample_action(self, prev: np.ndarray) -> np.ndarray:
        # smoothed uniform-random wheel commands (Dreamer seed-episode policy).
        target = self._rng.uniform(-6.0, 6.0, size=self._action_dim).astype(np.float32)
        return (0.7 * prev + 0.3 * target).astype(np.float32)

    def generate(self) -> EpisodeBatch:
        motors, ultras, lidars, masks, actions, rewards = ([] for _ in range(6))
        for ep in range(self._n):
            obs, info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
            prev = np.zeros(self._action_dim, dtype=np.float32)
            em, eu, el, ek, ea, er = ([] for _ in range(6))
            for _ in range(self._t):
                adapted = self._adapter.adapt(obs, info)
                action = self._sample_action(prev)
                # pad 2-DoF wheel action to the RSSM's 3-DoF [vx, vy=0, omega] space.
                padded = np.asarray([float(action[0]), 0.0, float(action[-1])], dtype=np.float32)
                em.append(adapted["motor"]); eu.append(adapted["ultrasonic"])
                el.append(adapted.get("lidar", np.zeros(0, dtype=np.float32)))
                ek.append(adapted["valid_mask"]); ea.append(padded)
                obs, reward, term, trunc, info = self._env.step(action)
                er.append(np.float32(reward))
                prev = action
                if term or trunc:
                    obs, info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
                    prev = np.zeros(self._action_dim, dtype=np.float32)
            motors.append(em); ultras.append(eu); lidars.append(el)
            masks.append(ek); actions.append(ea); rewards.append(er)
        _log.info("sim_episodes_generated", n_episodes=self._n, seq_len=self._t)

        def _stack(x: list[list[np.ndarray]]) -> Tensor:
            return torch.as_tensor(np.asarray(x, dtype=np.float32))

        return EpisodeBatch(
            motor=_stack(motors),
            ultrasonic=_stack(ultras),
            lidar=_stack(lidars),
            valid_mask=_stack(masks),
            action=_stack(actions),
            reward=torch.as_tensor(np.asarray(rewards, dtype=np.float32)),
        )
```

- [ ] **Step 4: Run + verify pass + lint/type + commit**

Run: `python -m pytest tests/unit/training/test_sim_episode_generator.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 2 passed.

Run: `python -m ruff check src/mousedroid/training/sim_episode_generator.py tests/unit/training/test_sim_episode_generator.py && python -m ruff format --check src/mousedroid/training/sim_episode_generator.py tests/unit/training/test_sim_episode_generator.py && python -m mypy --strict src/mousedroid/training/sim_episode_generator.py`
Expected: clean.

```bash
git add src/mousedroid/training/sim_episode_generator.py tests/unit/training/test_sim_episode_generator.py
git commit -m "feat(training): SimEpisodeGenerator + EpisodeBatch (in-memory, action padded to 3-DoF)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: `RSSMPretrainer`

**Files:**
- Create: `src/mousedroid/training/rssm_pretrainer.py`
- Test: `tests/unit/training/test_rssm_pretrainer.py`

- [ ] **Step 1: Write the failing test**

```python
"""RSSMPretrainer runs an Adam loop and writes a checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.training.rssm_pretrainer import RSSMPretrainer
from mousedroid.training.sim_episode_generator import EpisodeBatch
from mousedroid.world_model.rssm import RSSM


def _model() -> RSSM:
    return RSSM(ModelConfig(vision_dim=0, lidar_dim=16, lidar_proj_dim=32))  # type: ignore[arg-type]


def _batch(b: int = 3, t: int = 5) -> EpisodeBatch:
    return EpisodeBatch(
        motor=torch.randn(b, t, 4),
        ultrasonic=torch.rand(b, t, 1),
        lidar=torch.rand(b, t, 16),
        valid_mask=torch.ones(b, t, 5),
        action=torch.randn(b, t, 3),
        reward=torch.randn(b, t),
    )


def test_train_reduces_loss_and_writes_checkpoint(tmp_path: Path) -> None:
    model = _model()
    trainer = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu"))
    history = trainer.train([_batch()], epochs=15, checkpoint_path=tmp_path / "rssm.pt")
    assert history[-1] < history[0]
    assert (tmp_path / "rssm.pt").exists()


def test_checkpoint_is_loadable(tmp_path: Path) -> None:
    model = _model()
    trainer = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu"))
    trainer.train([_batch()], epochs=2, checkpoint_path=tmp_path / "rssm.pt")
    # weights_only=True: the checkpoint is a pure state_dict (tensors) — never
    # unpickle arbitrary objects from a model file.
    state = torch.load(tmp_path / "rssm.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # round-trips
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/training/test_rssm_pretrainer.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the pretrainer**

Create `src/mousedroid/training/rssm_pretrainer.py`:

```python
"""Adam pretraining loop for the RSSM dynamics core over sim episode batches."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mousedroid.logging.setup import get_logger
from mousedroid.training.sim_episode_generator import EpisodeBatch

_log = get_logger(__name__)


class RSSMPretrainer:
    """Owns the optimizer + epoch loop for ``RSSM.train_sequence``.

    AMP keeps the forward in mixed precision while the KL stays fp32 (handled
    inside ``train_sequence``). The loop is synchronous; the orchestrator runs
    it inside ``asyncio.to_thread`` so the event loop / thermal pause is not
    starved.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        grad_clip: float,
        amp: bool,
        device: torch.device,
    ) -> None:
        self._model = model.to(device)
        self._opt = torch.optim.Adam(model.parameters(), lr=lr)
        self._grad_clip = grad_clip
        self._amp = amp and device.type == "cuda"
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._amp)
        self._device = device

    def _to_device(self, batch: EpisodeBatch) -> dict[str, torch.Tensor]:
        return {
            "motor": batch.motor.to(self._device),
            "ultrasonic": batch.ultrasonic.to(self._device),
            "lidar": batch.lidar.to(self._device),
            "valid_mask": batch.valid_mask.to(self._device),
            "action": batch.action.to(self._device),
        }

    def train(self, batches: list[EpisodeBatch], *, epochs: int, checkpoint_path: Path) -> list[float]:
        history: list[float] = []
        self._model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in batches:
                tensors = self._to_device(batch)
                self._opt.zero_grad()
                with torch.autocast(device_type=self._device.type, enabled=self._amp):
                    out = self._model.train_sequence(tensors)
                loss = out["loss"]
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self._opt)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
                self._scaler.step(self._opt)
                self._scaler.update()
                epoch_loss += float(loss)
            mean = epoch_loss / max(1, len(batches))
            history.append(mean)
            _log.info(
                "rssm_pretrain_epoch",
                epoch=epoch,
                loss=mean,
                recon=float(out["recon"]),
                kl=float(out["kl"]),
                posterior_std=float(out["posterior_std"]),
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), checkpoint_path)
        _log.info("rssm_pretrain_checkpoint_written", path=str(checkpoint_path))
        return history
```

- [ ] **Step 4: Run + verify pass + lint/type + commit**

Run: `python -m pytest tests/unit/training/test_rssm_pretrainer.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 2 passed.

Run: `python -m ruff check src/mousedroid/training/rssm_pretrainer.py tests/unit/training/test_rssm_pretrainer.py && python -m ruff format --check src/mousedroid/training/rssm_pretrainer.py tests/unit/training/test_rssm_pretrainer.py && python -m mypy --strict src/mousedroid/training/rssm_pretrainer.py`
Expected: clean.

```bash
git add src/mousedroid/training/rssm_pretrainer.py tests/unit/training/test_rssm_pretrainer.py
git commit -m "feat(training): RSSMPretrainer (Adam loop, AMP, grad-clip, checkpoint)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: Wire `_train_rssm` in the orchestrator

**Files:**
- Modify: `src/mousedroid/training/pipeline_orchestrator.py`
- Test: `tests/unit/training/test_orchestrator_train_rssm.py`

- [ ] **Step 1: Write the failing test**

```python
"""_train_rssm is inert by default and runs the pretrainer when opted in."""
from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator


def _orch(settings: Settings) -> PipelineOrchestrator:
    # Construct via the same path the factory uses; see existing orchestrator tests
    # for the canonical builder. Here we assume a build_pipeline_orchestrator helper.
    from mousedroid.factory import build_pipeline_orchestrator

    return build_pipeline_orchestrator(settings)


@pytest.mark.asyncio
async def test_train_rssm_inert_when_disabled() -> None:
    cfg = Settings(mock_hardware=True)  # rssm_pretrain_enabled defaults False
    orch = _orch(cfg)
    await orch._train_rssm(batch_size=4)  # noqa: SLF001 — exercising the phase runner
    # inert: no checkpoint written
    assert not (cfg.training.weights_dir and __import__("pathlib").Path(cfg.training.weights_dir, cfg.training.rssm_checkpoint_name).exists())


@pytest.mark.asyncio
async def test_train_rssm_runs_when_enabled_and_mujoco(tmp_path) -> None:
    pytest.importorskip("mujoco")
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={
        "rover": cfg.rover.model_copy(update={"sim": cfg.rover.sim.model_copy(update={"backend": "mujoco"})}),
        "training": cfg.training.model_copy(update={
            "rssm_pretrain_enabled": True, "n_episodes": 2, "sequence_length": 4,
            "epochs": 2, "weights_dir": str(tmp_path),
        }),
    })
    orch = _orch(cfg)
    await orch._train_rssm(batch_size=2)  # noqa: SLF001
    assert (tmp_path / cfg.training.rssm_checkpoint_name).exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/training/test_orchestrator_train_rssm.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: FAIL — `_train_rssm` is the stub; no checkpoint behaviour. (If `build_pipeline_orchestrator` differs, adjust the builder import to the existing one — check `factory.py` for the real constructor name during Step 1.)

- [ ] **Step 3: Implement the wiring**

Replace the stub `_train_rssm` in `src/mousedroid/training/pipeline_orchestrator.py`:

```python
    async def _train_rssm(self, batch_size: int) -> None:
        """Run RSSM dynamics pretraining on MuJoCo-generated episodes.

        Inert (byte-identical to the prior stub) unless
        ``training.rssm_pretrain_enabled`` is True AND the rover backend is
        ``mujoco``. The synchronous torch loop runs in a worker thread so the
        orchestrator event loop (and the cooperative thermal-pause check) is
        not blocked.
        """
        tcfg = self._settings.training
        if not tcfg.rssm_pretrain_enabled:
            logger.info("rssm_training_skipped", reason="pretrain_disabled")
            return
        if self._settings.rover.sim.backend != "mujoco":
            logger.info("rssm_training_skipped", reason="non_mujoco_backend")
            return

        import torch  # local import keeps cold-start light

        from mousedroid.factory import build_rover_env, build_rssm_trainable
        from mousedroid.training.rover_obs_adapter import RoverObsAdapter
        from mousedroid.training.rssm_pretrainer import RSSMPretrainer
        from mousedroid.training.sim_episode_generator import SimEpisodeGenerator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_rssm_trainable(self._settings)
        env = build_rover_env(self._settings)
        adapter = RoverObsAdapter(battery_v=self._settings.rover.sim.mujoco.battery_voltage_const_v)
        generator = SimEpisodeGenerator(
            env, adapter, n_episodes=tcfg.n_episodes, seq_len=tcfg.sequence_length, seed=0
        )
        checkpoint = Path(tcfg.weights_dir) / tcfg.rssm_checkpoint_name

        def _run() -> list[float]:
            batch = generator.generate()
            trainer = RSSMPretrainer(
                model, lr=tcfg.learning_rate, grad_clip=tcfg.rssm_grad_clip,
                amp=self._config.use_amp, device=device,
            )
            return trainer.train([batch], epochs=tcfg.epochs, checkpoint_path=checkpoint)

        logger.info("rssm_training_started", n_episodes=tcfg.n_episodes, device=str(device))
        history = await asyncio.to_thread(_run)
        env.close()
        logger.info("rssm_training_done", first_loss=history[0], last_loss=history[-1])
```

Ensure `from pathlib import Path` and `import asyncio` are imported at the top of `pipeline_orchestrator.py` (asyncio already is; add `Path` if missing). Confirm the AMP flag name (`self._config.use_amp`) against the actual `TrainingPipelineConfig` field during Step 1 — adjust if the field is named differently.

- [ ] **Step 4: Run + verify pass**

Run: `python -m pytest tests/unit/training/test_orchestrator_train_rssm.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: 2 passed.

- [ ] **Step 5: Lint/format/type + commit**

Run: `python -m ruff check src/mousedroid/training/pipeline_orchestrator.py tests/unit/training/test_orchestrator_train_rssm.py && python -m ruff format --check src/mousedroid/training/pipeline_orchestrator.py tests/unit/training/test_orchestrator_train_rssm.py && python -m mypy --strict src/mousedroid/training/pipeline_orchestrator.py`
Expected: clean.

```bash
git add src/mousedroid/training/pipeline_orchestrator.py tests/unit/training/test_orchestrator_train_rssm.py
git commit -m "feat(training): wire _train_rssm to MuJoCo->RSSM pretraining (asyncio.to_thread)

Inert by default; runs only when rssm_pretrain_enabled AND backend==mujoco.
The blocking torch loop runs in a worker thread so the thermal-pause safety
check is not starved.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 15: End-to-end integration + golden + full CI gate

**Files:**
- Create: `tests/integration/test_phase5_rssm_pretrain.py`
- Create: `tests/regression/test_phase5_rssm_golden.py`

- [ ] **Step 1: Write the integration test (env → adapter → generator → pretrainer → checkpoint)**

```python
"""End-to-end: MuJoCo env -> episodes -> RSSM pretrain -> checkpoint."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import Settings
from mousedroid.factory import build_rover_env, build_rssm_trainable
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.rssm_pretrainer import RSSMPretrainer
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator


def test_end_to_end_pretrain_round_trip(tmp_path: Path) -> None:
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={"rover": cfg.rover.model_copy(update={"sim": cfg.rover.sim.model_copy(update={"backend": "mujoco"})})})
    env = build_rover_env(cfg)
    model = build_rssm_trainable(cfg)
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    gen = SimEpisodeGenerator(env, adapter, n_episodes=4, seq_len=6, seed=0)
    batch = gen.generate()
    trainer = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu"))
    history = trainer.train([batch], epochs=10, checkpoint_path=tmp_path / "rssm.pt")
    env.close()
    assert history[-1] < history[0]
    assert (tmp_path / "rssm.pt").exists()
```

- [ ] **Step 2: Write the golden test (non-gating, tolerance-based, CPU-deterministic)**

```python
"""Golden RSSM pretrain loss — monotone-ish decrease + final threshold (non-gating)."""
from __future__ import annotations

import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import Settings
from mousedroid.factory import build_rover_env, build_rssm_trainable
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.rssm_pretrainer import RSSMPretrainer
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator


def test_loss_decreases_deterministically() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={"rover": cfg.rover.model_copy(update={"sim": cfg.rover.sim.model_copy(update={"backend": "mujoco"})})})
    env = build_rover_env(cfg)
    model = build_rssm_trainable(cfg)
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    batch = SimEpisodeGenerator(env, adapter, n_episodes=4, seq_len=8, seed=0).generate()
    history = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu")).train(
        [batch], epochs=20, checkpoint_path=__import__("pathlib").Path("/tmp/_golden_rssm.pt")
    )
    env.close()
    # Tolerance-based, NOT point-wise ±1% (cross-platform float / MuJoCo drift).
    assert history[-1] < history[0] * 0.95  # at least 5% reduction
    assert history[-1] < 10.0  # sane absolute ceiling
```

- [ ] **Step 3: Run the new tests**

Run: `python -m pytest tests/integration/test_phase5_rssm_pretrain.py tests/regression/test_phase5_rssm_golden.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: pass (or skip if mujoco absent).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_phase5_rssm_pretrain.py tests/regression/test_phase5_rssm_golden.py
git commit -m "test(phase5): end-to-end pretrain round-trip + non-gating golden loss

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Full local CI gate (the finish bar)**

Install mujoco into the worktree's env if absent (so the gated tests actually run locally): `python -m pip install "mujoco>=3.0"`.

Run the full pipeline (matches `scripts/ci.sh`):

```bash
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
python -m mypy --strict src/mousedroid/
python -m pytest tests/ --import-mode=importlib --cov=src/mousedroid --cov-fail-under=85 -q
```

Expected: ruff clean, format clean, mypy clean, **all tests pass**, coverage ≥ 85%.
If coverage on the new modules is below the bar, add focused unit tests for the
uncovered branches (e.g. `body_velocity` action mode in `RoverMuJoCoEnv`,
lidar-absent path in `RoverObsAdapter`) before declaring done.

- [ ] **Step 6: Final commit if any coverage top-ups were needed**

```bash
git add tests/
git commit -m "test(phase5): coverage top-ups to clear the 85% gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Definition of done

- [ ] All 15 tasks committed.
- [ ] `ruff check` + `ruff format --check` + `mypy --strict` clean across `src/` + new tests.
- [ ] Full `pytest tests/` green with coverage ≥ 85%.
- [ ] Default-config behaviour byte-identical (encoder vision on; `mock` backend
      unchanged; pretrain disabled) — pinned by `tests/regression/test_phase5_backwards_compat.py`.
- [ ] `build_rover_env(backend="mujoco")` returns a working `RoverMuJoCoEnv`; the
      reserved `NotImplementedError` slot is gone.
- [ ] No hardcoded physics/training values — all via `MujocoSimConfig` / `TrainingConfig`.
- [ ] Then invoke **superpowers:finishing-a-development-branch**.

