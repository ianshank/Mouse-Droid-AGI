"""Hardware tests for GPU training operations on Jetson.

Exercises real CUDA forward/backward passes, AMP training, checkpoint
round-trips, GPU monitoring, and batch tuning against actual device VRAM.

Run on Jetson::

    pytest -m hardware -v --timeout=60 tests/hardware/test_training_on_gpu.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mousedroid.config.schema import Settings, TrainingPipelineConfig

pytestmark = pytest.mark.hardware

JETSON_PROD_CONFIG = os.getenv("MOUSEDROID_JETSON_CONFIG", "config/jetson_production.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Load Settings from jetson_production.yaml."""
    import yaml

    with open(JETSON_PROD_CONFIG) as fh:
        raw = yaml.safe_load(fh)
    return Settings(**raw)


@pytest.fixture
def pipeline_config(tmp_path: Path) -> TrainingPipelineConfig:
    """Create pipeline config with temp checkpoint dir."""
    return TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        thermal_pause_seconds=0.01,
    )


# ---------------------------------------------------------------------------
# 1. RSSM single AMP training step (forward, loss, backward)
# ---------------------------------------------------------------------------


def test_rssm_amp_training_step(settings: Settings) -> None:
    """Run a single AMP forward + backward pass on the RSSM model on GPU."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA required for training test"

    from mousedroid.world_model.rssm import RSSM

    model_cfg = settings.model
    device = torch.device("cuda")

    model = RSSM(
        obs_dim=model_cfg.obs_dim,
        action_dim=model_cfg.action_dim,
        hidden_dim=model_cfg.hidden_dim,
        latent_dim=model_cfg.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.training.learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    batch_size = 4
    seq_len = settings.training.sequence_length
    obs = torch.randn(batch_size, seq_len, model_cfg.obs_dim, device=device)
    actions = torch.randn(batch_size, seq_len, model_cfg.action_dim, device=device)

    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        result = model(obs, actions)
        # RSSM returns a dict or named tuple with reconstruction and KL losses
        if isinstance(result, dict):
            loss = result.get("loss", result.get("recon_loss", torch.tensor(0.0)))
        else:
            loss = result.loss if hasattr(result, "loss") else result[0]

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert loss.item() > 0.0 or loss.item() == 0.0, "Loss should be a finite number"
    assert not torch.isnan(loss), "Loss is NaN after AMP training step"
    assert not torch.isinf(loss), "Loss is Inf after AMP training step"


# ---------------------------------------------------------------------------
# 2. Checkpoint save/load round-trip with CUDA tensors
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_roundtrip(settings: Settings, tmp_path: Path) -> None:
    """Save a model checkpoint and reload it, verifying weight equality."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA required"

    from mousedroid.world_model.rssm import RSSM

    model_cfg = settings.model
    device = torch.device("cuda")

    model = RSSM(
        obs_dim=model_cfg.obs_dim,
        action_dim=model_cfg.action_dim,
        hidden_dim=model_cfg.hidden_dim,
        latent_dim=model_cfg.latent_dim,
    ).to(device)

    checkpoint_path = tmp_path / "rssm_test.pt"
    torch.save(model.state_dict(), checkpoint_path)
    assert checkpoint_path.exists(), "Checkpoint file not created"
    assert checkpoint_path.stat().st_size > 0, "Checkpoint file is empty"

    # Reload into a fresh model
    model2 = RSSM(
        obs_dim=model_cfg.obs_dim,
        action_dim=model_cfg.action_dim,
        hidden_dim=model_cfg.hidden_dim,
        latent_dim=model_cfg.latent_dim,
    ).to(device)
    model2.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # Verify weights match
    for (name1, p1), (_, p2) in zip(
        model.named_parameters(), model2.named_parameters(), strict=True
    ):
        assert torch.equal(p1, p2), f"Weight mismatch in {name1}"


# ---------------------------------------------------------------------------
# 3. GPU monitor reads real temperature
# ---------------------------------------------------------------------------


def test_gpu_monitor_reads_temperature(settings: Settings) -> None:
    """GPU monitor reads a plausible temperature from the real sysfs path."""
    import asyncio

    thermal_path = settings.jetson.thermal_zone_path
    if not thermal_path.exists():
        pytest.skip(f"Thermal zone {thermal_path} not found on this device")

    pipeline_cfg = TrainingPipelineConfig(
        thermal_sysfs_path=str(thermal_path),
    )

    from mousedroid.training.gpu_monitor import JetsonGPUMonitor

    monitor = JetsonGPUMonitor(pipeline_cfg)
    temp_c = asyncio.run(monitor.get_temperature())

    # Temperature should be plausible: between 10 C and critical threshold
    critical_c = settings.health.gpu_temp_critical_c
    assert 10.0 <= temp_c <= critical_c + 20.0, (
        f"GPU temperature {temp_c:.1f} C is outside plausible range [10, {critical_c + 20.0}]"
    )


# ---------------------------------------------------------------------------
# 4. Batch tuner returns valid size for device VRAM
# ---------------------------------------------------------------------------


def test_batch_tuner_returns_valid_size(settings: Settings) -> None:
    """Batch tuner returns a positive batch size that fits in device VRAM."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA required"

    pipeline_cfg = TrainingPipelineConfig()

    from mousedroid.training.batch_tuner import VRAMBatchTuner

    tuner = VRAMBatchTuner(pipeline_cfg)
    base_size = settings.training.batch_size

    tuned = tuner.tune_batch_size("rssm", base_size=base_size)
    assert isinstance(tuned, int), f"Expected int, got {type(tuned)}"
    assert tuned >= 1, f"Tuned batch size must be >= 1, got {tuned}"
    assert tuned <= base_size * 2, f"Tuned batch size {tuned} unexpectedly large (base={base_size})"


# ---------------------------------------------------------------------------
# 5. GPU memory fraction respected
# ---------------------------------------------------------------------------


def test_gpu_memory_fraction_reasonable(settings: Settings) -> None:
    """Verify configured GPU memory fraction leaves headroom."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA required"

    fraction = settings.jetson.gpu_memory_fraction
    assert 0.0 < fraction <= 1.0, f"Invalid GPU memory fraction: {fraction}"

    total_mem = torch.cuda.get_device_properties(0).total_mem
    expected_usable = total_mem * fraction
    # Verify we can actually allocate a tensor within that budget
    alloc_bytes = int(expected_usable * 0.1)  # 10% of budget
    n_floats = alloc_bytes // 4
    t = torch.empty(n_floats, dtype=torch.float32, device="cuda")
    assert t.device.type == "cuda"
    del t
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 6. AMP scaler state persistence
# ---------------------------------------------------------------------------


def test_amp_scaler_state_roundtrip(tmp_path: Path) -> None:
    """GradScaler state can be saved and loaded for training resumption."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA required"

    scaler = torch.amp.GradScaler("cuda")
    # Simulate a few scale updates
    model = torch.nn.Linear(8, 4).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = model(torch.randn(2, 8, device="cuda")).sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scaler_path = tmp_path / "scaler.pt"
    torch.save(scaler.state_dict(), scaler_path)
    assert scaler_path.exists()

    scaler2 = torch.amp.GradScaler("cuda")
    scaler2.load_state_dict(torch.load(scaler_path, map_location="cuda", weights_only=True))
    assert scaler2.get_scale() == scaler.get_scale(), "Scaler scale mismatch after load"


# ---------------------------------------------------------------------------
# 7. VRAM reporting matches device
# ---------------------------------------------------------------------------


def test_vram_reporting_consistent() -> None:
    """torch.cuda.mem_get_info returns values consistent with device properties."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA required"

    free, total = torch.cuda.mem_get_info()
    props_total = torch.cuda.get_device_properties(0).total_mem

    # Total from mem_get_info and device properties should be close
    assert abs(total - props_total) / props_total < 0.05, (
        f"VRAM total mismatch: mem_get_info={total}, properties={props_total}"
    )
    assert free > 0, "No free VRAM available"
    assert free <= total, "Free VRAM exceeds total"
