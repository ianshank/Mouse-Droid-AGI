"""Regression: cap inline type:ignore / noqa debt in src/.

Update the budgets DOWN as the purge lands; never up without justification.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"
# Budgets = the MEASURED post-purge residual of load-bearing suppressions
# (untyped 3rd-party boundaries, Pydantic A00x field shadows) PLUS the handful
# of prose mentions of the literal tokens inside explanatory comments/docstrings.
# Expected NON-ZERO. May only ratchet DOWN, never up without a documented reason.
#
# The ignore budget (8) = 7 live torch untyped-call directives
#   (Tensor.backward x5 — growth/distillation.py, learning/ewc.py,
#   learning/offline_rl.py, meta/maml.py, training/rssm_pretrainer.py;
#   torch.jit.trace/.load x2 — efficiency/tensorrt.py) + 1 prose mention
#   (voice/greeting.py docstring documenting a *removed* private-access ignore).
#   (The torch.amp.GradScaler attr-defined ignore was eliminated by importing
#   from torch.amp.grad_scaler directly — runtime-valid AND mypy-clean.)
# The lint-waiver budget (19) = 18 live directives (torch.nn.functional N812 x4
#   — curiosity/icm.py, growth/distillation.py, learning/offline_rl.py,
#   scaling/moe.py; InjectionRejected N818 — security/injection_filter.py;
#   non-crypto random S311 x3 — resilience/retry.py, voice/rocky.py,
#   telemetry/mock_source.py; fixed-http urllib S310 x4 — comms/wifi_driver.py;
#   Pydantic Field default paths S108/S104 x3 — config/schema.py;
#   Argus socket S108 — validation/runtime.py; watchdog systemd-notify
#   S603/S607 — health/watchdog.py) + 1 prose mention
#   (common/imports.py docstring).
#   (The 10 stale BLE001/RUF100 waivers in training/observability/
#   mlflow_logger.py were removed — BLE001 is not enabled in the ruff config,
#   so the directive was entirely dead.)
_MAX_TYPE_IGNORE = 8
_MAX_NOQA = 19


def _count(token: str) -> int:
    return sum(
        line.count(token)
        for p in _SRC.rglob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
    )


def test_type_ignore_within_budget() -> None:
    assert _count("type: ignore") <= _MAX_TYPE_IGNORE


def test_noqa_within_budget() -> None:
    assert _count("noqa") <= _MAX_NOQA
