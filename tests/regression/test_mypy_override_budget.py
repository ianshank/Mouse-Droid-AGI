"""Regression: ratchet the mypy ``disallow_subclassing_any`` override list.

pyproject.toml carries a ``[[tool.mypy.overrides]]`` block that relaxes
``disallow_subclassing_any`` for torch/numpy-heavy modules (real stub gaps:
``nn.Module`` resolves to ``Any`` without torch stubs). Unlike the inline
``type: ignore`` / ``noqa`` budgets (tests/regression/test_suppression_budget.py)
and the C901 gate (test_complexity_gate.py), this list previously had NO
ratchet — it structurally grows by one entry per new ML module with nothing
asking whether the relaxation is still needed.

Ratchet-down-only: removing a module from the pin is free; adding one means
extending ``_ALLOWED_SUBCLASS_ANY_MODULES`` in the same PR with the usual
documented-reason bar.
"""

from __future__ import annotations

from tests._pyproject import load_pyproject

# The measured override list at ratchet time (code-hygiene sprint).
_ALLOWED_SUBCLASS_ANY_MODULES = frozenset(
    {
        "mousedroid.config.schema",
        "mousedroid.llm_gateway.config",
        "mousedroid.world_model.encoder",
        "mousedroid.world_model.rssm",
        "mousedroid.world_model.mcts",
        "mousedroid.world_model.cfc_cell",
        "mousedroid.world_model.dual_stream_rssm",
        "mousedroid.world_model.stream_fusion",
        "mousedroid.scaling.moe",
        "mousedroid.scaling.adaptive",
        "mousedroid.reward.model",
        "mousedroid.learning.progressive",
        "mousedroid.curiosity.icm",
        "mousedroid.growth.distillation",
        "mousedroid.arm.control.sac_agent",
        "mousedroid.arm.perception.object_detector",
        "mousedroid.arm.perception.depth_processor",
        "mousedroid.arm.perception.hailo_detector",
        "mousedroid.hardware.accelerator.hailo_runtime",
        "mousedroid.cloud.pubsub_sink",
        "mousedroid.cloud.experience_exporter",
        "mousedroid.cloud.logging_sink",
        "mousedroid.cloud.monitoring_exporter",
        "mousedroid.cloud.firestore_sync",
        "mousedroid.cloud._auth",
    }
)


def _subclass_any_modules() -> set[str]:
    """Collect module names from overrides that relax disallow_subclassing_any."""
    mypy_cfg = load_pyproject()["tool"]["mypy"]  # type: ignore[index]
    modules: set[str] = set()
    for override in mypy_cfg.get("overrides", []):
        if override.get("disallow_subclassing_any") is False:
            listed = override.get("module", [])
            modules.update([listed] if isinstance(listed, str) else listed)
    return modules


def test_subclass_any_override_list_within_budget() -> None:
    actual = _subclass_any_modules()
    new_entries = actual - _ALLOWED_SUBCLASS_ANY_MODULES
    assert not new_entries, (
        "New modules relax mypy disallow_subclassing_any — extend "
        "_ALLOWED_SUBCLASS_ANY_MODULES consciously (is the stub gap real?) "
        f"or fix the typing instead: {sorted(new_entries)}"
    )
