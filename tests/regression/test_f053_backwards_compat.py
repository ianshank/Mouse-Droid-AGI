"""Backwards compatibility — F-053 must not weaken the eight original contracts.

Phases 2-4 moved prose and retired ``agent.md``, but the risk is quietly
softening an invariant while the Surface Map grows. This half of the regression
pair asserts:

* all **17** in-package ``CLAUDE.md`` files still exist and are indexed from the
  root Surface Map;
* each of the **eight** contracts that pre-dated F-053 Phase 2 still carries its
  purpose blockquote and the invariant titles that were live before the rewrite.

Discovered helpers (``tests/_claude_md.py``) keep the roster from going stale;
the purpose/invariant strings below are the load-bearing pins.
"""

from __future__ import annotations

from tests._claude_md import (
    CLAUDE_MD,
    discover_in_package_doc_packages,
    discover_nested_claude_md,
    repo_root,
    surface_map_indexes,
)

_REPO = repo_root()
_SRC = _REPO / "src" / "mousedroid"

# The eight subsystem contracts that existed before F-053 Phase 2 authored the
# nine that had only an agent.md. Titles and purpose lines are pinned, not
# discovered, so a rewrite that drops an invariant fails closed.
_ORIGINAL_CONTRACT_PINS: dict[str, tuple[str, tuple[str, ...]]] = {
    "orchestrator": (
        (
            "Sense-plan-act loop at 30 Hz (33.3 ms per tick). Coordinates sensor "
            "ingestion, latent world model planning, and actuator command generation."
        ),
        (
            "Strict 33.3 ms Cadence",
            "Emergency Stop (E-Stop)",
            "Telemetry Server & Shared Registry",
            "Safety Filter Projection",
            "Graceful Shutdown (S-1)",
        ),
    ),
    "llm_gateway": (
        (
            "Translation of operator natural language commands into structured "
            "`GoalVector` objectives across local and cloud backends."
        ),
        (
            "Protocol Conformance",
            "Never Raise on Backend Failure",
            "Explicit Task Cancellation",
            "Lazy SDK Imports",
            "Secret Protection",
            "Prompt Injection Sanitization",
            "Local-Only Failover",
        ),
    ),
    "hardware": (
        (
            "Physical sensor and actuator drivers for NVIDIA Jetson Orin Nano, "
            "CSI/USB cameras, LiDAR, IMU, ultrasonic, and ESP32 motor controller "
            "bridge."
        ),
        (
            "Factory DI Only",
            "Two-Level Hardware Gates",
            "USB-C Endpoint Discovery",
            "Ring Buffers",
            "Hardware Test Tier",
            "Robust Sysfs Reading",
        ),
    ),
    "telemetry": (
        (
            "REST endpoints, WebSocket live state streaming, and Prometheus metrics "
            "exposition on `/metrics` and `/api/v1/health`."
        ),
        (
            "Config-Gated Metric Families",
            "Pure-Add Render",
            "Keyword-Only Shared Registry",
            "Low-Cardinality Validated Labels",
            "Success-Path Recording Only",
            "Promtool Golden Samples",
        ),
    ),
    "learning": (
        (
            "Elastic Weight Consolidation (EWC), progressive neural networks, and "
            "experience replay preventing catastrophic forgetting across navigation "
            "domains."
        ),
        (
            "Inference with `torch.no_grad()`",
            "Deterministic Fisher Estimation",
            "Memory Footprint Bounds",
            "Isolated Task Boundaries",
        ),
    ),
    "growth": (
        ("Knowledge distillation and model compression (VLA teacher → compact student model)."),
        (
            "Default-OFF Off-Loop Execution",
            "Resource Throttling",
            "Loss Divergence Protection",
        ),
    ),
    "world_model": (
        (
            "Recurrent State Space Model (RSSM) latent dynamics and Monte Carlo Tree "
            "Search (MCTS) trajectory planning for autonomous navigation."
        ),
        (
            "Latent Space Planning",
            "ONNX / TensorRT Acceleration",
            "No In-Place Tensor Mutation",
            "`torch.no_grad()` on Rollouts",
        ),
    ),
    "arm": (
        (
            "Hierarchical robot-arm training platform (MuJoCo Gymnasium, SAC+HER "
            "reinforcement learning, PDDL symbolic replanner, and SO-ARM100 hardware "
            "driver)."
        ),
        (
            "Dual Cadence Planning",
            "Deterministic Simulation Seed",
            "Hardware Driver Isolation",
        ),
    ),
}


def test_all_seventeen_in_package_claude_md_still_load() -> None:
    nested = discover_nested_claude_md()
    assert len(nested) == 17, (
        f"expected 17 in-package CLAUDE.md, found {len(nested)}: "
        + ", ".join(sorted(p.parent.name for p in nested))
    )
    packages = discover_in_package_doc_packages()
    assert len(packages) == 17
    root = (_REPO / CLAUDE_MD).read_text(encoding="utf-8")
    missing = [pkg for pkg in packages if not surface_map_indexes(root, pkg)]
    assert not missing, "root Surface Map no longer indexes: " + ", ".join(missing)


def test_original_eight_contracts_keep_purpose_and_invariants() -> None:
    for pkg, (purpose, invariants) in _ORIGINAL_CONTRACT_PINS.items():
        path = _SRC / pkg / CLAUDE_MD
        assert path.is_file(), f"missing original contract {path}"
        text = path.read_text(encoding="utf-8")
        collapsed = " ".join(line[1:].strip() for line in text.splitlines() if line.startswith(">"))
        needle = purpose[:60]
        assert needle in collapsed, f"{pkg}/CLAUDE.md purpose no longer contains {needle!r}"
        for title in invariants:
            assert f"**{title}**" in text, f"{pkg}/CLAUDE.md lost invariant title {title!r}"


def test_original_eight_roster_is_exactly_the_pre_f053_set() -> None:
    assert set(_ORIGINAL_CONTRACT_PINS) == {
        "orchestrator",
        "llm_gateway",
        "hardware",
        "telemetry",
        "learning",
        "growth",
        "world_model",
        "arm",
    }
