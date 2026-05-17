"""F-006 regression: MOUSEDROID_LLM__N_GPU_LAYERS env override threads through to LLMConfig.

The Jetson production overlay (`config/jetson_production.yaml`) used to pin
``llm.n_gpu_layers: 0`` (CPU-only inference). On the live Jetson, Phi-3-mini-q4
ran at 0.5 tok/s under that setting (260 s per ``translate_mission``), blowing
the ``latency_target_ms`` budget. The fix flips the overlay to ``-1`` (offload
every layer to the iGPU). These tests pin the operator-side escape hatch:
``MOUSEDROID_LLM__N_GPU_LAYERS=<int>`` set in ``/etc/mousedroid/docker.env``
overrides whatever the YAML says, so an operator can fall back to CPU-only
inference on a host without flipping any committed file.

The env-nesting plumbing is pure Pydantic settings (``env_nested_delimiter="__"``
on the root ``Settings`` model). These tests assert the plumbing actually works
end-to-end through ``load_settings``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings


def test_env_override_n_gpu_layers_takes_precedence_over_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator can pin ``n_gpu_layers`` per-host without touching the overlay yaml.

    Loads the production overlay (which now sets ``n_gpu_layers: -1``) but
    forces the env var to ``8`` (a hand-picked mid-range value). The merged
    config must reflect the env, not the overlay.
    """
    monkeypatch.setenv("MOUSEDROID_LLM__N_GPU_LAYERS", "8")
    # Repo root is two parents up from this test file
    # (.../tests/unit/llm_gateway/test_*.py -> .../).
    repo_root = Path(__file__).resolve().parents[3]
    overlay = repo_root / "config" / "jetson_production.yaml"
    cfg = load_settings(overlay)
    assert cfg.llm.n_gpu_layers == 8


def test_no_env_override_falls_back_to_schema_default_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env var or any overlay, the schema default ``-1`` wins.

    Confirms the schema's documented default (``-1`` = offload-all) is what
    ``load_settings()`` returns when nothing else is in play — so a fresh
    deployment with no overlay still gets GPU inference, not CPU.
    """
    monkeypatch.delenv("MOUSEDROID_LLM__N_GPU_LAYERS", raising=False)
    cfg = load_settings()
    assert cfg.llm.n_gpu_layers == -1  # schema default per config/schema.py:620


def test_env_override_zero_is_valid_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``=0`` is the legitimate CPU-only fallback for hosts without an iGPU.

    Pins that the schema doesn't reject the zero case (which would block the
    very escape hatch this regression test set exists to protect).
    """
    monkeypatch.setenv("MOUSEDROID_LLM__N_GPU_LAYERS", "0")
    cfg = load_settings()
    assert cfg.llm.n_gpu_layers == 0
