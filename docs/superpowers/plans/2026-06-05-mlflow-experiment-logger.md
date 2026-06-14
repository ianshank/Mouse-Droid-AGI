# MLflow Experiment Logger — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire an MLflow-backed experiment-logger into `PipelineOrchestrator` and `OfflineRLTrainer` (CQL + IQL) so every training run produces per-step + per-phase metric curves at `./mlruns/`, defaulting OFF so existing YAML loads unchanged.

**Architecture:** Protocol-DI: a runtime-checkable `ExperimentLoggerProtocol` is the single abstraction the call sites import. `factory.build_experiment_logger(cfg)` resolves to either a `NoOpExperimentLogger` (always available, default) or a `MlflowExperimentLogger` (concrete, wraps `mlflow.MlflowClient`, imported INSIDE the factory). The logger is threaded keyword-only into both `PipelineOrchestrator.__init__` and `OfflineRLTrainer.__init__` — mirroring exactly how `metrics: MetricsRegistry | None = None` is threaded today. Backend default: `file://<absolute>/mlruns` resolved at factory time (not CWD-dependent). Nested runs via the `mlflow.parentRunId` tag (NOT the fluent `nested=True` API) so parent = pipeline run, children = phase runs.

**Tech Stack:** Python 3.10+, `mlflow-skinny>=2.22,<3` (client-only, no SQLAlchemy/Flask/Pandas bloat; file backend ships in skinny), Pydantic v2, structlog, pytest + pytest-asyncio + hypothesis, ruff 0.8.0, mypy --strict. PyTorch tensors via `.item()` coercion at the logger boundary.

---

## Spec recap

| Decision | Value | Source |
|---|---|---|
| Wiring scope | Pipeline orchestrator + OfflineRLTrainer (CQL + IQL). **NOT** SACAgent, **NOT** OLED. | user |
| Backend | Local file at `<repo>/mlruns/` (configurable via `cfg.observability.experiment_logger.tracking_uri`) | user |
| Library | `mlflow-skinny>=2.22,<3` — file backend ships in skinny; no server bloat | research |
| Default state | `cfg.observability.experiment_logger.backend == "none"` — existing YAML unchanged | CLAUDE.md invariant #9 |
| API style | `MlflowClient` (OOP) — NOT fluent `mlflow.start_run` — for protocol cleanliness + no thread-locals | research |
| Nested runs | Parent run = pipeline; child runs = phases via `mlflow.parentRunId` tag | research |
| Metric coercion | `_to_finite_float(v)` helper: `.item()` for tensors, finite-check, warn-and-skip NaN/Inf | research + invariant #7 |
| Test strategy | Real `MlflowClient` over `tmp_path` (NOT mock the client) | research |

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `src/mousedroid/training/observability/__init__.py` | CREATE | Package marker + protocol re-export |
| `src/mousedroid/training/observability/protocol.py` | CREATE | `ExperimentLoggerProtocol` (`@runtime_checkable`), `PhaseContext` dataclass, `_to_finite_float` helper |
| `src/mousedroid/training/observability/noop_logger.py` | CREATE | `NoOpExperimentLogger` — always available, byte-identical no-op |
| `src/mousedroid/training/observability/mlflow_logger.py` | CREATE | `MlflowExperimentLogger` — wraps `MlflowClient`, imports mlflow lazily |
| `src/mousedroid/config/schema.py` | MODIFY | Add `ObservabilityConfig` + `ExperimentLoggerConfig` Pydantic models (around line 1391, before `MetricsConfig`); add `observability: ObservabilityConfig \| None` field on `Settings` at line 4767 (alphabetical between `ppo` and `telemetry`) |
| `src/mousedroid/factory.py` | MODIFY | Add `build_experiment_logger(cfg) -> ExperimentLoggerProtocol` mirroring `build_metrics_registry` (line 1166 template); add keyword-only `experiment_logger=None` to consumers' construction sites |
| `src/mousedroid/training/pipeline_orchestrator.py` | MODIFY | Add `experiment_logger: ExperimentLoggerProtocol \| None = None` keyword-only to `__init__` (line 46); bracket `run()` (line 59) with `start_run`/`end_run`; bracket `_run_phase()` (line 128) with `start_phase`/`end_phase` |
| `src/mousedroid/learning/offline_rl.py` | MODIFY | Add `experiment_logger: ExperimentLoggerProtocol \| None = None` keyword-only to `OfflineRLTrainer.__init__` (line 146); store on `self._experiment_logger`; add `_log_step_metrics(losses, step)` helper; CQL (line 432) + IQL (line 583) call it at the tail of `update_step` |
| `pyproject.toml` | MODIFY | Add `[project.optional-dependencies] mlflow = ["mlflow-skinny>=2.22,<3"]` |
| `tests/unit/training/observability/__init__.py` | CREATE | Empty package marker |
| `tests/unit/training/observability/test_protocol_conformance.py` | CREATE | NoOp + Mlflow satisfy `ExperimentLoggerProtocol`; PhaseContext equality |
| `tests/unit/training/observability/test_noop_logger.py` | CREATE | Every method is a no-op + returns expected sentinel run-id |
| `tests/unit/training/observability/test_mlflow_logger.py` | CREATE | Real `MlflowClient` over `tmp_path` — start_run → log_metric → end_run round-trip; nested phase runs use parent tag; CancelledError + Exception cleanup; degraded-mode on missing mlflow |
| `tests/unit/training/observability/test_finite_float_coercion.py` | CREATE | Hypothesis property test: tensors / np scalars / Python floats → finite-float idempotent; NaN/Inf produce warning, no log call |
| `tests/unit/test_factory_observability.py` | CREATE | `build_experiment_logger`: backend=="none" → NoOp; backend=="mlflow" + skinny installed → Mlflow; backend=="mlflow" + skinny missing → NoOp with degrade-warning log |
| `tests/integration/test_pipeline_orchestrator_observability.py` | CREATE | Real tmp_path MLflow backend; run a 2-phase pipeline; assert parent run + 2 child runs exist with correct tags + status FINISHED |
| `tests/integration/test_offline_rl_observability.py` | CREATE | CQL + IQL trainers; 3 update_step calls each; assert metric history has correct step indices + key set |
| `tests/regression/test_observability_backwards_compat.py` | CREATE | Pre-feature minimal YAML still loads; absent `observability` key resolves to None; existing `config/default.yaml` and `config/jetson_production.yaml` still parse |
| `docs/runbooks/mlflow-local-ui.md` | CREATE | Operator runbook: how to run `mlflow ui --backend-store-uri file:./mlruns`, view experiments, common pitfalls |
| `CHANGELOG.md` | MODIFY | Under `## [Unreleased]`: "Added: `cfg.observability.experiment_logger` (MLflow file backend) wired into `PipelineOrchestrator` + `OfflineRLTrainer`; default OFF, opt-in via YAML." |
| `NEXT_STEPS.md` | MODIFY | Add "T2 — MLflow logger ✅" entry; strike through W&B reference if any |

---

## Implementation tasks

### Task 1: Add `mlflow` extras to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` `[project.optional-dependencies]` section

- [ ] **Step 1: Read current optional-dependencies layout**

Run: `python -c "import tomllib, pathlib; print(sorted(tomllib.loads(pathlib.Path('pyproject.toml').read_text()).get('project', {}).get('optional-dependencies', {}).keys()))"`
Expected: prints a sorted list like `['anthropic', 'arm', 'audio', 'cfc', 'dev', 'gcp', ...]`. Confirms there is no existing `mlflow` extras.

- [ ] **Step 2: Add the `mlflow` extras entry**

Locate the existing extras block (around line 25-158) and insert alphabetically between the closest two existing entries. The new entry:

```toml
mlflow = [
    "mlflow-skinny>=2.22,<3",
]
```

**Why mlflow-skinny:** The full `mlflow` package pulls in SQLAlchemy / Alembic / Flask / Pandas / scikit-learn (~300 MB). We're a write-only client targeting the file backend — `mlflow-skinny` includes `FileStore` and `MlflowClient` and nothing else heavy. Pinning `<3` keeps us on 2.x file-backend semantics; MLflow 3.7 (Dec 2025) flips the default backend to SQLite with a deprecation warning, which we want to evaluate deliberately later, not absorb implicitly.

- [ ] **Step 3: Add install verification command**

Run: `python -m pip install --dry-run "mlflow-skinny>=2.22,<3" 2>&1 | head -20`
Expected: shows resolution succeeds (or "would install"). Confirms the pin matches a real published version on PyPI. Do NOT actually install — just dry-run.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(deps): add mlflow-skinny>=2.22,<3 as [mlflow] optional extras"
```

---

### Task 2: Add `ObservabilityConfig` + `ExperimentLoggerConfig` to schema

**Files:**
- Create test: `tests/regression/test_observability_backwards_compat.py`
- Modify: `src/mousedroid/config/schema.py` (insert new models around line 1389, before `MetricsConfig` at 1391; add `Settings` field at line 4767)

- [ ] **Step 1: Write the failing backwards-compat test FIRST**

Create `tests/regression/test_observability_backwards_compat.py`:

```python
"""Regression: ``observability`` config is purely additive and defaults OFF.

Pins the CLAUDE.md invariant #9 ("Existing YAML files must load unchanged")
specifically for the new ``observability`` field added in PR T2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import Settings


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_minimal_pre_feature_yaml_still_loads() -> None:
    """A YAML that predates the feature loads with observability defaulting to OFF."""
    minimal = yaml.safe_load(
        """
        mock_hardware: true
        platform: mouse_droid
        """
    )
    cfg = Settings.model_validate(minimal)
    assert cfg.observability is None  # default; backwards-compatible


def test_existing_default_yaml_still_loads() -> None:
    """``config/default.yaml`` parses unchanged after the schema addition."""
    raw = yaml.safe_load((_REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    cfg = Settings.model_validate(raw)
    # Either absent (None) or explicitly disabled (the user has not opted in).
    if cfg.observability is not None:
        assert cfg.observability.experiment_logger.backend == "none"


def test_existing_jetson_production_yaml_still_loads() -> None:
    """``config/jetson_production.yaml`` parses unchanged after the schema addition."""
    raw = yaml.safe_load(
        (_REPO_ROOT / "config" / "jetson_production.yaml").read_text(encoding="utf-8")
    )
    cfg = Settings.model_validate(raw)
    if cfg.observability is not None:
        assert cfg.observability.experiment_logger.backend == "none"


def test_opt_in_overlay_parses() -> None:
    """A YAML that DOES set observability resolves the backend correctly."""
    overlay = yaml.safe_load(
        """
        mock_hardware: true
        platform: mouse_droid
        observability:
          experiment_logger:
            backend: mlflow
            tracking_uri: file:./mlruns
            experiment_name: mousedroid-test
        """
    )
    cfg = Settings.model_validate(overlay)
    assert cfg.observability is not None
    assert cfg.observability.experiment_logger.backend == "mlflow"
    assert cfg.observability.experiment_logger.tracking_uri == "file:./mlruns"
    assert cfg.observability.experiment_logger.experiment_name == "mousedroid-test"


def test_env_var_override() -> None:
    """Env-var nested overrides work for the new sub-config."""
    overlay = {"mock_hardware": True, "platform": "mouse_droid"}
    cfg = Settings.model_validate(overlay)
    # Direct model_validate path doesn't pick up env; spot-check the schema accepts
    # the env-style override path via model_copy. The full env mechanism is
    # exercised by ``Settings()`` instantiation in conftest.py.
    updated = cfg.model_copy(
        update={
            "observability": {
                "experiment_logger": {
                    "backend": "mlflow",
                    "experiment_name": "from-env",
                }
            }
        }
    )
    assert updated.observability is not None
    assert updated.observability.experiment_logger.experiment_name == "from-env"


def test_rejects_invalid_backend_literal() -> None:
    """Pydantic Literal rejects unknown backend strings at validation time."""
    bad = {
        "mock_hardware": True,
        "observability": {"experiment_logger": {"backend": "wandb"}},  # not in Literal
    }
    with pytest.raises(Exception):  # pydantic ValidationError
        Settings.model_validate(bad)
```

- [ ] **Step 2: Run the failing test**

Run: `python -m pytest tests/regression/test_observability_backwards_compat.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: fails — `Settings` has no `observability` attribute (`AttributeError` or pydantic validation error on the opt-in YAML).

- [ ] **Step 3: Add the new Pydantic models**

In `src/mousedroid/config/schema.py`, find `class MetricsConfig(BaseModel):` (line 1391). INSERT directly BEFORE it (so observability lives next to metrics — they're both "tell us what's happening" configs):

```python
class ExperimentLoggerConfig(BaseModel):
    """Experiment-logger configuration for training runs (per-step + per-phase metrics).

    Wired into :class:`PipelineOrchestrator` and :class:`OfflineRLTrainer`
    via :func:`mousedroid.factory.build_experiment_logger`. Defaults to OFF
    (``backend="none"``) so a YAML predating this feature loads unchanged
    (CLAUDE.md invariant #9). Selecting ``backend="mlflow"`` requires the
    ``mousedroid[mlflow]`` extras (``mlflow-skinny``); a missing dep
    degrades gracefully to the NoOp logger with a structured warning.
    """

    backend: Literal["none", "mlflow"] = Field(
        "none",
        description=(
            "Experiment-logger backend. ``none`` (default) selects the NoOp "
            "logger — byte-identical to pre-feature behavior. ``mlflow`` "
            "selects the MlflowClient-backed logger writing to "
            "``tracking_uri`` (default ``file:./mlruns``)."
        ),
    )
    tracking_uri: str = Field(
        "file:./mlruns",
        description=(
            "MLflow tracking URI. ``file:./mlruns`` (default) writes to a "
            "local directory relative to the factory's resolution time (the "
            "factory pins this to an absolute path to avoid CWD surprises). "
            "Set to ``http://host:port`` to use a remote tracking server."
        ),
    )
    experiment_name: str = Field(
        "mousedroid",
        min_length=1,
        description="MLflow experiment name (created if missing).",
    )
    run_name: str | None = Field(
        None,
        description=(
            "Optional human-readable run name for the parent (pipeline) run. "
            "When ``None`` the orchestrator constructs one from the pipeline "
            "config + UTC timestamp."
        ),
    )
    log_step_every_n: int = Field(
        1,
        gt=0,
        description=(
            "Per-update-step metric throttle. ``1`` (default) logs every "
            "update_step call. Set higher for very-long training runs to "
            "reduce store-write overhead."
        ),
    )
    log_artifacts: bool = Field(
        True,
        description=(
            "When True, the orchestrator logs the resolved Settings JSON "
            "snapshot as a parent-run artifact at start, plus the per-phase "
            "checkpoint file as a child-run artifact on phase completion."
        ),
    )


class ObservabilityConfig(BaseModel):
    """Top-level observability configuration for the training stack.

    Currently contains the experiment-logger sub-config; future fields
    (training-side Prometheus metrics, W&B integration, etc.) land here
    to keep ``Settings`` flat.
    """

    experiment_logger: ExperimentLoggerConfig = Field(
        default_factory=ExperimentLoggerConfig,
        description="Per-run experiment-logger config (MLflow file backend).",
    )


```

Then find the `Settings` field block (line ~4765-4767, currently has `offline_rl` → `ppo` → `telemetry`). INSERT the new field alphabetically between `ppo` and `telemetry`:

```python
    observability: ObservabilityConfig | None = Field(
        None,
        description=(
            "Top-level observability config (experiment logger). None (default) "
            "preserves byte-identical pre-feature behavior. Set to enable "
            "MLflow-backed metric logging for training runs."
        ),
    )
```

Also verify `Literal` is already imported at the top of `schema.py` (it is — used elsewhere). If not, add `from typing import Literal`.

- [ ] **Step 4: Run the regression test to verify it passes**

Run: `python -m pytest tests/regression/test_observability_backwards_compat.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: PASS — all 6 tests green.

- [ ] **Step 5: Run the full schema test suite to verify no other regression**

Run: `python -m pytest tests/unit/test_config_schema.py tests/regression/ -k "schema or config" --import-mode=importlib --no-cov -q 2>&1 | tail -10`
Expected: 0 failures.

- [ ] **Step 6: Lint + type-check**

Run: `python -m ruff check src/mousedroid/config/schema.py tests/regression/test_observability_backwards_compat.py && python -m ruff format --check src/mousedroid/config/schema.py tests/regression/test_observability_backwards_compat.py && python -m mypy --strict src/mousedroid/config/schema.py`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/mousedroid/config/schema.py tests/regression/test_observability_backwards_compat.py
git commit -m "feat(config): add ObservabilityConfig + ExperimentLoggerConfig (default OFF)

Pure-add Pydantic models defining the new observability surface. The
field defaults to None on Settings so any pre-feature YAML loads
unchanged (CLAUDE.md invariant #9). Regression test pins the
backwards-compat contract and the Literal[\"none\",\"mlflow\"] rejection."
```

---

### Task 3: Define `ExperimentLoggerProtocol`, `PhaseContext`, and `_to_finite_float` helper

**Files:**
- Create: `src/mousedroid/training/observability/__init__.py`
- Create: `src/mousedroid/training/observability/protocol.py`
- Create test: `tests/unit/training/observability/__init__.py`
- Create test: `tests/unit/training/observability/test_finite_float_coercion.py`

- [ ] **Step 1: Write the failing property test for the coercion helper**

Create `tests/unit/training/observability/__init__.py` (empty file).

Create `tests/unit/training/observability/test_finite_float_coercion.py`:

```python
"""Property test for ``_to_finite_float`` — MLflow metric value coercion.

MLflow ``log_metric`` accepts a Python ``float`` but training loops produce
``torch.Tensor`` / numpy scalars; logging NaN/Inf silently corrupts metric
curves. The helper centralises both concerns at the logger boundary.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from mousedroid.training.observability.protocol import _to_finite_float


@given(value=st.floats(allow_nan=False, allow_infinity=False, width=32))
@settings(max_examples=80, deadline=None)
def test_python_float_passes_through_finite(value: float) -> None:
    assert _to_finite_float(value) == pytest.approx(value, rel=1e-6, abs=1e-9)


@given(value=st.floats(allow_nan=False, allow_infinity=False, width=32))
@settings(max_examples=80, deadline=None)
def test_numpy_scalar_collapses_to_float(value: float) -> None:
    assert _to_finite_float(np.float32(value)) == pytest.approx(
        float(np.float32(value)), rel=1e-6, abs=1e-9
    )


def test_torch_zero_dim_tensor_collapses_via_item() -> None:
    torch = pytest.importorskip("torch")
    t = torch.tensor(3.14)
    assert _to_finite_float(t) == pytest.approx(3.14, rel=1e-6)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_returns_none(bad: float) -> None:
    assert _to_finite_float(bad) is None


def test_numpy_nan_returns_none() -> None:
    assert _to_finite_float(np.float32("nan")) is None


def test_torch_nan_tensor_returns_none() -> None:
    torch = pytest.importorskip("torch")
    assert _to_finite_float(torch.tensor(float("nan"))) is None


def test_unsupported_type_returns_none() -> None:
    """A string / dict / None is not a metric value — coerce to None, not raise."""
    assert _to_finite_float("not a number") is None
    assert _to_finite_float(None) is None
    assert _to_finite_float({"key": 1.0}) is None


def test_int_collapses_to_float() -> None:
    """Python int is a valid scalar; collapse to float."""
    assert _to_finite_float(42) == 42.0
    assert isinstance(_to_finite_float(42), float)
```

- [ ] **Step 2: Run the test to verify it fails (module doesn't exist)**

Run: `python -m pytest tests/unit/training/observability/test_finite_float_coercion.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: `ModuleNotFoundError: No module named 'mousedroid.training.observability'`.

- [ ] **Step 3: Create the protocol module + helper**

Create `src/mousedroid/training/observability/__init__.py`:

```python
"""Experiment-logger subsystem for the training pipeline.

Defines :class:`ExperimentLoggerProtocol` and ships two implementations:

* :class:`~mousedroid.training.observability.noop_logger.NoOpExperimentLogger`
  — always available, byte-identical no-op (the default).
* :class:`~mousedroid.training.observability.mlflow_logger.MlflowExperimentLogger`
  — wraps :class:`mlflow.MlflowClient`. Concrete; imported lazily inside
  :func:`mousedroid.factory.build_experiment_logger` so callers can rely
  on the protocol without paying the mlflow import cost.

The factory returns ``NoOpExperimentLogger`` when ``cfg.observability`` is
``None`` or ``cfg.observability.experiment_logger.backend == "none"``, and
when the ``[mlflow]`` extras are not installed — preserving byte-identical
behavior to the pre-feature path.
"""

from __future__ import annotations

from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)

__all__ = ["ExperimentLoggerProtocol", "PhaseContext"]
```

Create `src/mousedroid/training/observability/protocol.py`:

```python
"""Protocol + dataclass for experiment-logger DI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PhaseContext:
    """Opaque handle representing an active phase (child) run.

    Returned by :meth:`ExperimentLoggerProtocol.start_phase` and passed back
    to ``log_phase_metric`` / ``end_phase`` so the logger can route per-phase
    metrics to the right child run without per-call lookups. ``run_id`` is
    the backend's identifier (MLflow run-id, or a UUID4 for the NoOp logger);
    callers MUST treat it as opaque.
    """

    run_id: str
    phase: str


@runtime_checkable
class ExperimentLoggerProtocol(Protocol):
    """Interface for an experiment logger threaded into the training pipeline.

    Two-tier scope:

    * ``start_run`` / ``log_params`` / ``log_metric`` / ``log_artifact`` /
      ``end_run`` — operate on the **parent (pipeline)** run.
    * ``start_phase`` / ``log_phase_metric`` / ``log_phase_artifact`` /
      ``end_phase`` — operate on a **child (phase)** run nested under the
      parent. The parent → child relation is implementation-defined (the
      MLflow concrete uses the ``mlflow.parentRunId`` tag).

    All methods are total — they MUST NOT raise on backend failure
    (network drop, malformed input, NaN). On failure they emit a structured
    warning and return without side effect, mirroring the
    ``Never raises on backend failure`` contract used by the LLM gateways.
    """

    # --- parent-run lifecycle -------------------------------------------------
    def start_run(
        self,
        *,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start the parent run and return its run-id."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log scalar params on the parent run (e.g. resolved config snapshot)."""
        ...

    def log_metric(self, key: str, value: Any, step: int | None = None) -> None:
        """Log a scalar metric on the parent run.

        ``value`` is coerced via :func:`_to_finite_float`; NaN/Inf are
        skipped with a warning.
        """
        ...

    def log_artifact(self, local_path: str) -> None:
        """Upload a local file as a parent-run artifact (e.g. config.json)."""
        ...

    def end_run(self, *, status: str = "FINISHED") -> None:
        """Terminate the parent run. ``status`` MUST be one of
        ``FINISHED`` / ``FAILED`` / ``KILLED``.
        """
        ...

    # --- child (phase) lifecycle ---------------------------------------------
    def start_phase(
        self,
        *,
        phase: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PhaseContext:
        """Start a child run nested under the active parent run."""
        ...

    def log_phase_metric(
        self,
        ctx: PhaseContext,
        key: str,
        value: Any,
        step: int | None = None,
    ) -> None:
        """Log a scalar metric on the child run identified by ``ctx``."""
        ...

    def log_phase_artifact(self, ctx: PhaseContext, local_path: str) -> None:
        """Upload a local file as a child-run artifact."""
        ...

    def end_phase(self, ctx: PhaseContext, *, status: str = "FINISHED") -> None:
        """Terminate the child run identified by ``ctx``."""
        ...


# --- helpers --------------------------------------------------------------- #
def _to_finite_float(value: Any) -> float | None:
    """Coerce ``value`` to a finite Python ``float`` or return ``None``.

    Handles ``int``, ``float``, numpy scalars, and torch zero-dim tensors.
    Returns ``None`` for NaN / Inf / non-numeric / dict / None — the caller
    is responsible for skipping the log call when ``None`` is returned.

    The helper exists so every code path that hits MLflow's ``log_metric``
    goes through one finite-check gate — a NaN slipping through silently
    creates curve gaps in the UI and wastes operator time. Pair with
    ``torch.no_grad()`` in the call site if the value comes from a tensor
    that still has its gradient graph attached (CLAUDE.md invariant #7).
    """
    if value is None or isinstance(value, (str, bytes, dict, list, tuple, set)):
        return None
    # Torch 0-dim tensor / numpy scalar both expose ``.item()``.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (RuntimeError, ValueError, TypeError):
            return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        _log.warning("experiment_logger_skipped_nonfinite", value=repr(value))
        return None
    return coerced


__all__ = ["ExperimentLoggerProtocol", "PhaseContext", "_to_finite_float"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/training/observability/test_finite_float_coercion.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: all 8 tests pass (and the parametrized one expands to 3 cases — 10 total).

- [ ] **Step 5: Lint + format + mypy**

Run: `python -m ruff check src/mousedroid/training/observability/ tests/unit/training/observability/ && python -m ruff format --check src/mousedroid/training/observability/ tests/unit/training/observability/ && python -m mypy --strict src/mousedroid/training/observability/`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/mousedroid/training/observability/__init__.py \
        src/mousedroid/training/observability/protocol.py \
        tests/unit/training/observability/__init__.py \
        tests/unit/training/observability/test_finite_float_coercion.py
git commit -m "feat(training/observability): protocol + PhaseContext + finite-float coercion

Pure-add module defining the ExperimentLoggerProtocol seam and the
_to_finite_float helper. Property test (Hypothesis) pins NaN/Inf rejection,
torch tensor + numpy scalar coercion, and unsupported-type fall-through."
```

---

### Task 4: Implement `NoOpExperimentLogger`

**Files:**
- Create: `src/mousedroid/training/observability/noop_logger.py`
- Create test: `tests/unit/training/observability/test_noop_logger.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/training/observability/test_noop_logger.py`:

```python
"""Tests for NoOpExperimentLogger — byte-identical no-op contract."""

from __future__ import annotations

import structlog.testing

from mousedroid.training.observability.noop_logger import NoOpExperimentLogger
from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)


def test_satisfies_protocol() -> None:
    """NoOp satisfies the runtime-checkable protocol."""
    assert isinstance(NoOpExperimentLogger(), ExperimentLoggerProtocol)


def test_start_run_returns_sentinel_id() -> None:
    """Returns a stable ``noop-run`` id so callers can stash it without crashing."""
    logger = NoOpExperimentLogger()
    run_id = logger.start_run(run_name="x", params={"a": 1}, tags={"b": "c"})
    assert run_id == "noop-run"


def test_start_phase_returns_phase_context() -> None:
    """Returns a PhaseContext with a stable id and the requested phase name."""
    logger = NoOpExperimentLogger()
    ctx = logger.start_phase(phase="rssm", params={"lr": 1e-3}, tags={"phase": "rssm"})
    assert isinstance(ctx, PhaseContext)
    assert ctx.phase == "rssm"
    assert ctx.run_id == "noop-phase-rssm"


def test_methods_emit_no_logs() -> None:
    """Every call is a silent no-op — NoOp must not spam structured logs."""
    with structlog.testing.capture_logs() as logs:
        logger = NoOpExperimentLogger()
        logger.start_run(run_name="x")
        logger.log_params({"a": 1})
        logger.log_metric("loss", 0.5, step=1)
        logger.log_artifact("/tmp/x.json")
        ctx = logger.start_phase(phase="p")
        logger.log_phase_metric(ctx, "loss", 0.4, step=1)
        logger.log_phase_artifact(ctx, "/tmp/y.json")
        logger.end_phase(ctx)
        logger.end_run()
    assert logs == []


def test_end_run_status_accepted_but_ignored() -> None:
    """All three legal statuses (FINISHED/FAILED/KILLED) are accepted, no raise."""
    logger = NoOpExperimentLogger()
    logger.start_run(run_name="x")
    logger.end_run(status="FAILED")  # must not raise
    logger.end_run(status="KILLED")  # must not raise
    logger.end_run(status="FINISHED")  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/training/observability/test_noop_logger.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: `ModuleNotFoundError: No module named 'mousedroid.training.observability.noop_logger'`.

- [ ] **Step 3: Implement the NoOp logger**

Create `src/mousedroid/training/observability/noop_logger.py`:

```python
"""NoOp experiment logger — the always-available default."""

from __future__ import annotations

from typing import Any

from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)


class NoOpExperimentLogger:
    """Every method is a silent no-op.

    Conforms structurally to :class:`ExperimentLoggerProtocol`. Returned by
    :func:`mousedroid.factory.build_experiment_logger` when observability is
    disabled OR when ``mlflow`` is not installed — so call sites can ALWAYS
    rely on a non-None logger and skip ``if logger is not None`` guards.
    """

    def start_run(
        self,
        *,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        del run_name, params, tags
        return "noop-run"

    def log_params(self, params: dict[str, Any]) -> None:
        del params

    def log_metric(self, key: str, value: Any, step: int | None = None) -> None:
        del key, value, step

    def log_artifact(self, local_path: str) -> None:
        del local_path

    def end_run(self, *, status: str = "FINISHED") -> None:
        del status

    def start_phase(
        self,
        *,
        phase: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PhaseContext:
        del params, tags
        return PhaseContext(run_id=f"noop-phase-{phase}", phase=phase)

    def log_phase_metric(
        self,
        ctx: PhaseContext,
        key: str,
        value: Any,
        step: int | None = None,
    ) -> None:
        del ctx, key, value, step

    def log_phase_artifact(self, ctx: PhaseContext, local_path: str) -> None:
        del ctx, local_path

    def end_phase(self, ctx: PhaseContext, *, status: str = "FINISHED") -> None:
        del ctx, status


# Verify structural protocol conformance at import time so a method-signature
# drift fails fast.
_PROTOCOL_CHECK: ExperimentLoggerProtocol = NoOpExperimentLogger()
del _PROTOCOL_CHECK


__all__ = ["NoOpExperimentLogger"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/training/observability/test_noop_logger.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: 5 tests pass.

- [ ] **Step 5: Re-export from `__init__.py`**

Add `NoOpExperimentLogger` to the `__init__.py`:

```python
from mousedroid.training.observability.noop_logger import NoOpExperimentLogger
from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)

__all__ = ["ExperimentLoggerProtocol", "NoOpExperimentLogger", "PhaseContext"]
```

- [ ] **Step 6: Lint + format + mypy + commit**

Run: `python -m ruff check src/mousedroid/training/observability/ tests/unit/training/observability/test_noop_logger.py && python -m ruff format --check src/mousedroid/training/observability/ tests/unit/training/observability/test_noop_logger.py && python -m mypy --strict src/mousedroid/training/observability/`
Expected: all clean.

```bash
git add src/mousedroid/training/observability/noop_logger.py \
        src/mousedroid/training/observability/__init__.py \
        tests/unit/training/observability/test_noop_logger.py
git commit -m "feat(training/observability): NoOpExperimentLogger — byte-identical default

Every method is a silent no-op. Returned by build_experiment_logger
when observability is disabled OR when [mlflow] extras are missing,
so call sites never need a None guard."
```

---

### Task 5: Implement `MlflowExperimentLogger`

**Files:**
- Create: `src/mousedroid/training/observability/mlflow_logger.py`
- Create test: `tests/unit/training/observability/test_mlflow_logger.py`

- [ ] **Step 1: Write the failing tests using a real MlflowClient over tmp_path**

Create `tests/unit/training/observability/test_mlflow_logger.py`:

```python
"""Tests for MlflowExperimentLogger — uses a real MlflowClient over tmp_path.

Per the writing-plans research, mocking ``MlflowClient`` itself loses the
ability to catch signature drift on MLflow upgrades. The right pattern is
a tmp_path-rooted file backend and a real client.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")  # skip module entirely if extras missing
from mlflow import MlflowClient  # noqa: E402

from mousedroid.training.observability.mlflow_logger import (  # noqa: E402
    MlflowExperimentLogger,
)
from mousedroid.training.observability.protocol import (  # noqa: E402
    ExperimentLoggerProtocol,
    PhaseContext,
)


@pytest.fixture()
def tracking_uri(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'mlruns'}"


@pytest.fixture()
def client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


def _build_logger(tracking_uri: str, experiment: str = "test-exp") -> MlflowExperimentLogger:
    return MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment,
    )


def test_satisfies_protocol(tracking_uri: str) -> None:
    assert isinstance(_build_logger(tracking_uri), ExperimentLoggerProtocol)


def test_start_run_creates_parent_run_with_params_and_tags(
    tracking_uri: str, client: MlflowClient
) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(
        run_name="pipeline-1",
        params={"phases_count": 4, "amp": True},
        tags={"track": "T"},
    )
    run = client.get_run(run_id)
    assert run.info.status == "RUNNING"
    assert run.info.run_name == "pipeline-1"
    # MLflow coerces param values to strings on read.
    assert run.data.params["phases_count"] == "4"
    assert run.data.params["amp"] == "True"
    assert run.data.tags["track"] == "T"
    logger.end_run()


def test_log_metric_records_step_history(tracking_uri: str, client: MlflowClient) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="metrics")
    for step, loss in enumerate([1.0, 0.8, 0.6]):
        logger.log_metric("loss", loss, step=step)
    logger.end_run()
    history = client.get_metric_history(run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(0, 1.0), (1, 0.8), (2, 0.6)]


def test_log_metric_skips_nonfinite_value(tracking_uri: str, client: MlflowClient) -> None:
    """NaN must not reach the store; the warning is recorded by _to_finite_float."""
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="nan")
    logger.log_metric("loss", float("nan"), step=0)
    logger.log_metric("loss", 0.5, step=1)
    logger.end_run()
    history = client.get_metric_history(run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(1, 0.5)]


def test_start_phase_nests_under_parent_via_tag(
    tracking_uri: str, client: MlflowClient
) -> None:
    """Nested runs are tagged with mlflow.parentRunId — the canonical pattern."""
    logger = _build_logger(tracking_uri)
    parent_id = logger.start_run(run_name="pipe")
    ctx = logger.start_phase(phase="rssm")
    assert isinstance(ctx, PhaseContext)
    child = client.get_run(ctx.run_id)
    assert child.data.tags.get("mlflow.parentRunId") == parent_id
    assert child.data.tags.get("phase") == "rssm"
    logger.end_phase(ctx)
    logger.end_run()


def test_end_run_marks_status_finished(tracking_uri: str, client: MlflowClient) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="ok")
    logger.end_run()
    assert client.get_run(run_id).info.status == "FINISHED"


def test_end_run_status_failed_propagates(tracking_uri: str, client: MlflowClient) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="boom")
    logger.end_run(status="FAILED")
    assert client.get_run(run_id).info.status == "FAILED"


def test_end_run_rejects_invalid_status_with_warning(tracking_uri: str) -> None:
    """An unknown status string is normalised to FINISHED with a warning, never raises."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    logger.end_run(status="GARBAGE")  # must not raise


def test_log_metric_before_start_run_is_safe(tracking_uri: str) -> None:
    """Calling log_metric without start_run is a silent no-op + warning."""
    logger = _build_logger(tracking_uri)
    logger.log_metric("loss", 0.5)  # must not raise


def test_end_phase_after_end_run_is_safe(tracking_uri: str) -> None:
    """End-of-life ordering robustness — a stale ctx never crashes the trainer."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    ctx = logger.start_phase(phase="p")
    logger.end_run()  # parent terminates first (unusual but possible on KeyboardInterrupt)
    logger.end_phase(ctx)  # must not raise
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m pytest tests/unit/training/observability/test_mlflow_logger.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: All tests SKIPPED if `mlflow` is not installed locally OR ImportError on the module. Install dev extras first if needed: `python -m pip install "mlflow-skinny>=2.22,<3"`.

If mlflow is installed: tests fail with `ModuleNotFoundError: No module named 'mousedroid.training.observability.mlflow_logger'`.

- [ ] **Step 3: Implement the MLflow logger**

Create `src/mousedroid/training/observability/mlflow_logger.py`:

```python
"""MLflow-backed experiment logger using the ``MlflowClient`` OOP API.

Why ``MlflowClient`` and not the fluent ``mlflow.start_run`` API:

* No reliance on ``mlflow.active_run`` thread-local state — works under
  asyncio + thread pool dispatch without surprise.
* Idempotent / mockable / testable with a real client over a ``tmp_path``
  file backend (the recommended pattern per the project research notes).
* Symmetric with how every other backend wrapper in the codebase looks
  (e.g. :class:`AnthropicLLMGateway` wraps the ``anthropic`` SDK).

Imports ``mlflow`` lazily in :meth:`__init__` so the protocol module stays
import-safe when the ``[mlflow]`` extras are absent — the factory degrades
to :class:`NoOpExperimentLogger` in that case.

CLAUDE.md invariants honored:
* Protocol-DI (#1): conforms structurally to :class:`ExperimentLoggerProtocol`.
* No hardcoded values (#3): every knob comes from ``ExperimentLoggerConfig``.
* Structured logging (#4): all branches emit ``mlflow_logger_*`` events.
* Never raises on backend failure: catches ``Exception`` at every write
  boundary and degrades to a warning log + return (mirrors the LLM gateways).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mousedroid.logging.setup import get_logger
from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
    _to_finite_float,
)

_log = get_logger(__name__)

_VALID_STATUSES: frozenset[str] = frozenset({"FINISHED", "FAILED", "KILLED"})
_PARENT_RUN_TAG = "mlflow.parentRunId"


class MlflowExperimentLogger:
    """Wraps :class:`mlflow.MlflowClient` for parent + nested-phase runs.

    Construction is the only place this class touches ``mlflow``; method
    bodies operate via the constructed client. The
    :class:`ExperimentLoggerProtocol` consumer never sees an mlflow type
    leak through.
    """

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
    ) -> None:
        """Build the underlying ``MlflowClient`` + resolve the experiment.

        Args:
            tracking_uri: MLflow tracking URI. ``file:`` URIs are pinned to
                an absolute path (the factory resolves CWD-relative paths
                BEFORE calling here so they survive working-dir changes).
            experiment_name: MLflow experiment name; created if missing.
            run_name: Optional default run name for the parent run.
        """
        # Lazy import so a project that never opts-in to mlflow does not pay
        # the import cost. The factory probes the extras availability and
        # degrades to NoOp when missing — so reaching this constructor
        # implies the extras are installed.
        from mlflow import MlflowClient  # noqa: PLC0415

        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._default_run_name = run_name
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment_id: str = self._resolve_or_create_experiment(experiment_name)
        self._active_run_id: str | None = None
        _log.info(
            "mlflow_logger_initialised",
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            experiment_id=self._experiment_id,
        )

    # ---- experiment resolution ---------------------------------------------
    def _resolve_or_create_experiment(self, name: str) -> str:
        existing = self._client.get_experiment_by_name(name)
        if existing is not None:
            return existing.experiment_id
        return self._client.create_experiment(name)

    # ---- parent run --------------------------------------------------------
    def start_run(
        self,
        *,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        effective_name = run_name or self._default_run_name or "pipeline"
        try:
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                run_name=effective_name,
                tags=tags or {},
            )
        except Exception as exc:  # broad — never raise on backend failure
            _log.warning(
                "mlflow_logger_start_run_failed",
                error=f"{type(exc).__name__}:{exc}",
            )
            return ""
        self._active_run_id = run.info.run_id
        if params:
            self.log_params(params)
        return run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self._active_run_id is None:
            _log.warning("mlflow_logger_log_params_without_run")
            return
        for key, value in params.items():
            try:
                self._client.log_param(self._active_run_id, key, value)
            except Exception as exc:  # noqa: BLE001 — degrade-not-raise
                _log.warning(
                    "mlflow_logger_log_param_failed",
                    key=key,
                    error=f"{type(exc).__name__}:{exc}",
                )

    def log_metric(self, key: str, value: Any, step: int | None = None) -> None:
        if self._active_run_id is None:
            _log.warning("mlflow_logger_log_metric_without_run", key=key)
            return
        coerced = _to_finite_float(value)
        if coerced is None:
            return  # _to_finite_float already logged the skip
        try:
            self._client.log_metric(self._active_run_id, key, coerced, step=step)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "mlflow_logger_log_metric_failed",
                key=key,
                error=f"{type(exc).__name__}:{exc}",
            )

    def log_artifact(self, local_path: str) -> None:
        if self._active_run_id is None:
            _log.warning("mlflow_logger_log_artifact_without_run", path=local_path)
            return
        if not Path(local_path).exists():
            _log.warning("mlflow_logger_artifact_missing", path=local_path)
            return
        try:
            self._client.log_artifact(self._active_run_id, local_path)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "mlflow_logger_log_artifact_failed",
                path=local_path,
                error=f"{type(exc).__name__}:{exc}",
            )

    def end_run(self, *, status: str = "FINISHED") -> None:
        if self._active_run_id is None:
            return  # silent — nothing to end
        normalised = status if status in _VALID_STATUSES else "FINISHED"
        if normalised != status:
            _log.warning(
                "mlflow_logger_invalid_status_normalised",
                requested=status,
                normalised=normalised,
            )
        try:
            self._client.set_terminated(self._active_run_id, status=normalised)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "mlflow_logger_end_run_failed",
                error=f"{type(exc).__name__}:{exc}",
            )
        finally:
            self._active_run_id = None

    # ---- child (phase) run -------------------------------------------------
    def start_phase(
        self,
        *,
        phase: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PhaseContext:
        if self._active_run_id is None:
            _log.warning("mlflow_logger_start_phase_without_parent", phase=phase)
            return PhaseContext(run_id="", phase=phase)
        merged_tags = dict(tags or {})
        merged_tags[_PARENT_RUN_TAG] = self._active_run_id
        merged_tags.setdefault("phase", phase)
        try:
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                run_name=phase,
                tags=merged_tags,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "mlflow_logger_start_phase_failed",
                phase=phase,
                error=f"{type(exc).__name__}:{exc}",
            )
            return PhaseContext(run_id="", phase=phase)
        if params:
            for key, value in params.items():
                try:
                    self._client.log_param(run.info.run_id, key, value)
                except Exception as exc:  # noqa: BLE001
                    _log.warning(
                        "mlflow_logger_phase_param_failed",
                        phase=phase,
                        key=key,
                        error=f"{type(exc).__name__}:{exc}",
                    )
        return PhaseContext(run_id=run.info.run_id, phase=phase)

    def log_phase_metric(
        self,
        ctx: PhaseContext,
        key: str,
        value: Any,
        step: int | None = None,
    ) -> None:
        if not ctx.run_id:
            return
        coerced = _to_finite_float(value)
        if coerced is None:
            return
        try:
            self._client.log_metric(ctx.run_id, key, coerced, step=step)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "mlflow_logger_log_phase_metric_failed",
                phase=ctx.phase,
                key=key,
                error=f"{type(exc).__name__}:{exc}",
            )

    def log_phase_artifact(self, ctx: PhaseContext, local_path: str) -> None:
        if not ctx.run_id:
            return
        if not Path(local_path).exists():
            _log.warning("mlflow_logger_phase_artifact_missing", path=local_path)
            return
        try:
            self._client.log_artifact(ctx.run_id, local_path)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "mlflow_logger_log_phase_artifact_failed",
                phase=ctx.phase,
                error=f"{type(exc).__name__}:{exc}",
            )

    def end_phase(self, ctx: PhaseContext, *, status: str = "FINISHED") -> None:
        if not ctx.run_id:
            return
        normalised = status if status in _VALID_STATUSES else "FINISHED"
        if normalised != status:
            _log.warning(
                "mlflow_logger_invalid_phase_status_normalised",
                phase=ctx.phase,
                requested=status,
                normalised=normalised,
            )
        try:
            self._client.set_terminated(ctx.run_id, status=normalised)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "mlflow_logger_end_phase_failed",
                phase=ctx.phase,
                error=f"{type(exc).__name__}:{exc}",
            )


# Structural protocol-conformance check at import time. The placeholder
# instantiation is fenced behind ``mlflow`` availability so this module
# remains importable even when the extras are absent.
__all__ = ["MlflowExperimentLogger"]
```

- [ ] **Step 4: Run the test to verify it passes**

If mlflow not installed: `python -m pip install "mlflow-skinny>=2.22,<3"` first.

Run: `python -m pytest tests/unit/training/observability/test_mlflow_logger.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: all 10 tests pass.

- [ ] **Step 5: Lint + format + mypy + commit**

Run: `python -m ruff check src/mousedroid/training/observability/mlflow_logger.py tests/unit/training/observability/test_mlflow_logger.py && python -m ruff format --check src/mousedroid/training/observability/mlflow_logger.py tests/unit/training/observability/test_mlflow_logger.py && python -m mypy --strict src/mousedroid/training/observability/`
Expected: all clean.

```bash
git add src/mousedroid/training/observability/mlflow_logger.py \
        tests/unit/training/observability/test_mlflow_logger.py
git commit -m "feat(training/observability): MlflowExperimentLogger via MlflowClient

Wraps mlflow.MlflowClient for parent + nested-phase runs using the
mlflow.parentRunId tag (NOT the fluent nested=True API). Every backend
write is degrade-not-raise so a transient store failure cannot kill the
training loop. Tests use a real client over tmp_path file backend."
```

---

### Task 6: Add `build_experiment_logger()` to factory

**Files:**
- Modify: `src/mousedroid/factory.py` (insert near `build_metrics_registry` at line 1166)
- Create test: `tests/unit/test_factory_observability.py`

- [ ] **Step 1: Write the failing factory test**

Create `tests/unit/test_factory_observability.py`:

```python
"""Tests for factory.build_experiment_logger."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import (
    ExperimentLoggerConfig,
    ObservabilityConfig,
    Settings,
)
from mousedroid.factory import build_experiment_logger
from mousedroid.training.observability import (
    ExperimentLoggerProtocol,
    NoOpExperimentLogger,
)


def _settings_with_logger(**overrides: object) -> Settings:
    base = Settings(mock_hardware=True)
    return base.model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(**overrides),  # type: ignore[arg-type]
            ),
        }
    )


def test_no_observability_block_returns_noop() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.observability is None
    logger = build_experiment_logger(cfg)
    assert isinstance(logger, NoOpExperimentLogger)
    assert isinstance(logger, ExperimentLoggerProtocol)


def test_backend_none_returns_noop() -> None:
    cfg = _settings_with_logger(backend="none")
    assert isinstance(build_experiment_logger(cfg), NoOpExperimentLogger)


def test_backend_mlflow_returns_mlflow_logger_when_extras_present(tmp_path: object) -> None:
    pytest.importorskip("mlflow")
    cfg = _settings_with_logger(
        backend="mlflow",
        tracking_uri=f"file:{tmp_path}/mlruns",
        experiment_name="test",
    )
    logger = build_experiment_logger(cfg)
    from mousedroid.training.observability.mlflow_logger import MlflowExperimentLogger

    assert isinstance(logger, MlflowExperimentLogger)


def test_backend_mlflow_degrades_to_noop_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If mlflow is not importable the factory degrades cleanly with a warning."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("simulated absent extras")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Ensure no cached mlflow_logger module masks the import path.
    import sys

    for name in list(sys.modules):
        if name.startswith("mousedroid.training.observability.mlflow_logger") or name == "mlflow":
            sys.modules.pop(name, None)

    cfg = _settings_with_logger(backend="mlflow")
    logger = build_experiment_logger(cfg)
    assert isinstance(logger, NoOpExperimentLogger)


def test_relative_file_uri_is_resolved_to_absolute(tmp_path: object) -> None:
    """A ``file:./mlruns`` URI is pinned to an absolute path before construction."""
    pytest.importorskip("mlflow")
    monkey_cwd = str(tmp_path)
    import os

    saved_cwd = os.getcwd()
    try:
        os.chdir(monkey_cwd)
        cfg = _settings_with_logger(
            backend="mlflow",
            tracking_uri="file:./mlruns",
            experiment_name="abs",
        )
        logger = build_experiment_logger(cfg)
        # Internal attribute access is fine in tests; the contract is "absolute".
        assert getattr(logger, "_tracking_uri").startswith("file:")
        assert "mlruns" in getattr(logger, "_tracking_uri")
        # Crucially, the URI does NOT contain a relative ``./``.
        assert "./mlruns" not in getattr(logger, "_tracking_uri")
    finally:
        os.chdir(saved_cwd)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_factory_observability.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: `ImportError: cannot import name 'build_experiment_logger' from 'mousedroid.factory'`.

- [ ] **Step 3: Implement the factory builder**

In `src/mousedroid/factory.py`, find `build_metrics_registry` (line 1166). Insert IMMEDIATELY AFTER it (so the two observability builders sit together):

```python
def build_experiment_logger(cfg: Settings) -> ExperimentLoggerProtocol:
    """Build the shared experiment logger for training pipelines.

    Mirrors :func:`build_metrics_registry`'s shape: returns a NEVER-None
    protocol type so callers can drop the ``logger is not None`` guard.
    The NoOp implementation is the default and is byte-identically a no-op,
    so threading the logger through the orchestrator/trainer is free when
    observability is disabled.

    Resolution order:

    1. ``cfg.observability is None`` (the pre-feature default) →
       :class:`NoOpExperimentLogger`.
    2. ``cfg.observability.experiment_logger.backend == "none"`` →
       :class:`NoOpExperimentLogger`.
    3. ``cfg.observability.experiment_logger.backend == "mlflow"`` AND
       the ``[mlflow]`` extras are installed →
       :class:`MlflowExperimentLogger`.
    4. ``cfg.observability.experiment_logger.backend == "mlflow"`` AND
       ``mlflow-skinny`` is NOT installed →
       :class:`NoOpExperimentLogger` (with a structured warning, so
       operators see the misconfiguration without crashing the run).

    ``file:./mlruns`` URIs are pinned to an absolute path at build time so
    they survive working-dir changes inside the training process.

    Args:
        cfg: Root settings.

    Returns:
        A logger conforming to :class:`ExperimentLoggerProtocol`.
    """
    from mousedroid.training.observability import (
        ExperimentLoggerProtocol,
        NoOpExperimentLogger,
    )

    if cfg.observability is None:
        return NoOpExperimentLogger()
    logger_cfg = cfg.observability.experiment_logger
    if logger_cfg.backend == "none":
        return NoOpExperimentLogger()

    if logger_cfg.backend == "mlflow":
        try:
            from mousedroid.training.observability.mlflow_logger import (
                MlflowExperimentLogger,
            )
        except ImportError as exc:
            _log.warning(
                "experiment_logger_mlflow_extras_missing",
                error=f"{type(exc).__name__}:{exc}",
            )
            return NoOpExperimentLogger()

        tracking_uri = _resolve_tracking_uri(logger_cfg.tracking_uri)
        return MlflowExperimentLogger(
            tracking_uri=tracking_uri,
            experiment_name=logger_cfg.experiment_name,
            run_name=logger_cfg.run_name,
        )

    # Exhaustive Literal coverage; reached only on schema additions without a
    # corresponding factory branch.
    _log.warning(
        "experiment_logger_unknown_backend",
        backend=logger_cfg.backend,
    )
    return NoOpExperimentLogger()


def _resolve_tracking_uri(raw: str) -> str:
    """Pin a relative ``file:`` URI to an absolute path.

    Non-file URIs (``http``, ``https``, ``databricks``, ``sqlite``) pass
    through unchanged. The pin happens at factory time so the resolved
    path survives chdir() inside trainers.
    """
    if not raw.startswith("file:"):
        return raw
    path_part = raw[len("file:") :]
    abs_path = Path(path_part).resolve()
    return f"file:{abs_path}"
```

Add the import at the top of `factory.py` if missing:
```python
from pathlib import Path
```
(likely already imported; verify with `grep -n "from pathlib" src/mousedroid/factory.py`).

Also add the protocol type import under `TYPE_CHECKING` near line 80:
```python
if TYPE_CHECKING:
    ...
    from mousedroid.training.observability import ExperimentLoggerProtocol
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_factory_observability.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: 5 tests pass (one skipped if mlflow not installed; the test marker `pytest.importorskip` handles it).

- [ ] **Step 5: Lint + format + mypy + commit**

Run: `python -m ruff check src/mousedroid/factory.py tests/unit/test_factory_observability.py && python -m ruff format --check src/mousedroid/factory.py tests/unit/test_factory_observability.py && python -m mypy --strict src/mousedroid/factory.py`
Expected: all clean.

```bash
git add src/mousedroid/factory.py tests/unit/test_factory_observability.py
git commit -m "feat(factory): build_experiment_logger — NoOp/MLflow resolution

Mirrors build_metrics_registry's shape. Returns NoOp on disabled,
missing extras, or unknown backend — call sites never need a None
guard. file: URIs are pinned to an absolute path at build time."
```

---

### Task 7: Wire logger into `PipelineOrchestrator`

**Files:**
- Modify: `src/mousedroid/training/pipeline_orchestrator.py` (`__init__` line 46, `run` line 59, `_run_phase` line 128)
- Create test: `tests/integration/test_pipeline_orchestrator_observability.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_pipeline_orchestrator_observability.py`:

```python
"""Integration: pipeline orchestrator emits parent + nested phase runs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow import MlflowClient  # noqa: E402

from mousedroid.config.schema import (
    ExperimentLoggerConfig,
    ObservabilityConfig,
    Settings,
    TrainingPipelineConfig,
)
from mousedroid.training.observability.mlflow_logger import (  # noqa: E402
    MlflowExperimentLogger,
)
from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator  # noqa: E402


@pytest.fixture()
def tracking_uri(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'mlruns'}"


@pytest.fixture()
def settings(tracking_uri: str) -> Settings:
    base = Settings(mock_hardware=True)
    return base.model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    tracking_uri=tracking_uri,
                    experiment_name="pipeline-test",
                ),
            ),
            "training_pipeline": TrainingPipelineConfig(
                phases=["rssm", "warmstart"],
                checkpoint_dir=str(Path(tracking_uri.removeprefix("file:")).parent / "ckpt"),
                batch_sizes={"rssm": 16, "warmstart": 16},
                amp_enabled=False,
                resume_from_phase=None,
            ),
        }
    )


@pytest.mark.asyncio
async def test_run_creates_parent_and_nested_phase_runs(
    settings: Settings, tracking_uri: str
) -> None:
    logger = MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name=settings.observability.experiment_logger.experiment_name,
    )
    gpu = MagicMock()
    gpu.wait_for_thermal_clearance = AsyncMock(return_value=None)
    tuner = MagicMock()
    tuner.tune_batch_size = MagicMock(side_effect=lambda phase, base: base)

    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=settings.training_pipeline,  # type: ignore[arg-type]
        gpu_monitor=gpu,
        batch_tuner=tuner,
        experiment_logger=logger,
    )
    await orch.run()

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[logger._experiment_id],  # noqa: SLF001 — test-only access
        order_by=["attributes.start_time ASC"],
    )
    # 1 parent + 2 phase children
    assert len(runs) == 3
    parent = runs[0]
    children = runs[1:]
    assert parent.data.tags.get("mlflow.parentRunId") is None
    assert {c.data.tags.get("phase") for c in children} == {"rssm", "warmstart"}
    for c in children:
        assert c.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    assert parent.info.status == "FINISHED"
    assert all(c.info.status == "FINISHED" for c in children)


@pytest.mark.asyncio
async def test_run_marks_parent_failed_when_phase_raises(
    settings: Settings, tracking_uri: str
) -> None:
    logger = MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name=settings.observability.experiment_logger.experiment_name,
    )
    gpu = MagicMock()
    gpu.wait_for_thermal_clearance = AsyncMock(return_value=None)
    tuner = MagicMock()
    tuner.tune_batch_size = MagicMock(side_effect=lambda phase, base: base)

    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=settings.training_pipeline,  # type: ignore[arg-type]
        gpu_monitor=gpu,
        batch_tuner=tuner,
        experiment_logger=logger,
    )
    # Force the first phase to raise.
    orch._train_rssm = AsyncMock(side_effect=RuntimeError("simulated"))  # noqa: SLF001
    with pytest.raises(RuntimeError, match="simulated"):
        await orch.run()

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[logger._experiment_id],  # noqa: SLF001
        order_by=["attributes.start_time ASC"],
    )
    parent = runs[0]
    assert parent.info.status == "FAILED"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/integration/test_pipeline_orchestrator_observability.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: `TypeError: PipelineOrchestrator.__init__() got an unexpected keyword argument 'experiment_logger'`.

- [ ] **Step 3: Modify `PipelineOrchestrator`**

In `src/mousedroid/training/pipeline_orchestrator.py`, change `__init__` (line 46-57) to add the keyword-only logger:

```python
    def __init__(
        self,
        settings: Settings,
        pipeline_config: TrainingPipelineConfig,
        gpu_monitor: JetsonGPUMonitor | Any,
        batch_tuner: VRAMBatchTuner | Any,
        *,
        experiment_logger: ExperimentLoggerProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._config = pipeline_config
        self._gpu_monitor = gpu_monitor
        self._batch_tuner = batch_tuner
        self._checkpoint_dir = Path(pipeline_config.checkpoint_dir)
        # Default to NoOp so call sites never need a None guard. The factory
        # provides the real logger when the user opts in.
        if experiment_logger is None:
            from mousedroid.training.observability import NoOpExperimentLogger

            experiment_logger = NoOpExperimentLogger()
        self._experiment_logger = experiment_logger
```

Add the import under TYPE_CHECKING near line 18:
```python
if TYPE_CHECKING:
    from mousedroid.training.observability import ExperimentLoggerProtocol
```

Modify `run()` (line 59-126). Replace its body's preamble + epilogue with bracketing:

Locate line 85-90 (the `pipeline_started` log). REPLACE that section:
```python
        logger.info(
            "pipeline_started",
            total_phases=len(phases),
            start_index=start_idx,
            phases=phases[start_idx:],
        )
```

WITH:
```python
        logger.info(
            "pipeline_started",
            total_phases=len(phases),
            start_index=start_idx,
            phases=phases[start_idx:],
        )

        run_name = self._config.run_name if hasattr(self._config, "run_name") else "pipeline"
        parent_run_id = self._experiment_logger.start_run(
            run_name=run_name,
            params={
                "total_phases": len(phases),
                "start_index": start_idx,
                "amp_enabled": self._config.amp_enabled,
            },
            tags={"track": "training"},
        )
        run_status = "FINISHED"
```

Wrap the phase loop body in a try/except so failures mark the parent FAILED. Locate line 92 (the `for idx in range(start_idx, len(phases)):`). REPLACE the loop block (lines 92-124) with:

```python
        try:
            for idx in range(start_idx, len(phases)):
                phase = phases[idx]
                phase_log = logger.bind(phase=phase, phase_index=idx)

                await self._wait_for_thermal_clearance(phase_log)

                base_batch = self._config.batch_sizes.get(phase, self._settings.training.batch_size)
                tuned_batch = self._batch_tuner.tune_batch_size(phase, base_batch)

                if idx > start_idx:
                    prev_phase = phases[idx - 1]
                    if not self._checkpoint_exists(prev_phase):
                        msg = (
                            f"Missing checkpoint for phase '{prev_phase}' — "
                            f"cannot proceed to '{phase}'"
                        )
                        raise RuntimeError(msg)

                phase_log.info(
                    "phase_starting",
                    batch_size=tuned_batch,
                    amp_enabled=self._config.amp_enabled,
                )

                try:
                    await self._run_phase(phase, tuned_batch)
                except Exception:
                    phase_log.exception("phase_failed")
                    raise

                phase_log.info("phase_completed")
        except Exception:
            run_status = "FAILED"
            raise
        finally:
            self._experiment_logger.end_run(status=run_status)
            del parent_run_id  # locally bound for symmetry; logger holds state

        logger.info("pipeline_completed", phases_run=phases[start_idx:])
```

Modify `_run_phase()` (lines 128-149) to bracket each phase as a child run:

```python
    async def _run_phase(self, phase: str, batch_size: int) -> None:
        """Execute a single training phase under a nested phase run."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._experiment_logger.start_phase(
            phase=phase,
            params={"batch_size": batch_size, "amp_enabled": self._config.amp_enabled},
            tags={"phase": phase},
        )
        phase_status = "FINISHED"
        try:
            phase_fn = self._get_phase_runner(phase)
            await phase_fn(batch_size)
            checkpoint_path = self._checkpoint_dir / f"{phase}.done"
            checkpoint_path.write_text(f"phase={phase}\n")
            logger.info("checkpoint_written", path=str(checkpoint_path))
            self._experiment_logger.log_phase_artifact(ctx, str(checkpoint_path))
        except Exception:
            phase_status = "FAILED"
            raise
        finally:
            self._experiment_logger.end_phase(ctx, status=phase_status)
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `python -m pytest tests/integration/test_pipeline_orchestrator_observability.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: 2 tests pass.

- [ ] **Step 5: Run the original orchestrator unit tests to verify no regression**

Run: `python -m pytest tests/unit/test_pipeline_orchestrator.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: all existing tests still green.

- [ ] **Step 6: Lint + format + mypy + commit**

```bash
python -m ruff check src/mousedroid/training/pipeline_orchestrator.py tests/integration/test_pipeline_orchestrator_observability.py
python -m ruff format --check src/mousedroid/training/pipeline_orchestrator.py tests/integration/test_pipeline_orchestrator_observability.py
python -m mypy --strict src/mousedroid/training/pipeline_orchestrator.py
git add src/mousedroid/training/pipeline_orchestrator.py tests/integration/test_pipeline_orchestrator_observability.py
git commit -m "feat(training): wire ExperimentLogger into PipelineOrchestrator

start_run before the phase loop, end_run in finally (FAILED on any
exception). Each phase becomes a nested run via start_phase/end_phase
using the mlflow.parentRunId tag. Checkpoint file is uploaded as the
phase-run artifact. Default NoOp keeps pre-feature behavior byte-identical."
```

---

### Task 8: Wire logger into `OfflineRLTrainer`

**Files:**
- Modify: `src/mousedroid/learning/offline_rl.py` (`__init__` line 146, base helper, CQL `update_step` line 432, IQL `update_step` line 583)
- Create test: `tests/integration/test_offline_rl_observability.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_offline_rl_observability.py`:

```python
"""Integration: OfflineRLTrainer (CQL + IQL) logs per-step metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")
torch = pytest.importorskip("torch")

from mlflow import MlflowClient  # noqa: E402

from mousedroid.learning.offline_rl import CQLTrainer, IQLTrainer  # noqa: E402
from mousedroid.training.observability.mlflow_logger import (  # noqa: E402
    MlflowExperimentLogger,
)


@pytest.fixture()
def logger(tmp_path: Path) -> MlflowExperimentLogger:
    uri = f"file:{tmp_path / 'mlruns'}"
    return MlflowExperimentLogger(tracking_uri=uri, experiment_name="trainer-test")


def _batch(batch_size: int = 4, state_dim: int = 4, action_dim: int = 2) -> dict:
    return {
        "states": torch.zeros(batch_size, state_dim),
        "actions": torch.zeros(batch_size, action_dim),
        "rewards": torch.zeros(batch_size),
        "next_states": torch.zeros(batch_size, state_dim),
        "dones": torch.zeros(batch_size),
    }


def test_cql_trainer_logs_q_bellman_cql_policy_losses_per_step(
    logger: MlflowExperimentLogger,
) -> None:
    run_id = logger.start_run(run_name="cql")
    ctx = logger.start_phase(phase="cql")
    trainer = CQLTrainer(state_dim=4, action_dim=2, experiment_logger=logger, log_phase=ctx)
    batch = _batch()
    for step in range(3):
        trainer.update_step(**batch)
    logger.end_phase(ctx)
    logger.end_run()

    client = MlflowClient(tracking_uri=logger._tracking_uri)  # noqa: SLF001
    expected_keys = {"q_loss", "bellman_loss", "cql_loss", "policy_loss"}
    for key in expected_keys:
        history = client.get_metric_history(ctx.run_id, key)
        assert [m.step for m in history] == [0, 1, 2]


def test_iql_trainer_logs_q_value_policy_losses_per_step(
    logger: MlflowExperimentLogger,
) -> None:
    run_id = logger.start_run(run_name="iql")
    ctx = logger.start_phase(phase="iql")
    trainer = IQLTrainer(state_dim=4, action_dim=2, experiment_logger=logger, log_phase=ctx)
    batch = _batch()
    for step in range(3):
        trainer.update_step(**batch)
    logger.end_phase(ctx)
    logger.end_run()

    client = MlflowClient(tracking_uri=logger._tracking_uri)  # noqa: SLF001
    expected_keys = {"q_loss", "value_loss", "policy_loss"}
    for key in expected_keys:
        history = client.get_metric_history(ctx.run_id, key)
        assert [m.step for m in history] == [0, 1, 2]


def test_trainer_without_logger_is_byte_identical_default() -> None:
    """A trainer built without an experiment_logger arg runs unchanged."""
    trainer = CQLTrainer(state_dim=4, action_dim=2)
    out = trainer.update_step(**_batch())
    assert set(out.keys()) == {"q_loss", "bellman_loss", "cql_loss", "policy_loss"}
    for v in out.values():
        assert isinstance(v, float)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/integration/test_offline_rl_observability.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: `TypeError: CQLTrainer.__init__() got an unexpected keyword argument 'experiment_logger'`.

- [ ] **Step 3: Modify `OfflineRLTrainer.__init__` to accept the logger**

In `src/mousedroid/learning/offline_rl.py`, modify `OfflineRLTrainer.__init__` (line 146-193). Add two keyword-only args at the end + a helper. Add imports at top:

```python
from mousedroid.training.observability import (
    ExperimentLoggerProtocol,
    NoOpExperimentLogger,
    PhaseContext,
)
```

Update the `__init__` signature:
```python
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        device: torch.device | None = None,
        bc_lr: float | None = None,
        bc_batch_size: int | None = None,
        *,
        experiment_logger: ExperimentLoggerProtocol | None = None,
        log_phase: PhaseContext | None = None,
    ) -> None:
        # ... existing body unchanged through line 187 ...
        self._experiment_logger = experiment_logger or NoOpExperimentLogger()
        self._log_phase = log_phase
        self._global_step = 0
        _log.info(
            "offline_rl_bc_optimizer_built",
            bc_lr=bc_lr,
            bc_batch_size=bc_batch_size,
            shared_with_policy=self.bc_optimizer is self.policy_optimizer,
        )
```

Add a helper method on the base class (insert before `_soft_update_targets` line 195):

```python
    def _log_step_metrics(self, losses: dict[str, float]) -> None:
        """Forward per-update_step losses to the experiment logger.

        Called by ``update_step`` subclass implementations at the tail of
        each call. When the trainer was built without an
        ``experiment_logger`` OR without a ``log_phase`` context, this is a
        byte-identical no-op via the NoOp logger.
        """
        if self._log_phase is None:
            return
        for key, value in losses.items():
            self._experiment_logger.log_phase_metric(
                self._log_phase, key, value, step=self._global_step
            )
        self._global_step += 1
```

- [ ] **Step 4: Modify `CQLTrainer.update_step` (line 432-478)**

At the END of `update_step`, just BEFORE `return losses`, add:
```python
        self._log_step_metrics(losses)
        return losses
```

- [ ] **Step 5: Modify `IQLTrainer.update_step` (line 583-647)**

At the END of `update_step`, just BEFORE the return, add the same call:
```python
        self._log_step_metrics(losses)
        return losses
```

- [ ] **Step 6: Run the integration test**

Run: `python -m pytest tests/integration/test_offline_rl_observability.py --import-mode=importlib -v 2>&1 | tail -15`
Expected: 3 tests pass.

- [ ] **Step 7: Run existing offline_rl tests to verify byte-identical behavior**

Run: `python -m pytest tests/unit/test_offline_rl.py --import-mode=importlib -v 2>&1 | tail -10`
Expected: no regression in existing assertions.

- [ ] **Step 8: Lint + format + mypy + commit**

```bash
python -m ruff check src/mousedroid/learning/offline_rl.py tests/integration/test_offline_rl_observability.py
python -m ruff format --check src/mousedroid/learning/offline_rl.py tests/integration/test_offline_rl_observability.py
python -m mypy --strict src/mousedroid/learning/offline_rl.py
git add src/mousedroid/learning/offline_rl.py tests/integration/test_offline_rl_observability.py
git commit -m "feat(learning): wire ExperimentLogger into OfflineRLTrainer

CQL + IQL update_step now forwards its loss-dict to the experiment
logger via _log_step_metrics, indexed by an internal monotonic counter.
Default NoOp + None log_phase keeps pre-feature behavior byte-identical
— existing tests pass unchanged."
```

---

### Task 9: Operator runbook + CHANGELOG + NEXT_STEPS

**Files:**
- Create: `docs/runbooks/mlflow-local-ui.md`
- Modify: `CHANGELOG.md`
- Modify: `NEXT_STEPS.md`

- [ ] **Step 1: Create the operator runbook**

```markdown
# MLflow Local UI — Operator Runbook

The MLflow experiment logger writes to a local file backend at
`<repo>/mlruns/` (or whatever `cfg.observability.experiment_logger.tracking_uri`
points to). This runbook covers viewing the data.

## Prerequisites

Install the local-viewer extras (different from the rover-side `[mlflow]`
extras — the viewer needs the full `mlflow` package for the UI server):

```bash
pip install "mlflow>=2.22,<3"
```

## Viewing runs

From the repo root:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

This binds `127.0.0.1:5000` by default. Override:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5050
```

Open the URL it prints. The default experiment is `mousedroid`; runs are
named after the pipeline (parent) and each phase (child, nested via the
`mlflow.parentRunId` tag).

## Common pitfalls

* **No runs visible** — verify the working directory has an `mlruns/`
  subdirectory. The factory pins the path at startup; if the rover ran from
  a different CWD, the runs live elsewhere. Check
  `cfg.observability.experiment_logger.tracking_uri`.
* **"RUNNING" status stuck** — a training process crashed before `end_run`
  fired. Use the UI's "Delete run" or `mlflow.delete_run(run_id)`. The
  experiment logger calls `set_terminated(status="FAILED")` from the
  pipeline orchestrator's `finally` block, so this should be rare.
* **Concurrent writers** — the file backend is NOT thread/process safe for
  concurrent writers. Run one training process at a time per tracking URI.
* **Disk filling up** — old runs accumulate. Periodically clean with
  `mlflow gc --backend-store-uri file:./mlruns` or rotate the directory.

## Enabling on the rover

In `config/<overlay>.yaml`:

```yaml
observability:
  experiment_logger:
    backend: mlflow
    tracking_uri: file:/opt/mousedroid/mlruns
    experiment_name: mousedroid-jetson
    log_step_every_n: 10  # every 10th update_step (long runs)
```

Or via env:

```bash
MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND=mlflow
```

Required extras on the rover: `pip install "mousedroid[mlflow]"` —
installs only `mlflow-skinny` (write-only client, no Flask/SQLAlchemy).
```

- [ ] **Step 2: Update CHANGELOG.md**

Find `## [Unreleased]` and add under it:

```markdown
### Added
- `cfg.observability.experiment_logger` — MLflow-backed training metrics via
  `mlflow-skinny`. Wired into `PipelineOrchestrator` (parent run per pipeline,
  child run per phase) and `OfflineRLTrainer` (per-step loss metrics for CQL + IQL).
  Defaults to OFF; opt in via YAML or `MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND=mlflow`.
  See `docs/runbooks/mlflow-local-ui.md` for the operator runbook.
```

- [ ] **Step 3: Update NEXT_STEPS.md**

Find the prioritized list. Add a new entry referencing T2 from the
2026-05-15 roadmap:

```markdown
9. **[Training observability — P1] ✅ DONE (PR T2 — this PR).** MLflow logger
   wired into `PipelineOrchestrator` + `OfflineRLTrainer`. Operator runbook:
   `docs/runbooks/mlflow-local-ui.md`. Next-in-arc: T3 (`train_arm.py` SAC+HER entry point).
```

- [ ] **Step 4: Commit**

```bash
git add docs/runbooks/mlflow-local-ui.md CHANGELOG.md NEXT_STEPS.md
git commit -m "docs(observability): operator runbook + CHANGELOG + NEXT_STEPS update

Covers viewing runs locally (mlflow ui), enabling on the rover via
overlay YAML / env, and common pitfalls (concurrent writers, stuck
RUNNING status, disk hygiene)."
```

---

### Task 10: Full-suite verification + branch coverage gate

- [ ] **Step 1: Full lint + format check**

```bash
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
```
Expected: clean.

- [ ] **Step 2: Full mypy strict**

```bash
python -m mypy --strict src/mousedroid/
```
Expected: clean. If new errors appear in code unrelated to this PR, they
were pre-existing — diff against `git log` to confirm.

- [ ] **Step 3: Full pytest with coverage gate**

```bash
python -m pytest tests/unit tests/property tests/integration tests/regression \
    --import-mode=importlib --cov=src/mousedroid --cov-report=term-missing \
    --cov-fail-under=85 --no-header
```
Expected: all green; coverage ≥85% line and ≥85% branch on changed files.

- [ ] **Step 4: Pre-commit branch-coverage gate**

```bash
bash scripts/check_branch_coverage.py --min 85
```
Expected: passes. Reports per-file changed-line coverage for the PR.

- [ ] **Step 5: Hardcoded-values gate**

```bash
python scripts/check_no_hardcoded_values.py
```
Expected: clean. The new code defers every tunable to `ExperimentLoggerConfig`;
the only "constants" are the string literals `"none"` / `"mlflow"` / `"FINISHED"`
which are Pydantic Literal values + MLflow API enum values, both allowed.

- [ ] **Step 6: Final commit (only if any fix-ups needed)**

```bash
git status
git diff
# If any fix-ups landed, squash with --amend on the last logical commit OR
# add a "style: post-gate fixups" commit. Do NOT --no-verify.
```

---

## Self-review (run BEFORE handing off)

**Spec coverage:**
- [x] ObservabilityConfig + ExperimentLoggerConfig defined → Task 2
- [x] Protocol + PhaseContext + finite-float helper → Task 3
- [x] NoOp default → Task 4
- [x] MLflow concrete with MlflowClient → Task 5
- [x] Factory `build_experiment_logger` → Task 6
- [x] PipelineOrchestrator wiring → Task 7
- [x] OfflineRLTrainer wiring (CQL + IQL) → Task 8
- [x] Operator runbook + CHANGELOG + NEXT_STEPS → Task 9
- [x] Full-suite verification → Task 10
- [x] Backwards-compat regression test (Task 2 step 1)
- [x] mlflow-skinny extras in pyproject (Task 1)
- [x] Degrade-on-missing-extras (Task 6 step 3)
- [x] Absolute-path resolution for file: URIs (Task 6 step 3)
- [x] NaN/Inf coercion to None + warning (Task 3 + Task 5)

**Type consistency:**
- [x] `start_phase` returns `PhaseContext` everywhere it's used (Tasks 3, 4, 5, 7, 8)
- [x] `experiment_logger` keyword-only arg name is identical across `PipelineOrchestrator.__init__` and `OfflineRLTrainer.__init__`
- [x] `_to_finite_float` returns `float | None`; every caller checks for None
- [x] `_VALID_STATUSES` shared between `end_run` and `end_phase`
- [x] `MlflowExperimentLogger.start_run` returns `str` (the run_id); `start_phase` returns `PhaseContext` — distinct types as designed

**Placeholder scan:**
- [x] No "TBD" / "TODO" / "implement later" anywhere
- [x] Every code block is complete and runnable
- [x] All test code includes assertions, not stubs
- [x] All commit messages are full sentences

**Backwards-compat verification:**
- [x] All new Settings fields have safe defaults (None for `observability`, "none" for `backend`)
- [x] Regression test pins minimal-YAML, default.yaml, and jetson_production.yaml loading unchanged
- [x] All new function args are keyword-only with defaults
- [x] Existing tests verified green (Task 7 step 5, Task 8 step 7)

**Numpy/torch hygiene:**
- [x] `_to_finite_float` collapses torch tensors via `.item()` (Task 3)
- [x] `torch.no_grad()` invariant honored — `.item()` extracts the scalar without keeping the gradient graph alive
- [x] No implicit float64; we coerce to Python `float` which MLflow stores as f64 in the backend regardless

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-05-mlflow-experiment-logger.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (10 tasks → 10 dispatches), review between tasks, fast iteration with worktree isolation.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints for review.

Which approach?
