"""AQA pins for F-034 — mlflow tracking_uri default is sqlite, extras are present.

Complements ``tests/unit/training/observability/test_mlflow_logger.py`` and
``tests/unit/factory/test_factory_observability.py``, which prove *behaviour*
against a real ``MlflowClient`` — the latter also carries the direct,
dependency-free ``_resolve_tracking_uri`` passthrough pins (moved there from
an earlier draft of this file per ``.claude/skills/test-tier-mirror/SKILL.md``:
AQA is for schema properties, not behaviour). These two pins prove *shape* —
the specific values a refactor could quietly drift without any behavioural
test going red (e.g. reverting the default back to ``file:./mlruns``, or a
dependency-pin edit accidentally dropping ``sqlalchemy``/``alembic`` from the
``[mlflow]`` extra).

Both are Type A (revert-provable via ``scripts/prove_pin_fails.sh``).
"""

from __future__ import annotations

import pytest
from packaging.requirements import Requirement
from pydantic import ValidationError

from mousedroid.config.schema.telemetry import ExperimentLoggerConfig
from tests._pyproject import load_pyproject


def test_tracking_uri_default_is_sqlite() -> None:
    """The schema default is the sqlite backend, not the legacy file store.

    mlflow 3.x rejects the plain file-store backend outright; sqlite is
    mlflow's own recommended local backend and requires no explicit
    operator opt-in env var.
    """
    assert ExperimentLoggerConfig().tracking_uri == "sqlite:///mlflow.db"


def test_mlflow_extra_includes_sqlite_tracking_store_deps() -> None:
    """The [mlflow] extra carries sqlalchemy + alembic alongside mlflow-skinny.

    mlflow-skinny alone raises ``UnsupportedModelRegistryStoreURIException``
    for a ``sqlite:///`` tracking_uri without these two -- confirmed by
    constructing a real ``MlflowClient`` against a ``sqlite:///`` URI with
    only ``mlflow-skinny`` installed before this fix.
    """
    data = load_pyproject()
    project = data["project"]
    assert isinstance(project, dict)
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    mlflow_extra = optional["mlflow"]
    assert isinstance(mlflow_extra, list)
    names = {Requirement(d).name for d in mlflow_extra if isinstance(d, str)}
    assert "mlflow-skinny" in names
    assert "sqlalchemy" in names
    assert "alembic" in names


@pytest.mark.parametrize("blackhole", ["", "   ", "sqlite://", "SQLITE://"])
def test_tracking_uri_rejects_silently_discarding_values(blackhole: str) -> None:
    """Values that make the logger a silent black hole fail validation.

    Both of these validate as ordinary non-empty strings but discard every
    metric without ever raising, which is worse than failing loudly:
    whitespace-only falls back to mlflow's ambient default (so runs land
    somewhere unconfigured), and ``sqlite://`` (two slashes) is SQLAlchemy's
    in-memory database (so runs vanish at process exit). ``SQLITE://``
    covers the case-insensitive spelling.
    """
    with pytest.raises(ValidationError):
        ExperimentLoggerConfig(tracking_uri=blackhole)


@pytest.mark.parametrize(
    "legitimate",
    [
        "sqlite:///mlflow.db",
        "sqlite:///:memory:",
        "sqlite:////opt/mousedroid/mlflow.db",
        "file:./mlruns",
        "http://host:5000",
    ],
)
def test_tracking_uri_accepts_every_legitimate_backend_form(legitimate: str) -> None:
    """The validator rejects only the black-hole values, never a real backend.

    ``sqlite:///:memory:`` is deliberately allowed: it is the explicit,
    unambiguous way to request an in-memory store, unlike ``sqlite://``
    which is nearly always a typo for ``sqlite:///``.
    """
    assert ExperimentLoggerConfig(tracking_uri=legitimate).tracking_uri == legitimate
