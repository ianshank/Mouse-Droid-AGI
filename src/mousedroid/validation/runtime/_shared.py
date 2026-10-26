"""Shared runtime-validation helpers: config resolution + module-wide constants.

These names are genuinely shared across the domain modules in this package
(``_camera``, ``_hailo``, ``_storage``, ``_audio``, ``_lidar``) and are
imported by them as needed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from mousedroid.config.loader import load_settings
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


_CONFIG_LIST_ENV_VARS = ("MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS")
_CONFIG_SINGLE_ENV_VARS = ("MOUSEDROID_CONFIG", "MOUSEDROID_JETSON_CONFIG")

RUNTIME_CONFIG_ENV_VARS: Final[tuple[str, ...]] = _CONFIG_LIST_ENV_VARS + _CONFIG_SINGLE_ENV_VARS
"""Every variable :func:`resolve_runtime_config` reads, in precedence order.

Public so a caller that has to *tell an operator* which variables would have supplied an
overlay (the re-arm CLI's "no overlay" note) derives the list from the resolver instead of
keeping a copy that drifts.
"""

RuntimeConfigSource = Literal["cli", "env_csv", "env_single", "default"]
"""Where a resolved overlay stack came from.

``cli``: the caller passed paths explicitly (every in-repo caller forwards ``--config``).
``env_csv``: a comma-separated list variable. ``env_single``: a single-path variable.
``default``: nothing was supplied, so the caller loads ``default.yaml`` alone.
"""

# Named constants for paths and phrases used in validation helpers.
_ARGUS_SOCKET_PATH: str = "/tmp/argus_socket"  # noqa: S108 — fixed NVIDIA Argus socket path, not a temp write
_DEFAULT_SMOKE_PHRASE: str = "Hello hello! Rocky ready!"


@dataclass(frozen=True, slots=True)
class RuntimeConfigResolution:
    """The overlay stack a process should load, and where that answer came from."""

    paths: tuple[Path, ...]
    source: RuntimeConfigSource
    env_var: str | None = None
    """The variable that supplied ``paths`` when ``source`` is an ``env_*`` source."""


def resolve_runtime_config(
    config_paths: Sequence[Path | str] | None = None,
) -> RuntimeConfigResolution:
    """Resolve runtime config overlays and report where the answer came from.

    Precedence:
        1. Explicit ``config_paths`` passed by the caller (``source="cli"``).
        2. CSV lists in ``MOUSEDROID_CONFIGS`` or ``MOUSEDROID_JETSON_CONFIGS``
           (``source="env_csv"``).
        3. Single-path ``MOUSEDROID_CONFIG`` or legacy ``MOUSEDROID_JETSON_CONFIG``
           (``source="env_single"``).
        4. Nothing supplied (``source="default"``): the caller loads ``default.yaml``
           alone, which is almost never the deployed configuration.

    Blank values never count as supplied: a set-but-empty variable falls through to the
    next tier rather than resolving to an empty stack.

    Args:
        config_paths: Explicit config overlay paths.

    Returns:
        The normalized paths in precedence order, with their source.
    """
    explicit = tuple(Path(str(path)) for path in (config_paths or ()) if str(path).strip())
    if explicit:
        return RuntimeConfigResolution(explicit, "cli")

    for env_var in _CONFIG_LIST_ENV_VARS:
        raw_value = os.getenv(env_var, "").strip()
        if not raw_value:
            continue
        csv_paths = tuple(Path(part.strip()) for part in raw_value.split(",") if part.strip())
        if csv_paths:
            return RuntimeConfigResolution(csv_paths, "env_csv", env_var)

    for env_var in _CONFIG_SINGLE_ENV_VARS:
        single_path = os.getenv(env_var, "").strip()
        if single_path:
            return RuntimeConfigResolution((Path(single_path),), "env_single", env_var)

    return RuntimeConfigResolution((), "default")


def resolve_runtime_config_paths(
    config_paths: Sequence[Path | str] | None = None,
) -> tuple[Path, ...]:
    """Resolve runtime config overlays from explicit args or environment.

    The paths of :func:`resolve_runtime_config`, which owns the precedence; this is
    defined in terms of it so the two entry points cannot disagree.

    Args:
        config_paths: Explicit config overlay paths.

    Returns:
        Normalized config paths in precedence order.
    """
    return resolve_runtime_config(config_paths).paths


def load_runtime_settings(config_paths: Sequence[Path | str] | None = None) -> Settings:
    """Load runtime settings using the resolved config overlay list."""
    resolved_paths = resolve_runtime_config_paths(config_paths)
    return load_settings(*resolved_paths)


def _subprocess_timeout_s(cfg: Settings) -> float:
    """Return the per-subprocess timeout (seconds) for the verify_* probes."""
    return cfg.experience.diagnostics_subprocess_timeout_s


def _collect_configured_runtime_paths(cfg: Settings) -> dict[str, str]:
    """Build a mapping of schema-field -> resolved absolute path string.

    Schema fields covered (in the order they appear in ``Settings``):

    * ``experience.path`` — LMDB writer destination
    * ``jetson.tensorrt_cache_dir`` — compiled engine cache
    * ``cloud.weight_update.cache_dir`` — OTA download staging
    * ``harness.journal.path`` — operation journal (when configured)
    """
    paths: dict[str, str] = {}
    paths["experience.path"] = str(Path(cfg.experience.path).resolve())

    jetson_cfg = getattr(cfg, "jetson", None)
    if jetson_cfg is not None:
        cache_dir = getattr(jetson_cfg, "tensorrt_cache_dir", None)
        if cache_dir is not None:
            paths["jetson.tensorrt_cache_dir"] = str(Path(cache_dir).resolve())

    cloud_cfg = getattr(cfg, "cloud", None)
    weight_update_cfg = getattr(cloud_cfg, "weight_update", None) if cloud_cfg else None
    cache_dir = getattr(weight_update_cfg, "cache_dir", None) if weight_update_cfg else None
    if cache_dir is not None:
        paths["cloud.weight_update.cache_dir"] = str(Path(cache_dir).resolve())

    harness_cfg = getattr(cfg, "harness", None)
    journal_cfg = getattr(harness_cfg, "journal", None) if harness_cfg else None
    journal_path = getattr(journal_cfg, "path", None) if journal_cfg else None
    if journal_path is not None:
        paths["harness.journal.path"] = str(Path(journal_path).resolve())

    return paths
