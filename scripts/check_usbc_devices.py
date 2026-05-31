#!/usr/bin/env python3
"""USB-C endpoint smoke gate (config-driven, no hardcoded paths)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run from repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Force structlog to stderr BEFORE importing anything from mousedroid so the
# import-time configure() inside mousedroid.config.loader doesn't latch on to
# stdout (which would poison --json output).
import structlog  # noqa: E402

structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING and above
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=False,
)

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.diagnostics.usbc import (  # noqa: E402
    EndpointStatus,
    enumerate_usbc_devices,
)
from mousedroid.validation.runtime import (  # noqa: E402
    resolve_runtime_config_paths,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        action="append",
        default=None,
        type=Path,
        help=(
            "One or more YAML overlays (repeatable). When omitted, falls back "
            "to the resolve_runtime_config_paths() chain "
            "(MOUSEDROID_JETSON_CONFIGS env var + MOUSEDROID_CONFIG_DIR + "
            "config/default.yaml), aligning with the rest of the smoke "
            "surface so an operator can run this script with zero arguments."
        ),
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human output.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    # When --config is omitted, fall back to the runtime resolver so this
    # script behaves identically to the orchestrator + smoke wrappers
    # (single source of truth for which overlays apply on this host).
    config_paths = resolve_runtime_config_paths(args.config)
    settings = load_settings(*config_paths)
    if settings.usbc_discovery is None or not settings.usbc_discovery.enabled:
        print(
            "usbc_discovery not configured or disabled; nothing to check",
            file=sys.stderr,
        )
        return 0

    results = enumerate_usbc_devices(settings.usbc_discovery)
    if args.json:
        payload = {
            name: {
                "status": r.status.value,
                "resolved_path": str(r.resolved_path) if r.resolved_path else None,
                "required": r.required,
                "glob": r.glob,
            }
            for name, r in results.items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, r in results.items():
            marker = {
                EndpointStatus.PRESENT: "[OK]",
                EndpointStatus.WARN: "[WARN]",
                EndpointStatus.MISSING: "[FAIL]",
            }[r.status]
            location = r.resolved_path if r.resolved_path else f"missing ({r.glob})"
            print(f"  {marker} {name}: {location}")

    has_missing = any(r.status is EndpointStatus.MISSING for r in results.values())
    return 1 if has_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
