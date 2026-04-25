"""Module entry point — enables ``python -m mousedroid``.

Delegates to :func:`mousedroid.main.cli_entry` so external launchers
(MCP clients, systemd, container ENTRYPOINT, etc.) can use the standard
``python -m mousedroid [args...]`` form.
"""

from __future__ import annotations

from mousedroid.main import cli_entry


if __name__ == "__main__":  # pragma: no cover - entry point
    cli_entry()
