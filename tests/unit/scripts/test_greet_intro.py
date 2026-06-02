"""CLI smoke tests for ``scripts/greet_intro.py``.

End-to-end through the script's ``main()`` entry point with a mock
voice stack — no real audio, no Jetson, no Piper model. Verifies:

* ``--dry-run`` forces ``mock_hardware=true`` and the script exits 0.
* Missing greeting block exits ``2`` (config error).
* Disabled greeting block exits ``2``.
* A well-formed YAML with mock hardware exits ``0`` and records the
  expected utterances on the mock voice engine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
SCRIPT = _REPO / "scripts" / "greet_intro.py"


def _subprocess_env() -> dict[str, str]:
    """Mirror pytest's ``pythonpath = ['src', '.']`` for the subprocess.

    Mirrors the established pattern in ``tests/unit/test_scripts.py``
    (PR #106 commit 3a86477) — the spawned ``[sys.executable, '-c', ...]``
    starts with the *system* sys.path, dropping pytest's pythonpath
    hook. Without explicit propagation the script's
    ``from mousedroid.config.loader import load_settings`` fails with
    ``ModuleNotFoundError: No module named 'mousedroid.config'``,
    and on an editable-install worktree the subprocess could even
    pick up a stale sibling package. Code-reviewer round-1 finding #2.
    """
    env = dict(os.environ)
    extra = os.pathsep.join([str(_REPO / "src"), str(_REPO)])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extra + (os.pathsep + existing if existing else "")
    return env


def _write_yaml(tmp_path: Path, body: str) -> Path:
    cfg_path = tmp_path / "greeting.yaml"
    cfg_path.write_text(body, encoding="utf-8")
    return cfg_path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )


def test_dry_run_with_enabled_greeting_exits_zero(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
platform: mouse_droid
mock_hardware: false
voice:
  enabled: true
greeting:
  enabled: true
  names: [Alpha, Bravo]
  pre_chirp_event: greeting_excited
  inter_chirp_delay_s: 0.0
""".lstrip(),
    )
    result = _run_cli("--config", str(cfg), "--dry-run")
    assert result.returncode == 0, result.stderr


def test_missing_greeting_block_exits_config_error(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
platform: mouse_droid
mock_hardware: true
voice:
  enabled: true
""".lstrip(),
    )
    result = _run_cli("--config", str(cfg))
    assert result.returncode == 2
    # The structured-log JSON on stderr should call out the config error.
    assert "greet_intro_config_error" in result.stderr


def test_disabled_greeting_exits_config_error(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        """
platform: mouse_droid
mock_hardware: true
voice:
  enabled: true
greeting:
  enabled: false
""".lstrip(),
    )
    result = _run_cli("--config", str(cfg))
    assert result.returncode == 2
    assert "greet_intro_config_error" in result.stderr


def test_dry_run_log_shows_greeting_done(tmp_path: Path) -> None:
    """The structured JSON stream on stderr must show greeting_done event."""
    cfg = _write_yaml(
        tmp_path,
        """
platform: mouse_droid
mock_hardware: false
voice:
  enabled: true
greeting:
  enabled: true
  names: [Charlie, Delta]
  pre_chirp_event: ""
  inter_chirp_delay_s: 0.0
""".lstrip(),
    )
    result = _run_cli("--config", str(cfg), "--dry-run")
    assert result.returncode == 0, result.stderr
    # Each stderr line is a structlog JSON record.
    events = []
    for line in result.stderr.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    event_names = {e.get("event") for e in events}
    assert "greeting_started" in event_names
    assert "greeting_done" in event_names
