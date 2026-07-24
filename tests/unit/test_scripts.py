"""Tests for scripts/ — syntax validation, structure, and importability checks.

These tests validate the shell scripts (ci.sh, deploy_jetson.sh, flash_esp32.sh,
mousedroid.service) and Python-importable training module CLIs on Windows without
requiring Bash/systemd/hardware to be present. They verify structure, required
content, and Python module import integrity.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root relative to this test file
_REPO = Path(__file__).parent.parent.parent
_SCRIPTS = _REPO / "scripts"
_TRAINING = _REPO / "training"


def _subprocess_env() -> dict[str, str]:
    """Build a subprocess env that mirrors pytest's ``pythonpath = ['src', '.']``.

    The ``training/`` CLI scripts ``import mousedroid.<...>``, which only
    resolves when ``src/`` is on ``PYTHONPATH``. Pytest configures that
    via ``[tool.pytest.ini_options] pythonpath`` in ``pyproject.toml`` —
    but a bare ``subprocess.run([sys.executable, '-c', ...])`` spawns
    Python with the *system* sys.path, dropping that pytest hook. Without
    the explicit env propagation here, the import-smoke tests below all
    fail with ``ModuleNotFoundError: No module named 'mousedroid.config'``
    even though the package is reachable from the parent test process.
    """
    env = dict(os.environ)
    extra = os.pathsep.join([str(_REPO / "src"), str(_REPO)])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extra + (os.pathsep + existing if existing else "")
    return env


# ---------------------------------------------------------------------------
# Script file existence gates
# ---------------------------------------------------------------------------


class TestScriptFilesExist:
    def test_ci_sh_exists(self) -> None:
        assert (_SCRIPTS / "ci.sh").is_file()

    def test_check_no_hardcoded_values_py_exists(self) -> None:
        assert (_SCRIPTS / "check_no_hardcoded_values.py").is_file()

    def test_check_settings_identity_py_exists(self) -> None:
        assert (_SCRIPTS / "check_settings_identity.py").is_file()

    def test_deploy_jetson_sh_exists(self) -> None:
        assert (_SCRIPTS / "deploy_jetson.sh").is_file()

    def test_flash_esp32_sh_exists(self) -> None:
        assert (_SCRIPTS / "flash_esp32.sh").is_file()

    def test_mousedroid_service_exists(self) -> None:
        assert (_SCRIPTS / "mousedroid.service").is_file()


# ---------------------------------------------------------------------------
# Shell script structural validation (no Bash needed)
# ---------------------------------------------------------------------------


class TestCiSh:
    @pytest.fixture
    def content(self) -> str:
        return (_SCRIPTS / "ci.sh").read_text(encoding="utf-8")

    def test_shebang(self, content: str) -> None:
        assert content.startswith("#!/bin/bash")

    def test_errexit_set(self, content: str) -> None:
        assert "set -euo pipefail" in content

    def test_lint_step_present(self, content: str) -> None:
        assert "ruff check" in content

    def test_mypy_step_present(self, content: str) -> None:
        assert "mypy src/" in content

    def test_unit_test_step_present(self, content: str) -> None:
        assert "pytest tests/unit" in content

    def test_integration_test_step(self, content: str) -> None:
        assert re.search(
            r"(?:python\s+-m\s+)?pytest\b[^\n]*\btests/integration\b",
            content,
        )

    def test_e2e_step_present(self, content: str) -> None:
        assert "pytest tests/e2e/" in content

    def test_coverage_gate_present(self, content: str) -> None:
        assert "--cov-fail-under=85" in content

    def test_mock_hardware_env_set(self, content: str) -> None:
        assert "MOUSEDROID_MOCK_HARDWARE=true" in content

    def test_health_check_step_present(self, content: str) -> None:
        assert "mousedroid.main --health-check" in content

    def test_hardcoded_value_gate_present(self, content: str) -> None:
        assert "check_no_hardcoded_values.py" in content

    def test_settings_identity_gate_present(self, content: str) -> None:
        assert "check_settings_identity.py" in content

    def test_workforce_config_stage_present(self, content: str) -> None:
        assert "tools.claude_hooks.config" in content

    def test_workforce_hook_typecheck_present(self, content: str) -> None:
        # Governance code is held to --strict from day one, scoped to the hook
        # package because tools/ as a whole is not yet strict-clean.
        assert "mypy tools/claude_hooks/" in content

    def test_workforce_hook_coverage_stage_present(self, content: str) -> None:
        # The repo-wide gate measures src/mousedroid only; without this separate
        # invocation the hook package would ship unmeasured.
        assert "--cov=tools/claude_hooks" in content
        assert "--cov-branch" in content


class TestDeployJetsonSh:
    @pytest.fixture
    def content(self) -> str:
        return (_SCRIPTS / "deploy_jetson.sh").read_text(encoding="utf-8")

    def test_shebang(self, content: str) -> None:
        assert content.startswith("#!/bin/bash")

    def test_errexit_set(self, content: str) -> None:
        assert "set -euo pipefail" in content

    def test_installs_venv(self, content: str) -> None:
        assert "python3 -m venv" in content

    def test_installs_mousedroid(self, content: str) -> None:
        assert 'pip" install' in content or "pip install" in content

    def test_installs_hardware_extras(self, content: str) -> None:
        assert "hardware" in content

    def test_deploys_systemd_service(self, content: str) -> None:
        assert "systemctl" in content or "mousedroid.service" in content

    def test_config_files_copied(self, content: str) -> None:
        assert "default.yaml" in content

    def test_health_check_present(self, content: str) -> None:
        assert "health-check" in content


class TestFlashEsp32Sh:
    @pytest.fixture
    def content(self) -> str:
        return (_SCRIPTS / "flash_esp32.sh").read_text(encoding="utf-8")

    def test_shebang(self, content: str) -> None:
        assert content.startswith("#!/bin/bash")

    def test_errexit_set(self, content: str) -> None:
        assert "set -euo pipefail" in content

    def test_requires_port_argument(self, content: str) -> None:
        assert "${1:?" in content  # bash required arg

    def test_requires_firmware_argument(self, content: str) -> None:
        assert "${2:?" in content

    def test_checks_firmware_file_exists(self, content: str) -> None:
        assert "! -f" in content

    def test_checks_serial_port(self, content: str) -> None:
        assert "! -c" in content

    def test_uses_esptool(self, content: str) -> None:
        assert "esptool.py" in content

    def test_baud_rate_present(self, content: str) -> None:
        assert "460800" in content

    def test_installs_esptool_if_missing(self, content: str) -> None:
        assert "pip install esptool" in content


class TestMousedroidService:
    @pytest.fixture
    def content(self) -> str:
        return (_SCRIPTS / "mousedroid.service").read_text(encoding="utf-8")

    def test_is_systemd_unit_file(self, content: str) -> None:
        assert "[Unit]" in content or "[Service]" in content

    def test_has_service_section(self, content: str) -> None:
        assert "[Service]" in content

    def test_has_exec_start(self, content: str) -> None:
        assert "ExecStart" in content

    def test_references_mousedroid(self, content: str) -> None:
        assert "mousedroid" in content.lower()


# ---------------------------------------------------------------------------
# Python training modules — importability smoke tests
# ---------------------------------------------------------------------------


class TestTrainingModuleImports:
    """Ensure all training modules are syntactically valid and importable."""

    @pytest.mark.parametrize(
        "module_file",
        [
            "collect_annotations.py",
            "data_generator.py",
            "rssm_dataset.py",
            "train_bdi.py",
            "train_constitutional_rl.py",
            "train_rssm.py",
            "warmstart_policy.py",
        ],
    )
    def test_syntax_valid(self, module_file: str) -> None:
        """All training Python files must parse without SyntaxError."""
        src = (_TRAINING / module_file).read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"{module_file} has a syntax error: {exc}")

    def test_collect_annotations_importable(self) -> None:
        _import = "from training.collect_annotations import INTENTION_LABELS, label_intention"
        result = subprocess.run(
            [sys.executable, "-c", _import],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_data_generator_importable(self) -> None:
        _import = (
            "from training.data_generator import SyntheticSequenceGenerator, _bundle_to_tensors"
        )
        result = subprocess.run(
            [sys.executable, "-c", _import],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_rssm_dataset_importable(self) -> None:
        _import = "from training.rssm_dataset import RSSMSequenceDataset"
        result = subprocess.run(
            [sys.executable, "-c", _import],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_train_bdi_importable(self) -> None:
        _import = "from training.train_bdi import train_belief_encoder, train_bdi"
        result = subprocess.run(
            [sys.executable, "-c", _import],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_train_constitutional_rl_importable(self) -> None:
        _import = "from training.train_constitutional_rl import _gae, _ppo_update"
        result = subprocess.run(
            [sys.executable, "-c", _import],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_train_rssm_importable(self) -> None:
        _import = "from training.train_rssm import train_rssm"
        result = subprocess.run(
            [sys.executable, "-c", _import],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_warmstart_policy_importable(self) -> None:
        _import = (
            "from training.warmstart_policy import warmstart_policy, compute_latent_statistics"
        )
        result = subprocess.run(
            [sys.executable, "-c", _import],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Training module content / contract validation
# ---------------------------------------------------------------------------


class TestTrainRSSMContent:
    @pytest.fixture
    def content(self) -> str:
        return (_TRAINING / "train_rssm.py").read_text()

    def test_has_train_rssm_function(self, content: str) -> None:
        assert "def train_rssm(" in content

    def test_has_main_entrypoint(self, content: str) -> None:
        assert 'if __name__ == "__main__"' in content

    def test_uses_argparse(self, content: str) -> None:
        assert "argparse" in content

    def test_saves_checkpoint(self, content: str) -> None:
        assert "torch.save" in content

    def test_uses_adam_optimizer(self, content: str) -> None:
        assert "Adam" in content


class TestTrainBDIContent:
    @pytest.fixture
    def content(self) -> str:
        return (_TRAINING / "train_bdi.py").read_text()

    def test_has_train_bdi_function(self, content: str) -> None:
        assert "def train_bdi(" in content

    def test_has_belief_encoder(self, content: str) -> None:
        assert "def train_belief_encoder(" in content

    def test_has_desire_encoder(self, content: str) -> None:
        assert "def train_desire_encoder(" in content

    def test_has_intention_predictor(self, content: str) -> None:
        assert "def train_intention_predictor(" in content

    def test_has_affect_estimator(self, content: str) -> None:
        assert "def train_affect_estimator(" in content

    def test_saves_npz_files(self, content: str) -> None:
        assert "np.savez" in content


class TestTrainConstitutionalRLContent:
    @pytest.fixture
    def content(self) -> str:
        return (_TRAINING / "train_constitutional_rl.py").read_text()

    def test_has_gae_function(self, content: str) -> None:
        assert "def _gae(" in content

    def test_has_ppo_update_function(self, content: str) -> None:
        assert "def _ppo_update(" in content

    def test_has_train_constitutional_rl(self, content: str) -> None:
        assert "def train_constitutional_rl(" in content

    def test_uses_three_laws_checker(self, content: str) -> None:
        assert "RoboticsLawChecker" in content

    def test_uses_constitutional_checker(self, content: str) -> None:
        assert "ConstitutionalChecker" in content

    def test_saves_policy_and_value(self, content: str) -> None:
        assert "policy.save" in content
        assert "value_fn.save" in content


class TestWarmstartPolicyContent:
    @pytest.fixture
    def content(self) -> str:
        return (_TRAINING / "warmstart_policy.py").read_text()

    def test_has_warmstart_function(self, content: str) -> None:
        assert "def warmstart_policy(" in content

    def test_has_compute_latent_stats(self, content: str) -> None:
        assert "def compute_latent_statistics(" in content

    def test_has_tune_ucb(self, content: str) -> None:
        assert "def tune_ucb(" in content

    def test_has_run_warmstart(self, content: str) -> None:
        assert "def run_warmstart(" in content

    def test_ucb_candidates_present(self, content: str) -> None:
        # The implementation should try multiple UCB values
        assert "candidates" in content


class TestCollectAnnotationsContent:
    @pytest.fixture
    def content(self) -> str:
        return (_TRAINING / "collect_annotations.py").read_text()

    def test_intention_labels_has_10_items(self) -> None:
        from training.collect_annotations import INTENTION_LABELS

        assert len(INTENTION_LABELS) == 10

    def test_protect_human_label_present(self) -> None:
        from training.collect_annotations import INTENTION_LABELS

        assert "protect_human" in INTENTION_LABELS

    def test_obey_command_label_present(self) -> None:
        from training.collect_annotations import INTENTION_LABELS

        assert "obey_command" in INTENTION_LABELS

    def test_requires_mock_hardware(self, content: str) -> None:
        assert "mock_hardware" in content

    def test_saves_npz(self, content: str) -> None:
        assert "np.savez" in content


class TestDataGeneratorContent:
    @pytest.fixture
    def content(self) -> str:
        return (_TRAINING / "data_generator.py").read_text()

    def test_has_synthetic_sequence_generator(self, content: str) -> None:
        assert "class SyntheticSequenceGenerator" in content

    def test_requires_mock_hardware(self, content: str) -> None:
        assert "mock_hardware" in content

    def test_saves_pt_file(self, content: str) -> None:
        assert "torch.save" in content

    def test_bundle_to_tensors_present(self, content: str) -> None:
        assert "_bundle_to_tensors" in content


class TestRSSMDatasetContent:
    @pytest.fixture
    def content(self) -> str:
        return (_TRAINING / "rssm_dataset.py").read_text()

    def test_extends_dataset(self, content: str) -> None:
        assert "Dataset" in content

    def test_has_padding_for_short_seqs(self, content: str) -> None:
        # Zero-padding logic described in docstring
        assert "zeros" in content or "pad" in content.lower()

    def test_has_len_and_getitem(self, content: str) -> None:
        assert "__len__" in content
        assert "__getitem__" in content

    def test_returns_sequence_batch(self, content: str) -> None:
        # Dataset now returns a SequenceBatch dict keyed by modality name
        # (vision, ultrasonic, motor_state, valid_mask, lidar, actions) so the
        # sequence-of-Tensor tuple shape is replaced by a typed dict alias.
        assert "SequenceBatch" in content
        assert "dict[str, Tensor]" in content
