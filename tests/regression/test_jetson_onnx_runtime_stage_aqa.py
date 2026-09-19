"""AQA — the Jetson image's ONNX Runtime stage (openspec task 2.5).

The defect this pins shut is a *silent* one. `Dockerfile.jetson` runs
`pip install -e "."` with no extras, so the only ONNX Runtime in the image was
whatever `piper-tts` pulled in transitively — a CPU wheel. Meanwhile
`common/onnx_session.py::resolve_providers` falls back to
`("CPUExecutionProvider",)` on an empty intersection and logs it at INFO
(`:94-100`). Put together, `engine: onnx_trt` on the rover would have run
*slower* than the torch baseline while every health check stayed green.

Two properties matter and both are asserted here:

1. The image installs the `[onnx_world_model]` extra, so the version floors stay
   in `pyproject.toml` and are not respelt in the Dockerfile.
2. It does so only when the base image does not already offer an accelerated
   provider. NVIDIA ships the aarch64/L4T `onnxruntime-gpu` wheels on its own
   index and `dustynv/l4t-pytorch:r36.4.0` is expected to carry one; an
   unconditional `pip install` could replace a working L4T build with a
   CPU-only PyPI wheel, which is the very failure this stage prevents.

The stage is deliberately non-fatal. The default `engine: torch` needs no ONNX
Runtime, and the change's task-2.1 ceiling gate closed against enabling
`onnx_trt` (computed end-to-end ceiling 1.0016x-1.0039x), so a missing GPU wheel
must not break the image build. What it must not do is stay *quiet*: the probe
prints the resolved provider list so the build log is the evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.common.onnx_session import DEFAULT_ORT_PROVIDERS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "Dockerfile.jetson"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_EXTRA = "onnx_world_model"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _DOCKERFILE.read_text(encoding="utf-8")


class TestTheExtraIsInstalled:
    """One source of truth for the ONNX Runtime version floors."""

    def test_the_image_installs_the_onnx_world_model_extra(self, dockerfile: str) -> None:
        assert f".[{_EXTRA}]" in dockerfile, (
            f"Dockerfile.jetson must install the [{_EXTRA}] extra; without it the "
            "only ONNX Runtime in the image is the CPU wheel piper-tts pulls in"
        )

    def test_the_extra_exists_in_pyproject(self) -> None:
        assert f"{_EXTRA} = [" in _PYPROJECT.read_text(encoding="utf-8")

    def test_no_onnxruntime_version_literal_in_the_dockerfile(self, dockerfile: str) -> None:
        """Floors belong to pyproject.toml, not to a second place that can drift."""
        offenders = [
            line.strip()
            for line in dockerfile.splitlines()
            if "onnxruntime" in line and ">=" in line
        ]
        assert offenders == [], (
            f"pin onnxruntime in pyproject.toml's [{_EXTRA}] extra, not in the "
            f"Dockerfile: {offenders}"
        )


class TestTheInstallIsProbeGuarded:
    """Never clobber a working L4T runtime the base image already ships."""

    def test_a_provider_probe_precedes_the_install(self, dockerfile: str) -> None:
        probe_at = dockerfile.index("ORT_BASE_IMAGE_PROVIDERS")
        install_at = dockerfile.index(f".[{_EXTRA}]")
        assert probe_at < install_at, (
            "the install must be the probe's fallback branch, not unconditional — "
            "reinstalling over the base image's L4T wheel can downgrade it to a "
            "CPU-only PyPI build"
        )

    def test_the_probe_sources_provider_names_from_the_runtime_module(
        self, dockerfile: str
    ) -> None:
        """A respelt provider list could drift from the chain actually requested."""
        assert "DEFAULT_ORT_PROVIDERS" in dockerfile
        accelerated = [
            provider for provider in DEFAULT_ORT_PROVIDERS if provider != "CPUExecutionProvider"
        ]
        assert accelerated, "the default chain must offer at least one accelerator"
        for provider in accelerated:
            assert provider not in dockerfile, (
                f"{provider} is spelt out in the Dockerfile; read it from "
                "DEFAULT_ORT_PROVIDERS instead"
            )

    def test_the_providers_are_printed_at_the_decision_point(self, dockerfile: str) -> None:
        """Whatever the outcome, the build log must record it."""
        assert "ORT_BASE_IMAGE_PROVIDERS" in dockerfile
        assert "ORT_PROVIDERS_AFTER_EXTRA" in dockerfile


class TestTheFinalReadingComesLast:
    """The probe that decides the install is NOT the authoritative one.

    Stage 2 reads the image as it is at that moment, but Stage 5's
    `pip install piper-tts` runs ~80 lines later and can pull the CPU
    `onnxruntime` wheel over an accelerated build. Without a probe after every
    ORT-affecting install, the build log can print a GPU-capable provider list
    while the shipped image imports CPU ORT — the same silent downgrade this
    stage exists to prevent, just moved later in the build.

    The earlier version of this file asserted only that a provider list was
    printed *somewhere*, which is why it passed while that hole was open.
    """

    # The executable marker, not the bare word: "ORT_FINAL_PROVIDERS" also
    # appears in a comment ~130 lines earlier explaining why the late probe
    # exists, and matching that comment is what made these assertions read the
    # wrong file position on the first attempt.
    _PROBE = "print('ORT_FINAL_PROVIDERS="

    def test_a_final_probe_exists(self, dockerfile: str) -> None:
        assert self._PROBE in dockerfile

    def test_the_final_probe_runs_after_the_piper_tts_install(self, dockerfile: str) -> None:
        piper_at = dockerfile.index('pip install --no-cache-dir "piper-tts"')
        final_at = dockerfile.index(self._PROBE)
        assert piper_at < final_at, (
            "piper-tts can install the CPU onnxruntime wheel, so a provider "
            "reading taken before it is not the final image's"
        )

    def test_the_final_probe_is_the_last_ort_import_in_the_file(self, dockerfile: str) -> None:
        """Anything importing ORT after it would invalidate the reading again."""
        imports = [
            index
            for index, line in enumerate(dockerfile.splitlines())
            if "import onnxruntime" in line
        ]
        assert imports, "premise: the Dockerfile probes onnxruntime at all"
        final_line = next(
            index for index, line in enumerate(dockerfile.splitlines()) if self._PROBE in line
        )
        assert max(imports) <= final_line, (
            "an onnxruntime import appears after the final provider probe; "
            "move the probe below it or the reading is stale again"
        )

    def test_the_final_probe_states_whether_acceleration_survived(self, dockerfile: str) -> None:
        """A raw provider list makes the operator do the comparison."""
        assert "ORT_FINAL_ACCELERATED" in dockerfile


class TestTheStageIsNonFatal:
    """The torch default must never be held hostage to a GPU wheel."""

    def test_the_install_falls_through_to_a_warning(self, dockerfile: str) -> None:
        install_at = dockerfile.index(f".[{_EXTRA}]")
        tail = dockerfile[install_at : install_at + 400]
        assert "WARNING" in tail, (
            'follow this file\'s established `|| echo "WARNING: ..."` idiom for '
            "optional subsystems so a missing wheel does not fail the build"
        )

    def test_the_warning_says_the_torch_path_is_unaffected(self, dockerfile: str) -> None:
        install_at = dockerfile.index(f".[{_EXTRA}]")
        tail = dockerfile[install_at : install_at + 400]
        assert "engine: torch" in tail, (
            "the warning must tell the operator which paths are and are not "
            "degraded, or it reads as a broken build"
        )
