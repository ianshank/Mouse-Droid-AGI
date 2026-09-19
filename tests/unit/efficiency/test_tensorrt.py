"""Tests for TensorRT compilation -- protocol compliance, caching, fallback."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog.testing
import torch
import torch.nn as nn

from mousedroid.config.schema import JetsonConfig
from mousedroid.efficiency.tensorrt import (
    _IDENTITY_UNAVAILABLE,
    JetsonTensorRTCompiler,
    MockTensorRTCompiler,
    TensorRTCompilerProtocol,
    UntrustedEngineCacheError,
    _model_fingerprint,
    _runtime_identity,
    _safe_version,
    cache_dir_is_private,
)

#: POSIX mode bits are meaningless on Windows; ``test-windows`` runs in CI.
_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX permission bits; Windows st_mode carries no equivalent",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model for testing compilation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x)  # type: ignore[no-any-return]


@pytest.fixture
def tiny_model() -> nn.Module:
    return _TinyModel()


@pytest.fixture
def input_shapes() -> dict[str, list[int]]:
    return {"obs": [1, 4]}


@pytest.fixture
def jetson_cfg() -> JetsonConfig:
    return JetsonConfig(tensorrt_enabled=True, precision="fp16", workspace_gb=1.0)


@pytest.fixture
def jetson_cfg_disabled() -> JetsonConfig:
    return JetsonConfig(tensorrt_enabled=False)


@pytest.fixture
def jetson_cfg_fp32() -> JetsonConfig:
    return JetsonConfig(tensorrt_enabled=True, precision="fp32", workspace_gb=1.0)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify that concrete classes satisfy TensorRTCompilerProtocol."""

    def test_mock_satisfies_protocol(self) -> None:
        compiler = MockTensorRTCompiler()
        assert isinstance(compiler, TensorRTCompilerProtocol)

    def test_jetson_satisfies_protocol(self, jetson_cfg: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg)
        assert isinstance(compiler, TensorRTCompilerProtocol)


# ---------------------------------------------------------------------------
# MockTensorRTCompiler
# ---------------------------------------------------------------------------


class TestMockCompiler:
    """Tests for the mock compiler used in testing."""

    def test_is_available(self) -> None:
        compiler = MockTensorRTCompiler()
        assert compiler.is_available() is True

    @pytest.mark.asyncio
    async def test_compile_returns_model(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        compiler = MockTensorRTCompiler()
        result = await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert result is tiny_model

    @pytest.mark.asyncio
    async def test_compile_stores_in_cache(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        compiler = MockTensorRTCompiler()
        await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert len(compiler.compiled_models) == 1

    @pytest.mark.asyncio
    async def test_load_compiled(self) -> None:
        compiler = MockTensorRTCompiler()
        result = await compiler.load_compiled(Path("/fake/path"))
        assert isinstance(result, nn.Module)


# ---------------------------------------------------------------------------
# JetsonTensorRTCompiler
# ---------------------------------------------------------------------------


class TestJetsonCompiler:
    """Tests for the real JetsonTensorRTCompiler with torch2trt mocked out."""

    def test_is_available_when_enabled(self, jetson_cfg: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg)
        result = compiler.is_available()
        assert isinstance(result, bool)

    def test_is_available_when_disabled(self, jetson_cfg_disabled: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg_disabled)
        assert compiler.is_available() is False

    @pytest.mark.asyncio
    async def test_compile_disabled_returns_model(
        self,
        jetson_cfg_disabled: JetsonConfig,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
    ) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg_disabled)
        result = await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert result is tiny_model

    @pytest.mark.asyncio
    async def test_compile_falls_back_to_jit_trace(
        self,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
        tmp_path: Path,
    ) -> None:
        """When torch2trt is unavailable, falls back to JIT tracing."""
        cfg = JetsonConfig(
            tensorrt_enabled=True,
            precision="fp16",
            workspace_gb=1.0,
            tensorrt_cache_dir=tmp_path / "trt_cache",
        )
        compiler = JetsonTensorRTCompiler(cfg)
        with patch("mousedroid.efficiency.tensorrt._TORCH2TRT_AVAILABLE", False):
            result = await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert result is not None

    @pytest.mark.asyncio
    async def test_cache_hit(
        self,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
        tmp_path: Path,
    ) -> None:
        """Second compilation with same params should hit disk cache."""
        cfg = JetsonConfig(
            tensorrt_enabled=True,
            precision="fp16",
            workspace_gb=1.0,
            tensorrt_cache_dir=tmp_path / "trt_cache",
        )
        compiler = JetsonTensorRTCompiler(cfg)

        with patch("mousedroid.efficiency.tensorrt._TORCH2TRT_AVAILABLE", False):
            result1 = await compiler.compile_model(tiny_model, input_shapes, "fp16")
            result2 = await compiler.compile_model(tiny_model, input_shapes, "fp16")

        assert result1 is not None
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_cache_miss_different_precision(
        self,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
        tmp_path: Path,
    ) -> None:
        """Different precision should produce a different cache key."""
        cfg = JetsonConfig(
            tensorrt_enabled=True,
            precision="fp16",
            workspace_gb=1.0,
            tensorrt_cache_dir=tmp_path / "trt_cache",
        )
        compiler = JetsonTensorRTCompiler(cfg)

        with patch("mousedroid.efficiency.tensorrt._TORCH2TRT_AVAILABLE", False):
            result_fp16 = await compiler.compile_model(tiny_model, input_shapes, "fp16")
            result_fp32 = await compiler.compile_model(tiny_model, input_shapes, "fp32")

        assert result_fp16 is not None
        assert result_fp32 is not None

    @pytest.mark.asyncio
    async def test_load_compiled_file_not_found(self, jetson_cfg: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg)
        with pytest.raises(FileNotFoundError):
            await compiler.load_compiled(Path("/nonexistent/engine.pth"))

    @pytest.mark.asyncio
    async def test_load_compiled_from_disk(
        self,
        tiny_model: nn.Module,
        tmp_path: Path,
    ) -> None:
        """Can load a saved model from disk."""
        save_path = tmp_path / "test_engine.pth"
        torch.save(tiny_model, str(save_path))

        cfg = JetsonConfig(
            tensorrt_enabled=True,
            tensorrt_cache_dir=tmp_path,
        )
        compiler = JetsonTensorRTCompiler(cfg)
        loaded = await compiler.load_compiled(save_path)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    """Tests for the model fingerprint function."""

    def test_same_inputs_same_fingerprint(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        fp1 = _model_fingerprint(tiny_model, input_shapes, "fp16")
        fp2 = _model_fingerprint(tiny_model, input_shapes, "fp16")
        assert fp1 == fp2

    def test_different_precision_different_fingerprint(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        fp16 = _model_fingerprint(tiny_model, input_shapes, "fp16")
        fp32 = _model_fingerprint(tiny_model, input_shapes, "fp32")
        assert fp16 != fp32

    def test_different_shapes_different_fingerprint(
        self,
        tiny_model: nn.Module,
    ) -> None:
        fp1 = _model_fingerprint(tiny_model, {"obs": [1, 4]}, "fp16")
        fp2 = _model_fingerprint(tiny_model, {"obs": [2, 4]}, "fp16")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Test that JetsonConfig fields work correctly with the compiler."""

    def test_default_config_values(self) -> None:
        cfg = JetsonConfig()
        assert cfg.tensorrt_enabled is True
        assert cfg.precision == "fp16"
        assert cfg.workspace_gb == 1.0
        assert cfg.tensorrt_cache_dir == Path("/opt/mousedroid/tensorrt_cache")

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        cfg = JetsonConfig(tensorrt_cache_dir=tmp_path / "custom_cache")
        compiler = JetsonTensorRTCompiler(cfg)
        assert compiler._cache_dir == tmp_path / "custom_cache"

    def test_fp32_precision(self) -> None:
        cfg = JetsonConfig(precision="fp32")
        assert cfg.precision == "fp32"

    def test_int8_precision(self) -> None:
        cfg = JetsonConfig(precision="int8")
        assert cfg.precision == "int8"


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    """Test build_tensorrt_compiler factory function."""

    def test_build_returns_protocol(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_tensorrt_compiler

        cfg = Settings(mock_hardware=True)
        compiler = build_tensorrt_compiler(cfg)
        assert isinstance(compiler, TensorRTCompilerProtocol)

    def test_build_disabled_returns_mock(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_tensorrt_compiler

        cfg = Settings(mock_hardware=True, jetson={"tensorrt_enabled": False})
        compiler = build_tensorrt_compiler(cfg)
        assert isinstance(compiler, MockTensorRTCompiler)

    def test_build_enabled_returns_jetson_compiler(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_tensorrt_compiler

        cfg = Settings(mock_hardware=True, jetson={"tensorrt_enabled": True})
        compiler = build_tensorrt_compiler(cfg)
        assert isinstance(compiler, JetsonTensorRTCompiler)


# ---------------------------------------------------------------------------
# Runtime identity in the cache key (peer review D-7)
# ---------------------------------------------------------------------------


#: Each identity component, and a value that must move the fingerprint.
#: Table-driven so a component silently dropped from ``_runtime_identity``
#: fails the row named for it, rather than quietly weakening the cache key.
_IDENTITY_COMPONENTS = [
    ("torch", "torch.__version__", "1.13.0+cu117"),
    ("cuda", "torch.version.cuda", "11.7"),
    ("tensorrt", "mousedroid.efficiency.tensorrt._tensorrt_version", "10.16.2"),
    ("torch2trt", "mousedroid.efficiency.tensorrt._torch2trt_version", "0.5.0"),
    ("compute", "mousedroid.efficiency.tensorrt._device_capability", (8, 7)),
]


class TestRuntimeIdentity:
    """The cache key must describe what COMPILED an engine, not just what was
    compiled.

    Before this, ``_model_fingerprint`` hashed only model structure, shapes
    and precision, so an engine built under one TensorRT/CUDA/driver combo
    matched the cache key of one built under another. ``compile_model``
    treats a match as a hit, and ``docker-compose.jetson.yml`` mounts the cache
    from outside the image (the named volume ``mousedroid_tensorrt_cache``
    since F-051), so an engine outlives the image that produced it.
    """

    def test_identity_is_non_empty_and_stable(self) -> None:
        first = _runtime_identity()
        assert first
        assert first == _runtime_identity(), "a cache key cannot be nondeterministic"

    def test_every_component_is_labelled(self) -> None:
        """Bare values would collide across components (e.g. two ``'12.1'``)."""
        assert all("=" in component for component in _runtime_identity())

    @pytest.mark.parametrize(
        ("label", "target", "value"),
        _IDENTITY_COMPONENTS,
        ids=[component[0] for component in _IDENTITY_COMPONENTS],
    )
    def test_component_change_moves_the_fingerprint(
        self,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
        label: str,
        target: str,
        value: object,
    ) -> None:
        """The D-7 assertion, once per component.

        A component quietly removed from ``_runtime_identity`` fails exactly
        the row named for it, so the failure says which one.
        """
        before = _model_fingerprint(tiny_model, input_shapes, "fp16")
        patcher = (
            patch(target, value) if target.startswith("torch.") else patch(target, lambda: value)
        )
        with patcher:
            after = _model_fingerprint(tiny_model, input_shapes, "fp16")
        assert before != after, f"{label} does not reach the cache key"

    def test_an_absent_component_is_recorded_not_omitted(self) -> None:
        """Omitting it would collide a host that cannot report with one that can.

        This is also the CI and dev path: neither ``tensorrt`` nor
        ``torch2trt`` is installed there, so the sentinel branch is the
        common case and must not raise.
        """
        assert _safe_version(lambda: None) == _IDENTITY_UNAVAILABLE

    def test_a_raising_probe_does_not_break_the_cache_key(self) -> None:
        def _explode() -> str:
            raise RuntimeError("no driver")

        assert _safe_version(_explode) == _IDENTITY_UNAVAILABLE

    def test_fingerprint_still_stable_across_calls(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        """Identity is added without making the key nondeterministic."""
        assert _model_fingerprint(tiny_model, input_shapes, "fp16") == _model_fingerprint(
            tiny_model, input_shapes, "fp16"
        )


# ---------------------------------------------------------------------------
# Cache-directory privacy (peer review D-25)
# ---------------------------------------------------------------------------


def _compiler(cache: Path) -> JetsonTensorRTCompiler:
    return JetsonTensorRTCompiler(JetsonConfig(tensorrt_enabled=True, tensorrt_cache_dir=cache))


class TestCachePrivacy:
    """``weights_only=False`` is arbitrary pickle -- code execution on load.

    The comment guarding it claimed the cache directory "should have
    restricted permissions (0700)". Nothing enforced that: ``mkdir`` passed no
    ``mode``, so it landed at the umask default, and no ``chmod`` existed
    anywhere in ``src/``.
    """

    @_POSIX_ONLY
    async def test_saving_creates_a_private_directory(
        self, tmp_path: Path, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        cache = tmp_path / "trt_cache"
        await _compiler(cache).compile_model(tiny_model, input_shapes, "fp16")
        assert stat.S_IMODE(cache.stat().st_mode) == stat.S_IRWXU

    @_POSIX_ONLY
    async def test_a_loose_directory_is_a_cache_miss_not_an_error(
        self, tmp_path: Path, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        """The correction that matters most here.

        A cache is an optimization, so declining to trust an entry must mean
        *rebuild it*, never *fail*. Raising would drop every existing rig to
        eager PyTorch through ``OptimizedInference._ensure_compiled``'s
        ``except Exception`` -- a silent performance cliff, logged once.
        """
        cache = tmp_path / "trt_cache"
        compiler = _compiler(cache)
        await compiler.compile_model(tiny_model, input_shapes, "fp16")
        os.chmod(cache, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        result = await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert result is not None

    @_POSIX_ONLY
    async def test_the_recompile_self_heals_the_permissions(
        self, tmp_path: Path, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        cache = tmp_path / "trt_cache"
        compiler = _compiler(cache)
        await compiler.compile_model(tiny_model, input_shapes, "fp16")
        os.chmod(cache, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert stat.S_IMODE(cache.stat().st_mode) == stat.S_IRWXU

    @_POSIX_ONLY
    async def test_direct_load_refuses_a_pickle_from_a_loose_directory(
        self, tmp_path: Path
    ) -> None:
        """``load_compiled`` is public, so it is guarded independently of
        ``compile_model``'s miss-and-rebuild."""
        cache = tmp_path / "trt_cache"
        cache.mkdir(mode=stat.S_IRWXU)
        # A torch2trt engine is saved with ``torch.save``, not
        # ``torch.jit.save`` -- that is what makes the pickle path reachable.
        engine = cache / "engine_deadbeef.pth"
        torch.save({"weights": torch.zeros(2)}, str(engine))
        os.chmod(cache, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        with pytest.raises(UntrustedEngineCacheError, match="engine_deadbeef"):
            await _compiler(cache).load_compiled(engine)

    @_POSIX_ONLY
    async def test_a_private_directory_still_loads_a_pickle_engine(self, tmp_path: Path) -> None:
        """Proves the guard is not always-raise -- torch2trt engines must load."""
        cache = tmp_path / "trt_cache"
        cache.mkdir(mode=stat.S_IRWXU)
        os.chmod(cache, stat.S_IRWXU)
        engine = cache / "engine_cafe.pth"
        torch.save({"weights": torch.zeros(2)}, str(engine))

        loaded = await _compiler(cache).load_compiled(engine)
        assert "weights" in loaded

    async def test_a_torchscript_engine_never_reaches_the_pickle_path(
        self, tmp_path: Path, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        """The common case must not depend on the guard at all.

        Without torch2trt installed the compiler JIT-traces, so the cached
        engine IS TorchScript and ``torch.jit.load`` succeeds -- the
        ``except RuntimeError`` branch is never entered. Pinned because it is
        why the guard costs nothing in CI and on any host without TensorRT.
        """
        cache = tmp_path / "trt_cache"
        compiler = _compiler(cache)
        await compiler.compile_model(tiny_model, input_shapes, "fp16")
        engine = next(cache.glob("engine_*.pth"))

        with patch("torch.load", side_effect=AssertionError("pickle path taken")):
            assert await compiler.load_compiled(engine) is not None

    def test_privacy_check_is_true_where_mode_bits_are_meaningless(self, tmp_path: Path) -> None:
        """Windows has no POSIX mode bits; refusing every cache there would be
        a portability bug, not a security control."""
        with patch("mousedroid.efficiency.tensorrt.os.name", "nt"):
            assert cache_dir_is_private(tmp_path) is True

    @_POSIX_ONLY
    def test_a_directory_that_cannot_be_stat_ed_is_not_private(self, tmp_path: Path) -> None:
        """Fail closed: a directory we cannot vouch for is one we do not trust."""
        assert cache_dir_is_private(tmp_path / "does_not_exist") is False


class TestCacheChmodFailure:
    """A cache directory this process cannot chmod must degrade loudly.

    The likely case for the externally-mounted default: Docker created the
    named volume root-owned and the container runs unprivileged. The
    failure is otherwise invisible and self-perpetuating -- an unprivate
    directory makes every ``compile_model`` a cache miss, so the rover
    recompiles on every run with nothing saying why.
    """

    @_POSIX_ONLY
    async def test_a_failed_chmod_does_not_abort_the_save(
        self, tmp_path: Path, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        cache = tmp_path / "trt_cache"
        with patch(
            "mousedroid.efficiency.tensorrt.os.chmod",
            side_effect=PermissionError("not owner"),
        ):
            await _compiler(cache).compile_model(tiny_model, input_shapes, "fp16")
        assert list(cache.glob("engine_*.pth")), "the engine was not cached"

    @_POSIX_ONLY
    async def test_a_failed_chmod_is_reported_distinctly(
        self,
        tmp_path: Path,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
    ) -> None:
        """Not swallowed by the caller's generic ``tensorrt_cache_save_failed``.

        ``structlog.testing.capture_logs`` rather than ``caplog``: this repo
        logs through structlog, whose events do not reach pytest's stdlib
        capture. That is the existing house pattern (see
        ``tests/integration/test_pr109_greet_integration.py``).
        """
        cache = tmp_path / "trt_cache"
        with (
            structlog.testing.capture_logs() as logs,
            patch(
                "mousedroid.efficiency.tensorrt.os.chmod",
                side_effect=PermissionError("not owner"),
            ),
        ):
            await _compiler(cache).compile_model(tiny_model, input_shapes, "fp16")
        assert any(entry.get("event") == "tensorrt_cache_chmod_failed" for entry in logs)
