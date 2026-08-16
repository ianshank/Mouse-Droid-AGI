"""LLM gateway, VLA policy, and mission configuration models.

The NL command LLM gateway (local GGUF / OpenAI-compatible / Anthropic
cloud with local failover), the Vision-Language-Action policy block, the
rule-based mission parser, and the mission lifecycle state machine
(including its LLM-backed adaptive replanner adapters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from mousedroid.config.schema._primitives import VLABackendLiteral


class LLMConfig(BaseModel):
    """LLM Gateway configuration for NL command interface."""

    enabled: bool = Field(True, description="Enable LLM gateway")
    model_path: Path = Field(
        Path("/opt/mousedroid/models/llama-3-8b-instruct.Q4_K_M.gguf"),
        description="Path to GGUF model file",
    )
    model_url: str = Field(
        "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
        "/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        description="URL to download model from",
    )
    model_checksum: str = Field(
        "",
        description="SHA-256 checksum for model file verification (empty=skip)",
    )
    context_length: int = Field(2048, gt=0, description="Model context window in tokens")
    n_threads: int = Field(4, gt=0, description="CPU threads for inference")
    n_gpu_layers: int = Field(-1, description="GPU layers to offload (-1 = all)")
    n_batch: int = Field(512, gt=0, description="Prompt batch size for llama-cpp context")
    max_tokens: int = Field(256, gt=0, description="Max generation tokens")
    temperature: float = Field(0.1, ge=0, le=2, description="Sampling temperature")
    latency_target_ms: float = Field(
        500.0, gt=0, description="Target inference latency in milliseconds"
    )
    stop_tokens: list[str] = Field(
        default_factory=lambda: ["<|end|>", "<|endoftext|>"],
        description="Stop sequences",
    )
    max_command_len: int = Field(512, gt=0, description="Max NL command length in chars")
    max_vx_norm_mps: float = Field(0.5, gt=0, description="Max forward velocity norm (m/s)")
    max_vy_norm_mps: float = Field(0.3, gt=0, description="Max lateral velocity norm (m/s)")
    max_omega_norm_rads: float = Field(
        2.0,
        gt=0,
        description="Max angular velocity norm (rad/s)",
    )
    system_prompt: str = Field(
        "You are a Star Wars MSE-6 Mouse Droid navigation controller. "
        "Given a natural language mission, output a JSON object with keys "
        '"vx" (forward, -1 to 1), "vy" (lateral, -1 to 1), "omega" (rotation, -1 to 1). '
        "Respond with ONLY the JSON object.",
        description="System prompt for LLM mission translation",
    )
    query_system_prompt: str = Field(
        "You are Rocky, a friendly Star Wars MSE-6 Mouse Droid assistant. "
        "Answer the operator's question concisely in one or two short sentences. "
        "You are a small rover; you cannot perform actions from this channel — "
        "this is question-and-answer only, not a command channel.",
        description=(
            "System prompt for the conversational ``answer_query`` path "
            "(free-text Q&A), kept separate from ``system_prompt`` so the "
            "navigation translator keeps emitting JSON while the query path "
            "returns prose. Used by every backend's ``answer_query``."
        ),
    )
    query_max_tokens: int = Field(
        256,
        gt=0,
        description=(
            "Max generation tokens for the ``answer_query`` conversational "
            "path. Separate from ``max_tokens`` (which sizes the terse JSON "
            "GoalVector response) so operators can allow longer prose answers "
            "without enlarging every navigation translation."
        ),
    )
    injection_patterns: list[str] = Field(
        default_factory=lambda: [
            r"ignore (previous|above|all) instructions?",
            r"system prompt",
            r"you are now",
        ],
        description="Regex patterns to detect prompt injection attempts",
    )

    # Tier C2.3 — OpenAI-compatible HTTP backend knobs.
    backend: Literal["llama_cpp", "openai_compatible", "anthropic"] = Field(
        "llama_cpp",
        description=(
            "LLM backend dispatch. Default ``llama_cpp`` preserves pre-Tier-"
            "C2.3 behaviour — ``build_llm_gateway`` instantiates the existing "
            "in-process GGUF loader. ``openai_compatible`` instantiates the "
            "Tier C2.3 ``OpenAICompatibleLLMGateway`` which talks HTTP to "
            "``{base_url}/v1/chat/completions`` (Ollama 0.1.18+ exposes this "
            "endpoint; LM Studio and OpenAI also conform). ``anthropic`` "
            "instantiates the ``AnthropicLLMGateway`` (Claude Messages API) "
            "for cloud deliberative mission translation — it reuses "
            "``model_name`` (a Claude model id, e.g. "
            "``claude-haiku-4-5``), ``api_key`` (or the ``ANTHROPIC_API_KEY`` "
            "env var when unset), ``system_prompt``, ``temperature``, "
            "``max_tokens`` and ``request_timeout_s``. The ``anthropic`` SDK "
            "is an OPTIONAL dependency — install with "
            '``pip install -e ".[anthropic]"``.'
        ),
    )
    base_url: str = Field(
        "http://127.0.0.1:11434",
        description=(
            "Base URL for the ``openai_compatible`` backend. Default targets "
            "the canonical local Ollama port. Env override: "
            "``MOUSEDROID_LLM__BASE_URL``. Examples: "
            "``http://localhost:1234`` (LM Studio), "
            "``https://api.openai.com`` (OpenAI cloud)."
        ),
    )
    model_name: str = Field(
        "gemma-4-e4b",
        description=(
            "Model identifier passed in the ``model`` field of "
            "``/v1/chat/completions``. Default matches the operator's local "
            "Ollama tag. Env override: ``MOUSEDROID_LLM__MODEL_NAME``."
        ),
    )
    api_key: SecretStr | None = Field(
        None,
        description=(
            "Optional bearer token forwarded as ``Authorization: Bearer "
            "<key>``. ``None`` (default) is correct for anonymous local "
            "Ollama. Env override: ``MOUSEDROID_LLM__API_KEY``. Stored as "
            "``SecretStr`` so it never appears in repr / structlog output."
        ),
    )
    request_timeout_s: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Wall-clock timeout for a single ``/v1/chat/completions`` POST "
            "(``openai_compatible``) or ``messages.create`` call "
            "(``anthropic``). Default 10s covers the ``latency_target_ms`` "
            "(500ms) with 20x headroom for Jetson-on-battery deployments. "
            "Smaller than the orchestrator's tick budget so a slow LLM never "
            "starves the control loop. Cloud Claude round-trips are seconds — "
            "raise this (e.g. 15-30s) when ``backend='anthropic'``."
        ),
    )

    # Tier C-rover — cloud-primary / local-secondary failover knobs.
    fallback_backend: Literal["none", "llama_cpp", "openai_compatible"] = Field(
        "none",
        description=(
            "Optional LOCAL backend used when the primary ``backend`` is "
            "unavailable or degraded (e.g. the Jetson is off-network and "
            "``backend='anthropic'`` cannot reach the Claude API). Default "
            "``none`` disables failover so existing single-backend "
            "deployments are byte-identical. When set, "
            "``build_llm_gateway`` wraps the primary + this secondary in a "
            "``FallbackLLMGateway`` composite. Restricted to local backends "
            "(``llama_cpp`` GGUF, or ``openai_compatible`` pointed at a local "
            "Ollama / LM Studio) so the rover stays autonomous without "
            "connectivity. Set equal to ``backend`` is a no-op (the composite "
            "is skipped)."
        ),
    )
    fallback_model_name: str | None = Field(
        None,
        description=(
            "Optional ``model_name`` override applied ONLY to the "
            "``fallback_backend`` gateway. ``None`` (default) reuses "
            "``model_name``. Needed when the primary and secondary backends "
            "want different model identifiers — e.g. primary "
            "``backend='anthropic'`` with ``model_name='claude-haiku-4-5'`` "
            "and ``fallback_backend='openai_compatible'`` needing a local "
            "Ollama tag here. The canonical ``anthropic`` -> ``llama_cpp`` "
            "pairing needs no override (llama_cpp loads ``model_path``, not "
            "``model_name``)."
        ),
    )
    fallback_retry_cooldown_s: float = Field(
        30.0,
        gt=0.0,
        description=(
            "Seconds the ``FallbackLLMGateway`` composite waits before "
            "re-probing a degraded primary backend. A mobile rover sees "
            "transient WAN dropouts, so once the cloud primary degrades the "
            "composite periodically re-attempts it (rather than pinning to "
            "the local secondary until the next process restart). A "
            "successful re-probe clears the primary's degraded state and "
            "resumes cloud serving. Only consulted when "
            "``fallback_backend != 'none'``."
        ),
    )


class VLAConfig(BaseModel):
    """Vision-Language-Action policy configuration (Phase 3a).

    Default ``backend = "none"`` keeps the VLA branch fully disabled so
    pre-Phase-3a behavior is preserved byte-identical. Selecting
    ``"mock"`` activates the deterministic ``MockVLA`` reference; the
    Phase 3b ``"distilled_onnx"`` backend will reuse this same config
    block.
    """

    # ``model_filename`` / ``model_repo_id`` etc. clash with pydantic's
    # default protected ``model_`` namespace; opt out so the warnings do
    # not fire under tests / CI.
    model_config = {"protected_namespaces": ()}

    backend: VLABackendLiteral = Field(
        "none",
        description=(
            "VLA backend. 'none' (default) leaves the VLA branch unwired. "
            "'mock' selects the in-tree zero-dependency MockVLA. "
            "'distilled_onnx' is reserved for Phase 3b. "
            "See ``VLABackendLiteral`` in schema.py for the canonical type."
        ),
    )
    canned_action: list[float] | None = Field(
        None,
        description=(
            "Optional fixed action vector for MockVLA. Length must equal "
            "model.action_dim. None => zero action."
        ),
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence value emitted by MockVLA on every predict() call.",
    )
    fallback_on_timeout: bool = Field(
        True,
        description=(
            "When True, a VLA inference timeout transparently falls back "
            "to the nav_agent. When False, the orchestrator emits a "
            "'vla_timeout_safe_stop' event and returns a zero action so "
            "the safety monitor can escalate on the next tick."
        ),
    )
    # ----- Phase 3b: distilled ONNX backend -----
    model_repo_id: str | None = Field(
        None,
        description=(
            "HuggingFace repo id (e.g., 'lerobot/smolvla') used by "
            "weights_manager.download_weights_from_huggingface to fetch "
            "the ONNX file when ``backend='distilled_onnx'``. None => "
            "expect ``model_path`` to already exist locally under "
            "``cache_dir``."
        ),
    )
    model_filename: str = Field(
        "model.onnx",
        description="Filename of the ONNX graph inside the HF repo / cache dir.",
    )
    cache_dir: str | None = Field(
        "weights/vla",
        description=(
            "Local directory containing the ONNX model. Defaults to "
            "'weights/vla'; override via YAML to relocate the cache."
        ),
    )
    providers: list[str] | None = Field(
        None,
        description=(
            "Optional explicit ORT execution-provider chain. None => the "
            "default fallback chain "
            "['TensorrtExecutionProvider', 'CUDAExecutionProvider', "
            "'CPUExecutionProvider']. Unavailable providers are skipped "
            "automatically by ``DistilledVLAOnnx.warmup``."
        ),
    )
    warmup_iterations: int = Field(
        1,
        ge=0,
        description=(
            "Number of dummy inference passes after session creation to "
            "prime CUDA/TensorRT kernels. 0 disables warmup."
        ),
    )
    h_input_name: str = Field(
        "h",
        description="ONNX input name for the deterministic latent ``h``.",
    )
    z_input_name: str = Field(
        "z",
        description="ONNX input name for the stochastic latent ``z``.",
    )
    action_output_name: str = Field(
        "action",
        description="ONNX output name for the action tensor.",
    )


class MissionParserConfig(BaseModel):
    """NL mission parser configuration for speed and confidence mappings."""

    speed_map: dict[str, float] = Field(
        default_factory=lambda: {
            "slow": 0.3,
            "slowly": 0.3,
            "half speed": 0.5,
            "fast": 0.8,
            "quickly": 0.8,
            "full speed": 1.0,
        },
        description="Mapping of speed modifier keywords to normalised speed values",
    )
    default_speed: float = Field(0.5, gt=0, le=1, description="Default speed when no modifier")
    patrol_speed: float = Field(0.5, gt=0, le=1, description="Default patrol velocity (m/s)")
    avoid_speed: float = Field(0.3, gt=0, le=1, description="Default obstacle avoidance velocity")
    stop_confidence: float = Field(1.0, ge=0, le=1, description="Confidence for stop commands")
    direction_confidence: float = Field(
        0.9,
        ge=0,
        le=1,
        description="Confidence for directional movement commands",
    )
    patrol_confidence: float = Field(0.8, ge=0, le=1, description="Confidence for patrol commands")
    avoid_confidence: float = Field(
        0.7,
        ge=0,
        le=1,
        description="Confidence for obstacle avoidance commands",
    )
    llm_fallback_confidence: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Minimum parser confidence to skip LLM fallback",
    )


class MissionReplannerConfig(BaseModel):
    """Tier C2.3 — LLM-backed mission replanner adapter configuration.

    Tunables for ``LLMGatewayMissionReplanner`` (built by
    :func:`mousedroid.factory.build_mission_replanner` when
    ``mission.llm_replanner_enabled`` is ``True``). Distinct from
    :class:`LLMReplannerConfig` defined later in this module — that one
    configures the robot-arm symbolic planner. The two share a naming
    prefix but are unrelated subsystems.
    """

    max_prompt_chars: int = Field(
        512,
        gt=0,
        description=(
            "Maximum characters in the augmented goal_text prompt forwarded "
            "to the LLM gateway. The adapter clips longer prompts at this "
            "boundary so a runaway goal_text cannot exceed the gateway's "
            "context window. Default 512 mirrors the rule-based parser's "
            "command-length policy."
        ),
    )
    include_progress_in_prompt: bool = Field(
        True,
        description=(
            "When True (default), the adapter appends "
            "``(last_progress=<float>)`` to the prompt so the LLM sees the "
            "stall context. Operators can disable when their LLM is tuned "
            "for raw goals only."
        ),
    )


class MissionConfig(BaseModel):
    """Mission lifecycle state-machine configuration (Tier C2 / C2.2 / C2.3).

    Drives the ``MissionLifecycle`` state machine that wraps
    :class:`InMemoryTaskTracker` and adds VLM-driven goal-progress feedback
    plus LLM-driven adaptive replan. When ``replan_enabled=False`` (the
    default), the lifecycle never trips into ``REPLANNING`` and never calls
    the LLM gateway — existing deployments produce byte-identical pre-PR
    behaviour because the orchestrator does not build a lifecycle at all
    when this block is at defaults.

    Tier C2.3 adds four fields (``vlm_progress_enabled``,
    ``vlm_mock_progress_value``, ``llm_replanner_enabled``, ``replanner``)
    that gate the VLM progress head + LLM replanner wiring inside
    :func:`build_orchestrator`. All four default to safe values so
    existing YAML loads unchanged.
    """

    replan_enabled: bool = Field(
        False,
        description=(
            "Enable adaptive LLM-driven replan when VLM progress stalls. "
            "Default ``False`` preserves byte-identical pre-PR behaviour."
        ),
    )
    success_threshold: float = Field(
        0.90,
        ge=0.0,
        le=1.0,
        description=(
            "VLM progress score must cross this value to transition the mission to ``SUCCEEDED``."
        ),
    )
    stall_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description=(
            "VLM progress score below this value counts as a stalled tick. "
            "``stall_window_ticks`` consecutive stalls trip replan."
        ),
    )
    stall_window_ticks: int = Field(
        30,
        gt=0,
        description=(
            "Number of consecutive low-progress ticks before the lifecycle "
            "transitions to ``REPLANNING``. At 30 Hz this is ~1 second."
        ),
    )
    max_replans_per_mission: int = Field(
        3,
        ge=0,
        description=(
            "Hard cap on replans per mission. Once exceeded the lifecycle "
            "transitions to ``FAILED`` with reason='replan_limit_exceeded'."
        ),
    )
    vlm_progress_enabled: bool = Field(
        False,
        description=(
            "Tier C2.3: build a ``VLMProgressHead`` for the mission "
            "lifecycle. Default False preserves pre-Tier-C2.3 byte-identical "
            "behaviour (factory short-circuits to None). When True the head "
            "uses ``MockVLMProgress(mock_progress_value)`` by default; a "
            "real VLM backend is a separate sprint."
        ),
    )
    vlm_mock_progress_value: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Constant value the default ``MockVLMProgress`` backend returns. "
            "Default 0.95 sits above the default ``success_threshold=0.90`` "
            "so a smoke-mode mission transitions to SUCCEEDED on the first "
            "scored tick — useful for the boot-time smoke test."
        ),
    )
    llm_replanner_enabled: bool = Field(
        False,
        description=(
            "Tier C2.3: build an ``LLMGatewayMissionReplanner`` for the "
            "mission lifecycle. Default False preserves pre-Tier-C2.3 "
            "behaviour. Requires the LLM gateway to be enabled — when "
            "``cfg.llm.enabled is False`` the factory still short-circuits "
            "to None even with this flag True (with a structured warning)."
        ),
    )
    replanner: MissionReplannerConfig = Field(
        default_factory=MissionReplannerConfig,
        description="Sub-block tuning the LLM replanner adapter.",
    )


class LLMReplannerConfig(BaseModel):
    """Configuration for the LLM-backed arm replanner.

    Disabled by default; when enabled, ``backend`` selects the concrete
    implementation. ``model``, ``max_tokens``, ``temperature`` and the
    request envelope come from this config so no values are hardcoded
    in the backend modules.
    """

    enabled: bool = Field(
        False,
        description="Enable LLM-backed replanning (None=disabled)",
    )
    backend: Literal["null", "llama", "anthropic"] = Field(
        "null",
        description="Replanner backend selection",
    )
    model: str = Field(
        "claude-sonnet-4-6",
        description="Model identifier passed to the backend",
    )
    max_tokens: int = Field(
        1024,
        gt=0,
        description="Per-request max tokens",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    system_prompt: str = Field(
        "",
        description="System prompt passed to the backend",
    )
    api_key_env_var: str = Field(
        "ANTHROPIC_API_KEY",
        description="Env var holding the API key (Anthropic backend only)",
    )
    request_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Per-request timeout (s)",
    )
    max_retries: int = Field(
        3,
        ge=0,
        description="Max exponential-backoff retries on transient errors",
    )
