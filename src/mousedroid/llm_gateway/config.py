"""LLM Gateway configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    """LLM Gateway configuration for NL mission translation."""

    enabled: bool = Field(True, description="Enable LLM gateway (disable for headless operation)")
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
    context_length: int = Field(2048, gt=0, description="Model context window size in tokens")
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
        description="Stop sequences (model-specific)",
    )
    max_vx_norm_mps: float = Field(0.5, gt=0, description="Max forward velocity norm (m/s)")
    max_vy_norm_mps: float = Field(0.3, gt=0, description="Max lateral velocity norm (m/s)")
    max_omega_norm_rads: float = Field(2.0, gt=0, description="Max angular velocity norm (rad/s)")
    max_command_len: int = Field(512, gt=0, description="Max NL command length in characters")
    system_prompt: str = Field(
        "You are a Star Wars MSE-6 Mouse Droid navigation controller. "
        "Given a natural language mission, output a JSON object with keys "
        '"vx" (forward, -1 to 1), "vy" (lateral, -1 to 1), "omega" (rotation, -1 to 1). '
        "Respond with ONLY the JSON object.",
        description="System prompt for LLM mission translation",
    )
    injection_patterns: list[str] = Field(
        default_factory=lambda: [
            r"ignore (previous|above|all) instructions?",
            r"system prompt",
            r"you are now",
        ],
        description="Regex patterns to detect prompt injection attempts",
    )
