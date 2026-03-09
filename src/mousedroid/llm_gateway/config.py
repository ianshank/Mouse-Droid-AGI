"""LLM Gateway configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    """LLM Gateway configuration for NL mission translation."""

    model_path: Path = Field(..., description="Path to GGUF model file")
    n_threads: int = Field(4, gt=0, description="CPU threads for inference")
    n_gpu_layers: int = Field(-1, description="GPU layers to offload (-1 = all)")
    max_tokens: int = Field(256, gt=0, description="Max generation tokens")
    temperature: float = Field(0.1, ge=0, le=2, description="Sampling temperature")
    stop_tokens: list[str] = Field(
        default_factory=lambda: ["<|end|>", "<|endoftext|>"],
        description="Stop sequences (model-specific)",
    )
    max_vx_norm_mps: float = Field(0.5, gt=0, description="Max forward velocity norm (m/s)")
    max_vy_norm_mps: float = Field(0.3, gt=0, description="Max lateral velocity norm (m/s)")
    max_omega_norm_rads: float = Field(2.0, gt=0, description="Max angular velocity norm (rad/s)")
    system_prompt: str = Field(
        "You are a Star Wars MSE-6 Mouse Droid navigation controller. "
        "Given a natural language mission, output a JSON object with keys "
        '"vx" (forward, -1 to 1), "vy" (lateral, -1 to 1), "omega" (rotation, -1 to 1). '
        "Respond with ONLY the JSON object.",
        description="System prompt for LLM mission translation",
    )
