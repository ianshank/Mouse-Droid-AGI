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
