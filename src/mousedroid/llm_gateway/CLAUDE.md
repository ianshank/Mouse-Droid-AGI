# LLM Gateway Subsystem — Surface Contract (PR #107 Pattern)

> Translation of operator natural language commands into structured `GoalVector` objectives
> across local and cloud backends.

## Invariants & Design Rules

1. **Protocol Conformance**: Every backend conforms to `LLMGatewayProtocol` (`is_ready`, `start`,
   `translate_mission`, `stop`). Degraded state (`is_degraded`) is read via `getattr`.
2. **Never Raise on Backend Failure**: Return a neutral `GoalVector` and flip `_degraded = True`.
   Reset `_degraded` on a successful round-trip for self-healing.
3. **Explicit Task Cancellation**: Catch `asyncio.CancelledError` before the broad `except Exception`
   so cancellation propagates cleanly during e-stop or mission abort.
4. **Lazy SDK Imports**: Optional backend dependencies (anthropic, openai, etc.) are imported inside
   `start()`, never at module top-level or in `__init__`.
5. **Secret Protection**: API keys use Pydantic `SecretStr`. Unmask via `.get_secret_value()` once
   at client construction; never log secrets.
6. **Prompt Injection Sanitization**: All cloud-hitting backends sanitize user input via
   `RegexInjectionFilter.sanitize()` prior to API egress.
7. **Local-Only Failover**: `LLMConfig.fallback_backend` must target local backends (e.g. `mock`, `ollama`)
   to maintain off-network autonomy.

## Key Files

- `composite_gateway.py` — Multi-backend gateway with automatic fallback.
- `anthropic_gateway.py` / `mock_gateway.py` — Concrete gateway backends.
- `prompt_injection.py` — Pre-egress prompt sanitizer.
- `tests/unit/llm_gateway/` — Unit tests using `sdk=` test seams.
