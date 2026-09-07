# Tasks — HTTP gateway constructor sanitise + CLI filter threading

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/mousedroid/llm_gateway/openai_compatible.py src/mousedroid/factory/llm_gateway.py scripts/translate_mission.py scripts/ask_rover.py tests/unit/llm_gateway/test_openai_compatible_injection_filter.py tests/unit/scripts/test_translate_mission_cli.py tests/unit/scripts/test_ask_rover_cli.py tests/security/test_openai_compatible_cli_sanitize.py tests/regression/test_f037_aqa.py tests/regression/test_f037_backwards_compat.py
bash scripts/validations/F-037.sh
```

Task ordering is binding.

**Phase 1 — Constructor**

- [x] 1.1 `OpenAICompatibleLLMGateway` self-builds `RegexInjectionFilter` when None.
- [x] 1.2 `translate_mission` / `answer_query` always call `sanitize`.
- [x] 1.3 Factory docstring no longer claims HTTP skips sanitisation.

**Phase 2 — CLIs**

- [x] 2.1 `scripts/translate_mission.py` passes `build_injection_filter`.
- [x] 2.2 `scripts/ask_rover.py` passes `build_injection_filter`.

**Phase 3 — Tests + catalog**

- [x] 3.1 Rewrite None-skip unit tests to pin self-build.
- [x] 3.2 Security + CLI threading tests.
- [x] 3.3 Regression pair + `scripts/validations/F-037.sh` + `features.yaml` F-037.
