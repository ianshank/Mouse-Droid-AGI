# C4 Component — Validation Efficiency (latency stats · trend store · phase caching)

> The runtime/resource-efficiency layer bolted onto the existing Jetson
> validation harness. It answers three questions the binary, single-shot
> harness could not: *how slow is the tail?* (latency percentiles), *is the
> rover degrading over time?* (run-over-run trend store), and *can we skip work
> that hasn't changed?* (phase-1 caching). All three are additive and opt-in —
> defaults preserve byte-identical legacy behaviour.
>
> Companion to `docs/architecture/c4-usbc-smoke.md` (the smoke gate that runs
> just before this), `docs/architecture/c4-overview.md` (Levels 1–2), and
> `docs/runbooks/jetson-full-validation.md` (operator workflow).

## Component Diagram

```mermaid
flowchart TB
    subgraph External["External actors"]
        Operator(["Operator / CI"])
        Clock[("monotonic / wall clock")]
    end

    subgraph Pure["Pure helpers (dependency-free) — validation/latency_stats.py"]
        Summarize["summarize(samples_ms) -> LatencySummary\nmin/mean/p50/p95/p99/max"]
        Percentile["percentile(sorted, q)\nlinear interpolation (NumPy-default)"]
        Intervals["intervals_ms(timestamps_s)\narrival times -> inter-arrival gaps"]
    end

    subgraph Probes["Operator probes (tools/)"]
        LlmProbe["llm_latency_probe.py --iterations N\ngates on p95; emits llm_latency_summary"]
        LidarProbe["lidar_telemetry_probe.py\nemits lidar_frame_interval_summary"]
    end

    subgraph TrendStore["Trend store — validation/report_store.py"]
        Record["record_report(journal, report, run_id, git_sha)\n-> preflight_report event"]
        ReadHist["read_report_history(journal)\nfilter + sort by recorded_at_ns"]
        Detect["detect_regressions(history, slow_ratio, slow_floor_s)\nstatus downgrade / new FAIL / latency creep"]
    end

    subgraph CLI["validation CLI — cli/preflight.py"]
        PreflightCLI["preflight --journal-path PATH\n--trend --trend-slow-ratio --trend-slow-floor-s"]
    end

    subgraph Journal["Harness journal (reused) — harness/journal/"]
        BuildJournal["factory.build_journal(cfg)\n-> Null | JSONL | LMDB"]
        JournalProto["JournalProtocol\nappend / read_all (non-blocking)"]
    end

    subgraph Preflight["validation/preflight.py (reused)"]
        RunPreflight["run_preflight(cfg) -> PreflightReport"]
    end

    subgraph Script["scripts/jetson_full_validation.sh"]
        GitSha["git_clean_sha\nHEAD sha iff src/tests/scripts/config clean"]
        Phase1["phase1 (cache-aware wrapper)\nSKIP on matching sha; write cache on green"]
        Phases["--phases 0,1,3 / --no-cache"]
        Cache[(".cache/phase1_pass_sha\nunder report-root, gitignored")]
    end

    Operator --> PreflightCLI
    Operator --> LlmProbe
    Operator --> LidarProbe
    Operator --> Phases

    LlmProbe --> Summarize
    LidarProbe --> Intervals --> Summarize
    Summarize --> Percentile
    Clock --> LlmProbe
    Clock --> LidarProbe

    PreflightCLI --> RunPreflight --> PreflightCLI
    PreflightCLI --> Record
    PreflightCLI --> ReadHist --> Detect
    Record --> JournalProto
    ReadHist --> JournalProto
    BuildJournal --> JournalProto
    Clock --> Record

    Phases --> Phase1
    GitSha --> Phase1
    Phase1 --> Cache
```

## Key contracts (the non-negotiables)

| Contract | Where | Why |
|---|---|---|
| **Pure helpers stay pure.** `summarize` / `percentile` / `intervals_ms` do no I/O, read no clock, render no verdict. | `validation/latency_stats.py` | Deterministic + unit-testable; the *caller* owns the pass/fail decision against its config-supplied target (no hardcoded threshold in the maths). |
| **p95 gate, not single-sample.** `--iterations 1` == legacy single-shot (p95 of one sample = that sample); `>1` gates on p95. | `tools/llm_latency_probe.py` | Absorbs cloud/GPU tail variance without failing on one unlucky round-trip. Backwards compatible at the default. |
| **Reuse the journal — no parallel store.** The trend store writes a `preflight_report` event through `JournalProtocol`; works with Null/JSONL/LMDB via `factory.build_journal(cfg)`. | `validation/report_store.py` | The journal already solves non-blocking append, bounded-memory iteration, durable ordering. Modularity: the store depends only on the protocol. |
| **Wall-clock ordering.** `recorded_at_ns = time.time_ns()` in the payload; history sorts on it. | `validation/report_store.py` | Stable across reboots, unlike the journal's monotonic entry stamp (LMDB keys reset per process). |
| **Dual-gated latency creep.** A slowdown is flagged only when it exceeds BOTH `slow_ratio` AND `slow_floor_s`. | `detect_regressions` | The absolute floor suppresses noise on sub-50 ms checks where a 1.5× jump is meaningless. Both are operator-tunable via CLI (no hardcoded call site). |
| **Phase-1 cache keyed on clean source SHA.** `git_clean_sha` echoes HEAD only when the tree under `src/tests/scripts/config/pyproject.toml` is clean; a dirty tree forces a miss. Cache written only on a fully-green run. | `scripts/jetson_full_validation.sh` | Static CI is a pure function of committed source. A dirty tree must never be masked by a stale green. Hardware/live phases are never cached. |
| **Lazy sensor re-exports.** `validation/__init__` re-exports the numpy/cv2/pyaudio `runtime` helpers via PEP 562 `__getattr__`. | `validation/__init__.py` | Importing the pure modules never drags the sensor stack in. Backwards compatible — names still resolve on access. Locked by `tests/regression/test_validation_import_decoupling.py`. |

## Test surface

| Tier | File | Asserts |
|------|------|---------|
| Unit | `tests/unit/validation/test_latency_stats.py` | Percentile correctness, `intervals_ms`, edge cases. |
| Unit | `tests/unit/validation/test_report_store.py` | Record/read roundtrip, malformed-entry tolerance, all regression classes, custom thresholds, NullJournal safety. |
| Unit | `tests/unit/validation/test_init_lazy_exports.py` | In-process `__getattr__` resolution + AttributeError. |
| Unit | `tests/unit/cli/test_preflight_cli.py` | `--journal-path` / `--trend` / threshold flags / FAIL-exit. |
| Integration | `tests/integration/test_validation_report_store_integration.py` | Store through factory `build_journal` for JSONL **and** LMDB + NullJournal default. |
| Regression | `tests/regression/test_validation_import_decoupling.py` | Subprocess guard: pure modules don't import numpy/cv2; lazy re-exports resolve. |
| Smoke | `tests/smoke/test_jetson_full_validation_sanity.py` | Script arg surface (`--phases`, `--no-cache`, dry-run). |

## Structured-log events (operator grep recipes)

- `llm_latency_summary` — `gate_ms`, `p95_ms`, `passed` (probe verdict).
- `lidar_frame_interval_summary` — frame inter-arrival jitter percentiles.
- `preflight_report_recorded` — a run was appended to the trend journal.
- `trend_run_recorded` / `trend_evaluated` (DEBUG) — CLI trend path.
- `static CI (cached)` (bash `record PASS`) — phase-1 cache hit.
