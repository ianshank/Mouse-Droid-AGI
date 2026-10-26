# tests/regression/test_f059_aqa.py
"""AQA pins for F-059 (rover gate closeout). Each test names the round-3 finding it pins."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import WAVESHARE_STOCK_BAUD
from mousedroid.validation.preflight import PreflightStatus, _check_esp32

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_PROD = _CONFIG_DIR / "jetson_production.yaml"
_BENCH = _CONFIG_DIR / "jetson_bench_esp32.yaml"
_LIDAR_ONLY = _CONFIG_DIR / "jetson_lidar_only.yaml"


def _features() -> dict[str, dict]:
    raw = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text(encoding="utf-8"))
    return {f["id"]: f for f in raw["features"]}


def test_f059_is_reserved_and_critical() -> None:
    f = _features()["F-059"]
    assert f["priority"] == "critical"
    assert f["tier"] == "fast"
    assert f["validation_command"] == "bash scripts/validations/F-059.sh"


def test_f059_edge_points_the_right_way() -> None:
    """R3-1: F-059 makes F-008 passable, so F-008 waits on F-059 — never the reverse."""
    feats = _features()
    assert "F-008" not in feats["F-059"]["depends_on"]
    assert "F-059" in feats["F-008"]["depends_on"]


def test_select_next_returns_f059_while_it_is_in_progress() -> None:
    """F-059 wins the in-progress pick by priority (critical beats F-050 high / F-051 medium);
    the F-050/F-051 deferral is catalog honesty (EX-3), not the lever.
    """
    catalog = _features()
    if catalog["F-059"]["status"] != "in_progress":
        pytest.skip("F-059 is no longer in progress; the edge pin is the durable guarantee")
    if any(
        f["priority"] == "critical" and f["status"] == "in_progress"
        for fid, f in catalog.items()
        if fid != "F-059"
    ):
        pytest.skip("another critical feature is in progress; the selector tie-breaks on id")
    from mousedroid.harness import spec

    feats = spec.load_features(str(_REPO_ROOT / "features.yaml"))
    sel = spec.select_next(feats)
    assert sel.feature is not None, sel
    assert sel.feature["id"] == "F-059", sel


def test_bench_overlay_flips_exactly_esp32_enabled_on_top_of_production() -> None:
    """R3-2 / EX-1: one key, stacked after production; nothing else moves."""
    prod = load_settings(_PROD, config_dir=_CONFIG_DIR)
    bench = load_settings(_PROD, _BENCH, config_dir=_CONFIG_DIR)
    assert prod.esp32.enabled is False
    assert bench.esp32.enabled is True
    assert bench.esp32.model_copy(update={"enabled": False}) == prod.esp32
    assert bench.safety == prod.safety
    assert yaml.safe_load(_BENCH.read_text(encoding="utf-8")) == {"esp32": {"enabled": True}}


def test_bench_overlay_leaves_the_motion_gate_closed() -> None:
    """Review focus 2: enabling the board never opens the Pydantic motion gate."""
    bench = load_settings(_PROD, _BENCH, config_dir=_CONFIG_DIR)
    assert bench.esp32.smoke_test_allow_motion is False


def _bash_function(text: str, name: str) -> str:
    """Code of the top-level shell function ``name``: its opening line to the first bare ``}``.

    Whole-line ``#`` comments are dropped, so prose that mentions a token cannot satisfy a pin
    on the code that must contain it.
    """
    lines = text.splitlines()
    start = lines.index(f"{name}() {{")
    body = lines[start : lines.index("}", start)]
    return "\n".join(line for line in body if not line.lstrip().startswith("#"))


def _hardware_functions(module: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function defined in ``tests/hardware/<module>``, keyed by name."""
    src = (_REPO_ROOT / "tests" / "hardware" / module).read_text(encoding="utf-8")
    return {
        node.name: node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _code_names(fn: ast.AST) -> set[str]:
    """Names and attributes the *code* under ``fn`` uses.

    Docstrings, comments and message strings are never ``Name``/``Attribute`` nodes, so
    prose that merely mentions a token cannot satisfy a pin on the code that must use it.
    """
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _skips_when_the_board_is_configured_off(fn: ast.AST) -> bool:
    """Whether ``fn`` holds ``if not <...>.enabled:`` whose body calls ``pytest.skip``.

    Prose cannot satisfy it: a docstring, a message or an unrelated branch that merely mentions
    the setting is not an ``If`` node negating the attribute and skipping from its body. The
    ``not`` is pinned too: ``if <...>.enabled: pytest.skip(...)`` skips a live board and runs
    the test on the mock whenever the board is configured off.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        negates_enabled = (
            isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Attribute)
            and node.test.operand.attr == "enabled"
        )
        skips = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "skip"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pytest"
            for stmt in node.body
            for call in ast.walk(stmt)
        )
        if negates_enabled and skips:
            return True
    return False


@pytest.mark.parametrize(
    ("stage", "module", "passed_line"),
    [
        ("test_motor", "test_esp32_loopback.py", "test_send_velocity_moves_encoders PASSED"),
        ("test_power", "test_power_chain_smoke.py", "test_power_chain_within_budget PASSED"),
    ],
    ids=["motor", "power"],
)
def test_esp32_stages_run_their_hardware_modules_and_grep_passed(
    stage: str, module: str, passed_line: str
) -> None:
    """R-1 pin 1 + R3-2: the stage <-> test pairing the gate depends on, for both ESP32 stages.

    pytest exits 0 when every test skips (e.g. under esp32.enabled=false), so a bare exit status
    proves nothing: each stage must see its own test PASS. The explicit opt-out (the overlay
    variable set empty) SKIPs before pytest runs, and ANSI codes must never break the greps.
    """
    text = (_REPO_ROOT / "scripts" / "jetson_smoke_test.sh").read_text(encoding="utf-8")
    body = _bash_function(text, stage)
    # The assignment, not the bare path: the stage's own "not found" messages also name it.
    assignment = f'test_file="${{PROJECT_DIR}}/tests/hardware/{module}"'
    assert assignment in body, f"{stage} must run tests/hardware/{module}"
    assert f'grep -q "{passed_line}"' in body, (
        f"{stage} must require '{passed_line}' in the output, not just a zero exit status"
    )
    assert "esp32_stage_configs_csv" in body, (
        f"{stage} must stack the bench overlay through esp32_stage_configs_csv"
    )
    assert re.search(r"-m pytest -m hardware [^\n]*--color=no", body), (
        f"{stage} must run pytest with --color=no so the PASSED/SKIPPED greps never meet ANSI codes"
    )
    opt_out = re.search(
        r'\[\[ -z "\$\{BENCH_ESP32_OVERLAY\}" \]\]; then\s+record_skip [^\n]+\s+return\b', body
    )
    assert opt_out, f"{stage} must record a SKIP and return when the bench overlay is opted out"
    assert opt_out.start() < body.index("-m pytest -m hardware"), (
        f"{stage} must decide the opt-out before it invokes pytest"
    )


def test_loopback_motion_tests_branch_on_the_encoder_flag_and_the_motion_gate() -> None:
    """R-1 + review focus 2 + ruling 32: the encoder-less criterion is present, and both motion
    commands go through the Pydantic motion gate, capped by the one operator knob.
    """
    fns = _hardware_functions("test_esp32_loopback.py")
    velocity = _code_names(fns["test_send_velocity_moves_encoders"])
    estop = _code_names(fns["test_emergency_stop_latency"])
    assert "chassis_has_wheel_encoders" in velocity
    assert "imu_valid" in velocity
    assert "_commanded_velocity" in velocity
    assert "_commanded_velocity" in estop
    gate = _code_names(fns["_commanded_velocity"])
    assert "smoke_test_allow_motion" in gate
    # smoke_test_velocity_mps governs every smoke motion (0.0 locks all of it), as a ceiling:
    # `min` caps the command, a `max` would turn the setpoint into a floor.
    assert "smoke_test_velocity_mps" in gate, "the gate must cap the command at the one knob"
    assert "min" in gate, "smoke_test_velocity_mps is a ceiling on the command, not a floor"


def test_loopback_motion_gate_stays_closed_unless_the_pydantic_gate_opens() -> None:
    """CHARTER §3 polarity, which names cannot pin (an inverted ``if`` or a bare ``return 0.0``
    keeps every name): ``if not <...>.smoke_test_allow_motion: return 0.0``, then the ``min`` cap.
    """
    fn = _hardware_functions("test_esp32_loopback.py")["_commanded_velocity"]
    why = "CHARTER §3 gate: `if not <...>.smoke_test_allow_motion: return 0.0`, then min(...)"
    assert not any(isinstance(n, ast.Return) for n in fn.body[:-1]), why
    match fn.body[-2:]:
        case [
            ast.If(
                test=ast.UnaryOp(
                    op=ast.Not(), operand=ast.Attribute(attr="smoke_test_allow_motion")
                ),
                body=[ast.Return(value=ast.Constant(value=0.0))],
                orelse=[],
            ),
            ast.Return(value=ast.Call(func=ast.Name(id="min")) as cap),
        ]:
            assert {"max_velocity_mps", "smoke_test_velocity_mps"} <= _code_names(cap), why
        case _:
            pytest.fail(why)


def test_power_chain_test_skips_when_the_board_is_configured_off() -> None:
    """R3-2: esp32.enabled=false routes the power stage to MockESP32Driver (a 12.0 V green).

    The bench overlay turns it on, but an exported MOUSEDROID_ESP32__ENABLED=false beats
    the overlay; the test must skip with the remedy rather than measure the mock.
    """
    fns = _hardware_functions("test_power_chain_smoke.py")
    assert _skips_when_the_board_is_configured_off(fns["test_power_chain_within_budget"]), (
        "test_power_chain_within_budget must `if not <...>.enabled: pytest.skip(...)`"
    )


def test_loopback_velocity_test_skips_when_the_board_is_configured_off() -> None:
    """R3-2: the loopback velocity test must not pass against MockESP32Driver either.

    Same hazard as the power stage: an exported MOUSEDROID_ESP32__ENABLED=false beats the
    overlay and a mock answers every read, so the test skips with the remedy instead.
    """
    fns = _hardware_functions("test_esp32_loopback.py")
    assert _skips_when_the_board_is_configured_off(fns["test_send_velocity_moves_encoders"]), (
        "test_send_velocity_moves_encoders must `if not <...>.enabled: pytest.skip(...)`"
    )


def test_loopback_velocity_test_stops_the_board_in_a_finally_and_polls_within_the_budget() -> None:
    """Ruling 33: the board is left stopped whatever the command or a read does, and the
    encoder-less witness gets the driver's own timeout budget of reads, not a single poll.
    """
    fns = _hardware_functions("test_esp32_loopback.py")
    velocity = fns["test_send_velocity_moves_encoders"]
    stops_in_finally = any(
        isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "emergency_stop"
            for stmt in node.finalbody
            for call in ast.walk(stmt)
        )
        for node in ast.walk(velocity)
    )
    assert stops_in_finally, "emergency_stop() must run in a `finally` around the command and reads"
    names = _code_names(velocity)
    assert "max_consecutive_timeouts" in names, "the poll budget comes from the schema"
    assert "_read_until_feedback_frame" in names, "the T:1001 witness is polled, not read once"
    poll = _code_names(fns["_read_until_feedback_frame"])
    assert {"read_encoders", "imu_valid"} <= poll, "the poll must read until imu_valid answers"


async def test_preflight_esp32_warns_rather_than_ok_when_the_board_is_mocked() -> None:
    """R-4 / R3-2 (S-4): the shipped production overlay mocks the ESP32.

    ``config/jetson_production.yaml`` sets ``esp32.enabled: false``, so the factory
    hands back ``MockESP32Driver``; preflight used to report ``OK driver=...`` for
    that. It must be WARN (aggregate DEGRADED) -- there is no SKIP member (R-4).
    """
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    cfg.mock_hardware = False
    assert (await _check_esp32(cfg)).status is PreflightStatus.WARN


def test_trend_unit_never_runs_exclusive_device_checks() -> None:
    """R-11: the on-timer preflight must never open the serial port or the camera.

    The orchestrator container owns those devices while it runs. systemd honours the LAST
    ``Environment=`` assignment of a variable, so every such line is checked, not the first.
    """
    text = (_REPO_ROOT / "scripts" / "mousedroid-trend.service").read_text(encoding="utf-8")
    declared = re.findall(r"^Environment=MOUSEDROID_TREND_CHECKS=(\S+)$", text, re.M)
    assert declared, "default check list must be an Environment= line"
    listed = {name for value in declared for name in value.split(",")}
    assert not {"esp32", "lidar", "camera"} & listed


def test_production_overlay_speaks_stock_firmware_on_an_encoder_less_chassis() -> None:
    """D-2 / audit R3: the shipped rover speaks stock firmware on an encoder-less chassis."""
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    assert cfg.esp32.command_set == "waveshare_stock"
    assert cfg.esp32.serial_baud == WAVESHARE_STOCK_BAUD
    assert cfg.esp32.chassis_has_wheel_encoders is False
    assert cfg.esp32.enabled is False  # the flip to true lives only in the bench overlay


def test_production_overlay_arms_battery_thresholds_and_the_power_gate_has_no_floor() -> None:
    """CF-1 / R-6: thresholds are live; the runtime screens 0.0 V, the smoke gate does not."""
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    assert (cfg.safety.battery_warn_v, cfg.safety.battery_critical_v) == (10.5, 9.5)
    # runtime screens 0.0 V
    assert 0.0 < cfg.safety.battery_implausible_below_v < cfg.safety.battery_critical_v
    power = (_REPO_ROOT / "tests" / "hardware" / "test_power_chain_smoke.py").read_text(
        encoding="utf-8"
    )
    # R-6: the gate compares straight against the critical threshold and has no
    # plausibility floor of its own, so it is stricter than the runtime.
    assert "battery_voltage_v >= jetson_settings.safety.battery_critical_v" in power
    assert "battery_implausible_below_v" not in power


def test_latch_is_armed_on_production_and_explicitly_off_on_the_lidar_only_stack() -> None:
    """CF-2 / R-7: production latches an e-stop; the lidar-only stack opts out explicitly."""
    assert load_settings(_PROD, config_dir=_CONFIG_DIR).safety.emergency_latch.enabled is True
    # jetson_validate.sh's default stack.
    stack = load_settings(_PROD, _LIDAR_ONLY, config_dir=_CONFIG_DIR)
    # Premise: this is the stack that would latch.
    assert stack.safety.lidar_unavailable_policy == "emergency"
    assert stack.safety.emergency_latch.enabled is False
    raw = yaml.safe_load(_LIDAR_ONLY.read_text(encoding="utf-8"))
    assert raw["safety"]["emergency_latch"]["enabled"] is False  # explicit, not inherited (R-7)


_REARM_CONFIG = "--config /etc/mousedroid/jetson_production.yaml"


def test_runbooks_point_operators_at_the_rearm_cli() -> None:
    """CF-2 / Ruling 36: every documented re-arm names the production overlay.

    The CLI reads no environment to find its overlay. Without ``--config`` it loads
    ``default.yaml``, where the latch is off, prints "nothing to clear" and exits 0: a silent
    no-op on the production container that looks exactly like a clear rover. So the flag must
    sit on the command line of every invocation, not merely somewhere in the runbook.
    """
    for name in ("jetson-rover-smoke.md", "jetson-full-bringup.md", "emergency-stop-rearm.md"):
        text = (_REPO_ROOT / "docs" / "runbooks" / name).read_text(encoding="utf-8")
        assert re.search(
            r"mousedroid\.cli\.rearm --config /etc/mousedroid/jetson_production\.yaml", text
        ), name
        # Not one good command among bad ones: join the shell continuations, then check each
        # invocation. The boot log's quoted ``rearm_cmd`` hint is source output, not a command.
        joined = text.replace("\\\n", " ")
        for match in re.finditer(r"mousedroid\.cli\.rearm([^`\n]*)", joined):
            line_start = joined.rfind("\n", 0, match.start()) + 1
            args = " ".join(match.group(1).split())
            if args and "rearm_cmd=" not in joined[line_start : match.start()]:
                assert args.startswith(_REARM_CONFIG), (name, match.group(0))


def test_serial_driver_exclusive_open_is_a_constant_not_a_config_knob() -> None:
    """Task 4.1: ``exclusive=`` is fed by a module constant, and no config field can turn it off.

    Two openers on the ESP32 UART must collide loudly, so nothing reachable from YAML or an env
    var may release the lock. The unit test pins the value the driver passes, through a mock; this
    pins where that value comes from, which a literal ``exclusive=True`` or a new ESP32Config
    field would change without failing it.
    """
    from mousedroid.config.schema import ESP32Config

    path = _REPO_ROOT / "src" / "mousedroid" / "comms" / "serial_driver.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_SERIAL_EXCLUSIVE"
    ]
    assert len(constants) == 1, "serial_driver.py must define _SERIAL_EXCLUSIVE once"
    assert ast.unparse(constants[0].annotation) == "Final[bool]"
    assert isinstance(constants[0].value, ast.Constant)
    assert constants[0].value.value is True
    opens = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Serial"
    ]
    assert len(opens) == 1, "the driver opens the port in exactly one place"
    exclusive = {kw.arg: kw.value for kw in opens[0].keywords}.get("exclusive")
    assert isinstance(exclusive, ast.Name), "exclusive= must be the constant, not a literal"
    assert exclusive.id == "_SERIAL_EXCLUSIVE"
    assert not [name for name in ESP32Config.model_fields if "exclusive" in name]
