"""Property tests for the Agent-SDK reviewer's read guard (``tools/agent_reviewer/policy.py``).

Hypothesis throws arbitrary and near-miss paths, globs and environments at the guard and
asserts what must hold whatever the input:

* the checks themselves answer - they do not lean on the guard's catch-all - for any text:
  a Windows device name, a lone surrogate or a stray NUL is refused by name, not by crash;
* a read the guard accepts is a file inside the checkout, not ``.git``, not a secret name;
* a glob it accepts is spelled in the plain alphabet and cannot match a secret-file name;
* the environment scrub removes every name that has a credential word in it and nothing else
  it was not told to.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tools.agent_reviewer import policy

_TOKENS = [
    "*",
    "**",
    ".",
    "..",
    "/",
    "\\",
    "src",
    "app.py",
    ".git",
    ".GIT",
    ".env",
    "prod.env",
    "docker.env",
    "id_rsa",
    "nul",
    "CON",
    "com1",
    "lpt9.txt",
    ".py",
    "[",
    "]",
    "{",
    "}",
    "?",
    "!",
    ",",
    " ",
    "~",
    "$",
    "%",
    ":",
    "elsewhere",
    "checkout",
    "\x00",
    "é",
    "¹",
]
_near_misses = st.lists(st.sampled_from(_TOKENS), max_size=8).map("".join)
_values = st.one_of(_near_misses, st.text(max_size=60))
_PLAIN_GLOB = re.compile(r"[A-Za-z0-9_.@+*/-]+")
_CREDENTIAL_WORD = re.compile(
    r"(?:^|[^A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(?:[^A-Z0-9]|$)"
)


@pytest.fixture(scope="module")
def checkout() -> Iterator[Path]:
    """A small checkout with a directory beside it that must stay out of reach."""
    base = Path(tempfile.mkdtemp(prefix="read-guard-property-"))
    root = base / "checkout"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (base / "elsewhere").mkdir()
    yield root
    shutil.rmtree(base, ignore_errors=True)


@given(
    tool=st.sampled_from(["Read", "Grep", "Glob"]),
    field=st.sampled_from(["file_path", "path", "glob", "pattern"]),
    value=_values,
)
@settings(max_examples=300, deadline=None)
def test_the_checks_answer_for_any_text_without_leaning_on_the_catch_all(
    checkout: Path, tool: str, field: str, value: str
) -> None:
    result = policy._check_read_access(tool, {field: value}, root=str(checkout))
    assert result is None or isinstance(result, str)


@given(value=_values)
@settings(max_examples=300, deadline=None)
def test_a_read_the_guard_accepts_is_a_plain_file_inside_the_checkout(
    checkout: Path, value: str
) -> None:
    reason = policy.check_read_access("Read", {"file_path": value}, root=str(checkout))
    if reason is not None:
        return
    root_real = os.path.realpath(checkout)
    real = os.path.realpath(value if os.path.isabs(value) else os.path.join(checkout, value))
    assert policy.path_is_inside(real, root_real), (value, real)
    assert policy.read_secret_reason(real) is None, value
    below = os.path.relpath(real, root_real).replace("\\", "/").split("/")
    assert ".git" not in [part.lower() for part in below], value
    assert policy._device_name_in(value) is None, value
    assert not set(value) & set("*?[]{}$%\x00"), value
    assert not value.startswith("~"), value
    assert ":" not in os.path.splitdrive(value)[1], value


@given(value=_values)
@settings(max_examples=300, deadline=None)
def test_a_glob_the_guard_accepts_is_plain_and_cannot_match_a_secret_name(
    checkout: Path, value: str
) -> None:
    for tool, field in (("Glob", "pattern"), ("Grep", "glob")):
        reason = policy.check_read_access(tool, {field: value}, root=str(checkout))
        if reason is not None:
            continue
        assert _PLAIN_GLOB.fullmatch(value), (tool, value)
        segments = value.split("/")
        assert all(segments), (tool, value)  # no empty part
        assert ".git" not in [segment.lower() for segment in segments], (tool, value)
        assert not any(segment.startswith("..") for segment in segments), (tool, value)
        file_part = segments[-1].lower()
        for probe in policy._SECRET_PROBES:
            assert not fnmatch.fnmatchcase(probe, file_part), (tool, value, probe)
        assert policy.read_secret_reason(file_part) is None, (tool, value)
        assert policy._device_name_in(value) is None, (tool, value)


_env_names = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-."),
    min_size=1,
    max_size=24,
)


@given(environ=st.dictionaries(_env_names, st.text(max_size=8), max_size=12))
@settings(max_examples=300, deadline=None)
def test_the_scrub_removes_every_name_with_a_credential_word_and_keeps_the_rest(
    environ: dict[str, str],
) -> None:
    scrubbed = policy.scrubbed_environ(environ)
    for name, value in environ.items():
        word = _CREDENTIAL_WORD.search(name.upper()) is not None
        named = name.upper() in policy.CREDENTIAL_ENV_NAMES
        if word or named:
            assert name not in scrubbed, name
        else:
            assert scrubbed.get(name) == value, name
    assert set(scrubbed) <= set(environ)
