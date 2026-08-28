"""Unit: ``redact_uri_credentials`` masks inline URI credentials.

Unit tier per ``.claude/skills/test-tier-mirror/SKILL.md``: a pure function
over strings, no I/O, no factory. The proof that the *factory* actually calls
it lives one tier up, in
``tests/integration/test_experiment_logger_redaction.py``.
"""

from __future__ import annotations

import pytest

from mousedroid.logging.redaction import redact_uri_credentials, redact_uris_in_text


@pytest.mark.parametrize(
    "uri",
    [
        # The shipped default and its absolute form -- the overwhelmingly
        # common case, which must stay byte-identical so logs stay readable.
        "sqlite:///mlflow.db",
        "sqlite:////opt/mousedroid/mlflow.db",
        "sqlite:///:memory:",
        "file:./mlruns",
        "file:/opt/mousedroid/mlruns",
        # Remote, but credential-free.
        "http://mlflow.internal:5000",
        "https://mlflow.example.com/path?x=1",
        "databricks",
        # Degenerate inputs must not raise.
        "",
        "   ",
        "not a uri at all",
    ],
)
def test_credential_free_uris_pass_through_byte_identical(uri: str) -> None:
    """No userinfo component means nothing to mask -- and nothing changed."""
    assert redact_uri_credentials(uri) == uri


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        # user:password -- the canonical case.
        (
            "http://mlflow-user:S3cret@mlflow.internal:5000",
            "http://***@mlflow.internal:5000",
        ),
        # Bare userinfo with NO password. This is why the whole component is
        # masked rather than just ``parsed.password``: a personal access
        # token is conventionally passed exactly like this, so masking only
        # the password field would leak the entire credential.
        ("https://ghp_deadbeef@github.com/org/repo", "https://***@github.com/org/repo"),
        # Empty password still signals a credential form.
        ("http://user:@host:5000", "http://***@host:5000"),
        # Path, query and fragment are preserved -- they are the diagnostic
        # payload this logging exists to carry.
        (
            "postgresql://u:p@db.internal:5432/mlflow?sslmode=require",
            "postgresql://***@db.internal:5432/mlflow?sslmode=require",
        ),
        # A percent-encoded '@' inside the password must not confuse the
        # split: the LAST '@' is the real separator.
        ("http://u:p%40ss@host/x", "http://***@host/x"),
    ],
)
def test_userinfo_is_masked_while_the_target_stays_readable(uri: str, expected: str) -> None:
    assert redact_uri_credentials(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        # '@' in a path segment (S3-style key) or a query value (an email)
        # is not a credential separator, so these must survive intact.
        "file:./runs/user@host/mlruns",
        "http://host:5000/api?contact=ops%40example.com",
        "http://host:5000/api?contact=ops@example.com",
    ],
)
def test_an_at_sign_outside_the_authority_is_not_treated_as_a_credential(
    uri: str,
) -> None:
    assert redact_uri_credentials(uri) == uri


def test_an_unparseable_uri_containing_an_at_sign_is_redacted_wholesale() -> None:
    """Fail safe: if it cannot be parsed, assume it carries a credential.

    Losing a malformed URI as a diagnostic is the cheaper mistake; emitting
    one that happens to hold a password is not.

    An unterminated IPv6 literal is the trigger used here because it is one
    of the few inputs ``urlsplit`` itself rejects. Note what does *not*
    trigger it: an invalid port (``host:notaport``) parses fine and only
    raises on ``.port``, which this helper never touches -- that input takes
    the ordinary masking path instead, verified below.
    """
    assert redact_uri_credentials("http://u:p@[::1") == "<unparseable-uri-redacted>"
    # ...and the near-miss really does take the normal path, so this test is
    # pinning the branch it claims to and not passing by accident.
    assert redact_uri_credentials("http://u:p@host:notaport/x") == "http://***@host:notaport/x"


def test_redaction_is_idempotent() -> None:
    """Re-redacting an already-masked URI must not corrupt it."""
    once = redact_uri_credentials("http://u:p@host:5000/x")
    assert redact_uri_credentials(once) == once


def test_no_password_survives_any_credentialed_form() -> None:
    """Blanket guard: the secret must not appear anywhere in the output.

    Deliberately broader than the exact-equality cases above -- those pin
    the format, this pins the security property itself, so a future format
    change cannot silently reintroduce a leak.
    """
    secret = "hunter2SuperSecret"  # noqa: S105 - test fixture, not a real credential
    for uri in (
        f"http://user:{secret}@host:5000",
        f"https://{secret}@host/path",
        f"postgresql://user:{secret}@db:5432/name?opt=1",
    ):
        assert secret not in redact_uri_credentials(uri)


# --- redact_uris_in_text: URIs embedded in free-text messages ---------------


@pytest.mark.parametrize(
    "text",
    [
        # The messages this actually sees in production. None contains a URI,
        # and none may be altered -- mangling ordinary error text would make
        # the logs worse, not safer.
        "unable to open database file",
        "file is not a database",
        "No module named 'sqlalchemy'",
        "",
        # A URI with no credentials must survive intact.
        "got unsupported URI 'sqlite:///mlflow.db' for tracking",
        # Bare '@' in prose is not a URI.
        "contact ops@example.com about run 4",
    ],
)
def test_text_without_embedded_credentials_is_unchanged(text: str) -> None:
    assert redact_uris_in_text(text) == text


def test_the_real_mlflow_exception_message_is_redacted() -> None:
    """Pins the exact failure that motivated this helper.

    ``UnsupportedModelRegistryStoreURIException`` is the exception
    ``build_experiment_logger`` catches when the ``[mlflow]`` extra is thin,
    and mlflow quotes the offending URI back verbatim -- password included.
    Redacting only the ``configured_uri`` field while passing ``str(exc)``
    through would therefore have leaked the credential anyway.

    The message below is the real one, captured from mlflow 3.15.2 rather
    than invented.
    """
    secret = "hunter2SuperSecret"  # noqa: S105 - test fixture, not a real credential
    message = (
        "Model registry functionality is unavailable; got unsupported URI "
        f"'notascheme://user:{secret}@host/db' for model registry data storage."
    )
    redacted = redact_uris_in_text(message)

    assert secret not in redacted
    assert "notascheme://***@host/db" in redacted
    # The surrounding diagnostic prose must survive.
    assert redacted.startswith("Model registry functionality is unavailable")
    assert redacted.endswith("for model registry data storage.")


def test_multiple_embedded_uris_are_all_redacted() -> None:
    """One masked URI in a message must not let a second through."""
    text = "failed over from http://a:secret1@h1/x to https://b:secret2@h2/y"
    redacted = redact_uris_in_text(text)
    assert "secret1" not in redacted
    assert "secret2" not in redacted
    assert redacted == "failed over from http://***@h1/x to https://***@h2/y"


def test_text_redaction_is_idempotent() -> None:
    once = redact_uris_in_text("bad URI 'http://u:p@host/db' here")
    assert redact_uris_in_text(once) == once
