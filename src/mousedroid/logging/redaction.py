"""Credential redaction for values that reach structured log events.

Why this exists: several subsystems log a URI to make a failure diagnosable
("which store / host did this actually point at?"), and a URI is one of the
few config values that can carry a credential *inline* — the RFC 3986
``userinfo`` component, ``scheme://user:password@host/path``. A ``SecretStr``
field protects a credential that lives in its own field; it does nothing for
one embedded in an otherwise-loggable string.

The repo has no redaction processor in the structlog chain
(:mod:`mousedroid.logging.setup` builds the processor list; none of the
entries scrub values), and it deliberately renders JSON at ``INFO`` by
default. So anything handed to a log call is emitted verbatim to stdout, to
the telemetry ring buffer, and — where enabled — to Cloud Logging. Redaction
therefore has to happen at the call site, which is what this module is for.

Scope, stated precisely: this masks the ``userinfo`` component and nothing
else. It is *not* a general secret scrubber — it will not catch a token in a
query string or a path segment, and it is not a substitute for
``SecretStr`` on a field that is wholly a credential.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

__all__ = ["redact_uri_credentials", "redact_uris_in_text"]

# Replaces the whole ``userinfo`` component rather than just the password.
# A bare userinfo with no password is a real credential form -- a personal
# access token is conventionally passed as ``https://<token>@host/...`` --
# so masking only ``parsed.password`` would leak exactly that case. The
# marker is kept in place (rather than dropped) so an operator can still
# see that credentials *were* supplied, which is itself diagnostic.
_USERINFO_PLACEHOLDER = "***"

# Returned when a string contains an ``@`` but cannot be parsed. Redacting
# wholesale is the safe direction: an unparseable URI that might carry a
# credential is worth losing as a diagnostic, whereas emitting it is not.
_UNPARSEABLE = "<unparseable-uri-redacted>"


# Matches the ``scheme://userinfo@`` prefix of a URI embedded anywhere in a
# larger string. userinfo cannot contain whitespace, ``/`` or an unencoded
# ``@``, which is what keeps this from swallowing the rest of a sentence.
# Scheme grammar per RFC 3986 s3.1.
_EMBEDDED_USERINFO = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.\-]*://)([^\s/@]+)@")


def redact_uris_in_text(text: str) -> str:
    """Mask credentials in any URI embedded in a free-text string.

    For messages rather than bare URIs -- above all exception text, which is
    logged alongside the URI itself and can echo it back. This is not
    hypothetical: mlflow's ``UnsupportedModelRegistryStoreURIException``
    quotes the offending tracking URI verbatim, password included, and that
    is exactly the exception ``build_experiment_logger`` catches when the
    ``[mlflow]`` extra is thin. (SQLAlchemy, by contrast, already masks its
    own URLs -- verified, not assumed.)

    >>> redact_uris_in_text("got unsupported URI 'http://u:pw@host/db' for x")
    "got unsupported URI 'http://***@host/db' for x"
    >>> redact_uris_in_text("unable to open database file")
    'unable to open database file'

    Args:
        text: Arbitrary message text.

    Returns:
        The text with every embedded URI's userinfo component masked.
    """
    return _EMBEDDED_USERINFO.sub(rf"\1{_USERINFO_PLACEHOLDER}@", text)


def redact_uri_credentials(uri: str) -> str:
    """Return ``uri`` with any embedded credentials masked.

    Everything an operator needs to identify the target -- scheme, host,
    port, path, query -- is preserved; only the ``userinfo`` component is
    replaced. Values with no credential form are returned unchanged, so the
    common local cases (``sqlite:///mlflow.db``, ``file:./mlruns``) are
    byte-identical to their input and stay fully readable in logs.

    >>> redact_uri_credentials("sqlite:///mlflow.db")
    'sqlite:///mlflow.db'
    >>> redact_uri_credentials("http://user:pw@mlflow.internal:5000")
    'http://***@mlflow.internal:5000'
    >>> redact_uri_credentials("https://ghp_token@github.com/org/repo")
    'https://***@github.com/org/repo'

    Args:
        uri: Any URI-shaped string. Need not be well-formed.

    Returns:
        The URI with credentials masked, the input unchanged when it carries
        none, or a fixed marker when it both contains an ``@`` and cannot be
        parsed.
    """
    # Fast path, and the reason the common case stays readable: without an
    # ``@`` there is no userinfo component to mask, so there is nothing to
    # do and no need to risk a parse.
    if "@" not in uri:
        return uri

    try:
        parts = urlsplit(uri)
    except ValueError:
        # urlsplit itself rejects very little -- a malformed IPv6 literal
        # (``http://u:p@[::1``) is the realistic trigger. An invalid *port*
        # notably does NOT land here: it parses cleanly and only raises on
        # ``.port``, which this function never accesses, so such a URI takes
        # the ordinary masking path below. Both behaviours are pinned in
        # tests/unit/logging/test_redaction.py.
        return _UNPARSEABLE

    # ``@`` can appear in a path or query without being a credential
    # separator (an S3-style key, an email address in a query param). Only
    # a real userinfo component sets username/password.
    if not parts.username and not parts.password:
        return uri

    # The last ``@`` in the netloc is the separator: a literal ``@`` inside
    # userinfo must be percent-encoded, so rpartition cannot split wrongly.
    host_port = parts.netloc.rpartition("@")[2]
    return urlunsplit(
        (
            parts.scheme,
            f"{_USERINFO_PLACEHOLDER}@{host_port}",
            parts.path,
            parts.query,
            parts.fragment,
        )
    )
