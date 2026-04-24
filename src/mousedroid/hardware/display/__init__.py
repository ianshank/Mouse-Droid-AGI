"""MSE-6 face display drivers (SSD1306 OLED + mock).

The package exposes:

* :class:`Expression` — discrete facial-expression enum.
* :func:`render_expression` / :func:`render_text` — pure PIL renderers
  used by both the real and mock drivers (no I/O performed).
* :class:`SSD1306FaceDriver` — real I²C driver (lazy-imported by factory).
* :class:`MockFaceDriver` — in-memory mock used in CI and ``mock_hardware``
  mode.
"""

from __future__ import annotations
