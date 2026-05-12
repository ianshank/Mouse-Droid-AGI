"""Isaac Lab environment scaffolding for the 4WD rover.

Heavy imports (``isaaclab``, ``omni.isaac.*``, ``torch``) live inside the
concrete env class and are guarded behind a try/except so this package
loads cleanly on a workstation without Isaac Lab installed.
"""

from __future__ import annotations
