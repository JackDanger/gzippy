"""fulcrum.core — the project-agnostic decision engine.

Nothing in this package may import from fulcrum.adapters: the core knows no
project. Project specifics (taxonomy, knobs, guards, launch policy) arrive via
a fulcrum.adapters.base.ProjectAdapter instance.
"""

from .trace import InstrumentError  # noqa: F401  (the package-wide failure type)
