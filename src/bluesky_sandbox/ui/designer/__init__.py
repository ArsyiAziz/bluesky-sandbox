"""Environment Designer for bluesky-sandbox.

A declarative-spec layer over the simulation primitives (``bounds``,
``queryables``, ``spawn``, ``distributions``, ``scenarios``) that makes
environments serialisable, GUI-editable, and round-trippable.

The design seam follows the primitives themselves:

* **Structured data** - bounds/footprints, queryables, spawn regions, and
  distributions are plain dataclasses, so they serialise to a ``DesignSpec``
  (JSON-able dict) and reconstruct exactly. These are edited in the map tab.
* **Logic** - reward / termination / truncation functions and custom field
  or queryable classes are arbitrary callables. They cannot be values, so the
  spec references them by ``"module:attr"`` import string, resolved at build
  time. These live in the code tab.

The public surface:

* :func:`dump` / :func:`load` - object <-> spec-dict for any supported
  primitive (bounds, queryables, spawn, distributions).
* :class:`DesignSpec` - the top-level design document.
* :func:`build_scenario` / :func:`build_design_config` - compile a spec into the
  live :class:`~bluesky_sandbox.sim.sampling.Scenario` and static
  :class:`~bluesky_sandbox.config.EnvConfig` objects the runtime consumes.
* :mod:`bluesky_sandbox.ui.designer.nav` - navdb query layer scoped to an
  airspace, backing the map tab.
"""

from __future__ import annotations

from .builder import (
    BuildError,
    build_design_config,
    build_scenario,
)
from .spec import (
    DesignSpec,
    EnvSpec,
    SpecError,
    dump,
    dumps,
    load,
    loads,
)

__all__ = [
    "BuildError",
    "DesignSpec",
    "EnvSpec",
    "SpecError",
    "build_design_config",
    "build_scenario",
    "dump",
    "dumps",
    "load",
    "loads",
]
