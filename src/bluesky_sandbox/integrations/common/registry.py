"""Environment registry primitives shared by integrations."""

from __future__ import annotations

import importlib
from collections.abc import Callable

# Empty by default: task packages live outside this library (see ``rl/tasks``),
# so the simulator cannot name them without depending on its own consumer.
# Register your own with ``ENV_REGISTRY["my_task"] = "my_pkg.my_task:make_env"``,
# or skip the registry entirely and pass ``"module:attr"`` to
# :func:`resolve_make_env`, which is what every caller here already does.
ENV_REGISTRY: dict[str, str] = {}


def resolve_make_env(env: str) -> Callable:
    """Look up ``env`` in ``ENV_REGISTRY`` or treat it as ``module:attr``."""
    target = ENV_REGISTRY.get(env, env)
    if ":" not in target:
        raise ValueError(
            f"unknown env {env!r}: pass a registered name "
            f"({sorted(ENV_REGISTRY)}) or a 'module:attr' path"
        )
    mod_path, attr = target.rsplit(":", 1)
    return getattr(importlib.import_module(mod_path), attr)
