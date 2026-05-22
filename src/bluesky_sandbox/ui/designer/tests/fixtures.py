"""Trivial code-tab callables used to exercise ``"module:attr"`` resolution.

These stand in for the reward / termination / truncation functions a user would
write in the code tab; the builder resolves them by import string. Signatures
match the runtime contract (``bluesky_sandbox.interface.task``): every function receives
the agent-bound ``context``, through which queryables are evaluated for the
current aircraft via ``context.query("name")``.
"""

from __future__ import annotations

from typing import Any


def reward(
    obs: Any,
    action: Any,
    terminated: bool,
    truncated: bool,
    context: Any,
    info: Any,
    rng: Any,
) -> float:
    return 0.0


def terminated(obs: Any, action: Any, context: Any, info: Any, rng: Any) -> bool:
    return False


def truncated(obs: Any, action: Any, context: Any, info: Any, rng: Any) -> bool:
    return False
