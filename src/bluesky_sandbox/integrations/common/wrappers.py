"""Wrapper-chain helpers shared by integrations."""

from __future__ import annotations


def find_wrapper(env, wrapper_type: type):
    """Return the first wrapper of ``wrapper_type`` in an env chain."""
    cur = env
    while cur is not None:
        if isinstance(cur, wrapper_type):
            return cur
        cur = getattr(cur, "env", None)
    return None
