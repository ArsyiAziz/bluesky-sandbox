"""Shared integration primitives independent of a specific training library."""

from .parallel import build_parallel_env
from .registry import ENV_REGISTRY, resolve_make_env
from .spaces import widen_intruder_space
from .wrappers import find_wrapper

__all__ = [
    "ENV_REGISTRY",
    "build_parallel_env",
    "find_wrapper",
    "resolve_make_env",
    "widen_intruder_space",
]
