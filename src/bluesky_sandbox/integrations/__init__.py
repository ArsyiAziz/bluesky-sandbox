"""Optional integrations layered on top of the core environment primitives."""

from .common import (
    find_wrapper,
    widen_intruder_space,
    wrap_parallel_env,
)
from .vector import sb3_vec_env, vec_env

__all__ = [
    "find_wrapper",
    "widen_intruder_space",
    "sb3_vec_env",
    "vec_env",
    "wrap_parallel_env",
]
