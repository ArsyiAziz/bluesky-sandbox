"""How a field's values are compared with each other - one rule, for the
fields' own checks and the checker's alike.

Two ways of computing the same value (bulk against one aircraft at a time, a
batch normalization against one value's) must agree exactly as float32, what
reaches an observation. A value against a field's plain statement of it
(``expected``), computed independently, agrees to rounding: tight for float64
results, to float32 precision for the fields that emit float32. Each returns
what disagrees, as messages - none when they agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["beyond_rounding", "differs"]


def differs(got: Any, want: Any, what: str) -> list[str]:
    a = np.asarray(got, dtype=np.float32)
    b = np.asarray(want, dtype=np.float32)
    if a.shape != b.shape:
        return [f"{what}: shape {a.shape} vs {b.shape}"]
    return [] if np.array_equal(a, b, equal_nan=True) else [f"{what}: {_short(a)} vs {_short(b)}"]


def beyond_rounding(bulk: Any, reference: Any, what: str) -> list[str]:
    a = np.asarray(bulk)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        return [f"{what}: shape {a.shape} vs {b.shape}"]
    tol = 1e-6 if a.dtype == np.float32 else 1e-9
    close = np.allclose(a.astype(np.float64), b, rtol=tol, atol=tol, equal_nan=True)
    return [] if close else [f"{what}: {_short(a)} vs {_short(b)}"]


def _short(values: np.ndarray) -> str:
    flat = values.reshape(-1)
    return f"{flat[0]:g}" if flat.size == 1 else np.array2string(flat, precision=6, threshold=8)
