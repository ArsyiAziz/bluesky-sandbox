"""Which columns of an observation or an action belong to which field.

A layout names the columns of each part of a space: ``ownship``,
``intruders`` (one row per intruder) and the ``critic_*`` blocks of an
observation; ``continuous``, ``binary`` and ``discrete`` of an action. A space with a single
part is that part's space alone - an observation with only ownship fields is
its ``Box``, as is an action with only continuous fields - but its layout is
still keyed by the part.

A field's width depends only on the field and its normalizer, never on the
aircraft or the sim state, so a layout is fixed by the config: computed once,
in any process, without an environment. :meth:`observation_layout` and
:meth:`action_layout` on the environment are these for its config; a wrapper
that adds columns extends them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete, Space

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.interface.fields.base import ActionKind, action_kind

from .services import _action_parts, _field_output_size
from .slot import Slot, unique_names

__all__ = [
    "Slot",
    "action_applied",
    "action_layout",
    "flatten_action",
    "observation_layout",
    "observation_parts",
    "state_parts",
    "unflatten_action",
    "zero_action",
]


def slots(fields: Iterable[Any], start: int = 0) -> list[Slot]:
    """The slots of ``fields`` laid side by side from column ``start``."""
    out: list[Slot] = []
    for field in fields:
        width = _field_output_size(field)
        out.append(Slot(field.meta.name, slice(start, start + width)))
        start += width
    return unique_names(out)


def observation_parts(config: EnvConfig) -> dict[str, list[Any]]:
    """Each observation part and its fields, keyed as the observation is:
    ``ownship`` always, the intruder and critic parts when configured."""
    parts = {
        "ownship": config.obs_fields,
        "intruders": config.intruder_obs_fields,
        "critic_ownship": config.critic_obs_fields,
        "critic_intruders": config.critic_intruder_obs_fields,
    }
    return {
        key: list(fields or ())
        for key, fields in parts.items()
        if fields or key == "ownship"
    }


def state_parts(config: EnvConfig) -> dict[str, list[Any]]:
    """Each state part and its fields, keyed as hooks read them: ``ownship``
    and ``intruders``, each when configured. No agent observes them."""
    parts = {
        "ownship": getattr(config, "state_fields", None),
        "intruders": getattr(config, "intruder_state_fields", None),
    }
    return {key: list(fields) for key, fields in parts.items() if fields}


def observation_layout(config: EnvConfig) -> dict[str, list[Slot]]:
    """The columns of each observation part, keyed as the observation is."""
    return {key: slots(fields) for key, fields in observation_parts(config).items()}


def action_layout(config: EnvConfig) -> dict[str, list[Slot]]:
    """The columns of each action part: ``continuous``, ``binary``,
    ``discrete``."""
    return {
        kind.value: slots(fields)
        for kind, fields in _action_parts(config.action_fields).items()
    }


def flatten_action(config: EnvConfig, action: Any) -> np.ndarray:
    """``action`` as one vector, its fields in config order.

    What the action fields are applied from. A ``Box`` action space's action
    already is that vector; a ``Dict`` one's parts are interleaved back into
    config order.
    """
    parts = _action_parts(config.action_fields)
    if set(parts) <= {ActionKind.CONTINUOUS}:
        if isinstance(action, Mapping):
            raise TypeError(
                "this environment's actions are all continuous: pass one vector, "
                f"not a dict with {sorted(action)}."
            )
        return np.asarray(action, dtype=np.float32).reshape(-1)
    if not isinstance(action, Mapping):
        raise TypeError(
            "this environment's action space is a Dict: pass "
            f"{{{', '.join(repr(k.value) for k in parts)}}}, not one vector."
        )
    if set(action) != {kind.value for kind in parts}:
        raise ValueError(
            f"action has parts {sorted(action)}; this environment's are "
            f"{sorted(kind.value for kind in parts)}."
        )
    arrays = {
        kind: np.asarray(action[kind.value], dtype=np.float32).reshape(-1)
        for kind in parts
    }
    for kind, fields in parts.items():
        width = sum(_field_output_size(field) for field in fields)
        if arrays[kind].size != width:
            raise ValueError(
                f"action part {kind.value!r} has {arrays[kind].size} values; "
                f"its fields take {width}."
            )
    cursors = dict.fromkeys(parts, 0)
    out = []
    for field in config.action_fields:
        width = _field_output_size(field)
        kind = action_kind(field)
        start = cursors[kind]
        out.append(arrays[kind][start : start + width])
        cursors[kind] = start + width
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def action_applied(config: EnvConfig, applied: Sequence[bool] | None = None) -> Any:
    """Which values of an action took effect, shaped as the action: 1 for each
    value applied, 0 for each one skipped - masked by an ``ActionMask``, or on
    an axis a switch suppressed.

    ``applied`` has one flag per action field, in config order; ``None`` is
    every one applied. A ``Box`` action space's is one vector; a ``Dict`` one's
    a dict of the parts.
    """
    fields = config.action_fields
    if applied is None:
        applied = [True] * len(fields)
    if len(applied) != len(fields):
        raise ValueError(
            f"{len(applied)} applied flags for {len(fields)} action fields."
        )
    parts: dict[ActionKind, list[np.ndarray]] = {
        kind: [] for kind in _action_parts(fields)
    }
    for field, flag in zip(fields, applied):
        parts[action_kind(field)].append(
            np.full(_field_output_size(field), float(flag), dtype=np.float32)
        )
    arrays = {
        kind: np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
        for kind, values in parts.items()
    }
    if set(arrays) <= {ActionKind.CONTINUOUS}:
        return arrays.get(ActionKind.CONTINUOUS, np.zeros(0, dtype=np.float32))
    return {kind.value: values for kind, values in arrays.items()}


def unflatten_action(config: EnvConfig, flat: Any) -> Any:
    """The inverse of :func:`flatten_action`: one vector, fields in config
    order, as the action space takes it - itself for a ``Box``, a dict of the
    parts for a ``Dict``. For a trainer that keeps every action as one vector."""
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    parts = _action_parts(config.action_fields)
    width = sum(_field_output_size(f) for f in config.action_fields)
    if flat.size != width:
        raise ValueError(f"action has {flat.size} values; its fields take {width}.")
    if set(parts) <= {ActionKind.CONTINUOUS}:
        return flat
    out: dict[ActionKind, list[np.ndarray]] = {kind: [] for kind in parts}
    start = 0
    for field in config.action_fields:
        stop = start + _field_output_size(field)
        out[action_kind(field)].append(flat[start:stop])
        start = stop
    integer = {ActionKind.BINARY: np.int8, ActionKind.DISCRETE: np.int64}
    return {
        kind.value: (
            np.concatenate(values)
            if kind is ActionKind.CONTINUOUS
            else np.rint(np.concatenate(values)).astype(integer[kind])
        )
        for kind, values in out.items()
    }


def zero_action(space: Space) -> Any:
    """An all-zero action for ``space``: a vector, or a dict of the parts."""
    if isinstance(space, Dict):
        return {key: zero_action(part) for key, part in space.spaces.items()}
    if isinstance(space, (Box, MultiBinary, MultiDiscrete)):
        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(f"no zero action for {space!r}")
