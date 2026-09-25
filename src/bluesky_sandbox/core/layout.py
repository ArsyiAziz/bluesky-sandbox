"""Which columns of an observation or an action belong to which field.

A layout names the columns of each part of a space: ``ownship``,
``intruders`` (one row per intruder) and the ``critic_*`` blocks of an
observation; ``continuous`` and ``binary`` of an action. A space with a single
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

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary, Space

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.interface.fields.base import ActionKind

from .services import _action_parts, _field_output_size
from .slot import Slot, unique_names

__all__ = [
    "Slot",
    "action_layout",
    "flatten_action",
    "observation_layout",
    "observation_parts",
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


def observation_layout(config: EnvConfig) -> dict[str, list[Slot]]:
    """The columns of each observation part, keyed as the observation is."""
    return {key: slots(fields) for key, fields in observation_parts(config).items()}


def action_layout(config: EnvConfig) -> dict[str, list[Slot]]:
    """The columns of each action part: ``continuous`` and ``binary``."""
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
        start = cursors[field.kind]
        out.append(arrays[field.kind][start : start + width])
        cursors[field.kind] = start + width
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def zero_action(space: Space) -> Any:
    """An all-zero action for ``space``: a vector, or a dict of the parts."""
    if isinstance(space, Dict):
        return {key: zero_action(part) for key, part in space.spaces.items()}
    if isinstance(space, (Box, MultiBinary)):
        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(f"no zero action for {space!r}")
