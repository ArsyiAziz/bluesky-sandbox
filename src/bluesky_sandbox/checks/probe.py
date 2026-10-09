"""Probe one field at one moment: what it gives, how it got there, what it costs.

The env stands where its episode is - flown there by the caller - and the
probe reads one field for one aircraft (and, for a pair field, one other).
Nothing is listed by hand: what the field reads is watched as it runs.

- **Stages** - for an observation its raw value then its normalized one, for
  an action the policy's value, its steps, the value, the value on its grid,
  the target it commands and the command BlueSky takes - each as the env
  computes it (``actual``) and as the field states it plainly (``expected``)
  where it states it.
- **Trace** - the calls into the library and the design's code the field made
  for the aircraft, with what each returned, and every BlueSky traffic value
  it read, under the call that read it.
- **Overrides** - a traffic value set in BlueSky before the field runs
  (``bs.traf.alt`` for one aircraft), or what a call returns, whole or one of
  its numbers (``context.query`` - ``current.distance_nm``).
- **Lag ring** - for a field that is stacked, the frames its ring holds for
  the aircraft, and what each lagged field read at each step it was recorded
  (:class:`LagRecorder`), against the inner value then.
- **Cost** - each way the field can compute, timed on the traffic there is.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import importlib
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bluesky
import bluesky as bs
import numpy as np
from bluesky.stack.stackbase import Stack

from bluesky_sandbox.core import services
from bluesky_sandbox.interface.fields import _lag
from bluesky_sandbox.interface.fields._consistency import beyond_rounding, differs
from bluesky_sandbox.interface.fields.base import EnvBound, ObsField, PairObsField
from bluesky_sandbox.sim.geometry.conflict import invalidate_conflict_geometry

__all__ = ["LagRecorder", "Override", "ProbeResult", "Stage", "TraceNode", "probe"]

_PACKAGE = str(Path(__file__).resolve().parents[1])
_PROBE = str(Path(__file__).resolve().parent)
#: Where code no field is written in lives: Python's own, what is installed,
#: BlueSky and NumPy wherever they are.
_THIRD_PARTY = tuple(
    {
        *(str(Path(p).resolve()) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)),
        *(str(Path(p).resolve()) for p in sys.path if "site-packages" in p),
        str(Path(bluesky.__file__).resolve().parent),
        str(Path(np.__file__).resolve().parent),
    }
)


@dataclass(frozen=True)
class Override:
    """A value the probe sets before the field runs.

    ``target`` is ``"traf:<path>"`` - a BlueSky traffic value, ``alt`` or
    ``ap.trk``, set for ``aircraft`` - or ``"call:<module>:<qualname>"`` - what
    that function returns, whole, or the number at ``leaf`` in what it returns
    (``current.distance_nm``)."""

    target: str
    value: Any
    aircraft: str | None = None
    leaf: str = ""

    @property
    def kind(self) -> str:
        return self.target.split(":", 1)[0]


@dataclass(frozen=True)
class Stage:
    """One step from the field's inputs to what the policy sees."""

    name: str
    actual: Any
    expected: Any = None
    note: str = ""
    #: Whether ``actual`` and ``expected`` agree; None where there is no
    #: plain statement to hold it to.
    agrees: bool | None = None


@dataclass
class TraceNode:
    """A call the field made, or a traffic value it read."""

    label: str
    value: str = ""
    #: The override this node takes (``Override.target``), or None.
    key: str | None = None
    overridden: bool = False
    #: Numbers in what a call returned, each overridable: ``(leaf, value)``.
    leaves: list[tuple[str, str]] = field(default_factory=list)
    children: list[TraceNode] = field(default_factory=list)
    #: The traffic paths read under this call, filled in as it runs.
    reads: list[str] = field(default_factory=list, repr=False)


@dataclass
class ProbeResult:
    field: str
    kind: str
    aircraft: str
    other: str | None
    sim_time_s: float
    stages: list[Stage]
    trace: list[TraceNode]
    #: The field's unit, as its meta states it ("" when it has none).
    unit: str = ""
    #: An observation's bounds for the aircraft, and its normalizer across
    #: them and a fifth either side: ``[(raw, normalized)]``.
    bounds: tuple[float, float] | None = None
    curve: list[tuple[float, float]] = field(default_factory=list)
    #: A stepped action's every choice: its steps, value, value on the grid
    #: and the target it would command now.
    choices: list[dict[str, Any]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    lag: dict[str, Any] | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))


# --------------------------------------------------------------------------- #
# Probing                                                                     #
# --------------------------------------------------------------------------- #
def probe(
    env: Any,
    field_obj: Any,
    aircraft: str,
    *,
    other: str | None = None,
    overrides: tuple[Override, ...] | list[Override] = (),
    give: float | None = None,
    history: LagRecorder | None = None,
    timing_repeats: int = 5,
) -> ProbeResult:
    """Read ``field_obj`` - an observation or an action of ``env``'s config -
    for ``aircraft`` (about ``other``, for a pair field) as the env stands now.
    An action is given ``give``: what the policy gives it (a choice, for one
    with a step normalizer)."""
    given = field_obj
    if isinstance(field_obj, EnvBound) and getattr(field_obj, "env", None) is None:
        field_obj = field_obj.bind_env(env)
    ids = list(bs.traf.id)
    if aircraft not in ids:
        raise ValueError(f"{aircraft!r} is not in the air now")
    if other is not None and other not in ids:
        raise ValueError(f"{other!r} is not in the air now")
    own = ids.index(aircraft)
    oth = None if other is None else ids.index(other)
    is_action = callable(getattr(field_obj, "set", None)) and not isinstance(field_obj, (ObsField, PairObsField))
    is_pair = isinstance(field_obj, PairObsField)
    if is_pair and oth is None:
        raise ValueError(f"{type(field_obj).__name__} is read about another aircraft: name it")

    notes: list[str] = []
    names = {i: acid for i, acid in enumerate(ids)}
    recorder = _Recorder(names, own, oth)
    with _traffic_set(overrides, ids, own), _pinned_calls(overrides) as pinned:
        if is_action:
            stages, command, trace = _action(env, field_obj, own, give, recorder, notes)
        else:
            command = []
            stages, trace = _observation(field_obj, own, oth, is_pair, recorder, history, notes, bool(pinned))
        lag = None if is_action or is_pair else _lag_view(env, given, aircraft, own, history)
    _mark_overrides(trace, overrides, aircraft)
    bounds, curve = (None, []) if is_action else _bounds_and_curve(field_obj, own)
    return ProbeResult(
        field=_name(field_obj),
        kind="action" if is_action else "pair" if is_pair else "observation",
        aircraft=aircraft,
        other=other,
        sim_time_s=float(bs.sim.simt),
        stages=stages,
        trace=trace,
        unit=_unit(field_obj),
        bounds=bounds,
        curve=curve,
        choices=_choices(field_obj, own) if is_action else [],
        command=command,
        lag=lag,
        cost=_cost(env, field_obj, is_action, is_pair, timing_repeats),
        notes=notes,
    )


def _observation(field_obj, own, oth, is_pair, recorder, history, notes, pinned) -> tuple[list[Stage], list[TraceNode]]:
    every = np.arange(bs.traf.ntraf)
    if is_pair:
        actual = np.asarray(field_obj.get_pair_matrix(np.array([own])), dtype=np.float64)[0][oth]
        expected = field_obj.expected_pair(own, oth) if field_obj.states_expected() else None
        traced, trace = recorder.run(lambda: field_obj.get_pair(own, oth))
    else:
        actual = np.asarray(field_obj.get_many(every), dtype=np.float64)[own]
        expected = field_obj.expected(own) if field_obj.states_expected() else None
        traced, trace = recorder.run(lambda: field_obj.get(own))
    expected = _lag_expected(field_obj, own, history, expected)
    if pinned and beyond_rounding(np.asarray(traced, dtype=np.float64), actual, "raw"):
        # An override of a call applies where the call is made: the value is
        # the traced one's, which made it, and nothing it is held to need be.
        notes.append("the batched path does not make the call overridden: its value is one at a time's")
        raw = Stage("raw", np.asarray(traced, dtype=np.float64), expected)
        actual = raw.actual
    else:
        raw = _stage("raw", actual, expected)
    stages = [raw]
    if getattr(field_obj, "normalizer", None) is not None:
        batched = services._normalize_field_values_batch(field_obj, np.atleast_1d(actual)[None] if np.ndim(actual) else np.array([actual]), own)
        single = services._normalize_field_value(field_obj, expected if expected is not None else actual, own)
        batched = np.asarray(batched, dtype=np.float32).reshape(-1)[: np.size(single)]
        # What reaches the observation is float32: one value's normalization
        # and the batch's must agree there exactly.
        stages.append(Stage("normalized", batched, single, agrees=not differs(batched, single, "normalized")))
    return stages, trace


def _action(env, field_obj, own, give, recorder, notes) -> tuple[list[Stage], list[str], list[TraceNode]]:
    if give is None:
        raise ValueError(f"{type(field_obj).__name__} is an action: give it a value")
    stages = [Stage("policy", give)]
    acting = services._acting(field_obj, own)
    normalizer = getattr(acting, "normalizer", None)
    steps = getattr(normalizer, "steps", None)
    if callable(steps):
        choices = steps()
        stages.append(Stage("steps", choices[min(max(int(round(float(give))), 0), len(choices) - 1)]))
    value = services._denormalized(acting, np.array([give], dtype=np.float32), own)
    stages.append(Stage("value", value))
    on_grid = services._on_grid(acting, value, own)
    if getattr(acting, "grid", None) is not None:
        stages.append(Stage("on grid", on_grid))
    targets = getattr(acting, "targets", None)
    planned = float(np.asarray(targets(own, np.array([on_grid]))).reshape(-1)[0]) if callable(targets) else None
    queued = len(Stack.cmdstack)
    _applied, trace = recorder.run(lambda: env.apply_action(own, field_obj, value))
    if trace and trace[0].value == "False":
        notes.append(f"{type(field_obj).__name__} was held back: a mask, or a clearance's lock, keeps the axis")
    command = [line for line, _sender in Stack.cmdstack[queued:]]
    bs.stack.process()
    held = getattr(acting, "held", None)
    target = held(own) if callable(held) else None
    if planned is not None or target is not None:
        stages.append(_stage("target", target, planned))
    stages.append(Stage("BS command", "; ".join(command) if command else None))
    return stages, command, trace


def _choices(field_obj, own) -> list[dict[str, Any]]:
    acting = services._acting(field_obj, own)
    steps = getattr(getattr(acting, "normalizer", None), "steps", None)
    if not callable(steps):
        return []
    targets = getattr(acting, "targets", None)
    out = []
    for k, n in enumerate(steps()):
        value = services._denormalized(acting, np.array([k], dtype=np.float32), own)
        on_grid = services._on_grid(acting, value, own)
        target = float(np.asarray(targets(own, np.array([on_grid]))).reshape(-1)[0]) if callable(targets) else None
        out.append({"choice": k, "steps": n, "value": value, "on_grid": on_grid, "target": target})
    return out


def _bounds_and_curve(field_obj, own) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    try:
        low, high = (float(np.asarray(b).reshape(-1)[0]) for b in field_obj.bounds(own))
    except Exception:  # noqa: BLE001 - a field without bounds for one aircraft
        return None, []
    if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
        return (low, high), []
    if getattr(field_obj, "normalizer", None) is None or isinstance(field_obj, PairObsField):
        return (low, high), []
    margin = (high - low) / 5.0
    curve = []
    for raw in np.linspace(low - margin, high + margin, 43):
        try:
            normalized = services._normalize_field_value(field_obj, float(raw), own)
        except Exception:  # noqa: BLE001 - a normalizer that cannot take this value
            return (low, high), []
        if len(normalized) != 1:
            return (low, high), []
        curve.append((float(raw), float(normalized[0])))
    return (low, high), curve


def _unit(field_obj) -> str:
    unit = getattr(getattr(field_obj, "meta", None), "unit", "")
    text = str(getattr(unit, "value", unit) or "")
    return "" if text == "unitless" else text


def _stage(name: str, actual: Any, expected: Any) -> Stage:
    if expected is None or actual is None:
        return Stage(name, actual, expected)
    return Stage(name, actual, expected, agrees=not beyond_rounding(np.asarray(actual, dtype=np.float64), expected, name))


# --------------------------------------------------------------------------- #
# Overrides                                                                   #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _traffic_set(overrides, ids, own) -> Iterator[None]:
    """Each ``traf:`` override set in BlueSky - all that reads it sees it -
    and the value it had put back after."""
    before: list[tuple[Any, int, Any]] = []
    try:
        for ov in overrides:
            if ov.kind != "traf":
                continue
            idx = own if ov.aircraft is None else ids.index(ov.aircraft)
            owner, name = _traffic_owner(ov.target.split(":", 1)[1])
            values = getattr(owner, name)
            before.append((values, idx, copy.copy(values[idx])))
            values[idx] = ov.value
        if before:
            # Geometry cached for this sim time was computed before the change.
            invalidate_conflict_geometry()
        yield
    finally:
        for values, idx, value in reversed(before):
            values[idx] = value
        if before:
            invalidate_conflict_geometry()


def _traffic_owner(path: str) -> tuple[Any, str]:
    *parents, name = path.split(".")
    owner = bs.traf
    for part in parents:
        owner = getattr(owner, part)
    return owner, name


@contextlib.contextmanager
def _pinned_calls(overrides) -> Iterator[list[Override]]:
    """Each ``call:`` override patched in: its function returns the value
    given - whole, or at its leaf - wherever it is called, while in here."""
    calls = [ov for ov in overrides if ov.kind == "call"]
    restore: list[tuple[Any, str, Any]] = []
    try:
        by_target: dict[str, list[Override]] = {}
        for ov in calls:
            by_target.setdefault(ov.target, []).append(ov)
        for target, group in by_target.items():
            _kind, module, qualname = target.split(":", 2)
            *path, name = qualname.split(".")
            owner = importlib.import_module(module)
            for part in path:
                owner = getattr(owner, part)
            raw = vars(owner)[name]
            restore.append((owner, name, raw))
            setattr(owner, name, _pinning(raw, group))
        yield calls
    finally:
        for owner, name, raw in reversed(restore):
            setattr(owner, name, raw)


def _pinning(raw: Any, group: list[Override]) -> Any:
    def pin(result: Any) -> Any:
        for ov in group:
            result = ov.value if not ov.leaf else _with_leaf(result, ov.leaf.split("."), ov.value)
        return result

    if isinstance(raw, property):
        return property(lambda self, _get=raw.fget: pin(_get(self)))
    if isinstance(raw, functools.cached_property):
        return property(lambda self, _get=raw.func: pin(_get(self)))
    if isinstance(raw, staticmethod):
        return staticmethod(functools.wraps(raw.__func__)(lambda *a, **k: pin(raw.__func__(*a, **k))))
    if isinstance(raw, classmethod):
        return classmethod(functools.wraps(raw.__func__)(lambda cls, *a, **k: pin(raw.__func__(cls, *a, **k))))

    @functools.wraps(raw)
    def pinned(*args, **kwargs):
        return pin(raw(*args, **kwargs))

    return pinned


def _with_leaf(obj: Any, path: list[str], value: Any) -> Any:
    """A copy of ``obj`` with the number at ``path`` set to ``value`` - each
    object on the way copied, the original untouched, its type kept."""
    name, rest = path[0], path[1:]
    new_value = value if not rest else _with_leaf(getattr(obj, name), rest, value)
    new = copy.copy(obj)
    attr = getattr(type(obj), name, None)
    if isinstance(attr, property):
        # Computed on read: a subclass that reads the value given instead.
        sub = type(type(obj).__name__, (type(obj),), {name: property(lambda _self, v=new_value: v)})
        new.__class__ = sub
    else:
        object.__setattr__(new, name, new_value)
    return new


def _mark_overrides(nodes: list[TraceNode], overrides, aircraft: str) -> None:
    keys = {ov.target if ov.kind == "call" else f"{ov.target}@{ov.aircraft or aircraft}" for ov in overrides}
    for node in _walk(nodes):
        node.overridden = node.key in keys


# --------------------------------------------------------------------------- #
# Tracing                                                                     #
# --------------------------------------------------------------------------- #
class _Recorder:
    """Records the calls a field makes into the library and the design's code,
    and the BlueSky traffic values each reads."""

    def __init__(self, names: dict[int, str], own: int, other: int | None) -> None:
        self.names = names
        self.own = own
        self.other = other

    def run(self, call: Callable[[], Any]) -> tuple[Any, list[TraceNode]]:
        """What ``call`` returns, and the calls and reads it made."""
        root = TraceNode("root")
        stack: list[tuple[Any, TraceNode]] = [(None, root)]

        def profile(frame, event, arg):
            if event == "call" and _shown(frame):
                node = TraceNode(self._label(frame))
                code = frame.f_code
                node.key = f"call:{frame.f_globals.get('__name__', '')}:{code.co_qualname}"
                stack[-1][1].children.append(node)
                stack.append((frame, node))
            elif event == "return" and stack[-1][0] is frame:
                _frame, node = stack.pop()
                sys.setprofile(None)
                try:
                    node.value, node.leaves = _summary(arg, self.names)
                finally:
                    sys.setprofile(profile)

        def read(path: str) -> None:
            node = stack[-1][1]
            if path not in node.reads:
                node.reads.append(path)

        real = bluesky.traf
        bluesky.traf = _TrafficReads(real, "", read)
        sys.setprofile(profile)
        try:
            result = call()
        finally:
            sys.setprofile(None)
            bluesky.traf = real
        top = TraceNode(f"{getattr(call, '__name__', 'call')}", _summary(result, self.names)[0])
        top.children = root.children
        top.reads = root.reads
        for node in _walk([top]):
            node.children = [*self._reads(node.reads), *node.children]
        # The field's own call is the root shown: what it returned is the value.
        return result, top.children if len(top.children) == 1 and not top.reads else [top]

    def _label(self, frame) -> str:
        code = frame.f_code
        owner = frame.f_locals.get("self")
        name = f"{type(owner).__name__}.{code.co_name}" if owner is not None else code.co_name
        args = []
        for arg in code.co_varnames[1 if owner is not None else 0 : code.co_argcount]:
            value = frame.f_locals.get(arg)
            args.append(self._arg(arg, value))
        return f"{name}({', '.join(args)})"

    def _arg(self, name: str, value: Any) -> str:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) in self.names:
            return self.names[int(value)]
        if value is None or isinstance(value, (bool, int, float, str, np.number, np.ndarray, list, tuple)):
            return _short(value)
        return type(value).__name__

    def _reads(self, paths: list[str]) -> list[TraceNode]:
        out = []
        for path in paths:
            try:
                owner, name = _traffic_owner(path)
                values = getattr(owner, name)
            except Exception:  # noqa: BLE001 - shown as read, without a value
                out.append(TraceNode(f"bs.traf.{path}"))
                continue
            if isinstance(values, (np.ndarray, list)) and len(values) == bs.traf.ntraf:
                for idx in (self.own, self.other):
                    if idx is not None:
                        out.append(TraceNode(
                            f"bs.traf.{path}[{self.names[idx]}]", _short(values[idx]), key=f"traf:{path}",
                        ))
                        out[-1].leaves = []
                        out[-1].key = f"traf:{path}@{self.names[idx]}"
            elif isinstance(values, (int, float, np.number)):
                out.append(TraceNode(f"bs.traf.{path}", _short(values)))
        return out


class _TrafficReads:
    """``bs.traf`` as the field reads it: each attribute read reported, its
    value the real one (an object within it read the same way)."""

    _LEAVES = (np.ndarray, list, tuple, str, int, float, bool, np.number)

    def __init__(self, real: Any, prefix: str, report: Callable[[str], None]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_report", report)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        path = f"{self._prefix}{name}"
        if name.startswith("_") or callable(value) and not hasattr(value, "__dict__"):
            return value
        if isinstance(value, self._LEAVES) or value is None:
            self._report(path)
            return value
        if callable(value):
            return value
        return _TrafficReads(value, f"{path}.", self._report)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._real, name, value)


def _shown(frame) -> bool:
    """A call shown in the trace: public, into the library (outside the probe)
    or into the design's own code - not into Python, BlueSky or NumPy."""
    code = frame.f_code
    if code.co_name.startswith("_") or code.co_name.startswith("<"):
        return False
    filename = code.co_filename
    if filename.startswith(_PACKAGE):
        return not filename.startswith(_PROBE)
    if filename.startswith("<designer:"):
        return True
    return not filename.startswith("<") and not filename.startswith(_THIRD_PARTY)


def _walk(nodes: list[TraceNode]) -> Iterator[TraceNode]:
    for node in nodes:
        yield node
        yield from _walk(node.children)


def _summary(value: Any, names: dict[int, str]) -> tuple[str, list[tuple[str, str]]]:
    """``value`` in a few characters, and the numbers in it one can override."""
    if value is None or isinstance(value, (bool, int, float, np.number, str, np.ndarray, list, tuple, dict)):
        return _short(value), []
    leaves: list[tuple[str, str]] = []
    _leaves(value, "", leaves, depth=2)
    return type(value).__name__, leaves[:12]


def _leaves(obj: Any, prefix: str, out: list, depth: int) -> None:
    for name in _public(obj):
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 - a value that cannot be read is not shown
            continue
        if isinstance(value, bool) or callable(value):
            continue
        if isinstance(value, (int, float, np.number)):
            out.append((prefix + name, _short(value)))
        elif depth > 1 and not isinstance(value, (str, np.ndarray, list, tuple, dict)) and value is not None:
            _leaves(value, f"{prefix}{name}.", out, depth - 1)


def _public(obj: Any) -> list[str]:
    if dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
    else:
        names = [n for n, v in vars(type(obj)).items() if isinstance(v, (property, functools.cached_property))]
    return [n for n in names if not n.startswith("_")]


def _short(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return f"{float(flat[0]):.6g}" if flat.size == 1 else f"array({value.shape})"
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


# --------------------------------------------------------------------------- #
# Lag                                                                         #
# --------------------------------------------------------------------------- #
class LagRecorder:
    """What a stacked field's inner value was, and what each of its lagged
    fields read, for every aircraft at every step it is told to record - the
    independent account a lag ring is checked against."""

    def __init__(self, env: Any) -> None:
        self.groups: dict[str, tuple[Any, list[Any]]] = {}
        for item in _config_fields(env):
            if _is_lag(item):
                self.groups.setdefault(item._key, (item.inner, []))[1].append(item)
        #: Per stack and aircraft - ``(key, callsign)`` - each step recorded:
        #: ``(sim time, inner value, {steps: what that lag read})``.
        self.rows: dict[tuple[str, str], list[tuple[float, float, dict[int, float]]]] = {}
        #: The sim time of the first step recorded.
        self.start_s: float | None = None

    def record(self) -> None:
        ids = list(bs.traf.id)
        if not ids:
            return
        every = np.arange(len(ids))
        simt = float(bs.sim.simt)
        if self.start_s is None:
            self.start_s = simt
        for key, (inner, lags) in self.groups.items():
            values = np.asarray(inner.get_many(every), dtype=np.float64).reshape(len(ids), -1)[:, 0]
            reads = {int(lag.steps): np.asarray(lag.get_many(every), dtype=np.float64).reshape(len(ids), -1)[:, 0] for lag in lags}
            for i, acid in enumerate(ids):
                self.rows.setdefault((key, acid), []).append((simt, float(values[i]), {k: float(v[i]) for k, v in reads.items()}))


def _lag_expected(field_obj, own, history, expected):
    """A lagged field's plain statement: its inner value ``steps`` records
    back - held at the oldest where there are fewer."""
    if history is None or not _is_lag(field_obj):
        return expected
    rows = history.rows.get((field_obj._key, bs.traf.id[own]))
    if not rows:
        return expected
    back = min(int(field_obj.steps), len(rows) - 1)
    return rows[-1 - back][1]


def _is_lag(item: Any) -> bool:
    return isinstance(item, ObsField) and isinstance(getattr(item, "_key", None), str) and hasattr(item, "steps")


def _config_fields(env: Any) -> list[Any]:
    out = []
    for config_field in dataclasses.fields(env.config):
        value = getattr(env.config, config_field.name)
        if isinstance(value, (list, tuple)):
            out.extend(value)
    return out


def _lag_view(env, field_obj, acid, own, history) -> dict[str, Any] | None:
    """The ring a stacked field's lags read - ``field_obj`` the stack's live
    field or one of its lags - with each frame's value and who reads it."""
    inner = field_obj.inner if _is_lag(field_obj) else field_obj
    # The ring is keyed when the lag is built: matched by the field itself,
    # as what it prints can change after (a normalizer bound to the env).
    lags = [item for item in _config_fields(env) if _is_lag(item) and item.inner is inner]
    if not lags:
        return None
    key = lags[0]._key
    ring = _lag._LAG_HISTORY.get(("obs", key))
    frames: list[dict[str, Any]] = []
    depth = None
    if ring is not None:
        ring.sync(_lag.aircraft_keys(bs.traf.id))
        row = ring.row.get(_lag.aircraft_keys(bs.traf.id)[own])
        depth = ring.depth
        if row is not None and ring.values is not None:
            count, head = int(ring.count[row]), int(ring.head[row])
            for i in range(count):
                slot = (head - (count - 1 - i)) % ring.depth
                back = count - 1 - i
                readers = [int(lag.steps) for lag in lags if min(int(lag.steps), count - 1) == back]
                if back == 0:
                    readers.insert(0, 0)
                frames.append({
                    "back": back,
                    "value": float(np.asarray(ring.values[slot, row]).reshape(-1)[0]),
                    "read_by": readers,
                    "placeholder": [k for k in readers if k > count - 1],
                })
    rows = [] if history is None else history.rows.get((key, acid), [])
    if rows and len(rows) >= len(frames):
        for frame in frames:
            frame["sim_time_s"] = rows[-1 - frame["back"]][0]
    steps = sorted({int(lag.steps) for lag in lags})
    progression = []
    for n, (simt, _value, reads) in enumerate(rows[-(max(steps) + 2):], start=max(0, len(rows) - (max(steps) + 2))):
        cells = []
        for k in [0, *steps]:
            back = min(k, n)
            want = rows[n - back][1]
            got = rows[n][1] if k == 0 else reads.get(k)
            cells.append({"steps": k, "value": got, "expected": want, "placeholder": k > n,
                          "agrees": None if got is None else not beyond_rounding(np.float64(got), want, "")})
        progression.append({"sim_time_s": simt, "cells": cells})
    return {
        "inner": _name(inner),
        "lags": steps,
        "depth": depth,
        "frames": frames,
        "progression": progression,
        # When this aircraft's history begins, and the episode's: later, it
        # spawned since.
        "first_s": rows[0][0] if rows else None,
        "start_s": history.start_s if history is not None else None,
        "fields": [_name(inner), *[f"{_name(inner)}_lag{k}" for k in steps]],
    }


# --------------------------------------------------------------------------- #
# Cost                                                                        #
# --------------------------------------------------------------------------- #
def _cost(env, field_obj, is_action, is_pair, repeats) -> dict[str, Any]:
    n = bs.traf.ntraf
    if is_action or n == 0:
        return {}
    every = list(range(n))
    if is_pair:
        paths = {
            "pair matrix": lambda: field_obj.get_pair_matrix(np.array(every)),
            "per ownship": lambda: [field_obj.get_pairs(i, [j for j in every if j != i]) for i in every],
            "per pair": lambda: [field_obj.get_pair(i, j) for i in every for j in every if j != i],
        }
        batched = type(field_obj).get_pair_matrix is not PairObsField.get_pair_matrix
    else:
        paths = {
            "batched": lambda: field_obj.get_many(every),
            "one at a time": lambda: [field_obj.get(i) for i in every],
        }
        batched = type(field_obj).get_many is not ObsField.get_many
    timings = {name: _median_ms(path, repeats) for name, path in paths.items()}
    most = int(getattr(env, "episode_max_aircraft", n) or n)
    scale = (most / n) ** (2 if is_pair else 1) if n else 1.0
    return {
        "aircraft": n,
        "max_aircraft": most,
        "ms": timings,
        "ms_at_max": {name: ms * scale for name, ms in timings.items()},
        "batched_path": batched,
    }


def _median_ms(path: Callable[[], Any], repeats: int) -> float:
    times = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        path()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _name(field_obj: Any) -> str:
    meta = getattr(field_obj, "meta", None)
    return str(getattr(meta, "name", "") or type(field_obj).__name__)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
