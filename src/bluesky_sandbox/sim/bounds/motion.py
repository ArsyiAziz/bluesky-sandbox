"""How a region moves during an episode: motions, applied to its shape as the
simulation runs.

A region's :class:`MovingFootprint` holds the shape it starts the episode
with and its motions; :meth:`MovingFootprint.advance` sets it to how it is
``t`` seconds in. It is one object, held by everything referring to the region
- a queryable, the airspace, a spawn region, ``ctx.shape(...)`` - so all of
them see it move; the environment advances every one each simulation
substep. Each motion's parameters take a fixed value, a ``(low, high)`` range
or a scipy distribution, drawn once per episode (:meth:`Motion.realized`).

Motions compose in order, each applied to the shape the ones before it left:

- :class:`Drift` - moves at a heading and speed; ``within`` a region, it
  bounces off the region's box, the whole shape kept inside it.
- :class:`Spin` - turns about its center at a rate.
- :class:`Grow` - grows or shrinks about its center at a rate, within limits.

Each carries the shape through a point map (``Footprint.mapped``), so it keeps
its form - a circle stays a circle. A motion of your own is a subclass with
``apply(footprint, t_s)``, referenced by import path where a name is asked for.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from bluesky_sandbox.sim.sampling.distributions import sample_scalar

from .base import Bounds, Footprint, RegionBounds
from .coordinates import LatLon, LocalFrame
from .placement import moved_to

__all__ = ["Carrier", "Drift", "Grow", "Motion", "MovingFootprint", "Spin", "moving_in"]

#: A parameter: a fixed value, a ``(low, high)`` range, or a scipy distribution.
Value = Any


def _is_drawn(value: Any) -> bool:
    """Whether ``value`` is drawn per episode: a range or a distribution."""
    if hasattr(value, "rvs"):
        return True
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, (int, float)) for v in value)
    )


def scaled(footprint: Footprint, factor: float) -> Footprint:
    """``footprint`` scaled by ``factor`` about its center."""
    center = footprint.center_point()
    frame = LocalFrame(center)

    def point(lat: float, lon: float) -> tuple[float, float]:
        x, y = frame.to_xy_nm(LatLon(lat, lon))
        p = frame.from_xy_nm(x * factor, y * factor)
        return p.lat_deg, p.lon_deg

    return footprint.mapped(point)


@dataclass(frozen=True)
class Motion(ABC):
    """How a shape moves with time."""

    @abstractmethod
    def apply(self, footprint: Footprint, t_s: float) -> Footprint:
        """``footprint`` - as the motions before this one left it - ``t_s``
        seconds into the episode."""

    def about(self, t_s: float, pivot: LatLon, extent: Footprint):
        """How this motion moves a whole group ``t_s`` seconds in: a point map
        ``(lat, lon) -> (lat, lon)`` about the group's ``pivot``, its members
        spanning ``extent`` - so they move as one, keeping their places. A
        motion of your own moves a group once it has this."""
        raise NotImplementedError(
            f"{type(self).__name__} moves a bound, not a group: it has no about()"
        )

    def realized(self, rng: np.random.Generator) -> Motion:
        """This motion with each drawn parameter drawn - for an episode."""
        changes = {
            f.name: sample_scalar(getattr(self, f.name), rng)
            for f in dataclasses.fields(self)
            if f.init and _is_drawn(getattr(self, f.name))
        }
        return dataclasses.replace(self, **changes) if changes else self

    def turned(self, turn_deg: float, scale: float) -> Motion:
        """This motion in a scene turned ``turn_deg`` and scaled by ``scale``
        (a group's transform) - a heading turned with it, a speed scaled. As
        it is, for a motion with neither."""
        return self

    def mapped(self, point) -> Motion:
        """This motion carried through ``point`` - a region it keeps within
        moved with it."""

        def move(value: Any) -> Any:
            if isinstance(value, RegionBounds):
                return RegionBounds(value.footprint.mapped(point), value.altitude, value.name)
            if isinstance(value, Footprint):
                return value.mapped(point)
            return value

        changes = {f.name: move(getattr(self, f.name)) for f in dataclasses.fields(self) if f.init}
        return dataclasses.replace(self, **changes)


def _about(pivot: LatLon, fn):
    """The point map applying ``fn(x, y) -> (x, y)`` in nm about ``pivot``."""
    frame = LocalFrame(pivot)

    def point(lat: float, lon: float) -> tuple[float, float]:
        x, y = frame.to_xy_nm(LatLon(lat, lon))
        p = frame.from_xy_nm(*fn(x, y))
        return p.lat_deg, p.lon_deg

    return point


def _unit(lat: float, lon: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _carried_to(start: LatLon, end: LatLon):
    """The rigid move of the sphere taking ``start`` to ``end`` along the great
    circle between them: every point carried with it, distances and angles
    kept - a group moved anywhere keeps its formation."""
    a, b = _unit(start.lat_deg, start.lon_deg), _unit(end.lat_deg, end.lon_deg)
    axis = np.cross(a, b)
    sin_t, cos_t = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if sin_t < 1e-15:
        return lambda lat, lon: (lat, lon)
    k = axis / sin_t

    def point(lat: float, lon: float) -> tuple[float, float]:
        v = _unit(lat, lon)
        r = v * cos_t + np.cross(k, v) * sin_t + k * float(np.dot(k, v)) * (1.0 - cos_t)
        return math.degrees(math.asin(max(-1.0, min(1.0, r[2])))), math.degrees(math.atan2(r[1], r[0]))

    return point


def _turning(pivot: LatLon, turn_deg: float):
    """Turns about ``pivot`` by ``turn_deg`` - clockwise when positive."""
    a = math.radians(turn_deg)
    c, s_ = math.cos(a), math.sin(a)
    return _about(pivot, lambda x, y: (x * c + y * s_, -x * s_ + y * c))


def _bounce(x: float, lo: float, hi: float) -> float:
    """``x`` reflected back and forth between ``lo`` and ``hi``."""
    span = hi - lo
    if span <= 0.0:
        return (lo + hi) / 2.0
    u = (x - lo) % (2.0 * span)
    return lo + (u if u <= span else 2.0 * span - u)


@dataclass(frozen=True)
class Drift(Motion):
    """Moves at ``heading_deg`` (true) and ``speed_kts``. ``within`` a region,
    it bounces off the region's box - the box of its outline, in nm - keeping
    the whole shape inside it."""

    heading_deg: Value = 0.0
    speed_kts: Value = 10.0
    within: Bounds | Footprint | None = None

    def turned(self, turn_deg: float, scale: float) -> Motion:
        if _is_drawn(self.heading_deg) or _is_drawn(self.speed_kts):
            # Drawn after the turn: a range of headings is turned whole.
            heading = self.heading_deg
            if isinstance(heading, tuple):
                heading = (heading[0] + turn_deg, heading[1] + turn_deg)
            speed = self.speed_kts
            if isinstance(speed, tuple):
                speed = (speed[0] * scale, speed[1] * scale)
            return dataclasses.replace(self, heading_deg=heading, speed_kts=speed)
        return dataclasses.replace(
            self,
            heading_deg=(float(self.heading_deg) + turn_deg) % 360.0,
            speed_kts=float(self.speed_kts) * scale,
        )

    def _to(self, start: LatLon, extent: Footprint, t_s: float) -> LatLon:
        """Where ``start`` - spanning ``extent`` - drifts to in ``t_s``."""
        heading = math.radians(float(self.heading_deg))
        distance = float(self.speed_kts) * t_s / 3600.0
        if self.within is None:
            return LocalFrame(start).offset(math.degrees(heading), distance)
        region = self.within.footprint if isinstance(self.within, RegionBounds) else self.within
        frame = LocalFrame(region.center_point())
        reach = 0.0
        sx, sy = frame.to_xy_nm(start)
        for lat, lon in extent.vertices:
            x, y = frame.to_xy_nm(LatLon(lat, lon))
            reach = max(reach, abs(x - sx), abs(y - sy))
        lon_lo, lat_lo, lon_hi, lat_hi = region.outline().bounds
        x_lo, y_lo = frame.to_xy_nm(LatLon(lat_lo, lon_lo))
        x_hi, y_hi = frame.to_xy_nm(LatLon(lat_hi, lon_hi))
        x = _bounce(sx + distance * math.sin(heading), x_lo + reach, x_hi - reach)
        y = _bounce(sy + distance * math.cos(heading), y_lo + reach, y_hi - reach)
        return frame.from_xy_nm(x, y)

    def apply(self, footprint: Footprint, t_s: float) -> Footprint:
        return moved_to(footprint, self._to(footprint.center_point(), footprint, t_s))

    def about(self, t_s: float, pivot: LatLon, extent: Footprint):
        return _carried_to(pivot, self._to(pivot, extent, t_s))


@dataclass(frozen=True)
class Spin(Motion):
    """Turns about its center at ``rate_deg_s`` - clockwise when positive."""

    rate_deg_s: Value = 0.1

    def apply(self, footprint: Footprint, t_s: float) -> Footprint:
        return moved_to(footprint, footprint.center_point(), float(self.rate_deg_s) * t_s)

    def about(self, t_s: float, pivot: LatLon, extent: Footprint):
        return _turning(pivot, float(self.rate_deg_s) * t_s)


@dataclass(frozen=True)
class Grow(Motion):
    """Grows - or, at a negative rate, shrinks - about its center by
    ``rate_per_hr`` of its starting size an hour, kept between
    ``min_scale`` and ``max_scale`` of it."""

    rate_per_hr: Value = 0.25
    min_scale: float = 0.2
    max_scale: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.min_scale) <= float(self.max_scale):
            raise ValueError(
                f"Grow needs 0 < min_scale <= max_scale, got {self.min_scale}, {self.max_scale}"
            )

    def _factor(self, t_s: float) -> float:
        factor = 1.0 + float(self.rate_per_hr) * t_s / 3600.0
        return min(max(factor, float(self.min_scale)), float(self.max_scale))

    def apply(self, footprint: Footprint, t_s: float) -> Footprint:
        factor = self._factor(t_s)
        return footprint if factor == 1.0 else scaled(footprint, factor)

    def about(self, t_s: float, pivot: LatLon, extent: Footprint):
        k = self._factor(t_s)
        return _about(pivot, lambda x, y: (x * k, y * k))


@dataclass(frozen=True)
class Carrier:
    """A group's motion, carrying a member: ``motions`` (drawn for the
    episode) about the group's ``pivot``, its members spanning ``extent`` -
    both where the episode starts them."""

    motions: tuple[Motion, ...]
    pivot: LatLon
    extent: Footprint

    def apply(self, footprint: Footprint, t_s: float) -> Footprint:
        """``footprint`` carried ``t_s`` seconds: each motion about where the
        ones before it left the group's pivot."""
        pivot, extent = self.pivot, self.extent
        for motion in self.motions:
            point = motion.about(t_s, pivot, extent)
            footprint = footprint.mapped(point)
            pivot = LatLon(*point(pivot.lat_deg, pivot.lon_deg))
            extent = extent.mapped(point)
        return footprint

    def mapped(self, point) -> Carrier:
        """This carrier in a scene moved by ``point``."""
        north = LocalFrame(self.pivot).offset(0.0, 1.0)
        to = LatLon(*point(self.pivot.lat_deg, self.pivot.lon_deg))
        x, y = LocalFrame(to).to_xy_nm(LatLon(*point(north.lat_deg, north.lon_deg)))
        turn, scale = math.degrees(math.atan2(x, y)), math.hypot(x, y)
        return Carrier(
            tuple(m.mapped(point).turned(turn, scale) for m in self.motions),
            to,
            self.extent.mapped(point),
        )


@dataclass
class MovingFootprint(Footprint):
    """``footprint`` moving by ``motions``: as it is at the time last set by
    :meth:`advance` - at the episode's start until then.

    ``update`` - how often the environment moves it: once every environment
    ``"step"``, at its end (the default - between, it holds), or every
    simulation ``"substep"`` (what is inside is of the shape at each, at a
    substep's cost). ``carriers`` - the motions of the groups it is in, inner
    first: applied after its own, so a group moves its members as one. Its
    ``version`` counts its moves, for what caches its geometry."""

    footprint: Footprint
    motions: tuple[Motion, ...] = ()
    update: str = "step"
    carriers: tuple[Carrier, ...] = ()

    def __post_init__(self) -> None:
        if self.update not in ("substep", "step"):
            raise ValueError(f"MovingFootprint update must be 'substep' or 'step', got {self.update!r}")
        self.motions = tuple(self.motions)
        self.carriers = tuple(self.carriers)
        self.t_s = 0.0
        self.version = 0
        self._current = self.footprint

    def advance(self, t_s: float) -> None:
        """Set it to how it is ``t_s`` seconds into the episode."""
        t_s = float(t_s)
        if t_s == self.t_s and self.version:
            return
        current = self.footprint
        for motion in self.motions:
            current = motion.apply(current, t_s)
        for carrier in self.carriers:
            current = carrier.apply(current, t_s)
        self._current = current
        self.t_s = t_s
        self.version += 1

    @property
    def animatable(self) -> bool:
        """Whether it can be moved: every motion's ranges drawn - an episode's
        region, not a design's."""
        every = (*self.motions, *(m for c in self.carriers for m in c.motions))
        return bool(every) and not any(
            _is_drawn(getattr(m, f.name)) for m in every for f in dataclasses.fields(m)
        )

    @property
    def current(self) -> Footprint:
        """The shape now."""
        return self._current

    @property
    def shape(self):
        return self._current.shape

    @property
    def bounding_box(self):
        return self._current.bounding_box

    @property
    def vertices(self):
        return self._current.vertices

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        return self._current.contains(lat_deg, lon_deg)

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        return self._current.sample_point(rng)

    def outline(self):
        return self._current.outline()

    def circles(self):
        return self._current.circles()

    def center_point(self) -> LatLon:
        return self._current.center_point()

    def axis_deg(self) -> float | None:
        return self._current.axis_deg()

    def mapped(self, point) -> Footprint:
        # The scene's turn and scale here, so headings and speeds turn with it.
        center = self.footprint.center_point()
        north = LocalFrame(center).offset(0.0, 1.0)
        to = LatLon(*point(center.lat_deg, center.lon_deg))
        x, y = LocalFrame(to).to_xy_nm(LatLon(*point(north.lat_deg, north.lon_deg)))
        turn, scale = math.degrees(math.atan2(x, y)), math.hypot(x, y)
        moved = MovingFootprint(
            self.footprint.mapped(point),
            tuple(m.mapped(point).turned(turn, scale) for m in self.motions),
            self.update,
            tuple(c.mapped(point) for c in self.carriers),
        )
        if self.version:
            moved.advance(self.t_s)
        return moved


def moving_in(*objects: Any) -> list[MovingFootprint]:
    """Every moving footprint in ``objects`` - an episode's resources: its
    regions, queryables, spawn regions and their routes - each once."""
    found: dict[int, MovingFootprint] = {}
    seen: set[int] = set()

    def visit(obj: Any, depth: int) -> None:
        if obj is None or depth > 8 or id(obj) in seen:
            return
        if isinstance(obj, (str, bytes, int, float, bool, np.ndarray)):
            return
        seen.add(id(obj))
        if isinstance(obj, RegionBounds):
            if isinstance(obj.footprint, MovingFootprint):
                found[id(obj.footprint)] = obj.footprint
            return
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for value in obj:
                visit(value, depth + 1)
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                visit(getattr(obj, f.name, None), depth + 1)

    for obj in objects:
        visit(obj, 0)
    return list(found.values())
