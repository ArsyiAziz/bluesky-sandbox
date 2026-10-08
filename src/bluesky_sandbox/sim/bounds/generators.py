"""Shapes drawn anew each episode: a generator and its parameters, in place of
a fixed footprint.

A :class:`ShapeGenerator` draws a footprint from a random generator
(:meth:`ShapeGenerator.generate`), and says where every shape it can draw lies
(:meth:`ShapeGenerator.envelope`) - what stands for it wherever a fixed shape
is needed: an environment's support, the airspace checks, the map. A named
region whose footprint is a :class:`GeneratedFootprint` is redrawn each
episode by :func:`generate_regions`, and everything referencing it follows.

What a generator draws is one of the ordinary footprints - a polygon - so its
boundary, containment and drawing work as for any other. Its parameters take a
fixed value, a ``(low, high)`` range, or a scipy distribution, each drawn per
episode.

The generators and their methods:

- :class:`ConvexPolygon` - Valtr's algorithm: uniform over convex polygons of
  n vertices (P. Valtr, "Probability that n random points are in convex
  position", Discrete & Computational Geometry 13, 1995), turned and scaled
  to a radius.
- :class:`Blob` - a radius about a center that varies with bearing as a random
  Fourier series: a smooth, star-shaped closed curve, never self-intersecting
  (random Fourier descriptors).
- :class:`VoronoiSectors` - a Voronoi tessellation of random seed points,
  clipped to a parent region: its sectors (as in Voronoi-based airspace
  sectorization, e.g. M. Xue, "Airspace sector redesign based on Voronoi
  diagrams", JACIC 6, 2009). Computed by shapely.

A generator of your own is a subclass with ``_draw(rng)`` and ``envelope()``,
referenced by import path where a name is asked for.
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Point
from shapely.geometry import Polygon as _Polygon

from bluesky_sandbox.sim.sampling.distributions import sample_scalar

from .base import Bounds, Footprint, RegionBounds
from .coordinates import LatLon, LocalFrame
from .footprints import DiskFootprint, PolygonFootprint, ShapelyFootprint
from .motion import MovingFootprint
from .placement import PlacedFootprint, PlacementError

__all__ = [
    "Blob",
    "ConvexPolygon",
    "GeneratedFootprint",
    "GenerationError",
    "ShapeGenerator",
    "VoronoiSectors",
    "generate_regions",
]

#: A parameter: a fixed value, a ``(low, high)`` range, or a scipy distribution.
Value = Any


class GenerationError(ValueError):
    """A generator could not draw a shape meeting its constraints."""


def _upper(value: Value) -> float:
    """The largest value ``value`` can take."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(max(value))
    if hasattr(value, "support"):
        high = float(value.support()[1])
        if math.isfinite(high):
            return high
    raise ValueError(f"{value!r} has no finite upper bound: a generator's envelope needs one")


def _check_value(name: str, value: Value, low: float, high: float = math.inf) -> None:
    """``value`` - and the whole range it can take - within ``[low, high]``."""
    if isinstance(value, (int, float)):
        values = [float(value)]
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        values = [float(v) for v in value]
        if values[0] > values[1]:
            raise ValueError(f"{name} range must be (low, high), got {value!r}")
    elif hasattr(value, "rvs"):
        return
    else:
        raise ValueError(f"{name} must be a number, a (low, high) range or a distribution, got {value!r}")
    if any(not (low <= v <= high) for v in values):
        raise ValueError(f"{name} must lie in [{low:g}, {high:g}], got {value!r}")


def _points(frame: LocalFrame, xy: np.ndarray) -> list[tuple[float, float]]:
    """``(lat, lon)`` of each ``(x, y)`` nm in ``frame``."""
    p = frame.from_xy_nm(xy[:, 0], xy[:, 1])
    return list(zip(np.asarray(p.lat_deg).tolist(), np.asarray(p.lon_deg).tolist()))


def _xy(frame: LocalFrame, shape) -> Any:
    """A shapely shape in lon/lat, in ``frame``'s nm."""

    def to_xy(coords):
        lon, lat = np.asarray(coords, dtype=np.float64).T
        x, y = frame.to_xy_nm(LatLon(lat, lon))
        return np.column_stack([x, y])

    return shapely.transform(shape, to_xy)


def area_nm2(footprint: Footprint) -> float:
    """``footprint``'s area, nm²."""
    return float(_xy(LocalFrame(footprint.center_point()), footprint.outline()).area)


# --------------------------------------------------------------------------- #
# The interface                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShapeGenerator(ABC):
    """Draws a footprint per episode.

    ``contains`` - points every shape must contain; ``within`` - a region every
    shape must lie inside; ``min_area_nm2`` - the smallest shape allowed. A
    draw that misses one is redrawn, up to ``max_tries`` times, after which
    :class:`GenerationError` names the generator and what failed.
    """

    contains: Sequence[LatLon] = ()
    within: Bounds | Footprint | None = None
    min_area_nm2: float | None = None
    max_tries: int = 100

    #: How many shapes one draw gives: one, or - a partition - several, which
    #: a region of name ``n`` holds as ``n.0``, ``n.1``, ...
    count: int = field(default=1, init=False)
    #: For a partition, the parameter its number of shapes is set by - which a
    #: designer reads to name them - else None.
    partition: ClassVar[str | None] = None

    @abstractmethod
    def _draw(self, rng: np.random.Generator) -> list[Footprint]:
        """One unconstrained draw: ``count`` footprints."""

    @abstractmethod
    def envelope(self) -> Footprint:
        """A footprint every shape this generator can draw lies inside."""

    def mapped(self, point) -> ShapeGenerator:
        """This generator carried through ``point`` (a ``(lat, lon) -> (lat,
        lon)`` map: a move, a rotation): every point and region it is given
        moved with it - its center, a parent, a region to stay within. What it
        draws is drawn there; a turn is no matter to a shape drawn at random."""

        def move(value: Any) -> Any:
            if isinstance(value, LatLon):
                return LatLon(*point(value.lat_deg, value.lon_deg))
            if isinstance(value, RegionBounds):
                return RegionBounds(value.footprint.mapped(point), value.altitude, value.name)
            if isinstance(value, Footprint):
                return value.mapped(point)
            if isinstance(value, (list, tuple)) and value and all(isinstance(v, LatLon) for v in value):
                return tuple(move(v) for v in value)
            return value

        changes = {f.name: move(getattr(self, f.name)) for f in dataclasses.fields(self) if f.init}
        return dataclasses.replace(self, **changes)

    def generate(self, rng: np.random.Generator) -> list[Footprint]:
        """``count`` footprints meeting the constraints."""
        failure = ""
        for _ in range(max(1, int(self.max_tries))):
            shapes = self._draw(rng)
            failure = self._failure(shapes)
            if not failure:
                return shapes
        raise GenerationError(
            f"{type(self).__name__} drew no shape {failure} in {self.max_tries} tries"
        )

    def _failure(self, shapes: list[Footprint]) -> str:
        """What a draw misses, or "" when it meets every constraint."""
        within = self.within.footprint if isinstance(self.within, RegionBounds) else self.within
        for shape in shapes:
            for p in self.contains:
                if not shape.contains(p.lat_deg, p.lon_deg):
                    return f"containing ({p.lat_deg:g}, {p.lon_deg:g})"
            if within is not None and not within.outline().buffer(1e-9).covers(shape.outline()):
                return "within its region"
            if self.min_area_nm2 is not None and area_nm2(shape) < float(self.min_area_nm2):
                return f"of at least {float(self.min_area_nm2):g} nm²"
        return ""


def _rng_for(base: int, name: str) -> np.random.Generator:
    """The random stream of region ``name`` this episode: its own, so adding,
    removing or reordering generated regions draws the others no differently."""
    return np.random.default_rng([base, zlib.crc32(name.encode())])


@dataclass
class GeneratedFootprint(Footprint):
    """A named region's footprint drawn by ``generator`` each episode.

    As a fixed shape - before any episode, in an environment's support - it is
    the generator's envelope. :func:`generate_regions` replaces it with a draw.
    """

    generator: ShapeGenerator

    def __post_init__(self) -> None:
        self._envelope = self.generator.envelope()

    @property
    def shape(self):
        return self._envelope.shape

    @property
    def bounding_box(self):
        return self._envelope.bounding_box

    @property
    def vertices(self):
        return self._envelope.vertices

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        return self._envelope.contains(lat_deg, lon_deg)

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        return self._envelope.sample_point(rng)

    def outline(self):
        return self._envelope.outline()

    def circles(self):
        return self._envelope.circles()

    def center_point(self) -> LatLon:
        return self._envelope.center_point()

    def mapped(self, point) -> Footprint:
        # Still drawn each episode - by the generator, moved.
        return GeneratedFootprint(self.generator.mapped(point))


def _named_in(obj: Any) -> set[str]:
    """The names of the named regions a generator's or placement's params
    stand in for."""
    found: set[str] = set()
    for f in dataclasses.fields(obj):
        values = getattr(obj, f.name)
        for value in values if isinstance(values, (list, tuple)) else (values,):
            if isinstance(value, RegionBounds) and value.name:
                found.add(value.name)
    return found


def _with_drawn(obj: Any, drawn: Mapping[str, Bounds]) -> Any:
    """``obj`` - a generator or a placement - with each param standing in for
    a named region replaced by this episode's draw of it."""

    def swap(value: Any) -> Any:
        if isinstance(value, RegionBounds) and value.name in drawn:
            return drawn[value.name]
        if isinstance(value, tuple):
            return tuple(swap(v) for v in value)
        return value

    changes = {f.name: swap(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init}
    return dataclasses.replace(obj, **changes)


def _draw_and_place(
    footprint: Footprint,
    generator: ShapeGenerator | None,
    placement: Any,
    rng: np.random.Generator,
) -> list[Footprint]:
    """A region's shapes this episode: drawn by its generator, then placed by
    its placement. A generator's constraints hold where the shape ends up - so
    a placed shape is redrawn and replaced until they do."""
    if generator is None:
        return [footprint if placement is None else placement.place(footprint, rng)]
    if placement is None:
        return generator.generate(rng)
    failure = ""
    for _ in range(max(1, int(generator.max_tries))):
        placed = [placement.place(generator._draw(rng)[0], rng)]
        failure = generator._failure(placed)
        if not failure:
            return placed
    raise GenerationError(
        f"{type(generator).__name__} drew and placed no shape {failure} in {generator.max_tries} tries"
    )


def generate_regions(
    regions: Mapping[str, Bounds], rng: np.random.Generator
) -> dict[str, Bounds]:
    """``regions``, each one that changes per episode drawn for this one: its
    shape, where a generator draws it, then its spot, where a placement
    places it.

    A region of one shape is replaced by its draw; a partition's shapes are
    added as ``<name>.0``, ``<name>.1``, ... beside the region, which stays the
    envelope. Each keeps the region's altitude band. A region a generator or a
    placement refers to (a parent, a region to keep within or clear of) is
    drawn first, and its draw is what it refers to. One number is drawn from
    ``rng`` - only when there is such a region - and each region draws from
    its own stream seeded by it and its name.
    """
    out = dict(regions)
    pending = {
        name: b
        for name, b in regions.items()
        if isinstance(b, RegionBounds)
        and isinstance(b.footprint, (GeneratedFootprint, PlacedFootprint, MovingFootprint))
    }
    if not pending:
        return out
    base = int(rng.integers(2**63))

    def parts(bounds: RegionBounds):
        """A region's layers: its shape (or generator), its placement, its
        motions and how often they update - None or () where it has none."""
        footprint, placement, motions, update = bounds.footprint, None, (), "step"
        if isinstance(footprint, MovingFootprint):
            footprint, motions, update = footprint.footprint, footprint.motions, footprint.update
        if isinstance(footprint, PlacedFootprint):
            footprint, placement = footprint.footprint, footprint.placement
        generator = footprint.generator if isinstance(footprint, GeneratedFootprint) else None
        return footprint, generator, placement, motions, update

    def depends_on(bounds: RegionBounds) -> set[str]:
        _, generator, placement, motions, _ = parts(bounds)
        names = set()
        for obj in (generator, placement, *motions):
            if obj is not None:
                names |= _named_in(obj)
        # One of a partition's shapes: drawn with its partition.
        return {n if n in regions else n.rpartition(".")[0] for n in names}

    while pending:
        ready = [n for n, b in pending.items() if not (depends_on(b) & set(pending))]
        if not ready:
            raise ValueError(f"regions {sorted(pending)} refer to one another in a cycle")
        for name in ready:
            bounds = pending.pop(name)
            footprint, generator, placement, motions, update = parts(bounds)
            region_rng = _rng_for(base, name)
            partition = generator is not None and generator.count > 1
            if partition and (placement is not None or motions):
                raise PlacementError(
                    f"region {name!r}: a partition's shapes cover their parent - "
                    "place or move the parent instead"
                )
            try:
                shapes = _draw_and_place(
                    footprint,
                    None if generator is None else _with_drawn(generator, out),
                    None if placement is None else _with_drawn(placement, out),
                    region_rng,
                )
            except (GenerationError, PlacementError) as e:
                raise type(e)(f"region {name!r}: {e}") from e
            if motions:
                # Its motion for this episode: drawn once, run from its start.
                drawn = tuple(_with_drawn(m, out).realized(region_rng) for m in motions)
                shapes = [MovingFootprint(shapes[0], drawn, update)]
            if not partition:
                out[name] = RegionBounds(shapes[0], bounds.altitude)
            else:
                for i, shape in enumerate(shapes):
                    out[f"{name}.{i}"] = RegionBounds(shape, bounds.altitude)
    return out


# --------------------------------------------------------------------------- #
# ConvexPolygon: Valtr                                                        #
# --------------------------------------------------------------------------- #
def valtr_polygon(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random convex polygon of ``n`` vertices, uniform over such polygons
    in the unit square (Valtr 1995), counterclockwise, as ``(n, 2)``."""

    def components(values: np.ndarray) -> np.ndarray:
        values = np.sort(values)
        lo, hi = values[0], values[-1]
        out, last_a, last_b = [], lo, lo
        for v in values[1:-1]:
            if rng.random() < 0.5:
                out.append(v - last_a)
                last_a = v
            else:
                out.append(last_b - v)
                last_b = v
        out.append(hi - last_a)
        out.append(last_b - hi)
        return np.asarray(out)

    xs = components(rng.random(n))
    ys = components(rng.random(n))
    rng.shuffle(ys)
    vectors = np.column_stack([xs, ys])
    vectors = vectors[np.argsort(np.arctan2(vectors[:, 1], vectors[:, 0]))]
    return np.cumsum(vectors, axis=0)


@dataclass(frozen=True)
class ConvexPolygon(ShapeGenerator):
    """A random convex polygon about ``center``: ``vertices`` of them, its
    farthest vertex ``radius_nm`` from its centroid, turned at random.

    Valtr's algorithm - uniform over convex polygons of n vertices."""

    center: LatLon = LatLon(0.0, 0.0)
    radius_nm: Value = 10.0
    vertices: Value = 6

    def __post_init__(self) -> None:
        _check_value("radius_nm", self.radius_nm, 1e-6)
        _check_value("vertices", self.vertices, 3, 512)

    def envelope(self) -> Footprint:
        return DiskFootprint(self.center, _upper(self.radius_nm))

    def _draw(self, rng: np.random.Generator) -> list[Footprint]:
        n = int(round(sample_scalar(self.vertices, rng)))
        radius = sample_scalar(self.radius_nm, rng)
        xy = valtr_polygon(n, rng)
        xy = xy - shapely.centroid(_Polygon(xy)).coords[0]
        turn = rng.uniform(0.0, 2.0 * math.pi)
        c, s = math.cos(turn), math.sin(turn)
        xy = xy @ np.array([[c, s], [-s, c]])
        xy *= radius / np.hypot(xy[:, 0], xy[:, 1]).max()
        return [PolygonFootprint(_points(LocalFrame(self.center), xy))]


# --------------------------------------------------------------------------- #
# Blob: random Fourier series                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Blob(ShapeGenerator):
    """A smooth irregular shape about ``center``: radius ``radius_nm`` varying
    with bearing as a random Fourier series - up to ``± amplitude`` of it.

    ``harmonics`` is how many terms (how many lobes it can have); ``smoothness``
    how fast the higher ones fade (each term's size ~ 1/k^smoothness: higher is
    rounder). Always star-shaped about the center, so never self-intersecting;
    traced at ``points`` vertices."""

    center: LatLon = LatLon(0.0, 0.0)
    radius_nm: Value = 10.0
    amplitude: Value = 0.35
    harmonics: int = 5
    smoothness: float = 1.0
    points: int = 96

    def __post_init__(self) -> None:
        _check_value("radius_nm", self.radius_nm, 1e-6)
        _check_value("amplitude", self.amplitude, 0.0, 0.95)
        if int(self.harmonics) < 1:
            raise ValueError(f"harmonics must be >= 1, got {self.harmonics!r}")
        if int(self.points) < 12:
            raise ValueError(f"points must be >= 12, got {self.points!r}")

    def envelope(self) -> Footprint:
        return DiskFootprint(self.center, _upper(self.radius_nm) * (1.0 + _upper(self.amplitude)))

    def _draw(self, rng: np.random.Generator) -> list[Footprint]:
        radius = sample_scalar(self.radius_nm, rng)
        amplitude = sample_scalar(self.amplitude, rng)
        k = np.arange(1, int(self.harmonics) + 1)
        size = rng.normal(size=k.size) / k ** float(self.smoothness)
        phase = rng.uniform(0.0, 2.0 * math.pi, size=k.size)
        theta = np.linspace(0.0, 2.0 * math.pi, int(self.points), endpoint=False)
        series = (size[:, None] * np.cos(k[:, None] * theta + phase[:, None])).sum(axis=0)
        peak = np.abs(series).max()
        r = radius * (1.0 + amplitude * (series / peak if peak > 0 else series))
        xy = np.column_stack([r * np.sin(theta), r * np.cos(theta)])  # bearings
        return [PolygonFootprint(_points(LocalFrame(self.center), xy))]


# --------------------------------------------------------------------------- #
# VoronoiSectors                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VoronoiSectors(ShapeGenerator):
    """``sectors`` sectors partitioning ``parent``: the Voronoi cells of seed
    points drawn inside it, clipped to it. Seeds keep ``min_spacing_nm``
    apart. Numbered clockwise from north by their seed's bearing from the
    parent's center, so ``.0`` is the northernmost-first. ``min_area_nm2``
    holds for every sector."""

    parent: Bounds | Footprint | None = None
    sectors: int = 4
    min_spacing_nm: float = 0.0

    partition: ClassVar[str | None] = "sectors"

    def __post_init__(self) -> None:
        if self.parent is None:
            raise ValueError("VoronoiSectors needs a parent region to partition")
        if int(self.sectors) < 2:
            raise ValueError(f"sectors must be >= 2, got {self.sectors!r}")
        object.__setattr__(self, "count", int(self.sectors))

    @property
    def _parent(self) -> Footprint:
        return self.parent.footprint if isinstance(self.parent, RegionBounds) else self.parent

    def envelope(self) -> Footprint:
        return self._parent

    def _draw(self, rng: np.random.Generator) -> list[Footprint]:
        parent = self._parent
        center = parent.center_point()
        frame = LocalFrame(center)
        seeds: list[tuple[float, float]] = []
        for _ in range(int(self.sectors) * 200):
            if len(seeds) == int(self.sectors):
                break
            lat, lon = parent.sample_point(rng)
            x, y = frame.to_xy_nm(LatLon(lat, lon))
            if all(math.hypot(x - a, y - b) >= float(self.min_spacing_nm) for a, b in seeds):
                seeds.append((float(x), float(y)))
        if len(seeds) < int(self.sectors):
            raise GenerationError(
                f"VoronoiSectors placed {len(seeds)} of {self.sectors} seeds "
                f"{self.min_spacing_nm:g} nm apart"
            )
        region = _xy(frame, parent.outline())
        cells = shapely.voronoi_polygons(MultiPoint(seeds), extend_to=region)
        bearing = lambda s: math.degrees(math.atan2(s[0], s[1])) % 360.0  # noqa: E731
        out = []
        for seed in sorted(seeds, key=bearing):
            cell = next(c for c in cells.geoms if c.covers(Point(seed)))
            clipped = cell.intersection(region)
            out.append(_footprint(frame, clipped))
        return out


def _footprint(frame: LocalFrame, shape_xy) -> Footprint:
    """A shapely shape in ``frame``'s nm as a footprint: a polygon, or - for
    several parts or holes - the shape itself."""

    def to_lonlat(coords):
        xy = np.asarray(coords, dtype=np.float64)
        p = frame.from_xy_nm(xy[:, 0], xy[:, 1])
        return np.column_stack([p.lon_deg, p.lat_deg])

    shape = shapely.transform(shape_xy, to_lonlat)
    if shape.geom_type == "Polygon" and not shape.interiors:
        return PolygonFootprint([(lat, lon) for lon, lat in list(shape.exterior.coords)[:-1]])
    return ShapelyFootprint(shape)
