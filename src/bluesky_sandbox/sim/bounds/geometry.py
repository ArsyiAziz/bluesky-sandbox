"""A region's boundary as primitives: its faces and vertices, and where a point
projects onto them.

Every boundary is faces of two kinds - a :class:`Segment` (straight) and an
:class:`Arc` (part of a circle, or all of one) - meeting at :class:`Vertex`
points. A circle is one arc and no vertices; a sector two segments and an arc,
meeting at three vertices. Each primitive has one operation, :meth:`project`;
a :class:`Primitives` collection finds the nearest of any selection::

    b = ctx.shape("sector_a")
    n = b.boundary.nearest(ctx.position)
    n.face.distance_nm, n.vertex, n.signed_nm
    b.boundary.faces.of(Arc).nearest(ctx.position)
    (b.boundary.vertices + other.boundary.vertices).nearest(ctx.position)

Positions are absolute :class:`LatLon` throughout; :class:`Frame` is the view
relative to a region's center. Any point may hold arrays - one entry per
aircraft - and every number in what comes back is then an array too.

The outline is shapely's, in the region's flat local frame - the frame the
shapes test containment in - so every combination (holes, parts, shared edges)
is handled; a face is labeled an arc where it runs along one of the region's
circles, and projected onto it exactly. The distance and bearing reported from
a point to its projection are great-circle (BlueSky's ``qdrdist``), as the
waypoint fields report theirs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from bluesky.tools.geo import qdrdist
from shapely.geometry.polygon import orient

from .coordinates import LatLon, LocalFrame

__all__ = [
    "Arc",
    "Boundary",
    "Frame",
    "Nearest",
    "Primitive",
    "Primitives",
    "Projection",
    "Segment",
    "Vertex",
]

#: How close a point is to a circle to be on it: relative to the radius (an
#: outline vertex shapely puts on a chord sits within its sagitta), and at least
#: this many nm.
_ON_CIRCLE_REL = 1e-4
_ON_CIRCLE_NM = 1e-6
#: The widest step between two points on one arc of the outline, deg.
_ARC_STEP_DEG = 2.5
#: Two straight edges this close in direction are one face, rad.
_COLLINEAR_RAD = 1e-6


def _arrays(p: LatLon) -> tuple[np.ndarray, np.ndarray, bool]:
    """``p``'s lat and lon as arrays, and whether it was a single point."""
    lat = np.asarray(p.lat_deg, dtype=np.float64)
    lon = np.asarray(p.lon_deg, dtype=np.float64)
    return np.atleast_1d(lat), np.atleast_1d(lon), lat.ndim == 0 and lon.ndim == 0


def _out(value: np.ndarray, scalar: bool) -> Any:
    """``value`` as a float (or object) when the point was a single one."""
    if not scalar:
        return value
    item = value[0]
    return item.item() if isinstance(item, np.generic) else item


def _objects(values: list[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


@dataclass(frozen=True, eq=False)
class Projection:
    """Where a point projects onto a primitive - or onto the nearest of several.

    ``point`` is the closest point on it; ``distance_nm`` and ``bearing_deg``
    run from the projected point to it, great-circle, degrees true.
    ``along_nm`` is how far along the face from its ``start`` (0 on a vertex);
    ``cross_nm`` the signed offset from the face's line or circle, + on the
    region's inside. ``primitive`` is the primitive it landed on, ``at_vertex``
    the vertex ``point`` sits on, or ``None``.
    """

    point: LatLon
    distance_nm: float | np.ndarray
    bearing_deg: float | np.ndarray
    along_nm: float | np.ndarray
    cross_nm: float | np.ndarray
    primitive: Primitive
    at_vertex: Vertex | None


class Primitive:
    """A vertex or a face of a boundary."""

    def project(self, p: LatLon) -> Projection:
        """The closest point on this primitive to ``p``."""
        lat, lon, scalar = _arrays(p)
        foot_lat, foot_lon, along, cross, at_vertex = self._foot(lat, lon)
        bearing, distance = qdrdist(lat, lon, foot_lat, foot_lon)
        bearing = np.mod(np.asarray(bearing, dtype=np.float64), 360.0)
        distance = np.asarray(distance, dtype=np.float64)
        return Projection(
            point=LatLon(_out(foot_lat, scalar), _out(foot_lon, scalar)),
            distance_nm=_out(distance, scalar),
            bearing_deg=_out(bearing, scalar),
            along_nm=_out(along, scalar),
            cross_nm=_out(cross, scalar),
            primitive=self,
            at_vertex=_out(at_vertex, scalar),
        )

    def _foot(self, lat: np.ndarray, lon: np.ndarray):
        raise NotImplementedError


@dataclass(frozen=True)
class Vertex(Primitive):
    """A point where two faces meet."""

    point: LatLon

    def _foot(self, lat, lon):
        n = lat.shape[0]
        return (
            np.full(n, self.point.lat_deg),
            np.full(n, self.point.lon_deg),
            np.zeros(n),
            np.zeros(n),
            _objects([self] * n),
        )


@dataclass(frozen=True)
class Segment(Primitive):
    """A straight face from ``start`` to ``end``; the region is on its left."""

    start: LatLon
    end: LatLon
    _frame: LocalFrame = field(repr=False, compare=False)

    @cached_property
    def _ends(self) -> tuple[float, float, float, float]:
        ax, ay = self._frame.to_xy_nm(self.start)
        bx, by = self._frame.to_xy_nm(self.end)
        return ax, ay, bx, by

    @property
    def length_nm(self) -> float:
        ax, ay, bx, by = self._ends
        return math.hypot(bx - ax, by - ay)

    @property
    def bearing_deg(self) -> float:
        """Its direction, ``start`` to ``end``, degrees true."""
        ax, ay, bx, by = self._ends
        return math.degrees(math.atan2(bx - ax, by - ay)) % 360.0

    def _foot(self, lat, lon):
        ax, ay, bx, by = self._ends
        px, py = self._frame.to_xy_nm(LatLon(lat, lon))
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (length * length), 0.0, 1.0)
        foot = self._frame.from_xy_nm(ax + t * dx, ay + t * dy)
        cross = (dx * (py - ay) - dy * (px - ax)) / length
        at_vertex = _objects(
            [
                Vertex(self.start) if ti <= 0.0 else Vertex(self.end) if ti >= 1.0 else None
                for ti in t
            ]
        )
        return foot.lat_deg, foot.lon_deg, t * length, cross, at_vertex


@dataclass(frozen=True)
class Arc(Primitive):
    """Part of a circle: bearings ``from_deg`` clockwise through ``span_deg``
    about ``center`` - all of it when ``span_deg`` is 360. ``inside`` is
    whether the region is inside the circle (a disk) or outside it (a hole)."""

    center: LatLon
    radius_nm: float
    from_deg: float
    span_deg: float
    inside: bool = True

    @property
    def to_deg(self) -> float:
        return (self.from_deg + self.span_deg) % 360.0

    @property
    def full(self) -> bool:
        return self.span_deg >= 360.0 - 1e-9

    @cached_property
    def _frame(self) -> LocalFrame:
        return LocalFrame(self.center)

    @property
    def start(self) -> LatLon:
        return self._frame.offset(self.from_deg, self.radius_nm)

    @property
    def end(self) -> LatLon:
        return self._frame.offset(self.to_deg, self.radius_nm)

    @property
    def length_nm(self) -> float:
        return math.radians(self.span_deg) * self.radius_nm

    def _foot(self, lat, lon):
        r = self.radius_nm
        px, py = self._frame.to_xy_nm(LatLon(lat, lon))
        rho = np.hypot(px, py)
        theta = np.mod(np.degrees(np.arctan2(px, py)), 360.0)  # 0 at the center
        rel = np.mod(theta - self.from_deg, 360.0)
        on = np.full(rho.shape, True) if self.full else rel <= self.span_deg + 1e-9
        # Off the arc: the nearer of its ends.
        sx, sy = self._frame.to_xy_nm(self.start)
        ex, ey = self._frame.to_xy_nm(self.end)
        to_start = np.hypot(px - sx, py - sy)
        to_end = np.hypot(px - ex, py - ey)
        nearer_start = to_start <= to_end
        brg = np.radians(theta)
        fx = np.where(on, r * np.sin(brg), np.where(nearer_start, sx, ex))
        fy = np.where(on, r * np.cos(brg), np.where(nearer_start, sy, ey))
        foot = self._frame.from_xy_nm(fx, fy)
        along = np.where(on, np.radians(rel) * r, np.where(nearer_start, 0.0, self.length_nm))
        cross = (r - rho) if self.inside else (rho - r)
        at_vertex = _objects(
            [
                None
                if (o or self.full)
                else Vertex(self.start)
                if s
                else Vertex(self.end)
                for o, s in zip(on, nearer_start)
            ]
        )
        return foot.lat_deg, foot.lon_deg, along, cross, at_vertex


class Primitives(list):
    """Vertices and faces: a list, with :meth:`nearest`, :meth:`of` and ``+``."""

    def of(self, kind: type) -> Primitives:
        """Only the primitives of ``kind`` (``Arc``, ``Segment``, ``Vertex``)."""
        return Primitives(p for p in self if isinstance(p, kind))

    def __add__(self, other: Iterable[Primitive]) -> Primitives:
        return Primitives([*self, *other])

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return Primitives(item) if isinstance(index, slice) else item

    def nearest(self, p: LatLon) -> Projection | None:
        """The projection of ``p`` onto the nearest of these - the first on a
        tie - or ``None`` when there are none."""
        if not self:
            return None
        projections = [primitive.project(p) for primitive in self]
        _, _, scalar = _arrays(p)
        if scalar:
            return min(projections, key=lambda q: q.distance_nm)
        distance = np.stack([q.distance_nm for q in projections])
        best = np.argmin(distance, axis=0)
        cols = np.arange(best.shape[0])

        def pick(name: str) -> np.ndarray:
            return np.stack([np.asarray(getattr(q, name)) for q in projections])[best, cols]

        lat = np.stack([np.asarray(q.point.lat_deg) for q in projections])[best, cols]
        lon = np.stack([np.asarray(q.point.lon_deg) for q in projections])[best, cols]
        vertex = np.stack([q.at_vertex for q in projections])[best, cols]
        return Projection(
            point=LatLon(lat, lon),
            distance_nm=pick("distance_nm"),
            bearing_deg=pick("bearing_deg"),
            along_nm=pick("along_nm"),
            cross_nm=pick("cross_nm"),
            primitive=_objects([projections[i].primitive for i in best]),
            at_vertex=vertex,
        )


class Nearest:
    """What on a boundary is nearest a point: its ``face``, its ``vertex``,
    and its ``signed_nm``. Each is worked out when read."""

    def __init__(self, boundary: Boundary, p: LatLon) -> None:
        self._boundary = boundary
        self._p = p

    @cached_property
    def face(self) -> Projection:
        """The projection onto the nearest face: the nearest point on the
        boundary - a point's own vertex, on a boundary with no face."""
        return self._boundary.faces.nearest(self._p) or self._boundary.vertices.nearest(self._p)

    @cached_property
    def vertex(self) -> Projection | None:
        """The projection onto the nearest vertex - not the vertex the nearest
        point is at (that is ``face.at_vertex``) - or ``None`` with none."""
        return self._boundary.vertices.nearest(self._p)

    @cached_property
    def signed_nm(self) -> float | np.ndarray:
        """The distance to the boundary, + inside the region, - outside."""
        inside = self._boundary._contains(self._p)
        distance = self.face.distance_nm
        return np.where(inside, distance, -distance) if np.ndim(inside) else (
            distance if inside else -distance
        )


class Boundary:
    """A region's boundary: its ``faces`` and ``vertices``, absolute."""

    #: Every face: a Segment or an Arc, each traced with the region on its left.
    faces: Primitives
    #: Every vertex: where two faces meet (a circle has none).
    vertices: Primitives

    def __init__(self, faces: Primitives, vertices: Primitives, contains) -> None:
        self.faces = faces
        self.vertices = vertices
        self._contains = contains

    def nearest(self, p: LatLon) -> Nearest:
        """The nearest face and vertex to ``p``, and its signed distance."""
        return Nearest(self, p)

    def __repr__(self) -> str:
        kinds = ", ".join(
            f"{sum(isinstance(f, k) for f in self.faces)} {k.__name__.lower()}"
            for k in (Segment, Arc)
        )
        return f"Boundary({kinds}; {len(self.vertices)} vertices)"


class Frame:
    """Coordinates relative to a region's ``center``, in nm: east and north, or
    - oriented - along its axis and across it (+ to the right of it)."""

    def __init__(self, center: LatLon, axis_deg: float = 0.0, oriented: bool = False) -> None:
        self.center = center
        self.axis_deg = float(axis_deg)
        self.oriented = oriented
        self._local = LocalFrame(center)

    def __call__(self, oriented: bool = False) -> Frame:
        """The same frame, north-up or turned with the region's axis."""
        return Frame(self.center, self.axis_deg, oriented)

    def xy_nm(self, p: LatLon) -> tuple[float | np.ndarray, float | np.ndarray]:
        """``p`` relative to the center: ``(east, north)``, or oriented
        ``(along, across)``."""
        east, north = self._local.to_xy_nm(p)
        if not self.oriented:
            return east, north
        a = math.radians(self.axis_deg)
        return (
            east * math.sin(a) + north * math.cos(a),
            east * math.cos(a) - north * math.sin(a),
        )

    def latlon(self, x_nm: float | np.ndarray, y_nm: float | np.ndarray) -> LatLon:
        """The absolute point at ``(x_nm, y_nm)`` in this frame."""
        if self.oriented:
            a = math.radians(self.axis_deg)
            x_nm, y_nm = (
                x_nm * math.sin(a) + y_nm * math.cos(a),
                x_nm * math.cos(a) - y_nm * math.sin(a),
            )
        return self._local.from_xy_nm(x_nm, y_nm)


# --------------------------------------------------------------------------- #
# Building a boundary from an outline                                          #
# --------------------------------------------------------------------------- #
def boundary_of(outline, circles: list[tuple[LatLon, float]], frame: LocalFrame, contains) -> Boundary:
    """The faces and vertices of ``outline`` (shapely, lon/lat degrees): its
    rings, each traced with the region on its left, runs along one of
    ``circles`` as arcs and the rest as segments."""
    if outline.geom_type == "Point":
        # A point: one vertex, no face.
        return Boundary(Primitives([]), Primitives([Vertex(LatLon(outline.y, outline.x))]), contains)
    polygons = list(getattr(outline, "geoms", [outline]))
    faces: list[Primitive] = []
    vertices: list[Primitive] = []
    for polygon in polygons:
        polygon = orient(polygon, sign=1.0)
        for ring in (polygon.exterior, *polygon.interiors):
            ring_faces, ring_vertices = _trace(
                [LatLon(lat, lon) for lon, lat in list(ring.coords)[:-1]], circles, frame
            )
            faces.extend(ring_faces)
            vertices.extend(ring_vertices)
    return Boundary(Primitives(faces), Primitives(vertices), contains)


def _on_circle(points: list[LatLon], circles) -> list[set[int]]:
    """For each point, the circles it lies on."""
    out: list[set[int]] = [set() for _ in points]
    for k, (center, radius) in enumerate(circles):
        local = LocalFrame(center)
        tol = max(_ON_CIRCLE_NM, _ON_CIRCLE_REL * radius)
        for i, p in enumerate(points):
            x, y = local.to_xy_nm(p)
            if abs(math.hypot(x, y) - radius) <= tol:
                out[i].add(k)
    return out


def _bearing(center: LatLon, p: LatLon) -> float:
    x, y = LocalFrame(center).to_xy_nm(p)
    return math.degrees(math.atan2(x, y)) % 360.0


def _trace(points: list[LatLon], circles, frame: LocalFrame):
    """One ring's faces and vertices."""
    n = len(points)
    on = _on_circle(points, circles)
    # Each edge i (points[i] -> points[i+1]): the circle it runs along, if any.
    labels: list[int | None] = []
    for i in range(n):
        j = (i + 1) % n
        common = on[i] & on[j]
        label = None
        for k in sorted(common):
            center, _ = circles[k]
            step = abs((_bearing(center, points[j]) - _bearing(center, points[i]) + 180.0) % 360.0 - 180.0)
            if step <= _ARC_STEP_DEG:
                label = k
                break
        labels.append(label)

    xy = [frame.to_xy_nm(p) for p in points]

    def direction(i: int) -> float:
        j = (i + 1) % n
        return math.atan2(xy[j][1] - xy[i][1], xy[j][0] - xy[i][0])

    def same_face(i: int, j: int) -> bool:
        """Whether edges i and j (j following i) are one face."""
        if labels[i] != labels[j]:
            return False
        if labels[i] is not None:
            return True
        turn = abs((direction(j) - direction(i) + math.pi) % (2 * math.pi) - math.pi)
        return turn <= _COLLINEAR_RAD

    breaks = [i for i in range(n) if not same_face((i - 1) % n, i)]
    if not breaks:
        # One face all the way round: a whole circle (or a degenerate ring).
        if labels[0] is not None:
            center, radius = circles[labels[0]]
            return [Arc(center, radius, 0.0, 360.0, _arc_inside(center, points))], []
        breaks = [0]

    faces: list[Primitive] = []
    vertices: list[Primitive] = []
    for b, start in enumerate(breaks):
        stop = breaks[(b + 1) % len(breaks)]  # the run is edges start .. stop-1
        first, last = points[start], points[stop % n]
        vertices.append(Vertex(first))
        label = labels[start]
        if label is None:
            faces.append(Segment(first, last, frame))
            continue
        center, radius = circles[label]
        run = [points[(start + m) % n] for m in range(((stop - start) % n or n) + 1)]
        clockwise = _turns_clockwise(center, run)
        b_first, b_last = _bearing(center, first), _bearing(center, last)
        if clockwise:
            from_deg, span = b_first, (b_last - b_first) % 360.0
        else:
            from_deg, span = b_last, (b_first - b_last) % 360.0
        # Travel counterclockwise about the center has the region (on the left)
        # toward the center: inside the circle.
        faces.append(Arc(center, radius, from_deg, span, inside=not clockwise))
    return faces, vertices


def _turns_clockwise(center: LatLon, run: list[LatLon]) -> bool:
    """Whether ``run`` goes clockwise (bearing increasing) about ``center``."""
    local = LocalFrame(center)
    total = 0.0
    for a, b in zip(run, run[1:]):
        ax, ay = local.to_xy_nm(a)
        bx, by = local.to_xy_nm(b)
        total += ax * by - ay * bx
    return total < 0.0


def _arc_inside(center: LatLon, ring: list[LatLon]) -> bool:
    return not _turns_clockwise(center, [*ring, ring[0]])
