"""Trail-store and renderer-cache behaviour.

Both renderer caches are incremental, which makes them the kind of thing that
looks right on the frame you inspect and goes subtly wrong three minutes into
an episode.  These tests pin the two properties that matter: the geometry
drawn is identical to a from-scratch pass at EVERY step, and the per-step cost
is bounded rather than proportional to trail length.

Imports live inside the tests, as in the rest of this suite - the repo root
only reaches ``sys.path`` when pytest runs a test, not while it collects one.
"""

from __future__ import annotations

import math

import pytest

# Arbitrary but fixed plan-view projection parameters.
CX, CY, CLAT, CLON, LAT_PP, LON_PP = 500.0, 400.0, 56.0, 2.0, 0.001, 0.002


def _trail():
    from bluesky_sandbox.ui.drivers.common.trails import Trail

    return Trail


def _pixel_cache():
    from bluesky_sandbox.ui.drivers.pygame.views.horizontal import _TrailPixelCache

    return _TrailPixelCache


def _geometry():
    from bluesky_sandbox.ui.drivers.panda3d.views.world import (
        _TRAIL_CHUNK,
        _TrailGeometry,
    )

    return _TrailGeometry, _TRAIL_CHUNK


def _base(lat, lon):
    return (CX + (lon - CLON) / LON_PP, CY - (lat - CLAT) / LAT_PP)


def _reference_drawcalls(points):
    """The pre-cache run-batching algorithm, applied to the same points.

    Kept verbatim rather than factored out of the renderer: it is the oracle,
    so it has to be able to disagree with the implementation.
    """
    cached = [_base(lat, lon) for lat, lon, _alt, _key in points]
    n = len(points)
    out = []
    run_start = 0
    while run_start < n - 1:
        run_key = points[run_start][3]
        run_end = run_start
        while run_end + 1 < n - 1 and points[run_end + 1][3] == run_key:
            run_end += 1
        out.append((run_key, cached[run_start : run_end + 2]))
        run_start = run_end + 1
    return out


def _cache_drawcalls(cache, trail):
    """What ``HorizontalView._draw_trails`` would hand to pygame."""
    cache.sync(trail, CX, CY, CLAT, CLON, LAT_PP, LON_PP)
    total = cache.total
    points = [tuple(p) for p in cache.base[:total].tolist()]
    runs = cache.runs
    out = []
    for index, (offset, key) in enumerate(runs):
        stop = runs[index + 1][0] + 1 if index + 1 < len(runs) else total
        if stop - offset < 2:
            continue
        out.append((key, points[offset:stop]))
    return out


def _leg(step, phase):
    """Straight legs, a turning climb, and colour changes at the seams."""
    if phase == 0:
        return (56.0, 2.0 + step * 0.004, 20000.0, "normal")
    if phase == 1:
        angle = math.radians(step * 6.0)
        return (
            56.0 + 0.02 * (1 - math.cos(angle)),
            2.4 + 0.02 * math.sin(angle),
            20000.0 + step * 300.0,
            "normal",
        )
    if phase == 2:
        return (56.1 + step * 0.004, 2.42, 24000.0, "conflict")
    return (56.3 + step * 0.004, 2.42, 24000.0, "normal")


def _spiral(step):
    """A continuously curving track - nothing here is decimatable."""
    angle = math.radians(step * 3.0)
    return (
        56.0 + 0.3 * math.sin(angle),
        2.0 + 0.5 * math.cos(angle),
        20000.0 + 2000.0 * math.sin(angle / 3),
        "normal",
    )


def _segment_distance_nm(point, start, end):
    """Distance from ``point`` to the ``start``-``end`` segment, in nm."""
    cos_lat = math.cos(math.radians(start[0]))
    px = (point[1] - start[1]) * cos_lat * 60.0
    py = (point[0] - start[0]) * 60.0
    bx = (end[1] - start[1]) * cos_lat * 60.0
    by = (end[0] - start[0]) * 60.0
    length_sq = bx * bx + by * by
    if length_sq < 1e-15:
        return math.hypot(px, py)
    t = min(1.0, max(0.0, (px * bx + py * by) / length_sq))
    return math.hypot(px - t * bx, py - t * by)


class _FakeNode:
    def removeNode(self):
        pass


class _Builder:
    """Stands in for ``WorldView._build_trail_node``, recording vertex counts."""

    def __init__(self):
        self.calls = []

    def __call__(self, points):
        self.calls.append(len(points))
        return _FakeNode()


def _drawn_indices(geometry, trail, chunk):
    """Absolute point indices covered by the sealed chunks plus the tail."""
    covered = []
    for index in range(len(geometry._sealed)):
        lo = geometry._start + index * chunk
        covered.extend(range(lo, lo + chunk + 1))
    covered.extend(range(geometry._sealed_end, trail.end))
    ordered, seen = [], set()
    for i in covered:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


def test_decimation_collapses_a_straight_leg_to_its_endpoints():
    trail = _trail()()
    for step in range(20):
        trail.append((56.0, 2.0 + step * 0.01, 20000.0, "normal"))
    assert len(trail.points) == 2
    # The trail still ENDS at the aircraft: decimation replaces the
    # provisional point rather than skipping it, so the drawn trail cannot lag
    # behind the chevron.
    assert trail.points[-1][1] == pytest.approx(2.0 + 19 * 0.01)


def test_decimation_never_crosses_a_colour_key_change():
    trail = _trail()()
    for step in range(6):
        trail.append((56.0 + step * 0.01, 2.0, 20000.0, "normal"))
    for step in range(6, 12):
        trail.append((56.0 + step * 0.01, 2.0, 20000.0, "conflict"))
    # Perfectly collinear throughout, so geometry alone would leave 2 points;
    # the state transition has to survive as its own vertex pair.
    assert [point[3] for point in trail.points] == [
        "normal",
        "normal",
        "conflict",
        "conflict",
    ]


def test_decimation_keeps_turns_and_level_changes():
    Trail = _trail()
    straight = Trail()
    for step in range(40):
        straight.append((56.0, 2.0 + step * 0.004, 20000.0, "normal"))
    curved = Trail()
    for step in range(40):
        curved.append(_spiral(step))
    assert len(straight.points) == 2
    assert len(curved.points) > 10


def test_decimation_caps_how_long_one_chord_may_run():
    """The run cap bounds per-append cost, at the price of a spare vertex."""
    from bluesky_sandbox.ui.drivers.common.trails import _DECIMATE_MAX_RUN

    trail = _trail()()
    steps = 5 * _DECIMATE_MAX_RUN
    for step in range(steps):
        trail.append((56.0, 2.0 + step * 0.004, 20000.0, "normal"))
    # Dead straight, so geometry alone would leave 2 points.
    assert 2 < len(trail.points) <= steps // _DECIMATE_MAX_RUN + 2
    assert len(trail.pending) <= _DECIMATE_MAX_RUN


@pytest.mark.parametrize(
    "turn_deg_per_step", [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 6.0]
)
def test_decimation_never_strays_further_than_the_corridor(turn_deg_per_step):
    """The retained polyline must stay within tolerance of the FLOWN path.

    This is the property the point-count assertions above do not capture, and
    its absence hid a real bug: testing only the most recent dropped point let
    a gently curving 400-point arc collapse to 4 points and a 2.3 nm error,
    because the deepest deviation is in the MIDDLE of a dropped run.
    """
    from bluesky_sandbox.ui.drivers.common.trails import _DECIMATE_H_NM

    trail = _trail()()
    flown = []
    lat, lon, heading = 56.0, 2.0, 90.0
    for _ in range(400):
        heading += turn_deg_per_step
        step_nm = 450.0 * (5.0 / 3600.0)          # 450 kt over a 5 s step
        lat += step_nm * math.cos(math.radians(heading)) / 60.0
        lon += step_nm * math.sin(math.radians(heading)) / (
            60.0 * math.cos(math.radians(lat))
        )
        point = (lat, lon, 20000.0, "normal")
        flown.append(point)
        trail.append(point)

    kept = trail.points
    assert len(kept) >= 2
    worst = max(
        min(
            _segment_distance_nm(point, kept[i], kept[i + 1])
            for i in range(len(kept) - 1)
        )
        for point in flown
    )
    assert worst <= _DECIMATE_H_NM * 1.01, (
        f"{turn_deg_per_step} deg/step: retained polyline strays {worst:.3f} nm "
        f"from the flown path ({len(flown)} points -> {len(kept)})"
    )


def test_point_cap_evicts_the_front_and_tracks_absolute_indices():
    trail = _trail()()
    for step in range(600):
        trail.append(_spiral(step), max_points=40)
    assert trail.start > 0
    assert trail.start == trail.dropped
    assert trail.end - trail.start == len(trail.points)
    assert trail.at(trail.end - 1) == trail.points[-1]


# --------------------------------------------------------------------------- #
# pygame: incremental base projection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("max_points", [None, 40])
def test_pygame_cache_matches_a_from_scratch_pass_at_every_step(max_points):
    trail, cache = _trail()(), _pixel_cache()(0)
    for step in range(600):
        trail.append(_leg(step % 25, (step // 25) % 4), max_points)
        if trail.end - trail.start < 2:
            continue
        assert _cache_drawcalls(cache, trail) == _reference_drawcalls(trail.points)


def test_pygame_cache_projects_only_the_new_tail():
    trail, cache = _trail()(), _pixel_cache()(0)
    projected = []
    for step in range(720):
        trail.append(_spiral(step))
        if trail.end - trail.start < 2:
            continue
        before = cache.committed
        cache.sync(trail, CX, CY, CLAT, CLON, LAT_PP, LON_PP)
        projected.append(cache.committed - before)
    # At most one newly committed point per step; the provisional tail point
    # is the only other projection, and it is a single row.
    assert max(projected) <= 1
    assert cache.total > 100, "the trail did grow, so this was a real test"


def test_pygame_cache_is_a_no_op_when_nothing_changed():
    trail, cache = _trail()(), _pixel_cache()(0)
    for step in range(50):
        trail.append(_spiral(step))
    cache.sync(trail, CX, CY, CLAT, CLON, LAT_PP, LON_PP)
    settled = (cache.committed, cache.total, list(cache.runs))
    # Camera-only frames: pan/zoom is applied downstream of the cache, so a
    # second sync must not touch it.
    cache.sync(trail, CX, CY, CLAT, CLON, LAT_PP, LON_PP)
    assert (cache.committed, cache.total, list(cache.runs)) == settled


def test_pygame_cache_rebuilds_for_a_fresh_trail_under_the_same_callsign():
    """Trails toggled off and back on reuse callsigns, not Trail objects."""
    Trail = _trail()
    stale, cache = Trail(), _pixel_cache()(0)
    for step in range(400):
        stale.append(_spiral(step))
    _cache_drawcalls(cache, stale)

    fresh = Trail()
    for step in range(5):
        fresh.append(_leg(step, 0))
    assert _cache_drawcalls(cache, fresh) == _reference_drawcalls(fresh.points)


def test_pygame_screen_transform_is_cached_per_camera_pose():
    """Idle frames must not re-run the affine map over every point."""
    trail, cache = _trail()(), _pixel_cache()(0)
    for step in range(300):
        trail.append(_spiral(step))
    cache.sync(trail, CX, CY, CLAT, CLON, LAT_PP, LON_PP)

    first = cache.screen(1.0, 0.0, 0.0)
    assert cache.screen(1.0, 0.0, 0.0) is first, "recomputed an unchanged frame"

    moved = cache.screen(2.0, 30.0, -10.0)
    assert moved is not first
    assert len(moved[0]) == len(first[0])
    # The viewport is affine over base pixel space, so the transform is
    # exactly ``base * zoom + offset`` - checked against the first row.
    assert moved[0][0][0] == pytest.approx(first[0][0][0] * 2.0 + 30.0)
    assert moved[0][0][1] == pytest.approx(first[0][0][1] * 2.0 - 10.0)

    # A new point has to invalidate it even at an unchanged camera pose.
    trail.append(_spiral(300))
    cache.sync(trail, CX, CY, CLAT, CLON, LAT_PP, LON_PP)
    assert cache.screen(1.0, 0.0, 0.0) is not first


# --------------------------------------------------------------------------- #
# panda3d: chunked geometry
# --------------------------------------------------------------------------- #


def test_panda3d_chunks_tile_the_trail_without_gaps_or_overlap():
    Geometry, chunk = _geometry()
    trail, geometry, builder = _trail()(), Geometry(None), _Builder()
    for step in range(720):
        trail.append(_spiral(step))
        if trail.end - trail.start < 2:
            continue
        geometry.sync(trail, builder)
    assert len(geometry._sealed) >= 2, "no chunk was ever sealed"
    assert _drawn_indices(geometry, trail, chunk) == list(
        range(trail.start, trail.end)
    )


def test_panda3d_rebuild_cost_is_bounded_not_proportional_to_length():
    Geometry, chunk = _geometry()
    trail, geometry, builder = _trail()(), Geometry(None), _Builder()
    per_step = []
    for step in range(720):
        trail.append(_spiral(step))
        if trail.end - trail.start < 2:
            continue
        before = len(builder.calls)
        geometry.sync(trail, builder)
        per_step.append(sum(builder.calls[before:]))
    length = trail.end - trail.start
    assert length > 3 * chunk, "trail too short to tell the two apart"
    # A sealing step rebuilds one chunk plus the tail; every other step
    # rebuilds the tail alone.  Neither depends on how long the trail is.
    assert max(per_step) <= 2 * chunk + 2
    # The old cache rebuilt the whole trail on every step.
    assert sum(per_step) < sum(range(2, length + 1))


def test_panda3d_camera_only_frame_rebuilds_nothing():
    Geometry, _chunk = _geometry()
    trail, geometry, builder = _trail()(), Geometry(None), _Builder()
    for step in range(100):
        trail.append(_spiral(step))
    geometry.sync(trail, builder)
    settled = len(builder.calls)
    geometry.sync(trail, builder)
    geometry.sync(trail, builder)
    assert len(builder.calls) == settled


def test_panda3d_rebuilds_for_a_fresh_trail_under_the_same_callsign():
    Geometry, chunk = _geometry()
    Trail = _trail()
    stale, geometry, builder = Trail(), Geometry(None), _Builder()
    for step in range(400):
        stale.append(_spiral(step))
    geometry.sync(stale, builder)

    fresh = Trail()
    for step in range(5):
        fresh.append(_leg(step, 0))
    geometry.sync(fresh, builder)
    assert geometry._sealed == []
    assert _drawn_indices(geometry, fresh, chunk) == list(
        range(fresh.start, fresh.end)
    )


def test_panda3d_survives_front_eviction_by_the_point_cap():
    Geometry, chunk = _geometry()
    trail, geometry, builder = _trail()(), Geometry(None), _Builder()
    for step in range(600):
        trail.append(_spiral(step), max_points=80)
        if trail.end - trail.start < 2:
            continue
        geometry.sync(trail, builder)
    assert trail.start > 0
    assert _drawn_indices(geometry, trail, chunk) == list(
        range(trail.start, trail.end)
    )
