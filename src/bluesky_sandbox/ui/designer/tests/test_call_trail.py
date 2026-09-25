"""The designer's "computed with" trail under each built-in field's source.

A field's class source is often one call into a shared helper, so the catalog
lists the functions its value is computed with. The trail is derived from the
code; these pin that it resolves for every field, reaches the helpers that do
the work, and leaves out the plumbing.
"""

from __future__ import annotations

import pathlib

import pytest

import bluesky_sandbox
from bluesky_sandbox.interface.fields import actions, observations, queryables
from bluesky_sandbox.interface.fields.base import ActionField, ObsField, PairObsField
from bluesky_sandbox.ui.designer import catalog
from bluesky_sandbox.ui.designer.trail import call_trail

_PACKAGE = pathlib.Path(bluesky_sandbox.__file__).parent


def _built_in_fields() -> list[type]:
    return [
        *catalog._concrete_subclasses(observations, (ObsField, PairObsField)),
        *catalog._concrete_subclasses(queryables, (ObsField, PairObsField)),
        *catalog._concrete_subclasses(actions, ActionField),
    ]


@pytest.mark.parametrize("cls", _built_in_fields(), ids=lambda c: c.__name__)
def test_every_entry_points_at_real_source(cls):
    for entry in call_trail(cls):
        assert entry["depth"] >= 0
        if entry["external"]:
            assert entry["location"].startswith("bluesky.")
            assert entry["source"] is None
            continue
        path, line = entry["location"].rsplit(":", 1)
        lines = (_PACKAGE / path).read_text().splitlines()
        first = entry["source"].splitlines()[0].strip()
        # The location's line opens the definition the source starts with
        # (a decorator line above it is part of the source too).
        assert first in lines[int(line) - 1], entry["name"]


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (observations.DistToOwnNm, ["_pairs._pair_qdr_dist", "geo.qdrdist"]),
        (
            observations.ConflictTcpaS,
            ["_ConflictGeomPairField._pairs", "_GeomPairs", "conflict._compute"],
        ),
        (
            observations.IntruderFixArrivalDeltaS,
            ["_pairs._fix_projection", "_route._active_route_waypoint"],
        ),
        (observations.ActiveRouteWaypointEteS, ["_route._route_along_distance_nm"]),
        (observations.CasKts, ["_CasEnvelopeBounds._speed_bounds_ms"]),
        (observations.LaggedPair, ["_lag._lag_ring", "_LagRing"]),
        # Reached through env.query_batch, which only the explicit link follows.
        (queryables.WaypointDistanceNm, ["QueryBatch._waypoint_current"]),
        (queryables.WaypointRouteActive, ["QueryBatch._waypoint_route"]),
        (queryables.QueryRegionInside, ["QueryBatch._region"]),
        (queryables.WaypointSatisfiedTimeTotalS, ["QueryBatch._table_cells"]),
        (actions.ActiveRouteWaypointHdgDeltaDeg, ["_route._active_route_waypoint"]),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else "",
)
def test_the_trail_reaches_the_helpers_that_do_the_work(cls, expected):
    names = [entry["name"] for entry in call_trail(cls)]
    for name in expected:
        assert name in names, f"{cls.__name__}: {name} missing from {names}"


def test_a_helper_is_listed_below_the_one_that_calls_it():
    names = [(e["name"], e["depth"]) for e in call_trail(observations.ConflictTcpaS)]
    order = [name for name, _depth in names]
    depth = dict(names)
    assert order.index("_GeomPairs") < order.index("conflict.conflict_geometry")
    assert depth["conflict.conflict_geometry"] > depth["_GeomPairs"]


@pytest.mark.parametrize("cls", _built_in_fields(), ids=lambda c: c.__name__)
def test_plumbing_is_left_out(cls):
    names = {entry["name"].rsplit(".", 1)[-1] for entry in call_trail(cls)}
    assert not names & {"_indices_array", "_traf_array", "get_many", "get_pairs"}


def test_the_catalog_ships_the_trail():
    fields = {entry["name"]: entry for entry in catalog.obs_fields()}
    trail = fields["DistToOwnNm"]["profile"]["trail"]
    assert trail[0]["name"] == "_pairs._pair_qdr_dist"
    assert "def _pair_qdr_dist" in trail[0]["source"]
