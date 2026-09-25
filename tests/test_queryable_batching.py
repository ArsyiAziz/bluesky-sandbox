"""Queryable observation fields read a batch; it must match the result objects.

Observations read every aircraft's query result at once, as arrays
(:class:`~bluesky_sandbox.core.services.QueryBatch`). Tasks and rewards read
the same results one aircraft at a time, as objects, through
``agent_context(idx).query(name)``. Each field's batched value must be exactly
the attribute its ``queryable_spec.path`` names on that object - including
where the object raises, for temporal state a queryable does not track.

Covers tracked and untracked waypoints and regions, route legs the monitor
recorded at spawn and ones it only finds by searching the route, and the
active-waypoint choice across several queryables.
"""

from __future__ import annotations

import inspect

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import queryables as Q
from bluesky_sandbox.sim.bounds import BoxFootprint, ConstantAltitudeBand, RegionBounds
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_QUERYABLES = {
    "merge": Waypoint(
        lat=52.0,
        lon=4.8,
        alt_ft=9000,
        speed_kts=250,
        reach_radius_nm=8.0,
        alt_tolerance_ft=800,
        speed_tolerance_kts=15,
        track_temporal_state=True,
    ),
    "exit": Waypoint(lat=52.3, lon=5.3, alt_ft=12000, reach_radius_nm=None),
    "gate": Waypoint(
        lat=52.1,
        lon=5.0,
        speed_kts=240,
        speed_tolerance_kts=10,
        speed_tolerance_mach=0.02,
        track_temporal_state=True,
    ),
    # A tolerance with no altitude to hold it to: satisfied on position alone.
    "wide": Waypoint(lat=52.0, lon=4.6, reach_radius_nm=40.0, alt_tolerance_ft=500),
    "sector": QueryRegion(
        bounds=RegionBounds(
            BoxFootprint(51.85, 52.15, 4.4, 5.0), ConstantAltitudeBand(7000, 11000)
        ),
        track_temporal_state=True,
    ),
    "zone": QueryRegion(bounds=RegionBounds(BoxFootprint(51.9, 52.2, 4.6, 5.4))),
}
_TYPES = ["B744", "A320"]


class _Scenario:
    def __init__(self) -> None:
        box = RegionBounds(BoxFootprint(51.8, 52.2, 4.3, 4.7))
        params = {"alt_ft": (7000, 12000), "spd_kts": (220, 290)}
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=box, n_aircraft=8, params=params, route=["merge", "exit"]
                ),
                SpawnRegion(bounds=box, n_aircraft=6, params=params),
            ],
            aircraft_type=_TYPES,
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=self.spawn,
            queryables=_QUERYABLES,
            max_aircraft=14,
        )


def _field_classes() -> list[type]:
    abstract = (Q.QueryableObsField, Q.WaypointResultObsField, Q.ActiveWaypointObsField)
    return [
        cls
        for name in sorted(dir(Q))
        if not name.startswith("_")
        and inspect.isclass(cls := getattr(Q, name))
        and issubclass(cls, Q.QueryableObsField)
        and cls not in abstract
    ]


def _instances(cls: type) -> list:
    if issubclass(cls, Q.ActiveWaypointObsField):
        selections = [("merge", "exit"), ("exit", "gate", "merge")]
        if cls is not Q.ActiveWaypointOneHot:
            selections.append(())  # every configured waypoint
        return [cls(query_names=names) for names in selections]
    if issubclass(cls, Q.WaypointResultObsField):
        return [cls(query_name=name) for name in ("merge", "exit", "gate", "wide")]
    return [cls(query_name=name) for name in ("sector", "zone")]


def _one(field, idx: int):
    """The field's value for one aircraft, read off the result object."""
    if isinstance(field, Q.ActiveWaypointObsField):
        active = field.active_waypoint(idx)
        if isinstance(field, Q.ActiveWaypointAvailable):
            return float(active is not None)
        if isinstance(field, Q.ActiveWaypointOneHot):
            values = np.zeros(field.output_size(), dtype=np.float32)
            if active is not None:
                values[field.query_names.index(active[0])] = 1.0
            return values
        if isinstance(field, Q.ActiveWaypointRouteIndex):
            index = None if active is None else active[1].route.index
            return -1.0 if index is None else float(index)
        path = field.queryable_spec.path
        value = active[1] if active is not None else None
        for part in path.split("."):
            value = None if value is None else getattr(value, part)
        return 0.0 if value is None else float(value)
    value = field.query_result(idx)
    path = (
        field.queryable_spec.path
        if not isinstance(field, Q._WaypointRouteFlag)
        else (f"route.{field.flag_name}")
    )
    for part in path.split("."):
        value = getattr(value, part)
    if value is None:  # an unrouted waypoint's route index
        return -1.0
    return float(value)


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(
            dt=12.0, allowed_aircraft=_TYPES, obs_fields=[], action_fields=[]
        ),
    )
    env.reset(seed=0)
    # Routes the monitor never recorded: found only by searching the route.
    unrouted = [a for i, a in enumerate(bs.traf.id) if bs.traf.ap.route[i].nwp == 0]
    for acid in unrouted[:4]:
        bs.stack.stack(f"ADDWPT {acid} 52.1 5.0 9000 240")
        bs.stack.stack(f"ADDWPT {acid} 52.0 4.8")
    bs.stack.process()
    for _ in range(8):  # into the sector, past some legs
        env.step({})
    yield env
    env.close()


@pytest.mark.parametrize("cls", _field_classes(), ids=lambda c: c.__name__)
def test_the_batch_matches_each_aircrafts_result(env, cls):
    every = list(range(bs.traf.ntraf))
    for field in _instances(cls):
        field = field.bind_env(env)
        expected, raised = [], None
        for idx in every:
            try:
                expected.append(_one(field, idx))
            except Exception as exc:  # noqa: BLE001 - compared by type below
                raised = type(exc)
                break
        if raised is not None:
            with pytest.raises(raised):
                field.get_many(every)
            continue
        batched = np.asarray(field.get_many(every), dtype=np.float64)
        np.testing.assert_array_equal(
            batched, np.asarray(expected, dtype=np.float64), err_msg=repr(field)
        )
        for idx in every[::3]:
            np.testing.assert_array_equal(
                np.asarray(field.get(idx), dtype=np.float64),
                np.asarray(expected[idx], dtype=np.float64),
            )


def test_the_scenario_reaches_every_branch(env):
    """Guard against a vacuous pass: routed, searched and unrouted aircraft, and
    active legs on more than one waypoint."""
    active = Q.ActiveWaypointOneHot(query_names=("merge", "exit", "gate")).bind_env(env)
    chosen = np.asarray(active.get_many(list(range(bs.traf.ntraf))))
    assert (chosen.sum(axis=1) == 0).any(), "no aircraft without an active waypoint"
    assert (chosen.sum(axis=0) > 0).sum() >= 2, (
        "active legs on fewer than two waypoints"
    )
    satisfied = Q.WaypointSatisfied(query_name="wide").bind_env(env)
    assert satisfied.get_many(list(range(bs.traf.ntraf))).any(), (
        "nothing satisfies 'wide'"
    )
    inside = Q.QueryRegionInside(query_name="sector").bind_env(env)
    values = inside.get_many(list(range(bs.traf.ntraf)))
    assert 0 < values.sum() < len(values), "the sector holds all or none of the traffic"
