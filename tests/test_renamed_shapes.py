"""A design's named shapes were once its "bounds", and an element's shape its
``bounds``: the older names still construct and read the same objects."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import QueryRegion
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

BOX = RegionBounds(BoxFootprint(51.8, 52.2, 4.4, 5.0))
OTHER = RegionBounds(BoxFootprint(51.0, 51.2, 4.0, 4.2))


def test_an_elements_shape_by_either_name():
    q = QueryRegion(bounds=BOX)
    assert q.shape is BOX and q.bounds is BOX
    region = SpawnRegion(bounds=BOX, n_aircraft=1, params={"spd_kts": 250, "alt_ft": 10_000})
    assert region.shape is BOX and region.bounds is BOX
    assert replace(region, bounds=OTHER).shape is OTHER
    assert replace(region, shape=OTHER).bounds is OTHER


def test_the_designs_shapes_by_either_name():
    spawn = SpawnConfig(regions=[], aircraft_type="B744")
    spec = EpisodeSpec(airspace_bounds=None, spawn=spawn, queryables={}, max_aircraft=1, bounds={"core": BOX})
    assert spec.shapes == {"core": BOX} and spec.bounds is spec.shapes


def test_the_older_name_cannot_be_set_on_a_frozen_spec():
    spec = EpisodeSpec(airspace_bounds=None, spawn=None, queryables={}, max_aircraft=1)
    with pytest.raises(AttributeError):
        spec.bounds = {}
