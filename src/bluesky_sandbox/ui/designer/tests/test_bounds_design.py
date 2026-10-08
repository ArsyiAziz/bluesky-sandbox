"""A design's named bounds reach code: each episode carries them in its own
frame - turned with their rotation group - and a hook reads their geometry by
name, which the code editor completes."""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import DiskFootprint, LatLon, RegionBounds
from bluesky_sandbox.sim.bounds.geometry import Arc
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer.builder import build_scenario
from bluesky_sandbox.ui.designer.code_intel import code_intel

from .test_designer import _example_design_spec


def _with_regions():
    spec = _example_design_spec()
    spec.regions = {
        "core": {
            "type": "region",
            "footprint": {"type": "box", "lat_min_deg": 51.8, "lat_max_deg": 52.2,
                          "lon_min_deg": 4.4, "lon_max_deg": 5.0},
            "altitude": {"type": "constant", "min_ft": 2000, "max_ft": 9000},
        },
        "hold": {
            "type": "region",
            "footprint": {"type": "disk", "center": {"lat_deg": 52.0, "lon_deg": 4.7},
                          "radius_nm": 5.0},
        },
    }
    spec.airspace = {"ref": "core"}
    return spec


def test_every_episode_carries_the_named_bounds():
    episode = build_scenario(_with_regions()).sample(np.random.default_rng(0))
    assert set(episode.bounds) == {"core", "hold"}
    hold = episode.bounds["hold"]
    assert len(hold.boundary.faces.of(Arc)) == 1 and not hold.boundary.vertices


def test_a_named_bounds_turns_with_its_group_and_stays_its_primitive():
    spec = _with_regions()
    spec.transform = {
        "groups": [
            {"id": "g", "name": "g", "members": ["core", "hold"], "parent": None,
             "pivot": [52.0, 4.7], "angle_deg": 90.0}
        ]
    }
    episode = build_scenario(spec).sample(np.random.default_rng(0))
    # The airspace and the bounds it is drawn from turn together.
    assert np.allclose(episode.bounds["core"].bounding_box, episode.airspace_bounds.bounding_box)
    assert episode.bounds["core"].bounding_box != build_scenario(spec).support().bounds["core"].bounding_box
    hold = episode.bounds["hold"]
    assert isinstance(hold.footprint, DiskFootprint)  # about its own center: still there
    assert (hold.center.lat_deg, hold.center.lon_deg) == pytest.approx((52.0, 4.7))


def test_the_generated_scenario_hands_on_its_regions():
    files = codegen.generate_task(_with_regions(), "Geo")
    scenario = next(t for p, t in files.items() if p.endswith("scenario.py"))
    assert "bounds=REGIONS" in scenario


def test_the_code_editor_completes_shapes_by_name():
    types = code_intel(_with_regions())["types"]
    names = [item["name"] for item in types["design:shapes"]["items"]]
    assert names == ["core", "hold"]
    members = {a["name"] for a in types["bluesky_sandbox.sim.bounds.base.RegionBounds"]["attrs"]}
    assert {"boundary", "center", "frame", "floor_ft", "ceiling_ft", "parts"} <= members


class _OneDisk:
    """An episode with one named bounds, and nothing else."""

    def sample(self, rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
            bounds={"hold": RegionBounds(DiskFootprint(LatLon(52.0, 4.0), 10.0))},
        )


def test_a_hook_reads_a_bounds_geometry_from_its_context():
    env = BlueskyEnv(
        scenario=_OneDisk(),
        config=EnvConfig(dt=5.0, obs_fields=[obs.CasKts()], action_fields=[act.HdgDeg()]),
    )
    try:
        env.reset(seed=0)
        assert bs.traf.cre("GEO1", "B744", 52.0, 4.0, 90, 20_000 * ft, 250 * kts)
        env.set_aircraft_control_state("GEO1", AircraftControlState.CONTROLLED)
        ctx = env.agent_context(bs.traf.id.index("GEO1"))
        hold = ctx.shape("hold")
        assert hold is env.episode_shapes["hold"] is ctx.shapes["hold"]
        # The older names still read the same.
        assert ctx.bounds("hold") is hold is env.episode_bounds["hold"]
        n = hold.boundary.nearest(ctx.position)
        assert n.signed_nm == pytest.approx(10.0, rel=2e-3)  # at the center
        assert ctx.airspace_bounds is None
        with pytest.raises(KeyError, match="not in the design"):
            ctx.shape("nowhere")
    finally:
        env.close()


def test_shapes_complete_by_name_however_they_are_reached():
    from bluesky_sandbox.ui.designer.code_intel import code_intel as intel

    types = intel(_with_regions())["types"]
    ctx = next(v for k, v in types.items() if k.endswith(".AgentStepContext"))
    attrs = {a["name"]: a for a in ctx["attrs"]}
    # ctx.shapes["…"] and ctx.shape("…") offer the design's names (and the
    # older ctx.bounds("…") still does).
    assert attrs["shapes"]["type"] == "design:shapes"
    assert attrs["shape"]["params"][0]["keys"] == "design:shapes"
    assert attrs["bounds"]["params"][0]["keys"] == "design:shapes"
    assert attrs["queryables"]["type"] == "design:queryable"


def test_a_type_too_deep_to_describe_up_front_is_described_when_reached():
    from bluesky_sandbox.ui.designer.code_intel import code_intel as intel
    from bluesky_sandbox.ui.designer.code_intel import describe_type

    types = intel(_with_regions())["types"]
    # Some types are named only, past the up-front depth - described on reaching.
    partial = [name for name, t in types.items() if t.get("partial")]
    assert partial
    assert not describe_type(partial[0])[partial[0]].get("partial")
    projection = "bluesky_sandbox.sim.bounds.geometry.Projection"
    described = describe_type(projection)
    attrs = {a["name"]: a.get("type") for a in described[projection]["attrs"]}
    assert attrs["point"] == "bluesky_sandbox.sim.bounds.coordinates.LatLon"
    assert not described[projection].get("partial")
    assert described[attrs["point"]].get("partial")  # and so on, a level at a time
    # Cached properties and declared attributes are members too.
    nearest = describe_type("bluesky_sandbox.sim.bounds.geometry.Nearest")
    assert {"face", "vertex", "signed_nm"} <= {
        a["name"] for a in nearest["bluesky_sandbox.sim.bounds.geometry.Nearest"]["attrs"]
    }
    boundary = describe_type("bluesky_sandbox.sim.bounds.geometry.Boundary")
    assert {"faces", "vertices", "nearest"} <= {
        a["name"] for a in boundary["bluesky_sandbox.sim.bounds.geometry.Boundary"]["attrs"]
    }
    with pytest.raises(LookupError):
        describe_type("nowhere.Nothing")


def test_an_error_in_scenario_code_is_reported_with_its_line_not_a_crash():
    from starlette.testclient import TestClient

    from bluesky_sandbox.ui.designer.api import create_app

    client = TestClient(create_app())
    spec = _with_regions()
    spec.scenario_setup = "import math\nimport nowhere_at_all\n"
    spec.scenario_hooks = {"episode_geometry": "return geometry"}
    r = client.post("/api/spec/preview", json={"spec": spec.to_dict()})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("error in scenario setup, line 2: ModuleNotFoundError")
    spec.scenario_setup = ""
    spec.scenario_hooks = {"episode_geometry": "geometry = dict(geometry)\nreturn context"}
    r = client.post("/api/spec/preview", json={"spec": spec.to_dict()})
    assert r.status_code == 422
    assert "scenario hook episode_geometry, line 2: NameError" in r.json()["detail"]
    assert client.get("/api/python/type", params={"key": "nowhere.Nothing"}).status_code == 404
