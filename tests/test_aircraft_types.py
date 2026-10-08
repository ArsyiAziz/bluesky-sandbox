"""Aircraft types: each spawn region names its own; the types there are, and
what each is, are read from the performance model BlueSky flies - rotorcraft
too, a helicopter and drones, which fly with no Mach limit and from a hover."""

from __future__ import annotations

import warnings

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers.observations.normalizer import SymmetricNormalizer
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.performance import available_types, type_info
from bluesky_sandbox.sim.performance.envelope import _speed_band_kt
from bluesky_sandbox.sim.performance.speeds import above_crossover, cas_ceiling_ms, mach_limit
from bluesky_sandbox.sim.sampling.distributions import Categorical
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

ROTORCRAFT = {"ec35", "m600", "amzn", "mnet", "phan4", "m100", "m200", "mavic", "horsefly"}


# ---- the types there are, and what each is -------------------------------- #


def test_the_types_are_bluesky_models_rotorcraft_included():
    types = available_types("openap")
    assert ROTORCRAFT <= types and "b744" in types


@pytest.mark.parametrize(
    ("actype", "tags"),
    [
        ("B744", ["Fixed-wing", "Jet", "Wake Heavy", "397 t"]),
        ("A388", ["Fixed-wing", "Jet", "Wake Super", "560 t"]),
        ("EC35", ["Helicopter", "Turboshaft", "Wake Light", "2.8 t"]),
        ("M600", ["Drone", "6 rotors", "15.1 kg"]),
    ],
)
def test_each_type_says_what_it_is(actype, tags):
    info = type_info(actype, "openap")
    assert info["tags"] == tags and info["type"] == actype


def test_a_type_flown_on_anothers_data_says_so():
    info = type_info("AT72", "openap")
    assert info["name"] is None and info["tags"][-1].startswith("Flown as Embraer")


def test_the_designer_offers_every_type_with_its_tags():
    from bluesky_sandbox.ui.designer.catalog import aircraft_by_model

    offered = {t["type"]: t for t in aircraft_by_model()["openap"]}
    assert offered["M600"]["tags"][0] == "Drone" and offered["EC35"]["tags"][0] == "Helicopter"
    assert offered["B744"]["name"] == "Boeing 747-400"


# ---- each spawn region names its own ------------------------------------- #


class _Scenario:
    def __init__(self, regions, **spawn) -> None:
        self.regions, self.spawn = regions, spawn

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=list(self.regions), **self.spawn),
            queryables={},
            max_aircraft=sum(r.max_n() for r in self.regions),
        )


def _region(types, alt=(2_000.0, 6_000.0), spd=(20.0, 60.0), n=3, name="area"):
    return SpawnRegion(
        RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.6)),
        n_aircraft=n,
        params={"alt_ft": alt, "spd_kts": spd},
        aircraft_type=types,
        name=name,
    )


def _env(regions, config=None, **spawn):
    config = config or EnvConfig(dt=5.0, obs_fields=[obs.CasKts()], action_fields=[act.HdgDeltaDeg()])
    return BlueskyEnv(scenario=_Scenario(regions, **spawn), config=config)


def test_each_region_spawns_its_own_types():
    env = _env([_region("B744", spd=(250.0, 280.0), name="heavy"), _region(Categorical({"M600": 1.0, "EC35": 1.0}), alt=(400.0, 1_500.0), name="low")])
    try:
        env.reset(seed=1)
        by_region = {}
        for r in env.spawn_log:
            by_region.setdefault(r.region_index, set()).add(r.actype)
        assert by_region[0] == {"B744"} and by_region[1] <= {"M600", "EC35"}
    finally:
        env.close()


def test_a_region_naming_no_type_is_refused():
    env = _env([_region(None)])
    try:
        with pytest.raises(ValueError, match="spawn region 'area' has no aircraft_type"):
            env.reset(seed=0)
    finally:
        env.close()


def test_a_region_above_a_types_ceiling_is_refused():
    env = _env([_region("M600", alt=(20_000.0, 30_000.0))])
    try:
        with pytest.raises(ValueError, match="M600's ceiling is 8,202 ft"):
            env.reset(seed=0)
    finally:
        env.close()


def test_a_type_the_model_does_not_carry_is_refused():
    env = _env([_region("ZZZZ")])
    try:
        with pytest.raises(ValueError, match="does not carry"):
            env.reset(seed=0)
    finally:
        env.close()


def test_the_global_types_still_fill_in_but_are_deprecated():
    with pytest.warns(DeprecationWarning, match="allowed_aircraft is deprecated"):
        config = EnvConfig(dt=5.0, obs_fields=[obs.CasKts()], action_fields=[act.HdgDeltaDeg()], allowed_aircraft=["A320"])
    env = _env([_region(None, spd=(220.0, 260.0))], config=config)
    try:
        env.reset(seed=0)
        assert {r.actype for r in env.spawn_log} == {"A320"}
    finally:
        env.close()


# ---- a design's earlier global types, onto its regions --------------------- #


def _design(env_extra=None, spawn_extra=None, region_type=None):
    region = {"type": "spawn_region", "shape": {"type": "region", "footprint": {"type": "box", "lat_min_deg": 51.9, "lat_max_deg": 52.1, "lon_min_deg": 4.4, "lon_max_deg": 4.6}},
              "n_aircraft": 2, "params": {"alt_ft": {"type": "range", "low": 2000, "high": 6000}, "spd_kts": {"type": "range", "low": 200, "high": 260}}}
    if region_type is not None:
        region["aircraft_type"] = region_type
    return {
        "version": 1,
        "env": {"obs_fields": [], "action_fields": [], **(env_extra or {})},
        "spawn": {"type": "spawn_config", "regions": [region], **(spawn_extra or {})},
    }


@pytest.mark.parametrize(
    ("env_extra", "spawn_extra", "region_type", "expected"),
    [
        ({"allowed_aircraft": ["A320", "B738"]}, None, None, {"type": "categorical", "weights": {"A320": 1.0, "B738": 1.0}}),
        ({"allowed_aircraft": ["A320"]}, {"aircraft_type": "B744"}, None, "B744"),
        ({"allowed_aircraft": ["A320"]}, None, "EC35", "EC35"),  # its own kept
        (None, {"aircraft_type": None}, None, None),  # a null: nothing to carry
    ],
    ids=["allowed", "the spawn's", "its own", "a null"],
)
def test_an_earlier_designs_types_land_on_its_regions(env_extra, spawn_extra, region_type, expected):
    from bluesky_sandbox.ui.designer import spec as S

    spec = S.DesignSpec.from_dict(_design(env_extra, spawn_extra, region_type))
    assert spec.spawn["regions"][0].get("aircraft_type") == expected
    assert "aircraft_type" not in spec.spawn and "allowed_aircraft" not in spec.to_dict()["env"]


# ---- rotorcraft, flown ----------------------------------------------------- #


@pytest.fixture(scope="module")
def flying():
    env = _env([_region("B744", spd=(250.0, 280.0))])
    env.reset(seed=0)
    yield env
    env.close()


def _cre(env, acid, actype, alt_ft, cas_kts):
    assert bs.traf.cre(acid, actype, 52.0, 4.5, 90.0, alt_ft * ft, cas_kts * kts)
    env.set_aircraft_control_state(acid, AircraftControlState.CONTROLLED)
    return bs.traf.id.index(acid)


@pytest.mark.parametrize(("actype", "alt_ft"), [("M600", 400.0), ("EC35", 2_000.0)])
def test_a_rotorcraft_has_no_mach_limit_and_starts_from_a_hover(flying, actype, alt_ft):
    idx = _cre(flying, f"R{actype}", actype, alt_ft, 10.0)
    assert np.isinf(mach_limit(idx)[0])
    assert cas_ceiling_ms(idx) == pytest.approx(float(bs.traf.perf.vmax[idx]))  # not 0
    assert not above_crossover(idx)[0]
    lo, hi = _speed_band_kt(idx, alt_ft)
    assert lo == 0.0 and hi == pytest.approx(float(bs.traf.perf.vmax[idx]) / kts)


def test_a_crossover_speed_action_flies_a_rotorcraft_in_knots(flying):
    # Above a jet schedule's crossover altitude (FL293)? No: the EC135 has no Mach limit.
    idx = _cre(flying, "HELI1", "EC35", 15_000.0, 80.0)
    action = act.ApSpdDeltaCrossover(
        normalizer=SymmetricNormalizer(), above_crossover=act.MachRegime(act.Crossover(cas_kts=300, mach=0.78))
    )
    assert action.acting(idx) is action
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        low, high = action.bounds(idx)
    assert 0.0 <= 80.0 + low and 80.0 + high <= float(bs.traf.perf.vmax[idx]) / kts + 1e-6
