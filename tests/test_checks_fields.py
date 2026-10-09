"""Every field of a design checked against itself (bluesky_sandbox.checks):
a custom field is held to what a built-in is - its bulk values against one at
a time, against its own ``expected``, normalized as a batch or not."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.checks import check_fields
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import ObsField, ObsMeta, ObsQuantity, PairObsField, Unit
from bluesky_sandbox.interface.wrappers.observations.normalizer import MinMaxNormalizer

from test_env_bound import _Scenario


@dataclass(frozen=True)
class _Alt(ObsField):
    """Altitude in feet, in bulk and one at a time."""

    meta = ObsMeta("alt_check_ft", Unit.FT, ObsQuantity.ALTITUDE)
    low: float = 0.0
    high: float = 40_000.0

    def get(self, idx: Any) -> Any:
        return float(bs.traf.alt[int(idx)]) / 0.3048

    def get_many(self, indices: Any) -> Any:
        return np.asarray(bs.traf.alt, dtype=np.float64)[np.asarray(indices, dtype=np.intp)] / 0.3048

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class States(_Alt):
    meta = ObsMeta("alt_states_ft", Unit.FT, ObsQuantity.ALTITUDE)

    def expected(self, idx: int) -> Any:
        return float(bs.traf.alt[idx]) / 0.3048


@dataclass(frozen=True)
class StatesUnderItsOlderName(_Alt):
    meta = ObsMeta("alt_old_ft", Unit.FT, ObsQuantity.ALTITUDE)

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.alt[idx]) / 0.3048


@dataclass(frozen=True)
class BulkOff(_Alt):
    """Its bulk path is a foot off: what an agent observes is not what a reward reads."""

    meta = ObsMeta("alt_bulk_off_ft", Unit.FT, ObsQuantity.ALTITUDE)

    def get_many(self, indices: Any) -> Any:
        return super().get_many(indices) + 1.0


@dataclass(frozen=True)
class MeansMeters(_Alt):
    """Agrees with itself; says it means meters: its statement catches it."""

    meta = ObsMeta("alt_means_m", Unit.FT, ObsQuantity.ALTITUDE)

    def expected(self, idx: int) -> Any:
        return float(bs.traf.alt[idx])


@dataclass(frozen=True)
class AltAbove(PairObsField):
    """How far the other aircraft is above the ownship, in feet."""

    meta = ObsMeta("alt_above_ft", Unit.FT, ObsQuantity.ALTITUDE, is_pair=True)
    low: float = -40_000.0
    high: float = 40_000.0

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return (float(bs.traf.alt[int(other_idx)]) - float(bs.traf.alt[own_idx])) / 0.3048

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return (float(bs.traf.alt[other_idx]) - float(bs.traf.alt[own_idx])) / 0.3048

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class SometimesNaN(_Alt):
    """Not a number for the aircraft above 9,000 ft: it breaks training."""

    meta = ObsMeta("alt_nan_ft", Unit.FT, ObsQuantity.ALTITUDE)

    def get(self, idx: Any) -> Any:
        value = super().get(idx)
        return float("nan") if value > 9_000 else value

    def get_many(self, indices: Any) -> Any:
        return np.array([self.get(i) for i in indices])


@dataclass(frozen=True)
class Noisy(_Alt):
    """Draws its noise from an unseeded generator: no run repeats it."""

    meta = ObsMeta("alt_noisy_ft", Unit.FT, ObsQuantity.ALTITUDE)

    def get(self, idx: Any) -> Any:
        return self.get_many([idx])[0]

    def get_many(self, indices: Any) -> Any:
        return super().get_many(indices) + np.random.default_rng().random()


@dataclass(frozen=True)
class Cramped(_Alt):
    """Bounds narrower than the traffic it reads."""

    meta = ObsMeta("alt_cramped_ft", Unit.FT, ObsQuantity.ALTITUDE)
    high: float = 5_000.0


@dataclass(frozen=True)
class AltAboveUndefinedForItself(AltAbove):
    """NaN for an aircraft with itself - the one pair no observation holds."""

    meta = ObsMeta("alt_above_self_nan_ft", Unit.FT, ObsQuantity.ALTITUDE, is_pair=True)

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float("nan") if int(other_idx) == int(own_idx) else super().get_pair(own_idx, other_idx)


@pytest.fixture(scope="module")
def report():
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[obs.LatDeg(), States(normalizer=MinMaxNormalizer()), StatesUnderItsOlderName()],
            intruder_obs_fields=[AltAbove(normalizer=MinMaxNormalizer()), AltAboveUndefinedForItself()],
            state_fields=[BulkOff(), MeansMeters(), SometimesNaN(), Noisy(), Cramped()],
            action_fields=[act.HdgDeltaDeg(low=-30, high=30)],
        ),
    )
    yield {r.field: r for r in check_fields(env).results}
    env.close()


def test_fields_that_agree_with_themselves_pass(report):
    for name in ("LatDeg", "States", "StatesUnderItsOlderName", "AltAbove", "AltAboveUndefinedForItself"):
        assert report[name].ok, report[name].findings


def test_a_bulk_path_that_disagrees_is_named(report):
    assert not report["BulkOff"].ok
    assert "bulk vs one at a time" in report["BulkOff"].findings[0]


def test_a_field_that_means_something_else_than_it_states_is_named(report):
    assert not report["MeansMeters"].ok
    assert all("bulk vs expected" in f for f in report["MeansMeters"].findings)


def test_every_list_the_config_holds_is_checked(report):
    # Ownship, intruder and state fields alike.
    assert {"LatDeg", "AltAbove", "BulkOff"} <= set(report)


def test_an_older_name_for_expected_still_states_it():
    assert StatesUnderItsOlderName.states_expected()
    assert not _Alt.states_expected()


def test_a_value_that_is_not_a_number_fails(report):
    assert not report["SometimesNaN"].ok
    assert any("not a number" in f for f in report["SometimesNaN"].findings)


def test_a_field_that_differs_on_the_same_seed_fails(report):
    assert not report["Noisy"].ok
    assert any("not repeatable" in f for f in report["Noisy"].findings)


def test_a_value_outside_its_bounds_is_noted_not_failed(report):
    cramped = report["Cramped"]
    assert cramped.ok
    assert cramped.notes and "outside its bounds" in cramped.notes[0] and "bounds 0 to 5,000" in cramped.notes[0]
    assert not report["LatDeg"].notes


def test_each_field_is_timed_as_the_env_reads_it_and_found_where_it_is():
    config = EnvConfig(dt=5.0, obs_fields=[States(), obs.AltFt()], intruder_obs_fields=[obs.DistToOwnNm()], action_fields=[])
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    try:
        report = check_fields(env, steps=3)
    finally:
        env.close()
    by_name = {r.field: r for r in report.results}
    assert by_name["States"].where == ("obs_fields", 0) and by_name["AltFt"].where == ("obs_fields", 1)
    assert by_name["DistToOwnNm"].where == ("intruder_obs_fields", 0)
    assert all(r.ms is not None and r.ms >= 0 for r in report.results)
    assert by_name["AltFt"].batched is True and by_name["DistToOwnNm"].batched is True
    assert report.step_ms is not None and report.step_ms > 0 and report.aircraft > 0
