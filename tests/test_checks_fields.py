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
class MeansMetres(_Alt):
    """Agrees with itself; says it means metres: its statement catches it."""

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


@pytest.fixture(scope="module")
def report():
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[obs.LatDeg(), States(normalizer=MinMaxNormalizer()), StatesUnderItsOlderName()],
            intruder_obs_fields=[AltAbove(normalizer=MinMaxNormalizer())],
            state_fields=[BulkOff(), MeansMetres()],
            action_fields=[act.HdgDeltaDeg(low=-30, high=30)],
        ),
    )
    yield {r.field: r for r in check_fields(env).results}
    env.close()


def test_fields_that_agree_with_themselves_pass(report):
    for name in ("LatDeg", "States", "StatesUnderItsOlderName", "AltAbove"):
        assert report[name].ok, report[name].findings


def test_a_bulk_path_that_disagrees_is_named(report):
    assert not report["BulkOff"].ok
    assert "bulk vs one at a time" in report["BulkOff"].findings[0]


def test_a_field_that_means_something_else_than_it_states_is_named(report):
    assert not report["MeansMetres"].ok
    assert all("bulk vs expected" in f for f in report["MeansMetres"].findings)


def test_every_list_the_config_holds_is_checked(report):
    # Ownship, intruder and state fields alike.
    assert {"LatDeg", "AltAbove", "BulkOff"} <= set(report)


def test_an_older_name_for_expected_still_states_it():
    assert StatesUnderItsOlderName.states_expected()
    assert not _Alt.states_expected()
