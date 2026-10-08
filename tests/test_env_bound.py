"""Any field that reads its environment - an observation, a state field, an
action - is bound to it and reads what the hooks do: the design's shapes, its
queries, an aircraft's whole context (EnvBound)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import (
    ActionMeta,
    EnvActionField,
    EnvBound,
    EnvObsField,
    ObsField,
    ObsMeta,
    ObsQuantity,
    Unit,
)
from bluesky_sandbox.sim.bounds import (
    BoxFootprint,
    ConstantAltitudeBand,
    DiskFootprint,
    LatLon,
    RegionBounds,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

# Spawned in its west half: some aircraft inside the zone, some not.
_ZONE = RegionBounds(DiskFootprint(LatLon(52.0, 4.0), 25.0), ConstantAltitudeBand(0, 40_000))


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        region = SpawnRegion(
            bounds=RegionBounds(BoxFootprint(51.6, 52.4, 3.5, 5.5)),
            n_aircraft=4,
            params={"alt_ft": (10_000, 30_000), "spd_kts": (250, 250)},
        )
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[region], aircraft_type="B744", conflict_free_spawn=False),
            queryables={},
            max_aircraft=4,
            shapes={"zone": _ZONE},
        )


def _inside(context) -> bool:
    return bool(context.shape("zone").contains(context.position))


@dataclass(frozen=True)
class InZone(EnvObsField):
    meta = ObsMeta("in_zone", Unit.UNITLESS, ObsQuantity.INDICATOR)
    low: float = 0.0
    high: float = 1.0

    def get(self, idx: Any) -> Any:
        return float(_inside(self.agent_context(int(idx))))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# Built from the mixin alone, on a base the env has no name for: bound all the same.
@dataclass(frozen=True)
class ZoneRadius(ObsField, EnvBound):
    meta = ObsMeta("zone_radius_nm", Unit.NM, ObsQuantity.DISTANCE)
    low: float = 0.0
    high: float = 100.0

    def get(self, idx: Any) -> Any:
        return float(self.agent_context(int(idx)).shape("zone").footprint.radius_nm)

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


_APPLIED: list[tuple[str, bool]] = []


@dataclass(frozen=True)
class MarkInZone(EnvActionField):
    """Records, as it is applied, which aircraft it acts for and whether it is in the zone."""

    meta = ActionMeta("mark_in_zone", Unit.UNITLESS)
    low: float = 0.0
    high: float = 1.0

    def set(self, idx: int, value: float) -> None:
        context = self.agent_context(idx)
        _APPLIED.append((context.acid, _inside(context)))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[obs.LatDeg(), InZone()],
            state_fields=[ZoneRadius()],
            action_fields=[MarkInZone()],
        ),
    )
    env.reset(seed=0)
    yield env
    env.close()


def test_every_field_that_reads_its_env_is_bound(env):
    config = env.config
    for field in [*config.obs_fields, *config.state_fields, *config.action_fields]:
        if isinstance(field, EnvBound):
            assert field.bound_env is env


def test_an_observation_reads_the_hooks_context(env):
    field = env.config.obs_fields[1]
    inside = [field.get(i) for i in range(bs.traf.ntraf)]
    assert inside == [float(_inside(env.agent_context(i))) for i in range(bs.traf.ntraf)]
    assert 0.0 < sum(inside) < len(inside)  # some in, some out: it reads each aircraft


def test_a_field_on_the_mixin_alone_is_bound_and_reads_the_design(env):
    radius = env.config.state_fields[0]
    assert radius.get(0) == pytest.approx(25.0)


def test_an_action_reads_the_context_it_is_applied_from(env):
    _APPLIED.clear()
    # Before the step it commands: the state it is applied from.
    before = {acid: _inside(env.agent_context(bs.traf.id.index(acid))) for acid in env.agents}
    env.step({agent: np.zeros(1, dtype=np.float32) for agent in env.agents})
    assert dict(_APPLIED) == before
