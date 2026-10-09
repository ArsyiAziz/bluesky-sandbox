"""Probing one field at one moment (bluesky_sandbox.checks.probe): its stages
actual and expected, the calls and traffic it read, overrides of either, its
lag ring, an action's command and the cost of each way it computes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.checks.probe import LagRecorder, Override, probe
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.layout import zero_action
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import ObsField, ObsMeta, ObsQuantity, Unit
from bluesky_sandbox.interface.wrappers.observations.normalizer import MinMaxNormalizer, StepNormalizer

from test_env_bound import _Scenario


@dataclass(frozen=True)
class Reach:
    nm: float
    climb_ft: float


def reach_of(idx: int) -> Reach:
    """What an aircraft can reach: a value the field reads, not BlueSky's."""
    return Reach(nm=float(bs.traf.gs[idx]) / 10.0, climb_ft=float(bs.traf.alt[idx]) / 0.3048)


@dataclass(frozen=True)
class ReachNm(ObsField):
    meta = ObsMeta("reach_nm", Unit.NM, ObsQuantity.DISTANCE)
    low: float = 0.0
    high: float = 100.0

    def get(self, idx: Any) -> Any:
        return reach_of(int(idx)).nm

    def get_many(self, indices: Any) -> Any:
        return np.asarray(bs.traf.gs, dtype=np.float64)[np.asarray(indices, dtype=np.intp)] / 10.0

    def expected(self, idx: int) -> Any:
        return float(bs.traf.gs[idx]) / 10.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class SaysFeetMeansMeters(ReachNm):
    meta = ObsMeta("says_feet", Unit.FT, ObsQuantity.ALTITUDE)

    def get(self, idx: Any) -> Any:
        return float(bs.traf.alt[int(idx)])

    def get_many(self, indices: Any) -> Any:
        return np.asarray(bs.traf.alt, dtype=np.float64)[np.asarray(indices, dtype=np.intp)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.alt[idx]) / 0.3048


_ALT = obs.AltFt(normalizer=MinMaxNormalizer())


@pytest.fixture(scope="module")
def flown():
    config = EnvConfig(
        dt=1.0,
        obs_fields=[*_ALT.stacked(depth=4), ReachNm(), SaysFeetMeansMeters()],
        intruder_obs_fields=[obs.DistToOwnNm()],
        action_fields=[
            act.AltDeltaFt(grid=act.Grid(1000.0)),
            act.HdgDeltaDeg(grid=act.Grid(10.0, on="target"), normalizer=StepNormalizer(steps_each_way=3)),
        ],
    )
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    env.reset(seed=1)
    history = LagRecorder(env)
    history.record()
    for _ in range(2):
        env.step({a: zero_action(env.action_space(a)) for a in env.agents})
        history.record()
    yield env, history
    env.close()


def _field(env, name):
    return next(f for f in env.config.obs_fields if f.meta.name == name)


def _stage(result, name):
    return next(s for s in result.stages if s.name == name)


def _nodes(nodes):
    for node in nodes:
        yield node
        yield from _nodes(node.children)


def test_an_observation_reads_raw_then_normalized_each_actual_and_expected(flown):
    env, _ = flown
    own = bs.traf.id[0]
    result = probe(env, _field(env, "alt_ft"), own)
    raw, normalized = result.stages
    assert raw.name == "raw" and raw.agrees is True
    assert raw.actual == pytest.approx(float(bs.traf.alt[0]) / 0.3048)
    assert normalized.name == "normalized" and normalized.agrees is True
    assert result.kind == "observation" and result.aircraft == own


def test_a_field_whose_statement_disagrees_says_so(flown):
    env, _ = flown
    raw = _stage(probe(env, _field(env, "says_feet"), bs.traf.id[0]), "raw")
    assert raw.agrees is False and raw.expected == pytest.approx(raw.actual / 0.3048)


def test_the_trace_shows_the_calls_made_and_the_traffic_read_under_them(flown):
    env, _ = flown
    result = probe(env, _field(env, "reach_nm"), bs.traf.id[0])
    (top,) = result.trace
    assert top.label == f"ReachNm.get({bs.traf.id[0]})"
    (call,) = [n for n in _nodes(result.trace) if n.label.startswith("reach_of(")]
    assert call.key == f"call:{__name__}:reach_of"
    assert [leaf for leaf, _value in call.leaves] == ["nm", "climb_ft"]
    read = {n.label for n in call.children}
    assert {f"bs.traf.gs[{bs.traf.id[0]}]", f"bs.traf.alt[{bs.traf.id[0]}]"} <= read


def test_a_traffic_override_is_seen_by_all_that_reads_it_then_put_back(flown):
    env, _ = flown
    own = bs.traf.id[0]
    before = float(bs.traf.alt[0])
    result = probe(env, _field(env, "alt_ft"), own, overrides=[Override("traf:alt", 3048.0)])
    assert _stage(result, "raw").actual == pytest.approx(10_000.0)
    assert any(n.overridden and n.label == f"bs.traf.alt[{own}]" for n in _nodes(result.trace))
    assert float(bs.traf.alt[0]) == before


def test_a_call_override_replaces_one_number_in_what_it_returns(flown):
    env, _ = flown
    target = f"call:{__name__}:reach_of"
    result = probe(env, _field(env, "reach_nm"), bs.traf.id[0], overrides=[Override(target, 42.0, leaf="nm")])
    raw = _stage(result, "raw")
    # One at a time reads the override; the batched path never calls it.
    assert result.trace[0].value == "42"
    assert raw.agrees is None and result.notes
    assert any(n.overridden for n in _nodes(result.trace) if n.key == target)
    assert reach_of(0).nm != 42.0  # put back after


def test_a_lag_ring_shows_its_frames_and_how_each_lag_read_them(flown):
    env, history = flown
    own = bs.traf.id[0]
    result = probe(env, _field(env, "alt_ft_lag2"), own, history=history)
    lag = result.lag
    assert lag["fields"] == ["alt_ft", "alt_ft_lag1", "alt_ft_lag2", "alt_ft_lag3"]
    assert lag["depth"] == 4
    # Three pushes so far - reset and two steps: lag 3 reads the oldest.
    assert [f["back"] for f in lag["frames"]] == [2, 1, 0]
    assert lag["frames"][0]["read_by"] == [2, 3] and lag["frames"][0]["placeholder"] == [3]
    assert [f["sim_time_s"] for f in lag["frames"]] == [0.0, 1.0, 2.0]
    assert lag["first_s"] == lag["start_s"] == 0.0
    first, *_, last = lag["progression"]
    assert [c["placeholder"] for c in first["cells"]] == [False, True, True, True]
    assert [c["placeholder"] for c in last["cells"]] == [False, False, False, True]
    assert all(c["agrees"] for row in lag["progression"] for c in row["cells"])
    assert _stage(result, "raw").agrees is True


def test_a_pair_field_reads_about_another(flown):
    env, _ = flown
    own, other = bs.traf.id[0], bs.traf.id[1]
    result = probe(env, env.config.intruder_obs_fields[0], own, other=other)
    assert result.kind == "pair" and _stage(result, "raw").agrees is True
    assert set(result.cost["ms"]) == {"pair matrix", "per ownship", "per pair"}
    with pytest.raises(ValueError, match="another aircraft"):
        probe(env, env.config.intruder_obs_fields[0], own)


def test_an_action_goes_policy_to_value_to_grid_to_its_command(flown):
    env, _ = flown
    own = bs.traf.id[0]
    result = probe(env, env.config.action_fields[0], own, give=1400.0)
    stages = {s.name: s for s in result.stages}
    assert list(stages) == ["policy", "value", "on grid", "target", "BS command"]
    assert stages["on grid"].actual == 1000.0
    assert stages["target"].agrees is True
    assert result.command == [stages["BS command"].actual] and result.command[0].startswith(f"ALT {own} ")
    with pytest.raises(ValueError, match="give it a value"):
        probe(env, env.config.action_fields[0], own)


def test_cost_times_each_way_a_field_computes(flown):
    env, _ = flown
    cost = probe(env, _field(env, "reach_nm"), bs.traf.id[0]).cost
    assert set(cost["ms"]) == {"batched", "one at a time"} and cost["batched_path"] is True
    assert cost["aircraft"] == bs.traf.ntraf and cost["max_aircraft"] == env.episode_max_aircraft


def test_an_aircraft_not_in_the_air_is_refused(flown):
    env, _ = flown
    with pytest.raises(ValueError, match="not in the air"):
        probe(env, _field(env, "alt_ft"), "NOBODY")


def test_an_observation_carries_its_bounds_and_its_normalizer_across_them(flown):
    env, _ = flown
    result = probe(env, _field(env, "alt_ft"), bs.traf.id[0])
    low, high = result.bounds
    assert result.unit == "ft" and high > low
    raws = [raw for raw, _ in result.curve]
    normalized = [n for _, n in result.curve]
    assert raws[0] < low and raws[-1] > high  # a margin either side
    assert normalized == sorted(normalized)


def test_a_stepped_action_lists_each_choice_and_the_target_it_commands(flown):
    env, _ = flown
    own = bs.traf.id[0]
    result = probe(env, env.config.action_fields[1], own, give=4)
    assert [c["steps"] for c in result.choices] == [-3, -2, -1, 0, 1, 2, 3]
    targets = [c["target"] for c in result.choices]
    assert all(round(t) % 10 == 0 for t in targets)  # on the target grid
    assert _stage(result, "steps").actual == 1
    (command,) = result.command
    assert command.startswith(f"HDG {own} ") and float(command.split()[-1]) == pytest.approx(targets[4] % 360)
