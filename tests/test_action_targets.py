"""Speed and altitude actions: one computation per axis, in any unit.

Every action commanding a speed or an altitude is an axis (SI) plus a unit
mixin, absolute or a delta from a nominal, with an optional ``command_floor``
and ``command_ceiling`` on the target it sends. These pin the commands sent to
BlueSky - the part the rest of the suite cannot see, since it only reads the
action spaces.
"""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig


class _Empty:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
        )


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(
        scenario=_Empty(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    )
    yield env
    env.close()


@pytest.fixture
def aircraft(env):
    """A B744 at 12,000 ft and 250 kts CAS; its index."""
    env.reset(seed=0)
    assert bs.traf.cre("ACT001", "B744", 52.0, 4.5, 90, 12_000 * ft, 250 * kts) is True
    return bs.traf.id.index("ACT001")


@pytest.fixture
def sent(monkeypatch):
    """The stack commands a field sends, captured instead of executed."""
    commands: list[str] = []
    monkeypatch.setattr(bs.stack, "stack", lambda *lines, **_kw: commands.extend(lines))
    return commands


def _value(command: str) -> float:
    return float(command.split()[-1])


# --- a unit variant is its twin in another unit ------------------------------
_TWINS = [
    (act.SpdKts, act.SpdMs, kts, 240.0),
    (act.AltFt, act.AltM, ft, 15_000.0),
    (act.SpdDeltaKts, act.SpdDeltaMs, kts, -12.0),
    (act.AltDeltaFt, act.AltDeltaM, ft, 700.0),
    (act.ApAltDeltaFt, act.ApAltDeltaM, ft, -900.0),
]


@pytest.mark.parametrize(
    ("in_unit", "in_si", "si_per_unit", "value"),
    _TWINS,
    ids=[a.__name__ for a, *_ in _TWINS],
)
def test_twins_send_the_same_command(
    aircraft, sent, in_unit, in_si, si_per_unit, value
):
    in_unit().set(aircraft, value)
    in_si().set(aircraft, value * si_per_unit)
    assert len(sent) == 2
    assert sent[0].split()[:2] == sent[1].split()[:2]
    assert _value(sent[0]) == pytest.approx(_value(sent[1]), rel=1e-6)


_DELTA_TWINS = [
    (act.AltDeltaFt, act.AltDeltaM, ft),
    (act.SpdDeltaKts, act.SpdDeltaMs, kts),
    (act.ApAltDeltaFt, act.ApAltDeltaM, ft),
]


@pytest.mark.parametrize(
    ("in_unit", "in_si", "si_per_unit"),
    _DELTA_TWINS,
    ids=[a.__name__ for a, *_ in _DELTA_TWINS],
)
def test_twins_share_their_default_ranges(aircraft, in_unit, in_si, si_per_unit):
    # The metric delta actions used to default to half (altitude) and a tenth
    # (speed) of their twins' ranges; both now resolve from the envelope.
    low, high = in_unit().bounds(aircraft)
    low_si, high_si = in_si().bounds(aircraft)
    assert low_si == pytest.approx(low * si_per_unit, rel=1e-6)
    assert high_si == pytest.approx(high * si_per_unit, rel=1e-6)


# --- delta defaults -------------------------------------------------------------
def test_a_delta_defaults_to_the_reachable_span(aircraft):
    # 12,000 ft is nearer the ground than the ceiling: +/-12,000 ft reaches
    # both without a dead zone; the speed span is the nearer speed limit.
    assert act.AltDeltaFt().bounds(aircraft) == pytest.approx((-12_000.0, 12_000.0))
    cas = float(bs.traf.cas[aircraft])
    perf = bs.traf.perf
    half = min(cas - perf.vmin[aircraft], perf.vmax[aircraft] - cas) / kts
    assert act.SpdDeltaKts().bounds(aircraft) == pytest.approx((-half, half))


def test_a_fixed_delta_range_is_still_honored(aircraft):
    assert act.AltDeltaFt(low=-1000.0, high=1000.0).bounds(aircraft) == (
        -1000.0,
        1000.0,
    )


_ALIASES = [
    (act.ApAltDeltaFt, act.AltDeltaFt, 700.0),
    (act.ApAltDeltaM, act.AltDeltaM, 200.0),
    (act.ApSpdDeltaKts, act.SpdDeltaKts, -12.0),
]


@pytest.mark.parametrize(
    ("alias", "plain", "value"), _ALIASES, ids=[a.__name__ for a, *_ in _ALIASES]
)
def test_an_autopilot_delta_is_its_plain_twin_by_another_name(
    aircraft, sent, alias, plain, value
):
    assert issubclass(alias, plain)
    assert alias.meta.name == f"ap_{plain.meta.name}"
    assert alias().bounds(aircraft) == plain().bounds(aircraft)
    alias().set(aircraft, value)
    plain().set(aircraft, value)
    assert sent[0] == sent[1]


# --- floor and ceiling --------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (act.SpdKts(command_floor=220.0), 180.0, 220.0),
        (act.SpdKts(command_ceiling=260.0), 330.0, 260.0),
        (
            act.SpdMs(command_floor=220.0 * kts),
            100.0,
            220.0,
        ),  # floor in m/s, SPD in kts
        (act.AltFt(command_ceiling=14_000.0), 20_000.0, 14_000.0),
        (act.AltM(command_floor=5_000.0 * ft), 1_000.0, 5_000.0),
        (act.ApAltDeltaFt(command_ceiling=12_500.0), 2_000.0, 12_500.0),
        (act.AltDeltaFt(command_floor=11_800.0), -900.0, 11_800.0),
        (act.SpdDeltaKts(command_ceiling=255.0), 30.0, 255.0),
        (act.ActiveRouteWaypointAltDeltaFt(command_floor=13_000.0), 0.0, 13_000.0),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, act.ActionField) else "",
)
def test_the_floor_and_ceiling_bound_the_command(
    aircraft, sent, field, value, expected
):
    field.set(aircraft, value)
    assert _value(sent[-1]) == pytest.approx(expected, rel=1e-6)


def test_an_aircraft_below_the_floor_is_commanded_to_it(aircraft, sent):
    # Used to raise mid-episode: the floor asked for more climb than the delta
    # range allowed, and the bounds came out low > high.
    field = act.ApAltDeltaFt(command_floor=30_000.0)
    low, high = field.bounds(aircraft)
    assert low <= high
    for value in (low, 0.0, high, -5_000.0, 5_000.0):
        field.set(aircraft, value)
        assert _value(sent[-1]) == pytest.approx(30_000.0)


@pytest.mark.parametrize("kind", ["absolute", "delta"])
def test_a_floor_beyond_the_envelope_still_wins(aircraft, sent, kind):
    # A floor above the fastest the aircraft can fly leaves no target that
    # meets both; the floor is the one honored, and the bounds stay valid.
    floor = float(bs.traf.perf.vmax[aircraft]) / kts + 50.0
    field = (act.SpdKts if kind == "absolute" else act.ApSpdDeltaKts)(
        command_floor=floor
    )
    low, high = field.bounds(aircraft)
    assert low < high
    field.set(aircraft, low)
    assert _value(sent[-1]) == pytest.approx(floor)


def test_a_delta_never_commands_past_the_ceiling(aircraft, sent):
    act.AltDeltaFt(low=-50_000.0, high=50_000.0).set(aircraft, 45_000.0)
    ceiling_ft = float(bs.traf.perf.hmax[aircraft]) / ft
    assert _value(sent[-1]) == pytest.approx(ceiling_ft)


def test_a_floor_keeps_zero_meaning_the_nominal(aircraft):
    # The delta range stays symmetric within what may be commanded.
    low, high = act.ApAltDeltaFt(command_floor=11_500.0).bounds(aircraft)
    assert low == pytest.approx(-500.0, abs=1.0)
    assert high == pytest.approx(-low)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command_floor": -1.0},
        {"command_ceiling": -1.0},
        {"command_floor": 300.0, "command_ceiling": 200.0},
    ],
    ids=["negative floor", "negative ceiling", "floor above ceiling"],
)
def test_a_floor_or_ceiling_that_cannot_hold_is_refused(kwargs):
    with pytest.raises(ValueError, match="command_"):
        act.SpdKts(**kwargs)
