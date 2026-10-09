"""Aircraft placed exactly where a situation says, in a running env.

A situation's numbers are the ones a person works an answer out from on
paper - this track, this ground speed, 10 nm off at 090 - so each aircraft is
created through the env (``env.spawn``: its callsign, control state and
context, as the episode's own aircraft have them) and then set to exactly
that state in BlueSky's traffic arrays, with no wind between them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import KW_ONLY, dataclass, replace
from typing import Any, ClassVar

import bluesky as bs
from bluesky.tools.aero import ft, kts, vtas2cas, vtas2mach
from bluesky.tools.geo import qdrpos

__all__ = ["Aircraft", "Situation", "place", "positions", "without_traffic"]


@dataclass(frozen=True)
class Aircraft:
    """One aircraft of a situation: where it is - ``lat``/``lon``, or
    ``distance_nm`` along ``bearing_deg`` from an aircraft placed before it
    (``relative_to``) - and how it moves. ``gs_kts`` is its ground speed and
    true airspeed alike: a situation has no wind."""

    acid: str
    _: KW_ONLY
    track_deg: float
    gs_kts: float
    alt_ft: float
    actype: str
    lat: float | None = None
    lon: float | None = None
    relative_to: str | None = None
    distance_nm: float | None = None
    bearing_deg: float | None = None
    vs_fpm: float = 0.0

    #: The ways an aircraft is placed, each the fields given together: by
    #: position, or by distance and bearing from an aircraft placed before it.
    #: Exactly one - what the design schema reads too.
    PLACEMENTS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("lat", "lon"),
        ("relative_to", "distance_nm", "bearing_deg"),
    )

    def __post_init__(self) -> None:
        given = [[getattr(self, name) is not None for name in group] for group in self.PLACEMENTS]
        whole = sum(all(g) for g in given)
        partly = any(any(g) and not all(g) for g in given)
        if whole != 1 or partly:
            ways = " - or - ".join(", ".join(group) for group in self.PLACEMENTS)
            raise ValueError(f"aircraft {self.acid!r}: give one of {ways}")


@dataclass(frozen=True)
class Situation:
    """Aircraft placed by hand, in the design's geometry for episode ``seed``.
    The first aircraft is the ownship: a case reads its fields for it unless
    it names another."""

    name: str
    aircraft: tuple[Aircraft, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.aircraft:
            raise ValueError(f"situation {self.name!r} places no aircraft")
        seen: set[str] = set()
        for a in self.aircraft:
            if a.acid in seen:
                raise ValueError(f"situation {self.name!r}: two aircraft are {a.acid!r}")
            if a.relative_to is not None and a.relative_to not in seen:
                raise ValueError(
                    f"situation {self.name!r}: {a.acid!r} is placed relative to "
                    f"{a.relative_to!r}, which is not placed before it"
                )
            seen.add(a.acid)

    @property
    def ownship(self) -> str:
        return self.aircraft[0].acid


def positions(situation: Situation) -> dict[str, tuple[float, float]]:
    """Where each of the situation's aircraft is, ``(lat, lon)`` by callsign:
    its own position, or its distance and bearing from an aircraft placed
    before it (great-circle, BlueSky's ``qdrpos``)."""
    at: dict[str, tuple[float, float]] = {}
    for a in situation.aircraft:
        if a.relative_to is None:
            at[a.acid] = (float(a.lat), float(a.lon))
        else:
            ref_lat, ref_lon = at[a.relative_to]
            lat, lon = qdrpos(ref_lat, ref_lon, a.bearing_deg, a.distance_nm)
            at[a.acid] = (float(lat), float(lon))
    return at


def place(env: Any, situation: Situation) -> dict[str, int]:
    """Reset ``env`` for the situation's episode and create its aircraft, each
    in exactly the state it gives. Their traffic indices, by callsign.

    The env's own traffic is created too, unless it runs
    :func:`without_traffic`."""
    env.reset(seed=situation.seed)
    at = positions(situation)
    indices: dict[str, int] = {}
    for a in situation.aircraft:
        lat, lon = at[a.acid]
        tas, alt = a.gs_kts * kts, a.alt_ft * ft
        track = a.track_deg % 360.0
        acid = env.spawn(
            (lat, lon),
            alt_ft=a.alt_ft,
            spd_kts=float(vtas2cas(tas, alt)) / kts,
            hdg_deg=track,
            actype=a.actype,
            callsign=a.acid,
        )
        if acid is None:
            raise RuntimeError(f"situation {situation.name!r}: {a.acid!r} could not be created")
        idx = bs.traf.id.index(acid)
        traf = bs.traf
        traf.lat[idx], traf.lon[idx] = lat, lon
        traf.trk[idx] = traf.hdg[idx] = track
        traf.gs[idx] = traf.tas[idx] = tas
        traf.gsnorth[idx] = tas * math.cos(math.radians(track))
        traf.gseast[idx] = tas * math.sin(math.radians(track))
        traf.cas[idx] = vtas2cas(tas, alt)
        traf.M[idx] = vtas2mach(tas, alt)
        traf.alt[idx] = alt
        traf.vs[idx] = a.vs_fpm * ft / 60.0
        indices[a.acid] = idx
    return indices


class without_traffic:  # noqa: N801 - read as a phrase: ``with without_traffic(env):``
    """While in it, ``env`` resets to its design's geometry - shapes,
    queryables, routes - with none of its traffic: no spawn region or source
    creates an aircraft, so a situation's aircraft are the only ones. Its
    scenario is put back after; the env is left in the last situation."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self._scenario = env.scenario

    def __enter__(self) -> Any:
        self._env.scenario = _GeometryOnly(self._scenario)
        return self._env

    def __exit__(self, *_exc: Any) -> None:
        self._env.scenario = self._scenario


@dataclass(frozen=True)
class _GeometryOnly:
    """``scenario``'s episodes without their traffic."""

    scenario: Any

    def sample(self, rng: Any) -> Any:
        return _quiet(self.scenario.sample(rng))

    def support(self) -> Any:
        return _quiet(self.scenario.support())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scenario, name)


def _quiet(episode: Any) -> Any:
    with warnings.catch_warnings():
        # Rebuilding the spawn config repeats any warning it gave when the
        # design made it - already shown there.
        warnings.simplefilter("ignore", DeprecationWarning)
        spawn = replace(episode.spawn, regions=[], sources=[])
    return replace(episode, spawn=spawn)
