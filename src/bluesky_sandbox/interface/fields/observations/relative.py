"""Another aircraft relative to the ownship: its position and motion, the
closest point of approach, and shared route fixes.

Pair fields, so ``intruder_obs_fields`` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm
from bluesky.tools.geo import qdrdist

from bluesky_sandbox.sim.geometry.conflict import cd_hpz_m as _cd_hpz_m
from bluesky_sandbox.sim.geometry.conflict import cd_lookahead_s as _cd_lookahead_s
from bluesky_sandbox.sim.geometry.conflict import cd_rpz_m as _cd_rpz_m

from .._common import _M_TO_FT, _MS_TO_FTMIN, _MS_TO_KTS, _traf_array
from .._pairs import (
    _BroadcastPairs,
    _cd_pair_values,
    _fix_projection,
    _pair_horizontal_tcpa_s,
    _pair_qdr_dist,
    _pair_relative_motion,
    _per_own,
    _track_frame,
    _track_frame_at_cpa,
)
from .._reference import (
    one_confpair_value,
    one_pair_fix_projection,
    one_pair_motion,
    one_pair_track_frame,
)
from ..base import ObsMeta, ObsQuantity, PairObsField, Unit


@dataclass(frozen=True)
class DistToOwnNm(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder distance in nautical miles."""

    meta = ObsMeta("dist_to_own_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "distance nautical miles"] = 0.0
    high: Annotated[float, "distance nautical miles"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _qdr, dist = _pair_qdr_dist(own, other)
        return dist

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        own = (float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx]))
        other = (float(bs.traf.lat[other_idx]), float(bs.traf.lon[other_idx]))
        return float(qdrdist(*own, *other)[1])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TcpaS(_BroadcastPairs, PairObsField):
    """BlueSky ASAS time to closest point of approach, in seconds.

    ASAS only stores this value for detected conflict pairs. Non-conflict
    intruder rows return the field's high bound - the CD lookahead horizon, the
    largest time-to-CPA a detected conflict could carry.

    Bounds are dynamic: unless explicit ``low``/``high`` are given, they follow
    BlueSky's CD lookahead (:func:`_cd_lookahead_s`, read from
    ``asas_dtlookahead``) as ``(-lookahead, +lookahead)``. Detected conflicts
    fall in that range (``tcpa`` goes slightly negative just past CPA).
    """

    meta = ObsMeta(
        "tcpa_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    low: Annotated[
        float | None, "time seconds lower bound; None = -CD lookahead at runtime"
    ] = None
    high: Annotated[
        float | None, "time seconds upper bound; None = +CD lookahead at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``tcpa`` (one entry per ``confpairs``
        # row) each sim step - read it directly. Non-conflict intruders take the
        # high bound (the CD lookahead horizon), so the sentinel tracks config.
        fill = _per_own(own, lambda o: self.bounds(o)[1])
        return _cd_pair_values("tcpa", own, other, fill)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        value = one_confpair_value("tcpa", own_idx, other_idx)
        return self.bounds(own_idx)[1] if value is None else value

    def bounds(self, own_idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            look = _cd_lookahead_s()
            return -look, look

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class TlosS(_BroadcastPairs, PairObsField):
    """BlueSky ASAS predicted time to loss of separation (PZ entry), in seconds.

    Reads the conflict detector's cached ``tLOS`` (one entry per ``confpairs``
    row) each sim step - the time until the intruder is predicted to enter the
    protected zone. ASAS only stores this for detected conflict pairs, so
    non-conflict intruder rows take the high bound (the CD lookahead horizon).

    Bounds are dynamic: unless explicit ``low``/``high`` are given they follow
    BlueSky's CD lookahead (:func:`_cd_lookahead_s`, from ``asas_dtlookahead`` /
    ``config.lookahead_s``) as ``(0, lookahead)`` - the horizon within which a
    conflict is flagged. Unlike ``tcpa`` this is non-negative (time *to* PZ
    entry), so it is a cleaner imminence signal than time-to-CPA.
    """

    meta = ObsMeta(
        "tlos_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    low: Annotated[float | None, "time seconds lower bound; None = 0 at runtime"] = None
    high: Annotated[
        float | None, "time seconds upper bound; None = CD lookahead at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``tLOS`` (one entry per ``confpairs``
        # row) each sim step - read it directly. Non-conflict intruders take the
        # high bound (the CD lookahead horizon).
        fill = _per_own(own, lambda o: self.bounds(o)[1])
        return _cd_pair_values("tLOS", own, other, fill)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        value = one_confpair_value("tLOS", own_idx, other_idx)
        return self.bounds(own_idx)[1] if value is None else value

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_lookahead_s()))


@dataclass(frozen=True)
class ClosingRateKts(_BroadcastPairs, PairObsField):
    """Ownship-intruder horizontal closing rate, in knots.

    Positive means the pair is closing horizontally; negative means opening.
    """

    meta = ObsMeta("closing_rate_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "speed knots"] = -1000.0
    high: Annotated[float, "speed knots"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
            own, other
        )
        dist_m = np.maximum(np.hypot(rel_east_m, rel_north_m), 1e-6)
        range_rate_ms = (rel_east_m * rel_east_ms + rel_north_m * rel_north_ms) / dist_m
        return -range_rate_ms * _MS_TO_KTS

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        range_m = max(math.hypot(east, north), 1e-6)
        return -(east * v_east + north * v_north) / range_m / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BearingRateDegPerSec(_BroadcastPairs, PairObsField):
    """Ownship-intruder bearing rate, in degrees per second.

    Positive means the bearing from ownship to intruder rotates clockwise.
    """

    meta = ObsMeta(
        "bearing_rate_deg_per_sec",
        Unit.DEG_PER_SEC,
        ObsQuantity.BEARING,
        is_pair=True,
    )
    low: Annotated[float, "degrees per second"] = -10.0
    high: Annotated[float, "degrees per second"] = 10.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
            own, other
        )
        dist2_m = np.maximum(rel_east_m * rel_east_m + rel_north_m * rel_north_m, 1e-6)
        rate_rad_s = (rel_north_m * rel_east_ms - rel_east_m * rel_north_ms) / dist2_m
        return np.degrees(rate_rad_s)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        range2 = max(east * east + north * north, 1e-6)
        return math.degrees((north * v_east - east * v_north) / range2)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAlongTrackNm(_BroadcastPairs, PairObsField):
    """Intruder position relative to ownship ALONG the own track, nm (ahead +).

    Track-frame Cartesian. Unlike range x bearing, this gives relative position
    directly - no multiplicative decode whose bearing resolution scales with
    range. Pairs with :class:`RelPosCrossTrackNm`.
    """

    meta = ObsMeta(
        "rel_pos_along_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "along-track nm"] = -200.0
    high: Annotated[float, "along-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, _c, _va, _vc = _track_frame(own, other)
        return along / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[0] / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosCrossTrackNm(_BroadcastPairs, PairObsField):
    """Intruder position relative to ownship ACROSS the own track, nm (right +).

    Track-frame Cartesian companion to :class:`RelPosAlongTrackNm`.
    """

    meta = ObsMeta(
        "rel_pos_cross_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "cross-track nm"] = -200.0
    high: Annotated[float, "cross-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _a, cross, _va, _vc = _track_frame(own, other)
        return cross / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[1] / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelAlongTrackKts(_BroadcastPairs, PairObsField):
    """Intruder velocity relative to ownship ALONG the own track, kts.

    Track-frame Cartesian relative velocity (negative = intruder falling behind /
    ownship overtaking along-track). Together with the cross component this is the
    same information as closing rate + bearing rate, but singularity-free and
    without the range-dependent scaling.
    """

    meta = ObsMeta("rel_vel_along_track_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "along-track kts"] = -1000.0
    high: Annotated[float, "along-track kts"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _a, _c, v_along, _vc = _track_frame(own, other)
        return v_along * _MS_TO_KTS

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[2] / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelCrossTrackKts(_BroadcastPairs, PairObsField):
    """Intruder velocity relative to ownship ACROSS the own track, kts (right +).

    Track-frame Cartesian companion to :class:`RelVelAlongTrackKts`.
    """

    meta = ObsMeta("rel_vel_cross_track_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "cross-track kts"] = -1000.0
    high: Annotated[float, "cross-track kts"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _a, _c, _va, v_cross = _track_frame(own, other)
        return v_cross * _MS_TO_KTS

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[3] / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAtCpaAlongTrackNm(_BroadcastPairs, PairObsField):
    """Along-track relative position at predicted CPA, nm (ahead +).

    Cartesian replacement for the scalar horizontal-miss `dcpa`: together with the
    cross component it preserves the miss magnitude AND adds the pass direction.
    """

    meta = ObsMeta(
        "rel_pos_at_cpa_along_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "along-track nm"] = -200.0
    high: Annotated[float, "along-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, _cross = _track_frame_at_cpa(own, other)
        return along / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        along, cross, v_along, v_cross = one_pair_track_frame(own_idx, other_idx)
        v2 = max(v_along * v_along + v_cross * v_cross, 1e-9)
        tcpa = max(-(along * v_along + cross * v_cross) / v2, 0.0)  # future only
        return (along + v_along * tcpa) / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAtCpaCrossTrackNm(_BroadcastPairs, PairObsField):
    """Cross-track relative position at predicted CPA, nm (right +).

    The sign is which side the intruder passes at closest approach - the key cue
    for turn direction. ``|along, cross|`` == the horizontal miss dcpa.
    """

    meta = ObsMeta(
        "rel_pos_at_cpa_cross_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "cross-track nm"] = -200.0
    high: Annotated[float, "cross-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _along, cross = _track_frame_at_cpa(own, other)
        return cross / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        along, cross, v_along, v_cross = one_pair_track_frame(own_idx, other_idx)
        v2 = max(v_along * v_along + v_cross * v_cross, 1e-9)
        tcpa = max(-(along * v_along + cross * v_cross) / v2, 0.0)  # future only
        return (cross + v_cross * tcpa) / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVsFtMin(_BroadcastPairs, PairObsField):
    """Intruder vertical speed minus ownship vertical speed, in ft/min."""

    meta = ObsMeta(
        "rel_vs_ft_min",
        Unit.FT_PER_MIN,
        ObsQuantity.VERTICAL_SPEED,
        is_pair=True,
    )
    low: Annotated[float, "vertical speed feet per minute"] = -6000.0
    high: Annotated[float, "vertical speed feet per minute"] = 6000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        vs = _traf_array("vs")
        return (vs[other] - vs[own]) * _MS_TO_FTMIN

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return (float(bs.traf.vs[other_idx]) - float(bs.traf.vs[own_idx])) * 60.0 / ft

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class HorizontalDistAtCpaNm(_BroadcastPairs, PairObsField):
    """BlueSky ASAS horizontal distance at closest point of approach, in NM.

    Reads the conflict detector's cached ``dcpa`` (meters, one entry per
    ``confpairs`` row) and converts to nautical miles. ASAS only stores this for
    detected conflict pairs - and a conflict has ``dcpa < rpz`` - so non-conflict
    intruder rows take the high bound (the PZ radius): the smallest miss distance
    that is not a conflict.

    Bounds are dynamic: unless explicit ``low``/``high`` are given they follow
    BlueSky's horizontal protected-zone radius (:func:`_cd_rpz_m`, from
    ``config.pz_radius_nm`` / ``asas_pzr``) as ``(0, rpz)``. That is the true
    support of the cached ``dcpa``, so the normalized value grades the actual
    miss distance instead of collapsing every conflict toward 0.
    """

    meta = ObsMeta(
        "horizontal_dist_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "distance nm lower bound; None = 0 at runtime"] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``dcpa`` (meters, one entry per
        # ``confpairs`` row) each sim step - read it directly and convert to NM.
        # Non-conflict intruders take the high bound (the PZ radius).
        fill = _per_own(own, lambda o: self.bounds(o)[1])
        return _cd_pair_values("dcpa", own, other, fill, divisor=nm)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        value = one_confpair_value("dcpa", own_idx, other_idx)
        return self.bounds(own_idx)[1] if value is None else value / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class VerticalSepAtCpaFt(_BroadcastPairs, PairObsField):
    """Predicted absolute vertical separation at horizontal CPA, in feet.

    The horizontal CPA time is computed from current traffic vectors, for every
    intruder (not only detected conflicts). Negative CPA times are clipped to
    zero, so opening pairs report current vertical separation instead of
    extrapolating into the past.

    Bounds are dynamic: unless explicit ``low``/``high`` are given they follow
    BlueSky's vertical protected-zone height (:func:`_cd_hpz_m`, from
    ``config.pz_height_ft`` / ``asas_pzh``) as ``(0, hpz)`` - the minimum
    vertical separation. Because this reports a value for every intruder, that is
    a *normalization* choice, not the data range: intruders predicted to clear
    the vertical PZ saturate at the safe edge, focusing the signal on the danger
    band. Pass explicit bounds for a wider scale.
    """

    meta = ObsMeta(
        "vertical_sep_at_cpa_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "altitude ft lower bound; None = 0 at runtime"] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        tcpa_s = np.maximum(_pair_horizontal_tcpa_s(own, other), 0.0)
        alt, vs = _traf_array("alt"), _traf_array("vs")
        rel_alt_m = alt[other] - alt[own]
        rel_vs_ms = vs[other] - vs[own]
        return np.abs(rel_alt_m + rel_vs_ms * tcpa_s) * _M_TO_FT

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        v2 = max(v_east * v_east + v_north * v_north, 1e-6)
        tcpa = max(-(east * v_east + north * v_north) / v2, 0.0)
        rel_alt = float(bs.traf.alt[other_idx]) - float(bs.traf.alt[own_idx])
        rel_vs = float(bs.traf.vs[other_idx]) - float(bs.traf.vs[own_idx])
        return abs(rel_alt + rel_vs * tcpa) / ft

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_hpz_m() * _M_TO_FT))


@dataclass(frozen=True)
class IntruderFixApproachDistNm(_BroadcastPairs, PairObsField):
    """How near an intruder's *observed* path passes the ownship's active fix, nm.

    Projects the intruder's current position/velocity to its closest approach
    of the ownship's own fix (``route_offset`` 0 = active leg, 1 = next leg,
    e.g. the shared exit while still working a merge fix). Small => that
    aircraft is funnelling into my fix (a merge threat); large => crossing or
    unrelated. Captures the shared-destination geometry that pairwise
    ownship<->intruder CPA fields miss (same-heading in-trail traffic has low
    closing rate yet converges at the fix). Non-private: see :func:`_fix_projection`.
    Sentinel ``high`` (far) when the ownship has no usable fix.
    """

    meta = ObsMeta(
        "intruder_fix_approach_dist_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "distance nm lower bound"] = 0.0
    high: Annotated[float, "distance nm upper bound (also the no-fix sentinel)"] = 100.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        cpa_nm, _eta, _own_eta, _vsep, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, cpa_nm, float(self.high))

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return float(self.high) if projection is None else projection[0]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixArrivalDeltaS(_BroadcastPairs, PairObsField):
    """Signed queue order at the ownship's fix: intruder ETA minus ownship ETA, s.

    Both ETAs come from projecting *observed* motion onto the ownship's own fix
    (``route_offset`` 0 = active, 1 = next). Negative => the intruder reaches
    my fix *before* me (I should slot behind it); positive => I am ahead. This
    is the temporal-sequencing signal for a point merge - who arrives first -
    that lets the policy learn to decelerate and space in-trail rather than
    turn (which cannot separate traffic converging on a shared point).
    Non-private: see :func:`_fix_projection`. ``0`` (no order) when the ownship
    has no usable fix. Pair with :class:`IntruderFixApproachDistNm` so a
    consumer knows *whether* the intruder is heading to the fix at all.
    """

    meta = ObsMeta(
        "intruder_fix_arrival_delta_s", Unit.S, ObsQuantity.TIME, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "time s lower bound"] = -600.0
    high: Annotated[float, "time s upper bound"] = 600.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _cpa, intr_eta, own_eta, _vsep, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, intr_eta - own_eta, 0.0)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return 0.0 if projection is None else projection[1] - projection[2]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixVerticalSepFt(_BroadcastPairs, PairObsField):
    """Signed altitude gap at the ownship's fix: intruder minus ownship, ft.

    Both aircraft are projected to the fix by their observed vertical speed
    (current alt + VS x ETA), and the signed difference is taken there. Near 0
    => they will be *co-altitude* at the merge (a genuine conflict that lateral
    turning cannot separate - your same-altitude merge case); large magnitude
    => vertically clear, no sequencing needed regardless of horizontal
    approach. This is the vertical companion to
    :class:`IntruderFixApproachDistNm` (horizontal) and
    :class:`IntruderFixArrivalDeltaS` (temporal); together they predict a merge
    conflict against the ownship's fix the way the pairwise
    ``ConflictHorizontalDistAtCpaNm`` / ``ConflictVerticalSepAtCpaFt`` /
    ``ConflictTcpaS`` triple predicts a pairwise one. Signed so the policy
    knows which way to separate. Non-private: see :func:`_fix_projection`.
    ``0`` when the ownship has no usable fix - read alongside the approach
    field, whose sentinel flags that state.
    """

    meta = ObsMeta(
        "intruder_fix_vertical_sep_ft", Unit.FT, ObsQuantity.ALTITUDE, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "altitude ft lower bound"] = -5000.0
    high: Annotated[float, "altitude ft upper bound"] = 5000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _cpa, _eta, _own_eta, vsep_ft, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, vsep_ft, 0.0)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return 0.0 if projection is None else projection[3]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BrgFromOwnDeg(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder bearing in degrees.

    The true bearing from ownship to the intruder, signed in ``[-180, 180]``
    (0 = north, +90 = east, -90 = west) as returned by ``qdrdist``.
    """

    meta = ObsMeta(
        "brg_from_own_deg",
        Unit.DEG,
        ObsQuantity.BEARING,
        is_pair=True,
        circular=True,
    )
    low: Annotated[float, "bearing degrees"] = -180.0
    high: Annotated[float, "bearing degrees"] = 180.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        qdr, _dist = _pair_qdr_dist(own, other)
        return qdr

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        own = (float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx]))
        other = (float(bs.traf.lat[other_idx]), float(bs.traf.lon[other_idx]))
        return float(qdrdist(*own, *other)[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BrgFromOwnRelTrkDeg(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder bearing in the ownship track frame, degrees.

    The true bearing from ownship to the intruder minus the ownship track, so
    ``0`` means the intruder is dead ahead and ``+/-180`` directly behind. This
    is the egocentric (body-frame) polar angle, matching
    :class:`ActiveRouteWaypointTrackErrorDeg` for the waypoint; pair with
    :class:`DistToOwnNm` for a full egocentric polar intruder position. Unlike
    :class:`BrgFromOwnDeg` (an absolute compass bearing), this rotates with the
    ownship heading.
    """

    meta = ObsMeta(
        "brg_from_own_rel_trk_deg",
        Unit.DEG,
        ObsQuantity.BEARING,
        is_pair=True,
        circular=True,
    )
    low: Annotated[float, "bearing degrees"] = -180.0
    high: Annotated[float, "bearing degrees"] = 180.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        qdr, _dist = _pair_qdr_dist(own, other)
        return (qdr - _traf_array("trk")[own] + 540.0) % 360.0 - 180.0

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        own = (float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx]))
        other = (float(bs.traf.lat[other_idx]), float(bs.traf.lon[other_idx]))
        relative = float(qdrdist(*own, *other)[0]) - float(bs.traf.trk[own_idx])
        return (relative + 180.0) % 360.0 - 180.0

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()
