"""Conflict fields: BlueSky's conflict detection and the shared conflict
geometry, per intruder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from bluesky.tools.aero import ft, nm

from bluesky_sandbox.sim.geometry.conflict import (
    ConflictView,
    predicted_tlos_s,
    windowed_min_hsep_nm,
    windowed_min_vsep_ft,
    windowed_signed_vsep_at_entry_ft,
)
from bluesky_sandbox.sim.geometry.conflict import cd_hpz_m as _cd_hpz_m
from bluesky_sandbox.sim.geometry.conflict import cd_lookahead_s as _cd_lookahead_s
from bluesky_sandbox.sim.geometry.conflict import cd_rpz_m as _cd_rpz_m

from .._common import _M_TO_FT
from .._pairs import _BroadcastPairs, _cd_pair_values, _GeomPairs, _per_own
from .._reference import one_confpair_value
from ..base import ObsMeta, ObsQuantity, PairObsField, Unit


@dataclass(frozen=True)
class _ConflictGeomPairField(_BroadcastPairs, PairObsField):
    """Intruder field sourced from the shared per-step conflict geometry.

    Reads :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView` - the *same*
    continuous, all-pairs CPA primitives the cost and keep mask consume - so the
    feature is identical by construction to what drives the cost. Unlike the ASAS
    ``confpairs`` readers (:class:`HorizontalDistAtCpaNm`, :class:`TcpaS`,
    :class:`TlosS`) there is no detector-cache sentinel snapping and no
    detected-only gap: every intruder gets its true predicted geometry.
    """

    _geom_attr: ClassVar[str] = ""

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = _GeomPairs(own, other)
        return np.asarray(getattr(view, self._geom_attr), dtype=np.float64)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(getattr(view, self._geom_attr)[0])


@dataclass(frozen=True)
class _WindowedConflictPairField(_ConflictGeomPairField):
    """Conflict field whose value is measured over a 3-D conflict *window*.

    Adds an optional zone override. The window defaults to the live CD
    ``rpz``/``hpz``, which is right whenever the cost grades the true protected
    zone. Tasks that grade a *buffered* zone - a shaped margin wider than the PZ,
    so resolutions are not trained to the PZ edge - must widen the observation's
    window to match, or the marginal encounters the buffer exists to charge are
    reported to the policy as clean misses (they fall to the field's safe-miss
    branch) while the cost bills them.

    Overriding here rather than via ``config.pz_radius_nm`` / ``pz_height_ft`` is
    deliberate: those move BlueSky's CD zone, which is what ``bs.traf.cd.lospairs``
    - and therefore any loss-of-separation cost channel - is defined against.
    Widening the CD zone to fix an observation would silently redefine what counts
    as a LoS. The observation window and the LoS predicate are different things and
    are configured separately.

    Bound defaults still follow the CD zone, so an override wants explicit
    ``low``/``high`` alongside it - otherwise the normalizer re-clips away the very
    band the wider window just exposed.
    """

    rpz_nm: Annotated[
        float | None, "window horizontal radius nm; None = CD rpz at runtime"
    ] = None
    vpz_ft: Annotated[
        float | None, "window vertical half-height ft; None = CD hpz at runtime"
    ] = None

    def _window_zone(self) -> tuple[float, float]:
        """``(rpz_nm, vpz_ft)`` for the window: overrides, else the live CD zone."""
        rpz = _cd_rpz_m() / nm if self.rpz_nm is None else float(self.rpz_nm)
        vpz = _cd_hpz_m() * _M_TO_FT if self.vpz_ft is None else float(self.vpz_ft)
        if rpz <= 0.0 or vpz <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__} window zone must be positive, "
                f"got rpz_nm={rpz}, vpz_ft={vpz}."
            )
        return rpz, vpz


@dataclass(frozen=True)
class ConflictHorizontalDistAtCpaNm(_ConflictGeomPairField):
    """Predicted *horizontal* separation at CPA, in nm, from shared conflict geometry.

    The continuous, all-pairs counterpart of :class:`HorizontalDistAtCpaNm` (which
    reads the ASAS ``confpairs`` cache and sentinels non-detected pairs). Bounds
    are dynamic: unless given, ``(0, CD rpz)`` - so a clipped normalizer grades the
    danger band and saturates every safer miss at the PZ edge, matching the cost's
    ``r_h`` support.
    """

    meta = ObsMeta(
        "conflict_horizontal_dist_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    _geom_attr: ClassVar[str] = "dcpa_nm"
    low: Annotated[float | None, "distance nm lower bound; None = 0"] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class ConflictVerticalSepAtCpaFt(_WindowedConflictPairField):
    """Predicted minimum *vertical* separation over the 3-D conflict window, in ft.

    The continuous, all-pairs counterpart of :class:`VerticalSepAtCpaFt`, reading
    the SAME windowed measure the cost's penetration term (``r_v``) grades via
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_min_vsep_ft` - the rel-alt
    line at its in-window minimum, NOT the vertical sep at the *horizontal* CPA
    instant. The CPA-instant reading collapsed to ~0 for altitude-crossing traffic
    whose vertical crossing is offset in time from the horizontal CPA, hiding
    developing vertical conflicts from the policy that the cost was charging; the
    windowed minimum is exactly what the cost sees. A safe miss (no valid 3-D
    window) falls back to the classic CPA-instant value. Bounds are dynamic: unless
    given, ``(0, CD hpz)`` - a clipped normalizer grades the danger band and
    saturates every safe pair at the PZ edge.
    """

    meta = ObsMeta(
        "conflict_vertical_sep_at_cpa_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "altitude ft lower bound; None = 0"] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return windowed_min_vsep_ft(_GeomPairs(own, other), *self._window_zone())

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(windowed_min_vsep_ft(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_hpz_m() * _M_TO_FT))


@dataclass(frozen=True)
class ConflictHorizontalSepAtCpaNm(_WindowedConflictPairField):
    """Predicted minimum *horizontal* separation over the 3-D conflict window, nm.

    The horizontal twin of :class:`ConflictVerticalSepAtCpaFt`, reading the SAME
    windowed measure the cost's penetration term (``r_h``) grades via
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_min_hsep_nm` - the
    separation hyperbola at its in-window minimum, NOT the miss distance at the
    unconstrained CPA.

    **Why this is not just ``dcpa``.** :class:`ConflictHorizontalDistAtCpaNm` and
    the :class:`RelPosAtCpaAlongTrackNm` / :class:`RelPosAtCpaCrossTrackNm` pair
    all report ``dcpa``, the miss over ALL time. The cost charges the pair only
    while it is inside both bands at once, so when the horizontal CPA falls
    outside that window the two diverge - always with ``h_min >= dcpa``, i.e. the
    ``dcpa`` reading is the more alarming one. A policy steering on it maneuvers
    for encounters the cost never bills, and pays for the detour in track miles.

    This is the horizontal half of the same correction
    :class:`ConflictVerticalSepAtCpaFt` made vertically; that one was the more
    urgent because sampling vertical separation at the *horizontal* CPA instant
    was first-order wrong (the wrong axis's extremum time), whereas ``dcpa`` is
    only window-truncation wrong. Keep the signed ``RelPosAtCpa*`` pair alongside
    this field - they carry the pass DIRECTION, which this magnitude does not.

    Bounds are dynamic: unless given, ``(0, CD rpz)`` - a clipped normalizer
    grades the danger band and saturates every safer miss at the PZ edge, matching
    the cost's ``r_h`` support. Pass ``rpz_nm``/``vpz_ft`` (and matching
    ``low``/``high``) when the cost grades a buffered zone rather than the CD one.
    """

    meta = ObsMeta(
        "conflict_horizontal_sep_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "distance nm lower bound; None = 0"] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return windowed_min_hsep_nm(_GeomPairs(own, other), *self._window_zone())

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(windowed_min_hsep_nm(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class ConflictSignedVerticalSepAtEntryFt(_WindowedConflictPairField):
    """SIGNED vertical separation, in ft, as the 3-D conflict window opens.

    Positive = intruder ABOVE ownship. The vertical counterpart of the signed
    horizontal :class:`RelPosAtCpaAlongTrackNm` / :class:`RelPosAtCpaCrossTrackNm`
    pair, and the missing half of :class:`ConflictVerticalSepAtCpaFt`: that field
    grades HOW CLOSE the pair comes vertically (matching the cost's ``r_v`` by
    construction) but is an absolute value, so on its own it never says which way
    to go. This one supplies the side.

    Sampled at window entry rather than at the in-window minimum on purpose - the
    minimum of a crossing pair sits at the rel-alt zero, where the sign is
    undefined and flips, so a signed *minimum* would be blank exactly where the
    direction matters. See
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_signed_vsep_at_entry_ft`.

    Bounds are dynamic: unless given, ``(-CD hpz, +CD hpz)``. Pair with
    :class:`SymmetricNormalizer` so the whole PZ band spans ``[-1, 1]`` and every
    safe pair saturates at the correct end - unlike a wide ``relative_alt_ft``,
    where the danger band is squeezed into a few percent of the input range.
    """

    meta = ObsMeta(
        "conflict_signed_vertical_sep_at_entry_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None, "altitude ft lower bound; None = -CD PZ height at runtime"
    ] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = +CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = _GeomPairs(own, other)
        return windowed_signed_vsep_at_entry_ft(view, *self._window_zone())

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(windowed_signed_vsep_at_entry_ft(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        hpz_ft = _cd_hpz_m() * _M_TO_FT
        return self._dynamic_or_configured_bounds(lambda: (-hpz_ft, hpz_ft))


@dataclass(frozen=True)
class ConflictTcpaS(_ConflictGeomPairField):
    """Time to horizontal CPA (s), from shared conflict geometry; negative past it.

    The continuous, all-pairs counterpart of :class:`TcpaS` (ASAS ``confpairs``
    cache). Bounds are dynamic: unless given, ``(-lookahead, +lookahead)``.
    """

    meta = ObsMeta(
        "conflict_tcpa_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    _geom_attr: ClassVar[str] = "tcpa_s"
    low: Annotated[
        float | None, "time s lower bound; None = -CD lookahead at runtime"
    ] = None
    high: Annotated[
        float | None, "time s upper bound; None = +CD lookahead at runtime"
    ] = None

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (-_cd_lookahead_s(), _cd_lookahead_s())
        )


@dataclass(frozen=True)
class ConflictTlosS(_WindowedConflictPairField):
    """Predicted time to 3-D LoS entry (BlueSky ``tinconf``), s, from shared geometry.

    The continuous, all-pairs counterpart of :class:`TlosS` (ASAS ``confpairs``
    cache). Computed by
    :func:`~bluesky_sandbox.sim.geometry.conflict.predicted_tlos_s` - the same shared
    ``tinconf`` the cost's imminence term reads - using the CD ``rpz``/``hpz``.
    Non-conflict pairs saturate at the high bound (lookahead); an already-entered
    LoS reads 0. Bounds are dynamic: unless given, ``(0, lookahead)``.
    """

    meta = ObsMeta(
        "conflict_tlos_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    low: Annotated[float | None, "time s lower bound; None = 0"] = None
    high: Annotated[
        float | None, "time s upper bound; None = CD lookahead at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        tinconf = predicted_tlos_s(_GeomPairs(own, other), *self._window_zone())
        # +inf (no conflict) -> high bound (safe); <= 0 (already in LoS) -> 0.
        return np.clip(tinconf, 0.0, _per_own(own, lambda o: self.bounds(o)[1]))

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        tinconf = float(predicted_tlos_s(view, *self._window_zone())[0])
        return min(max(tinconf, 0.0), self.bounds(own_idx)[1])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_lookahead_s()))


@dataclass(frozen=True)
class InConf(_WindowedConflictPairField):
    """1.0 when the intruder is *currently flagged in conflict* with the ownship.

    The detection predicate itself, BlueSky ``StateBased.detect`` reproduced over
    the shared :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView`: the
    pair enters the 3-D protected zone within the lookahead horizon
    (``tinconf <= lookahead``, from
    :func:`~bluesky_sandbox.sim.geometry.conflict.predicted_tlos_s` - the same
    ``tinconf`` the cost's imminence term reads). Non-conflict pairs score ``0``,
    a pair already inside the zone (``tinconf <= 0``) scores ``1`` until it exits.

    Computed from the shared geometry rather than read from ``bs.traf.cd``
    (:class:`ConflictRisk`, :class:`TcpaS`, :class:`TlosS` do the latter), so it
    agrees with a ``ConflictView``-derived cost by construction and carries no
    detector-cache gap: every intruder is scored every step, not only the rows CD
    happened to cache. It is the binary companion of :class:`ConflictTlosS` -
    same window, same zone - stated where a network can read it.

    **Why a dedicated flag** rather than thresholding ``conflict_tlos_s``: that
    field clips ``+inf`` (no conflict at all) to its high bound, which is also
    where a conflict entering *exactly* at the horizon sits, so the two are
    indistinguishable at the top of the normalized range. This field separates
    them. The same argument :class:`InLosNow` makes for the LoS predicate.

    Pair the two: this one is the approach CD would alert on, :class:`InLosNow`
    is the breach. The implication runs one way only - a pair inside the zone is
    by construction inside its own conflict window and stays flagged until it
    exits, so ``in_los_now = 1`` always comes with ``in_conf = 1``, while the
    reverse is false for every conflict still minutes from entry. That gap is
    the point: it is the warning time a resolution has to act in, and only this
    field marks its opening.

    ``rpz_nm`` / ``vpz_ft`` / ``lookahead_s`` default to the live CD values and
    override together with the cost's zone - see
    :class:`_WindowedConflictPairField` for why a buffered zone belongs here and
    not in ``config.pz_radius_nm``.

    Leave ``normalizer`` unset - the value is already 0/1, so there is nothing
    to scale.
    """

    meta = ObsMeta("in_conf", Unit.UNITLESS, ObsQuantity.INDICATOR, is_pair=True)
    lookahead_s: Annotated[
        float | None, "detection horizon s; None = CD lookahead at runtime"
    ] = None
    low: Annotated[float, "indicator lower bound"] = 0.0
    high: Annotated[float, "indicator upper bound"] = 1.0

    def _horizon_s(self) -> float:
        look = (
            _cd_lookahead_s() if self.lookahead_s is None else float(self.lookahead_s)
        )
        if look <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__} lookahead_s must be positive, got {look}."
            )
        return look

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # ``+inf`` (no valid conflict window) compares False against any finite
        # horizon, so the horizon test alone is the full detection predicate.
        tinconf = predicted_tlos_s(_GeomPairs(own, other), *self._window_zone())
        return (tinconf <= self._horizon_s()).astype(np.float32)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        tinconf = float(predicted_tlos_s(view, *self._window_zone())[0])
        return float(tinconf <= self._horizon_s())

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class InLosNow(_BroadcastPairs, PairObsField):
    """1.0 when the intruder is *currently* inside the ownship's protected zone.

    The loss-of-separation predicate itself: ``horizontal < rpz AND |dalt| < hpz``,
    read from the shared :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView`
    (``horiz_dist_now_nm`` / ``dalt_now_ft``) against the live CD ``rpz``/``hpz``.
    Being current-state rather than predicted, it is the odd one out among the
    ``Conflict*`` fields - they all describe geometry at a future CPA, this one
    describes right now, hence the ``Now`` suffix mirroring the view's own
    attribute names.

    **Why a dedicated flag** rather than letting the network derive it. LoS is
    typically what a constrained task *counts* as its cost, and a policy that is
    penalized for it usually cannot see it: the continuous features either sit on
    the wrong side of a critic-only split, or bury the predicate in a few percent
    of their range (a 5 nm PZ inside a +-60 nm ``RelPosAlongTrackNm`` is ~4%), or
    - as with :class:`ConflictTlosS`, which reads exactly ``0`` once LoS is
    entered - encode it at a single endpoint of the normalized range, where it is
    indistinguishable from saturation. This field states it directly.

    Pair it with :class:`ConflictTlosS` for the approach and this for the entry
    (or with :class:`InConf`, the binary form of the same approach); the two
    carry different information and neither substitutes for the other.
    For a *graded* conflict signal see :class:`ConflictRisk` - but note that one
    reads BlueSky's ASAS ``confpairs`` cache, so it does not necessarily agree
    with a cost computed from ``ConflictView``, whereas this field does by
    construction.

    Leave ``normalizer`` unset - the value is already 0/1, so there is nothing
    to scale.
    """

    meta = ObsMeta("in_los_now", Unit.UNITLESS, ObsQuantity.INDICATOR, is_pair=True)
    low: Annotated[float, "indicator lower bound"] = 0.0
    high: Annotated[float, "indicator upper bound"] = 1.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = _GeomPairs(own, other)
        # Strict ``<`` on both axes, matching BlueSky's own LoS test (and the
        # keep-mask/cost predicates built on this view): a pair sitting exactly
        # ON the zone boundary is not yet a loss of separation.
        in_los = (view.horiz_dist_now_nm < _cd_rpz_m() / nm) & (
            view.dalt_now_ft < _cd_hpz_m() * _M_TO_FT
        )
        return in_los.astype(np.float32)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        horizontal = float(view.horiz_dist_now_nm[0]) < _cd_rpz_m() / nm
        vertical = float(view.dalt_now_ft[0]) < _cd_hpz_m() / ft
        return float(horizontal and vertical)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ConflictRisk(_BroadcastPairs, PairObsField):
    """Per-intruder graded conflict risk, ``1 - tcpa/lookahead`` in ``[0, 1]``.

    The per-element version of a CD-based safety cost: for an intruder BlueSky's
    conflict detector flags against the ownship, risk is
    ``1 - max(tcpa, 0)/lookahead`` (lookahead from :func:`_cd_lookahead_s`);
    non-conflict intruders score ``0``. Placed on the intruder token so attention
    can focus on the threatening aircraft, with the ownship's worst-case risk
    recoverable by max-pooling this field.
    """

    meta = ObsMeta("conflict_risk", Unit.UNITLESS, ObsQuantity.RISK, is_pair=True)
    low: Annotated[float, "risk fraction"] = 0.0
    high: Annotated[float, "risk fraction"] = 1.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``tcpa`` per ``confpairs`` row each sim
        # step. Non-conflict rows default to the lookahead horizon -> risk 0;
        # conflict rows use their CD tcpa, graded toward 1 as the conflict nears.
        lookahead = _cd_lookahead_s()
        tcpa = _cd_pair_values("tcpa", own, other, lookahead)
        risk = 1.0 - np.maximum(tcpa, 0.0) / lookahead
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        lookahead = _cd_lookahead_s()
        tcpa = one_confpair_value("tcpa", own_idx, other_idx)
        if tcpa is None:
            tcpa = lookahead
        return min(max(1.0 - max(tcpa, 0.0) / lookahead, 0.0), 1.0)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()
