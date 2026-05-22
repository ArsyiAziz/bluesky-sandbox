"""Shared per-step conflict geometry.

One vectorized pairwise closest-point-of-approach (CPA) computation, cached and
self-invalidating, read by cost functions, keep masks, and hooks so that
*costed* and *kept* describe the **same** geometry (this is what eliminates whole
classes of mismatch bugs, e.g. a cost that sees a conflict the keep mask does
not).

The :class:`ConflictView` exposes only **raw kinematic primitives** - unambiguous
facts about the linear extrapolation of each pair. Anything derived
(protected-zone penetration, imminence, severity, a breach flag) is a *policy*
with approximations and thresholds, so it is deliberately **not** a view field:
consumers compute it from these primitives with their own ``rpz``/``vpz``/horizon.
That keeps the primitives honest - every field means exactly what it says. The one
concession is :func:`predicted_tlos_s`, a *parameterized module function* (it takes
the caller's ``rpz``/``vpz``) rather than a view field - it exists so the
``tinconf`` (time-to-LoS) computation has a single shared implementation instead of
being copied into each consumer.

For each ordered pair ``(i, j)``, on a flat-earth ENU extrapolation (exact enough
for a ~100 nm sector):

- ``tcpa_s``             time to the *horizontal* CPA (s); negative once past it.
- ``dcpa_nm``           horizontal separation at that horizontal CPA (nm).
- ``vsep_at_cpa_ft``    vertical separation at that horizontal CPA (ft).
- ``horiz_dist_now_nm`` *current* horizontal separation (nm).
- ``dalt_now_ft``       *current* vertical separation (ft), unsigned.
- ``rel_hspd_kts``      horizontal relative speed (kts), ``|v_j - v_i|`` (>=0).
- ``rel_alt_now_ft``    *current* **signed** relative altitude (ft), ``alt_j - alt_i``.
- ``rel_vs_ft_min``     **signed** relative vertical speed (ft/min), ``vs_j - vs_i``.

All arrays are ``N x N`` in ``bs.traf`` order (diagonal 0). Unlike BlueSky's own
``confpairs`` this is not capped at ``DTLOOK`` and does not depend on the CD's
update timing - every pair is computed, and consumers threshold as they see fit.

Per-agent access is through :class:`ConflictView` (``context.conflicts``), which
exposes the ownship's row over the *other* aircraft with reductions and filters.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft as _FT
from bluesky.tools.aero import kts as _KTS
from bluesky.tools.aero import nm as _NM

_M_TO_FT = 1.0 / _FT
_MS_TO_KTS = 1.0 / _KTS
_MS_TO_FTMIN = _M_TO_FT * 60.0
_RE = 6371000.0  # earth radius, metres (matches BlueSky ``kwikqdrdist``)

_CACHE: dict = {"geom": None, "key": None}


class ConflictGeometry:
    """Immutable per-step pairwise CPA arrays (``N x N``, ``bs.traf`` order)."""

    __slots__ = (
        "dalt_now_ft",
        "dcpa_nm",
        "horiz_dist_now_nm",
        "n",
        "rel_alt_now_ft",
        "rel_hspd_kts",
        "rel_vs_ft_min",
        "tcpa_s",
        "vsep_at_cpa_ft",
    )

    def __init__(self, **arrays) -> None:
        for name, value in arrays.items():
            setattr(self, name, value)


def _compute() -> ConflictGeometry:
    n = len(bs.traf.id)
    if n < 2:
        z = np.zeros((n, n), dtype=np.float64)
        return ConflictGeometry(
            tcpa_s=z.copy(),
            dcpa_nm=z.copy(),
            vsep_at_cpa_ft=z.copy(),
            horiz_dist_now_nm=z.copy(),
            dalt_now_ft=z.copy(),
            rel_hspd_kts=z.copy(),
            rel_alt_now_ft=z.copy(),
            rel_vs_ft_min=z.copy(),
            n=n,
        )

    lat = np.asarray(bs.traf.lat, dtype=np.float64)
    lon = np.asarray(bs.traf.lon, dtype=np.float64)
    alt = np.asarray(bs.traf.alt, dtype=np.float64)
    vs = np.asarray(bs.traf.vs, dtype=np.float64)

    # Pairwise flat-earth (equirectangular) offset of j relative to i, matching
    # BlueSky's ``kwikqdrdist`` projection but vectorized over all pairs: the
    # compiled geo kernels do not broadcast 2-D inputs (they return a scalar), so
    # compute the projection directly with the same earth radius and mean-lat
    # cosine factor.
    dlat = np.radians(lat[None, :] - lat[:, None])  # [i, j]
    dlon = np.radians(lon[None, :] - lon[:, None])
    cavelat = np.cos(np.radians(lat[:, None] + lat[None, :]) * 0.5)
    dx = _RE * dlon * cavelat  # position of j relative to i (east), metres
    dy = _RE * dlat  # (north), metres
    dist = np.sqrt(dx * dx + dy * dy)  # metres, [i, j]
    horiz_dist_now_nm = dist / _NM

    trkrad = np.radians(np.asarray(bs.traf.trk, dtype=np.float64))
    gs = np.asarray(bs.traf.gs, dtype=np.float64)
    u = gs * np.sin(trkrad)
    v = gs * np.cos(trkrad)
    du = u[None, :] - u[:, None]  # relative velocity of j w.r.t. i (east)
    dv = v[None, :] - v[:, None]  # (north)
    dv2 = np.maximum(du * du + dv * dv, 1e-6)

    tcpa = -(du * dx + dv * dy) / dv2
    tcpa_pos = np.maximum(tcpa, 0.0)
    dcpa_nm = np.sqrt(np.abs(dist * dist - tcpa * tcpa * dv2)) / _NM
    rel_hspd_kts = np.sqrt(dv2) * _MS_TO_KTS  # horizontal relative speed (matches BlueSky vrel)
    rel_alt_now = alt[None, :] - alt[:, None]  # signed (j - i), metres
    rel_vs = vs[None, :] - vs[:, None]  # signed (j - i), m/s
    vsep_at_cpa_ft = np.abs(rel_alt_now + rel_vs * tcpa_pos) * _M_TO_FT
    dalt_now_ft = np.abs(rel_alt_now) * _M_TO_FT
    rel_alt_now_ft = rel_alt_now * _M_TO_FT
    rel_vs_ft_min = rel_vs * _MS_TO_FTMIN

    for m in (
        tcpa,
        dcpa_nm,
        vsep_at_cpa_ft,
        horiz_dist_now_nm,
        dalt_now_ft,
        rel_hspd_kts,
        rel_alt_now_ft,
        rel_vs_ft_min,
    ):
        np.fill_diagonal(m, 0.0)

    return ConflictGeometry(
        tcpa_s=tcpa,
        dcpa_nm=dcpa_nm,
        vsep_at_cpa_ft=vsep_at_cpa_ft,
        horiz_dist_now_nm=horiz_dist_now_nm,
        dalt_now_ft=dalt_now_ft,
        rel_hspd_kts=rel_hspd_kts,
        rel_alt_now_ft=rel_alt_now_ft,
        rel_vs_ft_min=rel_vs_ft_min,
        n=n,
    )


def conflict_geometry() -> ConflictGeometry:
    """Return the per-step conflict geometry, computing once and caching.

    Self-invalidating: keyed on ``(sim time, live-id tuple)``, so it recomputes
    exactly when the sim advances or the traffic set changes - no env hook or
    manual invalidation needed. Safe to call from cost functions, the keep mask,
    and hooks; the O(N^2) geometry runs at most once per step.
    """
    key = (getattr(getattr(bs, "sim", None), "simt", None), tuple(bs.traf.id))
    if _CACHE["geom"] is None or _CACHE["key"] != key:
        _CACHE["geom"] = _compute()
        _CACHE["key"] = key
    return _CACHE["geom"]


class ConflictView:
    """The ownship's per-step conflict view over the other aircraft.

    Exposed as ``context.conflicts``. Per-intruder arrays are aligned to *all*
    other live aircraft (``bs.traf`` order excluding self) - the same ordering as
    ``context.intruder_values`` and the obs intruder block. All reductions are
    safe when the agent is alone (return ``inf`` / empty arrays).
    """

    __slots__ = ("_g", "_i", "_others")

    def __init__(self, acidx: int, geom: ConflictGeometry | None = None, others=None):
        self._i = int(acidx)
        self._g = geom if geom is not None else conflict_geometry()
        if others is None:
            others = np.array(
                [j for j in range(self._g.n) if j != self._i], dtype=np.intp
            )
        self._others = np.asarray(others, dtype=np.intp)

    def _row(self, name: str) -> np.ndarray:
        m = getattr(self._g, name)
        if self._i >= self._g.n or self._others.size == 0:
            return np.empty(0, dtype=m.dtype)
        return m[self._i, self._others]

    # --- raw per-intruder arrays (aligned to other live aircraft) ---
    @property
    def tcpa_s(self) -> np.ndarray:
        return self._row("tcpa_s")

    @property
    def dcpa_nm(self) -> np.ndarray:
        return self._row("dcpa_nm")

    @property
    def vsep_at_cpa_ft(self) -> np.ndarray:
        return self._row("vsep_at_cpa_ft")

    @property
    def horiz_dist_now_nm(self) -> np.ndarray:
        return self._row("horiz_dist_now_nm")

    @property
    def dalt_now_ft(self) -> np.ndarray:
        return self._row("dalt_now_ft")

    @property
    def rel_hspd_kts(self) -> np.ndarray:
        return self._row("rel_hspd_kts")

    @property
    def rel_alt_now_ft(self) -> np.ndarray:
        return self._row("rel_alt_now_ft")

    @property
    def rel_vs_ft_min(self) -> np.ndarray:
        return self._row("rel_vs_ft_min")

    # --- scalar reductions (safe on empty) ---
    @property
    def min_tcpa_s(self) -> float:
        row = self.tcpa_s
        pos = row[row > 0.0]
        return float(pos.min()) if pos.size else float("inf")

    @property
    def min_dcpa_nm(self) -> float:
        row = self.dcpa_nm
        return float(row.min()) if row.size else float("inf")

    @property
    def min_horiz_dist_now_nm(self) -> float:
        row = self.horiz_dist_now_nm
        return float(row.min()) if row.size else float("inf")

    # --- filters ---
    def within(self, horizon_s: float) -> ConflictView:
        """A view restricted to intruders whose (converging) horizontal CPA is
        within ``horizon_s`` seconds."""
        t = self.tcpa_s
        if t.size == 0:
            return ConflictView(self._i, self._g, self._others)
        mask = (t > 0.0) & (t <= horizon_s)
        return ConflictView(self._i, self._g, self._others[mask])

    def __len__(self) -> int:
        return int(self._others.size)

    def __iter__(self):
        ids = bs.traf.id
        tcpa = self.tcpa_s
        dcpa = self.dcpa_nm
        vsep = self.vsep_at_cpa_ft
        dist = self.horiz_dist_now_nm
        for k, j in enumerate(self._others):
            yield _ConflictRecord(
                intruder_id=ids[int(j)],
                tcpa_s=float(tcpa[k]),
                dcpa_nm=float(dcpa[k]),
                vsep_at_cpa_ft=float(vsep[k]),
                horiz_dist_now_nm=float(dist[k]),
            )


class _ConflictRecord:
    """One (ownship, intruder) predicted encounter, yielded when iterating a view."""

    __slots__ = ("dcpa_nm", "horiz_dist_now_nm", "intruder_id", "tcpa_s", "vsep_at_cpa_ft")

    def __init__(self, intruder_id, tcpa_s, dcpa_nm, vsep_at_cpa_ft, horiz_dist_now_nm):
        self.intruder_id = intruder_id
        self.tcpa_s = tcpa_s
        self.dcpa_nm = dcpa_nm
        self.vsep_at_cpa_ft = vsep_at_cpa_ft
        self.horiz_dist_now_nm = horiz_dist_now_nm


def predicted_tlos_s(view: ConflictView, rpz_nm: float, vpz_ft: float) -> np.ndarray:
    """Predicted time to 3-D loss-of-separation entry (BlueSky ``tinconf``), seconds.

    A *derived, thresholded* quantity, so - by this module's contract - it is a
    parameterized **function** taking the caller's ``rpz``/``vpz``, not a
    ``ConflictView`` primitive. It exists as the single shared implementation so
    every consumer (the cost's imminence term, the critic's tLOS obs field) reads
    one ``tinconf``, never a divergent copy.

    Mirrors ``StateBased.detect`` over the row's raw primitives:
    ``tinconf = max(tinhor, tinver)`` of the horizontal (``dcpa < rpz``) and
    vertical (``|rel_alt| < vpz``) entry times, using the exposed ``rel_hspd_kts``
    (stable through CPA) and the signed ``rel_alt_now_ft`` / ``rel_vs_ft_min``.
    Returns ``+inf`` where no conflict is predicted (a band never breached, the two
    windows disjoint, or the encounter already fully passed) and ``<= 0`` while
    already inside the PZ.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    if tcpa.size == 0:
        return tcpa
    dcpa = np.asarray(view.dcpa_nm, dtype=np.float64)

    # Horizontal entry/exit (tinhor / touthor).
    swhor = dcpa < rpz_nm
    vrel_nm_s = np.asarray(view.rel_hspd_kts, dtype=np.float64) / 3600.0
    dxinhor = np.sqrt(np.maximum(0.0, rpz_nm**2 - dcpa**2))
    dtinhor = dxinhor / vrel_nm_s
    tinhor = np.where(swhor, tcpa - dtinhor, np.inf)
    touthor = np.where(swhor, tcpa + dtinhor, -np.inf)

    # Vertical band crossing (tinver / toutver) of the signed rel-alt line.
    d0 = np.asarray(view.rel_alt_now_ft, dtype=np.float64)
    dvs = np.asarray(view.rel_vs_ft_min, dtype=np.float64) / 60.0  # ft/s, signed
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)                  # guard, as BlueSky
    t_hi = (vpz_ft - d0) / dvs
    t_lo = (-vpz_ft - d0) / dvs
    tinver = np.minimum(t_hi, t_lo)
    toutver = np.maximum(t_hi, t_lo)

    tinconf = np.maximum(tinhor, tinver)
    toutconf = np.minimum(touthor, toutver)
    valid = swhor & (tinconf <= toutconf) & (toutconf > 0.0)
    return np.where(valid, tinconf, np.inf)


def _shared_conflict_window(
    view: ConflictView, rpz_nm: float, vpz_ft: float
) -> tuple[np.ndarray, ...]:
    """``(valid, tinconf, toutconf, d0, dvs)`` for the shared 3-D conflict window.

    The single implementation of the window :func:`windowed_min_vsep_ft` and
    :func:`windowed_signed_vsep_at_entry_ft` both key off, so the magnitude the
    cost grades and the sign the policy observes can never disagree about which
    encounter they describe. ``d0`` / ``dvs`` are the signed rel-alt line's
    intercept (ft) and slope (ft/s), returned so callers can sample it.

    Deliberately NOT shared with :func:`predicted_tlos_s`: that function divides
    by ``vrel_nm_s`` unguarded, and folding it in here would silently change the
    cost's imminence term. Left alone on purpose.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    dcpa = np.asarray(view.dcpa_nm, dtype=np.float64)

    # Horizontal entry/exit (tinhor / touthor) - mirrors predicted_tlos_s.
    swhor = dcpa < rpz_nm
    vrel_nm_s = np.asarray(view.rel_hspd_kts, dtype=np.float64) / 3600.0
    dxinhor = np.sqrt(np.maximum(0.0, rpz_nm**2 - dcpa**2))
    dtinhor = dxinhor / np.where(vrel_nm_s > 1e-9, vrel_nm_s, 1e-9)
    tinhor = np.where(swhor, tcpa - dtinhor, np.inf)
    touthor = np.where(swhor, tcpa + dtinhor, -np.inf)

    # Vertical band crossing (tinver / toutver) of the signed rel-alt line.
    d0 = np.asarray(view.rel_alt_now_ft, dtype=np.float64)
    dvs = np.asarray(view.rel_vs_ft_min, dtype=np.float64) / 60.0  # ft/s, signed
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)                  # guard, as BlueSky
    t_hi = (vpz_ft - d0) / dvs
    t_lo = (-vpz_ft - d0) / dvs
    tinver = np.minimum(t_hi, t_lo)
    toutver = np.maximum(t_hi, t_lo)

    tinconf = np.maximum(tinhor, tinver)
    toutconf = np.minimum(touthor, toutver)
    valid = swhor & (tinconf <= toutconf) & (toutconf > 0.0)
    return valid, tinconf, toutconf, d0, dvs


def windowed_min_vsep_ft(view: ConflictView, rpz_nm: float, vpz_ft: float) -> np.ndarray:
    """Minimum ``|relative altitude|`` (ft) over the shared 3-D conflict window.

    A *derived* quantity (parameterized by ``rpz``/``vpz`` per this module's
    contract, like :func:`predicted_tlos_s`), and the single shared implementation
    of the vertical separation the cost's penetration term (``r_v``) grades: the
    signed rel-alt line ``|d0 + dvs*t|`` sampled at its in-window minimum - the
    vertical zero-crossing ``-d0/dvs`` clipped into ``[max(tinconf, 0), toutconf]``
    (the SAME window :func:`predicted_tlos_s` keys off).

    Supersedes ``vsep_at_cpa_ft`` (vertical sep at the *horizontal* CPA instant)
    for conflict pairs: that measure reads ~0 for altitude-crossing traffic whose
    vertical crossing is offset in time from the horizontal CPA. Where no valid
    3-D window exists (a safe miss) it falls back to ``vsep_at_cpa_ft`` so a
    horizontally-distant pair keeps its classic, sensible reading.

    UNSIGNED by construction, and necessarily so: at the in-window minimum of a
    crossing pair the signed line is ~0, so a signed version of *this* measure
    would carry no direction exactly where direction matters. Escape direction
    comes from :func:`windowed_signed_vsep_at_entry_ft` instead.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    if tcpa.size == 0:
        return tcpa
    valid, tinconf, toutconf, d0, dvs = _shared_conflict_window(view, rpz_nm, vpz_ft)

    # Vertical sep at its in-window minimum (the rel-alt zero-crossing clamped
    # into the shared window), exactly the cost's ``v_min``.
    w0 = np.maximum(tinconf, 0.0)
    tv = np.clip(-d0 / dvs, w0, toutconf)
    v_min = np.abs(d0 + dvs * tv)
    return np.where(valid, v_min, np.asarray(view.vsep_at_cpa_ft, dtype=np.float64))


def windowed_min_hsep_nm(view: ConflictView, rpz_nm: float, vpz_ft: float) -> np.ndarray:
    """Minimum horizontal separation (nm) over the shared 3-D conflict window.

    The horizontal twin of :func:`windowed_min_vsep_ft`, and the single shared
    implementation of the separation the cost's penetration term (``r_h``)
    grades: the hyperbola ``sqrt(dcpa^2 + (vrel*(t - tcpa))^2)`` sampled at its
    in-window minimum - ``tcpa`` clipped into ``[max(tinconf, 0), toutconf]``
    (the SAME window the vertical measures key off).

    Differs from plain ``dcpa`` only by that clip, but the clip is the whole
    point: ``dcpa`` is the miss over ALL time, whereas the cost charges the pair
    only while it is simultaneously inside both bands. When the horizontal CPA
    falls outside that window - the pair is already separating horizontally by
    the time it converges vertically, or it leaves the vertical band before
    reaching the horizontal CPA - the two diverge, always with
    ``h_min >= dcpa``. Reading ``dcpa`` there overstates the horizontal threat
    relative to what the cost actually charges, inviting maneuvers that earn
    nothing on the cost channel and are paid for in track miles.

    Where no valid 3-D window exists (a safe miss) it falls back to ``dcpa_nm``,
    mirroring :func:`windowed_min_vsep_ft`'s fallback to ``vsep_at_cpa_ft``.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    if tcpa.size == 0:
        return tcpa
    valid, tinconf, toutconf, _d0, _dvs = _shared_conflict_window(view, rpz_nm, vpz_ft)

    dcpa = np.asarray(view.dcpa_nm, dtype=np.float64)
    vrel_nm_s = np.asarray(view.rel_hspd_kts, dtype=np.float64) / 3600.0
    # Horizontal sep at its in-window minimum (tcpa clamped into the shared
    # window), exactly the cost's ``h_min``.
    w0 = np.maximum(tinconf, 0.0)
    th = np.clip(tcpa, w0, toutconf)
    h_min = np.sqrt(dcpa**2 + (vrel_nm_s * (th - tcpa)) ** 2)
    return np.where(valid, h_min, dcpa)


def windowed_signed_vsep_at_entry_ft(
    view: ConflictView, rpz_nm: float, vpz_ft: float
) -> np.ndarray:
    """SIGNED relative altitude (ft) at entry to the shared 3-D conflict window.

    Positive = the intruder is ABOVE ownship as the encounter opens. The vertical
    counterpart of the signed horizontal ``RelPosAtCpa*`` primitives, and the
    direction half of what :func:`windowed_min_vsep_ft` reports the magnitude of:
    together they say "you will pass 200 ft apart, with the intruder above".

    Sampled at window ENTRY (``max(tinconf, 0)``), not at the in-window minimum,
    precisely because the minimum of a crossing pair sits AT the rel-alt zero
    where the sign is undefined and flips. Entry is the last instant at which the
    geometry still has an unambiguous side, so it is the instant that answers
    "climb or descend?". Where no valid 3-D window exists (a safe miss) it falls
    back to the signed rel-alt at the horizontal CPA, mirroring
    :func:`windowed_min_vsep_ft`'s fallback to ``vsep_at_cpa_ft``.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    if tcpa.size == 0:
        return tcpa
    valid, tinconf, _toutconf, d0, dvs = _shared_conflict_window(view, rpz_nm, vpz_ft)

    entry = d0 + dvs * np.maximum(tinconf, 0.0)
    miss = d0 + dvs * np.maximum(tcpa, 0.0)
    return np.where(valid, entry, miss)


def safety_margin_pairs(
    view: ConflictView,
    rpz_nm: float,
    vpz_ft: float,
    reach: float = 1.0,
) -> np.ndarray:
    """Per-intruder signed safety margin, clipped to ``[-1, reach]``.

    ``max(h/rpz - 1, |dz|/vpz - 1)``: positive iff the pair is clear in at least
    ONE axis (ICAO's rule, and exactly the complement of the LoS predicate
    ``h < rpz and |dz| < vpz``), zero on the protected-zone surface, negative
    inside it with magnitude equal to the fractional penetration of the axis that
    is least breached.

    Unlike :func:`windowed_min_vsep_ft` this is INSTANTANEOUS - no constant-
    velocity projection - because it is meant as the running payoff ``l(x)`` of a
    Hamilton-Jacobi safety value, which supplies the lookahead itself by
    propagating ``min`` over the trajectory. Baking a linear prediction in here
    would reintroduce the very assumption such a value function exists to avoid.

    ``reach`` caps the safe side. Uncapped, an intruder 100 nm away scores 19 and
    a regression target spans ``[-1, inf)``, so a learned head burns its capacity
    on how far away irrelevant traffic is; capping makes the informative band
    ``rpz .. (1 + reach) * rpz`` and everything beyond it indistinguishably safe.
    It is the analogue of the shaped-zone radius in the analytic cost.
    """
    h_nm = np.asarray(view.horiz_dist_now_nm, dtype=np.float64)
    dz_ft = np.asarray(view.dalt_now_ft, dtype=np.float64)  # already absolute
    margin = np.maximum(h_nm / rpz_nm - 1.0, dz_ft / vpz_ft - 1.0)
    return np.clip(margin, -1.0, float(reach))


def safety_margin(
    view: ConflictView,
    rpz_nm: float,
    vpz_ft: float,
    reach: float = 1.0,
) -> float:
    """Ownship safety margin: the ``min`` of :func:`safety_margin_pairs`.

    ``min`` because the state is unsafe if ANY intruder is in LoS, so the closest
    pair defines the margin. Taken over CLIPPED per-pair margins on purpose: on
    the raw values a single distant aircraft cannot mask a near one, but the
    saturated form also lets several near intruders each sit in the informative
    band instead of only the single closest registering.

    Returns ``reach`` (maximally safe) when there are no intruders.
    """
    pairs = safety_margin_pairs(view, rpz_nm, vpz_ft, reach)
    return float(pairs.min()) if pairs.size else float(reach)
