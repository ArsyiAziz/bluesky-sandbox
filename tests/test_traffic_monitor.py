"""The substep monitors only redo work when BlueSky's state actually changed.

``TrafficMonitor`` rebuilds conflict/LoS partners only when BlueSky's detector
has run (every ``asas_dt``), and ``QueryStateMonitor`` walks aircraft routes
only on a substep where some waypoint was reached. Both are optimisations that
must not change a single reported value, so each is checked against a reference
that does the work unconditionally, every substep - the algorithm as it was.
"""

from __future__ import annotations

from types import SimpleNamespace

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core import services
from bluesky_sandbox.core.services import TrafficMonitor
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_DT = 12.0
_STEPS = 6
_SEEDS = (0, 1)  # two episodes, so reset's in-place clear of confpairs is covered


class _RecomputeEverySubstep(TrafficMonitor):
    """Reference: rebuild partners and increments on every substep."""

    def record_substep(self) -> None:
        self.substep_count += 1
        simdt = float(self.env.config.simdt)
        self._sync_rows()
        conf_partners, los_partners = self._build_current_partner_sets()
        self._current_conflict_partners = tuple(
            () if p is None else tuple(sorted(p)) for p in conf_partners
        )
        self._current_los_partners = tuple(
            () if p is None else tuple(sorted(p)) for p in los_partners
        )
        n = len(self._table)
        inconf = np.asarray(bs.traf.cd.inconf, dtype=bool)[:n]
        if inconf.size < n:
            inconf = np.pad(inconf, (0, n - inconf.size), constant_values=False)
        self._table["conflict_step_substeps"] += inconf.astype(np.int32)
        self._table["conflict_total_s"] += inconf.astype(np.float64) * simdt
        los_mask = np.array([bool(p) for p in los_partners], dtype=bool)
        self._table["los_step_substeps"] += los_mask.astype(np.int32)
        self._table["los_total_s"] += los_mask.astype(np.float64) * simdt
        for store, current in (
            (self._conflict_step_partners, conf_partners),
            (self._los_step_partners, los_partners),
        ):
            for row, partners in enumerate(current):
                if partners:
                    if store[row] is None:
                        store[row] = set(partners)
                    else:
                        store[row].update(partners)


class _Scenario:
    def __init__(self, n_aircraft: int, *, route=None, queryables=None) -> None:
        self.queryables = queryables or {}
        self.n_aircraft = n_aircraft
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.95, 52.05, 4.5, 4.65)),
                    n_aircraft=n_aircraft,
                    params={"alt_ft": (8_000, 8_600), "spd_kts": (230, 270)},
                    route=route,
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=self.spawn,
            queryables=self.queryables,
            max_aircraft=self.n_aircraft,
        )


class _DeletesMidStep(BlueskyEnv):
    """Deletes one aircraft halfway through the second step of each episode."""

    def on_sim_step(self) -> None:
        if round(bs.sim.simt, 6) == _DT * 1.5 and bs.traf.ntraf > 1:
            bs.traf.delete(0)


def _config(**kwargs) -> EnvConfig:
    return EnvConfig(
        obs_fields=[], intruder_obs_fields=None, action_fields=[], dt=_DT, **kwargs
    )


def _separation_trace(env_cls, monitor_cls):
    env = env_cls(scenario=_Scenario(12), config=_config())
    if monitor_cls is not None:
        env._traffic_monitor = monitor_cls(env)
    trace = []
    try:
        for seed in _SEEDS:
            env.reset(seed=seed)
            for _ in range(_STEPS):
                env.step({})
                monitor = env._traffic_monitor
                for acidx, acid in enumerate(bs.traf.id):
                    trace.append(
                        (
                            seed,
                            acid,
                            monitor.substep_count,
                            monitor.build_separation_context(acid, acidx),
                        )
                    )
    finally:
        env.close()
    return trace


@pytest.mark.parametrize(
    "env_cls", [BlueskyEnv, _DeletesMidStep], ids=["steady", "aircraft-deleted-mid-step"]
)
def test_separation_matches_recomputing_every_substep(env_cls):
    reference = _separation_trace(env_cls, _RecomputeEverySubstep)
    cached = _separation_trace(env_cls, None)
    assert cached == reference
    # Guard against a vacuous pass: the traffic really was in conflict and LoS.
    assert sum(bool(ctx.conflict.step_partners) for *_, ctx in reference) > 50
    assert sum(ctx.los.substeps > 0 for *_, ctx in reference) > 5


def test_detection_is_reprocessed_once_per_detector_run_not_per_substep():
    env = BlueskyEnv(scenario=_Scenario(12), config=_config(asas_dt=2.0))
    refreshes = []
    monitor = env._traffic_monitor
    original = monitor._refresh_detection

    def counting(simdt):
        refreshes.append(bs.sim.simt)
        return original(simdt)

    monitor._refresh_detection = counting
    try:
        env.reset(seed=0)
        for _ in range(3):
            before = len(refreshes)
            env.step({})
            # The forced rebuild on the step's first substep, plus one per
            # detector run: dt / asas_dt = 6 - out of 240 substeps.
            assert len(refreshes) - before == 1 + 6
    finally:
        env.close()


def test_a_waypoint_reach_is_still_recorded_when_the_route_walk_is_skipped():
    merge = Waypoint(lat=52.0, lon=4.62, alt_ft=8_300, track_temporal_state=True)
    env = BlueskyEnv(
        scenario=_Scenario(6, route=["merge"], queryables={"merge": merge}),
        config=_config(),
    )
    monitor = env._query_state_monitor
    real_record, real_begin = monitor.record_substep, monitor.begin_step
    expected: dict[str, int] = {}

    def begin_step():
        real_begin()
        expected.clear()

    def record_substep():
        real_record()
        # Unconditional reference for the one column under test.
        table = monitor._table
        n = len(table)
        reached_rows = np.zeros(n, dtype=bool)
        idx = np.asarray(tuple(bs.traf.ap.idxreached), dtype=np.int64)
        reached_rows[idx[(idx >= 0) & (idx < n)]] = True
        active = np.array(
            [
                -1 if bs.traf.ap.route[i].iactwp is None
                else int(bs.traf.ap.route[i].iactwp)
                for i in range(n)
            ],
            dtype=np.int32,
        )
        swlnav = np.asarray(bs.traf.swlnav, dtype=bool)[:n]
        just_reached = np.where(swlnav, active - 1, active)
        route_indices = table["route_index"][:, table.col["merge"]]
        for row in np.flatnonzero(reached_rows & (route_indices == just_reached)):
            acid = table.ids[row]
            expected[acid] = expected.get(acid, 0) + 1

    monitor.begin_step = begin_step
    monitor.record_substep = record_substep
    reaches = 0
    try:
        env.reset(seed=0)
        table = monitor._table
        col = table.col["merge"]
        for _ in range(10):
            env.step({})
            reached = table["reached_step_substeps"]
            got = {
                acid: int(reached[row, col])
                for row, acid in enumerate(table.ids)
                if reached[row, col]
            }
            assert got == expected
            reaches += sum(got.values())
    finally:
        env.close()
    assert reaches > 0, "no aircraft reached the waypoint; the test proved nothing"


def test_step_partners_follow_their_aircraft_when_another_leaves(monkeypatch):
    # The equivalence test above cannot see this: its reference inherits the
    # same row sync. So pin it directly - partner sets are kept beside the
    # table and must be carried by callsign, like the table's own arrays.
    traf = SimpleNamespace(id=["A", "B", "C"])
    runtime = SimpleNamespace(aircraft_uids=np.array([10, 11, 12]))
    monkeypatch.setattr(services.bs, "traf", traf)
    monitor = TrafficMonitor(env=SimpleNamespace(_runtime=runtime))
    monitor.begin_step()
    monitor._conflict_step_partners[0] = {"C"}
    monitor._los_step_partners[2] = {"A"}

    # B leaves mid-step, and the order changes; the uids travel with them.
    traf.id = ["C", "A"]
    runtime.aircraft_uids = np.array([12, 10])
    monitor._sync_rows()

    assert monitor._conflict_step_partners == [None, {"C"}]
    assert monitor._los_step_partners == [{"A"}, None]
