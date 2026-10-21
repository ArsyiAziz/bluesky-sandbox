"""Static environment config for this generated task."""

from __future__ import annotations

from bluesky_sandbox.config import EnvConfig

from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields import queryables as qobs
from bluesky_sandbox.interface.wrappers.observations.normalizer import (
    CircularNormalizer, MinMaxNormalizer, PowerNormalizer, SignedPowerNormalizer, StepNormalizer, SymmetricNormalizer,
)

CONFIG = EnvConfig(
    obs_fields=list((
            obs.ActiveRouteWaypointDistanceNm(low=0, high=60, normalizer=MinMaxNormalizer(clipped=True)),
            obs.ActiveRouteWaypointTrackErrorDeg(normalizer=CircularNormalizer()),
            obs.AltFt(low=15000, high=25000, normalizer=MinMaxNormalizer(clipped=True)),
            obs.CasKts(low=200, high=350, normalizer=MinMaxNormalizer(clipped=True)),
            obs.VsFtMin(normalizer=SymmetricNormalizer(clipped=True)),
            obs.TrkDeg(normalizer=CircularNormalizer()),
            obs.GsKts(low=250, high=550, normalizer=MinMaxNormalizer(clipped=True)),
            obs.ApCasKts(low=200, high=350, normalizer=MinMaxNormalizer(clipped=True)),
            obs.PerfVminKts(normalizer=MinMaxNormalizer(clipped=True)),
            obs.PerfVmaxKts(normalizer=MinMaxNormalizer(clipped=True)),
        )),
    intruder_obs_fields=list((
            obs.RelPosAlongTrackNm(low=-60, high=60, normalizer=SymmetricNormalizer(clipped=True)),
            obs.RelPosCrossTrackNm(low=-60, high=60, normalizer=SymmetricNormalizer(clipped=True)),
            obs.AltFt().relative_to_own(low=-2000, high=2000, normalizer=SymmetricNormalizer(clipped=True)),
            obs.RelVelAlongTrackKts(low=-600, high=600, normalizer=SymmetricNormalizer(clipped=True)),
            obs.RelVelCrossTrackKts(low=-600, high=600, normalizer=SymmetricNormalizer(clipped=True)),
            obs.RelVsFtMin(normalizer=SymmetricNormalizer(clipped=True)),
            obs.DistToOwnNm(low=0, high=90, normalizer=MinMaxNormalizer(clipped=True)),
            obs.ConflictTcpaS(normalizer=MinMaxNormalizer(clipped=True)),
            obs.ConflictTlosS(normalizer=MinMaxNormalizer(clipped=True)),
            obs.InConf(),
            obs.InLosNow(),
        )),
    action_fields=list((
            act.Clearance(act.HdgDeg()),
            act.Clearance(act.SpdKts()),
            act.Clearance(act.AutopilotLnav()),
        )),
    dt=1,
    simdt=1,
    asas_dt=1,
    cd_method='CSTATEBASED',
    reso_method=None,
    pz_radius_nm=5,
    pz_height_ft=1000,
    lookahead_s=300,
    performance_model='openap',
    wind_dir_deg=270,
    wind_kts=0,
    turbulence_kts=0,
    gust_tau_s=30,
)
