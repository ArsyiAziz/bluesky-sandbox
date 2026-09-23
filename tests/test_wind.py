"""Wind: one vector, one sign convention.

``wind_dir_deg`` is aviation-standard - the direction the wind blows FROM - so
every consumer that wants a velocity has to negate it. That negation used to be
written out in both ``BlueSkyRuntime.apply_wind`` and
``clearance.predicted_conflict``; these tests pin it to one place.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from bluesky.tools.aero import kts

from bluesky_sandbox.sim.weather import UniformWind, wind_from_config


class _Cfg:
    wind_dir_deg = 240.0
    wind_kts = 25.0
    turbulence_kts = 12.0
    gust_tau_s = 20.0


# ---- sign convention ---------------------------------------------------- #


@pytest.mark.parametrize(
    ("dir_deg", "expect_n", "expect_e"),
    [
        (270.0, 0.0, +1.0),  # westerly: blows FROM the west, pushes east
        (90.0, 0.0, -1.0),  # easterly: pushes west
        (180.0, +1.0, 0.0),  # southerly: pushes north
        (0.0, -1.0, 0.0),  # northerly: pushes south
    ],
)
def test_mean_points_where_the_air_goes(dir_deg, expect_n, expect_e):
    """A 'westerly' pushes aircraft EAST - the vector is the reverse of dir_deg."""
    wind = UniformWind(dir_deg=dir_deg, speed_kts=30.0)
    vn, ve = wind.mean_ne_ms()
    speed_ms = 30.0 * kts
    assert vn == pytest.approx(expect_n * speed_ms, abs=1e-9)
    assert ve == pytest.approx(expect_e * speed_ms, abs=1e-9)


def test_magnitude_is_the_configured_speed():
    wind = UniformWind(dir_deg=37.0, speed_kts=42.0)
    assert math.hypot(*wind.mean_ne_ms()) == pytest.approx(42.0 * kts)


# ---- still / dynamic predicates ----------------------------------------- #


def test_no_wind_and_no_turbulence_is_still():
    assert UniformWind().is_still
    assert not UniformWind(speed_kts=5.0).is_still
    assert not UniformWind(turbulence_kts=5.0).is_still


def test_only_a_gusting_field_needs_reapplying_each_step():
    """A steady field is pushed to BlueSky once per episode, not per step."""
    assert not UniformWind(speed_kts=30.0).is_dynamic
    assert UniformWind(speed_kts=30.0, turbulence_kts=8.0).is_dynamic


# ---- the gust ------------------------------------------------------------ #


def test_steady_field_never_gusts():
    wind = UniformWind(dir_deg=270.0, speed_kts=20.0, turbulence_kts=0.0)
    before = wind.current_ne_ms()
    for _ in range(50):
        wind.advance(1.0, np.random.default_rng(0))
    assert wind.current_ne_ms() == before


def test_gust_is_zero_mean_with_the_requested_rms():
    """Stationary RMS must match ``turbulence_kts``, independent of dt."""
    wind = UniformWind(speed_kts=0.0, turbulence_kts=10.0, gust_tau_s=20.0)
    rng = np.random.default_rng(3)
    samples = []
    for _ in range(20_000):
        wind.advance(1.0, rng)
        samples.append(wind.current_ne_ms()[0])
    arr = np.asarray(samples[1000:])  # discard burn-in
    assert abs(arr.mean()) < 0.5 * kts  # zero-mean
    assert arr.std() == pytest.approx(10.0 * kts, rel=0.1)


def test_gust_rms_does_not_depend_on_step_size():
    """The OU step_sigma is dt-corrected; a finer dt must not shrink the gust."""
    stds = []
    for dt in (0.5, 4.0):
        wind = UniformWind(turbulence_kts=10.0, gust_tau_s=20.0)
        rng = np.random.default_rng(7)
        vals = []
        for _ in range(int(40_000 / dt)):
            wind.advance(dt, rng)
            vals.append(wind.current_ne_ms()[1])
        stds.append(np.asarray(vals[2000:]).std())
    assert stds[0] == pytest.approx(stds[1], rel=0.1)


def test_reset_drops_the_gust_but_keeps_the_mean():
    wind = UniformWind(dir_deg=270.0, speed_kts=20.0, turbulence_kts=15.0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        wind.advance(1.0, rng)
    assert wind.current_ne_ms() != wind.mean_ne_ms()  # a gust is live

    wind.reset()
    assert wind.current_ne_ms() == wind.mean_ne_ms()  # back to the mean


def test_mean_excludes_the_live_gust():
    """Predictive callers take the mean: gusts decorrelate within a step."""
    wind = UniformWind(dir_deg=270.0, speed_kts=20.0, turbulence_kts=15.0)
    mean_before = wind.mean_ne_ms()
    rng = np.random.default_rng(2)
    for _ in range(10):
        wind.advance(1.0, rng)
    assert wind.mean_ne_ms() == mean_before


# ---- config bridge ------------------------------------------------------- #


def test_wind_from_config_carries_every_setting():
    wind = wind_from_config(_Cfg())
    assert (wind.dir_deg, wind.speed_kts) == (240.0, 25.0)
    assert (wind.turbulence_kts, wind.gust_tau_s) == (12.0, 20.0)


def test_wind_from_config_tolerates_a_config_without_wind_fields():
    wind = wind_from_config(object())
    assert wind.is_still
