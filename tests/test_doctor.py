"""The doctor says whether BlueSky's compiled modules actually loaded.

Both fall back quietly when they don't, so a machine can end up slower (geo)
or with no conflict detection at all (``CDMETHOD CSTATEBASED`` fails without
raising) and nothing else says so.
"""

from __future__ import annotations

from types import SimpleNamespace

from bluesky.tools import geo

from bluesky_sandbox import doctor


def _backend(name: str):
    return lambda: SimpleNamespace(__name__=f"bluesky.tools.geo.{name}")


def test_compiled_modules_are_reported(monkeypatch):
    monkeypatch.setattr(geo, "select_backend", _backend("_cgeo"))
    lines, ok = doctor._compiled_modules()
    assert ok
    assert "geo functions      : compiled" in lines
    assert "conflict detection : compiled (CSTATEBASED)" in lines


def test_python_geo_is_reported_but_not_a_failure(monkeypatch):
    monkeypatch.setattr(geo, "select_backend", _backend("_geo"))
    lines, ok = doctor._compiled_modules()
    assert ok
    assert "geo functions      : Python" in lines


def test_a_missing_compiled_detector_fails_the_check(monkeypatch):
    # Not at module level: importing BlueSky's traffic package before
    # bs.init() breaks its performance-model selection for later tests.
    from bluesky.traffic.asas import statebased  # noqa: PLC0415

    monkeypatch.delattr(statebased, "CStateBased")
    lines, ok = doctor._compiled_modules()
    assert not ok
    assert "conflict detection : UNAVAILABLE (CSTATEBASED)" in lines
    assert any("default cd_method (CSTATEBASED) needs it" in line for line in lines)
