"""A waypoint's target arrival time in a design: its slack carried onto every
route step over it, built and generated - never passed to the Waypoint."""

from __future__ import annotations

from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer.builder import build_scenario

from .test_designer import _example_design_spec


def test_a_waypoints_arrival_slack_rides_on_every_route_over_it():
    spec = _example_design_spec()
    spec.queryables["fix"] = {
        "type": "waypoint",
        "lat": 52.0,
        "lon": 4.5,
        "arrival_slack_s": {"type": "range", "low": 0, "high": 120},
    }
    spec.spawn = {**spec.spawn, "routes": {"r": ["fix"]}}
    spawn = build_scenario(spec).support().spawn
    assert spawn.routes["r"] == [{"waypoint": "fix", "arrival_slack_s": (0.0, 120.0)}]
    files = codegen.generate_task(spec, "Timed")
    scenario = next(t for p, t in files.items() if p.endswith("scenario.py"))
    assert "'arrival_slack_s': (0, 120)" in scenario
    assert "arrival_slack_s=" not in scenario  # not a Waypoint argument
