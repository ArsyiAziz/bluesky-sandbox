"""Smoke test for the designer FastAPI server.

Standalone runnable (``python -m bluesky_sandbox.ui.designer.tests.test_api``).
Drives the app in-process via Starlette's TestClient - no network port needed.
"""

from __future__ import annotations

from scipy.stats import randint
from starlette.testclient import TestClient

from bluesky_sandbox.sim.bounds import BoxFootprint, ConstantAltitudeBand, RegionBounds
from bluesky_sandbox.sim.queryables import QueryRegion
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.api import create_app


def _example_spec_dict() -> dict:
    airspace = RegionBounds(
        BoxFootprint(51.5, 52.5, 4.0, 5.5), ConstantAltitudeBand(0, 20_000)
    )
    spawn = SpawnConfig(
        regions=[
            SpawnRegion(
                RegionBounds(BoxFootprint(51.6, 52.4, 4.1, 5.4)),
                n_aircraft=randint(2, 6),
                params={"alt_ft": (5_000, 15_000), "spd_kts": (200, 280)},
            )
        ]
    )
    goal = QueryRegion(
        RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.6), ConstantAltitudeBand(2_000, 8_000)),
        color="cyan",
    )
    env = S.EnvSpec(
        obs_fields=[S.FieldRef("LatDeg"), S.FieldRef("LonDeg"), S.FieldRef("AltFt")],
        intruder_obs_fields=[S.FieldRef("DistToOwnNm")],
        action_fields=[S.FieldRef("HdgDeg"), S.FieldRef("SpdKts")],
        allowed_aircraft=["A320", "B738"],
    )
    return S.DesignSpec(
        env=env,
        spawn=S.dump(spawn),
        airspace=S.dump(airspace),
        queryables={"goal": S.dump(goal)},
        metadata={"name": "demo"},
    ).to_dict()


def main() -> int:
    client = TestClient(create_app())
    failures = 0

    def check(label, cond):
        nonlocal failures
        if cond:
            print(f"  {label} OK")
        else:
            failures += 1
            print(f"  {label} FAILED")

    # health
    r = client.get("/api/health")
    check("health", r.status_code == 200 and r.json()["status"] == "ok")

    # catalog
    r = client.get("/api/catalog")
    cat = r.json()
    check(
        "catalog",
        r.status_code == 200
        and {"footprints", "altitude_bands", "obs_fields", "action_fields", "aircraft_types"}
        <= set(cat)
        and any(f["name"] == "BoxFootprint" for f in cat["footprints"])
        and any(f["name"] == "DistToOwnNm" and f["pair_only"] for f in cat["obs_fields"]),
    )

    # nav features within an airspace window
    bounds_spec = S.dump(RegionBounds(BoxFootprint(52.0, 52.6, 4.3, 5.0)))
    r = client.post("/api/nav/features", json={"bounds": bounds_spec, "airport_limit": 20})
    feats = r.json()
    check(
        "nav/features",
        r.status_code == 200
        and "EHAM" in {a["icao"] for a in feats["airports"]}
        and isinstance(feats["waypoints"], list),
    )

    # nav resolve
    r = client.get("/api/nav/airport/EHAM")
    check("nav/airport", r.status_code == 200 and r.json()["icao"] == "EHAM")
    r = client.get("/api/nav/waypoint/EKROS")
    check("nav/waypoint", r.status_code == 200 and abs(r.json()["lat_deg"] - 52.237) < 0.1)
    r = client.get("/api/nav/waypoint/ZZZZNOPE")
    check("nav/waypoint 404", r.status_code == 404)

    spec_dict = _example_spec_dict()

    # validate
    r = client.post("/api/spec/validate", json=spec_dict)
    body = r.json()
    check(
        "spec/validate ok",
        r.status_code == 200
        and body["ok"] is True
        and body["summary"]["max_aircraft"] == 5
        and body["summary"]["obs_fields"] == ["lat_deg", "lon_deg", "alt_ft"],
    )

    # validate a broken spec (unknown field ref) -> ok:false, not a crash
    broken = _example_spec_dict()
    broken["env"]["obs_fields"] = [{"field": "custom_fields:DoesNotExist"}]
    r = client.post("/api/spec/validate", json=broken)
    check("spec/validate error", r.status_code == 200 and r.json()["ok"] is False)

    # preview
    r = client.post("/api/spec/preview", json={"spec": spec_dict, "seed": 1})
    prev = r.json()
    check(
        "spec/preview",
        r.status_code == 200
        and prev["airspace"] is not None
        and len(prev["spawn_regions"]) == 1
        and len(prev["sampled_aircraft"]) <= prev["max_aircraft"]
        and all("lat" in ac and "lon" in ac for ac in prev["sampled_aircraft"]),
    )

    # nav search
    r = client.get("/api/nav/search", params={"q": "EHAM"})
    check("nav/search", r.status_code == 200 and "EHAM" in {a["icao"] for a in r.json()["airports"]})

    # generate task package
    r = client.post("/api/spec/generate", json={"spec": spec_dict, "package_name": "Demo Env"})
    gen = r.json()
    check(
        "spec/generate",
        r.status_code == 200
        and gen["package"] == "demo_env"
        and "demo_env/task.py" not in gen["files"]
        and "demo_env/design.py" not in gen["files"]
        and "demo_env/design.json" in gen["files"]
        and "RegionBounds(" in gen["files"]["demo_env/scenario.py"]
        and "def reward(self," in gen["files"]["demo_env/env.py"],
    )

    # store round-trip
    r = client.put("/api/specs/_smoke_test", json=spec_dict)
    check("specs PUT", r.status_code == 200)
    r = client.get("/api/specs")
    check("specs LIST", any(s["name"] == "_smoke_test" for s in r.json()))
    r = client.get("/api/specs/_smoke_test")
    check("specs GET", r.status_code == 200 and r.json()["metadata"]["name"] == "demo")
    r = client.delete("/api/specs/_smoke_test")
    check("specs DELETE", r.status_code == 200)

    total = 15
    print(f"\n{total - failures}/{total} checks passed.")
    return 1 if failures else 0


def test_api_smoke() -> None:
    """Run the checks above under pytest.

    ``main`` is written to be runnable standalone, so before this wrapper the
    module collected zero tests: pytest imported it (and could fail on that
    import) without ever executing a single check.
    """
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
